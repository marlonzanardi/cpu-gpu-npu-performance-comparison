import csv
import json
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from bench.hardware import collect_hardware
from bench.power import NvidiaPowerSampler, summarize_power
from bench.report import write_figures
from bench.runtime import load_onnxruntime

ROOT = Path(__file__).resolve().parents[1]
MACHINE_ID = "windows-workstation"
SESSION_GAP_S = 60
MIN_ENERGY_WINDOW_S = 3.0


def load_protocol():
    with (ROOT / "protocol" / "experiments.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_index():
    return json.loads((ROOT / "models" / "index.json").read_text(encoding="utf-8"))


def ensure_models():
    if (ROOT / "models" / "index.json").exists():
        return
    subprocess.check_call([sys.executable, "-m", "bench.export_models"], cwd=ROOT)


def make_session(ort, model_path, device):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": "0", "use_tf32": "0"}), "CPUExecutionProvider"]
    else:
        options.intra_op_num_threads = os_cpu_count()
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    active = session.get_providers()[0]
    expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if active != expected:
        raise RuntimeError(f"{device} session used {active}")
    return session


def os_cpu_count():
    import os

    return os.cpu_count() or 1


def input_tensor(seed, batch, precision):
    rng = np.random.default_rng(seed)
    dtype = np.float16 if precision == "fp16" else np.float32
    return rng.standard_normal((batch, 3, 224, 224)).astype(dtype)


def measure(session, feeds, iterations):
    latencies = []
    started = time.perf_counter()
    for _ in range(iterations):
        mark = time.perf_counter_ns()
        session.run(None, feeds)
        latencies.append((time.perf_counter_ns() - mark) / 1e6)
    return latencies, time.perf_counter() - started


def sustained_energy(session, feeds):
    sampler = NvidiaPowerSampler()
    sampler.start()
    count = 0
    started = time.perf_counter()
    while True:
        session.run(None, feeds)
        count += 1
        elapsed = time.perf_counter() - started
        if elapsed >= MIN_ENERGY_WINDOW_S:
            break
    sampler.stop()
    power_summary = summarize_power(sampler.samples)
    energy = None
    if power_summary["power_mean_w"] is not None and count:
        energy = power_summary["power_mean_w"] * elapsed / count
    return energy, elapsed, count, power_summary, sampler.samples


def validate_output(session, feeds, batch):
    output = np.asarray(session.run(None, feeds)[0])
    if output.shape[0] != batch or not np.isfinite(output).all():
        raise RuntimeError(f"unexpected output shape {output.shape}")


def configuration_seed(model_id, precision, batch):
    return zlib.adler32(f"{model_id}|{precision}|{batch}".encode("utf-8"))


def jobs_for(index):
    jobs = []
    for model_id, spec in index["models"].items():
        jobs.append((model_id, "fp32", "cpu", ROOT / "models" / spec["fp32"]["path"]))
        jobs.append((model_id, "fp32", "cuda", ROOT / "models" / spec["fp32"]["path"]))
        jobs.append((model_id, "fp16", "cuda", ROOT / "models" / spec["fp16"]["path"]))
    return jobs


def probe(ort, index):
    checks = [
        ("mobilenet_v2", "fp32", "cpu", 1),
        ("mobilenet_v2", "fp32", "cuda", 8),
        ("mobilenet_v2", "fp16", "cuda", 1),
        ("resnet50", "fp32", "cpu", 1),
        ("resnet50", "fp32", "cuda", 1),
    ]
    available = {(model_id, precision, device): path for model_id, precision, device, path in jobs_for(index)}
    for model_id, precision, device, batch in checks:
        path = available[(model_id, precision, device)]
        session = make_session(ort, path, device)
        feeds = {session.get_inputs()[0].name: input_tensor(1, batch, precision)}
        validate_output(session, feeds, batch)
        print(f"probe ok {model_id} {device} {precision} batch {batch}", flush=True)


def run_study(smoke):
    protocol = load_protocol()
    ensure_models()
    ort = load_onnxruntime()
    index = model_index()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CUDA provider missing: {ort.get_available_providers()}")
    probe(ort, index)

    if smoke:
        sessions = 1
        warmup = 3
        iterations = 5
        batches = [1]
        gap_s = 0
        mode = "smoke"
    else:
        sessions = int(protocol["sessions"])
        warmup = int(protocol["warmup_iterations"])
        iterations = int(protocol["measured_iterations"])
        batches = [int(batch) for batch in protocol["batch_sweep"]]
        gap_s = SESSION_GAP_S
        mode = "full"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "results" / MACHINE_ID / stamp
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    hardware = collect_hardware(ort)
    manifest = {
        "study": protocol["study"],
        "experiment": "A",
        "machine_id": MACHINE_ID,
        "mode": mode,
        "status": "running",
        "started_utc": stamp,
        "runtime": "onnxruntime",
        "scope": "inference",
        "precision_iso": "fp32",
        "cuda_use_tf32": False,
        "vendor_precision": "fp16",
        "latency_includes": "host_to_device_compute_device_to_host",
        "cpu_energy": "not_measured",
        "gpu_energy": "nvidia-smi board power averaged over a sustained repeat of the same inference, at least 3 seconds, times that window, divided by inferences in the window. The latency sample stays on the protocol iterations and does not include the power sampler.",
        "gpu_clocks": "not locked",
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "sessions": sessions,
        "session_gap_s": gap_s,
        "batches": batches,
        "input_seed_rule": "adler32(model|precision|batch), reused across sessions and devices",
        "intra_op_threads_cpu": os_cpu_count(),
        "hardware": hardware,
        "models": index,
        "protocol": "protocol/experiments.yaml",
    }
    manifest_path = run_dir / "manifest.json"
    samples_path = run_dir / "samples.jsonl"
    power_path = run_dir / "power.jsonl"
    summary_path = run_dir / "summary.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_fields = [
        "session",
        "model",
        "device",
        "precision",
        "batch",
        "iterations",
        "warmup_iterations",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_min_ms",
        "window_s",
        "throughput_ips",
        "energy_j_per_inference",
        "energy_source",
        "power_mean_w",
        "gpu_util_mean_pct",
        "gpu_memory_used_mib_mean",
        "power_samples",
        "energy_window_s",
        "energy_iterations",
    ]
    planned = [(model_id, precision, device, path, batch) for model_id, precision, device, path in jobs_for(index) for batch in batches]

    with samples_path.open("w", encoding="utf-8") as sample_file, power_path.open("w", encoding="utf-8") as power_file, summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        for session_index in range(1, sessions + 1):
            for model_id, precision, device, path, batch in planned:
                session = make_session(ort, path, device)
                input_name = session.get_inputs()[0].name
                tensor = input_tensor(configuration_seed(model_id, precision, batch), batch, precision)
                feeds = {input_name: tensor}
                for _ in range(warmup):
                    session.run(None, feeds)
                latencies, window_s = measure(session, feeds, iterations)
                if device == "cuda":
                    energy, energy_window_s, energy_iterations, power_summary, power_samples = sustained_energy(session, feeds)
                    energy_source = "nvidia-smi" if energy is not None else ""
                    for sample in power_samples:
                        power_file.write(json.dumps({
                            "session": session_index,
                            "model": model_id,
                            "device": device,
                            "precision": precision,
                            "batch": batch,
                            **sample,
                        }) + "\n")
                else:
                    energy = None
                    energy_source = ""
                    energy_window_s = None
                    energy_iterations = 0
                    power_summary = {
                        "power_mean_w": None,
                        "gpu_util_mean_pct": None,
                        "gpu_memory_used_mib_mean": None,
                        "power_samples": 0,
                    }
                for iteration, latency_ms in enumerate(latencies):
                    sample_file.write(json.dumps({
                        "session": session_index,
                        "model": model_id,
                        "device": device,
                        "precision": precision,
                        "batch": batch,
                        "iteration": iteration,
                        "latency_ms": latency_ms,
                    }) + "\n")
                sample_file.flush()
                row = {
                    "session": session_index,
                    "model": model_id,
                    "device": device,
                    "precision": precision,
                    "batch": batch,
                    "iterations": iterations,
                    "warmup_iterations": warmup,
                    "latency_median_ms": float(np.median(latencies)),
                    "latency_p95_ms": float(np.percentile(latencies, 95)),
                    "latency_min_ms": float(np.min(latencies)),
                    "window_s": window_s,
                    "throughput_ips": (batch * iterations) / window_s,
                    "energy_j_per_inference": energy,
                    "energy_source": energy_source,
                    "power_mean_w": power_summary["power_mean_w"],
                    "gpu_util_mean_pct": power_summary["gpu_util_mean_pct"],
                    "gpu_memory_used_mib_mean": power_summary["gpu_memory_used_mib_mean"],
                    "power_samples": power_summary["power_samples"],
                    "energy_window_s": energy_window_s,
                    "energy_iterations": energy_iterations,
                }
                writer.writerow(row)
                summary_file.flush()
                energy_text = f"{energy:.6f} J" if energy is not None else "n/a"
                print(
                    f"session {session_index} {model_id} {device} {precision} batch {batch} "
                    f"median {row['latency_median_ms']:.3f} ms p95 {row['latency_p95_ms']:.3f} ms {energy_text}",
                    flush=True,
                )
            if session_index < sessions and gap_s:
                print(f"session gap {gap_s}s", flush=True)
                time.sleep(gap_s)

    manifest["status"] = "completed"
    manifest["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_figures(run_dir)
    print(run_dir, flush=True)
    return run_dir
