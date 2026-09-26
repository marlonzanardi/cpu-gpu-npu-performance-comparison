from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import click

from benchmark.stats import median, percentile

COMPUTE_UNITS = ("cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine")
REQUESTED_DEVICE = {
    "cpuOnly": "cpu",
    "cpuAndGPU": "gpu",
    "cpuAndNeuralEngine": "neuralEngine",
}


def ensure_binary(root: Path) -> Path:
    source = root / "native" / "main.m"
    binary = root / "build" / "bench"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "clang",
            "-fobjc-arc",
            "-framework",
            "CoreML",
            "-framework",
            "Foundation",
            "-O2",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def swift_json(binary: Path, args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [str(binary), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or f"bench exited {exc.returncode}") from exc
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one JSON line, got {result.stdout!r}")
    payload = json.loads(lines[0])
    if not isinstance(payload, dict):
        raise RuntimeError("bench payload was not an object")
    return payload


def weight_share(plan: dict[str, Any], device: str) -> float | None:
    weights = plan.get("preferred_weight")
    counts = plan.get("preferred_counts")
    source: dict[str, float] | None = None
    if isinstance(weights, dict) and sum(float(value) for value in weights.values()) > 0:
        source = {str(key): float(value) for key, value in weights.items()}
    elif isinstance(counts, dict) and sum(int(value) for value in counts.values()) > 0:
        source = {str(key): float(value) for key, value in counts.items()}
    if source is None:
        return None
    total = sum(source.values())
    if total <= 0:
        return None
    return source.get(device, 0.0) / total


def dominant_device(plan: dict[str, Any]) -> tuple[str | None, float | None]:
    weights = plan.get("preferred_weight")
    counts = plan.get("preferred_counts")
    source: dict[str, float] | None = None
    if isinstance(weights, dict) and sum(float(value) for value in weights.values()) > 0:
        source = {str(key): float(value) for key, value in weights.items()}
    elif isinstance(counts, dict):
        source = {str(key): float(value) for key, value in counts.items()}
    if not source:
        return None, None
    total = sum(source.values())
    if total <= 0:
        return None, None
    name = max(source, key=source.get)
    return name, source[name] / total


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "precision",
        "batch",
        "compute_units",
        "placement_matches_request",
        "dominant_device",
        "dominant_weight_share",
        "cpu_weight_share",
        "gpu_weight_share",
        "neural_engine_weight_share",
        "samples",
        "sessions",
        "median_ms",
        "p95_ms",
        "throughput_per_s",
        "warmup_settled",
        "output_mean_abs",
    ]
    lines = [",".join(fieldnames)]
    for row in rows:
        cells: list[str] = []
        for field in fieldnames:
            value = row[field]
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _warmup_settled(warmup: list[float], measured_median: float) -> bool:
    if len(warmup) < 5 or measured_median <= 0:
        return False
    tail = median(warmup[-5:])
    return tail <= measured_median * 1.25


def _packages() -> dict[str, str]:
    import coremltools
    import numpy
    import torch
    import torchvision

    return {
        "coremltools": coremltools.__version__,
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def execute(
    root: Path,
    machine: dict[str, object],
    models: tuple[str, ...],
    precisions: tuple[str, ...],
    batches: tuple[int, ...],
    warmup: int,
    iterations: int,
    sessions: int,
    pause_seconds: float,
) -> Path:
    from datetime import datetime, timezone
    import time

    from benchmark.convert import convert_model

    if warmup < 1 or iterations < 1 or sessions < 1:
        raise ValueError("warmup, iterations, and sessions must be positive")
    if any(batch < 1 for batch in batches):
        raise ValueError("batch must be positive")

    binary = ensure_binary(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from benchmark.hardware import machine_slug

    output = root / "results" / machine_slug(machine) / stamp
    output.mkdir(parents=True)
    samples_path = output / "samples.jsonl"
    plans_path = output / "plans.jsonl"
    failures: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, int, str], list[float]] = {}
    session_counts: dict[tuple[str, str, int, str], int] = {}
    warmup_flags: dict[tuple[str, str, int, str], list[bool]] = {}
    output_means: dict[tuple[str, str, int, str], list[float]] = {}
    plans: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    model_records: list[dict[str, object]] = []

    for model_name in models:
        for precision in precisions:
            for batch in batches:
                key_label = f"{model_name} {precision} batch {batch}"
                click.echo(f"preparing {key_label}")
                try:
                    compiled = convert_model(root, model_name, precision, batch)
                except Exception as exc:
                    failures.append(
                        {
                            "stage": "convert",
                            "model": model_name,
                            "precision": precision,
                            "batch": batch,
                            "error": str(exc),
                        }
                    )
                    click.echo(f"convert failed for {key_label}: {exc}", err=True)
                    continue
                model_records.append(
                    {
                        "model": model_name,
                        "precision": precision,
                        "batch": batch,
                        "compiled_model_sha256": directory_sha256(compiled),
                        "path": str(compiled.relative_to(root)),
                    }
                )
                for compute_units in COMPUTE_UNITS:
                    plan_key = (model_name, precision, batch, compute_units)
                    try:
                        plan = swift_json(binary, ["plan", str(compiled), compute_units])
                    except Exception as exc:
                        failures.append(
                            {
                                "stage": "plan",
                                "model": model_name,
                                "precision": precision,
                                "batch": batch,
                                "compute_units": compute_units,
                                "error": str(exc),
                            }
                        )
                        click.echo(f"plan failed for {key_label} {compute_units}: {exc}", err=True)
                        continue
                    plan["model"] = model_name
                    plan["precision"] = precision
                    plan["batch"] = batch
                    plans[plan_key] = plan
                    with plans_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(plan, sort_keys=True) + "\n")
                    for session in range(1, sessions + 1):
                        if session > 1 and pause_seconds > 0:
                            time.sleep(pause_seconds)
                        click.echo(f"measuring {key_label} {compute_units} session {session}/{sessions}")
                        try:
                            prediction = swift_json(
                                binary,
                                [
                                    "predict",
                                    str(compiled),
                                    compute_units,
                                    str(warmup),
                                    str(iterations),
                                ],
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "stage": "predict",
                                    "model": model_name,
                                    "precision": precision,
                                    "batch": batch,
                                    "compute_units": compute_units,
                                    "session": session,
                                    "error": str(exc),
                                }
                            )
                            click.echo(
                                f"predict failed for {key_label} {compute_units} session {session}: {exc}",
                                err=True,
                            )
                            continue
                        latencies = [float(value) for value in prediction["latencies_ms"]]
                        warmup_latencies = [float(value) for value in prediction["warmup_latencies_ms"]]
                        grouped.setdefault(plan_key, []).extend(latencies)
                        session_counts[plan_key] = session_counts.get(plan_key, 0) + 1
                        settled = _warmup_settled(warmup_latencies, median(latencies))
                        warmup_flags.setdefault(plan_key, []).append(settled)
                        output_means.setdefault(plan_key, []).append(float(prediction["output_mean_abs"]))
                        with samples_path.open("a", encoding="utf-8") as handle:
                            for iteration, latency in enumerate(latencies):
                                handle.write(
                                    json.dumps(
                                        {
                                            "model": model_name,
                                            "precision": precision,
                                            "batch": batch,
                                            "compute_units": compute_units,
                                            "session": session,
                                            "iteration": iteration,
                                            "latency_ms": latency,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                        click.echo(
                            f"  median {median(latencies):.3f} ms  p95 {percentile(latencies, 0.95):.3f} ms"
                        )

    rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        model_name, precision, batch, compute_units = key
        latencies = grouped[key]
        plan = plans[key]
        requested = REQUESTED_DEVICE[compute_units]
        share = weight_share(plan, requested)
        dominant, dominant_share = dominant_device(plan)
        matches = share is not None and share >= 0.5
        median_ms = median(latencies)
        rows.append(
            {
                "model": model_name,
                "precision": precision,
                "batch": batch,
                "compute_units": compute_units,
                "placement_matches_request": matches,
                "dominant_device": dominant if dominant is not None else "",
                "dominant_weight_share": dominant_share if dominant_share is not None else 0.0,
                "cpu_weight_share": weight_share(plan, "cpu") or 0.0,
                "gpu_weight_share": weight_share(plan, "gpu") or 0.0,
                "neural_engine_weight_share": weight_share(plan, "neuralEngine") or 0.0,
                "samples": len(latencies),
                "sessions": session_counts[key],
                "median_ms": median_ms,
                "p95_ms": percentile(latencies, 0.95),
                "throughput_per_s": (1000.0 / median_ms) * batch if median_ms > 0 else 0.0,
                "warmup_settled": all(warmup_flags.get(key, [])),
                "output_mean_abs": median(output_means[key]),
            }
        )
    write_summary(output / "summary.csv", rows)
    manifest = {
        "created_at": stamp,
        "runtime": "coreml",
        "comparison": "same compiled mlprogram loaded with cpuOnly, cpuAndGPU, and cpuAndNeuralEngine",
        "accuracy_evaluated": False,
        "energy_collected": False,
        "energy_reason": "powermetrics requires sudo and was not sampled",
        "percentile_method": "linear_rank",
        "placement_rule": "a device result matches the request when that device holds at least half of the compute-plan weight",
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "sessions": sessions,
        "inter_session_pause_seconds": pause_seconds,
        "machine": machine,
        "packages": _packages(),
        "models": model_records,
        "failures": failures,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not rows:
        raise RuntimeError(f"no measurements were recorded; see {output}")
    _print_table(rows)
    click.echo(f"wrote {output}")
    return output


def _print_table(rows: list[dict[str, object]]) -> None:
    click.echo("")
    header = (
        f"{'model':<16} {'precision':<8} {'batch':>5} {'units':<20} "
        f"{'match':<5} {'dominant':<14} {'median_ms':>10} {'p95_ms':>10} {'img/s':>10}"
    )
    click.echo(header)
    for row in rows:
        click.echo(
            f"{str(row['model']):<16} {str(row['precision']):<8} {int(str(row['batch'])):>5} "
            f"{str(row['compute_units']):<20} {str(row['placement_matches_request']):<5} "
            f"{str(row['dominant_device']):<14} {float(str(row['median_ms'])):>10.3f} "
            f"{float(str(row['p95_ms'])):>10.3f} {float(str(row['throughput_per_s'])):>10.1f}"
        )
