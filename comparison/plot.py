import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SUMMARY = ROOT / "results" / "windows-workstation" / "20260926T195237Z" / "summary.csv"
MAC_SUMMARY = ROOT / "results" / "apple-m4-pro" / "20260926T204025Z" / "summary.csv"
OUT = ROOT / "results" / "comparison" / "20260926"

MODEL_LABELS = {"mobilenet_v2": "MobileNetV2", "resnet50": "ResNet-50"}
MODELS = ("mobilenet_v2", "resnet50")
COLORS = {"CPU": "#4C78A8", "CUDA": "#F58518", "GPU": "#E45756", "Neural Engine": "#54A24B"}
MAC_DEVICES = {"cpuOnly": "CPU", "cpuAndGPU": "GPU", "cpuAndNeuralEngine": "Neural Engine"}


def median(values):
    return float(statistics.median(values))


def load_windows():
    grouped = defaultdict(list)
    with WINDOWS_SUMMARY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["model"], row["device"], row["precision"], int(row["batch"]))].append(row)
    points = []
    for (model, device, precision, batch), rows in grouped.items():
        energy = [float(row["energy_j_per_inference"]) for row in rows if row["energy_j_per_inference"]]
        points.append({
            "machine": "Windows workstation",
            "model": model,
            "device": "CPU" if device == "cpu" else "CUDA",
            "precision": precision,
            "batch": batch,
            "latency_ms": median(float(row["latency_median_ms"]) for row in rows),
            "p95_ms": median(float(row["latency_p95_ms"]) for row in rows),
            "throughput": median(float(row["throughput_ips"]) for row in rows),
            "energy_j": median(energy) if energy else None,
            "valid": True,
        })
    return points


def load_mac():
    points = []
    with MAC_SUMMARY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            points.append({
                "machine": "MacBook Pro M4 Pro",
                "model": row["model"],
                "device": MAC_DEVICES[row["compute_units"]],
                "precision": "fp16" if row["precision"] == "float16" else "fp32",
                "batch": int(row["batch"]),
                "latency_ms": float(row["median_ms"]),
                "p95_ms": float(row["p95_ms"]),
                "throughput": float(row["throughput_per_s"]),
                "energy_j": None,
                "valid": row["placement_matches_request"] == "True",
            })
    return points


def pick(points, **filters):
    found = []
    for point in points:
        if all(point.get(key) == value for key, value in filters.items()):
            found.append(point)
    return found


def one(points, **filters):
    found = [point for point in pick(points, **filters) if point["valid"]]
    return found[0] if found else None


def grouped_bars(axis, points, devices, models):
    positions = np.arange(len(models))
    width = 0.8 / len(devices)
    for index, device in enumerate(devices):
        values = []
        for model in models:
            point = one(points, model=model, device=device)
            values.append(point["latency_ms"] if point else np.nan)
        offset = (index - (len(devices) - 1) / 2) * width
        axis.bar(positions + offset, values, width, label=device, color=COLORS[device])
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in models])
    axis.set_ylabel("Median latency (ms)")
    axis.legend(frameon=False)


def plot_latency(windows, mac, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    grouped_bars(axes[0], pick(windows, precision="fp32", batch=1), ("CPU", "CUDA"), MODELS)
    axes[0].set_title("Windows, FP32, batch 1")
    grouped_bars(axes[1], pick(mac, precision="fp16", batch=1), ("CPU", "GPU", "Neural Engine"), MODELS)
    axes[1].set_title("M4 Pro, float16, batch 1")
    figure.suptitle("Latency inside each machine. ResNet-50 has no valid M4 GPU bar at batch 1.")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_throughput(windows, mac, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for model in MODELS:
        for device, color in (("CPU", COLORS["CPU"]), ("CUDA", COLORS["CUDA"])):
            series = sorted(pick(windows, model=model, device=device, precision="fp32"), key=lambda item: item["batch"])
            axes[0].plot(
                [item["batch"] for item in series],
                [item["throughput"] for item in series],
                marker="o",
                color=color,
                linestyle="-" if model == "mobilenet_v2" else "--",
                label=f"{MODEL_LABELS[model]} {device}",
            )
    axes[0].set_title("Windows, FP32")
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Images per second")
    axes[0].legend(frameon=False, fontsize=8)
    for model in MODELS:
        for device in ("CPU", "GPU", "Neural Engine"):
            series = sorted(
                [item for item in pick(mac, model=model, device=device, precision="fp16") if item["valid"]],
                key=lambda item: item["batch"],
            )
            if not series:
                continue
            axes[1].plot(
                [item["batch"] for item in series],
                [item["throughput"] for item in series],
                marker="o",
                color=COLORS[device],
                linestyle="-" if model == "mobilenet_v2" else "--",
                label=f"{MODEL_LABELS[model]} {device}",
            )
    axes[1].set_title("M4 Pro, float16, valid placements only")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Images per second")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Throughput stays inside each machine")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_speedup(windows, mac, path):
    series = [
        ("Windows CUDA FP32", windows, "CUDA", "fp32", COLORS["CUDA"]),
        ("M4 GPU float16", mac, "GPU", "fp16", COLORS["GPU"]),
        ("M4 Neural Engine float16", mac, "Neural Engine", "fp16", COLORS["Neural Engine"]),
    ]
    positions = np.arange(len(MODELS))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for index, (label, points, device, precision, color) in enumerate(series):
        values = []
        for model in MODELS:
            target = one(points, model=model, device=device, precision=precision, batch=1)
            cpu_device = "CPU"
            baseline = one(points, model=model, device=cpu_device, precision=precision if points is mac else "fp32", batch=1)
            if points is windows:
                baseline = one(points, model=model, device="CPU", precision="fp32", batch=1)
            if target is None or baseline is None or target["latency_ms"] == 0:
                values.append(np.nan)
            else:
                values.append(baseline["latency_ms"] / target["latency_ms"])
        axis.bar(positions + (index - 1) * width, values, width, label=label, color=color)
    axis.axhline(1, color="#666666", linewidth=0.8)
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in MODELS])
    axis.set_ylabel("Speedup versus the CPU on the same machine")
    axis.set_title("Batch 1. Each bar uses that machine's own CPU at the same precision.")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_energy(windows, path):
    selected = [point for point in windows if point["device"] == "CUDA" and point["energy_j"] is not None]
    labels = []
    values = []
    colors = []
    for precision in ("fp32", "fp16"):
        for model in MODELS:
            for batch in (1, 8, 32):
                point = one(selected, model=model, precision=precision, batch=batch)
                if point is None:
                    continue
                labels.append(f"{MODEL_LABELS[model]}\n{precision} b{batch}")
                values.append(point["energy_j"] / batch)
                colors.append(COLORS["CUDA"] if precision == "fp32" else "#F2C14E")
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.bar(range(len(values)), values, color=colors)
    axis.set_xticks(range(len(labels)), labels, fontsize=8)
    axis.set_ylabel("Joules per image")
    axis.set_title("RTX 4070 Ti board energy. CPU and M4 Pro are absent.")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def write_table(windows, mac, path):
    fields = ["machine", "model", "device", "precision", "batch", "valid", "latency_ms", "p95_ms", "throughput", "energy_j"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for point in windows + mac:
            writer.writerow(point)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    mac = load_mac()
    plot_latency(windows, mac, OUT / "latency_within_machine.png")
    plot_throughput(windows, mac, OUT / "throughput_within_machine.png")
    plot_speedup(windows, mac, OUT / "speedup_vs_same_machine_cpu.png")
    plot_energy(windows, OUT / "energy_cuda_per_image.png")
    write_table(windows, mac, OUT / "points.csv")
    print(OUT)


if __name__ == "__main__":
    main()
