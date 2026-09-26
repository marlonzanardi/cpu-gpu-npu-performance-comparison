import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_samples(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _median(values):
    return float(np.median(values))


def _float(row, key):
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def plot_latency(samples, figure_path):
    selected = [row for row in samples if row["precision"] == "fp32" and row["batch"] == 1]
    models = list(dict.fromkeys(row["model"] for row in selected))
    devices = ["cpu", "cuda"]
    series = []
    labels = []
    for model in models:
        for device in devices:
            values = [
                row["latency_ms"]
                for row in selected
                if row["model"] == model and row["device"] == device
            ]
            if values:
                series.append(values)
                labels.append(f"{model}\n{device}")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(series, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Batch 1, FP32, TF32 off on CUDA")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def plot_throughput(summary_rows, figure_path):
    selected = [row for row in summary_rows if row["precision"] == "fp32"]
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["model"], row["device"], int(row["batch"]))].append(float(row["throughput_ips"]))
    figure, axis = plt.subplots(figsize=(9, 5))
    for model, device in sorted({(model, device) for model, device, _batch in grouped}):
        batches = sorted({batch for item_model, item_device, batch in grouped if item_model == model and item_device == device})
        medians = [_median(grouped[(model, device, batch)]) for batch in batches]
        axis.plot(batches, medians, marker="o", label=f"{model} {device}")
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Throughput (images/s)")
    axis.set_title("FP32 throughput, median across sessions")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def plot_energy(summary_rows, figure_path):
    selected = []
    for row in summary_rows:
        energy = _float(row, "energy_j_per_inference")
        if row["device"] == "cuda" and energy is not None:
            selected.append(row)
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["model"], row["precision"], int(row["batch"]))].append(_float(row, "energy_j_per_inference"))
    labels = []
    values = []
    for key in sorted(grouped):
        labels.append(f"{key[0]}\n{key[1]} b{key[2]}")
        values.append(_median(grouped[key]))
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Joules per inference")
    axis.set_title("CUDA board power integrated over the measured window")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def plot_latency_energy(summary_rows, figure_path):
    grouped = defaultdict(lambda: {"latency": [], "energy": []})
    for row in summary_rows:
        energy = _float(row, "energy_j_per_inference")
        if row["device"] != "cuda" or energy is None:
            continue
        key = (row["model"], row["precision"], int(row["batch"]))
        grouped[key]["latency"].append(float(row["latency_median_ms"]))
        grouped[key]["energy"].append(energy)
    figure, axis = plt.subplots(figsize=(8, 5))
    for key, payload in sorted(grouped.items()):
        axis.scatter(_median(payload["latency"]), _median(payload["energy"]))
        axis.annotate(f"{key[0]} {key[1]} b{key[2]}", (_median(payload["latency"]), _median(payload["energy"])))
    axis.set_xlabel("Median latency (ms)")
    axis.set_ylabel("Joules per inference")
    axis.set_title("CUDA latency against board energy")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def write_figures(run_dir):
    run_dir = Path(run_dir)
    summary_rows = read_summary(run_dir / "summary.csv")
    samples = read_samples(run_dir / "samples.jsonl")
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(samples, figure_dir / "latency_batch1_fp32.png")
    plot_throughput(summary_rows, figure_dir / "throughput_fp32.png")
    if any(_float(row, "energy_j_per_inference") is not None for row in summary_rows):
        plot_energy(summary_rows, figure_dir / "energy_cuda.png")
        plot_latency_energy(summary_rows, figure_dir / "latency_vs_energy_cuda.png")
