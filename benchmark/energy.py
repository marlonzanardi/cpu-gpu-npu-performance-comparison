from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


SAMPLE_HEADER = re.compile(
    r"\*\*\* Sampled system activity \((.+?)\) \(([0-9.]+)ms elapsed\) \*\*\*"
)
CPU_POWER = re.compile(r"^CPU Power:\s+([0-9.]+) mW", re.MULTILINE)
GPU_POWER = re.compile(r"^GPU Power:\s+([0-9.]+) mW", re.MULTILINE)
ANE_POWER = re.compile(r"^ANE Power:\s+([0-9.]+) mW", re.MULTILINE)
STAMP_FORMAT = "%a %b %d %H:%M:%S %Y %z"


def parse_powermetrics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    samples: list[dict[str, object]] = []
    for part in re.split(r"(?=\*\*\* Sampled system activity )", text):
        header = SAMPLE_HEADER.search(part)
        if header is None:
            continue
        cpu = CPU_POWER.search(part)
        gpu = GPU_POWER.search(part)
        ane = ANE_POWER.search(part)
        if cpu is None or gpu is None or ane is None:
            continue
        samples.append(
            {
                "time": datetime.strptime(header.group(1), STAMP_FORMAT),
                "elapsed_ms": float(header.group(2)),
                "cpu_mw": float(cpu.group(1)),
                "gpu_mw": float(gpu.group(1)),
                "ane_mw": float(ane.group(1)),
            }
        )
    return samples


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def integrate_window(
    samples: list[dict[str, object]],
    start_unix_ns: int,
    end_unix_ns: int,
    inferences: int,
) -> dict[str, object] | None:
    if inferences < 1 or end_unix_ns <= start_unix_ns:
        return None
    start = datetime.fromtimestamp(start_unix_ns / 1_000_000_000, tz=timezone.utc)
    end = datetime.fromtimestamp(end_unix_ns / 1_000_000_000, tz=timezone.utc)
    chosen = [sample for sample in samples if start <= sample["time"] <= end]
    if len(chosen) < 2:
        return None
    cpu = _mean([float(sample["cpu_mw"]) for sample in chosen])
    gpu = _mean([float(sample["gpu_mw"]) for sample in chosen])
    ane = _mean([float(sample["ane_mw"]) for sample in chosen])
    if cpu is None or gpu is None or ane is None:
        return None
    duration_s = (end_unix_ns - start_unix_ns) / 1_000_000_000
    joules = (cpu + gpu + ane) / 1000 * duration_s
    return {
        "duration_s": duration_s,
        "power_samples": len(chosen),
        "cpu_mw": cpu,
        "gpu_mw": gpu,
        "ane_mw": ane,
        "energy_j": joules / inferences,
    }
