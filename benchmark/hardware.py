from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def _profiler(data_type: str) -> list[dict[str, object]]:
    payload = json.loads(_run(["system_profiler", data_type, "-json"]))
    entries = payload[data_type]
    if not isinstance(entries, list):
        raise RuntimeError(f"unexpected {data_type} payload")
    return entries


def _processor_counts(raw: str) -> dict[str, int] | None:
    match = re.fullmatch(r"proc (\d+):(\d+):(\d+)", raw.strip())
    if match is None:
        return None
    total, performance, efficiency = (int(part) for part in match.groups())
    return {
        "cpu_cores": total,
        "performance_cores": performance,
        "efficiency_cores": efficiency,
    }


def _power_source() -> str:
    first = _run(["pmset", "-g", "batt"]).splitlines()[0]
    if "AC Power" in first:
        return "ac"
    if "Battery Power" in first:
        return "battery"
    return "unknown"


def _power_mode() -> str | None:
    names = {"0": "automatic", "1": "low", "2": "high"}
    for line in _run(["pmset", "-g"]).splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        if tokens[0] == "lowpowermode":
            return "low" if tokens[-1] == "1" else "normal"
        if tokens[0] == "powermode":
            return names.get(tokens[-1], tokens[-1])
    return None


def collect_machine() -> dict[str, object]:
    hardware = _profiler("SPHardwareDataType")[0]
    displays = _profiler("SPDisplaysDataType")
    chip = str(hardware.get("chip_type", "unknown"))
    processor_raw = str(hardware.get("number_processors", ""))
    gpu_cores = None
    metal = None
    if displays:
        gpu = displays[0]
        if "sppci_cores" in gpu:
            gpu_cores = int(str(gpu["sppci_cores"]))
        metal = gpu.get("spdisplays_mtlgpufamilysupport")
    versions = {
        "product": _run(["sw_vers", "-productName"]).strip(),
        "version": _run(["sw_vers", "-productVersion"]).strip(),
        "build": _run(["sw_vers", "-buildVersion"]).strip(),
    }
    power_mode = _power_mode()
    return {
        "machine_name": hardware.get("machine_name"),
        "model_identifier": hardware.get("machine_model"),
        "model_number": hardware.get("model_number"),
        "chip": chip,
        "memory": hardware.get("physical_memory"),
        "cpu": _processor_counts(processor_raw),
        "cpu_raw": processor_raw,
        "gpu_cores": gpu_cores,
        "metal": metal,
        "architecture": platform.machine(),
        "os": versions,
        "power_source": _power_source(),
        "power_mode": power_mode,
        "low_power_mode": None if power_mode is None else power_mode == "low",
    }


def machine_slug(machine: dict[str, object]) -> str:
    chip = str(machine["chip"]).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", chip).strip("-")
    return slug or "unknown-machine"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
