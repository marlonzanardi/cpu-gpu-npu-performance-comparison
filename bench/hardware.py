import importlib.metadata
import os
import platform
import subprocess

from bench.runtime import c_locale_env, hidden_process_flags


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi_csv(query):
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        env=c_locale_env(),
        creationflags=hidden_process_flags(),
    )
    return completed.stdout.strip().splitlines()[0]


def parse_metric(token):
    token = token.strip()
    if token in {"[N/A]", "N/A", "[Not Supported]", ""}:
        return None
    if "," in token and "." not in token:
        token = token.replace(",", ".")
    return float(token)


def collect_hardware(ort):
    gpu_fields = [part.strip() for part in nvidia_smi_csv("name,memory.total,driver_version").split(",")]
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "gpu_name": gpu_fields[0],
        "gpu_memory_mib": parse_metric(gpu_fields[1]),
        "nvidia_driver": gpu_fields[2],
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
        "onnx": package_version("onnx"),
        "onnxruntime": package_version("onnxruntime-gpu") or package_version("onnxruntime"),
        "available_providers": list(ort.get_available_providers()),
        "excluded_devices": ["amd_radeon_igpu"],
        "npu": "unavailable",
    }
