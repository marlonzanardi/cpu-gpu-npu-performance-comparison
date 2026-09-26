import subprocess
import threading
import time

from bench.hardware import parse_metric
from bench.runtime import c_locale_env, hidden_process_flags


class NvidiaPowerSampler:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="nvidia-power", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self):
        while not self._stop.is_set():
            sample = self._read_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(0.2)

    def _read_once(self):
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=c_locale_env(),
            creationflags=hidden_process_flags(),
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        parts = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 3:
            return None
        return {
            "t_unix": time.time(),
            "power_w": parse_metric(parts[0]),
            "gpu_util_pct": parse_metric(parts[1]),
            "memory_used_mib": parse_metric(parts[2]),
        }


def summarize_power(samples):
    powers = [sample["power_w"] for sample in samples if sample["power_w"] is not None]
    utils = [sample["gpu_util_pct"] for sample in samples if sample["gpu_util_pct"] is not None]
    memory = [sample["memory_used_mib"] for sample in samples if sample["memory_used_mib"] is not None]
    return {
        "power_mean_w": sum(powers) / len(powers) if powers else None,
        "gpu_util_mean_pct": sum(utils) / len(utils) if utils else None,
        "gpu_memory_used_mib_mean": sum(memory) / len(memory) if memory else None,
        "power_samples": len(powers),
    }
