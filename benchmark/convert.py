from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def _deployment_target(coremltools: object) -> object:
    target = coremltools.target
    version = platform.mac_ver()[0]
    major = int(version.split(".")[0]) if version else 0
    if major >= 15 and hasattr(target, "macOS15"):
        return target.macOS15
    return target.macOS14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_onnx(root: Path, model_name: str) -> tuple[str | None, str | None]:
    index_path = root / "models" / "index.json"
    if not index_path.exists():
        return None, None
    spec = json.loads(index_path.read_text(encoding="utf-8")).get("models", {}).get(model_name, {})
    fp32 = spec.get("fp32", {})
    weights = spec.get("weights")
    digest = fp32.get("sha256")
    return (str(weights) if weights else None, str(digest) if digest else None)


def ensure_onnx(root: Path, model_name: str) -> dict[str, object]:
    from bench.export_models import export_fp32

    destination = root / "models" / f"{model_name}-fp32.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    recorded_weights, windows_sha256 = _recorded_onnx(root, model_name)
    if not destination.exists():
        exported_weights = export_fp32(model_name, destination)
        recorded_weights = recorded_weights or str(exported_weights)
    digest = _sha256(destination)
    return {
        "onnx_path": str(destination.relative_to(root)),
        "onnx_sha256": digest,
        "weights": recorded_weights,
        "windows_onnx_sha256": windows_sha256,
        "matches_windows_onnx_file": bool(windows_sha256) and digest == windows_sha256,
    }


def _module_from_onnx(path: Path, batch: int) -> object:
    import torch
    from onnx2torch import convert as onnx_to_torch

    module = onnx_to_torch(str(path)).to("cpu").eval()
    example = torch.zeros(batch, 3, 224, 224)
    with torch.inference_mode():
        return torch.jit.trace(module, example)


def _compile(package_path: Path, compiled_path: Path) -> Path:
    import shutil

    import coremltools as ct

    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    produced = Path(ct.models.utils.compile_model(str(package_path), destination_path=str(compiled_path)))
    if produced.resolve() != compiled_path.resolve() and produced.exists() and not compiled_path.exists():
        produced.rename(compiled_path)
    if not compiled_path.exists():
        raise RuntimeError(f"compiled model missing at {compiled_path}")
    return compiled_path


def convert_model(root: Path, model_name: str, precision: str, batch: int) -> Path:
    import coremltools as ct

    if precision not in {"float16", "float32"}:
        raise ValueError(f"unknown precision: {precision}")
    if model_name not in {"mobilenet_v2", "resnet50"}:
        raise ValueError(f"unknown model: {model_name}")

    cache = root / "build" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}-{precision}-b{batch}"
    package_path = cache / f"{stem}.mlpackage"
    compiled_path = cache / f"{stem}.mlmodelc"
    meta_path = cache / f"{stem}.json"
    source = ensure_onnx(root, model_name)
    meta = {
        "batch": batch,
        "converter": "onnx2torch",
        "coremltools": ct.__version__,
        "matches_windows_onnx_file": source["matches_windows_onnx_file"],
        "model": model_name,
        "onnx_sha256": source["onnx_sha256"],
        "precision": precision,
        "source": "onnx-fp32",
        "weights": source["weights"],
        "windows_onnx_sha256": source["windows_onnx_sha256"],
    }
    if compiled_path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved == meta:
            return compiled_path

    traced = _module_from_onnx(root / str(source["onnx_path"]), batch)
    precision_constant = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(batch, 3, 224, 224), dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=precision_constant,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=_deployment_target(ct),
        skip_model_load=True,
    )
    if package_path.exists():
        import shutil

        shutil.rmtree(package_path)
    converted.save(str(package_path))
    _compile(package_path, compiled_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return compiled_path
