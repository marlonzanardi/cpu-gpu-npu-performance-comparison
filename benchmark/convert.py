from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _deployment_target(coremltools: object) -> object:
    target = coremltools.target
    if hasattr(target, "macOS15"):
        return target.macOS15
    return target.macOS14


WEIGHTS = {
    "mobilenet_v2": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
}


def _build_torch_model(model_name: str, batch: int) -> object:
    import torch
    from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, mobilenet_v2, resnet50

    if model_name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"unknown model: {model_name}")
    model = model.to("cpu").eval()
    example = torch.rand(batch, 3, 224, 224)
    with torch.inference_mode():
        return torch.jit.trace(model, example)


def _compile(package_path: Path, compiled_path: Path) -> Path:
    import coremltools as ct

    if compiled_path.exists():
        import shutil

        shutil.rmtree(compiled_path)
    produced = Path(ct.models.utils.compile_model(str(package_path), destination_path=str(compiled_path)))
    if produced.resolve() != compiled_path.resolve() and produced.exists():
        if compiled_path.exists():
            return compiled_path
        produced.rename(compiled_path)
    if not compiled_path.exists():
        raise RuntimeError(f"compiled model missing at {compiled_path}")
    return compiled_path


def convert_model(root: Path, model_name: str, precision: str, batch: int) -> Path:
    import coremltools as ct
    import torch

    cache = root / "build" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}-{precision}-b{batch}"
    package_path = cache / f"{stem}.mlpackage"
    compiled_path = cache / f"{stem}.mlmodelc"
    meta_path = cache / f"{stem}.json"
    if model_name not in WEIGHTS:
        raise ValueError(f"unknown model: {model_name}")
    meta = {
        "batch": batch,
        "coremltools": ct.__version__,
        "model": model_name,
        "precision": precision,
        "torch": torch.__version__,
        "weights": WEIGHTS[model_name],
    }
    if compiled_path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved == meta:
            return compiled_path

    traced = _build_torch_model(model_name, batch)
    precision_constant = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    if precision not in {"float16", "float32"}:
        raise ValueError(f"unknown precision: {precision}")
    example_shape = (batch, 3, 224, 224)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example_shape, dtype=np.float32)],
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
