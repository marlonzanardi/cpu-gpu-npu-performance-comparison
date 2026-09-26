import hashlib
import json
from pathlib import Path

import onnx
import torch
import torchvision
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_IDS = ("mobilenet_v2", "resnet50")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_fp32(model_id, path):
    weights = torchvision.models.get_model_weights(model_id).DEFAULT
    module = torchvision.models.get_model(model_id, weights=weights)
    module.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    with torch.inference_mode():
        try:
            torch.onnx.export(module, dummy, str(path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(module, dummy, str(path), **export_kwargs)
    return str(weights)


def export_fp16(src, dst):
    converted = float16.convert_float_to_float16(onnx.load(src))
    onnx.save(converted, dst)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    index = {"torch": torch.__version__, "torchvision": torchvision.__version__, "models": {}}
    for model_id in MODEL_IDS:
        fp32_path = MODEL_DIR / f"{model_id}-fp32.onnx"
        fp16_path = MODEL_DIR / f"{model_id}-fp16.onnx"
        weights_name = export_fp32(model_id, fp32_path)
        export_fp16(fp32_path, fp16_path)
        index["models"][model_id] = {
            "weights": weights_name,
            "fp32": {"path": fp32_path.name, "sha256": file_sha256(fp32_path)},
            "fp16": {"path": fp16_path.name, "sha256": file_sha256(fp16_path)},
        }
        print(f"exported {model_id}", flush=True)
    (MODEL_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
