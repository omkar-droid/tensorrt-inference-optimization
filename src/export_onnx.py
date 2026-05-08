"""Export pretrained ResNet18/34 to ONNX with a dynamic batch axis.

Usage:
    python -m src.export_onnx --model resnet18 --output engines/resnet18.onnx
    python -m src.export_onnx --model resnet34 --output engines/resnet34.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from torchvision import models


MODEL_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
}


def export(model_name: str, output_path: Path, opset: int = 17) -> None:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Choices: {list(MODEL_REGISTRY)}")

    builder, weights = MODEL_REGISTRY[model_name]
    model = builder(weights=weights).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print(f"[ok] exported {model_name} -> {output_path} (opset {opset})")

    _verify_parity(model, output_path, dummy)


def _verify_parity(torch_model: torch.nn.Module, onnx_path: Path, sample: torch.Tensor) -> None:
    """Run the ONNX graph through onnxruntime and compare against PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime not installed — skipping parity check")
        return

    with torch.no_grad():
        torch_out = torch_model(sample).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": sample.numpy()})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"[parity] max |torch - onnx| = {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-3, f"ONNX export diverged from PyTorch: {max_abs_diff}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_REGISTRY), default="resnet18")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    output = args.output or Path(f"engines/{args.model}.onnx")
    export(args.model, output, args.opset)


if __name__ == "__main__":
    main()
