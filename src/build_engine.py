"""Build a TensorRT engine from an ONNX model.

Supports FP32 (baseline), FP16 (Tensor Cores), and INT8 (post-training
quantization with entropy calibration).

Usage:
    python -m src.build_engine --onnx engines/resnet18.onnx --precision fp16 \\
        --output engines/resnet18_fp16.engine
    python -m src.build_engine --onnx engines/resnet18.onnx --precision int8 \\
        --output engines/resnet18_int8.engine --calib-dir data/calibration
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
import tensorrt as trt

from .calibrator import ImageNetCalibrator


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
PRECISIONS = ("fp32", "fp16", "int8")


def build_engine(
    onnx_path: Path,
    output_path: Path,
    precision: str,
    max_batch: int = 32,
    workspace_gb: int = 2,
    calib_dir: Path | None = None,
    calib_cache: Path | None = None,
) -> None:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "int8" and calib_dir is None:
        raise ValueError("--calib-dir is required when precision=int8")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Dynamic batch profile: opt for batch=8 (typical inference) but allow 1..max_batch.
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(max_batch, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[warn] platform reports no fast FP16 support — Tensor Cores may not engage")
        config.set_flag(trt.BuilderFlag.FP16)

    if precision == "int8":
        if not builder.platform_has_fast_int8:
            print("[warn] platform reports no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        # Calibration uses a fixed batch from the optimization profile.
        config.set_calibration_profile(profile)
        cache_path = calib_cache or output_path.parent / "calibration.cache"
        config.int8_calibrator = ImageNetCalibrator(
            image_dir=calib_dir,
            cache_path=cache_path,
            batch_size=8,
            max_batches=64,
        )

    print(f"[build] {onnx_path.name} -> {output_path.name} ({precision})")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed (build_serialized_network returned None)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[ok] wrote {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=PRECISIONS, default="fp16")
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--calib-cache", type=Path, default=None)
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        max_batch=args.max_batch,
        workspace_gb=args.workspace_gb,
        calib_dir=args.calib_dir,
        calib_cache=args.calib_cache,
    )


if __name__ == "__main__":
    main()
