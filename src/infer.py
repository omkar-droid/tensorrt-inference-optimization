"""TensorRT inference runner with async H2D/D2H transfers on a CUDA stream.

Works for any precision (FP32/FP16/INT8) — the engine file is opaque at runtime.

Usage:
    python -m src.infer --engine engines/resnet18_fp16.engine --image cat.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

from .preprocess import preprocess_image


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference:
    """Loads a serialized TRT engine and runs inference via async CUDA transfers."""

    def __init__(self, engine_path: Path | str, batch_size: int = 1):
        self.batch_size = batch_size
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Bind input shape (engine has a dynamic batch axis from the build profile).
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.context.set_input_shape(self.input_name, (batch_size, 3, 224, 224))

        in_shape = tuple(self.context.get_tensor_shape(self.input_name))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))

        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # Pinned host buffers + device buffers for async transfers.
        self.h_input = cuda.pagelocked_empty(int(np.prod(in_shape)), dtype=in_dtype)
        self.h_output = cuda.pagelocked_empty(int(np.prod(out_shape)), dtype=out_dtype)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.stream = cuda.Stream()

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Run a single inference. `batch` shape must match the configured batch size."""
        if batch.shape != self.in_shape:
            raise ValueError(f"expected input shape {self.in_shape}, got {batch.shape}")

        np.copyto(self.h_input.reshape(self.in_shape), batch.astype(self.h_input.dtype))
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.reshape(self.out_shape).copy()


def _load_imagenet_labels() -> list[str]:
    """Lazy import: only needed for the CLI demo."""
    from torchvision.models import ResNet18_Weights
    return ResNet18_Weights.DEFAULT.meta["categories"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    runner = TRTInference(args.engine, batch_size=1)
    batch = preprocess_image(args.image)
    logits = runner.infer(batch)[0]

    labels = _load_imagenet_labels()
    top = np.argsort(logits)[-args.top_k:][::-1]
    print(f"\nTop-{args.top_k} predictions for {args.image}:")
    for idx in top:
        print(f"  {labels[idx]:<40s}  logit={logits[idx]:+.3f}")


if __name__ == "__main__":
    main()
