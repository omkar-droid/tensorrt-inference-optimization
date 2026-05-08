"""Latency and throughput benchmarks across PyTorch and TensorRT backends.

Measures P50/P95/P99 latency and throughput (img/s) for each combination of:
  - backend: torch_fp32, torch_fp16, trt_fp32, trt_fp16, trt_int8
  - model:   resnet18, resnet34
  - batch:   1, 8, 32

Uses CUDA events for GPU timing — `time.time()` is unreliable for async kernels.

Usage:
    python -m src.benchmark --output results/benchmarks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import torch
from torchvision import models

from .infer import TRTInference


WARMUP_ITERS = 50
TIMED_ITERS = 200
BATCH_SIZES = (1, 8, 32)
MODELS = ("resnet18", "resnet34")


def _percentiles(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }


def _time_callable(run: Callable[[], None], device: torch.device) -> list[float]:
    """Run `run` for warmup + timed iters, return per-iter latency in ms."""
    if device.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        for _ in range(WARMUP_ITERS):
            run()
        torch.cuda.synchronize()

        latencies: list[float] = []
        for _ in range(TIMED_ITERS):
            start_evt.record()
            run()
            end_evt.record()
            torch.cuda.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt))
        return latencies

    # CPU fallback (unlikely path; included for completeness).
    import time
    for _ in range(WARMUP_ITERS):
        run()
    latencies = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        run()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


# ---------- PyTorch backends ----------

def bench_torch(model_name: str, batch: int, device: torch.device, fp16: bool) -> dict:
    model_cls = {"resnet18": models.resnet18, "resnet34": models.resnet34}[model_name]
    model = model_cls(weights=None).to(device).eval()
    if fp16:
        model = model.half()
    dtype = torch.float16 if fp16 else torch.float32
    x = torch.randn(batch, 3, 224, 224, device=device, dtype=dtype)

    @torch.inference_mode()
    def run() -> None:
        model(x)

    latencies = _time_callable(run, device)
    return _summarize(latencies, batch)


# ---------- TensorRT backend ----------

def bench_trt(engine_path: Path, batch: int) -> dict:
    runner = TRTInference(engine_path, batch_size=batch)
    x = np.random.randn(*runner.in_shape).astype(np.float32)

    # Warmup
    for _ in range(WARMUP_ITERS):
        runner.infer(x)

    start_evt = cuda.Event()
    end_evt = cuda.Event()
    latencies: list[float] = []
    for _ in range(TIMED_ITERS):
        start_evt.record(runner.stream)
        runner.infer(x)
        end_evt.record(runner.stream)
        end_evt.synchronize()
        latencies.append(start_evt.time_till(end_evt))
    return _summarize(latencies, batch)


def _summarize(latencies_ms: list[float], batch: int) -> dict:
    pct = _percentiles(latencies_ms)
    pct["throughput_imgs_per_sec"] = batch * 1000.0 / pct["p50_ms"]
    pct["batch"] = batch
    return pct


def run_all(engine_dir: Path, output: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("benchmark requires CUDA")

    results: dict = {"device": torch.cuda.get_device_name(0), "runs": []}

    for model_name in MODELS:
        for batch in BATCH_SIZES:
            print(f"\n=== {model_name}  batch={batch} ===")

            for backend, fn in [
                ("torch_fp32", lambda: bench_torch(model_name, batch, device, fp16=False)),
                ("torch_fp16", lambda: bench_torch(model_name, batch, device, fp16=True)),
                ("trt_fp32",   lambda: bench_trt(engine_dir / f"{model_name}_fp32.engine", batch)),
                ("trt_fp16",   lambda: bench_trt(engine_dir / f"{model_name}_fp16.engine", batch)),
                ("trt_int8",   lambda: bench_trt(engine_dir / f"{model_name}_int8.engine", batch)),
            ]:
                try:
                    r = fn()
                    r.update(model=model_name, backend=backend)
                    results["runs"].append(r)
                    print(
                        f"  {backend:<12s}  p50={r['p50_ms']:6.2f}ms  "
                        f"p95={r['p95_ms']:6.2f}ms  "
                        f"throughput={r['throughput_imgs_per_sec']:8.1f} img/s"
                    )
                except Exception as e:
                    print(f"  {backend:<12s}  SKIPPED: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] benchmarks -> {output}")
    _print_headline(results)


def _print_headline(results: dict) -> None:
    """Compute the headline 'X% latency reduction' number for the README."""
    by_key = {(r["model"], r["backend"], r["batch"]): r for r in results["runs"]}
    for model in MODELS:
        try:
            fp32 = by_key[(model, "trt_fp32", 1)]["p50_ms"]
            fp16 = by_key[(model, "trt_fp16", 1)]["p50_ms"]
            reduction = (fp32 - fp16) / fp32 * 100
            print(f"[headline] {model}: TRT FP16 vs FP32 latency reduction = {reduction:.1f}%")
        except KeyError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, default=Path("engines"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks.json"))
    args = parser.parse_args()
    run_all(args.engine_dir, args.output)


if __name__ == "__main__":
    main()
