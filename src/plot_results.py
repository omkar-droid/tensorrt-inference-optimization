"""Generate the latency and accuracy charts shown in the README.

Reads results/benchmarks.json and results/accuracy.json, writes PNG charts
to results/. Designed to be regenerated whenever the JSON changes.

Usage:
    python -m src.plot_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BACKEND_ORDER = ["torch_fp32", "torch_fp16", "trt_fp32", "trt_fp16", "trt_int8"]
BACKEND_LABELS = {
    "torch_fp32": "PyTorch FP32",
    "torch_fp16": "PyTorch FP16",
    "trt_fp32": "TRT FP32",
    "trt_fp16": "TRT FP16",
    "trt_int8": "TRT INT8",
}
COLORS = {
    "torch_fp32": "#888888",
    "torch_fp16": "#bbbbbb",
    "trt_fp32": "#1f77b4",
    "trt_fp16": "#2ca02c",
    "trt_int8": "#d62728",
}


def plot_latency(bench_path: Path, output: Path) -> None:
    data = json.loads(bench_path.read_text())
    by_key = {(r["model"], r["backend"], r["batch"]): r["p50_ms"] for r in data["runs"]}
    models = sorted({r["model"] for r in data["runs"]})
    batches = sorted({r["batch"] for r in data["runs"]})

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), sharey=False)
    if len(models) == 1:
        axes = [axes]

    width = 0.15
    for ax, model in zip(axes, models):
        x = np.arange(len(batches))
        for i, backend in enumerate(BACKEND_ORDER):
            vals = [by_key.get((model, backend, b), np.nan) for b in batches]
            offset = (i - len(BACKEND_ORDER) / 2) * width + width / 2
            ax.bar(x + offset, vals, width, label=BACKEND_LABELS[backend], color=COLORS[backend])
        ax.set_xticks(x)
        ax.set_xticklabels([f"bs={b}" for b in batches])
        ax.set_ylabel("P50 latency (ms)")
        ax.set_title(model)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[-1].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Inference latency by backend ({data.get('device', 'GPU')})")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[ok] {output}")


def plot_accuracy(acc_path: Path, output: Path) -> None:
    data = json.loads(acc_path.read_text())
    by_key = {(r["model"], r["backend"]): r["top1"] for r in data["runs"]}
    models = sorted({r["model"] for r in data["runs"]})
    backends = [b for b in BACKEND_ORDER if any((m, b) in by_key for m in models)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.18
    x = np.arange(len(models))
    for i, backend in enumerate(backends):
        vals = [by_key.get((m, backend), np.nan) for m in models]
        offset = (i - len(backends) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=BACKEND_LABELS[backend], color=COLORS[backend])

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0.6, 0.78)
    ax.set_title(f"Top-1 accuracy on ImageNet val (n={data.get('num_samples', '?')})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[ok] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=Path("results/benchmarks.json"))
    parser.add_argument("--accuracy", type=Path, default=Path("results/accuracy.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.bench.exists():
        plot_latency(args.bench, args.out_dir / "latency_comparison.png")
    else:
        print(f"[skip] {args.bench} not found")

    if args.accuracy.exists():
        plot_accuracy(args.accuracy, args.out_dir / "accuracy_comparison.png")
    else:
        print(f"[skip] {args.accuracy} not found")


if __name__ == "__main__":
    main()
