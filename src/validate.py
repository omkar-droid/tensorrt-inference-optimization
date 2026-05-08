"""Top-1 / Top-5 accuracy validation for all engines on an ImageNet val subset.

The 5K-image subset is fetched via HuggingFace `datasets`. Each engine is
evaluated identically; the headline check is that FP16 stays within 0.1%
and INT8 within 1.0% of the FP32 baseline.

Usage:
    python -m src.validate --engine-dir engines --output results/accuracy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import models
from tqdm import tqdm

from .infer import TRTInference
from .preprocess import preprocess_image


MODELS = ("resnet18", "resnet34")
BACKENDS = ("torch_fp32", "trt_fp32", "trt_fp16", "trt_int8")


def _load_val_subset(num_samples: int, cache_dir: Path) -> list[tuple[Path, int]]:
    """Download a 5K-image ImageNet val subset and return (image_path, label) pairs."""
    from datasets import load_dataset

    ds = load_dataset("imagenet-1k", split="validation", streaming=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, int]] = []
    for i, ex in enumerate(tqdm(ds, total=num_samples, desc="downloading val subset")):
        if i >= num_samples:
            break
        img_path = cache_dir / f"img_{i:05d}.jpg"
        if not img_path.exists():
            ex["image"].convert("RGB").save(img_path, "JPEG")
        pairs.append((img_path, int(ex["label"])))
    return pairs


def _eval_torch(model_name: str, samples: list[tuple[Path, int]]) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = {"resnet18": models.resnet18, "resnet34": models.resnet34}[model_name]
    weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
    }[model_name]
    model = model_cls(weights=weights).to(device).eval()

    top1 = top5 = 0
    with torch.inference_mode():
        for path, label in tqdm(samples, desc=f"torch_fp32 {model_name}"):
            x = torch.from_numpy(preprocess_image(path)).to(device)
            logits = model(x).cpu().numpy()[0]
            top1 += int(np.argmax(logits) == label)
            top5 += int(label in np.argsort(logits)[-5:])
    n = len(samples)
    return {"top1": top1 / n, "top5": top5 / n, "n": n}


def _eval_trt(engine_path: Path, samples: list[tuple[Path, int]]) -> dict:
    runner = TRTInference(engine_path, batch_size=1)
    top1 = top5 = 0
    for path, label in tqdm(samples, desc=engine_path.stem):
        x = preprocess_image(path)
        logits = runner.infer(x)[0]
        top1 += int(np.argmax(logits) == label)
        top5 += int(label in np.argsort(logits)[-5:])
    n = len(samples)
    return {"top1": top1 / n, "top5": top5 / n, "n": n}


def run_all(engine_dir: Path, output: Path, num_samples: int, val_cache: Path) -> None:
    samples = _load_val_subset(num_samples, val_cache)
    print(f"[ok] loaded {len(samples)} val samples")

    results: dict = {"num_samples": len(samples), "runs": []}

    for model_name in MODELS:
        for backend in BACKENDS:
            try:
                if backend == "torch_fp32":
                    r = _eval_torch(model_name, samples)
                else:
                    precision = backend.split("_")[1]
                    engine = engine_dir / f"{model_name}_{precision}.engine"
                    r = _eval_trt(engine, samples)
                r.update(model=model_name, backend=backend)
                results["runs"].append(r)
                print(f"  {model_name:9s} {backend:11s}  top1={r['top1']:.4f}  top5={r['top5']:.4f}")
            except Exception as e:
                print(f"  {model_name:9s} {backend:11s}  SKIPPED: {e}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] accuracy -> {output}")
    _print_drops(results)


def _print_drops(results: dict) -> None:
    """Show absolute top-1 drop vs torch_fp32 baseline for each model."""
    by_key = {(r["model"], r["backend"]): r for r in results["runs"]}
    for model in MODELS:
        try:
            base = by_key[(model, "torch_fp32")]["top1"]
        except KeyError:
            continue
        for backend in ("trt_fp32", "trt_fp16", "trt_int8"):
            try:
                top1 = by_key[(model, backend)]["top1"]
            except KeyError:
                continue
            drop_pct = (base - top1) * 100
            print(f"[drop] {model} {backend}: {drop_pct:+.3f} pp vs torch_fp32 (baseline={base:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, default=Path("engines"))
    parser.add_argument("--output", type=Path, default=Path("results/accuracy.json"))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--val-cache", type=Path, default=Path("data/val_subset"))
    args = parser.parse_args()
    run_all(args.engine_dir, args.output, args.num_samples, args.val_cache)


if __name__ == "__main__":
    main()
