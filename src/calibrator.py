"""INT8 entropy calibrator for TensorRT post-training quantization.

The calibrator feeds batches of representative images through the network
during engine build, so TensorRT can compute optimal quantization scales
for each tensor. Using random data here would destroy accuracy — feed
real ImageNet samples drawn from the same distribution as inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from .preprocess import preprocess_image


class ImageNetCalibrator(trt.IInt8EntropyCalibrator2):
    """Streams batches of preprocessed ImageNet images to TensorRT."""

    def __init__(
        self,
        image_dir: Path | str,
        cache_path: Path | str,
        batch_size: int = 8,
        max_batches: int = 64,
        input_shape: tuple[int, int, int] = (3, 224, 224),
    ):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.cache_path = Path(cache_path)

        image_paths = sorted(Path(image_dir).glob("*.[jJ][pP][gG]")) + sorted(
            Path(image_dir).glob("*.[jJ][pP][eE][gG]")
        ) + sorted(Path(image_dir).glob("*.[pP][nN][gG]"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir!r} for calibration")

        max_imgs = batch_size * max_batches
        self.image_paths = image_paths[:max_imgs]
        self._iter = self._batch_iterator()

        nbytes = batch_size * int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(nbytes)

        print(
            f"[calibrator] {len(self.image_paths)} images, "
            f"batch_size={batch_size}, max_batches={max_batches}"
        )

    def _batch_iterator(self) -> Iterator[np.ndarray]:
        buf = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        i = 0
        for path in self.image_paths:
            buf[i] = preprocess_image(path)[0]
            i += 1
            if i == self.batch_size:
                yield np.ascontiguousarray(buf)
                i = 0
        # Drop the last partial batch — TensorRT expects fixed batch size.

    # ---- IInt8EntropyCalibrator2 interface ----

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        try:
            batch = next(self._iter)
        except StopIteration:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.exists():
            print(f"[calibrator] using cached calibration: {self.cache_path}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        print(f"[calibrator] wrote calibration cache: {self.cache_path}")
