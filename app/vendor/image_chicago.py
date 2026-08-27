"""Chicago-style proxy measurements recovered exclusively from HRM PNG images.

The calibration is fitted before model training using normal-motility images
from the training partition only.  It never reads pressure CSV files.  Because
the exports contain no colour bar, the physical scale is identified by anchoring
the normal cohort to conservative physiological reference targets.  The output
must therefore be reported as *image-calibrated proxies*, not measurements made
by the acquisition system.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .chicago import METRIC_NAMES, swallow_metrics


@dataclass(frozen=True)
class ImageCalibration:
    pressure_offset: float
    pressure_scale: float
    sensors: int = 24
    samples: int = 241
    colour_bins: int = 32
    max_colour_distance: float = 0.18
    source: str = "train/normal_motility"
    normal_images: int = 0
    source_digest: str = ""
    objective: float = float("nan")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "ImageCalibration":
        return cls(**json.loads(path.read_text()))


def _jet(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0),
        ],
        axis=-1,
    )


def _make_lookup(bins: int) -> tuple[np.ndarray, np.ndarray]:
    levels = np.arange(bins, dtype=np.float32) / (bins - 1)
    rgb = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(-1, 3)
    coordinates = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    palette = _jet(coordinates)
    closest = np.empty(len(rgb), dtype=np.int16)
    error = np.empty(len(rgb), dtype=np.float32)
    for start in range(0, len(rgb), 1024):
        chunk = rgb[start : start + 1024]
        distance = ((chunk[:, None] - palette[None]) ** 2).sum(-1)
        index = distance.argmin(-1)
        closest[start : start + len(chunk)] = index
        error[start : start + len(chunk)] = np.sqrt(distance[np.arange(len(chunk)), index])
    return coordinates[closest], error


_LOOKUPS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def topography_from_image(
    image: Image.Image | np.ndarray,
    *,
    sensors: int = 24,
    samples: int = 241,
    colour_bins: int = 32,
    max_colour_distance: float = 0.18,
) -> tuple[np.ndarray, float]:
    """Invert the jet-like rendering into a unitless ``sensor x time`` map."""
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {rgb.shape}")
    if colour_bins not in _LOOKUPS:
        _LOOKUPS[colour_bins] = _make_lookup(colour_bins)
    lookup, errors = _LOOKUPS[colour_bins]
    q = np.rint(rgb.astype(np.float32) / 255.0 * (colour_bins - 1)).astype(np.int32)
    q = np.clip(q, 0, colour_bins - 1)
    index = q[..., 0] * colour_bins**2 + q[..., 1] * colour_bins + q[..., 2]
    coordinate, error = lookup[index], errors[index]
    valid = error <= max_colour_distance

    height, width = coordinate.shape
    y_edges = np.linspace(0, height, sensors + 1).astype(int)
    source_x = np.linspace(0.0, 1.0, width)
    target_x = np.linspace(0.0, 1.0, samples)
    grid = np.empty((sensors, samples), dtype=np.float32)
    for row in range(sensors):
        y0, y1 = y_edges[row], max(y_edges[row + 1], y_edges[row] + 1)
        band = coordinate[y0:y1].copy()
        band[~valid[y0:y1]] = np.nan
        # Text and rectangle annotations are sparse vertically; the median is a
        # robust template match for the underlying horizontal pressure band.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile = np.nanmedian(band, axis=0)
        missing = ~np.isfinite(profile)
        if missing.all():
            profile = np.median(coordinate[y0:y1], axis=0)
        elif missing.any():
            known = np.flatnonzero(~missing)
            profile[missing] = np.interp(np.flatnonzero(missing), known, profile[known])
        grid[row] = np.interp(target_x, source_x, profile)
    return grid, float(valid.mean())


def _load_topography(path: str | Path, calibration: ImageCalibration | None = None) -> tuple[np.ndarray, float]:
    options = calibration or ImageCalibration(0.0, 1.0)
    with Image.open(path) as image:
        return topography_from_image(
            image,
            sensors=options.sensors,
            samples=options.samples,
            colour_bins=options.colour_bins,
            max_colour_distance=options.max_colour_distance,
        )


def image_swallow_metrics(
    image: str | Path | Image.Image | np.ndarray, calibration: ImageCalibration
) -> tuple[dict[str, float], float]:
    if isinstance(image, (str, Path)):
        topography, confidence = _load_topography(image, calibration)
    else:
        topography, confidence = topography_from_image(
            image,
            sensors=calibration.sensors,
            samples=calibration.samples,
            colour_bins=calibration.colour_bins,
            max_colour_distance=calibration.max_colour_distance,
        )
    pressure = calibration.pressure_offset + calibration.pressure_scale * topography
    return swallow_metrics(pressure.astype(np.float32)), confidence


def _normal_objective(rows: list[dict[str, float]]) -> float:
    """Penalty against broad normal reference anchors, not patient measurements."""
    median_irp = float(np.median([row["irp4"] for row in rows]))
    median_dci = float(np.median([row["dci"] for row in rows]))
    median_latency = float(np.median([row["distal_latency"] for row in rows]))
    failed = np.mean([row["dci"] < 100.0 for row in rows])
    hyper = np.mean([row["dci"] > 8000.0 for row in rows])
    # Targets sit near the centres of the CC normal ranges. Log DCI prevents
    # its large numerical scale from dominating the calibration.
    return float(
        ((median_irp - 10.0) / 5.0) ** 2
        + ((np.log1p(median_dci) - np.log1p(1900.0)) / 0.8) ** 2
        + ((median_latency - 6.5) / 2.0) ** 2
        + 4.0 * failed**2
        + 4.0 * hyper**2
    )


def fit_image_calibration(
    image_paths: Iterable[str | Path], *, max_images: int = 32
) -> ImageCalibration:
    """Fit a global colour scale from normal training images only."""
    all_paths = sorted(str(Path(path).resolve()) for path in image_paths)
    if not all_paths:
        raise ValueError("At least one normal training image is required for calibration")
    # Spread the reference images across the sorted cohort instead of taking
    # adjacent swallows from the first few examinations.
    positions = np.linspace(0, len(all_paths) - 1, min(max_images, len(all_paths))).astype(int)
    paths = [all_paths[position] for position in positions]
    maps = [_load_topography(path)[0] for path in paths]
    best: tuple[float, float, float] | None = None

    def search(offsets: Iterable[float], scales: Iterable[float]) -> None:
        nonlocal best
        for offset in offsets:
            for scale in scales:
                rows = [swallow_metrics((offset + scale * grid).astype(np.float32)) for grid in maps]
                objective = _normal_objective(rows)
                if best is None or objective < best[0]:
                    best = (objective, float(offset), float(scale))

    # Coarse-to-fine search: offset controls blue baseline; scale controls red.
    search(np.arange(-30.0, 6.0, 5.0), np.arange(80.0, 241.0, 20.0))
    assert best is not None
    _, coarse_offset, coarse_scale = best
    search(
        np.arange(coarse_offset - 4.0, coarse_offset + 4.1, 2.0),
        np.arange(coarse_scale - 16.0, coarse_scale + 16.1, 8.0),
    )
    assert best is not None
    digest = hashlib.sha256("\n".join(paths).encode()).hexdigest()
    return ImageCalibration(
        pressure_offset=best[1],
        pressure_scale=best[2],
        normal_images=len(paths),
        source_digest=digest,
        objective=best[0],
    )


def metric_row(image_path: str, calibration_path: str) -> tuple[list[float], float]:
    """Pickle-friendly cache-worker entry point."""
    metrics, confidence = image_swallow_metrics(image_path, ImageCalibration.load(Path(calibration_path)))
    return [metrics[name] for name in METRIC_NAMES], confidence
