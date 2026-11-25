#!/usr/bin/env python3
"""Pick one textbook swallow per diagnosis and export it for the atlas.

The choice is made from the image-calibrated Chicago proxies alone, the same
numbers the service computes at request time, so it needs no GPU and cannot be
accused of showing whatever the classifier happens to like:

1. every held-out examination of the class is measured;
2. its study-level feature vector is compared against the class centroid, and
   the closest examination wins, i.e. the most typical study of that diagnosis;
3. inside it, the swallow whose CC v4.0 contractility pattern is the one that
   defines the diagnosis is exported.

The PNGs land in `assets/reference/` (ignored by git, since they are patient data)
and the choice is recorded in `assets/reference_manifest.json`, which is
committed so the gallery can be rebuilt anywhere the corpus is mounted.

    python3 scripts/export_reference_images.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS  # noqa: E402
from app.diseases import CLASS_ORDER  # noqa: E402
from app.vendor.chicago import METRIC_NAMES, study_features, swallow_pattern  # noqa: E402
from app.vendor.image_chicago import ImageCalibration, image_swallow_metrics  # noqa: E402

# When no swallow of the class reaches the signature pattern (the image-derived
# DCI is a proxy and under-reads the extremes), the most extreme swallow in the
# metric that defines the diagnosis is the honest second choice.
FALLBACK_RANK: dict[str, tuple[str, bool]] = {
    "hypercontractile_esophagus": ("dci", True),
    "achalasia_type_ii": ("pan_time_fraction", True),
    "egj_outflow_obstruction": ("irp4", True),
    "absent_contractility": ("dci", False),
    "achalasia_type_i": ("dci", False),
    "ineffective_esophageal_motility": ("dci", False),
}

# The swallow-level pattern that *defines* each study-level diagnosis; the
# exported example is the swallow showing it.
SIGNATURE_PATTERN: dict[str, tuple[str, ...]] = {
    "normal_motility": ("normal",),
    "ineffective_esophageal_motility": ("weak", "failed", "fragmented"),
    "absent_contractility": ("failed",),
    "achalasia_type_i": ("failed",),
    "achalasia_type_ii": ("panesophageal",),
    "egj_outflow_obstruction": ("normal",),
    "hypercontractile_esophagus": ("hypercontractile",),
}


def _measure(path_and_calibration: tuple[str, str]) -> list[float]:
    path, calibration = path_and_calibration
    with Image.open(path) as handle:
        metrics, _ = image_swallow_metrics(
            np.asarray(handle.convert("RGB")), ImageCalibration.load(Path(calibration))
        )
    return [float(metrics[name]) for name in METRIC_NAMES]


def measure_split(root: Path, split: str, calibration: Path, workers: int) -> dict[str, dict]:
    """Every examination of ``split``, with one metric row per swallow."""
    exams: dict[str, dict] = {}
    jobs: list[tuple[str, str]] = []
    owners: list[tuple[str, int]] = []
    for class_name in CLASS_ORDER:
        for exam_dir in sorted((root / split / class_name).glob("exam_*")):
            swallows = sorted(exam_dir.glob("swallow_*.png"))
            if not swallows:
                continue
            key = f"{class_name}/{exam_dir.name}"
            exams[key] = {"class": class_name, "exam": exam_dir.name, "swallows": swallows, "rows": []}
            for position, png in enumerate(swallows):
                jobs.append((str(png), str(calibration)))
                owners.append((key, position))

    print(f"medindo {len(jobs)} degluticoes de {len(exams)} exames ({split})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_measure, jobs, chunksize=32))
    for (key, _position), row in zip(owners, results):
        exams[key]["rows"].append(np.asarray(row, dtype=np.float32))
    return exams


def _fallback_swallow(class_name: str, rows: list[np.ndarray]) -> int:
    """The most representative swallow when none carries the signature pattern."""
    ranked = FALLBACK_RANK.get(class_name)
    if ranked is None:
        dci = np.array([row[METRIC_NAMES.index("dci")] for row in rows])
        return int(np.argmin(np.abs(dci - np.median(dci))))
    metric, highest = ranked
    values = np.array([row[METRIC_NAMES.index(metric)] for row in rows])
    return int(np.argmax(values) if highest else np.argmin(values))


def choose(exams: dict[str, dict]) -> dict[str, dict]:
    """The most typical examination of each class, and its signature swallow."""
    features = {key: study_features([dict(zip(METRIC_NAMES, row)) for row in value["rows"]])
                for key, value in exams.items()}
    stacked = np.stack(list(features.values()))
    mean = stacked.mean(axis=0)
    deviation = stacked.std(axis=0)
    deviation[deviation < 1e-6] = 1.0

    picked: dict[str, dict] = {}
    for class_name in CLASS_ORDER:
        members = [key for key, value in exams.items() if value["class"] == class_name]
        if not members:
            continue
        standardised = {key: (features[key] - mean) / deviation for key in members}
        centroid = np.stack([standardised[key] for key in members]).mean(axis=0)
        ranked = sorted(members, key=lambda key: float(np.linalg.norm(standardised[key] - centroid)))
        wanted = SIGNATURE_PATTERN.get(class_name, ("normal",))

        chosen: tuple[str, int, str] | None = None
        fallback: tuple[str, int, str] | None = None
        for key in ranked:
            exam = exams[key]
            patterns = [swallow_pattern(dict(zip(METRIC_NAMES, row))) for row in exam["rows"]]
            position = next(
                (index for index, pattern in enumerate(patterns) if pattern in wanted), None
            )
            if position is not None:
                chosen = (key, position, patterns[position])
                break
            if fallback is None:
                fallback = (key, _fallback_swallow(class_name, exam["rows"]), "")
                fallback = (fallback[0], fallback[1], patterns[fallback[1]])

        key, position, pattern = chosen or fallback  # type: ignore[misc]
        exam = exams[key]
        picked[class_name] = {
            "split": "test",
            "exam": exam["exam"],
            "swallow": exam["swallows"][position].name,
            "pattern": pattern,
            "signature_match": chosen is not None,
            "source": str(exam["swallows"][position]),
            "rank": ranked.index(key) + 1,
            "distance": round(float(np.linalg.norm(standardised[key] - centroid)), 4),
            "candidates": len(members),
        }
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=SETTINGS.dataset_root)
    parser.add_argument("--calibration", type=Path, default=SETTINGS.calibration_path)
    parser.add_argument("--out", type=Path, default=SETTINGS.reference_dir)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    if not args.dataset_root.is_dir():
        raise SystemExit(f"corpus ausente em {args.dataset_root}")
    if not args.calibration.is_file():
        raise SystemExit(f"calibracao ausente em {args.calibration}")

    exams = measure_split(args.dataset_root, args.split, args.calibration, args.workers)
    picked = choose(exams)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"split": args.split, "references": {}}
    for class_name, entry in picked.items():
        shutil.copyfile(entry["source"], args.out / f"{class_name}.png")
        manifest["references"][class_name] = {
            key: value for key, value in entry.items() if key != "source"
        }
        print(f"  {class_name:34s} {entry['exam']}/{entry['swallow']}  padrao={entry['pattern']}")

    manifest_path = Path(__file__).resolve().parent.parent / "assets" / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(picked)} imagens em {args.out}\nmanifesto em {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
