#!/usr/bin/env python3
"""Rank the trained runs that this service could serve.

Ordering is by **validation** macro-F1, which is the only number a checkpoint may
be chosen on; the test column is printed alongside so the cost of the choice is
visible, never to make it.

A run is only listed as servable if its Chicago features came from the images.
Some runs in the artefact tree were trained on features derived from the pressure
CSVs, which a report does not carry, and serving one of those would feed the
Chicago branch a distribution it never saw. The check is direct: rebuild the
run's own training split, recompute the standardisation from the image-derived
metrics, and compare against the constants the checkpoint stored. It needs the
`.cache` of the training repository; without it, every run is listed and flagged
as unverified.

    python3 scripts/compare_checkpoints.py
    python3 scripts/compare_checkpoints.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS  # noqa: E402


def image_metric_check(checkpoint: Path, seed: int) -> bool | None:
    """True when the run standardised its Chicago features from the images.

    None when the training cache is not on this machine and the question cannot
    be answered.
    """
    import numpy as np
    import torch

    cache = SETTINGS.classification_root / ".cache"
    experiments = SETTINGS.classification_root / "experiments"
    index_path = cache / "institutional_index.json"
    metrics_path = cache / "institutional_image_metrics.npy"
    if not (index_path.is_file() and metrics_path.is_file() and (experiments / "splits.py").is_file()):
        return None
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    try:
        from splits import strict_split
    except ImportError:
        return None

    from app.vendor.chicago import METRIC_NAMES, study_features

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = np.asarray(payload["study_mean"], dtype=np.float32)

    index = json.loads(index_path.read_text())
    exam_ids = [row["exam_id"] for row in index]
    labels = [row["label"] for row in index]
    metrics = np.load(metrics_path)

    order: list[str] = []
    rows_of: dict[str, list[int]] = {}
    for row, exam in enumerate(exam_ids):
        if exam not in rows_of:
            rows_of[exam] = []
            order.append(exam)
        rows_of[exam].append(row)
    features = np.stack(
        [
            study_features([dict(zip(METRIC_NAMES, metrics[row])) for row in rows_of[exam]])
            for exam in order
        ]
    ).astype(np.float32)

    position = {exam: index for index, exam in enumerate(order)}
    split = strict_split(exam_ids, labels, len(payload["classes"]), seed)
    train = sorted({exam_ids[row] for row in split["train"][0]})
    recomputed = features[[position[exam] for exam in train]].mean(axis=0)
    return bool(np.abs(recomputed - stored).max() < 1e-3)


def rows(root: Path) -> list[dict]:
    found: list[dict] = []
    for result in sorted(root.rglob("result.json")):
        checkpoint = result.parent / "model.pt"
        if not checkpoint.is_file():
            continue          # a run without saved weights cannot be served
        try:
            payload = json.loads(result.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tta = bool(payload.get("tta_selected", False))
        validation = payload.get("val_metrics_tta" if tta else "val_metrics") or {}
        test = payload.get("test_metrics_tta" if tta else "test_metrics_centre") or payload.get(
            "test_metrics", {}
        )
        seed = payload.get("seed")
        try:
            image_metrics = image_metric_check(checkpoint, int(seed)) if seed is not None else None
        except Exception:      # a stale or unreadable checkpoint answers "unknown"
            image_metrics = None
        found.append(
            {
                "run": str(result.parent.relative_to(root)),
                "image_metrics": image_metrics,
                "encoder": payload.get("encoder", "?"),
                "tta": tta,
                "megabytes": checkpoint.stat().st_size / (1024 * 1024),
                "val_macro_f1": float(validation.get("macro_f1", float("nan"))),
                "test_macro_f1": float(test.get("macro_f1", float("nan"))),
                "test_accuracy": float(test.get("accuracy", float("nan"))),
                "path": str(checkpoint),
            }
        )
    return sorted(found, key=lambda row: row["val_macro_f1"], reverse=True)


def main(argv: list[str] | None = None) -> int:
    default = SETTINGS.classification_root / "artifacts" / "leakage_fusion"
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=default)
    parser.add_argument("--all", action="store_true", help="lista tambem os runs nao servíveis")
    args = parser.parse_args(argv)

    found = rows(args.root)
    if not found:
        raise SystemExit(f"nenhum run com model.pt em {args.root}")

    label = {True: "img", False: "CSV", None: "?"}
    print(f"{'run':56s} {'MB':>5s} {'metr':>5s} {'tta':>4s} {'val F1':>7s} {'test F1':>8s} {'test acc':>9s}")
    for row in found:
        if not args.all and row["image_metrics"] is False:
            continue
        print(
            f"{row['run']:56s} {row['megabytes']:5.0f} {label[row['image_metrics']]:>5s} "
            f"{'sim' if row['tta'] else 'nao':>4s} {row['val_macro_f1']:7.4f} "
            f"{row['test_macro_f1']:8.4f} {row['test_accuracy']:9.4f}"
        )

    servable = [row for row in found if row["image_metrics"] is not False]
    rejected = [row for row in found if row["image_metrics"] is False]
    if rejected and not args.all:
        names = ", ".join(row["run"] for row in rejected)
        print(f"\nfora da lista (metricas de Chicago vindas dos CSV): {names}")
    if not servable:
        raise SystemExit("nenhum run servivel a partir de imagens")

    best = servable[0]
    print(f"\nmelhor na validacao: {best['run']}\n  MANOMETRY_CHECKPOINT={best['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
