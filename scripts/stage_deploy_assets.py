#!/usr/bin/env python3
"""Copy the model artefacts a deployed instance needs into `assets/model/`.

The light checkpoint is about 51 MB, below GitHub's 100 MB per-file limit, so
the deploy keeps all required artefacts in the repository:

* `model.pt` (the trained weights);
* `image_chicago_calibration.json` (two numbers, no patient data);
* `result.json` (the run's metrics, used by `/api/model`);
* `test_predictions.npz` (the run's held-out probabilities, used for the
  per-class table).

    python3 scripts/stage_deploy_assets.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS  # noqa: E402

TARGET = Path(__file__).resolve().parent.parent / "assets" / "model"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, default=SETTINGS.checkpoint_path)
    parser.add_argument("--calibration", type=Path, default=SETTINGS.calibration_path)
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args(argv)

    sources = {
        "model.pt": args.checkpoint,
        "image_chicago_calibration.json": args.calibration,
        "result.json": args.checkpoint.parent / "result.json",
        "test_predictions.npz": args.checkpoint.parent / "test_predictions.npz",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if not source.is_file():
            print(f"  {name:34s} ausente em {source}")
            continue
        shutil.copyfile(source, args.out / name)
        print(f"  {name:34s} {(args.out / name).stat().st_size / 1024:.1f} kB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
