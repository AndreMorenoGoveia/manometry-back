#!/usr/bin/env python3
"""Pre-build one demo laudo per diagnosis, so a deployed instance can offer them.

A hosted instance has no corpus mounted, so `/api/sample-report` can only answer
from files written ahead of time. This writes them into `assets/samples/`.

Those files carry real (de-identified) pressure topographies from the held-out
partition. Whether they ship with a public deployment is a decision about
publishing patient-derived images, so the directory is ignored by git until
whoever deploys decides otherwise.

    python3 scripts/build_sample_reports.py
    python3 scripts/build_sample_reports.py --only normal_motility achalasia_type_ii
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS  # noqa: E402
from app.diseases import CLASS_ORDER  # noqa: E402
from app.sample import build_docx, laudo_from_exam  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=SETTINGS.dataset_root)
    parser.add_argument("--out", type=Path, default=SETTINGS.sample_dir)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--only", nargs="*", default=list(CLASS_ORDER))
    parser.add_argument("--index", type=int, default=0, help="qual exame da classe usar")
    args = parser.parse_args(argv)

    if not args.dataset_root.is_dir():
        raise SystemExit(f"corpus ausente em {args.dataset_root}")

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for slug in args.only:
        if slug not in CLASS_ORDER:
            raise SystemExit(f"classe desconhecida: {slug}")
        exams = sorted((args.dataset_root / args.split / slug).glob("exam_*"))
        if not exams:
            print(f"  {slug:34s} sem exames em {args.split}")
            continue
        exam = exams[args.index % len(exams)]
        destination = args.out / f"{slug}.docx"
        build_docx(
            laudo_from_exam(exam),
            destination,
            text=[
                "SERVICO DE MOTILIDADE DIGESTIVA",
                "Manometria esofagica de alta resolucao - 10 degluticoes de agua",
                f"Exame: {exam.name}",
            ],
        )
        size = destination.stat().st_size / (1024 * 1024)
        print(f"  {slug:34s} {exam.name}  {size:.1f} MB")
        written += 1

    print(f"\n{written} laudos em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
