#!/usr/bin/env python3
"""Write a demo laudo `.docx` from an exam folder of swallow topographies.

    python3 scripts/make_sample_docx.py \
        --exam ../manometry-classification/data/test/achalasia_type_ii/exam_001282 \
        --out laudo_demo.docx

    # or let it pick an exam of a class for you
    python3 scripts/make_sample_docx.py --class normal_motility --out normal.docx

The panel rendering lives in `app/sample.py`, which the API also uses to serve
`/api/sample-report`.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS  # noqa: E402
from app.sample import build_docx, laudo_from_exam  # noqa: E402


def _pick_exam(dataset_root: Path, class_name: str, split: str, seed: int) -> Path:
    candidates = sorted((dataset_root / split / class_name).glob("exam_*"))
    if not candidates:
        raise SystemExit(f"Nenhum exame em {dataset_root / split / class_name}")
    return random.Random(seed).choice(candidates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exam", type=Path, help="pasta de um exame com swallow_*.png")
    parser.add_argument("--class", dest="class_name", help="classe a sortear quando --exam nao e dado")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--dataset-root", type=Path, default=SETTINGS.dataset_root)
    parser.add_argument("--out", type=Path, default=Path("laudo_demo.docx"))
    parser.add_argument("--placeholders", type=int, default=0, help="paineis 'sem dados' a acrescentar")
    parser.add_argument("--limit", type=int, default=None, help="numero maximo de degluticoes")
    parser.add_argument("--no-logo", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    exam = args.exam
    if exam is None:
        if not args.class_name:
            parser.error("informe --exam ou --class")
        exam = _pick_exam(args.dataset_root, args.class_name, args.split, args.seed)

    images = laudo_from_exam(
        exam, with_logo=not args.no_logo, placeholders=args.placeholders, limit=args.limit
    )
    path = build_docx(
        images,
        args.out,
        text=[
            "SERVICO DE MOTILIDADE DIGESTIVA",
            "Manometria esofagica de alta resolucao - 10 degluticoes de agua",
            f"Exame: {exam.name}",
        ],
    )
    print(f"{path}  ({len(images)} imagens, origem {exam})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
