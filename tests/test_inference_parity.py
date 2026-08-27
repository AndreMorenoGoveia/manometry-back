"""Re-score the run's own held-out examinations through the API's code path.

This is the test that keeps `app/model.py` honest.  The checkpoint ships the
probabilities it produced on its test partition; this rebuilds that partition,
feeds the same images through the service (not through the training loader) and
asserts the answers come back the same.  Any drift in the architecture
transcription, the preprocessing, the Chicago proxies or the standardisation
shows up here as a disagreement.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import SETTINGS
from app.inference import CLASSIFIER

pytestmark = pytest.mark.slow

# Scoring is a full ConvNeXt-Base forward per examination; a dozen is enough to
# catch a transcription error and still finish in seconds on a GPU.
EXAM_BUDGET = int(os.environ.get("MANOMETRY_PARITY_EXAMS", "12"))


def _requirements() -> tuple[list[dict], np.ndarray, np.ndarray, Path]:
    predictions = SETTINGS.checkpoint_path.parent / "test_predictions.npz"
    index_path = SETTINGS.classification_root / ".cache" / "institutional_index.json"
    experiments = SETTINGS.classification_root / "experiments"
    for path in (SETTINGS.checkpoint_path, predictions, index_path, experiments / "splits.py"):
        if not path.exists():
            pytest.skip(f"artefato de treino indisponivel: {path}")
    stored = np.load(predictions)
    index = json.loads(index_path.read_text())
    return index, stored["probabilities"], stored["targets"], experiments


def test_service_reproduces_the_runs_test_predictions():
    index, reference, targets, experiments = _requirements()
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    from splits import strict_split

    CLASSIFIER.load()
    exam_ids = [row["exam_id"] for row in index]
    labels = [row["label"] for row in index]
    images = [row["image"] for row in index]

    rows_of: dict[str, list[int]] = {}
    for row, exam in enumerate(exam_ids):
        rows_of.setdefault(exam, []).append(row)

    split = strict_split(exam_ids, labels, len(CLASSIFIER.classes), int(CLASSIFIER.card["seed"]))
    test_exams = sorted({exam_ids[row] for row in split["test"][0]})
    assert len(test_exams) == len(reference), (
        "a particao reconstruida nao tem o tamanho da guardada com o checkpoint; "
        "o corpus mudou desde o treino"
    )

    budget = min(EXAM_BUDGET, len(test_exams))
    mine = []
    for exam in test_exams[:budget]:
        swallows = [
            np.asarray(Image.open(images[row]).convert("RGB")) for row in rows_of[exam]
        ]
        mine.append(CLASSIFIER.analyse(swallows, thumbnails=False).probabilities)
    mine = np.stack(mine)
    expected = reference[:budget]

    agreement = float((mine.argmax(1) == expected.argmax(1)).mean())
    deviation = float(np.abs(mine - expected).mean())
    assert agreement == 1.0, (
        f"o servico discordou do treino em {(1 - agreement) * 100:.0f}% dos exames "
        f"(desvio medio {deviation:.4f})"
    )
    # bfloat16 autocast and a batch of one instead of six leave a small residue.
    assert deviation < 0.01, f"desvio medio de probabilidade alto demais: {deviation:.4f}"


def test_service_matches_the_reported_accuracy():
    index, reference, targets, experiments = _requirements()
    expected = float((reference.argmax(1) == targets).mean())
    reported = CLASSIFIER.card.get("test_metrics", {}).get("accuracy")
    if reported is None:
        pytest.skip("result.json sem test_metrics")
    assert abs(expected - reported) < 1e-6
