"""Re-score the fold's own held-out examinations through the API's code path.

This is the test that keeps `app/model.py` and `app/vendor/jbhi_readout.py` honest. The bundle ships the
probabilities `jbhi_deploy.py` produced on the test patients of partition 42 fold 0, from features the training
run extracted on the GPU (bfloat16 autocast). This feeds the same swallow images through the service, in float32
on whatever device is here, and asserts the answers come back the same.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import SETTINGS
from app.inference import CLASSIFIER

pytestmark = pytest.mark.slow

EXAM_BUDGET = int(os.environ.get("MANOMETRY_PARITY_EXAMS", "60"))


def _stored():
    predictions = SETTINGS.predictions_path
    data = SETTINGS.dataset_root
    for path in (SETTINGS.checkpoint_path, predictions, data / "labels_v4.csv"):
        if not path.exists():
            pytest.skip(f"artefato de treino indisponivel: {path}")
    return np.load(predictions), data


def _swallows(data: Path, exam: str) -> list[np.ndarray]:
    folder = next(data.glob(f"*/*/{exam}"), None)
    if folder is None:
        pytest.skip(f"exame {exam} ausente do corpus local")
    # The training index takes the swallows in file order and the model reads the first ten.
    return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(folder.glob("swallow_*.png"))]


def test_service_reproduces_the_folds_test_predictions():
    stored, data = _stored()
    budget = min(EXAM_BUDGET, len(stored["exam_ids"]))
    mine = np.stack([
        CLASSIFIER.analyse(_swallows(data, str(exam)), thumbnails=False).probabilities
        for exam in stored["exam_ids"][:budget]
    ])
    expected = stored["probabilities"][:budget]
    agreement = float((mine.argmax(1) == expected.argmax(1)).mean())
    deviation = float(np.abs(mine - expected).mean())
    # bfloat16 GPU features against float32 here (and half-precision storage of the weights) move the
    # probabilities by about 1e-4 on average (all 358 exams agree); a wrong feature, order or scaler moves
    # them by orders of magnitude more.
    assert agreement == 1.0, f"o servico discordou em {(1 - agreement) * 100:.1f}% dos exames (desvio {deviation:.4f})"
    assert deviation < 1e-3, f"desvio medio de probabilidade alto demais: {deviation:.5f}"


def test_the_card_publishes_the_numbers_of_the_path_it_runs():
    stored, _ = _stored()
    CLASSIFIER.load()
    expected = float((stored["probabilities"].argmax(1) == stored["targets"]).mean())
    reported = (CLASSIFIER.card.get("test_metrics") or {}).get("accuracy")
    assert reported is not None and abs(expected - reported) < 1e-6
