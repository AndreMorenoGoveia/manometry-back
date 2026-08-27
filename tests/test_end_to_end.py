"""The whole path, from a laudo file to a diagnosis, with the real checkpoint.

Everything else in the suite stubs the model. This one does not: it builds a
laudo out of real examinations, posts it, and checks both the shape of the answer
and that the answer is right often enough to be the model's and not an accident
of plumbing.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import SETTINGS
from app.main import app
from app.sample import build_docx_bytes, laudo_from_exam

pytestmark = pytest.mark.slow

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

client = TestClient(app)


def _exams(slug: str, count: int) -> list:
    exams = sorted((SETTINGS.dataset_root / "test" / slug).glob("exam_*"))
    usable = [exam for exam in exams if len(list(exam.glob("swallow_*.png"))) >= 8]
    if len(usable) < count:
        pytest.skip(f"corpus insuficiente para {slug}")
    return usable[:count]


def _analyse(exam, *, limit=None) -> dict:
    payload = build_docx_bytes(laudo_from_exam(exam, limit=limit))
    response = client.post(
        "/api/analyze", files={"file": (f"{exam.name}.docx", payload, DOCX_MIME)}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_real_laudo_comes_back_fully_described(model_ready, dataset_root):
    exam = _exams("achalasia_type_ii", 1)[0]
    body = _analyse(exam)

    swallows = len(list(exam.glob("swallow_*.png")))
    assert len(body["swallows"]) == swallows
    assert body["panels"]["accepted"] == swallows
    assert body["panels"]["logo"] == 1
    assert body["study"]["swallows"] == min(swallows, 10)
    assert abs(sum(entry["probability"] for entry in body["ranking"]) - 1.0) < 1e-6
    assert all(entry["metrics"] for entry in body["swallows"])
    assert all(entry["thumbnail"].startswith("data:image/png;base64,") for entry in body["swallows"])
    assert body["elapsed_ms"] > 0


def test_the_answer_is_the_models_and_not_a_coincidence(model_ready, dataset_root):
    """Type II achalasia is the class the run recognises best; most must land."""
    hits = 0
    exams = _exams("achalasia_type_ii", 3)
    for exam in exams:
        body = _analyse(exam)
        hits += body["diagnosis"]["slug"] == "achalasia_type_ii"
    assert hits >= 2, f"apenas {hits}/{len(exams)} exames de acalasia II foram reconhecidos"


def test_the_same_laudo_twice_gives_the_same_answer(model_ready, dataset_root):
    exam = _exams("normal_motility", 1)[0]
    first = _analyse(exam)
    second = _analyse(exam)
    assert first["diagnosis"]["slug"] == second["diagnosis"]["slug"]
    assert first["diagnosis"]["probability"] == pytest.approx(
        second["diagnosis"]["probability"], abs=1e-6
    )


def test_a_report_with_a_single_swallow_is_still_answered(model_ready, dataset_root):
    """Nine of the ten slots stay padded; the aggregation must not divide by zero."""
    exam = _exams("normal_motility", 1)[0]
    body = _analyse(exam, limit=1)

    assert len(body["swallows"]) == 1
    assert body["study"]["swallows"] == 1
    assert body["diagnosis"]["probability"] > 0
    assert any("deglutição válida" in note for note in body["warnings"])
