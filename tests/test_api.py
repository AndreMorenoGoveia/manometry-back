"""The HTTP contract the frontend is written against."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.diseases import CLASS_ORDER
from app.inference import CLASSIFIER, ExamReading, ModelUnavailable, SwallowReading, summarise_study
from app.main import app
from app.vendor.chicago import METRIC_NAMES

from app.sample import build_docx, render_logo, render_panel, render_placeholder
from test_report import _topography

client = TestClient(app)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _laudo_bytes(tmp_path: Path, panels: int = 3, placeholders: int = 1) -> bytes:
    images = [render_logo()]
    images += [render_panel(_topography(seed=index), index + 1) for index in range(panels)]
    images += [render_placeholder() for _ in range(placeholders)]
    return build_docx(images, tmp_path / "laudo.docx").read_bytes()


def _fake_reading(swallow_count: int = 3) -> ExamReading:
    """A reading shaped exactly like a real one, without touching the GPU."""
    rng = np.random.default_rng(0)
    probabilities = np.full(len(CLASS_ORDER), 0.02)
    probabilities[CLASS_ORDER.index("achalasia_type_ii")] = 1.0 - 0.02 * (len(CLASS_ORDER) - 1)
    rows = [
        {
            "dci": 40.0, "irp4": 22.0, "distal_latency": 5.0, "front_arrival": 3.0,
            "largest_break_cm": 7.0, "egj_basal": 30.0, "peak_distal_pressure": 45.0,
            "pressurization_fraction": 0.95, "pan_time_fraction": 0.4, "body_length_cm": 12.0,
        }
        for _ in range(swallow_count)
    ]
    swallows = [
        SwallowReading(
            index=position + 1,
            pattern="panesophageal",
            metrics=dict(row),
            colour_confidence=0.91,
            probabilities=rng.dirichlet(np.ones(len(CLASS_ORDER))),
            thumbnail="data:image/png;base64,AAAA",
            used=True,
        )
        for position, row in enumerate(rows)
    ]
    return ExamReading(
        probabilities=probabilities,
        classes=list(CLASS_ORDER),
        swallows=swallows,
        study_summary=summarise_study(rows),
        elapsed_ms=12.5,
    )


@pytest.fixture
def stub_model(monkeypatch):
    monkeypatch.setattr(CLASSIFIER, "analyse", lambda swallows, **kwargs: _fake_reading(len(swallows)))
    monkeypatch.setattr(
        CLASSIFIER, "card",
        {"encoder": "convnext_base_in22k", "fusion": "chicago_set", "classes": list(CLASS_ORDER)},
    )


# ------------------------------------------------------------------- meta


def test_health():
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert set(payload) >= {"version", "model_available", "model_loaded", "device", "checkpoint"}


def test_diseases_listing_matches_the_model_classes():
    payload = client.get("/api/diseases").json()["diseases"]
    assert tuple(sorted(entry["slug"] for entry in payload)) == CLASS_ORDER
    for entry in payload:
        assert entry["criteria"] and entry["summary"] and entry["accent"]


def test_single_disease_and_404():
    assert client.get("/api/diseases/achalasia_type_i").json()["short_name"] == "Acalásia I"
    assert client.get("/api/diseases/not_a_disease").status_code == 404


def test_vocabulary_covers_patterns_and_metrics():
    payload = client.get("/api/vocabulary").json()
    assert set(payload["metrics"]) == set(METRIC_NAMES)
    assert payload["patterns"]["premature"]["label"] == "Prematura"


def test_reference_image_is_png_or_404():
    response = client.get("/api/diseases/normal_motility/reference.png")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_reference_image_of_unknown_disease_is_404():
    assert client.get("/api/diseases/nope/reference.png").status_code == 404


# ---------------------------------------------------------------- refusals


def test_wrong_extension_is_415():
    response = client.post("/api/analyze", files={"file": ("laudo.pdf", b"%PDF-1.7", "application/pdf")})
    assert response.status_code == 415


def test_empty_file_is_400():
    response = client.post("/api/analyze", files={"file": ("laudo.docx", b"", DOCX_MIME)})
    assert response.status_code == 400


def test_corrupt_docx_is_422():
    response = client.post("/api/analyze", files={"file": ("laudo.docx", b"PK\x03\x04junk", DOCX_MIME)})
    assert response.status_code == 422


def test_report_without_swallows_is_422(tmp_path):
    payload = build_docx([render_logo(), render_placeholder()], tmp_path / "l.docx").read_bytes()
    response = client.post("/api/analyze", files={"file": ("laudo.docx", payload, DOCX_MIME)})
    assert response.status_code == 422
    assert "degluti" in response.json()["detail"].lower()


def test_oversized_upload_is_413(tmp_path, monkeypatch):
    import dataclasses

    from app import main as main_module

    monkeypatch.setattr(
        main_module, "SETTINGS", dataclasses.replace(main_module.SETTINGS, max_upload_bytes=1024)
    )
    payload = _laudo_bytes(tmp_path, panels=1, placeholders=0)
    response = client.post("/api/analyze", files={"file": ("laudo.docx", payload, DOCX_MIME)})
    assert response.status_code == 413


def test_missing_checkpoint_is_503(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise ModelUnavailable("checkpoint ausente")

    monkeypatch.setattr(CLASSIFIER, "analyse", unavailable)
    payload = _laudo_bytes(tmp_path, panels=1, placeholders=0)
    response = client.post("/api/analyze", files={"file": ("laudo.docx", payload, DOCX_MIME)})
    assert response.status_code == 503


# ------------------------------------------------------------- happy path


def test_analysis_payload_is_complete(tmp_path, stub_model):
    payload = _laudo_bytes(tmp_path, panels=3, placeholders=2)
    response = client.post("/api/analyze", files={"file": ("laudo.docx", payload, DOCX_MIME)})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["diagnosis"]["slug"] == "achalasia_type_ii"
    assert body["diagnosis"]["summary"] and body["diagnosis"]["accent"].startswith("#")
    assert [entry["slug"] for entry in body["ranking"]][0] == "achalasia_type_ii"
    assert len(body["ranking"]) == len(CLASS_ORDER)
    assert abs(sum(entry["probability"] for entry in body["ranking"]) - 1.0) < 1e-6
    # Ranking is sorted, descending.
    probabilities = [entry["probability"] for entry in body["ranking"]]
    assert probabilities == sorted(probabilities, reverse=True)

    assert len(body["swallows"]) == 3
    swallow = body["swallows"][0]
    assert swallow["pattern_label"] == "Pressurização panesofágica"
    assert {metric["key"] for metric in swallow["metrics"]} == set(METRIC_NAMES)
    assert swallow["top_class"]["slug"] in CLASS_ORDER

    assert body["panels"] == {
        "total": 6, "accepted": 3, "off_spec": 0,
        "logo": 1, "placeholder": 2, "not_localizable": 0, "unreadable": 0,
    }
    gates = {gate["key"]: gate for gate in body["study"]["gates"]}
    assert gates["irp_elevated"]["open"] is True
    assert "mmHg" in gates["irp_elevated"]["detail"]
    assert any("sem dados" in note for note in body["warnings"])
    assert any("10" in note for note in body["warnings"])
    assert body["model_card"]["encoder"] == "convnext_base_in22k"


def test_rtf_upload_is_accepted_by_extension(tmp_path, stub_model):
    # No PNGs inside, so it must reach the reader and fail there, not at the gate.
    response = client.post(
        "/api/analyze", files={"file": ("laudo.rtf", rb"{\rtf1\ansi}", "application/rtf")}
    )
    assert response.status_code == 422


def test_openapi_documents_the_analyze_route():
    schema = client.get("/openapi.json").json()
    assert "/api/analyze" in schema["paths"]
    assert "AnalysisResponse" in schema["components"]["schemas"]


def test_the_forward_pass_does_not_run_on_the_event_loop(tmp_path, monkeypatch):
    """Scoring is blocking work and belongs on a worker thread.

    Inline on the event loop it would stall every other request, health checks
    included, for the whole length of the analysis. A function running on the
    loop's own thread can see a running loop; one on a worker thread cannot,
    which is what this asserts.
    """
    import asyncio

    seen: dict[str, bool] = {}

    def record(swallows, **kwargs):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return _fake_reading(len(swallows))

    monkeypatch.setattr(CLASSIFIER, "analyse", record)
    monkeypatch.setattr(CLASSIFIER, "card", {"classes": list(CLASS_ORDER)})
    payload = _laudo_bytes(tmp_path, panels=1, placeholders=0)

    response = client.post("/api/analyze", files={"file": ("laudo.docx", payload, DOCX_MIME)})
    assert response.status_code == 200
    assert seen["on_loop"] is False, "a inferencia rodou na thread do event loop"
