"""The parts of the scoring service that do not need the GPU."""
from __future__ import annotations

import base64

import numpy as np
import pytest

from app.inference import _to_data_uri, summarise_study
from app.model import EVAL_OFFSET, IMAGE_SIZE, MAX_SWALLOWS, stack_exam, to_cache_window
from app.vendor.chicago import DCI_FAILED, DCI_HYPER, DCI_WEAK

BASE = {
    "dci": 1900.0, "irp4": 9.0, "distal_latency": 7.0, "front_arrival": 4.0,
    "largest_break_cm": 0.0, "egj_basal": 22.0, "peak_distal_pressure": 90.0,
    "pressurization_fraction": 0.2, "pan_time_fraction": 0.0, "body_length_cm": 14.0,
}


def rows(count: int, **overrides) -> list[dict[str, float]]:
    return [dict(BASE, **overrides) for _ in range(count)]


# ------------------------------------------------------------ study summary


def test_normal_study_opens_no_gate():
    summary = summarise_study(rows(10))
    assert summary["swallows"] == 10
    assert not any(summary[key] for key in summary if key.endswith("_gate"))
    assert summary["irp_elevated"] is False


def test_absent_contractility_gate_needs_every_swallow():
    nine = rows(9, dci=DCI_FAILED - 1) + rows(1)
    assert summarise_study(nine)["absent_contractility_gate"] is False
    assert summarise_study(rows(10, dci=DCI_FAILED - 1))["absent_contractility_gate"] is True


def test_ineffective_gate_is_above_seventy_percent():
    seventy = rows(7, dci=DCI_WEAK - 1) + rows(3)
    assert summarise_study(seventy)["ineffective_gate"] is False
    eighty = rows(8, dci=DCI_WEAK - 1) + rows(2)
    assert summarise_study(eighty)["ineffective_gate"] is True


def test_premature_gate_ignores_weak_swallows():
    """A short latency on a swallow that never contracted is not prematurity."""
    weak_and_early = rows(4, dci=DCI_WEAK - 1, distal_latency=2.0) + rows(6)
    assert summarise_study(weak_and_early)["premature_gate"] is False
    strong_and_early = rows(4, dci=3000.0, distal_latency=2.0) + rows(6)
    assert summarise_study(strong_and_early)["premature_gate"] is True


def test_hypercontractile_gate():
    assert summarise_study(rows(1, dci=DCI_HYPER + 1) + rows(9))["hypercontractile_gate"] is False
    assert summarise_study(rows(2, dci=DCI_HYPER + 1) + rows(8))["hypercontractile_gate"] is True


def test_irp_gate_uses_the_median_not_the_mean():
    """One outlier swallow must not open the achalasia branch on its own."""
    skewed = rows(1, irp4=90.0) + rows(9, irp4=8.0)
    summary = summarise_study(skewed)
    assert summary["median_irp4"] == pytest.approx(8.0)
    assert summary["irp_elevated"] is False


def test_panesophageal_fraction_follows_the_pattern_rule():
    pressurised = rows(3, pan_time_fraction=0.5, dci=50.0) + rows(7)
    summary = summarise_study(pressurised)
    assert summary["panesophageal_fraction"] == pytest.approx(0.3)
    assert summary["panesophageal_gate"] is True


def test_summary_is_json_serialisable_scalars():
    summary = summarise_study(rows(5))
    for key, value in summary.items():
        if key == "thresholds":
            continue
        assert isinstance(value, (int, float, bool)), key


# ------------------------------------------------------------ preprocessing


def test_cache_window_is_the_centre_224_of_a_256_square():
    rgb = np.random.default_rng(0).integers(0, 255, (500, 300, 3), dtype=np.uint8)
    window = to_cache_window(rgb)
    assert window.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert window.dtype == np.uint8
    assert EVAL_OFFSET == 16


def test_stack_pads_to_ten_slots_and_masks_the_rest():
    windows = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) for _ in range(4)]
    images, mask = stack_exam(windows)
    assert images.shape == (1, MAX_SWALLOWS, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert mask.tolist() == [[True] * 4 + [False] * 6]


def test_stack_truncates_beyond_ten():
    windows = [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) for _ in range(13)]
    images, mask = stack_exam(windows)
    assert images.shape[1] == MAX_SWALLOWS and bool(mask.all())


def test_thumbnail_is_a_png_data_uri():
    uri = _to_data_uri(np.zeros((600, 400, 3), dtype=np.uint8))
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_thumbnail_is_downscaled():
    from app.inference import THUMBNAIL_MAX_SIDE
    from PIL import Image
    import io

    uri = _to_data_uri(np.zeros((900, 600, 3), dtype=np.uint8))
    payload = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(io.BytesIO(payload)) as image:
        assert max(image.size) == THUMBNAIL_MAX_SIDE


# ------------------------------------------------------------- model card


def test_per_class_metrics_are_read_from_the_runs_own_predictions(tmp_path, monkeypatch):
    """The card reports the training run's numbers, not numbers invented here."""
    import dataclasses

    from app.inference import ManometryClassifier
    from app.config import SETTINGS

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    probabilities = np.eye(4, 3)[[0, 1, 2, 0]]      # 3 classes, one mistake
    predictions = tmp_path / "test_predictions.npz"
    np.savez(predictions, probabilities=probabilities, targets=np.array([0, 1, 2, 1]))

    classifier = ManometryClassifier(
        dataclasses.replace(SETTINGS, checkpoint_path=checkpoint, predictions_path=predictions)
    )
    classifier.classes = ["a", "b", "c"]
    rows = classifier._per_class_metrics()

    assert rows is not None
    assert [row["slug"] for row in rows] == ["a", "b", "c"]
    assert [row["support"] for row in rows] == [1, 2, 1]
    assert rows[0]["precision"] == pytest.approx(0.5)   # two predicted "a", one right
    assert rows[2]["recall"] == pytest.approx(1.0)


def test_per_class_metrics_are_absent_when_the_run_did_not_store_them(tmp_path):
    import dataclasses

    from app.inference import ManometryClassifier
    from app.config import SETTINGS

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    classifier = ManometryClassifier(
        dataclasses.replace(
            SETTINGS, checkpoint_path=checkpoint, predictions_path=tmp_path / "absent.npz"
        )
    )
    classifier.classes = ["a", "b"]
    assert classifier._per_class_metrics() is None


def test_per_class_metrics_refuse_a_mismatched_prediction_file(tmp_path):
    """A stale npz from a run with a different label set must not be reported."""
    import dataclasses

    from app.inference import ManometryClassifier
    from app.config import SETTINGS

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    predictions = tmp_path / "test_predictions.npz"
    np.savez(predictions, probabilities=np.eye(3), targets=np.arange(3))

    classifier = ManometryClassifier(
        dataclasses.replace(SETTINGS, checkpoint_path=checkpoint, predictions_path=predictions)
    )
    classifier.classes = ["a", "b"]           # two classes, file holds three
    assert classifier._per_class_metrics() is None
