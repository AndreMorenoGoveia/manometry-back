"""Score one HRM examination and describe how the score was reached.

The service loads the checkpoint once and answers `analyse()` calls under a
lock, so concurrent uploads queue instead of fighting over CPU and memory.

DCI, IRP 4s and distal latency per swallow are the model's own measurement
heads, the quantities its readout consumes, so the explanation follows the
prediction. The remaining per-swallow values (breaks, EGJ basal pressure,
pressurization) are image-calibrated colour proxies, shown as supporting
evidence only; the model does not read them.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SETTINGS, Settings, fetch_checkpoint
from .model import MAX_SWALLOWS, MultitaskSwallowModel, to_cache_window, to_model_input
from .vendor.chicago import (
    DCI_FAILED,
    DCI_HYPER,
    DCI_WEAK,
    DL_PREMATURE,
    IRP_UPPER,
    METRIC_NAMES,
    swallow_pattern,
)
from .vendor.image_chicago import ImageCalibration, image_swallow_metrics
from .vendor.jbhi_readout import RULE_ORDER, readout_features, swallow_states

THUMBNAIL_MAX_SIDE = 360


class ModelUnavailable(RuntimeError):
    """The checkpoint or the colour calibration is not on this machine."""


@dataclass
class SwallowReading:
    index: int
    pattern: str
    metrics: dict[str, float]
    colour_confidence: float
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thumbnail: str = ""
    used: bool = True


@dataclass
class ExamReading:
    probabilities: np.ndarray
    classes: list[str]
    swallows: list[SwallowReading]
    study_summary: dict[str, float | bool | int]
    elapsed_ms: float


def _to_data_uri(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))
    scale = THUMBNAIL_MAX_SIDE / max(image.size)
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def summarise_study(rows: list[dict[str, float]]) -> dict[str, float | bool | int]:
    """The CC v4.0 study-level statistics and gates, in reader's terms.

    In `analyse` the DCI, IRP and latency of each row are the model's measurement
    heads, and the fractions and gates computed here are then replaced by
    `state_patterns`; breaks and EGJ basal pressure stay image-calibrated colour
    proxies. None of them is a measurement read off the acquisition system.
    """
    dci = np.array([row["dci"] for row in rows], dtype=np.float32)
    irp = np.array([row["irp4"] for row in rows], dtype=np.float32)
    latency = np.array([row["distal_latency"] for row in rows], dtype=np.float32)
    breaks = np.array([row["largest_break_cm"] for row in rows], dtype=np.float32)
    egj = np.array([row["egj_basal"] for row in rows], dtype=np.float32)
    patterns = [swallow_pattern(row) for row in rows]

    median_irp = float(np.median(irp))
    failed = float(np.mean(dci < DCI_FAILED))
    ineffective = float(np.mean(dci < DCI_WEAK))
    premature = float(np.mean((latency < DL_PREMATURE) & (dci > DCI_WEAK)))
    hyper = float(np.mean(dci > DCI_HYPER))
    panesophageal = patterns.count("panesophageal") / len(patterns)

    return {
        "swallows": len(rows),
        "median_irp4": median_irp,
        "median_dci": float(np.median(dci)),
        "max_dci": float(dci.max()),
        "median_distal_latency": float(np.median(latency)),
        "min_distal_latency": float(latency.min()),
        "max_break_cm": float(breaks.max()),
        "median_egj_basal": float(np.median(egj)),
        "failed_fraction": failed,
        "ineffective_fraction": ineffective,
        "premature_fraction": premature,
        "hypercontractile_fraction": hyper,
        "panesophageal_fraction": panesophageal,
        # The four gates that open the CC v4.0 decision tree.
        "irp_elevated": bool(median_irp > IRP_UPPER),
        "absent_contractility_gate": bool(failed >= 1.0),
        "ineffective_gate": bool(ineffective > 0.7),
        "premature_gate": bool(premature >= 0.2),
        "hypercontractile_gate": bool(hyper >= 0.2),
        "panesophageal_gate": bool(panesophageal >= 0.2),
        "thresholds": {
            "irp_upper": IRP_UPPER,
            "dci_failed": DCI_FAILED,
            "dci_weak": DCI_WEAK,
            "dci_hyper": DCI_HYPER,
            "dl_premature": DL_PREMATURE,
        },
    }


def state_patterns(aux) -> tuple[list[str], dict[str, float | bool]]:
    """CC v4.0 swallow patterns and study fractions from the model's state heads.

    The fractions are the expected ones the readout consumes (jbhi_readout.readout_features); each swallow's
    pattern takes the most likely state in the precedence of `chicago.swallow_pattern`.
    """
    v, premature, pan, fragmented, _ = (t.numpy() for t in swallow_states(aux.double()))
    vigor = v.argmax(1)
    patterns = []
    for k in range(len(vigor)):
        if pan[k, 1] > 0.5 and vigor[k] != 3:
            patterns.append("panesophageal")
        elif vigor[k] in (0, 1):
            patterns.append(("failed", "weak")[vigor[k]])
        elif premature[k, 1] > 0.5:
            patterns.append("premature")
        elif vigor[k] == 3:
            patterns.append("hypercontractile")
        elif fragmented[k, 1] > 0.5:
            patterns.append("fragmented")
        else:
            patterns.append("normal")
    contracted = v[:, 2] + v[:, 3]
    failed = float(v[:, 0].mean())
    ineffective = float(np.minimum(v[:, 0] + v[:, 1] + contracted * fragmented[:, 1], 1).mean())
    early = float((contracted * premature[:, 1]).mean())
    hyper = float(v[:, 3].mean())
    pressurized = float(pan[:, 1].mean())
    return patterns, {
        "failed_fraction": failed,
        "ineffective_fraction": ineffective,
        "premature_fraction": early,
        "hypercontractile_fraction": hyper,
        "panesophageal_fraction": pressurized,
        "absent_contractility_gate": bool((vigor == 0).all()),
        "ineffective_gate": ineffective > 0.7,
        "premature_gate": early >= 0.2,
        "hypercontractile_gate": hyper >= 0.2,
        "panesophageal_gate": pressurized >= 0.2,
    }


class ManometryClassifier:
    """Lazily loaded singleton around the trained checkpoint."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._model: MultitaskSwallowModel | None = None
        self.device = None
        self._torch = None
        self.classes: list[str] = []
        self.calibration: ImageCalibration | None = None
        self.readout: dict = {}
        self.card: dict = {}

    # ------------------------------------------------------------- loading

    @property
    def available(self) -> bool:
        return self.settings.model_available

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self):
        import torch

        wanted = self.settings.device
        if wanted != "auto":
            return torch.device(wanted)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.available:
                raise ModelUnavailable(
                    f"Checkpoint nao encontrado em {self.settings.checkpoint_path} "
                    f"ou calibracao ausente em {self.settings.calibration_path}. "
                    "Aponte MANOMETRY_CLASSIFICATION_ROOT para o checkout de manometry-classification."
                )
            import torch

            self._torch = torch
            if self.settings.torch_threads > 0:
                torch.set_num_threads(self.settings.torch_threads)
            checkpoint = fetch_checkpoint(self.settings)
            # The bundle is written by `jbhi_deploy.py` upstream: plain tensors only.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.classes = list(payload["classes"])
            assert tuple(self.classes) == RULE_ORDER, "a ordem das classes do bundle difere da do readout"
            self.readout = {key: value.to(torch.float64) for key, value in payload["readout"].items()}
            self.calibration = ImageCalibration.load(self.settings.calibration_path)

            model = MultitaskSwallowModel(len(self.classes))
            # Stored in half precision to fit git; upcast to the float32 the run evaluated with.
            model.load_state_dict({k: v.float() for k, v in payload["state_dict"].items()}, strict=True)
            device = self._resolve_device()
            model.eval().to(device).to(memory_format=torch.channels_last)
            self.device = device
            self._model = model
            self.card = self._build_card(payload)

    def _result(self) -> dict:
        """The training run's `result.json`, or an empty dict when it is absent."""
        path = self.settings.result_path
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _per_class_metrics(self) -> list[dict] | None:
        """Precision, recall and F1 per class, from the run's own test predictions.

        The training script stores the probabilities it produced on the held-out
        examinations next to the checkpoint, so this is read off that file rather
        than recomputed: the interface reports the run's numbers, not numbers this
        process invented.
        """
        path = self.settings.predictions_path
        if not path.is_file():
            return None
        try:
            stored = np.load(path)
            probabilities = stored["probabilities"]
            targets = stored["targets"]
        except (OSError, KeyError, ValueError):
            return None
        if probabilities.shape[1] != len(self.classes):
            return None

        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            probabilities.argmax(1),
            labels=list(range(len(self.classes))),
            zero_division=0,
        )
        return [
            {
                "slug": slug,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, slug in enumerate(self.classes)
        ]

    def _build_card(self, payload: dict) -> dict:
        result = self._result()
        return {
            "encoder": payload.get("encoder"),
            "fusion": "jbhi_proposed",
            "seed": payload.get("partition"),
            "classes": self.classes,
            "max_swallows": MAX_SWALLOWS,
            "device": str(getattr(self, "device", "cpu")),
            "test_metrics": result.get("test_metrics"),
            "val_metrics": result.get("val_metrics"),
            "exam_counts": result.get("exam_counts"),
            "per_class": self._per_class_metrics(),
            "calibration": {
                "pressure_offset": self.calibration.pressure_offset if self.calibration else None,
                "pressure_scale": self.calibration.pressure_scale if self.calibration else None,
                "source": self.calibration.source if self.calibration else None,
            },
        }

    # ----------------------------------------------------------- inference

    def metric_rows(self, swallows: list[np.ndarray]) -> tuple[list[dict[str, float]], list[float]]:
        """Image-calibrated Chicago proxies, one row per swallow.

        Cast through float32 before use: the training cache stored these rows as
        a float32 array, and the study features are built from that array, so the
        aggregation must see the same precision it saw at fit time.
        """
        assert self.calibration is not None
        rows: list[dict[str, float]] = []
        confidences: list[float] = []
        for image in swallows:
            metrics, confidence = image_swallow_metrics(image, self.calibration)
            row = np.asarray([metrics[name] for name in METRIC_NAMES], dtype=np.float32)
            rows.append(dict(zip(METRIC_NAMES, row)))
            confidences.append(float(confidence))
        return rows, confidences

    def analyse(self, swallows: list[np.ndarray], *, thumbnails: bool = True) -> ExamReading:
        """Classify one examination from its cropped swallow topographies."""
        if not swallows:
            raise ValueError("A examination needs at least one swallow")
        self.load()
        torch = self._torch
        assert torch is not None and self._model is not None
        started = time.perf_counter()

        # Every swallow is measured, but the network reads at most ten;
        # the surplus is reported and marked unused rather than dropped silently.
        rows, confidences = self.metric_rows(swallows)
        windows = np.stack([to_cache_window(image) for image in swallows[:MAX_SWALLOWS]])

        with self._lock:
            with torch.inference_mode():
                pixels = to_model_input(torch.from_numpy(windows), self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    logits, aux = self._model(pixels)
        logits, aux = logits.float().cpu(), aux.float().cpu()
        exam_probabilities, swallow_probabilities = self._readout(logits, aux)

        # The measurement heads replace the colour proxies for the three values the readout is built on.
        for row, (log_dci, irp, latency) in zip(rows, aux[:, -3:].tolist()):
            row.update(dci=float(np.expm1(max(log_dci, 0.0))), irp4=irp, distal_latency=latency)
        used = rows[:MAX_SWALLOWS]
        patterns, state_summary = state_patterns(aux)

        readings: list[SwallowReading] = []
        for position, (image, row) in enumerate(zip(swallows, rows)):
            in_model = position < MAX_SWALLOWS
            readings.append(
                SwallowReading(
                    index=position + 1,
                    # Beyond the tenth swallow the model gives no states; the colour proxies are all there is.
                    pattern=patterns[position] if in_model else swallow_pattern(row),
                    metrics={name: float(value) for name, value in row.items()},
                    colour_confidence=confidences[position],
                    probabilities=swallow_probabilities[position] if in_model else np.zeros(len(self.classes)),
                    thumbnail=_to_data_uri(image) if thumbnails else "",
                    used=in_model,
                )
            )

        return ExamReading(
            probabilities=exam_probabilities,
            classes=list(self.classes),
            swallows=readings,
            study_summary={**summarise_study(used), **state_summary},
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _readout(self, logits, aux) -> tuple[np.ndarray, np.ndarray]:
        """jbhi_fusion's `proposed`: logistic over the readout features, then the validation temperature."""
        torch = self._torch
        r = self.readout
        swallow = torch.softmax(logits.double(), dim=1)
        mask = torch.ones(1, len(aux), dtype=torch.bool)
        x = readout_features(aux[None], mask, swallow.mean(0).log()[None])
        z = ((x - r["feature_mean"]) / r["feature_std"] - r["scaler_mean"]) / r["scaler_scale"]
        logp = torch.log_softmax(z @ r["coef"].T + r["intercept"], dim=1).clamp_min(np.log(1e-12))
        exam = torch.softmax(logp / r["temperature"], dim=1)[0]
        return exam.numpy(), swallow.numpy()


CLASSIFIER = ManometryClassifier()
