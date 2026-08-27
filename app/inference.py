"""Score one HRM examination and describe how the score was reached.

The service loads the checkpoint once and answers `analyse()` calls under a
lock: a single ConvNeXt-Base forward over ten 224px panels is already a
GPU-sized unit of work, so serialising is what keeps concurrent uploads from
fighting over VRAM instead of speeding anything up.

Everything the UI shows about *why* a study was called comes from the same
tensors the classifier itself consumed (the image-calibrated Chicago v4.0
proxies per swallow, their CC v4.0 contractility pattern, and the per-swallow
posteriors of the instance head), so the explanation cannot drift from the
prediction.
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
from .model import (
    MAX_SWALLOWS,
    TTA_OFFSETS,
    HeavyExamModel,
    stack_exam,
    to_cache_square,
    to_model_input,
    window,
)
from .vendor.chicago import (
    DCI_FAILED,
    DCI_HYPER,
    DCI_WEAK,
    DL_PREMATURE,
    IRP_UPPER,
    METRIC_NAMES,
    study_features,
    swallow_pattern,
)
from .vendor.image_chicago import ImageCalibration, image_swallow_metrics

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

    Same quantities the ``chicago`` branch of the network consumes, kept as named
    numbers here so the interface can state the reasoning instead of only the
    verdict.  They are *image-calibrated proxies* recovered from the rendered
    topography, not measurements read off the acquisition system.
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


class ManometryClassifier:
    """Lazily loaded singleton around the trained checkpoint."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._model: HeavyExamModel | None = None
        self.device = None
        self.tta = False
        self._torch = None
        self.classes: list[str] = []
        self.calibration: ImageCalibration | None = None
        self.study_mean: np.ndarray | None = None
        self.study_std: np.ndarray | None = None
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
            # The checkpoint is a local build artifact of the sibling training
            # repository and stores numpy standardisation arrays alongside the
            # tensors, which the weights-only unpickler cannot rebuild.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = payload.get("config", {})
            self.classes = list(payload["classes"])
            self.study_mean = np.asarray(payload["study_mean"], dtype=np.float32)
            self.study_std = np.asarray(payload["study_std"], dtype=np.float32)
            self.calibration = ImageCalibration.load(self.settings.calibration_path)
            # Reproduce the run by default: it chose between the centre window
            # and the five-crop average on its own validation examinations, and
            # recorded which one it kept in `result.json`, not in the checkpoint.
            # `MANOMETRY_TTA` overrides, because on a CPU-only host five forward
            # passes may not be worth the accuracy.
            result = self._result()
            self.tta = (
                bool(result.get("tta_selected", False))
                if self.settings.tta is None
                else self.settings.tta
            )

            model = HeavyExamModel(
                payload["encoder"],
                payload["fusion"],
                len(self.classes),
                int(payload["metric_dim"]),
                drop_path=0.0,
                hidden=int(config.get("hidden", 256)),
                dropout=float(config.get("dropout", 0.2)),
            )
            model.load_state_dict(payload["state_dict"], strict=True)
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
        selection = payload.get("selection", {})
        return {
            "encoder": payload.get("encoder"),
            "fusion": payload.get("fusion"),
            "seed": payload.get("seed"),
            "classes": self.classes,
            "max_swallows": MAX_SWALLOWS,
            "device": str(getattr(self, "device", "cpu")),
            "selected_epoch": selection.get("epoch"),
            "selected_weights": selection.get("weights"),
            "tta": self.tta,
            # Publish the numbers of the path this service actually runs, not
            # whichever line of the result file reads best.
            "test_metrics": (
                result.get("test_metrics_tta") if self.tta else result.get("test_metrics_centre")
            )
            or result.get("test_metrics"),
            "val_metrics": (
                result.get("val_metrics_tta") if self.tta else result.get("val_metrics")
            )
            or result.get("val_metrics"),
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

        # Every swallow is measured, but the network reads at most ten slots;
        # the surplus is reported and marked unused rather than dropped silently.
        rows, confidences = self.metric_rows(swallows)
        used = rows[:MAX_SWALLOWS]
        study = study_features(used)
        standardised = ((study - self.study_mean) / self.study_std).astype(np.float32)

        squares = [to_cache_square(image) for image in swallows[:MAX_SWALLOWS]]
        offsets = TTA_OFFSETS if self.tta else (TTA_OFFSETS[0],)
        study = torch.from_numpy(standardised).unsqueeze(0)

        exam_probabilities = np.zeros(len(self.classes))
        swallow_probabilities = np.zeros((MAX_SWALLOWS, len(self.classes)))
        with self._lock:
            with torch.inference_mode():
                # The crops are run one at a time rather than as one batch of
                # five: same arithmetic, a fifth of the peak memory.
                for offset in offsets:
                    images, mask = stack_exam([window(square, offset) for square in squares])
                    flat = images.reshape(-1, *images.shape[2:])
                    pixels = to_model_input(flat, self.device).reshape(
                        1, MAX_SWALLOWS, 3, images.shape[2], images.shape[3]
                    )
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
                    ):
                        exam_logits, instance_logits = self._model(
                            pixels, mask.to(self.device), study.to(self.device)
                        )
                    exam_probabilities += (
                        torch.softmax(exam_logits.float(), dim=1)[0].cpu().numpy().astype(np.float64)
                    )
                    swallow_probabilities += (
                        torch.softmax(instance_logits.float(), dim=2)[0]
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
        exam_probabilities /= len(offsets)
        swallow_probabilities /= len(offsets)

        readings: list[SwallowReading] = []
        for position, (image, row) in enumerate(zip(swallows, rows)):
            in_model = position < MAX_SWALLOWS
            readings.append(
                SwallowReading(
                    index=position + 1,
                    pattern=swallow_pattern(row),
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
            study_summary=summarise_study(used),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


CLASSIFIER = ManometryClassifier()
