"""Runtime configuration, all overridable by environment variable.

The trained checkpoint and the colour calibration are large, patient-derived
build outputs that live in ``manometry-classification`` and are deliberately not
committed here; the defaults point at a sibling checkout so a developer who has
both repositories side by side needs no configuration at all.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFICATION_ROOT = REPO_ROOT.parent / "manometry-classification"

# The winning run of the leakage/fusion study: a ConvNeXt-Base (ImageNet-22k)
# encoder under the `chicago_set` study-level head, seed 42, EMA weights of
# epoch 10. Test macro-F1 0.7425 over 336 held-out examinations.
DEFAULT_CHECKPOINT = (
    "artifacts/leakage_fusion/fusion_heavy_deploy/chicago_set/"
    "convnext_base_in22k/seed_42/model.pt"
)
DEFAULT_RESULT = DEFAULT_CHECKPOINT.replace("model.pt", "result.json")
DEFAULT_CALIBRATION = ".cache/image_chicago_calibration.json"


def _path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def _flag(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    classification_root: Path
    checkpoint_path: Path
    result_path: Path
    calibration_path: Path
    dataset_root: Path
    reference_dir: Path
    device: str
    max_upload_bytes: int
    allowed_origins: list[str] = field(default_factory=list)

    @property
    def model_available(self) -> bool:
        return self.checkpoint_path.is_file() and self.calibration_path.is_file()


def load_settings() -> Settings:
    root = _path("MANOMETRY_CLASSIFICATION_ROOT", DEFAULT_CLASSIFICATION_ROOT)
    origins = os.environ.get(
        "MANOMETRY_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    )
    return Settings(
        classification_root=root,
        checkpoint_path=_path("MANOMETRY_CHECKPOINT", root / DEFAULT_CHECKPOINT),
        result_path=_path("MANOMETRY_RESULT_JSON", root / DEFAULT_RESULT),
        calibration_path=_path("MANOMETRY_CALIBRATION", root / DEFAULT_CALIBRATION),
        dataset_root=_path("MANOMETRY_DATASET_ROOT", root / "data"),
        reference_dir=_path("MANOMETRY_REFERENCE_DIR", REPO_ROOT / "assets" / "reference"),
        device=os.environ.get("MANOMETRY_DEVICE", "auto"),
        max_upload_bytes=int(os.environ.get("MANOMETRY_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)),
        allowed_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
    )


SETTINGS = load_settings()

# Torch is loaded lazily so that `import app` stays cheap for the routes that do
# not touch the network: the disease reference, the model card and health.
LAZY_MODEL = _flag("MANOMETRY_LAZY_MODEL", True)
