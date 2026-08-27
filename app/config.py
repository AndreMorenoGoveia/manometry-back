"""Runtime configuration, all overridable by environment variable.

The trained checkpoint and the colour calibration are large, patient-derived
build outputs that live in ``manometry-classification`` and are deliberately not
committed here; the defaults point at a sibling checkout so a developer who has
both repositories side by side needs no configuration at all.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFICATION_ROOT = REPO_ROOT.parent / "manometry-classification"

# The image-only run of the leakage/fusion study: a ConvNeXt-Base (ImageNet-22k)
# encoder under the `chicago_set` study-level head, with the Chicago metrics
# recovered from the rendered topography instead of from the pressure CSVs, so
# that a report with nothing but images feeds the model what it was trained on.
# Seed 42, EMA weights of epoch 9, macro-F1 0.741 over 336 held-out examinations.
DEFAULT_CHECKPOINT = (
    "artifacts/leakage_fusion/fusion_heavy_image_deploy/chicago_set/"
    "convnext_base_in22k/seed_42/model.pt"
)
DEFAULT_RESULT = DEFAULT_CHECKPOINT.replace("model.pt", "result.json")
DEFAULT_PREDICTIONS = DEFAULT_CHECKPOINT.replace("model.pt", "test_predictions.npz")
DEFAULT_CALIBRATION = ".cache/image_chicago_calibration.json"


def _path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def _optional_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    predictions_path: Path
    calibration_path: Path
    dataset_root: Path
    reference_dir: Path
    sample_dir: Path
    device: str
    tta: bool | None
    torch_threads: int
    max_upload_bytes: int
    checkpoint_url: str = ""
    allowed_origins: list[str] = field(default_factory=list)
    allowed_origin_regex: str = ""

    @property
    def model_available(self) -> bool:
        return (
            self.checkpoint_path.is_file() or bool(self.checkpoint_url)
        ) and self.calibration_path.is_file()


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
        predictions_path=_path("MANOMETRY_PREDICTIONS", root / DEFAULT_PREDICTIONS),
        calibration_path=_path("MANOMETRY_CALIBRATION", root / DEFAULT_CALIBRATION),
        dataset_root=_path("MANOMETRY_DATASET_ROOT", root / "data"),
        reference_dir=_path("MANOMETRY_REFERENCE_DIR", REPO_ROOT / "assets" / "reference"),
        sample_dir=_path("MANOMETRY_SAMPLE_DIR", REPO_ROOT / "assets" / "samples"),
        device=os.environ.get("MANOMETRY_DEVICE", "auto"),
        # None means "do what the checkpoint's own run decided".
        tta=_optional_flag("MANOMETRY_TTA"),
        # Managed hosts report every core of the machine while giving the
        # container a fraction of one; letting torch spawn a thread per reported
        # core makes a forward pass slower, not faster.
        torch_threads=int(os.environ.get("MANOMETRY_TORCH_THREADS", "0")),
        max_upload_bytes=int(os.environ.get("MANOMETRY_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)),
        checkpoint_url=os.environ.get("MANOMETRY_CHECKPOINT_URL", "").strip(),
        allowed_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
        allowed_origin_regex=os.environ.get("MANOMETRY_ALLOWED_ORIGIN_REGEX", "").strip(),
    )


SETTINGS = load_settings()

# Torch is loaded lazily so that `import app` stays cheap for the routes that do
# not touch the network: the disease reference, the model card and health.
LAZY_MODEL = _flag("MANOMETRY_LAZY_MODEL", True)


def fetch_checkpoint(settings: Settings) -> Path:
    """Return the checkpoint, downloading it once if only a URL was configured.

    Platforms like Render build from a git checkout, and a 50-350 MB tensor file
    does not belong in git. Point ``MANOMETRY_CHECKPOINT_URL`` at a release asset
    and the first boot fetches it to ``MANOMETRY_CHECKPOINT`` (default: a path
    under the system temp dir, which survives for the life of the container).
    """
    destination = settings.checkpoint_path
    if destination.is_file():
        return destination
    if not settings.checkpoint_url:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download beside the target and move into place, so a boot interrupted
    # mid-transfer cannot leave a truncated checkpoint behind.
    handle, staging = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    os.close(handle)
    try:
        with urllib.request.urlopen(settings.checkpoint_url, timeout=300) as response:
            with open(staging, "wb") as target:
                shutil.copyfileobj(response, target, length=1 << 20)
        os.replace(staging, destination)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise
    return destination
