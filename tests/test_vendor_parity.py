"""The vendored modules must stay byte-identical to their upstream originals.

Inference reproduces training only because the cropper, the Chicago metrics and
the fusion head here are literally the same code that built the corpus and
trained the weights.  If someone edits a copy in place, this fails.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import SETTINGS

VENDOR = Path(__file__).resolve().parent.parent / "app" / "vendor"
UNIFIER = SETTINGS.classification_root.parent / "manometry-data-unifier"

ORIGINS = {
    "chicago.py": SETTINGS.classification_root / "experiments" / "chicago.py",
    "fusion_models.py": SETTINGS.classification_root / "experiments" / "fusion_models.py",
    "image_chicago.py": SETTINGS.classification_root / "experiments" / "image_chicago.py",
    "swallow_crop.py": UNIFIER / "unifier" / "images.py",
    "rtf_laudo.py": UNIFIER / "unifier" / "rtf.py",
}

# The one edit `scripts/sync_vendor.sh` applies, because these files live in a
# package here and are flat scripts upstream.
REWRITES = {"image_chicago.py": ("from chicago import", "from .chicago import")}


@pytest.mark.parametrize("name", sorted(ORIGINS))
def test_vendored_file_matches_upstream(name: str):
    origin = ORIGINS[name]
    if not origin.is_file():
        pytest.skip(f"repositorio de origem indisponivel: {origin}")
    expected = origin.read_text()
    if name in REWRITES:
        old, new = REWRITES[name]
        expected = expected.replace(old, new)
    assert (VENDOR / name).read_text() == expected, (
        f"{name} divergiu de {origin}; rode scripts/sync_vendor.sh"
    )


def test_every_vendored_module_is_declared():
    present = {path.name for path in VENDOR.glob("*.py")} - {"__init__.py"}
    assert present == set(ORIGINS)
