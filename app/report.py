"""Pull the ten swallow topographies out of an HRM report file.

A Multiplex laudo is a word-processor document that opens with the hospital
logo (landscape) and then carries one portrait panel per water swallow.  Each
panel is a screenshot of the plot framed by white UI chrome (the mmHg legend on
the left, the "Eventos"/swallow-number header on top, the time axis below), and
the model was trained on those panels with the chrome cropped away.  So the
reader here does exactly three things, in the same order the training corpus was
built:

1. read every embedded raster in document order (`.docx` and `.rtf`);
2. keep the portrait ones, which is what separates swallows from the logo;
3. crop the chrome with the upstream cropper and drop "no data" placeholders.

Step 3 is :mod:`app.vendor.swallow_crop`, copied verbatim from the dataset
builder, so a panel reaching the encoder here is the same array the encoder was
trained on.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image, UnidentifiedImageError

from .vendor import rtf_laudo
from .vendor.swallow_crop import _INSET, _plot_bbox, _white_mask, clean_swallow

MAX_PANELS = 200
# A .docx is a zip, so a small upload can decompress into gigabytes. Every entry
# is checked against its declared uncompressed size before it is read, and the
# package as a whole against a budget.
MAX_MEDIA_BYTES = 32 * 1024 * 1024
MAX_TOTAL_MEDIA_BYTES = 256 * 1024 * 1024

DOCX_DOCUMENT = "word/document.xml"
DOCX_RELS = "word/_rels/document.xml.rels"

# `r:embed` covers the modern DrawingML pictures, `r:id` the legacy VML ones
# that older Multiplex exports still emit.  Scanning the raw XML rather than
# walking the tree is what preserves document order across both forms.
_RELATION_REF = re.compile(rb'r:(?:embed|id|link)="([^"]+)"')

_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


class ReportError(ValueError):
    """The upload is not a report file this service can read."""


@dataclass(frozen=True)
class Panel:
    """One embedded image and what became of it."""

    position: int              # 1-based order of appearance in the document
    width: int
    height: int
    status: str                # accepted | accepted_off_spec | logo | placeholder | not_localizable | unreadable
    image: np.ndarray | None   # the cropped topography, RGB uint8, when accepted

    @property
    def accepted(self) -> bool:
        return self.status.startswith("accepted")


# --------------------------------------------------------------- raw rasters


def _natural_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def _docx_media_order(archive: zipfile.ZipFile) -> list[str]:
    """Names of `word/media/*` entries in the order the document references them."""
    try:
        document = archive.read(DOCX_DOCUMENT)
        rels = archive.read(DOCX_RELS)
    except KeyError as error:
        raise ReportError("O arquivo .docx nao contem word/document.xml") from error

    targets: dict[str, str] = {}
    for relationship in ElementTree.fromstring(rels):
        identifier = relationship.get("Id")
        target = relationship.get("Target", "")
        if identifier and target:
            # Targets are relative to `word/`, and may be written "../media/x".
            cleaned = target.replace("\\", "/").lstrip("/")
            while cleaned.startswith("../"):
                cleaned = cleaned[3:]
            targets[identifier] = (
                cleaned if cleaned.startswith("media/") else f"media/{Path(cleaned).name}"
            )

    names = set(archive.namelist())
    ordered: list[str] = []
    seen: set[str] = set()
    for match in _RELATION_REF.finditer(document):
        target = targets.get(match.group(1).decode("ascii", "ignore"))
        if target is None:
            continue
        entry = f"word/{target}"
        # One image reused by several references is one panel, not several.
        if entry in names and entry not in seen:
            seen.add(entry)
            ordered.append(entry)

    if not ordered:
        # A document whose relationship ids we could not follow still has its
        # media; `image10` must sort after `image9`, hence the numeric key.
        ordered = sorted(
            (name for name in names if name.startswith("word/media/")), key=_natural_key
        )
    return ordered


def _read_docx(data: bytes) -> list[bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ReportError("Arquivo .docx invalido ou corrompido") from error
    with archive:
        rasters: list[bytes] = []
        budget = MAX_TOTAL_MEDIA_BYTES
        for name in _docx_media_order(archive)[:MAX_PANELS]:
            if Path(name).suffix.lower() not in _RASTER_SUFFIXES:
                continue
            declared = archive.getinfo(name).file_size
            if declared > MAX_MEDIA_BYTES or declared > budget:
                raise ReportError(
                    "Uma das imagens do laudo é grande demais para ser uma topografia de pressão."
                )
            budget -= declared
            rasters.append(archive.read(name))
        return rasters


def _read_rtf(data: bytes) -> list[bytes]:
    return rtf_laudo.extract_pngs(data)[:MAX_PANELS]


def extract_rasters(data: bytes, filename: str) -> list[bytes]:
    """Every embedded raster of the report, in document order.

    What the file *is* wins over what it is called: laudos get re-saved and
    renamed on the way to a browser, and an .rtf named .docx is far more common
    than a genuinely corrupt package.
    """
    if data[:4] == b"PK\x03\x04":
        return _read_docx(data)
    if data[:5] == b"{\\rtf":
        return _read_rtf(data)
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".docx":
        return _read_docx(data)        # raises with the right message
    if suffix == ".rtf":
        return _read_rtf(data)
    raise ReportError("Formato nao suportado: envie o laudo em .docx ou .rtf")


# ------------------------------------------------------------------ cropping


def crop_panel(rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Crop the white UI chrome off one panel, or explain why it was dropped.

    The first attempt is the upstream cropper unchanged, so an in-spec panel
    reaches the encoder as the exact array the training corpus holds.  Upstream
    additionally requires the 420 px width of the Multiplex screenshots it was
    built from, a corpus-hygiene rule since nothing else in that batch was a
    swallow.  Here the file comes from whoever exported it, so a panel that is
    not 420 px wide but still localizes cleanly is kept and flagged
    ``accepted_off_spec`` rather than silently discarded.
    """
    cleaned = clean_swallow(rgb)
    if cleaned is not None:
        return cleaned, "accepted"

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None, "unreadable"
    if float(_white_mask(rgb).mean()) > 0.5:
        return None, "placeholder"      # the acquisition software's "no data" frame
    box = _plot_bbox(rgb)
    if box is None:
        return None, "not_localizable"
    x0, y0, x1, y1 = box
    crop = rgb[y0 + _INSET : y1 - _INSET + 1, x0 + _INSET : x1 - _INSET + 1]
    return (crop, "accepted_off_spec") if crop.size else (None, "not_localizable")


def read_panels(data: bytes, filename: str) -> list[Panel]:
    """Every embedded image of the report, classified and cropped."""
    panels: list[Panel] = []
    for position, raw in enumerate(extract_rasters(data, filename), start=1):
        try:
            with Image.open(io.BytesIO(raw)) as handle:
                width, height = handle.size
                rgb = np.asarray(handle.convert("RGB"))
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
            panels.append(Panel(position, 0, 0, "unreadable", None))
            continue
        if height <= width:
            # The hospital logo is the landscape image; swallow panels are portrait.
            panels.append(Panel(position, width, height, "logo", None))
            continue
        image, status = crop_panel(rgb)
        panels.append(Panel(position, width, height, status, image))
    return panels


def read_swallows(data: bytes, filename: str) -> tuple[list[np.ndarray], list[Panel]]:
    """The usable swallow topographies of a report, plus the full panel audit."""
    panels = read_panels(data, filename)
    swallows = [panel.image for panel in panels if panel.accepted and panel.image is not None]
    if not swallows:
        raise ReportError(
            "Nenhuma degluticao valida foi encontrada no laudo. Verifique se o arquivo "
            "e o laudo de manometria com as imagens de topografia de pressao."
        )
    return swallows, panels
