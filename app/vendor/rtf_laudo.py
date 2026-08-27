"""Parse a Multiplex esophageal-manometry laudo (.rtf).

Two things are pulled from each report:

* the diagnostic text (Nota / Conclusão), used to derive the disease label;
* the embedded PNG images. The report starts with the hospital logo (a
  landscape image); the 10 water-swallow pressure topographies follow, each a
  portrait image. We keep the portrait PNGs, in document order, as the swallows.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from striprtf.striprtf import rtf_to_text

# A run of >=800 hex-ish chars is an embedded-image payload. Blanking these out
# before striprtf turns a 17 MB parse into a few-KB one.
_HEX_BLOB = re.compile(rb"[0-9a-fA-F\r\n\t ]{800,}")
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNGBLIP = re.compile(rb"\\pngblip")
# The hex-encoded PNG signature. Anchoring on this (rather than "first long hex
# run") avoids swallowing the trailing digits of a preceding \pichgoalNNNN
# control word, which would shift the payload's byte parity and corrupt it.
_PNG_SIG_HEX = re.compile(rb"89504e470d0a1a0a", re.I)


@dataclass
class Laudo:
    path: Path
    text: str
    nota: str | None            # the report's own "Nota:" block, if present
    conclusao: str | None       # the Conclusão block, whitespace-collapsed
    swallows: list[bytes] = field(default_factory=list)  # portrait PNG bytes


def extract_text(data: bytes) -> str:
    """Plain text of an RTF, with image payloads stripped for speed."""
    stripped = _HEX_BLOB.sub(b" ", data)
    try:
        return rtf_to_text(stripped.decode("latin-1"), errors="ignore")
    except Exception:  # noqa: BLE001 - striprtf can choke on odd control words
        return ""


def _png_dimensions(raw: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR, or None if not a PNG."""
    if not raw.startswith(_PNG_SIG) or len(raw) < 24:
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


def extract_pngs(data: bytes) -> list[bytes]:
    """All embedded PNGs, in document order."""
    out: list[bytes] = []
    for m in _PNGBLIP.finditer(data):
        sig = _PNG_SIG_HEX.search(data, m.end())
        if not sig:
            continue
        end = data.find(b"}", sig.start())
        if end == -1:
            continue
        hexdata = re.sub(rb"[^0-9a-fA-F]", b"", data[sig.start():end])
        if len(hexdata) % 2:  # defend against a stray trailing nibble
            hexdata = hexdata[:-1]
        try:
            raw = bytes.fromhex(hexdata.decode("ascii"))
        except ValueError:
            continue
        if raw.startswith(_PNG_SIG):
            out.append(raw)
    return out


def select_swallows(pngs: list[bytes]) -> list[bytes]:
    """Keep portrait PNGs (the swallow topographies), dropping the landscape
    hospital logo. Order is preserved so index 0 == swallow 1."""
    swallows = []
    for raw in pngs:
        dims = _png_dimensions(raw)
        if dims is None:
            continue
        w, h = dims
        if h > w:  # portrait -> swallow panel; logo is landscape
            swallows.append(raw)
    return swallows


_CONCLUSAO_HEAD = re.compile(r"Conclus[aã]o\s*:?", re.I)
_NOTA_BLOCK = re.compile(r"\bNotas?\s*:\s*(.+?)(?:\||Abrevia[cç]|$)", re.S | re.I)


def _find_nota(text: str) -> str | None:
    """The report's own "Nota:" block, which follows the Conclusão.

    Anchored *after* the Conclusão heading on purpose: the clinical-history
    section routinely carries the referring physician's own wording ("paciente
    com diagnóstico de acalasia"), and a document-wide search lands on that
    instead of this exam's finding — silently attributing another doctor's
    (often less specific) diagnosis to this exam.
    """
    head = _CONCLUSAO_HEAD.search(text)
    m = _NOTA_BLOCK.search(text, head.end() if head else 0)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:600] if m else None


def _find_conclusao(text: str) -> str | None:
    m = re.search(r"Conclus[aã]o\s*:?(.+?)(?:\bNotas?\s*:|Abrevia|$)", text, re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:600] if m else None


def parse(path: Path, *, load_images: bool = True) -> Laudo:
    data = path.read_bytes()
    text = extract_text(data)
    swallows = select_swallows(extract_pngs(data)) if load_images else []
    return Laudo(
        path=path,
        text=text,
        nota=_find_nota(text),
        conclusao=_find_conclusao(text),
        swallows=swallows,
    )
