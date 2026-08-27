"""Build a laudo `.docx` out of an exam folder of swallow topographies.

The corpus this project trains on is stored *already cropped* — the white UI
chrome of the Multiplex screenshot was removed when the dataset was built.  A
real report carries the uncropped panels, so this module puts the chrome back:
the mmHg colour bar on the left, the "Eventos" header on top and the time axis
below, padded to the 420 px panel width the exporter produces.

That makes it useful twice: `scripts/make_sample_docx.py` writes demo files with
it, and the test-suite feeds its output through the real upload path, so the
cropper is exercised rather than bypassed.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PANEL_WIDTH = 420          # width of a raw Multiplex swallow panel
RIGHT_MARGIN = 8
HEADER_HEIGHT = 26
AXIS_HEIGHT = 34
LEGEND_X = 6
LEGEND_WIDTH = 16
EMU_PER_PIXEL = 9525       # 914400 EMU per inch / 96 px per inch

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOCUMENT_HEAD = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>"""

DOCUMENT_TAIL = "</w:body></w:document>"


def _jet(values: np.ndarray) -> np.ndarray:
    """The same jet-like ramp the topography exports are rendered with."""
    values = np.asarray(values, dtype=np.float32)
    return np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0),
        ],
        axis=-1,
    )


def render_panel(crop: np.ndarray, number: int) -> Image.Image:
    """Re-frame a cropped topography as the raw 420 px screenshot it came from."""
    height, width = crop.shape[:2]
    left = max(LEGEND_X + LEGEND_WIDTH + 6, PANEL_WIDTH - RIGHT_MARGIN - width)
    panel_width = max(PANEL_WIDTH, left + width + RIGHT_MARGIN)
    panel_height = HEADER_HEIGHT + height + AXIS_HEIGHT

    canvas = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
    canvas.paste(Image.fromarray(np.ascontiguousarray(crop.astype(np.uint8))), (left, HEADER_HEIGHT))

    draw = ImageDraw.Draw(canvas)
    # A thin plot border, as the acquisition software draws it.
    draw.rectangle(
        [left - 1, HEADER_HEIGHT - 1, left + width, HEADER_HEIGHT + height],
        outline=(120, 120, 120),
    )

    # mmHg colour bar: a narrow block, separated from the plot by white, which is
    # exactly the geometry the cropper relies on to exclude it.
    bar = (_jet(np.linspace(1.0, 0.0, height)) * 255).astype(np.uint8)
    bar = np.repeat(bar[:, None, :], LEGEND_WIDTH, axis=1)
    canvas.paste(Image.fromarray(bar), (LEGEND_X, HEADER_HEIGHT))
    for offset in (0, height // 2, height - 1):
        draw.line(
            [LEGEND_X + LEGEND_WIDTH, HEADER_HEIGHT + offset, LEGEND_X + LEGEND_WIDTH + 3, HEADER_HEIGHT + offset],
            fill=(60, 60, 60),
        )

    draw.text((left, 8), f"Eventos - Degluticao {number}", fill=(40, 40, 40))
    axis_y = HEADER_HEIGHT + height + 6
    draw.line([left, axis_y, left + width, axis_y], fill=(60, 60, 60))
    for tick in range(0, width + 1, max(width // 6, 1)):
        draw.line([left + tick, axis_y, left + tick, axis_y + 4], fill=(60, 60, 60))
    draw.text((left, axis_y + 8), "0s      5s     10s", fill=(60, 60, 60))
    return canvas


def render_placeholder() -> Image.Image:
    """The "no data" frame the software emits for a swallow it did not capture."""
    canvas = Image.new("RGB", (PANEL_WIDTH, 480), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([20, 20, PANEL_WIDTH - 20, 460], outline=(200, 200, 200))
    draw.rectangle([196, 226, 224, 254], outline=(160, 160, 160))
    draw.line([196, 226, 224, 254], fill=(160, 160, 160))
    draw.text((150, 270), "sem dados", fill=(150, 150, 150))
    return canvas


def render_logo() -> Image.Image:
    """A landscape header image, which the reader must drop as the hospital logo."""
    canvas = Image.new("RGB", (620, 130), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 619, 129], outline=(210, 210, 210))
    draw.ellipse([24, 34, 86, 96], fill=(24, 78, 140))
    draw.text((110, 46), "SERVICO DE MOTILIDADE DIGESTIVA", fill=(24, 78, 140))
    draw.text((110, 70), "Manometria esofagica de alta resolucao", fill=(90, 90, 90))
    return canvas


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _drawing(relationship_id: str, number: int, size: tuple[int, int]) -> str:
    width, height = size[0] * EMU_PER_PIXEL, size[1] * EMU_PER_PIXEL
    return (
        f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{width}" cy="{height}"/>'
        f'<wp:docPr id="{number}" name="Imagem {number}"/>'
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="{number}" name="image{number}.png"/>'
        f"<pic:cNvPicPr/></pic:nvPicPr>"
        f'<pic:blipFill><a:blip r:embed="{relationship_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width}" cy="{height}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        f"</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    )


def _paragraph(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<w:p><w:r><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"


def _write_package(target, images: list[Image.Image], text: list[str] | None) -> None:
    """Write a minimal but valid WordprocessingML package holding ``images``."""
    body: list[str] = [_paragraph(line) for line in (text or [])]
    relationships: list[str] = []
    media: dict[str, bytes] = {}
    for number, image in enumerate(images, start=1):
        relationship_id = f"rId{number + 10}"
        name = f"image{number}.png"
        media[f"word/media/{name}"] = _png(image)
        relationships.append(
            f'<Relationship Id="{relationship_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/{name}"/>'
        )
        body.append(_drawing(relationship_id, number, image.size))

    document = DOCUMENT_HEAD + "".join(body) + DOCUMENT_TAIL
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + "</Relationships>"
    )

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        for name, payload in media.items():
            archive.writestr(name, payload)


def build_docx(images: list[Image.Image], destination: Path, *, text: list[str] | None = None) -> Path:
    """Write the package to ``destination`` and return the path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        _write_package(handle, images, text)
    return destination


def laudo_from_exam(
    exam_dir: Path, *, with_logo: bool = True, placeholders: int = 0, limit: int | None = None
) -> list[Image.Image]:
    swallows = sorted(exam_dir.glob("swallow_*.png"))
    if not swallows:
        raise SystemExit(f"Nenhuma imagem swallow_*.png em {exam_dir}")
    if limit:
        swallows = swallows[:limit]
    images: list[Image.Image] = [render_logo()] if with_logo else []
    for number, path in enumerate(swallows, start=1):
        with Image.open(path) as handle:
            images.append(render_panel(np.asarray(handle.convert("RGB")), number))
    images.extend(render_placeholder() for _ in range(placeholders))
    return images


def build_docx_bytes(images: list[Image.Image], *, text: list[str] | None = None) -> bytes:
    """The same package, written to memory instead of to disk."""
    buffer = io.BytesIO()
    _write_package(buffer, images, text)
    return buffer.getvalue()
