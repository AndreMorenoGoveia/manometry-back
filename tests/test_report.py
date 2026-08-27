"""Reading swallow panels out of a laudo, and refusing everything else."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from app.report import Panel, ReportError, crop_panel, extract_rasters, read_panels, read_swallows

from app.sample import build_docx, render_logo, render_panel, render_placeholder


def _topography(height: int = 480, width: int = 330, seed: int = 0) -> np.ndarray:
    """A synthetic pressure plot: dense colour, nothing near white."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    band = np.exp(-((y - 0.65) ** 2) / 0.01) * np.exp(-((x - 0.5) ** 2) / 0.08)
    rgb = np.stack([band, np.clip(1.4 - 2.2 * band, 0, 1) * 0.6, 1.0 - band], axis=-1)
    rgb = np.clip(rgb + rng.normal(0, 0.01, rgb.shape), 0, 1)
    return (rgb * 200).astype(np.uint8)


def _docx(images) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        path = build_docx(list(images), Path(folder) / "l.docx")
        return path.read_bytes()


# ------------------------------------------------------------------ cropping


def test_crop_removes_the_chrome_exactly():
    plot = _topography()
    panel = np.asarray(render_panel(plot, 1))
    cropped, status = crop_panel(panel)
    assert status == "accepted"
    # The cropper insets two pixels off the detected box to drop the border line.
    assert cropped.shape[0] <= plot.shape[0] and cropped.shape[1] <= plot.shape[1]
    assert plot.shape[0] - cropped.shape[0] <= 4
    assert plot.shape[1] - cropped.shape[1] <= 4


def test_placeholder_panel_is_rejected():
    cropped, status = crop_panel(np.asarray(render_placeholder()))
    assert cropped is None and status == "placeholder"


def test_off_spec_width_is_accepted_and_flagged():
    plot = _topography(width=300)
    panel = np.asarray(render_panel(plot, 1))
    narrow = panel[:, :380]          # no longer the 420 px Multiplex panel
    cropped, status = crop_panel(narrow)
    assert status == "accepted_off_spec"
    assert cropped is not None and cropped.size


def test_a_panel_too_small_to_hold_a_plot_is_not_localizable():
    tiny = np.full((30, 30, 3), 40, dtype=np.uint8)   # dark, but no 40px run to find
    cropped, status = crop_panel(tiny)
    assert cropped is None and status == "not_localizable"


def test_greyscale_array_is_unreadable():
    cropped, status = crop_panel(np.zeros((50, 50), dtype=np.uint8))
    assert cropped is None and status == "unreadable"


# --------------------------------------------------------------------- docx


def test_reads_panels_in_document_order():
    plots = [_topography(seed=index) for index in range(3)]
    data = _docx([render_logo()] + [render_panel(plot, i + 1) for i, plot in enumerate(plots)])
    panels = read_panels(data, "l.docx")
    assert [panel.status for panel in panels] == ["logo", "accepted", "accepted", "accepted"]
    assert [panel.position for panel in panels] == [1, 2, 3, 4]


def test_recovered_swallows_are_the_original_pixels():
    plot = _topography(seed=7)
    data = _docx([render_logo(), render_panel(plot, 1)])
    swallows, _ = read_swallows(data, "l.docx")
    assert len(swallows) == 1
    recovered = swallows[0]
    height, width = recovered.shape[:2]
    best = min(
        np.abs(plot[dy : dy + height, dx : dx + width].astype(int) - recovered.astype(int)).mean()
        for dy in range(3)
        for dx in range(3)
        if plot.shape[0] - dy >= height and plot.shape[1] - dx >= width
    )
    assert best == 0.0


def test_placeholders_and_logo_do_not_reach_the_model():
    data = _docx([render_logo(), render_panel(_topography(), 1), render_placeholder()])
    swallows, panels = read_swallows(data, "l.docx")
    assert len(swallows) == 1
    assert {panel.status for panel in panels} == {"logo", "accepted", "placeholder"}


def test_a_report_without_usable_panels_is_rejected():
    data = _docx([render_logo(), render_placeholder()])
    with pytest.raises(ReportError):
        read_swallows(data, "l.docx")


def test_media_is_still_found_when_relationships_are_unreadable():
    """Word files in the wild sometimes carry ids the document body never cites."""
    data = _docx([render_panel(_topography(), 1), render_panel(_topography(seed=2), 2)])
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(buffer, "w") as target:
        for item in source.namelist():
            payload = source.read(item)
            if item == "word/document.xml":
                payload = payload.replace(b"rId11", b"rIdMissing").replace(b"rId12", b"rIdGone")
            target.writestr(item, payload)
    panels = read_panels(buffer.getvalue(), "l.docx")
    assert [panel.status for panel in panels] == ["accepted", "accepted"]


def test_media_order_is_numeric_not_lexicographic():
    """In the fallback path `image10` must still come after `image9`."""
    # Each panel gets its own height, so the decoded order is identifiable.
    images = [render_panel(_topography(height=400 + 10 * index), index) for index in range(12)]
    expected = [image.size for image in images]
    data = _docx(images)

    stripped = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(stripped, "w") as target:
        for item in source.namelist():
            payload = source.read(item)
            if item == "word/_rels/document.xml.rels":
                # No resolvable relationship: the reader must fall back to media order.
                payload = (
                    b'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org'
                    b'/package/2006/relationships"/>'
                )
            target.writestr(item, payload)

    rasters = extract_rasters(stripped.getvalue(), "l.docx")
    assert [Image.open(io.BytesIO(raw)).size for raw in rasters] == expected


def test_one_image_referenced_twice_is_one_panel():
    """A laudo that repeats a figure must not report the swallow twice."""
    data = _docx([render_panel(_topography(), 1)])
    duplicated = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(duplicated, "w") as target:
        for item in source.namelist():
            payload = source.read(item)
            if item == "word/document.xml":
                start = payload.index(b"<w:p><w:r><w:drawing>")
                end = payload.index(b"</w:drawing></w:r></w:p>") + len(b"</w:drawing></w:r></w:p>")
                block = payload[start:end]
                payload = payload[:end] + block + payload[end:]
            target.writestr(item, payload)
    panels = read_panels(duplicated.getvalue(), "l.docx")
    assert len(panels) == 1 and panels[0].status == "accepted"


# ------------------------------------------------------------------ refusals


def test_bad_zip_is_reported():
    with pytest.raises(ReportError):
        extract_rasters(b"PK\x03\x04 not really a zip", "l.docx")


def test_docx_without_document_xml_is_reported():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(ReportError):
        extract_rasters(buffer.getvalue(), "l.docx")


def test_unknown_format_is_reported():
    with pytest.raises(ReportError):
        extract_rasters(b"%PDF-1.7", "laudo.pdf")


def test_rtf_is_recognised_by_signature():
    # No images inside, but it must take the RTF branch rather than refuse.
    assert extract_rasters(rb"{\rtf1\ansi no images here}", "laudo.rtf") == []


def test_panel_accepted_property():
    assert Panel(1, 420, 600, "accepted", None).accepted
    assert Panel(1, 420, 600, "accepted_off_spec", None).accepted
    assert not Panel(1, 420, 600, "placeholder", None).accepted


# ------------------------------------------------------------------ hardening


def test_content_wins_over_the_extension():
    """A laudo re-saved as .rtf but named .docx must still be read."""
    assert extract_rasters(rb"{\rtf1\ansi}", "laudo.docx") == []


def test_a_docx_holding_an_absurd_image_is_refused():
    import zipfile as zf

    data = _docx([render_panel(_topography(), 1)])
    inflated = io.BytesIO()
    with zf.ZipFile(io.BytesIO(data)) as source, zf.ZipFile(inflated, "w", zf.ZIP_DEFLATED) as target:
        for item in source.namelist():
            payload = source.read(item)
            if item.startswith("word/media/"):
                # Highly compressible, enormous when inflated: the classic bomb.
                payload = b"\x89PNG\r\n\x1a\n" + b"\0" * (40 * 1024 * 1024)
            target.writestr(item, payload)
    assert len(inflated.getvalue()) < 1024 * 1024   # small on the wire
    with pytest.raises(ReportError):
        extract_rasters(inflated.getvalue(), "l.docx")
