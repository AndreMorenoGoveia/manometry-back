"""Crop and validate swallow topography images for model training.

The raw swallow PNGs are screenshots of the Multiplex plot framed by white/grey
UI chrome: the mmHg color legend (left), the "Eventos"/swallow-number header
(top), and the time axis (bottom). `clean_swallow` crops that chrome away and
returns the pressure topography unmodified — no pixel is edited, so the plot data
is exactly as captured.

The chrome width is not constant across exams, so the plot box is found per
image: the topography is a large block of low-"white" pixels, while the legend,
header, axis, and the gaps between them are near-white. The widest contiguous
run of low-white columns (and rows) is the plot; the narrower legend/header
blocks are separated from it by those white gaps and so are excluded.

`is_placeholder` rejects the software's "no data" panel (a mostly-white frame
with a broken-image glyph) that appears for swallows with no captured data;
exams left with no valid swallow are dropped downstream.
"""
from __future__ import annotations

import numpy as np

STD_WIDTH = 420
_WHITE_THRESH = 230      # a pixel is "white" if all channels exceed this
_RUN_MAX_WHITE = 0.5     # a plot row/col is mostly non-white
_INSET = 2               # trim a couple px to drop the plot's thin border line


def _white_mask(rgb: np.ndarray) -> np.ndarray:
    return (rgb > _WHITE_THRESH).all(axis=2)


def is_placeholder(rgb: np.ndarray) -> bool:
    """True if the image is the 'no data' placeholder rather than a real plot.
    The placeholder is ~92% pure white; real topographies are <25% white."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return True
    white_frac = float(_white_mask(rgb).mean())
    return white_frac > 0.5 or rgb.shape[1] != STD_WIDTH


def _widest_run(profile: np.ndarray, thresh: float) -> tuple[int, int]:
    """Start/end (inclusive) of the longest contiguous run of profile < thresh."""
    best_s = best_e = -1
    best_len = 0
    cur_s = None
    n = len(profile)
    for i in range(n):
        if profile[i] < thresh:
            if cur_s is None:
                cur_s = i
            if i - cur_s + 1 > best_len:
                best_len = i - cur_s + 1
                best_s, best_e = cur_s, i
        else:
            cur_s = None
    return best_s, best_e


def _plot_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the topography, excluding all white UI chrome."""
    white = _white_mask(rgb)
    col_white = white.mean(axis=0)
    row_white = white.mean(axis=1)
    x0, x1 = _widest_run(col_white, _RUN_MAX_WHITE)
    y0, y1 = _widest_run(row_white, _RUN_MAX_WHITE)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    return x0, y0, x1, y1


def clean_swallow(rgb: np.ndarray) -> np.ndarray | None:
    """Return the topography with the white UI chrome cropped off (pixels
    otherwise untouched), or None if the image is a placeholder / can't be
    localized and should be dropped."""
    if is_placeholder(rgb):
        return None
    box = _plot_bbox(rgb)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    crop = rgb[y0 + _INSET:y1 - _INSET + 1, x0 + _INSET:x1 - _INSET + 1]
    return crop if crop.size else None
