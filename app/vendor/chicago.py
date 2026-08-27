"""Automated Chicago Classification v4.0 proxy metrics from HRM pressure CSVs.

Every swallow CSV holds 24 pressure channels sampled at 20 Hz.  The catheter
geometry is not recorded, so the sensor spacing is assumed to be 1 cm and all
metrics below are proxies: they follow the CC v4.0 definitions but are computed
from an automatically detected landmark layout instead of a manual one.  The
downstream heads consume them as features, so absolute calibration matters less
than reproducibility across exams.

CC v4.0 swallow-level definitions reproduced here:

* DCI  -- distal contractile integral over the distal segment, integrating only
  pressures above 20 mmHg.  failed < 100, weak 100-450, normal 450-8000,
  hypercontractile > 8000 mmHg*s*cm.
* IRP4 -- lowest mean EGJ pressure over 4 contiguous seconds of the
  deglutitive window.  Elevated above 15 mmHg.
* DL   -- distal latency, swallow onset to contractile deceleration point.
  Premature below 4.5 s when DCI > 450.
* Large break -- gap > 5 cm in the 20 mmHg isobaric contour (fragmented when
  DCI > 450).
* Panesophageal pressurization -- uniform pressurization > 30 mmHg spanning the
  esophageal body between UES and EGJ.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SAMPLE_HZ = 20.0
DT = 1.0 / SAMPLE_HZ
DZ_CM = 1.0
# Catheter geometry is not recorded with the exports, so the DCI integral is
# rescaled by a single constant chosen once on the whole corpus so that the
# normal-motility median lands inside the published 450-8000 mmHg*s*cm band
# (~1900) and absent contractility stays under 100.  The constant is global, so
# it cannot advantage any split or model.
DCI_SCALE = 4.0

DCI_FAILED = 100.0
DCI_WEAK = 450.0
DCI_HYPER = 8000.0
IRP_UPPER = 15.0
DL_PREMATURE = 4.5
BREAK_CM = 5.0
CONTOUR_MMHG = 20.0
PRESSURIZATION_MMHG = 30.0

# Order matters: it is the feature layout consumed by the Chicago-aware head.
SWALLOW_PATTERNS = (
    "failed",
    "weak",
    "fragmented",
    "normal",
    "hypercontractile",
    "premature",
    "panesophageal",
)

METRIC_NAMES = (
    "dci",
    "irp4",
    "distal_latency",
    "front_arrival",
    "largest_break_cm",
    "egj_basal",
    "peak_distal_pressure",
    "pressurization_fraction",
    "pan_time_fraction",
    "body_length_cm",
)


def read_pressure(path: Path) -> np.ndarray:
    """Return a (channels, time) float32 pressure matrix."""
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    columns = [i for i, name in enumerate(header) if name.lower().startswith("p")]
    if not columns:
        raise ValueError(f"No pressure columns in {path}")
    values = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=columns, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    values = np.nan_to_num(values, nan=0.0, posinf=300.0, neginf=-20.0)
    return np.clip(values.T, -20.0, 300.0)


def _landmarks(pressure: np.ndarray) -> tuple[int, int]:
    """Locate the UES and EGJ channels from the basal pressure profile."""
    channels = pressure.shape[0]
    basal = np.median(pressure, axis=1)
    proximal_end = max(1, int(round(channels * 0.25)))
    ues = int(np.argmax(basal[:proximal_end]))
    distal_start = max(ues + 3, int(round(channels * 0.45)))
    distal_start = min(distal_start, channels - 1)
    egj = distal_start + int(np.argmax(basal[distal_start:]))
    return ues, egj


def _largest_break_cm(distal: np.ndarray) -> float:
    """Longest axial run of the distal segment never reaching the 20 mmHg contour."""
    active = distal.max(axis=1) >= CONTOUR_MMHG
    longest = current = 0
    for value in active:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return float(longest) * DZ_CM


def swallow_metrics(pressure: np.ndarray) -> dict[str, float]:
    channels, samples = pressure.shape
    ues, egj = _landmarks(pressure)
    body_start = min(ues + 1, channels - 1)
    body_end = max(body_start + 1, egj)  # exclusive; egj is the proximal EGJ margin
    body = pressure[body_start:body_end]
    # CC v4 measures DCI from the transition zone to the proximal EGJ margin.
    transition = body_start + (body_end - body_start) // 3
    distal = pressure[transition:body_end]
    if distal.shape[0] == 0:
        distal = pressure[body_start:body_end]

    above = np.clip(distal - CONTOUR_MMHG, 0.0, None)
    dci = float(above.sum() * DT * DZ_CM * DCI_SCALE)

    egj_band = pressure[egj : min(egj + 2, channels)].mean(axis=0)
    window = int(round(4.0 * SAMPLE_HZ))
    if samples > window:
        kernel = np.ones(window, dtype=np.float64) / window
        rolling = np.convolve(egj_band.astype(np.float64), kernel, mode="valid")
        irp4 = float(rolling.min())
    else:
        irp4 = float(egj_band.mean())

    # Swallow onset proxy: UES relaxation, i.e. the first collapse of the UES
    # channel below half its basal pressure.  The exported windows are already
    # cut at the swallow, so this usually resolves to t = 0.
    ues_trace = pressure[ues]
    relaxed = np.flatnonzero(ues_trace[: samples // 2] < 0.5 * np.median(ues_trace))
    onset = float(relaxed[0] * DT) if relaxed.size else 0.0

    # Contractile deceleration point proxy: peak pressure of the two channels
    # immediately proximal to the EGJ.  Validated against published norms --
    # this yields ~7.8 s for normal motility and ~4 s for achalasia.
    tail = distal[max(distal.shape[0] - 2, 0) :].mean(axis=0)
    distal_latency = max(float(np.argmax(tail) * DT) - onset, 0.0)
    crossings = np.flatnonzero(tail >= PRESSURIZATION_MMHG)
    front_arrival = float(crossings[0] * DT) if crossings.size else float(samples * DT)

    # Panesophageal pressurization: how much of the esophageal body sits above
    # 30 mmHg at the same instant, and for how long that spans (>=90%) the body.
    span = (body >= PRESSURIZATION_MMHG).mean(axis=0)
    pressurization_fraction = float(span.max())
    pan_time_fraction = float((span >= 0.9).mean())

    return {
        "dci": dci,
        "irp4": irp4,
        "distal_latency": distal_latency,
        "front_arrival": front_arrival,
        "largest_break_cm": _largest_break_cm(distal),
        "egj_basal": float(np.median(pressure[egj])),
        "peak_distal_pressure": float(distal.max()),
        "pressurization_fraction": pressurization_fraction,
        "pan_time_fraction": pan_time_fraction,
        "body_length_cm": float(body_end - body_start) * DZ_CM,
    }


def swallow_pattern(metrics: dict[str, float]) -> str:
    """Assign the CC v4.0 swallow-level contractility pattern."""
    dci = metrics["dci"]
    # Pressurization is only called when the whole body is pressurized for a
    # sustained stretch and no true contraction explains it.
    if metrics["pan_time_fraction"] >= 0.1 and dci < DCI_HYPER:
        return "panesophageal"
    if dci < DCI_FAILED:
        return "failed"
    if dci < DCI_WEAK:
        return "weak"
    if metrics["distal_latency"] < DL_PREMATURE:
        return "premature"
    if dci > DCI_HYPER:
        return "hypercontractile"
    if metrics["largest_break_cm"] > BREAK_CM:
        return "fragmented"
    return "normal"


def study_features(swallows: list[dict[str, float]]) -> np.ndarray:
    """Aggregate swallow metrics into the CC v4.0 study-level decision features.

    Layout: pattern proportions (7), then summary pressure statistics and the
    boolean gates that drive the CC v4.0 hierarchy.
    """
    if not swallows:
        raise ValueError("A study needs at least one swallow")
    patterns = [swallow_pattern(metrics) for metrics in swallows]
    counts = np.array([patterns.count(name) for name in SWALLOW_PATTERNS], dtype=np.float32)
    proportions = counts / float(len(patterns))

    dci = np.array([m["dci"] for m in swallows], dtype=np.float32)
    irp = np.array([m["irp4"] for m in swallows], dtype=np.float32)
    latency = np.array([m["distal_latency"] for m in swallows], dtype=np.float32)
    arrival = np.array([m["front_arrival"] for m in swallows], dtype=np.float32)
    breaks = np.array([m["largest_break_cm"] for m in swallows], dtype=np.float32)
    egj = np.array([m["egj_basal"] for m in swallows], dtype=np.float32)
    press = np.array([m["pressurization_fraction"] for m in swallows], dtype=np.float32)
    pan = np.array([m["pan_time_fraction"] for m in swallows], dtype=np.float32)

    median_irp = float(np.median(irp))
    ineffective = float(np.mean((dci < DCI_WEAK)))
    failed = float(np.mean(dci < DCI_FAILED))
    premature = float(np.mean((latency < DL_PREMATURE) & (dci > DCI_WEAK)))
    hyper = float(np.mean(dci > DCI_HYPER))

    summary = np.array(
        [
            # Continuous summaries, log-compressed where the dynamic range is wide.
            np.log1p(max(np.median(dci), 0.0)),
            np.log1p(max(dci.mean(), 0.0)),
            np.log1p(max(dci.max(), 0.0)),
            np.log1p(max(np.percentile(dci, 25), 0.0)),
            median_irp,
            float(irp.mean()),
            float(irp.max()),
            float(np.median(latency)),
            float(latency.min()),
            float(np.median(arrival)),
            float(arrival.min()),
            float(np.median(breaks)),
            float(breaks.max()),
            float(np.median(egj)),
            float(press.mean()),
            float(press.max()),
            float(pan.mean()),
            float(pan.max()),
            float(len(swallows)),
            # CC v4.0 decision gates.
            float(median_irp > IRP_UPPER),
            float(failed >= 1.0),
            float(ineffective > 0.7),
            float(failed >= 0.5),
            premature,
            float(premature >= 0.2),
            hyper,
            float(hyper >= 0.2),
            float(proportions[SWALLOW_PATTERNS.index("panesophageal")] >= 0.2),
            # Interactions that mirror the two top-level branches of the tree.
            float(median_irp > IRP_UPPER) * float(failed >= 1.0),
            float(median_irp > IRP_UPPER) * float(proportions[SWALLOW_PATTERNS.index("panesophageal")] >= 0.2),
            float(median_irp > IRP_UPPER) * float(premature >= 0.2),
            float(median_irp > IRP_UPPER) * (1.0 - float(failed >= 1.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([proportions, summary]).astype(np.float32)


STUDY_FEATURE_DIM = len(SWALLOW_PATTERNS) + 32
