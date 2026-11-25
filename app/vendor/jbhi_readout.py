"""Readout features of the JBHI proposed model, extracted from manometry-classification.

Unlike the other vendored files this one is an extract, not a whole-file copy: the upstream modules import the
training and benchmarking stack. The functions are copied unchanged in their arithmetic from
`studies/cbm_2026/experiments/chicago_cardinality.py` (count_distribution, ineffective_joint, clinical_events in
'exact' mode, RULE_ORDER, rule_compatibility) and `jbhi_fusion.py` (masked_median, swallow_states,
readout_features). Everything runs in float64, as upstream does after `torch.set_default_dtype(torch.float64)`.
`tests/test_inference_parity.py` is what keeps this extract honest.
"""
import torch
from torch.nn import functional as F

IRP_UPPER = 16.0
S = 10  # swallows per examination
AUX_SIZES = (4, 1, 1, 1, 3)  # vigor (failed, weak, normal, hyper), premature, fragmented, pan, [log DCI, IRP, DL]
RULE_ORDER = ("absent_contractility", "achalasia_type_i", "achalasia_type_ii", "egj_outflow_obstruction",
              "hypercontractile_esophagus", "ineffective_esophageal_motility", "normal_motility")


def count_distribution(probability, mask):
    """Poisson-binomial PMF [B,C,S+1]; padding contributes zero trials."""
    p = probability.masked_fill(~mask[..., None], 0)
    pmf = p.new_zeros(p.shape[0], p.shape[2], p.shape[1] + 1)
    pmf[..., 0] = 1
    for i in range(p.shape[1]):
        q = p[:, i, :, None]
        pmf = pmf * (1 - q) + F.pad(pmf[..., :-1], (1, 0)) * q
    return pmf


def ineffective_joint(failed, ineffective, mask):
    """Joint PMF P(K_failed=k,K_ineffective=l), with failed subset ineffective."""
    failed = failed.masked_fill(~mask, 0)
    ineffective = ineffective.masked_fill(~mask, 0)
    b, s = mask.shape
    pmf = failed.new_zeros(b, s + 1, s + 1)
    pmf[:, 0, 0] = 1
    for i in range(s):
        f = failed[:, i, None, None]
        weak = (ineffective[:, i] - failed[:, i])[:, None, None]
        other = (1 - ineffective[:, i])[:, None, None]
        pmf = pmf * other + F.pad(pmf[:, :, :-1], (1, 0, 0, 0)) * weak + F.pad(pmf[:, :-1, :-1], (1, 0, 1, 0)) * f
    return pmf


def clinical_events(vigor, timing, pressurization, integrity, mask):
    """[all failed, >=50% failed, >70% ineffective, IEM union, >=20% premature/hyper/pan]."""
    vigor, timing, pressurization, integrity = [p.to(torch.float64) for p in
                                                (vigor, timing, pressurization, integrity)]
    failed, weak, intact, hyper = vigor.unbind(-1)
    ineffective = (failed + weak + (intact + hyper) * integrity[..., 1]).clamp(max=1)
    premature = (intact + hyper) * timing[..., 1]
    pan = pressurization[..., 1]
    n = mask.sum(1)
    half = (n + 1) // 2
    seventy = (7 * n) // 10 + 1
    twenty = (n + 4) // 5
    joint = ineffective_joint(failed, ineffective, mask)
    axis = torch.arange(mask.shape[1] + 1, device=mask.device)
    all_failed = joint.sum(-1).gather(1, n[:, None]).squeeze(1)
    fh = axis[None, :, None] >= half[:, None, None]
    ie = axis[None, None, :] >= seventy[:, None, None]
    fhalf = (joint * fh).sum((1, 2))
    isevent = (joint * ie).sum((1, 2))
    iem = (joint * (fh | ie)).sum((1, 2))
    pmf = count_distribution(torch.stack([premature, hyper, pan], -1), mask)
    more = (pmf * (axis[None, None, :] >= twenty[:, None, None])).sum(-1).unbind(-1)
    return torch.stack([all_failed, fhalf, isevent, iem, *more], -1).clamp(0, 1)


def rule_compatibility(events, median_irp):
    """Chicago v4.0 scores, one column per class in RULE_ORDER (dataset v4, Nov 2026)."""
    all_failed, _, _, iem, prem, hyper, pan = events.unbind(-1)
    elevated = torch.sigmoid((median_irp.to(events.dtype) - IRP_UPPER) / 3)
    normal_irp = 1 - elevated
    residual = normal_irp * (1 - all_failed)
    columns = {
        "absent_contractility": normal_irp * all_failed * (1 - pan),
        "achalasia_type_i": elevated * all_failed * (1 - pan),
        "achalasia_type_ii": all_failed * pan,
        "egj_outflow_obstruction": elevated * (1 - all_failed),
        "hypercontractile_esophagus": residual * hyper,
        "ineffective_esophageal_motility": residual * (1 - hyper) * iem,
        "normal_motility": residual * (1 - hyper) * (1 - iem),
    }
    return torch.stack([columns[name] for name in RULE_ORDER], -1)


def masked_median(x, mask):
    x = x.masked_fill(~mask, float('nan'))
    return torch.nanmedian(x, 1).values


def swallow_states(aux):
    """Per-swallow state probabilities from the measurement heads (unit temperatures)."""
    vigor, premature, fragmented, pan, regression = aux.split(AUX_SIZES, -1)
    two = lambda z: torch.cat([1 - torch.sigmoid(z), torch.sigmoid(z)], -1)
    return vigor.softmax(-1), two(premature), two(pan), two(fragmented), regression


def readout_features(aux, mask, diag_logp):
    """Examination features of the proposed readout: exact count-event probabilities of the supervised states,
    expected fractions, median and mean IRP, Chicago v4.0 rule log-compatibilities and the image path.

    aux [B,S,10], mask [B,S] bool, diag_logp [B,classes] (log of the mean swallow softmax)."""
    aux, diag_logp = aux.to(torch.float64), diag_logp.to(torch.float64)
    v, prem, pan, frag, reg = swallow_states(aux)
    events = clinical_events(v, prem, pan, frag, mask)
    irp = masked_median(reg[..., 1], mask)
    compat = rule_compatibility(events, irp).clamp_min(1e-6).log()
    n = mask.sum(1).to(torch.float64)
    mean = lambda p: (p * mask).sum(1) / n
    fractions = [mean(v[..., k]) for k in range(4)] + [mean(pan[..., 1]), mean(frag[..., 1]), mean(prem[..., 1])]
    irp_mean = (reg[..., 1] * mask).sum(1) / n
    return torch.cat([events, torch.stack(fractions + [irp, irp_mean, mean(reg[..., 0]), n / S], 1), compat,
                      diag_logp], 1)
