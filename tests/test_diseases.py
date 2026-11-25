"""The clinical reference has to line up with the network's output vector."""
from __future__ import annotations

import re

import pytest

from app.diseases import BY_SLUG, CLASS_ORDER, DISEASES, get
from app.schemas import METRIC_LABELS, SWALLOW_PATTERNS
from app.vendor.chicago import METRIC_NAMES, SWALLOW_PATTERNS as CHICAGO_PATTERNS
from app.vendor.jbhi_readout import RULE_ORDER

# The seven classes of the trained model, in the order of its output vector.
EXPECTED = RULE_ORDER


def test_covers_exactly_the_model_classes():
    assert CLASS_ORDER == EXPECTED


def test_every_swallow_pattern_has_a_label():
    assert set(SWALLOW_PATTERNS) == set(CHICAGO_PATTERNS)


def test_every_metric_has_a_label():
    assert set(METRIC_LABELS) == set(METRIC_NAMES)


@pytest.mark.parametrize("disease", DISEASES, ids=lambda d: d.slug)
def test_entry_is_complete(disease):
    assert disease.name and disease.short_name and disease.english
    assert disease.tagline and disease.summary and disease.topography and disease.management
    assert len(disease.criteria) >= 3
    assert len(disease.symptoms) >= 2
    assert len(disease.differential) >= 2
    assert disease.metric_signature
    assert disease.references
    assert disease.prevalence > 0


@pytest.mark.parametrize("disease", DISEASES, ids=lambda d: d.slug)
def test_accent_is_a_hex_colour(disease):
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", disease.accent)


def test_accents_are_distinct():
    assert len({disease.accent for disease in DISEASES}) == len(DISEASES)


def test_prevalence_matches_the_corpus():
    # 1035 train + 349 val + 358 test examinations of partition 42 fold 0.
    assert sum(disease.prevalence for disease in DISEASES) == 1742


def test_the_two_entries_that_need_a_caveat_carry_one():
    assert BY_SLUG["egj_outflow_obstruction"].caveat
    assert BY_SLUG["hypercontractile_esophagus"].caveat


def test_unknown_slug_raises():
    with pytest.raises(KeyError):
        get("gastroparesis")


def test_to_dict_is_json_shaped():
    payload = BY_SLUG["achalasia_type_ii"].to_dict()
    assert payload["slug"] == "achalasia_type_ii"
    assert isinstance(payload["criteria"], list)
