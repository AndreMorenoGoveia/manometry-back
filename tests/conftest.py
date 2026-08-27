"""Shared fixtures.

The tests that need a real examination read one out of the training corpus.
That corpus is patient data and is not part of this repository, so every test
that depends on it skips cleanly when the sibling checkout is absent, and the
suite still passes on a bare clone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from app.config import SETTINGS  # noqa: E402


def _first_exam(class_name: str) -> Path | None:
    for split in ("test", "val", "train"):
        candidates = sorted((SETTINGS.dataset_root / split / class_name).glob("exam_*"))
        for candidate in candidates:
            if len(list(candidate.glob("swallow_*.png"))) >= 8:
                return candidate
    return None


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    if not SETTINGS.dataset_root.is_dir():
        pytest.skip(f"corpus indisponivel em {SETTINGS.dataset_root}")
    return SETTINGS.dataset_root


@pytest.fixture(scope="session")
def sample_exam(dataset_root: Path) -> Path:
    exam = _first_exam("normal_motility") or _first_exam("achalasia_type_ii")
    if exam is None:
        pytest.skip("nenhum exame com 8+ degluticoes no corpus")
    return exam


@pytest.fixture(scope="session")
def sample_docx(sample_exam: Path, tmp_path_factory) -> Path:
    from app.sample import build_docx, laudo_from_exam

    destination = tmp_path_factory.mktemp("laudo") / "laudo.docx"
    return build_docx(laudo_from_exam(sample_exam, placeholders=1), destination)


@pytest.fixture(scope="session")
def model_ready() -> None:
    if not SETTINGS.model_available:
        pytest.skip(f"checkpoint indisponivel em {SETTINGS.checkpoint_path}")


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the tests that load the checkpoint and score real examinations",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: loads the checkpoint and scores real examinations")


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="precisa de --runslow (carrega o checkpoint)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
