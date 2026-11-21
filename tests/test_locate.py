"""Use case 1: locating a work. Run with: python -m pytest tests -q"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from askmoma import Collection  # noqa: E402


@pytest.fixture(scope="module")
def moma():
    return Collection()


def test_locates_the_starry_night(moma):
    answer = moma.locate("Where is The Starry Night?")
    assert "MoMA, Floor 5, Gallery 501**" in answer.text   # no stray quote from the source file
    assert "https://www.moma.org/collection/works/79802" in answer.text


def test_tolerates_a_typo_and_a_missing_article(moma):
    assert "Gallery 501" in moma.locate("where is starry nite").text


def test_says_so_when_nothing_matches(moma):
    answer = moma.locate("Where is the Mona Lisa?")
    assert answer.matches == () and "couldn't find" in answer.text
