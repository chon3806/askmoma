"""Tests for evaluation/evaluate.py, using made-up pages shaped like moma.org work pages."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from evaluation import evaluate as ev

ON_VIEW_PAGE = """<html><head>
<script type="application/ld+json">{"@type": "VisualArtwork", "name": "Ana Lind. The Quiet Harbor at Night. 1921",
 "artist": [{"@type": "Person", "name": "Ana Lind"}]}</script>
<script type="application/ld+json">{"@type": "WebPage"}</script></head><body>
<div id="artwork-status-row"><a id="artwork-on-view" href="/x">On view</a></div>
<div id="artwork-location"><span id="artwork-floor">
  MoMA, Floor 5, <a id="artwork-gallery-link" href="/map?location=501">501</a><br>
</span><span id="artwork-donor-title">Some Galleries</span></div>
</body></html>"""

NOT_ON_VIEW_PAGE = """<html><head>
<script type="application/ld+json">{"@type": "VisualArtwork", "name": "Hugo Brant. Drip Study. 1950",
 "artist": [{"@type": "Person", "name": "Hugo Brant"}]}</script></head><body>
<div id="artwork-status-row"><p id="artwork-not-on-view">Not on view</p></div></body></html>"""


def test_parse_on_view_page():
    page = ev.parse_work_page(ON_VIEW_PAGE)
    assert page.name == "Ana Lind. The Quiet Harbor at Night. 1921"
    assert page.artists == ["Ana Lind"]
    assert page.status == "on view"
    assert page.location == "MoMA, Floor 5, 501"  # nested link and <br> handled, donor line excluded


def test_parse_not_on_view_page():
    page = ev.parse_work_page(NOT_ON_VIEW_PAGE)
    assert (page.status, page.location, page.artists) == ("not on view", "", ["Hugo Brant"])


def test_page_without_status_is_unknown():
    assert ev.parse_work_page("<html><body>Page not found</body></html>").status == "unknown"


@pytest.mark.parametrize(
    ("link", "url"),
    [
        ("http://www.moma.org/collection/works/79802", "https://www.moma.org/collection/works/79802"),
        ("https://www.moma.org/collection/works/79802/", "https://www.moma.org/collection/works/79802"),
        ("https://www.moma.org/artists/2206", None),
        ("https://evil.example/collection/works/79802", None),
        ("https://www.moma.org/collection/works/abc", None),
    ],
)
def test_work_page_url_accepts_only_moma_work_pages(link, url):
    assert ev.work_page_url(link) == url


@pytest.mark.parametrize(
    ("on_view", "floor", "gallery", "page", "ok"),
    [
        (True, 5, "501", ON_VIEW_PAGE, True),
        (True, 5, "502", ON_VIEW_PAGE, False),  # moved gallery
        (False, None, None, ON_VIEW_PAGE, False),  # AskMoMA says not on view, moma.org says it is
        (False, None, None, NOT_ON_VIEW_PAGE, True),
        (True, 4, "403", NOT_ON_VIEW_PAGE, False),  # taken off view since the data was pulled
    ],
)
def test_location_matches(on_view, floor, gallery, page, ok):
    assert ev._location_matches(on_view, floor, gallery, ev.parse_work_page(page)) is ok


def test_saved_pages_folder_is_used_without_network(tmp_path):
    (tmp_path / "79802.html").write_text(ON_VIEW_PAGE, encoding="utf-8")
    pages = ev.MomaPages(tmp_path, offline=True)
    assert "Quiet Harbor" in pages.get("https://www.moma.org/collection/works/79802")
    with pytest.raises(FileNotFoundError):
        pages.get("https://www.moma.org/collection/works/1")


def test_stops_asking_after_the_site_refuses(tmp_path, monkeypatch):
    calls = []

    def refuse(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(ev.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(ev, "REQUEST_GAP_SECONDS", 0)
    pages = ev.MomaPages(tmp_path)
    for oid in (1, 2, 3):
        with pytest.raises(urllib.error.HTTPError):
            pages.get(f"https://www.moma.org/collection/works/{oid}")
    assert len(calls) == 1


def test_report_separates_wrong_from_not_checked(tmp_path):
    right = ev.Result(
        "q1", "location", "Where is A?", "A (X)", right_work=True, link_ok=True, location_ok=True
    )
    wrong = ev.Result(
        "q2", "location", "Where is B?", "B (Y)", right_work=True, link_ok=True, location_ok=False
    )
    unchecked = ev.Result(
        "q3", "location", "Where is C?", "C (Z)", right_work=True, note="not checked: HTTP 403"
    )
    assert (right.passed, wrong.passed, unchecked.passed) == (True, False, False)
    assert not unchecked.checked
    ev.write_report([right, wrong, unchecked], {"date": "2026-11-21"}, tmp_path / "report")
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "1/2 checked answers fully correct" in md
    assert "1 of 3 answers could not be checked" in md
    assert "not checked" in md
    assert (tmp_path / "report.csv").read_text(encoding="utf-8").count("\n") == 4


def test_question_set_is_well_formed():
    questions = json.loads((Path(ev.QUESTIONS)).read_text(encoding="utf-8"))
    assert len(questions) == 20
    assert len({q["id"] for q in questions}) == 20
    for q in questions:
        assert set(q) == {"id", "scenario", "question", "title", "artist"}
        assert all(isinstance(v, str) and v for v in q.values())
