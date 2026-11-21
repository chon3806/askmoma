"""Check AskMoMA's answers against moma.org and time them (the design document's Evaluation Plan).

    uv run python -m evaluation.evaluate                 # accuracy + timing, templated answers
    uv run python -m evaluation.evaluate --llm HuggingFaceTB/SmolLM2-360M-Instruct
    uv run python -m evaluation.evaluate --offline       # reuse the moma.org pages saved last time

For each question in questions.json, AskMoMA answers from the local data, and
the work page it links to on moma.org is opened and read. Three checks:

* right work:  the work AskMoMA answered about is the one the question names.
* link:        the moma.org page AskMoMA linked to is that same work.
* location:    AskMoMA's floor and gallery (or "not on view") match what the
               moma.org page says today.

moma.org's robots.txt allows /collection/works/; pages are fetched one per
second with an identifying User-Agent and saved in .cache/moma_pages/ so a
re-run does not fetch them again (use --refresh to fetch fresh copies).
Results go to evaluation/results/<date>[-llm].md and .csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from askmoma import AskMoMA, config, llm, load
from askmoma.data import norm, parse_on_view

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE / "results"
USER_AGENT = "AskMoMA-evaluation/0.1 (student project; one request per second)"
REQUEST_GAP_SECONDS = 1.0

log = logging.getLogger("evaluation")


# ---------------------------------------------------------------- moma.org pages
@dataclass
class WorkPage:
    """What a moma.org work page says about the work."""

    name: str = ""  # "Vincent van Gogh. The Starry Night. Saint Rémy, June 1889"
    artists: list[str] = field(default_factory=list)
    status: str = "unknown"  # "on view", "not on view", or "unknown"
    location: str = ""  # "MoMA, Floor 5, 501"


class _WorkPageParser(HTMLParser):
    """Collects the JSON-LD blocks and the text of the status and location elements."""

    WATCHED = ("artwork-on-view", "artwork-not-on-view", "artwork-floor")
    VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.json_ld: list[str] = []
        self.text: dict[str, str] = {}
        self._in_json_ld = False
        self._capturing: str | None = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("type") == "application/ld+json":
            self._in_json_ld = True
            self.json_ld.append("")
        elif self._capturing:
            self._depth += tag not in self.VOID
        elif attrs.get("id") in self.WATCHED:
            self._capturing, self._depth = attrs["id"], 0
            self.text.setdefault(self._capturing, "")

    def handle_endtag(self, tag):
        if tag == "script" and self._in_json_ld:
            self._in_json_ld = False
        elif self._capturing:
            if self._depth == 0:
                self._capturing = None
            else:
                self._depth -= 1

    def handle_data(self, data):
        if self._in_json_ld:
            self.json_ld[-1] += data
        elif self._capturing:
            self.text[self._capturing] += data


def parse_work_page(html: str) -> WorkPage:
    parser = _WorkPageParser()
    parser.feed(html)
    page = WorkPage()
    for block in parser.json_ld:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "VisualArtwork":
            page.name = data.get("name", "")
            page.artists = [a.get("name", "") for a in data.get("artist", []) if isinstance(a, dict)]
    if "artwork-on-view" in parser.text:
        page.status = "on view"
        page.location = " ".join(parser.text.get("artwork-floor", "").split())
    elif "artwork-not-on-view" in parser.text:
        page.status = "not on view"
    return page


def work_page_url(link: str) -> str | None:
    """The https moma.org work page for an AskMoMA link, or None if the link is not one."""
    u = urlparse(link)
    parts = u.path.strip("/").split("/")
    if u.scheme in ("http", "https") and u.netloc == "www.moma.org" and parts[:2] == ["collection", "works"]:
        if len(parts) == 3 and parts[2].isdigit():
            return f"https://www.moma.org/collection/works/{parts[2]}"
    return None


class MomaPages:
    """Fetches moma.org work pages politely and keeps a copy on disk."""

    def __init__(self, cache_dir: Path, offline: bool = False, refresh: bool = False):
        self.cache_dir, self.offline, self.refresh = cache_dir, offline, refresh
        self._last_request = 0.0
        self.fetched = 0
        self.refused: urllib.error.HTTPError | None = None  # once refused, stop asking

    def get(self, url: str) -> str:
        path = self.cache_dir / f"{url.rsplit('/', 1)[-1]}.html"
        if path.exists() and not self.refresh:
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"{path.name} is not in {self.cache_dir}")
        if self.refused is not None:
            raise self.refused
        time.sleep(max(0.0, self._last_request + REQUEST_GAP_SECONDS - time.monotonic()))
        self._last_request = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):  # the site said no: respect it for the rest of the run
                self.refused = exc
            raise
        self.fetched += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(html, encoding="utf-8")
        tmp.replace(path)
        return html


# ---------------------------------------------------------------- scoring
@dataclass
class Result:
    id: str
    scenario: str
    question: str
    expected: str
    answered: str = "(no single work)"
    askmoma_location: str = ""
    moma_location: str = ""
    right_work: bool = False
    link_ok: bool | None = None  # None: the moma.org page could not be checked
    location_ok: bool | None = None
    seconds: float = 0.0
    llm_seconds: float | None = None
    llm_rewrite_used: bool | None = None
    note: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.right_work and self.link_ok and self.location_ok)

    @property
    def checked(self) -> bool:
        return self.link_ok is not None


def _same_work(title: str, artist: str, want_title: str, want_artist: str) -> bool:
    return norm(title) == norm(want_title) and norm(want_artist) in norm(artist)


def _location_matches(on_view: bool, floor, gallery, page: WorkPage) -> bool:
    if page.status == "not on view":
        return not on_view
    if page.status != "on view" or not on_view:
        return False
    page_floor, page_gallery, _ = parse_on_view(page.location)
    return page_floor == floor and page_gallery == gallery


def evaluate_question(bot: AskMoMA, q: dict, pages: MomaPages, use_llm: bool) -> Result:
    r = Result(q["id"], q["scenario"], q["question"], f"{q['title']} ({q['artist']})")
    started = time.perf_counter()
    answer = bot.ask(q["question"])
    r.seconds = time.perf_counter() - started
    if use_llm:
        started = time.perf_counter()
        r.llm_rewrite_used = llm.polish(q["question"], answer.text) != answer.text
        r.llm_seconds = time.perf_counter() - started
    if answer.intent in ("disambiguate", "not_found") or not answer.cards:
        r.note = f"AskMoMA answered with intent '{answer.intent}'"
        return r

    card = answer.cards[0]
    works = bot.col.works
    floor, gallery = works.at[card["row"], "floor"], works.at[card["row"], "gallery"]
    r.answered = f"{card['title']} ({card['artist']})"
    r.askmoma_location = card["location"] if card["on_view"] else "not on view"
    r.right_work = _same_work(card["title"], card["artist"], q["title"], q["artist"])

    url = work_page_url(card["link"])
    if url is None:
        r.note = f"link is not a moma.org work page: {card['link']}"
        return r
    try:
        page = parse_work_page(pages.get(url))
    except urllib.error.HTTPError as exc:
        r.note = (
            f"not checked: moma.org answered HTTP {exc.code}; save the page from a browser and use --pages"
        )
        return r
    except (OSError, ValueError) as exc:
        r.note = f"not checked: {exc}"
        return r
    r.moma_location = page.location or page.status
    title_on_page = norm(card["title"]) in norm(page.name)
    artist_on_page = any(norm(a) and norm(a) in norm(card["artist"]) for a in page.artists)
    r.link_ok = title_on_page and artist_on_page
    r.location_ok = _location_matches(card["on_view"], floor, gallery, page)
    if not r.link_ok:
        r.note = f"moma.org page is '{page.name}'"
    elif page.status == "unknown":
        r.note = "moma.org page shows no on-view status"
    return r


# ---------------------------------------------------------------- report
def _yes(flag: bool) -> str:
    return "not checked" if flag is None else "yes" if flag else "**no**"


def _timing(values: list[float]) -> str:
    if not values:
        return "n/a"

    def fmt(seconds: float) -> str:
        return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"

    return f"median {fmt(statistics.median(values))}, max {fmt(max(values))}"


def write_report(results: list[Result], meta: dict, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*asdict(results[0]), "passed"])
        writer.writeheader()
        for r in results:
            writer.writerow({**asdict(r), "passed": r.passed})

    n = len(results)
    checked = [r for r in results if r.checked]
    passed = sum(r.passed for r in checked)
    if checked:
        headline = f"**Accuracy: {passed}/{len(checked)} checked answers fully correct** ({passed / len(checked):.0%})."
    else:
        headline = "**Accuracy against moma.org: not measured** (no page could be read; see the notes)."
    if len(checked) < n:
        headline += f" {n - len(checked)} of {n} answers could not be checked against moma.org."
    lines = [
        f"# AskMoMA evaluation, {meta['date']}",
        "",
        headline,
        "",
        "| Check | Correct |",
        "| --- | --- |",
        f"| Right work (from AskMoMA's answer alone) | {sum(r.right_work for r in results)}/{n} |",
        f"| Link opens that work on moma.org | {sum(bool(r.link_ok) for r in checked)}/{len(checked)} checked |",
        f"| Location matches moma.org today | {sum(bool(r.location_ok) for r in checked)}/{len(checked)} checked |",
        "",
        f"**Response time** (AskMoMA only): {_timing([r.seconds for r in results])}.",
    ]
    llm_times = [r.llm_seconds for r in results if r.llm_seconds is not None]
    if llm_times:
        used = sum(bool(r.llm_rewrite_used) for r in results)
        lines.append(
            f"**Added by the LLM** ({meta['LLM']}): {_timing(llm_times)}; its rewrite was used for "
            f"{used}/{n} answers (the rest kept the template)."
        )
    lines += [
        "",
        "## Setup",
        "",
        *(f"- {k}: {v}" for k, v in meta.items()),
        "",
        "## Answers",
        "",
        "| ID | Question | Expected | AskMoMA answered | AskMoMA location | moma.org location | Work | Link | Location | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        cells = [r.id, r.question, r.expected, r.answered, r.askmoma_location, r.moma_location,
                 _yes(r.right_work), _yes(r.link_ok), _yes(r.location_ok), r.note]  # fmt: skip
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, help="folder with Artworks.csv and Artists.csv")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--llm", metavar="MODEL", help="also time the optional LLM layer with this model")
    ap.add_argument("--offline", action="store_true", help="use only moma.org pages saved by an earlier run")
    ap.add_argument("--pages", type=Path, help="folder of work pages saved from a browser as <ObjectID>.html")
    ap.add_argument("--refresh", action="store_true", help="fetch moma.org pages again even if saved")
    ap.add_argument(
        "--out", type=Path, help="report path without extension (default: evaluation/results/<date>)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    started = time.perf_counter()
    bot = AskMoMA(load(args.data))
    startup = time.perf_counter() - started
    llm_load = None
    if args.llm:
        config.LLM_MODEL = args.llm
        llm.reset()
        started = time.perf_counter()
        if llm.load() is None:
            log.error("The LLM %s could not be loaded; is the `llm` extra installed?", args.llm)
            return 1
        llm_load = time.perf_counter() - started

    if args.pages:
        pages = MomaPages(args.pages, offline=True)
    else:
        pages = MomaPages(config.CACHE_DIR / "moma_pages", offline=args.offline, refresh=args.refresh)
    results = []
    for q in questions:
        r = evaluate_question(bot, q, pages, use_llm=bool(args.llm))
        status = "PASS" if r.passed else ("FAIL" if r.checked or not r.right_work else "----")
        log.info("%s %s  %s", r.id, status, r.note or r.answered)
        results.append(r)

    today = dt.date.today().isoformat()
    meta = {
        "date": today,
        "questions": f"{args.questions.name} ({len(questions)})",
        "collection data as of": bot.col.as_of,
        "works on view in the data": f"{len(bot.col.on_view):,}",
        "search backend": bot.search.backend,
        "LLM": args.llm or "off",
        "startup (load data, build search)": f"{startup:.1f} s",
        "LLM load": f"{llm_load:.1f} s" if llm_load is not None else "n/a",
        "moma.org pages fetched this run": f"{pages.fetched} (others reused from .cache/moma_pages)",
        "machine": f"{platform.system()} {platform.release()}, {os.cpu_count()} CPU threads, "
        f"Python {platform.python_version()}",
    }
    stem = args.out or RESULTS_DIR / (today + ("-llm" if args.llm else ""))
    write_report(results, meta, stem)
    checked = [r for r in results if r.checked]
    log.info(
        "\n%d/%d checked answers fully correct; %d not checked. Report: %s.md",
        sum(r.passed for r in checked),
        len(checked),
        len(results) - len(checked),
        stem,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
