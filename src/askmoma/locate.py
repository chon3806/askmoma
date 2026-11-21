"""Use case 1: locate a work on view at MoMA.

"Where is The Starry Night?" -> floor, gallery, date, medium, and the work's
page on moma.org. Answers come only from MoMA's data (data/on_view.csv);
nothing is generated, so a location can always be traced to its source.

Matching is by title. Exact (case- and punctuation-insensitive) matches win;
otherwise the closest titles by string similarity are offered, so a typo such
as "starry nite" still finds the painting. If nothing is close enough, the
answer says so and points to moma.org instead of guessing.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "on_view.csv"
MOMA_SEARCH = "https://www.moma.org/collection/"
CUTOFF = 0.75   # how similar a title must be to count as a match (0 to 1)

# Words that start a question but are not part of the title.
LEAD_IN = re.compile(r"^(?:where(?: is| are|'s)?|find|locate|show me|can i see|i want to see)\s+",
                     re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase letters and digits only, one space apart."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def where(work: pd.Series) -> str:
    """ "MoMA, Floor 5, Gallery 501", or as much of that as the data gives."""
    if pd.notna(work["floor"]) and pd.notna(work["gallery"]):
        return f"{work['venue']}, Floor {work['floor']}, Gallery {work['gallery']}"
    return str(work["on_view"])


@dataclass(frozen=True)
class Answer:
    text: str                 # Markdown for the chat
    matches: tuple[int, ...]  # object_ids of the works found, best first


class Collection:
    """The works on view, with a title index for lookup."""

    def __init__(self, path: Path = DATA) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run python scripts/fetch_data.py first")
        self.works = pd.read_csv(path, dtype={"floor": str, "gallery": str})
        self._titles = self.works["title"].map(normalize)
        self._by_title: dict[str, list[int]] = {}
        for index, title in self._titles.items():
            self._by_title.setdefault(title, []).append(index)

    def find(self, question: str) -> list[int]:
        """Row indices of the works the question names, best first."""
        query = normalize(LEAD_IN.sub("", question.strip().rstrip("?")))
        query = re.sub(r"^the painting |^the work ", "", query)
        if not query:
            return []
        variants = (query, query.removeprefix("the "), "the " + query)
        for candidate in variants:
            if candidate in self._by_title:
                return self._by_title[candidate]
        # No exact title: the closest titles, trying the query with and without "the".
        scored: dict[str, float] = {}
        for candidate in variants:
            for title in difflib.get_close_matches(candidate, list(self._by_title), n=3, cutoff=CUTOFF):
                ratio = difflib.SequenceMatcher(None, candidate, title).ratio()
                scored[title] = max(scored.get(title, 0.0), ratio)
        best = sorted(scored, key=scored.get, reverse=True)[:3]
        return [index for title in best for index in self._by_title[title]]

    def locate(self, question: str) -> Answer:
        """The answer to "where is ...?" in Markdown."""
        rows = self.find(question)
        if not rows:
            return Answer("I couldn't find that title among the works currently on view. "
                          f"Try MoMA's collection search: {MOMA_SEARCH}", ())
        lines = []
        for index in rows[:3]:
            w = self.works.loc[index]
            lines.append(f"**{w['title']}** by {w['artist']} ({w['date']}) is on view at "
                         f"**{where(w)}**.  \n{w['medium']}. [View on moma.org]({w['url']})")
        intro = "" if len(rows) == 1 else "I found more than one match:\n\n"
        return Answer(intro + "\n\n".join(lines), tuple(int(self.works.loc[i, "object_id"]) for i in rows))
