"""The works on view, loaded once and shared by every feature."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pandas as pd

from askmoma.config import ON_VIEW_CSV

TITLE_CUTOFF = 0.75  # how similar a title must be to count as a match (0 to 1)


def normalize(text: str) -> str:
    """Lowercase letters and digits only, one space apart."""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def where(work: pd.Series) -> str:
    """ "MoMA, Floor 5, Gallery 501", or as much of that as the data gives."""
    if pd.notna(work["floor"]) and pd.notna(work["gallery"]):
        return f"{work['venue']}, Floor {work['floor']}, Gallery {work['gallery']}"
    return str(work["on_view"])


class Collection:
    """MoMA's works on view (data/on_view.csv), with a title index for lookup."""

    def __init__(self, path: Path = ON_VIEW_CSV) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `mise run data` first")
        self.works = pd.read_csv(path, dtype={"floor": str, "gallery": str})
        self._by_title: dict[str, list[int]] = {}
        for index, title in self.works["title"].map(normalize).items():
            self._by_title.setdefault(title, []).append(index)

    def __len__(self) -> int:
        return len(self.works)

    def find_title(self, title: str) -> list[int]:
        """Row indices of works with this title, best match first.

        Exact matches (ignoring case, punctuation, and a leading "the") win;
        otherwise the closest titles by string similarity, so "starry nite"
        still finds The Starry Night. An empty list means nothing was close.
        """
        query = normalize(title)
        if not query:
            return []
        variants = (query, query.removeprefix("the "), "the " + query)
        for candidate in variants:
            if candidate in self._by_title:
                return self._by_title[candidate]
        scored: dict[str, float] = {}
        for candidate in variants:
            for match in difflib.get_close_matches(candidate, list(self._by_title), n=3, cutoff=TITLE_CUTOFF):
                ratio = difflib.SequenceMatcher(None, candidate, match).ratio()
                scored[match] = max(scored.get(match, 0.0), ratio)
        best = sorted(scored, key=scored.get, reverse=True)[:3]
        return [index for match in best for index in self._by_title[match]]
