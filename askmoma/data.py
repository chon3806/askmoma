"""Load MoMA's Artworks.csv / Artists.csv and build lookup indexes.

Column names follow the public repo
(https://github.com/MuseumofModernArt/collection). Missing optional columns
are tolerated so the app still starts on older or trimmed copies of the data.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config

ARTWORK_COLS = [
    "Title", "Artist", "ConstituentID", "ArtistBio", "Nationality", "BeginDate",
    "EndDate", "Gender", "Date", "Medium", "Dimensions", "CreditLine",
    "AccessionNumber", "Classification", "Department", "DateAcquired",
    "Cataloged", "ObjectID", "URL", "ImageURL", "ThumbnailURL", "OnView",
]
ARTIST_COLS = [
    "ConstituentID", "DisplayName", "ArtistBio", "Nationality", "Gender",
    "BeginDate", "EndDate", "Wiki QID", "ULAN",
]

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(1[5-9]\d\d|20\d\d)")


def norm(text) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", text).strip()


def first_year(text) -> int | None:
    m = _YEAR.search(str(text)) if isinstance(text, str) else None
    return int(m.group(1)) if m else None


def parse_on_view(value) -> tuple[int | None, str | None, str]:
    """Turn the OnView text into (floor, gallery, label).

    The repo does not document the exact format, so this accepts the shapes we
    have seen in the wild ("MoMA, Floor 5, 501", "Floor 5, Gallery 501",
    "Gallery 501") and falls back to the raw text.
    """
    if not isinstance(value, str) or not value.strip():
        return None, None, ""
    raw = value.strip()
    floor_m = re.search(r"floor\s*(\d+)", raw, re.I)
    gallery_m = re.search(r"(?:gallery\s*)?\b(\d{3}[A-Za-z]?)\b", raw, re.I)
    floor = int(floor_m.group(1)) if floor_m else None
    gallery = gallery_m.group(1).upper() if gallery_m else None
    if floor is None and gallery and gallery[0].isdigit():
        floor = int(gallery[0])  # MoMA gallery numbers start with the floor
    if floor and gallery:
        label = f"Floor {floor}, Gallery {gallery}"
    elif floor:
        label = f"Floor {floor}"
    elif gallery:
        label = f"Gallery {gallery}"
    else:
        label = raw
    return floor, gallery, label


def _split_ids(value) -> list[str]:
    if not isinstance(value, str):
        if pd.isna(value):
            return []
        value = str(int(value)) if isinstance(value, float) else str(value)
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Collection:
    works: pd.DataFrame
    artists: pd.DataFrame
    as_of: str
    title_index: dict = field(default_factory=dict)     # norm title -> [row]
    title_tokens: dict = field(default_factory=dict)    # token -> {norm title}
    artist_index: dict = field(default_factory=dict)    # norm name -> cid
    surname_index: dict = field(default_factory=dict)   # last token -> [cid]
    works_by_artist: dict = field(default_factory=dict) # cid -> [row]
    nationalities: dict = field(default_factory=dict)   # norm nat -> label

    # ---- convenience -------------------------------------------------
    @property
    def on_view(self) -> pd.DataFrame:
        return self.works[self.works["on_view"]]

    def work(self, row: int) -> dict:
        return self.works.loc[row].to_dict()

    def artist(self, cid: str) -> dict | None:
        if cid in self.artists.index:
            rec = self.artists.loc[cid].to_dict()
            rec["ConstituentID"] = cid
            return rec
        return None

    def floors(self) -> list[int]:
        return sorted(int(f) for f in self.on_view["floor"].dropna().unique())


def _read_csv(path: Path, wanted: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, encoding="utf-8").columns
    cols = [c for c in wanted if c in header]
    return pd.read_csv(path, usecols=cols, dtype=str, encoding="utf-8",
                       keep_default_na=True, low_memory=False)


def load(data_dir: Path | None = None) -> Collection:
    data_dir = Path(data_dir or config.DATA_DIR)
    works_path = data_dir / "Artworks.csv"
    artists_path = data_dir / "Artists.csv"
    if not works_path.exists():
        raise FileNotFoundError(
            f"Artworks.csv not found in {data_dir}. See README: download it from "
            "https://github.com/MuseumofModernArt/collection")

    works = _read_csv(works_path, ARTWORK_COLS)
    for col in ARTWORK_COLS:
        if col not in works.columns:
            works[col] = pd.NA
    works = works.reset_index(drop=True)

    works["title_norm"] = works["Title"].map(norm)
    works["year"] = works["Date"].map(first_year)
    loc = works["OnView"].map(parse_on_view)
    works["floor"] = [l[0] for l in loc]
    works["gallery"] = [l[1] for l in loc]
    works["location"] = [l[2] for l in loc]
    works["on_view"] = works["location"].str.len() > 0
    works["cids"] = works["ConstituentID"].map(_split_ids)
    works["Artist"] = works["Artist"].fillna("Unknown artist")
    works["link"] = [
        u if isinstance(u, str) and u.startswith("http")
        else config.WORK_URL.format(object_id=o)
        for u, o in zip(works["URL"], works["ObjectID"])
    ]

    if artists_path.exists():
        artists = _read_csv(artists_path, ARTIST_COLS)
        artists["ConstituentID"] = artists["ConstituentID"].map(
            lambda v: _split_ids(v)[0] if _split_ids(v) else None)
        artists = artists.dropna(subset=["ConstituentID", "DisplayName"])
    else:
        # Rebuild a minimal artist table from the artwork rows.
        rows = []
        for names, cids in zip(works["Artist"], works["cids"]):
            for name, cid in zip(str(names).split(", "), cids):
                rows.append({"ConstituentID": cid, "DisplayName": name})
        artists = pd.DataFrame(rows)
    artists = artists.drop_duplicates("ConstituentID").set_index("ConstituentID")

    if config.DATA_AS_OF:
        as_of = config.DATA_AS_OF
    else:
        as_of = dt.date.fromtimestamp(works_path.stat().st_mtime).strftime("%B %d, %Y")

    col = Collection(works=works, artists=artists, as_of=as_of)
    _build_indexes(col)
    return col


def _build_indexes(col: Collection) -> None:
    works = col.works
    # Titles: on-view rows first so lookups prefer what a visitor can see.
    order = works.sort_values("on_view", ascending=False).index
    title_index: dict[str, list[int]] = defaultdict(list)
    title_tokens: dict[str, set] = defaultdict(set)
    for row in order:
        t = works.at[row, "title_norm"]
        if t:
            title_index[t].append(row)
            for tok in t.split():
                title_tokens[tok].add(t)
    col.title_index = dict(title_index)
    col.title_tokens = dict(title_tokens)

    works_by_artist: dict[str, list[int]] = defaultdict(list)
    for row, cids in zip(order, works.loc[order, "cids"]):
        for cid in cids:
            works_by_artist[cid].append(row)
    col.works_by_artist = dict(works_by_artist)

    artist_index, surname_index = {}, defaultdict(list)
    for cid, name in col.artists["DisplayName"].items():
        n = norm(name)
        if not n:
            continue
        # Several records can share a name; keep the one with most works.
        prev = artist_index.get(n)
        if prev is None or len(works_by_artist.get(cid, [])) > len(works_by_artist.get(prev, [])):
            artist_index[n] = cid
        toks = n.split()
        if len(toks) > 1 and len(toks[-1]) > 2:
            surname_index[toks[-1]].append(cid)
    for k, cids in surname_index.items():
        cids.sort(key=lambda c: -len(works_by_artist.get(c, [])))
    col.artist_index = artist_index
    col.surname_index = dict(surname_index)

    if "Nationality" in col.artists.columns:
        nats = col.artists["Nationality"].dropna().unique()
        col.nationalities = {norm(n): n for n in nats if norm(n) and norm(n) not in {"nationality unknown"}}
