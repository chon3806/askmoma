"""Download MoMA's open collection data and build the table AskMoMA uses.

    mise run data          (or: uv run python -m askmoma.data.fetch)

MoMA publishes its collection on GitHub under a CC0 public-domain dedication
(https://github.com/MuseumofModernArt/collection). The full files are large
(Artworks.csv is about 70 MB), so they are saved to data/raw/, which Git
ignores. Only the works currently listed as on view are kept, with each
location split into venue, floor, and gallery, in data/on_view.csv (small
enough to commit). data/SOURCE.md records which commit of MoMA's data was
used and when it was downloaded.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from askmoma.config import DATA_DIR, MOMA_DATA_REPO, ON_VIEW_CSV, RAW_DIR, USER_AGENT

# The CSVs are stored with Git LFS, so they are served from GitHub's media host.
MEDIA_URL = f"https://media.githubusercontent.com/media/{MOMA_DATA_REPO}/main/"

# "MoMA, Floor 5, 501" -> venue "MoMA", floor "5", gallery "501". Some works give only a
# floor ("MoMA, Floor 6") or another venue ("MoMA PS1, Second floor").
LOCATION = re.compile(
    r"^(?P<venue>[^,]+),\s*(?:Floor (?P<floor>\w+)|(?P<floor_name>\w+) floor)(?:,\s*(?P<gallery>.+))?$"
)

# MoMA's column names -> ours.
COLUMNS = {
    "ObjectID": "object_id", "Title": "title", "Artist": "artist", "ArtistBio": "artist_bio",
    "Date": "date", "Medium": "medium", "Classification": "classification",
    "Department": "department", "OnView": "on_view", "URL": "url", "ImageURL": "image_url",
}  # fmt: skip


def download(name: str) -> Path:
    """Fetch one of MoMA's files into data/raw/ (skipped if it is already there)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if not target.exists():
        print(f"downloading {name} ...")
        request = urllib.request.Request(MEDIA_URL + name, headers={"User-Agent": USER_AGENT})
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=300) as response:
            partial.write_bytes(response.read())
        partial.replace(target)  # only a complete download gets the real name
    return target


def latest_commit() -> str:
    """The commit of MoMA's repository the data came from, for the record."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{MOMA_DATA_REPO}/commits/main", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        return f"{data['sha'][:10]} ({data['commit']['committer']['date'][:10]})"
    except OSError:
        return "unknown (GitHub API unreachable)"


def split_locations(on_view: pd.Series) -> pd.DataFrame:
    """Split MoMA's location text into venue, floor, and gallery columns."""
    # MoMA's file wraps each location in literal quotes: "MoMA, Floor 5, 501".
    text = on_view.astype(str).str.strip().str.strip('"')
    parts = text.str.extract(LOCATION)
    return pd.DataFrame({
        "on_view": text,
        "venue": parts["venue"].str.strip(),
        "floor": parts["floor"].fillna(parts["floor_name"]),
        "gallery": parts["gallery"].str.strip(),
    })  # fmt: skip


def build_on_view(artworks: Path) -> tuple[pd.DataFrame, int]:
    """The works listed as on view, with their location split out; and the collection size."""
    works = pd.read_csv(artworks, low_memory=False, usecols=list(COLUMNS)).rename(columns=COLUMNS)
    total = len(works)
    works = works[works["on_view"].notna() & works["on_view"].astype(str).str.strip().ne("")].copy()
    works[["on_view", "venue", "floor", "gallery"]] = split_locations(works["on_view"])
    works["object_id"] = works["object_id"].astype(int)
    return works.sort_values(["floor", "gallery", "artist", "title"]).reset_index(drop=True), total


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    artworks = download("Artworks.csv")
    download("Artists.csv")
    on_view, total = build_on_view(artworks)
    on_view.to_csv(ON_VIEW_CSV, index=False)
    (DATA_DIR / "SOURCE.md").write_text(
        "# Data source\n\n"
        f"- Dataset: The Museum of Modern Art (MoMA) collection data, https://github.com/{MOMA_DATA_REPO}\n"
        "- License: CC0 1.0 (public domain dedication)\n"
        f"- Commit: {latest_commit()}\n"
        f"- Downloaded: {date.today().isoformat()}\n"
        f"- on_view.csv: {len(on_view):,} works listed as on view, of {total:,} in the collection\n\n"
        "Gallery locations change; check the work's page on moma.org for the latest.\n",
        encoding="utf-8",
    )
    print(f"wrote {ON_VIEW_CSV} ({len(on_view):,} works on view)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
