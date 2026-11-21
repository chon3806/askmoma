"""Download MoMA's open collection data and build the small table AskMoMA uses.

    python scripts/fetch_data.py

MoMA publishes its collection on GitHub under a CC0 public-domain dedication:
https://github.com/MuseumofModernArt/collection. The full files are large
(Artworks.csv is about 70 MB), so they are saved to data/raw/, which Git
ignores. This script keeps only the works currently listed as on view and
writes them, with their floor and gallery split out, to data/on_view.csv
(small enough to commit), plus data/SOURCE.md recording when and from which
commit the data was taken.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPO = "MuseumofModernArt/collection"
FILES = ("Artworks.csv", "Artists.csv")
# The CSVs are stored with Git LFS, so they are served from the media host.
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/main/"
USER_AGENT = "AskMoMA-course-project/0.1"

# "MoMA, Floor 5, 501" -> venue "MoMA", floor "5", gallery "501". Some works give only a
# floor ("MoMA, Floor 6") or another venue ("MoMA PS1, Second floor").
LOCATION = re.compile(r"^(?P<venue>[^,]+),\s*(?:Floor (?P<floor>\w+)|(?P<floor_name>\w+) floor)"
                      r"(?:,\s*(?P<gallery>.+))?$")

COLUMNS = {
    "ObjectID": "object_id", "Title": "title", "Artist": "artist", "ArtistBio": "artist_bio",
    "Date": "date", "Medium": "medium", "Classification": "classification",
    "Department": "department", "OnView": "on_view", "URL": "url", "ImageURL": "image_url",
}


def download(name: str) -> Path:
    """Fetch one file into data/raw/ (skipped if it is already there)."""
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / name
    if not target.exists():
        print(f"downloading {name} ...")
        request = urllib.request.Request(MEDIA + name, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response:
            target.write_bytes(response.read())
    return target


def latest_commit() -> str:
    """The commit of MoMA's repository the data came from, for the record."""
    request = urllib.request.Request(f"https://api.github.com/repos/{REPO}/commits/main",
                                     headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        return f"{data['sha'][:10]} ({data['commit']['committer']['date'][:10]})"
    except OSError:
        return "unknown (GitHub API unreachable)"


def build_on_view(artworks: Path) -> tuple[pd.DataFrame, int]:
    """Works with a gallery location, location split into floor and gallery; and the collection size."""
    works = pd.read_csv(artworks, low_memory=False, usecols=list(COLUMNS)).rename(columns=COLUMNS)
    total = len(works)
    works = works[works["on_view"].notna() & works["on_view"].astype(str).str.strip().ne("")].copy()
    # MoMA's file wraps each location in literal quotes: "MoMA, Floor 5, 501".
    works["on_view"] = works["on_view"].str.strip().str.strip('"')
    parts = works["on_view"].str.extract(LOCATION)
    works["venue"] = parts["venue"].str.strip()
    works["floor"] = parts["floor"].fillna(parts["floor_name"])
    works["gallery"] = parts["gallery"].str.strip()
    works["object_id"] = works["object_id"].astype(int)
    return works.sort_values(["floor", "gallery", "artist", "title"]).reset_index(drop=True), total


def main() -> int:
    artworks = download("Artworks.csv")
    download("Artists.csv")
    on_view, total = build_on_view(artworks)
    out = ROOT / "data" / "on_view.csv"
    on_view.to_csv(out, index=False)
    (ROOT / "data" / "SOURCE.md").write_text(
        "# Data source\n\n"
        f"- Dataset: The Museum of Modern Art (MoMA) collection data, https://github.com/{REPO}\n"
        "- License: CC0 1.0 (public domain dedication)\n"
        f"- Commit: {latest_commit()}\n"
        f"- Downloaded: {date.today().isoformat()}\n"
        f"- on_view.csv: {len(on_view):,} works listed as on view, of {total:,} in the collection\n\n"
        "Gallery locations change; check the work's page on moma.org for the latest.\n",
        encoding="utf-8")
    print(f"wrote {out} ({len(on_view)} works on view)")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
