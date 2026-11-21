"""Download MoMA's Artworks.csv and Artists.csv into the data folder.

    mise run data        (or: uv run python -m askmoma.download)

The files are stored with Git LFS upstream. A plain download of an LFS file
returns a small pointer file instead of the data, so the size is checked.
"""

from __future__ import annotations

import sys
import urllib.request

from askmoma.config import DATA_DIR

BASE_URL = "https://media.githubusercontent.com/media/MuseumofModernArt/collection/main"
FILES = ["Artworks.csv", "Artists.csv"]
MIN_BYTES = 1024  # a Git LFS pointer file is a few hundred bytes


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DATA_DIR / name
        print(f"Downloading {name} ...", flush=True)
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            target.write_bytes(response.read())
        if target.stat().st_size < MIN_BYTES:
            print(f"{name} is only {target.stat().st_size} bytes: that is a Git LFS pointer, not the data.")
            return 1
    print(f"Done. The data is in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
