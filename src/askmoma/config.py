"""Paths and constants shared across the package."""

from __future__ import annotations

from pathlib import Path

# The repository root: src/askmoma/config.py -> parents[2].
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # full MoMA download, ignored by Git
ON_VIEW_CSV = DATA_DIR / "on_view.csv"  # the works on view, committed

MOMA_DATA_REPO = "MuseumofModernArt/collection"
MOMA_COLLECTION_URL = "https://www.moma.org/collection/"
USER_AGENT = "AskMoMA-course-project/0.1"
