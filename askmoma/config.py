"""Runtime settings. Everything can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Folder holding Artworks.csv and Artists.csv from
# https://github.com/MuseumofModernArt/collection (CC0).
DATA_DIR = Path(os.environ.get("ASKMOMA_DATA_DIR", ROOT / "data"))

# Date shown to the visitor as "collection data as of ...".
# Set it to the date you pulled the repo; if empty we use the CSV file's date.
DATA_AS_OF = os.environ.get("ASKMOMA_DATA_AS_OF", "")

# Which works semantic search covers: "on_view" (fast, default) or "all".
SEARCH_SCOPE = os.environ.get("ASKMOMA_SEARCH_SCOPE", "on_view")

# Search backend: "auto" (embeddings if installed, else TF-IDF), "embeddings"
# (fail loudly if they cannot load), or "tfidf" (keyword search only).
SEARCH_BACKEND = os.environ.get("ASKMOMA_SEARCH_BACKEND", "auto")

# Embedding model for search-by-description (falls back to TF-IDF if missing).
EMBED_MODEL = os.environ.get("ASKMOMA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Optional small LLM that rewords answers. Empty = templated answers only.
# Candidates from the proposal: "Qwen/Qwen2.5-1.5B-Instruct",
# "HuggingFaceTB/SmolLM2-360M-Instruct".
LLM_MODEL = os.environ.get("ASKMOMA_LLM", "")

CACHE_DIR = Path(os.environ.get("ASKMOMA_CACHE_DIR", ROOT / ".cache"))

MOMA_HOME = "https://www.moma.org"
MOMA_VISIT = "https://www.moma.org/visit/"
WORK_URL = "https://www.moma.org/collection/works/{object_id}"
ARTIST_PAGE_URL = "https://www.moma.org/artists/{cid}-{slug}"
ARTIST_SEARCH_URL = "https://www.moma.org/search/?q={query}"
TOP_ARTISTS = 5
ARTIST_URL = "https://www.moma.org/artists/{constituent_id}"

MAX_LIST = 8  # rows shown in list-style answers
CHAT_LIST = 5  # rows shown in a chat reply; the rest link to moma.org
