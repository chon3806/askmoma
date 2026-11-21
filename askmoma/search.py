"""Search by description ("dreamy night sky paintings") and "similar works".

Two backends, chosen by ``config.SEARCH_BACKEND``:

* ``embeddings``: sentence embeddings from all-MiniLM-L6-v2, searched with a
  FAISS inner-product index (cosine similarity, since vectors are normalized).
  Needs the ``embeddings`` extra: ``uv sync --extra embeddings``.
* ``tfidf``: keyword search with scikit-learn, always available.

The default, ``auto``, tries embeddings and falls back to TF-IDF, so the app
always starts. Embeddings are cached in ``config.CACHE_DIR`` keyed by the model
and the corpus, so only the first start pays for encoding.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from . import config
from .data import Collection

log = logging.getLogger(__name__)

BACKENDS = ("auto", "embeddings", "tfidf")


def _doc(w) -> str:
    parts = [w["Title"], w["Artist"], w["Date"], w["Medium"], w["Classification"], w["Department"]]
    return ". ".join(str(p) for p in parts if isinstance(p, str) and p)


class Searcher:
    """Ranks works in the collection by how well they match a text query."""

    def __init__(self, col: Collection, backend: str | None = None):
        backend = backend or config.SEARCH_BACKEND
        if backend not in BACKENDS:
            raise ValueError(f"Unknown search backend {backend!r}; expected one of {', '.join(BACKENDS)}")
        self.col = col
        works = col.works
        corpus = works[works["on_view"]] if config.SEARCH_SCOPE == "on_view" else works
        if corpus.empty:  # no OnView data: search the whole collection instead
            corpus = works
        self.rows = corpus.index.to_numpy()
        self._position = {int(r): i for i, r in enumerate(self.rows)}  # collection row -> corpus position
        self.docs = [_doc(w) for _, w in corpus.iterrows()]
        self.backend = "none"
        self._model = self._index = self._tfidf = self._matrix = self._emb = None
        if self.docs:
            self._build(backend)

    # ---------------------------------------------------------------- build
    def _build(self, requested: str) -> None:
        if requested != "tfidf":
            try:
                self._build_embeddings()
            except Exception as exc:  # missing extra, no network for the model, etc.
                if requested == "embeddings":
                    raise
                log.warning("Embeddings unavailable (%s); using TF-IDF search.", exc)
            else:
                log.info(
                    "Search backend: %s, %d works, model %s", self.backend, len(self.docs), config.EMBED_MODEL
                )
                return
        self._build_tfidf()
        log.info("Search backend: tfidf, %d works", len(self.docs))

    def _build_embeddings(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        emb = self._cached_embeddings()
        self._emb = emb  # kept for profiles and variety (1,556 x 384 floats, about 2 MB)
        try:
            import faiss
        except ImportError:  # sentence-transformers without FAISS: brute force in NumPy
            self._matrix = emb
            self.backend = "embeddings"
            return
        self._index = faiss.IndexFlatIP(emb.shape[1])
        self._index.add(emb)
        self.backend = "embeddings+faiss"

    def _build_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._tfidf.fit_transform(self.docs)
        self.backend = "tfidf"

    def _cache_path(self) -> Path:
        digest = hashlib.sha256()
        digest.update(config.EMBED_MODEL.encode())
        for doc in self.docs:
            digest.update(b"\0" + doc.encode())
        return config.CACHE_DIR / f"emb_{digest.hexdigest()[:16]}.npy"

    def _cached_embeddings(self) -> np.ndarray:
        path = self._cache_path()
        if path.exists():
            try:
                emb = np.load(path)
                if emb.shape[0] == len(self.docs):
                    log.debug("Loaded embeddings from %s", path)
                    return emb
            except (OSError, ValueError) as exc:  # truncated or corrupt file: rebuild it
                log.warning("Ignoring unreadable embedding cache %s (%s)", path, exc)
        emb = self._model.encode(
            self.docs, batch_size=64, normalize_embeddings=True, show_progress_bar=len(self.docs) > 2000
        ).astype("float32")
        _save_atomic(path, emb)
        return emb

    # ---------------------------------------------------------------- query
    def _scores(self, vec) -> np.ndarray:
        if self.backend == "tfidf":
            return (self._matrix @ vec.T).toarray().ravel()
        return self._matrix @ vec.ravel()

    def _encode(self, text: str):
        if self.backend == "tfidf":
            return self._tfidf.transform([text])
        return self._model.encode([text], normalize_embeddings=True, show_progress_bar=False).astype(
            "float32"
        )

    def query(self, text: str, k: int = 8, min_score: float = 0.0) -> list[tuple[int, float]]:
        """Return up to ``k`` (row, score) pairs, best first, scoring above ``min_score``."""
        if self.backend == "none" or not text.strip():
            return []
        vec = self._encode(text)
        if self._index is not None:
            scores, idx = self._index.search(vec, k)
            pairs = [(int(self.rows[i]), float(s)) for i, s in zip(idx[0], scores[0], strict=True) if i >= 0]
        else:
            scores = self._scores(vec)
            top = np.argsort(-scores)[:k]
            pairs = [(int(self.rows[i]), float(scores[i])) for i in top]
        # TF-IDF scores run lower than cosine similarity between embeddings.
        floor = min_score if self.backend != "tfidf" else min_score * 0.3
        return [(r, s) for r, s in pairs if s > max(floor, 0.0)]

    # ---------------------------------------------------------------- vectors
    def vector(self, row: int) -> np.ndarray | None:
        """The unit-length vector for the work at ``row`` (a stored one if it is in the corpus)."""
        if self.backend == "none":
            return None
        pos = self._position.get(int(row))
        if self._emb is not None:
            vec = self._emb[pos] if pos is not None else self._encode(_doc(self.col.works.loc[row]))[0]
        else:
            src = (
                self._matrix[pos]
                if pos is not None
                else self._tfidf.transform([_doc(self.col.works.loc[row])])
            )
            vec = np.asarray(src.todense()).ravel()
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype("float32") if norm else None

    def rank(self, vec: np.ndarray, k: int = 30, exclude: set[int] = frozenset()) -> list[tuple[int, float]]:
        """Corpus works most similar to ``vec`` (cosine), best first, skipping ``exclude``."""
        if self.backend == "none" or vec is None:
            return []
        if self._emb is not None:
            scores = self._emb @ vec
        else:
            scores = np.asarray(self._matrix @ vec).ravel()
        order = np.argsort(-scores)
        out = []
        for i in order:
            row = int(self.rows[i])
            if row not in exclude and scores[i] > 0:
                out.append((row, float(scores[i])))
                if len(out) == k:
                    break
        return out

    def similar_to(self, row: int, k: int = 6) -> list[tuple[int, float]]:
        """Works most like the one at ``row``, excluding it."""
        w = self.col.works.loc[row]
        hits = self.query(_doc(w), k=k + 1)
        return [(r, s) for r, s in hits if r != row][:k]


def _save_atomic(path: Path, arr: np.ndarray) -> None:
    """Write to a temporary file and rename it, so a crash never leaves half a cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
