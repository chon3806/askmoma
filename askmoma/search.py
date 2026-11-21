"""Search by description ("dreamy night sky paintings") and "similar works".

Uses sentence embeddings (all-MiniLM-L6-v2) with a FAISS index when those
packages are installed, and falls back to TF-IDF (scikit-learn) otherwise,
so the app always starts.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from . import config
from .data import Collection

log = logging.getLogger(__name__)


def _doc(w) -> str:
    parts = [w["Title"], w["Artist"], w["Date"], w["Medium"], w["Classification"], w["Department"]]
    return ". ".join(str(p) for p in parts if isinstance(p, str) and p)


class Searcher:
    def __init__(self, col: Collection):
        self.col = col
        works = col.works
        corpus = works[works["on_view"]] if config.SEARCH_SCOPE == "on_view" else works
        if corpus.empty:  # no OnView data: search the whole collection instead
            corpus = works
        self.rows = corpus.index.to_numpy()
        self.docs = [_doc(w) for _, w in corpus.iterrows()]
        self.backend = "none"
        self._model = self._index = self._tfidf = self._matrix = None
        if self.docs:
            self._build()

    # ---------------------------------------------------------------- build
    def _build(self):
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(config.EMBED_MODEL)
            emb = self._cached_embeddings()
            try:
                import faiss

                self._index = faiss.IndexFlatIP(emb.shape[1])
                self._index.add(emb)
                self.backend = "embeddings+faiss"
            except ImportError:
                self._matrix = emb
                self.backend = "embeddings"
            return
        except Exception as exc:  # missing package, no network for the model, etc.
            log.warning("Embeddings unavailable (%s); using TF-IDF search.", exc)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._tfidf.fit_transform(self.docs)
        self.backend = "tfidf"

    def _cached_embeddings(self) -> np.ndarray:
        key = hashlib.md5(("\n".join(self.docs) + config.EMBED_MODEL).encode()).hexdigest()[:12]
        path = config.CACHE_DIR / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self._model.encode(
            self.docs, batch_size=64, normalize_embeddings=True, show_progress_bar=len(self.docs) > 2000
        ).astype("float32")
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    # ---------------------------------------------------------------- query
    def _scores(self, vec) -> np.ndarray:
        if self.backend == "tfidf":
            return (self._matrix @ vec.T).toarray().ravel()
        return self._matrix @ vec.ravel()

    def _encode(self, text: str):
        if self.backend == "tfidf":
            return self._tfidf.transform([text])
        return self._model.encode([text], normalize_embeddings=True).astype("float32")

    def query(self, text: str, k: int = 8, min_score: float = 0.0) -> list[tuple[int, float]]:
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
        floor = min_score if self.backend != "tfidf" else min_score * 0.3
        return [(r, s) for r, s in pairs if s > max(floor, 0.0)]

    def similar_to(self, row: int, k: int = 6) -> list[tuple[int, float]]:
        w = self.col.works.loc[row]
        hits = self.query(_doc(w), k=k + 1)
        return [(r, s) for r, s in hits if r != row][:k]
