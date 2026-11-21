"""Search tests on the fictional fixture.

The TF-IDF tests always run. Tests marked ``embeddings`` need the ``embeddings``
extra (``uv sync --extra embeddings``) and the all-MiniLM-L6-v2 model, which is
downloaded once from Hugging Face; they are skipped when the extra is missing.
Deselect them with ``pytest -m "not embeddings"``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

from askmoma import config, load
from askmoma.search import Searcher

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HAS_EMBEDDINGS = importlib.util.find_spec("sentence_transformers") is not None


def needs_embeddings(test):
    """Mark a test `embeddings` and skip it when the extra is not installed."""
    test = pytest.mark.embeddings(test)
    return pytest.mark.skipif(not HAS_EMBEDDINGS, reason="the `embeddings` extra is not installed")(test)


# No word in this query appears in the fixture, so keyword search cannot match it;
# its meaning is closest to "The Quiet Harbor at Night".
PARAPHRASE = "boats moored at the dock after dark"


@pytest.fixture(scope="module")
def col():
    return load(FIXTURES)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def no_embeddings(monkeypatch):
    """Make `import sentence_transformers` fail, as on an install without the extra."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


def titles(col, hits):
    return [col.works.loc[row, "Title"] for row, _ in hits]


# ------------------------------------------------------------------ TF-IDF


def test_tfidf_finds_keyword_matches(col):
    s = Searcher(col, backend="tfidf")
    assert s.backend == "tfidf"
    found = titles(col, s.query("night paintings", k=3))
    assert {"Night Garden", "The Quiet Harbor at Night"} & set(found)


def test_tfidf_cannot_match_a_paraphrase(col):
    assert Searcher(col, backend="tfidf").query(PARAPHRASE) == []


def test_blank_query_returns_nothing(col):
    assert Searcher(col, backend="tfidf").query("   ") == []


def test_similar_to_excludes_the_work_itself(col):
    s = Searcher(col, backend="tfidf")
    row = int(col.works.index[col.works["Title"] == "Moonlit Field"][0])
    hits = s.similar_to(row, k=4)
    assert hits and row not in [r for r, _ in hits]
    assert len(hits) <= 4


def test_scores_are_sorted_best_first(col):
    scores = [score for _, score in Searcher(col, backend="tfidf").query("oil on canvas painting", k=6)]
    assert scores == sorted(scores, reverse=True)


def test_unknown_backend_is_rejected(col):
    with pytest.raises(ValueError, match="Unknown search backend"):
        Searcher(col, backend="bm25")


def test_auto_falls_back_to_tfidf_without_the_extra(col, no_embeddings, caplog):
    with caplog.at_level(logging.WARNING, logger="askmoma.search"):
        s = Searcher(col, backend="auto")
    assert s.backend == "tfidf"
    assert "using TF-IDF search" in caplog.text


def test_embeddings_backend_fails_loudly_without_the_extra(col, no_embeddings):
    with pytest.raises(ImportError):
        Searcher(col, backend="embeddings")


# -------------------------------------------------------------- embeddings


@pytest.fixture
def emb(col, cache_dir):
    return Searcher(col, backend="embeddings")


@needs_embeddings
def test_embeddings_use_faiss(emb):
    assert emb.backend == "embeddings+faiss"
    assert emb._index.ntotal == len(emb.docs)


@needs_embeddings
def test_embeddings_match_meaning_not_just_words(col, emb):
    hits = emb.query(PARAPHRASE, k=3)
    assert titles(col, hits)[0] == "The Quiet Harbor at Night"
    assert all(-1.0 <= score <= 1.0 for _, score in hits)  # cosine similarity


@needs_embeddings
def test_embeddings_are_cached_and_reused(col, emb, cache_dir, monkeypatch):
    from sentence_transformers import SentenceTransformer

    (cache,) = cache_dir.glob("emb_*.npy")
    assert np.load(cache).shape == (len(emb.docs), 384)

    corpus_encodes = []
    original = SentenceTransformer.encode

    def spy(self, sentences, *args, **kwargs):
        if isinstance(sentences, list) and len(sentences) > 1:
            corpus_encodes.append(len(sentences))
        return original(self, sentences, *args, **kwargs)

    monkeypatch.setattr(SentenceTransformer, "encode", spy)
    again = Searcher(col, backend="embeddings")
    assert corpus_encodes == []  # the corpus came from the cache
    assert again.query(PARAPHRASE, k=1) == emb.query(PARAPHRASE, k=1)


@needs_embeddings
def test_corrupt_cache_is_rebuilt(col, emb, cache_dir):
    (cache,) = cache_dir.glob("emb_*.npy")
    cache.write_bytes(b"not a numpy file")
    rebuilt = Searcher(col, backend="embeddings")
    assert rebuilt.backend == "embeddings+faiss"
    assert np.load(cache).shape == (len(rebuilt.docs), 384)
    assert not list(cache_dir.glob("*.tmp"))
