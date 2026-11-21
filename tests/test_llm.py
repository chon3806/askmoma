"""Tests for the optional LLM layer.

Most tests replace the model with a fake that returns a chosen reply, so they
check the guardrails without downloading anything. The test marked ``llm``
loads a real small model; it needs the ``llm`` extra and is skipped otherwise.
Choose the model with ASKMOMA_TEST_LLM (default: SmolLM2-135M-Instruct).
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from askmoma import config, llm

PROSE = (
    "**The Starry Night** by Vincent van Gogh is on view on **Floor 5, Gallery 501**.\n\n1889 · Oil on canvas"
)
LINK = "[View it on moma.org](https://www.moma.org/collection/works/79802)"
FACTS = f"{PROSE}\n\n{LINK}"
FOOTER = "\n\n_Collection data as of November 21, 2026._"
QUESTION = "Where is The Starry Night?"
GOOD = "You'll find **The Starry Night** (1889), Vincent van Gogh's oil on canvas, on Floor 5 in Gallery 501."


class FakePipe:
    """Stands in for a transformers text-generation pipeline."""

    def __init__(self, reply: str | Exception):
        self.reply, self.calls = reply, []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return [{"generated_text": [*messages, {"role": "assistant", "content": self.reply}]}]


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    llm.reset()
    monkeypatch.setattr(config, "LLM_MODEL", "fake/model")
    yield
    llm.reset()


def use(monkeypatch, reply) -> FakePipe:
    pipe = FakePipe(reply)
    monkeypatch.setattr(llm, "_build_pipeline", lambda model: pipe)
    return pipe


# ------------------------------------------------------------------ split()


def test_split_sets_trailing_links_aside():
    assert llm.split(FACTS) == (PROSE, LINK)
    inline = "**Les Demoiselles d'Avignon** is Oil on canvas. [moma.org](https://www.moma.org/x)"
    assert llm.split(inline) == (
        "**Les Demoiselles d'Avignon** is Oil on canvas.",
        "[moma.org](https://www.moma.org/x)",
    )


def test_split_keeps_source_lines_whole():
    source = "Source: [Plan your visit, moma.org](https://www.moma.org/visit/) (checked November 21, 2026)."
    assert llm.split(f"Open 10:30 a.m. daily.\n\n{source}") == ("Open 10:30 a.m. daily.", source)


# ------------------------------------------------------------------ check()


def test_check_accepts_a_faithful_rewrite():
    assert llm.check(PROSE, GOOD) is None


@pytest.mark.parametrize(
    ("rewrite", "reason"),
    [
        ("", "empty reply"),
        ("Question: Where is The Starry Night?\nFACTS:\n" + PROSE, "echoed the prompt"),
        (GOOD + " See [this](https://example.com/x).", "added link"),
        (GOOD.replace("Gallery 501", "Gallery 510"), "added number 510"),
        (GOOD + " It was painted in 1890.", "added number 1890"),
        (GOOD.replace(" (1889)", ""), "dropped number 1889"),
        (GOOD.replace("**The Starry Night**", "this painting"), "dropped 'The Starry Night'"),
    ],
)
def test_check_rejects_unsafe_rewrites(rewrite, reason):
    assert llm.check(PROSE, rewrite).startswith(reason)


def test_check_rejects_a_dropped_link():
    assert llm.check(FACTS, GOOD).startswith("dropped link")


def test_check_ignores_thousands_separators():
    assert llm.check("There are 1,556 works on view.", "1556 works are on view.") is None


# ------------------------------------------------------------------ polish()


def test_off_by_default_never_loads_a_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_MODEL", "")
    pipe = use(monkeypatch, GOOD)
    assert llm.polish(QUESTION, FACTS) == FACTS
    assert pipe.calls == []


def test_safe_rewrite_gets_the_links_and_data_date_back(monkeypatch):
    pipe = use(monkeypatch, GOOD)
    assert llm.polish(QUESTION, FACTS + FOOTER) == f"{GOOD}\n\n{LINK}{FOOTER}"
    system, user = pipe.calls[0]
    assert user == {"role": "user", "content": QUESTION}
    assert "Gallery 501" in system["content"]
    assert "http" not in system["content"] and "Collection data as of" not in system["content"]


def test_unsafe_rewrite_falls_back_to_the_template(monkeypatch):
    use(monkeypatch, GOOD.replace("Gallery 501", "Gallery 510"))
    assert llm.polish(QUESTION, FACTS + FOOTER) == FACTS + FOOTER


def test_lists_are_not_reworded(monkeypatch):
    pipe = use(monkeypatch, GOOD)
    listing = "**Frida Kahlo**, 3 works on view:\n\n- [Fulang-Chang and I](https://www.moma.org/x)"
    assert llm.polish("What works by Frida Kahlo are on view?", listing) == listing
    assert pipe.calls == []


def test_generation_error_falls_back(monkeypatch):
    use(monkeypatch, RuntimeError("out of memory"))
    assert llm.polish(QUESTION, FACTS) == FACTS


def test_a_model_that_fails_to_load_is_tried_once(monkeypatch, caplog):
    attempts = []

    def broken(model):
        attempts.append(model)
        raise OSError("no such model")

    monkeypatch.setattr(llm, "_build_pipeline", broken)
    assert llm.polish(QUESTION, FACTS) == FACTS
    assert llm.polish(QUESTION, FACTS) == FACTS
    assert attempts == ["fake/model"]
    assert "using templated answers" in caplog.text


# ------------------------------------------------------------- real model


@pytest.mark.llm
@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="the `llm` extra is not installed"
)
def test_real_model_loads_and_output_is_checked(monkeypatch):
    monkeypatch.setattr(
        config, "LLM_MODEL", os.environ.get("ASKMOMA_TEST_LLM", "HuggingFaceTB/SmolLM2-135M-Instruct")
    )
    assert llm.load() is not None, "the model did not load; see the warning in the log"
    result = llm.polish(QUESTION, FACTS + FOOTER)
    assert result.endswith(f"{LINK}{FOOTER}")  # links and the data date always survive
    body = result[: -len(f"\n\n{LINK}{FOOTER}")]
    assert body == PROSE or llm.check(PROSE, body) is None
