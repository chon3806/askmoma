"""Recommendations and conversation repair, using lessons from earlier course projects.

- Week 11 recommender (CineMatch): several likes and dislikes form one profile
  (Rocchio), lists have variety (maximal marginal relevance), every pick says why,
  and suggestion chips let the visitor refine the list.
- Week 12 ethics audit: every stated reason must be true.
- Week 7 CUX redesign and week 9-10 usability heuristics: "None of these" is a way
  out when AskMoMA's guess about which work was meant is wrong.

Runs on the fictional fixture with keyword search, so the results are deterministic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from askmoma import AskMoMA, config, load
from askmoma.nlu import split_preferences

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def bot():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "SEARCH_BACKEND", "tfidf")
        yield AskMoMA(load(FIXTURES))


def titles(answer):
    return [c["title"] for c in answer.cards]


def assert_reasons_are_true(bot, answer, liked_titles=()):
    works = bot.col.works
    for card in answer.cards:
        why, row = card["reason"], card["row"]
        if m := re.fullmatch(r"In the same gallery \((\w+)\)", why):
            assert works.at[row, "gallery"] == m.group(1), card
        elif m := re.fullmatch(r"Also by (.+)", why):
            assert m.group(1) in card["artist"], card
        elif m := re.fullmatch(r"Same period \((\d+)–(\d+)\)", why):
            assert int(m.group(1)) <= works.at[row, "year"] <= int(m.group(2)), card
        elif m := re.fullmatch(r"Close to (.+)", why):
            assert m.group(1) in liked_titles, card
        else:
            assert why == "Similar in subject or style", card


# ------------------------------------------------------------ one liked work


def test_single_like_has_variety_and_true_reasons(bot):
    a = bot.ask("I loved The Quiet Harbor at Night. What else should I see?")
    assert a.intent == "recommend" and a.cards
    other_artists = [c["artist"] for c in a.cards if not c["reason"].startswith("Also by")]
    assert len(other_artists) == len(set(other_artists)), (
        "one work per artist unless the reason is the artist"
    )
    assert_reasons_are_true(bot, a)


# ------------------------------------------------- several likes and dislikes


def test_two_likes_build_one_profile(bot):
    a = bot.ask("I loved The Quiet Harbor at Night and Moonlit Field")
    assert a.text.startswith("Because you liked **The Quiet Harbor at Night** and **Moonlit Field**")
    assert {"The Quiet Harbor at Night", "Moonlit Field"}.isdisjoint(titles(a))
    assert_reasons_are_true(bot, a, liked_titles=("The Quiet Harbor at Night", "Moonlit Field"))
    assert a.context["liked"] and a.context["disliked"] == []


def test_dislike_is_named_and_left_out(bot):
    a = bot.ask("I loved Moonlit Field and Night Garden but not Blue Dancers")
    assert "but not **Blue Dancers**" in a.text
    assert "Blue Dancers" not in titles(a)
    assert_reasons_are_true(bot, a, liked_titles=("Moonlit Field", "Night Garden"))


def test_split_preferences_keeps_titles_that_contain_and(bot):
    liked, disliked = split_preferences(bot.col, "I loved Bottle and Glass and Night Garden but not Number 7")
    name = lambda rows: bot.col.works.at[rows[0], "Title"]  # noqa: E731
    assert [name(r) for r in liked] == ["Bottle and Glass", "Night Garden"]
    assert [name(r) for r in disliked] == ["Number 7"]


def test_feedback_chips_refine_the_list(bot):
    a = bot.ask("I loved The Quiet Harbor at Night. What else should I see?")
    first = a.cards[0]["title"]
    assert f"More like {first}" in a.suggestions
    refine = next(s for s in a.suggestions if " but not " in s)
    b = bot.ask(refine, a.context)
    assert b.intent == "recommend" and first not in titles(b)


# ------------------------------------------------------------ repair


def test_none_of_these_asks_for_the_artist_instead_of_guessing(bot):
    a = bot.ask("Tell me about Untitled")
    assert a.intent == "disambiguate" and "None of these" in a.suggestions
    b = bot.ask("None of these", a.context)
    assert b.intent == "clarify" and "artist" in b.text
    assert "choices" not in b.context
    # Only the question just asked can be answered with "none of these".
    assert bot.ask("none", b.context).intent != "clarify"
