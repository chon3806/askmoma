"""Conversation features in the chat window, from lessons in the course's earlier projects.

- Week 7 CUX redesign: say honestly that the bot is automated, close politely,
  and repair after a miss instead of repeating the same answer.
- Week 9-10 usability: "Was this helpful?" measures task success; repairs fit
  what went wrong (Langevin et al.'s heuristics for conversational agents).
- Week 12 ethics audit: transparency and privacy in what the bot says about itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app
from askmoma import AskMoMA, config, load

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def bot():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "SEARCH_BACKEND", "tfidf")
        yield AskMoMA(load(FIXTURES))


@pytest.mark.parametrize(
    "question", ["Are you a person?", "Is this a bot?", "am I talking to a human", "Who are you?"]
)
def test_says_it_is_automated(bot, question):
    a = bot.ask(question)
    assert a.intent == "identity"
    assert "automated guide, not a person" in a.text and "don't keep your conversation" in a.text


@pytest.mark.parametrize("message", ["Thanks!", "thank you so much", "ok, bye", "That's all"])
def test_closes_politely(bot, message):
    assert bot.ask(message).intent == "goodbye"


def test_thanks_inside_a_question_is_still_the_question(bot):
    assert bot.ask("Thanks, where is Moonlit Field?").intent == "location"


def test_context_remembers_the_last_question_for_repair(bot):
    a = bot.ask("Where is Moonlit Field?")
    assert a.context["last_intent"] == "location" and a.context["last_question"] == "Where is Moonlit Field?"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("find night paintings", "a few specific words"),
        ("I loved Moonlit Field and Night Garden", "what you didn't like"),
        ("Where is Moonlit Field?", "add the artist's name"),
        ("Where is Girl with a Pearl Earring?", "may not be in MoMA's collection"),
    ],
)
def test_not_helpful_offers_a_repair_that_fits(bot, question, expected):
    a = bot.ask(question)
    r = bot.not_helpful(a.context)
    assert r.intent == "repair" and r.text.startswith("Sorry that didn't help.")
    assert expected in r.text
    assert "moma.org/search/?q=" in r.text  # always a way forward
    assert r.text != a.text


def test_not_helpful_after_which_one_asks_for_the_artist(bot):
    a = bot.ask("Tell me about Untitled")
    assert bot.not_helpful(a.context).intent == "clarify"


def test_greeting_in_the_chat_window_is_transparent():
    html = app.thread_html([])
    assert "automated guide" in html and "don't keep this conversation" in html


def test_repair_turn_has_no_question_bubble():
    html = app.thread_html([["", app.md("Sorry that didn't help."), []]])
    assert "class='bubble'" not in html and "Sorry that didn't help." in html


def test_keyboard_escape_closes_the_chat():
    assert '"Escape"' in app.CHAT_RESIZE_HEAD and "#ask textarea" in app.CHAT_RESIZE_HEAD
