"""Scenario tests on the fictional fixture. Run: python -m pytest tests -q"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["ASKMOMA_DATA_AS_OF"] = "November 20, 2026"

from tests import make_fixture  # noqa: E402

make_fixture.main()

from askmoma import AskMoMA, load  # noqa: E402

BOT = AskMoMA(load(ROOT / "tests" / "fixtures"))


def ask(q, ctx=None, **kw):
    return BOT.ask(q, ctx, **kw)


def test_locate_work():
    a = ask("Where is The Quiet Harbor at Night?")
    assert "Floor 5, Gallery 501" in a.text and "moma.org" in a.text


def test_locate_without_article_and_typo():
    assert "Gallery 501" in ask("where can i find quiet harbor at night").text
    assert "Gallery 502" in ask("where is blue dancer").text  # fuzzy


def test_not_on_view():
    assert "not listed as on view" in ask("Where is Olive Grove in Wind?").text


def test_who_made():
    a = ask("Who painted Self-Portrait with Monkeys?")
    assert "Ines Calder" in a.text and a.intent == "who_made"


def test_followup_pronoun():
    first = ask("Who painted Number 7?")
    a = ask("When was it made?", first.context)
    assert "1950" in a.text
    b = ask("Where is it?", first.context)
    assert "Gallery 403" in b.text


def test_works_by_artist_and_surname():
    a = ask("What works by Tomas Arbel are on view?")
    assert "Moonlit Field" in a.text and "Blue Dancers" in a.text
    assert "Moonlit Field" in ask("show me paintings by Arbel").text


def test_artist_info():
    a = ask("Who was Mira Kestrel?")
    assert "Dutch" in a.text and "3" in a.text  # 3 works in fixture


def test_ambiguous_title():
    a = ask("Tell me about Untitled")
    assert a.intent == "disambiguate" and "Lena Ward" in a.text and "Ana Lind" in a.text


def test_recommend():
    a = ask("I loved The Quiet Harbor at Night. What else should I see?")
    assert "Moonlit Field" in a.text  # same gallery
    assert "Sunrise Over Water" in a.text  # same artist / gallery


def test_top_artists_floor():
    a = ask("Which artists have the most works on floor 5?")
    assert a.text.index("Tomas Arbel") < a.text.index("Ines Calder")


def test_floor_summary_counts_then_offers_next_steps():
    a = ask("What's on floor 4?").text
    assert "works on view by" in a and "artists" in a
    assert "top 5 artists" in a


def test_top_five_then_artist_link():
    a = ask("Which artists have the most works on floor 5?")
    chips = [c for c in a.suggestions if c.startswith("MoMA link for")]
    assert 1 <= len(chips) <= 5
    link = ask(chips[0])
    assert "https://www.moma.org/search/?q=" in link.text
    assert "works on view on floor 5" in link.text or "on floor 5" in link.text


def test_gallery_list_still_works():
    assert "Night Garden" in ask("what is in gallery 208").text


def test_nationality():
    a = ask("Which American artists are on view?")
    assert "Paul Okafor" in a.text and "Lena Ward" in a.text


def test_medium_dimensions_acquired():
    assert "Enamel on canvas" in ask("What medium is Number 7?").text
    assert "73.7" in ask("How big is Blue Dancers?").text
    assert "1939" in ask("When did MoMA acquire Blue Dancers?").text


def test_co_authored_work():
    a = ask("Who made Chair Study?")
    assert "Ana Lind, Hugo Brant" in a.text


def test_description_search():
    a = ask("find night paintings")
    assert any(w in a.text for w in ["Night Garden", "The Quiet Harbor at Night"])


def test_faq():
    assert "8:30" in ask("What time do you close on Friday?").text
    assert "$30" in ask("How much are tickets?").text
    assert "care partner" in ask("Is it wheelchair accessible?").text


def test_not_found_is_honest():
    a = ask("qwzx blorf")
    assert a.intent == "not_found" and "moma.org" in a.text


def test_every_artwork_answer_states_date():
    for q in ["Where is Moonlit Field?", "Who made Drip Study?", "What works by Ines Calder are on view?"]:
        assert "as of November 20, 2026" in ask(q).text


def test_followup_artist_pronoun_and_unknown_artist():
    first = ask("Where is Moonlit Field?")
    a = ask("What else did they make?", first.context)
    assert "Tomas Arbel" in a.text and "Blue Dancers" in a.text
    assert ask("Who is Pablo Nobody?").intent == "not_found"


def test_suggestion_chips_are_answerable():
    a = ask("Where is The Quiet Harbor at Night?")
    for s in a.suggestions:
        assert ask(s, a.context).intent != "not_found", s


def test_title_that_is_also_an_artist_name():
    # Add a work titled with an artist's name, as real photo records often are.
    from askmoma.data import load as _load

    col = _load(ROOT / "tests" / "fixtures")
    bot = AskMoMA(col)
    row = col.works.index[col.works["Title"] == "Drip Study"][0]
    col.title_index["ines calder"] = [row]
    a = bot.ask("What works by Ines Calder are on view?")
    assert "Self-Portrait with Monkeys" in a.text
    assert "Drip Study" in bot.ask("Where is Ines Calder?").text


def test_double_typo_and_descriptive_search():
    assert "Gallery 501" in ask("where is quiet harber at nite").text
    a = ask("find dreamy blue moonlit paintings")
    assert a.intent == "search"


def test_artist_typo():
    assert "Blue Dancers" in ask("show me paintings by Arbell").text
    assert "Okafor" in ask("who is paul okafr").text
