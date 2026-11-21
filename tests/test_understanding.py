"""Regression tests for mistakes found by asking AskMoMA questions on MoMA's real data.

Each case is rebuilt here with FICTIONAL artists and titles that reproduce the
same trap: an ordinary word that is also a surname, a full name that is also a
title, a title with and without "The", a MoMA PS1 location, and so on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from askmoma import AskMoMA, config, load
from askmoma.data import parse_on_view

ARTISTS = [
    # cid, name, bio, nationality
    ("1", "Nora Sky", "American, 1900–1980", "American"),
    ("2", "Nordic Pictures", "", "Nationality unknown"),
    ("3", "Paul French", "American, 1905–1988", "American"),
    ("4", "Anne Duval", "French, 1870–1950", "French"),
    ("5", "Lena Wearing", "British, born 1963", "British"),
    ("6", "J. A. Wery", "", "Nationality unknown"),
    ("7", "Tom Painter", "American, 1930–2010", "American"),
    ("8", "Rosa Chappell", "American, 1920–1999", "American"),
]
WORKS = [
    # title, cid, classification, on view
    ("Night Sky over the Bay", "4", "Painting", "MoMA, Floor 5, 501"),
    ("Starlit Harbor", "1", "Painting", "MoMA, Floor 5, 501"),
    ("Street Corner", "2", "Photograph", ""),
    ("Flag", "7", "Painting", "MoMA, Floor 4, 408"),
    ("Flag", "4", "Painting", "MoMA, Floor 5, 502"),
    ("Tom Painter", "5", "Photograph", "MoMA, Floor 3, 305"),
    ("The Quiet Night", "4", "Painting", "MoMA, Floor 5, 501"),
    ("Quiet Night", "3", "Print", ""),
    ("Roslin Chapel", "8", "Photograph", "MoMA, Floor 3, 305"),
    ("Colored Me", "1", "Painting", '"MoMA PS1, Second floor"'),
    ("Harbor Lights", "6", "Painting", ""),
    ("Window Study", "3", "Drawing", ""),
    ("Sea at Dusk", "4", "Painting", "MoMA, Floor 5, 502"),
    ("City Streets at Noon", "5", "Photograph", "MoMA, Floor 3, 305"),
]


@pytest.fixture(scope="module")
def bot(tmp_path_factory):
    folder = tmp_path_factory.mktemp("traps")
    pd.DataFrame(
        [(cid, name, bio, nat, "", "0", "0") for cid, name, bio, nat in ARTISTS],
        columns=[
            "ConstituentID",
            "DisplayName",
            "ArtistBio",
            "Nationality",
            "Gender",
            "BeginDate",
            "EndDate",
        ],
    ).to_csv(folder / "Artists.csv", index=False)
    names = {a[0]: a[1] for a in ARTISTS}
    bios = {a[0]: f"({a[2]})" if a[2] else "" for a in ARTISTS}
    pd.DataFrame(
        [
            {
                "Title": title,
                "Artist": names[cid],
                "ConstituentID": cid,
                "ArtistBio": bios[cid],
                "Date": "1950",
                "Medium": "Oil on canvas",
                "Classification": cls,
                "Department": "Painting & Sculpture",
                "ObjectID": str(100 + i),
                "URL": "",
                "OnView": on_view,
            }  # fmt: skip
            for i, (title, cid, cls, on_view) in enumerate(WORKS)
        ]
    ).to_csv(folder / "Artworks.csv", index=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "SEARCH_BACKEND", "tfidf")  # deterministic and offline
        yield AskMoMA(load(folder))


def ask(bot, question):
    return bot.ask(question)


# ------------------------------------------- ordinary words are not artists


@pytest.mark.parametrize(
    "question",
    ["Find dreamy night sky paintings", "find pictures of city streets", "show me night sky scenes"],
)
def test_description_search_is_not_taken_over_by_a_surname(bot, question):
    a = ask(bot, question)
    assert a.intent == "search"
    assert "Nora Sky" not in a.text.splitlines()[0] and "Nordic Pictures" not in a.text.splitlines()[0]


def test_nationality_word_in_a_title_is_not_a_surname(bot):
    a = ask(bot, "Where is French Window at Collioure?")
    assert a.intent == "not_found" and "Paul French" not in a.text


@pytest.mark.parametrize(
    ("question", "wrong"),
    [
        ("Where is Girl with a Pearl Earring?", "Wearing"),
        ("Where is the Sistine Chapel?", "Rosa Chappell"),
        ("asdf qwerty zxcv", "Wery"),
    ],
)
def test_unknown_works_are_reported_honestly(bot, question, wrong):
    a = ask(bot, question)
    assert a.intent == "not_found"
    assert wrong not in a.text.split("\n\n")[0]


def test_single_misspelled_surname_still_works(bot):
    a = ask(bot, "show me paintings by Duvall")
    assert a.intent == "works_by_artist" and "Anne Duval" in a.text


def test_full_artist_name_is_still_an_artist(bot):
    assert ask(bot, "Who is Paul French?").intent == "artist_info"


# ------------------------------------------------------------ titles


def test_title_without_the_finds_the_work_on_view(bot):
    a = ask(bot, "Where is Quiet Night?")
    assert "The Quiet Night" in a.text and "Gallery 501" in a.text


def test_artist_named_picks_between_works_sharing_a_title(bot):
    a = ask(bot, "Where is Flag by Tom Painter?")
    assert a.intent == "location" and "Floor 4, Gallery 408" in a.text


def test_question_about_a_place_without_a_work_name_searches(bot):
    assert ask(bot, "Where can I see paintings of the sea?").intent == "search"


# ------------------------------------------------------- visit questions


def test_floor_question_with_open_is_not_the_hours(bot):
    assert ask(bot, "What's open on floor 5?").intent == "floor_list"


def test_price_word_with_a_title_is_not_admission(bot):
    a = ask(bot, "How much is The Quiet Night worth?")
    assert a.intent == "work_info" and "The Quiet Night" in a.text


def test_faq_address(bot):
    a = ask(bot, "What is MoMA's address?")
    assert a.intent == "faq_address" and "11 West 53 Street" in a.text


# ------------------------------------------------------------ answers


def test_artist_bio_has_one_pair_of_brackets(bot):
    a = ask(bot, "Who painted Starlit Harbor?")
    assert "(American, 1900–1980)" in a.text and "((" not in a.text


def test_ps1_location_names_the_other_building(bot):
    a = ask(bot, "Where is Colored Me?")
    assert "MoMA PS1 in Queens" in a.text and '"' not in a.text.split("\n")[0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"MoMA, Floor 5, 501"', (5, "501", "Floor 5, Gallery 501")),
        (
            '"MoMA PS1, Second floor"',
            (None, None, "MoMA PS1 in Queens (not the Manhattan museum), Second floor"),
        ),
        ("MoMA PS1", (None, None, "MoMA PS1 in Queens (not the Manhattan museum)")),
        ('""', (None, None, "")),
    ],
)
def test_parse_on_view_strips_quotes_and_marks_ps1(raw, expected):
    assert parse_on_view(raw) == expected
