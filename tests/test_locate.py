"""Use case 1: locating a work."""

from askmoma.features.locate import locate, title_in


def test_locates_the_starry_night(moma):
    answer = locate(moma, "Where is The Starry Night?")
    assert "MoMA, Floor 5, Gallery 501**" in answer.text  # no stray quote from MoMA's file
    assert "https://www.moma.org/collection/works/79802" in answer.text
    assert answer.works == (79802,)


def test_tolerates_a_typo_and_a_missing_article(moma):
    assert "Gallery 501" in locate(moma, "where is starry nite").text


def test_floor_only_locations_are_reported_as_given(moma):
    work = moma.works[moma.works["gallery"].isna()].iloc[0]
    assert str(work["on_view"]) in locate(moma, f"Where is {work['title']}?").text


def test_says_so_when_nothing_matches(moma):
    answer = locate(moma, "Where is the Mona Lisa?")
    assert answer.works == () and "couldn't find" in answer.text


def test_question_words_are_removed_from_the_title():
    assert title_in("Where is the painting The Starry Night?") == "The Starry Night"
    assert title_in("find Water Lilies") == "Water Lilies"
