"""Parsing MoMA's location text, and the committed data file."""

import pandas as pd

from askmoma.data.fetch import split_locations


def test_location_text_is_split_into_venue_floor_and_gallery():
    parts = split_locations(pd.Series(['"MoMA, Floor 5, 501"', "MoMA, Floor 6", "MoMA PS1, Second floor"]))
    assert parts.iloc[0].tolist() == ["MoMA, Floor 5, 501", "MoMA", "5", "501"]
    assert parts.iloc[1]["floor"] == "6" and pd.isna(parts.iloc[1]["gallery"])
    assert parts.iloc[2]["venue"] == "MoMA PS1" and parts.iloc[2]["floor"] == "Second"


def test_committed_data_has_the_columns_the_features_use(moma):
    needed = {"object_id", "title", "artist", "date", "medium", "venue", "floor", "gallery", "url"}
    assert needed <= set(moma.works.columns) and len(moma) > 1000
