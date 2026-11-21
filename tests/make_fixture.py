"""Write a tiny FICTIONAL dataset with the same columns as MoMA's CSVs.

Used only by the tests so they run without the real 160k-row files. None of
these artists or titles are real MoMA records.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent / "fixtures"

ARTISTS = [
    # cid, name, bio, nationality, gender, begin, end
    ("1001", "Mira Kestrel", "Dutch, 1850–1891", "Dutch", "Female", "1850", "1891"),
    ("1002", "Tomas Arbel", "French, 1869–1954", "French", "Male", "1869", "1954"),
    ("1003", "Ines Calder", "Mexican, 1907–1954", "Mexican", "Female", "1907", "1954"),
    ("1004", "Paul Okafor", "American, 1912–1956", "American", "Male", "1912", "1956"),
    ("1005", "Ana Lind", "Swedish, born 1930", "Swedish", "Female", "1930", "0"),
    ("1006", "Hugo Brant", "Spanish, 1881–1973", "Spanish", "Male", "1881", "1973"),
    ("1007", "Lena Ward", "American, 1928–1987", "American", "Female", "1928", "1987"),
]

WORKS = [
    # title, cids, date, medium, classification, on_view, acquired
    (
        "The Quiet Harbor at Night",
        "1001",
        "1889",
        "Oil on canvas",
        "Painting",
        "MoMA, Floor 5, 501",
        "1941-01-01",
    ),
    ("Olive Grove in Wind", "1001", "1889", "Oil on canvas", "Painting", "", "1950-05-02"),
    ("Moonlit Field", "1002", "1895", "Oil on canvas", "Painting", "MoMA, Floor 5, 501", "1960-03-01"),
    ("Blue Dancers", "1002", "1910", "Oil on canvas", "Painting", "MoMA, Floor 5, 502", "1939-06-01"),
    ("Studio Window", "1002", "1912", "Oil on canvas", "Painting", "Floor 5, Gallery 502", "1955-01-01"),
    (
        "Self-Portrait with Monkeys",
        "1003",
        "1940",
        "Oil on canvas",
        "Painting",
        "MoMA, Floor 5, 503",
        "1945-01-01",
    ),
    ("Number 7", "1004", "1950", "Enamel on canvas", "Painting", "Floor 4, Gallery 403", "1968-01-01"),
    ("Drip Study", "1004", "1951", "Enamel on paper", "Drawing", "Floor 4, Gallery 403", "1970-01-01"),
    ("Untitled", "1007", "1962", "Acrylic on canvas", "Painting", "MoMA, Floor 4, 412", "1980-01-01"),
    ("Untitled", "1005", "1970", "Gelatin silver print", "Photograph", "MoMA Gallery 208", "1990-01-01"),
    ("Night Garden", "1005", "1965", "Oil on linen", "Painting", "MoMA Gallery 208", "1985-01-01"),
    ("Chair Study", "1005, 1006", "1958", "Bent plywood", "Design", "MoMA, Floor 3, 305", "1959-01-01"),
    ("The Guitar Player", "1006", "1913", "Oil on canvas", "Painting", "MoMA, Floor 5, 503", "1952-01-01"),
    ("Bottle and Glass", "1006", "1914", "Cut paper and charcoal", "Drawing", "", "1961-01-01"),
    ("Sunrise Over Water", "1001", "1888", "Oil on canvas", "Painting", "MoMA, Floor 5, 501", "1970-01-01"),
]


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        ARTISTS,
        columns=[
            "ConstituentID",
            "DisplayName",
            "ArtistBio",
            "Nationality",
            "Gender",
            "BeginDate",
            "EndDate",
        ],
    ).to_csv(HERE / "Artists.csv", index=False, lineterminator="\n")  # same bytes on every OS
    names = {a[0]: a[1] for a in ARTISTS}
    bios = {a[0]: a[2] for a in ARTISTS}
    rows = []
    for i, (title, cids, date, medium, cls, onview, acq) in enumerate(WORKS, start=1):
        ids = [c.strip() for c in cids.split(",")]
        rows.append(
            {
                "Title": title,
                "Artist": ", ".join(names[c] for c in ids),
                "ConstituentID": cids,
                "ArtistBio": " ".join(f"({bios[c]})" for c in ids),
                "Date": date,
                "Medium": medium,
                "Dimensions": "73.7 x 92.1 cm",
                "CreditLine": "Fictional test gift",
                "Classification": cls,
                "Department": "Painting & Sculpture",
                "DateAcquired": acq,
                "ObjectID": str(90000 + i),
                "URL": "",
                "OnView": onview,
            }
        )
    pd.DataFrame(rows).to_csv(HERE / "Artworks.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
