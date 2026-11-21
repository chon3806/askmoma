"""Use case 1: locate a work on view at MoMA.

"Where is The Starry Night?" -> venue, floor, gallery, date, medium, and the
work's page on moma.org. Answers come only from MoMA's data, so a location can
always be traced to its source. If no title is close enough, the answer says
so and points to moma.org instead of guessing.
"""

from __future__ import annotations

import re

from askmoma.config import MOMA_COLLECTION_URL
from askmoma.data.collection import Collection, where
from askmoma.features import Answer

# Words that start a question but are not part of the title.
LEAD_IN = re.compile(
    r"^(?:where(?: is| are|'s)?|find|locate|show me|can i see|i want to see)\s+(?:the (?:painting|work) )?",
    re.IGNORECASE,
)


def title_in(question: str) -> str:
    """The title part of a location question: "Where is The Starry Night?" -> "The Starry Night"."""
    return LEAD_IN.sub("", question.strip().rstrip("?").strip())


def locate(collection: Collection, question: str) -> Answer:
    """Answer "where is ...?" from the data."""
    rows = collection.find_title(title_in(question))
    if not rows:
        return Answer(
            "I couldn't find that title among the works currently on view. "
            f"Try MoMA's collection search: {MOMA_COLLECTION_URL}"
        )
    lines = []
    for index in rows[:3]:
        w = collection.works.loc[index]
        lines.append(
            f"**{w['title']}** by {w['artist']} ({w['date']}) is on view at **{where(w)}**.  \n"
            f"{w['medium']}. [View on moma.org]({w['url']})"
        )
    intro = "" if len(rows) == 1 else "I found more than one match:\n\n"
    ids = tuple(int(collection.works.loc[i, "object_id"]) for i in rows)
    return Answer(intro + "\n\n".join(lines), ids)
