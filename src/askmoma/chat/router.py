"""Decide which feature answers a message.

For now every message goes to use case 1 (locate). As features are added,
each gets a route here: a rule or, later, a model that recognises the
request ("top artists on Floor 5" -> rank, "I loved X" -> recommend). How to
recognise requests is one of the open design questions in docs/design-notes.md.
"""

from __future__ import annotations

from functools import lru_cache

from askmoma.data.collection import Collection
from askmoma.features import Answer
from askmoma.features.locate import locate

WELCOME = "Hi! I'm AskMoMA. Ask me where a work on view at MoMA is, for example *Where is The Starry Night?*"


@lru_cache(maxsize=1)
def collection() -> Collection:
    """The data, loaded on first use and shared."""
    return Collection()


def answer(message: str) -> Answer:
    """AskMoMA's reply to one message."""
    if not message or not message.strip():
        return Answer(WELCOME)
    return locate(collection(), message)
