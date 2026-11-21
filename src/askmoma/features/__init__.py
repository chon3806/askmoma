"""One module per use case. Each takes the Collection and the visitor's message and returns an Answer.

| Module        | Use case                                   | Status  |
| ------------- | ------------------------------------------ | ------- |
| locate.py     | 1. Where is a work?                        | working |
| (rank.py)     | 2. Top artists on view, museum or floor    | to do   |
| (recommend.py)| 3. Related works                           | to do   |
| (search.py)   | 4. Search by description (embeddings)      | to do   |
| (faq.py)      | Visitor FAQ                                | to do   |

A new feature: add its module here, then a route in askmoma/chat/router.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Answer:
    """What AskMoMA says back."""

    text: str  # Markdown for the chat
    works: tuple[int, ...] = ()  # MoMA object_ids mentioned, best first
