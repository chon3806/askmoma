"""AskMoMA's chat page (starter version: use case 1, locating a work).

    python app.py

Then open the address it prints. Later use cases (ranking artists, related
works, search by description, and the visitor FAQ) plug in here as the team
builds them; see the README's roadmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from askmoma import Collection  # noqa: E402

collection = Collection()

WELCOME = ("Hi! I'm AskMoMA. Ask me where a work on view at MoMA is, for example "
           "*Where is The Starry Night?*")


def respond(message: str, history: list) -> str:
    """One chat turn: for now, every question is treated as a location question."""
    return collection.locate(message).text


def build_interface() -> gr.ChatInterface:
    return gr.ChatInterface(
        respond,
        title="AskMoMA",
        description="A course project built on MoMA's open collection data (CC0). "
                    "Gallery locations change: check the linked moma.org page.",
        examples=["Where is The Starry Night?", "Where is The Persistence of Memory?"],
        chatbot=gr.Chatbot(value=[{"role": "assistant", "content": WELCOME}],
                           label="Conversation with AskMoMA"),
        textbox=gr.Textbox(label="Your question", placeholder="Where is ...?"),
    )


if __name__ == "__main__":
    build_interface().launch()
