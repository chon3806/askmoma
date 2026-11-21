"""The Gradio chat page.

    mise run app           (or: uv run askmoma)

Then open the address it prints. The page only displays; every answer comes
from askmoma.chat.router, so the logic can be tested without a browser.
"""

from __future__ import annotations

import gradio as gr

from askmoma.chat.router import WELCOME, answer

EXAMPLES = ["Where is The Starry Night?", "Where is The Persistence of Memory?"]


def respond(message: str, history: list) -> str:
    """One chat turn."""
    return answer(message).text


def build_interface() -> gr.ChatInterface:
    return gr.ChatInterface(
        respond,
        title="AskMoMA",
        description=(
            "A course project built on MoMA's open collection data (CC0). Not affiliated with MoMA. "
            "Gallery locations change: check the linked moma.org page."
        ),
        examples=EXAMPLES,
        chatbot=gr.Chatbot(
            value=[{"role": "assistant", "content": WELCOME}], label="Conversation with AskMoMA"
        ),
        textbox=gr.Textbox(label="Your question", placeholder="Where is ...?"),
    )


def main() -> None:
    build_interface().launch()


if __name__ == "__main__":
    main()
