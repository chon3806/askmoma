"""The router and the chat page's handler."""

from askmoma.chat.router import WELCOME, answer
from askmoma.ui.app import build_interface, respond


def test_empty_message_gets_the_welcome():
    assert answer("   ").text == WELCOME


def test_chat_handler_answers_from_the_data():
    assert "Floor 5, Gallery 501" in respond("Where is The Starry Night?", [])


def test_page_builds_with_labelled_controls():
    demo = build_interface()
    assert demo.chatbot.label and demo.textbox.label
