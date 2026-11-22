"""Tests for the HTML that app.py builds: answers, artwork cards, and the conversation."""

from __future__ import annotations

from pathlib import Path

import app

CARD = {
    "row": 0,
    "title": "Blue Dancers",
    "artist": "Tomas Arbel",
    "date": "1910",
    "location": "Floor 5, Gallery 502",
    "on_view": True,
    "link": "https://www.moma.org/collection/works/90004",
    "image": "",
    "reason": "In the same gallery (502)",
}


def test_markdown_bold_links_and_lists():
    html = app.md("**Blue Dancers** is here.\n\n- one\n- two\n\n[moma.org](https://www.moma.org/x)")
    assert "<strong>Blue Dancers</strong>" in html
    assert "<ul><li>one</li><li>two</li></ul>" in html
    assert "<a href='https://www.moma.org/x' target='_blank' rel='noopener'>moma.org</a>" in html


def test_markdown_escapes_html_and_refuses_script_links():
    html = app.md("<script>alert(1)</script> [click](javascript:alert(1))")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "href='javascript" not in html


def test_data_date_line_is_styled():
    assert "<p class='asof'>" in app.md("Answer.\n\n_Collection data as of November 21, 2026._")


def test_card_shows_location_reason_and_link():
    html = app.works_html([CARD])
    assert "href='https://www.moma.org/collection/works/90004'" in html
    assert "Floor 5, Gallery 502" in html and "In the same gallery (502)" in html


def test_card_escapes_text_from_the_data():
    html = app.works_html([{**CARD, "title": "<img src=x onerror=alert(1)>", "on_view": False}])
    assert "<img src=x" not in html and "&lt;img" in html
    assert "Not on view" in html


def test_conversation_is_a_live_region_and_escapes_the_question():
    html = app.thread_html([["<b>hi</b>", app.md("Hello."), []]])
    assert "role='log'" in html and "aria-live='polite'" in html
    assert "&lt;b&gt;hi&lt;/b&gt;" in html


def test_empty_conversation_shows_the_greeting():
    html = app.thread_html([])
    assert "I'm AskMoMA" in html and "role='log'" in html


def test_chat_window_can_be_resized_by_mouse_and_keyboard():
    head = app.CHAT_RESIZE_HEAD
    assert "chat-grip" in head and "pointerdown" in head  # drag
    assert "ArrowLeft" in head and "dblclick" in head  # keyboard, and reset
    assert "Resize the chat window" in head and "Make the chat window larger" in head  # labelled controls
    assert ".chat-size-bar" in app.CSS


def test_new_chat_button_sits_in_the_chat_header():
    src = Path(app.__file__).read_text(encoding="utf-8")
    head = src[
        src.index('elem_id="chat-head"') : src.index("popup_banner_html()", src.index('elem_id="chat-head"'))
    ]
    assert 'gr.Button("New chat", elem_id="newchat"' in head
    assert "#chat-head #newchat" in app.CSS
