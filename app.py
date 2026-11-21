"""AskMoMA: Gradio UI.

python app.py                         # web UI at http://127.0.0.1:7860
python app.py --port 7861             # if 7860 is busy
python app.py --data tests/fixtures   # small fictional test data
python app.py --cli                   # chat in the terminal
"""

from __future__ import annotations

import argparse
import functools
import html
import logging
import os
import re
import sys
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gradio 6.0.*")

STARTERS = [
    "Where is The Starry Night?",
    "What works by Frida Kahlo are on view?",
    "Which artists have the most works on floor 5?",
    "Find dreamy night sky paintings",
]
N_CHIPS = 4

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');

:root, .dark, body {
  --bg:#f6f7f9; --panel:#ffffff; --side:#0f172a; --side-2:#1e293b; --side-text:#cbd5e1;
  --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --accent:#2f5bea; --accent-soft:#eef2ff;
  --good:#047857; --good-bg:#ecfdf5;
  --ff:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
}
body, gradio-app {background:var(--bg) !important;}
.gradio-container {max-width:100% !important; padding:0 !important; margin:0 !important; background:var(--bg) !important;
  color:var(--ink) !important; font-family:var(--ff) !important;}
.gradio-container * {font-family:var(--ff);}
footer {display:none !important;}
.gradio-container .block, .gradio-container .form {border:0 !important; box-shadow:none !important; background:transparent !important; padding:0 !important;}
#shell {gap:0 !important; min-height:100vh; flex-wrap:nowrap !important; align-items:stretch !important;}

/* sidebar */
#side {background:var(--side) !important; padding:22px 16px 18px !important; gap:6px !important; flex:0 0 272px !important; max-width:272px;}
.brand {display:flex; align-items:center; gap:12px; padding:2px 6px 18px;}
.brand .logo {width:38px; height:38px; border-radius:10px; background:var(--accent); color:#fff; display:flex; align-items:center;
  justify-content:center; font-weight:700; font-size:1.05rem; letter-spacing:-.02em;}
.brand .name {color:#fff; font-weight:700; font-size:1.08rem; letter-spacing:-.01em; line-height:1.2;}
.brand .sub {color:#94a3b8; font-size:.78rem; margin-top:1px;}
.sec {color:#94a3b8; font-size:.74rem; font-weight:600; padding:16px 8px 4px; margin:0;}
#side button {width:100% !important; justify-content:flex-start !important; text-align:left !important; background:transparent !important;
  color:var(--side-text) !important; border:0 !important; box-shadow:none !important; border-radius:8px !important;
  font-size:.9rem !important; font-weight:500 !important; padding:9px 10px !important; min-width:0 !important; white-space:normal !important; line-height:1.3 !important;}
#side button:hover {background:var(--side-2) !important; color:#fff !important;}
#side button.primary {background:var(--side-2) !important; color:#fff !important; box-shadow:inset 3px 0 0 var(--accent) !important;}
#side #newchat {background:var(--accent) !important; color:#fff !important; justify-content:center !important; text-align:center !important;
  font-weight:600 !important; padding:10px !important; margin-bottom:4px;}
#side #newchat:hover {filter:brightness(1.08);}
.sidefoot {color:#94a3b8; font-size:.74rem; line-height:1.5; padding:18px 8px 0; border-top:1px solid var(--side-2); margin-top:16px;}
.sidefoot a {color:#cbd5e1;}
#side button:focus-visible, #chips button:focus-visible {outline:2px solid #93c5fd !important; outline-offset:2px;}

/* main */
#main {padding:0 !important; gap:0 !important; min-width:0 !important;}
.topbar {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; padding:16px 32px;
  background:var(--panel); border-bottom:1px solid var(--line);}
.topbar .t {font-weight:600; font-size:1rem; color:var(--ink);}
.topbar .t small {display:block; font-weight:400; font-size:.82rem; color:var(--muted); margin-top:2px;}
.status {display:flex; align-items:center; gap:8px; font-size:.82rem; color:var(--muted); background:var(--bg);
  border:1px solid var(--line); border-radius:999px; padding:6px 12px;}
.status i {width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block;}
.status b {color:var(--ink); font-weight:600;}

.thread {height:calc(100vh - 250px); min-height:380px; overflow-y:auto; display:flex; flex-direction:column-reverse; padding:0 24px;}
.thread .inner {width:100%; max-width:820px; margin:0 auto; display:flex; flex-direction:column; gap:22px; padding:28px 0 16px;}
.welcome {text-align:center; padding:8vh 0 2vh;}
.welcome .logo {width:56px; height:56px; border-radius:16px; background:var(--accent); color:#fff; display:inline-flex; align-items:center;
  justify-content:center; font-weight:700; font-size:1.5rem; box-shadow:0 10px 24px -10px rgba(47,91,234,.7);}
.welcome h2 {font-size:1.7rem; font-weight:700; letter-spacing:-.02em; margin:18px 0 8px; color:var(--ink);}
.welcome p {color:var(--muted); font-size:1rem; line-height:1.55; max-width:48ch; margin:0 auto;}
.caps {display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:28px auto 0; max-width:720px; text-align:left;}
.cap {background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px;}
.cap b {display:block; font-size:.92rem; font-weight:600; color:var(--ink); margin-bottom:3px;}
.cap span {font-size:.84rem; color:var(--muted); line-height:1.45;}

.row-u {display:flex; justify-content:flex-end;}
.row-u .bubble {background:var(--accent); color:#fff; border-radius:18px 18px 4px 18px; padding:10px 16px; font-size:.98rem; line-height:1.45; max-width:78%;}
.row-a {display:flex; gap:12px; align-items:flex-start;}
.avatar {flex:0 0 32px; height:32px; border-radius:9px; background:var(--side); color:#fff; display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:.82rem; margin-top:2px;}
.ans {min-width:0; flex:1;}
.ans .card {background:var(--panel); border:1px solid var(--line); border-radius:4px 16px 16px 16px; padding:14px 18px; box-shadow:0 1px 2px rgba(15,23,42,.04);
  font-size:.97rem; line-height:1.6; color:var(--ink);}
.ans .card p {margin:0 0 8px;} .ans .card p:last-child, .ans .card ul:last-child, .ans .card ol:last-child {margin-bottom:0;}
.ans .card ul, .ans .card ol {margin:0 0 8px; padding-left:1.2em;} .ans .card li {margin-bottom:5px;}
.ans .card a {color:var(--accent); font-weight:500; text-decoration:none;} .ans .card a:hover {text-decoration:underline;}
.ans .card .asof {font-size:.78rem; color:var(--muted); margin-top:10px; padding-top:8px; border-top:1px solid var(--line);}
.ans .card .asof em {font-style:normal;}
.works {display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:10px; margin-top:10px;}
.work {display:flex; gap:12px; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px; text-decoration:none !important;
  transition:border-color .15s, box-shadow .15s;}
.work:hover {border-color:var(--accent); box-shadow:0 6px 16px -8px rgba(15,23,42,.25);}
.work .thumb {flex:0 0 64px; height:64px; border-radius:10px; background:var(--accent-soft) center/cover no-repeat; color:var(--accent);
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.3rem;}
.work .info {min-width:0;}
.work .ti {font-weight:600; font-size:.9rem; color:var(--ink); line-height:1.3; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;}
.work .ar {font-size:.8rem; color:var(--muted); margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.tag {display:inline-flex; align-items:center; gap:5px; font-size:.72rem; font-weight:600; border-radius:999px; padding:2px 9px; margin-top:6px;}
.tag.on {background:var(--good-bg); color:var(--good);} .tag.off {background:#f1f5f9; color:var(--muted);}
.work .why {font-size:.74rem; color:var(--muted); margin-top:4px;}
.typing {display:inline-flex; gap:5px; padding:6px 2px;}
.typing i {width:7px; height:7px; border-radius:50%; background:#94a3b8;}
@media (prefers-reduced-motion:no-preference) {
  .typing i {animation:bounce 1.2s infinite ease-in-out;} .typing i:nth-child(2){animation-delay:.15s} .typing i:nth-child(3){animation-delay:.3s}
  @keyframes bounce {0%,80%,100%{transform:translateY(0);opacity:.5} 40%{transform:translateY(-5px);opacity:1}}
}

/* composer */
#composer {max-width:820px; width:100%; margin:0 auto !important; padding:0 24px 14px !important; gap:10px !important;}
#chips {gap:8px !important; flex-wrap:wrap !important;}
#chips button {flex:0 1 auto !important; min-width:0 !important; background:var(--panel) !important; color:var(--ink) !important;
  border:1px solid var(--line) !important; border-radius:999px !important; font-size:.84rem !important; font-weight:500 !important;
  padding:7px 14px !important; box-shadow:none !important; white-space:normal !important;}
#chips button:hover {border-color:var(--accent) !important; color:var(--accent) !important; background:var(--accent-soft) !important;}
#ask {background:var(--panel) !important; border:1px solid var(--line) !important; border-radius:16px !important; padding:6px 8px 6px 16px !important;
  box-shadow:0 8px 24px -14px rgba(15,23,42,.3) !important;}
#ask:focus-within {border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(47,91,234,.15) !important;}
#ask textarea, #ask input {background:transparent !important; color:var(--ink) !important; border:0 !important; box-shadow:none !important; font-size:1rem !important; padding:8px 0 !important;}
#ask textarea::placeholder {color:#94a3b8 !important;}
#ask button {background:var(--accent) !important; color:#fff !important; border:0 !important; border-radius:10px !important; font-weight:600 !important; padding:8px 18px !important;}
#ask button:hover {filter:brightness(1.08);}
.fine {text-align:center; font-size:.74rem; color:#94a3b8; margin:0;}
#opts {max-width:820px; margin:0 auto 14px !important; padding:0 24px !important;}
#opts, #opts * {font-size:.8rem !important; color:var(--muted) !important;}
#opts input[type=checkbox] {accent-color:var(--accent);}

@media (max-width:820px) {
  #shell {flex-wrap:wrap !important;}
  #side {flex:1 1 100% !important; max-width:100%;}
  .caps {grid-template-columns:1fr;} .topbar {padding:14px 18px;} .thread {height:60vh;}
}

/* ---- visual refresh */
.featured {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin-top:18px;}
.art {display:block; background:#ffffff; border-radius:18px; overflow:hidden; text-decoration:none !important;
  border:1px solid rgba(255,255,255,.85); box-shadow:0 10px 30px -18px rgba(15,23,42,.35);}
.art img {display:block; width:100%; height:200px; object-fit:cover; background:#e5e7eb;}
.art span {display:block; padding:10px 12px 12px; color:#4b5563; font-size:.84rem; line-height:1.35;}
.art b {display:block; color:#0b0b0c; font-size:.92rem; font-weight:600;}
@media (max-width: 760px) {.featured {grid-template-columns:1fr;}}
/* ---- visual refresh: light stage, serif display, pill controls ---- */
:root, .dark, body {
  --bg:#f4f4f5; --side:#ffffff; --side-2:#f1f3f6; --side-text:#1f2937;
  --ink:#0b0b0c; --muted:#5b6470; --line:#e5e7eb; --accent:#0b0b0c; --accent-soft:#f1f3f6;
  --display:'Instrument Serif', Georgia, 'Times New Roman', serif;
}
#side {border-right:1px solid var(--line) !important;}
#side .brand .logo {background:#0b0b0c;}
#side .sec {color:#6b7280;}
#side #newchat {background:#0b0b0c !important; color:#fff !important; border-radius:999px !important;}
#side button {color:#1f2937 !important;}
#side button:hover {background:#f1f3f6 !important; color:#0b0b0c !important;}
#side button:focus-visible {outline:2px solid #0b0b0c !important;}
.sidefoot {color:#6b7280; border-top:1px solid var(--line);}
.sidefoot a {color:#1f2937;}
.topbar {background:transparent; border-bottom:0;}
.status {background:#ffffff;}
.welcome {
  background:linear-gradient(180deg, rgba(228,228,231,.9) 0%, rgba(244,244,245,.92) 60%, rgba(244,244,245,.97) 100%),;
  border-radius:28px; padding:48px 44px 36px; margin:8px 0 0; text-align:left;
}
.welcome .kicker {display:inline-flex; align-items:center; gap:10px; font-size:.85rem; color:#374151;
  background:#ffffff; border-radius:999px; padding:6px 14px 6px 6px;}
.welcome .kicker .dot {width:26px; height:26px; border-radius:50%; background:#0b0b0c; color:#fff;
  display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:.8rem;}
.welcome h1 {font-family:var(--display); font-weight:400; font-size:clamp(2.6rem, 6vw, 4.4rem);
  line-height:.98; letter-spacing:-.02em; color:#0b0b0c; margin:22px 0 14px; max-width:14ch;}
.welcome h1 em {font-style:italic;}
.welcome .lede {font-size:1.05rem; line-height:1.55; color:#3f3f46; max-width:46ch; margin:0;}
.welcome .caps {margin:28px 0 0; max-width:none; grid-template-columns:repeat(3, minmax(0,1fr));}
.welcome .cap {background:rgba(255,255,255,.6); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,.85); border-radius:18px; padding:18px 18px 16px;}
.welcome .cap .tick {display:inline-block; width:22px; height:22px; border-radius:7px; background:#0b0b0c;
  position:relative; margin-bottom:12px;}
.welcome .cap .tick::after {content:""; position:absolute; left:8px; top:4px; width:5px; height:10px;
  border:solid #fff; border-width:0 2px 2px 0; transform:rotate(45deg);}
.welcome .cap b {font-size:1rem; color:#0b0b0c;}
.welcome .cap span {color:#4b5563;}
@media (max-width: 760px) {
  .welcome {padding:28px 22px;}
  .welcome .caps {grid-template-columns:1fr;}
}
.bubble {background:#0b0b0c !important; color:#ffffff !important;}
.card {border-radius:20px !important;}
#chips button {background:#ffffff !important; border:1px solid #e5e7eb !important; border-radius:999px !important;
  color:#111827 !important;}
#chips button::before {content:"\\2713"; display:inline-block; margin-right:8px; width:18px; height:18px;
  line-height:18px; text-align:center; border-radius:6px; background:#f97316; color:#ffffff; font-size:12px;}
#chips button:hover {border-color:#0b0b0c !important; background:#ffffff !important; color:#0b0b0c !important;}
#ask {border-radius:999px !important; padding-left:18px !important;}
#ask button {background:#0b0b0c !important; color:#ffffff !important; border-radius:999px !important;}
button.primary {background:#0b0b0c !important; border-color:#0b0b0c !important;}
"""


# ---------------------------------------------------------------- HTML helpers
def _e(v) -> str:
    return html.escape(str(v or ""), quote=True)


def md(text: str) -> str:
    """Small markdown-to-HTML converter for the answer templates."""
    links: list[tuple[str, str]] = []

    def keep(m):
        links.append((m.group(1), m.group(2)))
        return f"\x00{len(links) - 1}\x00"

    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", keep, text)
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<em>\1</em>", t)
    t = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<em>\1</em>", t)
    t = re.sub(
        r"\x00(\d+)\x00",
        lambda m: (
            f"<a href='{_e(links[int(m.group(1))][1])}' target='_blank' rel='noopener'>{html.escape(links[int(m.group(1))][0])}</a>"
        ),
        t,
    )

    out, para, items, kind = [], [], [], None

    def flush():
        nonlocal para, items, kind
        if para:
            body = " ".join(para)
            cls = " class='asof'" if body.startswith("<em>Collection data") else ""
            out.append(f"<p{cls}>{body}</p>")
            para = []
        if items:
            out.append(f"<{kind}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{kind}>")
            items, kind = [], None

    for line in t.split("\n"):
        s = line.strip()
        bullet = re.match(r"^[-•]\s+(.*)", s)
        number = re.match(r"^\d+\.\s+(.*)", s)
        if not s:
            flush()
        elif bullet or number:
            want = "ul" if bullet else "ol"
            if para or (kind and kind != want):
                flush()
            kind = want
            items.append((bullet or number).group(1))
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "".join(out)


FEATURED_TITLES = [
    "The Starry Night",
    "Water Lilies",
    "Les Demoiselles d'Avignon",
    "The Persistence of Memory",
]
_featured: list[dict] = []  # filled by build_ui from the works on view


def set_featured(works) -> None:
    """Pick the well-known works that are on view and have an image link."""
    _featured.clear()
    for title in FEATURED_TITLES:
        rows = works[
            (works["Title"] == title) & (works["Classification"] == "Painting") & works["ImageURL"].notna()
        ]
        if len(rows):
            row = rows.iloc[0]
            _featured.append(
                {"title": title, "artist": row["Artist"], "image": row["ImageURL"], "link": row["link"]}
            )
        if len(_featured) == 3:
            break


def _bg_style() -> str:
    """Background for the welcome card: a MoMA painting, faded behind the text."""
    if not _featured:
        return ""
    img = _e(_featured[0]["image"])
    return (
        "background: linear-gradient(180deg, rgba(228,228,231,.9) 0%, rgba(244,244,245,.92) 60%, "
        f"rgba(244,244,245,.97) 100%), url('{img}') center 30% / cover no-repeat;"
    )


def featured_html() -> str:
    if not _featured:
        return ""
    items = "".join(
        f"<a class='art' href='{_e(f['link'])}' target='_blank' rel='noopener'>"
        f"<img src='{_e(f['image'])}' alt='{_e(f['title'])}' loading='lazy'>"
        f"<span><b>{_e(f['title'])}</b>{_e(f['artist'])}</span></a>"
        for f in _featured
    )
    return f"<div class='featured'>{items}</div>"


def welcome_html() -> str:
    caps = [
        ("Find a work", "Get the floor and gallery for any artwork on view."),
        ("Learn about it", "Who made it, when, the medium, and how MoMA acquired it."),
        ("Decide what's next", "Recommendations nearby, by the same artist, or from the same period."),
    ]
    grid = "".join(
        f"<div class='cap'><span class='tick'></span><b>{t}</b><span>{d}</span></div>" for t, d in caps
    )
    return (
        f'<div class="welcome" style="{_bg_style()}">'
        "<div class='kicker'><span class='dot'>A</span><span>Museum guide for works on view at MoMA</span></div>"
        "<h1>Ask about the art <em>at MoMA</em></h1>"
        "<p class='lede'>Where a work hangs, who made it, and what to see next. "
        "Every answer links to moma.org.</p>"
        f"<div class='caps'>{grid}</div>{featured_html()}</div>"
    )


def works_html(cards: list[dict], limit: int = 6) -> str:
    if not cards:
        return ""
    out = []
    for c in cards[:limit]:
        img = c.get("image") or ""
        if img.startswith("http"):
            thumb = f"<div class='thumb' style=\"background-image:url('{_e(img)}')\"></div>"
        else:
            thumb = f"<div class='thumb'>{_e((c['title'] or '?')[:1].upper())}</div>"
        tag = (
            f"<span class='tag on'>{_e(c['location'])}</span>"
            if c["on_view"]
            else "<span class='tag off'>Not on view</span>"
        )
        why = f"<div class='why'>{_e(c['reason'])}</div>" if c.get("reason") else ""
        out.append(
            f"<a class='work' href='{_e(c['link'])}' target='_blank' rel='noopener'>{thumb}<div class='info'>"
            f"<div class='ti'>{_e(c['title'])}</div><div class='ar'>{_e(c['artist'])}, {_e(c['date'])}</div>{tag}{why}</div></a>"
        )
    return "<div class='works'>" + "".join(out) + "</div>"


def thread_html(history: list) -> str:
    if not history:
        return f"<div class='thread'><div class='inner'>{welcome_html()}</div></div>"
    rows = []
    for q, a, cards in history:
        rows.append(f"<div class='row-u'><div class='bubble'>{_e(q)}</div></div>")
        body = (
            f"<div class='card'>{a}</div>{works_html(cards)}"
            if a
            else "<div class='card'><span class='typing' aria-label='Working'><i></i><i></i><i></i></span></div>"
        )
        rows.append(f"<div class='row-a'><div class='avatar'>A</div><div class='ans'>{body}</div></div>")
    return "<div class='thread'><div class='inner'>" + "".join(rows) + "</div></div>"


def topbar_html(col) -> str:
    return (
        "<div class='topbar'><div class='t'>Museum guide<small>Answers come from MoMA's public collection data</small></div>"
        f"<div class='status'><i></i><span><b>{len(col.on_view):,}</b> works on view</span>"
        f"<span>Updated {_e(col.as_of)}</span></div></div>"
    )


# ---------------------------------------------------------------- UI
def build_ui(bot):
    import gradio as gr

    set_featured(bot.col.works)

    from askmoma import config, llm

    col = bot.col
    floors = sorted(col.floors(), reverse=True)
    counts = col.on_view["floor"].dropna().astype(int).value_counts()

    theme = gr.themes.Base(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate).set(
        body_background_fill="#f6f7f9",
        body_background_fill_dark="#f6f7f9",
        body_text_color="#0f172a",
        body_text_color_dark="#0f172a",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#ffffff",
        background_fill_secondary_dark="#ffffff",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
    )

    def chip_updates(suggestions):
        s = (suggestions or [])[:N_CHIPS]
        return [gr.update(value=x, visible=True) for x in s] + [gr.update(visible=False)] * (N_CHIPS - len(s))

    def add_user(message, history):
        message = (message or "").strip()
        history = list(history or [])
        if message:
            history.append([message, None, []])
        return "", history, thread_html(history)

    def answer(history, ctx, on_view_only):
        history = list(history or [])
        if not history or history[-1][1]:
            return [thread_html(history), history, ctx] + [gr.update()] * (N_CHIPS + len(floors))
        question = history[-1][0]
        ans = bot.ask(question, ctx, on_view_only=on_view_only)
        text = llm.polish(question, ans.text) if config.LLM_MODEL else ans.text
        history[-1][1], history[-1][2] = md(text), ans.cards
        hit = set()
        for c in ans.cards:
            f = col.works.at[c["row"], "floor"]
            if c["on_view"] and f is not None and f == f:
                hit.add(int(f))
        return (
            [thread_html(history), history, ans.context]
            + chip_updates(ans.suggestions)
            + [gr.update(variant="primary" if f in hit else "secondary") for f in floors]
        )

    def reset():
        return (
            [thread_html([]), [], {}] + chip_updates(None) + [gr.update(variant="secondary") for _ in floors]
        )

    # Gradio 6 moved theme and css from Blocks() to launch().
    gradio6 = int(gr.__version__.split(".")[0]) >= 6
    style = {"theme": theme, "css": CSS}
    with gr.Blocks(title="AskMoMA", fill_width=True, **({} if gradio6 else style)) as demo:
        hist = gr.State([])
        ctx = gr.State({})
        with gr.Row(elem_id="shell", equal_height=True):
            with gr.Column(scale=0, min_width=272, elem_id="side"):
                gr.HTML(
                    "<div class='brand'><div class='logo'>A</div><div><div class='name'>AskMoMA</div>"
                    "<div class='sub'>Museum guide</div></div></div>"
                )
                new_chat = gr.Button("New conversation", elem_id="newchat")
                gr.HTML("<p class='sec'>Browse by floor</p>")
                floor_btns = [
                    gr.Button(
                        f"Floor {f}  ({int(counts.get(f, 0)):,} work{'' if counts.get(f, 0) == 1 else 's'})",
                        variant="secondary",
                    )
                    for f in floors
                ]
                gr.HTML("<p class='sec'>Popular questions</p>")
                starter_btns = [
                    gr.Button(q) for q in STARTERS + ["What are the museum hours?", "How much is admission?"]
                ]
                gr.HTML(
                    f"<div class='sidefoot'>Collection data from MoMA's public "
                    f"<a href='https://github.com/MuseumofModernArt/collection' target='_blank' rel='noopener'>GitHub dataset</a> "
                    f"(CC0), {_e(col.as_of)}.<br>Student project for MSAI-631. Not affiliated with MoMA.</div>"
                )
            with gr.Column(scale=1, elem_id="main"):
                gr.HTML(topbar_html(col))
                thread = gr.HTML(thread_html([]))
                with gr.Column(elem_id="composer"):
                    with gr.Row(elem_id="chips"):
                        chips = [gr.Button(visible=False, size="sm") for _ in range(N_CHIPS)]
                    msg = gr.Textbox(
                        elem_id="ask",
                        placeholder="Ask about an artwork, an artist, or your visit",
                        show_label=False,
                        container=False,
                        autofocus=True,
                        submit_btn="Send",
                        lines=1,
                        max_lines=4,
                    )
                    gr.HTML(
                        "<p class='fine'>Gallery locations can change. Check the moma.org link before you go.</p>"
                    )
                with gr.Row(elem_id="opts"):
                    on_view_only = gr.Checkbox(
                        value=True, label="Only list works that are on view", container=False
                    )

        outs = [thread, hist, ctx, *chips, *floor_btns]
        run = dict(fn=answer, inputs=[hist, ctx, on_view_only], outputs=outs)
        msg.submit(add_user, [msg, hist], [msg, hist, thread], queue=False).then(**run)
        for b in chips + starter_btns:
            b.click(add_user, [b, hist], [msg, hist, thread], queue=False).then(**run)
        for f, b in zip(floors, floor_btns, strict=True):
            b.click(
                functools.partial(add_user, f"What's on floor {f}?"), [hist], [msg, hist, thread], queue=False
            ).then(**run)
        new_chat.click(reset, None, outs)
    demo.askmoma_launch_kwargs = style if gradio6 else {}
    return demo


def cli(bot):
    print(bot.ask("hi").text, "\n(type 'quit' to exit)\n")
    ctx = None
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        a = bot.ask(q, ctx)
        ctx = a.context
        print(f"\naskmoma [{a.intent}]>\n{a.text}\n")
        if a.suggestions:
            print("  try: " + " | ".join(a.suggestions) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder with Artworks.csv and Artists.csv")
    ap.add_argument("--cli", action="store_true", help="chat in the terminal")
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = ap.parse_args()

    from askmoma import AskMoMA, load
    from askmoma.config import DATA_DIR

    data_dir = Path(args.data) if args.data else DATA_DIR
    if not (data_dir / "Artworks.csv").exists():
        print(
            f"Artworks.csv is not in {data_dir}.\n"
            "Download MoMA's data first with: mise run data\n"
            "or try the app on the small test data: python app.py --data tests/fixtures",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = AskMoMA(load(args.data))
    if args.cli:
        cli(bot)
    else:
        demo = build_ui(bot)
        demo.launch(server_port=args.port, share=args.share, **demo.askmoma_launch_kwargs)


if __name__ == "__main__":
    main()
