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

EXPLORE = [
    "Where is The Starry Night?",
    "What works by Frida Kahlo are on view?",
    "Which artists have the most works on floor 5?",
    "What are the museum hours?",
]
STARTERS = [
    "Where is The Starry Night?",
    "What works by Frida Kahlo are on view?",
    "Which artists have the most works on floor 5?",
    "Find dreamy night sky paintings",
]
N_CHIPS = 4

CSS = """/* ---- paintings in the empty margins (wide screens only) ---- */
#body {position:relative;}
#gutters {position:absolute !important; top:0; left:0; right:0; bottom:0; height:auto !important; padding:0 !important; margin:0 !important; background:transparent !important; border:0 !important; pointer-events:none; display:none !important;}
.gut {position:absolute; top:40px; bottom:80px; width:200px; pointer-events:auto;}
.gut-left {left:calc(-200px - 16px);}
.gut-right {right:calc(-200px - 16px);}
.gut-slide {position:absolute; inset:0; border-radius:18px; background-size:cover; background-position:center; opacity:0;
  box-shadow:0 14px 30px -18px rgba(0,0,0,.5); animation-name:gut-fade; animation-iteration-count:infinite;
  animation-timing-function:ease-in-out; animation-fill-mode:both;}
@keyframes gut-fade {0% {opacity:0} 4% {opacity:1} 21% {opacity:1} 25% {opacity:0} 100% {opacity:0}}
@media (min-width: 1620px) {#gutters {display:block !important;}}

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
/* ---- polish: sidebar, header, hero ---- */
#side .sec {font-size:.72rem !important; letter-spacing:.08em; text-transform:uppercase; color:#6b7280 !important; font-weight:600; padding:0 12px; margin-top:18px;}
#side button:not(#newchat) {border-radius:12px !important; padding:9px 12px !important; font-size:.92rem !important; font-weight:500 !important;}
#side button:not(#newchat):hover {box-shadow:inset 3px 0 0 #0b0b0c !important;}
#side #newchat {box-shadow:0 8px 18px -10px rgba(11,11,12,.6);}
.topbar {padding:14px 28px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.7); backdrop-filter:blur(8px);}
.topbar .t {font-family:var(--display); font-size:1.35rem; font-weight:400; letter-spacing:-.01em; color:#0b0b0c;}
.topbar .t small {font-family:inherit; font-size:.85rem;}
.status {border:1px solid var(--line); border-radius:999px; padding:6px 14px; box-shadow:0 1px 2px rgba(15,23,42,.06);}
.welcome {box-shadow:0 30px 60px -40px rgba(15,23,42,.35);}
.welcome .cap {box-shadow:0 12px 28px -20px rgba(15,23,42,.45);}
.featured .art {transition:transform .18s ease, box-shadow .18s ease;}
.featured .art:hover {transform:translateY(-3px); box-shadow:0 18px 36px -20px rgba(15,23,42,.5);}
.featured .art img {height:220px;}

/* ---- dark gallery theme (header, hero, cards) ---- */
.topbar {background:#0b0b0c !important; border-bottom:1px solid #1f1f23 !important;}
.topbar .t {color:#f5f5f4 !important;}
.topbar .t small {color:#a1a1aa !important;}
.status {background:#18181b !important; border-color:#27272a !important; color:#d4d4d8 !important;}
.status b {color:#fafafa !important;}
.welcome {border:1px solid #27272a;}
.welcome h1 {font-family:"Inter", system-ui, sans-serif !important; font-weight:800 !important; text-transform:uppercase;
  letter-spacing:-.015em; line-height:.98; color:#fafafa !important; font-size:clamp(2.4rem, 5.4vw, 4.2rem) !important;}
.welcome h1 em {font-family:var(--display) !important; text-transform:none; font-weight:400; color:#fafafa !important;}
.welcome .kicker {font-family:ui-monospace, SFMono-Regular, Menlo, monospace !important; text-transform:uppercase;
  letter-spacing:.14em; font-size:.74rem !important; color:#e4e4e7 !important;}
.welcome .kicker span:last-child {background:transparent;}
.welcome .kicker .dot {background:#fafafa !important; color:#0b0b0c !important;}
.welcome .lede {color:#d4d4d8 !important;}
.welcome .cap {background:rgba(12,12,14,.55) !important; border-color:rgba(255,255,255,.14) !important; box-shadow:none !important;}
.welcome .cap b {color:#fafafa !important;}
.welcome .cap span {color:#a1a1aa !important;}
.welcome .cap .tick {background:#fafafa !important;}
.featured .art {background:#18181b !important; border-color:rgba(255,255,255,.1) !important;}
.featured .art b {color:#fafafa !important;}
.featured .art span {color:#a1a1aa !important;}

/* ---- roomier layout ---- */
.topbar {padding:24px 44px !important;}
.welcome {padding:88px 72px 72px !important; min-height:620px; display:flex; flex-direction:column; justify-content:center;}
.welcome .kicker {margin-bottom:28px;}
.welcome h1 {margin:0 0 26px !important;}
.welcome .lede {margin-bottom:0; max-width:52ch;}
.welcome .caps {gap:20px !important; margin-top:48px !important;}
.welcome .cap {padding:24px 24px 26px !important;}
.welcome .cap b {margin-bottom:8px;}
.featured {gap:24px !important; margin-top:40px !important;}
.featured .art img {height:300px !important;}
.featured .art span {padding:16px 18px 18px !important;}
#side button:not(#newchat) {padding:12px 16px !important; margin-bottom:2px;}
#side .sec {margin-top:28px !important; margin-bottom:6px;}
@media (max-width: 900px) {
  .welcome {padding:56px 28px 48px !important; min-height:0;}
  .featured {grid-template-columns:1fr !important;}
  .featured .art img {height:240px !important;}
}

/* ---- MoMA-style header, hero, and floating chat popup ---- */
.mh {display:flex; align-items:center; justify-content:space-between; padding:18px 40px; background:#ffffff; border-bottom:1px solid #e4e4e7;}
.mh-logo {font-weight:900; font-size:1.8rem; letter-spacing:-.02em; color:#0b0b0c;}
.mh-nav a {margin-left:28px; color:#0b0b0c; font-weight:700; font-size:1rem; text-decoration:none;}
.mh-nav a:hover {text-decoration:underline;}
.hero-wide {min-height:340px; padding:56px 48px; display:flex; flex-direction:column; justify-content:flex-end; color:#ffffff;}
.hero-wide h1 {font-weight:900; font-size:clamp(2.4rem, 6vw, 4.8rem); line-height:.95; letter-spacing:-.02em; color:#ffffff; margin:0;}
.hero-wide p {color:#e4e4e7; font-size:1.1rem; margin:12px 0 0; max-width:60ch;}
#chatpanel {position:fixed !important; right:24px; bottom:100px; width:min(420px, calc(100vw - 48px)) !important; height:min(620px, 72vh) !important; z-index:1000; background:#ffffff; border-radius:20px; box-shadow:0 24px 60px -20px rgba(0,0,0,.45); border:1px solid #e4e4e7; overflow:auto; padding:14px !important;}
#chat-launch {position:fixed !important; right:24px; bottom:24px; width:60px !important; min-width:60px !important; height:60px !important; border-radius:999px !important; z-index:1001; background:#0b0b0c !important; color:#ffffff !important; font-weight:700; box-shadow:0 12px 30px -10px rgba(0,0,0,.6); padding:0 !important;}

/* ---- body, floor cards, popup tabs and home ---- */
#body {max-width:1180px; margin:0 auto; padding:40px 40px 80px;}
.stats {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px;}
.stat {background:#ffffff; border:1px solid #e4e4e7; border-radius:18px; padding:22px 24px;}
.stat b {display:block; font-size:2.2rem; font-weight:800; color:#0b0b0c; letter-spacing:-.02em;}
.stat span {color:#52525b; font-size:.95rem;}
.about {margin-top:36px;}
.about h2 {font-weight:800; font-size:1.9rem; letter-spacing:-.01em; color:#0b0b0c; margin:0 0 12px;}
.about h3 {font-weight:700; font-size:1.2rem; margin:24px 0 8px; color:#0b0b0c;}
.about p, .about li {color:#3f3f46; font-size:1.05rem; line-height:1.6; max-width:70ch;}
.about a {color:#0b0b0c; font-weight:600;}
#floors {gap:14px !important; flex-wrap:wrap;}
.floor-card {flex:1 1 170px !important; min-height:104px !important; border-radius:18px !important; background:#ffffff !important; border:1px solid #e4e4e7 !important; color:#0b0b0c !important; font-weight:700 !important; white-space:pre-line !important; text-align:left !important; justify-content:flex-start !important; padding:18px !important;}
.floor-card:hover {border-color:#0b0b0c !important; box-shadow:0 14px 28px -20px rgba(0,0,0,.5) !important;}
.home-card {background:#0b0b0c; color:#ffffff; border-radius:18px; padding:22px 22px 20px; margin-bottom:12px;}
.home-card h2 {color:#ffffff; font-weight:800; font-size:1.5rem; margin:4px 0 6px;}
.home-card p {color:#d4d4d8; margin:0;}
.home-card .kicker {font-family:ui-monospace, Menlo, monospace; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; color:#a1a1aa; margin:0;}
.home-link {display:flex; justify-content:space-between; align-items:center; padding:14px 16px; border:1px solid #e4e4e7; border-radius:14px; color:#0b0b0c; font-weight:700; text-decoration:none; margin-bottom:12px;}
.home-link:hover {border-color:#0b0b0c;}
#send-btn {width:100%; margin-bottom:18px;}
.starter {justify-content:flex-start !important; text-align:left !important;}
#chattabs {min-height:0;}

/* ---- credits and footer ---- */
.credit {display:block; font-size:.78rem; color:#71717a; font-style:normal; margin-top:6px; text-decoration:none;}
.art em.credit a, a.credit {color:#52525b;}
.hero-credit {color:#e4e4e7 !important; margin-top:14px; font-size:.85rem; text-decoration:underline;}
.featured {margin-top:14px;}

/* ---- chat popup header and close button ---- */
#chat-head {display:flex !important; justify-content:space-between; align-items:center; margin:0 0 4px !important;}
.chat-title {font-weight:800; font-size:1.05rem; color:#0b0b0c; letter-spacing:-.01em;}
#chat-close {min-width:36px !important; max-width:36px !important; height:36px !important; padding:0 !important; border-radius:999px !important; background:#f4f4f5 !important; color:#0b0b0c !important; font-weight:700 !important; border:1px solid #e4e4e7 !important;}
#chat-close:hover {background:#e4e4e7 !important;}

/* ---- chat popup header: clear title and a solid close button ---- */
#chat-head {display:flex !important; justify-content:space-between; align-items:center; gap:12px; margin:-14px -14px 10px !important; padding:14px 16px !important; border-bottom:1px solid #e4e4e7; background:#ffffff; border-radius:20px 20px 0 0;}
.chat-head-text {display:flex; flex-direction:column;}
.chat-title {font-weight:800; font-size:1.1rem; color:#0b0b0c; letter-spacing:-.01em;}
.chat-sub {font-size:.8rem; color:#71717a;}
#chat-close {min-width:36px !important; max-width:36px !important; height:36px !important; padding:0 !important; border-radius:999px !important; background:#0b0b0c !important; color:#ffffff !important; font-size:1rem !important; font-weight:700 !important; border:none !important; box-shadow:0 6px 14px -8px rgba(0,0,0,.6);}
#chat-close:hover {background:#27272a !important;}
.pop-banner {margin:0 -14px 14px !important; padding:120px 20px 18px; min-height:170px; color:#ffffff; display:flex; flex-direction:column; justify-content:flex-end; gap:4px; border-bottom:1px solid #e4e4e7;}
.pop-title {font-family:"Instrument Serif", Georgia, serif; font-size:2rem; line-height:1.05; letter-spacing:-.01em;}
.pop-sub {font-size:.88rem; opacity:.9;}
.pop-credit, .pop-credit * {color:rgba(255,255,255,.85) !important; font-size:.7rem !important; text-decoration:underline;}
#chatnav {position:sticky !important; bottom:-14px; margin:14px -14px -14px !important; padding:8px 10px !important; gap:8px !important; background:#ffffff; border-top:1px solid #e4e4e7; z-index:2;}
#chatnav button {flex:1 1 0 !important; border-radius:12px !important; font-weight:600 !important; min-height:42px;}

/* ---- chat thread: compact, WhatsApp-style, newest message at the bottom ---- */
#chatpanel .thread {height:300px !important; min-height:0 !important; flex-direction:column !important; padding:0 4px !important; overflow-y:auto;}
#chatpanel .thread .inner {gap:6px !important; padding:8px 0 !important; max-width:none !important;}
#chatpanel .row-u .bubble {font-size:.88rem !important; padding:7px 11px !important; max-width:82% !important; border-radius:14px 14px 3px 14px !important;}
#chatpanel .row-a {gap:8px !important;}
#chatpanel .avatar {flex:0 0 24px !important; height:24px !important; font-size:.75rem; border-radius:50% !important;}
#chatpanel .ans .card {font-size:.88rem !important; line-height:1.4 !important; padding:7px 11px !important; border-radius:3px 14px 14px 14px !important; max-width:92%;}
#chatpanel .ans .card p {margin:0 0 4px !important;}
#chatpanel .ans .card .asof {font-size:.7rem !important; margin-top:6px !important; padding-top:5px !important;}
#chatpanel .ans .card li {margin-bottom:2px !important;}
#chatpanel .inner > .row-u + .row-a, #chatpanel .inner > .row-a + .row-u {margin-top:2px;}
#chatpanel #chips {margin-top:2px !important;}
#chatpanel #chips button {font-size:.78rem !important; padding:4px 10px !important;}
#chatpanel #ask textarea {font-size:.9rem !important;}
/* ---- chat popup layout: fixed header and composer, scrolling messages ---- */
#chatpanel {display:flex !important; flex-direction:column !important; overflow:hidden !important; padding:0 !important;
  height:min(640px, 74vh) !important; animation:pop-in .18s ease-out;}
@keyframes pop-in {from {opacity:0; transform:translateY(12px) scale(.98);} to {opacity:1; transform:none;}}
#chatpanel > *:not(#pane-home):not(#pane-messages):not(#chatnav) {flex:0 0 auto !important;}
#chatpanel #chat-head {margin:0 !important; border-radius:20px 20px 0 0 !important; padding:12px 16px !important;}
#pane-home {flex:1 1 auto !important; min-height:0 !important; overflow-y:auto !important; padding:0 0 10px !important;}
#pane-messages {flex:1 1 auto !important; min-height:0 !important; overflow:hidden !important; display:flex !important;
  flex-direction:column !important; padding:8px 12px 0 !important;}
#pane-messages > * {flex:0 0 auto;}
#pane-messages .thread {flex:1 1 auto !important; height:auto !important; min-height:120px !important; overflow-y:auto;}
#pane-home > * {margin-left:14px; margin-right:14px;}
#pane-home #send-btn {margin-left:14px !important; margin-right:14px !important; width:auto !important;}
#pane-home .pop-banner {margin:0 0 14px !important; border-radius:0 !important;}
#pane-home .home-card {margin:0 14px 10px !important;}
#pane-home .home-link {margin:0 14px 10px !important;}
#pane-home .sec {margin:12px 14px 6px !important;}
#pane-home .starter {margin:0 14px 6px !important; width:calc(100% - 28px) !important; font-size:.9rem !important; border-radius:12px !important;}
#chatpanel #chips {padding:0 !important; margin:4px 0 0 !important; gap:6px !important;}
#chatpanel #chips button {background:#f4f4f5 !important; border:0 !important; border-radius:999px !important; color:#0b0b0c !important;
  font-size:.8rem !important; padding:5px 11px !important;}
#chatpanel #chips button::before {content:none !important; display:none !important;}
#chatpanel #chips button:hover {background:#e4e4e7 !important; color:#0b0b0c !important;}
#chatpanel #composer {padding:6px 0 0 !important; gap:4px !important; max-width:none !important;}
#chatpanel #ask {border-radius:999px !important; padding:2px 4px 2px 14px !important; background:#f4f4f5 !important; border:0 !important;}
#chatpanel #ask:focus-within {box-shadow:0 0 0 2px #0b0b0c !important; background:#ffffff !important;}
#chatpanel #ask button {padding:6px 14px !important; font-size:.85rem !important;}
#chatpanel .fine {font-size:.7rem !important; margin:2px 0 0 !important; text-align:center;}
#chatpanel #opts {padding:0 !important; margin:2px 0 4px !important; max-width:none !important;}
#chatpanel #newchat {align-self:flex-end; margin:0 0 4px !important;}
#chatnav {margin:0 !important; border-radius:0 0 20px 20px !important; padding:10px 12px !important;}
#explore {gap:12px !important; flex-wrap:wrap !important;}
.ask-card {flex:1 1 220px !important; min-height:84px !important; border-radius:18px !important; background:#ffffff !important;
  border:1px solid #e4e4e7 !important; color:#0b0b0c !important; font-weight:600 !important; text-align:left !important;
  justify-content:flex-start !important; padding:18px !important; white-space:normal !important; line-height:1.35 !important;}
.ask-card:hover {border-color:#0b0b0c !important; box-shadow:0 14px 28px -20px rgba(0,0,0,.5) !important;}
.facts {display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px; margin-top:20px;}
.fact {background:#ffffff; border:1px solid #e4e4e7; border-radius:18px; padding:20px 22px;}
.fact .yr, .visit .yr {font-family:ui-monospace, Menlo, monospace; font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; color:#71717a; margin-bottom:6px;}
.fact h3 {font-size:1.05rem; font-weight:700; margin:0 0 6px; color:#0b0b0c;}
.fact p {margin:0; color:#3f3f46; font-size:.95rem; line-height:1.55;}
.visit {margin-top:16px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px; background:#0b0b0c; color:#e4e4e7; border-radius:18px; padding:22px 24px; font-size:.98rem; line-height:1.5;}
.visit b {color:#ffffff;}
.visit a {color:#ffffff; font-weight:700; text-decoration:none; border:1px solid #52525b; border-radius:999px; padding:10px 18px;}
.visit a:hover {background:#ffffff; color:#0b0b0c;}
@media (max-width: 760px) {.facts {grid-template-columns:1fr;}}
#chat-launch.secondary {background:#ffffff !important; color:#0b0b0c !important; border:1px solid #d4d4d8 !important; width:auto !important; min-width:0 !important; padding:0 20px !important; border-radius:999px !important;}
/* ---- footer: dark, to mark the end of the page ---- */
.ft {margin-top:64px; background:#0b0b0c !important; color:#a1a1aa !important; border-radius:24px 24px 0 0; padding:48px 40px 0; width:100%; box-sizing:border-box;}
.ft-grid {display:grid; grid-template-columns:1.4fr 1fr 1fr 1.2fr; gap:36px; padding-bottom:36px; border-bottom:1px solid #27272a;}
.ft h4 {font-family:ui-monospace, Menlo, monospace; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:#71717a; margin:0 0 12px; font-weight:500;}
.ft ul {list-style:none; padding:0; margin:0;}
.ft li {margin-bottom:9px; font-size:.92rem; line-height:1.45;}
.ft a {color:#e4e4e7; text-decoration:none; font-weight:500;}
.ft a:hover {color:#ffffff; text-decoration:underline;}
.ft-logo {font-weight:900; font-size:1.5rem; color:#ffffff; letter-spacing:-.02em; margin-bottom:12px;}
.ft-brand p {margin:0 0 10px; font-size:.92rem; line-height:1.6; max-width:40ch;}
.ft-note {font-size:.82rem !important; color:#71717a;}
.ft-bar {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; padding:20px 0 28px; font-size:.82rem;}
.ft-bar a {color:#e4e4e7; text-decoration:none;}
.ft, .ft p, .ft li, .ft span, .ft div, .ft h4 {color:#a1a1aa !important;}
.ft a, .ft b, .ft .ft-logo {color:#e4e4e7 !important;}
.ft .ft-logo {color:#ffffff !important;}
.ft h4 {color:#71717a !important;}
.ft .ft-note {color:#71717a !important;}
@media (max-width: 900px) {.ft-grid {grid-template-columns:1fr 1fr;}}
@media (max-width: 560px) {.ft {padding:36px 22px 0;} .ft-grid {grid-template-columns:1fr;}}

/* ---- header: stays on top while scrolling ---- */
.mh {position:sticky; top:0; z-index:60; backdrop-filter:saturate(1.4) blur(10px); background:rgba(255,255,255,.92) !important;}
.mh-nav a {transition:opacity .15s;}
.mh-nav a:hover {opacity:.6; text-decoration:none !important;}
/* ---- section headings: large, dark, readable ---- */
#body .sec {font-family:"Instrument Serif", Georgia, serif !important; font-size:clamp(1.9rem, 3.2vw, 2.4rem) !important;
  font-weight:400 !important; color:#0b0b0c !important; letter-spacing:-.015em; line-height:1.1; padding:44px 0 16px !important; margin:0 !important;}
#body .ask-card, #body .ask-card * {font-size:1.1rem !important;}
#body .ask-card {font-family:"Inter", system-ui, sans-serif !important;}
#body .stat span {font-size:.85rem !important;}
#body .stats {gap:12px !important;}
#body .stat {padding:14px 18px !important; border-radius:14px !important; min-height:0 !important; border:0 !important; background:#0b0b0c;}
#body .stat b {font-size:1.6rem !important; line-height:1.1; color:#ffffff !important;}
#body .stat.has-img span {color:#e4e4e7 !important;}
#body .fact.has-img {border:0 !important; color:#ffffff; min-height:170px;}
#body .fact.has-img h3 {color:#ffffff !important;}
#body .fact.has-img p {color:#e4e4e7 !important;}
#body .fact.has-img .yr {color:#d4d4d8 !important;}
#body .about p, #body .about li {font-size:1.12rem !important; color:#27272a !important;}
#body .about h2 {font-family:"Instrument Serif", Georgia, serif !important; font-weight:400 !important; font-size:clamp(2.2rem, 4vw, 3rem) !important;}
#body .fact h3 {font-size:1.2rem !important;}
#body .fact p {font-size:1.02rem !important; color:#27272a !important;}
#body .featured .art b, #body .featured .art strong {font-size:1.1rem;}
#pane-home .sec {font-family:"Inter", system-ui, sans-serif !important; font-size:1.2rem !important; font-weight:700 !important; padding:14px 14px 6px !important;}
#pane-home .starter {font-size:1rem !important;}
/* ---- header: brand, centred navigation, one call to action ---- */
.hdr {position:sticky; top:0; z-index:60; background:rgba(255,255,255,.9) !important; backdrop-filter:saturate(1.6) blur(14px);
  border-bottom:1px solid #e4e4e7; margin:0 !important;}
.hdr-in {max-width:1180px; margin:0 auto; height:68px; padding:0 28px; display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:16px;}
.hdr-brand {display:inline-flex; align-items:center; gap:10px; justify-self:start; text-decoration:none !important; color:#0b0b0c !important;}
.hdr-mark {width:32px; height:32px; border-radius:9px; background:#0b0b0c; color:#ffffff; display:inline-flex; align-items:center; justify-content:center;
  font-family:"Instrument Serif", Georgia, serif; font-size:1.25rem; line-height:1;}
.hdr-name {font-family:"Inter", system-ui, sans-serif; font-weight:800; font-size:1.12rem; letter-spacing:-.02em;}
.hdr-nav {display:flex; gap:6px; justify-self:center; background:#f4f4f5; border-radius:999px; padding:4px;}
.hdr-nav a {font-family:"Inter", system-ui, sans-serif; font-size:.95rem; font-weight:600; color:#3f3f46 !important; text-decoration:none !important;
  padding:8px 16px; border-radius:999px; transition:background .15s, color .15s;}
.hdr-nav a:hover {background:#ffffff; color:#0b0b0c !important; box-shadow:0 1px 2px rgba(0,0,0,.08);}
.hdr-cta {justify-self:end; font-family:"Inter", system-ui, sans-serif; font-size:.92rem; font-weight:700; color:#ffffff !important; background:#0b0b0c;
  padding:10px 18px; border-radius:999px; text-decoration:none !important; transition:transform .15s, box-shadow .15s;}
.hdr-cta:hover {transform:translateY(-1px); box-shadow:0 10px 20px -12px rgba(0,0,0,.6);}
@media (max-width: 760px) {
  .hdr-in {grid-template-columns:1fr auto; height:60px; padding:0 16px;}
  .hdr-nav {display:none;}
  .hdr-cta {font-size:.84rem; padding:9px 14px;}
}
#body .visit, #body .visit div {color:#d4d4d8 !important;}
#body .visit b {color:#ffffff !important;}
/* ---- one brand lockup everywhere: mark + name, same look in header, footer and chat ---- */
.brand-lockup {display:inline-flex; align-items:center; gap:10px;}
.brand-lockup .hdr-mark {width:28px; height:28px; font-size:1.05rem;}
.brand-lockup .hdr-name {font-size:1.05rem; color:#0b0b0c;}
.ft-logo {display:inline-flex; align-items:center; gap:10px; margin-bottom:14px !important;}
.ft-logo .hdr-mark {background:#ffffff !important; color:#0b0b0c !important; width:32px; height:32px;}
.ft-logo .hdr-name {color:#ffffff !important; font-size:1.25rem;}
@media (max-width: 600px) {
  #chatpanel {left:10px !important; right:10px !important; width:auto !important; bottom:86px !important; height:76vh !important;}
  #chat-launch {right:16px !important; bottom:16px !important;}
}




/* ---- accessibility: text contrast (WCAG 2.2 SC 1.4.3, at least 4.5:1) ----
   The refresh made the sidebar white and the header black; these rules still
   assumed the old dark sidebar and light header. Colors reuse the theme's own. */
#side .brand .name {color:#0b0b0c !important;}
#side .brand .sub {color:#5b6470 !important;}
#side button.primary {background:#e4e4e7 !important; color:#0b0b0c !important; box-shadow:inset 3px 0 0 #0b0b0c !important;}
.avatar {background:#0b0b0c !important; color:#ffffff !important;}
.status span {color:#d4d4d8 !important;}
.fine {color:#5b6470 !important;}
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
_pool: list[dict] = []  # more on-view paintings for card backgrounds, one per artist


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
    # A pool of other paintings on view (one per artist, none already featured) for card backgrounds.
    _pool.clear()
    used = {f["image"] for f in _featured}
    paint = works[
        (works["Classification"] == "Painting") & (works["on_view"].astype(bool)) & works["ImageURL"].notna()
    ]
    seen_artists: set = set()
    for _, row in paint.iterrows():
        if row["ImageURL"] in used or row["Artist"] in seen_artists:
            continue
        seen_artists.add(row["Artist"])
        used.add(row["ImageURL"])
        _pool.append({"title": row["Title"], "image": row["ImageURL"], "link": row["link"]})
        if len(_pool) >= 24:
            break


def _slot(k: int) -> dict | None:
    """The k-th painting for a background, so each place on the page shows a different one."""
    source = _pool or _featured
    return source[k % len(source)] if source else None


def _gutter_html(side: str, start: int, seconds: int) -> str:
    """A slow looping slideshow of paintings for the empty margin beside the page content."""
    slides = []
    for i in range(4):
        item = _slot(start + i)
        if not item:
            continue
        slides.append(
            f"<a class='gut-slide' href='{_e(item['link'])}' target='_blank' rel='noopener' "
            f"title='{_e(item['title'])}' style=\"background-image:url('{_e(item['image'])}');"
            f'animation-duration:{seconds}s;animation-delay:{i * seconds / 4:g}s"></a>'
        )
    return f"<div class='gut gut-{side}'>{''.join(slides)}</div>"


def gutters_html() -> str:
    """Left and right margins: two independent slideshows with paintings not used elsewhere."""
    return _gutter_html("left", 8, 28) + _gutter_html("right", 12, 36)


def _bg_style(item: dict | None = None) -> str:
    """Background for the hero and chat banner: a MoMA painting, faded behind the text."""
    item = item or _slot(0)
    if not item:
        return ""
    img = _e(item["image"])
    return (
        "background: linear-gradient(180deg, rgba(10,10,12,.35) 0%, rgba(10,10,12,.82) 70%, "
        f"rgba(10,10,12,.95) 100%), url('{img}') center 30% / cover no-repeat;"
    )


def featured_html() -> str:
    if not _featured:
        return ""
    items = "".join(
        f"<a class='art' href='{_e(f['link'])}' target='_blank' rel='noopener'>"
        f"<img src='{_e(f['image'])}' alt='{_e(f['title'])}' loading='lazy'>"
        f"<span><b>{_e(f['title'])}</b>{_e(f['artist'])}"
        f"<em class='credit'>Image: MoMA &#8599;</em></span></a>"
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
            f"<div class='ti'>{_e(c['title'])}</div><div class='ar'>{_e(c['artist'])}, {_e(c['date'])}</div>{tag}{why}"
            + ("<div class='credit'>Image: MoMA</div>" if img.startswith("http") else "")
            + "</div></a>"
        )
    return "<div class='works'>" + "".join(out) + "</div>"


def thread_html(history: list) -> str:
    if not history:
        greeting = (
            "<div class='row-a'><div class='avatar'>A</div><div class='ans'><div class='card'>"
            "<p>Hi! Ask about a work, an artist, or a floor.</p></div></div></div>"
        )
        return (
            "<div class='thread' role='log' aria-live='polite' aria-label='Conversation with AskMoMA'>"
            f"<div class='inner'>{greeting}</div></div>"
        )
    rows = []
    for q, a, cards in history:
        rows.append(f"<div class='row-u'><div class='bubble'>{_e(q)}</div></div>")
        body = (
            f"<div class='card'>{a}</div>{works_html(cards)}"
            if a
            else "<div class='card'><span class='typing' aria-label='Working'><i></i><i></i><i></i></span></div>"
        )
        rows.append(f"<div class='row-a'><div class='avatar'>A</div><div class='ans'>{body}</div></div>")
    return (
        "<div class='thread' role='log' aria-live='polite' aria-label='Conversation with AskMoMA'><div class='inner'>"
        + "".join(rows)
        + "</div></div>"
    )


def topbar_html(col) -> str:
    return (
        "<div class='topbar'><div class='t'>Museum guide<small>Answers come from MoMA's public collection data</small></div>"
        f"<div class='status'><i></i><span><b>{len(col.on_view):,}</b> works on view</span>"
        f"<span>Updated {_e(col.as_of)}</span></div></div>"
    )


# ---------------------------------------------------------------- UI
def header_html() -> str:
    return (
        "<header class='hdr'><div class='hdr-in'>"
        "<a class='hdr-brand' href='#'><span class='hdr-mark'>A</span><span class='hdr-name'>AskMoMA</span></a>"
        "<nav class='hdr-nav' aria-label='MoMA links'>"
        "<a href='https://www.moma.org/visit' target='_blank' rel='noopener'>Visit</a>"
        "<a href='https://www.moma.org/calendar/exhibitions' target='_blank' rel='noopener'>Exhibitions</a>"
        "<a href='https://www.moma.org/artists' target='_blank' rel='noopener'>Art and artists</a>"
        "</nav>"
        "<a class='hdr-cta' href='https://www.moma.org' target='_blank' rel='noopener'>Visit moma.org &#8599;</a>"
        "</div></header>"
    )


def hero_html() -> str:
    return (
        f"<section class='hero-wide' style=\"{_bg_style(_slot(0))}\">"
        "<h1>Welcome to AskMoMA</h1>"
        "<p>Your guide to the art on view at The Museum of Modern Art. Ask a question, or pick a floor.</p>"
        + (
            f"<a class='credit hero-credit' href='{_e(_slot(0)['link'])}' target='_blank' rel='noopener'>"
            f"Image: {_e(_slot(0)['title'])}, MoMA &#8599;</a>"
            if _featured
            else ""
        )
        + "</section>"
    )


def _img_bg(i: int) -> str:
    """Inline background: a distinct MoMA painting under a dark wash, so white text stays readable."""
    source = _pool or _featured
    if not source:
        return ""
    img = _e(source[i % len(source)]["image"])
    return (
        "background: linear-gradient(180deg, rgba(10,10,12,.30) 0%, rgba(10,10,12,.82) 100%), "
        f"url('{img}') center / cover no-repeat;"
    )


def about_html(col) -> str:
    on = col.works[col.works["on_view"]]
    artists = on.explode("cids")["cids"].dropna().nunique()
    stats = [
        (f"{len(on):,}", "works on view"),
        (f"{artists:,}", "artists"),
        (f"{len(col.floors())}", "floors"),
    ]
    cards = "".join(
        f"<div class='stat has-img' style=\"{_img_bg(i + 2)}\"><b>{n}</b><span>{label}</span></div>"
        for i, (n, label) in enumerate(stats)
    )
    facts = [
        (
            "1929",
            "Opened in New York",
            "The museum opened on November 7, 1929, with Alfred H. Barr, Jr. as its first director. "
            "Abby Aldrich Rockefeller, Lillie P. Bliss and Mary Quinn Sullivan were among its founders.",
        ),
        (
            "1941",
            "The Starry Night arrives",
            "Van Gogh's The Starry Night was acquired through the Lillie P. Bliss Bequest. "
            "It was the first van Gogh to enter a New York museum's collection.",
        ),
        (
            "~200,000",
            "Works in the collection",
            "Paintings, sculptures, drawings, prints, photographs, media, architecture and design. "
            "The collection began with a gift of eight prints and one drawing.",
        ),
    ]
    fact_cards = "".join(
        f"<div class='fact has-img' style=\"{_img_bg(i + 5)}\"><div class='yr'>{y}</div>"
        f"<h3>{_e(t)}</h3><p>{_e(d)}</p></div>"
        for i, (y, t, d) in enumerate(facts)
    )
    return (
        f"<div class='stats'>{cards}</div>"
        "<section class='about'><h2>About MoMA</h2>"
        "<p>The Museum of Modern Art connects people from around the world to the art of our time. "
        "It is a gathering place for artists, students and visitors, and it cares for one of the "
        "leading collections of modern and contemporary art.</p>"
        f"<div class='facts'>{fact_cards}</div>"
        "<div class='visit'><div><div class='yr'>Visit</div>"
        "<b>11 West 53rd Street</b>, between Fifth and Sixth Avenues, in Midtown Manhattan. "
        "Hours, tickets and gallery changes are on moma.org.</div>"
        "<a href='https://www.moma.org' target='_blank' rel='noopener'>Plan your visit &rarr;</a></div>"
        "</section>"
    )


def popup_banner_html() -> str:
    """Painting banner at the top of the chat popup, with the AskMoMA name over it."""
    credit = ""
    if _slot(1):
        credit = (
            f"<a class='credit pop-credit' href='{_e(_slot(1)['link'])}' target='_blank' rel='noopener'>"
            f"Image: {_e(_slot(1)['title'])}, MoMA &#8599;</a>"
        )
    return (
        f"<div class='pop-banner' style=\"{_bg_style(_slot(1))}\">"
        "<div class='pop-title'>Welcome to AskMoMA</div>"
        "<div class='pop-sub'>Ask about the art on view, or pick a floor.</div>"
        f"{credit}</div>"
    )


def home_html() -> str:
    return (
        "<div class='home-card'><p class='kicker'>Hi there</p>"
        "<h2>How can we help?</h2>"
        "<p>Ask about any work on view, an artist, or your visit.</p></div>"
        "<a class='home-link' href='https://www.moma.org' target='_blank' rel='noopener'>"
        "Visit moma.org <span>&rarr;</span></a>"
    )


def footer_html(col) -> str:
    link = "<a href='{u}' target='_blank' rel='noopener'>{t} &#8599;</a>"
    return (
        "<div class='ft'><div class='ft-grid'>"
        "<div class='ft-brand'><div class='ft-logo'><span class='hdr-mark'>A</span><span class='hdr-name'>AskMoMA</span></div>"
        "<p>A guide to the works on view at The Museum of Modern Art, built from MoMA's public collection data.</p>"
        "<p class='ft-note'>A student project for MSAI-631 at the University of the Cumberlands. "
        "Not affiliated with The Museum of Modern Art.</p></div>"
        "<div><h4>Visit MoMA</h4><ul>"
        + "".join(
            f"<li>{link.format(u=u, t=t)}</li>"
            for u, t in [
                ("https://www.moma.org/visit", "Plan your visit"),
                ("https://www.moma.org/calendar/exhibitions", "Exhibitions"),
                ("https://www.moma.org/calendar/events", "Events"),
                ("https://www.moma.org", "moma.org"),
            ]
        )
        + "</ul></div>"
        "<div><h4>Collection</h4><ul>"
        + "".join(
            f"<li>{link.format(u=u, t=t)}</li>"
            for u, t in [
                ("https://www.moma.org/collection/", "Browse the collection"),
                ("https://www.moma.org/artists", "Artists"),
                ("https://github.com/MuseumofModernArt/collection", "Open dataset (CC0)"),
            ]
        )
        + "</ul></div>"
        "<div><h4>References</h4><ul>"
        "<li>Artwork and artist data: MoMA public collection, CC0, "
        "<a href='https://github.com/MuseumofModernArt/collection' target='_blank' rel='noopener'>GitHub dataset</a>"
        f" (as of {_e(col.as_of)}).</li>"
        "<li>Images and work pages: <a href='https://www.moma.org' target='_blank' rel='noopener'>moma.org</a>, "
        "credited under each picture.</li>"
        "<li>Built with Gradio, pandas and scikit-learn.</li>"
        "<li>Fonts: <a href='https://fonts.google.com' target='_blank' rel='noopener'>Inter and Instrument Serif</a> "
        "(Google Fonts).</li>"
        "</ul></div>"
        "</div>"
        "<div class='ft-bar'><span>&copy; 2026 AskMoMA student project</span>"
        "<a href='https://github.com/chon3806/askmoma' target='_blank' rel='noopener'>Source code on GitHub &#8599;</a>"
        "</div></div>"
    )


# Keeps the chat on the newest message: whenever a thread's content changes, scroll it to the bottom.
SCROLL_JS = """
() => {
  if (window.__askmomaScroll) return;
  window.__askmomaScroll = true;
  const last = new WeakMap();
  const pin = () => {
    document.querySelectorAll('.thread').forEach((t) => {
      const n = t.innerHTML.length;
      if (last.get(t) !== n) {
        last.set(t, n);
        t.scrollTop = t.scrollHeight;
      }
    });
  };
  new MutationObserver(pin).observe(document.body, {childList: true, subtree: true});
}
"""


def build_ui(bot):
    import gradio as gr

    set_featured(bot.col.works)

    from askmoma import llm

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
        text = llm.polish(question, ans.text)  # unchanged unless ASKMOMA_LLM is set
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
        open_state = gr.State(False)
        gr.HTML(header_html())
        gr.HTML(hero_html())
        with gr.Column(elem_id="body"):
            gr.HTML(gutters_html(), elem_id="gutters")
            gr.HTML(about_html(col))
            gr.HTML("<p class='sec'>Paintings on view</p>" + featured_html())
            gr.HTML("<p class='sec'>Common FAQs</p>")
            with gr.Row(elem_id="explore"):
                explore_btns = [gr.Button(q, elem_classes=["ask-card"]) for q in EXPLORE]
            with gr.Row(elem_id="floors", visible=False):
                floor_btns = [
                    gr.Button(
                        f"Floor {f}\n{int(counts.get(f, 0)):,} works",
                        variant="secondary",
                        elem_classes=["floor-card"],
                    )
                    for f in floors
                ]
            gr.HTML(footer_html(col))
        with gr.Column(elem_id="chatpanel", visible=False) as panel:
            with gr.Row(elem_id="chat-head"):
                gr.HTML(
                    "<div class='chat-head-text'><div class='chat-title brand-lockup'><span class='hdr-mark'>A</span><span class='hdr-name'>AskMoMA</span></div>"
                    "<div class='chat-sub'>Museum guide for works on view</div></div>"
                )
                close_btn = gr.Button("✕", elem_id="chat-close", size="sm")
            with gr.Column(visible=True, elem_id="pane-home") as pane_home:
                gr.HTML(popup_banner_html())
                gr.HTML(home_html())
                send_btn = gr.Button("Send us a message", elem_id="send-btn", variant="primary")
                gr.HTML("<p class='sec'>Popular questions</p>")
                starter_btns = [
                    gr.Button(q, elem_classes=["starter"])
                    for q in STARTERS + ["What are the museum hours?", "How much is admission?"]
                ]
            with gr.Column(visible=False, elem_id="pane-messages") as pane_msgs:
                new_chat = gr.Button("New conversation", elem_id="newchat", size="sm")
                thread = gr.HTML(thread_html([]))
                with gr.Column(elem_id="composer"):
                    with gr.Row(elem_id="chips"):
                        chips = [gr.Button(visible=False, size="sm") for _ in range(N_CHIPS)]
                    msg = gr.Textbox(
                        elem_id="ask",
                        label="Your question",  # announced by screen readers even though it is not shown
                        placeholder="Ask about an artwork, an artist, or your visit",
                        show_label=False,
                        container=False,
                        autofocus=True,
                        submit_btn="Send",
                        lines=1,
                        max_lines=4,
                    )
                    gr.HTML("<p class='fine'>Check moma.org for hours and gallery changes.</p>")
                with gr.Row(elem_id="opts"):
                    on_view_only = gr.Checkbox(
                        value=True, label="Only list works that are on view", container=False
                    )
            with gr.Row(elem_id="chatnav"):
                nav_home = gr.Button("Home", elem_id="nav-home", variant="primary")
                nav_msgs = gr.Button("Messages", elem_id="nav-messages", variant="secondary")
        launch = gr.Button("Ask", elem_id="chat-launch")

        def launcher(is_open):
            return gr.update(
                value="✕ Close" if is_open else "Ask",
                variant="secondary" if is_open else "primary",
            )

        def toggle(is_open):
            now_open = not is_open
            return now_open, gr.update(visible=now_open), launcher(now_open)

        def close_chat():
            return False, gr.update(visible=False), launcher(False)

        pane_outs = [pane_home, pane_msgs, nav_home, nav_msgs]

        def show(tab):
            home = tab == "home"
            return (
                gr.update(visible=home),
                gr.update(visible=not home),
                gr.update(variant="primary" if home else "secondary"),
                gr.update(variant="primary" if not home else "secondary"),
            )

        def open_messages():
            return (gr.update(visible=True), *show("messages"))

        launch.click(toggle, [open_state], [open_state, panel, launch], queue=False)
        close_btn.click(close_chat, None, [open_state, panel, launch], queue=False)
        send_btn.click(lambda: show("messages"), None, pane_outs, queue=False)
        nav_home.click(lambda: show("home"), None, pane_outs, queue=False)
        nav_msgs.click(lambda: show("messages"), None, pane_outs, queue=False)

        outs = [thread, hist, ctx, *chips, *floor_btns]
        run = dict(fn=answer, inputs=[hist, ctx, on_view_only], outputs=outs)
        msg.submit(add_user, [msg, hist], [msg, hist, thread], queue=False).then(**run)
        for b in chips:
            b.click(add_user, [b, hist], [msg, hist, thread], queue=False).then(**run)
        for b in starter_btns + explore_btns:
            b.click(add_user, [b, hist], [msg, hist, thread], queue=False).then(
                open_messages, None, [panel, *pane_outs], queue=False
            ).then(**run)
        for f, b in zip(floors, floor_btns, strict=True):
            b.click(
                functools.partial(add_user, f"What's on floor {f}?"), [hist], [msg, hist, thread], queue=False
            ).then(open_messages, None, [panel, *pane_outs], queue=False).then(**run)
        new_chat.click(reset, None, outs)
        demo.load(None, None, None, js=SCROLL_JS)
    demo.askmoma_launch_kwargs = style if gradio6 else {}
    return demo


def cli(bot):
    from askmoma import llm

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
        print(f"\naskmoma [{a.intent}]>\n{llm.polish(q, a.text)}\n")
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
    from askmoma import config, llm

    if config.LLM_MODEL:
        llm.load()  # at startup, so the first visitor does not wait for it
    if args.cli:
        cli(bot)
    else:
        demo = build_ui(bot)
        demo.launch(server_port=args.port, share=args.share, **demo.askmoma_launch_kwargs)


if __name__ == "__main__":
    main()
