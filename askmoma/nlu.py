"""Understand a visitor's message: which scenario it is and what it is about.

Rule-based on purpose: it is fast, runs on any laptop, and every decision can
be traced. The optional LLM layer only rewords answers; it never decides
what the facts are.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .data import Collection, norm

# Words that describe the request rather than the artwork/artist.
STOP = set("""
a an the of on in at to for is are was were be been it its this that these those
who whom whose what which where when why how do does did can could would should
i me my we our you your he she him her they them their there here
made make makes making painted paint paints created create creates drew draw
designed design photographed sculpted produced did artist artists work works
artwork artworks piece pieces painting paintings by from about tell show find
give list any some all please moma museum see view on_view displayed display
floor gallery galleries same similar like liked love loved enjoyed enjoy else
other more year years old medium size big dimensions acquire acquired get got
info information details detail and or with
""".split())

PRONOUN_WORK = re.compile(r"\b(it|this|that|this one|that one|this work|that work|the painting|this painting)\b")
PRONOUN_ARTIST = re.compile(r"\b(they|them|he|she|him|her|this artist|that artist|the artist)\b")

INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("greeting", re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b|\b(help|what can you do|how do you work)\b")),
    ("faq_hours", re.compile(r"\b(hours?|open(ing)?|clos(e|ed|ing)|what time)\b")),
    ("faq_admission", re.compile(r"\b(tickets?|admission|price|prices|cost|how much|free (entry|admission|day|night))\b")),
    ("faq_access", re.compile(r"\b(wheelchair|accessib\w*|disabilit\w*|service animal|care partner)\b")),
    ("faq_address", re.compile(r"\b(address|where is moma|where is the museum|how do i get (there|to moma))\b")),
    ("top_artists", re.compile(r"\b(most|top|rank\w*|biggest|leading)\b.*\bartists?\b|\bartists?\b.*\b(most works|most pieces)\b")),
    ("recommend", re.compile(r"\b(i (really )?(loved|liked|love|like|enjoyed)|similar to|recommend\w*|suggest\w*|what else should i see|more like)\b")),
    ("location", re.compile(r"\b(where (is|are|can i (find|see))|which (floor|gallery|room)|is .* on view|can i see|on display)\b")),
    ("who_made", re.compile(r"\b(who (made|painted|created|did|drew|designed|photographed|sculpted|is the artist of|was the artist of)|artist (of|behind)|by whom)\b")),
    ("when_made", re.compile(r"\b(when was .* (made|painted|created|done)|what year|how old|date of)\b")),
    ("medium", re.compile(r"\b(medium|made of|materials?|what kind of (paint|material))\b")),
    ("dimensions", re.compile(r"\b(how (big|large|tall|wide)|size|dimensions)\b")),
    ("acquired", re.compile(r"\b(acquir\w*|when did moma (get|buy)|credit line|gift of|donat\w*)\b")),
    ("works_by_artist", re.compile(r"\b(works?|paintings?|pieces?|art|photos?|photographs?|sculptures?|prints?|drawings?) (by|from)\b|\bwhat (else )?(did|does|has) .*\b(make|made|paint|painted|create|created)\b|\bwhat else (by|from)\b")),
    ("floor_list", re.compile(r"\b(on|in) (floor|level) \d+\b|\bfloor \d+\b|\bgallery \d{3}\b|\bfifth floor|fourth floor|second floor|third floor\b")),
    ("artist_info", re.compile(r"\b(who (is|was)|tell me about|about the artist|biography|bio)\b")),
    ("search", re.compile(r"\b(find|search|looking for|show me|any)\b")),
]

ORDINAL_FLOORS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}


@dataclass
class Parsed:
    text: str
    intent: str
    work_rows: list[int] = field(default_factory=list)  # candidate artworks
    artist_cids: list[str] = field(default_factory=list)
    floor: int | None = None
    gallery: str | None = None
    nationality: str | None = None
    residue: str = ""  # message with intent words removed (for search)


def detect_intent(text: str) -> str:
    t = text.lower()
    for name, pattern in INTENT_RULES:
        if pattern.search(t):
            return name
    return "unknown"


def _ngrams(tokens: list[str], max_n: int = 12):
    for n in range(min(max_n, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            yield i, n, " ".join(tokens[i:i + n])


def find_title(col: Collection, tokens: list[str]) -> tuple[list[int], tuple[int, int] | None]:
    """Longest run of words in the message that is exactly a known title."""
    for i, n, gram in _ngrams(tokens):
        content = [t for t in gram.split() if t not in STOP]
        if not content:
            continue
        if n == 1 and (len(gram) < 4 or gram in STOP):
            continue
        for key in (gram, gram[4:] if gram.startswith("the ") else None):
            rows = col.title_index.get(key) if key else None
            if not rows:
                continue
            # A title with a single content word ("Night", "Untitled") only counts
            # if it is on view; otherwise generic words would match old records.
            if len(content) == 1 and not col.works.at[rows[0], "on_view"]:
                continue
            return rows, (i, i + n)
    return [], None


def find_artist(col: Collection, tokens: list[str]) -> tuple[list[str], tuple[int, int] | None]:
    for i, n, gram in _ngrams(tokens, 6):
        cid = col.artist_index.get(gram)
        if cid and (n > 1 or len(gram) > 3) and gram not in STOP:
            return [cid], (i, i + n)
    # Surname only: "Picasso", "van Gogh" -> artist with most works.
    for i, n, gram in _ngrams(tokens, 3):
        if gram in STOP or len(gram) < 4:
            continue
        last = gram.split()[-1]
        if last in col.surname_index and (n == 1 or gram.endswith(last)):
            return col.surname_index[last][:3], (i, i + n)
    return [], None


def _overlap(a, b) -> bool:
    return bool(a and b and not (a[1] <= b[0] or b[1] <= a[0]))


def fuzzy_title(col: Collection, residue: str) -> list[int]:
    if len(residue) < 4:
        return []
    toks = [t for t in residue.split() if t not in STOP]
    pool: set[str] = set()
    for t in toks:
        titles = col.title_tokens.get(t, ())
        if 0 < len(titles) < 5000:
            pool |= titles
    if len(pool) < 50:  # typos in every word: fall back to on-view titles
        pool |= set(col.on_view["title_norm"])
    # Compare on content words only, so "the ... at" don't dilute the score.
    stripped = {" ".join(w for w in t.split() if w not in STOP): t for t in pool}
    stripped.pop("", None)
    best = difflib.get_close_matches(" ".join(toks), list(stripped), n=1, cutoff=0.72)
    return col.title_index.get(stripped[best[0]], []) if best else []


def fuzzy_artist(col: Collection, residue: str) -> list[str]:
    if len(residue) < 4:
        return []
    best = difflib.get_close_matches(residue, list(col.artist_index), n=1, cutoff=0.82)
    if best:
        return [col.artist_index[best[0]]]
    for tok in sorted(residue.split(), key=len, reverse=True):  # "piccaso" -> "picasso"
        if len(tok) >= 5:
            hit = difflib.get_close_matches(tok, list(col.surname_index), n=1, cutoff=0.8)
            if hit:
                return col.surname_index[hit[0]][:3]
    return []


def parse(col: Collection, text: str, context: dict | None = None) -> Parsed:
    context = context or {}
    intent = detect_intent(text)
    lower = text.lower()
    tokens = norm(text).split()
    p = Parsed(text=text, intent=intent)

    m = re.search(r"\b(?:floor|level) (\d+)\b", lower) or re.search(r"\b(\d)(?:st|nd|rd|th) floor\b", lower)
    if m:
        p.floor = int(m.group(1))
    for word, num in ORDINAL_FLOORS.items():
        if f"{word} floor" in lower:
            p.floor = num
    g = re.search(r"\bgallery (\d{3}[a-z]?)\b", lower)
    if g:
        p.gallery = g.group(1).upper()

    if intent.startswith("faq") or intent == "greeting":
        return p

    artist_first = intent in ("works_by_artist", "artist_info", "top_artists", "nationality")
    if intent not in ("top_artists", "floor_list"):
        p.work_rows, t_span = find_title(col, tokens)
    else:
        t_span = None
    p.artist_cids, a_span = find_artist(col, tokens)
    if _overlap(t_span, a_span):
        # Same words matched both a title and an artist: let the question decide.
        if artist_first:
            p.work_rows = []
        else:
            p.artist_cids = []
            # ...but the artist may be named elsewhere ("Water Lilies by Monet").
            rest = tokens[:t_span[0]] + ["|"] + tokens[t_span[1]:]
            p.artist_cids, _ = find_artist(col, rest)

    for n_norm, label in col.nationalities.items():
        if re.search(rf"\b{re.escape(n_norm)}\b", norm(text)):
            p.nationality = label
            break

    keep = [t for t in tokens if t not in STOP]
    p.residue = " ".join(keep)

    # "find dreamy night sky paintings": a title covering only part of a
    # description is a coincidence, so search by meaning instead.
    if intent == "search" and p.work_rows and t_span:
        outside = [t for j, t in enumerate(tokens)
                   if not (t_span[0] <= j < t_span[1]) and t not in STOP]
        if outside:
            p.work_rows = []

    # Fuzzy fallbacks for typos ("starry nite", "piccaso").
    if not p.work_rows and not p.artist_cids and p.residue and intent not in ("top_artists", "floor_list", "search"):
        p.work_rows = fuzzy_title(col, p.residue)
        if not p.work_rows:
            p.artist_cids = fuzzy_artist(col, p.residue)

    # Follow-ups: "where is it?", "what else did they make?"
    if not p.work_rows and not p.artist_cids:
        if PRONOUN_WORK.search(lower) and context.get("work_row") is not None:
            p.work_rows = [context["work_row"]]
        elif PRONOUN_ARTIST.search(lower) and context.get("artist_cid"):
            p.artist_cids = [context["artist_cid"]]
        elif intent in ("location", "who_made", "when_made", "medium", "dimensions", "acquired") \
                and context.get("work_row") is not None and not keep:
            p.work_rows = [context["work_row"]]

    if p.intent == "unknown" or p.intent == "search":
        if p.work_rows:
            p.intent = "work_info"
        elif p.artist_cids and p.nationality is None:
            p.intent = "artist_info"
    if p.intent == "artist_info" and p.work_rows and not p.artist_cids:
        p.intent = "work_info"
    if p.intent == "works_by_artist" and not p.artist_cids and p.nationality:
        p.intent = "nationality"
    if p.nationality and not p.artist_cids and not p.work_rows and "artist" in lower:
        p.intent = "nationality"
    return p
