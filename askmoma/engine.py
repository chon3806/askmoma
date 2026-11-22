"""Turn a parsed request into a grounded answer.

Every artwork fact comes from a retrieved row; every answer links to its
source on moma.org and says how current the data is.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import Collection, norm
from .nlu import DESCRIPTIVE, Parsed, parse, split_preferences
from .search import Searcher

# Recommendations, from the week 11 recommender (CineMatch):
DISLIKE_WEIGHT = 0.5  # Rocchio: how far a disliked work pushes the profile away
MMR_LAMBDA = 0.7  # maximal marginal relevance: 1 = only relevance, 0 = only variety
NONE_OF_THESE = {"none of these", "none", "neither", "none of them", "not these"}

FAQ = json.loads((Path(__file__).parent / "faq.json").read_text(encoding="utf-8"))


@dataclass
class Answer:
    text: str
    cards: list[dict] = field(default_factory=list)  # artworks to show in the side panel
    suggestions: list[str] = field(default_factory=list)
    intent: str = ""
    context: dict = field(default_factory=dict)  # carried to the next turn


def _clean(v, fallback=""):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return fallback
    s = str(v).strip()
    return s if s and s.lower() != "nan" else fallback


class AskMoMA:
    def __init__(self, collection: Collection):
        self.col = collection
        self.search = Searcher(collection)

    # ------------------------------------------------------------ helpers
    def _card(self, row: int, reason: str = "") -> dict:
        w = self.col.works.loc[row]
        return {
            "row": int(row),
            "title": _clean(w["Title"], "Untitled"),
            "artist": _clean(w["Artist"], "Unknown artist"),
            "date": _clean(w["Date"], "n.d."),
            "medium": _clean(w["Medium"]),
            "classification": _clean(w["Classification"]),
            "location": _clean(w["location"]),
            "on_view": bool(w["on_view"]),
            "link": w["link"],
            "image": _clean(w.get("ImageURL")) or _clean(w.get("ThumbnailURL")),
            "reason": reason,
        }

    def _work_line(self, row: int, with_location: bool = True) -> str:
        c = self._card(row)
        loc = f" · {c['location']}" if with_location and c["on_view"] else ""
        return f"[{c['title']}]({c['link']}), {c['artist']} ({c['date']}){loc}"

    def _as_of(self) -> str:
        return f"\n\n_Collection data as of {self.col.as_of}._"

    def _artist_name(self, cid: str) -> str:
        a = self.col.artist(cid)
        return _clean(a.get("DisplayName")) if a else "this artist"

    def _pick_work(self, rows: list[int]) -> tuple[int | None, list[int]]:
        """Choose one work; return (row, alternatives) when the title is ambiguous."""
        if not rows:
            return None, []
        works = self.col.works
        on_view = [r for r in rows if works.at[r, "on_view"]]
        pool = on_view or rows
        artists = {works.at[r, "Artist"] for r in pool}
        if len(artists) <= 1:
            return pool[0], []
        return None, pool[: config.MAX_LIST]

    def _disambiguate(self, rows: list[int]) -> Answer:
        lines = "\n".join(f"{i}. {self._work_line(r)}" for i, r in enumerate(rows, 1))
        first = self._card(rows[0])
        return Answer(
            text=f"I found several works called **{first['title']}**. Which one do you mean?\n\n{lines}",
            cards=[self._card(r) for r in rows],
            suggestions=[
                f"Tell me about {self._card(r)['title']} by {self._card(r)['artist']}" for r in rows[:3]
            ]
            + ["None of these"],  # a way out when the guess is wrong (CUX Guide repair)
            intent="disambiguate",
            context={"choices": [int(r) for r in rows]},
        )

    def _none_of_these(self, context: dict) -> Answer:
        title = self._card(context["choices"][0])["title"]
        return Answer(
            text=(
                f"Sorry, none of those. Which **{title}** do you mean? Tell me the artist or the year, "
                f"for example *{title} by* followed by the artist's name. You can also search "
                f"[moma.org]({config.MOMA_HOME}/collection/)."
            ),
            suggestions=["What can you do?"],
            intent="clarify",
            context={k: v for k, v in context.items() if k != "choices"},
        )

    def _filter(self, df: pd.DataFrame, floor: int | None, on_view_only: bool) -> pd.DataFrame:
        if on_view_only:
            df = df[df["on_view"]]
        if floor:
            df = df[df["floor"] == floor]
        return df

    # ------------------------------------------------------------ entry
    def ask(
        self,
        message: str,
        context: dict | None = None,
        floor_filter: int | None = None,
        on_view_only: bool = True,
    ) -> Answer:
        message = (message or "").strip()
        if not message:
            return Answer(text="Ask me about an artwork, an artist, or your visit.")
        link = re.match(r"^MoMA link for (.+?)(?: on floor (\d+))?$", message)
        if link:
            return self._artist_link(link.group(1), int(link.group(2)) if link.group(2) else None)
        if norm(message) in NONE_OF_THESE and (context or {}).get("choices"):
            return self._none_of_these(context)
        p = parse(self.col, message, context)
        handler = getattr(self, f"_h_{p.intent}", None)
        work_intents = {
            "location",
            "who_made",
            "when_made",
            "medium",
            "dimensions",
            "acquired",
            "work_info",
            "recommend",
        }
        if p.intent in ("artist_info", "works_by_artist") and not p.artist_cids:
            handler = self._h_work_info if p.work_rows else self._h_unknown
        elif p.intent == "location" and not p.work_rows and (p.floor or p.gallery):
            handler = self._h_floor_list
        elif p.intent in work_intents and not p.work_rows and p.artist_cids:
            handler = self._h_works_by_artist if p.intent != "who_made" else self._h_artist_info
        elif p.intent in work_intents and not p.work_rows:
            handler = self._h_work_not_found
        elif handler is None:
            handler = self._h_unknown
        ans = handler(p, floor_filter=floor_filter, on_view_only=on_view_only)
        ans.intent = ans.intent or p.intent
        # Carry the conversation forward: keep old focus unless this answer sets a new one,
        # and remember the artist of the focused work so "they/them" resolves.
        new_ctx = {k: v for k, v in ans.context.items() if v is not None}
        if "work_row" in new_ctx and "artist_cid" not in new_ctx:
            cids = self.col.works.at[new_ctx["work_row"], "cids"]
            if cids:
                new_ctx["artist_cid"] = cids[0]
        ans.context = {**(context or {}), **new_ctx}
        if ans.intent != "disambiguate":
            ans.context.pop("choices", None)  # "None of these" only answers the question just asked
        # Remembered so "Not helpful" can offer a repair that fits the last answer.
        ans.context["last_intent"], ans.context["last_question"] = ans.intent, message
        return ans

    # ------------------------------------------------------------ conversation repair
    def not_helpful(self, context: dict | None) -> Answer:
        """The visitor marked the last answer as not helpful: apologise and offer a next step that
        fits what was asked (week 7 CUX redesign: repair, never repeat the same answer)."""
        context = dict(context or {})
        intent, question = context.get("last_intent", ""), context.get("last_question", "")
        if intent == "disambiguate" and context.get("choices"):
            return self._none_of_these(context)
        tips = {
            "search": "Try a few specific words, such as a subject, a colour, or a medium "
            "(*paintings of the sea*, *black and white photographs*), or name an artist.",
            "recommend": "Tell me what you didn't like, for example *I loved The Starry Night but not "
            "Water Lilies*, and I'll steer away from it.",
            "not_found": "It may not be in MoMA's collection. Check the title's spelling, or add the "
            "artist's name, for example *Flag by Jasper Johns*.",
        }
        work_tip = (
            "If you meant a different work with the same title, add the artist's name "
            "(*Water Lilies by Claude Monet*) or the year."
        )
        tip = tips.get(intent) or (work_tip if context.get("work_row") is not None else "")
        if not tip:
            tip = "Try naming the artwork's title, an artist, or a floor."
        text = f"Sorry that didn't help. {tip}"
        if question:
            link = f"{config.MOMA_HOME}/search/?q=" + "+".join(norm(question).split())
            text += f"\n\nYou can also [search moma.org]({link}) for *{question}*."
        return Answer(
            text=text,
            suggestions=["What can you do?", "What's on floor 5?", "Are you a person?"],
            intent="repair",
            context={**context, "last_intent": "repair"},
        )

    def _h_identity(self, p, **_):
        return Answer(
            text=(
                "I'm **AskMoMA**, an automated guide, not a person. I answer from MoMA's public collection "
                f"data ({len(self.col.on_view):,} works on view) and link every answer to moma.org. "
                "I don't keep your conversation after you close the page.\n\n"
                f"For anything I can't answer, MoMA's own staff can help: [Plan your visit]({config.MOMA_VISIT})."
            ),
            suggestions=["What can you do?", "Where is The Starry Night?"],
        )

    def _h_goodbye(self, p, **_):
        return Answer(
            text="You're welcome. Enjoy the museum! Ask me again any time you're looking for a work.",
            suggestions=["Where is The Starry Night?", "What are the museum hours?"],
        )

    # ------------------------------------------------------------ FAQ
    def _faq(self, key: str) -> Answer:
        f = FAQ[key]
        return Answer(
            text=f"{f['answer']}\n\nSource: [{f['source_title']}]({f['source']}) (checked {f['checked']}).",
            suggestions=f.get("suggestions", []),
        )

    def _h_faq_hours(self, p, **_):
        return self._faq("hours")

    def _h_faq_admission(self, p, **_):
        return self._faq("admission")

    def _h_faq_access(self, p, **_):
        return self._faq("accessibility")

    def _h_faq_address(self, p, **_):
        return self._faq("address")

    def _h_greeting(self, p, **_):
        n = len(self.col.on_view)
        return Answer(
            text=(
                "Hi! I'm AskMoMA. I can help you with the "
                f"{n:,} works currently listed as on view at MoMA. Try asking:\n\n"
                "- *Where is The Starry Night?*\n"
                "- *Who painted Water Lilies?*\n"
                "- *What works by Frida Kahlo are on view?*\n"
                "- *I loved The Starry Night. What else should I see?*\n"
                "- *Which artists have the most work on floor 5?*\n"
                "- *Find dreamy night sky paintings*\n"
                "- *What are the museum hours?*"
            ),
            suggestions=["Where is The Starry Night?", "What's on floor 5?", "What are the museum hours?"],
        )

    # ------------------------------------------------------------ artworks
    def _single(self, p: Parsed):
        row, alts = self._pick_work(p.work_rows)
        return row, (self._disambiguate(alts) if alts else None)

    def _work_answer(self, p: Parsed, body_fn) -> Answer:
        row, ambiguous = self._single(p)
        if ambiguous:
            return ambiguous
        c = self._card(row)
        first_cid = self.col.works.at[row, "cids"][0] if self.col.works.at[row, "cids"] else None
        text = body_fn(row, c) + self._as_of()
        return Answer(
            text=text,
            cards=[c],
            suggestions=[
                f"I loved {c['title']}. What else should I see?",
                f"What else by {c['artist'].split(', ')[0]} is on view?",
                f"When did MoMA acquire {c['title']}?",
            ],
            context={"work_row": row, "artist_cid": first_cid},
        )

    def _h_location(self, p, **_):
        def body(row, c):
            if c["on_view"]:
                return (
                    f"**{c['title']}** by {c['artist']} is on view on **{c['location']}**.\n\n"
                    f"{c['date']} · {c['medium']}\n\n[View it on moma.org]({c['link']})"
                )
            return (
                f"**{c['title']}** by {c['artist']} is in MoMA's collection but is **not listed as on view** "
                f"right now. [See its page on moma.org]({c['link']})."
            )

        return self._work_answer(p, body)

    def _h_who_made(self, p, **_):
        def body(row, c):
            w = self.col.works.loc[row]
            bio = _clean(w["ArtistBio"])  # MoMA's bios already carry brackets: "(French, 1840–1926)"
            bio = f" {bio}" if bio.startswith("(") else f" ({bio})" if bio else ""
            where = f" It's on view on {c['location']}." if c["on_view"] else ""
            return (
                f"**{c['title']}** was made by **{c['artist']}**{bio}, dated {c['date']}.{where}\n\n"
                f"[View it on moma.org]({c['link']})"
            )

        return self._work_answer(p, body)

    def _h_when_made(self, p, **_):
        return self._work_answer(
            p,
            lambda row, c: (
                f"**{c['title']}** by {c['artist']} is dated **{c['date']}**. [moma.org]({c['link']})"
            ),
        )

    def _h_medium(self, p, **_):
        return self._work_answer(
            p,
            lambda row, c: (
                f"**{c['title']}** is {c['medium'] or 'listed without a medium'}"
                f"{' (' + c['classification'] + ')' if c['classification'] else ''}. [moma.org]({c['link']})"
            ),
        )

    def _h_dimensions(self, p, **_):
        def body(row, c):
            dims = _clean(self.col.works.at[row, "Dimensions"], "not recorded")
            return f"**{c['title']}** measures {dims}. [moma.org]({c['link']})"

        return self._work_answer(p, body)

    def _h_acquired(self, p, **_):
        def body(row, c):
            w = self.col.works.loc[row]
            when = _clean(w["DateAcquired"], "on an unrecorded date")
            credit = _clean(w["CreditLine"])
            credit = f" Credit line: *{credit}*." if credit else ""
            return f"MoMA acquired **{c['title']}** on {when}.{credit} [moma.org]({c['link']})"

        return self._work_answer(p, body)

    def _h_work_info(self, p, **_):
        def body(row, c):
            w = self.col.works.loc[row]
            parts = [f"**{c['title']}**, {c['artist']}, {c['date']}"]
            for label, val in [
                ("Medium", c["medium"]),
                ("Dimensions", _clean(w["Dimensions"])),
                ("Department", _clean(w["Department"])),
                ("Acquired", _clean(w["DateAcquired"])),
                ("Where", c["location"] if c["on_view"] else "Not currently on view"),
            ]:
                if val:
                    parts.append(f"- {label}: {val}")
            parts.append(f"\n[View it on moma.org]({c['link']})")
            return "\n".join(parts)

        return self._work_answer(p, body)

    # ------------------------------------------------------------ recommendations
    def _h_recommend(self, p, floor_filter=None, **_):
        liked, disliked = split_preferences(self.col, p.text)
        if len(liked) >= 2 or (liked and disliked):
            return self._recommend_from_profile(liked, disliked, floor_filter)
        row, ambiguous = self._single(p)
        if ambiguous:
            return ambiguous
        works = self.col.works
        seed = works.loc[row]
        on_view = works[works["on_view"] & (works.index != row)]
        seed_vec = self.search.vector(row)
        picks: list[tuple[int, str]] = []
        seen, artists = set(), set()

        def add(rows, reason, limit, same_artist_ok=False):
            for r in rows:
                if len([1 for _, why in picks if why == reason]) >= limit:
                    return
                artist = works.at[r, "Artist"]
                # Variety: one work per artist, except when the reason is the artist.
                if r in seen or (artist in artists and not same_artist_ok):
                    continue
                seen.add(r)
                artists.add(artist)
                picks.append((r, reason))

        if seed["gallery"]:
            same_gallery = on_view[on_view["gallery"] == seed["gallery"]].index
            add(self._by_similarity(seed_vec, same_gallery), f"In the same gallery ({seed['gallery']})", 3)
        for cid in seed["cids"]:
            add(
                [r for r in self.col.works_by_artist.get(cid, []) if r in on_view.index],
                f"Also by {self._artist_name(cid)}",
                2,
                same_artist_ok=True,
            )
        if seed["year"]:
            near = on_view[(on_view["year"] - seed["year"]).abs() <= 10]
            if _clean(seed["Classification"]):
                near = near[near["Classification"] == seed["Classification"]]
            add(
                self._by_similarity(seed_vec, near.index),
                f"Same period ({int(seed['year']) - 10}–{int(seed['year']) + 10})",
                2,
            )
        for r, _score in self.search.similar_to(row, k=12):
            if r in on_view.index:
                add([r], "Similar in subject or style", 2)

        if floor_filter:
            picks = [(r, why) for r, why in picks if works.at[r, "floor"] == floor_filter] or picks
        picks = picks[: config.MAX_LIST]
        c = self._card(row)
        if not picks:
            return Answer(
                text=f"I couldn't find related works on view for **{c['title']}** right now. "
                f"[Browse what's on view]({config.MOMA_HOME}/collection/?on_view=1)" + self._as_of(),
                cards=[c],
                context={"work_row": row},
            )
        return self._recommendation_answer(f"If you liked **{c['title']}**, try these:", picks, [row], row)

    def _by_similarity(self, vec, rows) -> list[int]:
        """``rows`` ordered by similarity to ``vec`` (unchanged if there is no vector)."""
        if vec is None:
            return list(rows)
        scored = [(float(v @ vec) if (v := self.search.vector(r)) is not None else 0.0, r) for r in rows]
        return [r for _, r in sorted(scored, key=lambda x: -x[0])]

    def _recommend_from_profile(self, liked_rows, disliked_rows, floor_filter=None) -> Answer:
        """Several likes and dislikes, as in the week 11 recommender: a Rocchio profile
        (mean of liked works minus half the mean of disliked ones), then maximal
        marginal relevance so the list is not ten near-copies."""
        works = self.col.works

        def one(rows):  # the work on view, if the title is shared
            return next((r for r in rows if works.at[r, "on_view"]), rows[0])

        liked, disliked = [one(r) for r in liked_rows], [one(r) for r in disliked_rows]
        vecs = {r: self.search.vector(r) for r in liked + disliked}
        liked_v = [vecs[r] for r in liked if vecs[r] is not None]
        if not liked_v:
            return self._h_unknown(Parsed(text="", intent="unknown"))
        profile = np.mean(liked_v, axis=0)
        disliked_v = [vecs[r] for r in disliked if vecs[r] is not None]
        if disliked_v:
            profile = profile - DISLIKE_WEIGHT * np.mean(disliked_v, axis=0)
        profile = profile / (np.linalg.norm(profile) or 1.0)

        # Leave out the works named, and other panels of them (Monet made several "Water Lilies").
        named = {(works.at[r, "title_norm"], works.at[r, "Artist"]) for r in liked + disliked}
        same = works["title_norm"].isin({t for t, _ in named}) & works["Artist"].isin({a for _, a in named})
        exclude = set(liked) | set(disliked) | set(works.index[same])
        candidates = [
            (r, s) for r, s in self.search.rank(profile, k=40, exclude=exclude) if works.at[r, "on_view"]
        ]
        if floor_filter:
            candidates = [(r, s) for r, s in candidates if works.at[r, "floor"] == floor_filter] or candidates
        chosen = self._mmr(candidates, n=6)

        picks = []
        for r in chosen:
            v = self.search.vector(r)
            closest = max(liked, key=lambda lr: float(v @ vecs[lr]) if vecs[lr] is not None else -1.0)
            picks.append((r, f"Close to {self._card(closest)['title']}"))
        names = [f"**{self._card(r)['title']}**" for r in liked]
        intro = "Because you liked " + (
            " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + f", and {names[-1]}"
        )
        if disliked:
            intro += ", but not " + " or ".join(f"**{self._card(r)['title']}**" for r in disliked)
        if not picks:
            return Answer(text=intro + ", I couldn't find other works on view to suggest." + self._as_of())
        return self._recommendation_answer(intro + ", try these:", picks, liked, liked[0], disliked)

    def _mmr(self, candidates: list[tuple[int, float]], n: int) -> list[int]:
        """Maximal marginal relevance: each pick balances relevance against similarity to earlier picks."""
        vecs = {r: self.search.vector(r) for r, _ in candidates}
        chosen: list[int] = []
        pool = [(r, s) for r, s in candidates if vecs[r] is not None]
        while pool and len(chosen) < n:

            def gain(item):
                r, s = item
                overlap = max((float(vecs[r] @ vecs[c]) for c in chosen), default=0.0)
                return MMR_LAMBDA * s - (1 - MMR_LAMBDA) * overlap

            best = max(pool, key=gain)
            chosen.append(best[0])
            pool.remove(best)
        return chosen

    def _recommendation_answer(self, intro, picks, liked, focus, disliked=()) -> Answer:
        lines = "\n".join(f"- {self._work_line(r)} — *{why}*" for r, why in picks)
        first = self._card(picks[0][0])["title"]
        seed = " and ".join(self._card(r)["title"] for r in liked[:2])
        return Answer(
            text=f"{intro}\n\n{lines}" + self._as_of(),
            cards=[self._card(r, why) for r, why in picks],
            # Feedback, as in the week 11 recommender: refine the list from here.
            suggestions=[f"Where is {first}?", f"More like {first}", f"I loved {seed} but not {first}"],
            context={
                "work_row": focus,
                "liked": [int(r) for r in liked],
                "disliked": [int(r) for r in disliked],
            },
        )

    # ------------------------------------------------------------ artists
    def _h_artist_info(self, p, **_):
        cid = p.artist_cids[0]
        a = self.col.artist(cid) or {}
        name = _clean(a.get("DisplayName"), "This artist")
        rows = self.col.works_by_artist.get(cid, [])
        on_view = [r for r in rows if self.col.works.at[r, "on_view"]]
        bio = _clean(a.get("ArtistBio"))
        nat = _clean(a.get("Nationality"))
        life = ""
        b, e = _clean(a.get("BeginDate")), _clean(a.get("EndDate"))
        if b and b != "0":
            life = f"born {b}" + (f", died {e}" if e and e != "0" else "")
        facts = ", ".join(x for x in [nat, life] if x) or bio
        text = (
            f"**{name}**{f' ({facts})' if facts else ''}. MoMA's collection lists "
            f"**{len(rows):,}** work{'s' if len(rows) != 1 else ''} by {name}, "
            f"**{len(on_view)}** currently on view."
        )
        if on_view:
            text += "\n\nOn view now:\n" + "\n".join(
                f"- {self._work_line(r)}" for r in on_view[: config.MAX_LIST]
            )
        text += f"\n\n[{name} on moma.org]({config.ARTIST_URL.format(constituent_id=cid)})" + self._as_of()
        return Answer(
            text=text,
            cards=[self._card(r) for r in (on_view or rows)[: config.MAX_LIST]],
            suggestions=[f"What works by {name} are on view?"]
            + ([f"Where is {self._card(on_view[0])['title']}?"] if on_view else []),
            context={"artist_cid": cid, "work_row": on_view[0] if on_view else None},
        )

    def _h_works_by_artist(self, p, floor_filter=None, on_view_only=True, **_):
        if not p.artist_cids:
            return self._h_search(p, floor_filter=floor_filter, on_view_only=on_view_only)
        cid = p.artist_cids[0]
        name = self._artist_name(cid)
        rows = self.col.works_by_artist.get(cid, [])
        df = self._filter(self.col.works.loc[rows], p.floor or floor_filter, on_view_only)
        if df.empty:
            where = f" on floor {p.floor or floor_filter}" if (p.floor or floor_filter) else ""
            total = len(rows)
            return Answer(
                text=(
                    f"None of the {total:,} works by **{name}** in MoMA's collection are listed as on view{where} "
                    f"right now. [See all of {name}'s works on moma.org]"
                    f"({config.ARTIST_URL.format(constituent_id=cid)})" + self._as_of()
                ),
                suggestions=[f"Tell me about {name}"],
                context={"artist_cid": cid},
            )
        lines = "\n".join(f"- {self._work_line(r)}" for r in df.index[: config.MAX_LIST])
        more = f"\n\n…and {len(df) - config.MAX_LIST} more." if len(df) > config.MAX_LIST else ""
        scope = "on view" if on_view_only else "in the collection"
        return Answer(
            text=f"**{name}** — {len(df)} work{'s' if len(df) != 1 else ''} {scope}:\n\n{lines}{more}"
            + self._as_of(),
            cards=[self._card(r) for r in df.index[: config.MAX_LIST]],
            suggestions=[f"Tell me about {name}", f"Where is {self._card(df.index[0])['title']}?"],
            context={"artist_cid": cid, "work_row": int(df.index[0])},
        )

    def _artist_url(self, name: str) -> str:
        """MoMA's artist page (/artists/{id}-{slug}), or a MoMA search if the ID is unknown."""
        cid = self.col.artist_index.get(norm(name))
        if not cid:
            return config.ARTIST_SEARCH_URL.format(query=name.replace(" ", "+"))
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
        return config.ARTIST_PAGE_URL.format(cid=cid, slug=slug)

    def _artist_link(self, name: str, floor: int | None) -> Answer:
        works = self.col.on_view
        df = works[works["Artist"].str.contains(name, regex=False, na=False)]
        if floor:
            df = df[df["floor"] == floor]
        where = f" on floor {floor}" if floor else " on view"
        count = f"**{name}** has {len(df)} work{'s' if len(df) != 1 else ''}{where}."
        link = self._artist_url(name)
        return Answer(
            text=f"{count}\n\nTo know more about this artist, visit [{name}]({link})." + self._as_of(),
            suggestions=[f"Which artists have the most work on floor {floor}?"] if floor else [],
        )

    def _h_top_artists(self, p, floor_filter=None, **_):
        floor = p.floor or floor_filter
        df = self._filter(self.col.works, floor, True)
        counts = df.explode("cids")["cids"].dropna().value_counts().head(config.TOP_ARTISTS)
        if counts.empty:
            return Answer(
                text="I couldn't find any works listed as on view"
                + (f" on floor {floor}" if floor else "")
                + "."
            )
        scope = f"on floor {floor}" if floor else "across the museum"
        lines = "\n".join(
            f"{i}. {self._artist_name(cid)} — {n} work{'s' if n != 1 else ''}"
            for i, (cid, n) in enumerate(counts.items(), 1)
        )
        return Answer(
            text=f"Artists with the most works on view {scope}:\n\n{lines}" + self._as_of(),
            suggestions=[
                f"MoMA link for {self._artist_name(cid)}" + (f" on floor {floor}" if floor else "")
                for cid in counts.index
            ],
            context={"artist_cid": counts.index[0]},
        )

    def _h_nationality(self, p, floor_filter=None, **_):
        nat = p.nationality
        artists = self.col.artists
        cids = set(artists.index[artists["Nationality"] == nat]) if "Nationality" in artists else set()
        df = self._filter(self.col.works, p.floor or floor_filter, True)
        df = df[df["cids"].map(lambda ids: bool(cids.intersection(ids)))]
        if df.empty:
            return Answer(text=f"I couldn't find works by {nat} artists on view right now." + self._as_of())
        counts = df.explode("cids")
        counts = counts[counts["cids"].isin(cids)]["cids"].value_counts().head(config.MAX_LIST)
        lines = "\n".join(f"- {self._artist_name(c)} — {n} on view" for c, n in counts.items())
        return Answer(
            text=f"**{nat}** artists with work on view ({len(counts)} shown):\n\n{lines}" + self._as_of(),
            cards=[self._card(r) for r in df.index[: config.MAX_LIST]],
            suggestions=[f"Tell me about {self._artist_name(counts.index[0])}"],
        )

    # ------------------------------------------------------------ places
    def _h_floor_list(self, p, on_view_only=True, **_):
        works = self.col.on_view
        if p.gallery:
            df, scope = works[works["gallery"] == p.gallery], f"Gallery {p.gallery}"
        elif p.floor:
            df, scope = works[works["floor"] == p.floor], f"Floor {p.floor}"
        else:
            floors = ", ".join(str(f) for f in self.col.floors())
            return Answer(
                text=f"Which floor? Works are on view on floors {floors}.",
                suggestions=[f"What's on floor {f}?" for f in self.col.floors()[:3]],
            )
        if df.empty:
            return Answer(text=f"I don't have any works listed on {scope}." + self._as_of())
        galleries = df["gallery"].dropna().value_counts()
        lines = "\n".join(
            f"- {self._work_line(r, with_location=bool(p.floor))}" for r in df.index[: config.MAX_LIST]
        )
        gal = ""
        if p.floor and len(galleries):
            gal = "\n\nGalleries on this floor: " + ", ".join(sorted(galleries.index))
        if p.floor:
            n_artists = df.explode("cids")["cids"].dropna().nunique()
            return Answer(
                text=f"Floor {p.floor}: **{len(df)}** works by **{n_artists}** artists.\n\n"
                "Search an artist, or see the top 5?" + self._as_of(),
                suggestions=[f"Which artists have the most work on floor {p.floor}?"],
            )
        return Answer(
            text=f"**{len(df)}** works are on view on {scope}. A few of them:\n\n{lines}{gal}"
            + self._as_of(),
            cards=[self._card(r) for r in df.index[: config.MAX_LIST]],
            suggestions=[f"Which artists have the most work on floor {int(df['floor'].iloc[0])}?"],
        )

    # ------------------------------------------------------------ search
    def _h_search(self, p, floor_filter=None, on_view_only=True, **_):
        query = p.residue or p.text
        hits = self.search.query(query, k=12)
        works = self.col.works
        if floor_filter:
            hits = [(r, s) for r, s in hits if works.at[r, "floor"] == floor_filter] or hits
        hits = hits[:6]
        if not hits:
            return self._h_unknown(p)
        lines = "\n".join(f"- {self._work_line(r)}" for r, _ in hits)
        return Answer(
            text=f"Here are works on view that match **“{query}”**:\n\n{lines}"
            "\n\n_Matches are by meaning, so check each link to confirm it's what you're after._"
            + self._as_of(),
            cards=[self._card(r) for r, _ in hits],
            suggestions=[
                f"Tell me about {self._card(hits[0][0])['title']}",
                f"I loved {self._card(hits[0][0])['title']}. What else should I see?",
            ],
            intent="search",
            context={"work_row": hits[0][0]},
        )

    def _h_work_not_found(self, p, **kw):
        """A question about a work that is not in the data: say so, then offer close matches."""
        if DESCRIPTIVE.search(p.text.lower()):  # "where can I see paintings of the sea?"
            return self._h_search(p, **kw)
        asked = p.residue or p.text
        text = (
            f"I couldn't find a work called **“{asked}”** in MoMA's collection data. "
            f"It may not be in MoMA's collection; you can check on [moma.org]({config.MOMA_HOME}/collection/)."
        )
        hits = self.search.query(asked, k=4, min_score=0.35)
        if hits:
            lines = "\n".join(f"- {self._work_line(r)}" for r, _ in hits)
            text += f"\n\nWorks on view that may be related:\n\n{lines}"
        return Answer(
            text=text + self._as_of(),
            cards=[self._card(r) for r, _ in hits],
            suggestions=["What can you do?", "What's on floor 5?"],
            intent="not_found",
        )

    def _h_unknown(self, p, **kw):
        if p.residue and p.intent == "unknown":
            hits = self.search.query(p.residue, k=6, min_score=0.35)
            if hits:
                return self._h_search(p, **kw)
        return Answer(
            text=(
                "I couldn't find an answer to that in MoMA's collection data. "
                f"You can search directly on [moma.org]({config.MOMA_HOME}/collection/), "
                "or ask me about a specific artwork, artist, floor, or your visit."
            ),
            suggestions=["What can you do?", "What's on floor 5?", "What are the museum hours?"],
            intent="not_found",
        )
