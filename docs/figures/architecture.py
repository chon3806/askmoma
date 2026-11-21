"""Draw docs/figures/architecture.png, the one system diagram (README and design document).

    uv run python docs/figures/architecture.py

It shows the modules in askmoma/ and app.py as they are on main. Update it
when a module is added or renamed, so the README never shows an old design.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_suffix(".png")
INK, LINE = "#1f2937", "#475569"
FILL = {"ui": "#dbeafe", "logic": "#e0e7ff", "scenario": "#ecfccb", "data": "#fef3c7", "model": "#fce7f3"}


def box(ax, x, y, w, h, title, body="", kind="logic", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=FILL[kind], edgecolor=LINE, linewidth=1.2,
                                linestyle="--" if dashed else "-"))  # fmt: skip
    if body:
        ax.text(
            x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=9, fontweight="bold", color=INK
        )
        ax.text(
            x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=7.4, color=INK, linespacing=1.35
        )
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10, color=INK)


def arrow(ax, start, end, label="", dashed=False, offset=(0, 0.08)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=LINE,
                                 linewidth=1.2, linestyle="--" if dashed else "-"))  # fmt: skip
    if label:
        x, y = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
        ax.text(x, y, label, ha="center", fontsize=7.2, color=LINE)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    # Top row: the conversation path.
    box(ax, 0.1, 3.75, 1.4, 0.9, "Visitor")
    box(ax, 2.1, 3.5, 2.5, 1.35, "app.py", "Gradio web page:\nanswer cards, suggestion\nchips, floor buttons\n(also a terminal mode)", "ui")  # fmt: skip
    box(ax, 5.3, 3.5, 2.6, 1.35, "engine.py  (AskMoMA)", "one handler per scenario;\ncarries context to the next\nturn for follow-ups", "logic")  # fmt: skip
    box(ax, 8.6, 3.5, 2.3, 1.35, "nlu.py", "rule-based intents; finds\nthe title or artist named,\ntolerating typos", "logic")  # fmt: skip
    arrow(ax, (1.5, 4.4), (2.1, 4.4), "question")
    arrow(ax, (2.1, 3.95), (1.5, 3.95))
    ax.text(
        0.8,
        3.6,
        "answer + moma.org link\n+ date of the data",
        ha="center",
        va="top",
        fontsize=7.2,
        color=LINE,
    )
    arrow(ax, (4.6, 4.2), (5.3, 4.2))
    arrow(ax, (7.9, 4.4), (8.6, 4.4), "parse")
    arrow(ax, (8.6, 3.95), (7.9, 3.95), "intent,\nwork/artist", offset=(0, -0.38))

    # Middle: what the handlers answer.
    box(ax, 0.1, 1.75, 10.8, 1.15, "Scenarios (handlers in engine.py)",
        "Works: where is it? · who made it? · when, medium, size · when acquired      "
        "Artists: works by an artist · about an artist · top artists · by nationality\n"
        "Places: what's on a floor or in a gallery      Discovery: recommendations (same gallery, same artist, similar) · search by description\n"
        "Visit: hours, admission, accessibility, address (FAQ)      "
        "Conversation: follow-ups (\"where is it?\") · same title, many works → asks which · no answer → says so, links moma.org",
        "scenario")  # fmt: skip
    arrow(ax, (6.6, 3.5), (6.6, 2.9))

    # Bottom: where the facts come from.
    box(ax, 0.1, 0.2, 3.0, 1.2, "data.py  (Collection)", "MoMA's Artworks.csv + Artists.csv\n(CC0, fetched by download.py)\nOnView → floor and gallery; indexes", "data")  # fmt: skip
    box(ax, 3.4, 0.2, 2.5, 1.2, "search.py  (Searcher)", "all-MiniLM-L6-v2 embeddings\n+ FAISS; TF-IDF fallback\nwhen they are not installed", "model")  # fmt: skip
    box(ax, 6.2, 0.2, 2.0, 1.2, "faq.json", "visitor answers written\nfrom moma.org, with\nsource and date checked", "data")  # fmt: skip
    box(ax, 8.5, 0.2, 2.4, 1.2, "llm.py  (optional)", "off by default; Qwen2.5-1.5B\nor SmolLM2-360M rewords the\nretrieved facts, adds none", "model", dashed=True)  # fmt: skip
    for x in (1.6, 4.65, 7.2):
        arrow(ax, (x, 1.4), (x, 1.75))
    arrow(ax, (9.7, 1.75), (9.7, 1.4), dashed=True)

    ax.text(10.9, 0.02, "Dashed: optional, off by default", fontsize=7.2, color=LINE, ha="right")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
