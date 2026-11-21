"""Draw docs/figures/architecture.png, the system diagram used in the design document.

uv run python docs/figures/architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_suffix(".png")
INK, LINE = "#1f2937", "#475569"
FILL = {"ui": "#dbeafe", "logic": "#e0e7ff", "feature": "#ecfccb", "todo": "#f1f5f9",
        "data": "#fef3c7", "model": "#fce7f3"}  # fmt: skip


def box(ax, x, y, w, h, text, kind, dashed=False, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=FILL[kind], edgecolor=LINE, linewidth=1.2,
                                linestyle="--" if dashed else "-"))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=LINE,
                                 linewidth=1.2, linestyle="--" if dashed else "-"))  # fmt: skip


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Top row: the conversation path, and the optional language model beside the router.
    box(ax, 0.1, 4.0, 1.5, 0.8, "Visitor", "ui", size=10)
    box(ax, 2.4, 4.0, 1.9, 0.8, "Chat page\n(Gradio)", "ui", size=10)
    box(ax, 5.0, 4.0, 2.1, 0.8, "Router\nwhich feature answers?", "logic")
    box(ax, 7.8, 4.0, 2.1, 0.8, "Optional small LLM\nQwen2.5-1.5B / SmolLM2\nphrases retrieved records",
        "model", dashed=True, size=7.5)  # fmt: skip
    arrow(ax, (1.6, 4.55), (2.4, 4.55))
    arrow(ax, (2.4, 4.25), (1.6, 4.25))
    ax.text(2.0, 4.95, "question", fontsize=7.5, color=LINE, ha="center")
    ax.text(2.0, 3.75, "answer + moma.org link", fontsize=7.5, color=LINE, ha="center")
    arrow(ax, (4.3, 4.4), (5.0, 4.4))
    arrow(ax, (7.1, 4.4), (7.8, 4.4), dashed=True)

    # Middle row: one module per use case. Solid = built, dashed = planned.
    features = [("1. Locate a work\n(working)", "feature", False), ("2. Rank artists\non view", "todo", True),
                ("3. Recommend\nrelated works", "todo", True), ("4. Search by\ndescription", "todo", True),
                ("Visitor\nFAQ", "todo", True)]  # fmt: skip
    for i, (label, kind, dashed) in enumerate(features):
        x = 0.3 + i * 1.95
        box(ax, x, 2.1, 1.7, 0.9, label, kind, dashed=dashed, size=8.5)
        arrow(ax, (6.05, 4.0), (x + 0.85, 3.0), dashed=dashed)

    # Bottom row: what the features read.
    box(ax, 0.3, 0.2, 3.6, 1.1, "Collection: data/on_view.csv\n1,556 works on view: title, artist, date,\n"
        "medium, venue, floor, gallery, moma.org link\n(built by fetch.py from MoMA's open data, CC0)",
        "data", size=7.5)  # fmt: skip
    box(ax, 4.4, 0.2, 2.6, 1.1, "Embeddings\nall-MiniLM-L6-v2 + FAISS\n(use cases 3 and 4)", "model",
        dashed=True, size=8)  # fmt: skip
    box(ax, 7.5, 0.2, 2.4, 1.1, "FAQ entries\nwritten by the team from\nmoma.org, with source and date",
        "data", dashed=True, size=7.5)  # fmt: skip
    for i in range(4):
        arrow(ax, (2.1, 1.3), (0.3 + i * 1.95 + 0.85, 2.1), dashed=i > 0)
    arrow(ax, (5.7, 1.3), (4.9, 2.1), dashed=True)
    arrow(ax, (5.7, 1.3), (6.75, 2.1), dashed=True)
    arrow(ax, (8.7, 1.3), (8.7, 2.1), dashed=True)

    ax.text(9.95, 3.3, "Solid: built.  Dashed: planned.", fontsize=7.5, color=LINE, ha="right")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
