"""Optional small-LLM layer (off by default).

When ASKMOMA_LLM names a model (e.g. Qwen/Qwen2.5-1.5B-Instruct or
HuggingFaceTB/SmolLM2-360M-Instruct), the templated answer is reworded into
friendlier prose. The model only sees the facts we already retrieved and is
told not to add any; if it fails or drops a link, we keep the template.
"""

from __future__ import annotations

import logging
import re

from . import config

log = logging.getLogger(__name__)
_pipe = None
_failed = False

SYSTEM = (
    "You are AskMoMA, a friendly museum guide. Rewrite the FACTS as a short, natural reply "
    "to the visitor's question. Use only the facts given. Do not add artworks, dates, places, "
    "or prices. Keep every markdown link exactly as written. Keep it under 120 words."
)


def _load():
    global _pipe, _failed
    if _pipe is not None or _failed or not config.LLM_MODEL:
        return _pipe
    try:
        from transformers import pipeline

        _pipe = pipeline("text-generation", model=config.LLM_MODEL, device_map="auto")
    except Exception as exc:
        log.warning("LLM %s unavailable (%s); using templated answers.", config.LLM_MODEL, exc)
        _failed = True
    return _pipe


def polish(question: str, facts: str) -> str:
    pipe = _load()
    if pipe is None:
        return facts
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nFACTS:\n{facts}"},
    ]
    try:
        out = pipe(messages, max_new_tokens=220, do_sample=False)[0]["generated_text"][-1]["content"]
    except Exception as exc:
        log.warning("LLM generation failed (%s)", exc)
        return facts
    # Guardrail: every link in the facts must survive, or we fall back.
    links = set(re.findall(r"\]\((https?://[^)]+)\)", facts))
    if not links.issubset(set(re.findall(r"\]\((https?://[^)]+)\)", out))):
        return facts
    return out.strip()
