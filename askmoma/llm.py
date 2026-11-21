"""Optional small-LLM layer (off by default).

When ASKMOMA_LLM names a model (e.g. HuggingFaceTB/SmolLM2-360M-Instruct or
Qwen/Qwen2.5-1.5B-Instruct), the templated answer is reworded into friendlier
prose. Install the ``llm`` extra first: ``uv sync --extra llm``.

The model never decides facts, and never handles links:

* It sees only the plain sentences of the answer the engine already built.
  Links, "Source:" lines and the "data as of" line are set aside and put back
  unchanged after the rewrite, so a link cannot be dropped or altered.
* Its rewrite is used only if it passes ``check``: the same numbers (gallery,
  floor, date, price), every named title or person kept, no link added, and no
  echo of the prompt. Otherwise, or if the model fails, the template is used.
* Answers that are lists are not reworded.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from . import config

log = logging.getLogger(__name__)

# The facts go in the system message and the visitor's question is the user turn,
# so a small model answers the question instead of echoing a "Question/FACTS" form.
SYSTEM = (
    "You are AskMoMA, a friendly guide at the Museum of Modern Art. Answer the visitor in "
    "one or two natural sentences using only these facts. Keep every name, number, and date "
    "exactly as written, and add nothing.\n\n"
    "Facts:\n{facts}"
)
MAX_NEW_TOKENS = 220

_LINK = re.compile(r"\]\((https?://[^)\s]+)\)")
_TRAILING_LINK = re.compile(r"\s*\[[^\]]+\]\(https?://[^)\s]+\)\.?\s*$")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_LIST_LINE = re.compile(r"^\s*(?:[-•]|\d+\.)\s", re.M)
_FOOTER = re.compile(r"\s*_Collection data as of [^_\n]+_\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ECHO = re.compile(r"^\s*(?:question|facts?)\s*:", re.I | re.M)

_lock = threading.Lock()
_pipe = None
_failed_model: str | None = None  # a model that failed to load is not retried


def load():
    """Load the configured model once and return the pipeline, or None if off or unavailable."""
    global _pipe, _failed_model
    model = config.LLM_MODEL
    if not model or model == _failed_model:
        return None
    with _lock:
        if _pipe is None:
            try:
                _pipe = _build_pipeline(model)
            except Exception as exc:  # missing extra, no network, out of memory, ...
                log.warning("LLM %s unavailable (%s); using templated answers.", model, exc)
                _failed_model = model
                return None
    return _pipe


def _build_pipeline(model: str):
    import torch
    from transformers import pipeline

    # An explicit device, not device_map="auto", which would also need `accelerate`.
    cuda = torch.cuda.is_available()
    started = time.perf_counter()
    pipe = pipeline(
        "text-generation",
        model=model,
        device=0 if cuda else -1,
        dtype="auto" if cuda else torch.float32,
    )
    log.info("LLM %s loaded on %s in %.1fs", model, "cuda" if cuda else "cpu", time.perf_counter() - started)
    return pipe


def reset() -> None:
    """Forget the loaded model (used by tests and when the setting changes)."""
    global _pipe, _failed_model
    with _lock:
        _pipe, _failed_model = None, None


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def check(facts: str, rewrite: str) -> str | None:
    """Return why ``rewrite`` may not replace ``facts``, or None if it is safe to use."""
    if not rewrite.strip():
        return "empty reply"
    if _ECHO.search(rewrite):
        return "echoed the prompt"
    fact_links, new_links = set(_LINK.findall(facts)), set(_LINK.findall(rewrite))
    if missing := fact_links - new_links:
        return f"dropped link {sorted(missing)[0]}"
    if invented := new_links - fact_links:
        return f"added link {sorted(invented)[0]}"
    if added := _numbers(rewrite) - _numbers(facts):
        return f"added number {sorted(added)[0]}"
    if dropped := _numbers(facts) - _numbers(rewrite):
        return f"dropped number {sorted(dropped)[0]}"
    for term in _BOLD.findall(facts):  # titles and names; numbers are checked above
        if not any(c.isdigit() for c in term) and term.lower() not in rewrite.lower():
            return f"dropped {term!r}"
    return None


def split(body: str) -> tuple[str, str]:
    """Split an answer into (prose the model may reword, link text kept verbatim).

    A link at the end of a paragraph is set aside on its own; a paragraph with a
    link anywhere else (such as "Source: [...](...) (checked ...)") is kept whole.
    """
    prose, kept = [], []
    for para in (p.strip() for p in body.split("\n\n")):
        trailing = _TRAILING_LINK.search(para)
        head = para[: trailing.start()].strip() if trailing else para
        if _LINK.search(head):
            kept.append(para)
            continue
        if head:
            prose.append(head)
        if trailing:
            kept.append(para[trailing.start() :].strip())
    return "\n\n".join(prose), "\n\n".join(kept)


def polish(question: str, facts: str) -> str:
    """Reword ``facts`` with the configured model if that is on and safe; otherwise return them."""
    if not config.LLM_MODEL or _LIST_LINE.search(facts):
        return facts
    footer = _FOOTER.search(facts)
    body = facts[: footer.start()] if footer else facts
    prose, kept = split(body)
    if not prose:
        return facts
    pipe = load()
    if pipe is None:
        return facts
    messages = [
        {"role": "system", "content": SYSTEM.format(facts=prose)},
        {"role": "user", "content": question},
    ]
    started = time.perf_counter()
    try:
        with _lock:
            out = pipe(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        rewrite = out[0]["generated_text"][-1]["content"].strip()
    except Exception as exc:
        log.warning("LLM generation failed (%s); using the templated answer.", exc)
        return facts
    reason = check(prose, rewrite)
    log.info("LLM reply in %.1fs: %s", time.perf_counter() - started, reason or "accepted")
    if reason:
        return facts
    return rewrite + (f"\n\n{kept}" if kept else "") + (facts[footer.start() :] if footer else "")
