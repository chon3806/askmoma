# Design notes

Decisions the team researches and records as the project goes. Each entry says what was decided, why, and the source.

## Open questions

- **Language model**: does a small open model add enough to the templated answers? Candidates: Qwen2.5-1.5B-Instruct, SmolLM2-360M-Instruct. Check speed on a typical laptop before committing to one.
- **Embeddings for use cases 3 and 4**: all-MiniLM-L6-v2 is the starting point. Decide what text to embed per work (title, artist, medium, date?).
- **"Top artists"**: rank by number of works on view, or by paintings only? One large photo series can dominate a count.
- **Visitor FAQ**: which questions, and which moma.org page each answer comes from (record the date checked).

## Decisions

| Date | Decision | Why | Source |
| --- | --- | --- | --- |
| 2026-11-21 | Use MoMA's official GitHub data, not the Kaggle copy | Same two files (Artworks.csv, Artists.csv) and the same CC0 license, but the official repository was updated 2026-11-17, while Kaggle's `momanyc/museum-collection` was last updated 2017-02-15 (version 1), so its gallery locations are out of date | https://github.com/MuseumofModernArt/collection ; https://www.kaggle.com/datasets/momanyc/museum-collection |
| 2026-11-21 | Commit only `data/on_view.csv`; download the full files on demand | Artworks.csv is about 70 MB, which would slow every clone; the on-view table is 0.6 MB and rebuilt by `mise run data` | `src/askmoma/data/fetch.py` |
| 2026-11-21 | Python 3.12 with uv and mise | One pinned interpreter and a lock file, so every member runs the same versions | `mise.toml`, `uv.lock` |
| 2026-11-21 | Answers come only from the data, with a moma.org link | A location must be traceable; nothing is generated | Proposal, "Scope, Risks, and Evaluation" |
