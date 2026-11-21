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
| 2026-11-21 | Use MoMA's official GitHub data, not the Kaggle copy | Official, CC0, and updated 2026-11-17; the Kaggle copy is an older snapshot | https://github.com/MuseumofModernArt/collection |
| 2026-11-21 | Answers come only from the data, with a moma.org link | A location must be traceable; nothing is generated | Proposal, "Scope, Risks, and Evaluation" |
