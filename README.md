# AskMoMA

A conversational guide to the art on view at the Museum of Modern Art (MoMA): where a work is, which related works to see next, and answers to common visitor questions. Every answer links to the museum's own page so that visitors can verify it.

Group 4 project for MSAI-631-M80, *Artificial Intelligence for Human-Computer Interaction*, University of the Cumberlands, Fall 2026.

**Team:** Dipu Biswas, Heman Chongbang Limbu, Sarina Tamang, Soham Nitin Ghorpade, Xinran Peng

## Status

| Use case | Example | Status |
| --- | --- | --- |
| 1. Locate a work | *Where is The Starry Night?* → MoMA, Floor 5, Gallery 501, with a link to the moma.org page | **Working** |
| 2. Rank the artists on view, for the museum or one floor | *Who are the top artists on Floor 5?* | To do |
| 3. Recommend related works (same gallery, artist, or period) | *I loved The Starry Night* | To do |
| 4. Search by description (sentence embeddings) | *dreamy night-sky paintings* | To do |
| Visitor FAQ (hours, admission, accessibility), written from moma.org with source links | *What time does MoMA open?* | To do |

Optional after the core: a small open language model (Qwen2.5-1.5B-Instruct or SmolLM2-360M-Instruct) to phrase answers and read complex requests, answering only from the retrieved records. No OpenAI or paid services.

## Running it

Python 3.10 or later.

```bash
pip install -r requirements.txt
python scripts/fetch_data.py
python app.py
```

`fetch_data.py` downloads MoMA's open data (about 75 MB, kept out of Git in `data/raw/`) and writes `data/on_view.csv`, the 1,556 works currently listed as on view. `data/on_view.csv` is already committed, so the app also runs without this step.

Tests:

```bash
python -m pytest tests -q
```

## Layout

| Path | Contents |
| --- | --- |
| `app.py` | The Gradio chat page |
| `src/askmoma/locate.py` | Use case 1: finding a work and its location |
| `scripts/fetch_data.py` | Downloads MoMA's data and builds `data/on_view.csv` |
| `data/SOURCE.md` | Which version of MoMA's data is used, and when it was downloaded |
| `docs/design-notes.md` | Open questions and design decisions, recorded as the team researches them |
| `tests/` | Tests, run with pytest |

## Working together

- Work on a branch named after the feature, such as `feature/rank-artists`, and open a pull request into `main`.
- Another member reviews and merges each pull request.
- Run the tests before pushing.
- Never commit API keys or passwords; none are needed.

## Data and credits

- Collection data: The Museum of Modern Art (MoMA) collection data, https://github.com/MuseumofModernArt/collection, CC0 1.0. Gallery locations change, so each answer links to moma.org for the latest.
- Interface: Gradio (Abid et al., 2019).
- AskMoMA is an independent course project. It is not affiliated with or endorsed by The Museum of Modern Art.
