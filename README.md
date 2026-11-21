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

The model and the detailed design are being researched as the project goes; see [docs/design-notes.md](docs/design-notes.md).

## Getting started

The toolchain is pinned with [mise](https://mise.jdx.dev) (Python 3.12 and [uv](https://docs.astral.sh/uv/)), and the libraries are locked in `uv.lock`, so everyone runs the same versions.

```bash
mise install        # Python 3.12 and uv
mise run setup      # install AskMoMA and the dev tools into .venv
mise run app        # start the chat page, then open the address it prints
```

| Task | What it does |
| --- | --- |
| `mise run setup` | Install the project and dev tools from `uv.lock` |
| `mise run app` | Start the chat page |
| `mise run test` | Run the tests |
| `mise run lint` | Check style and common errors (ruff) |
| `mise run fmt` | Fix formatting |
| `mise run check` | Lint and test, the same as CI |
| `mise run data` | Re-download MoMA's data and rebuild `data/on_view.csv` |

**Without mise**, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use the same commands directly: `uv sync`, `uv run python -m askmoma`, `uv run pytest`, `uv run ruff check .`.

## Project structure

```
askmoma/
├── mise.toml                  toolchain pin and tasks
├── pyproject.toml             package metadata, dependencies, pytest and ruff settings
├── uv.lock                    exact library versions
├── src/askmoma/
│   ├── config.py              paths and constants
│   ├── data/
│   │   ├── fetch.py           downloads MoMA's data, builds data/on_view.csv
│   │   └── collection.py      loads the works on view; title lookup
│   ├── features/              one module per use case
│   │   └── locate.py          use case 1: where is a work?
│   ├── chat/router.py         decides which feature answers a message
│   └── ui/app.py              the Gradio chat page
├── data/
│   ├── on_view.csv            the 1,556 works on view (committed)
│   ├── SOURCE.md              which version of MoMA's data, downloaded when
│   └── raw/                   full Artworks.csv and Artists.csv (not committed)
├── docs/design-notes.md       open questions and decisions, with sources
├── tests/                     pytest
└── .github/workflows/ci.yml   lint and tests on every push and pull request
```

**Adding a use case:** create a module in `src/askmoma/features/` that takes the `Collection` and the message and returns an `Answer`, add a route for it in `src/askmoma/chat/router.py`, and add tests in `tests/`.

## Working together

- Create a branch for each piece of work, such as `feature/rank-artists`, and open a pull request into `main`.
- CI runs the lint and tests on every pull request; another member reviews and merges.
- Run `mise run check` before pushing.
- Record design decisions, with their sources, in `docs/design-notes.md`.
- Never commit API keys or passwords; none are needed.

## Data and credits

- **Collection data:** The Museum of Modern Art (MoMA) collection data, https://github.com/MuseumofModernArt/collection, CC0 1.0. `mise run data` downloads both files, `Artworks.csv` and `Artists.csv`. Only the works listed as on view are committed, in `data/on_view.csv`. The Kaggle dataset `momanyc/museum-collection` contains the same two files but was last updated in February 2017, so its gallery locations are out of date; we use MoMA's official repository instead.
- **Interface:** Gradio (Abid et al., 2019).
- AskMoMA is an independent course project. It is not affiliated with or endorsed by The Museum of Modern Art.
