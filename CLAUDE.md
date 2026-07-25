# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Early scaffold for a search engine over a movies corpus. As of now the repo contains:

- `pyproject.toml` — Python 3.14 project managed with `uv`; `dependencies = []` (no libraries added yet).
- `cli/keyword_search_cli.py` — an `argparse` CLI stub with a `search <query>` subcommand whose body is `pass`. The intended shape of the tool is `python cli/keyword_search_cli.py search "<query>"`.
- `data/movies.json` — ~26 MB dataset shaped as `{"movies": [{"id", "title", "description", ...}, ...]}`. **`/data/` is gitignored**, so the file is present on disk but never committed.
- `README.md` is empty.

The project name ("rag-search-engine") plus the keyword-CLI entry point suggest the direction is: start with keyword search over `movies.json`, then layer on retrieval/RAG. Don't assume any of that infrastructure exists yet — check before referencing it.

## Environment & commands

- Python is pinned via `.python-version` to **3.14** and dependencies are managed with **`uv`** (see `uv.lock`).
- Install / sync the environment: `uv sync`.
- Add a dependency: `uv add <pkg>` (updates `pyproject.toml` + `uv.lock`; do not hand-edit `dependencies`).
- Run the CLI: `uv run python cli/keyword_search_cli.py search "<query>"`.
- No test runner, linter, or formatter is configured yet. If you add one, wire it through `uv` (e.g. `uv run pytest`) rather than assuming a global install.

## Working notes

- The dataset lives outside version control. Any indexing/preprocessing artifacts should also go under `data/` (or another gitignored path) — don't commit derived indexes.
- The CLI uses `match/case` and PEP 604 type hints; keep new code targeting 3.14 features rather than back-porting.
