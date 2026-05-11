# Contributing

Brightwind is an **open-source library** and contributions from the wind and solar industry are warmly welcomed —
analysts, engineers, researchers and developers. Bug fixes, new features, docs improvements and feedback all help
build a standardised, validated toolkit for the industry.

For library usage and tutorials, see the [documentation site](https://brightwind-dev.github.io/brightwind-docs/).
This guide covers contributing to the **codebase itself**.

---

## Reporting issues

Search the [issue tracker](https://github.com/brightwind-dev/brightwind/issues) before opening a new issue. When
reporting a bug or requesting a feature, please include:

- A clear, descriptive title.
- What you expected to happen, and what actually happened.
- A minimal code example that reproduces the problem.
- Your brightwind version, Python version and operating system.

---

## Development workflow

The repo follows a modified git-flow:

- **`master`** — stable release.
- **`dev`** — newest features; tests must pass.
- **`iss<number>_<short_description>`** — feature branches off `dev`, one per issue.

### Branching and committing

1. Sync `dev`:
   ```bash
   git checkout dev
   git pull
   ```

2. Create a feature branch named after the issue:
   ```bash
   git checkout -b iss123_short_description
   ```

3. Keep changes small and focused. Verify before committing:
   ```bash
   git diff
   git status
   ```

4. Commit with a message that **starts with the issue number** (the space between `iss` and `#` links the commit
   to the issue on GitHub):
   ```bash
   git commit -m "iss #123 short description of the change"
   ```

5. Push:
   ```bash
   git push -u origin iss123_short_description
   ```

### Pull requests

Open PRs against **`dev`**, not `master`.

Before raising a PR:

- All tests pass — see [Running tests](#running-tests).
- Add a [CHANGELOG.md](CHANGELOG.md) entry under `[Unreleased]` for user-facing changes.
- Update docstrings for new or changed behaviour.
- Add a test for new functionality or bug fixes.

In the PR, keep the scope focused (ideally one issue), reference the issue number in the title
(e.g. `[iss123] Short description`), and clearly explain what changed and why.

---

## Development setup

Install brightwind in **editable mode** in a dedicated environment.

### Option 1 — venv

```bash
python -m venv brightwind_dev
# Windows
brightwind_dev\Scripts\activate
# macOS / Linux
source brightwind_dev/bin/activate
```

### Option 2 — conda

```bash
conda create --name brightwind_dev python=3.11
conda activate brightwind_dev
```

### Editable install

With the environment active, clone the repo and install brightwind in editable mode:

```bash
git clone https://github.com/brightwind-dev/brightwind.git
cd brightwind
pip install -e .
```

The `-e` flag links the install to your local clone, so code changes are picked up without reinstalling.

---

## Running tests

From the repo root:

```bash
pytest tests/
```

Single file or single test:

```bash
pytest tests/test_load.py
pytest tests/test_load.py::test_load_brighthub -v
```

All tests must pass before raising a PR. New functionality should include tests covering normal behaviour and
edge cases.

---

## Code standards

- **UK English** in comments, docstrings and user-facing strings (e.g. "analyse", "colour", "metres").
- **Docstrings** use reStructuredText style (`:param`, `:type`, `:return`, `:rtype`) — see existing functions
  for examples.
- **Follow existing patterns** in the codebase.
- **Wind and solar industry terminology** used correctly and consistently.

---

The library is licensed under the MIT license. By contributing you agree that your contributions will be licensed
under the same terms.
