# Contributing

Brightwind is an **open-source library** and contributions from the wind and solar industry are warmly welcomed — analysts, engineers, researchers and developers. Bug fixes, new features, docs improvements and feedback all help build a standardised, validated toolkit for the industry.

For library usage and tutorials, see the [documentation site](https://brightwind-dev.github.io/brightwind-docs/). This guide covers contributing to the **codebase itself**.

---

## Reporting issues

Search the [issue tracker](https://github.com/brightwind-dev/brightwind/issues) before opening a new issue. When reporting a bug or requesting a feature, please include:

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

### Fork, branch and commit

Brightwind follows the standard open-source fork-based workflow.

> **Internal BrightWind contributors** with push access to `brightwind-dev/brightwind` skip step 1, clone the upstream repo directly and push their feature branches to it rather than to a fork. This lets later PRs stack on branches that are not yet merged.

1. **Fork the repo** on GitHub — click the **Fork** button (top-right of the [brightwind repo page](https://github.com/brightwind-dev/brightwind)).

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/brightwind.git
   cd brightwind
   ```

3. **Add the upstream remote** so you can pull future changes from the main repo:
   ```bash
   git remote add upstream https://github.com/brightwind-dev/brightwind.git
   ```

4. **Sync `dev`** and create a feature branch named after the issue:
   ```bash
   git checkout dev
   git pull upstream dev
   git checkout -b iss123_short_description
   ```

5. Keep changes small and focused. Verify before committing:
   ```bash
   git diff
   git status
   ```

6. **Commit** with a message that **starts with the issue number** (the space between `iss` and `#` links the commit to the issue on GitHub):
   ```bash
   git commit -m "iss #123 short description of the change"
   ```

7. **Push to your fork**:
   ```bash
   git push -u origin iss123_short_description
   ```

### Pull requests

Every PR must have a matching issue. If there is none, open one first and describe the problem or proposal. The only exception is release preparation by a maintainer, which updates the CHANGELOG and version number.

Open PRs from your feature branch into **`brightwind-dev/brightwind:dev`** — not `master`. When stacking on work that is not yet merged, target that feature branch instead. Only release PRs (`dev` → `master`) target `master`. An urgent hotfix may target `master` only when a maintainer asks for it, and must then be merged back into `dev`.

Before raising a PR:

- All tests pass — see [Running tests](#running-tests).
- Add a [CHANGELOG.md](CHANGELOG.md) entry under `[Unreleased]` for user-facing changes.
- Update docstrings for new or changed behaviour.
- Add a test for new functionality or bug fixes.

In the PR, cover one issue, start the title with the issue number in the same format as commit messages (e.g. `iss #123 short description`), and clearly explain what changed and why.

Fill in the PR template (`.github/pull_request_template.md`), which GitHub adds automatically when a PR is opened in the browser.

PRs are reviewed and merged by the maintainers team.

---

## AI-assisted contributions

Brightwind is built by and for people working in wind and solar resource assessment. The domain knowledge of analysts, engineers and researchers is what makes the library useful, so contributions must come from people, not automated agents.

AI coding tools (e.g. Claude Code, Codex, Copilot, Cursor) may be used to help with a contribution, provided that:

- **A person is the author and is accountable for it.** The person who opens the issue or PR is responsible for every line, as if they had written it themselves.
- **You understand the change.** You can explain what the code does and why, and respond to review comments yourself.
- **You have reviewed and tested it.** Run the tests and check results against expected values. AI-generated code can look plausible and still be numerically wrong.
- **You have the right to contribute it** under the MIT licence (see below).

Issues, PRs or review comments created and posted by an agent without a person directing and checking them are not accepted.

Maintainers may close contributions that do not meet these requirements. AI agents working in this repo follow the instructions in [AGENTS.md](AGENTS.md).

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

To work on or test parquet loading, include the optional engine (pyarrow by default):

```bash
pip install -e .[parquet]
```

Or use fastparquet instead: `pip install -e .[parquet-fastparquet]`.

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

All tests must pass before raising a PR. New functionality should include tests covering normal behaviour and edge cases.

### Integration tests against BrightHub

Some tests hit the live [BrightHub](https://brighthub.io) API and require API credentials. They are automatically **skipped** if the credentials are not set, so the rest of the suite runs cleanly without them.

Tests covering `.parquet` downloads also require the `[parquet]` extra (see [Development setup](#development-setup)) — they are skipped otherwise.

To run the BrightHub tests, set **either** an API key pair **or** an email/password pair as environment variables before running `pytest`.

Using an API key (recommended — generate one from [brighthub.io/account-settings/settings](https://brighthub.io/account-settings/settings)):

```bash
# Windows (PowerShell, current session)
$env:BRIGHTHUB_CLIENT_ID="your-client-id"
$env:BRIGHTHUB_CLIENT_SECRET="your-client-secret"

# macOS / Linux
export BRIGHTHUB_CLIENT_ID="your-client-id"
export BRIGHTHUB_CLIENT_SECRET="your-client-secret"
```

Using email and password (deprecated — will be removed in the next major release (v3.0); not available for Microsoft SSO users):

```bash
# Windows (PowerShell, current session)
$env:BRIGHTHUB_EMAIL="your-email"
$env:BRIGHTHUB_PASSWORD="your-password"

# macOS / Linux
export BRIGHTHUB_EMAIL="your-email"
export BRIGHTHUB_PASSWORD="your-password"
```

---

## Code standards

- **UK English** in comments, docstrings and user-facing strings (e.g. "analyse", "colour", "metres").
- **Docstrings** use reStructuredText style (`:param`, `:type`, `:return`, `:rtype`) — see existing functions for examples.
- **Follow existing patterns** in the codebase.
- **Wind and solar industry terminology** used correctly and consistently.

---

The library is licensed under the MIT licence. By contributing you agree that your contributions will be licensed under the same terms.
