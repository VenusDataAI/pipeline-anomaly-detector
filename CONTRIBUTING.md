# Contributing to Pipeline Anomaly Detector

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Fork & Clone](#fork--clone)
- [Local Setup](#local-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Convention](#commit-convention)
- [Opening a Pull Request](#opening-a-pull-request)
- [Reporting Issues](#reporting-issues)

---

## Prerequisites

- Python **3.11** or higher
- Git
- A GitHub account
- (Optional) Docker, if you want to test the Airflow integration locally

---

## Fork & Clone

1. Click **Fork** on the top-right of the repository page.
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/pipeline-anomaly-detector.git
cd pipeline-anomaly-detector
```

3. Add the upstream remote so you can pull future changes:

```bash
git remote add upstream https://github.com/VenusDataAI/pipeline-anomaly-detector.git
```

---

## Local Setup

Create and activate a virtual environment, then install the package in editable mode with all development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, pytest-mock, pytest-cov, matplotlib, and notebook support.

To also install the optional Airflow integration:

```bash
pip install -e ".[dev,airflow]"
```

Verify the installation:

```bash
python -c "import pipeline_anomaly_detector; print(pipeline_anomaly_detector.__version__)"
pad --help
```

---

## Running Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ -v --cov=pipeline_anomaly_detector --cov-report=term-missing
```

Run only unit tests:

```bash
pytest tests/unit/ -v
```

Run only integration tests:

```bash
pytest tests/integration/ -v
```

Run a specific test file:

```bash
pytest tests/unit/test_feature_extractor.py -v
```

All tests must pass before a PR can be merged. If you add a feature, add a test for it.

---

## Code Style

This project uses [**Ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

Install it (already included via `[dev]` if you add it to pyproject.toml, otherwise):

```bash
pip install ruff
```

Check for lint errors:

```bash
ruff check .
```

Auto-fix safe issues:

```bash
ruff check . --fix
```

Format code:

```bash
ruff format .
```

Alternatively, [**Black**](https://black.readthedocs.io/) is also acceptable:

```bash
pip install black
black .
```

### Rules of thumb

- Line length: **88** characters (Black default).
- Type hints on all public functions and methods.
- Google-style docstrings on all public classes and methods.
- Use `structlog` for logging — do not use `print()` in library code.

---

## Commit Convention

This project follows [**Conventional Commits**](https://www.conventionalcommits.org/en/v1.0.0/).

Format:

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `chore` | Build process, dependency updates, CI changes |
| `ci` | Changes to CI/CD workflows |

### Examples

```
feat(detectors): add DBSCAN detector as an alternative to IsolationForest

fix(zscore): handle divide-by-zero when std dev is 0 in rolling window

docs(readme): add architecture diagram in Mermaid

test(ensemble): add test for mismatched detector weight normalization

chore(deps): bump scikit-learn from 1.4 to 1.5
```

Commits that introduce breaking changes must include `BREAKING CHANGE:` in the footer:

```
feat(models)!: rename AnomalyScore.score to AnomalyScore.anomaly_score

BREAKING CHANGE: field `score` renamed to `anomaly_score` in AnomalyScore dataclass.
```

---

## Opening a Pull Request

1. Create a feature branch from `main`:

```bash
git checkout -b feat/my-new-detector
```

2. Make your changes, write tests, update docs as needed.

3. Ensure all tests pass and there are no lint errors:

```bash
ruff check .
pytest tests/ -v
```

4. Push to your fork:

```bash
git push origin feat/my-new-detector
```

5. Open a PR against `main` on the upstream repository.

### PR checklist

- [ ] Tests added or updated for the change
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] No lint errors (`ruff check .`)
- [ ] Docstrings updated for any changed public API
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR title follows Conventional Commits format

A maintainer will review your PR, request changes if needed, and merge once approved.

---

## Reporting Issues

Please open an issue at [github.com/VenusDataAI/pipeline-anomaly-detector/issues](https://github.com/VenusDataAI/pipeline-anomaly-detector/issues) and include:

- Python version (`python --version`)
- Package version (`python -c "import pipeline_anomaly_detector; print(pipeline_anomaly_detector.__version__)"`)
- Minimal reproducible example
- Full traceback if applicable
