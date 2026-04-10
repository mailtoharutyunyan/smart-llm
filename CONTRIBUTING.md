# Contributing to ACS

Thank you for considering contributing to the Autonomous Cognitive System. This document outlines how to get started.

## Development Setup

```bash
# Clone and set up
git clone git@github.com:mailtoharutyunyan/smart-llm.git
cd smart-llm
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Code Standards

- **Formatting**: We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.
- **Type Hints**: Use type hints for all function signatures.
- **Docstrings**: Follow PEP 257. All public functions must have docstrings.
- **Logging**: Use `logging.getLogger(__name__)` — never `print()`.

Run the linter before committing:
```bash
ruff check .
ruff format .
```

## Testing

```bash
pytest
```

All new features should include corresponding tests in the `tests/` directory.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, ensuring tests pass and code is properly formatted.
3. Write a clear PR description explaining what the change does and why.
4. Request a review.

## Commit Messages

Use clear, descriptive commit messages:
- `feat: Add multi-GPU training support`
- `fix: Prevent evaluator score inflation during cold start`
- `docs: Update CLI reference with new inspection commands`
- `refactor: Extract trace scoring into dedicated module`

## Reporting Issues

Use GitHub Issues. Include:
- Steps to reproduce the problem.
- Expected vs actual behavior.
- Python version and OS.
- Relevant log output.
