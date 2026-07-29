# Contributing

Thanks for your interest in improving the Driver Monitoring System!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
python scripts/download_model.py
```

## Ground rules

- **Python 3.11**, type hints and docstrings throughout, no bare `except`.
- Keep the geometry/state modules dependency-light so they stay unit-testable
  without a camera, MediaPipe, or a display.
- Add or update tests under `tests/` for any behaviour change.
- Run the suite before opening a PR:

  ```bash
  python -m pytest -v
  ```

## Pull requests

- Branch from `main` using a `feat/…`, `fix/…`, or `docs/…` prefix.
- Keep each PR focused; describe what changed and how you tested it.
- CI (`.github/workflows/tests.yml`) must pass.

## Reporting bugs

Open an issue using the bug-report template with steps to reproduce, your OS,
Python version, and camera details.
