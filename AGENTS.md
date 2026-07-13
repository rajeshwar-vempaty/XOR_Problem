# AGENTS.md

## Cursor Cloud specific instructions

This is a single, self-contained educational Python project: a from-scratch 2-2-1
neural network trained with backpropagation to learn XOR. There are no servers,
databases, or external services — "running the app" means executing a script that
trains the network, prints a results table, and writes two PNG plots to `assets/`.

- Dependencies (`numpy`, `matplotlib`, plus `flake8`/`pytest` for CI parity) are
  installed by the startup update script; no manual install is needed.
- Run the application: `python3 xor_backprop.py`. It trains 10,000 epochs, prints an
  accuracy table (expect 100%), and saves `assets/training_loss.png` and
  `assets/decision_boundary.png`. It uses matplotlib's headless `Agg` backend, so no
  display/GUI is required.
- Lint (mirrors CI in `.github/workflows/python-app.yml`):
  `python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`.
- Tests: CI runs `python3 -m pytest`, but there are currently no test files, so pytest
  collects 0 tests and exits with code 5 — this is expected, not a failure.
- The notebook (`XOR Problem using Backpropagation in Neural Network.ipynb`) is optional
  and only needs the `notebook`/`jupyter` packages (installed via `requirements.txt`).
