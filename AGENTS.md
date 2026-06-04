# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

Single educational Jupyter notebook (`XOR Problem using Backpropagation in Neural Network.ipynb`) that trains a small neural network on the XOR truth table. There are no web apps, Docker services, or formal lint/test targets.

### Dependencies

Python 3 with **NumPy** and **Jupyter Notebook**. See `README.md` for the upstream install hint (`pip install notebook`). NumPy is required by the notebook code but not listed in the README.

### PATH

`pip install --user notebook` installs CLI tools under `~/.local/bin`. Ensure that directory is on `PATH` before running `jupyter` or `jupyter-nbconvert`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Run / verify (no long-lived service required)

**Headless E2E (preferred for agents):** execute the notebook and confirm XOR-like outputs (~0 for inputs (0,0) and (1,1); ~1 for (0,1) and (1,0)):

```bash
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
jupyter nbconvert --to notebook --execute "XOR Problem using Backpropagation in Neural Network.ipynb" --output /tmp/xor-executed.ipynb
```

**Interactive UI (optional):** Jupyter listens on port **8888** by default:

```bash
cd /workspace
export PATH="$HOME/.local/bin:$PATH"
jupyter notebook --no-browser --port=8888 --ip=127.0.0.1
```

Open the printed URL (with token) in a browser to run cells interactively.

### Lint / test / build

None defined in this repository. Validation is notebook execution and checking the final printed network outputs.
