# celeste-rl

A reinforcement learning agent that learns to play [Celeste](https://www.celestegame.com/) (Steam version) from game state, using vanilla game physics.

**Status:** Phase 0, setting up and testing the connection to the game. No training code yet.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | PyTorch groundwork exercises (tensors, a hand-written training loop fitting `sin(x)`) |
| `results/` | Plots produced by the experiments |
| `docs/planning/` | Kickoff prompt and answers, the phased roadmap, and the first feasibility measurements |

## Setup

Two separate virtual environments live in this repo, both ignored by git:

| Environment | Python | Used for |
|---|---|---|
| `.venv-rl` | 3.12.13 | The RL project: PyTorch 2.14.0 (CUDA 13.0), Gymnasium 1.3.0, Stable-Baselines3 2.9.0, pinned in `requirements-rl.lock`. |
| `.venv` | 3.14.6 | The PyTorch exercises in `experiments/` (torch 2.14.0+cu130). Left as-is. |

Create the RL environment with [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12.13
uv venv .venv-rl --python 3.12.13 --seed
uv pip sync requirements-rl.lock --python .venv-rl/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
.venv-rl/Scripts/python.exe scripts/check_env.py
```

`check_env.py` prints versions, runs a real calculation on the GPU, and trains PPO on CartPole briefly as a smoke test.

To change dependencies, edit `requirements-rl.in`, delete `requirements-rl.lock` (an existing lock file makes uv keep old versions), then regenerate it:

```bash
uv pip compile requirements-rl.in -o requirements-rl.lock --python-version 3.12 --python-platform windows --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match --generate-hashes --emit-index-annotation
```

Do not use `uv sync` or `uv run` here: uv's project mode manages a venv named `.venv` and would rebuild the Python 3.14 exercise environment.

Python 3.12 is used rather than 3.14 because the RL libraries (Gymnasium, Stable-Baselines3) only list support through 3.13. See the roadmap, section 1.4.

## Plan

See [`docs/planning/roadmap.md`](docs/planning/roadmap.md). In short: reliably clear the first room of Chapter 1, then every room, then a continuous Chapter 1 run, then the main story, then faster play.

## Ground rules

- Game physics are never modified. Savestates and speed control are training-only scaffolding and are absent from any demonstrated run.
- Game files and mods are never committed to this repository.
