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
| `.venv-rl` | 3.12.13 | The RL project. RL dependencies are not installed yet; exact versions will be pinned in a lock file. |
| `.venv` | 3.14.6 | The PyTorch exercises in `experiments/` (torch 2.14.0+cu130). Left as-is. |

Create the RL environment with [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12.13
uv venv .venv-rl --python 3.12.13 --seed
```

Python 3.12 is used rather than 3.14 because the RL libraries (Gymnasium, Stable-Baselines3) only list support through 3.13. See the roadmap, section 1.4.

## Plan

See [`docs/planning/roadmap.md`](docs/planning/roadmap.md). In short: reliably clear the first room of Chapter 1, then every room, then a continuous Chapter 1 run, then the main story, then faster play.

## Ground rules

- Game physics are never modified. Savestates and speed control are training-only scaffolding and are absent from any demonstrated run.
- Game files and mods are never committed to this repository.
