# celeste-rl

A reinforcement learning agent that learns to play [Celeste](https://www.celestegame.com/) (Steam version) from game state, using vanilla game physics.

**Status:** Phase 0, setting up and testing the connection to the game. No training code yet.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | PyTorch groundwork exercises (tensors, a hand-written training loop fitting `sin(x)`) |
| `results/` | Plots produced by the experiments |
| `docs/planning/` | Kickoff prompt and answers, the phased roadmap, and the first feasibility measurements |

## Plan

See [`docs/planning/roadmap.md`](docs/planning/roadmap.md). In short: reliably clear the first room of Chapter 1, then every room, then a continuous Chapter 1 run, then the main story, then faster play.

## Ground rules

- Game physics are never modified. Savestates and speed control are training-only scaffolding and are absent from any demonstrated run.
- Game files and mods are never committed to this repository.
