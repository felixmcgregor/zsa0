# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zsa0 (Zootopia Alpha-Zero) is an AlphaZero-style game AI for Zootopia, a Pacman-like maze game. A neural network is trained entirely via self-play to collect pellets while avoiding zookeepers. Based on [advait/c4a0](https://github.com/advait/c4a0).

## Commands

### Setup
```sh
uv sync                              # Install Python deps
cd rust && uv run maturin develop --release  # Compile Rust extension (required after Rust changes)
```

### Lint & Test
```sh
uv run ruff check                    # Python lint
uv run ruff check --fix              # Python lint with autofix
uv run pyright                       # Python type checking
uv run pytest                        # Python tests
uv run pytest tests/zsa0_tests/nn_test.py  # Single test file
cd rust && cargo test                # Rust tests
```

### Train & Play
```sh
uv run src/zsa0/main.py train --max-gens=10   # Train network
uv run src/zsa0/main.py play --model=best     # Play against network
uv run src/zsa0/main.py nn_sweep              # NN hyperparameter sweep
uv run src/zsa0/main.py mcts_sweep            # MCTS hyperparameter sweep
```

### CI runs
`cargo test` (in rust/), `ruff check`, and `maturin build --release`.

## Architecture

**Two-language design:** Rust handles all game logic and performance-critical computation; Python handles neural network training. They connect via PyO3/maturin.

### Training Loop (generation-based)
1. **Self-play** (Rust, multi-threaded): MCTS explores game trees, batching NN forward passes, producing training samples (position → policy target + Q-value target)
2. **NN training** (Python/PyTorch Lightning): Trains on self-play data with KL divergence loss (policy) + MSE loss (Q-values)
3. **Repeat**: Each generation's trained model feeds the next generation's self-play

### Rust modules (`rust/src/`)
- `zootopia.rs` — Game state (`Pos`), rules, movement, terminal conditions
- `mcts.rs` — Monte Carlo Tree Search with NN-guided exploration
- `self_play.rs` — Multi-threaded self-play orchestration with batched NN evaluation
- `pybridge.rs` — PyO3 interface exposing `play_games` to Python
- `types.rs` — Core types: `Policy`, `QValue`, `Sample`, `GameResult`
- `tui.rs` / `interactive_play.rs` — Terminal UI for interactive play

### Python modules (`src/zsa0/`)
- `main.py` — CLI entry point (typer): train, play, sweep commands
- `nn.py` — `BottinaNet`: ResNet-style CNN outputting policy + Q-values
- `training.py` — Generation-based training loop, data loading, model checkpointing
- `tournament.py` — Round-robin tournament evaluation between model generations
- `sweep.py` — Optuna hyperparameter optimization
- `utils.py` — Device selection, checkpoint callbacks

### Key conventions
- Config classes use Pydantic `BaseModel`
- Models/game results serialized as `.pkl` with JSON metadata per generation
- `EvalPosT` trait abstracts NN evaluation in Rust (enables testing with mock evaluators)
- Python provides NN evaluation callbacks to Rust during self-play
