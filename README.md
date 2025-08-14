# zta0: Zootopia Alpha-Zero

![CI](https://github.com/felixmcgregor/zsa0/actions/workflows/ci.yaml/badge.svg?ts=2)

> **Credit:** This project is based on amazing work of [advait/c4a0](https://github.com/advait/c4a0).

An Alpha-Zero-style Zootopia game engine trained entirely via self play.

Zootopia is a Pacman-like game where animals must collect pellets and power pellets while escaping from zookeepers in a maze environment.

The game logic, Monte Carlo Tree Search, and multi-threaded self play engine is written in rust
[here](https://github.com/felixmcgregor/zsa0/tree/master/rust).

The NN is written in Python/PyTorch [here](https://github.com/felixmcgregor/zsa0/tree/master/src/zsa0?ts=2)
and interfaces with rust via [PyO3](https://pyo3.rs/v0.22.2/)

![Terminal UI](https://raw.githubusercontent.com/felixmcgregor/zsa0/refs/heads/main/images/tui.png)

## Usage

1. Install clang
```sh
# Instructions for Ubuntu/Debian (other OSs may vary)
sudo apt install clang
```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for python dep/env management
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install deps and create virtual env:
```sh
uv sync
```

4. Compile rust code
```sh
cd rust
uv run maturin develop --release
```

4. Train a network
```sh
uv run src/zsa0/main.py train --max-gens=10
```

5. Play against the network
```sh
uv run src/zsa0/main.py play --model=best
```

6. (Optional) Download a game solver to objectively measure training progress (if available)

## Results
After X generations of training looks like its solving the Zootopia maze.

![Training Results](https://raw.githubusercontent.com/felixmcgregor/zsa0/refs/heads/main/images/learning.png)

## Architecture

### PyTorch NN [`src/zsa0/nn.py`](https://github.com/felixmcgregor/zsa0/blob/main/src/zsa0/nn.py?ts=2)

A resnet-style CNN that takes in as input a game position and outputs a Policy (probability
distribution over moves weighted by promise) and Q Value (predicted win/loss value [-1, 1]).

Various NN hyperparameters are sweepable via the `nn-sweep` command.

### Zootopia Game Logic [`rust/src/zootopia.rs`](https://github.com/felixmcgregor/zsa0/blob/main/rust/src/zootopia.rs?ts=2)

Implements compact representation of game state (`Pos`) and all Zootopia game rules
and logic including animal movement, pellet collection, and zookeeper interactions.

### Monte Carlo Tree Search (MCTS) [`rust/src/mcts.rs`](https://github.com/felixmcgregor/zsa0/blob/main/rust/src/mcts.rs?ts=2)

Implements Monte Carlo Tree Search - the core algorithm behind Alpha-Zero. Probabalistically
explores potential game pathways and optimally hones in on the optimal move to play from any
position.

MCTS relies on outputs from the NN. The output of MCTS helps train the next generation's NN.

### Self Play [`rust/src/self_play.rs`](https://github.com/felixmcgregor/zsa0/blob/main/rust/src/self_play.rs?ts=2)

Uses rust multi-threading to parallelize self play (training data generation).
