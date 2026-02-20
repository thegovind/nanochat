# Getting Started

## Prerequisites

- **Python** ≥ 3.10
- **Rust** (for building the BPE tokenizer)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **GPU** — an 8×H100 node for the full speedrun; single GPU works too (8× slower)

## Installation

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 3. Create virtual environment and install dependencies
uv venv
uv sync
source .venv/bin/activate

# 4. Build the Rust BPE tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Quick Start — Full Pipeline (~4 hours on 8×H100)

The fastest way to run everything end-to-end:

```bash
bash speedrun.sh
```

Or run inside a `screen` session (recommended for long runs):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

When complete, serve the chat UI:

```bash
python -m scripts.chat_web
```

Visit the URL shown (e.g. `http://<your-ip>:8000/`) to chat with your model.

## Quick Start — Exploring Without GPUs

Even without GPUs, you can explore the codebase, run the tokenizer, and run tests:

```bash
# Run tokenizer tests
python -m pytest tests/test_rustbpe.py -v -s

# Train a tokenizer on a small amount of data
python -m nanochat.dataset -n 1
python -m scripts.tok_train --max_chars=1000000
python -m scripts.tok_eval
```

## Directory Structure

```
nanochat/
├── nanochat/          # Core library (model, tokenizer, dataloader, engine, etc.)
├── scripts/           # CLI entrypoints (training, eval, serving)
├── tasks/             # Benchmark task definitions
├── rustbpe/           # Rust BPE tokenizer (compiled via maturin + PyO3)
├── tests/             # Test suite
├── dev/               # Development assets
├── speedrun.sh        # End-to-end pipeline script
└── pyproject.toml     # Project configuration
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_RUN` | W&B run name (`dummy` disables logging) | `dummy` |
| `OMP_NUM_THREADS` | OpenMP thread count | `1` |
| `NANOCHAT_BASE_DIR` | Cache directory for artifacts | `~/.cache/nanochat` |

## Running Tests

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## Next Steps

- Read the [Architecture Overview](architecture.md) to understand the pipeline
- Browse the [Core Modules](core-modules.md) for implementation details
- Check the [Scripts Reference](scripts.md) for CLI usage
