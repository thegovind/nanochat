# nanochat Documentation

Welcome to the nanochat documentation. nanochat is a full-stack, minimal implementation of a ChatGPT-like LLM — covering tokenization, pretraining, finetuning, evaluation, inference, and web serving — designed to run end-to-end on a single 8×H100 node for ~$100.

## Table of Contents

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation, prerequisites, and your first run |
| [Architecture Overview](architecture.md) | High-level system design and pipeline stages |
| [Core Modules](core-modules.md) | Detailed reference for the `nanochat/` library |
| [Scripts Reference](scripts.md) | CLI entrypoints for training, evaluation, and serving |
| [Tasks & Evaluation](tasks.md) | Benchmark tasks (ARC, GSM8K, HumanEval, MMLU) and metrics |
| [Tokenizer](tokenizer.md) | BPE tokenizer: Rust training engine and Python interface |

## Pipeline at a Glance

```
Tokenizer Training ──► Base Pretraining ──► Midtraining ──► SFT ──► RL (optional)
                                                                        │
                                                            Chat CLI / Web UI
```

Each stage produces a checkpoint consumed by the next. The entire pipeline is orchestrated by [`speedrun.sh`](../speedrun.sh).
