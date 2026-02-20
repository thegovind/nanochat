# Architecture Overview

nanochat implements a complete LLM pipeline in a single, minimal codebase. This document describes the high-level architecture and how the stages connect.

## Pipeline Stages

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│  Tokenizer   │───►│    Base      │───►│  Midtraining │───►│   SFT    │───►│    RL    │
│  Training    │    │  Pretraining │    │              │    │          │    │ (optional)│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────┘    └──────────┘
       │                   │                   │                 │               │
  tok_train.py        base_train.py       mid_train.py     chat_sft.py     chat_rl.py
  tok_eval.py         base_eval.py                         chat_eval.py    chat_eval.py
                      base_loss.py
```

### 1. Tokenizer Training

- **Script**: `scripts/tok_train.py`
- **Engine**: `rustbpe/` (Rust BPE implementation)
- **Input**: Raw text from FineWeb-Edu dataset
- **Output**: Trained BPE tokenizer with 65,536 vocabulary + special chat tokens
- **Eval**: `scripts/tok_eval.py` — compares compression ratio against GPT-2/GPT-4 tokenizers

### 2. Base Pretraining

- **Script**: `scripts/base_train.py` (multi-GPU via `torchrun`)
- **Input**: Tokenized FineWeb-Edu data shards (~240 shards, ~24GB)
- **Model**: GPT architecture (`nanochat/gpt.py`) with configurable depth
- **Optimizers**: Muon (matrix params) + DistAdamW (embeddings/lm_head)
- **Output**: Base model checkpoint
- **Eval**: CORE benchmark (`scripts/base_eval.py`) + BPB loss (`scripts/base_loss.py`)

### 3. Midtraining

- **Script**: `scripts/mid_train.py`
- **Input**: Task mixture (SmolTalk, MMLU, GSM8K) formatted with chat special tokens
- **Purpose**: Teach the model conversation structure, tool use patterns, and multiple-choice format
- **Output**: Midtrained checkpoint

### 4. Supervised Fine-Tuning (SFT)

- **Script**: `scripts/chat_sft.py`
- **Input**: Conversation data (ARC, GSM8K, SmolTalk) with assistant-only loss masking
- **Purpose**: Domain adaptation — each conversation trained independently with variable-length packing
- **Output**: SFT checkpoint
- **Eval**: `scripts/chat_eval.py` — ARC, MMLU, GSM8K, HumanEval benchmarks

### 5. Reinforcement Learning (Optional)

- **Script**: `scripts/chat_rl.py`
- **Input**: GSM8K problems
- **Method**: REINFORCE with advantage estimation (simplified GRPO)
- **Purpose**: Improve math reasoning via outcome-based rewards
- **Output**: RL checkpoint

### 6. Inference & Serving

- **CLI**: `scripts/chat_cli.py` — interactive terminal chat
- **Web**: `scripts/chat_web.py` — FastAPI server with ChatGPT-style UI
- **Engine**: `nanochat/engine.py` — KV-cached streaming generation with tool use support

## Model Architecture

The GPT model (`nanochat/gpt.py`) uses modern transformer techniques:

| Feature | Detail |
|---------|--------|
| Attention | Multi-Query Attention (MQA) with grouped KV heads |
| Positional encoding | Rotary Position Embeddings (RoPE) |
| Activation | ReLU² in MLP |
| Normalization | QK-norm in attention, RMSNorm |
| KV Cache | Dynamic resizing for efficient inference |

Default configuration (`depth=20`): **~561M parameters**

## Distributed Training

All training scripts support multi-GPU via PyTorch DDP (`torchrun`):

```bash
# 8-GPU training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Single GPU (omit torchrun, uses gradient accumulation)
python -m scripts.base_train
```

The system automatically compensates for fewer GPUs by increasing gradient accumulation steps, producing identical results (just slower).

## Data Flow

```
FineWeb-Edu (parquet)
       │
       ▼
  dataset.py ──► dataloader.py ──► tokenizer.py ──► GPU batches (B, T)
                                                          │
                                                          ▼
                                                    gpt.py (forward)
                                                          │
                                                          ▼
                                               muon.py + adamw.py (optimize)
                                                          │
                                                          ▼
                                              checkpoint_manager.py (save)
```

## Checkpoints

All checkpoints are stored under `~/.cache/nanochat/` with the structure:

```
~/.cache/nanochat/
├── data/              # Downloaded dataset shards
├── tok/               # Trained tokenizer artifacts
├── base_d20/          # Base pretrained model
├── mid_d20/           # Midtrained model
├── sft_d20/           # SFT model
├── rl_d20/            # RL model (optional)
├── eval_bundle/       # CORE evaluation data
└── report/            # Training report sections
```

## Report Generation

After training, `python -m nanochat.report generate` produces a `report.md` containing:
- System/environment info
- Tokenizer compression stats
- Training curves and sample generations
- Benchmark scores (CORE, ARC, MMLU, GSM8K, HumanEval)
- Final metrics summary table
