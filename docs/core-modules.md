# Core Modules Reference

This document covers the modules in the `nanochat/` package.

## gpt.py — GPT Model

The core transformer model implementation.

### `GPTConfig`
Dataclass holding architecture hyperparameters:

| Field | Default | Description |
|-------|---------|-------------|
| `sequence_len` | 1024 | Maximum sequence length |
| `vocab_size` | 65536 | Vocabulary size (2¹⁶) |
| `n_layer` | — | Number of transformer layers (set via `depth`) |
| `n_head` | — | Number of attention heads |
| `n_kv_head` | — | Number of KV heads (for grouped-query attention) |
| `n_embd` | — | Embedding dimension |

### `GPT`
Main model class.

- **`forward(idx, targets, kv_cache, loss_reduction)`** — Forward pass. Returns logits and optional loss.
- **`generate(tokens, max_tokens, temperature, top_k)`** — Autoregressive token generation with KV caching.
- **`setup_optimizers(lr, weight_decay, ...)`** — Creates paired Muon + AdamW optimizers.

### Supporting Classes
- **`CausalSelfAttention`** — Multi-Query Attention with RoPE and QK-norm
- **`MLP`** — Feed-forward network with ReLU² activation
- **`Block`** — Transformer block (attention + MLP with residual connections)

---

## engine.py — Inference Engine

Streaming inference with KV caching and tool use.

### `Engine`
Wraps a GPT model for efficient batched generation.

- **`generate(tokens, num_samples, max_tokens, temperature, top_k)`** — Streaming multi-sample generation. Yields tokens as they're generated. Supports tool use (detects `<|python_start|>` tokens and evaluates math expressions).
- **`generate_batch(...)`** — Non-streaming variant; returns all tokens at once.

### `KVCache`
Manages key/value caches with dynamic resizing for the decode phase.

---

## tokenizer.py — BPE Tokenizer

GPT-4-style byte-pair encoding with special chat tokens.

### Tokenizer Classes
- **`HuggingFaceTokenizer`** — Wrapper around HuggingFace Tokenizers library
- **`RustBPETokenizer`** — Uses `rustbpe` for training and `tiktoken` for inference

### Key Methods
- **`encode(text, prepend, append, num_threads)`** — Tokenize text to token IDs
- **`decode(ids)`** — Detokenize back to text
- **`render_conversation(conversation)`** — Convert a chat conversation to token IDs + loss mask (mask=1 for assistant tokens only)
- **`render_for_completion(conversation)`** — Prime model for autoregressive completion
- **`get_tokenizer()`** — Convenience function to load the trained tokenizer

### Special Tokens
`<|bos|>`, `<|eos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`

---

## dataloader.py — Training Data Pipeline

### `tokenizing_distributed_data_loader(B, T, split, tokenizer_threads)`
Streaming data loader that:
1. Reads documents from parquet shards
2. Tokenizes with multi-threading
3. Yields `(inputs, targets)` batches of shape `(B, T)`
4. DDP-aware: each rank processes different documents

---

## dataset.py — Data Source

Manages FineWeb-Edu dataset shards.

### Key Functions
- **`parquets_iter_batched(split, start, step)`** — Iterate parquet row groups (DDP-aware)
- **`list_parquet_files()`** — Find all downloaded parquet files
- **`download_single_file(index)`** — Download a single data shard with retry logic

### CLI Usage
```bash
python -m nanochat.dataset -n 8   # Download 8 data shards
```

---

## adamw.py — Distributed AdamW Optimizer

### `DistAdamW`
ZeRO-2 style distributed optimizer:
- Reduces and scatters gradients across ranks
- Each rank updates its assigned parameter slice
- All-gathers updated parameters after the step

Used for embedding and `lm_head` parameters.

---

## muon.py — Muon Optimizer

Novel orthogonal optimizer using Newton-Schulz iteration.

### Key Components
- **`zeropower_via_newtonschulz5(G, steps)`** — Orthogonalizes gradients via 5-step Newton-Schulz iteration
- **`Muon`** — Single-GPU variant
- **`DistMuon`** — Multi-GPU variant with reduce-scatter/all-gather

Used for all matrix parameters (attention, MLP layers).

---

## checkpoint_manager.py — Model Persistence

### Key Functions
- **`save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data)`** — Save model, optimizer, and metadata
- **`load_checkpoint(checkpoint_dir, step, device, load_optimizer)`** — Load from disk
- **`build_model(checkpoint_dir, step, device, phase)`** — Reconstruct a GPT model from a checkpoint
- **`load_model(source, device, phase)`** — Convenience loader (`source` = `"base"`, `"sft"`, `"rl"`, etc.)

---

## common.py — Shared Utilities

- **`compute_init()`** — Initialize CUDA, DDP, set precision (tf32)
- **`get_dist_info()`** — Return `(rank, world_size)`
- **`print0(...)`** — Print only on rank 0
- **`get_base_dir()`** — Returns `~/.cache/nanochat`

---

## core_eval.py — CORE Benchmark

Evaluates on DCLM paper's CORE tasks (multiple choice, schema matching, language modeling).

### Key Functions
- **`evaluate_task(model, task, ...)`** — Full distributed task evaluation
- **`evaluate_example(model, ...)`** — Single example evaluation
- **`render_prompts_*()`** — Format prompts by task type

---

## loss_eval.py — Bits Per Byte

### `evaluate_bpb(model, batches, steps, token_bytes)`
Computes a vocabulary-size-independent loss metric by normalizing cross-entropy loss by average token byte-length.

---

## execution.py — Sandboxed Code Execution

### `execute_code(code, timeout, maximum_memory_bytes)`
Safely runs model-generated Python code:
- Executes in a subprocess
- Enforces time (default 5s) and memory (default 256MB) limits
- Disables destructive functions (`os.system`, `shutil.rmtree`, etc.)
- Returns `ExecutionResult` with stdout, stderr, and exit code

Used by HumanEval evaluation and the inference engine's tool-use feature.

---

## report.py — Training Reports

### `Report`
Aggregates training logs and metrics into a final markdown report.

- **`log(section, data)`** — Write a section to the report
- **`generate()`** — Combine all sections + metrics summary table into `report.md`
- **`reset()`** — Clear sections and write header with system/git info

### Utilities
- `get_git_info()`, `get_gpu_info()`, `get_system_info()` — Environment metadata
- `estimate_cost()` — Estimate training cost from GPU type and wall-clock time

---

## configurator.py — Configuration

Simple Python-based config override system. Uses `exec()` for config files and `--key=value` CLI argument overrides.
