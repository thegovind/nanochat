# Scripts Reference

All scripts are run as Python modules (e.g., `python -m scripts.base_train`). Multi-GPU scripts use `torchrun`.

## Training Scripts

### `base_train.py` — Base Pretraining

Trains the base GPT model on tokenized FineWeb-Edu data.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=20 \
    --device_batch_size=32 \
    --run=$WANDB_RUN
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--depth` | 20 | Model depth (layers); determines model size |
| `--device_batch_size` | 32 | Batch size per GPU (reduce if OOM) |
| `--total_batch_size` | — | Total batch size (auto gradient accumulation) |
| `--max_seq_len` | 1024 | Maximum sequence length |
| `--num_iterations` | — | Training steps (auto-computed from target FLOPs) |
| `--target_flops` | — | Target compute budget |
| `--target_param_data_ratio` | 20 | Chinchilla-style tokens-to-params ratio |
| `--eval_every` | — | Steps between validation evaluations |
| `--core_metric_every` | — | Steps between CORE benchmark evaluations |
| `--run` | `dummy` | W&B run name |

---

### `mid_train.py` — Midtraining

Continues training on a curated task mixture (SmolTalk, MMLU, GSM8K).

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- \
    --device_batch_size=32 \
    --run=$WANDB_RUN
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_tag` | — | Base model checkpoint tag |
| `--step` | — | Base model checkpoint step |
| `--device_batch_size` | 32 | Batch size per GPU |
| `--max_seq_len` | 1024 | Maximum sequence length |
| `--init_lr_frac` | — | Initial learning rate fraction |
| `--eval_every` | — | Steps between evaluations |

---

### `chat_sft.py` — Supervised Fine-Tuning

Fine-tunes the midtrained model for chat using conversation data.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `mid` | Source checkpoint (`mid` or `base`) |
| `--model_tag` | — | Model checkpoint tag |
| `--device_batch_size` | — | Batch size per GPU |
| `--num_epochs` | — | Number of training epochs |
| `--target_examples_per_step` | — | Examples per gradient step |
| `--init_lr_frac` | — | Initial learning rate fraction |

---

### `chat_rl.py` — Reinforcement Learning

REINFORCE-based RL training on GSM8K (optional stage).

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `sft` | Source checkpoint (`sft` or `mid`) |
| `--device_batch_size` | — | Batch size per GPU |
| `--examples_per_step` | — | Examples per gradient step |
| `--num_samples` | — | Samples per example for advantage estimation |
| `--max_new_tokens` | — | Max tokens to generate per sample |
| `--temperature` | — | Sampling temperature |
| `--num_epochs` | — | Number of epochs |

---

## Evaluation Scripts

### `base_eval.py` — CORE Benchmark

Evaluates the base model on CORE tasks (in-context learning).

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

Outputs CSV with centered accuracies per task.

---

### `base_loss.py` — Validation Loss

Measures bits-per-byte (BPB) on train/val splits and generates samples.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_tag` | — | Model checkpoint tag |
| `--model_step` | — | Model checkpoint step |
| `--device_batch_size` | — | Batch size per GPU |

---

### `chat_eval.py` — Chat Model Evaluation

Evaluates chat models on ARC, MMLU, GSM8K, and HumanEval.

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --source` | `sft` | Model source (`mid`, `sft`, `rl`) |
| `-a, --task-name` | — | Specific task to evaluate (default: all) |
| `-t, --temperature` | — | Sampling temperature |
| `-n, --num-samples` | — | Number of samples for pass@k |
| `-b, --batch-size` | — | Batch size |
| `-x, --max-problems` | — | Max problems to evaluate |

---

### `tok_eval.py` — Tokenizer Evaluation

Benchmarks tokenizer compression ratio against GPT-2 and GPT-4.

```bash
python -m scripts.tok_eval
```

---

## Tokenizer Training

### `tok_train.py`

Trains a BPE tokenizer from FineWeb-Edu data.

```bash
python -m scripts.tok_train --max_chars=2000000000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_chars` | — | Maximum characters to train on |
| `--doc_cap` | — | Max characters per document |
| `--vocab_size` | 65536 | Vocabulary size |

---

## Inference & Serving

### `chat_cli.py` — CLI Chat

Interactive terminal chat with a trained model.

```bash
python -m scripts.chat_cli
python -m scripts.chat_cli -p "Why is the sky blue?"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --source` | `sft` | Model source (`sft`, `mid`, `rl`) |
| `-g, --model-tag` | — | Model checkpoint tag |
| `-t, --temperature` | — | Sampling temperature |
| `-k, --top-k` | — | Top-k sampling |
| `-p, --prompt` | — | Single prompt (non-interactive mode) |

---

### `chat_web.py` — Web UI Server

FastAPI server with a ChatGPT-style web interface and OpenAI-compatible API.

```bash
python -m scripts.chat_web
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `sft` | Model source |
| `--temperature` | — | Sampling temperature |
| `--top-k` | — | Top-k sampling |
| `--max-tokens` | — | Max tokens to generate |
| `--port` | 8000 | Server port |
| `--host` | — | Server host |

**Endpoints:**
- `GET /` — Chat web UI
- `POST /chat/completions` — OpenAI-compatible chat completion API (streaming supported)
