# Tasks & Evaluation

nanochat evaluates models on several benchmarks at different training stages. All task implementations live in the `tasks/` directory.

## Task Framework

### `tasks/common.py`

Provides base classes for all task datasets:

- **`Task`** — Abstract base class with indexing, slicing, and abstract `evaluate()` method
- **`TaskMixture`** — Combines multiple tasks with deterministic shuffling for mixed training
- **`TaskSequence`** — Sequential training on multiple tasks (curriculum learning)
- **`render_mc(question, choices)`** — Consistent multiple-choice formatting

## Benchmarks

### ARC (AI2 Reasoning Challenge)

**File:** `tasks/arc.py`

Science reasoning with multiple-choice questions. Includes both Easy and Challenge subsets.

- **Type:** Categorical (A/B/C/D)
- **Evaluation:** Exact match against ground truth letter
- **Used in:** SFT training, chat_eval

### GSM8K (Grade School Math)

**File:** `tasks/gsm8k.py`

Grade-school-level math word problems requiring step-by-step reasoning.

- **Type:** Generative with tool use
- **Answer format:** Numeric answer after `####` marker
- **Tool use:** Supports calculator expressions within `<< >>` tags
- **Evaluation:** Exact numeric match after normalization
- **Used in:** Midtraining, SFT, RL training, chat_eval

### HumanEval

**File:** `tasks/humaneval.py`

OpenAI's code generation benchmark — 164 Python programming problems.

- **Type:** Generative (code)
- **Evaluation:** Execute generated code against test cases via `nanochat/execution.py`
- **Metric:** pass@k
- **Used in:** chat_eval

### MMLU (Massive Multitask Language Understanding)

**File:** `tasks/mmlu.py`

General knowledge multiple-choice across 57 academic subjects.

- **Type:** Categorical (A/B/C/D)
- **Evaluation:** Exact match against ground truth letter
- **Used in:** Midtraining, chat_eval

### SmolTalk

**File:** `tasks/smoltalk.py`

HuggingFace conversational dataset (460K train, 24K test examples).

- **Type:** Multi-turn conversations
- **Evaluation:** No formal metric — used for supervised fine-tuning data only
- **Used in:** Midtraining, SFT

## Evaluation Flow

### Base Model Evaluation (CORE)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```
Evaluates on the CORE benchmark suite (from the DCLM paper) using in-context learning. Reports centered accuracies per task.

### Chat Model Evaluation
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```
Evaluates on ARC, MMLU, GSM8K, and HumanEval. Computes a **ChatCORE** aggregate metric.

## Metrics Summary

| Metric | Type | Stage | Description |
|--------|------|-------|-------------|
| BPB | Loss | Base | Bits-per-byte on validation data |
| CORE | Accuracy | Base | Aggregate of DCLM in-context tasks |
| ARC-Easy | Accuracy | Chat | Science reasoning (easy) |
| ARC-Challenge | Accuracy | Chat | Science reasoning (hard) |
| MMLU | Accuracy | Chat | General knowledge (57 subjects) |
| GSM8K | Accuracy | Chat/RL | Grade school math |
| HumanEval | pass@k | Chat | Code generation |
| ChatCORE | Aggregate | Chat | Combined chat evaluation score |

## Example Results ($100 Speedrun)

| Metric | BASE | MID | SFT | RL |
|--------|------|-----|-----|-----|
| CORE | 0.2219 | - | - | - |
| ARC-Challenge | - | 0.2875 | 0.2807 | - |
| ARC-Easy | - | 0.3561 | 0.3876 | - |
| GSM8K | - | 0.0250 | 0.0455 | 0.0758 |
| HumanEval | - | 0.0671 | 0.0854 | - |
| MMLU | - | 0.3111 | 0.3151 | - |
| ChatCORE | - | 0.0730 | 0.0884 | - |
