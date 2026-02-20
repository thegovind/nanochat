# Tokenizer

nanochat uses a GPT-4-style byte-pair encoding (BPE) tokenizer with a vocabulary size of 65,536 (2¹⁶). The tokenizer has two components: a fast Rust training engine and a Python interface for inference.

## Architecture

```
Training:  rustbpe (Rust/PyO3) ──► Vocabulary file
Inference: tiktoken (Python)   ◄── Vocabulary file
```

The Rust engine handles the compute-intensive BPE training, while tiktoken provides fast inference using the exported vocabulary.

## Rust BPE Engine (`rustbpe/`)

The Rust implementation provides high-performance BPE training:

- **Parallel text splitting** using `rayon` (processes buffers without GIL lock)
- **Octonary heap** for O(log n) merge prioritization
- **GPT-4 regex pattern** for pre-tokenization splitting
- **PyO3 bindings** for seamless Python integration

### Building

```bash
# Requires Rust toolchain
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Key Methods

```python
from rustbpe import Tokenizer

tok = Tokenizer()
tok.train_from_iterator(text_iterator, vocab_size=65536)
ranks = tok.get_mergeable_ranks()  # Export for tiktoken
encoded = tok.encode(text)
```

## Python Interface (`nanochat/tokenizer.py`)

### Tokenizer Classes

Two implementations are available:

| Class | Training | Inference | Description |
|-------|----------|-----------|-------------|
| `RustBPETokenizer` | rustbpe | tiktoken | Default — fast training + fast inference |
| `HuggingFaceTokenizer` | HF Tokenizers | HF Tokenizers | Alternative using HuggingFace library |

### Loading a Trained Tokenizer

```python
from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()
```

### Basic Usage

```python
# Encode text to token IDs
token_ids = tokenizer.encode("Hello, world!")

# Decode back to text
text = tokenizer.decode(token_ids)

# Encode with special tokens
token_ids = tokenizer.encode("Hello", prepend="bos", append="eos")
```

### Chat Conversation Rendering

The tokenizer handles chat-format conversations with special token wrapping:

```python
conversation = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

# For training: returns token IDs + loss mask (1 for assistant tokens)
token_ids, mask = tokenizer.render_conversation(conversation)

# For inference: prime model for completion
token_ids = tokenizer.render_for_completion(conversation)
```

### Special Tokens

| Token | Purpose |
|-------|---------|
| `<\|bos\|>` | Beginning of sequence |
| `<\|eos\|>` | End of sequence |
| `<\|user_start\|>` / `<\|user_end\|>` | User message boundaries |
| `<\|assistant_start\|>` / `<\|assistant_end\|>` | Assistant message boundaries |
| `<\|python_start\|>` / `<\|python_end\|>` | Python code block boundaries |
| `<\|output_start\|>` / `<\|output_end\|>` | Code output boundaries |

## Training a Tokenizer

```bash
# Download data first
python -m nanochat.dataset -n 8

# Train tokenizer on ~2B characters
python -m scripts.tok_train --max_chars=2000000000

# Evaluate compression ratio
python -m scripts.tok_eval
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_chars` | — | Maximum characters of training data |
| `--doc_cap` | — | Maximum characters per document |
| `--vocab_size` | 65536 | Target vocabulary size |

### Evaluation

`tok_eval.py` benchmarks the trained tokenizer against GPT-2 and GPT-4 tokenizers across several text categories (news, code, math, science, training data) and reports compression ratios.
