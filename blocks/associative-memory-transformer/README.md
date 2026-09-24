# Associative Memory Transformer (ARMT)

## Design Philosophy

A recurrent memory transformer that maintains a **persistent external memory** across sequence chunks, enabling long-context modeling without a linearly growing KV cache. The philosophy: chunk the input, process each chunk with a Transformer, and carry a fixed-size memory tensor forward — the model reads/writes to the memory via attention, so information persists across chunks without storing all tokens.

## Functionality

- **Chunk the input**: Split the sequence into chunks of length `L`.
- **Per chunk**: A Transformer block processes `[chunk_tokens; memory_tokens]`.
  - Self-attention over chunk + memory.
  - Memory tokens are read from the previous chunk's output and written to the next.
- **Memory**: A fixed-size set of learnable tokens `[n_memory, d]` carried forward.
- **Read/write**: Attention reads from old memory, writes new memory.

## Used By

| Model | Role |
|-------|------|
| ARMT (Associative Recurrent Memory Transformer) | The defining architecture |
- Related to RMT (Recurrent Memory Transformer), Block-Recurrent Transformers.

## Features

- **Fixed memory**: O(n_memory × d) regardless of sequence length.
- **Cross-chunk**: Information persists via memory, not a full KV cache.
- **Transformer-compatible**: Each chunk is processed by a standard Transformer block.

## Evolution

- **Predecessor**: Memory Networks (end-to-end memory networks); Transformer-XL (segment-level recurrence).
- **Successor**: Block-recurrent Transformers; RAG (retrieve-then-read); long-context architectures.
