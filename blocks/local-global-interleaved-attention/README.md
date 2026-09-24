# Local/Global Interleaved Attention

## Design Philosophy

Full attention at long context is quadratic; pure local attention loses global integration. Alternate the two at a fixed ratio — most layers cheap and local, a few layers global — so information can traverse the whole context while the average cost stays near-linear.

## Functionality

- Layer pattern: e.g., 5:1 (Gemma) or every 3rd layer (ModernBERT) or 128-window sliding + global alternation (gpt-oss); multi-view variant: frame-local vs all-view layers (VGGT).

## Used By

| Model | Role |
|-------|------|
| Gemma-4 12B | 40 sliding (1024-window) + 8 global layers, 262K context |
| gpt-oss-120b/20b | Alternating 128-window/full attention with attention-sink |
| ModernBERT | Global every 3rd layer, 128-token local window, native 8192 context |
| VGGT | Alternating frame-local (within-view) and global (all-view) attention for multi-view reasoning |

## Features

- **Near-linear average cost** with global information flow every k layers.
- **Fixed ratio = context-length budget** made explicit.

## Evolution

- **Predecessor**: Sparse Transformer (GPT-3's locally-banded sparse attention), Longformer.
- **Successor**: Jamba's ssm-attention-hybrid pushes the ratio to 7:1 with SSMs instead of local attention.
