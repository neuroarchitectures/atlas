# Parallel Attention+MLP Block (GPT-NeoX)

## Design Philosophy

The sequential pre-norm block (norm → attn → add → norm → mlp → add) serializes two residual streams. GPT-NeoX's parallel form feeds *one* pre-norm input to both attention and MLP and sums both into the same residual — saving one norm and one sequential dependency per layer, with negligible quality loss at scale.

## Functionality

- `x ← x + Attn(LN(x)) + MLP(LN(x))` (single LN, parallel branches).
- ~15% faster wall-clock than sequential at equal depth (GPT-NeoX measurements).

## Used By

| Model | Role |
|-------|------|
| Falcon-7B | 32 layers, hidden 4544 |
| Phi-2 | 32 layers, hidden 2560, partial RoPE on 40% of head dim |
| Pythia-1.4B | 24 layers, hidden 2048 (GPT-NeoX lineage) |

## Features

- **Fewer sequential ops per layer** — the parallelism win grows with tensor width.
- **Quality-neutral at scale** — validated across the NeoX/Pythia model suite.

## Evolution

- **Predecessor**: GPT-NeoX; earlier parallel residual in snapshot/DPFP experiments.
- **Successor**: nGPT's SLERP residual update reformulates the stream itself; most 2023+ LLMs returned to sequential with fused kernels.
