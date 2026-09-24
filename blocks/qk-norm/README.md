# QK-Norm

## Design Philosophy

Attention logits can explode with training length/width, destabilizing softmax. Normalize queries and keys per head (RMSNorm) *before* RoPE — bounding logit magnitude by construction and replacing ad-hoc bias tricks.

## Functionality

- `q = RMSNorm_q(q); k = RMSNorm_k(k)` per head, then RoPE and scaled dot-product as usual.
- Norm parameters are learned; removed QKV bias keeps parameter count flat.

## Used By

| Model | Role |
|-------|------|
| Qwen3-8B | Per-head RMSNorm on Q/K, 32:8 GQA heads, head dim 128, applied pre-RoPE each layer |
| ViT-22B / ViT-22B lineage | The original ViT-side adoption |

## Features

- **Logit stability at scale** — the ViT-22B finding, imported into LLMs.
- **Drops bias heuristics** — cleaner than per-channel bias corrections.

## Evolution

- **Predecessor**: ViT-22B QK-layer norm; post-hoc logit capping (Gemma 2).
- **Related**: cosine-attention (Swin-V2) — normalize inside the similarity instead of before it.
