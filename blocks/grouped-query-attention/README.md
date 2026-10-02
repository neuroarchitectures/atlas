# Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)

## Design Philosophy

Share K/V heads across multiple query heads. The philosophy: the KV cache (not compute) is the bottleneck at inference time, and most KV heads are redundant. MQA (Shazeer, 2019) uses 1 KV head for all query heads — extreme compression, slight quality loss. GQA (Ainslie, 2023) generalizes to `G` KV head groups (e.g., 8 KV heads for 32 query heads), recovering quality while keeping the cache ~4× smaller.

## Functionality

- `numHeads` query heads (e.g., 32), each with its own Q projection.
- `numKVHeads` KV heads (e.g., 8), shared across `numHeads / numKVHeads = 4` query heads each.
- Each query head rotates its K/V (via RoPE) and attends normally.
- At inference, the KV cache stores only `numKVHeads` worth of K/V — 4× smaller than MHA.

## Used By

| Model | Role |
|-------|------|
| LLaMA-3 8B | 32 query heads, 8 KV heads |
| Mistral-7B | 32 query heads, 8 KV heads |
| Mixtral 8×7B | Same GQA config |
| Gemma, Qwen2 | GQA variants |
| Falcon-7B | 71 query heads, 1 KV head, head dim 64, all 32 layers (MQA extreme: G=1) |
| Fast Transformer Decoder | Single shared K/V ("one write head") for decode speed |
| MobileNetV4 (Mobile MQA) | On-device attention: >39% faster than MHA on mobile accelerators |

## Features

- **Smaller KV cache**: Directly reduces inference memory and memory-bandwidth bottleneck.
- **Quality preservation**: GQA recovers MHA quality; MQA degrades slightly.
- **Drop-in**: Same I/O shape as MHA, so it's a direct swap.

## Evolution

- **Predecessor**: Multi-Head Attention (one KV head per query head).
- **Successor**: Multi-head Latent Attention (MLA, DeepSeek-V2) — compresses KV into a latent vector for even smaller cache.
