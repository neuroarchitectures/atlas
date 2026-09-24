# Multi-Query Attention (MQA)

## Design Philosophy

In autoregressive decoding, the KV cache — not compute — is the bottleneck. Let all query heads share *one* K/V head: quality barely drops, but the cache and the memory bandwidth shrink H-fold.

## Functionality

- H query heads, 1 shared K/V head; same softmax attention math with broadcast K/V.
- GQA (existing block) is the interpolating middle ground (g KV groups).

## Used By

| Model | Role |
|-------|------|
| Falcon-7B | 71 query heads, 1 KV head, head dim 64, all 32 layers |
| Fast Transformer Decoder | Single shared K/V ("one write head") for decode speed |
| MobileNetV4 (Mobile MQA) | On-device attention: >39% faster than MHA on mobile accelerators |

## Features

- **H× smaller KV cache** — direct decode throughput gain.
- **Mobile-friendly** — MQA's memory-bound profile suits NPUs better than GQA in some regimes.

## Evolution

- **Predecessor**: multi-head-attention.
- **Successor**: grouped-query-attention (quality/cache trade-off), multi-head-latent-attention (compress the cache instead of sharing).
