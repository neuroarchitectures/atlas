# SSM-Attention Hybrid Stack

## Design Philosophy

SSMs are cheap per token but weak on precise recall; attention is the reverse. Interleave them — Jamba uses 7 Mamba layers per 1 attention layer — so quadratic cost and KV cache apply to only 1/8 of the stack, while attention layers anchor retrieval quality.

## Functionality

- 8-layer blocks: 7 selective-SSM + 1 GQA attention; MoE on every other FFN layer.
- 32 total layers (4 blocks), 256K context, 16-expert top-2 MoE.

## Used By

| Model | Role |
|-------|------|
| Jamba | First production-scale Mamba-attention hybrid |
| Nemotron | Attention + linear blocks (Mamba2/GLA/Gated DeltaNet/RWKV/SConv) in a NAS-chosen ratio |

## Features

- **Amortized attention** — quality-critical recall from 12.5% of layers.
- **Long context at transformer-like quality** with SSM-side memory footprint.

## Evolution

- **Predecessor**: pure SSM stacks (Mamba), pure attention (Transformer); H3/Hyena early hybrids.
- **Related**: mambavision-mixer and local-global-interleaved-attention — the same hybridization idea in vision.
