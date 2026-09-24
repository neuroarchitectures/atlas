# SConv (Stateful Convolution) Block

## Design Philosophy

Nemotron's PostNAS searched hybrid attention alternatives and kept a compact "stateful convolution": time-mixing of past tokens feeds a generated kernel and a depthwise convolution — a linear-attention-class block that survives the architecture search on quality-per-dollar.

## Functionality

- Time-mix shift of previous token representations → kernel generator (produce per-channel convolution kernels from content) → depthwise conv over the sequence.
- Used as the linear block in hybrid stacks with frozen MLPs and full-attention layers.

## Used By

| Model | Role |
|-------|------|
| Nemotron | Hybrid layers: full attention + linear blocks (Mamba2 / GLA / Gated DeltaNet / RWKV / SConv) chosen by NAS |

## Features

- **NAS-discovered efficiency** — retained because measured latency/quality beat fixed recipes.
- **Content-generated kernels** — input-adaptive convolution without SSM machinery.

## Evolution

- **Predecessor**: RWKV time-mixing, Hyena-style long convolutions.
- **Related**: ssm-attention-hybrid — SConv is one of the mixers such stacks draw from.
