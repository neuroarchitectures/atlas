# Interleaved NoPE (iRoPE)

## Design Philosophy

Positional encoding ties attention to a length prior. Llama 4 interleaves layers *with* RoPE and layers *without any positional encoding* (NoPE): the position-free layers attend purely by content, giving the model attention layers that extrapolate to extreme context lengths while RoPE layers keep local precision.

## Functionality

- Every k-th layer (Llama 4: every 4th of 48) drops positional encoding entirely; the rest use RoPE.
- NoPE layers act as length-generalization "breathers" in the residual stream.

## Used By

| Model | Role |
|-------|------|
| Llama-4 Scout | 36 of 48 layers RoPE, 12 NoPE; GQA 40:8; 10M-token context claims |

## Features

- **Extrapolation via abstinence** — a positional-encoding-free path is the cheapest length generalizer.
- **No extra parameters** — a layer-pattern change only.

## Evolution

- **Predecessor**: NoPE ablations (2023) showing decoder LMs work without PE; YaRN/RoPE-scaling approaches.
- **Related**: yarn-rope-scaling (gpt-oss) — modify RoPE instead of removing it.
