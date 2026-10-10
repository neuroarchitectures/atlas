# Architecture: PatchMixer

## Motivation

Point-based time series models lose local context. Patching captures local patterns but needs efficient mixing.

## Core Idea

Divide the time series into patches, then use MLP-mixer style architecture to mix across patches (time) and across channels (variables) separately.

## Architecture

### Overview

![patchmixer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Time Series | `input` | shape: [96, 7] |
| 2 | Patch Embedding | `patchify` | patchLen: 16 |
| 3 | Time Mixing | `mlp-mixer` |  |
| 4 | Channel Mixing | `mlp-mixer` |  |
| 5 | Projection | `linear` |  |
| 6 | Forecast | `output` |  |

</details>

### Components

1. **Patch embedding** — Divide input into non-overlapping patches, project to embedding space. 2. **Time mixing** — MLP across the patch dimension. 3. **Channel mixing** — MLP across the channel dimension. 4. **Linear projection** — Map to forecast horizon.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Divide the time series into patches, then use MLP-mixer style architecture to mix across patches (time) and across chann
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Patch-TST, MLP-Mixer. Successor: TimeMixer.

## References

Chen et al. 2023
