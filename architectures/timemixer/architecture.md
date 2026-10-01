# Architecture: TimeMixer

## Motivation

Time series contain patterns at multiple temporal scales (micro, medium, macro). Existing methods handle scales separately. TimeMixer explicitly mixes information across scales.

## Core Idea

Decompose the series into multiple scales using downsampling. Use Past-Future Mixing (MLP for temporal prediction) and Cross-Scale Mixing (MLP for inter-scale information flow) at each scale.

## Architecture

### Overview

![timemixer architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [96, 1] |
| 2 | Scale 1 (raw) | `identity` | length 96 |
| 2 | Scale 2 (downsample 2x) | `identity` | length 48 |
| 2 | Scale 3 (downsample 4x) | `identity` | length 24 |
| 3 | Past-Future Mix | `linear` | MLP mapping past to future at each scale |
| 4 | Cross-Scale Mix | `linear` | MLP mixing across scales |
| 5 | Forecast | `output` |  |

</details>

### Components

1. **Multi-scale decomposition** — Average pooling at different rates (1, 2, 4, ...) creates multiple temporal scales. 2. **Past-Future Mixing (PFM)** — An MLP at each scale that maps the past segment to the future segment. 3. **Cross-Scale Mixing (CSM)** — An MLP that combines forecasts from different scales, enabling information flow from macro to micro scales. 4. **Pure MLP architecture** — No attention or recurrence; all mixing is done via MLPs for efficiency.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Decompose the series into multiple scales using downsampling. Use Past-Future Mixing (MLP for temporal prediction) and Cross-Scale Mixing (MLP for inter-scale information flow) at each scale.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: N-BEATS, MLP-based forecasting. Successor: TimeMixer variants.

## References

- Wang et al. 2024
