# Architecture: BYOL

## Motivation

Contrastive methods need many negative samples and can collapse. BYOL shows that self-supervised learning works without negatives by using a stop-gradient and a momentum-updated target network.

## Core Idea

Two networks: online (updated by gradient) and target (updated by momentum). The online network predicts the target network's output for a different augmented view. Stop-gradient prevents collapse.

## Architecture

### Overview

![byol architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (9 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape: [3, 224, 224] |
| 2 | View 1 | `conv` | augmentation |
| 2 | View 2 | `conv` | different augmentation |
| 3 | Online Encoder | `linear` | updated by gradient |
| 3 | Target Encoder | `linear` | momentum updated |
| 4 | Online Projector | `linear` | MLP projection head |
| 4 | Target Projector | `linear` | momentum updated |
| 5 | Predictor | `linear` | predicts target from online |
| 6 | Stop-Grad Loss | `output` | cosine similarity, stop-grad on target |

</details>

### Components

1. **Online network** — Encoder + projector + predictor. Updated by gradient descent. 2. **Target network** — Encoder + projector (no predictor). Updated by EMA of online. 3. **Stop-gradient** — The target network's output has stop-gradient, preventing the online network from collapsing to a constant. 4. **Symmetric loss** — Both (view1 -> online, view2 -> target) and (view2 -> online, view1 -> target) are computed. 5. **No negatives** — Unlike SimCLR/MoCo, BYOL uses no negative samples.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Two networks: online (updated by gradient) and target (updated by momentum). The online network predicts the target network's output for a different augmented view. Stop-gradient prevents collapse.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: SimCLR, MoCo. Successor: BYOL variants, DINO, data2vec.

## References

- Grill et al. 2020
