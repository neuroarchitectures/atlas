# Architecture: CvT

## Motivation

Convolutional Vision Transformer: convolutional token embedding + convolutional projection for efficient ViT.

## Core Idea

Convolutional Vision Transformer: convolutional token embedding + convolutional projection for efficient ViT.

## Architecture

### Overview

![cvt architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | shape=[3, 224, 224] |
| 2 | CvT Stage 1 | `transformer-encoder` | numLayers=1, hiddenSize=64, numHeads=1 |
| 3 | CvT Stage 2 | `transformer-encoder` | numLayers=4, hiddenSize=192, numHeads=3 |
| 4 | CvT Stage 3 | `transformer-encoder` | numLayers=16, hiddenSize=384, numHeads=6 |
| 5 | Classification Head | `linear` | outFeatures=1000 |
| 6 | Class Logits | `output` | — |

</details>

### Components

1. **Image** — input layer. 2. **CvT Stage 1** — transformer-encoder layer. 2. **CvT Stage 2** — transformer-encoder layer. 2. **CvT Stage 3** — transformer-encoder layer. 2. **Classification Head** — linear layer. 2. **Class Logits** — output layer.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Convolutional Vision Transformer: convolutional token embedding + convolutional projection for efficient ViT.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Earlier architectures in the same family. Successor: Improved variants and extensions.

## References

- See references/papers/ for related papers.
