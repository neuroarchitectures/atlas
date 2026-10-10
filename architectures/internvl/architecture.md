# Architecture: InternVL

## Motivation

Vision encoders (CLIP ViT-L) are small compared to LLMs (7B+). InternVL scales the vision encoder to 6B params (InternViT-6B), matching LLM capacity for better multimodal understanding.

## Core Idea

InternViT-6B (a massive ViT) extracts rich visual features. A projection layer maps them to LLM space. An LLM (InternLM) processes visual + text tokens. Progressive alignment training connects vision and language.

## Architecture

### Overview

![internvl architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (7 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Image | `input` | 3x448x448 |
| 2 | InternViT (6B) | `conv2d` | 6B param ViT |
| 3 | Projection | `linear` | 3200 -> 4096 |
| 4 | LLM Embedding | `linear` | text tokens |
| 5 | Concat | `identity` | visual + text tokens |
| 6 | LLM | `linear` | InternLM, 7B+ |
| 7 | Text Output | `output` | autoregressive |

</details>

### Components

1. **InternViT-6B** — 6B parameter vision transformer. 2. **Progressive alignment** — connect to multiple LLMs (LLaMA, InternLM). 3. **QLoRA** — efficient fine-tuning. 4. **Dynamic resolution** — handles variable image sizes. 5. **Scale matching** — vision encoder capacity matches LLM. 6. **Open data** — trained on open datasets.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — InternViT-6B (a massive ViT) extracts rich visual features. A projection layer maps them to LLM space. An LLM (InternLM)...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: CLIP (small vision encoder), LLaVA. Successor: InternVL 1.5, InternVL2.

## References

- Chen et al. 2023
