# Architecture: DeepSeek-R1

## Motivation

LLMs can reason but typically need supervised fine-tuning on chain-of-thought data. DeepSeek-R1 shows that pure RL (GRPO) can elicit reasoning capabilities, producing long chain-of-thought traces without human demonstrations.

## Core Idea

Start from DeepSeek-V3 base. Apply Group Relative Policy Optimization (GRPO) with rule-based rewards (correctness, format). The model learns to generate extended reasoning traces before answers. R1-Zero uses pure RL; R1 adds cold-start SFT.

## Architecture

### Overview

![deepseek-r1 architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Prompt | `input` | text tokens |
| 2 | Token Embedding | `linear` | 128K vocab -> 4096 dim |
| 3 | Transformer Layers | `linear` | 61 layers, MoE |
| 4 | RMSNorm | `identity` | final normalization |
| 5 | LM Head | `linear` | 4096 -> 128K vocab |
| 6 | Reasoning + Answer | `output` | CoT + answer |

</details>

### Components

1. **DeepSeek-V3 base** — MoE architecture with 671B total / 37B active params. 2. **GRPO** — Group Relative Policy Optimization, no critic needed. 3. **Rule-based rewards** — correctness + format, no learned reward model. 4. **Emergent reasoning** — model learns to produce long CoT traces via RL. 5. **R1-Zero** — pure RL, no SFT. 6. **R1** — cold-start SFT + multi-stage RL. 7. **Distillation** — smaller models distilled from R1 outputs.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Start from DeepSeek-V3 base. Apply Group Relative Policy Optimization (GRPO) with rule-based rewards (correctness, forma...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: DeepSeek-V3, OpenAI o1. Successor: DeepSeek-R1-Lite, distilled variants.

## References

- DeepSeek-AI 2025
