# Architecture: Command R

## Motivation

Enterprise RAG and tool-use applications need models optimized for these tasks specifically, not just general chat. Command R is designed for 128K context, RAG performance, and tool use at scale.

## Core Idea

A 35B parameter dense transformer with 128K context window. Trained with a focus on RAG and tool-use tasks. Uses RoPE positional encoding, SwiGLU activation, and multi-query attention. Optimized for inference efficiency.

## Architecture

### Overview

![command-r architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Prompt + Retrieved Context | `input` | 128K context |
| 2 | Token Embedding | `linear` | 256K vocab -> 4096 dim |
| 3 | Transformer Layers | `linear` | 40 layers, dense |
| 4 | RMSNorm | `identity` | final norm |
| 5 | LM Head | `linear` | 4096 -> 256K vocab |
| 6 | Response + Tool Calls | `output` | text + structured output |

</details>

### Components

1. **Dense transformer** — 35B params, no MoE. 2. **128K context** — long context via RoPE. 3. **Multi-query attention** — efficient inference. 4. **RAG-optimized** — trained on retrieval + generation tasks. 5. **Tool use** — trained to call external tools/APIs. 6. **Cohere For AI** — open weights for research.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — A 35B parameter dense transformer with 128K context window. Trained with a focus on RAG and tool-use tasks. Uses RoPE po...
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: Command (Cohere), LLaMA 2. Successor: Command R+ (104B).

## References

- Cohere 2024
