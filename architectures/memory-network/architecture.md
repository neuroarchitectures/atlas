# Architecture: Memory Network

## Motivation

RNNs and LSTMs compress all context into a fixed-size hidden state, losing information over long sequences. Memory Networks address this by maintaining an explicit memory bank of facts, with attention-based retrieval for question answering.

## Core Idea

Store input facts in a memory bank. For a given query, compute attention weights over all memory slots, then read a weighted sum of memory contents. The output is produced by combining the query with the retrieved memory. Multi-hop reasoning is achieved by chaining multiple read operations.

## Architecture

### Overview

![memory-network architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (10 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input Facts | `input` | shape: [10, 100] |
| 2 | Query | `input` | shape: [1, 100] |
| 3 | Input Memory Embed | `embedding` | numEmbeddings: 100, embeddingDim: 100 |
| 4 | Query Embed | `embedding` | numEmbeddings: 100, embeddingDim: 100 |
| 5 | Attention (dot product) | `custom` | type: content_addressing |
| 6 | Softmax | `softmax` |  |
| 7 | Weighted Sum (Read) | `custom` | type: weighted_read |
| 8 | Query + Memory | `add` |  |
| 9 | Output Layer | `linear` | outFeatures: 50, inFeatures: 100 |
| 10 | Answer | `output` |  |

</details>

Store input facts in a memory bank. For a given query, compute attention weights over all memory slots, then read a weighted sum of memory contents. The output is produced by combining the query with the retrieved memory. Multi-hop reasoning is achieved by chaining multiple read operations.

### Components

3. **Input Memory Embed** (`embedding`, scope: `memory`) — Params: numEmbeddings: 100, embeddingDim: 100
4. **Query Embed** (`embedding`, scope: `memory`) — Params: numEmbeddings: 100, embeddingDim: 100
5. **Attention (dot product)** (`custom`, scope: `hop.0`) — Params: type: content_addressing
6. **Softmax** (`softmax`, scope: `hop.0`) — Params: none
7. **Weighted Sum (Read)** (`custom`, scope: `hop.0`) — Params: type: weighted_read
8. **Query + Memory** (`add`, scope: `hop.0`) — Params: none
9. **Output Layer** (`linear`, scope: `output`) — Params: outFeatures: 50, inFeatures: 100

### Data Flow

The architecture processes input through a sequence of 10 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Memory Network.

## Evolution

Memory Networks introduced explicit memory with attention-based retrieval. Successors: End-to-End Memory Networks (differentiable multi-hop), Key-Value Memory Networks, and retrieval-augmented generation (RAG). The attention mechanism over memory directly inspired transformer attention.

## Source

- **Paper:** arXiv:1410.3916
- **Year:** 2014
- **Authors:** Weston et al.
