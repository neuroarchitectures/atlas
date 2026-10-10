# Architecture: Neural Turing Machine

## Motivation

Neural networks lack external memory — they store everything in their weights. RNNs have limited memory through hidden state. The NTM augments a neural controller with a large external memory matrix, enabling the network to read and write information, like a Turing machine with a differentiable tape.

## Core Idea

A controller network (LSTM or feedforward) interacts with an external memory matrix via read and write heads. Read heads use content-based addressing (similarity to keys) and location-based addressing (shifting). Write heads erase and add to memory locations. All operations are differentiable, enabling end-to-end training.

## Architecture

### Overview

![ntm architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (8 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Input | `input` | shape: [1, 10] |
| 2 | Controller (LSTM) | `lstm` | hiddenSize: 100, numLayers: 1 |
| 3 | Read Head | `custom` | type: read_head, memorySize: 128, wordSize: 20 |
| 4 | Write Head (Erase+Add) | `custom` | type: write_head, memorySize: 128, wordSize: 20 |
| 5 | Memory Matrix M | `custom` | type: external_memory, size: 128, wordSize: 20 |
| 6 | Read Vector r | `custom` | type: read_vector |
| 7 | Output | `linear` | outFeatures: 10, inFeatures: 110 |
| 8 | Output | `output` |  |

</details>

A controller network (LSTM or feedforward) interacts with an external memory matrix via read and write heads. Read heads use content-based addressing (similarity to keys) and location-based addressing (shifting). Write heads erase and add to memory locations. All operations are differentiable, enabling end-to-end training.

### Components

2. **Controller (LSTM)** (`lstm`, scope: `controller`) — Params: hiddenSize: 100, numLayers: 1
3. **Read Head** (`custom`, scope: `memory`) — Params: type: read_head, memorySize: 128, wordSize: 20
4. **Write Head (Erase+Add)** (`custom`, scope: `memory`) — Params: type: write_head, memorySize: 128, wordSize: 20
5. **Memory Matrix M** (`custom`, scope: `memory`) — Params: type: external_memory, size: 128, wordSize: 20
6. **Read Vector r** (`custom`, scope: `memory`) — Params: type: read_vector
7. **Output** (`linear`, scope: `output`) — Params: outFeatures: 10, inFeatures: 110

### Data Flow

The architecture processes input through a sequence of 8 components, with residual connections where applicable. Each component transforms the representation, building up the final output.

### State / Memory

See component descriptions above for state/memory details specific to Neural Turing Machine.

## Evolution

NTM introduced differentiable external memory. Successors: Memory Networks (attention-based read), Differentiable Neural Computer (DNC, improved addressing), and retrieval-augmented generation (RAG). The concept of external memory influenced modern transformer-XL and long-context architectures.

## Source

- **Paper:** arXiv:1410.5401
- **Year:** 2014
- **Authors:** Graves et al.
