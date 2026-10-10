# External Memory Bank

## Design Philosophy

Store representations in an external memory bank that can be read from and written to. This decouples memory from the model's parameters, enabling dynamic, input-dependent information storage.

## Functionality

Memory M is a matrix of shape [N, d]. Read: attention over M -> retrieved vector. Write: update M at specific locations. Both read and write use content-based addressing (similarity to query).

## Used By

Neural Turing Machine | Memory Network | Differentiable Neural Computer | RAG

## Features

- **Dynamic**: Memory content changes during inference.
- **Content-addressable**: Read/write based on similarity, not position.
- **Unbounded**: Can store arbitrary amounts of information.

## Evolution

Predecessor: LSTM internal memory. Successor: RAG, retrieval-augmented models.
