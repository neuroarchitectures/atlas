# Fast Feedforward Network (FFF)

## Design Philosophy

Replace the dense MLP FFN with a **conditional binary tree** of neurons, where each token traverses only one path (log-depth) from root to leaf. The philosophy: an MLP computes all neurons for all inputs; the FFF routes each input through a single leaf path, so inference cost is O(log N) instead of O(N). The trade-off: training is harder (sparse gradients), but inference is much faster.

## Functionality

- **Binary tree of depth `d`**: `2^d - 1` internal nodes, `2^d` leaves.
- **Routing**: At each internal node, a single neuron computes a value; the sign determines left/right child.
- **Leaf**: A simple linear (or small MLP) at the leaf produces the output.
- **Inference**: Each input traverses exactly one root-to-leaf path → O(d) neurons evaluated.
- **Training**: Soft routing (sigmoid) for gradient flow; hard routing at inference.

## Used By

| Model | Role |
|-------|------|
| FFF (Fast Feedforward Networks) | The defining architecture |
- A drop-in replacement for MLP FFN blocks in Transformers.

## Features

- **O(log N) inference**: Exponentially faster than an O(N) MLP of the same capacity.
- **Sparse compute**: Each token activates only one path.
- **Drop-in**: Same I/O shape as an MLP FFN.

## Evolution

- **Predecessor**: MLP (dense, all neurons computed); Mixture-of-Experts (sparse but flat routing).
- **Successor**: Tree-structured MoE; conditional computation research.
