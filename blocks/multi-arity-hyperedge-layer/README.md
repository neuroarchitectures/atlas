# Multi-Arity Expand-Reduce Hyperedge Layer

## Design Philosophy

Real hypergraphs mix arities — unary attributes, pairwise edges, triple relations. Process each arity with its own expand-reduce operations, then combine across arities by concatenation/permutation over sparse hyperedge tensors, so relations of different order coexist in one layer.

## Functionality

- Per arity a ∈ {0..3}: expand node/hyperedge features to the arity's tensor, apply the arity's reduce op.
- Cross-arity combination via concat and permutation; sparse tensor representation scales to 10K+-node graphs.

## Used By

| Model | Role |
|-------|------|
| SAL-HR | SpaLoc reasoning layers over sparse hypergraphs; with Hoyer sparsity pruning and information-sufficiency subgraph sampling |

## Features

- **Native mixed-arity support** — no flattening hyperedges to pairwise.
- **Sparse-tensor efficiency** — sub-linear in the dense hyperedge count.

## Evolution

- **Predecessor**: uniform-arity hypergraph convs (HGNN, THNN).
- **Related**: topological-message-passing — cell-complex generalization of mixed-dimension propagation.
