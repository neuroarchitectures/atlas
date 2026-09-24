# Meta-Path Guided Random Walk

## Design Philosophy

In heterogeneous graphs, arbitrary walks mix node types meaninglessly. Constrain walks to a *node-type sequence* (meta-path, e.g., APA, APVPA) so generated context respects heterogeneous semantics — a paper's authors and a venue's papers become comparable through the shared path schema.

## Functionality

- Walks transition only along edges matching the meta-path's next type.
- metapath2vec++: negative samples drawn from the *same node type* as the positive context, forcing within-type discrimination; walk length ~100, 10 walks/node, k=5 negatives.

## Used By

| Model | Role |
|-------|------|
| metapath2vec / metapath2vec++ | Heterogeneous network embeddings (d=128) |

## Features

- **Type-consistent context** — the semantic backbone of heterogeneous embedding.
- **Type-aware negatives** (++) — sharpen within-type boundaries.

## Evolution

- **Predecessor**: DeepWalk on homogeneous projections.
- **Related**: HAN semantic-level attention; metapath-relation-embeddings (HIN2Vec classification formulation).
