# Position-Dependent Hyperedge Embedding

## Design Philosophy

In an n-ary relation, the *slot* matters — (user, item, tag) is not (tag, item, user). Shift or convolve entity embeddings by a position-specific transform (HSimplE's shift / HypE's positional convolution) before scoring, making n-ary relation scoring position-aware and fully expressive.

## Functionality

- Score(v₁..v_n) = f( transform_1(e_1), ..., transform_n(e_n), r ) — per-slot transforms are learned.
- HSimplE: additive position shifts; HypE: position-indexed convolution filters.

## Used By

| Model | Role |
|-------|------|
| KHG (Knowledge Hypergraph) | n-ary relation scoring over entity + relation embeddings |

## Features

- **Slot-aware symmetry breaking** — the same entity contributes differently per position.
- **Fully expressive scoring** under the paper's analysis.

## Evolution

- **Predecessor**: SimplE/ComplEx pairwise scoring (positions fixed by construction).
- **Related**: tuplewise-hyperedge-similarity — deep non-linear scoring over the whole tuple.
