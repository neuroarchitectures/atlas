# Copy Mechanism

## Design Philosophy

Output vocabularies are closed, but inputs are open — names, numbers, node labels. A copy mechanism lets the decoder *copy tokens directly from the input* (here: graph nodes) instead of always generating from the vocabulary, blending a generation distribution with a pointer distribution.

## Functionality

- Per step: generation probability p_gen gates between vocabulary softmax and attention-weighted pointer over input tokens/graph nodes.
- Graph2Seq specifics: attention-based decoder over GCN node states; coverage vector tracks copied usage; decoder init from mean node state.

## Used By

| Model | Role |
|-------|------|
| Graph2Seq | Graph-to-text generation with copy from node labels/content |

## Features

- **Open-vocabulary output** grounded in the input.
- **Coverage control** — discourages re-copying the same node.

## Evolution

- **Predecessor**: pointer-generator networks (See et al. 2017), copy nets (Gu et al.).
- **Related**: retrieval-augmented generation — copying spans to copying documents.
