# Wide & Deep Block

## Design Philosophy

Joint-train a **wide** linear path (memorization of cross features) with a **deep** embedding MLP (generalization), summed at the logit. The philosophy: the wide path memorizes specific feature crosses (good for sparse, co-occurring features); the deep path generalizes to unseen combinations; the sum gets both behaviors in one model. The conceptual ancestor of most hybrid CTR architectures.

## Functionality

- **Wide path**: Crossed features → linear layer (one weight per cross).
- **Deep path**: Sparse categorical inputs → embedding table → flatten → MLP (FC-ReLU-FC-ReLU-FC).
- **Fusion**: `logit = wide_linear(x_wide) + deep_mlp(x_deep)`.
- **Output**: Sigmoid for CTR probability.

## Used By

| Model | Role |
|-------|------|
| Wide & Deep (Google) | Play-store ranking |
- Conceptual ancestor of DeepFM, DCN, and most hybrid CTR architectures.

## Features

- **Memorization + generalization**: The wide path memorizes; the deep path generalizes.
- **Joint training**: Both paths share the embedding table and are trained together.
- **Production-proven**: Deployed at Google scale.

## Evolution

- **Predecessor**: Linear models (memorization only); deep MLPs (generalization only).
- **Successor**: DeepFM (replaces wide crosses with a factorization machine); DCN (learned crosses); these remove the need for manual feature engineering.
