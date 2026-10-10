# Compressed Interaction Network (CIN)

## Design Philosophy

Feature interactions in CTR prediction can be modelled explicitly as a compressed, high-order tensor. Rather than enumerating every cross product, CIN keeps the interaction field at vector level and compresses it at each layer, giving an explicit but tractable order-vs-cost trade-off.

## Functionality

At each layer `k`, take the outer product of the previous layer's feature matrix `X^{k-1}` with the embedding matrix `X^0`, then compress:

```
Z^{k} = compress( outer( X^{k-1}, X^0 ) )      # outer product then conv/1x1
```

The compress step (a shared 1x1 kernel per layer) keeps the hidden dimension bounded; each layer's output is summed into the final representation. Vector-level (not bit-level) interactions are expanded layer by layer.

## Used By

xDeepFM

## Features

- **Explicit high-order interactions**: order grows with depth.
- **Compressed**: 1x1 compression bounds the parameter count.
- **Vector-wise**: preserves embedding structure, unlike bit-wise cross layers.
- **Complementary to DNN**: concatenated with a deep branch.

## Evolution

Predecessor: Cross network (DCN), FM. Sibling: DeepFM, DCN-v2. Successor: hybrid explicit/implicit interaction models in CTR (AutoInt, xDeepFM line).
