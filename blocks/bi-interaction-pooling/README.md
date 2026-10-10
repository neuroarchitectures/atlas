# Bi-Interaction Pooling (NFM)

## Design Philosophy

Second-order feature interactions carry most of the signal in sparse CTR data, but a full outer product is quadratically expensive. Bi-Interaction pooling computes the second-order interaction of embeddings in linear time by summing element-wise products and then taking one square.

## Functionality

Given a set of feature embeddings `{v_i}`, compute

```
f_BI( V ) = sum_{i<j} ( v_i . v_j )          -- naive, O(n^2)
          = 1/2 [ (sum_i v_i)^2 - sum_i v_i^2 ]   -- O(n), element-wise
```

The result is a single vector of the embedding dimension, which is fed to an MLP. FM's second-order term is the special case that sums it into a scalar.

## Used By

NFM

## Features

- **Linear-time second order**: the squared-sum identity avoids the outer product.
- **Vector output**: richer than FM's scalar, feeds a deep network.
- **No extra parameters**: pooling introduces none.
- **Bridges FM and DNN**: a clean second-order-to-MLP interface.

## Evolution

Predecessor: FM (scalar second-order term). Sibling: pairwise dot-interaction, CIN. Successor: higher-order interaction pooling in CTR models (AFM, NFM line).
