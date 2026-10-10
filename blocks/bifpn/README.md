# BiFPN (Weighted Bidirectional FPN)

## Design Philosophy

Multi-scale feature fusion should let the network learn *how much* each input scale matters rather than summing them equally. BiFPN treats feature fusion as weighted, bidirectional, and repeatable — a stackable module rather than a single top-down pass.

## Functionality

A BiFPN layer has a top-down path and a bottom-up path. Each intermediate node receives weighted contributions from its neighbors:

```
out = Conv( sum_i( w_i / (eps + sum_j w_j) ) * in_i )
```

The weights `w_i` are learnable scalars (fast normalized fusion). Nodes that have only one input edge (no fusion) are dropped to save compute; each bidirectional layer is repeated to build the feature pyramid.

## Used By

EfficientDet

## Features

- **Learnable scale weights**: fast normalized fusion, no softmax needed.
- **Bidirectional**: information flows top-down *and* bottom-up.
- **Stackable**: repeated layers improve accuracy at modest cost.
- **Efficient**: prunes single-input nodes; fewer channels per node.

## Evolution

Predecessor: FPN (top-down, equal weighting), PANet (adds bottom-up). Successor: bidirectional fusion variants in real-time detectors.
