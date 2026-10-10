# Factorized Embedding Parameterization (ALBERT)

## Design Philosophy

In BERT, the token-embedding matrix `V x H` dominates parameter count when the hidden size `H` is large. Decoupling the vocabulary size `V` from the hidden size by projecting through a small embedding dimension `E` shrinks that matrix dramatically.

## Functionality

Embed tokens into a small space `E`, then project to the hidden size `H`:

```
e = W_{H x E} ( Lookup_{V x E}(token) )
```

Because `E << H`, the embedding parameters drop from `V*H` to `V*E + E*H`. Paired with cross-layer parameter sharing, this makes large `H` affordable.

## Used By

ALBERT

## Features

- **Fewer embedding params**: `V*E` instead of `V*H`.
- **Decoupled dimensions**: vocabulary and hidden size set independently.
- **Pairs with parameter sharing**: the projection is part of the shared block.
- **Small quality cost**: with a large enough `E`, accuracy is preserved.

## Evolution

Predecessor: tied input/output embeddings (BERT, weight tying). Sibling: weight tying, cross-layer parameter sharing. Successor: factorized / low-rank embedding layers in compact LMs.
