# MetaFormer Block

## Design Philosophy

Most vision backbones are variations on one template: two residual sub-blocks, each `(norm -> token-mixer -> residual)`, where the only thing that changes is the mixer. Framing the architecture this way shows that the *meta*-architecture, not the specific mixer (attention, pooling, conv, MLP), carries much of the performance.

## Functionality

A MetaFormer block is:

```
X = X + Norm1( Mixer(X) )
X = X + Norm2( MLP(X) )
```

where `Mixer` can be any token-mixing operation — self-attention (Transformer), average pooling (PoolFormer), depthwise conv, or an MLP — and `MLP` is a two-layer channel-mixing feed-forward network. Norm is typically LayerNorm or GroupNorm.

## Used By

PoolFormer | EfficientFormer

## Features

- **Mixer-agnostic**: swap the token mixer without redesigning the block.
- **Unified view**: subsumes Transformer and conv blocks.
- **Cheap mixers work**: pooling alone performs competitively.
- **Drop-in**: standard residual + norm + FFN scaffolding.

## Evolution

Predecessor: Transformer encoder block (attention-only mixer), ConvNeXt block (conv mixer). Successor: MetaFormer baselines (PoolFormer, ConvFormer, CAFormer) probing which mixers suffice.
