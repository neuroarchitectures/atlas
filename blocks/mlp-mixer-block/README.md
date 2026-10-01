# MLP-Mixer Block

## Design Philosophy

Replace attention with two MLPs: one that mixes across spatial locations (token-mixing) and one that mixes across channels (channel-mixing). No attention, no convolution.

## Functionality

1. LayerNorm, transpose, token-mixing MLP (across S patches), transpose back, residual. 2. LayerNorm, channel-mixing MLP (across C channels), residual. Both MLPs have 2 layers with GELU.

## Used By

MLP-Mixer | gMLP (with spatial gating) | ResMLP

## Features

- **No attention**: Pure MLP architecture.
- **Position-independent**: Token-mixing MLP treats all positions equally.
- **O(S^2 + C^2)**: Parameters scale with patches and channels.

## Evolution

Predecessor: ViT (attention-based). Successor: gMLP, ResMLP, ConvMixer.
