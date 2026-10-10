# Spatial Transformer Module

## Design Philosophy

A differentiable module that learns spatial transformations (translation, rotation, scale, affine) of feature maps. This enables the network to learn invariance to geometric transformations.

## Functionality

1. Localisation network: predicts 6 affine parameters theta. 2. Grid generator: creates sampling grid from theta. 3. Sampler: bilinearly samples from input at grid locations. Output is the transformed feature map.

## Used By

Spatial Transformer Network | Attention-based localization | Medical image registration

## Features

- **Differentiable**: Gradients flow through the sampling.
- **Arbitrary transformations**: Can learn any affine transform.
- **Pluggable**: Can be inserted anywhere in a CNN.

## Evolution

Predecessor: Pooling (fixed). Successor: Deformable convolution (dense spatial transformation).
