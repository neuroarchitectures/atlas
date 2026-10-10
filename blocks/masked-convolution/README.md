# Masked Convolution (PixelCNN)

## Design Philosophy

Autoregressive image models need each pixel to depend only on previously generated pixels. A convolution can enforce this causality by zeroing the kernel weights that would look "ahead" in raster order — no recurrence or masking at the activation level is required.

## Functionality

For a `k x k` conv with the current pixel at the centre, mask the kernel:

- **Type A** (first layer): zero the centre weight, so a pixel never sees itself.
- **Type B** (subsequent layers): keep the centre weight, since the center already depends only on earlier pixels.

The mask is fixed and applied by element-wise multiplying the kernel before each forward pass.

## Used By

PixelCNN

## Features

- **Exact likelihood**: gives a tractable factorisation `p(x) = prod p(x_i | x_<i)`.
- **Parallel training**: unlike RNNs, all pixels train in one pass.
- **Blind spot**: a plain NxN masked conv misses some past pixels; gated variants fix this.
- **Receptive field grows with depth**: stacking widens the causal context.

## Evolution

Predecessor: fully-visible convs (non-causal), NADE. Successor: Gated PixelCNN, PixelCNN++, masked attention (Transformer decoders).
