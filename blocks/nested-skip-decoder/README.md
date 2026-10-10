# Nested Skip Decoder (UNet++)

## Design Philosophy

The original U-Net has one skip path per resolution, so the decoder sees only single-scale encoder features and the semantic gap between encoder and decoder at a given level is large. UNet++ densifies the skip connections into a nested grid, letting every decoder node fuse features from multiple encoder depths.

## Functionality

Arrange nodes on a grid `X^{i,j}` (encoder index `i`, decoder depth `j`). Each node receives the dense skip connection:

```
X^{i,j} = H( [ X^{i,0}, X^{i,1}, ..., X^{i,j-1} ] )
```

where `H` is a conv block, and `[ ]` concatenates all same-row predecessors. Interleaved up/down sampling aligns the resolutions. Deep supervision branches off at several depths.

## Used By

UNet++

## Features

- **Dense skips**: each decoder node sees features from all shallower levels.
- **Reduced semantic gap**: encoder and decoder features are closer.
- **Deep supervision**: intermediate branches speed convergence and improve accuracy.
- **Prunable**: shallower sub-networks can be extracted at inference.

## Evolution

Predecessor: U-Net (single skips), DenseNet (dense connectivity). Successor: nested/dense decoders in segmentation.
