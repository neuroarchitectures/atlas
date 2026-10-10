# Large-Kernel Convolution (RepLKNet)

## Design Philosophy

Vision transformers owe much of their advantage to large effective receptive fields. ConvNets were long limited to small kernels for efficiency. RepLKNet shows that a very large depthwise kernel (e.g. 31x31) can be used directly if it is made cheap, and that training large kernels (with small-kernel re-parameterization) beats stacking many 3x3 layers.

## Functionality

A large-kernel block is a depthwise `K x K` conv (`K` up to 31) surrounded by pointwise convs:

```
y = PW2( DW_{KxK}( PW1(x) ) )
```

Structural re-parameterization optionally adds parallel small-kernel and identity branches during training, then merges them into the single large kernel at inference.

## Used By

RepLKNet

## Features

- **Huge receptive field**: 31x31 depthwise kernel in one layer.
- **Cheap**: depthwise, so FLOPs stay modest.
- **Re-parameterizable**: small-kernel branches fold into the large kernel.
- **ConvNet scaling**: closes the gap to ViTs without attention.

## Evolution

Predecessor: 3x3 ConvNets, structural re-parameterization (RepVGG). Sibling: RepVGG blocks, depthwise separable conv. Successor: large-kernel ConvNets (SLaK, ConvNeXt) and hybrid conv/attention backbones.
