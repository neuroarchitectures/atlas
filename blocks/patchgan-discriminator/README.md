# PatchGAN Discriminator

## Design Philosophy

Deciding whether the *whole* image is real wastes capacity on global structure that a per-pixel L1 loss already handles. Much of image realism lives in local patches. A PatchGAN classifies each local patch independently, giving a discriminator that focuses on high-frequency texture and is cheap to run.

## Functionality

Run a fully-convolutional discriminator that outputs an `N x N` grid of real/fake logits instead of a single scalar:

```
D(x) in R^{N x N}     # one logit per receptive-field patch
```

The loss averages the per-patch adversarial decisions. No fully-connected head is used; the effective receptive field of each output unit defines the patch size.

## Used By

Pix2Pix

## Features

- **Local realism**: models texture via overlapping patches.
- **Fully convolutional**: applies to any input size.
- **Cheap**: far fewer parameters than a global classifier.
- **Markovian assumption**: good where local statistics drive realism.

## Evolution

Predecessor: global discriminator (whole-image real/fake). Sibling: multi-scale / multi-period discriminators. Successor: PatchGAN used widely in super-resolution, style transfer, and vocoders.
