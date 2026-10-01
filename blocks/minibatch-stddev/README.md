# Minibatch Standard Deviation

## Design Philosophy

GAN discriminators can fail to detect mode collapse (generating similar samples). Minibatch stddev adds a feature that measures the diversity of the batch, helping the discriminator penalize low-diversity samples.

## Functionality

Compute the standard deviation across the batch for each spatial location and channel. Average over spatial/channel dimensions. Broadcast and concatenate to the feature map.

## Used By

Progressive GAN | StyleGAN discriminator

## Features

- **Mode collapse detection**: Provides the discriminator with batch diversity information.
- **Single feature**: Adds just one extra channel.
- **Simple**: No parameters, just statistics.

## Evolution

Predecessor: Minibatch discrimination (DCGAN). Successor: Various GAN stabilization techniques.
