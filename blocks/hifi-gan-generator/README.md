# HiFi-GAN Generator

## Design Philosophy

Non-autoregressive audio generation using transposed convolutions for upsampling and multi-receptive field (MRF) blocks for high-fidelity waveform generation.

## Functionality

1. Transposed conv (upsample by stride). 2. MRF block: parallel residual blocks with different kernel sizes and dilations, summed. 3. Repeat for each upsampling stage. 4. Final 1x1 conv to audio.

## Used By

HiFi-GAN | VITS (decoder) | Audio generation

## Features

- **Fast**: Non-autoregressive, real-time generation.
- **MRF**: Multiple receptive fields for diverse patterns.
- **High quality**: Matches or exceeds autoregressive models.

## Evolution

Predecessor: WaveNet (autoregressive). Successor: BigVGAN, VITS2.
