# Neural Filter Block (UnivNet)

## Design Philosophy

A neural vocoder must upsample a low-rate mel-spectrogram to a high-rate waveform. Fixed upsampling kernels are generic; generating the filter *kernels* from the input lets the network adapt its upsampling to the signal, at a fraction of the parameters of a large fixed generator.

## Functionality

A kernel predictor (LSTM + convolutions) reads the mel-spectrogram and emits, for each upsampling stage, a set of filter kernels. A stack of transposed convolutions then upsamples using the *predicted* kernels:

```
kernels_k = KernelPredictor(mel)
x_k = TransposedConv( x_{k-1}, weight=kernels_k )
```

A multi-resolution STFT discriminator trains the generator adversarially.

## Used By

UnivNet

## Features

- **Input-conditioned upsampling**: filters adapt to the acoustic content.
- **Parameter-efficient**: small predictors replace huge fixed generators.
- **Parallel waveform synthesis**: no autoregression, single pass.
- **Multi-resolution STFT loss**: stabilises adversarial training.

## Evolution

Predecessor: fixed transposed-conv vocoders (HiFi-GAN generator, WaveNet). Sibling: HiFi-GAN generator. Successor: source-filter and neural-filter vocoders.
