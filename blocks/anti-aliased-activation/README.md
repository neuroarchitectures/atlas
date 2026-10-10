# Anti-Aliased Activation (BlurPool)

## Design Philosophy

Pointwise nonlinearities introduce high-frequency content that the sampling grid cannot represent, producing aliasing artifacts (jaggies, ringing) when the signal is later downsampled. Filtering the signal before the nonlinearity, rather than after, prevents the aliasing from ever being generated.

## Functionality

Low-pass filter the feature map with a fixed blur kernel (e.g. a binomial `[1,2,1]/4` separable filter), then apply the activation:

```
y = f( blur(x) )
```

The blur is a depthwise fixed convolution, so it adds no parameters. It is applied both before downsampling convs and before nonlinearities to keep the network band-limited.

## Used By

BigVGAN

## Features

- **Alias-free**: enforces the sampling theorem before any resolution change.
- **Zero parameters**: fixed binomial kernel.
- **Translation- and shift-equivariant**: removes the texture sticking artifact from strided ops.
- **Cheap**: one separable depthwise conv per nonlinearity.

## Evolution

Predecessor: plain activation + strided conv. Sibling: Snake activation (periodic) — combined in BigVGAN as an anti-aliased periodic activation. Successor: alias-free GAN generators (StyleGAN3).
