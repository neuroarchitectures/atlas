# Stochastic Depth (DropPath)

## Design Philosophy

Very deep residual networks are *over-parameterized for their gradient path*: with identity skip connections, a residual block can be removed at inference without breaking the network. Stochastic depth exploits that by **dropping entire residual branches during training** — each sample sees a shallower network — which regularizes, speeds up training, and (unlike dropout on activations) costs nothing at inference because all branches are kept.

## Functionality

```
# training, for residual block i with survival probability p_i
if bernoulli(p_i): x = x + f(x)
else:              x = x              # branch skipped

# inference: x = x + p_i * f(x)       # expectation; or just keep all branches
```

- **Linear decay rule** (original): survival probability decreases linearly from 1.0 at the input to ~0.5 at the last block — early layers are rarely dropped, deep layers often.
- Typically paired with a **higher learning rate** and longer training, since the effective network is shallower.
- Distinct from **Dropout**: dropout zeroes individual activations; drop-path zeroes a whole module's contribution. Distinct from **DropBlock**: that drops contiguous spatial regions of a feature map.
- Works on any residual/branch architecture: ResNet, ResNeXt, EfficientNet, ViT, ConvNeXt, MLP-Mixer.
- No inference cost: the full network is used at test time (scaling by survival probability is optional and usually unnecessary with the decay rule).

## Used By

| Architecture | How it is used |
|---|---|
| ResNet-1202 / ResNeXt | original stochastic depth (the paper that made 1000+ layers trainable) |
| EfficientNet / EfficientNetV2 | drop connect rate scaled with model depth/width (compound scaling) |
| ViT / DeiT / Swin | standard `drop_path` hyperparameter (0.1–0.5) |
| ConvNeXt, MLP-Mixer, gMLP | per-block drop path |
| NAS / supernet training | drop path as a way to train many sub-networks at once |

## Features

- **Regularization**: strong on deep models; often replaces dropout entirely in ViT-style architectures.
- **Training speed**: skipped branches are not computed → real wall-clock savings (up to ~25% in the original work).
- **Implicit ensemble**: training samples many depths; inference uses the full network (an expectation over them).
- **Depth-scaled rate**: deeper models need a higher drop rate; a single global value tuned per architecture.
- **Free at inference** — unlike dropout, no scaling trick needed if the decay rule is used consistently.

## Evolution

- **Dropout (2012)**: drop activations.
- **DropConnect (2013)**: drop weights.
- **Stochastic Depth (Huang et al., 2016)**: drop residual branches, linear survival decay.
- **DropBlock (2018)**: drop contiguous spatial regions (better for conv nets).
- **ViT / DeiT (2020)**: drop path becomes a standard transformer regularization knob alongside weight decay and layer scale.
- **Related**: **layer scale / stochastic depth on MLP branches only**, **Nested Dropout**, **PatchDropout** (drop input patches in MAE-style training).

## See Also

`layer-norm`, `ema-weights`, `gradient-checkpointing`
