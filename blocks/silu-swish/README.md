# SiLU / Swish

## Design Philosophy

A **self-gated** activation: the input gates itself with its own sigmoid, `x · σ(x)`. Found by automated search (Swish) and used in practice as SiLU, it is smooth, unbounded above, and bounded below — it keeps ReLU's "no saturation for large positive values" property while avoiding the hard zero and the dead-neuron problem.

## Functionality

`SiLU(x) = x · σ(x) = x / (1 + e^{−x})`.

- Non-monotonic: dips slightly below zero near `x ≈ −1.28`, so small negative activations survive.
- Smooth derivative everywhere, which helps optimization in very deep nets.
- `σ(βx)` with a learnable `β` is the "Swish-β" variant; `β = 1` is the standard.

## Used By

| Model | Role |
|-------|------|
| EfficientNet / EfficientNetV2 | Conv activation (replacing ReLU) |
| MobileNetV3 / MobileNetV4, YOLOv5–v11 | Conv-BN-SiLU block |
| ConvNeXt, RTMPose-style CSPNeXt backbones | Conv activation |
| SwiGLU FFN (LLaMA, Mistral, Qwen) | The gate function inside the gated MLP |

## Features

- **Smooth, self-gated, unbounded above** — combines properties of ReLU and gating.
- Typically a small but consistent accuracy gain over ReLU in CNNs, at negligible cost.
- Its most important modern use is as the **gate in SwiGLU**, not as a standalone conv activation.

## Evolution

- **Predecessors**: ReLU, LeakyReLU, GELU (same qualitative shape).
- **Itself**: Swish (Ramachandran et al., 2017, NAS-discovered) / SiLU (same function, independent naming).
- **Successors**: gated activations built on it — **SwiGLU** (`(xW) · SiLU(xV)`), now the default LLM FFN; also Hard-Swish for mobile-quantized inference.
