# MobileNetV3

## Overview

The stated goal is the **best possible mobile computer vision architecture optimizing the accuracy-latency tradeoff** on mobile devices. The method is to blend **automated search with novel architecture advances**, which the paper breaks into four contributions: complementary search techniques, new mobile-practical nonlinearities, new efficient network design, and a new efficient segmentation decoder.

- **Year:** 2019
- **Authors:** Howard et al. (Google AI, Google Brain)
- **Source:** arXiv:1905.02244 — *Searching for MobileNetV3*
- **Category:** DL/Efficient CNN

## Key Characteristics

1. **Complementary search techniques** — the paper reviews the efficient building blocks for mobile models and the complementary nature of the **MnasNet and NetAdapt** algorithms; NetAdapt is complementary to MnasNet and can be combined with it.
2. **New efficient nonlinearities** — mobile-practical versions of nonlinearities, notably **h-swish**, designed for the mobile setting where exact implementations of some activations are costly.
3. **New efficient network design** — architecture improvements that raise the efficiency of the models found by the joint search.
4. **New efficient segmentation decoder** — LR-ASPP, a lightweight decoder for mobile semantic segmentation.
5. **Thorough per-technique evaluation** — experiments across a wide range of use cases and mobile phones demonstrate the efficacy and value of each element separately.

Defines two released models: **MobileNetV3-Large** and **MobileNetV3-Small** for high- and low-resource use cases, intended to power next-generation on-device computer vision.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Howard_et_al._2019_1905.02244.md`](references/papers/Howard_et_al._2019_1905.02244.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (SqueezeNet, MobileNetV1, MobileNetV2, ShuffleNet, CondenseNet, ShiftNet, MnasNet, NetAdapt → MobileNetV3; successors: MobileNetV4; siblings: GhostNet, ShuffleNet V2; contrast: post-hoc compression methods — pruning, quantization and distillation — which are upper-bounded by the pre-trained networks they start from).
