# Slimmable Neural Networks

## Overview

Many applications — mobile phones, augmented reality devices, autonomous cars — require a short response time. Manually designed lightweight networks and NAS help, but **at runtime these networks are not re-configurable** to adapt across devices given the same response time budget. The scale of the problem: over **24,000 unique Android devices in 2015**, with drastically different runtimes for the same network. High-end phones could run larger models for higher accuracy; low-end phones must sacrifice accuracy.

- **Year:** 2018
- **Authors:** Yu et al. (University of Illinois at Urbana-Champaign, Adobe Research, Facebook AI Research)
- **Source:** arXiv:1812.08928 — *Slimmable Neural Networks*
- **Category:** DL/Dynamic Network

## Key Characteristics

1. **One network executable at different widths** — width meaning the number of channels in a layer; permits **instant and adaptive accuracy-efficiency trade-offs at runtime**.
2. **Shared network with switchable batch normalization** — instead of training individual networks per width configuration, a **shared** network is trained with **switchable batch normalization**; this is the enabling mechanism.
3. **Runtime width adjustment** — the network adjusts its width **on the fly** according to **on-device benchmarks and resource constraints**, rather than downloading and offloading different models.
4. **Matches or beats individually trained models** — similar, and in many cases better, ImageNet accuracy than individually trained MobileNet v1, MobileNet v2, ShuffleNet and ResNet-50 at different widths respectively.
5. **Transfer beyond classification** — better performance than individual models across COCO bounding-box object detection, instance segmentation and person keypoint detection **without tuning hyper-parameters**.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Yu_et_al._2018_1812.08928.md`](references/papers/Yu_et_al._2018_1812.08928.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (manually designed lightweight networks, NAS with on-device latency → Slimmable; siblings: Once-for-All, BranchyNet, MobileNetV1/V2, ShuffleNet; contrast: per-width individually trained models, which are the baselines matched and often beaten).
