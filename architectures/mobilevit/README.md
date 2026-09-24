# MobileViT

## Overview

Light-weight CNNs are the de-facto choice for mobile vision because their **spatial inductive biases** let them learn representations with few parameters — but they are **spatially local**. ViTs learn global representations but are **heavy-weight**, and shrinking them to a mobile parameter budget is worse than not doing so: at about 5–6M parameters, DeiT is 3% less accurate than MobileNetV3. MobileViT asks whether the two can be combined into one light-weight, low-latency network.

- **Year:** 2021
- **Authors:** Mehta and Rastegari (Apple)
- **Source:** arXiv:2110.02178 — *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer*
- **Category:** DL/Efficient ViT

## Key Characteristics

- **CNN strengths plus ViT strengths** — spatial inductive bias and locality from convolutions, global representation from transformers, in one light-weight and low-latency network.
- **A different perspective on global processing** — MobileViT does not just insert transformers into a CNN; it reformulates how global information is processed with transformers.
- **General-purpose, not classification-only** — the paper emphasizes generality across tasks and datasets rather than a single benchmark.
- **Beats both families at equal parameters** — **78.4% top-1** on ImageNet-1k with about **6M parameters**, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN) and DeIT (ViT) at a similar parameter count.
- **Transfer to detection** — 5.7% more accurate than MobileNetv3 on MS-COCO object detection at a similar number of parameters.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Mehta_et_al._2021_2110.02178.md`](references/papers/Mehta_et_al._2021_2110.02178.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (MobileNetV3, DeiT → MobileViT; siblings: RepViT, MobileNetV4, EfficientViT; contrast: ViT variants merely reduced to mobile size, which perform worse than light-weight CNNs).
