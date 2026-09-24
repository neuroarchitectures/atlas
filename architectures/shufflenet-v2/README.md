# ShuffleNet V2

## Overview

Efficient architecture design was being guided by **FLOPs**, but FLOPs is an indirect metric: speed also depends on **memory access cost and platform characteristics**. The paper makes the case with a concrete example — MobileNet v2 is much faster than NASNET-A at comparable FLOPs, and networks with similar FLOPs measurably differ in speed. ShuffleNet V2 evaluates the *direct* metric (speed on the target platform), derives practical guidelines from controlled experiments, and builds a new architecture from them.

- **Year:** 2018
- **Authors:** Ma et al. (Megvii Inc (Face++), Tsinghua University)
- **Source:** arXiv:1807.11164 — *ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design*
- **Category:** DL/Efficient CNN

## Key Characteristics

- **FLOPs is insufficient as the only metric** — it is an approximation of, but usually not equivalent to, the direct metric of speed or latency, and using it alone can lead to sub-optimal design.
- **Evaluate on the target platform** — the work proposes evaluating the direct metric on the platform rather than only considering FLOPs.
- **Practical guidelines from controlled experiments** — a series of controlled experiments yields guidelines for efficient network design, which the new architecture follows.
- **Direct predecessor of the FLOPS line of work** — the same FLOPs-versus-speed discrepancy that FasterNet later quantifies as FLOPS.
- **State-of-the-art speed/accuracy tradeoff** — verified by comprehensive ablation experiments.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Ma_et_al._2018_1807.11164.md`](references/papers/Ma_et_al._2018_1807.11164.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (Xception, MobileNet, MobileNetV2, ShuffleNet, CondenseNet → ShuffleNet V2; siblings: GhostNet, MobileNetV3, MobileNetV4; contrast: NASNET-A, which matches FLOPs but is much slower).
