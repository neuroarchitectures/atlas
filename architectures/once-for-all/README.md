# Once-for-All Network

## Overview

Efficient inference across many devices and resource constraints has a scaling problem: conventional approaches either manually design or use NAS to find a specialized network and **train it from scratch for each case**, which is computationally prohibitive — the paper cites \(CO_2\) emission as much as **5 cars' lifetime** — and thus unscalable. The optimal architecture also varies with hardware, and even on the same hardware under different battery conditions or workloads.

- **Year:** 2019
- **Authors:** Cai et al. (MIT)
- **Source:** arXiv:1908.09791 — *Once-for-All: Train One Network and Specialize it for Efficient Deployment*
- **Category:** DL/Dynamic Network

## Key Characteristics

1. **Train one, specialize many** — an OFA network supports diverse architectural settings by **decoupling training and search**, so cost drops.
2. **Sub-networks without additional training** — a specialized sub-network is obtained quickly by **selecting from the OFA network**; no retraining.
3. **Progressive shrinking** — a novel algorithm described as a **generalized pruning method** that reduces model size across **many more dimensions than pruning: depth, width, kernel size, and resolution**.
4. **A surprisingly large design space** — over **\(10^{19}\) sub-networks** fitting different hardware platforms and latency constraints, **while maintaining the same level of accuracy as training independently**.
5. **Beats SOTA NAS on real devices** — up to **4.0% ImageNet top-1 improvement over MobileNetV3**, or same accuracy but **1.5× faster than MobileNetV3** and **2.6× faster than EfficientNet** w.r.t. measured latency, while reducing **many orders of magnitude** of GPU hours and \(CO_2\) emission.
6. **New SOTA under mobile setting** — **80.0% ImageNet top-1** under <600M MACs; winning solution of the 3rd LPCVC (DSP classification) and 4th LPCVC (classification and detection).

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Cai_et_al._2019_1908.09791.md`](references/papers/Cai_et_al._2019_1908.09791.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (manual design, NAS, pruning → Once-for-All; siblings: Slimmable, BranchyNet, MobileNetV3; contrast: per-case NAS trained from scratch, which is the unscalable baseline).
