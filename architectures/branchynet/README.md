# BranchyNet

## Overview

The progression to deeper networks — 8 layers (AlexNet) to 19 (VGGNet) to 152 (ResNet) in four years — dramatically increased the **latency and energy** required for feedforward inference. The cited example: VGGNet versus AlexNet on a Titan X GPU shows a **20× increase in runtime and power** for about a 4% error reduction, and ResNet has an order of magnitude more layers than VGGNet. That trade makes deeper networks less tractable in real-world scenarios such as real-time control, where latency and energy matter.

- **Year:** 2017
- **Authors:** Teerapittayanon et al. (Harvard University)
- **Source:** arXiv:1709.01686 — *BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks*
- **Category:** DL/Dynamic Network

## Key Characteristics

1. **Side branches added to the main branch** — the main branch is the original baseline network; side branches introduce exit points so certain test samples can exit early.
2. **Exploits early-layer sufficiency** — features learned at earlier stages of a deep network can **correctly infer a large subset of the data population**; exiting those samples avoids layer-by-layer processing for all layers.
3. **Entropy-based confidence** — at each exit point the **entropy of the classification result** (e.g. from softmax) measures confidence; below a learned threshold the sample exits with that prediction, above it the sample continues. The last exit point always classifies.
4. **Significant runtime and energy reduction** — for the majority of samples, since most do not need the full depth.
5. **Joint optimization over exit losses** — BranchyNet is trained by solving a joint optimization problem on the **weighted sum of the loss functions associated with the exit points**, so all exits are trained together rather than separately.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Teerapittayanon_et_al._2017_1709.01686.md`](references/papers/Teerapittayanon_et_al._2017_1709.01686.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (AlexNet, VGGNet, ResNet → BranchyNet; siblings: Slimmable, Once-for-All; contrast: fixed-depth networks that process every sample through every layer regardless of difficulty).
