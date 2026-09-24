# FlowFormer

## Overview

RAFT-family flow models look up local cost volumes with convolutions and rely on many iterative refinements. FlowFormer instead **tokenizes the 4D cost volume** into a compact **cost memory** with a transformer encoder (using alternating-group attention to keep the cost tractable), then decodes flow with **recurrent dynamic positional cost queries** that retrieve from that memory.

- **Year:** 2022
- **Authors:** Huang et al.
- **Source:** arXiv:2203.16194 — *FlowFormer: A Transformer Architecture for Optical Flow*
- **Category:** DL/Optical Flow

## Key Characteristics

- **Cost volume encoder → cost memory** — the 4D cost volume is encoded into a compact latent memory, which is the transformer-native replacement for a raw correlation volume.
- **Alternating-group transformer** — makes attention over the cost volume affordable by splitting into groups alternately along the two axes.
- **Dynamic positional cost queries** — for each source pixel, a query is built from the local 9×9 cost patch around the *current* flow estimate plus positional embedding; it attends over the cost memory.
- **Recurrent refinement** — a ConvGRU regresses flow residuals iteratively; keys/values are computed once and reused, which is the stated efficiency benefit of the design.
- **Convex upsampling** — flow is estimated at reduced resolution and upsampled learnably, supervised at every iteration.

- See [`architecture.md`](architecture.md) for detailed architecture analysis.
- See [`model.json`](model.json) for the structural model graph.
- See [`assets/diagram.svg`](assets/diagram.svg) / [`diagram.png`](assets/diagram.png) for the architecture diagram.
- Primary source paper: [`Huang_et_al._2022_2203.16194.md`](references/papers/Huang_et_al._2022_2203.16194.md)

## Related Architectures

See `architecture.md` → Evolution section for predecessor/successor relationships (PWC-Net → RAFT → FlowFormer; siblings: GMFlow (global matching), SEA-RAFT, GMA; successor: FlowFormer++).
