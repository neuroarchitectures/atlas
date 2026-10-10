# MILE Coarsening-Refinement Block

## Overview

The core block of [MILE](../mile/) — a multi-level framework for scalable graph embedding. It expands the three phases of MILE into their individual operations: hybrid-matching graph coarsening (SEM + NHEM), base embedding on the coarsest graph using any existing method, and a learned graph-convolution refinement that propagates embeddings from the coarsest graph back to the original. The companion to the full [mile](../mile/) entry.

See `architecture.md` for detailed analysis and `references/` for source materials.

## Core Idea

Repeatedly coarsen a large graph into progressively smaller ones while preserving its backbone, embed the coarsest graph with an out-of-the-box method (DeepWalk, Node2Vec, GraRep, NetMF), then learn a graph convolution that refines those coarse embeddings back to every original node. MILE is agnostic to the underlying embedding technique.

## Key Characteristics

- **Hybrid matching coarsening** — combines Structural Equivalence Matching (SEM), which merges structurally equivalent nodes, with Normalized Heavy Edge Matching (NHEM), which merges nodes joined by heavy edges.
- **Matching matrices** — each level stores `M_{i,i+1}` with `A_{i+1} = M^T A_i M`, so embeddings can be propagated back up the hierarchy.
- **Black-box base embedding** — any existing graph embedding method is applied only to the smallest graph, making the cost of the expensive step negligible.
- **Learned refinement** — a graph convolution network learns to lift coarse embeddings to finer levels, improving quality via the holistic multi-level view.
- Scales to graphs with millions to billions of nodes that neither random-walk nor matrix-factorization methods can handle directly.

## Files

| File | Description |
|------|-------------|
| [`model.json`](model.json) | Structural model graph (nodes, layers, parameters). |
| [`assets/diagram.svg`](assets/diagram.svg) | Vector architecture diagram. |
| [`assets/diagram.png`](assets/diagram.png) | Raster architecture diagram. |
| [`architecture.md`](architecture.md) | Architecture analysis and documentation. |
| [`references/`](references/) | Source materials (papers, related work). |

## Related Architectures

See [`references/papers/`](references/papers/) for the source papers.
