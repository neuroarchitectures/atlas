# Tri-Level Hypergraph Contrast

## Design Philosophy

Hypergraph nodes live at three semantic levels: node content, group (hyperedge) cohesion, and node-group membership. TriCL contrasts *all three* between two augmented hypergraph views (membership corruption + feature corruption), with uniform random negatives — so the representation encodes the full incidence structure, not just features.

## Functionality

- Two views via membership and feature corruption.
- Node-, group-, and membership-level InfoNCE-style losses; shared hypergraph encoder; negatives drawn uniformly.

## Used By

| Model | Role |
|-------|------|
| TriCL | Self-supervised hypergraph representation learning |

## Features

- **Three aligned contrast levels** — captures incidence separately from content.
- **Membership corruption** — a hypergraph-specific augmentation.

## Evolution

- **Predecessor**: infonce-contrastive-loss (pairwise graphs), HyperGCL.
- **Related**: deep-graph-infomax — local-global contrast on pairwise graphs.
