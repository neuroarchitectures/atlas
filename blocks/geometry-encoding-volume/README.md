# Geometry Encoding Volume (GEV)

## Design Philosophy

A raw correlation volume says *how similar* two pixels are, but nothing about *geometry*. IGEV fuses all-pairs correlation with geometry/context information into one volume that encodes non-local 3D structure — good enough to regress an initial disparity directly, then indexed by ConvGRU iterations instead of blind refinement from zero.

## Functionality

- Build group-wise correlation from CNN features; combine with geometry/context encodings (combined correlation + geometry volume).
- Regress initial disparity (soft-argmax) from GEV; ConvGRU iterations index the GEV at warped positions for refinement.

## Used By

| Model | Role |
|-------|------|
| IGEV-Stereo | Initial disparity regression + iterative refinement over GEV |

## Features

- **Informed initialization** — iterations start near the answer, cutting count drastically.
- **Geometry-enriched matching** — better than raw correlation in textureless regions.

## Evolution

- **Predecessor**: RAFT's all-pairs-correlation-pyramid (pure similarity).
- **Related**: FlowFormer cost memory (learned retrieval from cost volume); CFNet initial-disparity regression.
