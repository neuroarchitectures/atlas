# Fine-Grained Distribution Refinement (FDR)

## Design Philosophy

A box coordinate is better described as a *distribution* over discrete offsets than 4 scalars. D-FINE makes each decoder layer refine the previous layer's per-edge probability distribution over quantized box offsets — coarse-to-fine localization where the expected value is the coordinate.

## Functionality

- Per edge (l, t, r, b): distribution over ~bin-count bins; layer ℓ takes layer ℓ−1's distribution, refines it (weights updated toward sharper, more accurate distributions); coordinate = expectation.
- Replaces the scalar box head entirely.

## Used By

| Model | Role |
|-------|------|
| D-FINE | 6-layer decoder refinement; combined with go-lsd-self-distillation |

## Features

- **Uncertainty as a first-class output** — distribution sharpness indicates localization confidence.
- **Free accuracy** — refinement reuses existing decoder layers.

## Evolution

- **Predecessor**: DFL (YOLOv8 distribution focal loss) — the same idea in the conv-head world.
- **Related**: look-forward-twice — supervision alignment across refinement layers.
