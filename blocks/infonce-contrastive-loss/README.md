# InfoNCE (Contrastive) Loss

## Design Philosophy

Learn representations by **comparison instead of reconstruction**: pull an anchor toward its positive and push it away from a large set of negatives, in a shared embedding space. InfoNCE is a multi-class classification over "which of these candidates is the positive", and it lower-bounds mutual information — which is why it works for self-supervised learning (SimCLR, MoCo), multimodal alignment (CLIP), and grounding (region-text in Grounding DINO).

## Functionality

`L = −log( exp(sim(a, p) / τ) / ( exp(sim(a, p) / τ) + Σ_n exp(sim(a, n) / τ) ) )`

- `sim` is usually cosine similarity after L2 normalization of both sides.
- `τ` (temperature) is critical: too high flattens the distribution (no hard negatives), too low makes training brittle; often learned.
- **Negatives matter more than positives**: performance scales with batch size / queue size / memory-bank size, hence MoCo's momentum queue and large-batch SimCLR.
- Superset of several losses: with a single negative it reduces to logistic BCE; symmetric versions average both directions.

## Used By

| Model | Role |
|-------|------|
| SimCLR, MoCo / MoCo-v3, BYOL variants | Self-supervised image representation |
| CLIP / ALIGN / SigLIP | Image-text alignment (bidirectional) |
| Grounding DINO / GLIP / DetCLIP | Region-text contrastive classification (open-set detection) |
| DINO (self-distillation), V-JEPA | Self-supervised representation / latent prediction |

## Features

- **No labels required** except the pairing signal; works with in-batch negatives.
- Scales with **negative count** — the main engineering constraint (memory bank, queue, big batch).
- Temperature and augmentation policy dominate final quality.

## Evolution

- **Predecessors**: triplet loss / margin ranking, NCE (Noise Contrastive Estimation), softmax classification over noise samples.
- **Itself**: InfoNCE / CPC (van den Oord et al., 2018).
- **Successors**: supervised contrastive (SupCon), SigLIP (sigmoid loss — no global normalization over negatives), SigLIP-2, and region-level contrastive losses in open-vocabulary detectors.
