# Dice Loss (and Soft Dice)

## Design Philosophy

Segmentation is evaluated with region overlap (Dice/IoU), but pixel-wise cross-entropy optimizes per-pixel accuracy — and is dominated by background pixels when the foreground is small. Dice loss optimizes the **metric itself** on the *soft* prediction: maximize overlap between the predicted probability map and the target mask. It is region-level, so a small foreground is not swamped by background.

## Functionality

```
Dice(p, y) = 2 * sum(p * y) / (sum(p) + sum(y) + eps)     # p: probabilities, y: binary/one-hot
DiceLoss   = 1 - Dice
```

- Works directly on probabilities (no thresholding), so it is differentiable — the "soft Dice".
- Per-class: compute per channel and average (standard for multi-class), which handles class imbalance naturally.
- **Combine with CE**: the usual recipe is `CE + Dice` — CE gives stable per-pixel gradients (especially early), Dice pushes region overlap.
- Sensitive to empty targets: with `eps` smoothing, an absent class can still receive gradient; masking empty classes out is a common fix.
- Gradient degenerates when both prediction and target are near zero — CE in the sum keeps early training stable.
- Variant: **Tversky loss** weights FP vs. FN (recall-biased), **Focal-Tversky** adds focusing.

## Used By

| Architecture | How it is used |
|---|---|
| U-Net / V-Net | the original Dice-loss architectures (medical, small foreground) |
| SAM | dice loss combined with focal loss for mask supervision |
| nnU-Net, SegFormer fine-tuning, nnDetection | CE + Dice default for medical segmentation |
| SAM 2 / video object segmentation | mask supervision on sparse, tiny objects |

## Features

- **Imbalance-robust**: region-level normalization means a 1%-area object still contributes fully.
- **Metric-aligned**: directly optimizes Dice, the reported metric.
- **Differentiable on probabilities** — no post-processing needed.
- **Empty-class and tiny-object pathologies**: needs `eps` and usually a CE partner.
- **Non-convex / noisy early**: pure Dice can stall; CE + Dice is the safe default.

## Evolution

- **Cross-entropy** (per-pixel) → **Dice / soft-Dice (2016, V-Net; U-Net 2015 used weighted CE)** for imbalanced regions.
- **Tversky (2017)**: asymmetric FP/FN weighting — recall-biased Dice.
- **Focal-Tversky / combo losses**: add focusing to hard pixels.
- **IoU / Lovász-softmax (2017)**: surrogate for the IoU metric itself.
- **Unified Focal Loss (2021)**: combines focal + asymmetric region terms.
- **Current practice**: `CE (or Focal) + Dice` is the default segmentation recipe; Dice alone is rare.

## See Also

`focal-loss`, `giou-loss`, `hungarian-matching-loss`
