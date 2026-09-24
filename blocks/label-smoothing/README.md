# Label Smoothing

## Design Philosophy

Hard one-hot targets tell the model to be *infinitely* confident about the correct class, which pushes logits apart without bound, overfits the label noise, and wrecks calibration. Label smoothing replaces the target with a mixture of the one-hot label and a uniform distribution: "the answer is class k with probability 1−ε, and anything else with a small residual". The model can never fully close the gap, so logits stay finite.

## Functionality

```
y_ls = (1 - eps) * one_hot(y) + eps / K
loss = CE(y_ls, logits)
```

- Typical `eps`: 0.1 for classification; 0.05–0.1 in detection heads; up to 0.2 in some distillation recipes.
- Equivalent view: CE with a uniform prior + a KL term pulling the output toward uniform.
- **Calibration**: directly improves ECE/calibration on top-1 accuracy datasets; the model's confidence becomes meaningful.
- Interactions: reduces the benefit of knowledge distillation from overconfident teachers; hurts when the number of classes is tiny or when metrics are ranking-based (e.g. retrieval AP can degrade because non-target logits are pushed up together).
- **Not** the same as mixup/cutmix (which smooth inputs), though both regularize confidence.

## Used By

| Architecture | How it is used |
|---|---|
| Inception-v3 / ResNet / EfficientNet | label smoothing 0.1 in the standard ImageNet recipe |
| ViT / DeiT | smoothing in the classification head |
| Transformer NMT / LLMs | smoothing in cross-entropy (often 0.1) |
| DETR-family classification heads | focal loss variants may use smoothing; used carefully because background dominates |

## Features

- **Regularization**: prevents unbounded confidence; mild but reliable accuracy gain on many classification tasks.
- **Calibration**: better expected calibration error — important when the score is thresholded downstream.
- **Robustness to label noise**: lowers the cost of a wrong hard label.
- **Cheap**: one line in the loss; no extra compute.
- **Caveats**: hurts with very few classes, can hurt retrieval/ranking metrics, and interacts with distillation and with temperature scaling.

## Evolution

- **Hard labels / one-hot CE** (the baseline).
- **Label smoothing (Szegedy et al., Inception-v3 2016)**: proposed as a regularizer; later shown to be mostly a calibration effect.
- **Analysis (2019–2020)**: shown to improve calibration and to "erase" logit information (penultimate-layer collapse toward class means).
- **Modern variants**: **online label smoothing** (use the model's own predictions to build the soft target), **zipfian / structured smoothing** (smooth toward similar classes rather than uniformly), and **knowledge distillation** as a learned alternative to uniform smoothing.

## See Also

`focal-loss`, `dice-loss`, `infonce-contrastive-loss`
