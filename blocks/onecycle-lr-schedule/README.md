# OneCycle LR Schedule

## Design Philosophy

Super-convergence (Smith & Topin 2018): with a single LR **cycle** — ramp up to a very high peak, then anneal all the way down below the starting value — plus coupled momentum that moves in the opposite direction, networks can reach the same accuracy in a fraction of the epochs. It is a *fast-training* schedule, not a maximum-quality one: the peak LR is deliberately uncomfortable for the optimizer, which acts as regularization.

## Functionality

```
phase 1 (p% of steps): lr    lr_max/10  -> lr_max      momentum 0.95 -> 0.85
phase 2 (rest)       : lr    lr_max     -> lr_max/1e4  momentum 0.85 -> 0.95
```

- Two phases, symmetric-ish: up for ~30% of steps, down for the rest (fastai default `pct_start=0.3`).
- **Cyclical momentum** is part of the recipe: high LR with low momentum, then the reverse.
- `lr_max` is found with an **LR range test** (increase LR exponentially for one epoch, take the value just before divergence), not guessed.
- At the end the LR is far below the initial value, so the last epochs act as a long fine-grained anneal.
- Optional: cyclical weight decay / cyclical batch size accompany it in the original recipe.

## Used By

| Architecture | How it is used |
|---|---|
| fastai / ResNet CIFAR-ImageNet recipes | the canonical super-convergence schedule |
| Ultralytics-style fine-tuning | OneCycle available as an alternative to linear/cosine |
| Kaggle / short-budget training | strong when the epoch budget is small |

## Features

- **Speed**: the headline property — competitive accuracy in far fewer epochs.
- **Regularization by high LR** — the peak acts like strong noise; often reduces the need for weight decay.
- **Coupled momentum** — required for the effect; LR-only OneCycle loses much of the benefit.
- **Requires an LR range test** — the peak is model- and batch-size-specific; guessing it usually diverges.
- **Not for long pretraining**: at LLM/pretraining scale, cosine or WSD is preferred; OneCycle shines in short fine-tuning runs.

## Evolution

- **Step decay → cosine**: decay-dominated schedules.
- **CLR (Smith 2015)**: cyclical LR between bounds.
- **Super-convergence / OneCycle (2017–2018)**: one cycle with coupled momentum; the fast-training schedule.
- **fastai default (2018+)**: popularized it with `fit_one_cycle`.
- **Related**: **WSD** (also horizon-bound but with a constant phase), **cosine** (safer default), **SGDR** (cosine with warm restarts — repeated cycles).

## See Also

`cosine-lr-schedule`, `wsd-lr-schedule`, `linear-warmup`, `sgd-momentum`
