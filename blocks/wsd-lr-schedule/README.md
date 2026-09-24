# WSD Schedule (Warmup–Stable–Decay)

## Design Philosophy

Cosine decay has a fatal operational property: you must know the total training length **before you start**. WSD separates the schedule into three explicit phases — **warmup**, a long **stable** (constant LR) phase, and a short **decay** to zero — so the expensive constant phase can be extended indefinitely and a decay can be run whenever you want a checkpoint. It is the schedule behind "train continuously, anneal on demand" (MiniCPM / Zyda-style recipes).

## Functionality

```
lr(t):
  t < T_warm  : lr_max * t / T_warm            # linear warmup
  T_warm..T_s : lr_max                          # stable (constant)
  decay phase : lr_max * f(progress) -> ~0      # e.g. 1-sqrt, linear, cosine
```

- Decay is usually short (≈5–10% of total steps) and can use linear, `1 - sqrt(x)`, or cosine shape.
- Because the stable phase has no built-in decay, **every intermediate checkpoint is a valid starting point** for a new decay run — the model does not "commit" to a horizon.
- After a decay, you can raise the LR again for another stable phase then decay again (multiple anneals from one run).
- Loss typically spikes when LR goes back up and drops below the previous minimum after each decay.

## Used By

| Architecture | How it is used |
|---|---|
| MiniCPM / InternLM-style LLM pretraining | WSD with repeated decay for continuous training |
| OLMo / Zyda-2 style open recipes | long stable phase + final anneal |
| Long or open-ended vision pretraining | avoids fixing the epoch budget up front |

## Features

- **No pre-committed horizon** — extend training without restarting; the main practical advantage.
- **Cheap checkpoint evaluation** — run a short decay from any stable-phase checkpoint to see the "final" model quality.
- **Re-annealing** — multiple decays from one long run, each producing a usable model.
- **Warmup still required** — the stable phase starts at the full LR, so warmup matters as much as with cosine.
- **Peak LR choice matters more** — with no decay shaping most of training, `lr_max` and weight decay carry the tuning burden.

## Evolution

- **Step decay** (AlexNet/ResNet): manually dropped LR — the original "decay when plateau" approach.
- **Cosine (SGDR 2017)**: smooth decay to zero; requires a fixed total length.
- **WSD (2024, MiniCPM)**: warmup + stable + decay; decouples training length from the schedule.
- **Related**: **OneCycle** (up-then-down in a single cycle, also horizon-bound), **constant-then-linear** variants, **WSD-S (with multiple decays)**.

## See Also

`linear-warmup`, `cosine-lr-schedule`, `onecycle-lr-schedule`, `adamw-optimizer`
