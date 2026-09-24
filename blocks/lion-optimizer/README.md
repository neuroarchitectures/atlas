# Lion Optimizer

## Design Philosophy

Discovered by symbolic program search over optimizer update rules (AutoML-zero style), Lion is a **sign-based** optimizer: it keeps only the *sign* of the momentum-of-gradients, so every parameter is updated by the same magnitude. That makes updates bounded, uniform across layers, and cheap — one momentum buffer instead of Adam's two.

## Functionality

```
c_t = beta2 * m_{t-1} + (1 - beta2) * g_t     # "momentum" (interpolated gradient)
theta_t = theta_{t-1} - lr * sign(c_t)        # uniform-magnitude update
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
```

- Typical `beta1 = 0.9`, `beta2 = 0.99`; learning rate is usually **3–10× smaller** than AdamW's because the update magnitude is fixed at `lr`.
- Weight decay is applied **separately** (decoupled, like AdamW), and usually at a larger value.
- State: one buffer (momentum) → roughly **half** AdamW's optimizer memory.
- The update magnitude being independent of gradient magnitude is both the strength (bounded, stable) and the weakness (no per-parameter adaptivity).

## Used By

| Architecture | How it is used |
|---|---|
| ViT / diffusion (Google Brain reports) | Lion matched or beat AdamW at lower memory |
| Language modeling | reported to save pretraining compute |
| Vision fine-tuning recipes | drop-in alternative to AdamW where optimizer memory matters |

## Features

- **Memory**: 1 buffer vs. AdamW's 2 → smaller optimizer state, important for billion-parameter models (and for on-device training).
- **Uniform update size**: `sign()` means every parameter moves by `lr` — this is what gives Lion its unexpectedly good generalization.
- **Quantization/robustness**: bounded updates behave better in low-precision training loops.
- **Hyperparameters differ**: do not reuse AdamW's lr/wd — scale lr down (≈1/10 as a starting point) and raise weight decay.
- **Batch-size sensitivity**: large batches work better than small ones.

## Evolution

- **SGD / momentum** (uniform-ish, non-adaptive) → **Adam / AdamW** (adaptive, 2 buffers).
- **SignSGD / Signum (2018)**: sign-based updates with momentum — Lion's direct ancestor, known to be robust in distributed/compressed settings.
- **Lion (2023, Google Brain)**: discovered by search; sign of the interpolated momentum, decoupled weight decay.
- **Related**: **Muon** (orthogonalized momentum for 2D weight matrices; strong for transformer training), **Sophia** (second-order, clipped diagonal Hessian), **Adafactor** (factored second moments for memory).

## See Also

`adamw-optimizer`, `sgd-momentum`, `cosine-lr-schedule`, `mixed-precision-amp`
