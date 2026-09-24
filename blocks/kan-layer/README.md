# KAN Layer (Kolmogorov-Arnold Network)

## Design Philosophy

Replace fixed activation functions on nodes with **learnable activation functions (splines) on edges**. The philosophy: MLPs place fixed activations on nodes with learnable linear weights on edges; KANs invert this — no linear weight matrices at all, every "weight" is a learnable univariate function. Inspired by the Kolmogorov-Arnold representation theorem, KANs are more accurate and interpretable on small-scale AI+Science tasks.

## Functionality

A KAN layer with `n_in` inputs and `n_out` outputs:
- Each edge `(i, j)` has a learnable univariate function `φ_{i,j}(x)` parametrized as a residual B-spline: `φ(x) = w_b · b(x) + spline(x)`.
- `x'_j = Σ_{i=1}^{n_in} φ_{i,j}(x_i)` — nodes simply sum incoming edge outputs.
- A depth-L KAN: `KAN(x) = (Φ_L ∘ ... ∘ Φ_1)(x)`.
- **Grid extension**: Refine the spline grid without retraining.
- **Simplification**: L1 sparsification → pruning → symbolification (replace learned splines with sin, exp, etc.).

## Used By

| Model | Role |
|-------|------|
| KAN (Liu et al.) | The defining model |
| KAN 2.0 | Adds multiplication nodes (MultKAN) |
| KAN-SR | KAN-guided symbolic regression |
| TempoRA, KAN-Transformers | Replacing MLP blocks with KAN blocks |

## Features

- **No weight matrices**: Every "weight" is a learnable function.
- **Interpretable**: Splines can be pruned and symbolified into readable formulas.
- **Accurate (small-scale)**: Outperforms MLPs on AI+Science tasks.
- **Grid extension**: Increase resolution without retraining.

## Evolution

- **Predecessor**: MLP (fixed activations on nodes); Kolmogorov-Arnold theorem (1957).
- **Successor**: KAN 2.0 (MultKAN); KAN-Transformers; scaling to large models remains open.
