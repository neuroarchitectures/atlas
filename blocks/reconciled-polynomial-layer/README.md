# Reconciled Polynomial Layer (RPN)

## Design Philosophy

Unify MLP, KAN, SVM, and probabilistic graphical models as special cases of one layer: a Taylor-inspired polynomial whose *data expansion* f(x) and *parameter reconciliation* g(w) are both learned, with a remainder term h(x) — output = f(x) · g(w) + h(x), the product of a data-dependent and a parameter-dependent function.

## Functionality

- f: ℝ^d → ℝ^D (data expansion: identity/conv/graph kernels), g: ℝ^m → ℝ^{D×d′} (parameter reconciliation, m ≪ D×d′), h: ℝ^d → ℝ^{d′} (remainder/identity).
- Stacked layers; v2 adds data-level and structural-level interdependence functions before expansion (subsuming CNN/RNN/GNN/Transformer as choices).

## Used By

| Model | Role |
|-------|------|
| RPN | Base reconciled polynomial network layers |
| RPN-v2 | Adds interdependence functions; identity recovers plain RPN |

## Features

- **One formulation, many architectures** — the unification claim, made concrete.
- **Reconciliation compresses parameters** — m ≪ D×d′.

## Evolution

- **Predecessor**: KAN (learned activations), Taylor networks.
- **Related**: kan-layer — the spline-special case.
