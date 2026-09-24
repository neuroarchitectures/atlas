# Poincaré Ball Embedding

## Design Philosophy

Hierarchies are trees, but Euclidean space needs O(depth) dimensions to embed them without distortion. Hyperbolic space's exponential distance growth fits hierarchies in *low dimension*: embed nodes in the Poincaré ball, link likelihood from hyperbolic distance `P = 1/(e^d + 1)`, and train with Riemannian SGD — Euclidean gradients scaled by the inverse metric tensor, retracted back into the ball.

## Functionality

- Per node: embedding in the unit ball; negative-sampling cross-entropy over hyperbolic distances.
- Riemannian SGD: gradient scaled by ((1−‖x‖²)/2)² metric factor; retraction `x ← ‖proj‖ < 1 ? proj : x/‖proj‖`.

## Used By

| Model | Role |
|-------|------|
| HGEmb | Hierarchical (taxonomic) network embeddings in low-dim hyperbolic space |

## Features

- **Exponential capacity per dimension** — 2–10 dims embed what Euclidean needs 100s for.
- **Metric-correct optimization** — plain SGD silently violates the geometry.

## Evolution

- **Predecessor**: Nickel & Kiela 2017 (Poincaré embeddings).
- **Related**: gaussian-node-embedding — probabilistic Euclidean alternative; Lorentz-model successors.
