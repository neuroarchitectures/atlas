# Dynamic Routing (Capsule Network)

## Design Philosophy

Instead of max pooling (which loses spatial hierarchies), capsules use dynamic routing-by-agreement. Lower-level capsules send their outputs to higher-level capsules that agree with them.

## Functionality

1. For each lower capsule i and higher capsule j: prediction u_hat_j|i = W_ij * u_i. 2. Coupling coefficient c_ij = softmax(b_ij). 3. Higher capsule: s_j = sum_i c_ij * u_hat_j|i. 4. Squash: v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||. 5. Update b_ij += u_hat_j|i . v_j. 6. Repeat 3 times.

## Used By

Capsule Network | CapsNet variants

## Features

- **Agreement-based**: Routing is determined by prediction agreement.
- **Iterative**: 3 routing iterations (standard).
- **Preserves hierarchy**: Unlike pooling, maintains part-whole relationships.

## Evolution

Predecessor: Max pooling, attention. Successor: EM routing, self-attention routing.
