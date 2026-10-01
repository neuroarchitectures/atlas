# Squash Function (Capsule)

## Design Philosophy

Non-linear activation for capsules that ensures short vectors get shrunk to near zero and long vectors get bounded to length < 1. This represents existence probability (length) and instantiation parameters (direction).

## Functionality

v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||). For ||s|| -> 0, v -> 0. For ||s|| -> inf, ||v|| -> 1. The function is continuous and differentiable.

## Used By

Capsule Network | All capsule-based architectures

## Features

- **Bounded**: Output length is always in [0, 1).
- **Non-linear**: Short vectors are suppressed more than long ones.
- **Direction-preserving**: Only scales, doesn't change direction.

## Evolution

Predecessor: ReLU, sigmoid. Successor: EM routing (no explicit squash).
