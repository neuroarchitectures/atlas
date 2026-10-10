# Coupling Layer (RealNVP)

## Design Philosophy

Invertible transformation for normalizing flows. Split the input into two parts, transform one part conditioned on the other, and keep the second part unchanged. This makes the Jacobian triangular and easy to compute.

## Functionality

Split x = (x1, x2). y1 = x1 * exp(s(x2)) + t(x2), y2 = x2. The Jacobian is lower triangular with diagonal exp(s(x2)), so log|det(J)| = sum(s(x2)).

## Used By

RealNVP | Glow (with modifications) | Multi-scale flows

## Features

- **Exact invertibility**: y -> x is trivially computed.
- **Trivial Jacobian**: Log-determinant is just sum of s.
- **Expressive**: s and t can be arbitrary neural networks.

## Evolution

Predecessor: NICE (additive coupling). Successor: Glow (1x1 invertible conv), Residual flows.
