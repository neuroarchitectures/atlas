# MultKAN Multiplication Node

## Design Philosophy

KAN layers approximate any function with univariate splines — but products need to be *learned* through that approximation, expensively. KAN 2.0 adds explicit multiplicative nodes: mult layers interleaved between spline KAN layers represent products directly.

## Functionality

- Mult node: takes p inputs, outputs their product (and optionally quotient); placed between KAN layers: input → KAN → Mult → KAN → Mult → KAN.

## Used By

| Model | Role |
|-------|------|
| KAN 2.0 | Mult layers for direct product/quotient structure (e.g., physics formulas) |

## Features

- **Exact symbolic structure** — products, not spline-approximated ones.
- **Convenience nodes** — the library also exposes addition/division analogues.

## Evolution

- **Predecessor**: kan-layer (pure splines); KAN 2.0's "MultKAN" extension.
- **Related**: star-operation-block — multiplicative feature interaction in conv nets.
