# Rational-Quadratic Spline Flow

## Design Philosophy

Normalizing flows need invertible transforms that are both flexible and cheap to invert. A monotonic rational-quadratic spline — a piecewise function of quadratic-over-quadratic segments — is exactly invertible in closed form and can approximate any monotone mapping, giving flows far more expressiveness per coupling layer than affine coupling.

## Functionality

Partition the domain into `K` bins with boundary knots; on each bin define

```
y = ( a_i z^2 + b_i z + c_i ) / ( d_i z^2 + e_i z + f_i )
```

Monotonicity is guaranteed by constraining the derivatives, and the spline's parameters are produced by a neural conditioner network from the input. The inverse requires only solving a quadratic per element.

## Used By

Neural Spline Flow

## Features

- **Exact invertibility**: closed-form forward and inverse.
- **Highly expressive**: many bins per dimension, learnable knot heights.
- **Linear cost in bins**: cheap density evaluation via log-derivative.
- **Coupling-friendly**: plugs straight into affine-coupling-style layers.

## Evolution

Predecessor: affine coupling (RealNVP), invertible 1x1 conv (Glow). Successor: spline-based flow matches and autoregressive flows (MAF/SOS).
