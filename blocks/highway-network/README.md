# Highway Network

## Design Philosophy

Very deep plain networks degrade because gradients vanish or explode on the way back. A highway network gives each layer a learned, element-wise gate that decides how much of the input to carry through unchanged versus transform, creating a data-dependent shortcut like an LSTM's cell path applied to feed-forward layers.

## Functionality

For a layer `H(x)` and a transform gate `T(x) = sigmoid(W_T x + b_T)`:

```
y = T(x) . H(x) + (1 - T(x)) . x
```

The carry term `(1 - T)` lets information bypass the nonlinearity, so gradients flow along the identity path when `T` is near zero. Multiple layers form a highway; a residual connection is the special case `T = 1`.

## Used By

Tacotron 2

## Features

- **Learned gates**: the network chooses how much to transform.
- **Gradient highway**: identity path mitigates vanishing gradients.
- **Element-wise**: each unit is gated independently.
- **Generalises residuals**: residual add is the `T ≡ 1` special case.

## Evolution

Predecessor: plain feed-forward / LSTM gating. Successor: residual connection (simplified, gating removed), and gated FFNs (GLU, SwiGLU, GeGLU).
