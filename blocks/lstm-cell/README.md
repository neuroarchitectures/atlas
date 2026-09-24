# LSTM Cell

## Design Philosophy

Maintain constant error flow through a self-connected cell (the **constant error carousel, CEC**) with weight 1.0 and identity activation, enforced by multiplicative **gates** that learn to open and close access. The philosophy: the vanishing gradient problem is solved by giving the gradient a path that doesn't attenuate, and gates provide context-sensitive control over what enters, stays in, and leaves the memory cell.

## Functionality

- **Cell state `c_t`**: The long-term memory, updated by `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g(x_t)`. The self-connection weight is 1.0, so gradients flow without vanishing.
- **Forget gate `f_t`**: `σ(W_f · [x_t, h_{t-1}])` — controls what to erase.
- **Input gate `i_t`**: `σ(W_i · [x_t, h_{t-1}])` — controls what to write.
- **Output gate `o_t`**: `σ(W_o · [x_t, h_{t-1}])` — controls what to read.
- **Hidden output**: `h_t = o_t ⊙ f(c_t)`, where `f` is tanh (or identity).

## Used By

| Model | Role |
|-------|------|
| LSTM (Hochreiter & Schmidhuber) | The defining cell |
| GRU | Simplified variant (update + reset gates, no separate cell) |
| DIEN | GRU-based interest evolution |
| EnCodec | LSTM in the bottleneck of the audio codec |
| CNN-LSTM-1D | LSTM on top of a conv feature extractor |

## Features

- **Long-term memory**: The CEC enables 1000+ step dependencies.
- **Gating**: Multiplicative gates resolve input/output weight conflicts.
- **O(1) per step**: Constant time per time step, but sequential (not parallelizable across sequence).

## Evolution

- **Predecessor**: Vanilla RNN (Elman, 1990) — suffers vanishing gradients.
- **Successor**: GRU (simplified); xLSTM (exponential gates + matrix memory); Transformers (parallel, replaced LSTM for most tasks); Mamba (selective SSM).
