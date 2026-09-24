# Xavier (Glorot) Initialization

## Design Philosophy

Keep the **variance of activations and of the back-propagated gradients equal across layers**. Glorot and Bengio derive the compromise `Var(W) = 2 / (fan_in + fan_out)` — the harmonic mean of fan-in and fan-out — which simultaneously preserves signal forward and gradient backward. It assumes a **linear or symmetric (tanh/sigmoid)** activation, which is why it predates and differs from Kaiming init.

## Functionality

Weights drawn from `U(−√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))` or `N(0, 2/(fan_in+fan_out))`.

- `fan_in`, `fan_out`: number of input and output units (for convs, times the receptive field size).
- Biases zero; gain factors exist for specific nonlinearities (`tanh`, `sigmoid`, `relu`, `linear`).
- Applies to fully connected and convolutional layers alike.

## Used By

| Model | Role |
|-------|------|
| Early deep nets (tanh/sigmoid MLPs) | Weight init |
| Transformer / BERT linear projections | Default PyTorch init for `nn.Linear` |
| Attention and FFN projection matrices | Init of `W_q, W_k, W_v, W_o, W_1, W_2` |
| LSTM/GRU weight matrices | Orthogonal or Xavier init variants |

## Features

- **Symmetric variance preservation** — good default when no normalization layer follows.
- Analytical and cheap; no tuning required.
- **Assumes an approximately linear activation**: with ReLU it under-scales by ~2×, which is exactly what Kaiming init corrects.

## Evolution

- **Predecessors**: uniform/normal small-random init used before 2010, which saturates sigmoid layers.
- **Itself**: Glorot & Bengio, 2010 (Understanding the difficulty of training deep feedforward neural networks).
- **Successors**: Kaiming/He init (rectifier-aware), orthogonal init (RNNs), Fixup / T-Fixup / LayerScale (norm-free deep nets), and truncated-normal small init for pretrained transformer stacks.
