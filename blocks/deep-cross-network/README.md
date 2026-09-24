# Deep & Cross Network (DCN / DCN-v2)

## Design Philosophy

Learn **explicit feature crosses** of bounded degree through a cross network, paired with a deep MLP. The philosophy: Wide & Deep needs manual crosses; DCN learns them. DCN-v2 upgrades the cross from rank-1 (a weight vector) to a full matrix (a learned linear), making crosses far more expressive, with a low-rank factorization to control cost.

## Functionality

**DCN (rank-1 cross)**: `x_{l+1} = x_0 ⊙ (w · x_l) + x_l` — element-wise cross with the input `x_0`, rank-1 weight `w`.

**DCN-v2 (matrix cross)**: `x_{l+1} = x_0 ⊙ (W x_l + b) + x_l` — full weight matrix `W`, far more expressive. In practice, `W` is low-rank: `W = U V^T` (bottleneck) to control parameters at production feature counts.

- **Cross network**: Stacked cross layers, each applying the cross with `x_0`.
- **Deep network**: Parallel MLP.
- **Fusion**: Concatenate cross + deep → logit.

## Used By

| Model | Role |
|-------|------|
| DCN (Google) | CTR with learned rank-1 crosses |
| DCN-v2 | Production successor, matrix crosses |
- Deployed across Google ad and feed ranking.

## Features

- **Learned crosses**: No manual feature engineering.
- **Bounded degree**: L cross layers model up to L-th order interactions.
- **Low-rank (v2)**: Bottleneck keeps the matrix cross affordable at scale.

## Evolution

- **Predecessor**: Wide & Deep (manual crosses); DeepFM (FM = 2nd order only).
- **Successor**: xDeepFM (compressed interaction); AutoInt (attention-based interactions).
