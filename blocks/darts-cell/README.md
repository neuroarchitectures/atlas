# DARTS Cell (Differentiable NAS)

## Design Philosophy

A cell is a DAG where each edge is a mixture of candidate operations weighted by softmax-normalized architecture parameters. The mixture enables gradient-based optimization of the architecture.

## Functionality

Each edge: o_bar(x) = sum_i softmax(alpha_i) * o_i(x), where o_i are candidate operations (3x3 conv, 5x5 conv, max pool, avg pool, identity, zero). Alphas are optimized via bilevel optimization.

## Used By

DARTS | PC-DARTS | DrNAS | Differentiable NAS

## Features

- **Differentiable**: Architecture is optimized via gradient descent.
- **Bilevel optimization**: Network weights on train set, alphas on validation set.
- **Discretization**: Final architecture selects the best operation per edge.

## Evolution

Predecessor: ENAS (RL), NASNet (evolution). Successor: PC-DARTS, DARTS+, DrNAS.
