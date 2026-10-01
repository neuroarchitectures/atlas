# Expert Choice Routing

## Design Philosophy

Instead of tokens choosing experts (top-k routing), experts choose tokens. Each expert selects its top-k tokens, ensuring perfect load balancing.

## Functionality

Router produces E x T score matrix. Each expert selects its top-k tokens (column-wise top-k). Tokens can be processed by 0, 1, or multiple experts. No auxiliary loss needed.

## Used By

Expert Choice MoE | EC-ST-MoE

## Features

- **Perfect load balancing**: Each expert processes exactly k tokens.
- **No auxiliary loss**: Load balancing is structural.
- **Variable token assignment**: Tokens can be processed by different numbers of experts.

## Evolution

Predecessor: Top-k routing. Successor: Soft MoE, brain-form MoE.
