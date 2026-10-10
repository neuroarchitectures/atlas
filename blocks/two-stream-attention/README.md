# Two-Stream Attention (XLNet)

## Design Philosophy

Permutation language modelling needs a *query* representation that predicts a target token using only its position, and a *content* representation that encodes the token itself for use as context. A single attention stream conflates the two roles, leaking the target. Two streams separate them.

## Functionality

Run self-attention with two parallel hidden states sharing the same layers:

- **Content stream** `h`: standard attention where each token attends to itself and its allowed context.
- **Query stream** `g`: each token attends to the *content* of its context only — never to its own content, and never through the query stream (a mask forbids `g` attending to `g`).

The query stream output predicts the token; the content stream feeds the next layer and provides keys/values.

## Used By

XLNet

## Features

- **No target leakage**: the query stream cannot see the token it predicts.
- **Permutation-aware**: with a permutation mask, learns all factorisations of the sequence.
- **Shared layers**: both streams reuse the same attention/FFN weights.
- **Relative-position compatible**: pairs with Transformer-XL segment recurrence.

## Evolution

Predecessor: masked LM (BERT), autoregressive LM (GPT) — unifies their objectives. Sibling: masked attention (decoder self-attention). Successor: two-stream designs in permutation and diffusion LMs.
