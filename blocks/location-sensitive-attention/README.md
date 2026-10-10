# Location-Sensitive Attention

## Design Philosophy

An autoregressive TTS decoder must walk through an utterance in order, never skipping or repeating input characters. Plain content-based attention has no notion of "where we already are", so it can stall or loop. Adding the accumulated attention history as an attention cue keeps decoding monotonic.

## Functionality

Attention uses the previous alignment plus a location-dependent term:

```
e_{i,t} = v^T tanh( W_q q_i + W_k k_t + W_l * F * alpha_{i-1} )
alpha_t = softmax(e_t)
```

where `alpha_{i-1}` is the cumulative attention weight over encoder positions and `F` is a 1-D conv (the "location" filter). The cumulative term biases the next attention toward positions just after the last attended one.

## Used By

Tacotron 2

## Features

- **Monotonic alignment**: discourages skipping/repeating input.
- **Cumulative features**: encodes the attention trajectory, not just the current step.
- **Fast convergence**: stabilises training of the autoregressive decoder.
- **Convolutional location**: a small conv summarises local alignment history.

## Evolution

Predecessor: content-based additive attention (Tacotron 1). Sibling: monotonic / windowed attention, duration-based alignment (FastSpeech). Successor: location-relative attention in later vocoder / TTS systems.
