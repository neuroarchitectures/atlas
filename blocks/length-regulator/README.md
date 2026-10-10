# Length Regulator

## Design Philosophy

A non-autoregressive TTS decoder needs a fixed-length, frame-level sequence, but the text encoder produces a variable-length, phoneme-level sequence. Converting phoneme rate to frame rate without recurrence is a simple expansion operation once each phoneme's duration is known.

## Functionality

Given a phoneme-level hidden sequence `H = [h_1..h_n]` and predicted durations `d = [d_1..d_n]`, repeat each hidden state `d_i` times:

```
H_exp = repeat_interleave(H, d)
```

so the expanded sequence has length `sum_i d_i` frames, aligned to the decoder's frame rate.

## Used By

FastSpeech

## Features

- **Parallel expansion**: no recurrence, ideal for non-autoregressive TTS.
- **Deterministic alignment**: the duration predictor fixes the mapping.
- **Differentiable**: durations can be predicted and trained end-to-end.
- **Simple**: a single gather/repeat op.

## Evolution

Predecessor: autoregressive attention alignment (Tacotron). Sibling: duration predictor (supplies `d`), variance adaptor. Successor: expansion ops in FastSpeech 2, VITS, and other NAR TTS models.
