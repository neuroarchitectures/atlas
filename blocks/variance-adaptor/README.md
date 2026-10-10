# Variance Adaptor (FastSpeech 2)

## Design Philosophy

Human speech varies in rhythm, pitch, and loudness independently of the text. A one-to-many TTS model must predict these variations explicitly rather than average them away. The variance adaptor takes phoneme-level encoder states and adds predicted duration, pitch, and energy so the decoder synthesises natural prosody.

## Functionality

Three lightweight predictors run in parallel on the encoder output:

- **Duration predictor**: predicts how many frames each phoneme spans, then a length regulator expands the sequence.
- **Pitch predictor**: predicts an `F0` contour (quantised + embedding, or direct regression).
- **Energy predictor**: predicts the per-frame energy.

Predicted values are embedded and added to the expanded hidden states before the decoder.

## Used By

FastSpeech 2

## Features

- **Explicit prosody control**: duration/pitch/energy are separate knobs.
- **Non-autoregressive**: durations give a deterministic, parallel alignment.
- **Trainable predictors**: variance can be predicted or fed at inference.
- **Naturalness**: avoids the "averaged" prosody of a pure text-to-mel model.

## Evolution

Predecessor: FastSpeech (duration only). Sibling: length regulator, duration predictor. Successor: VITS-style flow/prior prosody models, VALL-E and other codec LM TTS.
