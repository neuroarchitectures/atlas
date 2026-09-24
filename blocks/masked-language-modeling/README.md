# Masked Language Modeling (Cloze)

## Design Philosophy

Directional next-token prediction only sees one side. MLM instead masks ~15% of tokens and predicts them from *both-side* context — bidirectional pretraining that made deep encoder representations transferable. The 80/10/10 corruption mix (MASK/random/keep) mitigates the pretrain-finetune mismatch.

## Functionality

- Select 15% of tokens; replace 80% with [MASK], 10% with random tokens, 10% unchanged; predict the originals from bidirectional context.
- BERT adds Next-Sentence Prediction; RoBERTa drops it and increases masking dynamics.

## Used By

| Model | Role |
|-------|------|
| BERT-base | 12 layers, 768 hidden, post-norm, 512 context |
| BERT4Rec | The same Cloze mechanism applied to item sequences in recommendation |

## Features

- **Bidirectional context** — the defining contrast with causal LM.
- **[MASK] mismatch handled** by the corruption mix.

## Evolution

- **Predecessor**: Cloze tasks, Word2Vec skip-gram.
- **Related**: masked-autoencoder (vision twin); infonce-contrastive-loss as the alternative self-supervision for encoders.
