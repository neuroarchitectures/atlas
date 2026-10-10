# Replaced Token Detection (ELECTRA)

## Design Philosophy

Masked language modelling trains on only ~15% of tokens and learns from a placeholder, not from distinguishing real text. Replaced-token detection instead labels *every* token as original or replaced, giving a dense, sample-efficient signal — at the cost of needing a generator to produce plausible replacements.

## Functionality

Two networks:

- **Generator** `G`: a small masked LM that proposes a replacement token at each masked position.
- **Discriminator** `D`: a binary classifier deciding, for *every* position, whether the token is the original or a generator sample.

```
x_corrupt = replace( x, positions where G overrides )
loss = BCE( D(x_corrupt), is_original(x) )
```

Only `D` is used downstream; embeddings are typically tied between `G` and `D`.

## Used By

ELECTRA

## Features

- **Dense supervision**: every token gets a training signal.
- **Sample-efficient**: reaches MLM quality with far less compute.
- **Generator + discriminator**: the generator makes the task hard (not random noise).
- **Discriminator reused**: the pretrained encoder transfers directly.

## Evolution

Predecessor: masked language modelling (BERT), noise-contrastive estimation. Sibling: masked language modeling, replaced-token objectives. Successor: ELECTRA-style objectives in efficient pretraining.
