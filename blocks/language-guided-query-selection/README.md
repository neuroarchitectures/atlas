# Language-Guided Query Selection

## Design Philosophy

For open-vocabulary detection the text prompt should decide *where the decoder looks*. Rank encoder image features by their similarity to the text prompt and select decoder queries from the top-ranked ones — the language condition enters at query initialization, not just at classification.

## Functionality

- Feature enhancer: bidirectional image↔text cross-attention over backbone/FPN features and sub-sentence text embeddings.
- Query selection: score image features by prompt relevance; top features seed the decoder's content queries.

## Used By

| Model | Role |
|-------|------|
| Grounding DINO | Decoder init from text-ranked image features; combined with region-text contrastive head |

## Features

- **Prompt-conditioned attention budget** — queries concentrate on prompt-relevant regions.
- **Zero-shot transfer** — selection works for unseen category names.

## Evolution

- **Predecessor**: GLIP (language-image pre-training as detection); two-stage deformable DETR query selection.
- **Successor**: SAM 3's concept prompting generalizes selection to noun phrases + exemplars.
