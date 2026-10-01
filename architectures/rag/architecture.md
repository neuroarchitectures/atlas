# Architecture: RAG Architecture

## Motivation

Parametric language models have fixed knowledge baked into weights, leading to hallucination and inability to update. RAG augments the model with a retriever for external knowledge.

## Core Idea

Encode documents into a dense index using a retriever (DPR). At inference, retrieve top-k documents for the query, then condition the generator (BART/T5) on the query + retrieved documents.

## Architecture

### Overview

![rag architecture architecture](assets/diagram.png)

<details>
<summary><b>Layer-by-layer (6 nodes)</b></summary>

| # | Layer | Type | Params |
|---|---|---|---|
| 1 | Query | `input` | shape: [128] |
| 2 | Query Encoder | `embedding` | DPR encoder, 768 dim |
| 3 | Retriever (DPR) | `identity` | dot product with FAISS index |
| 4 | Retrieved Documents | `embedding` | top-k documents |
| 5 | Generator (BART) | `linear` | seq2seq conditioned on query + docs |
| 6 | Generated Text | `output` |  |

</details>

### Components

1. **Retriever (DPR)** — Dense Passage Retriever: two BERT encoders (one for queries, one for documents). Retrieval via maximum inner product search (MIPS) using FAISS. 2. **Generator** — A seq2seq model (BART or T5) that takes the query and retrieved documents as input and generates the answer. 3. **Two RAG variants** — RAG-Sequence: retrieves once, generates the whole sequence. RAG-Token: retrieves per generated token. 4. **Marginalization** — The generator marginalizes over retrieved documents, weighting by retrieval probability. 5. **Non-parametric memory** — The document index can be updated without retraining the model.

### Data Flow

1. **Input** — The input is fed into the first layer.
2. **Forward pass** — Each layer processes its input and passes the result to the next layer.
3. **Output** — The final layer produces the model's prediction.

### State / Memory

See component descriptions above for state/memory details.

## Design Decisions

1. **Architecture choice** — Encode documents into a dense index using a retriever (DPR). At inference, retrieve top-k documents for the query, then condition the generator (BART/T5) on the query + retrieved documents.
2. **Key tradeoff** — Balance between expressiveness and computational efficiency.

## Evolution

Predecessor: ORQA, REALM. Successor: RAG variants (RETRO, Atlas, In-Context RAG).

## References

- Lewis et al. 2020
