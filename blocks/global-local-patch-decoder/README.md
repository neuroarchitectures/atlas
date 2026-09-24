# Global/Local Patch Decoder (MegaByte)

## Design Philosophy

Byte-level AR transformers are O(L²) in raw tokens. MegaByte factorizes: a *global* transformer operates over patch embeddings (O((L/P)²)), while small *local* models autoregressively predict the bytes inside each patch — parallel intra-patch decoding, large per-patch FFNs, and no tokenizer.

## Functionality

- Patch embedder: P bytes → 1 embedding; global model over patches; per-patch local model (small transformer/CNN) conditioned on the global hidden state.
- Inference: global pass once per patch, local passes in parallel within.

## Used By

| Model | Role |
|-------|------|
| MegaByte | Multiscale byte-level generation without tokenization |

## Features

- **Tokenizer-free** — removes vocabulary artifacts, enables any-modality bytes.
- **Sub-quadratic in length** — L² / P² global cost.

## Evolution

- **Predecessor**: ByTet-level LMs (ByT5), hierarchical VQ models.
- **Related**: block-causal-diffusion — parallelism inside blocks via diffusion instead of local AR.
