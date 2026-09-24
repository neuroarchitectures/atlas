# References: CANE

## Papers

- [`CANE_Context_aware_network_embedding_for_relation_modeling_Tu_Liu_Sun_2017.md`](papers/CANE_Context_aware_network_embedding_for_relation_modeling_Tu_Liu_Sun_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **CANE official repository** — https://github.com/thunlp/CANE (official implementation by the authors; includes source code and datasets)
- **THUNLP — Network Representation Learning** — https://github.com/thunlp (Tsinghua THUNLP organization hosts CANE and related NE methods)

## Related

- **DeepWalk** (Perozzi et al., 2014) — https://github.com/phanein/deepwalk — Random-walk + Skip-Gram context-free embedding predecessor (CANE's backbone option).
- **LINE** (Tang et al., 2015) — First/second-order proximity context-free embedding predecessor (CANE's backbone option).
- **node2vec** (Grover & Leskovec, 2016) — https://github.com/aditya-grover/node2vec — Biased random walk context-free embedding predecessor (CANE's backbone option).
- **TADW** (Yang et al., 2015) — Text-Associated DeepWalk; incorporates text via matrix factorization (context-free).
- **MMDW** (Tu et al., 2016) — Max-margin DeepWalk; uses label info (context-free).
- **CENE** (Sun et al., 2016) — Context-enhanced network embedding; treats text as special vertices.
- **GENE** (Chen et al., 2016) — Group-enhanced network embedding.
- **GAT — Graph Attention Networks** (Veličković et al., 2018) — https://github.com/PetarV-/DGI — Successor extending attention-over-neighbors to broader GNNs.
- **Attention mechanisms** (Bahdanau et al., 2014; Vaswani et al., 2017) — The selective/mutual attention foundation CANE builds on.
- **DRNE** (Tu et al., 2018) — Sibling work from the same group (recursive regular-equivalence embedding).
