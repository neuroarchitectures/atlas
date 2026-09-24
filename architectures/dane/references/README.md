# References: DANE

## Papers

- [`Attributed_network_embedding_for_learning_in_a_dynamic_envir_Li_Dani_Hu_etal_2017.md`](papers/Attributed_network_embedding_for_learning_in_a_dynamic_envir_Li_Dani_Hu_etal_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **DANE — Jundong Li (ASU) group** — https://jundongl.github.io/ (author's page; check for code/data releases related to DANE)
- **Huan Liu's group (ASU — Data Mining and Machine Learning)** — https://www.public.asu.edu/~huanliu/ — Co-author group page with related dynamic network embedding resources.

## Related

- **DeepWalk** (Perozzi et al., 2014) — https://github.com/phanein/deepwalk — Static plain-network embedding predecessor.
- **LINE** (Tang et al., 2015) — First/second-order proximity static embedding predecessor.
- **node2vec** (Grover & Leskovec, 2016) — https://github.com/aditya-grover/node2vec — Biased random walk static embedding predecessor.
- **TADW** (Yang et al., 2015) — Text-associated DeepWalk; incorporates attributes but static.
- **AANE** (Huang et al.) — Attributed network embedding; static.
- **Spectral graph embedding / matrix factorization** — The algebraic foundation DANE's offline consensus model and perturbation updates build on.
- **Matrix perturbation theory** (Stewart & Sun) — The mathematical tool enabling DANE's efficient online update.
- **CTNE** (Nguyen et al., 2018) — Successor extending dynamic embedding to continuous-time temporal walks.
- **Dynamic GNNs / temporal graph networks** (TGAT, TGN, EvolveGCN) — Later neural methods for evolving graphs that build on the dynamic-attributed framing DANE introduced.
- **Incremental SVD / online matrix factorization** — Methods adopting DANE's offline-base + online-update paradigm.
