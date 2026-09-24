# References: DRNE

## Papers

- [`Deep_recursive_network_embedding_with_regular_equivalence_Cui_Wang_Yu_etal_2018.md`](papers/Deep_recursive_network_embedding_with_regular_equivalence_Cui_Wang_Yu_etal_2018.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **DRNE / THUNLP network embedding code** — https://github.com/thunlp (Tsinghua THUNLP organization hosts network embedding code; check for DRNE releases by Ke Tu et al.)
- **Peng Cui / Wenwu Zhu group (Tsinghua)** — https://cuip.thu.edu.cn/ — Author group page with related publications and resources.

## Related

- **DeepWalk** (Perozzi et al., 2014) — https://github.com/phanein/deepwalk — Random-walk + Skip-Gram embedding preserving structural equivalence (the paradigm DRNE moves beyond).
- **node2vec** (Grover & Leskovec, 2016) — https://github.com/aditya-grover/node2vec — Biased random walk embedding (structural equivalence).
- **LINE** (Tang et al., 2015) — First/second-order proximity embedding (structural equivalence).
- **SDNE** (Wang et al., 2016) — Deep autoencoder embedding (structural equivalence).
- **M-NMF** (Wang et al., 2016) — Incorporates community structure into embedding.
- **GraphSAGE** (Hamilton et al., 2017) — https://github.com/williamleif/GraphSAGE — Contemporary neighborhood-aggregation inductive method (structural equivalence focus).
- **CANE** (Tu et al., 2017) — https://github.com/thunlp/CANE — Context-aware network embedding (sibling work from the same group).
- **Structural/regular equivalence theory** (Lorrain & White, 1971; White & Reitz, 1983) — The sociological foundation DRNE operationalizes.
- **Centrality measures** (degree, betweenness, eigenvector, PageRank) — Role/importance metrics that DRNE unifies into a learned representation.
- **Role discovery / role-aware embedding** — Subsequent work on structural role embeddings builds on DRNE's regular-equivalence framing.
