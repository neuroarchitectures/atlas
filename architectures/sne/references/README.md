# References: SNE (Signed Network Embedding)

## Papers

- [`A_ributed_Signed_Network_Embedding_Wang_Aggarwal_Tang_etal_2017.md`](papers/A_ributed_Signed_Network_Embedding_Wang_Aggarwal_Tang_etal_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **SNEA** — No official public repository found; the method is described in full detail in the source paper
- **Paper (ACM DL)** — https://doi.org/10.1145/3132847.3132905 — Official CIKM '17 paper page

## Related

**Predecessors:**
- DeepWalk (Perozzi et al., 2014) — Skip-gram on random walks; unsigned only
- LINE (Tang et al., 2015) — First/second-order proximity; unsigned only
- SiNE (Wang et al., 2017) — First signed network embedding using balance theory; structure-only
- TADW (Yang et al., 2015) — Text-attributed DeepWalk; unsigned, shallow attribute fusion

**Successors:**
- SIDE (Kim et al., 2018) — Signed network embedding via directed random walks
- SGCN (Li et al., 2020) — Signed graph convolutional networks extending GCN to signed graphs
- SIGNet (Kim et al., 2020) — Scalable signed network embedding with feature learning

**Alternatives:**
- SDNE (Wang et al., 2016) — Deep autoencoder for unsigned network embedding
- node2vec (Grover & Leskovec, 2016) — Biased random walks; unsigned only
