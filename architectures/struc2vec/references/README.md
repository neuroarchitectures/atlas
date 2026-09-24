# References: struc2vec

## Papers

- [`struc2vec_Learning_Node_Representations_from_Structural_Iden_Identity_Ribeiro_Saverese_etal_2017.md`](papers/struc2vec_Learning_Node_Representations_from_Structural_Iden_Identity_Ribeiro_Saverese_etal_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** https://github.com/leoribeiro/struc2vec
- **arXiv:** https://arxiv.org/abs/1704.03165
- **DOI:** https://doi.org/10.1145/3097983.3098061

## Related

- **DeepWalk** — Predecessor; random walk + skip-gram. struc2vec reuses skip-gram but changes walk strategy to capture structural identity.
- **node2vec** — Predecessor; flexible biased random walks on the original network. struc2vec builds walks on a structural similarity multilayer graph instead.
- **RolX** — Predecessor; structural role discovery via matrix factorization. struc2vec provides a learned, embedding-based alternative.
- **GraRep** — Related; captures global structure via k-step transition matrices. Complementary proximity-based approach.
- **GraphWave** — Successor; spectral approach to structural role embedding using heat wavelet diffusion.
- **GraphSAGE** — Related; inductive representation learning that can incorporate structural features for role-aware embeddings.
- **HITS (Kleinberg, 1999)** — Early structural identity concept (hubs and authorities) in web graphs; foundational to structural equivalence.
