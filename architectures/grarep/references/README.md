# References: GraRep

## Papers

- [`GraRep_Learning_Graph_Representations_with_Global_Structural_Cao_2015.md`](papers/GraRep_Learning_Graph_Representations_with_Global_Structural_Cao_2015.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **arXiv:** https://arxiv.org/abs/1506.05941
- **DOI:** https://doi.org/10.1145/2806416.2806512

## Related

- **DeepWalk** — Predecessor; random walk + skip-gram. GraRep formalizes its implicit loss function and extends to global structure via k-step transition matrices.
- **LINE** — Predecessor; captures 1st and 2nd order proximity. GraRep generalizes to arbitrary k-step relationships.
- **node2vec** — Successor; flexible biased random walks. Complements GraRep's matrix factorization approach with a sampling-based method.
- **HARP** — Successor; hierarchical graph coarsening meta-strategy that can improve GraRep via multi-level initialization.
- **GraphGAN** — Alternative paradigm; adversarial framework combining generative and discriminative graph embedding.
- **word2vec / SGNS** — GraRep formally analyzes the connection between skip-gram with negative sampling and PMI matrix factorization.
- **struc2vec** — Related; focuses on structural identity rather than proximity. Complementary approach to GraRep's global structure capture.
