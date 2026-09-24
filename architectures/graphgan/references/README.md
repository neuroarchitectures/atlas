# References: GraphGAN

## Papers

- [`GraphGAN_Graph_Representation_Learning_with_Generative_Adver_Wang_Wang_Wang_etal_2018.md`](papers/GraphGAN_Graph_Representation_Learning_with_Generative_Adver_Wang_Wang_Wang_etal_2018.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** https://github.com/hongweilibao/GraphGAN
- **arXiv:** https://arxiv.org/abs/1711.08267

## Related

- **DeepWalk** — Predecessor (generative); random walk + skip-gram. GraphGAN's generator generalizes the connectivity distribution modeling.
- **node2vec** — Predecessor (generative); biased random walks. GraphGAN unifies generative and discriminative paradigms.
- **SDNE** — Predecessor (discriminative); autoencoder-based edge prediction. GraphGAN's discriminator is a simpler discriminative model.
- **LINE** — Predecessor; implicitly combines 1st-order (generative) and 2nd-order (discriminative) proximity. GraphGAN makes the combination explicit and adversarial.
- **GAN (Goodfellow et al., 2014)** — Foundational; original generative adversarial networks. GraphGAN adapts the GAN framework to graph-structured data.
- **NetGAN** — Successor; GAN-based graph generation via random walk sequences.
- **GraRep** — Related; matrix factorization approach to global structure. Alternative paradigm to adversarial learning.
- **HARP** — Related; hierarchical meta-strategy that could wrap GraphGAN for improved initialization.
