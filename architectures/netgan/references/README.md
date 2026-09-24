# References: NetGAN

## Papers

- [`NetGAN_Generating_Graphs_via_Random_Walks_Unknown_2018.md`](papers/NetGAN_Generating_Graphs_via_Random_Walks_Unknown_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- NetGAN official code — https://www.kdd.in.tum.de/netgan
- WGAN with gradient penalty (Gulrajani et al., 2017) — training objective
- Latent-space interpolation animation — https://goo.gl/bkNcVa

## Related

- **GAN** (Goodfellow et al., 2014) — the implicit generative framework NetGAN adapts to graph data.
- **WGAN** (Arjovsky et al., 2017) — the Wasserstein GAN objective NetGAN builds on.
- **node2vec** (Grover & Leskovec, 2016) — the biased second-order random-walk sampling strategy used to build the training set; also a node-embedding baseline.
- **DeepWalk** (Perozzi et al., 2014) and **LINE** (Tang et al., 2015) — node-embedding predecessors.
- **VGAE** (Kipf & Welling, 2016) — variational graph autoencoder; a node-embedding baseline that fails to produce realistic graphs when used for generation.
- **GraphGAN** (Wang et al., 2017) — a prescribed edge-level probabilistic model optimized with a GAN objective.
- **Liu et al. (2017)** and **Tavakoli et al. (2017)** — prior GAN attempts on graph data (subgraph topology; direct adjacency-matrix generation).
- **LSTM** (Hochreiter & Schmidhuber, 1997) — the recurrent architecture used for both generator and discriminator.
- **Straight-Through Gumbel-Softmax** (Jang et al., 2016) — the differentiable sampling estimator.
- **Prescribed graph generative models** — configuration model (Molloy & Reed, 1995), degree-corrected stochastic blockmodel (Karrer & Newman, 2011), ERGM (Holland & Leinhardt, 1981), BTER (Seshadhri et al., 2012) — explicit baselines.
