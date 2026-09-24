# References: SDGE (Deep Gaussian Embedding)

## Papers

- [`Deep_Gaussian_Embedding_of_Graphs_Unsupervised_Inductive_Lea_Ranking_2017.md`](papers/Deep_Gaussian_Embedding_of_Graphs_Unsupervised_Inductive_Lea_Ranking_2017.md) — primary source paper (from PaperVault `GNN/02-network-embedding/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Graph2Gauss (G2G) official implementation** — https://github.com/abojchevski/graph2gauss (reference implementation by the authors; includes attributed/plain, directed/undirected graph support and the ranking-based loss)
- **Aleksandar Bojchevski — project page** — https://www.bojchevski.me/ (author's site with related publications and code links)
- **PyTorch Geometric (PyG) — Gaussian embedding utilities** — https://github.com/pyg-team/pytorch_geometric (general GNN framework; supports custom distribution-based embeddings and inductive aggregation)

## Related

- **DeepWalk** (Perozzi et al., 2014) — https://github.com/phanein/deepwalk — Random-walk + Skip-Gram point embedding predecessor (transductive, plain graphs).
- **node2vec** (Grover & Leskovec, 2016) — https://github.com/aditya-grover/node2vec — Biased random walk embedding predecessor.
- **LINE** (Tang et al., 2015) — First/second-order proximity embedding predecessor.
- **SDNE** (Wang et al., 2016) — Deep autoencoder for first/second-order proximity; point embeddings.
- **GraphSAGE** (Hamilton et al., 2017) — https://github.com/williamleif/GraphSAGE — Inductive neighborhood aggregation embedding (contemporary inductive method).
- **GAE / Graph VAE** (Kipf & Welling, 2016) — https://github.com/tkipf/gae — Unsupervised graph autoencoder/VAE embedding.
- **DVNE** (Zhu et al., 2018) — Successor that also embeds nodes as Gaussians but uses 2-Wasserstein distance to preserve transitivity (addresses G2G's KL-divergence and shortest-path limitations).
- **Gaussian word embeddings** (Vilnis & McCallum, 2014) — The original Gaussian-distribution embedding idea that G2G extends to graphs.
