# References: GraphSAGE

## Papers

- [`GraphSage_17_Leskovec_2017.md`](papers/GraphSage_17_Leskovec_2017.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Related

- **Graph Convolutional Networks (GCN)** (Kipf & Welling, 2017) — direct predecessor; GraphSAGE extends the GCN framework to the inductive setting and generalizes it to use trainable aggregation functions beyond simple convolutions.
- **DeepWalk** (Perozzi et al., 2014) — representative factorization-based transductive baseline; GraphSAGE's unsupervised loss is inspired by random walk objectives but applied to learned aggregator outputs rather than per-node embeddings.
- **Planetoid-I** (Yang et al., 2016) — a notable inductive exception; however, it does not use graph structural information during inference, unlike GraphSAGE.
- **node2vec** (Grover & Leskovec, 2016) — random walk-based transductive embedding; cannot naturally generalize to unseen nodes without expensive re-training.
