# References: FastGCN

## Papers

- [`FASTGCN_Chen_2018.md`](papers/FASTGCN_Chen_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Related

- **Graph Convolutional Networks (GCN)** (Kipf & Welling, 2017) — the base model FastGCN generalizes; FastGCN reinterprets GCN's convolution as an integral transform to enable stochastic sampling.
- **GraphSAGE** (Hamilton et al., 2017) — the most directly related work; also proposes sampling to reduce computational footprint, but samples neighbors (product of sample sizes across layers) rather than vertices (sum of sample sizes), making FastGCN substantially more efficient.
- **ChebNet / Spectral filters** (Defferrard et al., 2016; Bruna et al., 2013; Henaff et al., 2015) — spectral graph convolution predecessors defining filters in the Fourier domain.
- **DeepWalk** (Perozzi et al., 2014) / **node2vec** (Grover & Leskovec, 2016) / **LINE** (Tang et al., 2015) — random walk and factorization-based transductive embedding methods.
