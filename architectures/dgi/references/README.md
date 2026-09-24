# References: Deep Graph Infomax (DGI)

## Papers

- [`Deep_Graph_Infomax_Brain_Fedus_Hamilton_2018.md`](papers/Deep_Graph_Infomax_Brain_Fedus_Hamilton_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Related

- **Deep InfoMax (DIM)** (Hjelm et al., 2018) — direct inspiration; DGI generalizes DIM's mutual information maximization between global and local representations from images to graphs. Paper: https://arxiv.org/abs/1808.06670
- **Contrastive Predictive Coding (CPC)** (Oord et al., 2018) — contemporaneous MI-maximization technique using autoregressive context; noted as unlikely to be trivially applicable to graphs due to node-ordering requirements.
- **Graph Convolutional Networks (GCN)** (Kipf & Welling, 2017) — used as the encoder for transductive tasks.
- **GraphSAGE** (Hamilton et al., 2017) — used as the encoder for inductive tasks; also the predecessor whose random walk objective DGI argues becomes redundant under GCN patch mixing.
