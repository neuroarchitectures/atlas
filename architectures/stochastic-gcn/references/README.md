# References: Stochastic GCN

## Papers

- [`Stochastic_Training_of_Graph_Convolutional_Networks_with_Var_Song_2018.md`](papers/Stochastic_Training_of_Graph_Convolutional_Networks_with_Var_Song_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- TensorFlow (Abadi et al., 2016) — auto-differentiation framework used for the implementation
- Titan X (Maxwell) GPU — reference hardware for the experiments
- Datasets: Citeseer, Cora, PubMed, NELL (Kipf & Welling, 2017); PPI, Reddit (Hamilton et al., 2017a)

## Related

- **GCN** (Kipf & Welling, 2017) — the base model whose batch training is the scalability bottleneck Stochastic GCN targets.
- **GraphSAGE / Neighbor Sampling** (Hamilton et al., 2017a) — the first stochastic GCN training attempt; no convergence guarantee; Stochastic GCN provides the missing guarantee and shrinks the sample size to 2.
- **FastGCN / Importance Sampling** (Chen et al., 2018) — another sampling-based algorithm; only converges as sample size → ∞; empirically worse than NS.
- **CNNs** (LeCun et al., 1995) — the convolutional antecedent GCNs generalize.
- **Dropout** (Srivastava et al., 2014) — the feature-randomness source that necessitates the CVD estimator.
- **Control variates** (Ripley, 2009, Ch. 5) — the Monte-Carlo variance-reduction technique the CV estimator is built on.
- **DeepWalk** (Perozzi et al., 2014), **LINE** (Tang et al., 2015), **node2vec** (Grover & Leskovec, 2016) — graph-embedding predecessors that do not use node features.
- **GAT** (Veličković et al., 2017), **R-GCN** (Schlichtkrull et al., 2017), **Graph Convolutional Matrix Completion** (Berg et al., 2017), **Variational Graph Autoencoder** (Kipf & Welling, 2016) — other GCN variants the algorithm is applicable to.
