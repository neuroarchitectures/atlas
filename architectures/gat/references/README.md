# References: GAT (Graph Attention Network)

## Papers

- [`Graph_Attention_Networks_Networks_2018.md`](papers/Graph_Attention_Networks_Networks_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Related

- **Graph Convolutional Networks (GCN)** (Kipf & Welling, 2017) — direct predecessor; GAT addresses GCN's spectral dependency on the Laplacian eigenbasis and its inability to assign different weights to neighbors.
- **GraphSAGE** (Hamilton et al., 2017) — inductive predecessor; GAT improves on its fixed-size neighborhood sampling and LSTM ordering assumption by attending to the entire neighborhood.
- **MoNet** (Monti et al., 2016) — unified spatial approach; GAT can be reformulated as a particular instance of MoNet, but uses node features rather than structural properties for similarity computation.
- **ChebNet** (Defferrard et al., 2016) — spectral approach using Chebyshev expansion of the graph Laplacian; a key spectral predecessor whose eigenbasis dependency GAT eliminates.
