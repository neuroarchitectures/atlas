# References: GraphVAE

## Papers

- [`GraphVAE_Towards_Generation_of_Small_Graphs_Using_Variationa_Variational_Autoencoders_2018.md`](papers/GraphVAE_Towards_Generation_of_Small_Graphs_Using_Variationa_Variational_Autoencoders_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- Edge-Conditioned Graph Convolutions (ECC) (Simonovsky & Komodakis, 2017) — encoder backbone
- Max-pooling matching / MPM (Cho et al., 2014) — approximate graph-matching algorithm used to compute the assignment matrix for the reconstruction loss
- QM9 (Ramakrishnan et al., 2014) and ZINC chemical databases — evaluation benchmarks

## Related

- **VAE** (Kingma & Welling, 2013; Rezende et al., 2014) — the variational autoencoder framework GraphVAE instantiates for graph data.
- **Graph autoencoders for link prediction** (Kipf & Welling, 2016b; Grover et al., 2017) — earlier graph-based autoencoders that learn node embeddings rather than generating whole graphs.
- **CharacterVAE / GrammarVAE / SDVAE** (Gómez-Bombarelli 2016; Kusner 2017; Dai 2018) — SMILES-based molecular VAE generators that GraphVAE positions itself against.
- **Junction Tree VAE** (Jin et al., 2018) and **NeVAE** (Samanta et al., 2018) — contemporaneous VAE-based graph generators for molecules.
- **MolGAN** (De Cao & Kipf, 2018) — implicit (GAN) successor that explicitly targets GraphVAE's permutation/graph-matching bottleneck by dropping the likelihood entirely.
