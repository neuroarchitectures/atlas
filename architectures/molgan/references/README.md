# References: MolGAN

## Papers

- [`MolGAN_An_implicit_generative_model_for_small_molecular_grap_Kipf_2018.md`](papers/MolGAN_An_implicit_generative_model_for_small_molecular_grap_Kipf_2018.md) — primary source paper (from PaperVault `GNN/03-gnn-architectures/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- RDKit Open-Source Cheminformatics Software — http://www.rdkit.org (external reward/property oracle used during training)
- Improved WGAN (Gulrajani et al., 2017) — gradient-penalty GAN training scheme adopted by MolGAN
- Relational-GCN (Schlichtkrull et al., 2017) — graph convolution with multiple edge types, used for the discriminator and reward network

## Related

- **ORGAN** (Guimaraes et al., 2017) — closest related work; SeqGAN on SMILES with REINFORCE for property optimization. MolGAN differs by working on graphs directly and using DDPG.
- **GraphVAE** (Simonovsky & Komodakis, 2018) — likelihood-based graph generator whose permutation/graph-matching bottleneck motivates MolGAN's implicit approach.
- **CharacterVAE / GrammarVAE / SDVAE** (Gómez-Bombarelli 2016; Kusner 2017; Dai 2018) — SMILES-based VAE baselines compared against in the QM9 experiments.
- **Junction Tree VAE** (Jin et al., 2018) and **NeVAE** (Samanta et al., 2018) — other VAE-based graph generators for molecules, cited as contemporaneous graph-direct generative work.
- **Junction Tree VAE / NeVAE / GraphRNN** (Johnson 2017; Li et al. 2018b; You et al. 2018) — sequential graph generators suggested as future extensions to overcome MolGAN's small-graph limitation.
- **DDPG** (Lillicrap et al., 2015) — off-policy actor-critic algorithm adapted for the RL component.
- **Gumbel-Softmax / Straight-Through estimator** (Jang et al., 2016; Maddison et al., 2016) — used for differentiable sampling of discrete atom/bond types.
