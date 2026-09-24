# References: Generative Distribution Embeddings

## Papers

- [`Generative_Distribution_Embeddings_Fishman_Gowri_Yin_etal_2025.md`](papers/Generative_Distribution_Embeddings_Fishman_Gowri_Yin_etal_2025.md) — primary source paper (from PaperVault `DL-Architectures/01-transformers/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** All experiment code available at the project repository linked in the paper (arXiv:2505.18150)
- **arXiv:** https://arxiv.org/abs/2505.18150

## Related

- **Kernel Mean Embeddings (KME)** — Nonparametric distribution representation in RKHS; GDE generalizes this as a particular distributionally invariant encoder.
- **Wasserstein Wormhole** — Represents distributions as points where Euclidean distances match Sinkhorn divergences; a particular instantiation of a GDE.
- **VAE / DDPM / GAN** — Conditional generative models that can be repurposed as GDE generators.
- **Deep Sets** — Permutation-invariant set encoders; related to but not fully distributionally invariant due to proportional sensitivity.
- **Optimal Transport / Wasserstein Geometry** — Theoretical foundation for GDE's geometric properties (latent distances ≈ W2 distances).
