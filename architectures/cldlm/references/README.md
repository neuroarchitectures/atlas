# References: Continuous Latent Diffusion Language Model

## Papers

- [`Continuous_Latent_Diffusion_Language_Model_Guo_Zhao_Zhao_etal_2026.md`](papers/Continuous_Latent_Diffusion_Language_Model_Guo_Zhao_Zhao_etal_2026.md) — primary source paper (from PaperVault `DL-Architectures/03-diffusion-models/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Project Page:** https://hongcanguo.github.io/Cola-DLM/
- **arXiv:** https://arxiv.org/abs/2605.06548
- **Institution:** ByteDance Seed

## Related

- **LLaDA** — Masked diffusion language model; used as a matched baseline for comparison.
- **Diffusion Transformers (DiT)** — Block-causal DiT is used for prior learning in the continuous latent space.
- **Text VAE / VAE** — Used to learn stable text-to-latent mapping in Stage 1.
- **Flow Matching** — Solver for the continuous-flow prior; implementation choice for learning the latent prior transport.
- **Autoregressive Language Models (GPT family)** — The dominant paradigm that Cola DLM provides an alternative to via non-autoregressive latent diffusion.
- **Discrete Diffusion Language Models** — Perform observation recovery in discrete token space; Cola DLM moves diffusion to continuous latent space.
- **Continuous Diffusion Language Models (Diffusion-LM, etc.)** — Token-embedding-based continuous methods; Cola DLM adds hierarchical latent-variable interpretation.
