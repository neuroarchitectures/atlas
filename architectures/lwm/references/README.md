# References: LWM (Large World Model)

## Papers

- [`Language_Is_Not_All_You_Need_Lv_Cui_Mohammed_etal_2023.md`](papers/Language_Is_Not_All_You_Need_Lv_Cui_Mohammed_etal_2023.md) — primary source paper (from PaperVault `DL-Architectures/01-transformers/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **Code:** https://github.com/microsoft/unilm
- **arXiv:** https://arxiv.org/abs/2302.14045

## Related

- **MetaLM** — The framework KOSMOS-1 follows; regards language models as general-purpose interfaces with perception modules docked.
- **CLIP (ViT-L/14)** — Pretrained vision encoder used as the embedding module for input images.
- **Magneto** — Transformer variant used as the backbone architecture, with extra LayerNorm for training stability.
- **XPOS** — Relative position encoding used for better long-context modeling and length generalization.
- **Resampler** — Attentive pooling mechanism used to reduce the number of image embeddings.
- **GPT / Transformer** — Foundation architecture; KOSMOS-1 extends the Transformer decoder to multimodal input.
