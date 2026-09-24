# References: GAN

## Papers

- [`Unknown_Gui_Sun_Wen_etal_2001.md`](papers/Unknown_Gui_Sun_Wen_etal_2001.md) — primary source paper (from PaperVault `DL-Architectures/06-gan-kan/`)

> Papers and blog posts stored as full-text markdown. PDFs converted with `pdftotext -layout`.

## Official

- **arXiv:** https://arxiv.org/abs/2001.06937 (survey paper)
- **Original GAN paper:** Goodfellow et al. (2014), arXiv:1406.2661

## Related

- **VAE (Variational Autoencoder)** — Explicit density generative model; GANs avoid the lower bound optimization limitation of VAEs.
- **Diffusion Models (DDPM)** — Eventually surpassed GANs in sample quality; iterative denoising vs. single-pass generation.
- **DCGAN** — Convolutional GAN architecture establishing stable training practices.
- **WGAN** — Uses Wasserstein distance for more stable training and meaningful loss curves.
- **StyleGAN** — Style-based generator with adaptive instance normalization for high-quality face generation.
- **CycleGAN** — Unpaired image-to-image translation using cycle consistency.
- **InfoGAN** — Disentangled representations via mutual information maximization.
- **cGAN (Conditional GAN)** — Conditional generation with auxiliary information.
- **PixelCNN / FVBNs** — Autoregressive generative models; GANs enable parallelizable generation unlike these sequential approaches.
