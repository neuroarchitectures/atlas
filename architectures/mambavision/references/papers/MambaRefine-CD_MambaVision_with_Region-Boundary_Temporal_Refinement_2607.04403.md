# Paper (Perera et al. 2026)

> Source: `https://arxiv.org/abs/2607.04403`

---

**MambaRefine-CD: MambaVision with Region-Boundary Temporal Refinement**

Dineth Perera, Thaariq Firdous, Oshadha Samarakoon, Roshan Godaliyadda, Parakrama Ekanayake, Vijitha Herath

**Abstract**

Binary change detection in remote sensing requires both complete changed-region localization and accurate boundary delineation. We present MambaRefine-CD, a region-boundary temporal refinement framework built on a shared MambaVision encoder. The proposed D-RBI module constructs temporal evidence from paired features, absolute differences, and signed differences, then separates it into region and Sobel-conditioned boundary streams. Region features are enhanced with CRAM-lite and decoded by an adaptive receptive-field FPN, while the finest boundary stream guides a bounded residual refinement of the coarse prediction. Experiments on DSIFN-CD and WHU-CD show strong changed-class F1 and IoU under verified evaluation settings, and ablations support the contribution of signed temporal evidence and the full region-boundary refinement pipeline.

---

- **arXiv ID**: `2607.04403`
- **Published**: 2026-07-05
- **Categories**: eess.IV, cs.CV
- **PDF**: https://arxiv.org/pdf/2607.04403
