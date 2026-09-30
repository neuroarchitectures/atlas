# Paper (Yang et al. 2025)

> Source: `https://doi.org/https://doi.org/10.1109/jbhi.2025.3588555`

---

**Restore-RWKV: Efficient and Effective Medical Image Restoration With RWKV**

Zhiwen Yang, Jiayin Li, Hui Zhang, Dan Zhao, Bingzheng Wei, Yan Lei Xu

**Abstract**

Transformers have revolutionized medical image restoration, but the quadratic complexity still poses limitations for their application to high-resolution medical images. The recent advent of the Receptance Weighted Key Value (RWKV) model in the natural language processing field has attracted much attention due to its ability to process long sequences efficiently. To leverage its advanced design, we propose Restore-RWKV, the first RWKV-based model for medical image restoration. Since the original RWKV model is designed for 1D sequences, we make two necessary modifications for modeling spatial relations in 2D medical images. First, we present a recurrent WKV (Re-WKV) attention mechanism that captures global dependencies with linear computational complexity. Re-WKV incorporates bidirectional attention as basic for a global receptive field and recurrent attention to effectively model 2D dependencies from various scan directions. Second, we develop an omnidirectional token shift (Omni-Shift) layer that enhances local dependencies by shifting tokens from all directions and across a wide context range. These adaptations make the proposed Restore-RWKV an efficient and effective model for medical image restoration. Even a lightweight variant of Restore-RWKV, with only 1.16 million parameters, achieves comparable or even superior results compared to existing state-of-the-art (SOTA) methods. Extensive experiments demonstrate that the resulting Restore-RWKV achieves SOTA performance across a range of medical image restoration tasks, including PET image synthesis, CT image denoising, MRI image super-resolution, and all-in-one medical image restoration.

---

- **arXiv ID**: `N/A`
- **Year**: 2025
- **DOI**: https://doi.org/10.1109/jbhi.2025.3588555
- **PDF**: N/A
