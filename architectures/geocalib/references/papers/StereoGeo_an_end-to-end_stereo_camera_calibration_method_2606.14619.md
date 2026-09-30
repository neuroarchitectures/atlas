# StereoGeo: an end-to-end stereo camera calibration method

> Source: `https://arxiv.org/abs/2606.14619`

---

**Authors:** Imane Meddour, Andréa Macario Barros, Cédric Gouy‐Pailler

**Published:** 2026

**Concepts:** Artificial intelligence, Computer vision, Computer science, Calibration, Monocular

## Abstract

In this work, we propose StereoGeo, an end-to-end network-based approach for stereo camera calibration. Our method estimates the focal lengths and gravity directions of the left and right cameras, as well as the relative extrinsic transformation relating them. Existing methods often rely on calibration patterns in structured environments or address only a single camera configuration, being limited to either intrinsic or extrinsic estimation, and depending on a multi-view setups. StereoGeo extends the GeoCalib algorithm, integrating deep neural network feature extraction with a differentiable optimizer. Extensive experiments on real-world benchmarks demonstrate that StereoGeo achieves competitive performance for intrinsic calibration and provides accurate stereo extrinsic estimation, outperforming existing methods that are limited to monocular settings. The dataset used in this work is partially publicly available at https://github.com/meddourimane/StereoGeo-dataset.

**PDF:** https://arxiv.org/pdf/2606.14619