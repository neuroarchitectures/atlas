# Paper (Fang et al. 2026)

> Source: `https://arxiv.org/abs/2605.19734`

---

**GeoMamba: A Geometry-driven MambaVision Framework and Dataset for Fine-grained Optical-SAR Object Retrieval**

Tiantong Fang, Xiuwei Wang, Jing Xiao, Wujie Zhou, Liang Liao, Mi Wang

**Abstract**

Multi-source remote sensing enables complementary observation of ground objects, while cross-modal fine-grained object retrieval remains challenging, especially under unaligned optical and SAR conditions. Unlike conventional retrieval settings that rely on paired or spatially aligned samples, practical optical-SAR retrieval is affected by substantial modality discrepancy, speckle noise, and structural inconsistency, which limit robust cross-modal representation learning. To address this problem, we propose GeoMamba, a geometry-driven framework tailored for optical-SAR fine-grained retrieval. Specifically, GeoMamba introduces a Geometric Feature Injection (GFI) module that enhances cross-modal feature interaction and incorporates structural priors, thereby improving the robustness of SAR representations and promoting geometry-consistent feature learning. In addition, a Geometric Consistency Constraint (GCC) module, together with a Deep Supervision (DS) strategy, imposes hierarchical geometric constraints using classical operators, which helps preserve informative object structures during representation learning. We further construct a new dataset, FGOS-as, containing 11 aerospace and maritime categories for evaluating unaligned cross-modal fine-grained object retrieval in realistic remote sensing scenarios. Extensive experiments on FGOS-as demonstrate that GeoMamba outperforms existing methods, achieving 63.3% mAP and 77.0% Rank-1 accuracy in all-to-all retrieval setting.

---

- **arXiv ID**: `2605.19734`
- **Published**: 2026-05-19
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2605.19734
