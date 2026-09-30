# Paper (Lin et al. 2022)

> Source: `https://arxiv.org/abs/2211.13357`

---

**MPT: Mesh Pre-Training with Transformers for Human Pose and Mesh Reconstruction**

Kevin Lin, Chung-Ching Lin, Lin Liang, Zicheng Liu, Lijuan Wang

**Abstract**

Traditional methods of reconstructing 3D human pose and mesh from single images rely on paired image-mesh datasets, which can be difficult and expensive to obtain. Due to this limitation, model scalability is constrained as well as reconstruction performance. Towards addressing the challenge, we introduce Mesh Pre-Training (MPT), an effective pre-training strategy that leverages large amounts of MoCap data to effectively perform pre-training at scale. We introduce the use of MoCap-generated heatmaps as input representations to the mesh regression transformer and propose a Masked Heatmap Modeling approach for improving pre-training performance. This study demonstrates that pre-training using the proposed MPT allows our models to perform effective inference without requiring fine-tuning. We further show that fine-tuning the pre-trained MPT model considerably improves the accuracy of human mesh reconstruction from single images. Experimental results show that MPT outperforms previous state-of-the-art methods on Human3.6M and 3DPW datasets. As a further application, we benchmark and study MPT on the task of 3D hand reconstruction, showing that our generic pre-training scheme generalizes well to hand pose estimation and achieves promising reconstruction performance.

---

- **arXiv ID**: `2211.13357`
- **Published**: 2022-11-24
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2211.13357
