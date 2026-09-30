# Paper (Wang et al. 2026)

> Source: `https://arxiv.org/abs/2609.16811`

---

**Hyper-RED: Scalable Event Pre-training via Semantic Hypergraph Distillation**

Meisen Wang, Zhiqiang Tian, Wei Bao, Chengjie Wang, Shaoyi Du, Siqi Li

**Abstract**

Event cameras have shown great potential for robust visual perception, yet scaling event representation learning remains challenging due to the scarcity of large-scale annotated event data. Pretrained image models provide scalable semantic supervision, but existing image-to-event methods rely on rigid pixel-wise or token-wise alignment that overlooks modality discrepancies in texture, density, and appearance, potentially causing semantic collapse and limiting transferability. To address this issue, we propose Hyper-RED, a simple, painless, and scalable image-to-event pretraining framework that transfers high-order semantic structures from images to events. Hyper-RED uses hypergraphs to model and align high-order semantic associations among multiple image and event tokens, enabling cross-modal knowledge transfer while accommodating modality-specific differences rather than enforcing rigid one-to-one correspondence. Specifically, given a paired event--image sample, Hyper-RED leverages DINOv3 to extract spatial token representations and constructs image, event, and cross-modal semantic hypergraphs, where each hyperedge connects multiple semantically correlated tokens. We further introduce a hypergraph relational distillation loss that imposes complementary intra- and cross-modal constraints, enabling the event encoder to inherit image-derived semantic organization while preserving local relational consistency and event-specific characteristics. Experiments on three tasks across five event datasets demonstrate consistent scaling from ViT-S to ViT-L and state-of-the-art performance (Fig.1). The code is available at: https://github.com/meisenwang/Hyper--RED.

---

- **arXiv ID**: `2609.16811`
- **Published**: 2026-09-15
- **Categories**: cs.CV
- **PDF**: https://arxiv.org/pdf/2609.16811
