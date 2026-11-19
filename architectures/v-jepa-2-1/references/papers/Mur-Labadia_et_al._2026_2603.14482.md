# V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

**Authors:** Corentin Mur-Labadia et al. (Meta FAIR)
**arXiv:** 2603.14482 (2026)

## Abstract

V-JEPA 2 demonstrated that masked latent prediction at internet scale produces strong video representations for understanding and planning. However, its patch-level features lack spatial structure, limiting performance on dense prediction tasks. We introduce V-JEPA 2.1, which unlocks dense features through four key components: (1) a dense prediction loss that supervises all tokens (visible and masked), (2) deep self-supervision applied at multiple encoder layers, (3) a multimodal tokenizer unifying image and video, and (4) effective scaling of model and data. V-JEPA 2.1 achieves state-of-the-art on Ego4D short-term object interaction anticipation (7.71 mAP), EPIC-KITCHENS high-level action anticipation (40.8 R@5), NYUv2 depth estimation (0.307 RMSE), and SSv2 (77.7), while improving robot grasping success by 20 percentage points when used with V-JEPA 2-AC.
