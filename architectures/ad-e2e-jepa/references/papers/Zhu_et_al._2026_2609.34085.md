# AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving

**Authors:** Zhiwu Zhu, Qingwen Zhang, Yann LeCun, Anna Choromanska
**arXiv:** 2609.34085 (2026)

## Abstract

We apply the JEPA world model framework to end-to-end autonomous driving. A SIGReg-regularized learnable projector compresses the JEPA encoder's spatial features by 16× in patch count and 4× in embedding dimension, enabling 100× inference acceleration: 0.8 seconds for an 8-frame rollout over 256 candidate trajectories. The world model performs zero-shot goal-conditioned planning without any trained driving policy. On NAVSIMv2, we achieve 67.3/72.9 EPDMS (with multiplicative safety metric) and 84.1/86.5 EPDMS† (without). Self-supervised pretraining of the projector improves downstream imitation learning from 80.2 to 85.4 EPDMS. Target displacement error averages 4.0/2.8 meters for 20-meter goals.
