# Paper (Vytla et al. 2025)

> Source: `https://arxiv.org/abs/2511.18243`

---

**Dreaming Falcon: Physics-Informed Model-Based Reinforcement Learning for Quadcopters**

Eashan Vytla, Bhavanishankar Kalavakolanu, Andrew Perrault, Matthew McCrink

**Abstract**

Current control algorithms for aerial robots struggle with robustness in dynamic environments and adverse conditions. Model-based reinforcement learning (RL) has shown strong potential in handling these challenges while remaining sample-efficient. Additionally, Dreamer has demonstrated that online model-based RL can be achieved using a recurrent world model trained on replay buffer data. However, applying Dreamer to aerial systems has been quite challenging due to its sample inefficiency and poor generalization of dynamics models. Our work explores a physics-informed approach to world model learning and improves policy performance. The world model treats the quadcopter as a free-body system and predicts the net forces and moments acting on it, which are then passed through a 6-DOF Runge-Kutta integrator (RK4) to predict future state rollouts. In this paper, we compare this physics-informed method to a standard RNN-based world model. Although both models perform well on the training data, we observed that they fail to generalize to new trajectories, leading to rapid divergence in state rollouts, preventing policy convergence.

---

- **arXiv ID**: `2511.18243`
- **Published**: 2025-11-23
- **Categories**: cs.RO
- **PDF**: https://arxiv.org/pdf/2511.18243
