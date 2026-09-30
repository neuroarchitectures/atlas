# Paper (Charabarty et al. 2026)

> Source: `https://arxiv.org/abs/2605.25527`

---

**DeepSeekMath Meets Order Book: Group-Aware Policy Optimization for High-Frequency Directional Trading**

Sayak Charabarty, Souradip Pal

**Abstract**

This paper studies reinforcement learning for high-frequency trading on limit order books by pairing an Order-Flow-based state model with policy-gradient methods. Instead of value-based RL techniques like tabular Q-learning, our approach deploys policy-based methods like vanilla PPO and DeepSeekMath-inspired variants like GRPO and GSPO, that use group-normalized updates and downside-aware shaping. On backtests with financial assets AMZN, AAPL, and GOOG under a simplified backtesting setup based on spread-scaled rewards, these new policies improve net average PnL, profitability, and drawdown over the Q-Learning baseline. Our results show that (1) Order-Flow signals are an adequate state for policy RL and (2) group-aware PPO surrogates are preferable over value-based baselines.

---

- **arXiv ID**: `2605.25527`
- **Published**: 2026-05-25
- **Categories**: cs.LG, cs.CE
- **PDF**: https://arxiv.org/pdf/2605.25527
