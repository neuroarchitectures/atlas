# Paper (Mavrin et al. 2026)

> Source: `https://arxiv.org/abs/2604.00362`

---

**In harmony with gpt-oss**

Borislav Mavrin

**Abstract**

No one has independently reproduced OpenAI's published scores for gpt-oss-20b with tools, because the original paper discloses neither the tools nor the agent harness. We reverse-engineered the model's in-distribution tools: when prompted without tool definitions, gpt-oss still calls tools from its training distribution with high statistical confidence -- a strong prior, not a hallucination. We then built a native harmony agent harness (https://github.com/borislavmavrin/harmonyagent.git) that encodes messages in the model's native format, bypassing the lossy Chat Completions conversion. Together, these yield the first independent reproduction of OpenAI's published scores: 60.4% on SWE Verified HIGH (published 60.7%), 53.3% MEDIUM (53.2%), and 91.7% on AIME25 with tools (90.4%).

---

- **arXiv ID**: `2604.00362`
- **Published**: 2026-04-01
- **Categories**: cs.AI, cs.LG
- **PDF**: https://arxiv.org/pdf/2604.00362
