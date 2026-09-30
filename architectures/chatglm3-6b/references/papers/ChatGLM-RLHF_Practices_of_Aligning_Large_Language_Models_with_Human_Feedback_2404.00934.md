# Paper (Hou et al. 2024)

> Source: `https://arxiv.org/abs/2404.00934`

---

**ChatGLM-RLHF: Practices of Aligning Large Language Models with Human Feedback**

Zhenyu Hou, Yilin Niu, Zhengxiao Du, Xiaohan Zhang, Xiao Liu, Aohan Zeng, Qinkai Zheng, Minlie Huang, Hongning Wang, Jie Tang et al.

**Abstract**

ChatGLM is a free-to-use AI service powered by the ChatGLM family of large language models (LLMs). In this paper, we present the ChatGLM-RLHF pipeline -- a reinforcement learning from human feedback (RLHF) system -- designed to enhance ChatGLM's alignment with human preferences. ChatGLM-RLHF encompasses three major components: the collection of human preference data, the training of the reward model, and the optimization of policies. Throughout the process of integrating ChatGLM-RLHF into production, we encountered and addressed several unprecedented challenges. We introduce the strategies to mitigate reward variance for stabilized large-scale training, implement model parallelism with fused gradient-descent, and design regularization constraints to avoid catastrophic forgetting in LLMs. Experiments show that ChatGLM-RLHF brings significant improvements in alignment tasks compared to the supervised fine-tuned (SFT) version of ChatGLM. For instance, it achieves on average 15\% more wins against ChatGLM-SFT in Chinese alignment tasks. The work presents our practices of aligning LLMs with human preferences, offering insights into the challenges and solutions in RLHF implementations.

---

- **arXiv ID**: `2404.00934`
- **Published**: 2024-04-01
- **Categories**: cs.CL
- **PDF**: https://arxiv.org/pdf/2404.00934
