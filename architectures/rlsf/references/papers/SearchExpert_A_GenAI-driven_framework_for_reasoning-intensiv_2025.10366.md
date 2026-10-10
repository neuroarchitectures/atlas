# SearchExpert: A GenAI-driven framework for reasoning-intensive multimedia information fusion through fine-tuning and reinforcement learning

> Source: `https://arxiv.org/abs/2025.10366`

---

**Authors:** Jinzheng Li, Yiqing Shen, Weitao Zhou, Hui CHEN

**Published:** 2025

**Concepts:** Reinforcement learning, Computer science, Information fusion, Multimedia, Artificial intelligence

## Abstract

The rapid advancement of Generative Artificial Intelligence (GenAI) has opened new frontiers in multimodal information fusion, yet current large language model (LLM)-driven search agents remain limited in their ability to handle reasoning-intensive queries and integrate multimedia data effectively. In this paper, we propose SearchExpert, a GenAI-enhanced framework that augments LLMs with powerful multimedia search and reasoning capabilities via a novel two-stage training paradigm. First, we introduce an efficient natural language representation for directed acyclic graph (DAG)-based search plans to reduce token overhead and support structured reasoning. We then propose Supervised Fine-Tuning for Searching (SFTS), enabled by an automated data construction pipeline that adapts LLMs to generate token-efficient, structured search plans from complex queries. Second, to further enhance reasoning ability, we introduce Reinforcement Learning from Search Feedback (RLSF), which uses reward signals based on semantic alignment and intrinsic quality assessments of retrieved results to optimize LLM behavior. To address the limitations of unimodal input and output, we integrate a multimedia understanding and generation module based on vision-language models and image synthesis tools (e.g., BLIP-2, and DALLE-3), enabling the GenAI-based fusion of text and visual data. We also establish SearchExpertBench-25, a benchmark comprising 200 multimedia-rich, reasoning-intensive queries spanning financial and global news domains, accompanied by a rigorous human evaluation framework. Experimental results demonstrate that SearchExpert surpasses state-of-the-art baselines such as FinSearch and Perplexity Pro, achieving up to 71.5% accuracy on complex benchmark tasks while reducing token consumption by over 40%. Human evaluations further highlight improvements in completeness, analytical integrity, and multimodal fluency. This work presents a scalable and generalizable GenAI framework for information fusion, with implications for real-time decision-making in complex, multi-source environments. The code is available at https://anonymous.4open.science/r/SearchExpert-2343/ . • A new GenAI framework enables reasoning-intensive multimedia information search. • Natural language DAGs cut token use while preserving structured reasoning plans. • Reinforcement learning improves search quality via semantic and intrinsic rewards. • Vision-language tools allow fusion of visual and textual data in user queries.

**PDF:** https://arxiv.org/pdf/2025.10366