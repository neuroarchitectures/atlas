# Paper (O'Brien et al. 2021)

> Source: `https://arxiv.org/abs/2108.13475`

---

**An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction**

Conor O'Brien, Kin Sum Liu, James Neufeld, Rafael Barreto, Jonathan J Hunt

**Abstract**

Industrial recommender systems are frequently tasked with approximating probabilities for multiple, often closely related, user actions. For example, predicting if a user will click on an advertisement and if they will then purchase the advertised product. The conceptual similarity between these tasks has promoted the use of multi-task learning: a class of algorithms that aim to bring positive inductive transfer from related tasks. Here, we empirically evaluate multi-task learning approaches with neural networks for an online advertising task. Specifically, we consider approximating the probability of post-click conversion events (installs) (CVR) for mobile app advertising on a large-scale advertising platform, using the related click events (CTR) as an auxiliary task. We use an ablation approach to systematically study recent approaches that incorporate both multitask learning and "entire space modeling" which train the CVR on all logged examples rather than learning a conditional likelihood of conversion given clicked. Based on these results we show that several different approaches result in similar levels of positive transfer from the data-abundant CTR task to the CVR task and offer some insight into how the multi-task design choices address the two primary problems affecting the CVR task: data sparsity and data bias. Our findings add to the growing body of evidence suggesting that standard multi-task learning is a sensible approach to modelling related events in real-world large-scale applications and suggest the specific multitask approach can be guided by ease of implementation in an existing system.

---

- **arXiv ID**: `2108.13475`
- **Published**: 2021-08-18
- **Categories**: cs.IR, cs.LG
- **PDF**: https://arxiv.org/pdf/2108.13475
