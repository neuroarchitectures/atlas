# Paper (Ouyang et al. 2022)

> Source: `https://arxiv.org/abs/2203.02155`

---

**Discernment and Social Learning as a Companion Training Layer**

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Ray, Alex et al.

**Abstract**

I propose a companion curriculum for language model training that explicitly practices how a model responds to error. A failed attempt pauses advancement in the affected skill, followed by observation of verified peer successes, judgments committed before outcome disclosure, diagnosis, and a fresh-task retry. Task performance and discernment remain separately assessed, with bounded interaction through curriculum scheduling and training allocation. Demonstrations show evidence use and recovery behavior; rotating social roles practice correction and justified disagreement. An advanced stage has capable teachers develop small student models, challenge them in controlled contests, and apply lessons from student failures to analogous errors in their own work. The central hypothesis is that these experiences can improve reliable correction and transfer beyond direct-answer supervision at matched resource budgets. The proposal specifies operational measures, controls for performed humility and strategic failure, and tests that distinguish context-supported adaptation from persistent changes to model weights. Existing research supports several component mechanisms, but the complete curriculum, three-success observation rule, and additional value of explicit teacher-to-self transfer remain untested. No experimental results are reported.

---

- **arXiv ID**: `2203.02155`
- **Year**: 2022
- **DOI**: https://doi.org/10.48550/arxiv.2203.02155
- **PDF**: https://arxiv.org/pdf/2203.02155
