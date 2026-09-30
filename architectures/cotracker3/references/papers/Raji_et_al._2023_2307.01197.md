# Paper (Raji et al. 2023)

> Source: `https://arxiv.org/abs/2307.01197v2`

---

Segment Anything Meets Point Tracking

Frano Rajič
Lei Ke
Yu-Wing Tai
Chi-Keung Tang
Martin Danelljan
Fisher Yu

Published: 2023-07-03T17:58:01Z

Categories: cs.CV

PDF: https://arxiv.org/pdf/2307.01197v2

Abstract
The Segment Anything Model (SAM) has established itself as a powerful zero-shot image segmentation model, enabled by efficient point-centric annotation and prompt-based models. While click and brush interactions are both well explored in interactive image segmentation, the existing methods on videos focus on mask annotation and propagation. This paper presents SAM-PT, a novel method for point-centric interactive video segmentation, empowered by SAM and long-term point tracking. SAM-PT leverages robust and sparse point selection and propagation techniques for mask generation. Compared to traditional object-centric mask propagation strategies, we uniquely use point propagation to exploit local structure information agnostic to object semantics. We highlight the merits of point-based tracking through direct evaluation on the zero-shot open-world Unidentified Video Objects (UVO) benchmark. Our experiments on popular video object segmentation and multi-object segmentation tracking benchmarks, including DAVIS, YouTube-VOS, and BDD100K, suggest that a point-based segmentation tracker yields better zero-shot performance and efficient interactions. We release our code that integrates different point trackers and video segmentation benchmarks at https://github.com/SysCV/sam-pt.
