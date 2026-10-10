# Paper (Teed et al. 2020)

> Source: `https://arxiv.org/abs/2003.12039v3`

---

RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

Zachary Teed
Jia Deng

Published: 2020-03-26T17:12:42Z

Categories: cs.CV

Comment: fixed a formatting issue, Eq 7. no change in content

PDF: https://arxiv.org/pdf/2003.12039v3

Abstract
We introduce Recurrent All-Pairs Field Transforms (RAFT), a new deep network architecture for optical flow. RAFT extracts per-pixel features, builds multi-scale 4D correlation volumes for all pairs of pixels, and iteratively updates a flow field through a recurrent unit that performs lookups on the correlation volumes. RAFT achieves state-of-the-art performance. On KITTI, RAFT achieves an F1-all error of 5.10%, a 16% error reduction from the best published result (6.10%). On Sintel (final pass), RAFT obtains an end-point-error of 2.855 pixels, a 30% error reduction from the best published result (4.098 pixels). In addition, RAFT has strong cross-dataset generalization as well as high efficiency in inference time, training speed, and parameter count. Code is available at https://github.com/princeton-vl/RAFT.
