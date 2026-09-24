# Paper (Wang et al. 2021)

> Source: `https://arxiv.org/abs/2008.13535`

---

DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi
Google Inc.
{ruoxi, rakeshshivanna, zcheng, sagarj, dongl, lichan, edchi}@google.com
Abstract.
Learning effective feature crosses is the key behind building recommender systems. However, the sparse and large feature space requires exhaustive search to identify effective crosses. Deep & Cross Network (DCN) was proposed to automatically and efficiently learn bounded-degree predictive feature interactions. Unfortunately, in models that serve web-scale traffic with billions of training examples, DCN showed limited expressiveness in its cross network at learning more predictive feature interactions. Despite significant research progress made, many deep learning models in production still rely on traditional feed-forward neural networks to learn feature crosses inefficiently.
In light of the pros/cons of DCN and existing feature interaction learning approaches, we
propose an improved framework DCN-V2 to make DCN more practical in large-scale industrial settings. In a comprehensive experimental study with extensive hyper-parameter search and model tuning, we observed that DCN-V2 approaches outperform all the state-of-the-art algorithms on popular benchmark datasets. The improved DCN-V2 is more expressive yet remains cost efficient at feature interaction learning, especially when coupled with a mixture of low-rank architecture. DCN-V2 is simple, can be easily adopted as building blocks, and has delivered significant offline accuracy and online business metrics gains across many web-scale learning to rank systems at Google.
1.
Introduction
Learning to rank (LTR)
(
Liu 2011
;
Cao
et al. 2007
)
has remained to be one of the most important problems in modern-day machine learning and deep learning. It has a wide range of applications in search, recommendation systems
(
Resnick and
Varian 1997
;
Herlocker et al. 2004
;
Schafer
et al. 1999
)
, and computational advertising
(
Broder 2008
;
Bottou
et al. 2013
)
. Among the crucial components of LTR models, learning effective feature crosses continues to attract lots of attention from both academia
(
Qu
et al. 2016
;
Lian
et al. 2018
;
Song et al. 2019
)
and industry
(
Wang
et al. 2017
;
Cheng et al. 2016
;
Guo
et al. 2017
;
Beutel et al. 2018
;
Naumov et al. 2019
)
.
Effective feature crosses are crucial to the success of many models. They provide additional interaction information beyond individual features. For example, the combination of “
country
” and “
language
” is more informative than either one of them.
In the era of linear models, ML practitioners rely on manually identifying such feature crosses
(
Seide
et al. 2011
)
to increase model’s expressiveness.
Unfortunately, this involves a combinatorial search space, which is large and sparse in web-scale applications where the data is mostly categorical.
Searching in such setting is exhaustive, often requires domain expertise, and makes the model harder to generalize.
Later on, embedding techniques have been widely adopted to project features from high-dimensional sparse vectors to much lower-dimensional dense vectors. Factorization Machines (FMs)
(
Rendle 2010
;
Rendle 2012
)
leverage the embedding techniques and construct pairwise feature interactions via the inner-product of two latent vectors. Compared to those traditional feature crosses in linear models, FM brings more generalization capabilities.
In the last decade, with more computing firepower and huge scale of data, LTR models in industry have gradually migrated from linear models and FM-based models to deep neural networks (DNN). This has significantly improved model performance for search and recommendation systems across the board
(
Cheng et al. 2016
;
Wang
et al. 2017
;
Guo
et al. 2017
)
. People generally consider DNNs as universal function approximators, that could potentially learn all kinds of feature interactions
(
Mhaskar 1996
;
Valiant 2014
;
Veit
et al. 2016
)
. However, recent studies
(
Beutel et al. 2018
;
Wang
et al. 2017
)
found that DNNs are inefficient to even approximately model 2nd or 3rd-order feature crosses.
To capture effective feature crosses more accurately, a common remedy is to further increase model capacity through wider or deeper networks. This naturally crafts a double edged sword that we are improving model performance while making models much slower to serve. In many production settings, these models are handling extremely high QPS, thus have very strict latency requirements for real-time inference. Possibly, the serving systems are already pushed to a stretch that cannot afford even larger models. Furthermore, deeper models often introduce trainability issues, making models harder to train.
This has shed light on critical needs to design a model that can efficiently and effectively learn predictive feature interactions, especially in a resource-constraint environment that handles real-time traffic from billions of users. Many recent works
(
Wang
et al. 2017
;
Cheng et al. 2016
;
Guo
et al. 2017
;
Beutel et al. 2018
;
Qu
et al. 2016
;
Lian
et al. 2018
;
Song et al. 2019
;
Naumov et al. 2019
)
tried to tackle this challenge. The common theme is to leverage those
implicit
high-order crosses learned from DNNs, with
explicit
and bounded-degree feature crosses which have been found to be effective in linear models.
Implicit
cross means the interaction is learned through an end-to-end function without any explicit formula modeling such cross.
Explicit
cross, on the other hand, is modeled by an explicit formula with controllable interaction order. We defer a detailed discussion of these models in
Section 2
.
Among these, Deep & Cross Network (DCN)
(
Wang
et al. 2017
)
is effective and elegant, however, productionizing DCN in large-scale industry systems faces many challenges. The expressiveness of its cross network is limited. The polynomial class reproduced by the cross network is only characterized by
O
⁡
(
input size
)
O(\text{input size})
parameters, largely limiting its flexibility in modeling random cross patterns.
Moreover, the allocated capacity between the cross network and DNN is unbalanced. This gap significantly increases when applying DCN to large-scale production data. An overwhelming portion of the parameters will be used to learn implicit crosses in the DNN.
In this paper, we propose a new model
DCN-V2
that improves the original DCN model. We have already successfully deployed DCN-V2 in quite a few learning to rank systems across Google with significant gains in both offline model accuracy and online business metrics. DCN-V2 first learns explicit feature interactions of the inputs (typically the embedding layer) through cross layers, and then combines with a deep network to learn complementary implicit interactions. The core of DCN-V2 is the cross layers, which inherit the simple structure of the cross network from DCN, however significantly more expressive at learning explicit and bounded-degree cross features. The paper studies datasets with clicks as positive labels, however DCN-V2 is label agnostic and can be applied to any learning to rank systems.
The main contributions of the paper are five-fold:
•
We propose a novel model—DCN-V2—to learn effective explicit and implicit feature crosses. Compared to existing methods, our model is more expressive yet remains efficient and simple.
•
Observing the low-rank nature of the learned matrix in DCN-V2, we propose to leverage low-rank techniques to approximate feature crosses in a subspace for better performance and latency trade-offs. In addition, we propose a technique based on the Mixture-of-Expert architecture
(
Shazeer et al. 2017
;
Jacobs
et al. 1991
)
to further decompose the matrix into multiple smaller sub-spaces. These sub-spaces are then aggregated through a gating mechanism.
•
We conduct and provide an extensive study using synthetic datasets, which demonstrates the inefficiency of traditional ReLU-based neural nets to learn high-order feature crosses.
•
Through comprehensive experimental analysis, we demonstrate that our proposed DCN-V2 models significantly outperform SOTA algorithms on Criteo and MovieLen-1M benchmark datasets.
•
We provide a case study and share lessons in productionizing DCN-V2 in a large-scale industrial ranking system, which delivered significant offline and online gains.
The paper is organized as follows.
Section 2
summarizes related work.
Section 3
describes our proposed model architecture DCN-V2 along with its memory efficient version.
Section 4
analyzes DCN-V2.
Section 5
raises a few research questions, which are answered through comprehensive experiments on both synthetic datasets in
Section 6
and public datasets in
Section 7
.
Section 8
describes the process of productionizing DCN-V2 in a web-scale recommender.
2.
Related Work
The core idea of recent feature interaction learning work is to leverage both explicit and implicit (from DNNs) feature crosses. To model explicit crosses, most recent work introduces multiplicative operations (
x
1
×
x
2
x_{1}\times x_{2}
) which is inefficient in DNN, and designs a function
f
⁡
(
𝐱
1
,
𝐱
2
)
f({\bf x}_{1},{\bf x}_{2})
to efficiently and explicitly model the pairwise interactions between features
𝐱
1
{\bf x}_{1}
and
𝐱
2
{\bf x}_{2}
. We organize the work in terms of how they combine the explicit and implicit components.
Parallel Structure.
One line of work jointly trains two parallel networks inspired from
the wide and deep model
(
Cheng et al. 2016
)
, where the wide component takes inputs as crosses of raw features; and the deep component is a DNN model. However, selecting cross features for the wide component falls back to the feature engineering problem for linear models. Nonetheless, the wide and deep model has inspired many works to adopt this parallel architecture and improve upon the wide component.
DeepFM
(
Guo
et al. 2017
)
automates the feature interaction learning in the wide component by adopting a FM model. DCN
(
Wang
et al. 2017
)
introduces a cross network, which learns explicit and bounded-degree feature interactions automatically and efficiently. xDeepFM
(
Lian
et al. 2018
)
increases the expressiveness of DCN by generating multiple feature maps, each encoding all the pairwise interactions between features at current level and the input level. Besides, it also considers each feature embedding
𝐱
i
{\bf x}_{i}
as a unit instead of each element
x
i
x_{i}
as a unit. Unfortunately, its computational cost is significantly high (10x of #params), making it impractical for industrial-scale applications. Moreover, both DeepFM and xDeepFM require all the feature embeddings to be of equal size, yet another limitation when applying to industrial data where the vocab sizes (sizes of categorical features) vary from
O
⁡
(
10
)
O(10)
to millions. AutoInt
(
Song et al. 2019
)
leverages the multi-head self-attention mechanism with residual connections. InterHAt
(
Li
et al. 2020
)
further employs Hierarchical Attentions.
Stacked Structure.
Another line of work introduces an interaction layer—which creates explicit feature crosses—in between the embedding layer and a DNN model. This interaction layer captures feature interaction at an early stage, and facilitates the learning of subsequent hidden layers. Product-based neural network (PNN)
(
Qu
et al. 2016
)
introduces inner (IPNN) and outer (OPNN) product layer as the pairwise interaction layers. One downside of OPNN lies in its high computational cost. Neural FM (NFM)
(
He and Chua 2017
)
extends FM by replacing the inner-product with a Hadamard product;
DLRM
(
Naumov et al. 2019
)
follows FM to compute the feature crosses through inner products;
These models can only create up to 2nd-order explicit crosses. AFN
(
Cheng
et al. 2019
)
transforms features into a log space and adaptively learns arbitrary-order feature interactions. Similar to DeepFM and xDeepFM, they only accept embeddings of equal sizes.
Despite many advances made, our comprehensive experiments (
Section 7
) demonstrate that DCN still remains to be a strong baseline. We attribute this to its simple structure that has facilitated the optimization. However, as discussed, its limited expressiveness has prevented it from learning more effective feature crosses in web-scale systems. In the following, we present a new architecture that inherits DCN’s simple structure while increasing its expressiveness.
3.
Proposed Architecture: DCN-V2
This section describes a novel model architecture — DCN-V2 — to learn both explicit and implicit feature interactions. DCN-V2 starts with an
embedding layer
, followed by a
cross network
containing multiple cross layers that models explicit feature interactions, and then combines with a
deep network
that models implicit feature interactions. The improvements made in DCN-V2 are
critical for putting DCN into practice for highly-optimized production systems
. DCN-V2 significantly improves the expressiveness of DCN
(
Wang
et al. 2017
)
in modeling complex explicit cross terms in web-scale production data, while maintaining its elegant formula for easy deployment. The function class modeled by DCN-V2 is a strict superset of that modeled by DCN. The overall model architecture is depicted in Fig.
1
, with two ways to combine the cross network with the deep network: (1) stacked and (2) parallel. In addition, observing the low-rank nature of the cross layers, we propose to leverage a mixture of low-rank cross layers to achieve healthier trade-off between model performance and efficiency.
(a)
Stacked
(b)
Parallel
Figure 1
.
Visualization of DCN-V2.
⊗
\otimes
represents the cross operation in Eq. (
1
),
i.e.
,
𝐱
l
+
1
=
𝐱
0
⊙
(
W
l
​
𝐱
l
+
𝐛
l
)
+
𝐱
l
{\bf x}_{l+1}={\bf x}_{0}\odot(W_{l}{\bf x}_{l}+{\bf b}_{l})+{\bf x}_{l}
.
3.1.
Embedding Layer
The embedding layer takes input as a combination of categorical (sparse) and dense features, and outputs
𝐱
0
∈
ℝ
d
{\bf x}_{0}\in\mathbb{R}^{d}
. For the
i
i
-th categorical feature, we project it from a high-dimensional sparse space to a lower-dimensional dense space via
𝐱
embed
,
i
=
W
embed
,
i
​
𝐞
i
{\bf x}_{\text{embed},i}=W_{\text{embed},i}{\bf e}_{i}
,
where
𝐞
i
∈
{
0
,
1
}
v
i
{\bf e}_{i}\in\{0,1\}^{v_{i}}
;
W
∈
ℝ
e
i
×
v
i
W\in\mathbb{R}^{e_{i}\times v_{i}}
is a learned projection matrix;
𝐱
embed
,
i
∈
ℝ
e
i
{\bf x}_{\text{embed},i}\in\mathbb{R}^{e_{i}}
is the dense embedded vector;
v
i
v_{i}
and
e
i
e_{i}
represents vocab and embedding sizes respectively. For multivalent features, we use the mean of the embedded vectors as the final vector.
The output is the concatenation of all the embedded vectors and the normalized dense features:
𝐱
0
=
[
𝐱
embed
,
1
;
…
;
𝐱
embed
,
n
;
x
dense
]
{\bf x}_{0}=[{\bf x}_{\text{embed},1};\ldots;{\bf x}_{\text{embed},n};x_{\text{dense}}]
.
Unlike many related works
(
Song et al. 2019
;
Lian
et al. 2018
;
Qu
et al. 2016
;
Guo
et al. 2017
;
Naumov et al. 2019
;
He and Chua 2017
)
which requires
e
i
=
e
j
​
∀
i
,
j
e_{i}=e_{j}\penalty\ \forall i,j
, our model accepts arbitrary embedding sizes. This is particularly important for industrial recommenders where the vocab size varies from
O
⁡
(
10
)
O(10)
to
O
⁡
(
10
5
)
O(10^{5})
. Moreover, our model isn’t limited to the above described embedding method; any other embedding techniques such as hashing could be adopted.
3.2.
Cross Network
The core of DCN-V2 lies in the cross layers that create explicit feature crosses. Eq. (
1
) shows the
(
l
+
1
)
th
(l+1)^{\text{th}}
cross layer.
(1)
𝐱
l
+
1
=
𝐱
0
⊙
(
W
l
​
𝐱
l
+
𝐛
l
)
+
𝐱
l
{\bf x}_{l+1}={\bf x}_{0}\odot(W_{l}{\bf x}_{l}+{\bf b}_{l})+{\bf x}_{l}
where
𝐱
0
∈
ℝ
d
{\bf x}_{0}\in\mathbb{R}^{d}
is the base layer that contains the original features of order 1, and is normally set as the embedding (input) layer.
𝐱
l
,
𝐱
l
+
1
∈
ℝ
d
{\bf x}_{l},{\bf x}_{l+1}\in\mathbb{R}^{d}
, respectively, represents the input and output of the
(
l
+
1
)
(l+1)
-th cross layer.
W
l
∈
ℝ
d
×
d
W_{l}\in\mathbb{R}^{d\times d}
and
𝐛
l
∈
ℝ
d
{\bf b}_{l}\in\mathbb{R}^{d}
are the learned weight matrix and bias vector.
Figure 2
shows how an individual cross layer functions.
Figure 2
.
Visualization of a cross layer.
For an
l
l
-layered cross network, the highest polynomial order is
l
+
1
l+1
and the network contains all the feature crosses up to the highest order. Please see
Section 4.1
for a detailed analysis, both from bitwise and feature-wise point of views. When
W
=
𝟏
×
𝐰
⊤
W={\bf 1}\times{\bf w}^{\top}
, where
𝟏
{\bf 1}
represents a vector of ones, DCN-V2 falls back to DCN.
The cross layers could only reproduce polynomial function classes of bounded degree; any other complex function space could only be approximated
1
1
1
Any function with certain smoothness assumptions can be well-approximated by polynomials. In fact, we’ve observed in our experiments that cross network alone was able to achieve similar performance as traditional deep networks.
. Hence, we introduce a deep network next to complement the modeling of the inherent distribution in the data.
3.3.
Deep Network
The
l
th
l^{\text{th}}
deep layer’s formula is given by
𝐡
l
+
1
=
f
⁡
(
W
l
​
𝐡
l
+
𝐛
l
)
{\bf h}_{l+1}=f(W_{l}{\bf h}_{l}+{\bf b}_{l})
,
where
𝐡
l
∈
ℝ
d
l
,
𝐡
l
+
1
∈
ℝ
d
l
+
1
{\bf h}_{l}\in\mathbb{R}^{d_{l}},{\bf h}_{l+1}\in\mathbb{R}^{d_{l+1}}
, respectively, are the input and output of the
l
l
-th deep layer;
W
l
∈
ℝ
d
l
×
d
l
+
1
W_{l}\in\mathbb{R}^{d_{l}\times d_{l+1}}
is the weight matrix and
𝐛
l
∈
ℝ
d
l
+
1
{\bf b}_{l}\in\mathbb{R}^{d_{l+1}}
is the bias vector;
f
⁡
(
⋅
)
f(\cdot)
is an elementwise activation function and we set it to be ReLU; any other activation functions are also suitable.
3.4.
Deep and Cross Combination
We seek structures to combine the cross network and deep network. Recent literature adopted two structures: stacked and parallel. In practice, we have found that which architecture works better is data dependent. Hence, we present both structures:
Stacked Structure (
1(a)
):
The input
𝐱
0
{\bf x}_{0}
is fed to the cross network followed by the deep network, and the final layer is given by
𝐱
final
=
𝐡
L
d
,
𝐡
0
=
𝐱
L
c
{\bf x}_{\text{final}}={\bf h}_{L_{d}},\penalty\ {\bf h}_{0}={\bf x}_{L_{c}}
, which models the data as
f
deep
∘
f
cross
f_{\text{deep}}\circ f_{\text{cross}}
.
Parallel Structure (
1(b)
):
The input
𝐱
0
{\bf x}_{0}
is fed in parallel to both the cross and deep networks; then, the outputs
𝐱
L
c
{\bf x}_{L_{c}}
and
𝐡
L
d
{\bf h}_{L_{d}}
are concatenated to create the final output layer
𝐱
final
=
[
𝐱
L
c
;
𝐡
L
d
]
{\bf x}_{\text{final}}=[{\bf x}_{L_{c}};{\bf h}_{L_{d}}]
. This structure models the data as
f
cross
+
f
deep
f_{\text{cross}}+f_{\text{deep}}
.
In the end, the prediction
y
^
i
\hat{y}_{i}
is computed as:
y
^
i
=
σ
⁡
(
𝐰
logit
⊤
​
𝐱
final
)
\hat{y}_{i}=\sigma({\bf w}_{\text{logit}}^{\top}{\bf x}_{\text{final}})
,
where
𝐰
logit
{\bf w}_{\text{logit}}
is the weight vector for the logit, and
σ
⁡
(
x
)
=
1
/
(
1
+
exp
⁡
(
−
x
)
)
\sigma(x)=1/(1+\exp(-x))
. For the final loss, we use the Log Loss that is commonly used for learning to rank systems especially with a binary label (e.g., click). Note that DCN-V2 itself is both prediction-task and loss-function agnostic.
loss
=
−
1
N
∑
i
=
1
N
y
i
log
(
y
^
i
)
+
(
1
−
y
i
)
log
(
1
−
y
^
i
)
+
λ
∑
l
∥
W
l
∥
2
2
,
\begin{split}\text{loss}=&-\frac{1}{N}\sum_{i=1}^{N}y_{i}\log(\hat{y}_{i})+(1-y_{i})\log(1-\hat{y}_{i})+\lambda\sum_{l}\|W_{l}\|^{2}_{2},\end{split}
where
y
^
i
\hat{y}_{i}
’s are predictions;
y
i
y_{i}
’s are the true labels;
N
N
is the total number of inputs; and
λ
\lambda
is the
L
2
L_{2}
regularization parameter.
3.5.
Cost-Effective Mixture of Low-Rank DCN
In real production models, the model capacity is often constrained by limited serving resources and strict latency requirements. It is often the case that we have to seek methods to reduce cost while maintaining the accuracy. Low-rank techniques
(
Golub and
Van Loan 1996
)
are widely used
(
Jaderberg
et al. 2014
;
Yu
et al. 2017
;
Chen
et al. 2018
;
Wang
et al. 2019
;
Halko
et al. 2011
;
Drineas and
Mahoney 2005
)
to reduce the computational cost. It approximates a dense matrix
M
∈
ℝ
d
×
d
M\in\mathbb{R}^{d\times d}
by two tall and skinny matrices
U
,
V
∈
ℝ
d
×
r
U,V\in\mathbb{R}^{d\times r}
. When
r
≤
d
/
2
r\leq d/2
, the cost will be reduced. However, they are most effective when the matrix shows a large gap in singular values or a fast spectrum decay. In many settings, we indeed observe that the learned matrix is numerically low-rank in practice.
Fig.
3(a)
shows the singular decay pattern of the learned matrix
W
W
in DCN-V2 (see Eq. (
1
)) from a production model. Compared to the initial matrix, the learned matrix shows a much faster spectrum decay pattern. Let’s define the numerical rank
R
T
R_{T}
with tolerance T to be
argmin
k
​
(
σ
k
<
T
⋅
σ
1
)
\text{argmin}_{k}(\sigma_{k}<T\cdot\sigma_{1})
, where
σ
1
≥
σ
2
≥
,
…
,
≥
σ
n
\sigma_{1}\geq\sigma_{2}\geq,\ldots,\geq\sigma_{n}
are the singular values. Then,
R
T
R_{T}
means majority of the mass up to tolerance
T
T
, is preserved in the top
k
k
singular values. In the field of machine learning and deep learning, a model could still work surprisingly well with a reasonably high tolerance
T
T
2
2
2
This is very different from the filed of scientific computing (
e.g.
, solving linear systems), where the approximation accuracy need to be very high. For problems such as CTR prediction, some errors could introduce regularization effect to the model.
.
(a)
Singular Values
(b)
Mixture of Low-rank Experts
Figure 3
.
Left: Singular value decay of the learned DCN-V2 weight matrix. The singular values are normalized and
1
=
σ
1
≥
σ
2
≥
…
≥
σ
k
1=\sigma_{1}\geq\sigma_{2}\geq\ldots\geq\sigma_{k}
.
+
+
represents the randomly initialized truncated normal matrix;
×
\times
represents the final learned matrix. Right: Visualization of mixture of low-rank cross layer.
Hence, it is well-motivated to impose a low-rank structure on
W
W
. Eq (
2
) shows the resulting
(
l
+
1
)
(l+1)
-th low-rank cross layer
(2)
𝐱
l
+
1
=
𝐱
0
⊙
(
U
l
​
(
V
l
⊤
​
𝐱
i
)
+
𝐛
l
)
+
𝐱
i
{\bf x}_{l+1}={\bf x}_{0}\odot\Big(U_{l}\big(V_{l}^{\top}{\bf x}_{i}\big)+{\bf b}_{l}\Big)+{\bf x}_{i}
where
U
l
,
V
l
∈
ℝ
d
×
r
U_{l},V_{l}\in\mathbb{R}^{d\times r}
and
r
≪
d
r\ll d
. Eq (
2
) has two
interpretations
: 1) we learn feature crosses in a subspace; 2) we project the input
𝐱
{\bf x}
to lower-dimensional
ℝ
r
\mathbb{R}^{r}
, and then project it back to
ℝ
d
\mathbb{R}^{d}
. The two interpretations have inspired the following two model improvements.
Interpretation 1 inspires us to adopt the idea from Mixture-of-Experts (MoE)
(
Shazeer et al. 2017
;
Jacobs
et al. 1991
;
Eigen
et al. 2013
;
Ma
et al. 2018
)
. MoE-based models consist of two components: experts (typically a small network) and gating (a function of inputs). In our case, instead of relying on one single expert (Eq (
2
)) to learn feature crosses, we leverage multiple such experts, each learning feature interactions in a different subspaces, and adaptively combine the learned crosses using a gating mechanism that depends on input
𝐱
{\bf x}
. The resulting mixture of low-rank cross layer formulation is shown in Eq. (
3
) and depicted in
3(b)
.
(3)
𝐱
l
+
1
=
∑
i
=
1
K
G
i
​
(
𝐱
l
)
​
E
i
​
(
𝐱
l
)
+
𝐱
l
E
i
​
(
𝐱
l
)
=
𝐱
0
⊙
(
U
l
i
​
(
V
l
i
⊤
​
𝐱
l
)
+
𝐛
l
)
\begin{split}{\bf x}_{l+1}&=\sum\nolimits_{i=1}^{K}G_{i}({\bf x}_{l})E_{i}({\bf x}_{l})+{\bf x}_{l}\\
E_{i}({\bf x}_{l})&={\bf x}_{0}\odot\Big(U_{l}^{i}\big(V_{l}^{i\top}{\bf x}_{l}\big)+{\bf b}_{l}\Big)\end{split}
where
K
K
is the number of experts;
G
i
​
(
⋅
)
:
ℝ
d
↦
ℝ
G_{i}(\cdot):\mathbb{R}^{d}\mapsto\mathbb{R}
is the gating function, common sigmoid or softmax;
E
i
​
(
⋅
)
:
ℝ
d
↦
ℝ
d
E_{i}(\cdot):\mathbb{R}^{d}\mapsto\mathbb{R}^{d}
is the
i
th
i^{\text{th}}
expert in learning feature crosses.
G
⁡
(
⋅
)
G(\cdot)
dynamically weights each expert for input
𝐱
{\bf x}
, and when
G
⁡
(
⋅
)
≡
1
G(\cdot)\equiv 1
, Eq (
3
) falls back to Eq (
2
).
Interpretation 2 inspires us to leverage the low-dimensional nature of the projected space. Instead of immediately projecting back from dimension
d
′
d^{\prime}
to
d
d
(
d
′
≪
d
d^{\prime}\ll d
), we further apply nonlinear transformations in the projected space to refine the representation
(
Fan et al. 2019
)
.
(4)
E
i
​
(
𝐱
l
)
=
𝐱
0
⊙
(
U
l
i
⋅
g
⁡
(
C
l
i
⋅
g
⁡
(
V
l
i
⊤
​
𝐱
l
)
)
+
𝐛
l
)
E_{i}({\bf x}_{l})={\bf x}_{0}\odot\Big(U_{l}^{i}\cdot g\big(C_{l}^{i}\cdot g\big(V_{l}^{i\top}{\bf x}_{l}\big)\big)+{\bf b}_{l}\Big)
where
g
⁡
(
⋅
)
g(\cdot)
represents any nonlinear activation function.
Discussions.
This section aims to make effective use of the fixed memory/time budget to learn meaningful feature crosses. From Eqs (
1
)–(
4
), each formula represents a strictly larger function class assuming a fixed #params.
Different from many model compression techniques where the compression is conducted post-training, our model imposes the structure prior to training and jointly learn the associated parameters with the rest of the parameters. Due to that, the cross layer is an integral part of the nonlinear system
f
(
𝐱
)
=
(
f
k
(
W
k
)
∘
⋯
∘
f
1
(
W
1
)
)
(
𝐱
)
f({\bf x})=\big(f_{k}(W_{k})\circ\cdots\circ f_{1}(W_{1})\big)({\bf x})
, where
(
f
i
+
1
∘
f
i
)
​
(
⋅
)
≔
f
i
+
1
​
(
f
i
​
(
⋅
)
)
(f_{i+1}\circ f_{i})(\cdot)\coloneqq f_{i+1}(f_{i}(\cdot))
. Hence, the training dynamics of the overall system might be affected, and it would be interesting to see how the global statistics, such as Jacobian and Hession matrices of
f
⁡
(
𝐱
)
f({\bf x})
, are affected. We leave such investigations to future work.
3.6.
Complexity Analysis
Let
d
d
denote the embedding size,
L
c
L_{c}
denote the number of cross layers,
K
K
denote the number of low-rank DCN experts. Further, for simplicity, we assume each expert has the same smaller dimension
r
r
(upper bound on the rank).
The time and space complexity for the cross network is
O
⁡
(
d
2
​
L
c
)
O(d^{2}L_{c})
, and for mixture of low-rank DCN (DCN-Mix) it’s efficient when
r
​
K
≪
d
rK\ll d
with
O
⁡
(
2
​
d
​
r
​
K
​
L
c
)
O(2drKL_{c})
.
4.
Model Analysis
This section analyzes DCN-V2 from polynomial approximation point of view, and makes connections to related work. We adopt the notations from
(
Wang
et al. 2017
)
.
Notations.
Let the embedding vector
𝐱
=
[
𝐱
1
;
𝐱
2
;
…
;
𝐱
k
]
=
[
x
1
,
x
2
,
…
,
x
d
]
∈
ℝ
d
{\bf x}=[{\bf x}_{1};{\bf x}_{2};\ldots;{\bf x}_{k}]=[x_{1},x_{2},\ldots,x_{d}]\in\mathbb{R}^{d}
be a column vector, where
𝐱
i
∈
ℝ
e
i
{\bf x}_{i}\in\mathbb{R}^{e_{i}}
represents the
i
i
-th feature embedding, and
x
i
x_{i}
represents the
i
i
-th element in
𝐱
{\bf x}
. Let multi-index
𝜶
=
[
α
1
,
⋯
,
α
d
]
∈
ℕ
d
{\bm{\alpha}}=[\alpha_{1},\cdots,\alpha_{d}]\in\mathbb{N}^{d}
and
|
𝜶
|
=
∑
i
=
1
d
α
i
|{\bm{\alpha}}|=\sum_{i=1}^{d}\alpha_{i}
.
C
a
b
≔
{
𝐲
∈
{
1
,
⋯
,
a
}
b
|
∀
i
<
j
,
y
i
>
y
j
}
C_{a}^{b}\coloneqq\bigl\{{\bf y}\in\{1,\cdots,a\}^{b}\mathrel{\big|}\forall i<j,y_{i}>y_{j}\bigr\}
. Let
𝟏
\mathbf{1}
be a vector of all 1’s, and
I
I
be an identity matrix. We use capital letters for matrices, bold lower-case letters for vectors, and normal lower-case letters for scalars.
4.1.
Polynomial Approximation
We analyze DCN-V2 from two perspectives of polynomial approximation —
1) Considering each element (bit)
x
i
x_{i}
as a unit, and analyzes interactions among the elements (
Theorem 4.1
); and 2) Considering each feature embedding
𝐱
i
{\bf x}_{i}
as a unit, and only analyzes the feature-wise interactions (
Theorem 4.2
) (proofs in Appendix).
Theorem 4.1 (Bitwise).
Assume the input to an
l
l
-layer cross network be
𝐱
∈
ℝ
d
{\bf x}\in\mathbb{R}^{d}
, the output be
f
l
​
(
𝐱
)
=
𝟏
⊤
​
𝐱
l
f_{l}({\bf x})={\bf 1}^{\top}{\bf x}^{l}
, and the
i
th
i^{\text{th}}
layer is defined as
𝐱
i
=
𝐱
⊙
W
(
i
−
1
)
​
𝐱
i
−
1
+
𝐱
i
−
1
{\bf x}^{i}={\bf x}\odot W^{(i-1)}{\bf x}^{i-1}+{\bf x}^{i-1}
. Then, the multivariate polynomial
f
l
​
(
𝐱
)
f_{l}({\bf x})
reproduces polynomials in the following class:
{
∑
𝜶
c
𝜶
(
W
(
1
)
,
…
,
W
(
l
)
)
x
1
α
1
x
2
α
2
…
x
d
α
d
|
0
≤
|
𝜶
|
≤
l
+
1
,
𝜶
∈
ℕ
d
}
,
\biggl\{\sum_{{\bm{\alpha}}}c_{{\bm{\alpha}}}\left(W^{(1)},\ldots,W^{(l)}\right)x_{1}^{\alpha_{1}}x_{2}^{\alpha_{2}}\ldots x_{d}^{\alpha_{d}}\mathrel{\bigg|}0\leq|{\bm{\alpha}}|\leq l+1,{\bm{\alpha}}\in\mathbb{N}^{d}\biggr\},
where
c
𝛂
=
∑
𝐣
∈
C
l
|
𝛂
|
−
1
∑
𝐢
∈
P
𝛂
∏
k
=
1
|
𝛂
|
−
1
w
i
k
​
i
k
+
1
(
j
k
)
c_{\bm{\alpha}}=\sum_{{\bf j}\in C_{l}^{|{\bm{\alpha}}|-1}}\sum_{{\bf i}\in P_{\bm{\alpha}}}\prod_{k=1}^{|{\bm{\alpha}}|-1}w_{i_{k}i_{k+1}}^{(j_{k})}
,
w
i
​
j
(
k
)
w_{ij}^{(k)}
is the
(
i
,
j
)
th
(i,j)^{\text{th}}
element of matrix
W
(
k
)
W^{(k)}
, and
P
𝛂
=
Permutations
(
∪
i
{
i
,
…
,
i
⏟
α
i
​
times
|
α
i
≠
0
}
)
P_{\bm{\alpha}}=\text{Permutations}\penalty\ (\cup_{i}\{\underbrace{i,\ldots,i}_{\alpha_{i}\text{times}}\mathrel{|}\alpha_{i}\neq 0\})
.
Theorem 4.2 (feature-wise).
With the same setting as in
Theorem 4.1
, we further assume input
𝐱
=
[
𝐱
1
;
…
;
𝐱
k
]
{\bf x}=[{\bf x}_{1};\ldots;{\bf x}_{k}]
contains
k
k
feature embeddings and consider each
𝐱
i
{\bf x}_{i}
as a unit. Then, the output
𝐱
l
{\bf x}^{l}
of an
l
l
-layer cross network creates all the feature interactions up to order
l
+
1
l+1
. Specifically, for features with their (repeated) indices in
I
I
, let
P
I
=
P
​
e
​
r
​
m
​
u
​
t
​
a
​
t
​
i
​
o
​
n
​
s
​
(
I
)
P_{I}=Permutations(I)
, then their order-
p
p
interaction is given by:
∑
𝐢
∈
P
I
∑
𝐣
∈
C
p
p
−
1
𝐱
i
1
⊙
(
W
i
1
,
i
2
(
j
1
)
​
𝐱
i
2
⊙
…
⊙
(
W
i
k
,
i
k
+
1
(
j
k
)
​
𝐱
i
l
+
1
)
)
\begin{split}\sum_{{\bf i}\in P_{I}}\sum_{{\bf j}\in C_{p}^{p-1}}{\bf x}_{i_{1}}\odot\left(W_{i_{1},i_{2}}^{(j_{1})}{\bf x}_{i_{2}}\odot\ldots\odot\left(W_{i_{k},i_{k+1}}^{(j_{k})}{\bf x}_{i_{l+1}}\right)\right)\end{split}
From both bitwise and feature-wise perspectives, the cross network is able to create all the feature interactions up to order
l
+
1
l+1
for an
l
l
-layered cross network. Compared to DCN-V, DCN-V2 characterizes the same polynomial class with more parameters and is more expressive. Moreover, the feature interactions in DCN-V2 is more expressive and can be viewed both bitwise and feature-wise, whereas in DCN it is only bitwise
(
Wang
et al. 2017
;
Lian
et al. 2018
;
Song et al. 2019
)
.
4.2.
Connections to Related Work
We study the connections between DCN-V2 and other SOTA feature interaction learning methods; we only focus on the feature interaction component of each model and ignore the DNN component. For comparison purposes, we assume the feature embeddings are of equal size
e
e
.
DCN.
Our proposed model was largely inspired from DCN
(
Wang
et al. 2017
)
. Let’s take the efficient projection view of DCN
(
Wang
et al. 2017
)
,
i.e.
, it implicitly generates all the pairwise crosses and then projects it to a lower-dimensional space; DCN-V2 is similar with a different projection structure.
𝐱
DCN
⊤
=
𝐱
pairs
​
[
𝐰
𝟎
…
𝟎
𝟎
𝐰
…
𝟎
⋱
𝟎
𝟎
…
𝐰
]
,
𝐱
DCN-V2
⊤
=
𝐱
pairs
​
[
𝐰
1
𝟎
…
𝟎
𝟎
𝐰
2
…
𝟎
⋱
𝟎
𝟎
…
𝐰
d
]
\begin{split}{\bf x}_{\text{DCN}}^{\top}={\bf x}_{\text{pairs}}\left[\begin{smallmatrix}{\bf w}&{\bf 0}&\ldots&{\bf 0}\\
{\bf 0}&{\bf w}&\ldots&{\bf 0}\vskip-3.01389pt\\
\vdots&\vdots&\ddots&\vdots\\
{\bf 0}&{\bf 0}&\ldots&{\bf w}\end{smallmatrix}\right],{\bf x}_{\text{{DCN-V2}}}^{\top}={\bf x}_{\text{pairs}}\left[\begin{smallmatrix}{\bf w}_{1}&{\bf 0}&\ldots&{\bf 0}\\
{\bf 0}&{\bf w}_{2}&\ldots&{\bf 0}\vskip-3.01389pt\\
\vdots&\vdots&\ddots&\vdots\\
{\bf 0}&{\bf 0}&\ldots&{\bf w}_{d}\end{smallmatrix}\right]\end{split}
where
𝐱
pairs
=
[
x
i
​
x
~
j
]
∀
i
,
j
{\bf x}_{\text{pairs}}=[x_{i}\tilde{x}_{j}]_{\forall i,j}
contains all the
d
2
d^{2}
pairwise interactions between
𝐱
0
{\bf x}_{0}
and
𝐱
~
\tilde{\bf x}
;
𝐰
∈
ℝ
d
{\bf w}\in\mathbb{R}^{d}
is the weight vector in DCN-V;
𝐰
i
∈
ℝ
d
{\bf w}_{i}\in\mathbb{R}^{d}
is the
i
th
i^{\text{th}}
column of the weight matrix in DCN-V2 (Eq.(
1
)).
DLRM and DeepFM.
Both are essentially 2nd-order FM without the DNN component (ignoring small differences). Hence, we simplify our analysis and compare with FM which has formula
𝐱
⊤
​
𝜷
+
∑
i
<
j
w
i
​
j
​
⟨
𝐱
i
,
𝐱
j
⟩
{\bf x}^{\top}{\bm{\beta}}+\sum_{i<j}w_{ij}\langle{\bf x}_{i},{\bf x}_{j}\rangle
.
This is equivalent to 1-layer DCN-V2 (Eq. (
1
) without residual term) with a structured weight matrix.
𝟏
⊤
​
(
[
𝐱
1
𝐱
2
𝐱
k
]
⊙
(
[
𝟎
w
12
​
I
⋯
w
1
​
k
​
I
𝟎
𝟎
⋯
w
2
​
k
​
I
⋱
𝟎
𝟎
⋯
𝟎
]
​
[
𝐱
1
𝐱
2
𝐱
k
]
+
[
𝜷
1
𝜷
2
𝜷
k
]
)
)
\begin{split}\mathbf{1}^{\top}\left(\left[\begin{smallmatrix}{\bf x}_{1}\\
{\bf x}_{2}\vskip-3.01389pt\\
\vdots\\
{\bf x}_{k}\end{smallmatrix}\right]\odot\left(\left[\begin{smallmatrix}{\bf 0}&w_{12}I&\cdots&w_{1k}I\\
{\bf 0}&{\bf 0}&\cdots&w_{2k}I\vskip-3.01389pt\\
\vdots&\vdots&\ddots&\vdots\\
{\bf 0}&{\bf 0}&\cdots&{\bf 0}\end{smallmatrix}\right]\left[\begin{smallmatrix}{\bf x}_{1}\\
{\bf x}_{2}\vskip-3.01389pt\\
\vdots\\
{\bf x}_{k}\end{smallmatrix}\right]+\left[\begin{smallmatrix}{\bm{\beta}}_{1}\\
{\bm{\beta}}_{2}\vskip-3.01389pt\\
\vdots\\
{\bm{\beta}}_{k}\end{smallmatrix}\right]\right)\right)\end{split}
xDeepFM.
The
h
h
-th feature map at the
k
k
-th layer is given by:
𝐱
h
,
∗
k
=
∑
i
=
1
k
−
1
∑
j
=
1
m
w
i
​
j
k
,
h
​
(
𝐱
i
,
∗
k
−
1
⊙
𝐱
j
)
{\bf x}_{h,*}^{k}=\sum\nolimits_{i=1}^{k-1}\sum\nolimits_{j=1}^{m}w_{ij}^{k,h}({\bf x}_{i,*}^{k-1}\odot{\bf x}_{j})
The
h
h
-th feature map at the 1st layer is equivalent to 1-layer DCN-V2 (Eq. (
1
) without residual term).
𝐱
h
,
∗
1
=
[
I
,
I
,
⋯
,
I
]
(
𝐱
⊙
(
W
𝐱
)
)
=
∑
i
=
1
k
𝐱
i
⊙
(
W
i
,
:
𝐱
)
{\bf x}_{h,*}^{1}=[I,I,\cdots,I]\left({\bf x}\odot(W{\bf x})\right)=\sum\nolimits_{i=1}^{k}{\bf x}_{i}\odot(W_{i,:}{\bf x})
where the
(
i
,
j
)
(i,j)
-th block
W
i
,
j
=
w
i
​
j
⋅
I
W_{i,j}=w_{ij}\cdot I
, and
W
i
,
:
≔
[
W
i
,
1
,
…
,
W
i
,
k
]
W_{i,:}\coloneqq[W_{i,1},\ldots,W_{i,k}]
.
AutoInt.
The interaction layer of AutoInt adopted the multi-head self-attention mechanism. For simplicity, we assume a single head is used in AutoInt; multi-head case could be compared summarily using concatenated cross layers.
From a high-level view, the 1st layer of AutoInt outputs
𝐱
~
=
[
𝐱
~
1
;
𝐱
~
2
;
…
;
𝐱
~
k
]
\widetilde{\bf x}=[\widetilde{\bf x}_{1};\widetilde{\bf x}_{2};\ldots;\widetilde{\bf x}_{k}]
, where
𝐱
~
i
\widetilde{\bf x}_{i}
encodes all the 2nd-order feature interactions with the i-th feature. Then,
𝐱
~
\widetilde{\bf x}
is fed to the 2nd layer to learn higher-order interactions. This is the same as DCN-V2.
From a low-level view (ignoring the residual terms),
𝐱
~
i
=
R
​
e
​
L
​
U
​
(
∑
j
=
1
k
exp
⁡
(
⟨
W
q
​
𝐱
i
,
W
k
​
𝐱
j
⟩
)
∑
j
exp
⁡
(
⟨
W
q
​
𝐱
i
,
W
k
​
𝐱
j
⟩
)
​
(
W
v
​
𝐱
j
)
)
=
R
​
e
​
L
​
U
​
(
∑
j
=
1
k
softmax
​
(
𝐱
i
⊤
​
W
~
​
𝐱
j
)
​
W
v
​
𝐱
j
)
\small\begin{split}\widetilde{\bf x}_{i}&=ReLU\left(\sum\nolimits_{j=1}^{k}\frac{\exp\left(\langle W_{\text{q}}{\bf x}_{i},W_{\text{k}}{\bf x}_{j}\rangle\right)}{\sum\nolimits_{j}\exp\left(\langle W_{\text{q}}{\bf x}_{i},W_{\text{k}}{\bf x}_{j}\rangle\right)}(W_{\text{v}}{\bf x}_{j})\right)\\
&=ReLU\big(\sum\nolimits_{j=1}^{k}\text{softmax}({\bf x}_{i}^{\top}\widetilde{W}{\bf x}_{j})\penalty\ W_{\text{v}}{\bf x}_{j}\big)\end{split}
where
⟨
⋅
,
⋅
⟩
\langle\cdot,\cdot\rangle
represents inner (dot) product, and
W
~
=
W
q
​
W
k
\widetilde{W}=W_{\text{q}}W_{\text{k}}
.
While in DCN-V2,
(5)
𝐱
~
i
=
∑
j
=
1
k
𝐱
i
⊙
(
W
i
,
j
𝐱
j
)
=
𝐱
i
⊙
(
W
i
,
:
𝐱
)
\begin{split}\widetilde{\bf x}_{i}=\sum\nolimits_{j=1}^{k}{\bf x}_{i}\odot(W_{i,j}{\bf x}_{j})={\bf x}_{i}\odot(W_{i,:}{\bf x})\end{split}
where
W
i
,
j
W_{i,j}
represents the
(
i
,
j
)
(i,j)
-th block of
W
W
. It is clear that the difference lies in how we model the feature interactions. AutoInt claims the non-linearity was from ReLU(
⋅
\cdot
); we consider each summation term to also contribute. Differently, DCN-V2 used
𝐱
i
⊙
W
i
,
j
​
𝐱
j
{\bf x}_{i}\odot W_{i,j}{\bf x}_{j}
.
PNN.
The inner-product version (IPNN) is similar to FM. For the outer-product version (OPNN), it first explicitly creates all the
d
2
d^{2}
pairwise interactions,
and then projects them to a lower dimensional space
d
′
d^{\prime}
using a
d
′
d^{\prime}
by
d
2
d^{2}
dense matrix. Differently, DCN-V2 implicitly creates the interactions using a structured matrix.
5.
Research Questions
We are interested to seek answers for these following research questions:
RQ1
When would feature interaction learning methods become more efficient than ReLU-based DNNs?
RQ2
How does the feature-interaction component of each baseline perform without integrating with DNN?
RQ3
How does the proposed mDCN approaches compare to the baselines? Could we achieve healthier trade-off between model accuracy and cost through mDCN and the mixture of low-rank DCN?
RQ4
How does the settings in mDCN affect model quality?
RQ5
Is mDCN capturing important feature crosses? Does the model provide good understandability?
Throughout the paper, “CrossNet" or “CN" represents the cross network; suffix “Mix" denotes the mixture of low-rank version.
6.
Empirical understanding of feature crossing techniques (RQ1)
Many recent works
(
Wang
et al. 2017
;
Cheng et al. 2016
;
Guo
et al. 2017
;
Beutel et al. 2018
;
Qu
et al. 2016
;
Lian
et al. 2018
;
Naumov et al. 2019
)
proposed to model explicit feature crosses that couldn’t be learned efficiently from traditional neural networks. However, most works only studied public datasets with unknown cross patterns and noisy data; few work has studied in a clean setting with known ground-truth models. Hence, it’s important to understand : 1) in which cases would traditional neural nets become inefficient; 2) the role of each component in the cross network of DCN-V2.
We use the cross network in DCN models to represent those feature cross methods and compare with ReLUs, which are commonly used in industrial recommender systems.
To simplify experiments and ease understanding, we assume each feature
x
i
x_{i}
is of dimension one, and monomial
x
1
α
1
x
2
α
2
⋯
x
d
α
d
x_{1}^{\alpha_{1}}x_{2}^{\alpha_{2}}\cdots x_{d}^{\alpha_{d}}
represents a
|
𝜶
|
|{\bm{\alpha}}|
-order interaction between features.
Performance with increasing difficulty.
Consider only 2nd-order feature crosses and let the ground-truth model be
f
⁡
(
𝐱
)
=
∑
|
𝜶
|
=
2
w
𝜶
​
x
1
α
1
​
x
2
α
2
​
…
​
x
d
α
d
f({\bf x})=\sum_{|{\bm{\alpha}}|=2}w_{{\bm{\alpha}}}x_{1}^{\alpha_{1}}x_{2}^{\alpha_{2}}\ldots x_{d}^{\alpha_{d}}
.
Then, the difficulty of learning
f
⁡
(
𝐱
)
f({\bf x})
depends on: 1) sparsity (
w
𝜶
=
0
w_{{\bm{\alpha}}}=0
), the number of crosses, and 2) similarity of the cross patterns (characterized by
Var
⁡
(
w
𝜶
)
\mathrm{Var}(w_{{\bm{\alpha}}})
), meaning a change in one feature would simultaneously affect most feature crosses by similar amount. We create synthetic datasets with increasing difficulty in Eq. (
6
).
(6)
f
1
​
(
𝐱
)
=
x
1
2
+
x
1
​
x
2
+
x
3
​
x
1
+
x
4
​
x
1
f
2
​
(
𝐱
)
=
x
1
2
+
0.1
​
x
1
​
x
2
+
x
2
​
x
3
+
0.1
​
x
3
2
f
3
​
(
𝐱
)
=
∑
(
i
,
j
)
∈
S
w
i
​
j
​
x
i
​
x
j
,
𝐱
∈
ℝ
100
,
|
S
|
=
100
\begin{split}f_{1}({\bf x})&=x_{1}^{2}+x_{1}x_{2}+x_{3}x_{1}+x_{4}x_{1}\\
f_{2}({\bf x})&=x_{1}^{2}+0.1x_{1}x_{2}+x_{2}x_{3}+0.1x_{3}^{2}\\
f_{3}({\bf x})&=\sum\nolimits_{(i,j)\in S}w_{ij}x_{i}x_{j},\penalty\ \penalty\ {\bf x}\in\mathbb{R}^{100},|S|=100\end{split}
where set
S
S
and weights
w
i
​
j
w_{ij}
are randomly assigned, and
x
i
x_{i}
’s are uniformly sampled from interval [-1, 1].
Table 1
reports mean RMSE out of 5 runs and the model size. When the cross patterns are simple (
f
1
f_{1}
), both DCN-V2 and DCN are efficient. When the patterns become more complicated (
f
3
f_{3}
), DCN-V2 remains accurate while DCN degrades. DNN’s performance remains poor even with a wider and deeper structure (layer sizes [200, 200] for
f
1
f_{1}
and
f
2
f_{2}
, [1024, 512, 256] for
f
3
f_{3}
). This suggests the inefficiency of DNN in modeling monomial patterns.
Table 1
.
RMSE and Model Size (# Parameters) for Polynomial Fitting of Increasing Difficulty.
DCN (1Layer)
DCN-V2 (1Layer)
DNN (1Layer)
DNN (large)
RMSE
Size
RMSE
Size
RMSE
Size
RMSE
Size
f
1
f_{1}
8.9E-13
12
5.1E-13
24
2.7E-2
24
4.7E-3
41K
f
2
f_{2}
1.0E-01
9
4.5E-15
15
3.0E-2
15
1.4E-3
41K
f
3
f_{3}
2.6E+00
300
6.7E-07
10K
2.7E-1
10K
7.8E-2
758K
Role of each component.
We also conducted ablation studies on homogeneous polynomials of order 3 and 4, respectively. For each order, we randomly selected 20 cross terms from
𝐱
∈
ℝ
50
{\bf x}\in\mathbb{R}^{50}
.
Figure 4
shows the change in mean RMSE with layer depth. Clearly,
𝐱
0
⊙
(
W
​
𝐱
i
)
{\bf x}_{0}\odot(W{\bf x}_{i})
models order-
d
d
crosses at layer
d
d
-1, which is verified by that the best performance for order-3 polynomial is achieved at layer 2 (similar for order-4). At other layers, however, the performance significantly degrades. This is where the bias and residual terms are helpful — they create and maintain all the crosses up to the highest order. This reduces the performance gap between layers, and stabilizes the model when redundant crosses are introduced. This is particularly important for real-world applications with unknown cross patterns.
Fig.
4
also reveals the limited expressiveness of DCN in modeling complicated cross patterns.
Figure 4
.
Homogeneous polynomial fitting of order 3 and 4.
x
x
-axis represents the number of layers used;
y
y
-axis represents RMSE (the lower the better). In the legend, the top 3 models are DCN-V2 with different component(s) included.
Performance with increasing layer depth.
We now study scenarios closer to real-world settings, where the cross terms are of a combined order.
f
⁡
(
𝐱
)
=
𝐱
⊤
𝐰
+
∑
𝜶
∈
S
w
𝜶
x
1
α
1
x
2
α
2
⋯
x
d
α
d
+
0.1
sin
(
2
𝐱
⊤
𝐰
s
+
0.1
)
+
0.01
ϵ
\small\begin{split}f({\bf x})=&{\bf x}^{\top}{\bf w}+\sum_{{\bm{\alpha}}\in S}w_{{\bm{\alpha}}}x_{1}^{\alpha_{1}}x_{2}^{\alpha_{2}}\cdots x_{d}^{\alpha_{d}}+0.1\sin(2{\bf x}^{\top}{\bf w}_{s}+0.1)+0.01\epsilon\end{split}
where the randomly chosen set
S
=
S
2
∪
S
3
∪
S
4
S=S_{2}\cup S_{3}\cup S_{4}
,
|
S
2
|
=
20
,
|
S
3
|
=
10
,
|
S
4
|
=
5
|S_{2}|=20,|S_{3}|=10,|S_{4}|=5
, and
∀
𝜶
∈
S
i
,
|
𝜶
|
=
i
\forall{\bm{\alpha}}\in S_{i},|{\bm{\alpha}}|=i
; sine introduces perturbations and
ϵ
\epsilon
represents Gaussian noises.
Table 2
reports the mean RMSE out of 5 runs. With the increase of layer depth, CN-M was able to capture higher-order feature crosses in the data, resulting in improved performance. Thanks to the bias and residual terms, the performance didn’t degrade beyond layer 3, where redundant feature interactions were introduced.
Table 2.
Combined-order (1 - 4) Polynomial Fitting.
#Layers
1
2
3
4
5
DCN-V2
1.43E-01
2.89E-02
9.82E-03
9.87E-03
9.92E-03
DNN
1.32E-01
1.03E-01
1.03E-01
1.09E-01
1.05E-01
To summarize, ReLUs are inefficient in capturing explicit feature crosses (multiplicative relations) even with a deeper and larger network. This is well aligned with previous studies
(
Beutel et al. 2018
)
. The accuracy considerably degrades when the cross patterns become more complicated. DCN accurately captures simple cross patterns but fails at more complicated ones. DCN-V2, on the other hand, remains accurate and efficient for complicated cross patterns.
7.
Experimental Results (RQ2 - RQ5)
This section empirically verifies the effectiveness of DCN-V2 in feature interaction learning across 3 datasets and 2 platforms, compared with SOTA. In light of recent concerns about poor reproducibility of published results
(
Dacrema
et al. 2019
;
Musgrave
et al. 2020
;
Rendle
et al. 2020
)
, we conducted a fair and comprehensive experimental study with extensive hyper-parameter search to properly tune all the baselines and proposed approaches. In addition, for each optimal setup, we train 5 models with different random initialization, and report the mean and standard deviation.
Section 7.2
studies the performance of the feature-cross learning components (
RQ2
) between baselines
without
integrating with DNN ReLU layers (similar to
(
Lian
et al. 2018
;
Song et al. 2019
)
); only sparse features are considered for a clean comparison.
Section 7.3
compares DCN-V2 with all the baselines comprehensively (
RQ3
).
Section 7.5
evaluates the influence of hyper-parameters on the performance of DCN-V2 (
RQ4
).
Section 7.6
focuses on model understanding (
RQ5
) of whether we are indeed discovering meaningful feature crosses with DCN-V2.
7.1.
Experiment Setup
This section describes the experiment setup, including training datasets, baseline approaches, and details of the hyper-parameter search and training process.
7.1.1.
Datasets
Table 3
lists the statistics of each dataset:
Table 3.
Datasets.
Data
# Examples
# Features
Vocab Size
Criteo
45M
39
2.3M
MovieLen-1M
740k
7
3.5k
Production
>
>
100B
NA
NA
Criteo
3
3
3
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
.
The most popular click-through-rate (CTR) prediction benchmark dataset contains user logs over a period of 7 days. We follow
(
Wang
et al. 2017
;
Song et al. 2019
)
and use first 6 days for training, and randomly split the last day’s data into validation and test set equally. We log-normalize (
log
⁡
(
x
+
4
)
\log(x+4)
for feature-2 and
log
⁡
(
x
+
1
)
\log(x+1)
for others) the 13 continuous features and embed the remaining 26 categorical features.
MovieLen-1M
4
4
4
https://grouplens.org/datasets/movielens
.
The most popular dataset for recommendation systems research. Each training example includes a
⟨
\langle
user-features, movie-features, rating
⟩
\rangle
triplet. Similar to AutoInt
(
Song et al. 2019
)
, we formalize the task as a regression problem. All the ratings for 1s and 2s are normalized to be 0s; 4s and 5s to be 1s; and rating 3s are removed. 6 non-multivalent categorical features are used and embedded. The data is randomly split into 80% for training, 10% for validation and 10% for testing.
7.1.2.
Baselines.
We compare our proposed approaches with 6 SOTA feature interaction learning algorithms. A brief comparison between the approaches is highlighted in
Table 4
.
Table 4.
High-level comparison between models. Assuming the input
𝐱
0
=
[
𝐯
1
;
…
;
𝐯
k
]
{\bf x}_{0}=[{\bf v}_{1};\ldots;{\bf v}_{k}]
contains
k
k
feature embeddings that each represented as
𝐯
i
{\bf v}_{i}
.
⊕
\oplus
denotes concatenation;
⊗
\otimes
denotes outer-product;
⊙
\odot
denotes Hadamard-product.
f
i
​
(
⋅
)
f_{i}(\cdot)
represents implicit feature interactions,
i.e.,
ReLU layers. In the last column, the ‘+’ sign is on the logit level.
Model
Explicit Interactions (
f
e
f_{e}
)
Final
Objective
Order
(Simplified) Key Formula
PNN
(
Qu
et al. 2016
)
2
𝐱
o
=
[
𝐯
i
⊤
​
𝐯
j
|
∀
i
,
j
]
{\bf x}_{o}=[{\bf v}_{i}^{\top}{\bf v}_{j}\mathrel{|}\forall i,j]
(IPNN)
f
i
∘
f
e
f_{i}\circ f_{e}
𝐱
o
=
[
vec
​
(
𝐯
i
⊗
𝐯
j
)
|
∀
i
,
j
]
{\bf x}_{o}=[\text{vec}({\bf v}_{i}\otimes{\bf v}_{j})\mathrel{|}\forall i,j]
(OPNN)
DeepFM
(
Guo
et al. 2017
)
2
2
𝐱
o
=
[
𝐯
i
⊤
​
𝐯
j
|
∀
i
,
j
]
{\bf x}_{o}=[{\bf v}_{i}^{\top}{\bf v}_{j}\mathrel{|}\forall i,j]
f
i
+
f
e
f_{i}+f_{e}
DLRM
(
Naumov et al. 2019
)
2
𝐱
o
=
[
𝐯
i
⊤
​
𝐯
j
|
∀
i
,
j
]
{\bf x}_{o}=[{\bf v}_{i}^{\top}{\bf v}_{j}\mathrel{|}\forall i,j]
f
i
∘
f
e
f_{i}\circ f_{e}
DCN
(
Wang
et al. 2017
)
≥
2
\geq 2
𝐱
i
+
1
=
𝐱
0
⊗
𝐱
i
​
𝐰
i
{\bf x}_{i+1}={\bf x}_{0}\otimes{\bf x}_{i}{\bf w}_{i}
f
i
+
f
e
f_{i}+f_{e}
xDeepFM
(
Lian
et al. 2018
)
≥
2
\geq 2
𝐯
h
k
=
∑
i
,
j
w
i
​
j
k
​
h
​
(
𝐯
i
k
−
1
⊙
𝐯
j
)
{\bf v}_{h}^{k}=\sum_{i,j}w_{ij}^{kh}({\bf v}_{i}^{k-1}\odot{\bf v}_{j})
f
i
+
f
e
f_{i}+f_{e}
AutoInt
(
Song et al. 2019
)
NA
OPEN
𝐯
~
i
=
g
⁡
(
∑
j
exp
⁡
(
⟨
W
q
​
𝐯
i
,
W
k
​
𝐯
j
⟩
)
​
W
v
​
𝐯
j
∑
j
exp
⁡
(
⟨
W
q
​
𝐯
i
,
W
k
​
𝐯
j
⟩
)
)
)
\widetilde{\bf v}_{i}=g\left(\frac{\sum_{j}\exp(\langle W_{q}{\bf v}_{i},W_{k}{\bf v}_{j}\rangle)W_{v}{\bf v}_{j}}{\sum_{j}\exp\left(\langle W_{q}{\bf v}_{i},W_{k}{\bf v}_{j}\rangle\right)})\right)
f
i
+
f
e
f_{i}+f_{e}
DCN-V2 (ours)
≥
2
\geq 2
𝐱
i
=
𝐱
0
⊙
(
W
i
​
𝐱
i
)
{\bf x}_{i}={\bf x}_{0}\odot(W_{i}{\bf x}_{i})
f
i
∘
f
e
f_{i}\circ f_{e}
f
i
+
f
e
f_{i}+f_{e}
7.1.3.
Implementation Details.
All the baselines and our approaches are implemented in TensorFlow v1. For a fair comparison, all the implementations were identical across all the models except for the feature interaction component
5
5
5
We adopted implementation from
https://github.com/Leavingseason/xDeepFM
,
https://github.com/facebookresearch/dlrm
and
https://github.com/shenweichen/DeepCTR
.
Embeddings.
All the baselines require each feature’s embedding size to be the same except for DNN and DCN. Hence, we fixed it to be
Avg
​
(
∑
vocab
6
⋅
(
vocab cardinality
)
1
4
)
\text{Avg}\big(\sum_{\text{vocab}}6\cdot(\text{vocab cardinality})^{\frac{1}{4}}\big)
(39 for Criteo and 30 for Movielen-1M) for all the models
6
6
6
This formula is a rule-of-thumb number that is widely used
(
Wang
et al. 2017
)
, also see
https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
.
Optimization.
We used Adam
(
Kingma and Ba 2014
)
with a batch size of
512
512
(128 for MovieLen). The kernels were initialized with He Normal
(
He
et al. 2015
)
, and biases to
𝟎
{\bf 0}
; the gradient clipping norm was 10; an exponential moving average with decay 0.9999 to trained parameters was applied.
Reproducibility and fair comparisons: hyper-parameters tuning and results reporting.
For all the baselines, we conducted a coarse-level (larger-range) grid search over the hyper-parameters, followed by a finer-level (smaller-range) search. To ensure reproducibility and mitigate model variance, for each approach and dataset, we report the mean and stddev out of 5 independent runs for the best configuration. We describe detailed settings below for Criteo; and follow a similar process for MovieLens with different ranges.
We first describe the hyper-parameters shared across the baselines. The learning rate was tuned from
10
−
4
10^{-4}
to
10
−
1
10^{-1}
on a log scale and then narrowed down to
10
−
4
10^{-4}
to
5
×
10
−
4
5\times 10^{-4}
on a linear scale. The training steps were searched over {150k, 160k, 200k, 250k, 300k}. The number of hidden layers ranged in {1, 2, 3, 4} with their layer sizes in {562, 768, 1024}. And the regularization parameter
λ
\lambda
was in {0,
3
×
10
−
5
3\times 10^{-5}
,
10
−
4
10^{-4}
}.
We then describe each model’s own hyper-parameters, where the search space is designed based on reported setting. For DCN, the number of cross layers ranged from 1 to 4. For AutoInt, the number of attention layers was from 2 to 4; the attention embedding size was in {20, 32, 40}; the number of attention head was from 2 to 3; and the residual connection was either on or off. For xDeepFM, the CIN layer size was in {100, 200}, depth in {2, 3, 4}, activation was identity, computation was either direct or indirect. For DLRM, the bottom MLP layer sizes and numbers was in {(512,256,64), (256,64)}. For PNN, we ran for IPNN, OPNN and PNN*, and for the latter two, the kernel type ranged in {full matrix, vector, number}.
For all the models, the total number of parameters was capped at
1024
2
×
5
1024^{2}\times 5
to limit the search space and avoid overly expensive computations.
7.2.
Performance of Feature Interaction Component Alone (RQ2)
We consider the feature interaction component alone of each model
without their DNN component
. Moreover, we only consider the categorical features, as the dense features were processed differently among baselines.
Table 5
shows the results on Criteo dataset. Each baseline was tuned similarly as in
Section 7.1.3
. There are two major observations. 1). Higher-order methods demonstrate a superior performance over 2nd-order methods. This suggests high-order crosses are meaningful in this dataset. 2). Among the high-order methods, cross network achieved the best performance and was on-par or slightly better compared to DNN.
Table 5.
LogLoss (test) of feature interaction component of each model (no DNN). Only categorical features were used. In the ‘Setting’ column,
l
l
stands for number of layers.
Model
LogLoss
Best Setting
2nd
PNN
(
Qu
et al. 2016
)
0.4715
±
\pm
4.430e-04
OPNN, kernel=matrix
FM
0.4736
±
\pm
3.04E-04
–
>
>
2
CIN
(
Lian
et al. 2018
)
0.4719
±
\pm
9.41E-04
l=3, cinLayerSize=100
AutoInt
(
Song et al. 2019
)
0.4711
±
\pm
1.62E-04
l=2, head=3, attEmbed=40
DNN
0.4704
±
\pm
1.57E-04
l=2, size=1024
CrossNet
0.4702
±
\pm
3.80E-04
l=2
CrossNet-Mix
0.4694
±
\pm
4.35E-04
l=5, expert=4, gate=
1
1
+
e
−
x
\frac{1}{1+e^{-x}}
7.3.
Performance of Baselines (RQ3)
This section compares the performance between DCN-V2 approaches and the baselines in an end-to-end fashion. Note that the best setting reported for each model was searched over a
wide-ranged model capacity and hyper-parameter space
including the baselines. And if two settings performed on-par, we report the
lower-cost
one.
Table 6
shows the best LogLoss and AUC (Area Under the ROC Curve) on testset for Criteo and MovieLen. For Criteo, a
0.001-level improvement
is considered significant (see
(
Song et al. 2019
;
Wang
et al. 2017
;
Guo
et al. 2017
)
). We see that DCN-V2 consistently outperformed the baselines (including DNN) and achieved a healthy quality/cost trade-off. It’s also worth mentioning that the baselines’ performances reported in
Table 6
were improved over the numbers reported by previous papers (see
Table 9
in Appendix); however, when integrated with DNN, their performance gaps are closing up (compared to
Table 5
) with their performances on-par and sometimes worse than the ReLU-based DNN with fine-granular model tuning.
Best Settings.
The optimal hyper-parameters are in
Table 6
. For DCN-V2 models, both the ‘stacked’ and ‘parallel’ structures outperformed all the baselines, while ‘stacked’ worked better on Criteo and ‘parallel’ worked better on Movielen-1M. On Criteo, the setting was gate as constant, hard_tanh activation for DCN-Mix; gate as softmax and identity activation for CrossNet. The best training steps was 150k for all the baselines; learning rate varies for all the models.
Model Quality — Comparisons among baselines.
When integrating the feature cross learning component with a DNN, the advantage of higher-order methods is less pronounced, and the performance gap among all the models are closing up on Criteo (compared to
Table 5
).
This suggests the importance of implicit feature interactions and the power of a well-tuned DNN model.
For 2nd-order methods, DLRM performed inferiorly to DeepFM although they are both derived from FM. This might be due to DLRM’s omission of the 1st-order sparse features after the dot-product layer. PNN models 2nd-order crosses more expressively and delivered better performance on MovieLen-1M; however on Criteo, its mean LogLoss was driven up by its high standard deviation. For higher-order methods, xDeepFM, AutoInt and DCN behaved similarly on Criteo, while on MovieLens xDeepFm showed a high variance.
DCN-V2 achieved the best performance (0.001 considered to be significant on Criteo
(
Wang
et al. 2017
;
Lian
et al. 2018
;
Song et al. 2019
)
) by explicitly modeling up to 3rd-order crosses beyond those implicit ones from DNN. DCN-Mix, the mixture of low-rank DCN, efficiently utilized the memory and reduced the cost by 30% while maintaining the accuracy. Interestingly, CrossNet alone outperformed DNN on both datasets; we defer more discussions to
Section 7.4
.
Model Quality — Comparisons with DNN.
DNNs are universal approximators and are tough-to-beat baselines when highly-optimized. Hence, we finely tuned DNN along with all the baselines, and used a larger layer size than those used in literature (
e.g.
, 200 - 400 in
(
Lian
et al. 2018
;
Song et al. 2019
)
).
To our surprise, DNN performed neck to neck with most baselines and even outperformed certain models.
Our hypothesis is that those explicit feature crosses from baselines were not modeled in an
expressive
and
easy-to-optimize
manner. The former makes its performanc easy to be matched by a DNN with large capacity. The latter would easily lead to trainability issues, making the model unstable, hard to identify a good local optima or to generalize. Hence, when integrated with DNN, the overall performance is dominated by the DNN component. This becomes especially true with a large-capacity DNN, which could already approximate some simple cross patterns.
In terms of expressiveness, consider the 2nd-order methods. PNN models crosses more expressively than DeepFM and DLRM, which resulted in its superior performance on MovieLen-1M. This also explains the inferior performance of DCN compared to DCN-V2.
In terms of trainability, certain models might be inherently more difficult to train and resulted in unsatisfying performance. Consider PNN. On MoiveLen-1M, it outperformed DNN, suggesting the effectiveness of those 2nd-order crosses. On Criteo, however, PNN’s advantage has diminished and the averaged performance was on-par with DNN. This was caused by the instability of PNN. Although its best run was better than DNN, its high stddev from multiple trials has driven up the mean loss. xDeepFM also suffers from trainability issue (see its high stddev on MovieLens). In xDeepFM, each feature map encodes all the pair-wise crosses while only relies on a single variable to learn the importance of each cross. In practice, a single variable is difficult to be learned when jointly trained with magnitudes more parameters. Then, an improperly learned variable would lead to noises.
DCN-V2, on the other hand, consistently outperforms DNN. It successfully leveraged both the explicit and implicit feature interactions. We attribute this to the balanced number of parameters between the cross network and the deep network (
expressive
), as well as the simple structure of cross net which eased the optimization (
easy-to-optimize
). It’s worth noting that the high-level structure of DCN-V2 shares a similar spirit of the self-attention mechanism adopted in AutoInt, where each feature embedding attends to a weighed combination of other features. The difference is that during the attention, higher-order interactions were modeled explicitly in DCN-V2 but implicitly in AutoInt.
Model Efficiency.
Table 6
also provides details for model size and FLOPS
7
7
7
FLOPS is a close estimation of run time, which is subjective to implementation details.
. The reported setting was properly tuned over the hyper-parameters of each model and the DNN component. For most models, the FLOPS is roughly 2x of the #params; for xDeepFM, however, the FLOPS is one magnitude higher, making it impractical in industrial-scale applications (also observed in
(
Song et al. 2019
)
). Note that for DeepFM and DLRM, we’ve also searched over larger-capacity models; however, they didn’t deliver better quality. Among all the methods, DCN-V2 delivers the best performance while remaining relatively efficient; DCN-Mix further reduced the cost, achieving a better trade-off between model efficiency and quality.
Table 6.
LogLoss and AUC (test) on Criteo and Movielen-1M. The metrics were averaged over 5 independent runs with their stddev in the parenthesis. In the ‘Best Setting’ column, the left reports DNN setting and the right reports model-specific setting.
l
l
denotes layer depth;
n
n
denotes CIN layer size;
h
h
and
e
e
, respectively, denotes #heads and att-embed-size;
K
K
denotes #experts and
r
r
denotes total rank.
Baseline
Criteo
MovieLens-1M
Logloss
AUC
Params
FLOPS
Best Setting
Logloss
AUC
Params
FLOPS
PNN
0.4421 (5.8E-4)
0.8099 (6.1E-4)
3.1M
6.1M
(3, 1024)
OPNN
0.3182 (1.4E-3)
0.8955 (3.3E-4)
54K
110K
DeepFm
0.4420 (1.4E-4)
0.8099 (1.5E-4)
1.4M
2.8M
(2, 768)
–
0.3202 (1.0E-3)
0.8932 (7.7E-4)
46K
93K
DLRM
0.4427 (3.1E-4)
0.8092 (3.1E-4)
1.1M
2.2M
(2, 768)
[512,256,64]
0.3245 (1.1E-3)
0.8890 (1.1E-3)
7.7K
16K
xDeepFm
0.4421 (1.6E-4)
0.8099 (1.8E-4)
3.7M
32M
(3, 1024)
l
l
=2,
n
n
=100
0.3251 (4.3E-3)
0.8923 (8.6E-4)
160K
990K
AutoInt+
0.4420 (5.7E-5)
0.8101 (2.6E-5)
4.2M
8.7M
(4, 1024)
l
l
=2,
h
h
=2,
e
e
=40
0.3204 (4.4E-4)
0.8928 (3.9E-4)
260K
500K
DCN
0.4420 (1.6E-4)
0.8099 (1.7E-4)
2.1M
4.2M
(2, 1024)
l
l
=4
0.3197 (1.9E-4)
0.8935 (2.1E-4)
110K
220K
DNN
0.4421 (6.5E-5)
0.8098 (5.9E-5)
3.2M
6.3M
(3, 1024)
–
0.3201 (4.1E-4)
0.8929 (2.3E-4)
46K
92K
Ours
DCN-V2
0.4406 (6.2E-5)
0.8115 (7.1E-5)
3.5M
7.0M
(2, 768)
l
l
=2
0.3170 (3.6E-4)
0.8950 (2.7E-4)
110K
220K
DCN-Mix
0.4408 (1.0E-4)
0.8112 (9.8E-5)
2.4M
4.8M
(2, 512)
l
l
=3,
K
K
=4,
r
r
=258
0.3160 (4.9E-4)
0.8964 (2.9E-4)
110K
210K
CrossNet
0.4413 (2.5E-4)
0.8107 (2.4E-4)
2.1M
4.2M
–
l
l
=4,
K
K
=4,
r
r
=258
0.3185 (3.0E-4)
0.8937 (2.7E-4)
65K
130K
7.4.
Can Cross Layers Replace ReLU layers?
The solid performance of DCN-V2 approaches has inspired us to further study the efficiency of their cross layers (CrossNet) in learning explicit high-order feature crosses.
In a realistic setting with resource constraints, we often have to limit model capacity. Hence, we fixed the model capacity (memory / # of parameters) at different levels, and compared the performance between a model with only cross layers (Cross Net), and a ReLU based DNN.
Table 7
reports the best test LogLoss for different memory constraints. The memory was controlled by varying the number of cross layers and its rank ({128, 256}), the number of hidden layers and their sizes. The best performance was achieved by the cross network (5-layer), suggesting the ground-truth model could be well-approximated by polynomials. Moreover, the best performance per memory limit was also achieved by the cross network, indicating both solid effectiveness and efficiency.
It is well known that ReLU layers are the backbone for various Neural Nets models including DNN, Recurrent Neural Net (RNN)
(
Rumelhart
et al. 1985
;
Hochreiter and
Schmidhuber 1997
;
Mikolov et al. 2011
)
and Convolutional Neural Net (CNN)
(
LeCun et al. 1989
;
Schmidhuber 2015
;
Lawrence
et al. 1997
)
. It is quite surprising and encouraging to us that we may potentially replace ReLU layers by Cross Layers entirely for certain applications. Obviously we need significant more analysis and experiments to verify the hypothesis. Nonetheless, this is a very interesting preliminary study and sheds light for our future explorations on cross layers.
Table 7.
Logloss and AUC (test) with a fixed memory budget.
#Params
7.9E+05
1.3E+06
2.1E+06
2.6E+06
LogLoss
CrossNet
0.4424
0.4417
0.4416
0.4415
DNN
0.4427
0.4426
0.4423
0.4423
AUC
CrossNet
0.8096
0.8104
0.8105
0.8106
DNN
0.8091
0.8094
0.8096
0.80961
7.5.
How the Choice of Hyper-parameters Affect DCN-V2 Model Performance (RQ4)
This section examines the model performance as a function of hyper-parameters that include 1) depth of cross layers; 2) matrix rank of DCN-Mix; 3) number of experts in DCN-Mix.
Depth of Cross Layers.
By design, the highest feature cross order captured by the cross net increases with layer depth. Hence, we constrain ourselves to the full-rank cross layers, and evaluate the performance change with layer depth
5(a)
shows the test LogLoss and AUC while increasing layer depth on the Criteo dataset. We see a steady quality improvement with a deeper cross network, indicating that it’s able to capture more meaningful crosses. The rate of improvement, however, slowed down when more layers were used. This suggests the contribution from that of higher-order crosses is less significant than those from lower-order crosses. We also used a same-sized DNN as a reference. When there were
≤
2
\leq 2
layers, DNN outperformed the cross network; when more layers became available, the cross network started to close the performance gap and even outperformed DNN. In the small-layer regime, the cross network could only approximate very low-order crosses (
e.g.,
1
∼
\sim
2); in the large-layer regime, those low-order crosses were characterized with more parameters, and those high-order interactions were started to be captured.
(a)
Layer depth
(b)
Matrix rank
Figure 5.
Logloss and AUC (test) v.s. depth & matrix rank.
Rank of Matrix.
The rank of the weight matrix controls the number of parameters as well as the portion of low-frequency signals passing through the cross layers. Hence, we study its influence on model quality. The model is based on a well-performed setting with 3 cross layers followed by 3 hidden layers of size 512. We approximate the dense matrix
W
W
in each cross layer by
U
​
V
⊤
UV^{\top}
where
U
,
V
∈
ℝ
d
×
r
U,V\in\mathbb{R}^{d\times r}
, and we vary
r
r
. We loosely consider the smaller dimension
r
r
to be the rank.
5(b)
shows the test LogLoss and AUC v.s. matrix’s rank
r
r
on Criteo. When
r
r
was as small as 4, the performance was on-par with other baselines. When
r
r
was increased from 4 to 64, the LogLoss decreased almost linearly with
r
r
(
i.e.
, model’s improving). When
r
r
was further increased from 64 to full, the improvement on LogLoss slowed down. We refer to 64 as the
threshold rank
. The significant slow down from 64 suggests that the important signals characterizing feature crosses could be captured in the top-64 singular values.
Our hypothesis for the value of this
threshold rank
is
O
⁡
(
k
)
O(k)
where
k
k
represents # features (39 for Criteo). Consider the
(
i
,
j
)
(i,j)
-th block of matrix
W
W
, we can view
W
i
,
j
=
W
i
,
j
L
+
W
i
,
j
H
\small W_{i,j}=W_{i,j}^{L}+W_{i,j}^{H}
, where
W
i
,
j
L
W_{i,j}^{L}
stores the dominant signal (low-frequency) and
W
i
,
j
H
\small W_{i,j}^{H}
stores the rest (high-frequency). In the simplest case where
W
i
,
j
L
=
c
i
​
j
​
𝟏𝟏
⊤
\small W_{i,j}^{L}=c_{ij}{\bf 1}{\bf 1}^{\top}
, the entire matrix
W
L
\small W^{L}
will be of rank
k
k
. The effectiveness of this hypothesis remains to be verified across multiple datasets.
Number of Experts.
We study how the number of low-rank experts affects the quality. We’ve observed that 1) best-performed setting (#expert, gate, matrix activation type) was subjective to datasets and model architectures; 2) the best-performed model of each setting yielded similar results. For example, for a 2-layered cross net with total rank 256 on Criteo, the LogLoss for 1, 4, 8, 16, and 32 experts, respectively, was 0.4418, 0.4416, 0.4416, 0.4422, and 0.4420. The fact that more lower-ranked experts wasn’t performing better than a single higher-ranked expert might be caused by the naïve gating functions and optimizations adopted. We believe more sophisticated gating
(
Jang
et al. 2016
;
Louizos
et al. 2017
;
Ma
et al. 2019
)
and optimization techniques (
e.g.
, alternative training, special initialization, temperature adjustment) would leverage more from a mixture of experts. This, however, is beyond the scope of this paper and we leave it to future work.
7.6.
Model Understanding (RQ5)
One key research question is whether the proposed approaches are indeed learning meaningful feature crosses. A good understanding about the learned feature crosses helps improve model understandability, and is especially crucial to fields like ML fairness and ML for health. Fortunately, the weight matrix
W
W
in DCN-V2 exactly reveals what feature crosses the model has learned to be important. Specifically, we assume that each input
𝐱
=
[
𝐱
1
;
𝐱
2
;
…
;
𝐱
k
]
{\bf x}=[{\bf x}_{1};{\bf x}_{2};\ldots;{\bf x}_{k}]
contains
k
k
features with each represented by an embedding
𝐱
i
{\bf x}_{i}
. Then, the block-wise view of the feature crossing component (ignoring the bias) in Eq. (
7
) shows that the importance of feature interaction between
i
i
-th and
j
j
-th feature is characterized by the
(
i
,
j
)
(i,j)
-th block
W
i
,
j
W_{i,j}
.
(7)
𝐱
⊙
W
​
𝐱
=
[
𝐱
1
𝐱
2
𝐱
k
]
⊙
[
W
1
,
1
W
1
,
2
⋯
W
1
,
k
W
2
,
1
W
2
,
2
⋯
W
2
,
k
⋱
W
k
,
1
W
k
,
2
⋯
W
k
,
k
]
​
[
𝐱
1
𝐱
2
𝐱
k
]
{\bf x}\odot W{\bf x}=\left[\begin{smallmatrix}{\bf x}_{1}\\
{\bf x}_{2}\vskip-3.01389pt\\
\vdots\\
{\bf x}_{k}\end{smallmatrix}\right]\odot\left[\begin{smallmatrix}W_{1,1}&W_{1,2}&\cdots&W_{1,k}\\
W_{2,1}&W_{2,2}&\cdots&W_{2,k}\vskip-3.01389pt\\
\vdots&\vdots&\ddots&\vdots\\
W_{k,1}&W_{k,2}&\cdots&W_{k,k}\end{smallmatrix}\right]\left[\begin{smallmatrix}{\bf x}_{1}\\
{\bf x}_{2}\vskip-3.01389pt\\
\vdots\\
{\bf x}_{k}\end{smallmatrix}\right]
Figure 6
shows the learned weight matrix
W
W
in the first cross layer. Subplot (a) shows the entire matrix with orange boxes highlighting some notable feature crosses. The off-diagonal block corresponds to crosses that are known to be important, suggesting the effectiveness of DCN-V2. The diagonal block represents self-interaction (
x
2
x^{2}
’s).
Subplot (b) shows each block’s Frobenius norm and indicates some strong interactions learned,
e.g.
,
Gender
×
\times
UserId
,
MovieId
×
\times
UserId
.
(a)
Production data
(b)
Movielen-1M
Figure 6
.
Visualization of learned weight matrix in DCN-V2. Rows and columns represents real features. For (a), feature names were not shown for proprietary reasons; darker pixel represents larger weight in its absolute value. For (b), each block represents the Frobenius norm of each matrix block.
8.
Productionizing DCN-V2 at Google
This section provides a case study to share our experience productionizing DCN-V2 in a large-scale recommender system in Google. We’ve achieved significant gains through DCN-V2 in both offline model accuracy, and online key business metrics.
The Ranking Problem:
Given a user and a large set of candidates, our problem is to return the top-
k
k
items the user is most likely to engage with. Let’s denote the training data to be
{
(
𝐱
i
,
y
i
)
}
i
=
1
N
\{({\bf x}_{i},y_{i})\}_{i=1}^{N}
, where
𝐱
i
{\bf x}_{i}
’s represents features of multiple modalities, such as user’s interests, an item’s metadata and contextual features;
y
i
y_{i}
’s are labels representing a user’s action (
e.g.
, a click). The goal is to learn a function
f
:
ℝ
d
↦
ℝ
f:\mathbb{R}^{d}\mapsto\mathbb{R}
that predicts the probability
P
⁡
(
y
|
𝐱
)
P(y|{\bf x})
, the user’s action
y
y
given features
𝐱
{\bf x}
.
Production Data and Model:
The production data are sampled user logs consisting of hundreds of billions of training examples. The vocabulary sizes of sparse features vary from 2 to millions. The baseline model is a fully-connected multi-layer perceptron (MLP) with ReLU activations.
Comparisons with Production Models:
When compared with production model, DCN-V2 yielded 0.6% AUCLoss (1 - AUC) improvement. For this particular model, a gain of 0.1% on AUCLoss is considered a significant improvement. We also observed significant online performance gains on key metrics.
Table 8
further verifies the amount of gain from DCN-V2 by replacing cross layers with same-sized ReLU layers.
Table 8.
Relative AUCLoss of DCN-V2 v.s. same-sized ReLUs
1layer ReLU
2layer ReLU
1layer DCN-V2
2layer DCN-V2
0%
-0.15%
-0.19%
-0.45%
Practical Learnings.
We share some practical lessons we have learned through productionizing DCN-V2.
•
It’s better to insert the cross layers in between the input and the hidden layers of DNN (also observed in
(
Shan
et al. 2016
)
). Our hypothesis is that the physical meaning of feature representations and their interactions becomes weaker as it goes farther away from the input layer.
•
We saw consistent accuracy gains by stacking or concatenating 1 - 2 cross layers. Beyond 2 cross layers, the gains start to plateau.
•
We observed that both stacking cross layers and concatenating cross layers work well. Stacking layers learns higher-order feature interactions, while concatenating layers (similar to multi-head mechanism
(
Vaswani et al. 2017
)
) captures complimentary interactions.
•
We observed that using low-rank DCN with rank
(input size)
/
4
\text{(input size)}/4
consistently preserved the accuracy of a full-rank DCN-V2.
9.
Conclusions and Future Work
In this paper, we propose a new model—DCN-V2—to model explicit crosses in an expressive yet simple manner. Observing the low-rank nature of the weight matrix in the cross network, we also propose a mixture of low-rank DCN (DCN-Mix) to achieve a healthier trade-off between model performance and latency. DCN-V2 has been successfully deployed in multiple web-scale learning to rank systems with significant offline model accuracy and online business metric gains. Our experimental results also have demonstrated DCN-V2’s effectiveness over SOTA methods.
For future work, we are interested in advancing our understanding of 1). the interactions between DCN-V2 and optimization algorithms such as second-order methods; 2). the relation between embedding, DCN-V2 and its rank of matrix. Further, we would like to improve the gating mechanism in DCN-Mix. Moreover, observing that cross layers in DCN-V2 may serve as potential alternatives to ReLU layers in DNNs, we are very interested to verify this observation across more complex model architectures (
e.g.
, RNN, CNN).
Acknowledgement.
We would like to thank Bin Fu, Gang (Thomas) Fu, and Mingliang Wang for their early contributions of DCN-V2; Tianshuo Deng, Wenjing Ma, Yayang Tian, Shuying Zhang, Jie (Jerry) Zhang, Evan Ettinger, Samuel Ieong and many others for their efforts and supports in productionizing DCN-V2; Ting Chen for his initial idea of mixture of low-rank; and Jiaxi Tang for his valuable comments.
References
(1)
Beutel et al
.
(2018)
Alex Beutel, Paul
Covington, Sagar Jain, Can Xu,
Jia Li, Vince Gatto, and
Ed H Chi. 2018.
Latent cross: Making use of context in recurrent
recommender systems. In
Proceedings of the Eleventh
ACM International Conference on Web Search and Data Mining
.
46–54.
Bottou
et al
.
(2013)
Léon Bottou, Jonas
Peters, Joaquin Quiñonero-Candela,
Denis X Charles, D Max Chickering,
Elon Portugaly, Dipankar Ray,
Patrice Simard, and Ed Snelson.
2013.
Counterfactual reasoning and learning systems: The
example of computational advertising.
The Journal of Machine Learning Research
14, 1 (2013),
3207–3260.
Broder (2008)
Andrei Z Broder.
2008.
Computational advertising and recommender systems.
In
Proceedings of the 2008 ACM conference on
Recommender systems
. 1–2.
Cao
et al
.
(2007)
Zhe Cao, Tao Qin,
Tie-Yan Liu, Ming-Feng Tsai, and
Hang Li. 2007.
Learning to rank: from pairwise approach to
listwise approach. In
Proceedings of the 24th
international conference on Machine learning
. 129–136.
Chen
et al
.
(2018)
Ting Chen, Ji Lin,
Tian Lin, Song Han,
Chong Wang, and Denny Zhou.
2018.
Adaptive mixture of low-rank factorizations for
compact neural modeling.
(2018).
Cheng et al
.
(2016)
Heng-Tze Cheng, Levent
Koc, Jeremiah Harmsen, Tal Shaked,
Tushar Chandra, Hrishi Aradhye,
Glen Anderson, Greg Corrado,
Wei Chai, Mustafa Ispir, et al
.
2016.
Wide & Deep Learning for Recommender Systems.
arXiv preprint arXiv:1606.07792
(2016).
Cheng
et al
.
(2019)
Weiyu Cheng, Yanyan Shen,
and Linpeng Huang. 2019.
Adaptive Factorization Network: Learning
Adaptive-Order Feature Interactions.
arXiv preprint arXiv:1909.03276
(2019).
Dacrema
et al
.
(2019)
Maurizio Ferrari Dacrema,
Paolo Cremonesi, and Dietmar Jannach.
2019.
Are we really making much progress? A worrying
analysis of recent neural recommendation approaches. In
Proceedings of the 13th ACM Conference on
Recommender Systems
. 101–109.
Drineas and
Mahoney (2005)
Petros Drineas and
Michael W Mahoney. 2005.
On the Nyström method for approximating a Gram
matrix for improved kernel-based learning.
journal of machine learning research
6, Dec (2005),
2153–2175.
Eigen
et al
.
(2013)
David Eigen, Marc’Aurelio
Ranzato, and Ilya Sutskever.
2013.
Learning factored representations in a deep mixture
of experts.
arXiv preprint arXiv:1312.4314
(2013).
Fan et al
.
(2019)
Yuwei Fan, Jordi
Feliu-Faba, Lin Lin, Lexing Ying, and
Leonardo Zepeda-Núnez.
2019.
A multiscale neural network based on hierarchical
nested bases.
Research in the Mathematical Sciences
6, 2 (2019),
21.
Golub and
Van Loan (1996)
Gene H Golub and
Charles F Van Loan. 1996.
Matrix Computations Johns Hopkins University
Press.
Baltimore and London
(1996).
Guo
et al
.
(2017)
Huifeng Guo, Ruiming
Tang, Yunming Ye, Zhenguo Li, and
Xiuqiang He. 2017.
DeepFM: a factorization-machine based neural
network for CTR prediction.
arXiv preprint arXiv:1703.04247
(2017).
Halko
et al
.
(2011)
Nathan Halko, Per-Gunnar
Martinsson, and Joel A Tropp.
2011.
Finding structure with randomness: Probabilistic
algorithms for constructing approximate matrix decompositions.
SIAM review
53,
2 (2011), 217–288.
He
et al
.
(2015)
Kaiming He, Xiangyu
Zhang, Shaoqing Ren, and Jian Sun.
2015.
Delving deep into rectifiers: Surpassing
human-level performance on imagenet classification. In
Proceedings of the IEEE international conference on
computer vision
. 1026–1034.
He and Chua (2017)
Xiangnan He and Tat-Seng
Chua. 2017.
Neural factorization machines for sparse predictive
analytics. In
Proceedings of the 40th International
ACM SIGIR conference on Research and Development in Information Retrieval
.
355–364.
Herlocker et al
.
(2004)
Jonathan L Herlocker,
Joseph A Konstan, Loren G Terveen, and
John T Riedl. 2004.
Evaluating collaborative filtering recommender
systems.
ACM Transactions on Information Systems
(TOIS)
22, 1 (2004),
5–53.
Hochreiter and
Schmidhuber (1997)
Sepp Hochreiter and
Jürgen Schmidhuber. 1997.
Long short-term memory.
Neural computation
9,
8 (1997), 1735–1780.
Jacobs
et al
.
(1991)
Robert A Jacobs, Michael I
Jordan, Steven J Nowlan, and Geoffrey E
Hinton. 1991.
Adaptive mixtures of local experts.
Neural computation
3,
1 (1991), 79–87.
Jaderberg
et al
.
(2014)
Max Jaderberg, Andrea
Vedaldi, and Andrew Zisserman.
2014.
Speeding up convolutional neural networks with low
rank expansions.
arXiv preprint arXiv:1405.3866
(2014).
Jang
et al
.
(2016)
Eric Jang, Shixiang Gu,
and Ben Poole. 2016.
Categorical reparameterization with
gumbel-softmax.
arXiv preprint arXiv:1611.01144
(2016).
Kingma and Ba (2014)
Diederik Kingma and
Jimmy Ba. 2014.
Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980
(2014).
Lawrence
et al
.
(1997)
Steve Lawrence, C Lee
Giles, Ah Chung Tsoi, and Andrew D
Back. 1997.
Face recognition: A convolutional neural-network
approach.
IEEE transactions on neural networks
8, 1 (1997),
98–113.
LeCun et al
.
(1989)
Yann LeCun, Bernhard
Boser, John S Denker, Donnie Henderson,
Richard E Howard, Wayne Hubbard, and
Lawrence D Jackel. 1989.
Backpropagation applied to handwritten zip code
recognition.
Neural computation
1,
4 (1989), 541–551.
Li
et al
.
(2020)
Zeyu Li, Wei Cheng,
Yang Chen, Haifeng Chen, and
Wei Wang. 2020.
Interpretable Click-Through Rate Prediction through
Hierarchical Attention. In
Proceedings of the 13th
International Conference on Web Search and Data Mining
.
313–321.
Lian
et al
.
(2018)
Jianxun Lian, Xiaohuan
Zhou, Fuzheng Zhang, Zhongxia Chen,
Xing Xie, and Guangzhong Sun.
2018.
xdeepfm: Combining explicit and implicit feature
interactions for recommender systems. In
Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
Discovery & Data Mining
. 1754–1763.
Liu (2011)
Tie-Yan Liu.
2011.
Learning to rank for information
retrieval
.
Springer Science & Business Media.
Louizos
et al
.
(2017)
Christos Louizos, Max
Welling, and Diederik P Kingma.
2017.
Learning Sparse Neural Networks through
L
​
_
​
0
L\_0
Regularization.
arXiv preprint arXiv:1712.01312
(2017).
Ma
et al
.
(2019)
Jiaqi Ma, Zhe Zhao,
Jilin Chen, Ang Li,
Lichan Hong, and Ed H Chi.
2019.
Snr: Sub-network routing for flexible parameter
sharing in multi-task learning. In
Proceedings of
the AAAI Conference on Artificial Intelligence
, Vol. 33.
216–223.
Ma
et al
.
(2018)
Jiaqi Ma, Zhe Zhao,
Xinyang Yi, Jilin Chen,
Lichan Hong, and Ed H Chi.
2018.
Modeling task relationships in multi-task learning
with multi-gate mixture-of-experts. In
Proceedings
of the 24th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining
. 1930–1939.
Mhaskar (1996)
Hrushikesh N Mhaskar.
1996.
Neural networks for optimal approximation of smooth
and analytic functions.
Neural computation
8,
1 (1996), 164–177.
Mikolov et al
.
(2011)
Tomáš Mikolov,
Stefan Kombrink, Lukáš Burget,
Jan Černockỳ, and Sanjeev
Khudanpur. 2011.
Extensions of recurrent neural network language
model. In
2011 IEEE international conference on
acoustics, speech and signal processing (ICASSP)
. IEEE,
5528–5531.
Musgrave
et al
.
(2020)
Kevin Musgrave, Serge
Belongie, and Ser-Nam Lim.
2020.
A metric learning reality check.
arXiv preprint arXiv:2003.08505
(2020).
Naumov et al
.
(2019)
Maxim Naumov, Dheevatsa
Mudigere, Hao-Jun Michael Shi, Jianyu
Huang, Narayanan Sundaraman, Jongsoo
Park, Xiaodong Wang, Udit Gupta,
Carole-Jean Wu, Alisson G Azzolini,
et al
.
2019.
Deep learning recommendation model for
personalization and recommendation systems.
arXiv preprint arXiv:1906.00091
(2019).
Qu
et al
.
(2016)
Yanru Qu, Han Cai,
Kan Ren, Weinan Zhang,
Yong Yu, Ying Wen, and
Jun Wang. 2016.
Product-based neural networks for user response
prediction. In
2016 IEEE 16th International
Conference on Data Mining (ICDM)
. IEEE, 1149–1154.
Rendle (2010)
Steffen Rendle.
2010.
Factorization machines. In
2010 IEEE International Conference on Data Mining
.
IEEE, 995–1000.
Rendle (2012)
Steffen Rendle.
2012.
Factorization Machines with libFM.
ACM Trans. Intell. Syst. Technol.
3, 3, Article 57
(May 2012), 22 pages.
Rendle
et al
.
(2020)
Steffen Rendle, Walid
Krichene, Li Zhang, and John
Anderson. 2020.
Neural Collaborative Filtering vs. Matrix
Factorization Revisited.
arXiv preprint arXiv:2005.09683
(2020).
Resnick and
Varian (1997)
Paul Resnick and Hal R
Varian. 1997.
Recommender systems.
Commun. ACM
40,
3 (1997), 56–58.
Rumelhart
et al
.
(1985)
David E Rumelhart,
Geoffrey E Hinton, and Ronald J
Williams. 1985.
Learning internal representations by error
propagation
.
Technical Report.
California Univ San Diego La Jolla Inst for Cognitive
Science.
Schafer
et al
.
(1999)
J Ben Schafer, Joseph
Konstan, and John Riedl.
1999.
Recommender systems in e-commerce. In
Proceedings of the 1st ACM conference on Electronic
commerce
. 158–166.
Schmidhuber (2015)
Jürgen Schmidhuber.
2015.
Deep learning in neural networks: An overview.
Neural networks
61
(2015), 85–117.
Seide
et al
.
(2011)
Frank Seide, Gang Li,
Xie Chen, and Dong Yu.
2011.
Feature engineering in context-dependent deep
neural networks for conversational speech transcription. In
2011 IEEE Workshop on Automatic Speech Recognition
& Understanding
. IEEE, 24–29.
Shan
et al
.
(2016)
Ying Shan, T Ryan Hoens,
Jian Jiao, Haijing Wang,
Dong Yu, and JC Mao.
2016.
Deep Crossing: Web-Scale Modeling without Manually
Crafted Combinatorial Features. In
Proceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining
. ACM, 255–262.
Shazeer et al
.
(2017)
Noam Shazeer, Azalia
Mirhoseini, Krzysztof Maziarz, Andy
Davis, Quoc Le, Geoffrey Hinton, and
Jeff Dean. 2017.
Outrageously large neural networks: The
sparsely-gated mixture-of-experts layer.
arXiv preprint arXiv:1701.06538
(2017).
Song et al
.
(2019)
Weiping Song, Chence Shi,
Zhiping Xiao, Zhijian Duan,
Yewen Xu, Ming Zhang, and
Jian Tang. 2019.
Autoint: Automatic feature interaction learning via
self-attentive neural networks. In
Proceedings of
the 28th ACM International Conference on Information and Knowledge
Management
. 1161–1170.
Valiant (2014)
Gregory Valiant.
2014.
Learning polynomials with neural networks.
(2014).
Vaswani et al
.
(2017)
Ashish Vaswani, Noam
Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia
Polosukhin. 2017.
Attention is all you need. In
Advances in neural information processing systems
.
5998–6008.
Veit
et al
.
(2016)
Andreas Veit, Michael J
Wilber, and Serge Belongie.
2016.
Residual Networks Behave Like Ensembles of
Relatively Shallow Networks.
In
Advances in Neural Information Processing
Systems 29
, D. D. Lee,
M. Sugiyama, U. V. Luxburg,
I. Guyon, and R. Garnett (Eds.).
Curran Associates, Inc., 550–558.
Wang
et al
.
(2017)
Ruoxi Wang, Bin Fu,
Gang Fu, and Mingliang Wang.
2017.
Deep & Cross Network for Ad Click Predictions.
In
Proceedings of the ADKDD’17
.
1–7.
Wang
et al
.
(2019)
Ruoxi Wang, Yingzhou Li,
Michael W Mahoney, and Eric Darve.
2019.
Block Basis Factorization for Scalable Kernel
Evaluation.
SIAM J. Matrix Anal. Appl.
40, 4 (2019),
1497–1526.
Yu
et al
.
(2017)
Xiyu Yu, Tongliang Liu,
Xinchao Wang, and Dacheng Tao.
2017.
On compressing deep models by low rank and sparse
decomposition. In
Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition
.
7370–7379.
Appendix
10.
Baseline performance reported in papers
Tab.
9
lists the quoted Logloss and AUC metrics reported in papers for each baseline.
Table 9.
Baseline performance reported in papers. The metrics (Logloss, AUC) are quoted from papers. Each row represents a baseline, each column represents the paper where the metrics are being reported. The best metric for each baseline is marked in bold.
Model
Paper
DeepFM
(
Guo
et al. 2017
)
(2017)
DCN
(
Wang
et al. 2017
)
(2017)
xDeepFM
(
Lian
et al. 2018
)
(2018)
DLRM
(
Naumov et al. 2019
)
(2019)
AutoInt
(
Song et al. 2019
)
(2019)
DCN-V2 (ours)
DeepFM
(0.45083, 0.8007)
–
(0.4468, 0.8025)
–
(0.4449, 0.8066)
(0.4420, 0.8099)
DCN
–
(0.4419, -)
(0.4467, 0.8026)
(-,
∼
\sim
0.789)
(0.4447, 0.8067)
(0.4420, 0.8099)
xDeepFM
–
–
(
0.4418
, 0.8052)
–
(0.4447, 0.8070)
(0.4421,
0.8099)
DLRM
–
–
–
(-,
∼
\sim
0.790)
–
(0.4427,
0.8092)
AutoInt
–
–
–
–
(0.4434, 0.8083)
(0.4420, 0.8101)
DCN-V2
–
–
–
–
–
(0.4406, 0.8115)
DNN
–
(0.4428, -)
(0.4491, 0.7993)
–
–
(0.4421, 0.8098)
11.
Theorem Proofs
11.1.
Proofs for
Theorem 4.2
Proof.
We start with notations; then prove by induction.
Notations.
Let
[
k
]
:=
{
1
,
…
,
k
}
[k]:=\{1,\ldots,k\}
. Let’s denote the embedding as
𝐱
=
[
𝐱
1
;
𝐱
2
;
…
;
𝐱
c
]
{\bf x}=[{\bf x}_{1};{\bf x}_{2};\ldots;{\bf x}_{c}]
, the output from the
l
l
-th cross layer to be
𝐱
l
=
[
𝐱
1
l
;
𝐱
2
l
;
…
;
𝐱
c
l
]
{\bf x}^{l}=[{\bf x}_{1}^{l};{\bf x}_{2}^{l};\ldots;{\bf x}_{c}^{l}]
where
𝐱
i
,
𝐱
i
l
∈
ℝ
e
i
{\bf x}_{i},{\bf x}_{i}^{l}\in\mathbb{R}^{e_{i}}
and
e
i
e_{i}
is the embedding size for the
i
i
-th feature.
To simplify the notations, let’s also define the feature interaction between features in an ordered set
I
I
(
e.g.,
(
i
1
,
i
3
,
i
4
)
(i_{1},i_{3},i_{4})
) with weights characterized by an ordered set
J
J
as
(8)
g
⁡
(
I
,
J
,
𝐱
,
W
)
=
𝐱
i
1
⊙
(
W
i
1
,
i
2
j
1
​
𝐱
i
2
⊙
…
⊙
(
W
i
k
,
i
k
+
1
j
k
​
𝐱
i
l
+
1
)
)
g(I,J;{\bf x},W)={\bf x}_{i_{1}}\odot\left(W_{i_{1},i_{2}}^{j_{1}}{\bf x}_{i_{2}}\odot\ldots\odot\left(W_{i_{k},i_{k+1}}^{j_{k}}{\bf x}_{i_{l+1}}\right)\right)
where weights
W
i
a
,
i
b
j
W_{i_{a},i_{b}}^{j}
represents the
(
i
a
,
i
b
)
(i_{a},i_{b})
-th block in weight
W
j
W^{j}
at the
j
j
-th cross layer, and it serves as two purposes: align the dimensions between features and increase the impressiveness of the feature cross representations. Note that given the order of
𝐱
i
{\bf x}_{i}
’s, the subscripts of matrix
W
W
’s are uniquely determined.
Proposition.
We first proof by induction that
𝐱
i
l
{\bf x}_{i}^{l}
has the following formula:
(9)
𝐱
i
l
=
∑
p
=
2
l
+
1
∑
I
∈
S
p
i
∑
J
∈
C
l
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
{\bf x}_{i}^{l}=\sum_{p=2}^{l+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{l}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{i}
where
S
p
i
S_{p}^{i}
is a set which represents all the combinations of choosing
p
p
elements from
[
c
]
[c]
with replacement, and with first element fixed to be
i
i
:
S
p
i
=
:
{
𝐲
∈
[
c
]
p
|
y
1
=
i
}
,
∀
I
∈
S
p
,
I
=
(
i
1
,
…
,
i
p
)
;
S_{p}^{i}=:\bigl\{{\bf y}\in[c]^{p}\mathrel{\big|}y_{1}=i\bigr\},\penalty\ \forall I\in S_{p},\penalty\ I=(i_{1},\ldots,i_{p});
and
C
l
p
−
1
C_{l}^{p-1}
is a set that represents choosing a combination of
p
−
1
p-1
indices out of integers
[
l
]
[l]
at a time:
C
l
p
−
1
:=
{
𝐲
∈
[
l
]
p
−
1
|
∀
i
<
j
,
y
i
>
y
j
}
.
C_{l}^{p-1}:=\bigl\{{\bf y}\in[l]^{p-1}\mathrel{\big|}\forall i<j,y_{i}>y_{j}\bigr\}.
Base case.
When
l
=
1
l=1
,
𝐱
i
1
=
∑
j
W
i
,
j
1
​
𝐱
j
+
𝐱
i
{\bf x}_{i}^{1}=\sum_{j}W_{i,j}^{1}{\bf x}_{j}+{\bf x}_{i}
.
Induction step.
Let’s assume that when
l
=
k
l=k
,
𝐱
i
k
=
∑
p
=
2
k
+
1
∑
I
∈
S
p
i
∑
J
∈
C
k
p
−
1
g
J
​
(
𝐱
,
I
)
+
𝐱
i
{\bf x}_{i}^{k}=\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k}^{p-1}}g_{J}({\bf x};I)+{\bf x}_{i}
Then, for
l
=
k
+
1
l=k+1
, we have
𝐱
i
k
+
1
=
𝐱
i
⊙
∑
q
=
1
c
W
i
,
q
k
+
1
​
𝐱
q
k
+
𝐱
i
k
\displaystyle{\bf x}_{i}^{k+1}={\bf x}_{i}\odot\sum_{q=1}^{c}W_{i,q}^{k+1}{\bf x}_{q}^{k}+{\bf x}_{i}^{k}
=
\displaystyle=
𝐱
i
⊙
∑
q
=
1
c
W
i
,
q
k
+
1
​
(
∑
p
=
2
k
+
1
∑
I
∈
S
p
q
∑
J
∈
C
k
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
q
)
+
\displaystyle\penalty\ {\bf x}_{i}\odot\sum_{q=1}^{c}W_{i,q}^{k+1}\left(\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{q}}\sum_{J\in C_{k}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{q}\right)+
∑
p
=
2
k
+
1
∑
I
∈
S
p
i
∑
J
∈
C
k
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
\displaystyle\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{i}
=
\displaystyle=
∑
q
=
1
c
∑
p
=
2
k
+
1
∑
I
∈
S
p
q
∑
J
∈
C
k
p
−
1
𝐱
i
⊙
(
W
i
,
q
k
+
1
​
g
​
(
I
,
J
,
𝐱
,
W
)
)
+
\displaystyle\sum_{q=1}^{c}\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{q}}\sum_{J\in C_{k}^{p-1}}{\bf x}_{i}\odot\left(W_{i,q}^{k+1}g(I,J;{\bf x},W)\right)+
∑
q
=
1
c
𝐱
i
⊙
W
i
,
q
k
+
1
​
𝐱
q
+
∑
p
=
2
k
+
1
∑
I
∈
S
p
i
∑
J
∈
C
k
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
\displaystyle\sum_{q=1}^{c}{\bf x}_{i}\odot W_{i,q}^{k+1}{\bf x}_{q}+\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{i}
=
\displaystyle=
∑
p
=
2
k
+
1
∑
J
∈
C
k
p
−
1
∑
q
=
1
c
∑
I
∈
S
p
q
𝐱
i
⊙
(
W
i
,
q
k
+
1
​
g
​
(
I
,
J
,
𝐱
,
W
)
)
+
\displaystyle\sum_{p=2}^{k+1}\sum_{J\in C_{k}^{p-1}}\sum_{q=1}^{c}\sum_{I\in S_{p}^{q}}{\bf x}_{i}\odot\left(W_{i,q}^{k+1}g(I,J;{\bf x},W)\right)+
∑
p
=
2
∑
J
=
k
+
1
∑
I
∈
S
2
i
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
∑
p
=
2
k
+
1
∑
I
∈
S
p
i
∑
J
∈
C
k
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
\displaystyle\sum_{p=2}\sum_{J=k+1}\sum_{I\in S_{2}^{i}}g(I,J;{\bf x},W)+\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{i}
=
\displaystyle=
∑
p
=
2
k
+
1
∑
J
∈
k
+
1
⊕
C
k
p
−
1
∑
I
∈
S
p
+
1
i
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
\displaystyle\sum_{p=2}^{k+1}\sum_{J\in{k+1}\oplus C_{k}^{p-1}}\sum_{I\in S_{p+1}^{i}}g(I,J;{\bf x},W)+
∑
p
=
2
∑
J
=
k
+
1
∑
I
∈
S
2
i
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
∑
p
=
2
k
+
1
∑
I
∈
S
p
i
∑
J
∈
C
k
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
\displaystyle\sum_{p=2}\sum_{J=k+1}\sum_{I\in S_{2}^{i}}g(I,J;{\bf x},W)+\sum_{p=2}^{k+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{i}
=
\displaystyle=
(
∑
p
=
3
k
+
2
∑
J
∈
k
+
1
⊕
C
k
p
−
2
∑
I
∈
S
p
i
+
∑
p
=
3
k
+
1
∑
I
∈
S
p
i
∑
J
∈
C
k
p
−
1
)
g
(
I
,
J
;
𝐱
,
W
)
+
\displaystyle\left(\sum_{p=3}^{k+2}\sum_{J\in{k+1}\oplus C_{k}^{p-2}}\sum_{I\in S_{p}^{i}}+\sum_{p=3}^{k+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k}^{p-1}}\right)g(I,J;{\bf x},W)+
(
∑
p
=
2
∑
I
∈
S
2
i
∑
J
∈
C
k
1
g
(
I
,
J
;
𝐱
,
W
)
+
∑
p
=
2
∑
J
=
k
+
1
∑
I
∈
S
2
i
)
g
(
I
,
J
;
𝐱
,
W
)
+
𝐱
i
\displaystyle\left(\sum_{p=2}\sum_{I\in S_{2}^{i}}\sum_{J\in C_{k}^{1}}g(I,J;{\bf x},W)+\sum_{p=2}\sum_{J=k+1}\sum_{I\in S_{2}^{i}}\right)g(I,J;{\bf x},W)+{\bf x}_{i}
=
\displaystyle=
∑
p
=
3
k
+
2
∑
J
∈
C
k
+
1
p
−
1
∑
I
∈
S
p
i
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
∑
p
=
2
∑
J
=
C
k
+
1
p
−
1
∑
I
∈
S
p
i
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
\displaystyle\sum_{p=3}^{k+2}\sum_{J\in C_{k+1}^{p-1}}\sum_{I\in S_{p}^{i}}g(I,J;{\bf x},W)+\sum_{p=2}\sum_{J=C_{k+1}^{p-1}}\sum_{I\in S_{p}^{i}}g(I,J;{\bf x},W)+{\bf x}_{i}
=
\displaystyle=
∑
p
=
2
k
+
2
∑
I
∈
S
p
i
∑
J
∈
C
k
+
1
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
𝐱
i
\displaystyle\sum_{p=2}^{k+2}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{k+1}^{p-1}}g(I,J;{\bf x},W)+{\bf x}_{i}
where
⊕
\oplus
denotes adding index
k
+
1
k+1
to each element in the set of
C
k
p
−
1
C_{k}^{p-1}
. The first
5
5
equalities are are straightforward.
For the
6
th
6^{\text{th}}
equality, we first interchanged variable
p
′
=
p
+
1
p^{\prime}=p+1
for the first term, and separated the third term into cases of
p
=
2
p=2
and
p
>
2
p>2
. Then, we group the terms into two cases:
p
=
2
p=2
and
p
>
2
p>2
.
For the second to the last equality, we combined the summations over
J
J
. Consider the set of choosing a combination of
p
−
1
p-1
indices from
k
+
1
k+1
integers, it could be separated into two sets, with index
k
+
1
k+1
and without. Hence,
C
k
+
1
p
−
1
=
C
k
p
−
1
∪
(
(
k
+
1
)
⊕
C
k
p
−
2
)
C_{k+1}^{p-1}=C_{k}^{p-1}\cup\left((k+1)\oplus C_{k}^{p-2}\right)
.
Conclusion.
Since both the base case and the induction step hold, we conclude that
∀
l
≥
1
\forall\penalty\ l\geq 1
, Eq (
9
) holds. This completes the proof.
In such case, the
l
l
-th cross layer contains all the feature interactions (feature-wise) of order up to
l
+
1
l+1
. The interactions between different feature set is parameterized differently, specifically, the interactions between features in set
I
I
(feature’s can be repeated) of order
p
p
is
∑
𝐢
∈
I
′
∑
𝐣
∈
C
p
p
−
1
{
g
(
𝐢
,
𝐣
;
𝐱
,
W
)
=
𝐱
i
1
⊙
(
W
i
1
,
i
2
j
1
𝐱
i
2
⊙
…
⊙
(
W
i
k
,
i
k
+
1
j
k
𝐱
i
l
+
1
)
)
}
\begin{split}\sum_{{\bf i}\in I^{\prime}}\sum_{{\bf j}\in C_{p}^{p-1}}\left\{g({\bf i},{\bf j};{\bf x},W)={\bf x}_{i_{1}}\odot\left(W_{i_{1},i_{2}}^{j_{1}}{\bf x}_{i_{2}}\odot\ldots\odot\left(W_{i_{k},i_{k+1}}^{j_{k}}{\bf x}_{i_{l+1}}\right)\right)\right\}\end{split}
where
I
′
I^{\prime}
contains all the permutations of elements in
I
I
.
∎
11.2.
Proofs for
Theorem 4.1
Proof.
Instead of treating each feature embedding as a unit, we treat each element
x
i
x_{i}
in input embedding
𝐱
=
[
x
1
,
x
2
,
…
,
x
d
]
{\bf x}=[x_{1},x_{2},\ldots,x_{d}]
as a unit. This is a special case of
Theorem 4.2
where all the feature embedding sizes are 1. In such case, all the computations are interchangeable. Hence, we adopt the notations and also the result of
Equation 9
, that is, the
i
i
-th element in the
l
l
-th layer of cross network
𝐱
l
{\bf x}^{l}
has the following formula:
(10)
𝐱
i
l
=
∑
p
=
2
l
+
1
∑
I
∈
S
p
i
∑
J
∈
C
l
p
−
1
g
⁡
(
I
,
J
,
𝐱
,
W
)
+
x
i
{\bf x}_{i}^{l}=\sum_{p=2}^{l+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{l}^{p-1}}g(I,J;{\bf x},W)+x_{i}
To ease the proof and simplify the final formula, we assume the final logit for a
l
l
-layer cross network is
𝟏
⊤
​
𝐱
l
{\bf 1}^{\top}{\bf x}^{l}
, then
𝟏
⊤
​
𝐱
l
=
∑
i
=
1
d
∑
p
=
2
l
+
1
∑
I
∈
S
p
i
∑
J
∈
C
l
p
−
1
x
i
1
⊙
(
w
i
1
​
i
2
(
j
1
)
​
x
i
2
⊙
…
⊙
(
w
i
k
​
i
k
+
1
(
j
k
)
​
x
i
l
+
1
)
)
+
∑
i
=
1
d
x
i
=
∑
p
=
2
l
+
1
∑
I
∈
S
p
∑
J
∈
C
l
p
−
1
w
i
1
​
i
2
(
j
1
)
​
…
​
w
i
k
​
i
k
+
1
(
j
k
)
​
x
i
1
​
x
i
2
​
…
​
x
i
l
+
1
+
∑
i
=
1
d
x
i
=
∑
p
=
2
l
+
1
∑
|
𝜶
|
=
p
∑
J
∈
C
l
p
−
1
∑
𝐢
∈
P
𝜶
∏
k
=
1
|
𝜶
|
−
1
w
i
k
​
i
k
+
1
(
j
k
)
x
1
α
1
x
2
α
2
⋯
x
d
α
d
+
∑
i
=
1
d
x
i
=
∑
𝜶
∑
𝐣
∈
C
l
|
𝜶
|
−
1
∑
𝐢
∈
P
𝜶
∏
k
=
1
|
𝜶
|
−
1
w
i
k
​
i
k
+
1
(
j
k
)
x
1
α
1
x
2
α
2
⋯
x
d
α
d
+
∑
i
=
1
d
x
i
\begin{split}{\bf 1}^{\top}{\bf x}^{l}&=\sum_{i=1}^{d}\sum_{p=2}^{l+1}\sum_{I\in S_{p}^{i}}\sum_{J\in C_{l}^{p-1}}x_{i_{1}}\odot\left(w_{i_{1}i_{2}}^{(j_{1})}x_{i_{2}}\odot\ldots\odot\left(w_{i_{k}i_{k+1}}^{(j_{k})}x_{i_{l+1}}\right)\right)+\sum_{i=1}^{d}x_{i}\\
&=\sum_{p=2}^{l+1}\sum_{I\in S_{p}}\sum_{J\in C_{l}^{p-1}}w_{i_{1}i_{2}}^{(j_{1})}\ldots w_{i_{k}i_{k+1}}^{(j_{k})}x_{i_{1}}x_{i_{2}}\ldots x_{i_{l+1}}+\sum_{i=1}^{d}x_{i}\\
&=\sum_{p=2}^{l+1}\sum_{|{\bm{\alpha}}|=p}\sum_{J\in C_{l}^{p-1}}\sum_{{\bf i}\in P_{\bm{\alpha}}}\prod_{k=1}^{|{\bm{\alpha}}|-1}w_{i_{k}i_{k+1}}^{(j_{k})}x_{1}^{\alpha_{1}}x_{2}^{\alpha_{2}}\cdots x_{d}^{\alpha_{d}}+\sum_{i=1}^{d}x_{i}\\
&=\sum_{{\bm{\alpha}}}\sum_{{\bf j}\in C_{l}^{|{\bm{\alpha}}|-1}}\sum_{{\bf i}\in P_{\bm{\alpha}}}\prod_{k=1}^{|{\bm{\alpha}}|-1}w_{i_{k}i_{k+1}}^{(j_{k})}x_{1}^{\alpha_{1}}x_{2}^{\alpha_{2}}\cdots x_{d}^{\alpha_{d}}+\sum_{i=1}^{d}x_{i}\end{split}
where
P
𝜶
P_{\bm{\alpha}}
is the set of all the permutations of
(
1
⋯
1
⏟
α
1
​
times
⋯
d
⋯
d
⏟
α
d
​
times
)
(\underbrace{1\cdots 1}_{\alpha_{1}\penalty\ \text{times}}\cdots\underbrace{d\cdots d}_{\alpha_{d}\penalty\ \text{times}})
,
C
l
|
𝜶
|
−
1
C_{l}^{|{\bm{\alpha}}|-1}
is a set that represents choosing a combination of
|
𝜶
|
−
1
|{\bm{\alpha}}|-1
indices out of integers
{
1
,
⋯
,
l
}
\{1,\cdots,l\}
at a time, specifically,
C
l
|
𝜶
|
−
1
≔
{
𝐲
∈
[
l
]
|
𝜶
|
−
1
|
(
y
i
≠
y
j
)
∧
(
y
j
1
>
y
j
2
>
…
>
y
j
|
𝜶
|
−
1
)
}
.
C_{l}^{|{\bm{\alpha}}|-1}\coloneqq\bigl\{{\bf y}\in[l]^{|{\bm{\alpha}}|-1}\mathrel{\big|}(y_{i}\neq y_{j})\penalty\ \wedge\penalty\ (y_{j_{1}}>y_{j_{2}}>\ldots>y_{j_{|{\bm{\alpha}}|-1}})\bigr\}.
The second equality combined the first and the third summations into a single one summing over a new set
S
p
c
:=
[
c
]
p
S_{p}^{c}:=[c]^{p}
. The third equality re-represented the cross terms (monomials) using multi-index
𝜶
{\bm{\alpha}}
, and modified the index for weights
w
w
’s accordingly. The last equality combined the first two summations. Thus the proof.
∎