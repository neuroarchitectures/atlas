# Paper (Sun et al. 2019)

> Source: `https://arxiv.org/abs/1904.06690`

---

BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer
Conference:
The 28th ACM International Conference on Information and Knowledge Management; November 3–7, 2019; Beijing, China
The 28th ACM International Conference on Information and Knowledge Management (CIKM ’19), November 3–7, 2019, Beijing, China
Price:
15.00
DOI:
10.1145/3357384.3357895
ISBN:
978-1-4503-6976-3/19/11
CCS:
Information systems Recommender systems
Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang
Affiliation:
Alibaba Group, Beijing, China
email:
ofey.sf, yanhan.lj, joshuawu.wujian, changhua.pch, hc.lx, santong.oww, jiangpeng.jp@alibaba-inc.com
2019© , 2019;
Abstract.
Modeling users’ dynamic preferences from their historical behaviors is challenging and crucial for recommendation systems.
Previous methods employ sequential neural networks to encode users’ historical interactions from left to right into hidden representations for making recommendations.
Despite their effectiveness, we argue that such left-to-right unidirectional models are sub-optimal due to the limitations including:
a
)
unidirectional architectures restrict the power of hidden representation in users’ behavior sequences;
b
)
they often assume a rigidly ordered sequence which is not always practical.
To address these limitations, we proposed a sequential recommendation model called
BERT4Rec
, which employs the deep bidirectional self-attention to model user behavior sequences.
To avoid the information leakage and efficiently train the bidirectional model, we adopt the Cloze objective to sequential recommendation, predicting the random masked items in the sequence by jointly conditioning on their left and right context.
In this way, we learn a bidirectional representation model to make recommendations by allowing each item in user historical behaviors to fuse information from both left and right sides.
Extensive experiments on four benchmark datasets show that our model outperforms various state-of-the-art sequential models consistently.
Keywords:
Sequential Recommendation; Bidirectional Sequential Model; Cloze
1.
Introduction
Accurately characterizing users’ interests lives at the heart of an effective recommendation system.
In many real-world applications, users’ current interests are intrinsically dynamic and evolving, influenced by their historical behaviors.
For example, one may purchase accessories (
e.g.
, Joy-Con controllers) soon after buying a Nintendo Switch, though she/he will not buy console accessories under normal circumstances.
To model such sequential dynamics in user behaviors, various methods have been proposed to make
sequential recommendations
based on users’ historical interactions
(
Rendle et al. 2010
;
Hidasi et al. 2016
;
Kang and McAuley [n. d.]
)
.
They aim to predict the successive item(s) that a user is likely to interact with given her/his past interactions.
Recently, a surge of works employ sequential neural networks,
e.g.
, Recurrent Neural Network (RNN), for sequential recommendation and obtain promising results
(
Hidasi et al. 2016
;
Yu
et al. 2016
;
Wu
et al. 2017
;
Donkers
et al. 2017
;
Hidasi and
Karatzoglou 2018
)
.
The basic paradigm of previous work is to encode a user’s historical interactions into a vector (
i.e.
, representation of user’s preference) using a left-to-right sequential model and make recommendations based on this hidden representation.
Despite their prevalence and effectiveness, we argue that such left-to-right unidirectional models are not sufficient to learn optimal representations for user behavior sequences. The major limitation, as illustrated in Figure
1
c and
1
d, is that such unidirectional models restrict the power of hidden representation for items in the historical sequences, where each item can only encode the information from previous items.
Another limitation is that previous unidirectional models are originally introduced for sequential data with natural order,
e.g.
, text and time series data.
They often assume a rigidly ordered sequence over data which is not always true for user behaviors in real-world applications.
In fact, the choices of items in a user’s historical interactions may not follow a rigid order assumption
(
Hu
et al. 2017
;
Wang
et al. 2018
)
due to various unobservable external factors
(
Covington
et al. 2016
)
. In such a situation, it is crucial to incorporate context from both directions in user behavior sequence modeling.
To address the limitations mentioned above, we seek to use a bidirectional model to learn the representations for users’ historical behavior sequences.
Specifically, inspired by the success of BERT
(
Devlin
et al. 2018
)
in text understanding, we propose to apply the deep bidirectional self-attention model to sequential recommendation, as illustrated in Figure
1
b.
For representation power, the superior results for deep bidirectional models on text sequence modeling tasks show that it is beneficial to incorporate context from both sides for sequence representations learning
(
Devlin
et al. 2018
)
.
For rigid order assumption, our model is more suitable than unidirectional models in modeling user behavior sequences since all items in the bidirectional model can leverage the contexts from both left and right side.
However, it is not straightforward and intuitive to train the bidirectional model for sequential recommendation. Conventional sequential recommendation models are usually trained left-to-right by predicting the next item for each position in the input sequence.
As shown in Figure
1
, jointly conditioning on both left and right context in a deep bidirectional model would cause information leakage,
i.e.
, allowing each item to indirectly “see the target item”.
This could make predicting the future become trivial and the network would not learn anything useful.
To tackle this problem, we introduce the Cloze task
(
Devlin
et al. 2018
;
Taylor 1953
)
to take the place of the objective in unidirectional models (
i.e.
, sequentially predicting the next item).
Specifically, we randomly mask some items (
i.e.
, replace them with a special token
[mask]
) in the input sequences, and then predict the ids of those masked items based on their surrounding context. In this way, we avoid the information leakage and learn a bidirectional representation model by allowing the representation of each item in the input sequence to fuse both the left and right context.
In addition to training a bidirectional model, another advantage of the Cloze objective is that it can produce more samples to train a more powerful model in multiple epochs. However, a downside of the Cloze task is that it is not consistent with the final task (
i.e.
, sequential recommendation).
To fix this, during the test, we append the special token “
[mask]
” at the end of the input sequence to indicate the item that we need to predict, and then make recommendations base on its final hidden vector.
Extensive experiments on four datasets show that our model outperforms various state-of-the-art baselines consistently.
The contributions of our paper are as follows:
•
We propose to model user behavior sequences with a bidirectional self-attention network through Cloze task. To the best of our knowledge, this is the first study to introduce deep bidirectional sequential model and Cloze objective into the field of recommendation systems.
•
We compare our model with state-of-the-art methods and demonstrate the effectiveness of both bidirectional architecture and the Cloze objective through quantitative analysis on four benchmark datasets.
•
We conduct a comprehensive ablation study to analyze the contributions of key components in the proposed model.
2.
Related Work
In this section, we will briefly review several lines of works closely related to ours, including general recommendation, sequential recommendation, and attention mechanism.
2.1.
General Recommendation
Early works on recommendation systems typically use Collaborative Filtering (CF) to model users’ preferences based on their interaction histories
(
Sarwar
et al. 2001
;
Koren and Bell 2011
)
.
Among various CF methods, Matrix Factorization (MF) is the most popular one, which projects users and items into a shared vector space and estimate a user’s preference on an item by the inner product between their vectors
(
Salakhutdinov and
Mnih 2007
;
Koren
et al. 2009
;
Koren and Bell 2011
)
.
Another line of work is item-based neighborhood methods
(
Sarwar
et al. 2001
;
Linden
et al. 2003
;
Koren 2008
;
Kabbur
et al. 2013
)
.
They estimate a user’s preference on an item via measuring its similarities with the items in her/his interaction history using a precomputed item-to-item similarity matrix.
Recently, deep learning has been revolutionizing the recommendation systems dramatically.
The early pioneer work is a two-layer Restricted Boltzmann Machines (RBM) for collaborative filtering, proposed by
Salakhutdinov
et al. 2007
(
Salakhutdinov
et al. 2007
)
in Netflix Prize
1
1
1
https://www.netflixprize.com
.
One line of deep learning based methods seeks to improve the recommendation performance by integrating the distributed item representations learned from auxiliary information,
e.g.
, text
(
Wang
et al. 2015
;
Kim
et al. 2016
)
, images
(
Wang et al. 2017
;
Kang
et al. 2017
)
, and acoustic features
(
van den
Oord et al. 2013
)
into CF models.
Another line of work seeks to take the place of conventional matrix factorization.
For example, Neural Collaborative Filtering (NCF)
(
He
et al. 2017b
)
estimates user preferences via Multi-Layer Perceptions (MLP) instead of inner product, while AutoRec
(
Sedhain
et al. 2015
)
and CDAE
(
Wu
et al. 2016
)
predict users’ ratings using Auto-encoder framework.
2.2.
Sequential Recommendation
Unfortunately, none of the above methods is for sequential recommendation since they all ignore the order in users’ behaviors.
Early works on sequential recommendation usually capture sequential patterns from user historical interactions using Markov chains (MCs).
For example,
Shani
et al. 2005
(
Shani
et al. 2005
)
formalized recommendation generation as a sequential optimization problem and employ Markov Decision Processes (MDPs) to address it.
Later,
Rendle et al. 2010
(
Rendle et al. 2010
)
combine the power of MCs and MF to model both sequential behaviors and general interests by Factorizing Personalized Markov Chains (FPMC).
Besides the first-order MCs, high-order MCs are also adopted to consider more previous items
(
He and McAuley 2016
;
He
et al. 2017a
)
.
Recently, RNN and its variants, Gated Recurrent Unit (GRU)
(
Cho et al. 2014
)
and Long Short-Term Memory (LSTM)
(
Hochreiter and
Schmidhuber 1997
)
, are becoming more and more popular for modeling user behavior sequences
(
Hidasi et al. 2016
;
Yu
et al. 2016
;
Wu
et al. 2017
;
Donkers
et al. 2017
;
Quadrana et al. 2017
;
Li
et al. 2017
;
Hidasi and
Karatzoglou 2018
)
.
The basic idea of these methods is to encode user’s previous records into a vector (
i.e.
, representation of user’s preference which is used to make predictions) with various recurrent architectures and loss functions, including session-based GRU with ranking loss (GRU4Rec)
(
Hidasi et al. 2016
)
, Dynamic REcurrent bAsket Model (DREAM)
(
Yu
et al. 2016
)
, user-based GRU
(
Donkers
et al. 2017
)
, attention-based GRU (NARM)
(
Li
et al. 2017
)
, and improved GRU4Rec with new loss function (
i.e.
,
BPR-max
and
TOP1-max
) and an improved sampling strategy
(
Hidasi and
Karatzoglou 2018
)
.
Other than recurrent neural networks, various deep learning models are also introduced for sequential recommendation
(
Chen
et al. 2018
;
Tang and Wang 2018
;
Liu
et al. 2018b
;
Kang and McAuley [n. d.]
)
. For example,
Tang and Wang 2018
(
Tang and Wang 2018
)
propose a Convolutional Sequence Model (Caser) to learn sequential patterns using both horizontal and vertical convolutional filters.
Chen
et al. 2018
(
Chen
et al. 2018
)
and
Huang
et al. 2018
(
Huang
et al. 2018
)
employ Memory Network to improve sequential recommendation.
STAMP captures both users’ general interests and current interests using an MLP network with attention
(
Liu
et al. 2018b
)
.
2.3.
Attention Mechanism
Attention mechanism has shown promising potential in modeling sequential data,
e.g.
, machine translation
(
Bahdanau
et al. 2015
;
Vaswani et al. 2017
)
and text classification
(
Yang:naacl2016:HAN
)
.
Recently, some works try to employ the attention mechanism to improve recommendation performances and interpretability
(
Li
et al. 2017
;
Liu
et al. 2018b
)
.
For example,
Li
et al. 2017
(
Li
et al. 2017
)
incorporate an attention mechanism into GRU to capture both the user’s sequential behavior and main purpose in session-based recommendation.
𝒗
1
\bm{v}_{1}
⋯
\cdots
𝒗
t
−
1
\bm{v}_{t-1}
𝒗
[mask]
\bm{v}_{\mathchoice{\raisebox{-1.0pt}{\resizebox{840841}{196608}{\hbox{\raisebox{1.75pt}{$\displaystyle\text{[mask]}$}}}}}{\raisebox{-1.0pt}{\resizebox{840841}{196608}{\hbox{\raisebox{1.75pt}{$\textstyle\text{[mask]}$}}}}}{\raisebox{-0.99998pt}{\resizebox{736919}{196608}{\hbox{\raisebox{1.22499pt}{$\scriptstyle\text{[mask]}$}}}}}{\raisebox{-1.0pt}{\resizebox{736915}{196608}{\hbox{\raisebox{0.875pt}{$\scriptscriptstyle\text{[mask]}$}}}}}}
v
1
v_{1}
⋯
\cdots
[mask]
v
t
−
1
v_{t-1}
Trm
⋮
\vdots
Trm
+
Trm
⋮
\vdots
Trm
+
Trm
⋮
\vdots
Trm
+
L
×
L\times
𝒑
1
\bm{p}_{1}
⋯
\cdots
𝒑
t
−
1
\bm{p}_{t-1}
𝒑
t
\bm{p}_{t}
Projection
𝒉
1
L
\bm{h}_{1}^{L}
𝒉
t
−
1
L
\bm{h}_{t-1}^{L}
v
t
v_{t}
𝒉
t
L
\bm{h}_{t}^{L}
Embedding
Layer
⋯
\cdots
⋮
\vdots
⋯
\cdots
Multi-Head
Attention
Dropout
Add & Norm
Position-wise
Feed-Forward
Dropout
Add & Norm
input
Trm
(b) BERT4Rec model architecture.
(a) Transformer Layer.
⋯
\cdots
v
1
v_{1}
…
\ldots
v
t
v_{t}
v
t
−
1
v_{t-1}
GRU
GRU
GRU
…
\ldots
v
2
v_{2}
…
\ldots
v
t
v_{t}
v
t
+
1
v_{t+1}
(d) RNN based sequential recommendation methods.
Trm
Trm
Trm
Trm
Trm
Trm
v
1
v_{1}
v
t
−
1
v_{t-1}
v
t
v_{t}
…
\ldots
…
\ldots
…
\ldots
…
\ldots
v
2
v_{2}
v
t
v_{t}
v
t
+
1
v_{t+1}
(c) SASRec model architecture.
Figure 1
.
Differences in sequential recommendation model architectures. BERT4Rec learns a bidirectional model via Cloze task, while SASRec and RNN based methods are all left-to-right unidirectional model which predict next item sequentially.
The works mentioned above basically treat attention mechanism as an additional component to the original models. In contrast, Transformer
(
Vaswani et al. 2017
)
and BERT
(
Devlin
et al. 2018
)
are built solely on multi-head self-attention and achieve state-of-the-art results on text sequence modeling.
Recently, there is a rising enthusiasm for applying purely attention-based neural networks to model sequential data for their effectiveness and efficiency
(
Lin et al. 2017
;
Shaw
et al. 2018
;
Liu et al. 2018a
;
Radford et al. 2018
;
Radford:2019:Language
)
.
For sequential recommendation,
Kang and McAuley [n. d.]
(
Kang and McAuley [n. d.]
)
introduce a two-layer
Transformer decoder
(
i.e.
, Transformer language model) called SASRec to capture user’s sequential behaviors and achieve state-of-the-art results on several public datasets.
SASRec is closely related to our work.
However, it is still a unidirectional model using a casual attention mask.
While we use a bidirectional model to encode users’ behavior sequences with the help of Cloze task.
3.
BERT4Rec
Before going into the details, we first introduce the research problem, the basic concepts, and the notations in this paper.
3.1.
Problem Statement
In sequential recommendation, let
𝒰
=
{
u
1
,
u
2
,
…
,
u
|
𝒰
|
}
\mathcal{U}{=}\{u_{1},u_{2},\dots,u_{|\mathcal{U}|}\}
denote a set of users,
𝒱
=
{
v
1
,
v
2
,
…
,
v
|
𝒱
|
}
\mathcal{V}{=}\{v_{1},v_{2},\dots,v_{|\mathcal{V}|}\}
be a set of items, and list
𝒮
u
=
[
v
1
(
u
)
,
…
,
v
t
(
u
)
,
…
,
v
n
u
(
u
)
]
\mathcal{S}_{u}{=}[v_{1}^{(u)},\dots,v_{t}^{(u)},\dots,v_{n_{u}}^{(u)}]
denote the interaction sequence in chronological order for user
u
∈
𝒰
u\in\mathcal{U}
, where
v
t
(
u
)
∈
𝒱
v_{t}^{(u)}\in\mathcal{V}
is the item that
u
u
has interacted with at time step
2
2
2
Here, following
(
Rendle et al. 2010
;
Kang and McAuley [n. d.]
)
, we use the relative time index instead of absolute time index for numbering interaction records.
t
t
and
n
u
n_{u}
is the the length of interaction sequence for user
u
u
.
Given the interaction history
𝒮
u
\mathcal{S}_{u}
, sequential recommendation aims to predict the item that user
u
u
will interact with at time step
n
u
+
1
n_{u}+1
.
It can be formalized as modeling the probability over all possible items for user
u
u
at time step
n
u
+
1
n_{u}{+}1
:
p
⁡
(
v
n
u
+
1
(
u
)
=
v
|
𝒮
u
)
p\bigl(v_{n_{u}+1}^{(u)}=v|\ \mathcal{S}_{u}\bigr)
3.2.
Model Architecture
Here, we introduce a new sequential recommendation model called
BERT4Rec
, which adopts
B
idirectional
E
ncoder
R
epresentations from
T
ransformers to a new task, sequential
R
ecommendation.
It is built upon the popular self-attention layer, “Transformer layer”.
As illustrated in Figure
1
b, BERT4Rec is stacked by
L
L
bidirectional Transformer layers.
At each layer, it iteratively revises the representation of every position by exchanging information across all positions at the previous layer in parallel with the Transformer layer.
Instead of learning to pass relevant information forward step by step as RNN based methods did in Figure
1
d, self-attention mechanism endows BERT4Rec with the capability to directly capture the dependencies in any distances.
This mechanism results in a global receptive field, while CNN based methods like Caser usually have a limited receptive field.
In addition, in contrast to RNN based methods, self-attention is straightforward to parallelize.
Comparing Figure
1
b,
1
c, and
1
d, the most noticeable difference is that SASRec and RNN based methods are all left-to-right unidirectional architecture, while our BERT4Rec uses bidirectional self-attention to model users’ behavior sequences.
In this way, our proposed model can obtain more powerful representations of users’ behavior sequences to improve recommendation performances.
3.3.
Transformer Layer
As illustrated in Figure
1
b, given an input sequence of length
t
t
, we iteratively compute hidden representations
𝒉
i
l
\bm{h}_{i}^{l}
at each layer
l
l
for each position
i
i
simultaneously by applying the Transformer layer from
(
Vaswani et al. 2017
)
.
Here, we stack
𝒉
i
l
∈
ℝ
d
\bm{h}_{i}^{l}\in\mathbb{R}^{d}
together into matrix
𝑯
l
∈
ℝ
t
×
d
\bm{H}^{l}{\in}\mathbb{R}^{t\times d}
since we compute attention function on all positions simultaneously in practice.
As shown in Figure
1
a, the Transformer layer
Trm
contains two sub-layers, a
Multi-Head Self-Attention
sub-layer and a
Position-wise Feed-Forward Network
.
Multi-Head Self-Attention
.
Attention mechanisms have become an integral part of sequence modeling in a variety of tasks, allowing capturing the dependencies between representation pairs without regard to their distance in the sequences.
Previous work has shown that it is beneficial to jointly attend to information from different representation subspaces at different positions
(
Vaswani et al. 2017
;
Li
et al. 2018
;
Devlin
et al. 2018
)
.
Thus, we here adopt the multi-head self-attention instead of performing a single attention function.
Specifically, multi-head attention first linearly projects
𝑯
l
\bm{H}^{l}
into
h
h
subspaces, with different, learnable linear projections, and then apply
h
h
attention functions in parallel to produce the output representations which are concatenated and once again projected:
(1)
𝙼𝙷
⁡
(
𝑯
l
)
\displaystyle\mathtt{MH}(\bm{H}^{l})
=
[
head
1
;
head
2
;
…
;
head
h
]
​
𝑾
O
\displaystyle=[\text{head}_{1};\text{head}_{2};\dots;\text{head}_{h}]\bm{W}^{O}
head
i
\displaystyle\text{head}_{i}
=
𝙰𝚝𝚝𝚎𝚗𝚝𝚒𝚘𝚗
⁡
(
𝑯
l
​
𝑾
i
Q
,
𝑯
l
​
𝑾
i
K
,
𝑯
l
​
𝑾
i
V
)
\displaystyle=\mathtt{Attention}\bigl(\bm{H}^{l}\bm{W}_{i}^{Q},\bm{H}^{l}\bm{W}_{i}^{K},\bm{H}^{l}\bm{W}_{i}^{V}\bigr)
where the projections matrices for each head
𝑾
i
Q
∈
ℝ
d
×
d
/
h
\bm{W}_{i}^{Q}\in\mathbb{R}^{d\times d/h}
,
𝑾
i
K
∈
ℝ
d
×
d
/
h
\bm{W}_{i}^{K}\in\mathbb{R}^{d\times d/h}
,
𝑾
i
V
∈
ℝ
d
×
d
/
h
\bm{W}_{i}^{V}\in\mathbb{R}^{d\times d/h}
, and
𝑾
i
O
∈
ℝ
d
×
d
\bm{W}_{i}^{O}\in\mathbb{R}^{d\times d}
are learnable parameters. Here, we omit the layer subscript
l
l
for the sake of simplicity.
In fact, these projection parameters are not shared across the layers.
Here, the
𝙰𝚝𝚝𝚎𝚗𝚝𝚒𝚘𝚗
\mathtt{Attention}
function is
Scaled Dot-Product Attention
:
(2)
𝙰𝚝𝚝𝚎𝚗𝚝𝚒𝚘𝚗
⁡
(
𝑸
,
𝑲
,
𝑽
)
=
softmax
⁡
(
𝑸
​
𝑲
⊤
d
/
h
)
​
𝑽
\mathtt{Attention}(\bm{Q},\bm{K},\bm{V})=\mathrm{softmax}\biggl(\frac{\bm{Q}\bm{K}^{\top}}{\sqrt{d/h}}\biggr)\bm{V}
where
query
𝑸
\bm{Q}
,
key
𝑲
\bm{K}
, and
value
𝑽
\bm{V}
are projected from the same matrix
𝑯
l
\bm{H}^{l}
with different learned projection matrices as in Equation
1
. The
temperature
d
/
h
\sqrt{d/h}
is introduced to produce a softer attention distribution for avoiding extremely small gradients
(
Hinton
et al. 2015
;
Vaswani et al. 2017
)
.
Position-wise Feed-Forward Network
.
As described above, the self-attention sub-layer is mainly based on linear projections.
To endow the model with nonlinearity and interactions between different dimensions, we apply a
Position-wise Feed-Forward Network
to the outputs of the self-attention sub-layer, separately and identically at each position.
It consists of two affine transformations with a Gaussian Error Linear Unit (
GELU
) activation in between:
(3)
PFFN
​
(
𝑯
l
)
\displaystyle\texttt{PFFN}(\bm{H}^{l})
=
[
𝙵𝙵𝙽
​
(
𝒉
1
l
)
⊤
;
…
;
𝙵𝙵𝙽
​
(
𝒉
t
l
)
⊤
]
⊤
\displaystyle=\bigl[\mathtt{FFN}(\bm{h}_{1}^{l})^{\top};\dots;\mathtt{FFN}(\bm{h}_{t}^{l})^{\top}\bigr]^{\top}
𝙵𝙵𝙽
⁡
(
𝒙
)
\displaystyle\mathtt{FFN}(\bm{x})
=
GELU
​
(
𝒙
​
𝑾
(
1
)
+
𝒃
(
1
)
)
​
𝑾
(
2
)
+
𝒃
(
2
)
\displaystyle=\texttt{GELU}\bigl(\bm{x}\bm{W}^{(1)}+\bm{b}^{(1)}\bigr)\bm{W}^{(2)}+\bm{b}^{(2)}
GELU
​
(
x
)
\displaystyle\texttt{GELU}(x)
=
x
​
Φ
​
(
x
)
\displaystyle=x\Phi(x)
where
Φ
⁡
(
x
)
\Phi(x)
is the cumulative distribution function of the standard gaussian distribution,
𝑾
(
1
)
∈
ℝ
d
×
4
​
d
\bm{W}^{(1)}\in\mathbb{R}^{d\times 4d}
,
𝑾
(
2
)
∈
ℝ
4
​
d
×
d
\bm{W}^{(2)}\in\mathbb{R}^{4d\times d}
,
𝒃
(
1
)
∈
ℝ
4
​
d
\bm{b}^{(1)}\in\mathbb{R}^{4d}
and
𝒃
(
2
)
∈
ℝ
d
\bm{b}^{(2)}\in\mathbb{R}^{d}
are learnable parameters and shared across all positions.
We omit the layer subscript
l
l
for convenience.
In fact, these parameters are different from layer to layer.
In this work, following OpenAI GPT
(
Radford et al. 2018
)
and BERT
(
Devlin
et al. 2018
)
, we use a smoother
GELU
(
Hendrycks and
Gimpel 2016
)
activation rather than the standard
ReLu
activation.
Stacking Transformer Layer
.
As elaborated above, we can easily capture item-item interactions across the entire user behavior sequence using self-attention mechanism.
Nevertheless, it is usually beneficial to learn more complex item transition patterns by stacking the self-attention layers.
However, the network becomes more difficult to train as it goes deeper.
Therefore, we employ a residual connection
(
He
et al. 2016
)
around each of the two sub-layers as in Figure
1
a, followed by layer normalization
(
Ba
et al. 2016
)
.
Moreover, we also apply dropout
(
Srivastava et al. 2014
)
to the output of each sub-layer, before it is normalized. That is, the output of each sub-layer is
LN
​
(
𝒙
+
Dropout
​
(
sublayer
​
(
𝒙
)
)
)
\texttt{LN}(\bm{x}+\texttt{Dropout}(\text{sublayer}(\bm{x})))
, where
sublayer
​
(
⋅
)
\text{sublayer}(\cdot)
is the function implemented by the sub-layer itself,
LN
is the layer normalization function defined in
(
Ba
et al. 2016
)
. We use
LN
to normalize the inputs over all the hidden units in the same layer for stabilizing and accelerating the network training.
In summary, BERT4Rec refines the hidden representations of each layer as follows:
(4)
𝑯
l
\displaystyle\bm{H}^{l}
=
𝚃𝚛𝚖
⁡
(
𝑯
l
−
1
)
,
∀
i
∈
[
1
,
…
,
L
]
\displaystyle=\mathtt{Trm}\bigl(\bm{H}^{l-1}\bigr),\quad\forall i\in[1,\dots,L]
(5)
𝚃𝚛𝚖
⁡
(
𝑯
l
−
1
)
\displaystyle\mathtt{Trm}(\bm{H}^{l-1})
=
𝙻𝙽
⁡
(
𝑨
l
−
1
+
𝙳𝚛𝚘𝚙𝚘𝚞𝚝
⁡
(
𝙿𝙵𝙵𝙽
⁡
(
𝑨
l
−
1
)
)
)
\displaystyle=\mathtt{LN}\Bigl(\bm{A}^{l-1}+\mathtt{Dropout}\bigl(\mathtt{PFFN}(\bm{A}^{l-1})\bigr)\Bigr)
(6)
𝑨
l
−
1
\displaystyle\bm{A}^{l-1}
=
𝙻𝙽
(
𝑯
l
−
1
+
𝙳𝚛𝚘𝚙𝚘𝚞𝚝
(
𝙼𝙷
(
𝑯
l
−
1
)
)
)
\displaystyle=\mathtt{LN}\Bigr(\bm{H}^{l-1}+\mathtt{Dropout}\bigl(\mathtt{MH}(\bm{H}^{l-1})\bigr)\Bigr)
3.4.
Embedding Layer
As elaborated above, without any recurrence or convolution module, the Transformer layer
𝚃𝚛𝚖
\mathtt{Trm}
is not aware of the order of the input sequence.
In order to make use of the sequential information of the input, we inject
Positional Embeddings
into the input item embeddings at the bottoms of the Transformer layer stacks.
For a given item
v
i
v_{i}
, its input representation
𝒉
i
0
\bm{h}_{i}^{0}
is constructed by summing the corresponding item and positional embedding:
𝒉
i
0
\displaystyle\bm{h}_{i}^{0}
=
𝒗
i
+
𝒑
i
\displaystyle=\bm{v}_{i}+\bm{p}_{i}
where
𝒗
i
∈
𝑬
\bm{v}_{i}{\in}\bm{E}
is the
d
−
d-
dimensional embedding for item
v
i
v_{i}
,
𝒑
i
∈
𝑷
\bm{p}_{i}{\in}\bm{P}
is the
d
−
d-
dimensional positional embedding for position index
i
i
.
In this work, we use the learnable positional embeddings instead of the fixed sinusoid embeddings in
(
Vaswani et al. 2017
)
for better performances.
The positional embedding matrix
𝑷
∈
ℝ
N
×
d
\bm{P}\in\mathbb{R}^{N\times d}
allows our model to identify which portion of the input it is dealing with.
However, it also imposes a restriction on the maximum sentence length
N
N
that our model can handle.
Thus, we need to truncate the the input sequence
[
v
1
,
…
,
v
t
]
[v_{1},\dots,v_{t}]
to the last
N
N
items
[
v
t
−
N
+
1
u
,
…
,
v
t
]
[v_{t-N+1}^{u},\dots,v_{t}]
if
t
>
N
t>N
.
3.5.
Output Layer
After
L
L
layers that hierarchically exchange information across all positions in the previous layer, we get the final output
𝑯
L
\bm{H}^{L}
for all items of the input sequence.
Assuming that we mask the item
v
t
v_{t}
at time step
t
t
, we then predict the masked items
v
t
v_{t}
base on
𝒉
t
L
\bm{h}_{t}^{L}
as shown in Figure
1
b.
Specifically, we apply a two-layer feed-forward network with
GELU
activation in between to produce an output distribution over target items:
(7)
P
⁡
(
v
)
=
softmax
⁡
(
𝙶𝙴𝙻𝚄
⁡
(
𝒉
t
L
​
𝑾
P
+
𝒃
P
)
​
𝑬
⊤
+
𝒃
O
)
P(v)=\mathrm{softmax}\bigl(\mathtt{GELU}(\bm{h}_{t}^{L}\bm{W}^{P}+\bm{b}^{P})\bm{E}^{\top}+\bm{b}^{O}\bigr)
where
𝑾
P
\bm{W}^{P}
is the learnable projection matrix,
𝒃
P
\bm{b}^{P}
, and
𝒃
O
\bm{b}^{O}
are bias terms,
𝑬
∈
ℝ
|
𝒱
|
×
d
\bm{E}\in\mathbb{R}^{|\mathcal{V}|\times d}
is the embedding matrix for the item set
𝒱
\mathcal{V}
.
We use the shared item embedding matrix in the input and output layer for alleviating overfitting and reducing model size.
3.6.
Model Learning
Training
.
Conventional unidirectional sequential recommendation models usually train the model by predicting the next item for each position in the input sequence as illustrated in Figure
1
c and
1
d.
Specifically, the target of the input sequence
[
v
1
,
…
,
v
t
]
[v_{1},\dots,v_{t}]
is a shifted version
[
v
2
,
…
,
v
t
+
1
]
[v_{2},\dots,v_{t+1}]
.
However, as shown in Figure
1
b, jointly conditioning on both left and right context in a bidirectional model would cause the final output representation of each item to contain the information of the target item.
This makes predicting the future become trivial and the network would not learn anything useful.
A simple solution for this issue is to create
t
−
1
t-1
samples (subsequences with next items like ([
v
1
v_{1}
],
v
2
v_{2}
) and ([
v
1
,
v
2
v_{1},v_{2}
],
v
3
v_{3}
)) from the original length
t
t
behavior sequence and then encode each historical subsequence with the bidirectional model to predict the target item.
However, this approach is very time and resources consuming since we need to create a new sample for each position in the sequence and predict them separately.
In order to efficiently train our proposed model, we apply a new objective:
Cloze
task
(
Taylor 1953
)
(also known as “Masked Language Model” in
(
Devlin
et al. 2018
)
) to sequential recommendation.
It is a test consisting of a portion of language with some words removed, where the participant is asked to fill the missing words.
In our case, for each training step, we randomly mask
ρ
\rho
proportion of all items in the input sequence (
i.e.
, replace with special token “
[mask]
”), and then predict the original ids of the masked items based solely on its left and right context.
For example:
Input
:
[
v
1
,
v
2
,
v
3
,
v
4
,
v
5
]
[v_{1},v_{2},v_{3},v_{4},v_{5}]
[
v
1
,
[v_{1},
[
𝚖𝚊𝚜𝚔
]
1
\mathtt{[mask]}_{1}
,
v
3
,
,v_{3},
[
𝚖𝚊𝚜𝚔
]
2
\mathtt{[mask]}_{2}
,
v
5
]
,v_{5}]
Labels
:
[
𝚖𝚊𝚜𝚔
]
1
\mathtt{[mask]}_{1}
=
v
2
=v_{2}
,
[
𝚖𝚊𝚜𝚔
]
2
\mathtt{[mask]}_{2}
=
v
4
=v_{4}
randomly mask
The final hidden vectors corresponding to “
[mask]
” are fed into an output softmax over the item set, as in conventional sequential recommendation.
Eventually, we define the loss for each masked input
𝒮
u
′
\mathcal{S}_{u}^{\prime}
as the negative log-likelihood of the masked targets:
(8)
ℒ
=
1
|
𝒮
u
m
|
∑
v
m
∈
𝒮
u
m
−
log
P
(
v
m
=
v
m
∗
|
𝒮
u
′
)
\mathcal{L}=\frac{1}{|\mathcal{S}_{u}^{m}|}\sum_{v_{m}\in\mathcal{S}_{u}^{m}}-\log P(v_{m}=v_{m}^{*}|\mathcal{S}_{u}^{\prime})
where
𝒮
u
′
\mathcal{S}_{u}^{\prime}
is the masked version for user behavior history
𝒮
u
\mathcal{S}_{u}
,
𝒮
u
m
\mathcal{S}_{u}^{m}
is the random masked items in it,
v
m
∗
v_{m}^{*}
is the true item for the masked item
v
m
v_{m}
, and the probability
P
⁡
(
⋅
)
P(\cdot)
is defined in Equation
7
.
An additional advantage for Cloze task is that it can generate more samples to train the model.
Assuming a sequence of length
n
n
, conventional sequential predictions in Figure
1
c and
1
d produce
n
n
unique samples for training, while BERT4Rec can obtain
(
n
k
)
\binom{n}{k}
samples (if we randomly mask
k
k
items) in multiple epochs.
It allows us to train a more powerful bidirectional representation model.
Test
.
As described above, we create a mismatch between the training and the final sequential recommendation task since the Cloze objective is to predict the current masked items while sequential recommendation aims to predict the future.
To address this, we append the special token “
[mask]
” to the end of user’s behavior sequence, and then predict the next item based on the final hidden representation of this token.
To better match the sequential recommendation task (
i.e.
, predict the last item), we also produce samples that only mask the last item in the input sequences during training.
It works like fine-tuning for sequential recommendation and can further improve the recommendation performances.
3.7.
Discussion
Here, we discuss the relation of our model with previous related work.
SASRec
. Obviously, SASRec is a left-to-right unidirectional version of our BERT4Rec with single head attention and causal attention mask.
Different architectures lead to different training methods.
SASRec predicts the next item for each position in a sequence, while BERT4Rec predicts the masked items in the sequence using Cloze objective.
CBOW & SG
. Another very similar work is Continuous Bag-of-Words (CBOW) and Skip-Gram (SG)
(
Mikolov
et al. 2013a
)
.
CBOW predicts a target word using the average of all the word vectors in its context (both left and right).
It can be seen as a simplified case of BERT4Rec, if we use one self-attention layer in BERT4Rec with uniform attention weights on items, unshare item embeddings, remove the positional embedding, and only mask the central item.
Similar to CBOW, SG can also be seen as a simplified case of BERT4Rec following similar reduction operations (mask all items except only one).
From this point of view, Cloze can be seen as a general form for the objective of CBOW and SG.
Besides, CBOW uses a simple aggregator to model word sequences since its goal is to learn good word representations, not sentence representations.
On the contrary, we seek to learn a powerful behavior sequence representation model (deep self-attention network in this work) for making recommendations.
BERT
. Although our BERT4Rec is inspired by the BERT in NLP, it still has several differences from BERT:
a
)
The most critical difference is that BERT4Rec is an end-to-end model for sequential recommendation, while BERT is a pre-training model for sentence representation.
BERT leverages large-scale task-independent corpora to pre-train the sentence representation model for various text sequence tasks since these tasks share the same background knowledge about the language.
However, this assumption does not hold in the recommendation tasks.
Thus we train BERT4Rec end-to-end for different sequential recommendation datasets.
b
)
Different from BERT, we remove the next sentence loss and segment embeddings since BERT4Rec models a user’s historical behaviors as only one sequence in sequential recommendation task.
4.
Experiments
4.1.
Datasets
Table 1
.
Statistics of datasets.
Datasets
#users
#items
#actions
Avg. length
Density
Beauty
40,226
\mathrm{40{,}226}
54,542
\mathrm{54{,}542}
0.35m
8.8
0.02%
Steam
281,428
\mathrm{281{,}428}
13,044
\mathrm{13{,}044}
3.5m
12.4
0.10%
ML-1m
6040
\mathrm{6040}
3416
\mathrm{3416}
1.0m
163.5
4.79%
ML-20m
138,493
\mathrm{138{,}493}
26,744
\mathrm{26{,}744}
20m
144.4
0.54%
We evaluate the proposed model on four real-world representative datasets which vary significantly in domains and sparsity.
•
Amazon
Beauty
3
3
3
http://jmcauley.ucsd.edu/data/amazon/
: This is a series of product review datasets crawled from Amazon.com by
McAuley et al. 2015
(
McAuley et al. 2015
)
.
They split the data into separate datasets according to the top-level product categories on Amazon.
In this work, we adopt the “Beauty” category.
•
Steam
4
4
4
https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
:
This is a dataset collected from Steam, a large online video game distribution platform, by
Kang and McAuley [n. d.]
(
Kang and McAuley [n. d.]
)
.
•
MovieLens
(
Harper and
Konstan 2015
)
: This is a popular benchmark dataset for evaluating recommendation algorithms.
In this work, we adopt two well-established versions, MovieLens 1m (
ML-1m
)
5
5
5
https://grouplens.org/datasets/movielens/1m/
and MovieLens 20m (
ML-20m
)
6
6
6
https://grouplens.org/datasets/movielens/20m/
.
For dataset preprocessing, we follow the common practice in
(
Rendle et al. 2010
;
Tang and Wang 2018
;
Kang and McAuley [n. d.]
)
.
For all datasets, we convert all numeric ratings or the presence of a review to implicit feedback of 1 (
i.e.
, the user interacted with the item).
After that, we group the interaction records by users and build the interaction sequence for each user by sorting these interaction records according to the timestamps.
To ensure the quality of the dataset, following the common practice
(
Rendle et al. 2010
;
He
et al. 2017b
;
Tang and Wang 2018
;
Kang and McAuley [n. d.]
)
, we keep users with at least five feedbacks.
The statistics of the processed datasets are summarized in Table
1
.
4.2.
Task Settings & Evaluation Metrics
To evaluate the sequential recommendation models, we adopted the
leave-one-out
evaluation (
i.e.
, next item recommendation) task, which has been widely used in
(
He
et al. 2017b
;
Tang and Wang 2018
;
Kang and McAuley [n. d.]
)
.
For each user, we hold out the last item of the behavior sequence as the test data, treat the item just before the last as the validation set, and utilize the remaining items for training.
For easy and fair evaluation, we follow the common strategy in
(
He
et al. 2017b
;
Tang and Wang 2018
;
Kang and McAuley [n. d.]
)
, pairing each ground truth item in the test set with 100 randomly sampled
negative
items that the user has not interacted with.
To make the sampling reliable and representative
(
Huang
et al. 2018
)
, these 100 negative items are sampled according to their popularity.
Hence, the task becomes to rank these negative items with the ground truth item for each user.
Evaluation Metrics
.
To evaluate the ranking list of all the models, we employ a variety of evaluation metrics, including
Hit Ratio
(HR),
Normalized Discounted Cumulative Gain
(NDCG), and
Mean Reciprocal Rank
(MRR).
Considering we only have one ground truth item for each user, HR@
k
k
is equivalent to Recall@
k
k
and proportional to Precision@
k
k
; MRR is equivalent to Mean Average Precision (MAP).
In this work, we report HR and NDCG with
k
=
1
,
5
,
10
k={1,5,10}
.
For all these metrics, the higher the value, the better the performance.
4.3.
Baselines & Implementation Details
To verify the effectiveness of our method, we compare it with the following representative baselines:
•
POP
: It is the simplest baseline that ranks items according to their popularity judged by the number of interactions.
•
BPR-MF
(
Rendle et al. 2009
)
: It optimizes the matrix factorization with implicit feedback using a pairwise ranking loss.
•
NCF
(
He
et al. 2017b
)
: It models user–item interactions with a MLP instead of the inner product in matrix factorization.
•
FPMC
(
Rendle et al. 2010
)
: It captures users’ general taste as well as their sequential behaviors by combing MF with first-order MCs.
•
GRU4Rec
(
Hidasi et al. 2016
)
: It uses GRU with ranking based loss to model user sequences for session based recommendation.
•
GRU4Rec
+
(
Hidasi and
Karatzoglou 2018
)
: It is an improved version of GRU4Rec with a new class of loss functions and sampling strategy.
•
Caser
(
Tang and Wang 2018
)
: It employs CNN in both horizontal and vertical way to model high-order MCs for sequential recommendation.
•
SASRec
(
Kang and McAuley [n. d.]
)
: It uses a left-to-right Transformer language model to capture users’ sequential behaviors, and achieves state-of-the-art performance on sequential recommendation.
For NCF
7
7
7
https://github.com/hexiangnan/neural_collaborative_filtering
, GRU4Rec
8
8
8
https://github.com/hidasib/GRU4Rec
, GRU4Rec
+
8
, Caser
9
9
9
https://github.com/graytowne/caser_pytorch
, and SASRec
10
10
10
https://github.com/kang205/SASRec
, we use code provided by the corresponding authors.
For BPR-MF and FPMC, we implement them using
TensorFlow
.
For common hyper-parameters in all models, we consider the hidden dimension size
d
d
from
{
16
,
32
,
64
,
128
,
256
}
\{16,32,64,128,256\}
, the
ℓ
2
\ell_{2}
regularizer from {1, 0.1, 0.01, 0.001, 0.0001}, and dropout rate from {
0
,
0.1
,
0.2
,
⋯
,
0.9
0,0.1,0.2,\cdots,0.9
}.
All other hyper-parameters (
e.g.
, Markov order in Caser) and initialization strategies are either followed the suggestion from the methods’ authors or tuned on the validation sets.
We report the results of each baseline under its optimal hyper-parameter settings.
We implement BERT4Rec
11
11
11
https://github.com/FeiSun/BERT4Rec
with
TensorFlow
.
All parameters are initialized using truncated normal distribution in the range
[
−
0.02
,
0.02
]
[-0.02,0.02]
.
We train the model using Adam
(
Kingma and Ba 2015
)
with learning rate of 1e-4,
β
1
=
0.9
,
β
2
=
0.999
\beta_{1}=0.9,\beta_{2}=0.999
,
ℓ
2
\ell_{2}
weight decay of 0.01, and linear decay of the learning rate.
The gradient is clipped when its
ℓ
2
\ell_{2}
norm exceeds a threshold of 5. For fair comparison, we set the layer number
L
=
2
L=2
and head number
h
=
2
h=2
and use the same maximum sequence length as in
(
Kang and McAuley [n. d.]
)
,
N
=
200
N=200
for ML-1m and ML-20m,
N
=
50
N=50
for Beauty and Steam datasets. For head setting, we empirically set the dimensionality of each head as 32 (single head if
d
<
32
d<32
).
We tune the mask proportion
ρ
\rho
using the validation set, resulting in
ρ
=
0.6
\rho=0.6
for Beauty,
ρ
=
0.4
\rho=0.4
for Steam, and
ρ
=
0.2
\rho=0.2
for ML-1m and ML-20m. All the models are trained from scratch without any pre-training on a single NVIDIA GeForce GTX 1080 Ti GPU with a batch size of 256.
4.4.
Overall Performance Comparison
Table 2
.
Performance comparison of different methods on next-item prediction. Bold scores are the best in each row, while underlined scores are the second best. Improvements over baselines are statistically significant with
p
<
0.01
p<0.01
.
Datasets
Metric
POP
BPR-MF
NCF
FPMC
GRU4Rec
GRU4Rec
+
Caser
SASRec
BERT4Rec
Improv.
Beauty
HR@1
0.0077
0.0415
0.0407
0.0435
0.0402
0.0551
0.0475
0.0906
0.0953
5.19%
HR@5
0.0392
0.1209
0.1305
0.1387
0.1315
0.1781
0.1625
0.1934
0.2207
14.12%
HR@10
0.0762
0.1992
0.2142
0.2401
0.2343
0.2654
0.2590
0.2653
0.3025
14.02%
NDCG@5
0.0230
0.0814
0.0855
0.0902
0.0812
0.1172
0.1050
0.1436
0.1599
11.35%
NDCG@10
0.0349
0.1064
0.1124
0.1211
0.1074
0.1453
0.1360
0.1633
0.1862
14.02%
MRR
0.0437
0.1006
0.1043
0.1056
0.1023
0.1299
0.1205
0.1536
0.1701
10.74%
Steam
HR@1
0.0159
0.0314
0.0246
0.0358
0.0574
0.0812
0.0495
0.0885
0.0957
8.14%
HR@5
0.0805
0.1177
0.1203
0.1517
0.2171
0.2391
0.1766
0.2559
0.2710
5.90%
HR@10
0.1389
0.1993
0.2169
0.2551
0.3313
0.3594
0.2870
0.3783
0.4013
6.08%
NDCG@5
0.0477
0.0744
0.0717
0.0945
0.1370
0.1613
0.1131
0.1727
0.1842
6.66%
NDCG@10
0.0665
0.1005
0.1026
0.1283
0.1802
0.2053
0.1484
0.2147
0.2261
5.31%
MRR
0.0669
0.0942
0.0932
0.1139
0.1420
0.1757
0.1305
0.1874
0.1949
4.00%
ML-1m
HR@1
0.0141
0.0914
0.0397
0.1386
0.1583
0.2092
0.2194
0.2351
0.2863
21.78%
HR@5
0.0715
0.2866
0.1932
0.4297
0.4673
0.5103
0.5353
0.5434
0.5876
8.13%
HR@10
0.1358
0.4301
0.3477
0.5946
0.6207
0.6351
0.6692
0.6629
0.6970
4.15%
NDCG@5
0.0416
0.1903
0.1146
0.2885
0.3196
0.3705
0.3832
0.3980
0.4454
11.91%
NDCG@10
0.0621
0.2365
0.1640
0.3439
0.3627
0.4064
0.4268
0.4368
0.4818
10.32%
MRR
0.0627
0.2009
0.1358
0.2891
0.3041
0.3462
0.3648
0.3790
0.4254
12.24%
ML-20m
HR@1
0.0221
0.0553
0.0231
0.1079
0.1459
0.2021
0.1232
0.2544
0.3440
35.22%
HR@5
0.0805
0.2128
0.1358
0.3601
0.4657
0.5118
0.3804
0.5727
0.6323
10.41%
HR@10
0.1378
0.3538
0.2922
0.5201
0.5844
0.6524
0.5427
0.7136
0.7473
4.72%
NDCG@5
0.0511
0.1332
0.0771
0.2239
0.3090
0.3630
0.2538
0.4208
0.4967
18.04%
NDCG@10
0.0695
0.1786
0.1271
0.2895
0.3637
0.4087
0.3062
0.4665
0.5340
14.47%
MRR
0.0709
0.1503
0.1072
0.2273
0.2967
0.3476
0.2529
0.4026
0.4785
18.85%
Table
2
summarized the best results of all models on four benchmark datasets.
The last column is the improvements of BERT4Rec relative to the best baseline.
We omit the NDCG@1 results since it is equal to HR@1 in our experiments.
It can be observed that:
The non-personalized POP method gives the worst performance
12
12
12
What needs to be pointed out is that such low scores for POP are because the negative samples are sampled according to the items’ popularity.
on all datasets since it does not model user’s personalized preference using the historical records.
Among all the baseline methods, sequential methods (
e.g.
, FPMC and GRU4Rec
+
) outperforms non-sequential methods (
e.g.
, BPR-MF and NCF) on all datasets consistently.
Compared with BPR-MF, the main improvement of FPMC is that it models users’ historical records in a sequential way.
This observation verifies that considering sequential information is beneficial for improving performances in recommendation systems.
Among sequential recommendation baselines, Caser outperforms FPMC on all datasets especially for the dense dataset ML-1m, suggesting that high-order MCs is beneficial for sequential recommendation.
However, high-order MCs usually use very small order
L
L
since they do not scale well with the order
L
L
.
This causes Caser to perform worse than GRU4Rec
+
and SASRec, especially on sparse datasets.
Furthermore, SASRec performs distinctly better than GRU4Rec and GRU4Rec
+
, suggesting that self-attention mechanism is a more powerful tool for sequential recommendation.
According to the results, it is obvious that BERT4Rec performs best among all methods on four datasets in terms of all evaluation metrics.
It gains
7.24%
HR@10,
11.03%
NDCG@10, and
11.46%
MRR improvements (on average) against the strongest baselines.
Question 1
:
Do the gains come from the bidirectional self-attention model or from the Cloze objective?
To answer this question, we try to isolate the effects of these two factors by constraining the Cloze task to mask only one item at a time.
In this way, the main difference between our BERT4Rec (with 1 mask) and SASRec is that BERT4Rec predicts the target item jointly conditioning on both left and right context.
We report the results on Beauty and ML-1m with
d
=
256
d=256
in Table
3
due to the space limitation.
The results show that BERT4Rec with 1 mask significantly outperforms SASRec on all metrics.
It demonstrates the importance of bidirectional representations for sequential recommendation.
Besides, the last two rows indicate that the Cloze objective also improves the performances.
Detailed analysis of the mask proportion
ρ
\rho
in Cloze task can be found in §
4.6
Table 3
.
Analysis on bidirection and Cloze with
d
=
256
d=256
.
Model
Beauty
ML-1m
HR@10
NDCG@10
MRR
HR@10
NDCG@10
MRR
SASRec
0.2653
0.1633
0.1536
0.6629
0.4368
0.3790
BERT4Rec (1 mask)
0.2940
0.1769
0.1618
0.6869
0.4696
0.4127
BERT4Rec
0.3025
0.1862
0.1701
0.6970
0.4818
0.4254
Question 2
:
Why and how does bidirectional model outperform unidirectional models?
(a)
Layer 1, head 1
(b)
Layer 1, head 2
(c)
Layer 2, head 2
(d)
Layer 2, head 4
Figure 2
.
Heat-maps of average attention weights on Beauty, the last position “9” denotes “
[mask]
” (best viewed in color).
To answer this question, we try to reveal meaningful patterns by visualizing the average attention weights of the last 10 items during the test on Beauty in Figure
2
.
Due to the space limitation, we only report four representative attention heat-maps in different layers and heads.
16
32
64
128
256
0.1
0.1
0.15
0.15
Dimensionality
NDCG@10
Beauty
16
32
64
128
256
0.15
0.15
0.2
0.2
Dimensionality
NDCG@10
Steam
16
32
64
128
256
0.3
0.3
0.4
0.4
0.5
0.5
Dimensionality
NDCG@10
ML-1m
16
32
64
128
256
0.2
0.2
0.3
0.3
0.4
0.4
0.5
0.5
Dimensionality
NDCG@10
ML-20m
16
32
64
128
256
0.15
0.15
0.2
0.2
0.25
0.25
0.3
0.3
Dimensionality
HR@10
Beauty
16
32
64
128
256
0.25
0.25
0.3
0.3
0.35
0.35
0.4
0.4
Dimensionality
HR@10
Steam
16
32
64
128
256
0.5
0.5
0.6
0.6
0.7
0.7
Dimensionality
HR@10
ML-1m
16
32
64
128
256
0.4
0.4
0.5
0.5
0.6
0.6
0.7
0.7
Dimensionality
HR@10
ML-20m
4
Figure 3
.
Effect of the hidden dimensionality
d
d
on HR@10 and NDCG@10 for neural sequential models.
We make several observations from the results.
a
)
Attention varies across different heads.
For example, in layer 1, head 1 tends to attend on items at the left side while head 2 prefers to attend on items on the right side.
b
)
Attention varies across different layers.
Apparently, attentions in layer 2 tend to focus on more recent items.
This is because layer 2 is directly connected to the output layer and the recent items play a more important role in predicting the future.
Another interesting pattern is that heads in Figure
2(a)
and
2(b)
also tend to attend on
[mask]
13
13
13
This phenomenon also exists in text sequence modeling using BERT.
.
It may be a way for self-attention to propagate sequence-level state to the item level.
c
)
Finally and most importantly, unlike unidirectional model can only attend on items at the left side, items in BERT4Rec tend to attend on the items at both sides.
This indicates that bidirectional is essential and beneficial for user behavior sequence modeling.
In the following studies, we examine the impact of the hyper-parameters, including the hidden dimensionality
d
d
, the mask proportion
ρ
\rho
, and the maximum sequence length
N
N
.
We analyze one hyper-parameter at a time by fixing the remaining hyper-parameters at their optimal settings.
Due to space limitation, we only report NDCG@10 and HR@10 for the follow-up experiments.
4.5.
Impact of Hidden Dimensionality
d
d
We now study how the hidden dimensionality
d
d
affects the recommendation performance.
Figure
3
shows NDCG@10 and HR@10 for neural sequential methods with the hidden dimensionality
d
d
varying from 16 to 256 while keeping other optimal hyper-parameters unchanged.
We make some observations from this figure.
The most obvious observation from these sub-figures is that the performance of each model tends to converge as the dimensionality increases.
A larger hidden dimensionality does not necessarily lead to better model performance, especially on sparse datasets like Beauty and Steam.
This is probably caused by overfitting.
In terms of details, Caser performs unstably on four datasets, which might limit its usefulness.
Self-attention based methods (
i.e.
, SASRec and BERT4Rec) achieve superior performances on all datasets. Finally, our model consistently outperforms all other baselines on all datasets even with a relatively small hidden dimensionality.
Considering that our model achieves satisfactory performance with
d
≥
64
d{\geq}64
, we only report the results with
d
=
64
d{=}64
in the following analysis.
4.6.
Impact of Mask Proportion
ρ
\rho
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
0.2
0.2
0.3
0.3
0.4
0.4
0.5
0.5
0.6
0.6
0.7
0.7
Dimensionality
HR@10
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
0.1
0.1
0.2
0.2
0.3
0.3
0.4
0.4
0.5
0.5
Dimensionality
NDCG@10
4
Figure 4
.
Performance with different mask proportion
ρ
\rho
on
d
=
64
d=64
. Bold symbols denote the best scores in each line.
As described in §
3.6
, mask proportion
ρ
\rho
is a key factor in model training, which directly affects the loss function (Equation
8
).
Obviously, mask proportion
ρ
\rho
should not be too small or it is not enough to learn a strong model.
Meanwhile, it should not be too large, otherwise, it would be hard to train since there are too many items to guess based on a few contexts in such case.
To examine this, we study how mask proportion
ρ
\rho
affects the recommendation performances on different datasets.
Figure
4
shows the results with varying mask proportion
ρ
\rho
from 0.1 to 0.9.
Considering the results with
ρ
>
0.6
\rho>0.6
on all datasets, a general pattern emerges, the performances decreasing as
ρ
\rho
increases.
From the results of the first two columns, it is easy to see that
ρ
=
0.2
\rho=0.2
performs better than
ρ
=
0.1
\rho=0.1
on all datasets.
These results verify what we claimed above.
In addition, we observe that the optimal
ρ
\rho
is highly dependent on the sequence length of the dataset.
For the datasets with short sequence length (
e.g.
, Beauty and Steam), the best performances are achieved at
ρ
=
0.6
\rho{=}0.6
(Beauty) and
ρ
=
0.4
\rho{=}0.4
(Steam), while the datasets with long sequence length (
e.g.
, ML-1m and ML-20m) prefer a small
ρ
=
0.2
\rho{=}0.2
.
This is reasonable since, compared with short sequence datasets, a large
ρ
\rho
in long sequence datasets means much more items that need to be predicted. Take ML-1m and Beauty as example,
ρ
=
0.6
\rho{=}0.6
means we need to predict
98
=
⌊
163.5
×
0.6
⌋
98{=}\lfloor 163.5{\times}0.6\rfloor
items on average per sequence for ML-1m, while it is only
5
=
⌊
8.8
×
0.6
⌋
5{=}\lfloor 8.8{\times}0.6\rfloor
items for Beauty.
The former is too hard for model training.
4.7.
Impact of Maximum Sequence Length
N
N
Table 4
.
Performance with different maximum length
N
N
.
10
20
30
40
50
Beauty
#samples/s
5504
3256
2284
1776
1441
HR@10
0.3006
0.3061
0.3057
0.3054
0.3047
NDCG@10
0.1826
0.1875
0.1837
0.1833
0.1832
10
50
100
200
400
ML-1m
#samples/s
14255
8890
5711
2918
1213
HR@10
0.6788
0.6854
0.6947
0.6955
0.6898
NDCG@10
0.4631
0.4743
0.4758
0.4759
0.4715
We also investigate the effect of the maximum sequence length
N
N
on model’s recommendation performances and efficiency.
Table
4
shows recommendation performances and training speed with different maximum length
N
N
on Beauty and ML-1m.
We observe that the proper maximum length
N
N
is also highly dependent on the average sequence length of the dataset.
Beauty prefers a smaller
N
=
20
N=20
, while ML-1m achieves the best performances on
N
=
200
N=200
.
This indicates that a user’s behavior is affected by more recent items on short sequence datasets and less recent items for long sequence datasets.
The model does not consistently benefit from a larger
N
N
since a larger
N
N
tends to introduce both extra information and more noise.
However, our model performs very stably as the length
N
N
becomes larger.
This indicates that our model can attend to the informative items from the noisy historical records.
A scalability concern about BERT4Rec is that its computational complexity per layer is
𝒪
⁡
(
n
2
​
d
)
\mathcal{O}(n^{2}d)
, quadratic with the length
n
n
.
Fortunately, the results in Table
4
shows that the self-attention layer can be effectively parallelized using GPUs.
4.8.
Ablation Study
Finally, we perform ablation experiments over a number of key components of BERT4Rec in order to better understand their impacts, including positional embedding (PE), position-wise feed-forward network (PFFN), layer normalization (LN), residual connection (RC), dropout, the layer number
L
L
of self-attention, and the number of heads
h
h
in multi-head attention. Table
5
shows the results of our default version (
L
=
2
,
h
=
2
L=2,h=2
) and its eleven variants on all four datasets with dimensionality
d
=
64
d=64
while keeping other hyper-parameters (
e.g.
,
ρ
\rho
) at their optimal settings.
We introduce the variants and analyze their effects respectively:
(1)
PE
. The results show that removing positional embeddings causes BERT4Rec’s performances decreasing dramatically on long sequence datasets (
i.e.
, ML-1m and ML-20m). Without the positional embeddings, the hidden representation
𝑯
i
L
\bm{H}_{i}^{L}
for each item
v
i
v_{i}
depends only on item embeddings.
In this situation, we predict different target items using the same hidden representation of “
[mask]
”. This makes the model ill-posed. This issue is more serious on long sequence datasets since they have more masked items to predict.
(2)
PFFN
. The results show that long sequence datasets (
e.g.
, ML-20m) benefit more from PFFN.
This is reasonable since a purpose of PFFN is to integrate information from many heads which are preferred by long sequence datasets as discussed in the analysis about head number
h
h
in ablation study (
5
).
(3)
LN
,
RC
, and
Dropout
. These components are introduced mainly to alleviate overfitting.
Obviously, they are more effective on small datasets like Beauty.
To verify their effectiveness on large datasets, we conduct an experiment on ML-20m with layer
L
=
4
L{=}4
.
The results show that NDCG@10 decreases about 10% w/o RC.
(4)
Number of layers
L
L
. The results show that stacking Transformer layer can boost performances especially on large datasets (
e.g
, ML-20m).
This verifies that it is helpful to learn more complex item transition patterns via deep self-attention architecture.
The decline in Beauty with
L
=
4
L=4
is largely due to overfitting.
(5)
Head number
h
h
. We observe that long sequence datasets (
e.g.
, ML-20m) benefit from a larger
h
h
while short sequence datasets (
e.g.
, Beauty) prefer a smaller
h
h
. This phenomenon is consistent with the empirical result in
(
Tang
et al. 2018
)
that large
h
h
is essential for capturing long distance dependencies with multi-head self-attention.
Table 5
.
Ablation analysis (NDCG@10) on four datasets. Bold score indicates performance better than the default version, while
↓
\downarrow
indicates performance drop more than 10%.
Architecture
Dataset
Beauty
Steam
ML-1m
ML-20m
L
=
2
L=2
,
h
=
2
h=2
0.1832
0.2241
0.4759
0.4513
w/o PE
0.1741
0.2060
0.2155
↓
\downarrow
0.2867
↓
\downarrow
w/o PFFN
0.1803
0.2137
0.4544
0.4296
w/o LN
0.1642
↓
\downarrow
0.2058
0.4334
0.4186
w/o RC
0.1619
↓
\downarrow
0.2193
0.4643
0.4483
w/o Dropout
0.1658
0.2185
0.4553
0.4471
1 layer (
L
=
1
L=1
)
0.1782
0.2122
0.4412
0.4238
3 layers  (
L
=
3
L=3
)
0.1859
0.2262
0.4864
0.4661
4 layers  (
L
=
4
L=4
)
0.1834
0.2279
0.4898
0.4732
1 head  (
h
=
1
h=1
)
0.1853
0.2187
0.4568
0.4402
4 heads  (
h
=
4
h=4
)
0.1830
0.2245
0.4770
0.4520
8 heads  (
h
=
8
h=8
)
0.1823
0.2248
0.4743
0.4550
5.
Conclusion and Future Work
Deep bidirectional self-attention architecture has achieved tremendous success in language understanding.
In this paper, we introduce a deep bidirectional sequential model called BERT4Rec for sequential recommendation.
For model training, we introduce the Cloze task which predicts the masked items using both left and right context.
Extensive experimental results on four real-world datasets show that our model outperforms state-of-the-art baselines.
Several directions remain to be explored.
A valuable direction is to incorporate rich item features (
e.g.
, category and price for products, cast for movies) into BERT4Rec instead of just modeling item ids. Another interesting direction for the future work would be introducing user component into the model for explicit user modeling when the users have multiple sessions.
References
(1)
Ba
et al
.
(2016)
Lei Jimmy Ba, Ryan Kiros,
and Geoffrey E. Hinton. 2016.
Layer Normalization.
CoRR
abs/1607.06450
(2016).
Bahdanau
et al
.
(2015)
Dzmitry Bahdanau,
Kyunghyun Cho, and Yoshua Bengio.
2015.
Neural Machine Translation by Jointly Learning to
Align and Translate. In
Proceedings of ICLR
.
Chen
et al
.
(2018)
Xu Chen, Hongteng Xu,
Yongfeng Zhang, Jiaxi Tang,
Yixin Cao, Zheng Qin, and
Hongyuan Zha. 2018.
Sequential Recommendation with User Memory
Networks. In
Proceedings of WSDM
.
ACM, New York, NY, USA,
108–116.
Cho et al
.
(2014)
Kyunghyun Cho, Bart van
Merrienboer, Caglar Gulcehre, Dzmitry
Bahdanau, Fethi Bougares, Holger
Schwenk, and Yoshua Bengio.
2014.
Learning Phrase Representations using RNN
Encoder–Decoder for Statistical Machine Translation. In
Proceedings of EMNLP
.
Association for Computational Linguistics,
1724–1734.
Covington
et al
.
(2016)
Paul Covington, Jay
Adams, and Emre Sargin.
2016.
Deep Neural Networks for YouTube Recommendations.
In
Proceedings of RecSys
.
ACM, New York, NY, USA,
191–198.
Devlin
et al
.
(2018)
Jacob Devlin, Ming-Wei
Chang, Kenton Lee, and Kristina
Toutanova. 2018.
BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding.
CoRR
abs/1810.04805
(2018).
Donkers
et al
.
(2017)
Tim Donkers, Benedikt
Loepp, and Jürgen Ziegler.
2017.
Sequential User-based Recurrent Neural Network
Recommendations. In
Proceedings of RecSys
.
ACM, New York, NY, USA,
152–160.
Harper and
Konstan (2015)
F. Maxwell Harper and
Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Trans. Interact. Intell. Syst.
5, 4, Article 19
(Dec. 2015), 19 pages.
He
et al
.
(2016)
Kaiming He, Xiangyu
Zhang, Shaoqing Ren, and Jian Sun.
2016.
Deep Residual Learning for Image Recognition. In
Proceedings of CVPR
. 770–778.
He
et al
.
(2017a)
Ruining He, Wang-Cheng
Kang, and Julian McAuley.
2017a.
Translation-based Recommendation. In
Proceedings of RecSys
. ACM,
New York, NY, USA, 161–169.
He and McAuley (2016)
Ruining He and Julian
McAuley. 2016.
Fusing Similarity Models with Markov Chains for
Sparse Sequential Recommendation. In
Proceedings
of ICDM
. 191–200.
He
et al
.
(2017b)
Xiangnan He, Lizi Liao,
Hanwang Zhang, Liqiang Nie,
Xia Hu, and Tat-Seng Chua.
2017b.
Neural Collaborative Filtering. In
Proceedings of WWW
. 173–182.
Hendrycks and
Gimpel (2016)
Dan Hendrycks and Kevin
Gimpel. 2016.
Bridging Nonlinearities and Stochastic Regularizers
with Gaussian Error Linear Units.
CoRR
abs/1606.08415
(2016).
Hidasi and
Karatzoglou (2018)
Balázs Hidasi and
Alexandros Karatzoglou. 2018.
Recurrent Neural Networks with Top-k Gains for
Session-based Recommendations. In
Proceedings of
CIKM
. ACM, New York, NY, USA,
843–852.
Hidasi et al
.
(2016)
Balázs Hidasi,
Alexandros Karatzoglou, Linas Baltrunas,
and Domonkos Tikk. 2016.
Session-based Recommendations with Recurrent Neural
Networks. In
Proceedings of ICLR
.
Hinton
et al
.
(2015)
Geoffrey Hinton, Oriol
Vinyals, and Jeff Dean.
2015.
Distilling the knowledge in a neural network. In
Deep Learning and Representation Learning
Workshop
.
Hochreiter and
Schmidhuber (1997)
Sepp Hochreiter and
Jürgen Schmidhuber. 1997.
Long Short-Term Memory.
Neural Computation
9,
8 (Nov. 1997),
1735–1780.
Hu
et al
.
(2017)
Liang Hu, Longbing Cao,
Shoujin Wang, Guandong Xu,
Jian Cao, and Zhiping Gu.
2017.
Diversifying Personalized Recommendation with
User-session Context. In
Proceedings of IJCAI
.
1858–1864.
Huang
et al
.
(2018)
Jin Huang, Wayne Xin
Zhao, Hongjian Dou, Ji-Rong Wen, and
Edward Y. Chang. 2018.
Improving Sequential Recommendation with
Knowledge-Enhanced Memory Networks. In
Proceedings
of SIGIR
. ACM, New York, NY, USA,
505–514.
Kabbur
et al
.
(2013)
Santosh Kabbur, Xia Ning,
and George Karypis. 2013.
FISM: Factored Item Similarity Models for top-N
Recommender Systems. In
Proceedings of KDD
.
ACM, New York, NY, USA,
659–667.
Kang
et al
.
(2017)
Wang-Cheng Kang, Chen
Fang, Zhaowen Wang, and Julian
McAuley. 2017.
Visually-Aware Fashion Recommendation and Design
with Generative Image Models. In
Proceedings of
ICDM
. IEEE Computer Society,
207–216.
Kang and McAuley ([n. d.])
Wang-Cheng Kang and
Julian McAuley. [n. d.].
Self-Attentive Sequential Recommendation. In
Proceedings of ICDM
. 197–206.
Kim
et al
.
(2016)
Donghyun Kim, Chanyoung
Park, Jinoh Oh, Sungyoung Lee, and
Hwanjo Yu. 2016.
Convolutional Matrix Factorization for Document
Context-Aware Recommendation. In
Proceedings of
RecSys
. ACM, New York, NY, USA,
233–240.
Kingma and Ba (2015)
Diederik P. Kingma and
Jimmy Ba. 2015.
Adam: A Method for Stochastic Optimization. In
Proceedings of ICLR
.
Koren (2008)
Yehuda Koren.
2008.
Factorization Meets the Neighborhood: A
Multifaceted Collaborative Filtering Model. In
Proceedings of KDD
. ACM,
426–434.
Koren and Bell (2011)
Yehuda Koren and Robert
Bell. 2011.
Advances in Collaborative Filtering.
In
Recommender Systems Handbook
.
Springer US, Boston, MA,
145–186.
Koren
et al
.
(2009)
Yehuda Koren, Robert
Bell, and Chris Volinsky.
2009.
Matrix Factorization Techniques for Recommender
Systems.
Computer
42,
8 (Aug. 2009),
30–37.
Li
et al
.
(2017)
Jing Li, Pengjie Ren,
Zhumin Chen, Zhaochun Ren,
Tao Lian, and Jun Ma.
2017.
Neural Attentive Session-based Recommendation. In
Proceedings of CIKM
. ACM,
New York, NY, USA, 1419–1428.
Li
et al
.
(2018)
Jian Li, Zhaopeng Tu,
Baosong Yang, Michael R. Lyu, and
Tong Zhang. 2018.
Multi-Head Attention with Disagreement
Regularization. In
Proceedings of tEMNLP
.
2897–2903.
Lin et al
.
(2017)
Zhouhan Lin, Minwei Feng,
Cícero Nogueira dos Santos, Mo Yu,
Bing Xiang, Bowen Zhou, and
Yoshua Bengio. 2017.
A Structured Self-attentive Sentence Embedding. In
Proceedings of ICLR
.
Linden
et al
.
(2003)
Greg Linden, Brent Smith,
and Jeremy York. 2003.
Amazon.Com Recommendations: Item-to-Item
Collaborative Filtering.
IEEE Internet Computing
7, 1 (Jan.
2003), 76–80.
Liu et al
.
(2018a)
Peter J. Liu, Mohammad
Saleh, Etienne Pot, Ben Goodrich,
Ryan Sepassi, Lukasz Kaiser, and
Noam Shazeer. 2018a.
Generating Wikipedia by Summarizing Long
Sequences. In
Proceedings of ICLR
.
Liu
et al
.
(2018b)
Qiao Liu, Yifu Zeng,
Refuoe Mokhosi, and Haibin Zhang.
2018b.
STAMP: Short-Term Attention/Memory Priority Model
for Session-based Recommendation. In
Proceedings
of KDD
. ACM, New York, NY, USA,
1831–1839.
McAuley et al
.
(2015)
Julian McAuley,
Christopher Targett, Qinfeng Shi, and
Anton van den Hengel. 2015.
Image-Based Recommendations on Styles and
Substitutes. In
Proceedings of SIGIR
.
ACM, New York, NY, USA,
43–52.
Mikolov
et al
.
(2013a)
Tomas Mikolov, Kai Chen,
Greg Corrado, and Jeffrey Dean.
2013a.
Efficient Estimation of Word Representations in
Vector Space.
CoRR
abs/1301.3781
(2013).
Mikolov et al
.
(2013b)
Tomas Mikolov, Ilya
Sutskever, Kai Chen, Greg Corrado, and
Jeffrey Dean. 2013b.
Distributed Representations of Words and Phrases
and Their Compositionality. In
Proceedings of
NIPS
. Curran Associates Inc., USA,
3111–3119.
Quadrana et al
.
(2017)
Massimo Quadrana,
Alexandros Karatzoglou, Balázs
Hidasi, and Paolo Cremonesi.
2017.
Personalizing Session-based Recommendations with
Hierarchical Recurrent Neural Networks. In
Proceedings of RecSys
. ACM,
New York, NY, USA, 130–137.
Radford et al
.
(2018)
Alec Radford, Karthik
Narasimhan, Tim Salimans, and Ilya
Sutskever. 2018.
Improving language understanding by generative
pre-training. In
OpenAI Technical report
.
Rendle et al
.
(2009)
Steffen Rendle, Christoph
Freudenthaler, Zeno Gantner, and Lars
Schmidt-Thieme. 2009.
BPR: Bayesian Personalized Ranking from Implicit
Feedback. In
Proceedings of UAI
.
AUAI Press, Arlington, Virginia,
United States, 452–461.
Rendle et al
.
(2010)
Steffen Rendle, Christoph
Freudenthaler, and Lars Schmidt-Thieme.
2010.
Factorizing Personalized Markov Chains for
Next-basket Recommendation. In
Proceedings of
WWW
. ACM, New York, NY, USA,
811–820.
Salakhutdinov and
Mnih (2007)
Ruslan Salakhutdinov and
Andriy Mnih. 2007.
Probabilistic Matrix Factorization. In
Proceedings of NIPS
. Curran
Associates Inc., USA, 1257–1264.
Salakhutdinov
et al
.
(2007)
Ruslan Salakhutdinov,
Andriy Mnih, and Geoffrey Hinton.
2007.
Restricted Boltzmann Machines for Collaborative
Filtering. In
Proceedings of ICML
.
791–798.
Sarwar
et al
.
(2001)
Badrul Sarwar, George
Karypis, Joseph Konstan, and John
Riedl. 2001.
Item-based Collaborative Filtering Recommendation
Algorithms. In
Proceedings of WWW
.
ACM, New York, NY, USA,
285–295.
Sedhain
et al
.
(2015)
Suvash Sedhain,
Aditya Krishna Menon, Scott Sanner, and
Lexing Xie. 2015.
AutoRec: Autoencoders Meet Collaborative
Filtering. In
Proceedings of WWW
.
ACM, New York, NY, USA,
111–112.
Shani
et al
.
(2005)
Guy Shani, David
Heckerman, and Ronen I. Brafman.
2005.
An MDP-Based Recommender System.
J. Mach. Learn. Res.
6
(Dec. 2005), 1265–1295.
Shaw
et al
.
(2018)
Peter Shaw, Jakob
Uszkoreit, and Ashish Vaswani.
2018.
Self-Attention with Relative Position
Representations. In
Proceedings of NAACL
.
Association for Computational Linguistics,
464–468.
Srivastava et al
.
(2014)
Nitish Srivastava,
Geoffrey Hinton, Alex Krizhevsky,
Ilya Sutskever, and Ruslan
Salakhutdinov. 2014.
Dropout: A Simple Way to Prevent Neural Networks
from Overfitting.
J. Mach. Learn. Res.
15,
1 (Jan. 2014),
1929–1958.
Tang
et al
.
(2018)
Gongbo Tang, Mathias
Müller, Annette Rios, and Rico
Sennrich. 2018.
Why Self-Attention? A Targeted Evaluation of Neural
Machine Translation Architectures. In
Proceedings
of EMNLP
. 4263–4272.
Tang and Wang (2018)
Jiaxi Tang and Ke
Wang. 2018.
Personalized Top-N Sequential Recommendation via
Convolutional Sequence Embedding. In
Proceedings
of WSDM
. 565–573.
Taylor (1953)
Wilson L. Taylor.
1953.
“Cloze Procedure”: A New Tool for Measuring
Readability.
Journalism Bulletin
30,
4 (1953), 415–433.
van den
Oord et al
.
(2013)
Aaron van den Oord, Sander
Dieleman, and Benjamin Schrauwen.
2013.
Deep content-based music recommendation.
In
Proceedings of NIPS
.
2643–2651.
Vaswani et al
.
(2017)
Ashish Vaswani, Noam
Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez,
Łukasz Kaiser, and Illia
Polosukhin. 2017.
Attention is All you Need.
In
NIPS
. Curran
Associates, Inc., 5998–6008.
Wang
et al
.
(2015)
Hao Wang, Naiyan Wang,
and Dit-Yan Yeung. 2015.
Collaborative Deep Learning for Recommender
Systems. In
Proceedings of KDD
.
ACM, New York, NY, USA,
1235–1244.
Wang
et al
.
(2018)
Shoujin Wang, Liang Hu,
Longbing Cao, Xiaoshui Huang,
Defu Lian, and Wei Liu.
2018.
Attention-Based Transactional Context Embedding for
Next-Item Recommendation.
, 2532–2539 pages.
Wang et al
.
(2017)
Suhang Wang, Yilin Wang,
Jiliang Tang, Kai Shu,
Suhas Ranganath, and Huan Liu.
2017.
What Your Images Reveal: Exploiting Visual Contents
for Point-of-Interest Recommendation. In
Proceedings of WWW
. 391–400.
Wu
et al
.
(2017)
Chao-Yuan Wu, Amr Ahmed,
Alex Beutel, Alexander J. Smola, and
How Jing. 2017.
Recurrent Recommender Networks. In
Proceedings of WSDM
. ACM,
New York, NY, USA, 495–503.
Wu
et al
.
(2016)
Yao Wu, Christopher
DuBois, Alice X. Zheng, and Martin
Ester. 2016.
Collaborative Denoising Auto-Encoders for Top-N
Recommender Systems. In
Proceedings of WSDM
.
ACM, New York, NY, USA,
153–162.
Yu
et al
.
(2016)
Feng Yu, Qiang Liu,
Shu Wu, Liang Wang, and
Tieniu Tan. 2016.
A Dynamic Recurrent Model for Next Basket
Recommendation. In
Proceedings of SIGIR
.
ACM, New York, NY, USA,
729–732.