# Paper (Zhou et al. 2019)

> Source: `https://arxiv.org/abs/1809.03672`

---

Deep Interest Evolution Network for Click-Through Rate Prediction
Guorui Zhou
Thanks:
Corresponding author is Guorui Zhou.
Na Mou
Thanks:
This author is the one who did the really hard work for Online Testing. The source code is available at https://github.com/mouna99/dien.
Ying Fan
Qi Pi
Weijie Bian
Affiliation:
Chang Zhou, Xiaoqiang Zhu  and Kun Gai
Affiliation:
Alibaba Inc, Beijing, China
Affiliation:
{guorui.xgr, mouna.mn, fanying.fy, piqi.pq, weijie.bwj, ericzhou.zc, xiaoqiang.zxq, jingshi.gk}@alibaba-inc.com
Abstract
Click-through rate (CTR) prediction, whose goal is to estimate the probability of a user clicking on the item, has become one of the core tasks in the advertising system.
For CTR prediction model, it is necessary to capture the latent user interest behind the user behavior data.
Besides, considering the changing of the external environment and the internal cognition, user interest evolves over time dynamically.
There are several CTR prediction methods for interest modeling, while most of them regard the representation of behavior as the interest directly, and lack specially modeling for latent interest behind the concrete behavior. Moreover, little work considers the changing trend of the interest.
In this paper, we propose a novel model, named Deep Interest Evolution Network (DIEN), for CTR prediction. Specifically, we design interest extractor layer to capture temporal interests from history behavior sequence. At this layer,
we introduce an auxiliary loss to supervise interest extracting at each step. As user interests are diverse, especially in the e-commerce system, we propose interest evolving layer to capture interest evolving process that is relative to the target item. At interest evolving layer, attention mechanism is embedded into the sequential structure novelly, and the effects of relative interests are strengthened during interest evolution.
In the experiments on both public and industrial datasets, DIEN significantly outperforms the state-of-the-art solutions. Notably, DIEN has been deployed in the display advertisement system of Taobao, and obtained 20.7% improvement on CTR.
Introduction
Cost per click (CPC) billing is one of the commonest billing forms in the advertising system, where advertisers are charged for each click on their advertisement. In CPC advertising system, the performance of click-through rate (CTR) prediction not only influences the final revenue of whole platform, but also impacts user experience and satisfaction. Modeling CTR prediction has drawn more and more attention from the communities of academia and industry.
In most non-searching e-commerce scenes, users do not express their current intention actively. Designing model to capture user’s interests as well as their dynamics is the key to advance the performance of CTR prediction.
Recently, many CTR models transform from traditional methodologies
[
2001
,
2010
]
to deep CTR models
[
2017
,
2016
,
2018
]
. Most deep CTR models focus on capturing interaction between features from different fields and pay less attention to user interest representation.
Deep Interest Network (DIN)
[
2018c
]
emphasizes that user interests are diverse, it uses attention based model to capture relative interests to target item, and obtains adaptive interest representation.
However, most interest models including DIN regard the behavior as the interest directly, while latent interest is hard to be fully reflected by explicit behavior. Previous methods neglect to dig the true user interest behind behavior. Moreover, user interest keeps evolving, capturing the dynamic of interest is important for interest representation.
Based on all these observations, we propose Deep Interest Evolution Network (DIEN) to improve the performance of CTR prediction.
There are two key modules in DIEN, one is for extracting latent temporal interests from explicit user behaviors,
and the other one is for modeling interest evolving process.
Proper interest representation is the footstone of interest evolving model. At interest extractor layer,
DIEN chooses GRU
[
2014
]
to model the dependency between behaviors.
Following the principle that interest leads to the consecutive behavior directly, we propose auxiliary loss
which uses the next behavior to supervise the learning of current hidden state. We call these hidden states with extra supervision as interest states.
These extra supervision information helps to capture more semantic meaning for interest representation and push hidden states of GRU to represent interests effectively.
Moreover, user interests are diverse, which leads to interest drifting phenomenon: user’s intentions can be very different in adjacent visitings,
and one behavior of a user may depend on the behavior that takes long time ago.
Each interest has its own evolution track.
Meanwhile, the click actions of one user on different target items are effected by different parts of interests.
At interest evolving layer, we model the interest evolving trajectory that is relative to target item.
Based on the interest sequence obtained from interest extractor layer, we design GRU with attentional update gate (AUGRU). Using interest state and target item to compute relevance, AUGRU strengthens relative interests’ influence on interest evolution, while weakens irrelative interests’ effect that results from interest drifting. With the introduction of attentional mechanism into update gate, AUGRU can lead to the specific interest evolving processes for different target items.
The main contributions of DIEN are as following:
•
We focus on interest evolving phenomenon in e-commerce system, and propose a new structure of network to model interest evolving process. The model for interest evolution leads to more expressive interest representation and more precise CTR prediction.
•
Different from taking behaviors as interests directly, we specially design interest extractor layer.
Pointing at the problem that hidden state of GRU is less targeted for interest representation, we propose one auxiliary loss. Auxiliary loss uses consecutive behavior to supervise the learning of hidden state at each step. which makes hidden state expressive enough to represent latent interest.
•
We design interest evolving layer novelly, where GPU with attentional update gate (AUGRU) strengthens the effect from relevant interests to target item and overcomes the inference from interest drifting.
In the experiments on both public and industrial datasets, DIEN significantly outperforms the state-of-the-art solutions. It is notable that DIEN has been deployed in Taobao display advertisement system and obtains significant improvement under various metrics.
Related Work
By virtue of the strong ability of deep learning on feature representation and combination, recent CTR models transform from traditional linear or nonlinear models
[
2001
,
2010
]
to deep models. Most deep models follow the structure of Embedding and Multi-ayer Perceptron (MLP)
[
2018c
]
.
Based on this basic paradigm, more and more models pay attention to the interaction between features: Both Wide & Deep
[
2016
]
and deep FM
[
2017
]
combine low-order and high-order features to improve the power of expression; PNN
[
2016
]
proposes a product layer to capture interactive patterns between interfield categories.
However, these methods can not reflect the interest behind data clearly. DIN
[
2018c
]
introduces the mechanism of attention to activate the historical behaviors w.r.t. given target item locally, and captures the diversity characteristic of user interests successfully. However, DIN is weak in capturing the dependencies between sequential behaviors.
In many application domains, user-item interactions can be recorded over time. A number of recent studies show that this information can be used to build richer individual user
models and discover additional behavioral patterns.
In recommendation system, TDSSM
[
2016
]
jointly optimizes long-term and short-term user interests to improve the recommendation quality; DREAM
[
2016
]
uses the structure of recurrent neural network (RNN) to investigate the dynamic representation of each user and the global sequential behaviors of item-purchase history.
?
(
?
) build visually-aware recommender system which surfaces products that more closely match users’ and communities’ evolving interests.
?
(
?
)
measures users’ similarities based on user’s interest sequence, and improves the performance of collaborative filtering recommendation.
?
(
?
) improves native ads CTR prediction by using large scale event embedding and attentional output of recurrent networks.
ATRank
[
2018a
]
uses attention-based sequential framework to model heterogeneous behaviors.
Compared to sequence-independent approaches, these methods can significantly improve the prediction accuracy.
However, these traditional RNN based models have some problems. On the one hand, most of them regard hidden states of sequential structure as latent interests directly, while these hidden states lack special supervision for interest representation. On the other hand, most existing RNN based models deal with all dependencies between adjacent behaviors successively and equally. As we know, not all user’s behaviors are strictly dependent on each adjacent behavior. Each user has diverse interests, and each interest has its own evolving track. For any target item, these models can only obtain one fixed interest evolving track, so these models can be disturbed by interest drifting.
In order to push hidden states of sequential structure to represent latent interests effectively, extra supervision for hidden states should be introdueced.
DARNN
[
2018
]
uses click-level sequential prediction, which models the click action at each time when each ad is shown to the
user. Besides click action, ranking information can be further introduced. In recommendation system, ranking loss has been widely used for ranking task
[
2009
,
2017
]
. Similar to these ranking losses, we propose an auxiliary loss for interest learning. At each step, the auxiliary loss uses consecutive clicked item against non-clicked item to supervise the learning of interest representation.
For capturing interest evolving process that is related to target item, we need more flexible sequential learning structure. In the area of question answering (QA), DMN+
[
2016
]
uses attention based GRU (AGRU) to push the attention mechanism to be sensitive to both the position and ordering of the inputs facts. In AGRU, the vector of the update gate is replaced by the scalar of attention score simply. This replacement neglects the difference between all dimensions of update gates, which contains rich information transmitted from previous sequence. Inspired by the novel sequential structure used in QA, we propose GRU with attentional gate (AUGRU) to activate relative interests during interest evolving. Different from AGRU, attention score in AUGRU acts on the information computed from update gate. The combination of update gate and attention score pushes the process of evolving more specifically and sensitively.
Deep Interest Evolution Network
In this section, we introduce Deep Interest Evolution Network (DIEN) in detail. First, we review the basic Deep CTR model, named BaseModel. Then we show the overall structure of DIEN and introduce the techniques that are used for capturing interests and modeling interest evolution process.
Review of BaseModel
The BaseModel is introduced from the aspects of feature representation, model structure and loss function.
Feature Representation
In our online display system, we use four categories of feature:
User Profile
,
User Behavior
,
Ad
and
Context
.
It is notable that the ad is also item. For generation, we call the ad as the target item in this paper.
Each category of feature has several fields,
User Profile
’s fields are
gender
,
age
and so on;
The fields of
User Behavior
’s are the list of user
visited goods id
;
Ad
’s fields are
ad_id
,
shop_id
and so on;
Context
’s fields are
time
and so on.
Feature in each field can be encoded into one-hot vector, e.g., the female feature in the category of
User Profile
are encoded as
[
0
,
1
]
[0,1]
. The concat of different fields’ one-hot vectors from
User Profile
,
User Behavior
,
Ad
and
Context
form
𝐱
p
,
𝐱
b
,
𝐱
a
,
𝐱
c
\mathbf{x}_{p},\mathbf{x}_{b},\mathbf{x}_{a},\mathbf{x}_{c}
, respectively. In sequential CTR model, it’s remarkable that each field contains a list of behaviors, and each behavior corresponds to a one-hot vector, which can be represented by
𝐱
b
=
[
𝐛
1
;
𝐛
2
;
⋯
;
𝐛
T
]
∈
ℝ
K
×
T
,
𝐛
t
∈
{
0
,
1
}
K
,
\mathbf{x}_{b}=[\mathbf{b}_{1};\mathbf{b}_{2};\cdots;\mathbf{b}_{T}]\in\mathbb{R}^{K\times T},\mathbf{b}_{t}\in\{0,1\}^{K},
where
𝐛
t
\mathbf{b}_{t}
is encoded as one-hot vector and represents
t
t
-th behavior,
T
T
is the number of user’s history behaviors,
K
K
is the total number of goods that user can click.
The Structure of BaseModel
Most deep CTR models are built on the basic structure of embedding & MLR. The basic structure is composed of several parts:
•
Embedding
Embedding is the common operation that transforms the large scale sparse feature into low-dimensional dense feature.
In the embedding layer, each field of feature is corresponding to one embedding matrix, e.g., the embedding matrix of visited goods can be represented by
E
g
​
o
​
o
​
d
​
s
=
[
𝐦
1
;
𝐦
2
;
⋯
;
𝐦
K
]
∈
ℝ
n
E
×
K
,
\mathrm{E}_{goods}=[\mathbf{m}_{1};\mathbf{m}_{2};\cdots;\mathbf{m}_{K}]\in\mathbb{R}^{n_{E}\times K},
where
𝐦
j
∈
ℝ
n
E
\mathbf{m}_{j}\in\mathbb{R}^{n_{E}}
represents a embedding vector with dimension
n
E
n_{E}
.
Especially, for behavior feature
𝐛
t
\mathbf{b}_{t}
, if
𝐛
t
​
[
j
t
]
=
1
\mathbf{b}_{t}[j_{t}]=1
, then its corresponding embedding vector is
𝐦
j
t
\mathbf{m}_{j_{t}}
, and the ordered embedding vector list of behaviors for one user can be represented by
𝐞
b
=
[
𝐦
j
1
;
𝐦
j
2
;
⋯
,
𝐦
j
T
]
.
\mathbf{e}_{b}=[\mathbf{m}_{j_{1}};\mathbf{m}_{j_{2}};\cdots,\mathbf{m}_{j_{T}}].
Similarly,
𝐞
a
\mathbf{e}_{a}
represents the concatenated embedding vectors of fields in the category of advertisement.
•
Multilayer Perceptron
(MLP) First, the embedding vectors from one category are fed into pooling operation. Then all these pooling vectors from different categories are concatenated. At last, the concatenated vector is fed into the following MLP for final prediction.
Loss Function
The widely used loss function in deep CTR models is negative log-likelihood function, which uses the label of target item to supervise overall prediction:
L
t
​
a
​
r
​
g
​
e
​
t
=
−
1
N
∑
(
𝐱
,
y
)
∈
𝒟
N
(
y
log
p
(
𝐱
)
+
(
1
−
y
)
log
(
1
−
p
(
𝐱
)
)
)
,
L_{target}=-\frac{1}{N}\sum_{(\mathbf{x},y)\in\mathcal{D}}^{N}(y\log p(\mathbf{x})+(1-y)\log(1-p(\mathbf{x}))),
(1)
where
𝐱
=
[
𝐱
p
,
𝐱
a
,
𝐱
c
,
𝐱
b
]
∈
𝒟
\mathbf{x}=[\mathbf{x}_{p},\mathbf{x}_{a},\mathbf{x}_{c},\mathbf{x}_{b}]\in\mathcal{D}
,
𝒟
\mathcal{D}
is the training set of size
N
N
.
y
∈
{
0
,
1
}
y\in\{0,1\}
represents whether the user clicks target item.
p
⁡
(
𝐱
)
p(\mathbf{x})
is the output of network, which is the predicted probability that the user clicks target item.
Deep Interest Evolution Network
Different from sponsored search, in many e-commerce platforms like online display advertisement, users do not show their intention clearly, so capturing user interest and their dynamics is important for CTR prediction. DIEN devotes to capture user interest and models interest evolving process.
As shown in Fig.
1
, DIEN is composed by several parts. First, all categories of features are transformed by embedding layer. Next, DIEN takes two steps to capture interest evolving:
interest extractor layer extracts interest sequence based on behavior sequence; interest evolving layer models interest evolving process that is relative to target item.
Then final interest’s representation and embedding vectors of ad, user profile, context are concatenated. The concatenated vector is fed into MLP for final prediction.
In the remaining of this section, we will introduce two core modules of DIEN in detail.
Figure 1:
The structure of DIEN. At the behavior layer, behaviors are sorted by time, the embedding layer transforms the one-hot representation
𝐛
⁡
[
t
]
\mathbf{b}[t]
to embedding vector
𝐞
⁡
[
t
]
\mathbf{e}[t]
. Then interest extractor layer extracts each interest state
𝐡
⁡
[
t
]
\mathbf{h}[t]
with the help of auxiliary loss. At interest evolving layer, AUGRU models the interest evolving process that is relative to target item. The final interest state
𝐡
′
​
[
T
]
\mathbf{h}^{\prime}[T]
and embedding vectors of remaining feature are concatenated, and fed into MLR for final CTR prediction.
Interest Extractor Layer
In e-commerce system, user behavior is the carrier of latent interest, and interest will change after user takes one behavior. At the interest extractor layer, we extract a series of interest states from sequential user behaviors.
The click behaviors of user in e-commerce system are rich, where the length of history behavior sequence is long even in a short period of time, like two weeks.
For the balance between efficiency and performance, we take GRU to model the dependency between behaviors, where the input of GRU is ordered behaviors by their occur time.
GRU overcomes the vanishing gradients problem of RNN and is faster than LSTM
[
1997
]
, which is suitable for e-commerce system.
The formulations of GRU are listed as follows:
𝐮
t
=
σ
⁡
(
W
u
​
𝐢
t
+
U
u
​
𝐡
t
−
1
+
𝐛
u
)
,
\displaystyle\mathbf{u}_{t}=\sigma(W^{u}\mathbf{i}_{t}+U^{u}\mathbf{h}_{t-1}+\mathbf{b}^{u}),
(2)
𝐫
t
=
σ
⁡
(
W
r
​
𝐢
t
+
U
r
​
𝐡
t
−
1
+
𝐛
r
)
,
\displaystyle\mathbf{r}_{t}=\sigma(W^{r}\mathbf{i}_{t}+U^{r}\mathbf{h}_{t-1}+\mathbf{b}^{r}),
(3)
𝐡
~
t
=
tanh
⁡
(
W
h
​
𝐢
t
+
𝐫
t
∘
U
h
​
𝐡
t
−
1
+
𝐛
h
)
,
\displaystyle\mathbf{\tilde{h}}_{t}=\tanh(W^{h}\mathbf{i}_{t}+\mathbf{r}_{t}\circ U^{h}\mathbf{h}_{t-1}+\mathbf{b}^{h}),
(4)
𝐡
t
=
(
𝟏
−
𝐮
t
)
∘
𝐡
t
−
1
+
𝐮
t
∘
𝐡
~
t
,
\displaystyle\mathbf{h}_{t}=(\mathbf{1}-\mathbf{u}_{t})\circ\mathbf{h}_{t-1}+\mathbf{u}_{t}\circ\mathbf{\tilde{h}}_{t},
(5)
where
σ
\sigma
is the sigmoid activation function,
∘
\circ
is element-wise product,
W
u
,
W
r
,
W
h
∈
ℝ
n
H
×
n
I
W^{u},W^{r},W^{h}\in\mathbb{R}^{n_{H}\times n_{I}}
,
U
z
,
U
r
,
U
h
∈
n
H
×
n
H
U^{z},U^{r},U^{h}\in{n_{H}\times n_{H}}
,
n
H
n_{H}
is the hidden size, and
n
I
n_{I}
is the input size.
𝐢
t
\mathbf{i}_{t}
is the input of GRU,
𝐢
t
=
𝐞
b
​
[
t
]
\mathbf{i}_{t}=\mathbf{e}_{b}[t]
represents the
t
t
-th behavior that the user taken,
𝐡
t
\mathbf{h}_{t}
is the
t
t
-th hidden states.
However, the hidden state
𝐡
t
\mathbf{h}_{t}
which only captures the dependency between behaviors can not represent interest effectively. As the click behavior of target item is triggered by final interest, the label used in
L
t
​
a
​
r
​
g
​
e
​
t
L_{target}
only contains the ground truth that supervises final interest’s prediction, while history state
𝐡
t
​
(
t
<
T
)
\mathbf{h}_{t}~(t<T)
can’t obtain proper supervision. As we all know, interest state at each step leads to consecutive behavior directly.
So we propose auxiliary loss, which uses behavior
𝐛
t
+
1
\mathbf{b}_{t+1}
to supervise the learning of interest state
𝐡
t
\mathbf{h}_{t}
.
Besides using the real next behavior as positive instance, we also use negative instance that samples from item set except the clicked item.
There are
N
N
pairs of behavior embedding sequence:
{
𝐞
b
i
,
𝐞
^
b
i
}
∈
𝒟
ℬ
,
i
∈
1
,
2
,
⋯
,
N
\{\mathbf{e}^{i}_{b},\hat{\mathbf{e}}^{i}_{b}\}\in\mathcal{D_{B}},i\in{1,2,\cdots,N}
, where
𝐞
b
i
∈
ℝ
T
×
n
E
\mathbf{e}^{i}_{b}\in\mathbb{R}^{T\times n_{E}}
represents the clicked behavior sequence, and
𝐞
^
b
i
∈
ℝ
T
×
n
E
\hat{\mathbf{e}}^{i}_{b}\in\mathbb{R}^{T\times n_{E}}
represent the negative sample sequence.
T
T
is the number of history behaviors,
n
E
n_{E}
is the dimension of embedding,
𝐞
b
i
​
[
t
]
∈
𝒢
\mathbf{e}^{i}_{b}[t]\in\mathcal{G}
represents the
t
t
-th item’s embedding vector that user
i
i
click,
𝒢
\mathcal{G}
is the whole item set.
𝐞
^
b
i
​
[
t
]
∈
𝒢
−
𝐞
b
i
​
[
t
]
\hat{\mathbf{e}}^{i}_{b}[t]\in\mathcal{G}-\mathbf{e}^{i}_{b}[t]
represents the embedding of item that samples from the item set except the item clicked by user
i
i
at
t
t
-th step. Auxiliary loss can be formulated as:
L
a
​
u
​
x
=
−
\displaystyle L_{aux}=-
1
N
​
(
∑
i
=
1
N
∑
t
log
⁡
σ
⁡
(
𝐡
t
i
,
𝐞
b
i
​
[
t
+
1
]
)
CLOSE
\displaystyle\frac{1}{N}(\sum_{i=1}^{N}\sum_{t}\log\sigma(\mathbf{h}^{i}_{t},\mathbf{e}^{i}_{b}[t+1])
(6)
OPEN
+
log
⁡
(
1
−
σ
⁡
(
𝐡
t
i
,
𝐞
^
b
i
​
[
t
+
1
]
)
)
)
,
\displaystyle+\log(1-\sigma(\mathbf{h}^{i}_{t},\hat{\mathbf{e}}^{i}_{b}[t+1]))),
where
σ
⁡
(
𝐱
𝟏
,
𝐱
𝟐
)
=
1
1
+
exp
⁡
(
−
[
𝐱
1
,
𝐱
2
]
)
\sigma(\mathbf{x_{1}},\mathbf{x_{2}})=\frac{1}{1+\exp(-[\mathbf{x}_{1},\mathbf{x}_{2}])}
is sigmoid activation function,
𝐡
t
i
\mathbf{h}^{i}_{t}
represents the
t
t
-th hidden state of GRU for user
i
i
.
The global loss we use in our CTR model is:
L
=
L
t
​
a
​
r
​
g
​
e
​
t
+
α
∗
L
a
​
u
​
x
,
L=L_{target}+\alpha*L_{aux},
(7)
where
α
\alpha
is the hyper-parameter which balances the interest representation and CTR prediction.
With the help of auxiliary loss, each hidden state
𝐡
t
\mathbf{h}_{t}
is expressive enough to represent interest state after user takes behavior
𝐢
t
\mathbf{i}_{t}
. The concat of all
T
T
interest points
[
𝐡
1
,
𝐡
2
,
⋯
,
𝐡
T
]
[\mathbf{h}_{1},\mathbf{h}_{2},\cdots,\mathbf{h}_{T}]
composes the interest sequence that interest evolving layer can model interest evolving on.
Overall, the introduction of auxiliary loss has several advantages:
from the aspect of interest learning, the introduction of auxiliary loss helps each hidden state of GRU represent interest expressively.
As for the optimization of GRU, auxiliary loss reduces the difficulty of back propagation when GRU models long history behavior sequence.
Last but not the least, auxiliary loss gives more semantic information for the learning of embedding layer, which leads to a better embedding matrix.
Interest Evolving Layer
As the joint influence from external environment and internal cognition, different kinds of user interests are evolving over time. Using the interest on clothes as an example, with the changing of population trend and user taste, user’s preference for clothes evolves.
The evolving process of the user interest on clothes will directly decides CTR prediction for candidate clothes.
The advantages of modeling the evolving process is as follows:
•
Interest evolving module could supply the representation of final interest with more relative history information;
•
It is better to predict the CTR of target item by following the interest evolution trend.
Notably, interest shows two characteristics during evolving:
•
As the diversity of interests, interest can drift. The effect of interest drifting on behaviors is that user may interest in kinds of books during a period of time, and need clothes in another time.
•
Though interests may affect each other, each interest has its own evolving process, e.g. the evolving process of books and clothes is almost individually. We only concerns the evolving process that is relative to target item.
In the first stage, with the help of auxiliary loss, we has obtained expressive representation of interest sequence.
By analyzing the characteristics of interest evolving,
we combine the local activation ability of attention mechanism and sequential learning ability from GRU to model interest evolving. The local activation during each step of GRU can intensify relative interest’s effect, and weaken the disturbance from interest drifting, which is helpful for modeling interest evolving process that relative to target item.
Similar to the formulations shown in Eq. (
2
-
5
), we use
𝐢
t
′
\mathbf{i}_{t}^{\prime}
,
𝐡
t
′
\mathbf{h}_{t}^{\prime}
to represent the input and hidden state in interest evolving module, where the input of second GRU is the corresponding interest state at Interest Extractor Layer:
𝐢
t
′
=
𝐡
t
\mathbf{i}_{t}^{\prime}=\mathbf{h}_{t}
. The last hidden state
𝐡
T
′
\mathbf{h}_{T}^{\prime}
represents final interest state.
And the attention function we used in interest evolving module can be formulated as:
a
t
=
exp
⁡
(
𝐡
t
​
W
​
𝐞
a
)
∑
j
=
1
T
exp
⁡
(
𝐡
j
​
W
​
𝐞
a
)
,
a_{t}=\frac{\exp(\mathbf{h}_{t}W\mathbf{e}_{a})}{\sum_{j=1}^{T}\exp(\mathbf{h}_{j}W\mathbf{e}_{a})},
(8)
where
𝐞
a
\mathbf{e}_{a}
is the concat of embedding vectors from fields in category ad,
W
∈
ℝ
n
H
×
n
A
W\in\mathbb{R}^{n_{H}\times n_{A}}
,
n
H
n_{H}
is the dimension of hidden state and
n
A
n_{A}
is the dimension of advertisement’s embedding vector. Attention score can reflect the relationship between advertisement
𝐞
a
\mathbf{e}_{a}
and input
𝐡
t
\mathbf{h}_{t}
, and strong relativeness leads to a large attention score.
Next, we will introduce several approaches that combine the mechanism of attention and GRU to model the process of interest evolution.
•
GRU with attentional input (AIGRU)
In order to activate relative interests during interest evolution, we propose a naive method, named GRU with attentional input (AIGRU). AIGRU uses attention score to affect the input of interest evolving layer.
As shown in Eq. (
9
):
𝐢
t
′
=
𝐡
t
∗
a
t
\mathbf{i}_{t}^{\prime}=\mathbf{h}_{t}*a_{t}
(9)
Where
𝐡
t
\mathbf{h}_{t}
is the
t
t
-th hidden state of GRU at interest extractor layer,
𝐢
t
′
\mathbf{i}_{t}^{\prime}
is the input of the second GRU which is for interest evolving, and
∗
*
means scalar-vector product.
In AIGRU, the scale of less related interest can be reduced by the attention score. Ideally, the input value of less related interest can be reduced to zero. However, AIGRU works not very well. Because even zero input can also change the hidden state of GRU, so the less relative interests also affect the learning of interest evolving.
•
Attention based GRU(AGRU)
In the area of question answering
[
2016
]
, attention based GRU (AGRU) is firstly proposed. After modifying the GRU architecture by embedding information from the attention mechanism, AGRU can extract key information in complex queries effectively.
Inspired by the question answering system, we transfer the using of AGRU from extracting key information in query to capture relative interest during interest evolving novelly.
In detail, AGRU uses the attention score to replace the update gate of GRU, and changes the hidden state directly. Formally:
𝐡
t
′
=
(
1
−
a
t
)
∗
𝐡
t
−
1
′
+
a
t
∗
𝐡
~
t
′
,
\mathbf{h}_{t}^{\prime}=(1-a_{t})*\mathbf{h}_{t-1}^{\prime}+a_{t}*\tilde{\mathbf{{h}}}_{t}^{\prime},
(10)
where
𝐡
t
′
\mathbf{h}_{t}^{\prime}
,
𝐡
t
−
1
′
\mathbf{h}_{t-1}^{\prime}
and
𝐡
~
t
′
\tilde{\mathbf{h}}_{t}^{\prime}
are the hidden state of AGRU.
In the scene of interest evolving, AGRU makes use of the attention score to control the update of hidden state directly. AGRU weakens the effect from less related interest during interest evolving.
The embedding of attention into GRU improves the influence of attention mechanism, and helps AGRU overcome the defects of AIGRU.
•
GRU with attentional update gate (AUGRU)
Although AGRU can use attention score to control the update of hidden state directly, it use s a scalar (the attention score
a
t
a_{t}
) to replace a vector (the update gate
u
t
u_{t}
), which ignores the difference of importance among different dimensions. We propose the GRU with attentional update gate (
AUGRU
) to combine attention mechanism and GRU seamlessly:
𝐮
~
t
′
\displaystyle\mathbf{\tilde{u}}_{t}^{\prime}
=
a
t
∗
𝐮
t
′
,
\displaystyle=a_{t}*\mathbf{u}_{t}^{\prime},
(11)
𝐡
t
′
\displaystyle\mathbf{h}_{t}^{\prime}
=
(
1
−
𝐮
~
t
′
)
∘
𝐡
t
−
1
′
+
𝐮
~
t
′
∘
𝐡
~
t
′
,
\displaystyle=(1-\mathbf{\tilde{u}}_{t}^{\prime})\circ\mathbf{h}_{t-1}^{\prime}+\mathbf{\tilde{u}}_{t}^{\prime}\circ\mathbf{\tilde{h}}_{t}^{\prime},
(12)
where
𝐮
t
′
\mathbf{u}_{t}^{\prime}
is the original update gate of AUGRU,
𝐮
~
t
′
\mathbf{\tilde{u}}_{t}^{\prime}
is the attentional update gate we design for AUGRU,
𝐡
t
′
,
𝐡
t
−
1
′
\mathbf{h}_{t}^{\prime},\mathbf{h}_{t-1}^{\prime}
, and
𝐡
~
t
′
\tilde{\mathbf{h}}_{t}^{\prime}
are the hidden states of AUGRU.
In AUGRU, we keep original dimensional information of update gate, which decides the importance of each dimension. Based on the differentiated information, we use attention score
a
t
a_{t}
to scale all dimensions of update gate, which results that less related interest make less effects on the hidden state. AUGRU avoids the disturbance from interest drifting more effectively, and pushes the relative interest to evolve smoothly.
Experiments
In this section, we compare DIEN with the state of the art on both public and industrial datasets. Besides, we design experiments to verify the effect of auxiliary loss and AUGRU, respectively. For observing the process of interest evolving, we show the visualization result of interest hidden states. At last, we share the results and techniques for online serving.
Datasets
We use both public and industrial datasets to verify the effect of DIEN. The statistics of all datasets are shown in Table
1
.
Table 1:
The statistics of datasets
Dataset
User
Goods
Categories
Samples
Books
603,668
367,982
1,600
603,668
Electronics
192,403
63,001
801
192,403
Industrial dataset
0.8 billion
0.82 billion
18,006
7.0 billion
public Dataset
Amazon Dataset
[
2015
]
is composed of product reviews and metadata from Amazon.
We use two subsets of Amazon dataset: Books and Electronics, to verify the effect of DIEN.
In these datasets, we regard reviews as behaviors, and sort the reviews from one user by time. Assuming there are
T
T
behaviors of user
u
u
, our purpose is to use the
T
−
1
T-1
behaviors to predict whether user
u
u
will write reviews that shown in
T
T
-th review.
Industrial Dataset
Industrial dataset is constructed by impression and click logs from our online display advertising system.
For training set, we take the ads that clicked at last
49
49
days as the target item. Each target item and its corresponding click behaviors construct one instance. Using one target item
a
a
as example, we set the day that
a
a
is clicked as the last day, the behaviors that this user takes in previous
14
14
days as history behaviors. Similarly, the target item in test set is choose from the following one day, and the behaviors are built as same as training data.
Compared Methods
We compare DIEN with some mainstream CTR prediction methods:
•
BaseModel
BaseModel takes the same setting of embedding and MLR with DIEN, and uses sum pooling operation to integrate behavior embeddings.
•
Wide&Deep
[
2016
]
Wide & Deep consists of two parts: its deep model is the same as Base Model, and its wide model is a linear model.
•
PNN
[
2016
]
PNN uses a product layer to capture interactive patterns between interfield categories.
•
DIN
[
2018c
]
DIN uses the mechanism of attention to activate related user behaviors.
•
Two layer GRU Attention
Similar to
[
2018
]
, we use two layer GRU to model sequential behaviors, and takes an attention layer to active relative behaviors.
Results on Public Datasets
Overall, as shown in Fig.
1
, the structure of DIEN consists GRU, AUGRU and auxiliary loss and other normal components. In public datase. Each experiment is repeated 5 times.
Table 2:
Results (AUC) on public datasets
Model
Electronics (mean
±
\pm
std)
Books (mean
±
\pm
std)
BaseModel
[
2018c
]
0.7435
±
0.00128
0.7435\pm 0.00128
0.7686
±
0.00253
0.7686\pm 0.00253
Wide&Deep
[
2016
]
0.7456
±
0.00127
0.7456\pm 0.00127
0.7735
±
0.00051
0.7735\pm 0.00051
PNN
[
2016
]
0.7543
±
0.00101
0.7543\pm 0.00101
0.7799
±
0.00181
0.7799\pm 0.00181
DIN
[
2018c
]
0.7603
±
0.00028
0.7603\pm 0.00028
0.7880
±
0.00216
0.7880\pm 0.00216
Two layer GRU Attention
0.7605
±
0.00059
0.7605\pm 0.00059
0.7890
±
0.00268
0.7890\pm 0.00268
DIEN
0.7792
±
0.00243
\bm{0.7792\pm 0.00243}
0.8453
±
0.00476
\bm{0.8453\pm 0.00476}
From Table
2
, we can find
Wide & Deep
with manually designed features performs not well, while automative interaction between features (
PNN
) can improve the performance of
BaseModel
.
At the same time, the models aiming to capture interests can improve AUC obviously:
DIN
actives the interests that relative to ad,
Two layer GRU attention
further activates relevant interests in interest sequence, all these explorations obtain positive feedback.
DIEN not only captures sequential interests more effectively, but also models the interest evolving process that is relative to target item. The modeling for interest evolving helps DIEN obtain better interest representation, and capture dynamic of interests precisely, which improves the performance largely.
Results on Industrial Dataset
We further conduct experiments on the dataset of real display advertisement.
There are 6 FCN layers used in industrial dataset, the dimensions area 600, 400, 300, 200, 80, 2, respectively, the max length of history behaviors is set as
50
50
.
As shown in Table
3
,
Wide & Deep
and
PNN
obtain better performance than
BaseModel
.
Different from only one category of goods in Amazon dataset, the dataset of online advertisement contains all kinds of goods at the same time. Based on this characteristic, attention-based methods improve the performance largely, like
DIN
.
DIEN captures the interest evolving process that is relative to target item, and obtains best performance.
Table 3:
Results (AUC) on industrial dataset
Model
AUC
BaseModel
[
2018c
]
0.6350
Wide&Deep
[
2016
]
0.6362
PNN
[
2016
]
0.6353
DIN
[
2018c
]
0.6428
Two layer GRU Attention
0.6457
BaseModel + GRU + AUGRU
0.6493
DIEN
0.6541
\bm{0.6541}
Application Study
In this section, we will show the effect of AUGRU and auxiliary loss, respectively.
Effect of GRU with attentional update gate (AUGRU)
Table 4:
Effect of AUGRU and auxiliary loss (AUC)
Model
Electronics (mean
±
\pm
std)
Books (mean
±
\pm
std)
BaseModel
0.7435
±
0.00128
0.7435\pm 0.00128
0.7686
±
0.00253
0.7686\pm 0.00253
Two layer GRU attention
0.7605
±
0.00059
0.7605\pm 0.00059
0.7890
±
0.00268
0.7890\pm 0.00268
BaseModel + GRU + AIGRU
0.7606
±
0.00061
0.7606\pm 0.00061
0.7892
±
0.00222
0.7892\pm 0.00222
BaseModel + GRU + AGRU
0.7628
±
0.00015
0.7628\pm 0.00015
0.7890
±
0.00268
0.7890\pm 0.00268
BaseModel + GRU + AUGRU
0.7640
±
0.00073
0.7640\pm 0.00073
0.7911
±
0.00150
0.7911\pm 0.00150
DIEN
0.7792
±
0.00243
\bm{0.7792\pm 0.00243}
0.8453
±
0.00476
\bm{0.8453\pm 0.00476}
Table
4
shows the results of different methods for interest evolving.
Compared to
BaseMode
,
Two layer GRU Attention
obtains improvement, while the lack for modeling evolution limits its ability.
AIGRU
takes the basic idea to model evolving process,
though it has advances, the splitting of attention and evolving lost information during interest evolving.
AGRU
further tries to fuse attention and evolution, as we proposed previously, its attention in GRU can’t make fully use the resource of update gate.
AUGRU
obtains obvious improvements, which reflects it fuses the attention mechanism and sequential learning ideally, and captures the evolving process of relative interests effectively.
Effect of auxiliary loss
Based on the model that obtained with AUGRU, we further explore the effect of auxiliary loss.
In the public datasets, the negative instance used in the auxiliary loss is randomly sampled from item set except the item shown in corresponding review.
As for industrial dataset, the ads shown while not been clicked are as negative instances.
(a)
Books
(b)
Electronics
Figure 2:
Learning curves on public datasets.
α
\alpha
is set as
1
1
.
As shown in Fig.
2
, the loss of both whole loss
L
L
and auxiliary loss
L
​
_
​
a
​
u
​
x
L\_aux
keep similar descend trend, which means both global loss for CTR prediction and auxiliary loss for interest representation make effect.
As shown in Table
4
, auxiliary loss bring great improvements for both public datasets, it reflects the importance of supervision information for the learning of sequential interests and embedding representation.
For industrial dataset shown in Table
3
, model with auxiliary loss improves performance further. However, we can see that the improvement is not as obvious as that in public dataset. The difference derives from several aspects.
First, for industrial dataset, it has a large number of instances to learn the embedding layer, which makes it earn less from auxiliary loss. Second, different from all items from one category in amazon dataset, the behaviors in industrial dataset are clicked goods from all scenes and all catagories in our platform. Our goal is to predict CTR for ad in one scene. The supervision information from auxiliary loss may be heterogeneous from the target item, so the effect of auxiliary loss for the industrial dataset may be less for public datasets, while the effect of AUGRU is magnified.
Visualization of Interest Evolution
The dynamic of hidden states in AUGRU can reflect the evolving process of interest. In this section, we visualize these hidden states to explore the effect of different target item for interest evolution.
The selective history behaviors are from category
Computer Speakers, Headphones, Vehicle GPS, SD & SDHC Cards, Micro SD Cards, External Hard Drives, Headphones, Cases
, successively.
The hidden states in AUGRU are projected into a two dimension space by principal component analysis (PCA)
[
1987
]
. The projected hidden states are linked in order. The moving routes of hidden states activated by different target item are shown in Fig.
3
(a). The yellow curve which is with
None
target represents the attention score used in eq. (
12
) are equal, that is the evolution of interest are not effected by target item. The blue curve shows the hidden states are activated by item from category
Screen Protectors
, which is less related to all history behaviors, so it shows similar route to yellow curve. The red curve shows the hidden states are activated by item from category
Cases
, the target item is strong related to the last behavior, which moves a long step shown in Fig.
3
(a). Correspondingly, the last behavior obtains a large attention score showed in Fig.
3
(b).
(a)
Visualization of hidden states in AUGRU
(b)
Attention score of different history behaviors
Figure 3:
Visualization of interest evolution, (a) The hidden states of AUGRU reduced by PCA into two dimensions. Different curves shows the same history behaviors are activated by different target items.
None
means interest evolving is not effected by target item. (b) Faced with different target item, attention scores of all history behaviors are shown.
Online Serving & A/B testing
Table 5:
Results from Online A/B testing
Model
CTR Gain
PPC Gain
eCPM Gain
BaseModel
0%
0%
0%
DIN
[
2018c
]
+ 8.9%
- 2.0%
+ 6.7%
DIEN
+ 20.7%
- 3.0%
+ 17.1%
From 2018-06-07 to 2018-07-12, online A/B testing was conducted in the display advertising system of Taobao. As shown in Table
5
, compared to the BaseModel, DIEN has improved CTR by 20.7% and effective cost per mille (eCPM) by 17.1%. Besides, DIEN has decayed pay per click (PPC) by 3.0%. Now, DIEN has been deployed online and serves the main traffic, which contributes a significant business revenue growth.
It is worth noticing that online serving of DIEN is a great challenge for commercial system.
Online system holds really high traffic in our display advertising system, which serves more than 1 million users per second at traffic peak.
In order to keep low latency and high throughput, we deploy several important techniques to improve serving performance: i)
element parallel GRU & kernel fusion
[
2010
]
, we fuse as many independent kernels as possible. Besides, each element of the hidden state of GRU can be calculated in parallel. ii)
Batching
: adjacent requests from different users are merged into one batch to take advantage of GPU. iii)
Model compressing with Rocket Launching
[
2018b
]
: we use the method proposed in
[
2018b
]
to train a light network, which has smaller size but performs close to the deeper and more complex one. For instance, the dimension of GRU hidden state can be compressed from 108 to 32 with the help of Rocket Launching. With the help of these techniques, latency of DIEN serving can be reduced from 38.2 ms to 6.6 ms and the QPS (Query Per Second) capacity of each worker can be improved to 360.
Conclusion
In this paper, we propose a new structure of deep network, namely Deep Interest Evolution Network (DIEN), to model interest evolving process. DIEN improves the performance of CTR prediction largely in online advertising system.
Specifically, we design interest extractor layer to capture interest sequence particularly, which uses auxiliary loss to provide the interest state with more supervision.
Then we propose interest evolving layer, where DIEN uses GRU with attentional update gate (AUGRU) to model the interest evolving process that is relative to target item. With the help of AUGRU, DIEN can overcome the disturbance from interest drifting.
Modeling for interest evolution helps us capture interest effectively, which further improves the performance of CTR prediction.
In future, we will try to construct a more personalized interest model for CTR prediction.
References
[2016]
Cheng, H.-T.; Koc, L.; Harmsen, J.; Shaked, T.; Chandra, T.; Aradhye, H.;
Anderson, G.; Corrado, G.; Chai, W.; Ispir, M.; et al.
2016.
Wide & deep learning for recommender systems.
In
Proceedings of the 1st Workshop on Deep Learning for
Recommender Systems
, 7–10.
ACM.
[2014]
Chung, J.; Gulcehre, C.; Cho, K.; and Bengio, Y.
2014.
Empirical evaluation of gated recurrent neural networks on sequence
modeling.
arXiv preprint arXiv:1412.3555
.
[2001]
Friedman, J. H.
2001.
Greedy function approximation: a gradient boosting machine.
Annals of statistics
1189–1232.
[2017]
Guo, H.; Tang, R.; Ye, Y.; Li, Z.; and He, X.
2017.
Deepfm: a factorization-machine based neural network for ctr
prediction.
In
Proceedings of the 26th International Joint Conference on
Artificial Intelligence
, 2782–2788.
[2016]
He, R., and McAuley, J.
2016.
Ups and downs: Modeling the visual evolution of fashion trends with
one-class collaborative filtering.
In
Proceedings of the 25th international conference on world
wide web
, 507–517.
[2017]
Hidasi, B., and Karatzoglou, A.
2017.
Recurrent neural networks with top-k gains for session-based
recommendations.
arXiv preprint arXiv:1706.03847
.
[1997]
Hochreiter, S., and Schmidhuber, J.
1997.
Long short-term memory.
Neural computation
9(8):1735–1780.
[2018]
Lian, J.; Zhou, X.; Zhang, F.; Chen, Z.; Xie, X.; and Sun, G.
2018.
xdeepfm: Combining explicit and implicit feature interactions for
recommender systems.
In
Proceedings of the 24th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining
.
[2015]
McAuley, J.; Targett, C.; Shi, Q.; and Van Den Hengel, A.
2015.
Image-based recommendations on styles and substitutes.
In
Proceedings of the 38th International ACM SIGIR Conference on
Research and Development in Information Retrieval
, 43–52.
ACM.
[2018]
Parsana, M.; Poola, K.; Wang, Y.; and Wang, Z.
2018.
Improving native ads ctr prediction by large scale event embedding
and recurrent networks.
arXiv preprint arXiv:1804.09133
.
[2016]
Qu, Y.; Cai, H.; Ren, K.; Zhang, W.; Yu, Y.; Wen, Y.; and Wang, J.
2016.
Product-based neural networks for user response prediction.
In
Proceedings of the16th International Conference on Data
Mining
, 1149–1154.
IEEE.
[2018]
Ren, K.; Fang, Y.; Zhang, W.; Liu, S.; Li, J.; Zhang, Y.; Yu, Y.; and Wang, J.
2018.
Learning multi-touch conversion attribution with dual-attention
mechanisms for online advertising.
arXiv preprint arXiv:1808.03737
.
[2009]
Rendle, S.; Freudenthaler, C.; Gantner, Z.; and Schmidt-Thieme, L.
2009.
Bpr: Bayesian personalized ranking from implicit feedback.
In
Proceedings of the twenty-fifth conference on uncertainty in
artificial intelligence
, 452–461.
AUAI Press.
[2010]
Rendle, S.
2010.
Factorization machines.
In
Proceedings of the 10th International Conference on Data
Mining
, 995–1000.
IEEE.
[2016]
Song, Y.; Elkahky, A. M.; and He, X.
2016.
Multi-rate deep learning for temporal recommendation.
In
Proceedings of the 39th International ACM SIGIR conference on
Research and Development in Information Retrieval
, 909–912.
ACM.
[2010]
Wang, G.; Lin, Y.; and Yi, W.
2010.
Kernel fusion: An effective method for better power efficiency on
multithreaded gpu.
In
Proceedings of the 2010 IEEE/ACM Int’L Conference on Green
Computing and Communications & Int’L Conference on Cyber, Physical and
Social Computing
, 344–350.
[1987]
Wold, S.; Esbensen, K.; and Geladi, P.
1987.
Principal component analysis.
Chemometrics and intelligent laboratory systems
2(1-3):37–52.
[2016]
Xiong, C.; Merity, S.; and Socher, R.
2016.
Dynamic memory networks for visual and textual question answering.
In
Proceedings of the 33rd International Conference on
International Conference on Machine Learning
, 2397–2406.
[2016]
Yu, F.; Liu, Q.; Wu, S.; Wang, L.; and Tan, T.
2016.
A dynamic recurrent model for next basket recommendation.
In
Proceedings of the 39th International ACM SIGIR conference on
Research and Development in Information Retrieval
, 729–732.
ACM.
[2014]
Zhang, Y.; Dai, H.; Xu, C.; Feng, J.; Wang, T.; Bian, J.; Wang, B.; and Liu,
T.-Y.
2014.
Sequential click prediction for sponsored search with recurrent
neural networks.
In
Proceedings of the 28th AAAI Conference on Artificial
Intelligence
, 1369–1375.
[2018a]
Zhou, C.; Bai, J.; Song, J.; Liu, X.; Zhao, Z.; Chen, X.; and Gao, J.
2018a.
Atrank: An attention-based user behavior modeling framework for
recommendation.
In
Proceedings of the 32nd AAAI Conference on Artificial
Intelligence
.
[2018b]
Zhou, G.; Fan, Y.; Cui, R.; Bian, W.; Zhu, X.; and Gai, K.
2018b.
Rocket launching: A universal and efficient framework for training
well-performing light net.
In
Proceedings of the 32nd AAAI Conference on Artificial
Intelligence
.
[2018c]
Zhou, G.; Zhu, X.; Song, C.; Fan, Y.; Zhu, H.; Ma, X.; Yan, Y.; Jin, J.; Li,
H.; and Gai, K.
2018c.
Deep interest network for click-through rate prediction.
In
Proceedings of the 24th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining
, 1059–1068.
ACM.