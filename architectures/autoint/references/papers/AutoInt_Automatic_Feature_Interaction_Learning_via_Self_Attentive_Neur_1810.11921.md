# Paper (Song et al. 2018)

> Source: `https://arxiv.org/abs/1810.11921`

---

AutoInt: Automatic Feature Interaction Learning via
Self-Attentive Neural Networks
Weiping Song∗
Department of Computer Science,
School of EECS, Peking University
weiping.song@pku.edu.cn
Chence Shi
Department of Computer Science,
School of EECS, Peking University
chenceshi@pku.edu.cn
Zhiping Xiao
Department of Computer Science,
University of California, Los Angeles
patriciaxiao@g.ucla.edu
Zhijian Duan, Yewen Xu
Department of Computer Science,
School of EECS, Peking University
{zjduan,xuyewen}@pku.edu.cn
Ming Zhang†
Department of Computer Science,
School of EECS, Peking University
mzhang_cs@pku.edu.cn
Jian Tang†
Mila-Quebec AI Institute,
HEC Montreal & CIFAR AI Chair
jian.tang@hec.ca
ABSTRACT
Click-through rate (CTR) prediction, which aims to predict the
probability of a user clicking on an ad or an item, is critical to many
online applications such as online advertising and recommender
systems. The problem is very challenging since (1) the input features
(e.g., the user id, user age, item id, item category) are usually sparse
and high-dimensional, and (2) an effective prediction relies on high-
order combinatorial features (a.k.a. cross features), which are very
time-consuming to hand-craft by domain experts and are impossible
to be enumerated. Therefore, there have been efforts in finding low-
dimensional representations of the sparse and high-dimensional
raw features and their meaningful combinations.
In this paper, we propose an effective and efficient method called
the AutoInt to automatically learn the high-order feature interac-
tions of input features. Our proposed algorithm is very general,
which can be applied to both numerical and categorical input fea-
tures. Specifically, we map both the numerical and categorical fea-
tures into the same low-dimensional space. Afterwards, a multi-
head self-attentive neural network with residual connections is
proposed to explicitly model the feature interactions in the low-
dimensional space. With different layers of the multi-head self-
attentive neural networks, different orders of feature combinations
of input features can be modeled. The whole model can be efficiently
fit on large-scale raw data in an end-to-end fashion. Experimental
results on four real-world datasets show that our proposed ap-
proach not only outperforms existing state-of-the-art approaches
for prediction but also offers good explainability. Code is available
at: https://github.com/DeepGraphLearning/RecommenderSystems.
∗Part of this work was performed when the first author was visiting Mila.
†Corresponding authors.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.
CIKM ’19, November 3–7, 2019, Beijing, China
© 2019 Association for Computing Machinery.
ACM ISBN 978-1-4503-6976-3/19/11...$15.00
https://doi.org/10.1145/3357384.3357925
CCS CONCEPTS
• Information systems →Recommender systems; • Comput-
ing methodologies →Neural networks; Learning latent repre-
sentations;
KEYWORDS
High-order feature interactions, Self attention, CTR prediction,
Explainable recommendation
ACM Reference Format:
Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming
Zhang, and Jian Tang. 2019. AutoInt: Automatic Feature Interaction Learn-
ing via Self-Attentive Neural Networks. In The 28th ACM International
Conference on Information and Knowledge Management (CIKM ’19), No-
vember 3–7, 2019, Beijing, China. ACM, New York, NY, USA, 10 pages.
https://doi.org/10.1145/3357384.3357925
1
INTRODUCTION
Predicting the probabilities of users clicking on ads or items (a.k.a.,
click-through rate prediction) is a critical problem for many appli-
cations such as online advertising and recommender systems [8,
10, 15]. The performance of the prediction has a direct impact on
the final revenue of the business providers. Due to its importance,
it has attracted growing interest in both academia and industry
communities.
Machine learning has been playing a key role in click-through
rate prediction, which is usually formulated as supervised learn-
ing with user profiles and item attributes as input features. The
problem is very challenging for several reasons. First, the input fea-
tures are extremely sparse and high-dimensional [8, 11, 13, 21, 32].
In real-world applications, a considerable percentage of user’s de-
mographics and item’s attributes are usually discrete and/or cat-
egorical. To make supervised learning methods applicable, these
features are first converted to a one-hot encoding vector, which
can easily result in features with millions of dimensions. Taking
the well-known CTR prediction data Criteo1 as an example, the
feature dimension is approximately 30 million with sparsity over
99.99%. With such sparse and high-dimensional input features, the
machine learning models are easily overfitted. Second, as shown in
extensive literature [8, 11, 19, 32], high-order feature interactions2
1http://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/
2In this paper, we will use “combinatorial feature” and “feature interaction” inter-
changeably as they are both used in the literature [11, 19, 32] .
arXiv:1810.11921v2  [cs.IR]  23 Aug 2019

are crucial for a good performance. For example, it is reasonable
to recommend Mario Bros., a famous video game, to David, who
is a ten-year-old boy. In this case, the third-order combinatorial
feature <Gender=Male, Age=10, ProductCategory=VideoGame> is
very informative for prediction. However, finding such meaningful
high-order combinatorial features heavily relies on domain experts.
Moreover, it is almost impossible to hand-craft all the meaningful
combinations [8, 26]. One may ask that we can enumerate all the
possible high-order features and let machine learning models se-
lect the meaningful ones. However, enumerating all the possible
high-order features will exponentially increase the dimension and
sparsity of the input features, leading to a more serious problem
of model overfitting. Therefore, there has been extensive efforts
in the communities in finding low-dimensional representations of
the sparse and high-dimensional input features and meanwhile
modeling different orders of feature combinations.
For example, Factorization Machines (FM) [26], which combine
polynomial regression models with factorization techniques, are
developed to model feature interactions and have been proved
effective for various tasks [27, 28]. However, limited by its polyno-
mial fitting time, it is only effective for modeling low-order feature
interactions and impractical to capture high-order feature interac-
tions. Recently, many works [8, 11, 13, 38] based on deep neural
networks have been proposed to model the high-order feature inter-
actions. Specifically, multiple layers of non-linear neural networks
are usually used to capture the high-order feature interactions.
However, such kinds of methods suffer from two limitations. First,
fully-connected neural networks have been shown inefficient in
learning multiplicative feature interactions [4]. Second, since these
models learn the feature interactions in an implicit way, they lack
good explanation on which feature combinations are meaningful.
Therefore, we are looking for an approach that is able to explicitly
model different orders of feature combinations, represent the entire
features into low-dimensional spaces, and meanwhile offer good
model explainability.
In this paper, we propose such an approach based on the multi-
head self-attention mechanism [36]. Our proposed approach learns
effective low-dimensional representations of the sparse and high-
dimensional input features and is applicable to both the categorical
and/or numerical input features. Specifically, both the categorical
and numerical features are first embedded into low-dimensional
spaces, which reduces the dimension of the input features and
meanwhile allows different types of features to interact with each
other via vector arithmetic (e.g., summation and inner product).
Afterwards, we propose a novel interacting layer to promote the
interactions between different features. Within each interacting
layer, each feature is allowed to interact with all the other features
and is able to automatically identify relevant features to form mean-
ingful higher-order features via the multi-head attention mecha-
nism [36]. Moreover, the multi-head mechanism projects a feature
into multiple subspaces, and hence it can capture different feature
interactions in different subspaces. Such an interacting layer models
the one-step interaction between the features. By stacking multiple
interacting layers, we are able to model different orders of feature
interactions. In practice, the residual connection [12] is added to
the interacting layer, which allows combining different orders of
feature combinations. We use the attention mechanism for mea-
suring the correlations between features, which offers good model
explainability.
To summarize, in this paper we make the following contribu-
tions:
• We propose to study the problem of explicitly learning high-
order feature interactions and meanwhile finding models
with good explainability for the problem.
• We propose a novel approach based on self-attentive neu-
ral network, which can automatically learn high-order fea-
ture interactions and efficiently handle large-scale high-
dimensional sparse data.
• We conducted extensive experiments on several real-world
data sets. Experimental results on the task of CTR predic-
tion show that our proposed approach not only outperforms
existing state-of-the-art approaches for prediction but also
offers good model explainability.
Our work is organized as follows. In Section 2, we summarize
the related work. Section 3 formally defines our problem. Section
4 presents the proposed approach to learn feature interactions. In
Section 5, we present the experimental results and detailed analysis.
We conclude this paper and point out the future work in Section 6.
2
RELATED WORK
Our work is relevant to three lines of work: 1) Click-through rate
prediction in recommender systems and online advertising, 2) tech-
niques for learning feature interactions, and 3) self-attention mech-
anism and residual networks in the literature of deep learning.
2.1
Click-through Rate Prediction
Predicting click-through rates is important to many Internet com-
panies, and various systems have been developed by different com-
panies [8–10, 15, 21, 29, 43]. For example, Google developed the
Wide&Deep[8] learning system for recommender systems, which
combines the advantages of both the linear shallow models and
deep models. The system achieves remarkable performance in APP
recommendation. The problem also receives a lot of attention in
the academic communities. For example, Shan et al. [31] proposed a
context-aware CTR prediction method which factorized three-way
<user, ad, context> tensor. Oentaryo et al. [24] developed hierar-
chical importance-aware factorization machine to model dynamic
impacts of ads.
2.2
Learning Feature Interactions
Learning feature interactions is a fundamental problem and there-
fore extensively studied in the literature. A well-known exam-
ple is Factorization Machines (FM) [26], which were proposed to
mainly capture the first- and second-order feature interactions
and have been proved effective for many tasks in recommender
systems [27, 28]. Afterwards, different variants of factorization ma-
chines have been proposed. For example, Field-aware Factorization
Machines (FFM) [16] modeled fine-grained interactions between
features from different fields. GBFM [7] and AFM [40] considered
the importance of different second-order feature interactions. How-
ever, all these approaches focus on modeling low-order feature
interactions.

There are some recent works that model high-order feature in-
teractions. For example, NFM [13] stacked deep neural networks on
top of the output of the second-order feature interactions to model
higher-order features. Similarly, PNN [25], FNN [41], DeepCross-
ing [32], Wide&Deep [8] and DeepFM [11] utilized feed-forward
neural networks to model high-order feature interactions. How-
ever, all these approaches learn the high-order feature interactions
in an implicit way and therefore lack good model explainability.
On the contrary, there are three lines of works that learn fea-
ture interactions in an explicit fashion. First, Deep&Cross [38]
and xDeepFM [19] took outer product of features at the bit- and
vector-wise level respectively. Although they perform explicit fea-
ture interactions, it is not trivial to explain which combinations are
useful. Second, some tree-based methods [39, 42, 44] combined the
power of embedding-based models and tree-based models but had
to break training procedure into multiple stages. Third, HOFM [5]
proposed efficient training algorithms for high-order factorization
machines. However, HOFM requires too many parameters and only
its low-order (usually less than 5) form can be practically used. Dif-
ferent from existing work, we explicitly model feature interactions
with attention mechanism in an end-to-end manner, and probe the
learned feature combinations via visualization.
2.3
Attention and Residual Networks
Our proposed model makes use of the latest techniques in the lit-
erature of deep learning: attention [2] and residual networks [12].
Attention is first proposed in the context of neural machine trans-
lation [2] and has been proved effective in a variety of tasks such
as question answering [35], text summarization [30], and recom-
mender systems [14, 33, 43]. Vaswani et al. [36] further proposed
multi-head self-attention to model complicated dependencies be-
tween words in machine translation.
Residual networks [12] achieved state-of-the-art performance
in the ImageNet contest. Since the residual connection, which can
be simply formalized as y = F(x) + x, encourages gradient flow
through interval layers, it becomes a popular network structure for
training very deep neural networks.
3
PROBLEM DEFINITION
We first formally define the problem of click-through rate (CTR)
prediction as follows:
DEFINITION 1. (CTR Prediction) Let x ∈Rn denotes the con-
catenation of user u’s features and item v’s features, where categor-
ical features are represented with one-hot encoding, and n is the
dimension of concatenated features. The problem of click-through
rate prediction aims to predict the probability of user u clicking on
item v according to the feature vector x.
A straightforward solution for CTR prediction is to treat x as the
input features and deploy the off-the-shelf classifiers such as logistic
regression. However, since the original feature vector x is very
sparse and high-dimensional, the model will be easily overfitted.
Therefore, it is desirable to represent the raw input features in low-
dimensional continuous spaces. Moreover, as shown in existing
literature, it is crucial to utilize the higher-order combinatorial
features to yield good prediction performance [6, 8, 11, 23, 26, 32].
…
…
Multi-head    
Self-Attention
Interacting 
Layer
Output Layer: Estimated CTR
…
…
0.3
0.5
1
0
0
Input Layer: sparse feature X
Embedding 
Layer
Feature field 1
Feature field M
Figure 1: Overview of our proposed model AutoInt. The de-
tails of embedding layer and interacting layer are illustrated
in Figure 2 and Figure 3 respectively.
Specifically, we define the high-order combinatorial features as
follows:
DEFINITION 2. (p-order Combinatorial Feature) Given input
feature vector x ∈Rn, a p-order combinatorial feature is defined
as д(xi1, ...,xip ) , where each feature comes from a distinct field, p
is the number of involved feature fields, and д(·) is a non-additive
combination function, such as multiplication [26] and outer prod-
uct [19, 38]. For example, xi1 × xi2 is a second-order combinatorial
feature involving xi1 and xi2.
Traditionally, meaningful high-order combinatorial features are
hand-crafted by domain experts. However, this is very time-consuming
and hard to generalize to other domains. Besides, it is almost impos-
sible to hand-craft all meaningful high-order features. Therefore,
we aim to develop an approach that is able to automatically discover
the meaningful high-order combinatorial features and meanwhile
map all these features into low-dimensional continuous spaces.
Formally, we define our problem as follows:
DEFINITION 3. (Problem Definition) Given an input feature
vector x ∈Rn for click-through rate prediction, our goal is to learn
a low-dimensional representation of x, which models the high-order
combinatorial features.
4
AUTOINT: AUTOMATIC FEATURE
INTERACTION LEARNING
In this section, we first give an overview of the proposed approach
AutoInt, which can automatically learn feature interactions for CTR
prediction. Next, we present a comprehensive description of how
to learn a low-dimensional representation that models high-order
combinatorial features without manual feature engineering.
4.1
Overview
The goal of our approach is to map the original sparse and high-
dimensional feature vector into low-dimensional spaces and mean-
while model the high-order feature interactions. As shown in Fig-
ure 1, our proposed method takes the sparse feature vector x as
input, followed by an embedding layer that projects all features

…
…
0.3
0.5
1
0
0
x1
xm
Embedding 
layer 
𝑒$
𝑒%
𝑒&
xM
Input layer 
Figure 2: Illustration of input and embedding layer, where
both categorical and numerical fields are represented by low-
dimensional dense vectors.
(i.e., both categorical and numerical features) into the same low-
dimensional space. Next, we feed embeddings of all fields into a
novel interacting layer, which is implemented as a multi-head self-
attentive neural network. For each interacting layer, high-order
features are combined through the attention mechanism, and dif-
ferent kinds of combinations can be evaluated with the multi-head
mechanisms, which map the features into different subspaces. By
stacking multiple interacting layers, different orders of combinato-
rial features can be modeled.
The output of the final interacting layer is the low-dimensional
representation of the input feature, which models the high-order
combinatorial features and is further used for estimating the click-
through rate through a sigmoid function. Next, we introduce the
details of our proposed method.
4.2
Input Layer
We first represent user’s profiles and item’s attributes as a sparse
vector, which is the concatenation of all fields. Specifically,
x = [x1; x2; ...; xM],
(1)
where M is the number of total feature fields, and xi is the feature
representation of the i-th field. xi is a one-hot vector if the i-th field
is categorical (e.g., x1 in Figure 2). xi is a scalar value if the i-th
field is numerical (e.g., xM in Figure 2).
4.3
Embedding Layer
Since the feature representations of the categorical features are very
sparse and high-dimensional, a common way is to represent them
into low-dimensional spaces (e.g., word embeddings). Specifically,
we represent each categorical feature with a low-dimensional vector,
i.e.,
ei = Vixi,
(2)
where Vi is an embedding matrix for field i, and xi is an one-hot
vector. Often times categorical features can be multi-valued, i.e., xi
is a multi-hot vector. Take movie watching prediction as an example,
there could be a feature field Genre which describes the types of
a movie and it may be multi-valued (e.g., Drama and Romance
for movie “Titanic”). To be compatible with multi-valued inputs,
we further modify the Equation 2 and represent the multi-valued
feature field as the average of corresponding feature embedding
vectors:
ei = 1
q Vixi,
(3)
where q is the number of values that a sample has for i-th field and
xi is the multi-hot vector representation for this field.
To allow the interaction between categorical and numerical fea-
tures, we also represent the numerical features in the same low-
dimensional feature space. Specifically, we represent the numerical
feature as
em = vmxm,
(4)
where vm is an embedding vector for field m, and xm is a scalar
value.
By doing this, the output of the embedding layer would be a con-
catenation of multiple embedding vectors, as presented in Figure 2.
4.4
Interacting Layer
Once the numerical and categorical features live in the same low-
dimensional space, we move to model high-order combinatorial
features in the space. The key problem is to determine which fea-
tures should be combined to form meaningful high-order features.
Traditionally, this is accomplished by domain experts who create
meaningful combinations based on their knowledge. In this pa-
per, we tackle this problem with a novel method, the multi-head
self-attention mechanism [36].
Multi-head self-attentive network [36] has recently achieved
remarkable performance in modeling complicated relations. For
example, it shows superiority for modeling arbitrary word depen-
dency in machine translation [36] and sentence embedding [20],
and has been successfully applied to capturing node similarities
in graph embedding [37]. Here we extend this latest technique to
model the correlations between different feature fields.
Specifically, we adopt the key-value attention mechanism [22] to
determine which feature combinations are meaningful. Taking the
feature m as an example, next we explain how to identify multiple
meaningful high-order features involving feature m. We first define
the correlation between feature m and feature k under a specific
attention head h as follows:
α(h)
m,k =
exp(ψ (h)(em, ek))
ÍM
l=1 exp(ψ (h)(em, el))
,
ψ (h)(em, ek) = ⟨W(h)
Queryem, W(h)
Keyek⟩,
(5)
whereψ (h)(·, ·) is an attention function which defines the similarity
between the feature m and k. It can be defined as a neural network
or as simple as inner product, i.e., ⟨·, ·⟩. In this work, we use inner
product due to its simplicity and effectiveness. W(h)
Query, W(h)
Key ∈
Rd′×d in Equation 5 are transformation matrices which map the
original embedding space Rd into a new space Rd′. Next, we update
the representation of feature m in subspace h via combining all
relevant features guided by coefficients α(h)
m,k:
ee(h)
m =
M
Õ
k=1
α(h)
m,k(W(h)
Valueek),
(6)
where W(h)
Value ∈Rd′×d.
Since ee(h)
m ∈Rd′ is a combination of feature m and its relevant
features (under head h), it represents a new combinatorial feature
learned by our method. Furthermore, a feature is also likely to be
involved in different combinatorial features, and we achieve this by
using multiple heads, which create different subspaces and learn
distinct feature interactions separately. We collect combinatorial
features learned in all subspaces as follows:

𝑒"
𝑊$%&'(
(*)
𝑒,
𝑊-./%&
(*)
𝑒0
𝛼"
(*)
𝑒̃"
(*)
𝑊3&(
(*)
..
..
..
..
0.1
0.8
0.02
Figure 3: The architecture of interacting layer. Combinato-
rial features are conditioned on attention weights, i.e., α(h)
m .
eem = ee(1)
m ⊕ee(2)
m ⊕· · · ⊕ee(H)
m ,
(7)
where ⊕is the concatenation operator, and H is the number of total
heads.
To preserve previously learned combinatorial features, including
raw individual (i.e., first-order) features, we add standard residual
connections in our network. Formally,
eRes
m
= ReLU(eem + WResem),
(8)
where WRes ∈Rd′H×d is the projection matrix in case of dimen-
sion mismatching [12], and ReLU(z) = max(0,z) is a non-linear
activation function.
With such an interacting layer, the representation of each feature
em will be updated into a new feature representation eRes
m , which
is a representation of high-order features. We can stack multiple
such layers with the output of the previous interacting layer as the
input of the next interacting layer. By doing this, we can model
arbitrary-order combinatorial features.
4.5
Output Layer
The output of the interacting layer is a set of feature vectors {eRes
m }M
m=1,
which includes raw individual features reserved by residual block
and combinatorial features learned via the multi-head self-attention
mechanism. For final CTR prediction, we simply concatenate all of
them and then apply a non-linear projection as follows:
ˆy = σ(wT(eRes
1
⊕eRes
2
⊕· · · ⊕eRes
M ) + b),
(9)
where w ∈Rd′HM is a column projection vector which linearly
combines concatenated features,b is the bias, andσ(x) = 1/(1+e−x)
transforms the values to users clicking probabilities.
4.6
Training
Our loss function is Log loss, which is defined as follows:
Loдloss = −1
N
N
Õ
j=1
(yj log(ˆyj) + (1 −yj) log(1 −ˆyj)),
(10)
where yj and ˆyj are ground truth of user clicks and estimated
CTR respectively, j indexes the training samples, and N is the total
number of training samples. The parameters to learn in our model
are {Vi, vm, W(h)
Query, W(h)
Key, W(h)
Value, WRes, w,b}, which are updated
via minimizing the total Logloss using gradient descent.
4.7
Analysis Of AutoInt
Modeling Arbitrary Order Combinatorial Features. Given fea-
ture interaction operator defined by Equation 5 - 8, we now analyze
how low-order and high-order combinatorial features are modeled
in our proposed model.
For simplicity, let’s assume there are four feature fields (i.e., M=4)
denoted byx1,x2,x3 andx4 respectively. Within the first interacting
layer, each individual feature interacts with any other features
through attention mechanism (i.e. Equation 5) and therefore a set
of second-order feature combinations such asд(x1,x2),д(x2,x3) and
д(x3,x4) are captured with distinct correlation weights, where the
non-additive property of interaction function д(·) (in DEFINITION
2) can be ensured by the non-linearity of activation function ReLU(·).
Ideally, combinatorial features that involve x1 can be encoded into
the updated representation of the first feature field eRes
1
. As the
same can be derived for other feature fields, all second-order feature
interactions can be encoded in the output of the first interacting
layer, where attention weights distill useful feature combinations.
Next, we prove that higher-order feature interactions can be
modeled within the second interacting layer. Given the representa-
tion of the first feature field eRes
1
and the representation of the third
feature field eRes
3
generated by the first interacting layer, third-order
combinatorial features that involve x1, x2 and x3 can be modeled
by allowing eRes
1
to attend on eRes
3
because eRes
1
contains the inter-
action д(x1,x2) and eRes
3
contains the individual feature x3 (from
residual connection). Moreover, the maximum order of combina-
torial features grows exponentially with respect to the number of
interacting layers. For example, fourth-order feature interaction
д(x1,x2,x3,x4) can be captured by the combination of eRes
1
and eRes
3
,
which contain the second-order interactions д(x1,x2) and д(x3,x4)
respectively. Therefore a few interacting layers will suffice to model
high-order feature interactions.
Based on above analysis, we can see that AutoInt learns feature
interactions with attention mechanism in a hierarchical manner,
i.e., from low-order to high-order, and all low-order feature inter-
actions are carried by residual connections. This is promising and
reasonable because learning hierarchical representation has proven
quite effective in computer vision and speech processing with deep
neural networks [3, 18].
Space Complexity. The embedding layer, which is a shared com-
ponent in neural network-based methods [11, 19, 32], contains nd
parameters, where n is the dimension of sparse representation of in-
put feature and d is the embedding size. As an interacting layer con-
tains following weight matrices: {W(h)
Query, W(h)
Key, W(h)
Value, WRes},
the number of parameters in an L-layer network is L×(3dd′+d′Hd),
which is independent of the number of feature fields M. Finally,
there are d′HM + 1 parameters in the output layer. As far as in-
teracting layers are concerned, the space complexity is O(Ldd′H).
Note that H and d′ are usually small (e.g., H = 2 and d′ = 32 in our
experiments), which makes the interacting layer memory-efficient.
Time Complexity. Within each interacting layer, the computation
cost is two-fold. First, calculating attention weights for one head

takes O(Mdd′ + M2d′) time. Afterwards, forming combinatorial
features under one head also takes O(Mdd′ + M2d′) time. Because
we have H heads, it takes O(MHd′(M + d)) time altogether. It is
therefore efficient because H,d andd′ are usually small. We provide
running time of AutoInt in Section 5.2.
5
EXPERIMENT
In this section, we move forward to evaluate the effectiveness of
our proposed approach. We aim to answer the following questions:
RQ1 How does our proposed AutoInt perform on the problem of
CTR prediction? Is it efficient for large-scale sparse and
high-dimensional data?
RQ2 What are the influences of different model configurations?
RQ3 What are the dependency structures between different
features? Is our proposed model explainable?
RQ4 Will integrating implicit feature interactions further
improve the performance?
We first describe the experimental settings before answering these
questions.
5.1
Experiment Setup
5.1.1
Data Sets. We use four public real-world data sets. The
statistics of the data sets are summarized in Table 1. Criteo3 This is
a benchmark dataset for CTR prediction, which has 45 million users’
clicking records on displayed ads. It contains 26 categorical feature
fields and 13 numerical feature fields. Avazu4 This dataset contains
users’ mobile behaviors including whether a displayed mobile ad
is clicked by a user or not. It has 23 feature fields spanning from
user/device features to ad attributes. KDD125 This data set was
released by KDDCup 2012, which originally aimed to predict the
number of clicks. Since our work focuses on CTR prediction rather
than the exact number of clicks, we treat this problem as a binary
classification problem (1 for clicks>0, 0 for without click), which is
similar to FFM [16]. MovieLens-1M6 This dataset contains users’
ratings on movies. During binarization, we treat samples with a
rating less than 3 as negative samples because a low score indicates
that the user does not like the movie. We treat samples with a rating
greater than 3 as positive samples and remove neutral samples, i.e.,
a rating equal to 3.
Data Preparation First, we remove the infrequent features (ap-
pearing in less than threshold instances) and treat them as a single
feature “<unknown>”, where threshold is set to {10, 5, 10} for Criteo,
Avazu and KDD12 data sets respectively. Second, since numerical
features may have large variance and hurt machine learning algo-
rithms, we normalize numerical values by transforming a value
z to loд2(z) if z > 2, which is proposed by the winner of Criteo
Competition7. Third, we randomly select 80% of all samples for
training and randomly split the rest into validation and test sets of
equal size.
5.1.2
Evaluation Metrics. We use two popular metrics to evalu-
ate the performance of all methods.
3https://www.kaggle.com/c/criteo-display-ad-challenge
4https://www.kaggle.com/c/avazu-ctr-prediction
5https://www.kaggle.com/c/kddcup2012-track2
6https://grouplens.org/datasets/movielens/
7https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
Table 1: Statistics of evaluation data sets.
Data
#Samples
#Fields
#Features (Sparse)
Criteo
45,840,617
39
998,960
Avazu
40,428,967
23
1,544,488
KDD12
149,639,105
13
6,019,086
MovieLens-1M
739,012
7
3,529
AUC Area Under the ROC Curve (AUC) measures the probability
that a CTR predictor will assign a higher score to a randomly chosen
positive item than a randomly chosen negative item. A higher AUC
indicates a better performance.
Logloss Since all models attempt to minimize the Logloss defined
by Equation 10, we use it as a straightforward metric.
It is noticeable that a slightly higher AUC or lower Logloss at
0.001-level is regarded significant for CTR prediction task, which
has also been pointed out in existing works [8, 11, 38].
5.1.3
Competing Models. We compare the proposed approach
with three classes of previous models. (A) the linear approach that
only uses individual features. (B) factorization machines-based
methods that take into account second-order combinatorial features.
(C) techniques that can capture high-order feature interactions. We
associate the model classes with model names accordingly.
LR (A). LR only models the linear combination of raw features.
FM [26] (B). FM uses factorization techniques to model second-
order feature interactions.
AFM [40] (B). AFM is one of the state-of-the-art models that
capture second-order feature interactions. It extends FM by using
attention mechanism to distinguish the different importance of
second-order combinatorial features.
DeepCrossing [32] (C). DeepCrossing utilizes deep fully-connected
neural networks with residual connections to learn non-linear fea-
ture interactions in an implicit fashion.
NFM [13] (C). NFM stacks deep neural networks on top of
second-order feature interaction layer. High-order feature interac-
tions are implicitly captured by the nonlinearity of neural networks.
CrossNet [38] (C). Cross Network, which is the core of Deep&Cross
model, takes outer product of concatenated feature vector at the
bit-wise level to model feature interactions explicitly.
CIN [19] (C). Compressed Interaction Network, which is the
core of xDeepFM model, takes outer product of stacked feature
matrix at vector-wise level.
HOFM [5] (C). HOFM proposes efficient kernel-based algorithms
for training high-order factorization machines. Follow settings
in Blondel et al. [5] and He and Chua [13], we build a third-order
factorization machine using public implementation.
We will compare with the full models of CrossNet and CIN, i.e.,
Deep&Cross and xDeepFM, under the setting of joint training with
plain DNN later (i.e., Section 5.5).
5.1.4
Implementation Details. All methods are implemented in
TensorFlow[1]. For AutoInt and all baseline methods, we empirically
set embedding dimension d to 16 and batch size to 1024. AutoInt
has three interacting layers and the number of hidden units d′ is
32 in default setting. Within each interacting layer, the number of

Table 2: Effectiveness Comparison of Different Algorithms. We highlight that our proposed model almost outperforms all
baselines across four data sets and both metrics. Further analysis is provided in Section 5.2.
Model Class
Model
Criteo
Avazu
KDD12
MovieLens-1M
AUC
Logloss
AUC
Logloss
AUC
Logloss
AUC
Logloss
First-order
LR
0.7820
0.4695
0.7560
0.3964
0.7361
0.1684
0.7716
0.4424
Second-order
FM [26]
0.7836
0.4700
0.7706
0.3856
0.7759
0.1573
0.8252
0.3998
AFM[40]
0.7938
0.4584
0.7718
0.3854
0.7659
0.1591
0.8227
0.4048
High-order
DeepCrossing [32]
0.8009
0.4513
0.7643
0.3889
0.7715
0.1591
0.8448
0.3814
NFM [13]
0.7957
0.4562
0.7708
0.3864
0.7515
0.1631
0.8357
0.3883
CrossNet [38]
0.7907
0.4591
0.7667
0.3868
0.7773
0.1572
0.7968
0.4266
CIN [19]
0.8009
0.4517
0.7758
0.3829
0.7799
0.1566
0.8286
0.4108
HOFM [5]
0.8005
0.4508
0.7701
0.3854
0.7707
0.1586
0.8304
0.4013
AutoInt (ours)
0.8061**
0.4455**
0.7752
0.3824
0.7883**
0.1546**
0.8456*
0.3797**
AutoInt outperforms the strongest baseline w.r.t. Criteo, KDD12 and MovieLens-1M data at the: ** 0.01 and * 0.05 level, unpaired t-test.
LR
FM
AFM
DC
CN
CIN
HOFM
NFM
AutoInt
0
500
1000
1500
2000
2500
3000
3500
4000
4500
5000
Seconds
Run time per epoch
(a) Criteo
LR
FM
AFM
DC
CN
CIN
HOFM
NFM
AutoInt
0
500
1000
1500
2000
2500
3000
3500
Seconds
Run time per epoch
(b) Avazu
LR
FM
AFM
DC
CN
CIN
HOFM
NFM
AutoInt
0
5000
10000
15000
20000
25000
30000
35000
Seconds
Run time per epoch
(c) KDD12
LR
FM
AFM
DC
CN
CIN
HOFM
NFM
AutoInt
0
2
4
6
8
10
12
14
16
18
20
Seconds
Run time per epoch
(d) MovieLens-1M
Figure 4: Efficiency Comparison of Different Algorithms in terms of Run Time. “DC” and “CN” are DeepCrossing and CrossNet
for short, respectively. Since HOFM cannot be fit on one GPU card for the KDD12 dataset, extra communication cost makes it
most time-consuming. Further analysis is presented in Section 5.2.
attention head is two8. To prevent overfitting, we use grid search
to select dropout rate [34] from {0.1 - 0.9} for MovieLens-1M data
set, and we found dropout is not necessary for other three large
data sets. For baseline methods, we use one hidden layer of size 200
on top of Bi-Interaction layer for NFM as recommended by their
paper. For CN and CIN, we use three interaction layers following
AutoInt. DeepCrossing has four feed-forward layers and the number
of hidden units is 100, because it performs poorly when using
three neural layers. Once all network structures are fixed, we also
apply grid search to baseline methods for optimal hype-parameters.
Finally, we use Adam [17] to optimize all deep neural network-based
models.
5.2
Quantitative Results (RQ1)
Evaluation of Effectiveness
We summarize the results averaged over 10 different runs into Ta-
ble 2. We have the following observations: (1) FM and AFM, which
explore second-order feature interactions, consistently outperform
LR by a large margin on all datasets, which indicates that individ-
ual features are insufficient in CTR prediction. (2) An interesting
observation is the inferiority of some models which capture high-
order feature interactions. For example, although DeepCrossing
8We also tried different number of attention heads. The performance of using one
head is inferior to that of two heads, and the improvement of further increasing head
number is not significant.
and NFM use the deep neural network as a core component to
learning high-order feature interactions, they do not guarantee
improvement over FM and AFM. This may attribute to the fact
that they learn feature interactions in an implicit fashion. On the
contrary, CIN does it explicitly and outperforms low-order models
consistently. (3) HOFM significantly outperforms FM on Criteo and
MovieLens-1M datasets, which indicates that modeling third-order
feature interactions can be beneficial to prediction performance. (4)
AutoInt achieves the best performance overall baseline methods on
three of four real-world data sets. On Avazu data set, CIN performs
a little better than AutoInt in AUC evaluation, but we get lower
Logloss. Note that our proposed AutoInt shares the same structures
as DeepCrossing except the feature interacting layer, which indi-
cates using the attention mechanism to learn explicit combinatorial
features is crucial.
Evaluation of Model Efficiency
We present the runtime results of different algorithms on four data
sets in Figure 4. Unsurprisingly, LR is the most efficient algorithm
due to its simplicity. FM and NFM perform similarly in terms of
runtime because NFM only stacks a single feed-forward hidden
layer on top of the second-order interaction layer. Among all listed
methods, CIN, which achieves the best performance for prediction
among all the baselines, is much more time-consuming due to its
complicated crossing layer. This may make it impractical in the

Table 3: Efficiency Comparison of Different Algorithms in
terms of Model Size on Criteo data set. “DC” and “CN”
are DeepCrossing and CrossNet for short, respectively. The
counted parameters exclude the embedding layer.
Model
DC
CN
CIN
NFM
AutoInt
#Params
1.6 × 105
3 × 103
1.9 × 106
4 × 103
3.9 × 104
Table 4: Ablation study comparing the performance of Au-
toInt with and without residual connections. AutoIntw/ is
the complete model while the AutoIntw/o is the model with-
out residual connection.
Data Sets
Models
AUC
Logloss
Criteo
AutoIntw/
0.8061
0.4454
AutoIntw/o
0.8033
0.4478
Avazu
AutoIntw/
0.7752
0.3823
AutoIntw/o
0.7729
0.3836
KDD12
AutoIntw/
0.7888
0.1545
AutoIntw/o
0.7831
0.1557
MovieLens-1M
AutoIntw/
0.8460
0.3784
AutoIntw/o
0.8299
0.3959
industrial scenarios. Note that AutoInt is sufficiently efficient, which
is comparable to the efficient algorithms DeepCrossing and NFM.
We also compare the sizes of different models (i.e., the number
of parameters) as another criterion for efficiency evaluation. As
shown in Table 3, comparing to the best model CIN in the baseline
models, the number of parameters in AutoInt is much smaller.
To summarize, our proposed AutoInt achieves the best perfor-
mance among all the compared models. Compared to the most
competitive baseline model CIN, AutoInt requires much fewer pa-
rameters and is much more efficient during online inference.
5.3
Analysis (RQ2)
To further validate and gain deep insights into the proposed model,
we conduct ablation study and compare several variants of AutoInt.
5.3.1
Influence of Residual Structure. The standard AutoInt uti-
lizes residual connections, which carry through all learned com-
binatorial features and therefore allow modeling very high-order
combinations. To justify the contribution of residual units, we tease
apart them from our standard model and keep other structures as
they are. As presented in Table 4, we observe that the performance
decrease on all datasets if residual connections are removed. Specif-
ically, the full model outperforms the variant by a large margin
on the KDD12 and MovieLens-1M data, which indicates residual
connections are crucial to model high-order feature interactions in
our proposed method.
5.3.2
Influence of Network Depths. Our model learns high-order
feature combinations by stacking multiple interacting layers (in-
troduced in Section 4). Therefore, we are interested in how the
performance change w.r.t. the number of interacting layers, i.e.,
the order of combinatorial features. Note that when there is no
interacting layer (i.e., Number of layers equals zero), our model
0
1
2
3
4
Number of layers
0.77
0.78
0.79
0.80
0.81
0.82
0.83
0.84
0.85
AUC
0.73
0.74
0.75
0.76
0.77
0.78
0.79
AUC
MovieLens-1M
KDD12
(a) AUC
0
1
2
3
4
Number of layers
0.37
0.38
0.39
0.40
0.41
0.42
0.43
0.44
0.45
Logloss
0.154
0.156
0.158
0.160
0.162
0.164
0.166
0.168
0.170
Logloss
MovieLens0M
KDD12
(b) Logloss
Figure 5: Performance w.r.t. the number of interacting lay-
ers. Results on Criteo and Avazu data sets are similar and
hence omitted.
8
16
24
32
#Dimension
0.836
0.838
0.840
0.842
0.844
0.846
0.848
0.850
Logloss
0.783
0.784
0.785
0.786
0.787
0.788
Logloss
MovieLens-1M
KDD12
(a) AUC
8
16
24
32
#Dimension
0.374
0.376
0.378
0.380
0.382
0.384
Logloss
0.1544
0.1546
0.1548
0.1550
0.1552
0.1554
Logloss
MovieLens-1M
KDD12
(b) Logloss
Figure 6: Performance w.r.t. number of embedding dimen-
sions. Results on Criteo and Avazu data sets are similar and
hence omitted.
takes the weighted sum of raw individual features as input, i.e., no
combinatorial features are considered.
The results are summarized in Figure 5. We can see that if one
interacting layer is used, i.e., feature interactions are taken into
account, the performance increase dramatically on both data sets,
showing that combinatorial features are very informative for pre-
diction. As the number of interacting layers further increases, i.e.,
higher-order combinatorial features are taken into account, the
performance of the model further increases. When the number
of layers reaches three, the performance becomes stable, showing
that adding extremely high-order features are not informative for
prediction.
5.3.3
Influence of Different Dimensions. Next, we investigate the
performance w.r.t. the parameter d, which is the output dimension
of the embedding layer. On the KDD12 dataset, we can see that the
performance continuously increase as we increase the dimension
size since larger models are used for prediction. The results are
different on the MovieLens-1M dataset. When the dimension size
reaches 24, the performance begins to decrease. The reason is that
this data set is small, and the model is overfitted when too many
parameters are used.
5.4
Explainable Recommendations (RQ3)
A good recommender system can not only provide good recommen-
dations but also offer good explainability. Therefore, in this part,
we present how our AutoInt is able to explain the recommendation
results. We take the MovieLens-1M dataset as an example.
Let’s look at a recommendation result suggested by our algo-
rithm, i.e., a user likes an item. Figure 7 (a) presents the correlations
between different fields of input features, which are obtained by

Table 5: Results of Integrating Implicit Feature Interactions. We indicate the base model behind each method. The last two
columns are average changes of AUC and Logloss compared to corresponding base models (“+”: increase, “-”: decrease).
Model
Criteo
Avazu
KDD12
MovieLens-1M
Avg. Changes
AUC
Logloss
AUC
Logloss
AUC
Logloss
AUC
Logloss
AUC
Logloss
Wide&Deep (LR)
0.8026
0.4494
0.7749
0.3824
0.7549
0.1619
0.8300
0.3976
+0.0292
-0.0213
DeepFM (FM)
0.8066
0.4449
0.7751
0.3829
0.7867
0.1549
0.8437
0.3846
+0.0142
-0.0113
Deep&Cross (CN)
0.8067
0.4447
0.7731
0.3836
0.7872
0.1549
0.8446
0.3809
+0.0200
-0.0164
xDeepFM (CIN)
0.8070
0.4447
0.7770
0.3823
0.7820
0.1560
0.8463
0.3808
+0.0068
-0.0096
AutoInt+ (ours)
0.8083**
0.4434**
0.7774*
0.3811**
0.7898**
0.1543**
0.8488**
0.3753**
+0.0023
-0.0020
AutoInt+ outperforms the strongest baseline w.r.t. each data at the: ** 0.01 and * 0.05 level, unpaired t-test.
(a) Label=1, Predicted CTR=0.89
(b) Overall feature interactions
Figure 7: Heat maps of attention weights for both case-
and global-level feature interactions on MovieLens-1M. The
axises represent feature fields <Gender, Age, Occupation, Zip-
code, RequestTime, RealeaseTime, Genre>. We highlight some
learned combinatorial features in rectangles.
the attention score. We can see that AutoInt is able to identify
the meaningful combinatorial feature <Gender=Male, Age=[18-24),
MovieGenre=Action&Triller> (i.e., red dotted rectangle). This is very
reasonable since young men are very likely to prefer action&triller
movies.
We are also interested in what the correlations between different
feature fields in the data are. Therefore, we measure the correlations
between the feature fields according to their average attention score
in the entire data. The correlations between different fields are sum-
marized into Figure 7 (b). We can see that <Gender, Genre>, <Age,
Genre>, <RequestTime, ReleaseTime> and <Gender, Age, Genre> (i.e.,
solid green region) are strongly correlated, which are the explain-
able rules for recommendation in this domain.
5.5
Integrating Implicit Interactions (RQ4)
Feed-forward neural networks are capable of modeling implicit fea-
ture interactions and have been widely integrated into existing CTR
prediction methods [8, 11, 19]. To investigate whether integrating
implicit feature interactions further improves the performance, we
combine AutoInt with a two-layer feed-forward neural network by
joint training. We name the joint model AutoInt+ and compare it
with the following algorithms:
• Wide&Deep [8]. Wide&Deep integrates the outputs of logistic
regression and feed-forward neural networks.
• DeepFM [11]. DeepFM combines trainditional second-order fac-
torization machines and feed-forward neural network, with a
shared embedding layer.
• Deep&Cross [38]. Deep&Cross is the extension of CrossNet by
integrating feed-forward neural networks.
• xDeepFM [19]. xDeepFM is the extension of CIN by integrating
feed-forward neural networks.
Table 5 presents the averaged results (over 10 runs) of joint-
training models. We have the following observations: 1) The perfor-
mance of our method improves by joint training with feed-forward
neural networks on all datasets. This indicates that integrating im-
plicit feature interactions indeed boosts the predictive ability of our
proposed model. However, as can be seen from last two columns,
the magnitude of performance improvement is fairly small com-
pared to other models, showing that our individual model AutoInt
is quite powerful. 2) After integrating implicit feature interactions,
AutoInt+ outperforms all competitive methods, and achieves new
state-of-the-art performances on used CTR prediction data sets.
6
CONCLUSION AND FUTURE WORK
In this work, we propose a novel CTR prediction model based on
self-attention mechanism, which can automatically learn high-order
feature interactions in an explicit fashion. The key to our method
is the newly-introduced interacting layer, which allows each fea-
ture to interact with the others and to determine the relevance
through learning. Experimental results on four real-world data sets
demonstrate the effectiveness and efficiency of our proposed model.
Besides, we provide good model explainability via visualizing the
learned combinatorial features. When integrating with implicit
feature interactions captured by feed-forward neural networks,
we achieve better offline AUC and Logloss scores compared to the
previous state-of-the-art methods.
For future work , we are interested in incorporating contextual
information into our method and improving its performance for on-
line recommender systems. Besides, we also plan to extend AutoInt
for general machine learning tasks, such as regression, classification
and ranking.
7
ACKNOWLEDGEMENT
The authors would like to thank all the anonymous reviewers for
their insightful comments. We thank Xiao Xiao and Jianbo Dong for
the discussion on recommendation mechanism in China University
MOOC platform. We also thank Meng Qu for reviewing the initial
version of this paper. Weiping Song and Ming Zhang are supported
by National Key Research and Development Program of China with
Grant No. SQ2018AAA010010, Beijing Municipal Commission of
Science and Technology under Grant No. Z181100008918005 as well
as the National Natural Science Foundation of China (NSFC Grant
Nos.61772039 and 91646202). Weiping Song is also supported by

Chinese Scholarship Council. Jian Tang is supported by the Natural
Sciences and Engineering Research Council of Canada, as well as
the Canada CIFAR AI Chair Program.
REFERENCES
[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey
Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, et al. 2016. TensorFlow:
A System for Large-Scale Machine Learning.. In OSDI, Vol. 16. 265–283.
[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine
translation by jointly learning to align and translate. In International Conference
on Learning Representations.
[3] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013.
Representation
learning: A review and new perspectives. IEEE transactions on pattern analysis
and machine intelligence 35, 8 (2013), 1798–1828.
[4] Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, and Ed H
Chi. 2018. Latent Cross: Making Use of Context in Recurrent Recommender
Systems. In Proceedings of the Eleventh ACM International Conference on Web
Search and Data Mining. ACM, 46–54.
[5] Mathieu Blondel, Akinori Fujino, Naonori Ueda, and Masakazu Ishihata. 2016.
Higher-order factorization machines. In Advances in Neural Information Process-
ing Systems. 3351–3359.
[6] Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, and Naonori Ueda. 2016.
Polynomial Networks and Factorization Machines: New Insights and Efficient
Training Algorithms. In International Conference on Machine Learning. 850–858.
[7] Chen Cheng, Fen Xia, Tong Zhang, Irwin King, and Michael R Lyu. 2014. Gradient
boosting factorization machines. In Proceedings of the 8th ACM Conference on
Recommender systems. ACM, 265–272.
[8] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra,
Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al.
2016. Wide & deep learning for recommender systems. In Proceedings of the 1st
Workshop on Deep Learning for Recommender Systems. ACM, 7–10.
[9] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks
for youtube recommendations. In Proceedings of the 10th ACM Conference on
Recommender Systems. ACM, 191–198.
[10] Thore Graepel, Joaquin Quiñonero Candela, Thomas Borchert, and Ralf Her-
brich. 2010. Web-scale Bayesian Click-through Rate Prediction for Sponsored
Search Advertising in Microsoft’s Bing Search Engine. In Proceedings of the 27th
International Conference on International Conference on Machine Learning. 13–20.
[11] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017.
DeepFM: A Factorization-machine Based Neural Network for CTR Prediction.
In Proceedings of the 26th International Joint Conference on Artificial Intelligence.
AAAI Press, 1725–1731.
[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual
learning for image recognition. In Proceedings of the IEEE conference on computer
vision and pattern recognition. 770–778.
[13] Xiangnan He and Tat-Seng Chua. 2017. Neural factorization machines for sparse
predictive analytics. In Proceedings of the 40th International ACM SIGIR conference
on Research and Development in Information Retrieval. ACM, 355–364.
[14] Xiangnan He, Zhankui He, Jingkuan Song, Zhenguang Liu, Yu-Gang Jiang, and
Tat-Seng Chua. 2018. NAIS: Neural attentive item similarity model for recom-
mendation. IEEE Transactions on Knowledge and Data Engineering 30, 12 (2018),
2354–2366.
[15] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine
Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting
clicks on ads at facebook. In Proceedings of the Eighth International Workshop on
Data Mining for Online Advertising. ACM, 1–9.
[16] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. 2016. Field-
aware factorization machines for CTR prediction. In Proceedings of the 10th ACM
Conference on Recommender Systems. ACM, 43–50.
[17] Diederick P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic opti-
mization. In International Conference on Learning Representations.
[18] Honglak Lee, Roger Grosse, Rajesh Ranganath, and Andrew Y Ng. 2011. Unsu-
pervised learning of hierarchical representations with convolutional deep belief
networks. Commun. ACM 54, 10 (2011), 95–103.
[19] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and
Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature
Interactions for Recommender Systems. In Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining. ACM, 1754–
1763.
[20] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang,
Bowen Zhou, and Yoshua Bengio. 2017. A structured self-attentive sentence
embedding. In International Conference on Learning Representations.
[21] H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner,
Julian Grady, Lan Nie, Todd Phillips, et al. 2013. Ad Click Prediction: A View from
the Trenches. In Proceedings of the 19th ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining. ACM, 1222–1230.
[22] Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bor-
des, and Jason Weston. 2016. Key-Value Memory Networks for Directly Reading
Documents. In Proceedings of the 2016 Conference on Empirical Methods in Natural
Language Processing. Association for Computational Linguistics, 1400–1409.
[23] Alexander Novikov, Mikhail Trofimov, and Ivan Oseledets. 2016. Exponential
machines. arXiv preprint arXiv:1605.03795 (2016).
[24] Richard J Oentaryo, Ee-Peng Lim, Jia-Wei Low, David Lo, and Michael Finegold.
2014. Predicting response in mobile advertising with hierarchical importance-
aware factorization machine. In Proceedings of the 7th ACM international confer-
ence on Web search and data mining. ACM, 123–132.
[25] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang.
2016. Product-based neural networks for user response prediction. In Data Mining
(ICDM), 2016 IEEE 16th International Conference on. IEEE, 1149–1154.
[26] Steffen Rendle. 2010. Factorization machines. In Data Mining (ICDM), 2010 IEEE
10th International Conference on. IEEE, 995–1000.
[27] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factor-
izing personalized markov chains for next-basket recommendation. In Proceedings
of the 19th international conference on World wide web. ACM, 811–820.
[28] Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, and Lars Schmidt-Thieme.
2011. Fast context-aware recommendations with factorization machines. In Pro-
ceedings of the 34th international ACM SIGIR conference on Research and develop-
ment in Information Retrieval. ACM, 635–644.
[29] Matthew Richardson, Ewa Dominowska, and Robert Ragno. 2007. Predicting
clicks: estimating the click-through rate for new ads. In Proceedings of the 16th
international conference on World Wide Web. ACM, 521–530.
[30] Alexander M. Rush, Sumit Chopra, and Jason Weston. 2015. A Neural Atten-
tion Model for Abstractive Sentence Summarization. In Proceedings of the 2015
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, 379–389.
[31] Lili Shan, Lei Lin, Chengjie Sun, and Xiaolong Wang. 2016. Predicting ad click-
through rates via feature-based fully coupled interaction tensor factorization.
Electronic Commerce Research and Applications 16 (2016), 30–42.
[32] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. 2016.
Deep crossing: Web-scale modeling without manually crafted combinatorial
features. In Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining. ACM, 255–262.
[33] Weiping Song, Zhiping Xiao, Yifan Wang, Laurent Charlin, Ming Zhang, and Jian
Tang. 2019. Session-based Social Recommendation via Dynamic Graph Attention
Networks. In Proceedings of the Twelfth ACM International Conference on Web
Search and Data Mining. ACM, 555–563.
[34] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan
Salakhutdinov. 2014. Dropout: A simple way to prevent neural networks from
overfitting. The Journal of Machine Learning Research 15, 1 (2014), 1929–1958.
[35] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. 2015. End-to-end memory
networks. In Advances in neural information processing systems. 2440–2448.
[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need. In Advances in Neural Information Processing Systems. 6000–6010.
[37] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
Lio, and Yoshua Bengio. 2018. Graph Attention Networks. In International Con-
ference on Learning Representations.
[38] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network
for Ad Click Predictions. In Proceedings of the ADKDD’17. ACM, 12:1–12:7.
[39] Xiang Wang, Xiangnan He, Fuli Feng, Liqiang Nie, and Tat-Seng Chua. 2018.
TEM: Tree-enhanced Embedding Model for Explainable Recommendation. In Pro-
ceedings of the 2018 World Wide Web Conference on World Wide Web. International
World Wide Web Conferences Steering Committee, 1543–1552.
[40] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua.
2017. Attentional factorization machines: learning the weight of feature in-
teractions via attention networks. In Proceedings of the 26th International Joint
Conference on Artificial Intelligence. AAAI Press, 3119–3125.
[41] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field
categorical data. In European conference on information retrieval. Springer, 45–57.
[42] Qian Zhao, Yue Shi, and Liangjie Hong. 2017. GB-CENT: Gradient Boosted Cate-
gorical Embedding and Numerical Trees. In Proceedings of the 26th International
Conference on World Wide Web. International World Wide Web Conferences
Steering Committee, 1311–1319.
[43] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui
Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-
Through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining. ACM, 1059–1068.
[44] Jie Zhu, Ying Shan, JC Mao, Dong Yu, Holakou Rahmanian, and Yi Zhang. 2017.
Deep embedding forest: Forest-based serving with deep embedding features.
In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge
Discovery and Data Mining. ACM, 1703–1711.
