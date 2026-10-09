# Paper (He et al. 2017)

> Source: `https://arxiv.org/abs/1708.05027`

---

Neural Factorization Machines for Sparse Predictive Analytics∗
Xiangnan He
School of Computing
National University of Singapore
Singapore 117417
dcshex@nus.edu.sg
Tat-Seng Chua
School of Computing
National University of Singapore
Singapore 117417
dcscts@nus.edu.sg
ABSTRACT
Many predictive tasks of web applications need to model categori-
cal variables, such as user IDs and demographics like genders and
occupations. To apply standard machine learning techniques, these
categorical predictors are always converted to a set of binary fea-
tures via one-hot encoding, making the resultant feature vector
highly sparse. To learn from such sparse data eﬀectively, it is crucial
to account for the interactions between features.
Factorization Machines (FMs) are a popular solution for eﬃciently
using the second-order feature interactions. However, FM mod-
els feature interactions in a linear way, which can be insuﬃcient
for capturing the non-linear and complex inherent structure of
real-world data. While deep neural networks have recently been
applied to learn non-linear feature interactions in industry, such as
the Wide&Deep by Google and DeepCross by Microsof, the deep
structure meanwhile makes them diﬃcult to train.
In this paper, we propose a novel model Neural Factorization Ma-
chine (NFM) for prediction under sparse setings. NFM seamlessly
combines the linearity of FM in modelling second-order feature
interactions and the non-linearity of neural network in modelling
higher-order feature interactions. Conceptually, NFM is more ex-
pressive than FM since FM can be seen as a special case of NFM
without hidden layers. Empirical results on two regression tasks
show that with one hidden layer only, NFM signiﬁcantly outper-
forms FM with a 7.3% relative improvement. Compared to the recent
deep learning methods Wide&Deep and DeepCross, our NFM uses
a shallower structure but oﬀers beter performance, being much
easier to train and tune in practice.
CCS CONCEPTS
•Information systems →Information retrieval; Recommender
systems; •Computing methodologies →Neural networks; Fac-
torization methods;
KEYWORDS
Factorization Machines, Neural Networks, Deep Learning, Sparse
Data, Regression, Recommendation
∗NExT research is supported by the National Research Foundation, Prime Minister’s
Oﬃce, Singapore under its IRC@SG Funding Initiative.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for proﬁt or commercial advantage and that copies bear this notice and the full citation
on the ﬁrst page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permited. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior speciﬁc permission and/or a
fee. Request permissions from permissions@acm.org.
SIGIR’17, August 7–11, 2017, Shinjuku, Tokyo, Japan
© 2017 ACM. 978-1-4503-5022-8/17/08...$15.00
DOI: htp://dx.doi.org/10.1145/3077136.3080777
1
INTRODUCTION
Predictive analytics is one of the most important techniques for
many information retrieval (IR) and data mining (DM) tasks, rang-
ing from recommendation systems [2, 16], targeted advertising [21],
to search ranking [19, 39], visual analysis [35], and event detec-
tion [40]. Typically, a predictive task is formulated as estimating
a function that maps predictor variables to some target, for ex-
ample, real valued target for regression and categorical target for
classiﬁcation. Distinct from continuous predictor variables that are
naturally found in images and audios, such as raw features, the
predictor variables for web applications are mostly discrete and
categorical. For example, in online advertising, we need to predict
how likely (target) a user (ﬁrst predictor variable) of a particular
occupation (second predictor variable) will click on an ad (third
predictor variable). To build predictive models with these categor-
ical predictor variables, a common solution is to convert them to
a set of binary features (a.k.a. feature vector) via one-hot encod-
ing [2, 9, 16, 30, 31]. Tereafer, standard machine learning (ML)
techniques such as logistic regression and support vector machines
can be applied.
Depending on the possible values of categorical predictor vari-
ables, the generated feature vector can be of very high dimension
but sparse. To build eﬀective ML models with such sparse data,
it is crucial to account for the interactions between features [4,
23, 31]. Many successful solutions in both industry and academia
largely rely on manually crafing combinatorial features [9], i.e., con-
structing new features by combining multiple predictor variables,
also known as cross features. For example, we can cross variable
occupation = {banker,doctor} withдender = {M, F} and get a new
occupation дender = {banker M,banker F,doctor M,doctor F}.
It is well known that top data scientists are usually masters of
crafing combinatorial features, which play a key role in their win-
ning formulas [31]. However, the power of such features comes at a
high cost, since it requires heavy engineering eﬀorts and useful do-
main knowledge to design eﬀective features. Tus these solutions
can be diﬃcult to generalize to new problems or domains.
Instead of augmenting feature vectors manually, another solution
is to design a ML model to learn feature interactions from raw
data automatically. A popular approach is factorization machines
(FMs) [27], which embeds features into a latent space and models the
interactions between features via inner product of their embedding
vectors. While FM has yielded great promise1 in many prediction
tasks [2, 8, 21, 24], we argue that its performance can be limited
by its linearity, as well as the modelling of pairwise (i.e., second-
order) feature interactions only. Speciﬁcally, for real-world data
1htps://securityintelligence.com/factorization-machines-a-new-way-of-looking-at-
machine-learning
arXiv:1708.05027v1  [cs.LG]  16 Aug 2017

that have complex and non-linear underlying structure, FM may
not be expressive enough. Although higher-order FMs have been
proposed [27], they still belong to the family of linear models and
are claimed to be diﬃcult to estimate [28]. Moreover, they are
known to have only marginal improvements over FM, which we
suspect the reason might be due to the modelling of higher-order
interactions in a linear way.
In this work, we propose a novel model for sparse data prediction
named Neural Factorization Machines (NFMs), which enhances FMs
by modelling higher-order and non-linear feature interactions. By
devising a new operation in neural network modelling — Bilinear
Interaction (Bi-Interaction) pooling — we subsume FM under the
neural network framework for the ﬁrst time. Trough stacking
non-linear layers above the Bi-Interaction layer, we are able to
deepen the shallow linear FM, modelling higher-order and non-
linear feature interactions eﬀectively to improve FM’s expressive-
ness. In contrast to traditional deep learning methods that simply
concatenate [9, 31, 44] or average [16, 36] embedding vectors in
the low level, our use of Bi-Interaction pooling encodes more in-
formative feature interactions, greatly facilitating the following
“deep” layers to learn meaningful information. We conduct ex-
tensive experiments on two public benchmarks for context-aware
prediction and personalized tag recommendation. With one hid-
den layer only, our NFM signiﬁcantly outperforms FM (the oﬃ-
cial LibFM [28] implementation) with a 7.3% improvement. Com-
pared to the state-of-the-art deep learning methods — the 3-layer
Wide&Deep [9] and 10-layer DeepCross [31] — our 1-layer NFM
shows consistent improvements with a much simpler structure
and fewer model parameters. Our implementation is available at:
htps://github.com/hexiangnan/neural factorization machine.
Te main contributions of this work are summarized as follows.
(1) To the best of our knowledge, we are the ﬁrst to introduce the
Bi-Interaction pooling operation in neural network modelling,
and present a new neural network view for FM.
(2) Based on this new view, we develop a novel NFM model to
deepen FM under the neural network framework for learning
higher-order and non-linear feature interactions.
(3) We conduct extensive experiments on two real-world tasks to
study the Bi-Interaction pooling and NFM model, demonstrat-
ing the eﬀectiveness of NFM and great promise in using neural
networks for prediction under sparse setings.
2
MODELLING FEATURE INTERACTIONS
Due to the large space of combinatorial features, traditional solu-
tions typically rely on manual feature engineering eﬀorts or feature
selection techniques like boosted decision trees to select impor-
tant feature interactions. One limitation of such solutions is that
they cannot generalize to combinatorial features that have not
appeared in the training data. In recent years, embedding-based
methods become increasingly popular, which try to learn feature
interactions from raw data [9, 18, 24, 30, 31, 44]. By embedding
high-dimensional sparse features into a low-dimensional latent
space, the model can generalize to unseen feature combinations.
Regardless of domain, we can categorize the approaches into two
types: 1) factorization machine-based linear models, and 2) neu-
ral network-based non-linear models. In what follows, we shortly
recapitulate the two representative techniques.
2.1
Factorization Machines
Factorization machines are originally proposed for collaborative
recommendation [27, 30]. Given a real valued feature vector x ∈Rn,
FM estimates the target by modelling all interactions between each
pair of features via factorized interaction parameters:
ˆyF M(x) = w0 +
n
Õ
i=1
wixi +
n
Õ
i=1
n
Õ
j=i+1
vT
i vj · xixj,
(1)
where w0 is the global bias, wi models the interaction of the i-th
feature to the target. Te vT
i vj term denotes the factorized inter-
action, where vi ∈Rk denotes the embedding vector for feature
i, and k is the size of embedding vector, also termed as number of
latent factors in literature. Note that due to the coeﬃcient xixj,
only interactions of non-zero features are considered.
One main power of FM stems from its generality — in contrast
to matrix factorization (MF) that models the relation of two entities
only [17], FM is a general predictor working with any real valued
feature vector for supervised learning. Clearly, it enhances lin-
ear/logistic regression (LR) using the second-order factorized inter-
actions between features. By specifying input features, Rendle [27]
showed that FM can mimic many speciﬁc factorization models
such as the standard MF, parallel factor analysis, and SVD++ [22],
Owing to such genericity, FM has been recognized as one of the
most eﬀective embedding methods for sparse data prediction. It has
been successfully applied to many predictive tasks, ranging from
online advertising [21], microblog retrieval [26], to open relation
extraction [25].
2.1.1
Expressiveness Limitation of FM. Despite eﬀective-
ness, we point out that FM still belongs to the family of (multivari-
ate) linear models. In other words, the predicted target ˆy(x) is linear
w.r.t. each model parameter [28]. Formally, for each model parame-
ter θ ∈{w0, {wi }, {vif }}, we can have ˆy(x) = д + hθ, where д and
h are expressions independent of θ. Unfortunately, the underlying
structure of real-world data is ofen highly non-linear and cannot
be accurately approximated by linear models [12]. As such, FM
may suﬀer from insuﬃcient representation ability for modelling
real data with complex inherent structure and regularities.
In terms of methodology, many variants [18, 21, 24, 38] of FM
have been developed. For example, Hong et al. [18] proposed Co-
FMs to learn from multi-view data; Oentaryo et al. [24] encoded
prior knowledge of features to FM by designing a hierarchical
regularizer; and Lin et al. [21] proposed ﬁeld-aware FM, which
learned multiple embedding vectors for a feature to diﬀerentiate its
interaction with features of diﬀerent ﬁelds. More recently, Xiao et
al. [38] proposed atentional FM, using an atention network [7, 39]
to learn the importance of each feature interaction. However, these
variants are all linear extensions of FM and model the second-order
feature interactions only. As such, they can suﬀer from the same
expressiveness issue for modelling real-world data.
In this work, we contribute improvements on the expressiveness
of FM by endowing it the ability of non-linearity modelling. Te
idea is to perform non-linear transformation on the latent space

0
0.1
0.2
0.3
0.4
0.5
0
20
40
60
80
100
RMSE 
Epoch
(a) Random initialization
0
0.1
0.2
0.3
0.4
0.5
0
20
40
60
80
100
RMSE 
Epoch
Wide&Deep(test)
Wide&Deep(train)
DeepCross(test)
DeepCross(train)
LibFM(test)
(b) FM as pre-training
Figure 1:
Training and test error of each epoch of
Wide&Deep and DeepCross on Frappe. Random initializa-
tion of embeddings leads to poor performance, worse than
LibFM (a). Initializing using the embeddings learned by FM
improves the performance signiﬁcantly (b). Tis reveals op-
timization diﬃculties for training deep neural networks.
of the second-order feature interactions; meanwhile, higher-order
feature interactions can be captured.
2.2
Deep Neural Networks
In recent ﬁve years, deep neural networks (DNNs) have achieved
immense success and have been widely used on speech recognition,
computer vision and natural language processing. However, the
use of DNNs is not as widespread among the IR and DM commu-
nity. In our view, we think one reason might be that most data
of IR and DM tasks are naturally sparse, such as user behaviors,
documents/queries and various features converted from categorical
variables. Although DNNs have exhibited strong ability to learn
paterns from dense data [14], the use of DNNs on sparse data has
received less scrutiny, and it is unclear how to employ DNNs for
eﬀectively learning feature interactions under sparse setings.
Until very recently, some work [6, 9, 16, 31, 44] started to explore
DNNs for some scenarios of sparse predictive analytics. In [16],
He et al. presented a neural collaborative ﬁltering (NCF) frame-
work to learn interactions between users and items. Later, the
NCF framework was extended to model atribute interactions for
atribute-aware CF [37]. However, their methods are only applica-
ble to learn interactions between two entities and do not directly
support the general seting of supervised learning. Zhang et al. [44]
developed a FM-supported Neural Network (FNN), which uses the
feature embeddings learned by FM to initialize DNNs. Cheng et
al. [9] proposed Wide&Deep for App recommendation, where the
deep part is a multi-layer perceptron (MLP) on the concatenation
of feature embedding vectors to learn feature interactions. Shan et
al. [31] proposed DeepCross for ads prediction, which shares a
similar framework with Wide&Deep by replacing the MLP with
the state-of-the-art residual network [14].
2.2.1
Optimization Diﬀiculties of DNN. It is worthwhile to
mention the common structure of these neural network-based ap-
proaches, i.e., stacking multiple layers above the concatenation of
embedding vectors to learn feature interactions. Te expectation
is that the multiple layers can learn combinatorial features of ar-
bitrary orders in an implicit way [31]. However, we ﬁnd a key
weakness of such an architecture is that simply concatenating fea-
ture embedding vectors carries too litle information about feature
interactions in the low level. An empirical evidence is from He et
al.’s recent work [16], which shows that simply concatenating user
and item embedding vectors leads to very poor results for collabora-
tive ﬁltering. To remedy this, one has to rely on the following deep
layers to learn meaningful interaction function. While it is claimed
that multiple non-linear layers are able to learn feature interactions
well [9, 31], such a deep architecture can be diﬃcult to optimize
in practice due to the notorious problems of vanishing/exploding
gradients, overﬁting, degradation, among others [14].
To demonstrate optimization diﬃculties of DNNs empirically,
we plot the training and test error of each epoch of Wide&Deep and
DeepCross on the Frappe data in Figure 1. We use the same architec-
ture and parameters as reported in their papers, where Wide&Deep
applies a 3-layer tower structure and DeepCross uses a 10-layer
residual network with successively decreasing hidden units. From
Figure 1a, we can see that training the two models from scratch
leads to a performance much worse than the shallow FM model.
For Wide&Deep, the training error is relatively high, which is likely
because of the degradation problem [14]. For DeepCross, the very
low training error but high test error implies that the model is over-
ﬁting. Inspired by FNN [44], we further explore the use of feature
embeddings learned by FM to initialize DNNs, which can be seen
as a pre-training step. As can be seen from Figure 1b, both models
achieve much beter performance (over 11% improvements). For
Wide&Deep, the degradation problem is well addressed, evidenced
by the much lower training error, and the test performance is beter
than LibFM. However for the 10-layer DeepCross, it still suﬀers
from severe overﬁting and underperforms LibFM. Tese relatively
negative results reveal optimization diﬃculties for training DNNs.
In this work, we present a new paradigm of neural networks for
sparse data prediction. Instead of concatenating feature embedding
vectors, we propose a new Bi-Interaction operation that models
the second-order feature interactions. Tis results in a much more
informative representation in the low level, greatly helping the
subsequent non-linear layers to learn higher-order interactions.
3
NEURAL FACTORIZATION MACHINES
We ﬁrst present the NFM model that uniﬁes the strengths of FMs
and neural networks for sparse data modelling. We then discuss
the learning procedure and how to employ some useful techniques
in neural networks — dropout and batch normalization — for NFM.
3.1
Te NFM Model
Similar to factorization machine, NFM is a general machine learner
working with any real valued feature vector. Given a sparse vector
x ∈Rn as input, where a feature value xi = 0 means the i-th feature
does not exist in the instance, NFM estimates the target as:
ˆyN F M(x) = w0 +
n
Õ
i=1
wixi + f (x),
(2)
where the ﬁrst and second terms are the linear regression part
similar to that for FM, which models global bias of data and weight

0
1
0
1
0
0
0.2
……
Input Feature Vector (sparse)
Embedding Layer
v2
v4
v7
Bi-Interaction Pooling
…
B-Interaction Layer
Hidden Layers
Layer 1
Layer 2
… …
Layer L
ŷ
Prediction Score
Figure 2: Neural Factorization Machines model (the ﬁrst-
order linear regression part is not shown for clarity).
of features. Te third term f (x) is the core component of NFM
for modelling feature interactions, which is a multi-layered feed-
forward neural network as shown in Figure 2. In what follows, we
elaborate the design of f (x) layer by layer.
Embedding Layer. Te embedding layer is a fully connected
layer that projects each feature to a dense vector representation.
Formally, let vi ∈Rk be the embedding vector for the i-th feature.
Afer embedding, we obtain a set of embedding vectors Vx =
{x1v1, ...,xnvn} to represent the input feature vector x. Owing to
sparse representation of x, we only need to include the embedding
vectors for non-zero features, i.e., Vx = {xivi } where xi , 0. Note
that we have rescaled an embedding vector by its input feature
value, rather than simply an embedding table lookup, so as to
account for the real valued features [27].
Bi-Interaction Layer. We then feed the embedding set Vx into
the Bi-Interaction layer, which is a pooling operation that converts
a set of embedding vectors to one vector:
fBI (Vx) =
n
Õ
i=1
n
Õ
j=i+1
xivi ⊙xjvj,
(3)
where ⊙denotes the element-wise product of two vectors, that is,
(vi ⊙vj)k = vikvjk. Clearly, the output of Bi-Interaction pooling
is a k-dimension vector that encodes the second-order interactions
between features in the embedding space.
It is worth pointing out that our proposal of Bi-Interaction pool-
ing does not introduce extra model parameter, and more impor-
tantly, it can be eﬃciently computed in linear time. Tis prop-
erty is the same with average/max pooling and concatenation
that are rather simple but commonly used in neural network ap-
proaches [16, 31, 36]. To show the linear time complexity of evalu-
ating Bi-Interaction pooling, we reformulate Equation (3) as:
fBI (Vx) = 1
2
"
(
n
Õ
i=1
xivi)2 −
n
Õ
i=1
(xivi)2
#
,
(4)
where we use the symbol v2 to denote v ⊙v. By considering the
sparsity of x, we can actually perform Bi-Interaction pooling in
O(kNx) time, where Nx denotes the number of non-zero entries in
x. Tis is a very appealing property, meaning that the beneﬁt of
Bi-Interaction pooling in modelling pairwise feature interactions
does not involve any additional cost.
Hidden Layers. Above the Bi-Interaction pooling layer is a
stack of fully connected layers, which are capable of learning higher-
order interactions between features [31]. Formally, the deﬁnition
of fully connected layers is as follows:
z1 = σ1(W1fBI (Vx) + b1),
z2 = σ2(W2z1 + b2),
......
zL = σL(WLzL−1 + bL),
(5)
where L denotes the number of hidden layers, Wl , bl and σl denote
the weight matrix, bias vector and activation function for the l-th
layer, respectively. By specifying non-linear activation functions,
such as sigmoid, hyperbolic tangent (tanh), and Rectiﬁer (ReLU),
we allow the model to learn higher-order feature interactions in
a non-linear way. Tis is advantageous to existing methods for
higher-order interaction learning, such as higher-Order FM [3] and
Exponential Machines [23], which only support the learning of
higher-order interactions in a linear way. As for the structure of
fully connected layers (i.e., size of each layer), one can freely choose
tower [9, 16], constant [31], and diamond [44], among others.
Prediction Layer. At last, the output vector of the last hidden
layer zL is transformed to the ﬁnal prediction score:
f (x) = hT zL,
(6)
where vector h denotes the neuron weights of the prediction layer.
To summarize, we give the formulation NFM’s predictive model as:
ˆyN F M(x) = w0 +
n
Õ
i=1
wixi
+ hT σL(WL(...σ1(W1fBI (Vx) + b1)...) + bL),
(7)
with all model parameters Θ = {w0, {wi, vi }, h, {Wl, bl }}. Com-
pared to FM, the additional model parameters of NFM are mainly
{Wl, bl }, which are used for learning higher-order interactions
between features. In remainder of this subsection, we ﬁrst show
how NFM generalizes FM and discuss the connection of NFM be-
tween existing deep learning methods; we then analyze the time
complexity of evaluating NFM model.
3.1.1
NFM Generalizes FM. FM is a shallow and linear model,
which can be seen as a special case of NFM with no hidden layer.
To show this, we set L to zero and directly project the output of
Bi-Interaction pooling to prediction score. We term this simpliﬁed
model as NFM-0, which is formulated as:
ˆyN F M−0 = w0 +
n
Õ
i=1
wixi + hT
n
Õ
i=1
n
Õ
j=i+1
xivi ⊙xjvj
= w0 +
n
Õ
i=1
wixi +
n
Õ
i=1
n
Õ
j=i+1
k
Õ
f =1
hf vif vjf · xixj.
(8)
As can be seen, by ﬁxing h to a constant vector of (1, ..., 1), we can
exactly recover the FM model2.
2Note that for NFM-0, a trainable h can not improve the expressiveness of FM, since
its impact on prediction can be absorbed into feature embeddings.

It is worth pointing out that, to our knowledge, this is the ﬁrst
time FM has been expressed under the neural network framework.
While a recent work by Blondel et al. [4] has uniﬁed FM and Poly-
nomial network via kernelization, their kernel view of FM only
provides a new training algorithm and does not provide insight for
improving the FM model. Our new view of FM is highly instructive
and provides more insight for improving FM. Particularly, we allow
the use of various neural network techniques on FM to improve
its learning and generalization ability. For example, we can use
dropout [33] — a well-known technique in deep learning commu-
nity to prevent overﬁting — on the Bi-Interaction layer as a way
to regularize FM, which we ﬁnd can be even more eﬀective than
the conventional L2 regularization (see Figure 3 of Section 4).
3.1.2
Relation to Wide&Deep and DeepCross. NFM has
a similar multi-layered neural architecture with several existing
deep learning solutions [9, 16, 31]. Te key diﬀerence is in the
Bi-Interaction pooling component, which is uniquely used in NFM.
Speciﬁcally, if we replace the Bi-Interaction pooling with concate-
nation and apply a tower-structure MLP (or residual units [14])
to hidden layers, we can recover the Wide&Deep (or DeepCross)
model. An obvious limitation of the concatenation operation is that
it does not account for any interaction between features. As such,
these deep learning approaches have to rely entirely on the fol-
lowing deep layers to learn meaningful feature interactions, which
unfortunately can be diﬃcult to train in practice (cf. Section 2.2.1).
Our use of Bi-Interaction pooling captures second-order feature
interactions in the low level, which is more informative than the
concatenation operation. Tis greatly facilitates the subsequent hid-
den layers of NFM to learn useful higher-order feature interactions
in a much easier way.
3.1.3
Time Complexity Analysis. We have shown in Equa-
tion (4) that Bi-Interaction pooling can be eﬃciently computed in
O(kNx) time, which is the same as FM. Ten the additional costs
are caused by the hidden layers. For hidden layer l, the matrix-
vector multiplication is the main operation which can be done in
O(dl−1dl), where dl denotes the dimension of the l-th hidden layer
and d0 = k. Te prediction layer only involves inner product of two
vectors, for which the complexity isO(dL). As such, the overall time
complexity for evaluating a NFM model is O(kNx + ÍL
l=1 dl−1dl),
which is the same as that of Wide&Deep and DeepCross.
3.2
Learning
NFM can be applied to a variety of prediction tasks, including re-
gression, classiﬁcation and ranking. To estimate model parameters
of NFM, we need to specify an objective function to optimize. For
regression, a commonly used objective function is the squared loss:
Lreд =
Õ
x∈X
(ˆy(x) −y(x))2,
(9)
where X denotes the set of instances for training, and y(x) denotes
the target of instance x. Te regularization terms are optional
and omited here, since we found that some techniques in neural
network modelling such as dropout can well prevent NFM from
overﬁting. For classiﬁcation task, we can optimize the hinge loss
or log loss [16]. For ranking task, we can optimize pairwise person-
alized ranking loss [29, 37] or contrastive max-margin loss [43]. In
this work, we focus on the regression task and optimize the squared
loss of Equation (9). Te optimization for ranking/classiﬁcation
tasks can be done in the same way.
Stochastic gradient descent (SGD) is a universal solver for opti-
mizing neural network models. It iteratively updates the parameters
until convergence. In each time, it randomly selects a training in-
stance x, updating each model parameter towards the direction of
its negative gradient:
θ = θ −η · 2(ˆy(x) −y(x))d ˆy(x)
dθ
,
(10)
where θ ∈Θ is a trainable model parameter, and η > 0 is the
learning rate that controls the step size of gradient descent. As
NFM is a multi-layered neural network model, the gradient of ˆy(x)
w.r.t. to each model parameter can be derived using the chain
rule. Here we give only the diﬀerentiation of the Bi-Interaction
pooling layer, since other layers are just standard operations in
neural network modelling and have been widely implemented in
ML toolkits like TensorFlow and Keras.
d fBI (Vx)
dvi
= (
n
Õ
j=1
xjvj)xi −x2
i vi =
n
Õ
j=1,j,i
xixjvj.
(11)
As such, for end-to-end neural methods, upon plugging in the Bi-
Interaction pooling layer, they can still be learned end-to-end. To
leverage the vectorization and parallelism speedup of modern com-
puting platforms, mini-batch SGD is more widely used in practice,
which samples a batch of training instances and updates model
parameters based on the batch. In our implementation, we use
mini-batch Adagrad [10] as the optimizer, rather than the vanilla
SGD. Its main advantage is that the learning rate can be self adapted
during the training phase, which eases the pain of choosing a proper
learning rate and leads to faster convergence than the vanilla SGD.
3.2.1
Dropout. While neural network models have strong rep-
resentation ability, they are also easy to overﬁt the training data.
Dropout [33] is a regularization technique for neural networks to
prevent overﬁting. Te idea is to randomly drop neurons (along
with their connections) of the neural network during training. Tat
is, in each parameter update, only part of the model parameters that
contributes to the prediction of ˆy(x) will be updated. Trough this
process, it can prevent complex co-adaptations of neurons on train-
ing data. It is important to note that in the testing phase, dropout
must be disabled and the whole network is used for estimating ˆy(x).
As such, dropout can also be seen as performing model averaging
with smaller neural networks [33].
In NFM, to avoid feature embeddings co-adapt to each other and
overﬁt the data, we propose to adopt dropout on the Bi-Interaction
layer. Speciﬁcally, afer obtaining fBI (Vx) which is ak-dimensional
vector of latent factors, we randomly drop ρ percent of latent fac-
tors, where ρ is termed as the dropout ratio. Since NFM with no
hidden layer degrades to the FM model, it can be seen as a new
way to regularize FM. Moreover, we also apply dropout on each
hidden layer of NFM to prevent the learning of higher-order feature
interactions from co-adaptations and overﬁting.
3.2.2
Batch Normalization. One diﬃculty of training multi-
layered neural networks is caused by the fact of covariance shif [20].
It means that the distribution of each layer’s inputs changes during

training, as the parameters of the previous layers change. As a
result, the later layer needs to adapt to these changes (which are
ofen noisy) when updating its parameters, which adversely slows
down the training. To address the problem, Ioﬀe and Szegedy [20]
proposed batch normalization (BN), which normalizes layer inputs
to a zero-mean unit-variance Gaussian distribution for each training
mini-batch. It has been shown that BN leads to faster convergence
and beter performance in several computer vision tasks [14, 42].
Formally, let the input vector to a layer be xi ∈Rd and all
input vectors to the layer of the mini-batch be B = {xi }, then BN
normalizes xi as:
BN(xi) = γ ⊙(xi −µB
σB
) + β,
(12)
where µB =
1
|B |
Í
i ∈B xi denotes the mini-batch mean, σ2
B =
1
|B |
Í
i ∈B(xi −µB)2 denotes the mini-batch variance, and γ and β
are trainable parameters (vectors) to scale and shif the normalized
value to restore the representation power of the network. Note
that in testing, BN also needs to be applied, where µB and σB are
estimated from the whole training data.
In NFM, to avoid the update of feature embeddings changing the
input distribution to hidden layers or prediction layer, we perform
BN on the output of the Bi-Interaction pooling. For each successive
hidden layer, the BN is also applied.
4
EXPERIMENTS
As the key contributions of this work are on Bi-Interaction pooling
in neural network modelling and the design of NFM for sparse
data prediction, we conduct experiments to answer the following
research questions:
RQ1 Can Bi-Interaction pooling eﬀectively capture the second-
order feature interactions? How does dropout and batch
normalization work for Bi-Interaction pooling?
RQ2 Are hidden layers of NFM useful for capturing higher-
order interactions between features and improving the
expressiveness of FM?
RQ3 How does NFM perform as compared to higher-order FM
and the state-of-the-art deep learning methods Wide&Deep
and DeepCross?
In what follows, we ﬁrst present the experimental setings, fol-
lowed by answering the above research questions one by one.
4.1
Experimental Settings
4.1.1
Datasets. We experimented with two publicly accessible
datasets: Frappe3 and MovieLens4:
1. Frappe. Frapp´e is a context-aware app discovery tool. Tis
dataset is constructed by Baltrunas et al. [1]. It contains 96, 203 app
usage logs of users under diﬀerent contexts. Besides user ID and
app ID, each log contains 8 context variables, including weather,
city and daytime (e.g., morning or afernoon). We converted each
log (i.e., user ID, app ID and all context variables) to a feature vector
using one-hot encoding, resulting in 5, 382 features in total. A target
value of 1 means the user has used the app under the context.
3htp://baltrunas.info/research-menu/frappe
4htp://grouplens.org/datasets/movielens/latest
Table 1: Statistics of the evaluation datasets.
Dataset
Instance#
Feature#
User#
Item#
Frappe
288,609
5, 382
957
4,082
MovieLens
2,006,859
90, 445
17,045
23,743
2. MovieLens. Tis is the Full version of the latest Movie-
Lens data published by GroupLens [13]. As this work concerns
higher-order interactions between features, we study the task of
personalized tag recommendation rather than collaborative ﬁlter-
ing [16] that considers the second-order interactions only. Te
tagging part of the data includes 668, 953 tag applications of 17, 045
users on 23, 743 items with 49, 657 distinct tags. We converted each
tag application (i.e., user ID, movie ID and tag) to a feature vector,
resulting in 90, 445 features in total. A target value of 1 means the
user has assigned the tag to the movie.
As both original datasets contain positive instances only (i.e., all
instances have target value 1), we sampled two negative instances
to pair with one positive instance to ensure the generalization of
the predictive model. For each log of Frappe, we randomly sampled
two apps that the user has not used in the context; for each tag
application of MovieLens, we randomly sampled two tags that
the user has not assigned to the movie. Each negative instance is
assigned to a target value of −1. Table 1 summarizes the statistics
of the ﬁnal evaluation datasets.
4.1.2
Evaluation Protocols. We randomly split the dataset
into training (70%), validation (20%), and test (10%) sets. Te val-
idation set was used for tuning hyper-parameters and the ﬁnal
performance comparison was conducted on the test set. Te study
of NFM properties (i.e., the answering of RQ1 and RQ2) was per-
formed on the validation set, which can also reﬂect how we choose
the optimal hyper-parameters. To evaluate the performance, we
adopted root mean square error (RMSE), where a lower RMSE score
indicates a beter performance. Note that RMSE has been widely
used for evaluating regression tasks such as recommendation with
explicit ratings [5, 30] and click-through rate prediction [24]. We
rounded up the prediction of each model to 1 or −1 if it was out
of the range. Te one-sample paired t-test was performed to judge
the statistical signiﬁcance where necessary.
4.1.3
Baselines. We implemented NFM using TensorFlow5.
We compared with the following competitive embedding-based
models that are speciﬁcally designed for sparse data prediction:
- LibFM [28]. Tis is the oﬃcial implementation6 of FM released
by Rendle. It has shown strong performance for personalized tag
recommendation and context-aware prediction [30]. We used the
SGD learner for a fair comparison with other methods which were
all optimized with SGD (or its variants).
- HOFM. Tis is the TensorFlow implementation7 of higher-
order FM, as described in [27]. We experimented with order size 3,
since the MovieLens data concerns the ternary relationship between
users, movies and tags.
- Wide&Deep [9]. As introduced in Section 2.2, the deep part
ﬁrst concatenates feature embeddings, followed by a MLP to model
5Codes are available at htps://github.com/hexiangnan/neural factorization machine
6htp://www.libfm.org/
7htps://github.com/geﬀy/tﬀm

0.3
0.35
0.4
0.45
0.5
0.55
0.6
0/0
0.2/1e-5 0.4/1e-4 0.6/1e-3 0.8/1e-2
RMSE (validation)
dropout_ratio / reg
Frappe
LR
NFM-0(reg)
NFM-0(dropout)
0.47
0.49
0.51
0.53
0.55
0.57
0.59
0/0
0.2/1e-5 0.4/1e-4 0.6/1e-3 0.8/1e-2
RMSE (validation)
dropout_retio / reg
MovieLens
LR
NFM-0(reg)
NFM-0(dropout)
Figure 3: Validation error of NFM-0 w.r.t. dropout on the Bi-
Interaction layer and L2 regularization on embeddings.
feature interactions. As the structure of a DNN is diﬃcult to be fully
tuned, we used the same structure as reported in their paper, which
has three layers with size 1024, 512 and 256, respectively. While
the wide part (which is a linear regression model) is subjected to
design to incorporate cross features, we used the raw features only
for a fair comparison with FM and NFM.
- DeepCross [31]. It applies a multi-layered residual network
on the concatenation of feature embeddings for learning feature
interactions. We used the same structure as reported in their paper,
which stacks 5 residual units (each unit has two layers) with the
hidden dimension 512, 512, 256, 128 and 64, respectively.
4.1.4
Parameter Setings. To fairly compare models’ capabil-
ity, we learned all models by optimizing the square loss (Equation
(9)). Te learning rate was searched in [0.005, 0.01, 0.02, 0.05] for all
methods. To prevent overﬁting, we tuned the L2 regularization for
linear models LibFM and HOFM in [1e−6, 5e−6, 1e−5, ..., 1e−1], and
the dropout ratio for neural network models Wide&Deep, Deep-
Cross and NFM in [0, 0.1, 0.2, ..., 0.9]. Note that we found that
dropout can well regularize the hidden layers of Wide&Deep and
NFM; however it did not work well for the residual units of Deep-
Cross. Besides LibFM that optimized FM with the vanilla SGD,
all other methods were optimized with mini-batch Adagrad [10],
where the batch size was set to 128 for Frappe and 4096 for Movie-
Lens. Note that the batch size was selected by considering both
training time and convergence rate, as a larger batch size usually
led to faster training per epoch but slower convergence. For all
methods, the early stopping strategy was performed, where we
stopped training if the RMSE on validation set increased for 4 suc-
cessive epochs. Without special mention, we show the results of
embedding size 64, and more results of larger embedding sizes are
shown in Section 4.4.
4.2
Study of Bi-Interaction Pooling (RQ1)
We empirically study the Bi-Interaction pooling operation. To avoid
other components (e.g., hidden layers) aﬀecting the analysis, we
study the NFM-0 model that directly projects the output of Bi-
Interaction pooling to prediction score with no hidden layer. As
discussed in Section 3.1.1, NFM-0 is identical to FM as the trainable h
does not impact model’s expressiveness. We ﬁrst compare dropout
with traditional L2 regularization for preventing model overﬁting,
and then explore the impact of batch normalization.
4.2.1
Dropout Improves Generalization. Figure 3 shows the
validation error of NFM-0 w.r.t. dropout ratio on the Bi-Interaction
0
0.1
0.2
0.3
0.4
0.5
0.6
0
20
40
60
80
100
RMSE 
Epoch
Frappe
0
0.1
0.2
0.3
0.4
0.5
0.6
0
20
40
60
80
100
RMSE 
Epoch
MovieLens
Dropout=0(validation)
Dropout=0(train)
Dropout=0.3(validation)
Dropout=0.3(train)
Figure 4: Training and validation error of each epoch of
NFM-0 with and without dropout on Bi-Interaction layer.
0.1
0.2
0.3
0.4
0.5
0.6
0
20
40
60
80
100
RMSE 
Epoch
Frappe
0
0.1
0.2
0.3
0.4
0.5
0.6
0
20
40
60
80
100
RMSE 
Epoch
MovieLens
Dropout(train)
Dropout(validation)
Dropout+BN(train)
Dropout+BN(validation)
Figure 5: Training and validation error of each epoch of
NFM-0 with and without BN on the Bi-Interaction layer.
layer and L2 regularization on feature embeddings. Te perfor-
mance of linear regression (LR) is also shown for benchmarking
the performance of prediction that does not consider feature inter-
actions. First, LR leads to very poor performance, highlighting the
importance of modelling interactions between sparse features for
prediction. Second, we see that both L2 regularization and dropout
can well prevent overﬁting and improve NFM-0’s generalization
to unseen data. Between the two strategies, dropout oﬀers beter
performance. Speciﬁcally, on Frappe, using a dropout ratio of 0.3
leads to a lowest validation error of 0.3562, which is signiﬁcantly
beter than that of L2 regularization 0.3799. One reason might be
that enforcing L2 regularization only suppresses the values of pa-
rameters in each update numerically, while using dropout can be
seen as ensembling multiple sub-models [33], which can be more
eﬀective. Considering the genericity of FM that subsumes many
factorization models, we believe this is a new interesting ﬁnding,
meaning that dropout can also be an eﬀective strategy to address
overﬁting of linear latent-factor models.
To be more clear about the eﬀect of dropout, we show the train-
ing and validation error of each epoch of NFM-0 with and without
dropout in Figure 4. Both datasets show that with a dropout ratio
of 0.3, although the training error is higher, the validation error be-
comes lower. Tis demonstrates the ability of dropout in preventing
overﬁting and as such, beter generalization can be achieved.
4.2.2
Batch Normalization Speeds up Training. Figure 5
shows the training and validation error of each epoch of NFM-0
with and without BN on the Bi-Interaction layer. Te dropout is

0.3
0.32
0.34
0.36
0.38
0.4
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
RMSE (validation)
Dropout-1
Frappe
LibFM
NFM-0
NFM-identity
NFM-tanh
NFM-sigmoid
NFM-ReLU
0.46
0.47
0.48
0.49
0.5
0.51
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
RMSE (validation)
Dropout-1
MovieLens
LibFM
NFM-0
NFM-identity
NFM-tanh
NFM-sigmoid
NFM-ReLU
Figure 6: Validation error of LibFM, NFM-0 and NFM with
diﬀerent activation functions on the ﬁrst hidden layer.
enabled with a ratio of 0.3, and the learning rate is set to 0.02. Fo-
cusing on the training error, we can see that BN leads to a faster
convergence; on Frappe, when BN is applied, the training error
of epoch 20 is even lower than that of epoch 60 without BN; and
the validation error indicates that the lower training error is not
overﬁting — in fact, [14, 20] showed that by addressing the in-
ternal covariate shif with BN, the model’s generalization ability
can be improved. Our result also veriﬁes this point, where using
BN leads to slight improvement (although the improvement is not
statistically signiﬁcant). Furthermore, we notice that BN makes
the learning less stable, as evidenced by the larger performance
ﬂuctuation of blue lines. Tis is caused by our use of dropout and
BN together, as randomly dropping neurons can change the in-
put distribution normalized by BN. It is an interesting direction to
eﬀectively combine BN and dropout.
4.3
Impact of Hidden Layers (RQ2)
Te hidden layers of NFM play a pivotal role in capturing higher-
order interactions between features. To explore the impact, we ﬁrst
add one hidden layer above the Bi-Interaction layer and slightly
overuse the name NFM to indicate this speciﬁc model. To ensure
the same model capability with NFM-0, we set the size of hidden
layer the same as the embedding size.
Figure 6 shows the validation error of NFM w.r.t. diﬀerent ac-
tivation functions and dropout ratios for the hidden layer. Te
performance of LibFM and NFM-0 are also shown for benchmark-
ing purposes. First and foremost, we observe that by using non-
linear activations, NFM’s performance is improved with a large
margin — compared to NFM-0 which has a similar performance
with LibFM, the relative improvement is 11.3% and 5.2% for Frappe
and MovieLens, respectively. Tis highlights the importance of
modelling higher-order feature interactions for quality prediction.
Among the diﬀerent non-linear activation functions, there is no
obvious winner. Second, when we use the identity function as the
activation function, i.e., the hidden layer performs a linear transfor-
mation, NFM does not perform that well. Tis provides evidence
to the necessity of learning higher-order feature interactions with
non-linear functions.
To see whether a deeper NFM can further improve the perfor-
mance, we stack more ReLU layers above the Bi-Interaction layer.
As it is computationally expensive to tune the size and dropout
ratio for each hidden layer separately, we use the same seting for
all layers and tune them the same way as NFM-1. As can be seen
0.1
0.2
0.3
0.4
0.5
0.6
0
20
40
60
80
100
RMSE 
Epoch
Frappe
NFM(train)
NFM(validation)
NFM-pre(train)
NFM-pre(validation)
0.1
0.2
0.3
0.4
0.5
0.6
0
20
40
60
80
100
RMSE 
Epoch
MovieLens
NFM(train)
NFM(validation)
NFM-pre(train)
NFM-pre(validation)
Figure 7: Training and validation error of each epoch of
NFM-1 with and without pre-training.
from Table 2, when we stack more layers, the performance is not
further improved, and best performance is when we use one hidden
layer only. We have also explored other designs for hidden layers,
such as the tower structure and residual units, however, the per-
formance is still not improved. We think the reason is because the
Bi-Interaction layer has encoded informative second-order feature
interactions, and based on which, a simple non-linear function is
suﬃcient to capture higher-order interactions. To verify this, we
replaced the Bi-Interaction layer with concatenation (which leads
to the same architecture as Wide&Deep), and found that the perfor-
mance can be gradually improved with more hidden layers (up to
three); however, the best performance achievable is still inferior to
that of NFM-1. Tis demonstrates the value of using a more infor-
mative operation for low-level layers, which can ease the burden
of higher-level layers for learning meaningful information. As a
result, a deep structure becomes not necessarily required.
Table 2: NFM w.r.t. diﬀerent number of hidden layers.
Methods
Frappe
MovieLens
NFM-0
0.3562
0.4901
NFM-1
0.3133
0.4646
NFM-2
0.3193
0.4681
NFM-3
0.3219
0.4752
NFM-4
0.3202
0.4703
4.3.1
Pre-training Speeds up Training. It is known that pa-
rameter initialization can greatly aﬀect the convergence and per-
formance of DNNs [11, 16], since gradient-based methods can only
ﬁnd local optima for DNNs. As have been shown in Section 2.2.1,
initializing with feature embeddings learned by FM can signiﬁcantly
enhance Wide&Deep and DeepCross. Now the question arises, how
does pre-training impact NFM?
Figure 7 shows the state of each epoch of NFM-1 with and with-
out pre-training. First, we can see that by using FM embeddings
as pre-training, NFM exhibits extremely fast convergence — on
both datasets, with 5 epochs only, the performance is on par with
40 epochs of NFM that is trained from scratch (with BN enabled).
Second, we ﬁnd that pre-training does not improve NFM’s ﬁnal
performance, and a random initialization can achieve a result that
is slightly beter than that with pre-training. Tis demonstrates
the robustness of NFM, which is relatively insensitive to parameter
initialization. In contrast to the huge impact of pre-training on
Wide&Deep and DeepCross (cf. Figure 1) that improves both their
convergence and ﬁnal performance, we draw the conclusion that

Table 3: Test error and number of trainable parameters for diﬀerent methods on latent factors 128 and 256. M denotes “mil-
lion”; ∗and ∗∗denote the statistical signiﬁcance for p < 0.05 and p < 0.01, respectively, compared to the best baseline.
Frappe
MovieLens
Factors=128
Factors=256
Factors=128
Factors=256
Method
Param#
RMSE
Param#
RMSE
Param#
RMSE
Param#
RMSE
LibFM [28]
0.69M
0.3437
1.38M
0.3385
11.67M
0.4793
23.24M
0.4735
HOFM
1.38M
0.3405
2.76M
0.3331
23.24M
0.4752
46.40M
0.4636
Wide&Deep [9]
2.66M
0.3621
4.66M
0.3661
12.72M
0.5323
24.69M
0.5313
Wide&Deep (pre-train)
2.66M
0.3311
4.66M
0.3246
12.72M
0.4595
24.69M
0.4512
DeepCross [31]
4.47M
0.4025
8.93M
0.4071
12.71M
0.5885
25.42M
0.5907
DeepCross (pre-train)
4.47M
0.3388
8.93M
0.3548
12.71M
0.5084
25.42M
0.5130
NFM
0.71M
0.3127∗∗
1.45M
0.3095∗∗
11.68M
0.4557∗
23.31M
0.4443∗
0.3
0.32
0.34
0.36
0.38
0.4
16
32
64
128
256
RMSE
Factors
Frappe
LibFM
HOFM
Wide&Deep
DeepCross
NFM
0.44
0.46
0.48
0.5
0.52
0.54
16
32
64
128
256
RMSE
Factors
MovieLens
LibFM
HOFM
Wide&Deep
DeepCross
NFM
Figure 8: Performance comparison on the test set w.r.t. diﬀerent embedding sizes. LibFM, HOFM and HOFM are trained from
random initialization; Wide&Deep and DeepCross are pre-trained with FM feature embeddings.
NFM is much easier to train and optimize, which is due largely to
the informative and eﬀective Bi-Interaction pooling operation.
4.4
Performance Comparison (RQ3)
We now compare with state-of-the-art methods. For NFM, we use
one hidden layer with ReLU as the activation function, since the
baselines DeepCross and Wide&Deep also choose ReLU in their
original papers. Note that the most important hyper-parameter for
NFM is the dropout ratio, which we use 0.5 for the Bi-Interaction
layer and tune the value for the hidden layer.
Figure 8 plots the test RMSE w.r.t. diﬀerent number of latent
factors (i.e., embedding sizes), where Wide&Deep and DeepCross
are pre-trained with FM to beter explore the two methods. Table 3
shows the concrete scores obtained on factors 128 and 256, and
the number of model parameters of each method. Te scores of
Wide&Deep and DeepCross without pre-training are also shown.
We have the following three key observations.
First and foremost, NFM consistently achieves the best perfor-
mance on both datasets with the fewest model parameters be-
sides FM. Tis demonstrates the eﬀectiveness and rationality of
NFM in modelling higher-order and non-linear feature interactions
for prediction with sparse data. Te performance is followed by
Wide&Deep, which uses a 3-layer MLP to learn feature interactions.
We have also tried deeper layers for Wide&Deep, however the per-
formance has not been improved. Tis further veriﬁes the utility of
using the informative Bi-Interaction pooling in the low level.
Second, we observe that HOFM shows slight improvement over
FM with 1.45% and 1.04% average improvement on Frappe and
MovieLens, respectively. Tis sheds light on the limitation of FM
that models only the second-order feature interactions, and thus
the usefulness of modelling higher-order interactions. Meanwhile,
the large performance gap between HOFM and NFM reﬂects the
value of modelling higher-order interactions in a non-linear way,
since HOFM models higher-order interactions linearly and uses
much more parameters than NFM.
Lastly, the relatively weak performance of DeepCross reveals
that deeper learnings are not always beter, as DeepCross is the
deepest method among all baselines that utilizes a 10-layer network.
On Frappe, DeepCross only achieves a comparable performance
with the shallow FM model, while it underperforms FM signiﬁcantly
on MovieLens. We believe that the reasons are due to optimization
diﬃculties and overﬁting (as evidenced by the worse performance
on factors 128 and 256).
5
CONCLUSION AND FUTURE WORK
In this work, we proposed a novel neural network model NFM,
which brings together the eﬀectiveness of linear factorization ma-
chines with the strong representation ability of non-linear neural
networks for sparse predictive analytics. Te key of NFM’s archi-
tecture is the newly proposed Bi-Interaction operation, based on
which we allow a neural network model to learn more informative
feature interactions at the lower level. Extensive experiments on
two real-world datasets show that with one hidden layer only, NFM
signiﬁcantly outperforms FM, higher-order FM, and state-of-the-art
deep learning approaches Wide&Deep and DeepCross.

Te work represents the ﬁrst step towards bridging the gap
between linear models and deep learning. Linear models, such
as various factorization methods, have shown to be eﬀective for
many IR and DM tasks and are easy to interpret. However, their
limited expressiveness may hinder the performance when mod-
elling real-world data with complex inherent paterns. While deep
learning models have exhibited great expressive power and yielded
immense success on speech processing and computer vision, their
performance is still unsatisfactory for IR tasks, such as collaborative
ﬁltering [16]. In our view, one reason is that most data of IR and
DM tasks are naturally sparse; and to date, there still lacks eﬀective
deep learning solutions for prediction with sparse data. By connect-
ing neural networks with FM — one of the most powerful linear
models for supervised learning — we are able to design a simple
yet eﬀective deep learning solution for sparse data prediction.
With recent developments on GPU platforms, it is not technically
diﬃcult to build very deep models with hundreds or even thousands
of layers [14]. However, deeper models do not necessarily lead to
beter results, since deeper models are less transparent and more
diﬃcult to optimize and tune. As such, we expect future research
on deep learning for IR should focus more on designing beter
neural components or architectures for speciﬁc tasks, rather than
relying on deeper models for minor improvements. Our proposed
Bi-Interaction pooling is an eﬀective neural component for sparse
feature modelling, reducing the demand for deeper structure for
quality prediction. In future, we will improve the eﬃciency of
NFM by resorting to hashing techniques [32, 41] to make it more
suitable for large-scale applications and study its performance for
other IR tasks, such as search ranking and targeted advertising.
While this work endows FM with non-linearities from predictive
model perspective, another viable solution for incorporating non-
linearities is to extend the objective function with regularizers like
the graph Laplacian [15, 34]. Lastly, we are interested in exploring
the Bi-Interaction pooling for recurrent neural networks (RNNs)
for sequential data modelling.
REFERENCES
[1] L. Baltrunas, K. Church, A. Karatzoglou, and N. Oliver. Frappe: Understanding
the usage and perception of mobile app recommendations in-the-wild. CoRR,
abs/1505.03014, 2015.
[2] I. Bayer, X. He, B. Kanagal, and S. Rendle. A generic coordinate descent frame-
work for learning from implicit feedback. In WWW, 2017.
[3] M. Blondel, A. Fujino, N. Ueda, and M. Ishihata. Higher-order factorization
machines. In NIPS, 2016.
[4] M. Blondel, M. Ishihata, A. Fujino, and N. Ueda. Polynomial networks and
factorization machines: New insights and eﬃcient training algorithms. In ICML,
2016.
[5] D. Cao, X. He, L. Nie, X. Wei, X. Hu, S. Wu, and T.-S. Chua. Cross-platform app
recommendation by jointly modeling ratings and texts. ACM TOIS, 2017.
[6] J. Chen, B. Sun, H. Li, H. Lu, and X.-S. Hua. Deep ctr prediction in display
advertising. In MM, 2016.
[7] J. Chen, H. Zhang, X. He, L. Nie, W. Liu, and T.-S. Chua. Atentive collaborative
ﬁltering: Multimedia recommendation with feature- and item-level atention. In
SIGIR, 2017.
[8] T. Chen, X. He, and M.-Y. Kan. Context-aware image tweets modelling and
recommendation. In MM, 2016.
[9] H.-T. Cheng, L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye, G. Anderson,
G. Corrado, W. Chai, M. Ispir, R. Anil, Z. Haque, L. Hong, V. Jain, X. Liu, and
H. Shah. Wide & deep learning for recommender systems. In DLRS, 2016.
[10] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online
learning and stochastic optimization. Journal of Machine Learning Research,
2011.
[11] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio.
Why does unsupervised pre-training help deep learning? Journal of Machine
Learning Research, 2010.
[12] M. Genzel and G. Kutyniok. A mathematical framework for feature selection from
real-world data with non-linear observations. arXiv preprint arXiv:1608.08852,
2016.
[13] F. M. Harper and J. A. Konstan. Te movielens datasets: History and context.
ACM Transactions on Interactive Intelligent Systems, 2015.
[14] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition.
In CVPR, 2016.
[15] X. He, M. Gao, M.-Y. Kan, Y. Liu, and K. Sugiyama. Predicting the popularity of
web 2.0 items based on user comments. In SIGIR, 2014.
[16] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua. Neural collaborative
ﬁltering. In WWW, 2017.
[17] X. He, H. Zhang, M.-Y. Kan, and T.-S. Chua. Fast matrix factorization for online
recommendation with implicit feedback. In SIGIR, 2016.
[18] L. Hong, A. S. Doumith, and B. D. Davison. Co-factorization machines: Modeling
user interests and predicting individual decisions in twiter. In WSDM, 2013.
[19] R. Hong, Y. Yang, M. Wang, and X.-S. Hua. Learning visual semantic relationships
for eﬃcient visual retrieval. IEEE Transactions on Big Data, 2015.
[20] S. Ioﬀe and C. Szegedy. Batch normalization: Accelerating deep network training
by reducing internal covariate shif. In ICML, 2015.
[21] Y. Juan, Y. Zhuang, W.-S. Chin, and C.-J. Lin. Field-aware factorization machines
for ctr prediction. In RecSys, 2016.
[22] Y. Koren. Factorization meets the neighborhood: A multifaceted collaborative
ﬁltering model. In KDD, 2008.
[23] A. Novikov, M. Troﬁmov, and I. Oseledets. Exponential machines. In ICLR
Workshop, 2017.
[24] R. J. Oentaryo, E.-P. Lim, J.-W. Low, D. Lo, and M. Finegold. Predicting response
in mobile advertising with hierarchical importance-aware factorization machine.
In WSDM, 2014.
[25] F. Petroni, L. Del Corro, and R. Gemulla. Core: Context-aware open relation
extraction with factorization machines. In EMNLP, 2015.
[26] R. Qiang, F. Liang, and J. Yang. Exploiting ranking factorization machines for
microblog retrieval. In CIKM, 2013.
[27] S. Rendle. Factorization machines. In ICDM, 2010.
[28] S. Rendle. Factorization machines with libfm. ACM Transactions on Intelligent
Systems and Technology, 2012.
[29] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Tieme. Bpr: Bayesian
personalized ranking from implicit feedback. In UAI, 2009.
[30] S. Rendle, Z. Gantner, C. Freudenthaler, and L. Schmidt-Tieme. Fast context-
aware recommendations with factorization machines. In SIGIR, 2011.
[31] Y. Shan, T. R. Hoens, J. Jiao, H. Wang, D. Yu, and J. Mao. Deep crossing: Web-scale
modeling without manually crafed combinatorial features. In KDD, 2016.
[32] F. Shen, Y. Mu, Y. Yang, W. Liu, L. Liu, J. Song, and H. T. Shen. Classiﬁcation by
retrieval: Binarizing data and classiﬁer. In SIGIR, 2017.
[33] N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov.
Dropout: a simple way to prevent neural networks from overﬁting. Journal of
Machine Learning Research, 2014.
[34] M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu. Scalable semi-supervised learning
by eﬃcient anchor graph regularization. IEEE Transaction on Knowledge and
Data Engineering, 2016.
[35] M. Wang, X. Liu, and X. Wu. Visual classiﬁcation by l1-hypergraph modeling.
IEEE Transaction on Knowledge and Data Engineering, 2015.
[36] P. Wang, J. Guo, Y. Lan, J. Xu, S. Wan, and X. Cheng. Learning hierarchical
representation model for nextbasket recommendation. In SIGIR, 2015.
[37] X. Wang, X. He, L. Nie, and T.-S. Chua. Item silk road: Recommending items
from information domains to social users. In SIGIR, 2017.
[38] J. Xiao, H. Ye, X. He, H. Zhang, F. Wu, and T.-S. Chua. Atentional factorization
machines: Learning the weight of feature interactions via atention networks.
In IJCAI, 2017.
[39] C. Xiong, J. Callan, and T.-Y. Liu. Learning to atend and to rank with word-entity
duets. In SIGIR, 2017.
[40] C. Zhang, G. Zhou, Q. Yuan, H. Zhuang, Y. Zheng, L. Kaplan, S. Wang, and J. Han.
Geoburst: Real-time local event detection in geo-tagged tweet streams. In SIGIR,
2016.
[41] H. Zhang, F. Shen, W. Liu, X. He, H. Luan, and T.-S. Chua. Discrete collaborative
ﬁltering. In SIGIR, 2016.
[42] H. Zhang, M. Wang, R. Hong, and T.-S. Chua. Play and rewind: Optimizing
binary representations of videos by self-supervised temporal hashing. In MM,
2016.
[43] H. Zhang, Z.-J. Zha, Y. Yang, S. Yan, Y. Gao, and T.-S. Chua. Atribute-augmented
semantic hierarchy: Towards bridging semantic gap and intention gap in image
retrieval. In MM, 2013.
[44] W. Zhang, T. Du, and J. Wang. Deep learning over multi-ﬁeld categorical data.
In ECIR, 2016.
