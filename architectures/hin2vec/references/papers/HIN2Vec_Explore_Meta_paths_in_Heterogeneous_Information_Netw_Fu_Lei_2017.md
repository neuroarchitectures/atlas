# HIN2Vec Explore Meta paths in Heterogeneous Information Netw Fu Lei 2017

> Source: `HIN2Vec_Explore_Meta_paths_in_Heterogeneous_Information_Netw_Fu_Lei_2017.pdf`

---

Session 9B: Representation Learning                                                                                           CIKM’17, November 6-10, 2017, Singapore




      HIN2Vec: Explore Meta-paths in Heterogeneous Information
                Networks for Representation Learning
                     Tao-yang Fu                                               Wang-Chien Lee                                              Zhen Lei
       The Pennsylvania State University                             The Pennsylvania State University                       The Pennsylvania State University
        University Park, PA 16802, USA                                University Park, PA 16802, USA                          University Park, PA 16802, USA
             txf225@cse.psu.edu                                             wlee@cse.psu.edu                                           zlei@psu.edu

ABSTRACT                                                                                             relationships amongst nodes is much needed to serve as input fea-
In this paper, we propose a novel representation learning frame-                                     tures to supervised machine learning algorithms. This is a prepro-
work, namely HIN2Vec, for heterogeneous information networks                                         cessing step for data mining and knowledge discovery, well known
(HINs). The core of the proposed framework is a neural network                                       as feature engineering. A typical approach of feature engineering is
model, also called HIN2Vec, designed to capture the rich seman-                                      to involve domain experts to manually design domain-specific rep-
tics embedded in HINs by exploiting different types of relation-                                     resentation of data, i.e., feature vectors of data, for specific predic-
ships among nodes. Given a set of relationships specified in forms                                   tion tasks. This approach, heavily relying on prior knowledge and
of meta-paths in an HIN, HIN2Vec carries out multiple prediction                                     experiences of domain experts, is time-consuming and expensive.
training tasks jointly based on a target set of relationships to learn                               This issue has given rise to a great deal of interest in representation
latent vectors of nodes and meta-paths in the HIN. In addition to                                    learning in networks that aims to embed a network into a low-
model design, several issues unique to HIN2Vec, including regular-                                   dimensional space and represent each node as a low-dimensional
ization of meta-path vectors, node type selection in negative sam-                                   feature vector for supervised learning.
pling, and cycles in random walks, are examined. To validate our                                        In this paper, we propose a new neural network (NN) model,
ideas, we learn latent vectors of nodes using four large-scale real                                  namely Heterogeneous Information Network to Vector (HIN2Vec) for
HIN datasets, including Blogcatalog, Yelp, DBLP and U.S. Patents,                                    representation learning of nodes in heterogeneous information net-
and use them as features for multi-label node classification and                                     works (HINs). The HIN2Vec model aims to capture the rich in-
link prediction applications on those networks. Empirical results                                    formation in an HIN by exploiting various types of relationships
show that HIN2Vec soundly outperforms the state-of-the-art repre-                                    among nodes and the network structure. HINs, such as Yelp social
sentation learning models for network data, including DeepWalk,                                      network [4], DBLP collaboratoin network [3], and U.S. patent ci-
LINE, node2vec, PTE, HINE and ESim, by 6.6% to 23.8% of micro-f 1                                    tation network [2], are networks with nodes and edges belonging
in multi-label node classification and 5% to 70.8% of MAP in link                                    to different types. With heterogeneous types of nodes and edges,
prediction.                                                                                          HINs are able to describe various types of relationships among
                                                                                                     nodes and thus contain very rich information. A meta-path, con-
KEYWORDS                                                                                             sisted of a sequence of node types and/or edge types, is usually
                                                                                                     used to denote a particular relationship between node pairs. There-
representation learning, heterogeneous information network
                                                                                                     fore, different meta-paths may have different semantics. For ex-
                                                                                                     ample, consider a DBLP collaboration network, which consists of
                                                                                                     three node types: Author, Paper and Venue, and two edge types:
1 INTRODUCTION                                                                                       an author writes a paper, and a paper is published in a venue. A
Network data analysis and mining is an important research field                                      meta-path Author-Paper-Author describes the collaboration be-
because network data, capturing phenomena in various networks,                                       tween two authors, while Author-Paper-Venue-Paper-Author
such as social networks, paper citation networks, and World Wide                                     describes the relationship where both authors have papers pub-
Web, are ubiquitous in the real world [6, 15, 29]. Network analy-                                    lished in the same conference venue. We claim that encoding the
sis often involves prediction tasks over nodes or edges, e.g., node                                  rich information embedded in meta-paths and the whole network
classification [14], node clustering [23] and link prediction [20]. In                               structure would help learning meaningful representation which is
order to achieve good performance in these tasks, a proper repre-                                    useful for various applications, because the different semantics of
sentation of network nodes and edges that captures embedded in-                                      relationships are better captured.
formation in the network structure while preserving the original                                        To achieve this goal and to train the HIN2Vec model, we design a
                                                                                                     new learning framework (also called HIN2Vec) (shown in Figure 1)
Permission to make digital or hard copies of all or part of this work for personal or                which, given an HIN and a set of targeted relationships specified
classroom use is granted without fee provided that copies are not made or distributed                in forms of meta-paths, learns latent vectors of both nodes and the
for profit or commercial advantage and that copies bear this notice and the full cita-
tion on the first page. Copyrights for components of this work owned by others than
                                                                                                     targeted relationships in the HIN by predicting relationships be-
the author(s) must be honored. Abstracting with credit is permitted. To copy other-                  tween nodes. Compared with previous works, the HIN2Vec model
wise, or republish, to post on servers or to redistribute to lists, requires prior specific          preserves more contextual information, not only assuming that
permission and/or a fee. Request permissions from permissions@acm.org.
CIKM’17, November 6–10, 2017, Singapore.                                                             two nodes are relevant if there exists a relationship between them
© 2017 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ISBN 978-1-4503-4918-5/17/11…$15.00
DOI: https://doi.org/10.1145/3132847.3132953




                                                                                              1797
Session 9B: Representation Learning                                                                       CIKM’17, November 6-10, 2017, Singapore




                                                                                  A novel idea for representation learning in HINs. With mul-
                                                                                  tiple types of nodes and edges, HINs are able to describe various
                                                                                  types of relationships between nodes, which have different seman-
                                                                                  tics. We show that by capturing various types of relationships be-
                                                                                  tween nodes would help representation learning because it better
                                                                                  captures more detailed and precise information embedded in the
                                                                                  network structure.
                                                                                  A new representation learning framework for HINs. We pro-
                                                                                  pose a two-phase framework to learn representations of nodes and
                                                                                  meta-paths in HINs. The training data preparation algorithm in
                                                                                  Phase 1 adopts random walk generation and negative sampling
       Figure 1: Overview of the HIN2Vec framework                                to prepare training data tailored for HIN2Vec. The HIN2Vec NN
                                                                                  model, core of phase 2, is designed as a logistic binary classifier
                                                                                  that predicts whether two input nodes has a specific relationship
(as previous works do) but also distinguishing the different rela-                in order to efficiently learn model parameters, i.e., node vectors
tionships between nodes and treating them differently by learn-                   and meta-path vectors. Issues in these two phases, such as cycles
ing relationship vectors jointly. In specific, the HIN2Vec frame-                 in random walks, node selection in negative sampling, and regu-
work consists of two phases: (1) Training data preparation: A data                larization of meta-path vectors are studied.
preparation approach based on random walk and negative sampling                   Empirical study using real-world data. We evaluate HIN2Vec
is developed to prepare training data in accordance with the tar-                 by conducting a comprehensive evaluation with two different ap-
geted relationships for representation learning of nodes in HINs;                 plications, node classification and link prediction, using four large-
and (2) Representation learning: The novel neural network model,                  scale real HIN datasets, including Blogcatalog, Yelp, DBLP and
HIN2Vec model, is designed to learn both node vectors and re-                     U.S. patents, in comparison with six state-of-the-art representation
lationship vectors by maximizing the likelihood of predicting re-                 learning models. Empirical result shows that HIN2Vec soundly
lationships among nodes jointly. The proposed neural network                      outperforms all existing models. In addition, an analysis on the
model is trained by predicting multiple heterogeneous relationships               learned meta-path vectors shows that the learned representation
between node pairs simultaneously and jointly. This multi-task                    of relationships capture their semantics.
learning approach allows the proposed model to jointly embed the                     In the rest of this paper, we first review the related work in
rich information of different relationships and the overall network               Section 2, and introduce HINs, meta-paths, problem definition and
structure into node vectors. Conceptually, relevant nodes which                   analysis in Section 3. We present the proposed HIN2Vec frame-
have relationships are close to each other. On the other hand, the                work in Section 4 and show experiment results in Section 5. Fi-
relationship vectors indicate which dimension captures certain re-                nally, we conclude the paper in Section 6.
lationships, and are useful for providing analytical insights, such
as grouping meta-paths with similar semantics. Furthermore, it
is essential for the proposed model to be scalable for large-scale
real-world datasets. We leverage asynchronous stochastic gradi-                   2 RELATED WORK
ent descent in representation learning in parallel.                               Recent development on representation learning has shed a light
   There are a few previous works on representation learning in                   on alleviating the dependence of feature engineering on human
homogeneous information networks [10, 24, 28]. Although these                     knowledge and labors [7, 24, 28]. The goal of representation learn-
prior works all claim that their approaches are able to capture the               ing aims to automatically learn useful latent representations of
structural information of networks, the specific objective functions              data that are effective and discriminative as input features to su-
used in their models tend to consider only part of aggregated in-                 pervised machine learning algorithms for various prediction tasks.
formation among nodes or limited types of relationships between                   Among the various approaches of representation learning, the neu-
nodes separately. There are also several existing works on rep-                   ral network based learning models have received significant at-
resentation learning in HINs [8, 11, 13, 25, 27]. However, some                   tention in recent years, and achieved successes in several empir-
models aim to only capture limited types of relationships between                 ical research studies of various domains, including speech recog-
nodes (e.g., one-hop or two-hop neighberhood between two nodes)                   nition [12, 22], computer vision [9, 16], and natural language pro-
[8, 11, 27], and some tend to miss the different semantics of rela-               cessing (NLP) [21].
tionships between nodes and only capture the aggregated informa-                     Recently, research on representation learning has been extended
tion of relationships [8, 11, 13, 27]. Only one study [25] tries to cap-          to network data [8, 10, 11, 13, 24, 25, 27, 28]. However, instead of
ture different relationships between nodes. However, it highly de-                the complex Heterogeneous information networks (HINs) targeted
pends on a user-guided way to determine a user-given meta-path                    in this paper, some prior works focus only on learning node vec-
set and the weight of each meta-path for representation learning.                 tors in homogeneous information networks [10, 24, 28]. Moreover,
Moreover, some parts of its objective function to encode relation-                while they all claim that their approaches are able to capture the
ships between nodes, e.g., the multiplication of vectors of meta-                 embedded structures of information networks, these models tend
paths, are illy defined.                                                          to consider only aggregated information among nodes or limited
   The major contributions of our work are summarized as follows.                 types of relationships. For instance, DeepWalk [24] and node2vec




                                                                           1798
Session 9B: Representation Learning                                                                     CIKM’17, November 6-10, 2017, Singapore




[10] learn feature vectors of nodes by capturing the nearby neigh-
borhood to each node by simulating uniform and parameterized
random walks, respectively. LINE [28] captures 1-hop and 2-hop
neighborhood relationships, separately, to learn two representa-
tions of nodes.
   There are also several existing works on representation learn-
                                                                                  Figure 2: A conceptual Figure 3: A paper-author
ing on HINs [8, 11, 13, 25, 27]. Some models aim to only capture
                                                                                  model for HIN2Vec      HIN
limited types of relationships between nodes. Specifically, PTE
[27] and HNE [8] learn feature vectors of nodes by capturing 1-                    In this paper, we propose a neural network model to tackle the
hop neighborhood relationships between nodes. HEBE [11] cap-                    representation learning problem on HIN. Our idea is to explore the
tures 2-hop neighberhood between multiple nodes. Some tend to                   rich information and network structure in HIN by capturing mul-
miss the different semantics of relationships between nodes and                 tiple relationships (i.e., meta-paths) between nodes and use them
only capture the aggregated information of relationships [8, 11, 13,            as prediction objectives simultaneously and jointly for learning of
27]. Only one study [25] tries to capture different relationships be-           the node vectors. To realize this idea, nevertheless, we face the
tween nodes. However, it relies heavily on user guidance to deter-              following challenges: (1) Model design. A well designed NN model
mine a user-given meta-path set and the weight of each meta-path                is critical to effective and efficient learning of the HIN2Vec frame-
for representation learning. Moreover, some parts of its objective              work. A conceptual neural network model, developed in our pre-
function to encode relationships between nodes, e.g., the multipli-             liminary design, trains a multi-label classifier to learn node vec-
cation of vectors of meta-paths, are illy defined.                              tors, but it faces excessive overhead in both training data prepara-
                                                                                tion and model learning processes. The proposed HIN2Vec model
3 PRELIMINARIES                                                                 is a better design. (2) Regularization. Due to the semantics and
In this section, we first introduce the notions of HINs and meta-               implications of latent vectors in the learning process, proper regu-
paths, then define the tackled problem, and discuss the challenges.             larization on certain model parameters are required. (3) Training
                                                                                data preparation. Training data need to be prepared and tailored
3.1 Information Network and Meta-Path                                           based on the learning logic behind the proposed HIN2Vec model.
We first define information networks and meta-paths.                            There is a trade-off, especially for large-scale HINs, between the
                                                                                computational/spacial efficiency and the quality of training data.
   Definition 1. Information Network. An information network
is a directed graph G = (V ,E, Φ, Ψ), where V is the set of nodes;
E ⊆ V × V is the set of edges in V . Φ : V → A and Ψ : E → R
                                                                                4 THE HIN2VEC FRAMEWORK
are type mapping functions for nodes and edges, respectively. Here              As introduced earlier, the HIN2Vec framework consists of two
each node v ∈ V is mapped to one particular node type in A, i.e.,               phases: Training data preparation and Representation learning (See
Φ(v) ∈ A, and each link e ∈ E belongs to a particular edge type in              Figure 1). In the following, we first present our approach for the
R, i.e., Ψ(e) ∈ R. When |A| > 1 or |R| > 1, the network is called a             representation learning phase (Section 4.1), where we discuss a
heterogeneous information network (HIN); otherwise, it is a homoge-             conceptual design of NN model for our framework and its pitfalls,
neous information network.                                                      and then introduce the proposed HIN2Vec NN model and some
                                                                                issues to consider. Next, for the training data sampling phase, we
   The concept of meta-paths has been proposed to describe how                  propose an efficient training data sampling method, based on ideas
nodes are related to each other [18] [26]. In the following, we de-             of random walks and negative sampling, to generate training data
fine meta-paths.                                                                for the proposed model and discuss related issues (Section 4.2).
    Definition 2. Meta-path. Given a heterogeneous information
network G = (V ,E, Φ, Ψ), a meta-path π is a sequence of node types             4.1 Representation Learning
a 1 ,a 2 , ...,an and/or edge types r 1 ,r 2 , ...,r n−1 :                      As discussed, our idea to learn node vectors for HIN applications
                             r1       ri    rn−1                                lies in jointly learning a model for multiple prediction tasks, one
                    π = a 1 → ...ai → ... → an                                  for each meta-path. Thus, an intuitive approach is to develop an
   A path p, which goes though nodes v 1 ,v 2 , ...,vn , is an instance         NN model that predicts a set of targeted relationships between any
of the meta-path π , if ∀i = 1, ...,n,ai = Φ(vi ) and r i = Ψ(vi ,vi +1 ).      given pair of nodes.
                                                                                   Conceptually, we can adopt a single-hidden-layer feedforward
3.2 Problem Definition and Analysis                                             neural network model (as illustrated in Figure 2) that takes a pair of
                                                                                nodes x,y ∈ V as the input to predict the probabilities P(r i |x,y)(i =
The goal of this work is to learn a representation of nodes in an
                                                                                1..|R|) of relationships between x and y in the target relationship
HIN. In the following, we formally define the problem.
                                                                                set R as the output. In the model, the input layer takes in two one-
   Definition 3. Representation Learning on HIN. Given a heteroge-              hot vectors, x⃗ and y⃗ both of length |V |, representing x and y. In
neous information network (HIN), denoted as a graph G = (V ,E, Φ, Ψ).           the latent layer, x⃗ and y⃗ are transformed into latent vectors WX′ x⃗
Representation learning aims to learn a function f : V → Rd that                and WY′ y⃗, where WX and WY are two |V | ×d matrices representing
projects each node v ∈ V to a vector in a d-dimensional space Rd ,              the transformation, WX′ and WY′ are their transpose matrices, and
where d ≪ |V |.                                                                 d is the dimensionality of the hidden layer. Finally, in the output




                                                                         1799
Session 9B: Representation Learning                                                                                    CIKM’17, November 6-10, 2017, Singapore




                                                                                             in the learning process. Here we apply a regularization function
                                                                                             f 01 (.) to restrict the values of latent vector for r to be between 0
                                                                                             and 1. We apply Hadamard function, i.e., element-wise multipli-
                                                                                             cation, to aggregate the three vectors, denoted by WX′ x⃗ ⊙ WY′ y⃗ ⊙
                                                                                             f 01 (WR′ r⃗) and then apply the Identity function for activation. Fi-
                                                                                             nally, the output layer, taking Summation as the input (function
                                                                                                                                                               ∑ ′
                                                                                             and Sigmoid function for activation, computes siдmoid                WX x⃗
                                                                                             ⊙WY′ y⃗ ⊙ f 01 (WR′ r⃗) to realize logistic classification. Notice that, in
                                                                                             the framework, we can regularize WX and WY to be the same ma-
                                                                                             trix or two different matrices, depending on whether we would like
                  Figure 4: The HIN2Vec NN model
                                                                                             to emphasize on the individual roles of input nodes in the relation-
layer, the model outputs a vector of length |R| that predicts the                            ships. In our experiments, we make WX and WY to be identical, so
probability of each relationship r ∈ R between x and y, where WR                             WX and WR consist of learned node vectors and meta-path vectors,
is a |R| × d matrix representing the transformation from the latent                          respectively.
layer to output. In summary, this conceptual model can be seen                                   4.1.2 Regularization of WR . As discussed, the regularization
as a multi-label classifier, and the three matrices, WX , WY and WR ,                        function f 01 (.) of WR regularizes values in WR within 0 and 1. The
collect the feature vectors of input node pairs and their relation-                          goal is two-fold: 1) it avoids negative values in WR , which would
ships.1                                                                                      inverse the sign in the multiplication of node vectors, thus interfer-
   To illustrate the conceptual NN model and issues arising in the                           ing the learning of node vectors. Suppose we do not apply f 01 (.)
HIN2Vec framework, Figure 3 shows a simple HIN consisting of                                 on WR , and some values in a relationship vector are negative. The
four papers (P) P1 … P4 , and two authors (A) A1 and A2 . In this                            signs of corresponding values in the two node vectors may become
example, an edge between two papers denotes their citation rela-                             opposite to optimize WR′ (WX′ x⃗ ⊙ WY′ y⃗) in the model. However, if
tionship (we ignore the direction of citations for simplicity) and an                        these nodes are actually positively related, the signs of values in
edge between a paper and an author denotes the authorship. Let                               these two nodes vectors should be the same so as to stay positive
R denote a target relationship set that includes all relationships be-                       after passing through the Sigmoid function at the output layer. 2)
tween nodes within 2 hops, i.e., R = {P-P, P-A, A-P, P-P-P, P-                               it prevents values in WR from becoming too large, which decreases
P-A, P-A-P, A-P-P, A-P-A}, specified by meta-paths. To train                                 the effectiveness of learning. In specific, because HIN2Vec adopts
the conceptual model based on our idea of HIN2Vec, training data                             element-wise multiplication to aggregate the three vectors WX′ x⃗ ,
need to be prepared for eight prediction tasks corresponding to R.                           WY′ y⃗, and WR′ r⃗. If a value of WR′ r⃗ is greater than 1, it decreases the
For instance, P1 and A1 have two relationships, P-A and P-P-A. A                             importance of values of WX′ x⃗ and WY′ y⃗ in the multiplication, thus
                        ⟨                                                ⟩
training data entry is x : P 1 ,y : A1 ,output : [0, 1, 0, 0, 1, 0, 0, 0] .                  interfering the learning of node vectors.
   In practice, however, an issue arising in preparing the training                              Several functions, e.g., Sigmoid function or Binary Step func-
set for this model is the high cost in scanning the whole network                            tion, may serve the purpose described above. We empirically show
to find all the relationships in R for each pair of nodes in the net-                        that the Binary Step function outperforms Sigmoid.
work. In addition, during the training process, the transformation
matrices, i.e., WX , WY and WR , need to be updated in accordance                               4.1.3 Optimization Objective. The learning of HIN2Vec model
with all the relationships, whether they are in presence (positive)                          parameters (i.e., node vectors and meta-path vectors) is realized by
between two input nodes or not (negative).                                                   setting multiple optimization objectives, one for each meta-path in
                                                                                             R, to facilitate iterative training upon the training data. It is critical
    4.1.1 The HIN2Vec model. To address the aforementioned is-                               to set the objective functions properly.
sues, we propose the HIN2Vec NN model that reduces the predic-                                  To train HIN2Vec, a training set D containing training data in
tion tasks of the conceptual NN model (i.e., predicting the probabil-                                      ⟨                 ⟩
                                                                                             the form of x,y,r ,L(x,y,r ) is extracted from the HIN. Here
ities of relationships between two nodes) into new prediction tasks                          L(x,y,r ), a binary value, indicates whether x and y have r . With
— whether two nodes, x and y, have a specific relationship r . As to                         D, HIN2Vec is trained by the backpropagation training algorithm
be discussed later, this design avoids scanning for all relationships                        in conjunction with stochastic gradient descent. It goes backwards
in data preparation as well as examining/updating for all relation-                          to adjust the weights in WX ,WY , and WR for each entry in D, at-
ships during training.                                                                       tempting to maximize the objective function O, which is the multi-
    As shown in Figure 4, the HIN2Vec model is a binary classi-                              plication of O x,y,r (x,y,r ) for each training data entry in D, where
fier that takes a pair of nodes x and y, and a certain relationship                          O x,y,r (x,y,r ) quantifies how HIN2Vec correctly predicts L(x,y,r ).
r ∈ R as the input to predict whether the relationship r exists be-                          To ease the computation in the optimization process, we maximize
tween x and y. The input layer takes in three one-hot vectors, x⃗ , y⃗,                      log O rather than directly maximize O. The objective function O
and r⃗, which are transformed into latent vectors WX′ x⃗ , WY′ y⃗, and                       and derivation of log O are shown below.
f 01 (WR′ r⃗) in the latent layer. Note that the treatment on r is differ-                                                  ∑
                                                                                                            O ∝ log O = x,y,r ∈D log O x,y,r (x,y,r )
ent from x and y due to their different semantics and implications
                                                                                                 In the above, O x,y,r (x,y,r ) quantifies how HIN2Vec correctly
                                                                                                                       ⟨                ⟩
1 As a conceptual model, here we do not discuss some implementation details such as          predicts a training data x,y,r ,L(x,y,r ) based on P(r |x,y), which
aggregation and activation operations in detail.                                             is the output of HIN2Vec and is the predicted probability that x and




                                                                                      1800
Session 9B: Representation Learning                                                                             CIKM’17, November 6-10, 2017, Singapore




y have the relationship r in the network. Specifically, for a train-                                      Table 1: Statistics of Datasets
                ⟨                ⟩
ing data entry x,y,r ,L(x,y,r ) , when L(x,y,r ) is 1, O x,y,r (x,y,r )                                      User     Group
aims to maximize P(r |x,y); otherwise, O x,y,r (x,y,r ) aims to mini-                      Blogcatalog
                                                                                                            10312        39
mize P(r |x,y). Thus, O x,y,r (x,y,r ), log O x,y,r (x,y,r ) and P(r |x,y)                                   User    Business       City       Category
are derived as below,                                                                          Yelp
                                                                                                            630639     86810         10          807
                           
                           P(r |x,y),         if L(x,y,r ) = 1                                             Paper     Author       Venue
        O x,y,r (x,y,r ) = 
                            1 − P(r |x,y), if L(x,y,r ) = 0
                                                                                              DBLP
                                                                                                            53464      54949         20
                           
                                                                                                            Patent   Inventor     Assignee       Class
                                                                                           U.S. Patents
      log O x,y,r (x,y,r ) =L(x,y,r ) log P(r |x,y)                                                         295145    293848       31805          14
                           + [1 − L(x,y,r )] log[1 − P(r |x,y)]
                             (∑                                 )
         P(r |x,y) = siдmoid    WX′ x⃗ ⊙ WY′ y⃗ ⊙ f 01 (WR′ r⃗)
                                                                                      for the first node P1 as ⟨P 1 ,P2 , P-P⟩ and ⟨P 1 ,A1 , P-P-A⟩, for the
   We then apply the stochastic gradient descent algorithm to max-                    second node P2 as ⟨P 2 ,A1 , P-A⟩ and ⟨P 2 ,A3 , P-A-P⟩, and so on.
imize the objective function O. Specifically, for each training data                     There are issues in applying random walks to HIN2Vec data
        ⟨                    ⟩
entry, x,y,r ,L(x,y,r ) , it goes backwards to adjust the weights                     preparation. First, a random walk may give rise to cycles, which
in WX′ x⃗ ,WY′ y⃗, and WR′ r⃗ based on the gradients of log O x,y,r (x,y,r )          hurt the quality of the training data, because a path instance with
differentiated by WX′ x⃗ ,WY′ y⃗, and WR′ r⃗, respectively, by                        cycles may not comply to the semantics of the corresponding meta-
                                      d log O x,y,r (x,y,r )                          path. For example, the third node A1 in the above random walk
                  WX′ x⃗ := WX′ x⃗ +         dWX′ x ⃗                                 generates a training data entry ⟨A1 ,A1 , A-P-A⟩ by the path
                                     d  log O x,y,r (x,y,r )
                  WY′ y⃗ := WY′ y⃗ +         dWY′ y⃗
                                                                                      A1 ,P 3 ,A1 . However, an author has no coauthorship with herself.
                    ′         ′      d log O x,y,r (x,y,r )                           Thus, we eliminate data entries with any cycle by checking dupli-
                  WR r⃗ := WR r⃗ +           dWR′ r⃗                                  cate nodes.
                                                                                         Second, random walks are used to sample positive data, but the
4.2 Training Data Preparation                                                         model also needs negative data for learning. Thus, while generat-
We develop an efficient algorithm to sample the HIN while extract-                    ing positive samples via random walks, we also generate negative
                                                ⟨              ⟩
ing training data for HIN2Vec in the form of x,y,r ,L(x,y,r ) . No-                   data entries following the ideas of negative sampling in Word2Vec
                                                                                                                                     ⟨     ⟩
tice that there is a trade-off in data collection between the compu-                  [21]. For each sampled positive entry, x,y,r , we generate neg-
tational efficiency (e.g., to sample training data randomly instead                   ative data entries by randomly replacing one of the three values
of enumerating) and the quality (e.g., the training dataset should                    with other x ′ , y ′ , or r ′ , where x ′ and y ′ are randomly selected
cover as many nodes pairs with correct semantics in their relation-                   nodes, and r ′ is a randomly selected relationship from R. A gen-
                                                                                                                       ⟨           ⟩
ships as possible). Therefore, it’s essential to design an efficient                  erated negative data entry, x ′′ ,y ′′ ,r ′′ indicates that x ′′ and y ′′
data sample extraction scheme for training data preparation.                          are not expected to have a certain relationship r ′′ . However, for
    We apply random walks in the task for several advantages. First,                  HIN2Vec, more erroneous negative data (which actually a positve
it is computationally efficient for both space and time complexity.                   data) might be generated by replacement of r compared with by re-
The space complexity of storing the direct neighbors of a node to                     placement of x or y since |V | is usually much larger than |R|, while
select the next node in a random walk is O(|V |), and the space com-                  |R| is usually not very large (tens to hundreds). To maintain effi-
plexity of storing the path of a random walk with length l is O(l).                   ciency and the cleanness of the negative samples, we only sample
The time complexity of selecting the next node is O(1) (by some                       negative data by randomly replacing x or y, and filter out erro-
random selection methods, such as Alias method [5, 17]), and the                      neously generated positive samples. To do so, we also ensure that
time complexity of generating a path with length l by a random                        the randomly selected x ′ or y ′ has the same node type as x or y.
walk is O(l). Moreover, subsequences of a random walk are also                        For example, for a positive data ⟨P1 ,P2 , P-P⟩, we randomly select
random walks, which can be used to generate training data with-                       a paper, says P3 , to replace P2 (or P1 ) to generate a negative data
out re-generating new random walks. Suppose we consider all re-                       that indicates P2 (or P 1 ) does not have a citation relationship with
lationships within w-hop neighborhood. Given a random walk of                         P3 . In other words, we do not select a node with a different node
length l > w, it can generate w training data for l − w nodes. Thus,                  type, says authors, for replacement. We empirically examine the
the time complexity of generating training data given a random                        effect of avoiding cycles in random walks and ensuring the same
walk is O(w(l −w)). Therefore, if we consider 1 to w-hop neighbor                     node type in negative sampling.
relationships, the space complexity and time complexity of gener-
ating training data by a random walk with length l is O(|V | + l)
and O(l + w(l − w)), respectively. When l and w are both much
smaller than |V |, using random walks to generate training data is                    5 EXPERIMENTS
scalable for large-scale networks.                                                    In this section, we conduct a comprehensive evaluation on HIN2Vec.
    Consider the example of paper-author network discussed ear-                       We first introduce four real-world HINs used for experiments and
lier. Suppose we generate a random walk P 1 ,P2 ,A1 ,P3 ,A1 . Let                     six models for representation learning on networks. Then, we eval-
the target relationship set R include all relationships with meta-                    uate HIN2Vec and those models by two applications: multi-label
path length no greater than 2-hop. We can generate training data                      classification for nodes and link prediction for edges.




                                                                               1801
Session 9B: Representation Learning                                                                                             CIKM’17, November 6-10, 2017, Singapore




5.1 Datasets and Models for Evaluation                                                            5.2 Multi-label Classification of Nodes
Our evaluation involves two social network datasets (Blogcatalog                                  In this section, we evaluate the models by multi-label classification
and Yelp) and two scientific publication datasets (DBLP and U.S.                                  of nodes. We first introduce the experimental setup, including the
Patents). Some statistics of the HINs extracted from these datasets                               process of classification, the preparation of labeled datasets and
are summarized in Table 1.                                                                        the default settings of parameters in the compared models. Then,
Blogcatalog is a blog dataset of Blogcatalog, released by Arizona                                 we perform sensitivity tests on parameters of HIN2Vec to deter-
State University [1]. We use all bloggers (U) and their groups (G)                                mine their default settings. Next, we examine several issues in the
as nodes to form a social network, which contains friendships (U-                                 HIN2Vec framework, including the regularization on meta-path
U) and users’ groups (U-G) as edges.                                                              vectors WR , cycles in random walks, and negative sampling with
Yelp is a social media dataset, released in Yelp Dataset Challenge                                the same node type. Finally, we show the experimental results and
[4]. We extract data of the top 10 cities with the most businesses to                             analyze the meta-path vectors learned by HIN2Vec.
form a network, which includes users (U), businesses (B), cities(C)
and categories (T) as nodes, and friendships (U-U), users’ reviews                                    5.2.1 Experimental Setup. After learning the node vectors, we
(B-U), businesses’ cities (B-L) and businesses’ categories (B-C) as                               select a set of nodes, which are assigned one or more labels from
edges. We filter small categories with less than 10 businesses.                                   a finite label set, and use their representations as feature vectors
DBLP is a bibliographic dataset in computer science [3]. We ex-                                   to learn and test a linear SVM classifier with five-fold cross vali-
tract papers published between 1994 to 2014 of 20 conferences in 4                                dation. We use micro-f 1 score and macro-f 1 score as metrics for
research fields to form a network 2 . The network includes papers                                 evaluation.
(P), authors (A) and venues (V) as nodes, and authorships (P-A),                                      In Blogcatalog, users are categorized by 39 groups, which serve
papers’ venues (P-V) as edges.                                                                    as labels. Thus, all users are included in the labeled dataset for
U.S. Patent is a patent dataset of United States Patent and Trade-                                user group classification. In Yelp, as the majority of businesses are
mark Office (USPTO) [2]. We extract patents issued between 1998                                   restaurants, we select 10 main cuisines in restaurants’ categories
to 2012 in 14 drug related patent classes to form a network 3 . The                               as labels, and select 13,111 restaurants with at least one of those
network contains patents (P), inventors (I), assignees (A) and patent                             labels to form a labeled dataset for restaurant type classification 4 .
classes (C) as nodes, and inventorships (P-I), patents’ assignees (P-                             In DBLP, we use the 4 research fields as labels, and assign an au-
A), patents’ classes (P-C) and citations (P→P) as edges.                                          thor with a field if he/she has a publication in a conference in that
   We evaluate HIN2Vec against six state-of-the-art representation                                field to form a labeled dataset for author classification. Finally, for
learning models. Among them, DeepWalk, LINE and node2vec, are                                     U.S. Patents, we use 14 drug related patent classes as labels, and in-
designed for homogeneous information networks. The are applied                                    volve all patents to form a labeled dataset for patent classification.
by treating all node and edges in the HINs as homogeneous ones.                                   For these node classifications, we eliminate from HINs, groups in
DeepWalk [24] learns d-dimensional node vectors by capturing                                      Blogcatalog, categories in Yelp, venues in DBLP and patent classes
node pairs within w-hop neighborhood via uniform random walks                                     in U.S. Patents, when we learn the representation of nodes.
in the network.                                                                                       Regarding default parameters, the dimensionality of node vec-
LINE [28] learns node vectors by considering first- and second-                                   tors, d, is set to 128 for all approaches. The negative sampling rate n
order proximities of nodes in a network separately. We apply LINE                                 is set to 5. The initial learning rate α in stochastic gradient descent
to learn d/2 dimensions by capturing first-order information, and                                 is 0.025. The context window for DeepWalk and node2vec is set to
the other d/2 dimensions by capturing the second-order informa-                                   4 because they achieve good performance. Also, we use all meta-
tion, and then use both to form d-dimensional node vectors.                                       paths with length w ≤ 4 for HINE, ESim and HIN2Vec. For ESim,
node2vec [10] is generalized from DeepWalk. It learns d-                                          we set the weight of training data sampling of each meta-path
dimensional node vectors by capturing node pairs within w-hop                                     based on its length. For example, a meta-path with length l, its sam-
neighborhood via parameterized random walks.                                                      pling weight is 1/l. We also try equal weighting of each meta-path,
PTE [27] decomposes an HIN to a set of bipartite networks by                                      but weighting by meta-path length performs better. For node2vec,
edge tyes, and learns d-dimensional node vectors by capturing 1-                                  the two parameters q and p for parameterized random walks are
hop neighborhood of the resulting bipartite networks.                                             set to 4 and 1, respectively. The number of training data sampling
HINE [13] learns d-dimensional node vectors by capturing path                                     or the length of random walks for generating training data varies
counts or path constrained random walks [19] of node pairs within                                 for each model and for each dataset, in order to obtain converged
w-hop neighborhood.                                                                               results.
ESim [25] learns d-dimensional node vectors by paths between
                                                                                                     5.2.2 Parameter Tuning in HIN2Vec. Parameters settings in
nodes along a given set of meta-paths. This model also learns meta-
                                                                                                  HIN2Vec affect the node representation learning and the applica-
path vectors, which is used to shift node vectors in the designed
                                                                                                  tion performance. To decide the default settings, we vary the val-
objective function.
                                                                                                  ues of important parameters to observe how the micro-f 1 changes
                                                                                                  in node classification in the networks. The results are shown in
2 20 conferences are ”AAAI”, ”CVPR”, ”ECML”, ”IJCAI”, ”SIGMOD”, ”VLDB”, ”PODS”,
”EDBT”, ”ICDE”, ”ICDM”, ”KDD”, ”PAKDD”, ”PKDD”, ”SDM”, ”ECIR”, ”SIGIR”,
                                                                                                  Figure 5.
”WSDM”, ”WWW”, ”CIKM”, that belong to 4 research fields, including DM, DB, IR,
and ML.
3 Drug related patent classes are 128, 351, 433, 424, 435, 623, 514, 600, 601, 602, 604,          4 10 main cuisines are ”American”, ”Mexican”, ”Italian”, ”Chinese”, ”Japanese”, ”Thai”,
606, 607, 800.                                                                                    ”Indian”, ”Canadian”, ”Middle Eastern” and ”Greek”




                                                                                           1802
Session 9B: Representation Learning                                                                    CIKM’17, November 6-10, 2017, Singapore




        (a) Dimensionality d         (b) Number of Negative Samples n           Figure 6: Comparison of approaches to issues in HIN2Vec

                                                                               networks. Also, these parameter values are set for the fair compar-
                                                                               ison with the compared models. The length of random walks in
                                                                               HIN2Vec is set to 1280 to obtained converged performance.

                                                                                   5.2.3 Study of unique issues in HIN2Vec. As discussed previ-
                                                                               ously, there are several unique issues arising due to HIN data prepa-
                                                                               ration and HIN2Vec design. In this section, we examine the issues
    (c) Length of Random Walks l        (d) Length of Meta-paths w             of i) regularization functions on relationship vectors WR , ii) nega-
                                                                               tive sampling with the same node types, and iii) cycle elimination
                   Figure 5: Parameter Tuning
                                                                               in random walks. For (i), HIN2Vec uses the Binary Step function
                                                                               instead of Sigmoid; for (ii), HIN2Vec ensures negative sampling
No. of dimensionality. First of all, Figure 5(a) shows that set-               choosing the same types of nodes as the corresponding positive
ting the number of dimension d at 128 is reasonable. General                   samples rather than arbitrary node types, and for (iii), HIN2Vec
speaking, a small d is not sufficient to capture the information em-           eliminates cycles from random walks instead of leaving them in
bedded in relationships between nodes, but a large d may lead to               the data samples. We perform experiments to compare the differ-
noises and cause overfitting. In Blogcatalog, the best performance             ent choices in these issues and justify our decisions.
is achieved when d is 128. In the other three networks, increasing                 Specifically, we compare 5 settings. First, the Baseline setting
d up to 256 continuously improves the performance, though their                adopts Sigmoid function (Sigmod), uses arbitrary node types in
improvements are small (about 0.7% to 6.2%). Generally, a larger               negative sampling (Arbitrary Type), and has cycles in random
network may need a larger d to capture the information embedded                walks (Cycle). On the other hand, the Best setting applies Bi-
in relationships between nodes.                                                nary Step function (Binary Step), ensures the same node type in
No. of negative samples per positive sample. As shown in Fig-                  negative sampling (Same Type), and eliminates cycles in random
ure 5(b), the performance does not change much when the number                 walks (No Cycle). Three additional settings regarding particular
of negative sampling per positive sample n is set at between 3 to 7.           issues are tested: a) Sigmod+ Same Type + No Cycle, b) Bi-
Thus, setting n to 5 is a good choice.                                         nary Step+ Arbitrary Type + No Cycle, and c) Binary Step +
Length of Random Walks. A longer random walk can generates                     Same Type + Cycle. Figure 6 shows that the Best setting used in
more sample data. Figure 5(c) suggests that when the length of                 HIN2Vec performs the best, outperforming the Baseline setting in
random walks l is increased (resulting in more sample data), the               all the networks by about 3.4% to 9.2%.
performance continues to improve and converge when l is set to                     Regarding the issue of regularization functions, we find that us-
1280 or 2560.                                                                  ing Binary Step is better than using Sigmoid in all networks, by
Length of Meta-Paths. Finally, we study the impact of the maxi-                about 0.5% to 7.4%. The main difference is that Sigmoid function
mum length w of meta-paths of interest. Figure 5(d) shows that the             regularizes values to a continuous number between 0 to 1 and Bi-
maximum length of meta-paths does not affect the performance                   nary Step function regularizes values to 0 or 1 by the threshold
significantly in Blogcatalog, Yelp and U.S. Patents, because that 1-           0.5. When a value in a meta-path vector is regularized to a num-
hop and 2-hop relationships, e.g., friendships in Blogcatalog and              ber between 0 to 1 by Sigmoid, it amplifies the importance of the
citations in U.S. Patents, are the most important relationship in              multiplication of two input node vectors, thus mis-guiding repre-
these networks. However, a larger w is still helpful in these net-             sentation learning.
works, and the performance improves by 9% when w is 4. On the                      The result also shows that Same Type is better than Arbitrary
other hand, capturing meta-paths with larger w is crucial in DBLP              Type, by a margin of about 1.6% to 5% (excluding Blogcatalog net-
because some longer meta-paths have important semantic mean-                   work because that it only contains one node type, and thus does
ing. For examples, A-P-A-P-A (two authors have co-authorship                   not have this issue.) The result suggests that ensuring the same
with the same author) in DBLP would help the classification as                 node type in negative sampling is necessary.
these two authors are likely to be in the same research field.                     Finally, No Cycle and Cycle do not show significant difference.
   Based on these results, for HIN2Vec, the dimensionality of node             Actually, eliminating cycles in random walks does not always per-
representation, d, is set to 128, the negative sampling rate n is set          form better than leaving cycles in training data. Perhaps this is
to 5, and to use all meta-paths with length w ≤ 4, because that                because the number of erroneous data is relatively small. How-
these parameter values can achieve good performance in the all                 ever, our experiments show that avoiding the training data with




                                                                        1803
Session 9B: Representation Learning                                                                      CIKM’17, November 6-10, 2017, Singapore




cycles improve the efficiency while maintaining the effectiveness.               In Cluster 4, the patents’ assignees, P-A, has unique semantics,
Thus, HIN2Vec chooses to remove cycles from training data.                       and thus no other meta-path is similar to it. Cluster 5 contains re-
                                                                                 lationships between patents via bibliographic information (inven-
   5.2.4 Evaluation of models. The performance of node classifi-
                                                                                 tors and assignees) (except P-I). However, Cluster 6 also contains
cation by all evaluated models is summarized in Table 2. HIN2Vec
                                                                                 several meta-paths with no similar semantics.
outperforms all the state-of-the-art models. As shown, the im-
                                                                                    These analyses suggest that the representations of meta-paths
provement ratio (compared with the best of these existing models,
                                                                                 in HIN2Vec capture semantics of meta-paths very well, placing
marked by ’*’) ranges from 6.4% to 23.8%. We have the following
                                                                                 meta-path vectors in the latent space closely with other vectors
observations from the comparison.
                                                                                 of relevant semantics.
Exploring meta-paths of longer length is useful. Compared
with LINE and PTE, which only capture 1-hop or 2-hop neigh-                      5.3 Link Prediction
borhood of nodes, other models usually have better performance
(except ESim) because they also capture “longer” relationships be-               In this section, we demonstrate that the same node vectors learned
tween nodes.                                                                     by HIN2Vec can be used effectively for another application, link
Distinguishing different meta-paths between nodes is use-                        prediction. We first introduce the experimental setup including
ful. Comparing with DeepWalk, node2vec, HINE and HIN2Vec, all                    the experimental flow of link prediction and metrics for evaluation.
of which consider relationships of nodes within w(=4)-hop neigh-                 Then, we study different vector functions that transform two node
borhood, HIN2Vec further distinguishes different relationships be-               vectors into one vector for the purpose of link prediction. Finally,
tween nodes, whereas others only capture aggregated information.                 we show the experimental results of HIN2Vec, in comparison with
Thus, HIN2Vec is able to capture more detailed information of net-               other models.
work structure and thus can achieve higher performance in node                      5.3.1 Experimental Setup. We model the link prediction prob-
classification.                                                                  lem as a recommendation problem that aims to rank node pairs
Appropriate model design to distinguish relationships be-                        in terms of their relevancy, which may lead to potential linkage
tween nodes is crucial. Comparing with ESim which also cap-                      between them. Specifically, given a network, we first generate a
tures meta-path relationships between nodes, HIN2Vec outperforms                 sub-network by selecting an edge class and randomly removing a
ESim in all four networks. We argue the superiority of HIN2Vec                   certain fraction (20% in our experiments) of edges of the selected
is due to a better model design that precisely captures the relation-            edge class as missing edges. After removing edges, we apply rep-
ships between nodes.                                                             resentation learning models on the resulting sub-network. Then,
                                                                                 to perform the recommendation on the sub-network, we apply su-
   5.2.5 Analysis of Meta-paths. In addition to node vectors,
                                                                                 pervised models to rank node pairs which are more likely to have
HIN2Vec also learns meta-path vectors as a side-product. In this
                                                                                 missing edges. In specific, we randomly select 2000 nodes to form
section, we analyze the representations of relationships (i.e., meta-
                                                                                 a training set. For a selected node, we first use its neighbors within
path vectors) by clustering those vectors to examine whether meta-
                                                                                 a certain number of hops as candidates to form a set of candidate
paths with relevant semantics are clustered together. We use Yelp
                                                                                 node pairs (a pair contains the node and a candidate node). This
and U.S. Patent networks as the study cases because they have
                                                                                 candidate selection step aims to reduce the number of node pairs
more complex schema of networks, with more meta-paths than
                                                                                 to be ranked. We choose an appropriate hop number for different
Blogcatalog and DBLP. We select top 20 meta-paths in these two
                                                                                 networks to ensure the coverage of missing edges is greater than
networks with the highest frequency when generating the train-
                                                                                 85%. To form a training set, for each node pair, we label it with 1 if
ing data by random walks, and then apply K-Means algorithm to
                                                                                 they have a missing edge, and with 0, otherwise. We study differ-
group meta-paths into 6 clusters. The results are shown in Table 3.
                                                                                 ent vector functions (to be discussed later) to construct a feature
   In Yelp, we observe that each of clusters 1 to 5 have clear and dis-
                                                                                 vector for the node pair based on the two nodes’ representations.
tinct semantics. Specifically, Cluster 1 groups meta-paths of friend-
                                                                                 Finally, we apply linear SVM to perform ranking, by using the pre-
ships (except 3-hop friendship). Cluster 2 groups meta-paths be-
                                                                                 dicted confidence of each data with five-fold cross validation. We
tween two businesses via only friendships (except B-U-U-B which
                                                                                 use Mean Average Precision (MAP) and The top-k recall (recall@k)
is not in top 20 frequent meta-path set). Cluster 3 contains meta-
                                                                                 as metrics for evaluation.
paths between business to users via multiple hops of friendships.
                                                                                    (1) Mean Average Precision (MAP): the mean of the average pre-
In Cluster 4, B-U has unique semantics that no other meta-path is
                                                                                 cision scores for the ranking result of each node v. The average pre-
similar with it. Cluster 5 contains meta-paths with cities. Cluster
                                                                                 cision score (AveP) is the sum of the top-k precision (precision@k),
6 contains several meta-paths that do not share semantics, and due
                                                                                 the percentage of top-k ranking results that hit the ground truth
to the lack of space, we skip to show meta-paths in Cluster 6.
                                                                                 over k. The equations of AveP and MAP are shown below.
   Among the 20 meta-paths in the U.S. Patent network, each of
                                                                                                         pr ecision@k                ΣK   Ave P (v )
clusters 1 to 5 also shows obvious semantics. Specifically, Clus-                     AveP(v) = Σnk =1                  MAP = i =1 K       i
                                                                                                              k
ters 1 contains co-citing and co-cited meta-paths, which are rele-                  (2) The top-k recall (recall@k): the percentage of the ground
vant because two patents are likely in the same research field if                truth or indicators ranked in the top-k returned results.
they are co-cited or co-cite the same patents. Cluster 2 contains 1-                                              # of hit s in t op−k
hop and 2-hop citation meta-paths, which both indicate knowledge                      recall@k = # of posit ive document s in t heдr ound t r ut h
flow among patents. Cluster 3 has meta-paths describing the rela-                  To construct a feature vector for a node pair based their node
tionships between inventors and his/her citing and cited patents.                vectors, we study four vector functions, Hadamard, Average, Minus




                                                                          1804
Session 9B: Representation Learning                                                                                CIKM’17, November 6-10, 2017, Singapore




                                                Table 2: Performance Evaluation of Node Classification
                        Blogcatalog                                 Yelp                                  DBLP                            U.S. Patents
                  micro-f1      macro-f1                   micro-f1      macro-f1                micro-f1      macro-f1             micro-f1       macro-f1
   DeepWalk        0.244          0.140                     0.276          0.165                   0.481        0.463                0.675           0.676
     LINE          0.239          0.128                     0.270          0.163                   0.449        0.429                 0.66           0.663
   node2vec        0.246          0.141                     0.276          0.166                   0.491        0.470                0.676           0.677
     PTE           0.179          0.096                     0.222          0.130                   0.417        0.394                0.547           0.555
     HINE          *0.250         *0.144                    *0.278        *0.169                   0.475        0.461                *0.681         *0.685
     ESim          0.207          0.102                     0.229          0.132                  *0.514        *0.496               0.610           0.562
   HIN2Vec      0.272(9.9%) 0.158(11.3%)                 0.302(7.9%) 0.192(12.0%)              0.605(23.8%) 0.594(20.1%)          0.729(6.6%) 0.732(6.4%)

                Table 3: Clusters of Meta-paths                                            one pair is distant and the other pair is close. On the other hand,
  Cluster             Yelp                             U.S. Patents                        Minus function yields the difference between two node vectors but
     1       U-U, U-U-U, U-U-U-U-U                 P→P←P, P←P→P                            does not capture their similarity. For example, two pairs of nodes
     2       B-U-B, B-U-U-U-B                      P←P, P←P←P                              can be similar (close in the latent space) but quite different in terms
     3       B-U-U, B-U-U-U-U                      I-P→P, I-P←P                            of their node vectors.
     4       B-U                                   P-A                                        5.3.3 Evaluation. The performance of link prediction is summa-
     5       B-C, B-C-B, U-B-C                     P-A-P, P-I-P, P-I                       rized in Table 6. ESim fails in Yelp and U.S. Patent networks, be-
     6       …                                     …                                       cause it exhausts all memory (320G) on our server. HIN2Vec out-
                                                                                           performs the existing representation learning models, and the im-
            Table 4: Vector Functions of Node Pairs                                        provement ratio (compared with the best one of the existing works,
                                                                                           which is marked by ’*’) varies from 5.0% to 70.8% for MAP, and 10.8
 Functions      Hadamard         Average         Minus           Abs. Minus                to 24.3% for recall@100. We have the following observations.
                                 v⃗1 i +v⃗2 i
 Description     v⃗1 i ∗ v⃗2 i         2         v⃗1 i − v⃗2 i   |v⃗1 i − v⃗2 i |          Capturing meta-paths of longer length is useful for link pre-
                                                                                           diction in complex HINs. Compared with LINE and PTE, which
and Absolute Minus. The equations are shown in Table 4. Specifi-                           only capture 1-hop or 2-hop neighborhood of nodes, other models
cally, Hadamard function is the element-wise multiplication of two                         usually have better performance in most of the networks (except
vectors. Average function obtains the centroid of two vectors in                           for Blogcatalog), perhaps because they captures relationships be-
the latent space. Minus function yields the difference between two                         tween nodes with larger hop number. For friendship recommen-
vectors in the latent space. Finally, Absolute Minus function cal-                         dation in Blogcatalog, 1-hop or 2-hop friendship in the network al-
culates the distance of the two vectors in each dimension of the                           ready provides useful information, consistent with the notion that
latent space.                                                                              two users are more likely to be friends if they have multiple com-
   Regarding the edge class of missing edges in each network for                           mon friends. Thus, capturing friendships with larger hop numbers
experiments, we choose friendships (U-U) in Blogcatalog, users’                            does not help in other models except for HIN2Vec. The reason why
business reviews (B-U) in Yelp, authorships (P-A) in DBLP and                              HIN2Vec still outperforms LINE in Blogcatalog is that not only it
patent citations (P→P) in U.S. Patent network.                                             captures multiple hop relationships, but also it distinguishes them.
   Regarding default settings, we use the same parameter values                            Distinguishing different relationships between nodes is use-
used in node classification experiments. Additionally, the default                         ful. Compared with DeepWalk, node2vec, HINE, HIN2Vec dis-
feature vector function for node pairs is Hadamard function be-                            tinguishes relationships between nodes, rather than only captur-
cause it has the best performance for all the models.                                      ing aggregated information. Thus, when applying SVM to train a
    5.3.2 Vector Functions for Node Pairs. In this section, we study                       model, the model can increase or reduce the feature weights of the
the effect of aforementioned vector functions for node pairs and                           dimensions of the relationship if it is helpful or not for prediction,
summarize their performance in Table 5. First of all, Hadamard                             respectively. Thus, it achieves higher performance in link predic-
function outperforms all the other functions because it matches                            tion. This is also the reason why HIN2Vec outperforms LINE in
the objective function of HIN2Vec (and other models). In specific,                         Blogcatalog.
if two nodes are more relevant (i.e., with more relationships be-
tween them), the sum of element-wise multiplication of their vec-                          6 CONCLUSIONS
tors should be larger (i.e., their vectors are closer in the latent                        This study focuses on representation learning in HINs. Prior works
space). This is also the reason why Absolute Minus performs well                           in representation learning in networks only consider limited types
because it captures the distance in each dimension of the two nodes’                       of relationships among nodes, or only capture aggregated informa-
vectors. However, Average function and Minus function do not                               tion of relationships. To fill in this gap, we design a novel neural
have good performance. Average function gets the centroid of two                           network model, HIN to Vectors (HIN2Vec), that enables users to
nodes’ vectors but does not well capture the similarity between the                        capture rich semantics of relationships and the details of the net-
two nodes (i.e., the closeness of the two nodes in the latent space).                      work structure to learn representations of nodes in HINs. More-
For example, two pairs of nodes may have the same centroid, but                            over, the proposed model also learn representations of meta-paths,




                                                                                    1805
Session 9B: Representation Learning                                                                                            CIKM’17, November 6-10, 2017, Singapore




                                                    Table 5: Performance Evaluation of Vector Functions
                                               Blogcatalog                        Yelp                          DBLP                    U.S. Patents
                                           MAP recall@100               MAP        recall@100          MAP       recall@100         MAP recall@100
                        Hadamard           0.141      0.279             0.028         0.138            0.265        0.751           0.176      0.602
                         Average           0.074      0.245             0.004         0.033            0.005        0.124           0.008       0.063
                          Minus            0.050      0.171             0.004         0.030            0.004        0.114           0.009       0.059
                        Abs. minus         0.130      0.238             0.023         0.119            0.249        0.750           0.130       0.540

                                                     Table 6: Performance Evaluation of Link Prediction
                          Blogcatalog                                 Yelp                                       DBLP                                   U.S. Patents
                      MAP        recall@100                    MAP         recall@100                      MAP       recall@100                    MAP           recall@100
 DeepWalk            0.124           0.227                    *0.021           0.110                       0.230        *0.710                     0.093             0.500
   LINE              *0.134         *0.249                     0.017           0.104                       0.086        0.580                      0.091             0.400
 node2vec            0.125           0.229                    *0.021          *0.111                      *0.231        *0.710                     0.095            *0.503
   PTE               0.067           0.139                     0.004           0.034                       0.071        0.324                      0.030             0.243
   HINE              0.085           0.179                     0.016           0.097                       0.205        0.697                     *0.103             0.495
   ESim              0.132           0.185                       x               x                         0.179        0.633                        x                 x
   MPE            0.141(5.0%) 0.279(10.8%)                 0.028(31.8%) 0.138(24.3%)                   0.265(12.8%) 0.751(5.8%)                0.176(70.8%) 0.602(19.9%)


which can be used for meta-path analysis. Empirically, we demon-                                      Tara N Sainath. 2012. Deep neural networks for acoustic modeling in speech
strate that the proposed HIN2Vec model is able to automatically                                       recognition: The shared views of four research groups. IEEE Signal Processing
                                                                                                      Magazine 29, 6 (2012), 82–97.
learn feature vectors for nodes in HINs to support a variety of HIN                              [13] Zhipeng Huang and Nikos Mamoulis. Heterogeneous Information Network Em-
applications, including multi-label node classification and link pre-                                 bedding for Meta Path based Proximity. arXiv preprint arXiv:1701.05291 (⁇⁇).
                                                                                                 [14] Ming Ji, Jiawei Han, and Marina Danilevsky. 2011. Ranking-based classification
diction, in several real-world networks. HIN2Vec also soundly out-                                    of heterogeneous information networks. In Proceedings of the ACM International
performs all the compared models in those experiments.                                                Conference on Knowledge Discovery and Data Mining (SIGKDD 2011).
   As for our next step, we plan to explore the regularization for                               [15] Jon M Kleinberg. 2000. Navigation in a small world. Nature 406, 6798 (2000).
                                                                                                 [16] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classi-
representations of nodes to learn sparse representations, which                                       fication with deep convolutional neural networks. In Proceedings of the Annual
may capture more distinct latent topics of each nodes and meta-                                       Conference on Neural Information Processing Systems (NIPS 2012). 1097–1105.
paths.                                                                                           [17] Richard A Kronmal and Arthur V Peterson Jr. 1979. On the alias method for gen-
                                                                                                      erating random variables from a discrete distribution. The American Statistician
                                                                                                      33, 4 (1979), 214–218.
ACKNOWLEDGMENTS                                                                                  [18] Ni Lao and William W. Cohen. 2010. Relational Retrieval Using a Combination
                                                                                                      of Path-constrained Random Walks. Machine Learning 81 (Oct. 2010), 53–67.
This work is supported in part by the National Science Foundation                                [19] Ni Lao and William W Cohen. 2010. Relational retrieval using a combination of
under Grant No. IIS-1717084 and SMA-1360205.                                                          path-constrained random walks. Machine learning 81, 1 (2010), 53–67.
                                                                                                 [20] David Liben-Nowell and Jon Kleinberg. 2007. The link-prediction problem for
                                                                                                      social networks. Journal of the American society for information science and
REFERENCES                                                                                            technology 58, 7 (2007), 1019–1031.
                                                                                                 [21] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.
 [1] 2009. BlogCatalog3. http://socialcomputing.asu.edu/datasets/BlogCatalog3.
                                                                                                      Distributed representations of words and phrases and their compositionality. In
     (2009).
                                                                                                      Proceedings of the Annual Conference on Neural Information Processing Systems
 [2] 2014.     USPTO PatentView.          http://www.dev.patentsview.org/workshop/
                                                                                                      (NIPS 2013). 3111–3119.
     participants.html. (2014).
                                                                                                 [22] Abdel-rahman Mohamed, George E Dahl, and Geoffrey Hinton. 2012. Acoustic
 [3] 2017. How can I download the whole dblp dataset. http://dblp.uni-trier.de/faq/
                                                                                                      modeling using deep belief networks. IEEE Transactions on Audio, Speech, and
     How+can+I+download+the+whole+dblp+dataset. (2017).
                                                                                                      Language Processing 20, 1 (2012), 14–22.
 [4] 2017. Yelp dataset. https://www.yelp.com/dataset_challenge. (2017).
                                                                                                 [23] Tore Opsahl and Pietro Panzarasa. 2009. Clustering in weighted networks. Social
 [5] Joachim H Ahrens and Ulrich Dieter. 1989. An alias method for sampling from
                                                                                                      networks 31, 2 (2009), 155–163.
     the normal distribution. Computing 42, 2-3 (1989), 159–170.
                                                                                                 [24] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learn-
 [6] Albert-László Barabási and Réka Albert. 1999. Emergence of scaling in random
                                                                                                      ing of social representations. In Proceedings of the ACM International Conference
     networks. Science 286, 5439 (1999), 509–512.
                                                                                                      on Knowledge Discovery and Data Mining (SIGKDD 2014). ACM, 701–710.
 [7] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013. Representation learn-
                                                                                                 [25] Jingbo Shang, Meng Qu, Jialu Liu, Lance M Kaplan, Jiawei Han, and Jian Peng.
     ing: A review and new perspectives. IEEE Transactions on Pattern Analysis and
                                                                                                      2016. Meta-Path Guided Embedding for Similarity Search in Large-Scale Het-
     Machine Intelligence 35, 8 (2013), 1798–1828.
                                                                                                      erogeneous Information Networks. arXiv preprint arXiv:1610.09769 (2016).
 [8] Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C Aggarwal, and
                                                                                                 [26] Yizhou Sun, Rick Barber, Manish Gupta, Charu C. Aggarwal, and Jiawei Han.
     Thomas S Huang. 2015. Heterogeneous network embedding via deep archi-
                                                                                                      2011. Co-author Relationship Prediction in Heterogeneous Bibliographic Net-
     tectures. In Proceedings of the 21th ACM SIGKDD International Conference on
                                                                                                      works. In Proc. of the 2011 International Conference on Advances in Social Net-
     Knowledge Discovery and Data Mining. ACM, 119–128.
                                                                                                      works Analysis and Mining (ASONAM 2011). Kaohsiung, Taiwan, 121–128.
 [9] Dan Ciregan, Ueli Meier, and Jürgen Schmidhuber. 2012. Multi-column deep
                                                                                                 [27] Jian Tang, Meng Qu, and Qiaozhu Mei. 2015. Pte: Predictive text embedding
     neural networks for image classification. In Proceedings of the IEEE International
                                                                                                      through large-scale heterogeneous text networks. In Proceedings of the ACM In-
     Conference on Computer Vision and Pattern Recognition (CVPR 2012). IEEE.
                                                                                                      ternational Conference on Knowledge Discovery and Data Mining (SIGKDD 2015).
[10] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable Feature Learning
                                                                                                 [28] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
     for Networks. (2016).
                                                                                                      2015. LINE: Large-scale information network embedding. In Proceedings of the
[11] Huan Gui, Jialu Liu, Fangbo Tao, Meng Jiang, Brandon Norick, and Jiawei Han.
                                                                                                      International Conference on World Wide Web (WWW 2015). ACM, 1067–1077.
     2016. Large-Scale Embedding Learning in Heterogeneous Event Data. (2016).
                                                                                                 [29] Duncan J Watts and Steven H Strogatz. 1998. Collective dynamics of small-world
[12] Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed,
                                                                                                      networks. Nature 393, 6684 (1998), 440–442.
     Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, and




                                                                                          1806

