# Deep Variational Network Embedding in Wasserstein Space Cui Wang Zhu 2018

> Source: `Deep_Variational_Network_Embedding_in_Wasserstein_Space_Cui_Wang_Zhu_2018.pdf`

---

       Deep Variational Network Embedding in Wasserstein Space
                                   Dingyuan Zhu∗                                                                          Peng Cui
                                  Tsinghua University                                                               Tsinghua University
                                   zhudy11@126.com                                                                 cuip@tsinghua.edu.cn

                                     Daixin Wang                                                                       Wenwu Zhu
                                Tsinghua University                                                               Tsinghua University
                              dxwang0826@gmail.com                                                               wwzhu@tsinghua.edu.cn

ABSTRACT                                                                                        August 19–23, 2018, London, United Kingdom. ACM, New York, NY, USA,
Network embedding, aiming to embed a network into a low di-                                     10 pages. https://doi.org/10.1145/3219819.3220052
mensional vector space while preserving the inherent structural
properties of the network, has attracted considerable attentions
recently. Most of the existing embedding methods embed nodes                                    1   INTRODUCTION
as point vectors in a low-dimensional continuous space. In this                                 Network embedding has attracted considerable research attentions
way, the formation of the edge is deterministic and only determi-                               in the past few years. The basic idea is to embed a network into a
ned by the positions of the nodes. However, the formation and                                   low-dimensional vector space to preserve the network structure.
evolution of real-world networks are full of uncertainties, which                               Many network embedding methods are demonstrated to be effective
makes these methods not optimal. To address the problem, we                                     in a variety of applications, such as link prediction [42, 44], classi-
propose a novel Deep Variational Network Embedding in Wasser-                                   fication [8, 26] and clustering [35, 46]. However, most of existing
stein Space (DVNE) in this paper. The proposed method learns                                    network embedding methods represent each node by a single point
a Gaussian distribution in the Wasserstein space as the latent re-                              in a low-dimensional vector space. In this way, the formation of
presentation of each node, which can simultaneously preserve the                                the whole network structure is deterministic.
network structure and model the uncertainty of nodes. Specifically,                                Actually, real-world networks are much more complex than we
we use 2-Wasserstein distance as the similarity measure between                                 assume. The formation and evolution of the networks are full of
the distributions, which can well preserve the transitivity in the                              uncertainties. For example, for the nodes with low degree, they
network with a linear computational cost. Moreover, our method                                  contain less information and thus their representations bear more
implies the mathematical relevance of mean and variance by the                                  uncertainties than others. For the nodes across multiple communi-
deep variational model, which can well capture the position of the                              ties, the possible contradiction between their neighboring nodes
node by the mean vectors and the uncertainties of nodes by the va-                              may also be larger and thus cause the uncertainty. Furthermore, in
riance. Additionally, our method captures both the local and global                             social network, human behavior is multi-faceted which also ma-
network structure by preserving the first-order and second-order                                kes the generation of edges uncertain [47]. For all of these cases,
proximity in the network. Our experimental results demonstrate                                  without considering the uncertainty of networks, the learned em-
that our method can effectively model the uncertainty of nodes in                               beddings will be less effective in network analysis and inference
networks, and show a substantial gain on real-world applications                                tasks.
such as link prediction and multi-label classification compared with                               Gaussian distribution innately represents the uncertainty pro-
the state-of-the-art methods.                                                                   perty [43]. Therefore, it is promising to represent a node by Gauss-
                                                                                                ian distributions, i.e. the mean and the variance, rather than a point
KEYWORDS                                                                                        vector to incorporate the uncertainty. Motivated by this, to model
Network Embedding, Wasserstein space, Deep Learning                                             the uncertainty of each node using Gaussian distributions, there
ACM Reference Format:                                                                           are some basic requirements for network embedding methods to
Dingyuan Zhu, Peng Cui, Daixin Wang, and Wenwu Zhu. 2018. Deep Varia-                           meet.
tional Network Embedding in Wasserstein Space. In KDD ’18: The 24th ACM                             • Transitivity: The embedding space should be a metric space
SIGKDD International Conference on Knowledge Discovery & Data Mining,
                                                                                                      to preserve the transitivity in networks. Transitivity is a
∗ Beijing National Research Center for Information Science and Technology(BNRist)                     very important property in networks, especially in social
                                                                                                      networks [25]. For example, the friend of my friend is more
Permission to make digital or hard copies of all or part of this work for personal or                 likely to be my friend than some randomly chosen users.
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation             Moreover, the transitivity measures the density of triangles
on the first page. Copyrights for components of this work owned by others than the                    in a network, which plays an important role in calculating
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
                                                                                                      clustering coefficient [7]. If the metric space satisfies the
and/or a fee. Request permissions from permissions@acm.org.                                           triangle inequality, the transitivity in the network can be
KDD ’18, August 19–23, 2018, London, United Kingdom                                                   well preserved.
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.                   • Uncertainty: By using Gaussian distributions to represent
ACM ISBN 978-1-4503-5552-0/18/08. . . $15.00
https://doi.org/10.1145/3219819.3220052                                                               a node, the mean and the variance should preserve different
      properties to make such representations informative. Speci-              • We comprehensively evaluate the effectiveness of DVNE on
      fically, the mean vectors should reflect the position of the               several real-world networks in various applications.
      nodes and variance terms should contain the uncertainty of         The rest of the paper is organized as follows. In Section 2, we review
      the nodes. In this way, the representations based on distribu-     the related work. In Section 3, we summarize the notations used
      tions can preserve the uncertainty while supporting network        in this paper and give the problem formulation. We introduce the
      applications.                                                      framework of the method in Section 4 and report the experimental
    • Strcutural Proximity: The network structures, especially           results in Section 5. We conclude the paper in Section 6.
      high-order proximity, should be preserved in a effective and
      efficient way. The high-order proximity is critical for captu-     2     RELATED WORK
      ring the network structure, which has been demonstrated to
                                                                         Because of the popularity of networked data, network embedding
      be useful in many real-world applications [36].
                                                                         has received more and more attentions in recent years. We briefly
   Recently, some works attempt to use Gaussian distributions to         review some network embedding methods, and readers can referred
represent a node for network embedding [3, 17, 24] to integrate un-      to [13] for a comprehensive survey. Deepwalk [37] first uses the
certainty. However, these methods use the Kullback-Leibler (abbre-       language modeling technique to learn the latent representations
viated as KL) divergence [28] to measure the similarity between          of a network by truncated random walks. LINE [39] embeds the
distributions. However, the KL divergence is asymmetric and does         network into a low-dimensional space where the first-order and
not satisfy triangle inequality. Thus, it can not well preserve the      second-order proximity between nodes are preserved. Node2vec
transitivity of proximity in networks, especially in undirected net-     [22] learns a mapping of nodes to a low-dimensional space of fe-
works. Additionally, these methods regard the variance terms as          atures that maximizes the likelihood of preserving network neig-
additional dimensions of mean vectors, and use similarity measure        hborhoods of nodes. HOPE [36] proposes a high-order proximity
to constrain their learning. In this way, they do not reflect the in-    preserved embedding method. Furthermore, deep learning method
trinsic relationship between variance terms and mean vectors in          for network embedding is also studied. SDNE [44] first considers
the model. Finally, very few of these works preserve the high-order      the high nonlinearity in network embedding and proposes a deep
proximity in network embedding, except Graph2Gauss [3]. But              autoencoder to preserve the first- and the second-order proximities.
Graph2Gauss needs to calculate the shortest path between any two         The graph variational autoencoder (GAE) [27] learns node embed-
nodes, which is unaffordable in large-scale networks.                    dings in an unsupervised manner with variational autoencoder
   To address these problems, we propose a novel Deep Variational        (VAE) [16].
Network Embedding in Wasserstein Space method in this paper,                All the aforementioned methods learn a point-vector for each
named DVNE. The proposed method learns a Gaussian Embedding              node as its embedding. However, as we stated before, these methods
for each node in the Wasserstein Space by the deep variational mo-       have the limitation to model the uncertainty, which is a critical
del. Specifically, we employ 2-Wasserstein distance to measure the       property needed to be considered for network embedding. Then
similarity between the distributions, i.e. the embeddings of the no-     some following works start to consider the uncertainty problem.
des. The 2-Wasserstein distance is a real metric that able to preserve   Inspired by [43], which learns the Gaussian word embeddings to
the transitivity in embedding space. In this way, the proposed deep      model uncertainty, KG2E [24] learns Gaussian embeddings for kno-
model is able to simultaneously preserve the transitivity and model      wledge graphs. HCGE [17] similarly learns Gaussian embeddings
the node uncertainty with linear time complexity. Meanwhile, we          for heterogeneous graphs. And Aleksandar et al. [3] proposes a
use a deep variational model to minimize the Wasserstein distance        deep model to learn Gaussian embeddings on the attributed net-
between the model distribution and the data distribution, which          work. All of these methods use the KL divergence or its variant
can extract the intrinsic relationship between mean vectors and          JensenShannon divergence [19] as the similarity measure between
variance terms. Furthermore, our method efficiently preserve the         the distributions. However, both the KL divergence and the Jen-
first-order and second-order proximity of the nodes in networks,         senShannon divergence are not the true metrics. These metrics
empowering the learned node representations to reflect both local        do not satisfy the triangle inequality. In this way, these methods
and global network structure [44].                                       cannot preserve the transitivity to get effective representations for
   The main contributions of our method are summarized as fol-           networks. Furthermore, these methods regard the variance terms as
lows:                                                                    the extra dimensions, then use the similarity measure to constrain
    • We propose DVNE, an novel method that learns the Gaussian          their learning. In this way, it is difficult to capture the intrinsic
      embedding in the Wasserstein space, which can well preserve        relationships between the mean and the variance terms.
      the transitivity in networks and reflect the uncertainties of
      nodes.                                                             3     NOTATIONS AND PROBLEM DEFINITION
    • We imply the mathematical relevance of mean vectors and            In this section, we summarize the notations used in this paper and
      variance terms by the deep variational model, where the            give the problem formulation.
      mean vectors denote the position of the nodes and the vari-
      ance terms represent the uncertainties of the nodes.               3.1     Notations
    • We efficiently preserve the first-order and second-order prox-     We first summarize the notations used in this paper. A network
      imity between nodes, thus the learned representations cap-         is defined as G = {V, E}, where V = {v 1 , v 2 , ..., v N } denotes a set
      ture the local and global network structure.                       of nodes and N is the number of the nodes. E is the set of edges
between the nodes, and M = |E| is the number of the edges. In
                                                                                                                               Parameter sharing


this paper, we mainly consider undirected networks. Let Nbrsi =
{v j |(vi , v j ) ∈ E} denote the set of neighbors of node vi . Let P ∈                    𝒙i              D                   𝒙j           D            𝒙k           D

RN ×N be the transition matrix, where P(i, :) and P(:, j) denote its
i th row and j t h column respectively and P(i, j) is the element of                  decoder              …              decoder            …      decoder           …

the i t h row and j t h column. If there is an edge from vi to v j and               sample εi
                                                                                                 zi
                                                                                                                         sample εj
                                                                                                                                     zj
                                                                                                                                                   sample εk
                                                                                                                                                               zk

the degree of node vi is di , then we set P(i, j) to d1 , otherwise        Ranking
                                                         i                                                 σi             µj                σj
                                                                                      µi                                                            µk                σk
we mark P(i, j) with zero. We define hi = N (µ i , Σi ) as a lower-         Loss
                                                                                                           …                                …                         …
dimensional Gaussian distribution embedding for node vi , where                       encoder                             encoder                   encoder

µ i ∈ RL , Σi ∈ RL×L . L is the embedding dimension, which satisfies                                  µi

                                                                                           𝒙i              D                   𝒙j            D           𝒙k            D
L ≪ N . In this paper, we focus on diagonal covariance matrices.                                      Node i                                                        Node k
                                                                                                                                          Node j


3.2    Problem Definition                                                                         Figure 1: The framework of DVNE.
In this paper, we focus on the problem of network embedding with
first-order and second-order proximity preserved.

  Definition 3.1. (First-Order Proximity) The first-order proximity       4  DEEP VARIATIONAL NETWORK
describes the pairwise proximity between nodes. For any pair of
                                                                             EMBEDDING
nodes, if P(i, j) > 0, there exists positive first-order proximity bet-
ween vi and v j . Otherwise, the first-order proximity between vi         4.1 Framework
and v j is 0.                                                             In this paper, we propose a novel model to perform network em-
                                                                          bedding, namely DVNE, whose framework is shown in Figure 1.
   The first-order proximity implies that two nodes in real-world         Basically, we propose a deep architecture, which is composed of
networks are similar if they are linked by an observed edge. For ex-      multiple nonlinear mapping functions to map the input data to the
ample, if two users build a relationship between them on the social       Wasserstein space to preserve the uncertainties of the nodes and
network, they may have a common interest. However, real-world             capture the network structure. Specifically, we first use a ranking
networks are usually so sparse that we can only observe a very            based loss function on the Wasserstein embedding space, aiming to
limited number of links. Only capturing the first-order proximity         make nodes with edges similar and without edges dissimilar. In this
is not sufficient,thus we introduce the second-order proximity to         way, the first-order proximity is preserved. Furthermore, we use a
capture the global network structure.                                     deep variational model to preserve the second-order proximity, by
                                                                          reconstructing the neighborhood structure of each node. Meanw-
   Definition 3.2. (Second-Order Proximity) The second-order prox-        hile, the whole deep variational model implies the the mathematical
imity between a pair of nodes denotes the similarity between their        relevance of mean vectors and variance terms explicitly by the sam-
neighborhood network structures. Then the second-order proxi-             pling process. In this way, the mean vectors find an approximate
mity between vi and v j is determined by the similarity between           position of the node and the variance term capture the uncertainty.
Nbrsi and Nbrsj . If none of nodes is linked with both vi and v j ,       In the following sections, we will introduce how to realize the deep
the second-order proximity between vi and v j is 0.                       model in detail.

   Intuitively, the second-order proximity assumes that if two nodes
share common neighbors, they tend to be similar. The second-order
                                                                          4.2        Similarity Measure
proximity has been demonstrated to be a good metric to define the         To support network applications, we need to define a suitable si-
similarity of a pair of nodes, even if there is no edge between them      milarity measure between the latent representations of two nodes.
[31]. Moreover, the second-order proximity has been proved to be          Since we use distributions to represent our latent representations
able to alleviate the sparsity problem of the first-order proximity       to incorporate uncertainty, the similarity measure should be able to
and better preserve the global structure of the network [39].             measure the similarity between the distributions. Furthermore, as
   With the first- and second-order proximity, then we define our         transitivity is a important property of the network, the similarity
network embedding problem as follows:                                     measure should simultaneously preserve the transitivity between
                                                                          nodes. Through extensive studies, we find that the Wasserstein
   Definition 3.3. (Gaussian-Based Network Embedding) Given a             distance is able to measure the similarity between two distributions
network G = {V, E}, we aim to represent each node vi as a lower-          while simultaneously satisfies the triangle inequality [9], which gua-
dimensional Gaussian distribution hi = N (µ i , Σi ), where µ i cap-      rantees its ability to preserve the transitivity of similarity between
tures the position of the nodes in the embedding space and Σi             nodes.
investigates the uncertainty of the nodes. Meanwhile, the latent             The p th Wasserstein distance between two probability measures
representations aim to preserve the first-order proximity and the         µ and ν is defined as:
second-order proximity between the nodes to preserve the network
                                                                                                                Wp (µ, ν )p = inf E d(X , Y )p ,
                                                                                                                                             
structure.                                                                                                                                                                   (1)
where E[Z ] denotes the expected value of a random variable Z and                      pairs, which makes the energy of positive examples to be lower
the infimum is taken over all joint distributions of the random vari-                  than that of negative examples. Equivalently, it will make the simi-
ables X and Y with marginals µ and ν respectively. Moreover, when                      larity between the positive examples larger than that of negative
p ≥ 1, the p t h Wasserstein distance preserves all properties of a                    examples, thus helps preserve the first-order proximity.
metric [1], including both the symmetry and the triangle inequality                        For second-order proximity, we use the transition matrix P as our
[6]. In this way, Wasserstein distance is suitable to be a similarity                  input features and propose a variant of Wasserstein Auto-Encoders
measure between the latent representation of nodes, especially for                     (WAE) [41] as the model to preserve the neighborhood structure.
an undirected network.                                                                 WAE is a deep variational model, which can imply the mathematical
    But the calculation of the general-formed Wasserstein distance                     relevance of mean vectors and variance terms by the sampling
is limited by a heavy computational cost, which poses a great chal-                    process. The objective of original WAE is composed of two terms,
lenge to network applications. To reduce the computational cost, in                    the reconstruction cost and the regularizer. The reconstruction
our case since we use Gaussian distributions for the latent represen-                  cost aims to capture the information of the input. The regularizer
tation of nodes, the 2t h Wasserstein distance (abbreviated as W2 )                    encourages the encoded training distributions to match the prior
has the closed form solution to speed-up the calculation process.                      distribution. As for our problem, the P(i, :) shows the neighborhood
The W2 distance has also been widely used in in computer vision                        structure of node vi , thus we use P(i, :) as the input feature to the
[4, 11], computer graphics [5, 15] or machine learning [12, 14].                       WAE for node vi and reconstruct it to preserve its neighborhood
    More specifically, we have the following formula to calculate W2                   structure. For the regularization term, it is hard to define the prior
distance between two Gaussian distributions [20]:                                      distribution of each node in the network. Therefore, we focus only
       dist = W2 (N (µ 1 , Σ1 ), N (µ 2 , Σ2 ))                                        on the reconstruction cost to preserve the neighborhood structure.
                                                                  (2)                      Let PX denote the data distribution, and PG denote the encoded
                                                 1/2   1/2
       dist 2 = ∥µ 1 − µ 2 ∥22 + Tr(Σ1 + Σ2 − 2(Σ1 Σ2 Σ1 )1/2 )                        training distribution. The reconstruction cost can be represented
   In this paper we focus on diagonal covariance matrices1 , thus                      as:
Σ1 Σ2 = Σ2 Σ1 . Then the formula (2) can be simplified as:                                    DW AE (PX , PG ) =           EPX EQ (Z |X ) c(X , G(Z )) , (6)
                                                                                                                                                     
                                                                                                                    inf
                                                                                                                     Q (Z |X )∈Q
                                                             1/2      1/2
  W2 (N (µ 1 , Σ1 ); N (m 2 , Σ2 ))2 = ∥µ 1 − µ 2 ∥22 + ∥Σ1        − Σ2 ∥F2 . (3)
                                                                                       where Q is the encoders and G is the decoders, X ∼ PX and Z ∼
   According to the above equation, the time complexity of calcula-                    Q(Z |X ). It aims to minimize Wasserstein distance between the PX
ting W2 distance between the latent representation of two nodes is                     and PG .
linear with the embedding dimension L. Therefore, we choose W2                            According to [41], when using c(x, y) = ∥x − y∥22 , the above loss
distance as the similarity measure, and the computational costs no                     function (6) minimizes the W2 distance between PX and PG , thus
longer constitute limitations.                                                         PG captures the information of the input data in the Wasserstein
                                                                                       space.
4.3     Loss Functions                                                                    Considering the sparsity of the transition matrix P, we focus on
Our overall loss functions for DVNE consists of two parts, the                         non-zero elements in P to speed up our model. Thus, we present
ranking-based loss to preserve the first-order proximity and the                       the loss function as follows to preserve the second-order proximity:
reconstruction loss to preserve second-order proximity.
                                                                                                 L2 =            EPX EQ (Z |X ) ∥X ◦(X − G(Z ))∥22 ,
                                                                                                                                                 
   First, we consider how to preserve the first-order proximity.                                           inf                                           (7)
                                                                                                       Q (Z |X )∈Q
Intuitively, we want all nodes which are linked with vi to be closer
to vi w.r.t. their embedding, compared to the nodes that have no                       where ◦ means the element-wise multiplication.
edge with vi . More specifically, we propose the following pairwise                        In our model, we use the transition matrix P as the input feature
constraints to preserve the first-order proximity:                                     X . The reconstruction process will make the nodes with similar
                                                                                       neighborhoods have similar latent representations. Therefore, the
 W2 (hi , hj ) < W2 (hi , hk ), ∀vi ∈ V, ∀v j ∈ Nbrsi , ∀vk < Nbrsi . (4)              second-order proximity between nodes is preserved.
where hi is the latent representation of node vi , Nbrsi is the set of                     To preserve first-order proximity and second-order proximity of
neighbors of node vi . The smaller the W2 distance, the larger the                     networks simultaneously, we jointly minimize the loss function by
similarities between nodes.                                                            combining Eq. (5) and Eq. (7):
   Then we use a energy based learning approach [29] to incorpo-                                                      L = L1 + α L2 .                            (8)
rate all of the pairwise constraints defined in the above equation.
Mathematically, denoting Ei j = W2 (hi , hj ) as the energy between
                                                                                       4.4     Optimization
two nodes, we present the objective function as follows:
                          Õ                                                            For large graphs, optimizing objective function (5) is computatio-
                  L1 =         (Ei j 2 + exp(−Eik )),              (5)                 nally expensive, which requires to calculate the all valid triplets in
                            (i, j,k )∈D                                                D. Therefore, we sample triplets from D uniformly, which replace
                                                                                       Í
where D is the set of all valid triplets given in Eq. (4). The above                     (i, j,k )∈D with E(i, j,k )∼D in Eq. (5). In details, for each iteration, we
objective function penalizes ranking errors by the energy of the                       sample M triplets from D to calculate the estimates of the gradient.
1 When the covariance matrices is not diagonal, Wang proposed an fast iterative
                                                                                          Considering objective function (7), we need sample Z from
algorithm (called BADMM) to solve the Wasserstein distance [45]. It is not the focus   Q(Z |X ), which is a non-continuous operation and has no gradient.
of the paper and we will not discuss it.                                               In this case, it is difficult for the deep models to optimize the loss
function. To solve the problem, inspired by the Variational Auto-            Algorithm 1 Training algorithm of DVNE
Encoders (VAE) [16], we can use the "reparameterization trick" to            Input: The network G = {V, E} with the transition matrix P, the
optimize the above objective equation. Mathematically, we first                  parameter α
sample ϵ ∼ N (0, I), then compute Z = µ(X ) + Σ1/2 (X ) ∗ ϵ. Given a         Output: Network embeddings {hi }i=1    N and updated parameters
fixed X and ϵ, the objective function (7) is deterministic and con-              θ = {W , b }i=1
                                                                                          (i) (i)  5
tinuous in the parameters of encoders Q and decoders G. In this               1: Initial parameters θ by xavier initialization
way, the whole model can get the gradient when performing the                 2: while L do not converge do
back-propagation, and thus we can use stochastic gradient descent             3:    Sample M triplets from D uniformly
to optimize the model.                                                        4:    Split these triplets to a number of batches
                                                                              5:    calculate partial derivative ∂L/∂θ with backpropagation
4.5         Implementation Details                                                  algorithm to update θ
For all the experiments in this paper we used an encoder and a                6: end while
decoder with a single hidden layer of size S = 512 respectively.
More specifically, to obtain the embeddings for a node vi , we have
                                                                                 • DeepWalk [37]: This algorithm learns embedding by simula-
            = Relu(xi W(1) + b(1) ), W(1) ∈ RN ×S , b(1) ∈ RS
      (1)
    yi
                                                                                   ting several uniform random walks. It assumes that a pair of
      µ i = yi W(2) + b(2) , W(2) ∈ RS ×L , b(2) ∈ RL
               (1)
                                                                                   nodes are similar if they are close in the random walks.
                                                                                 • LINE [39]: This algorithm preserves the first-order and second-
      σi = Elu(yi W(3) + b(3) ) + 1, W(3) ∈ RS ×L , b(3) ∈ RL
                     (1)
                                                                       (9)         order proximity between nodes respectively, and directly
      zi = µ i + σi ∗ ϵ, ϵ ∼ N (0, I)                                              concatenates the representations for the first-order and second-
                                                                                   order proximity.
            = Relu(zi W(4) + b(4) ), W(4) ∈ RL×S , b(4) ∈ RS
      (2)
    yi
                                                                                 • SDNE [44]: This method learns a point-vector for each node
      bi = Siдmoid(y(2) W(5) + b(5) ), W(5) ∈ RS ×N , b(5) ∈ RN ,
      x                                                                            with preserving the first and the second order proximities
where x i is P(i, :), Relu [34] and Elu [10] are the rectified linear unit         simultaneously using deep models.
and exponential linear unit. We use elu() + 1 to guarantee that σi               • Graph2Gauss(G2G_oh) [2]: This method aims to learn the
is positive. Because the range of values in x i is between [0, 1], we              lower-dimensional Gaussian distribution embedding by ran-
use the sigmoid function as the output function of the last hidden                 king similarity based on the shortest path between nodes. As
layer.                                                                             the datasets have no attribute information, we compare with
                                                                                   the one-hot encoding version of Graph2Gauss as described
4.6         Complexity analysis                                                    in the paper.
Algorithm 1 lists the procedures of our method. During the training             5.1.2 Dataset. In order to comprehensively evaluate the effecti-
procedure, the time complexity of calculating gradients and upda-            veness of our proposed method, we use four different real-world
ting parameters is O(T × M × (dave S + SL + L)), where M is the              datasets, including citation networks and social networks. The de-
number of the edges, dave is the average degree of all nodes, L is the       tailed information is shown as follows:
dimension of embedding vectors, S is the size of hidden layer of the             • Cora : This is a research paper set constructed by McCal-
encoder and decoder, T is the number of iterations. Since we only                  lum et al. [33], which consists of 2708 scientific publications
reconstruct non-zero elements in x i , the computational complexity                classified into one of seven classes.
of the first and last hidden layers is O(dave S). The computational              • Facebook : It is a typical social network dataset without node
complexity of other hidden layers is O(SL), and it takes O(L) to                   labels constructed by J. McAuley et al. [30].
calculate the W2 distance between the distributions. In practice we              • BlogCatalog[38]: This is a network of social relationships of
found that a small number of iterations T (T ≤ 50 for all shown                    the bloggers listed on the BlogCatalog website. The labels
experiments) is needed for convergence.                                            represent the topic categories provided by the authors.
                                                                                 • Flickr [38]: It is a social network where node represents
5     EXPERIMENT                                                                   users and edges correspond to friendships between users.
In this section, we empirically evaluate the effectiveness of the our              The labels represent the interest groups of the users.
method.                                                                      All the networks are undirected, and the detailed statistics of the
                                                                             datasets are summarized in Table 1.
5.1         Experiment Setting
We first introduce the experiment setting before presenting results             5.1.3 Parameter Settings. In all experiments, we set the em-
of the experiments.                                                          bedding dimension L = 128 unless stated. For the equality, all the
                                                                             methods that learn the embedding as the distribution use the length
   5.1.1 Baseline Methods. We use the following five methods as              of mean vector and variance terms to match L. Specifically, our met-
the baselines.                                                               hod actually uses half of the dimensionality L as the length of mean
     • DVNE_kl : In order to show the advantages of W2 distance              vector in all experiments.
        in undirected network. We replace the similarity measure in             For DVNE and DVNE_kl, the hyper-parameters of α are tuned by
        our method with the KL divergence.                                   using grid search on the validation set. We use xavier initialization
Table 1: Statistics of datasets. |V | denotes the number of no-          experiments, we randomly hide 20% of the edges as the testing
des , |E| denotes the number of edges and |C | denotes the               network and train the embeddings on the rest of the network. After
number of classes.                                                       the training, we can obtain the embedding for each node and then
                                                                         use the embeddings to predict the unobserved edges. The pairs of
               Cora    Facebook    BlogCatalog       Flickr              nodes are ranked in a similar way as network reconstruction and
        |V |   2,708     4,039        10,312        80,513               the top ranking pairs are evaluated on the testing network. Unlike
        |E|    5,429    88,234       333,983       5,899,882             the reconstruction task, this task predicts the unobserved edges
        |C |     7         -            39            195                in testing network instead of reconstructing the existing edges in
                                                                         training network. We still use AUC as the evaluation metric.

[21] for all weight matrices. The parameters are optimized using
RMSProp [40] with a fixed learning rate of 0.001.                                   Table 3: AUC scores for Link Prediction.
   The parameters for baselines are tuned to be optimal. For Deep-
Walk, we set window size as 10, walk length as 40, walks per node                            Cora     Facebook     BlogCatalog     Flickr
as 10. For LINE, we set the number of negative samples as 5, and                 DVNE        0.947      0.982         0.945        0.942
line search for the optimal value of the training samples on dif-              DVNE_kl       0.919      0.930         0.917        0.908
ferent datasets. For SDNE, we use the default parameter settings               DeepWalk      0.880      0.923         0.827        0.931
and the multi-layer deep structure in the author’s implementation.                Line       0.854      0.882         0.802        0.919
For G2G_oh, we use the default parameter settings and the fixed                  SDNE        0.917      0.931         0.920        0.927
learning rate in the implementation details of the paper.                       G2G_oh       0.901      0.925         0.903        0.906

5.2    Network Reconstruction
The most primal objective for network embedding is to reconstruct            From the results in Table 3, our proposed method still outper-
the given network, ans a good network embedding method should            forms the baselines in all datasets. Especially on the facebook data-
ensure that the learned embeddings can preserve the original net-        set, our method significantly improve AUC scores by 0.05 than the
work structure. Thus, we first provide a basic evaluation on different   baselines. From the results, we have the following analysis:
network embedding methods with respect to their capability of net-           Deepwalk can introduce high-order proximity by changing the
work reconstruction. More specifically, we use different network         parameter of window size, but it can not balance the weight of the
embedding methods to learn the embedding vectors on the different        first-order proximity and the high-order proximity. This means it
real-world networks. Then we rank pairs of nodes according to            can not handle well both reconstruction task and prediction task at
their trained similarities between the embedding of nodes, i.e. the      the same time, which is evident from the experimental results. We
W2 distance for our method, the KL divergence for G2G_oh. The            also find that LINE does not achieve as good performance as other
larger the similarities between pairs of nodes, the more likely they     methods do in most cases. The reason may be twofold. Firstly, LINE
have the edges. Then we can use the top ranking pairs to recon-          adopts shallow structure, which is difficult to capture the highly
struct the edges of the original networks. For the evaluation metric,    non-linear structure [44] in the network. Moreover, LINE directly
we use Area Under Curve (AUC) [18].                                      concatenates the embeddings for the first-order and second-order
                                                                         proximity, which is sub-optimal than jointly optimizing them in
      Table 2: AUC scores for Network Reconstruction.                    our method.
                                                                             Although DVNE and SDNE both exploit the first-order and
                   Cora     Facebook     BlogCatalog     Flickr          second-order proximity to preserve the network structure, DVNE
        DVNE       0.996      0.998         0.962        0.959           achieves better performance. The reason is that our method learns
      DVNE_kl      0.940      0.958         0.937        0.925           a Gaussian distribution as an embedding for each node, allowing us
      DeepWalk     0.986      0.984         0.864        0.950           to capture uncertainty in the network by the latent representations.
                                                                         Actually, adding a new edge between two nodes is a uncertain event,
         Line      0.952      0.934         0.891        0.939
                                                                         it is more natural to describe this event from the perspective of the
        SDNE       0.992      0.960         0.958        0.917
                                                                         distributions.
       G2G_oh      0.921      0.942         0.924        0.901
                                                                             We also find that DVNE achieves a substantial gain over DVNE_kl
                                                                         on all the datasets. The reason is two fold. Firstly, the KL divergence
   The results are shown in Table 2. Our proposed method outper-         is not suitable for undirected network because of the asymmetric
forms the baseline methods in all datasets. The results demonstrate      property of the KL divergence. Secondly, the KL divergence does
that our proposed method can effectively preserve the original net-      not necessarily guarantee the transitivity of similarities between
work structure and reconstruct the network. It lays the foundation       the nodes, which makes KL-based methods worse link prediction
for other real-world applications of network embedding.                  results.
                                                                             Compared with DVNE_kl and G2G_oh, which both use the KL di-
5.3    Link Prediction                                                   vergence as the similarity measures, DVNE_kl outperforms G2G_oh.
Link prediction, aiming to predict which pairs of nodes will form        It is because that G2G_oh use the variance terms as the added di-
edges in the future, is a typical task of network embedding. In our      mensions while DVNE_kl relates the variance terms and the mean
                          BlogCatalog Dataset                                           BlogCatalog Dataset
                                                                           0.28                                               our method can encode more proximity-based information into the
                                                                           0.26
             0.4                                                                                                              mean vectors and thus perform much better than G2G_oh. Second,
                                                                           0.24

                                                                           0.22
                                                                                                                              similar to the previous task, the KL divergence is not a suitable
                                                                Macro F1
            0.35

 Micro F1
                                                                            0.2                                               similarity measure to capture the transitivity for the undirected
             0.3
                                                 DVNE
                                                 DVNE_kl
                                                                           0.18                                DVNE
                                                                                                               DVNE_kl
                                                                                                                              networks.
                                                 DeepWalk                  0.16                                DeepWalk
                                                 LINE                                                          LINE
            0.25                                 SDNE                      0.14                                SDNE

               0     0.2      0.4       0.6
                                                 G2G_oh
                                                 0.8        1
                                                                           0.12
                                                                               0   0.2      0.4       0.6
                                                                                                               G2G_oh
                                                                                                               0.8        1
                                                                                                                              5.5    Embedding Uncertainty
                       Percentage of labeled nodes                                   Percentage of labeled nodes
                                                                                                                              Learning an embedding as a distribution rather than a point-vector
                   Figure 2: Micro-F1 and Macro-F1 on BlogCatalog.                                                            allows us to capture uncertainty of the nodes. With our intuition,
                                                                                                                              the nodes that have less links with other nodes, are harder to get a
                                                                                                                              exact point-vector in the latent space. In other words, the lower the
vectors by the sampling process. Thus, DVNE is able to better cap-                                                            degree of a node, the less discriminative information it contains,
ture the uncertainties of nodes and get a better link prediction                                                              thus making its embedding more uncertain. Then we conduct the
result.                                                                                                                       following experiment to evaluate the intuition. For each node, we
   Overall, the results demonstrate that our proposed method works                                                            select its 10 dimensions with the largest variance and averaged
well for network inference tasks.                                                                                             the variance of the 10 dimensions as the variance value for the
                                                                                                                              node. Then for each network dataset, we divide the total nodes
5.4                Multi-label Classification                                                                                 into 10 parts based on their degrees. For each part of the nodes, we
Multi-label classification is another task commonly used to eva-                                                              report the relationship between their degree and their averaged
luate the effectiveness of the learned embeddings. We evaluate                                                                variance values. The Figure 4a shows the result on the all datasets.
the multi-label classification performance for three datasets (Cora,                                                          The horizontal axis represents the log10 () values of degree. Because
Blogcatalog and Flickr) that have ground-truth labels. The represen-                                                          the max degree of the node is no more than 200 in Cora, the line of
tations for the nodes are generated from the network embedding                                                                Cora is different from the other datasets.
methods and are used as features to classify each node into a set of                                                             From Figure 4a, we find that the experimental results support our
labels. For all methods based on the distribution, we only use the                                                            intuition. The nodes with higher degree contains rich information,
mean vectors as the input features in this task. We adopt a linear                                                            thus making their variance smaller. Meanwhile, we can see that
SVC [23] as the classifiers for all methods. Then, following [37],                                                            when the network is denser like Facebook and Flickr, the average
we randomly sample a portion of the labeled nodes as the training                                                             variance of embeddings is smaller. This means that our learned
data and the rest as the test. For BlogCatalog, we randomly sample                                                            embeddings of variance can reflect the density of the network.
10% to 90% of the nodes as the training samples and use the left                                                                 Moreover, to demonstrate that the uncertainty in variance terms
nodes to test the performance. For Cora and for Flickr, we randomly                                                           can help to deal with the noise edges in networks, we conduct an
sample 1% to 10% of the nodes as the training samples and use the                                                             experiment to show the benefits of the uncertainty. First, following
left nodes to test the performance on even more sparsely labeled                                                              the setting in link prediction, we randomly hide 20% of the edges
networks. We use the Micro-F 1 and Macro-F 1 scores to evaluate the                                                           as the testing network and use the rest of network as the training
performance and report results averaged over 10 trials. The results                                                           network. Then we randomly choose some pairs of nodes as the noise
are shown in Figure 2 and Figure 3 respectively.                                                                              edges and add them into the training network. We use different
   In Figure 2 and Figure 3, the curve of our method is consistently                                                          network embedding methods to learn the representations of nodes
above the curves of baseline methods. It demonstrates that our                                                                in the modified training network. Similar to link prediction task, we
method can achieve a better classification performance than ba-                                                               use the similarity between the learned node embeddings to predict
selines even if the labelled data is limited. Such an advantage is                                                            the unobserved edges in testing network. We use the results of each
meaningful for real-world applications, because the labelled data                                                             method reported in link prediction as the benchmark to calculate
in real-world network is usually scarce. The variance terms of the                                                            the percentage of AUC decline. We vary the percentage of noise
representation can help us to deal with the noise information in                                                              edges from 0.05 to 0.5, then show the percentage of AUC decline
the network, which makes the mean vectors to better capture the                                                               with respect to it in Figure 4b.
network structure. Therefore, the learned network embedding of                                                                   From the results shown in Figure 4b, we can see that the per-
our method can better generalize to the classification task than                                                              formance of our method is least affected by the noise edges. It
baselines.                                                                                                                    demonstrates that our method can better deal with the noise edges
   In most cases, the performance of G2G_oh is the worst among all                                                            in networks by capturing the uncertainties of the nodes. DeepWalk
the compared network embedding methods. The reasons are two-                                                                  adopts random walk to generate network representations. Each
fold. First, G2G_oh uses the variance terms as the added dimensions,                                                          node walks to other communities with a lower probability in the
causing part of the information of the proximity between nodes                                                                modified training network. Thus, DeepWalk can still preserve the
included in variance terms. In this way, the performance of G2G_oh                                                            original network structure and the result of DeepWalk is also good.
greatly degrades. Our method, by using the deep variational model,                                                            DVNE_kl uses the KL divergence as the similarity measure, which
makes the mean vectors and the variance terms capture different                                                               can not well preserve the transitivity in the networks. The noise
properties of the network, i.e. the mean vector captures the proxi-                                                           edges between nodes will further damage this property, leading to
mity and the variance term captures the uncertainty. In this way,                                                             worse results. For G2G_oh, there is a weak connection between
                                                        Cora Dataset                                                                                           Cora Dataset                                                                       Flickr Dataset                                                                         Flickr Dataset
                               0.75
                                                                                                                                         0.7                                                                                                                                                              0.2

                                                                                                                                                                                                                 0.3
                                0.7
                                                                                                                                        0.65                                                                                                                                                             0.15


                    Micro F1                                                                                                 Macro F1                                                                Micro F1                                                                                 Macro F1
                                                                                                                                                                                                                0.25
                               0.65

                                                                               DVNE                                                      0.6                                       DVNE                                                                              DVNE                                 0.1                                               DVNE
                                                                                                                                                                                                                 0.2
                                                                               DVNE_kl                                                                                             DVNE_kl                                                                           DVNE_kl                                                                                DVNE_kl
                                0.6                                            DeepWalk                                                                                            DeepWalk                                                                          DeepWalk                                                                               DeepWalk
                                                                               LINE                                                                                                LINE                                                                              LINE                                                                                   LINE
                                                                                                                                        0.55                                                                    0.15                                                                                     0.05
                                                                               SDNE                                                                                                SDNE                                                                              SDNE                                                                                   SDNE
                                                                               G2G_oh                                                                                              G2G_oh                                                                            G2G_oh                                                                                 G2G_oh
                               0.55
                                   0         0.02     0.04       0.06     0.08           0.1                                               0         0.02     0.04       0.06     0.08         0.1                 0                    0.02     0.04       0.06     0.08         0.1                              0           0.02     0.04       0.06     0.08     0.1
                                                Percentage of labeled nodes                                                                             Percentage of labeled nodes                                                        Percentage of labeled nodes                                                            Percentage of labeled nodes




                                                                                                                                               Figure 3: Micro-F1 and Macro-F1 on Cora and Flickr.

                                                                                                                                                          Cora Dataset
                    0.35                                                                                                                                                                                                        0.96
                                                                       Cora                                                                    DVNE
                                                                       Facebook                                            0.16                DVNE_kl                                                                          0.94                                                                                   0.96




                                                                                               Percentage of AUC decline
                      0.3                                              Blogcatalog                                                             DeepWalk
                                                                                                                           0.14                                                                                                                                                                                        0.94
                                                                       Flickr                                                                  LINE                                                                             0.92




 Average variance
                                                                                                                           0.12                SDNE                                                                                                                                                                    0.92



                                                                                                                                                                                                                   AUC scores                                                                             AUC scores
                    0.25                                                                                                                       G2G_oh                                                                            0.9
                                                                                                                            0.1                                                                                                                                                                                         0.9
                                                                                                                                                                                                                                0.88
                                                                                                                           0.08                                                                                                                                                                                        0.88
                      0.2
                                                                                                                                                                                                                                0.86
                                                                                                                           0.06                                                                                                                                                                                        0.86
                                                                                                                                                                                                                                0.84
                    0.15                                                                                                   0.04                                                                                                                                                                                        0.84

                                                                                                                           0.02                                                                                                 0.82                                                                                   0.82
                      0.1                                                                                                      0                                                                                                 0.8                                                                                    0.8
                         0             0.5       1       1.5      2      2.5         3                                          0                0.1      0.2       0.3      0.4         0.5                                        0        50     100   150      200      250         300                                0       0.2       0.4      0.6      0.8         1
                                                     Log (degree)                                                                                   Percentage of noise edges                                                                        Dimensionality L                                                                    Hyper−parameter alpha
                                                        10



(a) The average variance wrt the (b) The performance wrt the                                                                                                                                                      (a) The AUC scores wrt the di- (b) The AUC scores wrt the hyper-
degree of nodes.                 noise edges.                                                                                                                                                                     mensionality L                 parameter α

                                        Figure 4: Results of embedding uncertainty.                                                                                                                                                               Figure 6: Results of parameter sensitivity.



                                                                                                                                                                                                                    The visualization results are shown in Figure 5, we compare
                                                                                                                                                                                                                 the DVNE with DVNE_kl. For DVNE_kl, in the center part the
                                                                                                                                                                                                                 nodes of different classes are mixed with each other. Obviously,
                                                                                                                                                                                                                 the visualization of DVNE looks better because points of the same
                                                                                                                                                                                                                 color form segmented classes, and the boundaries of each class are
                                                                                                                                                                                                                 clearer. It demonstrate the superiority of our method that using the
                                                                                                                                                                                                                 W2 distance as the similarity measure in the visualization task.

                                                (a) DVNE                                                                                            (b) DVNE_kl
                                                                                                                                                                                                                 5.7                      Parameter Sensitivity
                                       Figure 5: Visualization of network embedding.                                                                                                                             In this section, we investigate the parameter sensitivity. More spe-
                                                                                                                                                                                                                 cifically, we evaluate how different numbers of the embedding
                                                                                                                                                                                                                 dimensions and different values of hyper-parameter α can affect
                                                                                                                                                                                                                 the results. We report AUC scores on the dataset of Cora.
variance terms and mean vectors in the model, which means the                                                                                                                                                       First, we show how the dimension of the embedding vectors
variance terms can not well capture the uncertainties of the nodes.                                                                                                                                              affects the performance in Figure 6a. We can see that initially the
Through the sampling process proposed by our method, DVNE is                                                                                                                                                     performance raises when the number of dimension increases. Ho-
more natural to learn the variance terms that contains the uncer-                                                                                                                                                wever, when the number of dimensions continuously increases,
tainties of the nodes. Therefore, DVNE and DVNE_kl achieve better                                                                                                                                                the performance tends to be stable. This is because most of the
performance than G2G_oh. SDNE and LINE treat each edge equally,                                                                                                                                                  useful information is already encoded into the embeddings. Addi-
thus the similarities between nodes in latent space are easily be                                                                                                                                                tional dimensions consume more computing resources, but have
destroyed by the noise edges.                                                                                                                                                                                    less effect on performance. Overall, it is important to determine
                                                                                                                                                                                                                 the appropriate number of dimensions for the latent space. When
5.6                                    Visualization                                                                                                                                                             the number of dimensions is not too small (L ≥ 32), DVNE is not
Visualization is another important application for network embed-                                                                                                                                                sensitive to this parameter.
ding. Therefore, we visualize the learned embeddings of the Cora                                                                                                                                                    Then, we fix the number of dimensions to 128. The Figure 6b
network. Following [39], we first learn a lower-dimensional L = 128                                                                                                                                              shows how the value of α affects the performance . The parameter of
embedding for each node and then map those representations in                                                                                                                                                    α balances the weight of the first-order proximity and second-order
2-dimension space by t-SNE [32]. For nodes with different labels,                                                                                                                                                proximity between nodes. When α = 0, our method only preserves
we use different colors. Thus, a good visualization result is that the                                                                                                                                           the first-order proximity between nodes and the performance is
points of the same color are near from each other.                                                                                                                                                               worse than that of other parameter settings. It demonstrates that
both first-order and second-order proximity are essential for net-                              and machine intelligence 39, 9 (2017), 1853–1865.
work embedding methods to capture the network structure. When                              [13] Peng Cui, Xiao Wang, Jian Pei, and Wenwu Zhu. 2017. A Survey on Network
                                                                                                Embedding. arXiv preprint arXiv:1711.08752 (2017).
α > 0, we observe that DVNE is also not very sensitive to the choice                       [14] Marco Cuturi and Arnaud Doucet. 2014. Fast computation of Wasserstein bary-
of this hyper-parameter.                                                                        centers. In International Conference on Machine Learning. 685–693.
                                                                                           [15] Fernando De Goes, Katherine Breeden, Victor Ostromoukhov, and Mathieu Des-
                                                                                                brun. 2012. Blue noise through optimal transport. ACM Transactions on Graphics
6     CONCLUSIONS                                                                               (TOG) 31, 6 (2012), 171.
                                                                                           [16] Carl Doersch. 2016. Tutorial on variational autoencoders. arXiv preprint
In this paper, we propose a method to learn the Gaussian embedding                              arXiv:1606.05908 (2016).
by the deep variational model, namely DVNE, which can model                                [17] Ludovic Dos Santos, Benjamin Piwowarski, and Patrick Gallinari. 2016. Multila-
the uncertainties of nodes. It is the first unsupervised method that                            bel classification on heterogeneous graphs with gaussian embeddings. In Joint
                                                                                                European Conference on Machine Learning and Knowledge Discovery in Databases.
represents nodes in networks as Gaussian distributions in Was-                                  Springer, 606–622.
serstein space. The method preserves first-order proximity and                             [18] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern recognition letters
second-order proximity between nodes to capture the local and                                   27, 8 (2006), 861–874.
                                                                                           [19] Bent Fuglede and Flemming Topsoe. 2004. Jensen-Shannon divergence and
global network structure. Moreover, DVNE uses the 2-Wasserstein                                 Hilbert space embedding. In Information Theory, 2004. ISIT 2004. Proceedings.
distance as the similarity measure to better preserve the transitivity                          International Symposium on. IEEE, 31.
                                                                                           [20] Clark R Givens, Rae Michael Shortt, et al. 1984. A class of Wasserstein metrics
in the network with the linear time complexity. The empirical study                             for probability distributions. The Michigan Mathematical Journal 31, 2 (1984),
demonstrates the superiority of our proposed method. Our future                                 231–240.
direction is to find a good Gaussian prior for each node to better                         [21] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training
                                                                                                deep feedforward neural networks. In Proceedings of the Thirteenth International
capture the network structure and model the uncertainties of nodes.                             Conference on Artificial Intelligence and Statistics. 249–256.
                                                                                           [22] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for
7     ACKNOWLEDGEMENTS                                                                          networks. In Proceedings of the 22nd ACM SIGKDD International Conference on
                                                                                                Knowledge Discovery and Data Mining. ACM, 855–864.
This work was supported in part by National Program on Key Basic                           [23] Steve R Gunn et al. 1998. Support vector machines for classification and regression.
Research Project (No. 2015CB352300), National Natural Science                                   ISIS technical report 14, 1 (1998), 5–16.
                                                                                           [24] Shizhu He, Kang Liu, Guoliang Ji, and Jun Zhao. 2015. Learning to represent
Foundation of China (No. 61772304, No. 61521002, No. 61531006,                                  knowledge graphs with gaussian embedding. In Proceedings of the 24th ACM
No. 61702296), National Natural Science Foundation of China Major                               International on Conference on Information and Knowledge Management. ACM,
                                                                                                623–632.
Project (No.U1611461), the research fund of Tsinghua-Tencent Joint                         [25] Paul W Holland and Samuel Leinhardt. 1972. Holland and Leinhardt reply: some
Laboratory for Internet Innovation Technology, and the Young Elite                              evidence on the transitivity of positive interpersonal sentiment.
Scientist Sponsorship Program by CAST. All opinions, findings,                             [26] Zhipeng Huang and Nikos Mamoulis. 2017. Heterogeneous Information Network
                                                                                                Embedding for Meta Path based Proximity. arXiv preprint arXiv:1701.05291 (2017).
conclusions and recommendations in this paper are those of the                             [27] Thomas N Kipf and Max Welling. 2016. Variational graph auto-encoders. arXiv
authors and do not necessarily reflect the views of the funding                                 preprint arXiv:1611.07308 (2016).
agencies.                                                                                  [28] Solomon Kullback and Richard A Leibler. 1951. On information and sufficiency.
                                                                                                The annals of mathematical statistics 22, 1 (1951), 79–86.
                                                                                           [29] Yann LeCun, Sumit Chopra, Raia Hadsell, M Ranzato, and F Huang. 2006. A
REFERENCES                                                                                      tutorial on energy-based learning. Predicting structured data 1, 0 (2006).
 [1] Luigi Ambrosio, Nicola Gigli, and Giuseppe Savaré. 2008. Gradient flows: in metric    [30] Jure Leskovec and Julian J Mcauley. 2012. Learning to discover social circles in
     spaces and in the space of probability measures. Springer Science &amp; Business           ego networks. In Advances in neural information processing systems. 539–547.
     Media.                                                                                [31] David Liben-Nowell and Jon Kleinberg. 2007. The link-prediction problem for
 [2] Aleksandar Bojchevski and Stephan Günnemann. 2017. Deep gaussian embedding                 social networks. journal of the Association for Information Science and Technology
     of attributed graphs: Unsupervised inductive learning via ranking. arXiv preprint          58, 7 (2007), 1019–1031.
     arXiv:1707.03815 (2017).                                                              [32] Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE.
 [3] A. Bojchevski and S. Günnemann. 2017. Deep Gaussian Embedding of Graphs:                   Journal of Machine Learning Research 9, Nov (2008), 2579–2605.
     Unsupervised Inductive Learning via Ranking. ArXiv e-prints (July 2017).              [33] Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore.
     arXiv:stat.ML/1707.03815                                                                   2000. Automating the construction of internet portals with machine learning.
 [4] Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. 2015. Sliced          Information Retrieval 3, 2 (2000), 127–163.
     and radon wasserstein barycenters of measures. Journal of Mathematical Imaging        [34] Vinod Nair and Geoffrey E Hinton. 2010. Rectified linear units improve re-
     and Vision 51, 1 (2015), 22–45.                                                            stricted boltzmann machines. In Proceedings of the 27th international conference
 [5] Nicolas Bonneel, Michiel Van De Panne, Sylvain Paris, and Wolfgang Heidrich.               on machine learning (ICML-10). 807–814.
     2011. Displacement interpolation using Lagrangian mass transport. In ACM              [35] Feiping Nie, Wei Zhu, and Xuelong Li. 2017. Unsupervised Large Graph Embed-
     Transactions on Graphics (TOG), Vol. 30. ACM, 158.                                         ding.. In AAAI. 2422–2428.
 [6] Victor Bryant. 1985. Metric spaces: iteration and application. Cambridge University   [36] Mingdong Ou, Peng Cui, Jian Pei, Ziwei Zhang, and Wenwu Zhu. 2016. Asym-
     Press.                                                                                     metric transitivity preserving graph embedding. In Proc. of ACM SIGKDD. 1105–
 [7] Chen Chen and Hanghang Tong. 2015. Fast eigen-functions tracking on dynamic                1114.
     graphs. In Proceedings of the 2015 SIAM International Conference on Data Mining.      [37] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning
     SIAM, 559–567.                                                                             of social representations. In Proceedings of the 20th ACM SIGKDD international
 [8] Siheng Chen, Sufeng Niu, Leman Akoglu, Jelena Kovačević, and Christos Falout-              conference on Knowledge discovery and data mining. ACM, 701–710.
     sos. 2017. Fast, Warped Graph Embedding: Unifying Framework and One-Click             [38] Zafarani Reza and Liu Huan. 2009. Social Computing Data Repository. (2009).
     Algorithm. arXiv preprint arXiv:1702.05764 (2017).                                    [39] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
 [9] Philippe Clement and Wolfgang Desch. 2008. An elementary proof of the triangle             2015. Line: Large-scale information network embedding. In Proceedings of the
     inequality for the Wasserstein metric. Proc. Amer. Math. Soc. 136, 1 (2008), 333–          24th International Conference on World Wide Web. ACM, 1067–1077.
     339.                                                                                  [40] Tijmen Tieleman and Geoffrey Hinton. 2012. Lecture 6.5-rmsprop: Divide the
[10] Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. 2015. Fast and                gradient by a running average of its recent magnitude. COURSERA: Neural
     accurate deep network learning by exponential linear units (elus). arXiv preprint          networks for machine learning 4, 2 (2012), 26–31.
     arXiv:1511.07289 (2015).                                                              [41] Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. 2017.
[11] Nicolas Courty, Rémi Flamary, and Mélanie Ducoffe. 2017. Learning Wasserstein              Wasserstein Auto-Encoders. arXiv preprint arXiv:1711.01558 (2017).
     Embeddings. arXiv preprint arXiv:1710.07457 (2017).                                   [42] Ke Tu, Peng Cui, Xiao Wang, Fei Wang, and Wenwu Zhu. 2017. Structural Deep
[12] Nicolas Courty, Rémi Flamary, Devis Tuia, and Alain Rakotomamonjy. 2017.                   Embedding for Hyper-Networks. arXiv preprint arXiv:1711.10146 (2017).
     Optimal transport for domain adaptation. IEEE transactions on pattern analysis        [43] Luke Vilnis and Andrew McCallum. 2014. Word representations via gaussian
                                                                                                embedding. arXiv preprint arXiv:1412.6623 (2014).
[44] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural deep network em-        [46] Xiao Wang, Peng Cui, Jing Wang, Jian Pei, Wenwu Zhu, and Shiqiang Yang. 2017.
     bedding. In Proceedings of the 22nd ACM SIGKDD international conference on          Community Preserving Network Embedding. (2017).
     Knowledge discovery and data mining. ACM, 1225–1234.                           [47] Chengxi Zang, Peng Cui, Christos Faloutsos, and Wenwu Zhu. 2017. Long Short
[45] Huahua Wang and Arindam Banerjee. 2014. Bregman alternating direction               Memory Process: Modeling Growth Dynamics of Microscopic Social Connectivity.
     method of multipliers. In Advances in Neural Information Processing Systems.        In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge
     2816–2824.                                                                          Discovery and Data Mining. ACM, 565–574.

