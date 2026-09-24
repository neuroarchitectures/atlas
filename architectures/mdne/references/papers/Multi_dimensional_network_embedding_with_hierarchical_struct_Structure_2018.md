# Multi dimensional network embedding with hierarchical struct Structure 2018

> Source: `Multi_dimensional_network_embedding_with_hierarchical_struct_Structure_2018.pdf`

---

                            Multi-Dimensional Network Embedding with
                                      Hierarchical Structure
                        Yao Ma∗†                                                Zhaochun Ren†                                             Ziheng Jiang
       Data Science and Engineering Lab                                         Data Science Lab                                        Data Science Lab
          Michigan State University                                                 JD.com                                                   JD.com
            mayao4@msu.edu.com                                                renzhaochun@jd.com                                      jiangziheng@jd.com

                                                     Jiliang Tang                                            Dawei Yin‡
                                      Data Science and Engineering Lab                                    Data Science Lab
                                         Michigan State University                                            JD.com
                                           tangjili@msu.edu.com                                          yindawei@acm.org

ABSTRACT                                                                                      KEYWORDS
Information networks are ubiquitous in many applications. A pop-                              Network Embedding, Multi-dimensional Networks, Hierarchical
ular way to facilitate the information in a network is to embed                               Structure
the network structure into low-dimension spaces where each node                               ACM Reference Format:
is represented as a vector. The learned representations have been                             Yao Ma, Zhaochun Ren, Ziheng Jiang, Jiliang Tang, and Dawei Yin. 2018.
proven to advance various network analysis tasks such as link                                 Multi-Dimensional Network Embedding with Hierarchical Structure. In
prediction and node classification. The majority of existing em-                              WSDM 2018: WSDM 2018: The Eleventh ACM International Conference on Web
bedding algorithms are designed for the networks with one type                                Search and Data Mining , February 5–9, 2018, Marina Del Rey, CA, USA. ACM,
of nodes and one dimension of relations among nodes. However,                                 New York, NY, USA, 9 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
many networks in the real-world complex systems have multiple
types of nodes and multiple dimensions of relations. For example,                             1     INTRODUCTION
an e-commerce network can have users and items, and items can                                 We are living in a connected world where information networks
be viewed or purchased by users, corresponding to two dimensions                              are ubiquitous. Some examples of information networks include
of relations. In addition, some types of nodes can present hier-                              social networks, publication networks, the World Wide Web and
archical structure. For example, authors in publication networks                              e-commerce networks. Network embedding, aiming to learn vector
are associated to affiliations; and items in e-commerce networks                              representations for nodes, has attracted increasing attention in
belong to categories. Most of existing methods cannot be natu-                                recent years. Many advanced network embedding algorithms have
rally applicable to these networks. In this paper, we aim to learn                            emerged such as Deepwalk [26], LINE [29] and Metapath2vec [11],
representations for networks with multiple dimensions and hierar-                             which have been proven to help numerous network analysis tasks
chical structure. In particular, we provide an approach to capture                            such as link prediction [18], node classification [5][33] and network
independent information from each dimension and dependent infor-                              visualization [21][28].
mation across dimensions and propose a framework MINES, which                                    Most of existing embedding algorithms are designed for net-
performs Multi-dImension Network Embedding with hierarchical                                  works with one type of nodes and one dimension of relations
Structure. Experimental results on a network from a real-world                                among nodes. However, many networks in real-world complex
e-commerce website demonstrate the effectiveness of the proposed                              systems contain multiple dimensions of relations among nodes. For
framework.                                                                                    example, in social networking sites such as Facebook, two users
                                                                                              could be connected by friend relations, and via various social in-
                                                                                              teractions; in the transportation network [3], two cities could be
∗ Work performed during an internship at Data Science Lab, JD.com.
† These two authors contributed equally.
                                                                                              connected via various means of transportations such as train, high-
‡ Corresponding author                                                                        way and airplane; while in e-commerce networks, items can be
                                                                                              viewed and purchased by users, corresponding to two dimensions
                                                                                              of relations between users and items. In addition, some of the nodes
                                                                                              can present certain hierarchical structure. For example, in publi-
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed         cation networks, authors are associated to affiliations; while in
for profit or commercial advantage and that copies bear this notice and the full citation     e-commerce networks, items are organized by categories. A typical
on the first page. Copyrights for components of this work owned by others than ACM
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
                                                                                              example of multi-dimensional networks with hierarchical struc-
to post on servers or to redistribute to lists, requires prior specific permission and/or a   ture is illustrated in Figure 1 where there are two types of nodes
fee. Request permissions from permissions@acm.org.                                            U = {u 1 , u 2 , u 3 , u 4 } and T = {t 1 , t 2 , t 3 , t 4 }, and C = {c 1 , c 2 , c 3 }
WSDM 2018, February 5–9, 2018, Marina Del Rey, CA, USA
© 2018 Association for Computing Machinery.
                                                                                              is the set of parent nodes. The relations of nodes in U, nodes in
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM. . . $15.00                                                  T and nodes between U and T are two-dimensional; while each
https://doi.org/10.1145/nnnnnnn.nnnnnnn                                                       node in U is associated to one parent in C. The vast majority of
                                 t1     t2        t3   t4             2.1    Multi-dimensional Network Analysis
                                                                      Network analysis has been extensively studied for many years
        Dimension 2                                                   [34][35][6][14][2][7]. Multidimensional networks, which are quite
                                                                      ubiquitous in the real-world applications, have attracted increasing
                       u1    u2        u3    u4
                                                                      attention. In [3], the authors introduced a few examples of real-
                                                                      world multidimensional networks, and they also defined measures
                                                                      such as degree, neighbors for the multidimensional networks. More
                                 t1     t2        t3   t4             measures for the multidimensional networks are introduced in [22].
                                                                      The classic link prediction problem has been extended to multidi-
        Dimension 1                                                   mensional networks with the new problem “what is the probability
                                                                      that a new link between two nodes will form in a specific dimen-
                      u1    u2          u3        u4
                                                                      sion?” [27]. Multidimensional versions of the Common Neighbors
                                                                      and Adamic-Adar have been introduced to solve this problem [27].
                                                                      In [4], the authors studied the community discovery problem in the
                                                                      multidimensional network setting. In [16], the authors investigated
                       c1         c2         c3                       friendship maintenance and prediction in multidimensional social
                                                                      networks.
Figure 1: An illustrative example of a multi-dimensional net-
work with hierarchical structure                                      2.2    Network Embedding
                                                                      Networks can be represented by adjacency matrices; however, these
                                                                      representations are too sparse and high-dimensional. Many classic
existing embedding algorithms cannot be naturally applicable to       methods such as Laplacian eigenmap [1] and IsoMap [30] have
multi-dimensional networks with hierarchical structure as shown       been proposed to learn low-dimensional representations. These
in Figure 1.                                                          methods work fine on small size networks but cannot be scaled
   In this paper, we aim to learn representations of nodes in net-    to very large networks. Inspired by word2vec [23][25], DeepWalk
works with multiple dimensions and hierarchical structure. In par-    and LINE are proposed recently which can be applied to very large
ticular, we study approaches (1) to mathematically capture multi-     scale networks. DeepWalk regards the nodes in the network as the
dimensional information and hierarchical structure; and (2) to in-    “words” of an artificial language and uses random walk to generate
corporate such information simultaneously for embedding. Conse-       the “sentences” for this language. Then, following the procedure of
quently, we propose a framework MINES for Multi-dImensional           word2vec, the representations for the nodes can be learned. LINE
Network Embedding with hierarchical Structure. Our major con-         tries to capture both the first order and second order proximity in
tributions are summarized as follows:                                 the representations. node2vec [13] extends DeepWalk by adding
    • Providing a principled approach to model multi-dimensional      parameters to introduce the biased random walk. These network
      networks, which can capture independent information from        embedding methods have shown effectiveness in various tasks
      each dimension and dependent information across dimen-          on many homogeneous networks. In [11], the authors extended
      sions;                                                          DeepWalk method to heterogeneous networks by introducing meta-
    • Proposing a framework MINES, which incorporates multi-          path based random walks. In [9], the authors also facilitate the
      dimensional relations and hierarchical structure into a co-     meta-path to learn the heterogeneous network embedding, while
      herent model for node representation learning; and              they focus on the selection of the meta-path. In [8], the authors
    • Validating the effectiveness of the proposed framework in a     facilitate deep architectures to perform heterogeneous network
      real-world e-commerce network.                                  embedding. In [32], a signed network embedding algorithm SiNE is
                                                                      proposed based on the notion that a user should be closer to their
   The rest of this paper is arranged as follows. In Section 2, we    “friend” than their “enemy”. In [20], the authors try to preserve both
review some works that are related to our problem. The problem        local and global information in the network for network embedding.
of embedding networks with multiple dimensions and hierarchi-         There are also works on attributed network embedding [17][31].
cal structure to vector space is formally defined in Section 3. The   Two recent surveys [10][12] give a comprehensive overview of
approach to model networks with multiple dimensions and hierar-       network embedding algorithms. However, most of the existing
chical structure and the proposed framework with an optimization      methods cannot naturally be applicable to networks with multiple
method are introduced in Section 4. The experiments on a real-        dimensions of relations and hierarchical structure. In this paper,
world e-commerce network with discussions are presented in Sec-       we aim to model the multi-dimensional relations and hierarchical
tion 5. The conclusion and future work are presented in Section 6.    structure and propose a framework to embed these networks to
                                                                      vector space.

2   RELATED WORK                                                      3     PROBLEM STATEMENT
Our work is related to multi-dimensional network analysis and         In the multi-dimensional networks, we have different types of nodes
network embedding. In this section, we briefly review them.           and multiple dimensions of relations. Assume that there are K types
                                           (i)   (i)       (i)
of nodes in total and let Vi = {v 1 , v 2 , . . . , v N } be the set of the        and completely ignores information across dimensions. Hence the
                                                       i
i-th type with Ni nodes. Let V denote the set of all the nodes                     learned representations for different dimensions are not related.
       K                                                                           However, dimensions are inherently related since they share the
V =
       Ð
         Vi . Some types of nodes in the network might present                     same set of nodes. Thus, in this subsection, we study how to model
        i=1
hierarchical structure. In other words, these nodes are associated                 multi-dimensional relations.
with categories. For simplicity, we assume all the types of nodes                     Intuitively, each dimension should have its independent infor-
have hierarchical structures with a depth of 2, and we name the                    mation individually; while all dimensions should share dependent
parent nodes as categories in this case. Note that though in this                  information across dimensions. Therefore, the learned represen-
work, we focus on the hierarchical structures with a depth of 2,                   tations for each dimension should not only preserve independent
it is straightforward to apply the proposed framework for deeper                   information from the dimension but also keep dependent informa-
                                        (i) (i)             (i)                    tion across dimensions. To achieve this goal, for a given dimension
hierarchical structures. We set Ci = {c 1 , c 2 , . . . , c M } as the set
                                                                       i
                                                                                   d, the representation ud for a node contains two components – (1)
of Mi categories for the i-th type of nodes, and set T(i) ∈ RNi ×Mi
                                                                                   one component u for the information shared across dimensions;
as the matrix that describes the category information, for the i-th
                                                                                   and (2) one component ed specific to the dimension d. With these
type of nodes.
                                                                                   two components, we can rewrite ud as:
   Two nodes could be connected via multiple relations, and we
regard each type of relations as a dimension. Thus, nodes from the                                           ud = f (u, ed ),                     (1)
same type or different types can be connected in the same dimen-                   where f is a function to combine the shared component u and the
sion. These connections can be described by adjacency matrices                     specific component ed . The shared component u not only captures
(for the same type nodes) and the interaction matrices (for different              dependent information across dimensions but also helps the learned
types of nodes). Let Ad ∈ RNi ×Ni be the adjacency matrix of the i-
                      (i)
                                                                                   representations of all dimensions to be related. The specific com-
                          (i,j)                                                    ponent ed preserves independent information from the dimension
th node type and Hd ∈ RNi ×Nj be the interaction matrix between
the i-th and j-th types of node in the d-th dimension. We target                   d.
to learn representations for each node in each dimension of the
                            (i)   (i)           (i)
network. Let Ud (i) = {ud 1 , ud 2 , . . . , ud N } denote the represen-
                                                                                   4.2   Capturing Hierarchical Structure
                                                  i
tations of i-th type of nodes in the dimension d (d = 1, . . . D) where            For these nodes which have the hierarchical structure, we also need
D is the number of dimensions.                                                     to model its category information (or parents). The category infor-
   With the aforementioned notations and definitions, our problem                  mation is actually shared by all the dimensions, hence it should be
can be formally defined as follow:                                                 indicated in the shared component of representations. For the nodes
                                                                                   in the same category, they should also share the similar character-
Given                                                                              istics. Therefore, to model the hierarchical structure, the shared
                                                        (i)      (i)         (i)   component of the node representation should further contain two
      • K different sets of nodes, i.e., Vi = {v 1 , v 2 , . . . , v N } (i =
                                                                      i            components – (1) one component cu indicates category informa-
        1, . . . , K);                                                             tion which is shared by all the nodes in the category, and (2) one
                                                                      (i)
      • multi-dimensional relations among the nodes, i.e., Ad (i =                 component su is specific to the node. With the defined components,
                                  (i, j)
        1, . . . , K) and Hd (i, j = 1, . . . , K; i , j; d = 1, . . . , D);       we can further rewrite u in Eq. (1) as:
      • the hierarchical structure information, i.e., Ti (i = 1, . . . , K).                                 u = д(cu , su )                      (2)
We aim to learn a set of representations for all nodes, i.e.,                      where д is the function to combine the category shared information
                (i)         (i)    (i)            (i)                              cu and the node specific information su . Note that Eq. (2) can be
           Ud         = {ud 1 , ud 2 , . . . , ud N }   (i = 1, . . . , K)
                                                    i                              easily extended to deeper hierarchical structure by further decom-
in each dimension d (d = 1, . . . D).                                              posing the category shared information cu .

4     THE MULTI-DIMENSIONAL EMBEDDING                                              4.3   The Proposed Framework
      FRAMEWORK WITH HIERARCHICAL                                                  With approaches to capture multi-dimensional relations and hier-
      STRUCTURE                                                                    archical structure, in this subsection, we introduce the embedding
                                                                                   framework MINES.
In this section, we will first introduce how to model multi-dimensional
                                                                                      To learn the embeddings for the nodes in each dimension, we
relations and hierarchical structure; and then discuss the proposed
                                                                                   follow the idea of skip-gram model [24], which is an effective and
framework with an optimization method.
                                                                                   efficient way to learn distributed representations of words. The
                                                                                   skip-gram model predicts surrounding context given a center word,
4.1     Capturing Multi-Dimensional Relations                                      which can be formulated as follows:
In a multi-dimensional network, all dimensions share the same
set of nodes, while having their own network structures in each                                              p(N (wc )|wc ),                      (3)
dimension. A straightforward way to learn representations for each                 where N (wc ) is the set of words that surround word wc .
dimension is to perform the network embedding for each dimen-                         Similarly, we can use the skip-gram to model the network in a
sion, separately. This strategy treats each dimension independently                given dimension d. For a node v, we define all nodes connected to
                                                                                                                                                        (i)
v as the “context” of v, which is formally defined as:                                                   method, we replace each log pd (v j |v) with
                                                 K
                                                            (i)
                                                 Ø
                                    Nd (v) =              Nd (v);                                  (4)                                                        Ne
                                                                                                               Od (v, v j ) = log σ (uTd ud j ) +                   log σ (−uTd ud (n) );
                                                                                                                           (i)                    (i)                               (i)
                                                                                                                                                              Õ
                                                 i=1                                                                                                                                              (10)
                                                                                                                                                              n=1
           (i)
where Nd is the set of i-th type of nodes that are connected to
node v in the dimension d. Note that the “context” of v consists of                                      where σ (x) = 1/(1 + exp(x)) is the sigmoid function, and Ne is
different types of nodes, and we treat them differently, which will                                      the number of negative samples. The negative samples are ran-
be further explained later.                                                                                                                                                      (i)
                                                                                                         domly sampled from some noise distribution. For log pd (v j |v),
   Then, given a center node v, we need to predict its “context” as:                                     the negative samples are sampled from node set Vi according to
                                                                                                          (i)              3/4
                       K                                  K                                              Pi(v) (v (i) ) ∼ d (i ) , as proposed in [25], where dv (i ) is the in-degree
                                      (i)                                                (i)                                v
                       Ö                                  Ö               Ö
  pd (Nd (v)|v) =               p(Nd (v)|v) =                                       pd (v j |v); (5)
                                                                                                         of v (i) corresponding to the i(v)-th type of nodes and i(v) indicates
                        i=1                                i=1 v (i ) ∈N (i ) (v)                                                                                (i)
                                                                  j         d                            the type of node v. Note again, for i-th type of node v j , we sample
          (i)                                                                                            the negative samples from the i-th type of nodes set Vi instead of
where p(v j |v) can be modeled using a softmax function as:
                                                                                                         the whole nodes set V.
                                                                                                             We adopt mini-batch Stochastic Gradient Descent (SGD) to opti-
                                                  exp(uTd ud j )
                                                                      (i)
                              (i)                                                                        mize the problem. In each step, a mini-batch of edges of the same
                     pd (v j |v) =                                    .                            (6)
                                                     exp(uTd ud (i) )
                                                 Í
                                                                                                         type are sampled according to their weights. Here, by “same” type
                                             v (i ) ∈Vi                                                  of edge, we mean, these edges have same types of nodes for source
In (6), the softmax function is over the i-th type nodes Vi instead                                      and target nodes respectively and also the relations between them
of the whole nodes set V.                                                                                is in the same dimension. For each sampled edge, the source node
                                                                                                                                                               (i)
   To learn the representations for the dimension d, we model this                                       is treated as v and the target node is treated as v j in (10). The
problem as a maximum likelihood problem. In other words, we                                                                       (i)       (i)
                                                                                                         derivatives for v, v j and v (n) are
need to maximize the probability that Nd (v) is the “context” of
node v for all the nodes v ∈ V. Hence, we need to maximize:
                                                                                                                     (i)                                            Ne
                            Ö                                                                             ∂Od (v, v j )
                      Pd =       pd (Nd (v)|v).                                                                             = (1 − σ (udT ud j ))ud j −                  (1 − σ (−udT ud (n) ))ud (n) ;
                                                                                                                                             (i)         (i)                                (i)    (i)
                                                                                                                                                                    Õ
                                                                 (7)
                                        v ∈V                                                                   ∂ud                                                  n=1
                                                                                                                     (i)
   With all D dimensions, we need to jointly maximize the following                                       ∂Od (v, v j )
                                                                                                                        = (1 − σ (udT ud j ))ud ;
                                                                                                                                         (i)
term:                                                                                                                                                                                             (11)
                                                                                                                 (i)
                                                                                                            ∂ud j
                                                 D
                                                 Ö
                                            P=            Pd .                                     (8)              (i)
                                                                                                          ∂Od (v, v j )
                                                                                                                        = −(1 − σ (−udT ud (n) ))ud , n = 1, . . . , N e.
                                                                                                                                             (i)
                                                 d =1
                                                                                                                (i)
                                                                                                            ∂ud (n)
   Instead of maximizing Eq. (8), we equivalently minimize its neg-
                                                      (i)
ative logarithm with respect to the representations Ud as:
                                                                                                            Next we discuss how to choose f and д functions. In fact, f is
                                                                                                         used to combine the dimension shared component u and dimension-
                 min            − log P                                                                  specific component cd . It can be a linear function, a non-linear
                 =1, . . ., D
              (i )
          {Ud }id=1, . . ., K                                                                            function (e.g., exponential functions) or even can be automatically
                                    D
                                    Õ                                                                    learned (e.g., neural networks). In this work, we choose a linear
      ⇔          min            −           log Pd                                                 (9)   function f . In other words, we define as – f (u, ed ) = u+cd . We also
                 =1, . . ., D
              (i )
          {Ud }id=1, . . ., K       d =1                                                                 use a similar function for д. We would like to leave the investigation
                                    Õ Õ K
                                    D                                                                    of other choices of f and д as one future direction. With choices of
                                                                                         (i)             f and д, the representations for nodes can be rewritten as:
                                        Õ                         Õ
      ⇔          min            −                                               log pd (v j |v).
            (i ) =1, . . ., D
          {Ud }id=1, . . ., K       d =1 v ∈V i=1 v (i ) ∈N (i ) (v)
                                                            j         d
                                                                                                                                 ud = cu + su + ed ;                                              (12)
4.4    An Optimization Method                                                                                                       (i)     (i)     (i)
                                                                                                                                 ud j = cu j + su j + ed j ;
                                                                                                                                                             (i)
                                                                                                                                                                                                  (13)
There are two challenges to address when optimizing Eq. (9). First,                                                                 (i)      (i)      (i)      (i)
the minimization of Eq. (9) is computationally expensive due to                                                                  ud (n) = cu (n) + su (n) + ed (n) .                              (14)
summation over the whole set of nodes Vi when calculating each
               (i)
term log pd (v j |v). Second, how to choose the functions of f in                                          We need to update cu , su , ed , cu j , su j , ed j , cu (n) , su (n) and ed (n) .
Eq. (1) and д in Eq. (2).                                                                                We update these representations using Gradient Decent (GD).
   To solve computational challenge, we adopt the negative sam-                                            To update the representations for v, we need to update its three
pling approach proposed in [25]. By using the negative sampling                                          components cu , su and cd according to (15).
                                                                                             Algorithm 1: Optimization procedure
                                                  (i)
                                         ∂Od (v, vh )                                         Input: N e, m, S, ρ, dim, E
                                                                                                           (i) d =1, ..., D
                     cu ← cu + ρ ·                              ;                             Output: {Ud }i=1,     ..., K
                                                ∂ud
                                                                                                               (i)    (i)          (i)
                                                        (i)                                 1 Initialize cu j , su j and ed j , as dim dimension vectors
                                         ∂Od (v, vh )
                     su ← su + ρ ·                              ;                    (15)       randomly, for d = 1, . . . , D, i = 1, . . . , K and j = 1, . . . , Ni ;
                                                ∂ud
                                                                                            2 s = 0;
                                                         (i)
                                         ∂Od (v, vh )                                       3 while s < S do
                     ed ← ed + ρ ·                              .
                                                ∂ud                                         4     Sample a set of m edges of the same type SE from E;
                                                                                                                     (i)
                                                                     (i)                    5        for e = (v, v j ) ∈ SE do
   Similarly, to update the representations for v j , we need to up-                                                                                     (i)
                                  (i)     (i)              (i)
                                                                                            6             Sample a set of N e negative samples {v (n) }n=1, ..., N e ;
date its three components cu j , su j and cd j according to (16).
                                                                                            7        Calculate the gradients according to (11);
                                                               (i)                          8        Update the corresponding vectors according to (15),
                     (i)      (i)
                                            ∂Od (v, v j )
                   cu j ← cu j + ρ ·                                 ;                                (16) and (17);
                                                         (i)
                                                  ∂ud j                                      9    end
                                                              (i)                           10    s ← s + m.
                     (i)      (i)
                                           ∂Od (v, v j )
                                                                                            11 end
                   su j ← su j + ρ ·                                 ;               (16)
                                                         (i)
                                                  ∂ud j                                     12
                                                                                                    (i)      (i)      (i)    (i)
                                                                                                 ud j = cu j + su j + ed j ; for d = 1, . . . , D, i = 1, . . . , K and
                                            ∂Od (v, v j )
                                                               (i)                                j = 1, . . . , Ni ;
                     (i)      (i)                                                                            d =1, ..., D
                   ed j ← ed j + ρ ·                                 .                                        (i)
                                                                                                 return {Ud }i=1,
                                                  ∂ud j
                                                         (i)                                13
                                                                                                                  ..., K .


                                                          (i)
Finally, to update the representations for v (n) , n = 1, . . . , N e, we                   5      EXPERIMENTS
                                                          (i)            (i)   (i)
need to update their three components cu (n) , su (n) and cd (n) ac-                        In this section, we present the experimental details to verify the
cording to (17) respectively.                                                               effectiveness of the proposed framework. We first introduce the
                                                                                            dataset we will use in the evaluation. Then, we describe the experi-
                                                  (i)
             (i)      (i)
                                    ∂Od (v, v j )                                           mental settings. Finally, we present the experimental results with
          cu (n) ← cu (n) + ρ ·                         , n = 1, . . . N e;                 discussions and study the key parameter in the proposed frame-
                                            (i)
                                        ∂ud (n)
                                                                                            work.
                                                  (i)
             (i)       (i)
                                  ∂Od (v, v j )
          su (n) ← su (n) + ρ ·                         , n = 1, . . . N e;          (17)   5.1      Dataset
                                            (i)
                                        ∂ud (n)                                             In our experiments, we sample data from JD.com, which is one of the
                                                  (i)                                       largest e-commerce companies. In our dataset, we have two types
             (i)       (i)
                                    ∂Od (v, v j )
          ed (n) ← ed (n) + ρ ·                         , n = 1, . . . N e.                 of nodes: users and items. The items have hierarchical structure and
                                            (i)
                                        ∂ud (n)                                             each item belongs to some predefined categories. Users can perform
                                                                                            various behaviors on items such as “view", “save", and “purchase".
   We summarize the optimization procedure in Algorithm 1. In                               In this work, we collect two behaviors, i.e., “view” and “purchase”,
the algorithm, the input includes the number of mini-batch size                             to construct two-dimensional relations between users and items.
m, the training size S, the dimension of representations dim, the                           In addition, we collect two other relations: one is the “view session”
number of negative samples N e, the learning rate ρ and the set                             of a user, while another is the “purchase basket” of the user.
of all the edges E in the network. In line 1, we initialize all the                            A view session is a sequence of items that are viewed by a user
components for all the representations. Then, we sample a set of                            within a period of time. It is intuitively to understand the items that
same type edges SE from E in line 4. In line 6, for each edge, we                           are viewed within a short period by the same user should be similar.
sample N e negative samples. We calculate the gradients and update                          To incorporate these relations into the network, we construct an
the components in lines 7 and 8, respectively. Finally, we combine                          item-item view network by connecting the items that are viewed w
the components to form the representations for each node in each                            items before or after a given item in a session with this item, where
dimension in line 12.                                                                       w is the window size. In this work, we set the window size to 5.
   To efficiently sample the edges and negatives samples, we adopt                          These edges are in the “view” dimension and they are weighted
the alias methods proposed in [15], which can generate a random                             where the weight is the co-occurrence frequency.
variable from a discrete distribution in constant time O(1). The                               A purchase basket is a set of items that are purchased by a user
optimization with negative sampling takes O(dim · (N e + 2) + N e)                          at the same time. Items that are purchased in the same basket are
time, where N e is the number of negative samples. hence, each step                         supposed to be related to each other. To incorporate these relations,
of MINES takes O(dim · N e) operations. If the training size is S, the                      we construct an item-item purchase network. In particular, we
overall time complexity of MINES is O(S · dim · N e).                                       connect two items if they are purchased in the same basket. These
                          # items             401,922                           • Element-wise multiplication Given two dim dimension
                          # users              17,806                             representations of two nodes, we multiply them element-
                       # categories            2,788                              wisely and get a new dim dimension vector as the represen-
                   # item-item (view)        6,402,586                            tation for this pair of nodes.
                   # user-user (view)       13,651,206                        For all the methods, we use both ways to form the representations
                   # user-item (view)         962,362                      for the pairs of nodes and report the results for each method.
                # item-item (purchase)       3,211,660                        After we form the representations for the pairs of the nodes in
                # user-user (purchase)       6,870,510                     the training set and the testing set, we train a binary classifier using
                # user-item (purchase)        485,656                      logistic regression on the training set and perform link prediction
             Table 1: The statistics of the network                        on the testing set. In this work, we will use Micro-F 1 , Macro-F 1 and
                                                                           AUC as the metric to evaluate the link prediction performance.

                                                                           5.3    Performance Comparison
edges are weighted where the weight is the frequency of the two            To evaluate the performance of our algorithm, we compare the
items presenting in the same basket. In the other way around,              performance of our algorithm with the following representative
users that have “viewed” or “purchased” the same item also shows           baselines:
similarity. In each dimension, we connected users that have “viewed”           • LINE [29]: As LINE can only work for one-dimensional net-
or “purchased” the same items.                                                    work, we apply LINE to the two dimensions separately and
    To sum up, in the constructed multidimensional e-commerce                     learn one set of representations for each dimension, respec-
network, we have two types of nodes, i.e., the users and the items,               tively. We treat categories as nodes, and add item-category
and the items have hierarchical structure. There are two dimensions,              edges into the networks for LINE.
i.e., the “view” dimension and the “purchase” dimension. We can                • DeepWalk [26]: We apply DeepWalk to the two dimensions
conclude that the constructed network has all the characteristics                 separately and learn two sets of representations. We treat
of networks we want to study in this work; hence it is suitable for               categories as nodes, and add item-category edges into the
us to use the dataset to evaluate the proposed framework. Some                    networks for DeepWalk. DeepWalk can only work for un-
statistics of the network are shown in Table 1.                                   weighted networks, hence, we convert our network to un-
                                                                                  weighted network by ignoring the weights.
5.2     Experimental Setting                                                   • Non-negative Matrix Factorization (NMF) [19]: We ap-
                                                                                  ply it to the user-item interaction matrix and use the factor-
Following the common way to assess network embedding algo-                        ized two matrices as the embeddings for the users and items.
rithms [13], we choose link prediction as the evaluation task. The                NMF is also applied to the two dimensions, separately.
intuition is that a better embedding algorithm should learn bet-               • Co-NMF: In Co-NMF, we perform a co-factorization on the
ter node representations, which will lead to better link prediction               multi-dimensional networks and learn unified user repre-
performance.                                                                      sentations for all dimensions. Basically Co-NMF assumes all
   In the link prediction task, a certain fraction of edges are removed,          dimensions share the same embeddings, which completely
and we would like to predict whether these “missing” edges exist.                 ignores independent information from each dimension.
   In our evaluation, we perform the link prediction task on the               • MINES(S): This is a variant of our framework MINES. In-
two dimensions, separately. For each dimension, we remove the                     stead of using all three components cu , su and ed , we only
user-item edges and use them as parts of the testing set. We set up               use the shared components cu and su to form the represen-
3 groups of experiments, where 10%, 30% and 50% of the user-item                  tation for a node v.
edges are removed, respectively. To form the training set, we first
                                                                             We summarize the experiments results for the “view” dimension
put all the remaining user-item edges into the training set, and then,
                                                                           and “purchase” dimension in Table 2, and Table 3, respectively. We
for each user-item edge in the training set, we randomly sample
                                                                           make the following observations from Table 2:
an item that is not connected to this user and use this user and
non-connected item pair as the negative sample in the training set.            • For all methods, using the Element-wise multiplication
We form the testing set in the same way.                                          is better than Element-wise addition, which is consistent
   After removing the edges, we use the remaining network to                      with the observation in [13].
learn the representations for all the nodes. Then, to perform the              • The performance of Co-NMF is worse than that of NMF,
link prediction task, the representations for the edges (or the user              which indicates that the independent information from each
item pairs) should be learned. We use two different ways to combine               dimension is very important to accurately predict links in
the representations of two nodes as the representation of the edge                that dimension.
(or user item pair) as used in [13].                                           • MINES shows better performance than MINES(S), which
                                                                                  further shows the importance of the dimension specific in-
      • Element-wise addition Given two dim dimension repre-                      formation.
        sentations of two nodes, we add them element-wisely and                • As we remove more percent of edges, the performance of
        get a new dim dimension vector as the representation for                  all methods decrease in all three measures when using the
        this pair of nodes.                                                       Element-wise multiplication.
                                                             Addition                   Multiplication
                                  % removed edges    10%       30%       50%      10%        30%         50%
                                     MINES           72.77    72.76      72.65    83.89    82.57      81.32
                                      LINE           71.00    70.83      70.72    79.42     78.26     76.84
                                    DeepWalk         69.39    69.03      68.87    76.45     76.10     74.94
                  Micro-F 1 (%)
                                      NMF            59.80    59.79      59.86    78.16     78.16     77.98
                                    Co-NMF           56.66    56.83      56.93    76.96     77.00     76.87
                                    MINES(S)         67.22    66.78      66.82    76.98     76.28     75.78
                                     MINES           72.74    72.72      72.58    83.75    82.35      80.98
                                      LINE           70.94    70.82      70.78    79.17     77.95     76.44
                                    DeepWalk         69.28    68.92      68.79    76.22     75.79     74.63
                  Macro-F 1 (%)
                                      NMF            59.77    59.77      59.86    78.08     78.07     77.90
                                    Co-NMF           56.66    56.83      56.93    76.84     76.87     76.74
                                    MINES(S)         67.21    66.77      68.81    76.96     76.25     75.77
                                     MINES          0.8037   0.8040     0.8036   0.9261    0.9180    0.9146
                                      LINE          0.7757   0.7757     0.7732   0.8879    0.8759    0.8636
                     AUC
                                    DeepWalk        0.7478   0.7434     0.7392   0.8522    0.8517    0.8351
                                      NMF           0.6516   0.6527     0.6529   0.8741    0.8739    0.8729
                                    Co-NMF          0.5912   0.5927     0.5933   0.8603    0.8605    0.8596
                                    MINES(S)        0.7315   0.7271     0.7278   0.8530    0.8456    0.8409
                       Table 2: Link Prediction Performance Comparison: The View Dimension


                                                             Addition                   Multiplication
                                  % removed edges    10%       30%       50%      10%        30%         50%
                                      MINES          78.48    77.94     77.62    90.93     89.74     88.57
                                       LINE          71.55     70.84     70.63    87.88     86.94     85.69
                                     DeepWalk        67.08     67.46     67.53    82.76     82.35     80.54
                  Micro-F 1 (%)
                                       NMF           68.20     68.11     68.18    83.43     83.65     83.44
                                     Co-NMF          56.28     56.36     56.31    76.98     76.83     76.75
                                     MINES(S)        70.36     70.77     70.54    83.63     82.97     82.20
                                      MINES         78.47     77.92     77.59    90.89     89.66     88.49
                                       LINE          71.48     70.83     70.51    87.85     86.87     85.58
                                     DeepWalk        67.03     67.38     67.52    82.70     82.27     80.41
                  Macro-F 1 (%)
                                       NMF           68.19     68.10     68.18    83.36     83.59     83.36
                                     Co-NMF          56.24     56.33     56.27    76.84     76.67     76.60
                                     MINES(S)        70.34     70.75     70.52    83.61     82.96     82.19
                                      MINES         0.8662    0.8644    0.8623   0.9762    0.9725    0.9690
                                       LINE         0.7892    0.7791    0.7759   0.9614    0.9518    0.9455
                                     DeepWalk       0.7109    0.7134    0.7128   0.9278    0.9256    0.9252
                      AUC
                                       NMF          0.7544    0.7528    0.7538   0.9346    0.9347    0.9342
                                     Co-NMF         0.5826    0.5834    0.5820   0.8592    0.8585    0.8568
                                     MINES(S)       0.7740    0.7775    0.7758   0.9146    0.9036    0.9101
                    Table 3: Link Prediction Performance Comparison: The Purchase Dimension



• The performance of DeepWalk is worse than LINE and NMF.               moved. The major reason is the proposed framework has
  This is mainly because DeepWalk can only work for un-                 two components to capture the multi-dimensional relations
  weighted networks and cannot take advantage of the edge               and the hierarchical structure.
  weights.
• The proposed framework MINES obtains the best perfor-
                                                                    We have similar observations for the “purchase” dimension –
  mance. For example, MINES boosts the performance 3% − 5%
                                                                 (1) using the Element-wise multiplication is better than using
  compared to the best baseline when 10% − 50% edges are
                                          Figure 2: Parameter Analysis: The View Dimension




                                        Figure 3: Parameter Analysis: The Purchase Dimension


Element-wise addition and (2) MINES outperforms all the base-           6   CONCLUSION
lines; for example, MINES obtains over 2% improvement in terms          In this paper, we propose an approach to model multi-dimensional
of all the measures compared to the best baseline.                      networks, which can capture independent information from each
                                                                        dimension and dependent information across dimensions. Based
                                                                        on this approach, we propose the MINES framework which can
5.4    Parameter Analysis                                               embed multi-dimensional network with hierarchical structure to
In this section, we analyze how the dimension of the learned rep-       low-dimensional vector spaces. We can learn a set of node repre-
resentations in our method affects the performance of the link          sentations for each dimension using this framework. The learned
prediction task. In particular, we set the dimension of the represen-   representations for each dimension will contain the hierarchical in-
tations to {16, 32, 64, 128} with the setting of 50% edges removed.     formation, the independent information from the specific dimension
The results are reported in in Figure 2 and Figure 3 for view and       and also dependent information across dimensions. We evaluate the
purchase dimensions, separately. Note that we ignore the results        effectiveness of our framework on a multi-dimensional e-commerce
with other settings since we can make similar observations.             network. The results of our experiments show the advancement of
   As shown in Figure 2 and Figure 3, in both view and purchase         our framework.
dimensions, the Micro-F 1 and Macro-F 1 first increase as the dimen-       In this work, we utilize linear functions to model the across di-
sion of the learned representations gets large, and then decrease.      mension information and the hierarchical structure information.
Both the Micro-F 1 and Macro-F 1 reach the maximum when the             In our future work, more complicated non-linear functions such
dimension of the representations is 32 in both view and purchase        as exponential functions or even the neural networks can be used.
dimension. AUC also increases first and then decrease as the dimen-     Meanwhile, as a limitation in our work, we only focus on hierarchi-
sion of the representations gets larger. However, the AUC score         cal structures with depth of 2 in this paper. As another direction in
reaches the maximum when the dimension of the representation is         our future work, we would like to investigate the proposed frame-
64.                                                                     work with deeper hierarchical structures. Real-world networks
   In summary, the performance first increases and then decreases       typically evolve such as addition of new nodes and links, and dele-
as the dimension of the representations gets larger. The dimension      tion of old nodes and links. Therefore, multi-dimensional network
of the representations affects the different measures differently.      embedding with dynamics should provide new insights in future.
ACKNOWLEDGEMENTS                                                                               and Social Media. ACM, 83–92.
                                                                                          [17] Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, and Huan Liu. 2017.
The authors wish to thank the anonymous reviewers for their help-                              Attributed Network Embedding for Learning in a Dynamic Environment. arXiv
ful comments. Yao Ma and Jiliang Tang are supported by the Na-                                 preprint arXiv:1706.01860 (2017).
                                                                                          [18] David Liben-Nowell and Jon M. Kleinberg. 2003. The link prediction problem for
tional Science Foundation (NSF) under grant number IIS-1714741                                 social networks. In Proceedings of the 2003 ACM CIKM International Conference on
and IIS-1715940.                                                                               Information and Knowledge Management, New Orleans, Louisiana, USA, November
                                                                                               2-8, 2003. ACM, 556–559.
                                                                                          [19] Chih-Jen Lin. 2007. Projected gradient methods for nonnegative matrix factor-
REFERENCES                                                                                     ization. Neural computation 19, 10 (2007), 2756–2779.
 [1] Mikhail Belkin and Partha Niyogi. 2002. Laplacian eigenmaps and spectral             [20] Yao Ma, Suhang Wang, ZhaoChun Ren, Dawei Yin, and Jiliang Tang. 2017. Pre-
     techniques for embedding and clustering. In Advances in neural information                serving Local and Global Information for Network Embedding. arXiv preprint
     processing systems. 585–591.                                                              arXiv:1710.07266 (2017).
 [2] Michael GH Bell, Yasunori Iida, et al. 1997. Transportation network analysis.        [21] Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE.
     (1997).                                                                                   Journal of Machine Learning Research 9, Nov (2008), 2579–2605.
 [3] Michele Berlingerio, Michele Coscia, Fosca Giannotti, Anna Monreale, and Dino        [22] MARGGF Magnani, Anna Monreale, Giulio Rossetti, and Fosca Giannotti. 2013.
     Pedreschi. 2013. Multidimensional networks: foundations of structural analysis.           On multidimensional network measures. In Italian conference on Sistemi Evoluti
     World Wide Web 16, 5-6 (2013), 567–593.                                                   per le Basi di Dati (SEBD).
 [4] Michele Berlingerio, Fabio Pinelli, and Francesco Calabrese. 2013. Abacus: fre-      [23] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
     quent pattern mining-based community discovery in multidimensional networks.              estimation of word representations in vector space. arXiv preprint arXiv:1301.3781
     Data Mining and Knowledge Discovery 27, 3 (2013), 294–320.                                (2013).
 [5] Smriti Bhagat, Graham Cormode, and S Muthukrishnan. 2011. Node classification        [24] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
     in social networks. In Social network data analytics. Springer, 115–148.                  Estimation of Word Representations in Vector Space. CoRR abs/1301.3781 (2013).
 [6] Ronald L Breiger, Scott A Boorman, and Phipps Arabie. 1975. An algorithm             [25] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.
     for clustering relational data with applications to social network analysis and           Distributed representations of words and phrases and their compositionality. In
     comparison with multidimensional scaling. Journal of mathematical psychology              Advances in neural information processing systems. 3111–3119.
     12, 3 (1975), 328–383.                                                               [26] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. DeepWalk: online learning
 [7] Carter T Butts. 2009. Revisiting the foundations of network analysis. science 325,        of social representations. In The 20th ACM SIGKDD International Conference on
     5939 (2009), 414–416.                                                                     Knowledge Discovery and Data Mining, KDD ’14, New York, NY, USA - August 24
 [8] Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C Aggarwal, and                     - 27, 2014, Sofus A. Macskassy, Claudia Perlich, Jure Leskovec, Wei Wang, and
     Thomas S Huang. 2015. Heterogeneous network embedding via deep archi-                     Rayid Ghani (Eds.). ACM, 701–710.
     tectures. In Proceedings of the 21th ACM SIGKDD International Conference on          [27] Giulio Rossetti, Michele Berlingerio, and Fosca Giannotti. 2011. Scalable link
     Knowledge Discovery and Data Mining. ACM, 119–128.                                        prediction on multidimensional networks. In Data Mining Workshops (ICDMW),
 [9] Ting Chen and Yizhou Sun. 2017. Task-Guided and Path-Augmented Hetero-                    2011 IEEE 11th International Conference on. IEEE, 979–986.
     geneous Network Embedding for Author Identification. In Proceedings of the           [28] Jian Tang, Jingzhou Liu, Ming Zhang, and Qiaozhu Mei. 2016. Visualization
     Tenth ACM International Conference on Web Search and Data Mining, WSDM                    Large-scale and High-dimensional Data. CoRR abs/1602.00370 (2016).
     2017, Cambridge, United Kingdom, February 6-10, 2017, Maarten de Rijke, Milad        [29] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
     Shokouhi, Andrew Tomkins, and Min Zhang (Eds.). ACM, 295–304.                             2015. LINE: Large-scale Information Network Embedding. In Proceedings of the
[10] Peng Cui, Xiao Wang, Jian Pei, and Wenwu Zhu. 2017. A Survey on Network                   24th International Conference on World Wide Web, WWW 2015, Florence, Italy,
     Embedding. arXiv preprint arXiv:1711.08752 (2017).                                        May 18-22, 2015, Aldo Gangemi, Stefano Leonardi, and Alessandro Panconesi
[11] Yuxiao Dong, Nitesh V Chawla, and Ananthram Swami. 2017. metapath2vec:                    (Eds.). ACM, 1067–1077.
     Scalable Representation Learning for Heterogeneous Networks. (2017).                 [30] Joshua B Tenenbaum, Vin De Silva, and John C Langford. 2000. A global geometric
[12] Palash Goyal and Emilio Ferrara. 2017. Graph Embedding Techniques, Applica-               framework for nonlinear dimensionality reduction. science 290, 5500 (2000), 2319–
     tions, and Performance: A Survey. arXiv preprint arXiv:1705.02801 (2017).                 2323.
[13] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for       [31] Suhang Wang, Charu C. Aggarwal, Jiliang Tang, and Huan Liu. [n. d.]. Attributed
     networks. In Proceedings of the 22nd ACM SIGKDD international conference on               Signed Network Embedding. In Proceedings of CIKM.
     Knowledge discovery and data mining. ACM, 855–864.                                   [32] Suhang Wang, Jiliang Tang, Charu C. Aggarwal, Yi Chang, and Huan Liu. 2017.
[14] Gueorgi Kossinets and Duncan J Watts. 2006. Empirical analysis of an evolving             Signed Network Embedding in Social Media. In Proceedings of SDM. 327–335.
     social network. science 311, 5757 (2006), 88–90.                                     [33] Suhang Wang, Jiliang Tang, Charu C. Aggarwal, and Huan Liu. 2016. Linked
[15] Richard A Kronmal and Arthur V Peterson Jr. 1979. On the alias method for gen-            Document Embedding for Classification. In Proceedings of CIKM. 115–124.
     erating random variables from a discrete distribution. The American Statistician     [34] Stanley Wasserman and Katherine Faust. 1994. Social network analysis: Methods
     33, 4 (1979), 214–218.                                                                    and applications. Vol. 8. Cambridge university press.
[16] Ka-Wei Roy Lee and Ee-Peng Lim. 2016. Friendship maintenance and prediction in       [35] Barry Wellman. 1983. Network analysis: Some basic principles. Sociological
     multiple social networks. In Proceedings of the 27th ACM Conference on Hypertext          theory (1983), 155–200.

