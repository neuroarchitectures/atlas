# Representation Learning for Scale free Networks Hu Wu Zhuang 2018

> Source: `Representation_Learning_for_Scale_free_Networks_Hu_Wu_Zhuang_2018.pdf`

---

                                                                     Representation Learning for Scale-free Networks

                                                                  Rui Feng∗ , Yang Yang†∗ , Wenjie Hu, Fei Wu, and Yueting Zhuang
                                                                      College of Computer Science and Technology, Zhejiang University, China
                                                                                    †
                                                                                      Corresponding author: yangya@zju.edu.cn




arXiv:1711.10755v1 [cs.SI] 29 Nov 2017
                                                                     Abstract                                       100                             100                            100
                                                                                                                 10 1                           10 1                           10 1
                                           Network embedding aims to learn the low-dimensional repre-
                                                                                                                 10 2                           10 2                           10 2
                                           sentations of vertexes in a network, while structure and inher-    PDF                             PDF                            PDF
                                           ent properties of the network is preserved. Existing network          10 3
                                                                                                                                                10 3                           10 3
                                           embedding works primarily focus on preserving the micro-              10 4
                                                                                                                                                10 4 0                         10 4 0
                                           scopic structure, such as the first- and second-order proxim-              100   101   102   103        10      101   102   103        10     101   102   103
                                                                                                                             degree                         degree                        degree
                                           ity of vertexes, while the macroscopic scale-free property is
                                           largely ignored. Scale-free property depicts the fact that ver-           (a) Original                         (b) LE                   (c) DP-Walker
                                           tex degrees follow a heavy-tailed distribution (i.e., only a few
                                           vertexes have high degrees) and is a critical property of real-    Figure 1: Scale-free property of real-world networks. (a)
                                           world networks, such as social networks. In this paper, we
                                           study the problem of learning representations for scale-free
                                                                                                              is the degree distribution of an academic network. (b) and
                                           networks. We first theoretically analyze the difficulty of em-     (c) are respectively the degree distribution of of the net-
                                           bedding and reconstructing a scale-free network in the Eu-         work reconstructed based on vertex representations learned
                                           clidean space, by converting our problem to the sphere pack-       by Laplacian Eigenmap (LE) and our proposed method (DP-
                                           ing problem. Then, we propose the “degree penalty” principle       Walker).
                                           for designing scale-free property preserving network embed-
                                           ding algorithm: punishing the proximity between high-degree
                                           vertexes. We introduce two implementations of our principle        tent space, and represent each vertex by a vector in that
                                           by utilizing the spectral techniques and a skip-gram model         space. For example, a number of recent works apply ad-
                                           respectively. Extensive experiments on six datasets show that
                                           our algorithms are able to not only reconstruct heavy-tailed
                                                                                                              vances in natural language processing (NLP), most notably
                                           distributed degree distribution, but also outperform state-of-     models known as word2vec (Mikolov et al. 2013), to net-
                                           the-art embedding models in various network mining tasks,          work embedding and propose word2vec-based algorithms,
                                           such as vertex classification and link prediction.                 such as DeepWalk (Perozzi, Al-Rfou, and Skiena 2014) and
                                                                                                              node2vec (Grover and Leskovec 2016). Besides, researchers
                                                                                                              also consider network embedding as part of dimension-
                                                              1     Introduction                              ality reduction techniques. For instance, Laplacian Eigen-
                                         Network analysis has attracted considerable research efforts         map (LE) (Belkin and Niyogi 2003) aims to learn the low-
                                         in many areas of artificial intelligence, as networks are able       dimensional representation to expand the manifold where
                                         to encode rich and complex data, such as human relation-             the data lie.
                                         ships and interactions. One major challenge of network anal-            Essentially, these methods mainly focus on preserving mi-
                                         ysis is how to represent network properly so that the network        croscopic structure of network, like pairwise relationship
                                         structure can be preserved. The most straightforward way             between vertexes. Nevertheless, scale-free property, one of
                                         is to represent the network by its adjacency matrix. How-            the most fundamental macroscopic properties of networks,
                                         ever, it suffers from the data sparsity. Other traditional net-      is largely ignored.
                                         work representation relies on handcrafted network feature               Scale-free property depicts that the vertex degrees fol-
                                         design (e.g., clustering coefficient), which is inflexible, non-     low a power-law distribution, which is a common knowl-
                                         scalable, and requires hard human labor.                             edge for many real-world networks. We take an academic
                                            In recent years, network representation learning, also            network as the example, where each edge indicates if a ver-
                                         known as network embedding, has been proposed and                    tex (researcher) has cited at least one publication of an-
                                         aroused considerable research interest. It aims to automat-          other vertex (researcher). Figure 1(a) demonstrates the de-
                                         ically project a given network into a low-dimensional la-            gree distribution of this network. The linearity on log-log
                                            ∗
                                              Equal contribution. Ordering determined by dice rolling.        scale suggests a power-law distribution: the probability de-
                                         Copyright c 2018, Association for the Advancement of Artificial      creases when the vertex degree grows, with a long tail
                                         Intelligence (www.aaai.org). All rights reserved.                    tending to zero (Faloutsos, Faloutsos, and Faloutsos 1999;
Adamic and Huberman 1999). In other words, there are only            significant improvement on six datasets and three tasks
a few vertexes of high degrees. The majority of vertexes con-        compared to several state-of-the-art baselines.
nected to a high-degree vertex is, however, of low degree,
and not likely connected to each other.                                           2    Theoretical Analysis
   Moreover, compared with the microscopic structure, the          In this section, we try to study why most network embedding
macroscopic scale-free property imposes a higher level con-        algorithms will overestimate higher degrees theoretically,
straint on the vertex representations: only a few vertexes can     and analyze if there exists a solution of scale-free property
be close to many others in the learned latent space. Incorpo-      preserving network embedding in the Euclidean space. This
rating scale-free property in network embedding can reflect        section also provides intuitions of our approach in Section 3.
and preserve the sparsity of real-world networks and, in turn,
provide effective information to make the vertex representa-       2.1   Preliminaries
tions more discriminative.
   In this paper, we study the problem of learning scale-          Notations. We consider an undirected network G = (V, E),
free property preserving network embedding. As the repre-          where V = {v1 , · · · , vn } is the vertex set containing n ver-
sentation of a network, vertex embedding vectors are ex-           texes and E is the edge set. Each eij ∈ E indicates an undi-
pected to well reconstruct the network. Most existing al-          rected edge between vi and vj . We define the adjacency ma-
gorithms learn network embedding in the Euclidean space.           trix of G as A = [Aij ] ∈ Rn×n , where Aij = 1 if eij ∈ E
However, we find that most traditional network embedding           and AijP= 0 otherwise. Let D be a diagonal matrix where
algorithms will overestimate the number of higher degrees.         Dii = j Aij is the degree of vi .
Figure 1(b) gives an example, where the representation is          Network embedding. In this paper, we focus on the repre-
learned by Laplacian Eigenmap. We analyze and try to un-           sentation learning for undirected networks. Given an undi-
derstand this theoretically, and study the feasibility of recov-   rected graph G = (V, E), the problem of graph embedding
ering power-law distributed vertex degree in the Euclidean         aims to represent each vertex vi ∈ V into a low-dimensional
space, by converting our problem to the Sphere Packing             space Rk , i.e., learning a function f : V 7→ Un×k , where U
problem. Through our analysis we find that theoretically,          is the embedding matrix, k  n and network structures can
moderately increasing the dimension of embedding vectors           be preserved in U.
can help to preserve the scale-free property. See details in       Network reconstruction. As the representation of a net-
Section 2.                                                         work, the learned embedding matrix is expected to well re-
   Inspired by our theoretical analysis, we then propose the       construct the network. In particular, one can reconstruct the
degree penalty principle for designing scale-free property         network edges based on distances between vertexes in the
preserving network embedding algorithms in Section 3: pun-         latent space Rk . For example, the probability of an edge ex-
ishing the proximity between vertexes with high degrees.           isting between vi and vj can be defined as
We further introduce two implementations of our approach
                                                                                                      1
based on the spectral techniques (Belkin and Niyogi 2003)                              pi,j =                                  (1)
and the skip-gram model (Mikolov et al. 2013). As Fig-                                          1 + ekui −uj k
ure 1(c) suggests, our approach can better preserve the scale-     where the Euclidean distance between embedding vectors ui
free property of networks. In particular, the Kolmogorov-          and uj of the vertex vi and vj , in respective, is considered.
Smirnov (aka. K-S statistic) distance between the obtained         In practice, a threshold ε ∈ [0, 1] is chosen and an edge
degree distribution and its theoretical power-law distribution     eij will be created if pi,j ≥ ε. We call the above method
is 0.09, while the value for the degree distribution obtained      as ε-NN, which is geometrically informative and a common
by LE is 0.2 (the smaller the better).                             method used in many existing work (Shaw and Jebara 2009;
   To verify the effectiveness of our approach, we con-            Belkin and Niyogi 2003; Alanis-Lobato, Mier, and Andrade-
duct experiments on both synthetic data and five real-world        Navarro 2016).
datasets in Section 4. The experimental results show that our
approach is able to not only preserve the scale-free property      2.2   Reconstructing Scale-free Networks
of networks, but also outperform several state-of-the-art em-      Given a network, we call it as a scale-free network, when
bedding algorithms in different network analysis tasks.            its vertex degrees follow a power-law distribution. In other
   We summarize our contribution of this paper as follows:         words, there are only a few vertexes of high degrees. The
                                                                   majority of vertexes connected to a high-degree vertex is,
• We analyze the difficulty and feasibility of reconstructing
                                                                   however, of low degree, and not likely connected to each
  a scale-free network based on learned vertex representa-
                                                                   other. Formally, the probability density function of vertex
  tions in the Euclidean space, by converting our problem
                                                                   degree Dii has the following form:
  to the Sphere Packing problem.
• We propose the degree penalty principle and two imple-                     pDii (d) = Cd−α , α > 1, d > dmin > 0             (2)
  mentations to preserve the scale-free property of networks
                                                                   where α is the exponent parameter and C is the normaliza-
  and improve the effectiveness of vertex representations.
                                                                   tion term. In practice, the above power-law form only ap-
• We validate our proposed principle by conducting exten-          plies for vertexes with degrees greater than a certain mini-
  sive experiments and find that our approach achieves a           mum value dmin (Clauset, Shalizi, and Newman 2009a).
                                                                                               (failed)
                                 embedding
                                                                                            reconstruction
                                                                                     1/2
                                              1                            3/2


                                                          Sphere Packing




Figure 2: An illustration of an ego-network centered by a hub vertex, a potential embedding solution, which is equivalent to a
sphere packing solution, and leads to a failed reconstruction, where higher degrees are overestimated.


   However, it is difficult to reconstruct a scale-free network            packing density in Rk as ∆k , which means no packing of
in the Euclidean space by ε-NN. As we see in Figure 1(b),                  spheres in Rk achieves a packing density larger than ∆k .
the higher degrees will be overestimated. We aim to explain                   As Theorem 1 suggests, we aim to find a packing solution
this theoretically.                                                        with large density so that more points in a closed sphere can
Intuition. While reconstructing the network by ε-NN, we                    keep their distance larger than 1. However, in general cases,
select a certain ε, and for a vertex vi with embedding vec-                finding the optimal packing density remains an open prob-
tor ui , we regard all points that fall in the closed ball of              lem. Still, we are able to derive the upper and lower bounds
radius ε centered at ui , denoted by B(ui , ε), as ones having             of ∆k for sufficiently large k.
edges with vi . When vi is a high-degree vertex, there will be
                                                                           Theorem 2 (Upper and lower bounds for ∆k ).
many points in B(ui , ε). We expect these points are far away
from each other, keeping more vertexes with low degree and                                            1
                                                                                         −1 ≤ lim log2 ∆k ≤ −0.599
thus in turn keep the power-law distribution. However, in-                                      k→∞ k
tuitively, as these points are placed in the same closed ball              Specifically, we have
B(ui , ε), it will be more likely that their distances from each
other are less than ε. As a result, there will be many edges                                      ∆k ≥ 2−k , k ≥ 1                   (3)
created among those points, which violates the assumption                  And the following inequality holds for sufficiently large k:
of the scale-free property.
   Following the above idea, we introduce a theorem below,                                         ∆k ≤ 2−0.599k                     (4)
which is discussed in Rk , and without loss of generality, we              Eq. 4 is one of the best upper bound when k ≥ 115 (Cohn
set ε as 1.                                                                and Zhao 2014). The proof of the above theorem can be
Theorem 1 (Sphere Packing). There are m points in B(0, 1)                  found in the work of Kabatiansky and Levenshtein.
whose distances from each other are larger than or equal to                Theorem 3. Suppose x is in Rk , ε > 0, and there can be at
1, if and only if, there exists m disjoint spheres of radius 1/2           most Mk points in B(x, ε) whose distances from each other
in B(0, 3/2).                                                              are larger than ε, then

Proof. The center of any sphere of radius 1/2 in B(0, 3/2)
                                                                                     $  %
                                                                                         k
                                                                                       3
                                                                                            ≤ Mk ≤ 3k 2−0.599k
                                                                                                              
falls in B(0, 1) and the distance between any two centers of                                                                         (5)
two different spheres of radius 1/2 is larger than or equal to                         2
1.                                                                         where b·c means taking the integer part. The upper bound
                                                                           holds for sufficiently large
Remark. Theorem 1 converts our problem of reconstruct-
ing a scale-free network to the Sphere Packing Problem,                    Proof. By Theorem 1 we only need to estimate the num-
which seeks to find the optimal packing of spheres in high                 ber of disjoint spheres of radius 12 ε that can be fitted in
dimensional spaces (Cohn and Elkies 2003; Vance 2011;                      B(x, 32 ε). The volume of a k-dimensional ball of radius R
Venkatesh 2012). Figure 2 gives an example, where a net-                                 π  k/2
                                                                           is given by Γ(k/2+1) Rk . Plugging in the radius, the vol-
work centered by a high-degree vertex is embedded into a
two-dimensional space. The embedding result corresponds                    umes of B(x, 2 ε) and a sphere of radius 21 ε are respectively
                                                                                        3
                                                                                     k/2                           k/2
to an equivalent Sphere Packing solution, which fails to                            π
                                                                           V1 = Γ(k/2+1)  ( 23 ε)k and V2 = Γ(k/2+1)
                                                                                                               π
                                                                                                                     ( 12 ε)k . Since the
place enough disjoint spheres and leads to a failed network                optimal packing density is given by ∆k , we have
reconstruction (many nonexistent edges are created). For-                                              
mally, we define the packing density as follows:                                                     V1
                                                                                                          = 3k ∆ k
                                                                                                                  
                                                                                        M k = ∆k
Definition 1 (Packing Density). The packing density is the                                           V2
fraction of the space filled by the spheres making up the                  Combing with Eq. 3 and Eq. 4, we obtain the inequality as
packing. For convenience, we define the optimal sphere                     desired.
Discussion. The lower bound of Mk in Eq. 5 suggests the            Dii is the degree of vi . Thus W is proportional to C0 and is
feasibility to reconstruct scale-free network by ε-NN in the       inversely proportional to vertex degrees.
Euclidean space, when k, the dimension of embedding vec-           Objective. Our goal is to learn vertex representations U,
tor, is sufficiently large. For instance, when k = 100, we         where ui ∈ Rk is the ith row of U and represents the em-
have Mk > 4.06 × 1017 , which is enough to keep scale-free         bedding vector of vertex vi , and minimize
property holds for most real-world networks.                                            X
                                                                                             kui − uj k2 Wij                 (9)
                   3    Our Approach                                                      i,j
General idea. Inspired by our theoretical analysis, we pro-        under the constraint
pose a principle of degree penalty for designing scale-free
property preserving embedding algorithms: while preserv-                                        UT DU = I                        (10)
ing first- and second-order proximity, the proximity between
vertexes that have high degrees shall be punished. We give         Eq. 10 is provided such that the embedding vectors will not
the general idea behind this principle below.                      collapse onto a subspace of dimension less than k (Belkin
   Scale-free property is featured by the ubiquitous exis-         and Niyogi 2003).
tence of “big hubs” which attract most edges in the net-           Optimization. In order to minimize Eq. 9, we utilize graph
work. Most of existing network embedding algorithms, ex-           Laplacian, which is an analogy to the Laplacian operator in
plicitly or implicitly, attempt to keep these big hubs close       multivariate calculus. Formally, we define graph Laplacian,
to their neighbors by preserving first-order and/or second-        L, as follows:
order proximity (Belkin and Niyogi 2003; Tang et al. 2015;
Perozzi, Al-Rfou, and Skiena 2014). However, connecting to                                      L,D−W                            (11)
big hubs does not imply proximity as strong as connecting          Observe that
to vertexes with mediocre degrees. Taking social network
as an example, where a famous celebrity may receive a lot                    X
                                                                                     Wij kui − uj k2 = trace (UT LU)             (12)
of followers. However, the celebrity may not be familiar or
                                                                               i,j
similar to her followers. Besides, two followers of the same
celebrity may not know each other at all and can be totally        The desired U minimizing the objective function (9) is ob-
dissimilar. As a comparison, a mediocre user is more likely        tained by putting
to known and to be similar to her followers.
   From another perspective, a high-degree vertex vi is more                              U = [t1 , . . . , tk ]                 (13)
likely to hurt the scale-free property, as placing more disjoint
                                                                   where ti is an eigenvector of L. In practice, we use
spheres in a closed ball is more difficult (Section 2).
   The degree penalty principle can be implemented by vari-        normalized form of W and L, (i.e., D−1/2 WD1/2 and
ous methods. In this paper, we introduce two proposed mod-         I − D−1/2 WD1/2 ).
els based on our principle, implemented by spectral tech-
niques and skip-gram models respectively.                          3.2   Model II: DP-Walker
                                                                   Our second method, Degree Penalty based Random Walk
3.1   Model I: DP-Spectral                                         (DP-Walker), utilizes a skip-gram model, word2vec.
Our first model, Degree Penalty based Spectral Embedding              We start with a brief description of the word2vec model
(DP-Spectral), mainly utilizes graph spectral techniques.          and its applications on network embedding tasks. Given a
Given a network G = (V, E) and its adjacency matrix A,             text corpus, Mikolov et al. proposed word2vec to learn the
we define a matrix C to indicate the common neighbors of           representation of each word by facilitating the prediction of
any two vertexes in G:                                             its context. Inspired by it, some network embedding algo-
                                                                   rithms like DeepWalk (Perozzi, Al-Rfou, and Skiena 2014)
                 C , AT A − diag (AT A)                     (6)    and node2vec (Grover and Leskovec 2016) define a vertex’s
                                                                   “context” by co-occurred vertexes in random walk generated
where Cij is the number of common neighbors of vi and vj .         paths.
C can be also regarded as a measurement of second-order               Specifically, the Random Walk rooted at vertex vi is a
proximity, and can be easily extended to further consider          stochastic process Vvki , k = 1, . . . , m, where Vvk+1   is a ver-
first-order proximity by                                                                                                i
                                                                   tex sampled from the neighbors of Vvki and m is the path
                        C0 , C + A                          (7)    length. Traditional methods regard P (Vvk+1   i
                                                                                                                    |Vvki ) as a uni-
                                                                   form distribution where each neighbor of Vvki has the equal
As we aim to model the influence of vertex degrees in our
                                                                   chance to be sampled.
model, we further extend C0 to consider degree penalty as
                                                                      However, as our proposed Degree Penalty principle sug-
                                                                   gests, a neighbor vi of a vertex vj with high degree may not
                   W , (D−β )T C0 D−β                       (8)    be similar to vj . In other words, vj shall have less chance
where β ∈ R is the model parameter used to indicate the            to be sampled as one of vi ’s context. Thus, we define the
strength of degree penalty, and D is a diagonal matrix where       probability of the random walk jumping from vi to vj as
                                                                                 Synthetic   Facebook   Twitter   Coauthor   Citation   Mobile
                                    C0ij                                  |V |    10000        4039      81306      5242      48521     198959
                   Pr(vj |vi ) ∝                         (14)             |E|
                                 (Dii Djj )β                                      399580      88234     1768149    28980     357235     1151003

where C0 can be found in Eq. 7 and β is the model parame-           Table 1: Statistics of datasets. |V | indicates the number of ver-
ter. According to Eq. 14, we find that vj will have a greater       texes and |E| denotes the number of edges.
chance to be sampled when it has more common neighbors
with vi and has a lower degree. After obtaining random walk
generated paths, we enable skip-gram to learn effective ver-        • DeepWalk (Perozzi, Al-Rfou, and Skiena 2014): This rep-
tex representations for G by predicting each vertex’s context.        resents skip-gram model based network embedding algo-
This results in an optimization problem                               rithms. It first generates random walks on the network,
                                                                      and defines the context of a vertex by its co-occurred ver-
       arg min − log Pr ({vi−w , . . . , vi+w } vi |ui )     (15)     texes in paths. Then, it learns vertex representations by
          U                                                           predicting each vertex’s context. Specifically, we perform
where 2 × w is the path length we consider. Specifically,             10 random walks starting from each vertex, and each ran-
for each random walk Vvi , we feed it to skip-gram algo-              dom walk will have a length of 40.
rithm (Mikolov et al. 2013) to update the vertex repre-             • DP-Spectral: This is a spectral technique based imple-
sentations. For implementation, we use Hierarchical Soft-             mentation of our degree penalty principle.
max (Morin and Bengio 2005; Mnih and Hinton 2009) to                • DP-Walker: This is another implementation of our ap-
estimate the concerned probability distribution.                      proach. It is based on a skip-gram model.
                     4    Experiments                               Unless otherwise specified, the embedding dimension for
                                                                    our experiments is 200.
4.1    Experiment Setup                                             Tasks. We first utilize different algorithms to learn vertex
Datasets. We use four datasets, whose statistics are summa-         representations for a certain dataset, then apply the learned
rized in Table 1 for the evaluations.                               embeddings in three different tasks:
• Synthetic: We generate a synthetic dataset by the Pref-           • Network reconstruction: this task aims to validate if an
  erential Attachment model (Vazquez 2003), which de-                 algorithm is able to preserve the scale-free property of
  scribes the generation of scale-free networks.                      networks. We evaluate the performance of different algo-
• Facebook (Leskovec and Mcauley 2012): This dataset is               rithms by the correlation coefficients between the recon-
  a subnetwork of Facebook1 , where vertexes indicate users           structed degrees and the degrees in the given network.
  of Facebook, and edges denote friendship between users.           • Link prediction: given two vertexes vi and vj , we feed
• Twitter (Leskovec and Mcauley 2012): This dataset is a              their embedding vectors, ui and uj , to a linear regression
  subnetwork of Twitter2 , where vertexes indicate users of           classifier and determine whether there exists an edge be-
  Twitter, and edges denote following relationships.                  tween vi and vj . Specifically, we use ui − uj as the fea-
                                                                      ture, and randomly select about 1% pairs of vertexes for
• Coauthor (Leskovec, Kleinberg, and Faloutsos 2007):
                                                                      training and evaluation.
  This network covers scientific collaborations between au-
  thors. Vertexes are authors. Vertexes are authors. An undi-       • Vertex classification: on this task, we consider the vertex
  rected edge exists between two authors if they have coau-           labels. For instance, in Citation, each vertex has a label to
  thored at least one paper.                                          indicate the researcher’s research area. Specifically, given
                                                                      a vertex vi , we define its feature vector as ui , and train a
• Citation (Tang et al. 2008): Similar to Coauthor, this is
                                                                      linear regression classifier to determine its label.
  also an academic network, where vertexes are authors.
  Edges indicate citations instead of coauthor relationship.        4.2      Network Reconstruction
• Mobile: This is a mobile network provided by PPDai3 .             Comparison results. In this task, after obtaining vertex rep-
  Vertexes are PPDai registered users. An edge between two          resentations, we use the ε-NN algorithm introduced in Sec-
  users indicates that one of the users has called the other.       tion 2 to reconstruct the network. We then evaluate the per-
  Overall, it consists of over one million calling logs.            formance of different methods on reconstructing power-law
Baseline methods. We compare the following four network             distributed vertex degrees by considering three different cor-
embedding algorithms in our experiments:                            relation coefficients of the reconstructed degrees and origi-
• Laplacian Eigenmap (LE) (Belkin and Niyogi 2003): This            nal degrees: Pearson’s r (Shao 2007), Spearman’s ρ (Spear-
  represents spectral-based network embedding algorithms.           man 1904), and Kendall’s τ (Kendall 1938). All of these
  It aims to learn the low-dimensional representation to ex-        statistics are used to evaluate some relationship between
  pand the manifold where the data lie.                             paired samples. Pearson’s r is used to detect linear relation-
                                                                    ships, while Spearman’s ρ and Kendall’s τ are capable of
   1
     http://facebook.com                                            finding monotonic relationships.
   2
     http://twitter.com                                                To select ε, for each algorithm, we roll over all values
   3
     The largest unsecured micro-credit loan platform in China.     of ε ranging from 0.01 to 1 with step 0.01, and choose the
                       1.0                                                 1.0                                                 1.0                                                  1.0




   Performance (Pearson)                               Performance (Pearson)                               Performance (Pearson)                                Performance (Pearson)
                       0.8                                                 0.8                                                 0.8                                                  0.8
                       0.6                                                 0.6                                                 0.6                                                  0.6
                       0.4                                                 0.4                                                 0.4                                                  0.4
                       0.2               DP-Walker                         0.2               DP-Walker                         0.2                DP-Walker                         0.2                DP-Walker
                                         DP-Spectral                                         DP-Spectral                                          DP-Spectral                                          DP-Spectral
                       0.00    50 100 150 200 250                          0.00    50 100 150 200 250                          0.0.5 1 1.5 2 2.5 3 3.5 4 4.5                        0.0.5 1 1.5 2 2.5 3 3.5 4 4.5
                              Embedding dimension                                 Embedding dimension                                            β                                                  β
                              (a) Synthetic.                                       (b) Facebook.                                         (c) Synthetic.                                    (d) Facebook.

Figure 3: Model parameter analysis. (a) and (b) demonstrate the sensitivity of the embedding dimension k in Synthetic and
Facebook dataset respectively. (c) and (d) present the sensitivity of the degree penalty parameter β. We omit the results on other
datasets due to space limitation.

 Dataset                       Method (ε)           Pearson                       Spearman   Kendall                               network, for each embedding algorithm, we validate its ef-
                               LE (0.55)             0.14                           0.054     0.039
                               DeepWalk (0.91)       0.47                           -0.22     -0.18
                                                                                                                                   fectiveness to preserve the scale-free property, by fitting
 Synthetic                                                                                                                         the reconstructed degree distribution (e.g., Figure 1(b) and
                               DP-Spectral (0.52)    0.92                            0.79     0.63
                               DP-Walker (0.95)      0.94                            0.63     0.52                                 (c)) to a theoretical power-law distribution (Alstott, Bull-
                               LE (0.52)             0.48                            0.18     0.12                                 more, and Plenz 2014). We then calculate the Kolmogorov-
                               DeepWalk (0.81)       0.73                            0.65     0.49                                 Smirnov distance between these two distributions and find
 Facebook
                               DP-Spectral (0.52)    0.87                            0.67     0.51                                 that the proposed methods can always obtain better results
                               DP-Walker (0.84)      0.75                            0.73     0.57
                               LE (0.81)             0.17                            0.19     0.17                                 compared to baselines (0.115 vs 0.225 averagely).
                               DeepWalk (0.51)       0.08                            0.21     0.26                                 Parameter analysis. We further study the sensitivity
 Twitter
                               DP-Spectral (0.93)    0.50                            0.34     0.27                                 of model parameters: embedding dimension and degree
                               DP-Walker (0.087)     0.40                            0.33     0.27                                 penalty weight β. We only present the results on Synthetic
                               LE (0.50)             0.32                            0.04     0.03                                 and Facebook and omit others due to space limitation. Fig-
                               DeepWalk (0.92)       0.66                            0.31     0.25
 Coauthor
                               DP-Spectral (0.51)    0.64                            0.69     0.55
                                                                                                                                   ure 3(a) and 3(b) shows the Pearson correlation coefficients
                               DP-Walk (0.93)        0.75                            0.44     0.35                                 resulted by our algorithm with different embedding dimen-
                               LE (0.99)             0.11                           -0.27     -0.20                                sions. When the embedding dimension grows, the perfor-
 Citation
                               DeepWalk (0.97)       0.51                            0.28     0.20                                 mance increases significantly. The figures suggest that when
                               DP-Spectral (0.50)    0.45                            0.72     0.54                                 embedding a network into Euclidean space, there is a dimen-
                               DP-Walker (0.98)      0.62                            0.65     0.50                                 sion which is sufficient for preserving scale-free property,
                               LE (0.51)             0.10                            0.05     0.04
                               DeepWalk (0.71)       0.77                            0.22     0.20                                 and further increase of the embedding dimension has lim-
 Mobile                                                                                                                            ited effect. Correlation does not change drastically for DP-
                               DP-Spectral (0.50)    0.40                            0.68     0.60
                               DP-Walker (0.78)      0.93                            0.22     0.20                                 Walker as the dimension increases, as the figure suggests.
                                                                                                                                   The figures also show that DP-Walker is able to obtain fairly
Table 2: Performance of different methods on scale-free                                                                            high Pearson correlation even in a lower dimension and it
property reconstruction. For each method, ε (indicated af-                                                                         requires an embedding dimension lower than DP-Spectral
ter the corresponding method) is chosen so that Pearson is                                                                         does for a satisfactory performance.
maximized.                                                                                                                            We also study how β influences the performance. Fig-
                                                                                                                                   ure 3(c) and 3(d) shows that DP-Spectral is more sensitive
                                                                                                                                   to the choice of β. This is largely due to the fact that in DP-
value which maximizes the Pearson’s correlation coefficient                                                                        Spectral, the influence of β is manifested in the objective
between the original and recovered degrees. Our selection of                                                                       function (Eq. 9), which imposes a stronger restriction than
Pearson’s r as evaluation metric is because of the scale-free                                                                      its counterpart in DP-Walker, which is embodied in the sam-
property of degree distribution.                                                                                                   pling process. Figure 3(c) and 3(d) shows that the optimal
   From Table 2, we see that our algorithms outperform the                                                                         choice of β varies from graph to graph. It makes sense, since
baselines significantly. Especially, Pearson’s r of DP-Walker                                                                      β can be viewed as a punishment on the degrees, and the in-
achieves at least 44.5% improvement, and Spearman’s ρ of                                                                           fluence of β is therefore supposedly related to the topology
DP-Spectral achieves at least 84.3% improvement. The good                                                                          of the original network.
fitness of the vertex degree reconstructed by our algorithms
suggests that we can better preserve the scale-free property                                                                       4.3    Link Prediction
of the network. Besides, we see that the best ε to maximize                                                                        In this task, we consider the following evaluation metrics:
Pearson’s r for DP-Spectral is more stable (i.e., around 0.51)                                                                     Precision, Recall, and F1-score. Table 3 demonstrates the
than other methods. Thus one can tune DP-Spectral’s param-                                                                         performance of different methods on the link prediction task.
eters more easily in practice.                                                                                                     For our methods, we use the model parameter β as optimized
Preserving Scale-free Property. After reconstructing the                                                                           in Table 2. We see that, in most cases, DP-Spectral obtains
      Dataset       Method        Precision    Recall          F1                         5       Related Work
                      LE            0.52        0.53           0.53
                   Deepwalk         0.51        0.51           0.51
  Synthetic                                                            Network embedding. Network embedding aims to learn
                  DP-Spectral       0.64        0.68           0.66
                  DP-Walker         0.61        0.63           0.62    representations for vertexes in a given network. Some re-
                      LE            0.75        0.92           0.83    searchers regard network embedding as part of dimensional-
                   Deepwalk         0.84        0.97           0.90    ity reduction techniques. For example, Laplacian Eigenmaps
  Facebook
                  DP-Spectral       0.76        0.98           0.85    (LE) (Belkin and Niyogi 2003) aims to learn the vertex rep-
                  DP-Walker         0.82        0.95           0.89
                      LE            0.58        0.35           0.43
                                                                       resentation to expand the manifold where the data lie. As
                   Deepwalk         0.65        0.77           0.70    a variant of LE, Locality Preserving Projections (LPP) (He
      Twitter                                                          et al. 2005) learns a linear projection from feature space
                  DP-Spectral       0.59        0.98           0.73
                  DP-Walker         0.54        0.58           0.56    to embedding space. Besides, there are other linear (Jol-
                      LE            0.61        0.83           0.70    liffe 2002) and non-linear (Tenenbaum, De Silva, and Lang-
                   Deepwalk         0.55        0.58           0.56    ford 2000) network embedding algorithms for dimension-
   Coauthor
                  DP-Spectral       0.62        0.89           0.73
                  DP-Walker         0.56        0.66           0.61
                                                                       ality reduction. Recent network embedding works take ad-
                      LE            0.54        0.56           0.55    vancements in natural language processing, most notably
                   Deepwalk         0.54        0.56           0.55    models known as word2vec (Mikolov et al. 2013), which
   Citation                                                            learns the distributed representations of words. Building on
                  DP-Spectral       0.52        0.99           0.68
                  DP-Walker         0.55        0.57           0.56    word2vec, Perozzi et al. define a vertex’s “context” by their
                      LE            0.75        0.36           0.48    co-occurrence in a random walk path (Perozzi, Al-Rfou,
                   Deepwalk         0.55        0.60           0.57    and Skiena 2014). More recently, Grover et al. propose a
      Mobile
                  DP-Spectral       0.63        0.89           0.74
                  DP-Walker         0.54        0.58           0.56    mixture of width-first and breadth-first search based proce-
                                                                       dure to generate paths of vertexes (Grover and Leskovec
                                                                       2016). Dong et al. further develop the model to handle het-
        Table 3: Experimental results of link prediction.              erogeneous networks (Dong, Chawla, and Swami 2017).
                                                                       LINE (Tang et al. 2015) decomposes a vertex’s context into
   Method       Acrh   CN        CS    DM     THM       GRA     UNK    first-order (neighbors) and second-order (two-degree neigh-
     LE         0.36   0.75     0.14   0.37   0.46      0.13    0.86   bors) proximity. Wang et al. preserve community informa-
  Deepwalk      0.54   0.54     0.52   0.56   0.56      0.56    0.85   tion in their vertex representations (Wang et al. 2017). How-
 DP-Walker      0.56   0.57     0.54   0.58   0.58      0.55    0.85   ever, all of above methods focus on preserving microscopic
 DP-Spectral    0.71   0.74     0.78   0.76   0.74      0.75    0.85   network structure and ignore macroscopic scale-free prop-
                                                                       erty of networks.
Table 4: Accuracy of multi-classification task. The labels             Scale-free Networks. The scale-free property has been dis-
stand Architecture, Computer Network, Computer Science,                covered to be ubiquitous over a variety of network sys-
Data Mining, Theory, Graphics, and Unknown, respectively.              tems (Mood 1950; Newman 2005; Clauset, Shalizi, and
                                                                       Newman 2009b), such as the Internet Autonomous System
                                                                       graph (Faloutsos, Faloutsos, and Faloutsos 1999), the In-
the best performance, which suggests that with the help of             ternet router graph (Govindan and Tangmunarunkit 2000),
the proposed principle, we can not only preserve the scale-            the degree distributions of subsets of the world wide
free property of networks, but also improve the effectiveness          web (Barabási and Albert 1999). Newman provides a com-
of the embedding vectors.                                              prehensive list of such work (Newman 2005). However, in-
                                                                       vestigating the scale-free property in a low-dimensional vec-
                                                                       tor space and establishing its cooperation with network em-
                                                                       bedding have not been fully considered.
4.4     Vertex Classification

Table 4 lists the accuracy of vertex classification task on Ci-                               6    Conclusion
tation. Our task is to determine an author’s research area,
which is a multi-classification problem. We define features            In this paper, we study the problem of learning the scale-free
as vertex representation obtained by the four different em-            property preserving network embeddings. We first analyze
bedding algorithms. Generally, from the table, we see that             the feasibility of reconstructing a scale-free network based
DP-Walker and DP-Spectral beat respectively Deepwalk and               on learned vertex representations in the Euclidean space by
Laplacian Eigenmap. In particular, DP-Spectral achieves the            converting our problem to the Sphere Packing problem. We
best result for 5 out of 7 labels. Besides, we can also observe        then propose the degree penalty principle as our solution
its stability of the performance. DP-Spectral’s result on all          and introduce two implementations by leveraging spectral
labels is more stable than other methods. In comparison, LE            techniques and a skip-gram model respectively. The pro-
achieves a satisfactory result for two labels, but for others          posed principle can also be implemented using other meth-
the result can be poor. Specifically, the standard deviation of        ods, which is left as our future work. We at last conduct
DP-Spectral is 0.04, while the value is 0.26 for LE and 0.1            extensive experiments on both synthetic data and five real-
for DeepWalk.                                                          world datasets to verify the effectiveness of our approach.
 Acknowledgements. The work is supported by the                 [Kabatiansky and Levenshtein 1978] Kabatiansky, G. A.,
 Fundamental Research Funds for the Central Universi-            and Levenshtein, V. I. 1978. On bounds for packings on
 ties, 973 Program (2015CB352302), NSFC (U1611461,               a sphere and in space. Problemy Peredachi Informatsii
 61625107, 61402403), and key program of Zhejiang                14(1):3–25.
 Province (2015C01027).                                         [Kendall 1938] Kendall, M. G. 1938. A new measure of rank
                                                                 correlation. Biometrika 30(1/2):81–93.
                        References                              [Leskovec and Mcauley 2012] Leskovec, J., and Mcauley, J.
[Adamic and Huberman 1999] Adamic, L., and Huberman,             2012. Learning to discover social circles in ego networks.
 B. A. 1999. The nature of markets in the world wide web.        In NIPS’12, 539–547.
 Q. J. Econ.
                                                                [Leskovec, Kleinberg, and Faloutsos 2007] Leskovec,          J.;
[Alanis-Lobato, Mier, and Andrade-Navarro 2016] Alanis-          Kleinberg, J. M.; and Faloutsos, C. 2007. Graph evolution:
 Lobato, G.; Mier, P.; and Andrade-Navarro, M. A. 2016.          Densification and shrinking diameters. ACM Transactions
 Efficient embedding of complex networks to hyperbolic           on Knowledge Discovery From Data 1(1):2.
 space via their laplacian. In Scientific reports.
                                                                [Mikolov et al. 2013] Mikolov, T.; Chen, K.; Corrado, G.;
[Alstott, Bullmore, and Plenz 2014] Alstott, J.; Bullmore,       and Dean, J. 2013. Efficient estimation of word representa-
 E.; and Plenz, D. 2014. powerlaw: A Python Package              tions in vector space. In NIPS’13, 3111–3119.
 for Analysis of Heavy-Tailed Distributions. PLoS ONE
                                                                [Mnih and Hinton 2009] Mnih, A., and Hinton, G. E. 2009.
 9:e85777.
                                                                 A scalable hierarchical distributed language model. In
[Barabási and Albert 1999] Barabási, A.-L., and Albert, R.     Koller, D.; Schuurmans, D.; Bengio, Y.; and Bottou, L.,
 1999. Emergence of scaling in random networks. science          eds., Advances in Neural Information Processing Systems
 286(5439):509–512.                                              21. Curran Associates, Inc. 1081–1088.
[Belkin and Niyogi 2003] Belkin, M., and Niyogi, P. 2003.       [Mood 1950] Mood, A. M. 1950. Introduction to the theory
 Laplacian eigenmaps for dimensionality reduction and data       of statistics.
 representation. Neural Comput. 15(6):1373–1396.
                                                                [Morin and Bengio 2005] Morin, F., and Bengio, Y. 2005.
[Clauset, Shalizi, and Newman 2009a] Clauset, A.; Shalizi,       Hierarchical probabilistic neural network language model.
 C. R.; and Newman, M. E. J. 2009a. Power-law distribu-          In AISTATS05, 246–252.
 tions in empirical data. SIAM Review 51(4):661–703.
                                                                [Newman 2005] Newman, M. E. 2005. Power laws,
[Clauset, Shalizi, and Newman 2009b] Clauset, A.; Shalizi,       pareto distributions and zipf’s law. Contemporary physics
 C. R.; and Newman, M. E. 2009b. Power-law distributions         46(5):323–351.
 in empirical data. SIAM review 51(4):661–703.
                                                                [Perozzi, Al-Rfou, and Skiena 2014] Perozzi, B.; Al-Rfou,
[Cohn and Elkies 2003] Cohn, H., and Elkies, N. 2003. New        R.; and Skiena, S. 2014. Deepwalk: Online learning of so-
 upper bounds on sphere packings i. Annals of Mathematics        cial representations. In KDD’14, 701–710.
 689–714.
                                                                [Shao 2007] Shao, J.       2007.    Mathematical Statistics.
[Cohn and Zhao 2014] Cohn, H., and Zhao, Y. 2014. Sphere         Springer, 2 edition.
 packing bounds via spherical codes. Duke Mathematical
 Journal 163(10):1965–2002.                                     [Shaw and Jebara 2009] Shaw, B., and Jebara, T. 2009.
                                                                 Structure preserving embedding. In Proceedings of the
[Dong, Chawla, and Swami 2017] Dong, Y.; Chawla, N.;             26th Annual International Conference on Machine Learn-
 and Swami, A. 2017. metapath2vec: Scalable representation       ing, ICML ’09, 937–944. New York, NY, USA: ACM.
 learning for heterogeneous networks. In KDD’17, 135–144.
                                                                [Spearman 1904] Spearman, C. 1904. The proof and mea-
[Faloutsos, Faloutsos, and Faloutsos 1999] Faloutsos, M.;        surement of association between two things. The American
 Faloutsos, P.; and Faloutsos, C. 1999. On power-law rela-       Journal of Psychology 15(1):72–101.
 tionships of the internet topology. In COMPUT COMMUN
 REV, volume 29, 251–262.                                       [Tang et al. 2008] Tang, J.; Zhang, J.; Yao, L.; Li, J.; Zhang,
                                                                 L.; and Su, Z. 2008. Arnetminer: extraction and mining of
[Govindan and Tangmunarunkit 2000] Govindan, R., and
                                                                 academic social networks. In KDD’08, 990–998.
 Tangmunarunkit, H. 2000. Heuristics for internet map dis-
 covery. In INFOCOM’00, volume 3, 1371–1380.                    [Tang et al. 2015] Tang, J.; Qu, M.; Wang, M.; Zhang, M.;
                                                                 Yan, J.; and Mei, Q. 2015. Line: Large-scale information
[Grover and Leskovec 2016] Grover, A., and Leskovec, J.
                                                                 network embedding. In WWW’15, 1067–1077.
 2016. node2vec: Scalable feature learning for networks. In
 KDD’16, 855–864.                                               [Tenenbaum, De Silva, and Langford 2000] Tenenbaum,
                                                                 J. B.; De Silva, V.; and Langford, J. C. 2000. A global ge-
[He et al. 2005] He, X.; Yan, S.; Hu, Y.; Niyogi, P.; and
                                                                 ometric framework for nonlinear dimensionality reduction.
 Zhang, H. 2005. Face recognition using laplacianfaces.
                                                                 science 290(5500):2319–2323.
 IEEE Transactions on Pattern Analysis and Machine Intelli-
 gence 27(3):328–340.                                           [Vance 2011] Vance, S. 2011. Improved sphere packing
                                                                 lower bounds from hurwitz lattices. Advances in Mathemat-
[Jolliffe 2002] Jolliffe, I. 2002. Principal component analy-
                                                                 ics 227(5):2144–2156.
 sis. Wiley Online Library.
[Vazquez 2003] Vazquez, A. 2003. Growing network with           search notices 2013(7):1628–1642.
 local rules: Preferential attachment, clustering hierarchy,   [Wang et al. 2017] Wang, X.; Cui, P.; Wang, J.; Pei, J.; Zhu,
 and degree correlations. Physical Review E 67(5):056104.       W.; and Yang, S. 2017. Community preserving network
[Venkatesh 2012] Venkatesh, A. 2012. A note on sphere           embedding. In AAAI’17.
 packings in high dimension. International mathematics re-

