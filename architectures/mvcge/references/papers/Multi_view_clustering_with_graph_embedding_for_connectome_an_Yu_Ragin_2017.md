# Multi view clustering with graph embedding for connectome an Yu Ragin 2017

> Source: `Multi_view_clustering_with_graph_embedding_for_connectome_an_Yu_Ragin_2017.pdf`

---

                     Multi-view Clustering with Graph Embedding for
                                  Connectome Analysis

                                          Guixiang Ma∗ , Lifang He∗ , Chun-Ta Lu∗ , Weixiang Shao† ,

                                                      Philip S. Yu ¶,∗ , Alex D. Leow∗ , Ann B. Ragin‡
                                                        ∗ University of Illinois at Chicago, Chicago, IL, USA
                                                                     † Google Inc, Mountain View, USA
                ¶ Shanghai Institute for Advanced Communication and Data Science, Fudan University, Shanghai, China
                                                             ‡ Northwestern University, Chicago, IL, USA

          {gma4,clu29,psyu}@uic.edu,{lifanghescut,software.shao,alexfeuillet}@gmail.com,ann-ragin@northwestern.edu

ABSTRACT                                                                                      instead of feature vectors as in traditional data. Brain networks, for
Multi-view clustering has become a widely studied problem in the                              example, are comprised of anatomic regions as nodes, and func-
area of unsupervised learning. It aims to integrate multiple views                            tional/structural connectivities between the brain regions as links.
by taking advantages of the consensus and complimentary informa-                              Linkage structures often come from different sources, called as
tion from multiple views. Most of the existing works in multi-view                            multi-view data. For instance, fMRI (functional magnetic resonance
clustering utilize the vector-based representation for features in                            imaging) and DTI (diffusion tensor imaging) are two major neu-
each view. However, in many real-world applications, instances are                            roimaging approaches widely used in neuroscience research and
represented by graphs, where those vector-based models cannot                                 in clinical applications [8, 24, 47]. Connections in brain networks
fully capture the structure of the graphs from each view. To solve                            derived from fMRI brain images encode correlations in functional
this problem, in this paper we propose a Multi-view Clustering                                activity among brain regions, whereas DTI networks provide infor-
framework on graph instances with Graph Embedding (MCGE).                                     mation concerning structural connections (i.e. white matter fiber
Specifically, we model the multi-view graph data as tensors and                               paths) between different brain regions. The different networks af-
apply tensor factorization to learn the multi-view graph embed-                               ford two different views of the brain connectivity.
dings, thereby capturing the local structure of graphs. We build an                              Multi-view clustering has received considerable attention for un-
iterative framework by incorporating multi-view graph embedding                               labeled data with multiple views from diverse domains. While there
into the multi-view clustering task on graph instances, jointly per-                          have been advances in multi-view clustering, most approaches are
forming multi-view clustering and multi-view graph embedding                                  based on vector representation of features in each view and com-
simultaneously. The multi-view clustering results are used for refin-                         bining vectors from different views for the clustering task [21, 46].
ing the multi-view graph embedding, and the updated multi-view                                However, the complex structures and the lack of vector represen-
graph embedding results further improve the multi-view clustering.                            tations within graph data, pose serious challenges for this kind
Extensive experiments on two real brain network datasets (i.e., HIV                           of vector-based approach. It is desirable to find a way that can
and Bipolar) demonstrate the superior performance of the proposed                             better capture and exploit graph structural information for multi-
MCGE approach in multi-view connectome analysis for clinical                                  view clustering of graph instances. To address this problem, this
investigation and application.                                                                paper explores an approach involving multi-view clustering of
                                                                                              graph instances based on graph embedding and its application to
KEYWORDS                                                                                      connectome analysis in multi-view brain networks on HIV and
                                                                                              Bipolar. The goal of graph embedding is to find low-dimensional
Multi-view Clustering, Graph Embedding, Connectome Analysis
                                                                                              representations of graphs that can preserve the inherent structure
                                                                                              and properties [26, 45]. While graph embedding technology has
1     INTRODUCTION                                                                            been broadly used for graph mining, to the best of our knowledge,
Advances in capabilities for data acquisition have given rise to an                           this approach has not been used for multi-view clustering of graph
explosion of new information in the form of graph representations.                            instances. There are two main challenges that must be addressed
These data are inherently represented as a set of nodes and links,                            for the problem of multi-view clustering with graph embedding:
Permission to make digital or hard copies of all or part of this work for personal or             • How to learn the graph embedding for each graph instance
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation           with multiple views, such that the graph embeddings can
on the first page. Copyrights for components of this work owned by others than ACM                  encode the multi-view structure information of the graphs?
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,             Specifically, the embeddings of the similar nodes within the
to post on servers or to redistribute to lists, requires prior specific permission and/or a
fee. Request permissions from permissions@acm.org.                                                  graph instance should be close.
CIKM’17 , November 6–10, 2017, Singapore, Singapore                                               • How to leverage the multi-view graph embedding results to
© 2017 Association for Computing Machinery.                                                         facilitate the multi-view clustering task on graph instances?
ACM ISBN 978-1-4503-4918-5/17/11. . . $15.00
https://doi.org/10.1145/3132847.3132909                                                             The graph embeddings mainly captures the local structure
                      fMRI   DTI                         Embedding                             Table 1: List of basic symbols.
         Subject 1


                                                                                  Symbol     Definition and description
         Subject 2
                                   Learning Multi-view
                                    Graph Embedding
                                                                 Group 1
                                                                                  x          each lowercase letter represents a scale
         Subject 3                                                                x          each boldface lowercase letter represents a vector
                                                                                  X          each boldface uppercase letter represents a matrix
          Subject 4
                                                                                  X          each calligraphic letter represents a tensor
                                                                                  ⟨·, ·⟩     denotes inner product
          Subject 5
                                                                 Group 2
                                                                                  ◦          denotes tensor product (outer product)
                                                                                  ⊗          denotes Kronecker product
                                                                                  ⊙          denotes Khatri-Rao product
        Figure 1: An example of the MCGE problem

                                                                           2    PRELIMINARIES
      of graphs, while the similarity between the graph instances          In this section, we first introduce some notations and terminologies
      holds their global structure information. For multi-view clus-       that we will use throughout the paper. Then we formulate the
      tering on graph data, it is critical to appropriately fuse these     problem of interest formally.
      two kinds of graph structure information .                               Notations. Vectors are denoted by boldface lowercase letters,
                                                                           matrices are denoted by boldface capital letters, and tensors are
   To address the above challenges, we propose the MCGE (multi-            denoted by calligraphic letters. An element of a vector x, a matrix
view clustering with graph embedding) framework. Our contribu-             X, or a tensor X is denoted by x i , x i j , x i jk , etc., depending on the
tions can be summarized as:                                                number of indices (also known as modes). For a matrix X ∈ Rn×m ,
    • We model the multi-view graph data as tensors, and apply             its i-th row and j-th column are denoted by xi and xj , respectively.
                                                                                                                                       qÍ
      tensor factorization to learn the multi-view embeddings of           The Frobenius norm of X is defined as ∥X∥ F =                  n ∥xi ∥ 2 . For
                                                                                                                                          i=1     2
      graphs. In this manner, the graph embeddings can capture
                                                                           any vector x ∈ Rn , Diaд(x) ∈ Rn×n is the diagonal matrix whose
      the key local structure of the graphs in all the views, while
                                                                           diagonal elements are x i . In denotes an identity matrix with size n.
      also encoding the latent correlations between different views.
                                                                           We denote an undirected graph as G = (V , E), where V is the set of
    • We employ graph kernel to measure the similarity between
                                                                           nodes and E ⊂ V × V is the set of edges. An overview of the basic
      graph instances in each view, construct a multi-view kernel
                                                                           symbols used in this paper can be found in Table 1.
      tensor based on kernel matrices, and obtain the common
      latent factors that encode the global structure information.           Definition 2.1 (Tensor). An nth-order tensor X ∈ RI1 ×I2 ×···×In is
    • We propose to jointly perform the multi-view graph embed-            an element of the outer product of n vector spaces, each of which
      ding stage and the multi-view clustering stage in an iterative       has its own coordinate system.
      manner. Considering the fact that the graphs clustered into
      the same group tend to have similar local structure, for each            Definition 2.2 (Outer product). The outer product of vectors x(k ) ∈
      graph, we use the multi-view embeddings of the neighbour             RIk for k = 1, 2, · · · , n is an n-th order tensor and defined elemen-
                                                                                                                                     (1) (2)           (n)
                                                                           twise by x(1) ◦ x(2) ◦ · · · ◦ x(n) i 1,i 2, ··· ,i n = x i 1 x i 2 · · · x i n =
                                                                                                                  
      graphs clustered in the same group to refine its multi-view
                                                                           În      (k)
      embedding. Then the updated multi-view embeddings of the                   x for all values of the indices.
                                                                             k =1 i k
      graphs will be used for the multi-view clustering stage in the
      next iteration. Following this iterative two-stage process, the        Definition 2.3 (Kronecker Product). The Kronecker product of
      multi-view graph embedding and multi-view clustering will            two matrices A ∈ RI ×J , B ∈ RK ×L is a matrix in the dimension of
      be improved until we obtain an optimal clustering results.           IK × JL:
    • We apply the proposed MCGE framework for unsupervised                                           a 11 B      a 12 B   ···    a 1J B
      multi-view connectome analysis on HIV and Bipolar. Specif-                                    © a B         a 22 B   ···    a 2J B ª®
                                                                                                    ­ 21
      ically, we study the connectome of fMRI and DTI brain net-                           A ⊗ B = ­­ .              ..     ..       .. ®®              (1)
      works and aim to cluster the subjects with similar neuro-                                     ­ ..              .      .        . ®
      logical status into the same group as shown in Figure 1.                                      « aI 1B       aI 2B    ···    aI J B ¬
      Experimental results on the HIV and Bipolar datasets show
                                                                               Definition 2.4 (Khatri-Rao Product). The Khatri-Rao product of
      the effectiveness of MCGE for multi-view clustering in con-
                                                                           two matrices A ∈ RI ×K , B ∈ R J ×K is a matrix in dimension of
      nectome analysis.
                                                                           I J × K:
The rest of this paper is organized as follows. In the next section,                     A ⊙ B = (a 1 ⊗ b1 , a 2 ⊗ b2 , · · · , a K ⊗ bK ) (2)
problem formulation and background are given. The details of the
                                                                           where a 1 , a 2 , · · · , a K are the columns of A and b1 , b2 , · · · , bK are
proposed MCGE framework are presented in Sections 3 and 4. Ex-
                                                                           the columns of B.
tensive experimental results and analysis are shown in Section 5.
Related work is discussed in Section 6 and followed by the conclu-            Definition 2.5 (Mode-k Matricization). The mode-k matricization
sion in Section 7.                                                         of a tensor X ∈ RI1 ×I2 ×···×In , denoted by X(k ) ∈ RIk ×J , where
                                          (3)           (3)                  (3)
                                      x1               x2                   xR                 clustering with graph embedding. Then we describe the optimiza-
                             ≈                                                                 tion scheme of our framework.
                                          (2)           (2)                  (2)
                                      x1               x2                   xR

                      X          x1
                                    (1)          (1)
                                                x2                   xR
                                                                       (1)                     3.1    Multi-view Graph Embedding
                                                                                               Graph embedding is an important tool in topological graph the-
 Figure 2: The CP Factorization for a third-order tensor X                                     ory, which has been widely used in data analysis [2, 13, 45]. In
                                                                                               the unsupervised situation, conventional methods for multi-view
                                                                                               graph embedding either glued the graph affinity matrices from
     n
J = Πq=1,q,k Iq . Each tensor element with indices (i 1 , i 2 , · · · , i n )                  all the views together into a big graph [11, 14], or collaboratively
maps to a matrix element (i k , j), such that                                                  explored the consensus embedding from different views (individual
                          m
                          Õ                                                                    affinity matrices) [43, 44, 48]. However these methods can only
           j =1+                 (ip − 1)Jp , with                                             capture the linear relationships in multi-view graph data. In or-
                      p=1,p,k                                                                  der to achieve better embeddings, here we develop a multilinear
                (                                                                              embedding approach via tensorization as follows.
                    1,                     if p = 1 or (p = 2 and k = 1)                          To model the multiple views for each graph instance G i , we
         Jp =         p−1                                                                (3)
                    Πq=1,q,k Iq ,          otherwise.                                          build a tensor Ti by stacking the graph affinity matrices from all
                                                                                               the v views of the graph. Assume that the dimension of the row
    Definition 2.6 (CP Factorization). For a general tensor X ∈ RI1 ×···×In ,                  vectors in the graph embeddings is c, and let Fi ∈ Rm×c be the
its CANDECOMP / PARAFAC (CP) factorization is                                                  graph embedding of G i , i.e., the j-th row vector of Fi represent the
                                                                                               embedding of node j on graph instance G i . Then we can formulate
                                                       R
                                                       Õ
                                                               (1)                 (n)         the multi-view graph embedding as the following optimization
             X = JX(1) , · · · , X(n) K ≡                     xr ◦ · · · ◦ xr ,          (4)
                                                                                               problem based on CP factorization:
                                                       r =1
                                                                                                                                             2
                                                       (k )          (k )                                            min Ti − JFi , Fi , Hi K F
where for k = 1, 2, · · · , n, X(k ) = [x1 , · · · , xR ] are factor matri-                                          Fi,Hi
ces of size Ik × R, R is the number of factors, and J·K is used for                                                    s.t. Fi T Fi = Ic                         (7)
shorthand. Figure 2 shows the form of the CP Factorization for a
third-order tensor example.                                                                    where Fi ∈ Rm×c and Hi ∈ Rv×c are the latent factor matrices.
                                                                                                  Besides, as we discussed earlier, the graphs clustered into the
   To obtain the CP factorization JX(1) , · · · , X(n) K, the objective is                     same group tend to have more similar local structure. That is, for
to minimize the following estimation error:                                                    two graphs in the same cluster, the closer they are, the more similar
                L=         min            ∥X − JX(1) , · · · , X(n) K∥F2                 (5)   local structure they tend to have. Based on this assumption, we
                       X(1), ··· ,X(n)                                                         incorporate such global cluster information to further improve the
                                                                                               multi-view graph embedding result in Equation (7). Assuming we
However, L is not jointly convex w.r.t. X(1) , · · · , X(n) . A widely
                                                                                               can obtain a weight matrix W, where w i j denotes the weight of G j
used optimization technique is the Alternating Least Squares (ALS)
                                                                                               for G i and a larger w i j implies a closer distance between G i and
algorithm, which alternatively minimize L for each variable while
                                                                                               G j in the same cluster. By incorporating the weighted influence
fixing the other, that is,
                                                                                               from the neighbor graphs into Equation (7), we have the following
                                              n
             X(k ) ← arg min ∥X(k ) − X(k ) (⊙i,k X(i) )T ∥F2                            (6)   objective function:
                            X(k )                                                                                                                         2
                                                                                                                                   2
                                                                                                                                               Õ
         n X(i) = X(n) ⊙ · · · X(k +1) ⊙ X(k−1) · · · ⊙ X(1) .                                            min Ti − JFi , Fi , Hi K F + β Fi −     wi j Fj
where ⊙i,k                                                                                                Fi,Hi                                           F
                                                                                                                                                j                (8)
   Problem Definition We study the problem of multi-view clus-
tering of graph instances with multi-view graph embedding. As-                                            s.t. Fi T Fi = Ic
sume we are given a set of instances D = {G 1 , G 2 , · · · , G n } with                       where β is a parameter balancing the two parts.
v views, where each instance is represented with a graph with                                    In the following section, we will show how to incorporate the
m nodes in each view. For the j-th view, we have a set of graphs                               graph embeddings into the multi-view clustering framework and
                                        (j) (j)          (j)
with the affinity matrices D (j) = {G1 , G2 , · · · , Gn }. The goal                           how to obtain the weight matrix W from the clustering results.
of multi-view clustering on D is to cluster the graphs in D into k
subsets. Figure 1 shows a simple two-view example of the MCGE                                  3.2    Multi-view Clustering via Graph
problem intuitively. Given the fMRI and DTI brain networks of five                                    Embedding
subjects, MCGE aims to learn multi-view graph embedding for each                               Since graph embedding usually encodes local structure of graphs,
of them, and cluster these subjects into different groups based on                             and the original affinity matrix holds the global structure, we pro-
the obtained multi-view graph embeddings.                                                      pose to consider both of these two kinds of structure information
                                                                                               for the multi-view clustering task. Specifically, we employ the graph
3    MCGE FRAMEWORK                                                                            kernel to measure the similarity of the global structure between
In this section, we first present the proposed MCGE framework con-                             different graphs. Graph kernel is a pervasive method for comparing
sisting of two stages: multi-view graph embedding and multi-view                               graphs [37]. Here we employ the random walk graph kernel [37],
                                                                                                                                  Multi-View Graph Embedding
                                                                                                               multi-view


                                                           {
                                                                           (v)
                                                                          Gi      Stack                     graph embedding
                                                                                                                                       Fi
                                        graph instance
                                     from multiple views            (1)
                                                                   Gi
                                                                                             τi                               graph embedding
                                                                                   multi-view tensor instance                               compute similarity S




                                                           G(1)
                                     view 1
                                              {    (1)
                                                  G1
                                                            n


                                                                     calculate
                                                                                  kernel
                                                                                  matrix

                                                                                                 Stack
                                                                                                                         Multi-view
                                                                                                                         Clustering
                                                                                                                                            compute weight W


                                                                                                                                                K-means
                                                                   graph kernel
                                                                                                           X                           B

                                              {          Gn(v)                                            multi-view             latent factor     cluster indicator
                                                                                  kernel                 kernel tensor              matrix
                                     view v        (v)                            matrix
                                                  G1
                                                                                                                              Multi-View Clustering of Graphs




                                              Figure 3: The framework of the proposed model MCGE.


which is one of the most widely used graph kernels, to measure                                             following optimization problem:
the similarity between the affinity matrices of different graphs in                                                                                     2
                                                                                                                                                                     
each view. Since we have n graphs in v views, we will get v kernel                                                                    min X − JB, B, AK F + αTr BT LB
                                                                                                                                      B,A                                         (12)
matrices, each with dimension of n × n. In order to integrate the
multiple views, we propose to stack the v kernel matrices together,                                                                    s.t. BT B = Ik
which form a tensor X ∈ Rn×n×v . Then we apply CP factorization                                            where α is a parameter balancing two parts.
on the tensor X to get the common factor matrices across all the                                              After we obtain matrix B, we can apply k-means clustering on
views. Suppose the number of factors is k, X can be factorized as:                                         the row vectors of B and then we know which graphs are clustered
                                                                                                           into the same group and which ones are not. This result will help de-
                               X = JB, B, AK                                               (9)             termine the weight matrix W for the multi-view graph embedding
                                                                                                           stage. Specifically, for graph G i we consider the graphs from the
where B ∈ Rn×k and A ∈ Rv×k are the latent factor matrices.                                                same cluster with G i , and we aim to infer the weights of influence
Notably, B can be interpreted as the common latent factor across                                           they should have on G i . Suppose we use Xi to represent both the
all the views, which can be used for clustering the graphs.                                                global and local structure of G i ,then this problem can be formulated
     Now let us consider how to incorporate the results of multi-                                          as the following minimization problem based on LLE method [31]:
view graph embedding into the multi-view clustering stage. As we                                                                     Õ           Õ           2
discussed above, the multi-view graph embeddings imply the local                                                                min       Xi −       wi j Xj
                                                                                                                                              W                            F
structure of the graphs, and graphs with similar local structure tend                                                                                 i                j
                                                                                                                                                    Õ
to be close to each other in the original multi-view feature space.                                                                            s.t.  wi j = 1                     (13)
     Suppose we have obtained a set of graph embeddings F =                                                                                           j
{F1 , F2 , · · · ,Fn }, where Fi ∈ Rm×c is the multi-view graph em-
bedding for G i , we can build a similarity matrix S ∈ Rn×n , where                                        where w i j denotes the weight of G j for G i , and w i j = 0 if G j and G i
si j denotes the similarity between two examples G i and G j in terms                                      are not in the same cluster. Note that there is no need for an explicit
of graph embedding, and we define it as:                                                                   definition of Xi here, as it will be implicitly represented with both
                                                                                                           the affinity matrices and the multi-view graph embedding results,
                                                  2                                                        which will be used for the optimization of W. The details will be
                            si j = 1 − Fi − Fj F                                      (10)
                                                                                                           illustrated in Section 4.

  Then we can formulate the following objective function on the                                            3.3       The Overall Framework: MCGE
basis of the spectral analysis [38]:
                                                                                                           With the two stages discussed above, we can formulate the over-
                        n                         2                                                        all iterative process for the MCGE framework. As the multi-view
                        Õ        bi     bj                                
                                                                                                           graph embedding and multi-view clustering depend on each other,
          min =            si j √     −p        = Tr BT LB
            B
                    i, j=1       d ii   d j j 2                                       (11)                 we propose to jointly perform these two stages. In each iteration,
                    T                                                                                      we first perform the multi-view graph embedding on each graph,
            s.t. B B = Ik
                                                                                                           and then utilize the obtained graph embedding in the multi-view
                1              1                                                                           clustering stage. Then the resulted graph cluster information will
 where L = D− 2 (D − S)D− 2 is the symmetric normalized Laplacian                                          be used for refining the multi-view graph embeddings in the next
matrix, and D is a diagonal matrix with dii = nj=1 si j .
                                              Í
                                                                                                           iteration. Following this alternate two-stage process, the multi-view
By combining the above tensor CP factorization strategy with Equa-                                         graph embedding and multi-view clustering will be improved by
tion (11), we can formulate the multi-view clustering task as the                                          each other until convergence.
   An overview of our framework is shown in Figure 3. The upper                  where U ∈ Rn×k are Lagrange multipliers, and µ is the penalty
part demonstrates the multi-view graph embedding stage in MCGE,                  parameter. Then the objective function with respect to B can be
and the lower part shows the multi-view clustering stage, while                  derived as:
the blue arrow and red arrow indicate the interaction of the two                                      2   µ          1 2               
stages. Overall, given a set of graph instances D = {G 1 , G 2 , · · · , G n }       min BQT − X(1) +        B − P− U + αTr BT LB
                                                                                      B               F   2          µ F                     (16)
with v views, we aim to obtain a multi-view graph embedding for
each of these graph instances, and then use the multi-view graph                      s.t. BT B = Ik
embeddings as key features for clustering graph instances.
   As shown in Figure 3, in the multi-view graph embedding stage,                where Q = P ⊙ A ∈ R(n∗v)×k and X(1) ∈ Rn×(n∗v) is the mode-1
for each graph instance G i , we stack its affinity matrices from all            matricization of X.
the v views together to form a multi-view tensor instance Ti . Then                 As such an optimization problem with orthogonal constraint
we apply tensor factorization in Equation (8) to learn its multi-view            has been well studied, and can be solved by a few solvers [1, 41],
embedding, which partially depend on the embeddings of the other                 here we employ the solver Algorithm 2 in [41] to solve Equation
graphs from the same cluster that is determined by the multi-view                (16), which is a more efficient optimization algorithm with code
clustering stage. Meanwhile, in the multi-view clustering stage, we              publicly available. Since this algorithm requires the derivative of
first measure the similarity between each pair of the graphs by                  the objective function as one input, we obtain the derivative of
calculating the graph kernel from each view, and then we stack                   Equation (16) with respect to B:
the kernel matrices from all the views, resulting in a multi-view                               ∇B L (B) =2BQT Q − 2X(1) Q + µ (B − P) −
kernel tensor X. By utilizing the CP factorization on X, we can                                                                                  (17)
                                                                                                          U + α 2LB + LT B
get the common factor B across all the views. Considering the
importance of graph embedding in capturing graph structure, we                     Then the auxiliary matrix P can be optimized by setting the
compute the similarity between graphs based on the multi-view                    derivative of Equation (15) with respect to P as 0. We have:
graph embedding results and incorporate it into the CP factorization                                                              −1
scheme with a spectral analysis term, as shown in Equation (12). The                           P = 2X(2) O + µB − U 2OT O + µI                (18)
latent factor B obtained from this step will indicate which graphs
are closer to each other, thus can be further used for computing the             where O = B ⊙ A ∈ R(n∗v)×k and X(2) ∈ Rn×(n∗v) is the mode-2
weight matrix W, which will be used for updating the multi-view                  matricization of tensor X.
graph embeddings in the next iteration. Vice versa, the new multi-                 After updating B and P, we optimize the Lagrangian multipliers
view graph embeddings will be used for updating the similarity S,                U by gradient ascent:
thus improving the multi-view clustering stage.
                                                                                                             U ← U + µ (P − B)                    (19)

4    OPTIMIZATION                                                                Note that in our experiment, we initialize µ as 10−6 , and set µmax =
Since the objective function in Equation (12) is not convex with                 107 . Each time after U is updated, we adjust µ by µ = min(ρµ, µmax ),
respect to A and B jointly, and Equation (8) is not convex with                  where we set ρ = 1.05.
respect to Fi , there is no closed-form solution for such problem. We            Update factor matrix A. Next, we fix B and optimize A. Following
employ an Alternating Direction Method of Multipliers (ADMM)                     Equation (12), the objective function with respect to A is:
scheme [4, 40] to solve these problems, which alternately updates                                                             2
one variable while fixing others until convergence.                                                      min AZT − X(3)                           (20)
                                                                                                             A                F
   We first solve the optimization problem in Equation (12). The
variables to be estimated include B and A.                                       where Z = B ⊙ P ∈ R(n∗n)×k and X(3) ∈ Rv×(n∗n) is the mode-3
Update factor matrix B. We first update B while fixing A. Due                    matricization of X, thus this can be solved directly.
to the fourth-order term, the objective function in Equation (12) is                By performing the above optimization steps iteratively until
not convex with respect to B, thus being difficult to optimize. We               convergence, we can obtain the optimal indicator matrix B for the
employ the variable substitution technique to solve this problem.                multi-view clustering stage, thus knowing which graphs are clus-
By substituting the second B with P in Equation (12), we obtain the              tered together by performing k-means algorithm on the row vectors
equivalent form of Equation (12):                                                of B. The resulted cluster information will be used for determining
                                                                                 the weight matrix W in the multi-view graph embedding stage.
                  min ∥X − JB, P, AK∥F2 + αTr(BT LB)                                Now we solve the optimization problem in Equation (13) with
                   B                                                     (14)    respect to the weight matrix W. According to the locally linear
                   s.t. P = B, BT B = Ik                                         embedding approach proposed in [32], such a minimization problem
                                                                                 with respect to vectors can be solved as a constrained least squares
where P is auxiliary variable. The augmented Lagrangian function                 problem. Since the Frobenius norm for matrices can be regarded as
for (14) is:                                                                     a generalization of the l 2 norm for vectors, we can directly derive
                                                                               the following equation based on the analysis in [32]:
                                       2
              L (B, P) = X − JB, P, AK F + Tr UT (P − B)                                               Õ           2   Õ
                          µ                                            (15)                     Xi −      wi j Xj =      w i j w ir Cjr        (21)
                        − ∥P − B ∥ 2F + α Tr BT LB                                                                  F
                          2                                                                              j               jr
Algorithm 1 MCGE                                                                imaging (DTI) for each subject, from which we can construct
Input: X , { T1, · · · , Tn }, c , k , α , β                                    the fMRI and DTI brain networks.
Output: B, F
                                                                              • Bipolar: This dataset consists of the resting-state fMRI and
 1: Initialize B s.t. BT0 B0 = Ik ;
 2: Initialize Fi for i = 1, 2, · · · , n s.t. Fi T0 Fi 0 = Ic ;                DTI image data of 52 bipolar I subjects who are in euthymia
 3: while not converge do                                                       and 45 healthy controls with matched age and gender [9, 25].
 4:     Compute W according to Equation (24);
 5:     for i = 1 : n do                                                    We perform preprocessing on the HIV dataset using the standard
             t ← 0;
 6:
 7:          while not converge do
                                                                        process as illustrated in [8]. First, we use the DPARSF toolbox1 to
 8:             Compute Fi t +1 by solving Equation (8);                process the fMRI data. We realign the images to the first volume,
 9:             t ← t + 1;
10:          end while                                                  do the slice timing correction and normalization, and then use an
11:      end for                                                        8-mm Gaussian kernel to smooth the image spatially. The band-pass
12:      Update B by solving Equation (14);
13:      Update A by solving Equation (20);                             filtering (0.01-0.08 Hz) and linear trend removing of the time series
14:      Cluster B by k -means;                                         are also performed. We focus on the 116 anatomical volumes of
15: end while
                                                                        interest (AVOI), each of which represents a specific brain region, and
                                                                        extract a sequence of responds from them. Finally, we construct
where G j and G r are two neighbor graphs of G i in the same cluster.   a brain network with the 90 cerebral regions. Each node in the
Cjr is the local covariance matrix, and it can be computed by           graph represents a brain region, and links are created based on the
                                                                        correlations between different brain regions. For the DTI data, we
                         1                                              use FSL toolbox2 for the preprocessing and then construct the brain
                           (M j + Mr − m jr − M 0 )
                              Cjr =                            (22)
                         2                                              networks. The preprocessing includes distortion correction, noise
where m jr denotes the squared distance between the jth and r th        filtering, repetitive sampling from the distributions of principal
neighbors of G i , and we compute it based on both the original         diffusion directions for each voxel. We parcellate the DTI images
affinity matrices from v views and the graph embeddings of G j and      into the 90 regions same with fMRI via the propagation of the
G r by                                                                  Automated Anatomical Labeling (AAL) on each DTI image [36].
                                    v                                       For the Bipolar dataset, the brain networks were constructed
                1 1 Õ (d )        (d ) 2    1          2                using the CONN3 toolbox [42]. The raw EPI images were first
              m jr =
                 (       Gj − Gr         ) + ( Fj − Fr F )  (23)
                2 d                    F    2                           realigned and co-registered, after which we perform the normaliza-
                    d =1
                                                                        tion and smoothing. Then the confound effects from motion artifact,
M j = z m jz , Mr = z mr z and M 0 = jr m jr . Then the optimal
     Í              Í                    Í
                                                                        white matter, and CSF were regressed out of the signal. Finally, the
weights can be obtained by:
                                Í −1                                    brain networks were derived using the pairwise signal correlations
                                  r C jr                                based on the 82 labeled Freesurfer-generated cortical/subcortical
                         wi j = Í      −1
                                                            (24)        gray matter regions.
                                 l z Cl z

   For details about the above derivation for the solution, readers     5.2      Baselines and Metrics
can refer to the illustrations in [32].                                 We compare our MCGE framework with six other baseline methods
   Once the weight matrix W is obtained, we can easily solve the        for the multi-view clustering task on brain networks. To the best
optimization problem in Equation (8) following the same ADMM            of our knowledge, our proposed framework is the first work that
steps as shown above for solving (12). The overall optimization         jointly performs multi-view graph embedding and multi-view clus-
algorithm of MCGE is summarized in Algorithm 1.                         tering of graph instances. Therefore, for the evaluation, we apply
                                                                        the following state-of-the-art multi-view clustering methods and
5      EXPERIMENTS AND EVALUATION                                       adapt them to perform the multi-view clustering task here.
In order to evaluate the performance of the proposed method for               • SingleBest applies spectral clustering on each single view
multi-view clustering of graphs, we test our framework on real fMRI             and reports the best performance among them.
and DTI brain network data for connectome analysis and compare                • SEC is a single view spectral embedding clustering frame-
with a few of state-of-the-art multi-view clustering methods.                   work proposed in [28]. It imposes a linearity regularization
                                                                                on the spectral clustering model and uses both local and
5.1        Data Collection and Preprocessing                                    global discriminative information for the embedding.
In this work, we use two real datasets as follows:                            • CoRegSc is the co-regularized based multi-view spectral
       • Human Immunodeficiency Virus Infection (HIV): This dataset             clustering framework proposed in [19]. The centroid based
         is collected from the Chicago Early HIV Infection Study at             approach is applied for the multi-view clustering task.
         Northwestern University[30]. This clinical study involves 77         • MultiNMF is the multi-view clustering method based on
         subjects, 56 of which are early HIV patients (positive) and            joint nonnegative matrix factorization proposed by [21]. It
         the other 21 subjects are seronegative controls (negative).            aims to search for a factorization that gives compatible clus-
         These two groups of subjects do not differ in demographic              tering solutions across multiple views.
         characteristics such as age, gender, racial composition and    1 http://rfmri.org/DPARSF.
         education level. This dataset contains both the functional     2 http://fsl.fmrib.ox.ac.uk/fsl/fslwiki.
         magnetic resonance imaging (fMRI) and diffusion tensor         3 http://www.nitrc.org/projects/conn
        Table 2: Results on HIV dataset (mean ± std).                       5.3    Performance Evaluations
                                                                               5.3.1 Clustering Accuracy and NMI. As shown in Table 2 and
              Methods         Accuracy           NMI                        Table 3, our MCGE framework performs the best in the multi-view
              SingleBest    0.561 ± 0.010   0.104 ± 0.007                   clustering task on both of the two datasets in terms of accuracy and
              SEC           0.523 ± 0.012   0.092 ± 0.011                   NMI. Among the seven methods, the first two methods are single
              AMGL          0.563 ± 0.002   0.132 ± 0.008                   view clustering methods, both of which achieve lower accuracy
              SCMV-3DT      0.576 ± 0.013   0.123 ± 0.019
                                                                            and NMI compared with the multi-view methods. In particular,
              MultiNMF      0.613 ± 0.016   0.197 ± 0.021
              CoRegSc       0.626 ± 0.020   0.254 ± 0.013
                                                                            the lowest accuracy is from SEC, which is a single view clustering
              MCGE          0.682 ± 0.019   0.390 ± 0.015                   method applied here by concatenating the features of all the views.
                                                                            Although the SEC method considers both global structure and
                                                                            local structure of graphs, it does not distinguish the features from
       Table 3: Results on Bipolar dataset (mean ± std).                    different views, which leads to a poor performance in the multi-
                                                                            view clustering. The SingleBest achieves its best performance on the
              Methods         Accuracy           NMI                        fMRI brain networks for both datasets, which means that the fMRI
                                                                            data provide more discriminative information for the SingleBest
              SingleBest    0.553 ± 0.012   0.098 ± 0.006
                                                                            method. By comparing SingleBest with SEC, we can find that if
              SEC           0.536 ± 0.012   0.103 ± 0.009
              AMGL          0.558 ± 0.026   0.101 ± 0.012                   the multiple views are combined improperly, it may perform even
              SCMV-3DT      0.585 ± 0.009   0.132 ± 0.010                   worse than only using information from a single view.
              MultiNMF      0.642 ± 0.011   0.192 ± 0.015                      Among the multi-view clustering methods, CoRegSc and Mult-
              CoRegSc       0.619 ± 0.024   0.170 ± 0.008                   iNMF have quite good performance, though not as good as the
              MCGE          0.703 ± 0.013   0.264 ± 0.012                   proposed MCGE method. This is mainly because that they consider
                                                                            the interactions between different views via joint modeling with
                                                                            the multiple views, while the other two multi-view methods do not.
    • AMGL is a recently proposed multi-view spectral learn-                Comparatively, CoRegSc achieves slightly better results than the
      ing framework [27] that can automatically learn an optimal            MultiNMF method on HIV dataset and vice versa on the Bipolar
      weight for each graph without introducing additive parame-            dataset. Compared to the proposed MCGE method, the common
      ters.                                                                 property of the other four multi-view clustering methods is that
    • SCMV-3DT is a tensor based multi-view clustering method               the features they learn for each view are based on vector represen-
      recently proposed in [46]. It uses t-product in third-order           tations. However, for graph instances, the structural information
      tensor space, and represents multi-view data by a t-linear            could barely be preserved by such vector representations, which
      combination with sparse and low-rank penalty based on the             could be the underlying reason of why these methods could not out-
      circular convolution.                                                 perform our MCGE method. Moreover, by using tensor technique
    • MCGE is the proposed multi-view clustering framework in               to model the multi-view graph-graph affinity as illustrated in Equa-
      this paper, which jointly performs multi-view graph embed-            tion (9), MCGE can not only encode the latent interaction across
      ding and multi-view clustering of the graph instances.                different views, but also capture the graph-specific features through
                                                                            the graph kernels. From Table 2 and Table 3, we can see that, as
There are three main parameters in our model, which include the             another tensor-based multi-view method, the SCMV-3DT does not
α in objective function (12), the β in objective function (8), and the      achieve compatible results to MCGE. The reason behind this might
dimension c of the row vectors in the graph embeddings. We apply            be that although SCMV-3DT models the data into third-order ten-
the grid search to find the optimal values for the parameters. For          sor, it does not consider the local structure of graphs, making it less
details, we do grid search for α and β in {10−4 , 10−3 , · · · 104 }, and   effective for the multi-view clustering of graphs.
the optimal c is selected by the grid search from {2, 3, · · · , 12}. For
evaluation, since there are two possible labels of the brain network           5.3.2 MCGE for Connectome Analysis. To evaluate the effective-
instances in both of the two datasets, we set the number of clusters k      ness of the proposed MCGE framework for connectome analysis,
to be 2, and test how well our method can group the brain networks          we investigate this approach for capturing the inner structure of
of subjects with disorders and those of normal controls into two            connectomes in analysis of brain alterations induced by HIV infec-
different clusters.                                                         tion and Bipolar affective disorder, respectively.
   For fair comparisons of the baseline methods, we employ Litek-              HIV is associated with heterogeneous changes in the brain and
means [6] for all the k-means clustering step if it is needed in the        in cognitive function [39]. In many CNS(Central Nervous System)
implementation of the six methods listed above. We repeat cluster-          disorders, etiology is unknown. In contrast, HIV involves a known
ing for 20 times with random initialization as k-means depends on           viral etiology. Therefore it is possible to study the brain in the
initialization.                                                             early stages of injury. Studies of early HIV infection have found
   To evaluate the quality of the clusters produced by different            alterations in both structural and functional connectivity [39]. More-
approaches, we use Accuracy and Normalized Mutual Information               over, a hallmark of HIV is neuroinflammation, which is a common
(NMI) as the evaluation metrics. For each experiment, we repeat             characteristic of neurological injury from diverse causes, including
50 times and report the mean value along with standard deviation            traumatic, ischemic, developmental and neurodegenerative brain
(std) as the results.                                                       disorders. Since HIV infection is broadly relevant to many other
neurological disorders, it represents an ideal model for evaluating
the sensitivity of new frameworks for neuroimaging analysis.
   We apply the proposed MCGE framework on the multi-view
brain networks of the HIV dataset and obtain the clustering results
as well as the multi-view graph embedding for each brain network.
We further employ k-means algorithm (with k = 6) on the row
vectors of the multi-view graph embedding for each brain network,
and obtain the clustering relationship of their inner nodes, i.e., the                   (a) normal control                        (b) HIV patient
brain regions. Figure 4 shows an example of the resulting brain
region clustering map of a normal control and that of an HIV patient.     Figure 4: Comparison of the connectomes captured from the
In this figure, each node represents a brain region, and each edge        brain networks of a normal control and an HIV patient
indicates the correlation between two brain regions. Nodes of the
same color represent the brain regions that are grouped into the
same cluster by MCGE. As we can see from Figure 4, the clustering
pattern of the HIV patient is quite different from the normal control.
Nodes of the normal brain network are well grouped into several
clusters, while nodes in the HIV brain network are less coherent.
In addition, for the normal control, edges within each cluster are
much more intense than the edges across different clusters. For
example, in Figure 4(a), the pink nodes in the lower left and the                        (a) normal control                       (b) bipolar subject
pink nodes in the upper right are strongly connected with each
other. While in Figure 4(b), the corresponding nodes in the lower         Figure 5: Comparison of the connectomes captured from the
left, which are mostly marked in yellow, have very few connections        brain networks of a normal control and a bipolar subject
with those yellow nodes in the upper right. By looking into the
connections, we can find that for the normal control, there are
                                                                                   0.8                                      0.8

several pink nodes in the center of the brain which bridge the                     0.7                                      0.7
                                                                                   0.6                                      0.6

lower left part and the upper right part, while these intermediate                 0.5

                                                                                   0.4
                                                                                                                 ACC
                                                                                                                 NMI
                                                                                                                            0.5                           ACC
                                                                                                                                                          NMI
                                                                                                                            0.4

nodes in the HIV brain are clustered in blue or pink instead of the                0.3

                                                                                   0.2
                                                                                                                            0.3

                                                                                                                            0.2

same color (yellow) as the lower left part and the upper right part.               0.1

                                                                                    0
                                                                                                                            0.1

                                                                                                                             0
                                                                                     2     4    6       8   10         12     2    4     6       8   10         12
This implies that the intermediate regions are probably injured so                                  c                                        c



that they are no longer the bridges (or hubs) across other related                             (a) HIV                                 (b) Bipolar
regions. Some studies in neuroscience [12] show that the highly-
interconnected hub nodes are biologically costly due to higher blood               Figure 6: Accuracy and NMI with different c
flow or connection distances, and thus tend to be more sensitive to
injury. Our observations in Figure 4 potentially reflect this evidence.
   Then we apply the MCGE framework on the Bipolar dataset with           5.4    Parameter Sensitivity Analysis
the same steps as illustrated above for HIV dataset. The visualized       In this section, we study the sensitivity of the proposed MCGE
results of a normal control and a bipolar subject are shown in Figure     framework to the three parameters α, β, and c, and explore how
5. Similarly to the observations above, as we can see from Figure 5,      the different values for parameters would affect the performance of
the cluster information of normal control is quite different from the     MCGE in the multi-view clustering. We first look into the parameter
bipolar subject. The connectomes of the normal control are well           c, which is the dimension of the row vectors in graph embedding.
organized, while the corresponding nodes in the brain network of          Figure 6 shows the multi-view clustering performance of MCGE on
the bipolar subject spread out irregularly across different clusters.     the two datasets with the c value varying from 2 to 12. From the
We can also find that for normal control, edges within each cluster       figure, we can see that the value for c affects the performance of
are much more intense than the edges across different clusters,           MCGE in both accuracy and NMI. The highest accuracy is achieved
while this is less the case for bipolar subject. The reason behind this   when c equals to 8 for HIV dataset and the best NMI occurs at 9. For
observation is probably that the collaborative activities of different    Bipolar dataset, both the accuracy and NMI reach the peak when
brain regions of the bipolar subject are not organized in a proper        c equals to 6. The changing of accuracy and NMI with different c
order as those of normal controls are.                                    values has similar trend on the two datasets. With the increase of
   These findings indicate that our proposed MCGE framework can           the c value, the performance first keeps rising up until it reaches the
distinguish brain alterations in neurological disorders from healthy      peak, and then it starts to decline. This changing trend is reasonable
controls. It also yields new information and insights concerning          as when the dimension of graph embedding is too small, it could
network perturbations in brain injury and neuroinflammation for           not encode enough local structure information of the graph, result-
further investigation and interpretation.                                 ing in poor performance for the clustering. When the dimension
                                                                          of graph embedding is set to be a large number, it may include
                                                                          much redundant information, making it less discriminative for the
                                                                          clustering task.
                                                                                                projections using multiple views[3]. In [10], a CCA based method
                                                                                                is proposed and applied for audio-visual speaker clustering and
                                                                                                hierarchical Wikipedia document clustering. Another main cate-
                                                                                                gory of algorithms aim to integrate multiple views in the cluster-
                                                                                                ing process directly by optimizing the loss functions[3]. A typical
                                                                                                work from this category is the co-regularized multi-view spec-
                                                                                                tral clustering method proposed by [19], which is also a baseline
            (a) Accuracy on HIV                   (b) NMI on HIV                                method used in our experiment. It performs multi-view clustering
                                                                                                by co-regularizing the clustering hypotheses. In addition, matrix
                                                                                                factorization based methods also form a category of multi-view
                                           0.8

                                           0.6                                                  clustering methods[18, 21], which use constraints to push multiple
                                     NMI   0.4

                                           0.2
                                                                                                views towards consensus.
                                            0
                                            −4
                                                                                                    Graph embedding is a hot research topic in graph mining. The
                                                 −2                                      4

                                     α (log scale)
                                                      0
                                                          2            −2
                                                                            0
                                                                                   2
                                                                                                goal of graph embedding is to find low-dimensional representations
                                                              4                 β (log scale)
                                                                  −4
                                                                                                of nodes in graphs that can preserve the important structure and
           (c) Accuracy on Bipolar               (d) NMI on Bipolar                             properties of graphs [45]. It has drawn great interest from the data
                                                                                                mining community, and has been extensively studied for various
       Figure 7: Accuracy and NMI with different α, β                                           kinds of applications. In [26], a new graph embedding algorithm
                                                                                                is proposed based on Laplacian type operator on manifold, and
                                                                                                it is applied for recovering the geometry of data and extending
   Now we evaluate the sensitivity of MCGE to α and β. As illus-                                a function on new data points. Recently, a high-order proximity
trated in Equation (12), α is the weight parameter which determines                             preserved embedding method is proposed in [29]. They first de-
the extent that the local embedding structure is utilized for the                               rive a general formulation that covers multiple popular high-order
multi-view clustering task. The higher the value for α, the more                                proximity measurements, and then propose a scalable embedding
emphasis we put on the graph embeddings for multi-view clus-                                    algorithm to approximate the high-order proximity measurements.
tering modeling. Similarly, the parameter β balances how much                                       Connectome analysis is a prominent emphasis area in the field of
influence the embeddings of neighbor graphs would have on the                                   medical data mining. The "connectome", refers to the vast connec-
multi-view graph embedding of each graph. For the evaluation, we                                tivity of neural systems at different levels involving both global and
set c to be 8 and run the MCGE framework with different values                                  local structure information of the connections [17]. Connectome
of α and β. The clustering accuracy and NMI achieved at different                               analysis has been the focus of intense investigation owing to the
values of parameters for the two datasets are shown in Figure 7(a),                             tremendous potential to provide more comprehensive understand-
Figure 7(b), Figure 7(c) and Figure 7(d), respectively. As we can                               ing of normal brain function and to yield new insights concerning
see from the figures, MCGE achieves different levels of accuracy                                many different brain disorders [7, 25, 35]. Most connectome analy-
and NMI when the values of α and β vary. The highest accuracy                                   ses, however, aim to learn the structure from brain networks based
on HIV dataset is achieved when α = 103 , and β = 102 , while the                               on an individual neuroimaging modality [9, 15, 16, 20]. For exam-
best NMI on HIV is achieved at α = 103 , and β = 103 . On Bipolar                               ple, in [9], the identification of discriminative subgraph patterns is
dataset, both the highest accuracy and the best NMI are achieved                                studied on fMRI brain networks for bipolar affective disorder anal-
when α = 103 , and β = 103 . Notably, when the value for α is too                               ysis. In [23], a multi-graph clustering method is proposed based on
small, both the accuracy and NMI achieved by MCGE are quite low,                                interior-node clustering for connectome analysis in fMRI resting-
and the same situation holds for β. This is mainly because that if                              state networks. Although some recent work [5] use multi-view
we set a small value to α, little information of graph embeddings                               brain networks in connectome analysis, they focus on the group-
would be used for the multi-view clustering stage. Similarly, when                              wise functional community detection problem instead of doing
β is too small, the graph embeddings of neighbor graphs would                                   multi-view clustering of the subjects. Here, we apply the proposed
hardly influence the multi-view graph embedding stage of each                                   graph embedding based approach to facilitate the multi-view clus-
graph. On the other hand, when α and β are set to be large values,                              tering of multiple brain networks simultaneously, thus providing
the performance drops as well, as the influence imposed on those                                a more comprehensive strategy for further neurological disorder
parts is too much. Therefore, finding an optimal combination of                                 identification.
these parameter values is very important when applying MCGE
framework for multi-view clustering.
                                                                                                7   CONCLUSION
6   RELATED WORK
                                                                                                In this paper, we present MCGE, a Multi-view Clustering framework
Our work relates to several branches of studies, which include
                                                                                                with Graph Embedding, to solve multi-view clustering problem on
multi-view clustering, graph embedding and connectome analysis.
                                                                                                graph instances. MCGE first models the multi-view graph data
   Multi-view clustering is a clustering strategy for analyzing data
                                                                                                as tensors and then learns the multi-view graph embeddings via
with multiple views [3] and it has been widely studied and ap-
                                                                                                tensor factorization. We further incorporate multi-view graph em-
plied in various domains [22, 33, 34]. For example, the Canonical
                                                                                                bedding into an iterative multi-view clustering framework, jointly
Correlation Analysis (CCA) based methods focus on constructing
performing multi-view clustering and graph embedding simulta-                           [22] Chun-Ta Lu, Lifang He, Weixiang Shao, Bokai Cao, and Philip S Yu. 2017. Multi-
neously. The results of multi-view clustering are used to refine                             linear Factorization Machines for Multi-Task Multi-View Learning. In WSDM.
                                                                                        [23] Guixiang Ma, Lifang He, Bokai Cao, Jiawei Zhang, S Yu Philip, and Ann B Ragin.
the multi-view graph embeddings, in turn, the updated multi-view                             2016. Multi-graph Clustering Based on Interior-Node Topology with Applica-
graph embedding results are used to improve the multi-view clus-                             tions to Brain Networks. In Joint European Conference on Machine Learning and
                                                                                             Knowledge Discovery in Databases. Springer, 476–492.
tering. By updating the clustering results and graph embeddings                         [24] Guixiang Ma, Lifang He, Chun-Ta Lu, Philip S Yu, Linlin Shen, and Ann B Ragin.
iteratively, the proposed MCGE framework will result in a better                             2016. Spatio-temporal tensor analysis for whole-brain fMRI classification. In
multi-view clustering solution. We apply our MCGE framework                                  Proceedings of the 2016 SIAM International Conference on Data Mining. SIAM,
                                                                                             819–827.
for unsupervised multi-view connectome analysis on HIV-induced                          [25] Guixiang Ma, Chun-Ta Lu, Lifang He, S Yu Philip, and B Ragin Ann. 2017. Multi-
brain alterations and bipolar affective disorder. Extensive exper-                           view Graph Embedding with Hub Detection for Brain Network Analysis. In
imental results on real multi-view HIV brain network data and                                ICDM.
                                                                                        [26] Saman Mousazadeh and Israel Cohen. 2015. Embedding and function extension
Bipolar brain network data show the effectiveness of MCGE for                                on directed graph. Signal Processing 111 (2015), 137–149.
multi-view clustering in connectome analysis.                                           [27] Feiping Nie, Jing Li, Xuelong Li, and others. 2016. Parameter-Free Auto-Weighted
                                                                                             Multiple Graph Learning: A Framework for Multiview Clustering and Semi-
ACKNOWLEDGMENTS                                                                              Supervised Classification. International Joint Conferences on Artificial Intelli-
                                                                                             gence.
This work is supported in part by NSF through grants IIS-1526499,                       [28] Feiping Nie, Zinan Zeng, Ivor W Tsang, Dong Xu, and Changshui Zhang. 2011.
and CNS-1626432, and NSFC 61672313, and NSFC 61503253.                                       Spectral embedded clustering: A framework for in-sample and out-of-sample
                                                                                             spectral clustering. IEEE Trans on Neural Networks 22, 11 (2011), 1796–1808.
                                                                                        [29] M Ou, Peng Cui, Jian Pei, Z Zhang, and W Zhu. 2016. Asymmetric transitivity
REFERENCES                                                                                   preserving graph embedding. In SIGKDD.
 [1] P-A Absil, Robert Mahony, and Rodolphe Sepulchre. 2009. Optimization algo-         [30] Ann B Ragin, Hongyan Du, Renee Ochs, Ying Wu, Christina L Sammet, Alfred
     rithms on matrix manifolds. Princeton University Press.                                 Shoukry, and Leon G Epstein. 2012. Structural brain alterations can be detected
 [2] Mikhail Belkin and Partha Niyogi. 2001. Laplacian eigenmaps and spectral                early in HIV infection. Neurology 79, 24 (2012), 2328–2334.
     techniques for embedding and clustering. In NIPS.                                  [31] Sam T Roweis and Lawrence K Saul. 2000. Nonlinear dimensionality reduction
 [3] Steffen Bickel and Tobias Scheffer. 2004. Multi-View Clustering. In ICDM.               by locally linear embedding. Science 290, 5500 (2000), 2323–2326.
 [4] Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. 2011.   [32] Lawrence K Saul and Sam T Roweis. 2000. An introduction to locally linear
     Distributed optimization and statistical learning via the alternating direction         embedding. http://www. cs. toronto. edu/˜ roweis/lle/publications. html (2000).
     method of multipliers. Foundations and Trends® in Machine Learning 3, 1 (2011),    [33] Weixiang Shao, Lifang He, Chun-Ta Lu, Xiaokai Wei, and S Yu Philip. 2016. Online
     1–122.                                                                                  unsupervised multi-view feature selection. In ICDM.
 [5] Nathan D Cahill, Harmeet Singh, Chao Zhang, Daryl A Corcoran, Alison M             [34] Weixiang Shao, Lifang He, and S Yu Philip. 2015. Clustering on multi-source
     Prengaman, Paul S Wenger, John F Hamilton, Peter Bajorski, and Andrew M                 incomplete data via tensor modeling and factorization. In PAKDD.
     Michael. 2016. Multiple-View Spectral Clustering for Group-wise Functional         [35] Olaf Sporns, Giulio Tononi, and Rolf Kötter. 2005. The human connectome: a
     Community Detection. arXiv preprint arXiv:1611.06981 (2016).                            structural description of the human brain. PLoS Comput Biol 1, 4 (2005), e42.
 [6] D Cai. 2011. Litekmeans: the fastest matlab implementation of kmeans. Software     [36] Nathalie Tzourio-Mazoyer, Brigitte Landeau, Dimitri Papathanassiou, Fabrice
     available at: http://www. zjucadcg. cn/dengcai/Data/Clustering. html (2011).            Crivello, Olivier Etard, Nicolas Delcroix, Bernard Mazoyer, and Marc Joliot. 2002.
 [7] Bokai Cao, Lifang He, Xiaokai Wei, Mengqi Xing, Philip S. Yu, Heide Klumpp,             Automated anatomical labeling of activations in SPM using a macroscopic anatom-
     and Alex D. Leow. 2017. t-BNE: Tensor-based Brain Network Embedding. In                 ical parcellation of the MNI MRI single-subject brain. Neuroimage 15, 1 (2002),
     SDM.                                                                                    273–289.
 [8] Bokai Cao, Xiangnan Kong, Jingyuan Zhang, S Yu Philip, and Ann B Ragin.            [37] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M
     2015. Identifying HIV-induced subgraph patterns in brain networks with side             Borgwardt. 2010. Graph kernels. Journal of Machine Learning Research 11, Apr
     information. Brain informatics 2, 4 (2015), 211–223.                                    (2010), 1201–1242.
 [9] Bokai Cao, Liang Zhan, Xiangnan Kong, Philip S Yu, Nathalie Vizueta, Lori L        [38] Ulrike Von Luxburg. 2007. A tutorial on spectral clustering. Statistics and
     Altshuler, and Alex D Leow. 2015. Identification of discriminative subgraph             computing 17, 4 (2007), 395–416.
     patterns in fMRI brain networks in bipolar affective disorder. In BIH.             [39] Xue Wang, Paul Foryt, Renee Ochs, Jae-Hoon Chung, Ying Wu, Todd Parrish,
[10] Kamalika Chaudhuri, Sham M Kakade, Karen Livescu, and Karthik Sridharan.                and Ann B Ragin. 2011. Abnormalities in resting-state functional connectivity in
     2009. Multi-view clustering via canonical correlation analysis. In ICML.                early human immunodeficiency virus infection. Brain connectivity 1, 3 (2011),
[11] Rodrigo Cilla Ugarte. 2012. Action recognition in visual sensor networks: a data        207–217.
     fusion perspective. (2012).                                                        [40] Yichen Wang, Robert Chen, Joydeep Ghosh, Joshua C Denny, Abel Kho, You
[12] Nicolas A Crossley, Andrea Mechelli, Jessica Scott, Francesco Carletti, Peter T         Chen, Bradley A Malin, and Jimeng Sun. 2015. Rubik: Knowledge guided tensor
     Fox, Philip McGuire, and Edward T Bullmore. 2014. The hubs of the human                 factorization and completion for health data analytics. In SIGKDD.
     connectome are generally implicated in the anatomy of brain disorders. Brain       [41] Zaiwen Wen and Wotao Yin. 2013. A feasible method for optimization with
     137, 8 (2014), 2382–2395.                                                               orthogonality constraints. Mathematical Programming 142, 1-2 (2013), 397–434.
[13] Yun Fu and Yunqian Ma. 2012. Graph embedding for pattern analysis. Springer        [42] Susan Whitfield-Gabrieli and Alfonso Nieto-Castanon. 2012. Conn: a functional
     Science & Business Media.                                                               connectivity toolbox for correlated and anticorrelated brain networks. Brain
[14] Jing Gao, Nan Du, Wei Fan, Deepak Turaga, Srinivasan Parthasarathy, and Jiawei          connectivity 2, 3 (2012), 125–141.
     Han. 2013. A multi-graph spectral framework for mining multi-source anomalies.     [43] Tian Xia, Dacheng Tao, Tao Mei, and Yongdong Zhang. 2010. Multiview spec-
     In Graph Embedding for Pattern Analysis. Springer, 205–227.                             tral embedding. IEEE Transactions on Systems, Man, and Cybernetics, Part B
[15] Lifang He, Xiangnan Kong, Philip S Yu, Xiaowei Yang, Ann B Ragin, and Zhifeng           (Cybernetics) 40, 6 (2010), 1438–1446.
     Hao. 2014. Dusk: A dual structure-preserving kernel for supervised tensor          [44] Bo Xie, Yang Mu, Dacheng Tao, and Kaiqi Huang. 2011. m-SNE: Multiview sto-
     learning with applications to neuroimages. In SDM.                                      chastic neighbor embedding. IEEE Transactions on Systems, Man, and Cybernetics,
[16] Lifang He, Chun-Ta Lu, Guixiang Ma, Shen Wang, Linlin Shen, S Yu Philip, and            Part B (Cybernetics) 41, 4 (2011), 1088–1096.
     Ann B Ragin. 2017. Kernelized support tensor machines. In ICML.                    [45] Shuicheng Yan, Dong Xu, Benyu Zhang, Hong-Jiang Zhang, Qiang Yang, and
[17] Marcus Kaiser. 2011. A tutorial in connectome analysis: topological and spatial         Stephen Lin. 2007. Graph embedding and extensions: a general framework for
     features of brain networks. Neuroimage 57, 3 (2011), 892–907.                           dimensionality reduction. IEEE transactions on pattern analysis and machine
[18] Mahdi M Kalayeh, Haroon Idrees, and Mubarak Shah. 2014. Nmf-knn: Image                  intelligence 29, 1 (2007), 40–51.
     annotation using weighted multi-view non-negative matrix factorization. In         [46] Ming Yin, Shengli Xie, Yi Guo, and others. 2016. Low-rank Multi-view Clustering
     CVPR.                                                                                   in Third-Order Tensor Space. arXiv preprint arXiv:1608.08336 (2016).
[19] Abhishek Kumar, Piyush Rai, and Hal Daume. 2011. Co-regularized multi-view         [47] Jingyuan Zhang, Bokai Cao, Sihong Xie, Chun-Ta Lu, Philip S Yu, and Ann B
     spectral clustering. In NIPS.                                                           Ragin. 2016. Identifying connectivity patterns for brain diseases via multi-side-
[20] Chia-Tung Kuo, Xiang Wang, Peter Walker, Owen Carmichael, Jieping Ye, and               view guided deep architectures. In SDM.
     Ian Davidson. 2015. Unified and contrasting cuts in multiple graphs: application   [48] Lefei Zhang, Qian Zhang, Liangpei Zhang, Dacheng Tao, Xin Huang, and Bo
     to medical imaging segmentation. In SIGKDD.                                             Du. 2015. Ensemble manifold regularized sparse low-rank approximation for
[21] Jialu Liu, Chi Wang, Jing Gao, and Jiawei Han. 2013. Multi-view clustering via          multiview feature embedding. Pattern Recognition 48, 10 (2015), 3102–3112.
     joint nonnegative matrix factorization. In SDM.

