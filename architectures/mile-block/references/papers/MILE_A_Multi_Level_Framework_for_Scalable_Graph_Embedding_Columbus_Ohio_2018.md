# MILE A Multi Level Framework for Scalable Graph Embedding Columbus Ohio 2018

> Source: `MILE_A_Multi_Level_Framework_for_Scalable_Graph_Embedding_Columbus_Ohio_2018.pdf`

---

                                         MILE: A Multi-Level Framework for Scalable Graph Embedding
                                                       Jiongqian Liang                                   Saket Gurukar                      Srinivasan Parthasarathy
                                                  The Ohio State University                        The Ohio State University                  The Ohio State University
                                                       Columbus, Ohio                                  Columbus, Ohio                              Columbus, Ohio
                                                     liang.420@osu.edu                                gurukar.1@osu.edu                        srini@cse.ohio-state.edu
                                         ABSTRACT                                                                    Node2Vec [9], require a large amount of CPU time to generate
                                         Recently there has been a surge of interest in designing graph              a sufficient number of walks and train the embedding model.
                                         embedding methods. Few, if any, can scale to a large-sized                  As another example, embedding methods based on matrix
                                                                                                                     factorization, including GraRep [3] and NetMF [21], requires




arXiv:1802.09612v1 [cs.AI] 26 Feb 2018
                                         graph with millions of nodes due to both computational com-
                                         plexity and memory requirements. In this paper, we relax                    constructing an enormous objective matrix (usually much
                                         this limitation by introducing the MultI-Level Embedding                    denser than adjacency matrix) on which matrix factorization
                                         (MILE) framework – a generic methodology allowing contem-                   is performed. Even a medium-size graph with 100K nodes
                                         porary graph embedding methods to scale to large graphs.                    can easily require hundreds of GB of memory using those
                                         MILE repeatedly coarsens the graph into smaller ones us-                    methods. On the other hand, many graph datasets in the
                                         ing a hybrid matching technique to maintain the backbone                    real world tend to be large-scale with millions or even billions
                                         structure of the graph. It then applies existing embedding                  of nodes. For instance, Google knowledge graph covers over
                                         methods on the coarsest graph and refines the embeddings                    570M entities while Facebook friendship graph contains at
                                         to the original graph through a novel graph convolution neu-                least 1.39B user dataset with over 1 trillion connections [6].
                                         ral network that it learns. The proposed MILE framework                     To the best of our knowledge, none of the existing efforts
                                         is agnostic to the underlying graph embedding techniques                    examines how to scale up graph embedding in a generic way.
                                         and can be applied to many existing graph embedding meth-                   We make the first attempt to close this gap. We are also
                                         ods without modifying them. We employ our framework on                      interested in the related question of whether the quality of
                                         several popular graph embedding techniques and conduct                      such embeddings can be improved along the way. Specifically,
                                         embedding for real-world graphs. Experimental results on                    we ask:
                                         five large-scale datasets demonstrate that MILE significantly                 (1) Can we scale up the existing embedding techniques
                                         boosts the speed (order of magnitude) of graph embedding                          in an agnostic manner so that they can be directly
                                         while also often generating embeddings of better quality for                      applied to larger datasets?
                                         the task of node classification. MILE can comfortably scale                   (2) Can the quality of such embedding methods be strength-
                                         to a graph with 9 million nodes and 40 million edges, on                          ened by incorporating the holistic view of the graph?
                                         which existing methods run out of memory or take too long
                                         to compute on a modern workstation.                                            To tackle these problems, we propose a MultI-Level Embed-
                                                                                                                     ding (MILE) framework for graph embedding. Our approach
                                         ACM Reference Format:
                                         Jiongqian Liang, Saket Gurukar, and Srinivasan Parthasarathy.
                                                                                                                     relies on a three-step process: first, we repeatedly coarsen
                                         2018. MILE: A Multi-Level Framework for Scalable Graph Embed-               the original graph into smaller ones by employing a hybrid
                                         ding. In Proceedings of ACM conference (Conference’18). ACM,                matching strategy; second, we compute the embeddings on
                                         New York, NY, USA, 11 pages. https://doi.org/xxx                            the coarsest graph using an existing embedding mechanism -
                                                                                                                     note that graph embedding on the coarsest graph is inexpen-
                                         1    INTRODUCTION                                                           sive to compute and utilizes far less memory, and moreover
                                                                                                                     intuitively can capture the global structure of the original
                                         In recent years, graph embedding has attracted much inter-
                                                                                                                     graph [14, 23]; and third, we propose a novel refinement
                                         est due to its broad applicability for tasks such as vertex
                                                                                                                     model based on learning a graph convolution network to
                                         classification [20] and full network visualization [24]. How-
                                                                                                                     refine the embeddings from the coarsest graph to the original
                                         ever, such methods rarely scale to large datasets (e.g., graphs
                                                                                                                     graph – learning a graph convolution network allows us to
                                         with over 1 million nodes) since they are computationally
                                                                                                                     compute a refinement procedure that levers the dependencies
                                         expensive and often memory intensive. For example, random-
                                                                                                                     inherent to the graph structure and the embedding method
                                         walk-based embedding techniques, such as DeepWalk [20] and
                                                                                                                     of choice. To train this model for embeddings refinement, we
                                                                                                                     design a particular learning task on the coarsest graph, which
                                         Permission to make digital or hard copies of part or all of this work
                                         for personal or classroom use is granted without fee provided that          is efficient to perform. To summarize, we find that:
                                         copies are not made or distributed for profit or commercial advantage       1) MILE is generalizable. Our MILE framework is agnostic
                                         and that copies bear this notice and the full citation on the first page.
                                         Copyrights for third-party components of this work must be honored.
                                                                                                                     to the underlying graph embedding techniques and treats
                                         For all other uses, contact the owner/author(s).                            them as black boxes. We report results on DeepWalk[20],
                                         Conference’18, Jan 2018, DC, Washington USA                                 Node2Vec[9], GraRep[3], and NetMF[21].
                                         © 2018 Copyright held by the owner/author(s).                               2) MILE is scalable. We show that the proposed framework
                                         ACM ISBN 978-x-xxxx-xxxx-x/YY/MM.
                                         https://doi.org/xxx                                                         can significantly improve the scalability of the embedding
Conference’18, Jan 2018, DC, Washington USA                         Jiongqian Liang, Saket Gurukar, and Srinivasan Parthasarathy


methods (up to 30-fold), by reducing the running time and                                  Input graph 𝒢$
the memory consumption.
3) MILE generates high-quality embeddings. In many cases,                                                Final Embedding ℰ$

we find that the quality of embeddings improves by levering
MILE (in some cases is in excess of 10%).                                                           𝒢%           ℰ%
4) MILE’s ability to learn a data- and embedding- sensitive
refinement procedure is key to its effectiveness. Other design
choices such as the hybrid coarsening strategy also enable                                                  𝒢"
                                                                                     Coarsening                          Refining
MILE to produce quality embeddings in a scalable fashion.

                                                                                                  Base Embedding ℰ"
2   RELATED WORK                                                        Figure 1: An overview of the multi-level embedding framework.
Graph Embedding: Many techniques for graph or network em-
bedding have been proposed in recent years. DeepWalk and            some ideas at a conceptual level with such efforts, the objec-
Node2Vec generate truncated random walks on graphs and              tives are distinct in that we focus on graph embeddings while
apply the Skip Gram by treating the walks as sentences [9, 20].     these methods work on graph partitioning and community
LINE learns the node embeddings by preserving the first-            discovery.
order and second-order proximities [24]. Following LINE,
SDNE leverages deep neural networks to capture the highly           3     PROBLEM FORMULATION
non-linear structure [25]. Other methods construct a particu-
                                                                    Let 𝒢 = (𝑉, 𝐸) be the input graph (weighted or unweighted),
lar objective matrix and use matrix factorization techniques
                                                                    where 𝑉 and 𝐸 are respectively the node set and edge set.
to generate embeddings, e.g., GraRep [3] and NetMF [21].
                                                                    Let 𝐴 be the |𝑉 | × |𝑉 | adjacency matrix of the graph with
This also led to the proliferation of network embedding meth-
                                                                    each entry 𝐴(𝑢, 𝑣) denoting the weight of the edge between
ods for information-rich graphs, including heterogeneous infor-
                                                                    node 𝑢 and 𝑣. Without ambiguity, we refer to the graph as
mation networks [4, 8] and attributed graphs [15, 17, 19, 26].
                                                                    𝒢 and 𝐴 interchangeably in the rest of the paper. We also
On the other hand, there are very few efforts, focusing on the
                                                                    assume 𝒢 is undirected, though our problem can be easily
scalability of network embedding [1, 13, 27]. Such efforts are
                                                                    extended to directed graph. We first define graph embedding:
specific to a particular embedding strategy and do not offer
a generic strategy to scale other embedding techniques. Yet,        Definition 3.1. Graph Embedding Given a graph 𝒢 = (𝑉, 𝐸)
they still cannot scale to very large graphs. Incidentally, these   and a pre-defined dimensionality 𝑑 (𝑑 ≪ |𝑉 |), the problem
efforts at scalability are actually orthogonal to our strategy      of graph embedding is to learn a 𝑑-dimension vector repre-
and can potentially be employed along with our efforts to           sentation for each node in graph 𝒢 so that graph properties
afford even greater speedup.                                        are best preserved.
   Tangentially related to our work are the recent efforts that
develop embedding strategies on multi-layered networks [16,            Following this, a graph embedding method is essentially
18]. Distinct from our effort, the networks they studied con-       a mapping function 𝑓 : R|𝑉 |×|𝑉 | ↦→ R|𝑉 |×𝑑 , whose input
tain multiple layers in nature with a hierarchical structure.       is the adjacency matrix 𝐴 (or 𝒢) and output is a lower
The closest work to this paper is the very recently proposed        dimension matrix. Motivated by the fact that the majority of
HARP [5], which proposes a hierarchical paradigm for graph          graph embedding methods cannot scale to large datasets, we
embedding based on iterative learning methods (e.g., Deep-          seek to speed up existing graph embedding methods without
Walk and Node2Vec). However, HARP focuses on improving              sacrificing quality. We formulate the problem as:
the quality of embeddings by using the learned embeddings           Given a graph 𝒢 = (𝑉, 𝐸) and a graph embedding method 𝑓 (·),
from the previous level as the initialized embeddings for the       we aim to realize a strengthened graph embedding method
next level, which introduces a huge computational overhead.         𝑓^(·) so that it is more scalable than 𝑓 (·) while generating
Moreover, HARP cannot be easily extended to other graph             embeddings of comparable or even better quality.
embedding techniques (e.g., GraRep and NetMF) since it                 We refer to the process of applying 𝑓 (·) on a graph as base
needs to modify the embedding methods to preset their ini-          embedding, where 𝑓 (·) is called the base embedding method.
tialized embeddings. In this paper, we focus on designing a
general purpose framework to scale up embedding methods             4     METHODOLOGY
treating them as black boxes.                                       To address the aforementioned problem, we propose a scalable
   Multi-level Community Detection: The multi-level approach        MultI-Level Embedding (MILE) framework. Our framework
has been widely studied for efficient community detection [2,       is similar to Metis, MLR-MCL, and Graculus [7, 14, 23],
7, 14, 22, 23]. The key idea of these multi-level algorithms        which are popular multi-level graph clustering algorithm.
is to coarsen the original graph into a much smaller one,           Figure 1 shows the overview of our MILE framework, which
which is then partitioned into clusters. The partitions are         contains three key phases: graph coarsening, base embedding,
then recovered from the coarse-grained graph to the original        and embeddings refining. On the whole, we reduce the size
graph in a recursive manner. While our framework shares             of the graph through repeated coarsening and run graph
MILE: A Multi-Level Framework for Scalable Graph Embedding                                              Conference’18, Jan 2018, DC, Washington USA



                                                                                                              A   B   C   D   E                 A   BC DE
  D           E                    DE                                  DE                  DE
                                                                                                                                                            A
                                                                   2
                                                                                                                                                            B
   1          1                    2                           3∗ 2                       2
                      SEM                  Normalization                           NHEM                                                                     C
        A                              A                               A                      A                                                             D
                                                           1                   1
   1           1               1           1           3∗ 2
                                                                                                                                                            E
                                                                              3∗ 2        2
                                                                        1
                                                                                                                                     0 2        2
                                                                       2∗ 2                       2                     '
   B              C            B           C                   B              C            BC                     𝐴" = 𝑀%," 𝐴%𝑀%," = 2 2        0
         1                             1                                                                                             2 0        0
                          (a) Using SEM and NHEM for graph coarsening                                         (b) Adjacency matrix and matching matrix

Figure 2: Toy example for illustrating graph coarsening1 . (a) shows the process of applying Structural Equivalence Matching (SEM) and
Normalized Heavy Edge Matching (NHEM) for graph coarsening. (b) presents the adjacency matrix 𝐴0 of the input graph, the matching matrix
𝑀0,1 corresponding to the SEM and NHEM matchings, and the derivation of the adjacency matrix 𝐴1 of the coarsened graph using Eq. 2.


         Symbol       Definition                                                             This can be proved by reasoning on the fact that the two
         𝒢𝑖           the graph after 𝑖 iterations of coarsening                          nodes are non-distinguishable and interchangeable on 𝒢 if
         𝑉𝑖 , 𝐸𝑖      vertex set, edge set of 𝒢𝑖                                          they share the same set of the neighborhoods (details of proof
         𝐴𝑖 , 𝐷𝑖      the adjacency and degree matrix of 𝒢𝑖
                                                                                          omitted due to the limit of space). Base on Theorem 1, we
         𝑑            dimensionality of the embeddings
         𝑚            the total number of coarsening levels                               define a structural equivalence matching as a set of nodes that
         𝑓 (·)        the base embedding method applicable on 𝒢𝑖                          are structurally equivalent to each other. For the example
         ℰ𝑖           the embeddings of nodes in 𝒢𝑖                                       in Figure 2a, nodes 𝐷 and 𝐸 are considered a structural
         𝑀𝑖,𝑖+1       the matching matrix from 𝒢𝑖 to 𝒢𝑖+1                                 equivalent matching.
         ℛ(·)         the embeddings refinement model
                      # layers in the graph convolution network
                                                                                            4.1.2 Normalized Heavy Edge Matching (NHEM). Heavy
         𝑙
                       Table 1: Table of notations                                        edge matching is a popular matching method for graph coars-
                                                                                          ening [14]. For an unmatched node 𝑢 in 𝒢𝑖 , its heavy edge
embedding on the coarsest graph, after which we perform                                   matching is a pair of vertices (𝑢, 𝑣) such that the weight of
embeddings refinement to recover the embeddings on the                                    the edge between 𝑢 and 𝑣 is the largest. In this paper, we
original graph. We summarize some important notations in                                  propose to normalize the edge weights when applying heavy
Table 1 and describe our framework in detail below.                                       edge matching using the formula as follows

4.1 Graph Coarsening                                                                                    𝑊𝑖 (𝑢, 𝑣) = √︀
                                                                                                                          𝐴𝑖 (𝑢, 𝑣)
                                                                                                                                            .               (1)
In this phase, the input graph 𝒢 (or 𝒢0 ) is repeatedly coars-                                                        𝐷𝑖 (𝑢, 𝑢) · 𝐷𝑖 (𝑣, 𝑣)
ened into a series smaller graphs 𝒢1 , 𝒢2 , ..., 𝒢𝑚 such that
|𝑉0 | > |𝑉1 | > ... > |𝑉𝑚 |. In order to coarsen a graph from                             In Eq. 1, the weight of an edge is normalized by the degree
𝒢𝑖 to 𝒢𝑖+1 , multiple nodes in 𝒢𝑖 are collapsed to form super-                            of the two vertices on which the edge is incident. Intuitively,
nodes in 𝒢𝑖+1 , and the edges incident on a super-node are                                it penalizes the weights of edges connected with high-degree
the union of the edges on the original nodes in 𝒢𝑖 [14]. Here                             nodes. For the example in Figure 2a, node 𝐵 is equally
the set of nodes forming a super-node is called a matching.                               likely to be matched with node 𝐴 and node 𝐶 without edge
The key part of this step is to design a matching approach                                weight normalization. With normalization, node 𝐵 will be
that can efficiently coarsen the graph while retaining the                                matched with 𝐶, which is a better matching since 𝐵 is more
global structure. In this paper, we propose a hybrid matching                             structurally similar to 𝐶. As we will show in Sec. 4.3, this
technique containing two matching strategies.                                             normalization is tightly connected with the graph convolution
                                                                                          kernel.
   4.1.1 Structural Equivalence Matching (SEM). Given two
vertices 𝑢 and 𝑣 in an unweighted graph 𝒢, we call they are                                  4.1.3 A Hybrid Matching Method. In this paper, we use a
structurally equivalent if they are incident on the same set of                           hybrid of two matching methods above for graph coarsening.
neighborhoods.                                                                            To construct 𝒢𝑖+1 from 𝒢𝑖 , we first find out all the structural
                                                                                          equivalence matching (SEM) ℳ1 , where 𝒢𝑖 is treated as an
   Theorem 1. If two vertices 𝑢 and 𝑣 in an unweighted                                    unweighted graph. This is followed by the searching of the
graph 𝒢 are structurally equivalent, then their node embed-                               normalized heavy edge matching (NHEM) ℳ2 on 𝒢𝑖 . Nodes
dings derived from 𝒢 will be identical.                                                   in each matching are then collapsed into a super-node in
                                                                                          𝒢𝑖+1 . Note that some nodes might not be matched at all and
    1
      We follow the strategy in existing work [23] for weighting self-loops
                                                                                          they will be directly copied to 𝒢𝑖+1 . Figure 2a provides a toy
(the weight of the self-loop on node 𝐵𝐶 is 2 instead of 1).                               example for illustrating the process.
Conference’18, Jan 2018, DC, Washington USA                          Jiongqian Liang, Saket Gurukar, and Srinivasan Parthasarathy


   Formally, we build the adjacency matrix 𝐴𝑖+1 of 𝒢𝑖+1
through matrix operations. To this end, we define the match-
ing matrix storing the matching information from graph 𝒢𝑖
to 𝒢𝑖+1 as a binary matrix 𝑀𝑖,𝑖+1 ∈ {0, 1}|𝑉𝑖 |×|𝑉𝑖+1 | . The
𝑟-th row and 𝑐-th column of 𝑀𝑖,𝑖+1 is set to 1 if node 𝑟 in 𝒢𝑖
will be collapsed to super-node 𝑐 in 𝒢𝑖+1 , and is set to 0 if
otherwise. Each column of 𝑀𝑖,𝑖+1 represents a matching with
the 1s representing the nodes in it. Each unmatched vertex
appears as an individual column in 𝑀𝑖,𝑖+1 with merely one
entry set to 1. For the toy example in Figure 2, matching
matrix 𝑀0,1 of dimension 5 × 3 indicates the mapping infor-          Figure 3: Architecture of the embeddings refinement model. The
mation from the original graph to the coarsened graph. In            input layer is the embeddings ℰ𝑖+1 of the coarsened graph 𝒢𝑖+1 . The
particular, 2nd row and 3rd row means node 𝐵 and node 𝐶              projection layer computes the projected embeddings ℰ𝑖𝑝 based on the
form a matching and are mapped to super-node 𝐵𝐶 in the               matching matrix 𝑀𝑖,𝑖+1 using Eq. 4. Following this, the projected
coarsened graph (similar for 4th and 5th row). Following this        embeddings go through 𝑙 graph convolution layers and output the
formulation, we construct the adjacency matrix of 𝒢𝑖+1 by            refined embeddings ℰ𝑖 of graph 𝒢𝑖 at the end. Note the model pa-
using                                                                rameters Θ (𝑘) (𝑘 = 1...𝑙) are shared among all the refinement steps
                                                                     (𝒢𝑖+1 to 𝒢𝑖 , where 𝑖 = 𝑚 − 1...0).
                             𝑇
                  𝐴𝑖+1 = 𝑀𝑖,𝑖+1  𝐴𝑖 𝑀𝑖,𝑖+1 .               (2)
   Algorithm 1 summarizes the steps of graph coarsening. For         In this paper, we use a wide range of popular embedding
each iteration of coarsening, SEM is generated followed by           methods for base embedding, which includes DeepWalk [20],
NHEM (line 2-9). A key part of NHEM is to visit the vertices         Node2Vec [9], GraRep [3], and NetMF [21].
following the ascending order of the number of neighbors
(line 5). This is important to ensure most vertices can be
                                                                     4.3    Embeddings Refinement
matched and the graph can be coarsened significantly. The
intuition here is vertices with a smaller number of neighbors        The final phase of MILE is the embeddings refinement phase.
have limited choice of finding a match and should be given           Given a series of coarsened graph 𝒢0 , 𝒢1 , 𝒢2 , ..., 𝒢𝑚 , their cor-
higher priority for matching (otherwise, once their neighbors        responding matching matrix 𝑀0,1 , 𝑀1,2 , ..., 𝑀𝑚−1,𝑚 , and
are matched by others, these vertices cannot be matched).            the node embeddings ℰ𝑚 on 𝒢𝑚 , we seek to develop an ap-
                                                                     proach to derive the node embeddings of 𝒢0 from 𝒢𝑚 . To this
Algorithm 1 Graph Coarsening
                                                                     end, we first study an easier subtask: given a graph 𝒢𝑖 , its
Input: A input graph 𝒢0 , and # levels for coarsening 𝑚.             coarsened graph 𝒢𝑖+1 , the matching matrix 𝑀𝑖,𝑖+1 and the
Output: Graph 𝒢𝑖 and matching matrix 𝑀𝑖−1,𝑖 , ∀𝑖 ∈ [1, 𝑚].
                                                                     node embeddings ℰ𝑖+1 on 𝒢𝑖+1 , how to infer the embeddings
 1: for 𝑖 = 1...𝑚 do
                                                                     ℰ𝑖 on graph 𝒢𝑖 . Once we solved this subtask, we can then
 2:     ℳ1 ← all the structural equivalence matching in 𝒢𝑖−1 .
                                                                     iteratively apply the technique on each pair of consecutive
 3:     Mark vertices in ℳ1 as matched.
 4:     ℳ2 = ∅.          ◁ storing normalized heavy edge matching
                                                                     graphs from 𝒢𝑚 to 𝒢0 and eventually derive the node embed-
 5:     Sort 𝑉𝑖−1 by the number of neighbors in ascending order.     dings on 𝒢0 . In this work, we propose to use a graph-based
 6:     for 𝑣 ∈ 𝑉𝑖−1 do                                              neural network model to perform embeddings refinement.
 7:         if 𝑣 and 𝑢 are not matched and 𝑢 ∈ Neighbors(𝑣) then
 8:             (𝑣, 𝑢) ← the normalized heavy edge matching for 𝑣.     4.3.1 Graph Convolution Network for Embeddings Refine-
 9:             ℳ2 = ℳ2 ∪ (𝑣, 𝑢), and mark both as matched.          ment. Since we know the matching information between the
10:     Compute matching matrix 𝑀𝑖−1,𝑖 based on ℳ1 and ℳ2 .          two consecutive graphs 𝒢𝑖 and 𝒢𝑖+1 , we can easily project
11:     Derive the adjacency matrix 𝐴𝑖 for 𝒢𝑖 using Eq. 2.           the node embeddings from the coarse-grained graph 𝒢𝑖+1 to
12: Return graph 𝒢𝑖 and matching matrix 𝑀𝑖−1,𝑖 , ∀𝑖 ∈ [1, 𝑚].        the fine-grained graph 𝒢𝑖 using

                                                                                                                                      (4)
                                                                                               𝑝
                                                                                              ℰ𝑖 = 𝑀𝑖,𝑖+1 ℰ𝑖+1
4.2   Base Embedding on Coarsened Graph
The size of the graph reduces drastically after each iteration       In this case, embedding of a super-node is directly copied to
                                                                     its original node(s). We call ℰ𝑖 the projected embeddings from
                                                                                                    𝑝
of coarsening, halving the size of the graph in the best case.
We coarsen the graph for 𝑚 iterations and apply the graph            𝒢𝑖+1 to 𝒢𝑖 , or simply projected embeddings without ambiguity.
embedding method 𝑓 (·) on the coarsest graph 𝒢𝑚 . Denoting           While this way of simple projection maintain some informa-
the embeddings on 𝒢𝑚 as ℰ𝑚 , we have                                 tion of node embeddings, it has obvious limitations that
                                                                     nodes will share the same embeddings if they are matched
                         ℰ𝑚 = 𝑓 (𝒢𝑚 ).                        (3)    and collapsed into a super-node during the coarsening phase.
   Since our framework is agnostic to the adopted graph              This problem will be more serious when the embedding re-
embedding method, we can use any graph embedding algo-               finement is performed iteratively from 𝒢𝑚 , ..., 𝒢0 . To address
rithm for base embedding. Therefore, many existing graph             this issue, we propose to use a graph convolution network for
embedding methods can be scaled up using this framework.             embedding refinement. Specifically, we design a graph-based
MILE: A Multi-Level Framework for Scalable Graph Embedding                               Conference’18, Jan 2018, DC, Washington USA


neural network model ℰ𝑖 = ℛ(ℰ𝑖 , 𝐴𝑖 ), which derives the em-              predicted ones as the loss function for training. We propose to
                                       𝑝

beddings ℰ𝑖 on graph 𝒢𝑖 based on the projected embeddings                 learn Θ (𝑘) on the coarsest graph and reuse them across all the
ℰ𝑖 and the graph adjacency matrix 𝐴𝑖 .
 𝑝
                                                                          levels for refinement. Specifically, given the coarsest graph 𝒢𝑚 ,
  Given a graph 𝐺 with adjacency matrix 𝐴 ∈ R|𝑉 |×|𝑉 | , we               we first perform base embedding to get ℰ𝑚 = 𝑓 (𝒢𝑚 ), which
consider the graph convolution [12] of 𝑑-channel input signals            serves as the “ground truth” for embeddings refinement. We
𝑋 with filters 𝑔 on 𝐺 as                                                  then further coarsen graph 𝒢𝑚 into graph 𝒢𝑚+1 and perform
                                                                          another base embedding: ℰ𝑚+1 = 𝑓 (𝒢𝑚+1 ). Following the em-
                      𝑋 *𝐺 𝑔 = 𝑈 𝜃𝑔 𝑈 𝑇 𝑋.                          (5)   bedding refinement procedures, we can predict the refined em-
Here, 𝜃𝑔 = diag(𝜃) is parameterized by spectral multipliers               beddings on 𝒢𝑚 as ℛ(ℰ𝑚 , 𝐴𝑚 ) = 𝐻 (𝑙) (𝑀𝑚,𝑚+1 ℰ𝑚+1 , 𝐴𝑚 ).
                                                                                                     𝑝

𝜃 ∈ R|𝑉 | in the Fourier domain, 𝑈 is the matrix of eigenvec-             Considering the “ground truth” from base embedding and in-
tors of the normalized graph Laplacian 𝐿 = 𝐼 − 𝐷∑︀ − 12     1
                                                        𝐴𝐷− 2 ,           ferred embeddings from the refinement model, we can define
where 𝐷 is a diagonal matrix with entries 𝐷(𝑖, 𝑖) = 𝑗 𝐴(𝑖, 𝑗).            the loss function as the mean square error as follows
   Since Eq. 5 can be computationally expensive, we use its                                 ⃦                                  ⃦
                                                                                        1 ⃦                             ⃦2
fast approximate version from [15]:                                              𝐿=                (𝑙)
                                                                                            ⃦ℰ𝑚 − 𝐻 (𝑀𝑚,𝑚+1 ℰ𝑚+1 , 𝐴𝑚 ) ⃦ .             (9)
                                                                                      |𝑉𝑚 |
                                       1            1
                           ˜ − 2 𝐴˜𝐷
                  𝑋 *𝐺 𝑔 ≈ 𝐷       ˜ − 2 𝑋Θ                         (6)      We refer to the learning task associated with the above loss
                             ∑︀ ˜                                         function as double-base embedding learning since it requires
where 𝐴˜ = 𝐴 + 𝜆𝐷, ˜
                   𝐷(𝑖, 𝑖) =    𝐴(𝑖, 𝑗), Θ ∈ R𝑑×𝑑 , and
                                               𝑗                          conducting two times of base embedding in the consecutive
𝜆 ∈ [0, 1] is a hyper-parameter for controlling the weight                layers. We point out, however, there are two key drawbacks to
of self-loop. As this approximate convolution model can be                this method. First of all, the above loss function requires one
regarded as a layer-wise linear model, we can stack multiple              more level of coarsening to construct 𝒢𝑚+1 and an extra base
such layers to achieve a model of higher capacity. The 𝑘-th               embedding on 𝒢𝑚+1 . These two steps, especially the latter,
layer of this neural network model is                                     introduce non-negligible overheads to the MILE framework,
                      (︁                                       )︁
                     ˜ − 2 𝐴˜𝐷
    𝐻 (𝑘) (𝑋, 𝐴) = 𝜎 𝐷
                              1            1
                             ˜ − 2 𝐻 (𝑘−1) (𝑋, 𝐴)Θ (𝑘)              (7)   which contradicts our motivation of scaling up graph embed-
                                                                          ding. More importantly, ℰ𝑚 might not be a desirable “ground
where 𝜎(·) is an activation function, Θ (𝑘) is a layer-specific           truth” for the refined embeddings, which are predicted based
trainable weight matrix, and 𝐻 (0) (𝑋, 𝐴) = 𝑋.                            on ℰ𝑚+1 . This is because most of the embedding methods
   In this paper, we define our embedding refinement model                are invariant to an orthogonal transformation of the embed-
as a 𝑙-layer graph convolution model                                      dings, i.e., the embeddings can be rotated by an arbitrary
                       (︀ 𝑝       )︀               (︀ 𝑝   )︀              orthogonal matrix [10]. In other words, the embedding spaces
               ℰ𝑖 = ℛ ℰ𝑖 , 𝐴𝑖 ≡ 𝐻 (𝑙) ℰ𝑖 , 𝐴𝑖 .                 (8)       of graph 𝒢𝑚 and 𝒢𝑚+1 can be totally different since the two
   The architecture of the refinement model is shown in Fig-              base embeddings are learned independently. Even if we follow
ure 3. The intuition behind this refinement model is to in-               the paradigm in [5] and conduct base embedding on 𝒢𝑚 using
                                                                          the simple projected embeddings from 𝒢𝑚+1 (ℰ𝑚 ) as initial-
                                                                                                                            𝑝
tegrate the structural information of the current graph 𝒢𝑖
into the projected embedding ℰ𝑖 by repeatedly performing
                                  𝑝                                       ization, the embedding space does not naturally generalize
the spectral graph convolution. To some extent, each layer of             and can drift during re-training. One possible solution is to
graph convolution network in Eq. 7 can be regarded as one                 use an alignment procedure to force the embeddings to be
iteration of embedding propagation in the graph following the             aligned between the two graphs [11]. But it could be very
re-normalized adjacency matrix 𝐷   ˜ − 12 𝐴˜𝐷
                                            ˜ − 12 . Note that this       computationally expensive.
re-normalized matrix is well aligned with the way we conduct                 In this paper, we propose a very simple method to address
normalized heavy edge matching in Eq. 1, where we apply                   the above issues. Instead of conducting an additional level of
the same way of re-normalization on the adjacency matrix                  coarsening, we construct a dummy coarsened graph by simply
for edge matching. However, we point out that the graph                   copying 𝒢𝑚 , i.e., 𝑀𝑚,𝑚+1 = 𝐼 and 𝒢𝑚+1 = 𝒢𝑚 . By doing
convolution model goes beyond just simple propagation in                  this, we not only reduce one iteration of graph coarsening,
that the activation function is applied for each iteration of             but also avoid performing base embedding on 𝒢𝑚+1 simply
propagation and each dimension of the embedding interacts                 because ℰ𝑚+1 = ℰ𝑚 . Moreover, the embeddings of 𝒢𝑚 and
with other dimensions controlled by the weight matrix Θ (𝑘) .             𝒢𝑚+1 are guaranteed to be in the same space in this case
                                                                          without any drift. With this strategy, we change the loss
We next discuss how the weight matrix Θ (𝑘) is learned.
                                                                          function for model learning as follows
   4.3.2 Refinement Model Learning. The learning of the re-                                       ⃦                      ⃦
                                                                                              1 ⃦                    ⃦2
finement model is essentially learning Θ (𝑘) for each 𝑘 ∈ [1, 𝑙]                       𝐿=                (𝑙)
                                                                                                  ⃦ℰ𝑚 − 𝐻 (ℰ𝑚 , 𝐴𝑚 ) ⃦ .               (10)
                                                                                            |𝑉𝑚 |
according to Eq. 7. Here we study how to design the learning
task and construct the loss function.                                        With the above loss function, we adopt gradient descent
   Since the graph convolution model 𝐻 (𝑙) (·) aims to predict            with back-propagation to learn the parameters Θ (𝑘) , 𝑘 ∈ [1, 𝑙].
the embeddings ℰ𝑖 on graph 𝒢𝑖 , we can directly run a base                In the subsequent refinement steps, we apply the same set of
embedding on 𝒢𝑖 to generate the “ground-truth” embeddings                 parameters Θ (𝑘) to infer the refined embeddings. We point out
and use the difference between these embeddings and the                   that the training of the refinement model is rather efficient
Conference’18, Jan 2018, DC, Washington USA                          Jiongqian Liang, Saket Gurukar, and Srinivasan Parthasarathy


as it is done on the coarsest graph, which is usually much             treating the walks as sentences. Following the original
smaller than the original graph. The embeddings refinement             work [20], we set the length of random walks as 80, number
process involves merely sparse matrix multiplications using            of walks per node as 10, and context windows size as 10.
Eq. 8 and is relatively affordable compared to conducting            ∙ Node2Vec (NV) [9]: This is an improved version of Deep-
embedding on the original graph.                                       Walk, where it generates random walks with more flexibility
   With these different components, we summarize the whole             controlled through parameters 𝑝 and 𝑞. We use the same
algorithm of our MILE framework in Algorithm 2.                        setting as DeepWalk for those common hyper-parameters
                                                                       while setting 𝑝 = 4.0 and 𝑞 = 1.0, which we found empiri-
Algorithm 2 Multi-Level Algorithm for Graph Embedding                  cally to generate better results across all the datasets.
Input: A input graph 𝒢0 = (𝑉0 , 𝐸0 ), # coarsening levels 𝑚, and a   ∙ GraRep (GR) [3]: This method considers different powers
base embedding method 𝑓 (·).                                           (up to 𝑘) of the adjacency matrix to preserve higher-order
Output: Graph embeddings ℰ0 on 𝒢0 .                                    graph proximity for graph embedding. It uses SVD decom-
 1: Use Algorithm 1 to coarsen 𝒢0 into 𝒢1 , 𝒢2 , ..., 𝒢𝑚 .             position to generate the low-dimensional representation of
 2: Perform base embedding on the coarsest graph 𝒢𝑚 (See Eq. 3).       nodes. We set 𝑘 = 4 as suggested in the original work.
 3: Learn the weights Θ (𝑘) using the loss function in Eq. 10.       ∙ NetMF (NM) [21]: It is a recent effort that supports graph
 4: for 𝑖 = (𝑚 − 1)...0 do                                             embedding via matrix factorization. We set the window size
 5:     Compute the projected embeddings ℰ𝑖𝑝 on 𝒢𝑖 using Eq. 4.        to 10 and the rank ℎ to 1024, and lever the approximate
 6:     Use Eq. 7 and Eq. 8 to compute refined embeddings ℰ𝑖 .         version, as suggested and reported by the authors.
 7: Return graph embeddings ℰ0 on 𝒢0 .
                                                                     MILE-specific Settings: When applying our MILE framework,
                                                                     we vary the coarsening levels 𝑚 from 1 to 10 whenever possi-
5       EXPERIMENTS AND ANALYSIS                                     ble. For the graph convolution network model, the self-loop
In this section, we conduct extensive experiments to gain            weight 𝜆 is set to 0.05, the number of hidden layers 𝑙 is 2,
more insights on the proposed MILE framework.                        and tanh(·) is used as the activation function, the learning
                                                                     rate is set to 0.001 and the number of training epochs is 200.
               Dataset   # Nodes      # Edges     # Classes
                 PPI         3,852       37,841          50          The Adam Optimizer is used for model training.
                 Blog       10,312      333,983          39          System Specification: The experiments were conducted on a
                Flickr      80,513    5,899,882        195           machine running Linux with an Intel Xeon E5-2680 CPU
               YouTube   1,134,890    2,987,624          47
                                                                     (28 cores, 2.40GHz) and 128 GB of RAM. For all the four
                 Yelp    8,938,630   39,821,123          22
                                                                     base embedding methods, we adapt the original code from
                    Table 2: Dataset Information                     the authors3 . We additionally use TensorFlow package for
                                                                     the embeddings refinement learning component. We lever the
5.1 Experimental Configuration                                       available parallelism (on 28 cores) for each method (e.g., the
Datasets: The datasets used in our experiments is shown in           generation of random walks in DeepWalk and Node2Vec, the
Table 2 and are detailed below:                                      training of the refinement model in MILE, etc.).
∙ PPI is a Protein-Protein Interaction graph constructed             Evaluation Metrics: To evaluate the quality of the embed-
  based on the interplay activity between proteins of Homo           dings, we follow the typical method in existing work to per-
  Sapiens, where the labels represent biological states.             form multi-label node classification [9, 20]. Specifically, after
∙ Blog is a network of social relationship of bloggers on Blog-      the graph embeddings are learned for nodes (label is not used
  Catalog and the labels indicate interests of the bloggers.         for this part), we run a 10-fold cross validation using the
∙ Flickr is a social network of the contacts between users on        embeddings as features and report the average Micro-F1 and
  flickr.com with labels denoting the interest groups.               average Macro-F1. We also record the end-to-end wallclock
∙ YouTube is a social network between users on YouTube,              time consumed by each method for scalability comparisons.
  where labels represent genres of groups subscribed by users.
∙ Yelp is a social network of friends on Yelp and labels             5.2   MILE Framework Performance
  indicate the business categories on which the users review.        We first evaluate the performance of our MILE framework
The first four datasets have been previously used to evaluate        when applied to different graph embedding methods. For
graph embedding strategies [9, 20, 21], while Yelp is a dataset      each dataset, we show the results of MILE under two settings
preprocessed by us following similar procedures in [13]2 .           of coarsening levels 𝑚 and expand on the remaining results
Baseline Methods: To demonstrate that MILE can work with             in the next section. Table 3 summarizes the performance
different graph embedding methods, we explore several pop-           of MILE on different datasets with various base embedding
ular methods for graph embedding.                                    methods4 . We make the following observations:
∙ DeepWalk (DW) [20]: This method generates truncated                   3
                                                                          DeepWalk: https://github.com/phanein/deepwalk; Node2Vec:
  random walks on graphs and applies the Skip Gram by                https://github.com/aditya-grover/node2vec; GraRep: https://github.
                                                                     com/thunlp/OpenNE; NetMF: https://github.com/xptree/NetMF
    2                                                                   4
        Raw data: https://www.yelp.com/dataset_challenge/dataset          We discuss the results of Yelp later.
MILE: A Multi-Level Framework for Scalable Graph Embedding                                  Conference’18, Jan 2018, DC, Washington USA

        Method                Micro-F1        Macro-F1        Time (mins)     Method             Micro-F1        Macro-F1          Time (mins)
        DeepWalk              23.0            18.6            2.42            DeepWalk           37.0            21.0              8.02
        MILE (DW, 𝑚 = 1)      25.6(11.3%↑)    20.4(9.7%↑)     1.22(2.0×)      MILE (DW, 𝑚 = 1)   42.9(15.9%↑)    27.0(28.6%↑)      4.69(1.7×)
        MILE (DW, 𝑚 = 2)      25.5(10.9%↑)    20.7(11.3%↑)    0.67(3.6×)      MILE (DW, 𝑚 = 2)   39.4(6.5%↑)     23.5(11.9%↑)      2.71(3.0×)
        Node2Vec              24.3            19.6            4.01            Node2Vec           39.1            23.0              13.04
        MILE (NV, 𝑚 = 1)      25.9(6.6%↑)     20.6(5.1%↑)     1.77(2.3×)      MILE (NV, 𝑚 = 1)   42.8(9.5%↑)     26.4(14.8%↑)      6.99(1.9×)
        MILE (NV, 𝑚 = 2)      26.0(7.0%↑)     21.1(7.7%↑)     0.98(4.1×)      MILE (NV, 𝑚 = 2)   40.2(2.8%↑)     23.9(3.9%↑)       3.89(3.4×)
        GraRep                25.5            20.0            2.99            GraRep             40.6            23.3              28.76
        MILE (GR, 𝑚 = 1)      25.6(0.4%↑)     19.8(-1.0%↓)    1.11(2.7×)      MILE (GR, 𝑚 = 1)   41.7(2.7%↑)     24.0(3.0%↑)       12.25(2.3×)
        MILE (GR, 𝑚 = 2)      25.3(-0.8%↓)    19.5(-2.5%↓)    0.43(6.9×)      MILE (GR, 𝑚 = 2)   38.3(-5.7%↓)    20.4(-12.4%↓)     4.22(6.8×)
        NetMF                 24.6            20.1            0.65            NetMF              41.4            25.0              2.64
        MILE (NM, 𝑚 = 1)      26.9(9.3%↑)     21.6(7.5%↑)     0.27(2.5×)      MILE (NM, 𝑚 = 1)   43.8(5.8%↑)     27.6(10.4%↑)      1.98(1.3×)
        MILE (NM, 𝑚 = 2)      26.7(8.5%↑)     21.1(5.0%↑)     0.17(3.9×)      MILE (NM, 𝑚 = 2)   42.4(2.4%↑)     25.5(2.0%↑)       1.27(2.1×)

                                (a) PPI Dataset                                                    (b) Blog Dataset

        Method               Micro-F1        Macro-F1        Time (mins)      Method             Micro-F1       Macro-F1         Time (mins)
        DeepWalk             40.0            26.5            50.08            DeepWalk           45.2           34.7             604.83
        MILE (DW, 𝑚 = 1)     40.4(1.0%↑)     27.3(3.0%↑)     34.48(1.5×)      MILE (DW, 𝑚 = 6)   46.1(2.0%↑)    38.5(11.0%↑)     55.20(11.0×)
        MILE (DW, 𝑚 = 2)     39.3(-1.8%↓)    26.1(-1.5%↓)    26.88(1.9×)      MILE (DW, 𝑚 = 8)   44.3(-2.0%↓)   35.3(1.7%↑)      37.35(16.2×)
        Node2Vec             40.5            27.3            78.21            Node2Vec           45.5           34.6             951.27
        MILE (NV, 𝑚 = 1)     40.7(0.5%↑)     27.7(1.5%↑)     50.54(1.5×)      MILE (NV, 𝑚 = 6)   46.3(1.8%↑)    38.3(10.7%↑)     83.52(11.4×)
        MILE (NV, 𝑚 = 2)     38.8(-4.2%↓)    25.8(-5.5%↓)    36.85(2.1×)      MILE (NV, 𝑚 = 8)   44.3(-2.6%↓)   35.8(3.5%↑)      55.55(17.1×)
        GraRep               N/A             N/A             > 2343.37        GraRep             N/A            N/A              > 3167.00
        MILE (GR, 𝑚 = 1)     36.7            18.6            697.39(>3.4×)    MILE (GR, 𝑚 = 6)   43.2           32.7             1644.89(>1.9×)
        MILE (GR, 𝑚 = 2)     36.3            18.6            163.05(>14.4×)   MILE (GR, 𝑚 = 8)   42.3           30.9             673.95(>4.7×)
        NetMF5               31.8            14.0            69.72            NetMF              N/A            N/A              > 574.75
        MILE (NM, 𝑚 = 1)     39.3(23.6%↑)    24.5(75.0%↑)    24.03(2.9×)      MILE (NM, 𝑚 = 6)   40.9           27.8             35.22(>16.3×)
        MILE (NM, 𝑚 = 2)     39.5(24.2%↑)    25.9(85.0%↑)    15.84(4.4×)      MILE (NM, 𝑚 = 8)   39.2           25.5             19.22(>29.9×)
                               (c) Flickr Dataset                                                (d) YouTube Dataset

Table 3: Performance of MILE compared to the original embedding methods. DeepWalk, Node2Vec, GraRep, and NetMF denotes the original
method without using our MILE framework. We set the number of coarsening levels 𝑚 to 1 and 2 for PPI, Blog and Flickr, while choosing 6
and 8 for YouTube (due to its larger scale). The Micro-F1 and Macro-F1 are in 10−2 scale while the column Time shows the running time in
minutes. The numbers within the parenthesis by the reported Micro-F1 and Macro-F1 scores are the relative percentage of change compared to
the original method, e.g., MILE (DW, 𝑚 = 1) vs. DeepWalk. “↑” and “↓” respectively indicate improvement and decline. Numbers along with
“×” is the speedup compared to the original method. “N/A” indicates the method runs out of 128 GB memory and we show the amount of
running time spent when it happens.


∙ MILE is scalable. MILE greatly boosts the speed of the                      ∙ MILE improves quality. For the smaller coarsening levels
  explored embedding methods. With a single level of coars-                     across all the datasets and methods, MILE-enhanced em-
  ening (𝑚=1), we are able to achieve speedup ranging from                      beddings almost always offer a qualitative improvement
  1.5× to 3.4× (on PPI, Blog, and Flickr) while improving                       over the original embedding method as evaluated by the
  qualitative performance. Larger speedups are typically ob-                    Micro-F1 score and Macro-F1 score (as high as 28.6% while
  served on GraRep and NETMF. Increasing the coarsening                         many others also show an 10%+ increase). Evident exam-
  level 𝑚 to 2, the speedup increases further (up to 14.4×),                    ples include MILE (DW, 𝑚 = 1) on Blog/PPI and MILE
  while the quality of the embeddings is comparable with                        (NM, 𝑚 = 1) on PPI/Blog/Flickr. Even with the higher
  the original methods reflected by Micro-F1 and Macro-F1.                      number of coarsening level (𝑚 = 2 for PPI/Blog/Flickr;
  On the largest datasets among the four (YouTube) where                        𝑚 = 8 for YouTube), MILE in addition to being much faster
  the coarsening level is 6 and 8, we observe more than 10×                     can still improve, qualitatively, over the original methods
  speedup for DeepWalk and Node2Vec. For NetMF, the                             on all datasets, e.g., MILE(NM, 𝑚 = 2) ≫ NETMF on PPI,
  speedup is even larger (more than 16×) – original NetMF                       Blog, and Flickr. We conjecture the observed improvement
  runs out of memory within 9.5 hours while MILE (NM) only                      on quality is because the embeddings begin to rely on a
  takes around 35 minutes (𝑚 = 6) or 20 minutes (𝑚 = 8).                        more holistic view of the graph.
                                                                              ∙ MILE supports multiple embedding strategies. We make
                                                                                some embedding-specific observations here. We observe
                                                                                that MILE consistently improves both the quality and the
    5
      The NetMF paper [21], reports different results on Flickr with            efficiency of NetMF on all four datasets (for YouTube the
𝑑 = 128 and rank ℎ = 1024, which we were unable to replicate. In                base method runs out of memory). For the largest dataset
personal communication, its first author promptly acknowledged the
error - a much larger rank ℎ is needed to achieve the reported results,
                                                                                the speedups afforded exceed 30-fold. We observe that
which comes at a significant computation and memory cost (their                 for GraRep, while speedups with MILE are consistently
results are on a machine with 1TB of memory).                                   observed, the qualitative improvements, if any, are smaller
Conference’18, Jan 2018, DC, Washington USA                       Jiongqian Liang, Saket Gurukar, and Srinivasan Parthasarathy


 (for both YouTube and Flickr, the base method runs out of                                 PPI            Blog            Flickr          YouTube
                                                                                      Mi-F1 Time     Mi-F1 Time      Mi-F1 Time        Mi-F1 Time
 memory). For DeepWalk and Node2Vec, we again observe              DeepWalk           23.0    2.42   37.0     8.02   40.0      50.08   45.2   604.83
 consistent improvements in scalability (up to 11-fold on the      MILE (DW)          25.6    1.22   42.9     4.69   40.4      34.48   46.1   55.20
                                                                   MILE-rm (DW)       25.3    1.01   40.4     3.62   38.9      26.67   44.9   55.10
 largest dataset) as well as quality using MILE with a single      MILE-proj (DW)     20.9    1.12   34.5     3.92   35.5      25.99   40.7   53.97
 level of coarsening (or 𝑚 = 6 for YouTube). However, when         MILE-avg (DW)      23.5    1.07   37.7     3.86   37.2      25.99   41.4   55.26
                                                                   MILE-untr (DW)     23.5    1.08   35.5     3.96   37.6      26.02   41.8   54.52
 the coarsening level is increased, the additional speedup         MILE-2base (DW)    25.4    2.22   35.6     6.74   37.7      53.32   41.6   94.74
 afforded (up to 17-fold) comes at a mixed cost to quality         MILE-gs (DW)       22.4    2.03   35.3     6.44   36.4      44.81   43.6   394.72
 (micro-F1 drops slightly while macro-F1 improves slightly).       NetMF              24.6   0.65    41.4    2.64    31.8     69.72    N/A    >574
                                                                   MILE (NM)          26.9   0.27    43.8    1.98    39.3     24.03    40.9   35.22
   To summarize, our MILE framework not only significantly         MILE-rm (NM)       25.2   0.22    41.0    1.69    37.6     20.00    39.6   33.52
speeds up the embedding methods, but also improves the             MILE-proj (NM)     23.5   0.12    38.7    1.06    34.5     15.10    26.4   26.48
                                                                   MILE-avg (NM)      24.5   0.13    39.9    1.05    36.4     14.86    26.4   27.71
quality of the node embeddings. We do notice that it nega-         MILE-untr (NM)     24.8   0.13    39.4    1.08    36.4     15.23    30.2   27.20
tively affects the embeddings in some case when the number         MILE-2base (NM)    26.6   0.29    41.3    2.33    37.7     31.65    34.7   55.18
                                                                   MILE-gs (NM)       24.8   1.08    40.0    3.70    35.1     34.25    36.4   345.28
of coarsening levels is large, e.g., MILE (GR, 𝑚 = 2) on PPI
and Blog. But we point out the decrease is mostly minor com-      Table 4: Comparisons of graph embeddings between MILE and its
                                                                  variants. Except for the original methods (DeepWalk and NetMF),
pared to large speedup achieved. Moreover, we can reduce the
                                                                  the number of coarsening level 𝑚 is set to 1 on PPI/Blog/Flickr and
coarsening levels in order to generate better embeddings (e.g.,
                                                                  6 on YouTube. Mi-F1 is the Micro-F1 score in 10−2 scale while Time
𝑚 = 1) if the quality of the embeddings is valued over effi-      column shows the running time of the method in minutes. “N/A”
ciency. We discuss the trade-off between quality and efficiency   denotes the method consumes more than 128 GB RAM.
of graph embedding in Sec. 5.4
                                                                  ∙ GraphSAGE as Refinement Model (MILE-gs): It replaces
5.3      MILE Drilldown: Design Choices
                                                                    the graph convolution network in our refinement method
We now study the role of the design choices we make within          with GraphSAGE [10]7 . We choose max-pooling for aggre-
the MILE framework related to the coarsening and refinement         gation and set the number of sampled neighbors as 100, as
procedures described. To this end, we examine alternative           suggested by the authors. Also, concatenation is conducted
design choices and systematically examine their performance.        instead of replacement during the process of propagation.
The alternatives we consider are:
                                                                     Table 4 shows the comparison of performance on these
∙ Random Matching (MILE-rm): We replace Algorithm 1               methods across the four datasets. Due to limit of space, we
  with a simple random matching approach for graph coars-         focus on using DeepWalk and NetMF for base embedding
  ening. For each iteration of coarsening, we repeatedly pick     with a smaller coarsening level (𝑚 = 1 for PPI, Blog, and
  a random pair of connected nodes as a match and merge           Flickr; 𝑚 = 6 for YouTube). Results are similar for the other
  them into a super-node until no more matching can be            embedding options we consider. We hereby summarize the
  found. The rest of the algorithm is the same as our MILE.       key information derived from Table 4 as follows:
∙ Simple Projection (MILE-proj): We replace our embedding
                                                                  ∙ The matching methods used within MILE offer a qualitative
  refinement model with a simple projection method. In other
                                                                    benefit at a minimal cost to execution time. Comparing
  words, we directly copy the embedding of a super-node to
                                                                    MILE with MILE-rm for all the datasets, we can see that
  its original node(s) without any refinement (see Eq. 4).
                                                                    MILE generates better embeddings than MILE-rm using
∙ Averaging Neighborhoods (MILE-avg): For this baseline
                                                                    either DeepWalk or NetMF as the base embedding method.
  method, the refined embedding of each node is a weighted
                                                                    Though MILE-rm is slightly faster than MILE due to its
  average node embeddings of its neighborhoods (weighted by
                                                                    random matching, its Micro-F1 score and Macro-F1 score
  the edge weights). This can be regarded as an embeddings
                                                                    are consistently lower than of MILE.
  propagation method. We add self-loop to each node6 and
                                                                  ∙ The graph convolution based refinement learning method-
  conduct the embeddings propagation for two rounds.
                                                                    ology in MILE is particularly effective. Simple projection
∙ Untrained Refinement Model (MILE-untr): Instead of train-
                                                                    based MILE-proj, performs significantly worse than MILE.
  ing the refinement model to minimize the loss defined in
                                                                    The other two variants (MILE-avg and MILE-untr) which
  Eq. 10, this baseline merely uses a fix set of values for
                                                                    do not train the refinement model at all, also perform
  parameters Θ (𝑘) without training (values are randomly            much worse than the proposed method. Note MILE-untr
  generated; other parts of the model in Eq. 7 are the same,        is the same as MILE except it uses a default set of pa-
  including 𝐴˜ and ˜𝐷).                                             rameters instead of learning those parameters. Clearly, the
∙ Double-base Embedding for Refinement Training (MILE-              model learning part of our refinement method is a funda-
  2base): This method replaces the loss function in Eq. 10          mental contributing factor to the effectiveness of MILE.
  with the alternative one in Eq. 9 for model training. It          Through training, the refinement model is tailored to the
  conducts one more layer of coarsening and base embedding          specific graph under the base embedding method in use.
  (level 𝑚 + 1), from which the embeddings are projected to         The overhead cost of this learning (comparing MILE with
  level 𝑚 and used as the input for model training.                 MILE-untr), can vary depending on the base embedding
   6                                                                 7
       Self-loop weights are tuned to the best performance.              Adapt code from https://github.com/williamleif/GraphSAGE
  MILE: A Multi-Level Framework for Scalable Graph Embedding                                                                                        Conference’18, Jan 2018, DC, Washington USA

                                                  MILE (DeepWalk)                               MILE (Node2Vec)                            MILE (GraRep)          MILE (NetMF)
              0.30                                                                                                             0.45                                             0.52
                                                                        0.45
              0.28                                                                                                             0.40                                             0.50
                                                                        0.40
                                                                                                                               0.35                                             0.48
              0.26                                                      0.35
                                                                                                                                                                                0.46
Micro-f1                                                  Micro-f1                                               Micro-f1                                         Micro-f1
                                                                        0.30                                                   0.30
              0.24                                                                                                                                                              0.44
                                                                        0.25                                                   0.25
              0.22                                                                                                                                                              0.42
                                                                        0.20                                                   0.20
              0.20                                                                                                                                                              0.40
                                                                        0.15                                                   0.15                                             0.38
              0.18 0         1      2         3       4                        0     1    2      3 4     5   6                        0 1 2 3 4 5 6 7 8                                0 1 2 3 4 5 6 7 8
                                 # Levels                                                     # Levels                                       # Levels                                         # Levels
                         (a) PPI (Micro-F1)                                        (b) Blog (Micro-F1)                                  (c) Flickr (Micro-F1)                          (d) YouTube (Micro-F1)



                                                                        101                                                                                                     103


                                                          Time (mins)                                            Time (mins)                                      Time (mins)
                                                                                                                               102
Time (mins)
               100

                                                                                                                                                                                102
                                                                        100
                                                                                                                               101
              10 1
                     0       1      2         3       4                        0    1    2       3 4     5   6                        0 1 2 3 4 5 6 7 8                                0 1 2 3 4 5 6 7 8
                                 # Levels                                                     # Levels                                       # Levels                                         # Levels
                     (e) PPI (Running Time)                                    (f) Blog (Running Time)                                (g) Flickr (Running Time)                    (h) YouTube (Running Time)

  Figure 4: Changes in performance as the number of coarsening levels in MILE increases (best viewed in color). Micro-F1 and running-time are
  reported in the first and second row respectively. Running time in minutes is shown in logarithm scale. Note that # level = 0 represents the
  original embedding method without using MILE. Lines/points are missing for algorithms that use over 128 GB of RAM.


    employed (for instance on the YouTube dataset, it is an                                                                     contains less than 128 nodes (it is trivial to embed such a
    insignificant 1.2% on DeepWalk - while being up to 20%                                                                      graph into 128 dimensions). Figure 4 shows the changes of
    on NetMF) but is still worth it due to qualitative benefits                                                                 Micro-F1 for node classification and running time of MILE
    (Micro-F1 up from 30.2 to 40.9 with NetMF on YouTube).                                                                      as 𝑚 increases. We underline the following observations:
  ∙ Graph convolution refinement learning outperforms Graph-
                                                                                                                                ∙ When coarsening level 𝑚 is small, MILE tends to signif-
    SAGE. Replacing the graph convolution network with
                                                                                                                                  icantly improve the quality of embeddings while taking
    GraphSAGE for embeddings refinement, MILE-gs does
                                                                                                                                  much less time. From 𝑚 = 0 (i.e., without applying the
    not perform as well as MILE. It is also computationally
                                                                                                                                  MILE framework) to 𝑚 = 1, we see a clear jump of the
    more expensive, partially due to its reliance on embeddings
                                                                                                                                  Micro-F1 score on all the datasets across the four base
    concatenation, instead of replacement, during the process
                                                                                                                                  embedding methods. This observation is more evident on
    the embeddings propagation (higher model complexity).
                                                                                                                                  larger datasets (Flickr and YouTube). On YouTube, MILE
  ∙ Double-base embedding learning is not effective. In Sec. 4.3.2,
                                                                                                                                  (DeepWalk) with 𝑚=1 increases the Micro-F1 score by
    we discuss the issues with unaligned embeddings of the
                                                                                                                                  5.3% while only consuming half of time compared to the
    double-base embedding method for the refinement model
                                                                                                                                  original DeepWalk. MILE (DeepWalk) continues to gen-
    learning. The performance gap between MILE and MILE-
                                                                                                                                  erate embeddings of better quality than DeepWalk until
    2base in Table 4 provides empirical evidence supporting
                                                                                                                                  𝑚 = 7, where the speedup is 13×.
    our argument. This gap is likely caused by the fact that
                                                                                                                                ∙ As the coarsening level 𝑚 in MILE increases, the running
    the base embeddings of level 𝑚 and level 𝑚 + 1 might
                                                                                                                                  time drops dramatically while the quality of embeddings
    not lie in the same embedding space (rotated by some
                                                                                                                                  only decreases slightly. The running time decreases at an
    orthogonal matrix) [10]. As a result, using the projected
                                                                                                                                  almost exponential rate (logarithm scale on the y-axis in
    embeddings ℰ𝑚 as input for model training (MILE-2base)
                   𝑝
                                                                                                                                  the second row of Figure 4). On the other hand, the Micro-
    is not as good as directly using ℰ𝑚 (MILE). Moreover,
                                                                                                                                  F1 score descends much more slowly (first row of Figure 4).
    Table 4 shows that the additional round of base embed-
                                                                                                                                  Sacrificing a tiny fraction of quality on embeddings can
    ding in MILE-2base introduces a non-trivial overhead. On
                                                                                                                                  save a huge amount of computational resource.
    YouTube, the running time of MILE-2base is 1.6 times as
    much as MILE.
                                                                                                                                5.5       MILE Drilldown: Memory Consumption
  5.4 MILE Drilldown: Varying Coarsening Levels                                                                                 We now study the impact of MILE on reducing memory
 We now study the performance of the MILE framework as                                                                          consumption. For this purpose, we focus on MILE (GraRep)
 we vary the number of coarsening levels 𝑚. Starting from                                                                       and MILE (NetMF), with GraRep and NetMF as base em-
 𝑚 = 0, we increase 𝑚 until it reaches 8 or the coarsest graph                                                                  bedding methods respectively. Both of these are embedding
Conference’18, Jan 2018, DC, Washington USA                                                                                            Jiongqian Liang, Saket Gurukar, and Srinivasan Parthasarathy


                                                                                      1.6                                              5.7              MILE: Large Graph Embedding
                                                                                      1.4
                                                                                                                                       We now explore the scalability of our MILE framework on
                    10
                                                                                      1.2




   Memory in (GB)                                                    Memory in (GB)
                     8

                     6
                                                                                      1.0
                                                                                      0.8
                                                                                                                                       the large Yelp dataset. To the best our knowledge, Yelp is
                     4                                                                0.6                                              one of the largest datasets for a graph embedding task in
                     2
                                                                                      0.4
                                                                                      0.2
                                                                                                                                       the literature with around 9 million nodes and 40 million
                     0
                         0   1     2    3    4      5   6
                                                                                      0.0
                                                                                             0   1     2    3    4      5   6          edges. None of the four graph embedding methods studied
                                 Coarsening level                                                    Coarsening level
                                                                                                                                       in this paper can successfully conduct graph embedding on
                         (a) MILE (GraRep)                                                   (b) MILE (NetMF)
                                                                                                                                       Yelp within 60 hours on a modern machine with 28 cores
Figure 5: Memory consumption of MILE (GraRep) and MILE                                                                                 and 128 GB RAM (two run out of memory). Leveraging the
(NetMF) on Blog with varied coarsening levels. Coarsening level 0
                                                                                                                                       proposed MILE framework, however, makes it possible to
corresponds to the original embedding method without applying the
                                                                                                                                       perform graph embedding on this scale of datasets. To this
MILE framework.
                                                                                                                                       end, we run the MILE framework on Yelp using the four
                                       PPI                       Blog                           Flickr                   YouTube       graph embedding techniques as the base embedding methods
                                  Mi-F1 Time                Mi-F1 Time                      Mi-F1 Time               Mi-F1    Time     with various coarsening levels (see Figure 6 for the results).
 DeepWalk                         23.0    2.42              37.0     8.02                   40.0     50.08           45.2    604.83    We observe that MILE significantly reduces the running time
 MILE (DW)                        25.6    1.22              42.9     4.69                   40.4     34.48           46.1    55.20
 HARP (DW)                        24.1    3.08              41.3     9.85                   40.6     78.21           46.6    1727.78   while the Micro-F1 score remains almost unchanged. For
                             Table 5: Comparisons of MILE with HARP.                                                                   example, MILE reduces the running time of DeepWalk from
                                                                                                                                       53 hours (coarsening level 4) to 2 hours (coarsening level
methods based on matrix factorization, which possibly in-                                                                              22) while reducing the Micro-F1 score just by 1% (from
volves a dense objective matrix and could be rather memory                                                                             0.643 to 0.634). Meanwhile, there is no change in the Micro-
expensive. We do not explore DeepWalk and Node2Vec here                                                                                F1 score from coarsening level 4 to 10, where the running
since their embedding learning methods generate truncated                                                                              time is improved by a factor of two. These results affirm the
random walks (training data) on the fly with almost negli-                                                                             power of the proposed MILE framework on scaling up graph
gible memory consumption (compared to the space storing                                                                                embedding algorithms while generating quality embeddings.
the graph and the embeddings). Figure 5 shows the memory                                                                                                     MILE (DeepWalk)    MILE (Node2Vec)                       MILE (GraRep)   MILE (NetMF)
consumption of MILE (GraRep) and MILE(NetMF) as the                                                                                                   0.70
coarsening level increases on Blog (results on other dataset                                                                                          0.68
are similar). We observe that MILE significantly reduces the                                                                                                                                                    103

                                                                                                                                                                                                  Time (mins)
                                                                                                                                                      0.66
memory consumption as the coarsening level increases. Even                                                                                 Micro-f1   0.64
with one level of coarsening, the memory consumption of
                                                                                                                                                      0.62
GraRep and NetMF reduces by 64% and 42% respectively.
                                                                                                                                                      0.60 0 2 4 6 8 10 12 14 16 18 20 22                       102 0 2 4 6 8 10 12 14 16 18 20 22
The dramatic reduction continues as the coarsening level                                                                                                             # Levels                                                 # Levels
increases until it reaches 4, where the memory consumption                                                                                                       (a) Micro-F1                                            (b) Running Time
is mainly contributed by the storage of the graph and the                                                                              Figure 6: Running MILE on Yelp dataset. Lines/points are missing
embeddings. This memory reduction is consistent with our                                                                               for algorithms that do not finish within 60 hours or use over 128 GB
intuition, since both # rows and # columns in the objective                                                                            of RAM.
matrix for factorization reduce almost by half with one level
of coarsening.
                                                                                                                                       6        CONCLUSION
5.6 Comparing MILE with HARP                                                                                                           In this work, we propose a novel multi-level embedding
HARP is a recent multi-level method primarily for improv-                                                                              (MILE) framework to scale up graph embedding techniques,
ing the quality of graph embeddings. We compare HARP                                                                                   without modifying them. Our framework incorporates exist-
with our MILE framework using DeepWalk as the base em-                                                                                 ing embedding techniques as black boxes, and significantly
bedding method8 . Table 5 shows the performance of these                                                                               improves the scalability of extant methods by reducing both
two methods on the four datasets (coarsening level is 1 on                                                                             the running time and memory consumption. Additionally,
PPI/Blog/Flickr and 6 on YouTube). From the table we can                                                                               MILE also provides a lift in the quality of node embeddings
observe that MILE generates embeddings of comparable qual-                                                                             in most of the cases. A fundamental contribution of MILE is
ity with HARP. MILE performs much better than HARP on                                                                                  its ability to learn a refinement strategy that depends on both
PPI and Blog but falls slightly behind on Flickr and YouTube.                                                                          the underlying graph properties and the embedding method
However, MILE is significant faster than HARP on all the                                                                               in use. In the future, we plan to generalize our framework for
four datasets (e.g. on YouTube, MILE affords a 31× speedup).                                                                           information-rich graphs, such as heterogeneous information
This is because HARP requires running the whole embedding                                                                              networks and attributed graphs.
algorithm on each coarsened graph, which introduces a huge                                                                             Acknowledgments: This work is supported by the National
computational overhead (see Sec. 2 for more discussions).                                                                              Science Foundation under grants EAR-1520870 and DMS-
  8
                                                                                                                                       1418265. Computational support was provided by the Ohio
    We use the source code from the authors: https://github.com/
GTmac/HARP. Results on Node2Vec are similar and hence omitted.
                                                                                                                                       Supercomputer Center under grant PAS0166. All content
MILE: A Multi-Level Framework for Scalable Graph Embedding                Conference’18, Jan 2018, DC, Washington USA


represents the opinion of the authors, which is not necessarily
shared or endorsed by their sponsors.

REFERENCES
 [1] N. K. Ahmed and et al. A framework for generalizing graph-based
     representation learning methods. In arXiv’17’.
 [2] V. D. Blondel and et al. Fast unfolding of communities in large
     networks. In J. Stat. Mech. Theory Exp. ’08.
 [3] S. Cao, W. Lu, and Q. Xu. Grarep: Learning graph representations
     with global structural information. In CIKM’15.
 [4] S. Chang and et al. Heterogeneous network embedding via deep
     architectures. In KDD’15.
 [5] H. Chen, B. Perozzi, Y. Hu, and S. Skiena. Harp: Hierarchical
     representation learning for networks. In AAAI’18.
 [6] A. Ching, S. Edunov, M. Kabiljo, D. Logothetis, and S. Muthukr-
     ishnan. One trillion edges: Graph processing at facebook-scale.
     In VLDB’15.
 [7] I. S. Dhillon, Y. Guan, and B. Kulis. Weighted graph cuts without
     eigenvectors a multilevel approach. In PAMI’07.
 [8] Y. Dong, N. V. Chawla, and A. Swami. metapath2vec: Scalable
     representation learning for heterogeneous networks. In KDD’17.
 [9] A. Grover and J. Leskovec. node2vec: Scalable feature learning
     for networks. In KDD’16.
[10] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation
     learning on large graphs. In NIPS’17.
[11] W. L. Hamilton, J. Leskovec, and D. Jurafsky. Diachronic word
     embeddings reveal statistical laws of semantic change. In ACL’16.
[12] M. Henaff, J. Bruna, and Y. LeCun. Deep convolutional networks
     on graph-structured data. In arXiv’15’.
[13] X. Huang, J. Li, and X. Hu. Accelerated attributed network
     embedding. In SDM’17.
[14] G. Karypis and V. Kumar. Multilevelk-way partitioning scheme
     for irregular graphs. In JPDC’98.
[15] T. N. Kipf and M. Welling. Semi-supervised classification with
     graph convolutional networks. In ICLR’17.
[16] J. Li, C. Chen, H. Tong, and H. Liu. Multi-layered network
     embedding. In SDM’18.
[17] J. Liang, P. Jacobs, J. Sun, and S. Parthasarathy. Semi-supervised
     embedding in attributed networks with outliers. In SDM’18.
[18] W. Liu, P.-y. Chen, S. Yeung, T. Suzumura, and L. Chen. Princi-
     pled multilayer network embedding. In ICDM’17.
[19] S. Pan, J. Wu, X. Zhu, C. Zhang, and Y. Wang. Tri-party deep
     network representation. In IJCAI’16.
[20] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk: Online learning
     of social representations. In KDD’14.
[21] J. Qiu, Y. Dong, H. Ma, J. Li, K. Wang, and J. Tang. Network
     embedding as matrix factorization: Unifyingdeepwalk, line, pte,
     and node2vec. In WSDM’18.
[22] Y. Ruan, D. Fuhry, J. Liang, Y. Wang, and S. Parthasarathy.
     Community discovery: Simple and scalable approaches. In User
     Community Discovery. 2015.
[23] V. Satuluri and S. Parthasarathy. Scalable graph clustering using
     stochastic flows: applications to community discovery. In KDD’09.
[24] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line:
     Large-scale information network embedding. In WWW’15.
[25] D. Wang, P. Cui, and W. Zhu. Structural deep network embedding.
     In KDD’2016.
[26] C. Yang, Z. Liu, D. Zhao, M. Sun, and E. Y. Chang. Network
     representation learning with rich text information. In IJCAI’15.
[27] C. Yang, M. Sun, Z. Liu, and C. Tu. Fast network embedding en-
     hancement via high order proximity approximation. In IJCAI’17.

