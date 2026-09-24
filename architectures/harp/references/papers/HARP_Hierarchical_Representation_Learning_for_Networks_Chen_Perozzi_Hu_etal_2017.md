# HARP Hierarchical Representation Learning for Networks Chen Perozzi Hu etal 2017

> Source: `HARP_Hierarchical_Representation_Learning_for_Networks_Chen_Perozzi_Hu_etal_2017.pdf`

---

                                                            HARP: Hierarchical Representation Learning for Networks
                                                                       Haochen Chen                                           Bryan Perozzi
                                                                    Stony Brook University                                    Google Research
                                                                 haocchen@cs.stonybrook.edu                                  bperozzi@acm.org

                                                                           Yifan Hu                                           Steven Skiena
                                                                        Yahoo! Research                                   Stony Brook University
                                                                       yifanhu@oath.com                                  skiena@cs.stonybrook.edu




arXiv:1706.07845v2 [cs.SI] 16 Nov 2017
                                                                     Abstract
                                           We present HARP, a novel method for learning low dimen-
                                           sional embeddings of a graph’s nodes which preserves higher-
                                           order structural features. Our proposed method achieves this
                                           by compressing the input graph prior to embedding it, effec-
                                           tively avoiding troublesome embedding configurations (i.e.
                                           local minima) which can pose problems to non-convex op-
                                           timization.                                                             (a) Can 187           (b) LINE            (c) HARP
                                           HARP works by finding a smaller graph which approximates
                                           the global structure of its input. This simplified graph is used
                                           to learn a set of initial representations, which serve as good
                                           initializations for learning representations in the original, de-
                                           tailed graph. We inductively extend this idea, by decompos-
                                           ing a graph in a series of levels, and then embed the hierarchy
                                           of graphs from the coarsest one to the original graph.
                                           HARP is a general meta-strategy to improve all of the state-
                                           of-the-art neural algorithms for embedding graphs, including           (d) Poisson 2D         (e) LINE            (f) HARP
                                           DeepWalk, LINE, and Node2vec. Indeed, we demonstrate that
                                           applying HARP’s hierarchical paradigm yields improved im-           Figure 1: Comparison of two-dimensional embeddings from
                                           plementations for all three of these methods, as evaluated on       LINE and our proposed method, for two distinct graphs. Ob-
                                           classification tasks on real-world graphs such as DBLP, Blog-       serve how HARP’s embedding better preserves the higher
                                           Catalog, and CiteSeer, where we achieve a performance gain          order structure of a ring and a plane.
                                           over the original implementations by up to 14% Macro F1.


                                                                 Introduction                                  then be used as features for common tasks on graphs such as
                                         From social networks to the World Wide Web, graphs are a              multi-label classification, clustering, and link prediction.
                                         ubiquitous way to organize a diverse set of real-world infor-            Traditional methods for graph dimensionality reduction
                                         mation. Given a network’s structure, it is often desirable to         (Belkin and Niyogi 2001; Roweis and Saul 2000; Tenen-
                                         predict missing information (frequently called attributes or          baum, De Silva, and Langford 2000) perform well on small
                                         labels) associated with each node in the graph. This missing          graphs. However, the time complexity of these methods are
                                         information can represent a variety of aspects of the data –          at least quadratic in the number of graph nodes, makes them
                                         for example, on a social network they could represent the             impossible to run on large-scale networks.
                                         communities a person belongs to, or the categories of a doc-             A recent advancement in graph representation learning,
                                         ument’s content on the web.                                           DeepWalk (Perozzi, Al-Rfou, and Skiena 2014) proposed
                                            Because many information networks can contain billions             online learning methods using neural networks to address
                                         of nodes and edges, it can be intractable to perform complex          this scalability limitation. Much work has since followed
                                         inference procedures on the entire network. One technique             (Cao, Lu, and Xu 2015; Grover and Leskovec 2016; Perozzi
                                         which has been proposed to address this problem is dimen-             et al. 2017; Tang et al. 2015). These neural network-based
                                         sionality reduction. The central idea is to find a mapping            methods have proven both highly scalable and performant,
                                         function which converts each node in the graph to a low-              achieving strong results on classification and link prediction
                                         dimensional latent representation. These representations can          tasks in large networks.
                                                                                                                  Despite their success, all these methods have several
                                         Copyright c 2018, Association for the Advancement of Artificial       shared weaknesses. Firstly, they are all local approaches –
                                         Intelligence (www.aaai.org). All rights reserved.                     limited to the structure immediately around a node. Deep-
Walk (Perozzi, Al-Rfou, and Skiena 2014) and Node2vec                sification tasks on several real-world networks, with im-
(Grover and Leskovec 2016) adopt short random walks to               provements as large as 14% Macro F1 .
explore the local neighborhoods of nodes, while LINE (Tang
et al. 2015) is concerned with even closer relationships                            Problem Formulation
(nodes at most two hops away). This focus on local structure
implicitly ignores long-distance global relationships, and the     We desire to learn latent representations of nodes in a
learned representations can fail to uncover important global       graph. Formally, let G = (V, E) be a graph, where V is
structural patterns. Secondly, they all rely on a non-convex       the set of nodes and E is the set of edges. The goal of
optimization goal solved using stochastic gradient descent         graph representation learning is to develop a mapping func-
(Goldberg and Levy 2014; Mikolov et al. 2013) which can            tion Φ : V 7→ R|V |×d , d  |V |. This mapping Φ de-
become stuck in a local minima (e.g. perhaps as a result           fines the latent representation (or embedding) of each node
of a poor initialization). In other words, all previously pro-     v ∈ V . Popular methods for learning the parameters of
posed techniques for graph representation learning can ac-         Φ (Perozzi, Al-Rfou, and Skiena 2014; Tang et al. 2015;
cidentally learn embedding configurations which disregard          Grover and Leskovec 2016) suffer from two main disadvan-
important structural features of their input graph.                tages: (1) higher-order graph structural information is not
   In this work, we propose HARP, a meta strategy for em-          modeled, and (2) their stochastic optimization can fall vic-
bedding graph datasets which preserves higher-order struc-         tim to poor initialization.
tural features. HARP recursively coalesces the nodes and              In light of these difficulties, we introduce the hierarchi-
edges in the original graph to get a series of succes-             cal representation learning problem for graphs. At its core,
sively smaller graphs with similar structure. These coalesced      we seek to find a graph, Gs = (Vs , Es ) which captures the
graphs, each with a different granularity, provide us a view       essential structure of G, but is smaller than our original (i.e.
of the original graph’s global structure. Starting from the        |Vs | << |V |, |Es | << |E|). It is likely that Gs will be easier
most simplified form, each graph is used to learn a set of         to embed for two reasons. First, there are many less pairwise
initial representations which serve as good initializations for    relationships (|Vs |2 versus |V |2 ) which can be expressed in
embedding the next, more detailed graph. This process is re-       the space. As the sample space shrinks, there is less variation
peated until we get an embedding for each node in the orig-        in training examples – this can yield a smoother objective
inal graph.                                                        function which is easier to optimize. Second, the diameter
   We illustrate the effectiveness of this multilevel paradigm     of Gs may be smaller than G, so algorithms with a local
in Figure 1, by visualizing the two-dimension embeddings           focus can exploit the graph’s global structure.
from an existing method (LINE (Tang et al. 2015)) and our             In summary, we define the hierarchical representation
improvement to it, HARP(LINE). Each of the small graphs            learning problem in graphs as follows:
we consider has an obvious global structure (that of a ring       Given a large graph G(V, E) and a function f , which embeds
(1a) and a grid (1d)) which is easily exposed by a force             G using initialization θ, f : G × θ 7→ ΦG ,
direced layout (Hu 2005). The center figures represent the         Simplify G to a series of successively smaller graphs
two-dimensional embedding obtained by LINE for the ring              G0 . . . GL ,
(1b) and grid (1e). In these embeddings, the global struc-        Learn a coarse embedding ΦGL = f (GL , ∅),
ture is lost (i.e. that is, the ring and plane are unidenti-      Refine the coarse embedding into ΦG by iteratively applying
fiable). However, the embeddings produced by using our               ΦGi = f (Gi , ΦGi+1 ), 0 ≤ i < L.
meta-strategy to improve LINE (right) clearly capture both
the local and global structure of the given graphs (1c, 1f).
   Our contributions are the following:                                                       Method
• New Representation Learning Paradigm. We propose                 Here we present our hierarchical paradigm for graph repre-
  HARP, a novel multilevel paradigm for graph representa-          sentation learning. After discussing the method in general,
  tion which seamlessly blends ideas from the graph draw-          we present a structure-preserving algorithm for its most cru-
  ing (Fruchterman and Reingold 1991) and graph repre-             cial step, graph coarsening.
  sentation learning (Perozzi, Al-Rfou, and Skiena 2014;
  Tang et al. 2015; Grover and Leskovec 2016) communi-             Algorithm: HARP
  ties to build substantially better graph embeddings.             Our method for multi-level graph representation learning,
• Improved Optimization Primitives. We demonstrate                 HARP, is presented in Algorithm 1. It consists of three parts
  that our approach leads to improved implementations of           - graph coarsening, graph embedding, and representation re-
  all state-of-the-art graph representation learning methods,      finement - which we detail below:
  namely DeepWalk (DW), LINE and Node2vec (N2V). Our              1. Graph Coarsening (line 1): Given a graph G, graph
  improvements on these popular methods for learning la-             coarsening algorithms create a hierarchy of successively
  tent representations illustrate the broad applicability of         smaller graphs G0 , G1 , · · · , GL , where G0 = G. The
  our hierarchical approach.                                         coarser (smaller) graphs preserve the global structure of
• Better Embeddings for Downstream Tasks. We demon-                  the original graph, yet have significantly fewer nodes and
  strate that HARP(DW), HARP(LINE) and HARP(N2V)                     edges. Algorithms for generating this hierarchy of graphs
  embeddings consistently outperform the originals on clas-          will be discussed in detail below.
              (a) Edge Collapsing.               (b) Edge Collapsing fails to collapse stars.            (c) Star Collapsing.

Figure 2: Illustration of graph coarsening algorithms. 2a: Edge collapsing on a graph snippet. 2b: How edge collapsing fails to
coalesce star-like structures. 2c: How star collapsing scheme coalesces the same graph snippet efficiently.


Algorithm 1 HARP(G, Embed())                                              Algorithm 2 GraphCoarsening(G)
Input:                                                                    Input: graph G(V, E)
    graph G(V, E)                                                         Output: Series of Coarsened Graphs G0 , G1 , · · · , GL
    arbitrary graph embedding algorithm E MBED()                           1: L ← 0
Output: matrix of vertex representations Φ ∈ R|V |×d                       2: G0 ← G
 1: G0 , G1 , · · · , GL ← G RAPH C OARSENING(G)                           3: while |VL | ≥ threshold do
 2: Initialize Φ0GL by assigning zeros                                     4:     L←L+1
 3: ΦGL ← E MBED(GL , Φ0GL )                                               5:     GL ← E DGE C OLLAPSE(S TAR C OLLAPSE(G))
 4: for i = L − 1 to 0 do                                                  6: end while
 5:     Φ0Gi ← P ROLONGATE(ΦGi+1 , Gi+1 , Gi )                             7: return G0 , G1 , · · · , GL
 6:     ΦGi ← E MBED(Gi , Φ0Gi )
 7: end for
 8: return ΦG0                                                            based on the shared neighborhood structure of the nodes.
                                                                             Edge Collapsing. Edge collapsing (Hu 2005) is an effi-
                                                                          cient algorithm for preserving first-order proximity. It se-
2. Graph Embedding on the Coarsest Graph (line 2-3): The                  lects E 0 ⊆ E, such that no two edges in E 0 are incident
   graph embedding is obtained on the coarsest graph GL                   to the same vertex. Then, for each (ui , vi ) ∈ E 0 , it merges
   with the provided graph embedding algorithm. As the size               (ui , vi ) into a single node wi , and merge the edges incident
   of GL is usually very small, it is much easier to get a high-          to ui and vi . The number of nodes in the coarser graph is
   quality graph representation.                                          therefore at least half of that in the original graph. As il-
                                                                          lustrated in Figure 2a, the edge collapsing algorithm merges
3. Graph Representation Prolongation and Refinement (line                 node pairs (v1 , v2 ) and (v3 , v4 ) into supernodes v1,2 and v3,4
   4-7): We prolong and refine the graph representation from              respectively, resulting in a coarser graph with 2 nodes and 1
   the coarsest to the finest graph. For each graph Gi , we               edge. The order of merging is arbitrary; we find different
   prolong the graph representation of Gi+1 as its initial em-            merging orders result in very similar node embeddings in
   bedding Φ0Gi . Then, the embedding algorithm Embed()                   practice.
   is applied to (Gi , Φ0Gi ) to further refine Φ0Gi , resulting in          Star Collapsing. Real world graphs are often scale-free,
   the refined embedding ΦGi . We discuss this step in the                which means they contain a large number of star-like struc-
   embedding prolongation section below.                                  tures. A star consists of a popular central node (sometimes
4. Graph Embedding of the Original Graph (line 8): We re-                 referred to as hubs) connected to many peripheral nodes. Al-
   turn ΦG0 , which is the graph embedding of the original                though the edge collapsing algorithm is simple and efficient,
   graph.                                                                 it cannot sufficiently compress the star-like structures in a
                                                                          graph. Consider the graph snippet in Figure 2b, where the
   We can easily see that this paradigm is algorithm inde-                only central node v7 connects to all the other nodes. As-
pendent, relying only on the provided functions Embed().                  sume the degree of the central node is k, it is clear that the
Thus, with minimum effort, this paradigm can be incorpo-                  edge collapsing scheme can only compress this graph into
rated into any existing graph representation learning meth-               a coarsened graph with k − 1 nodes. Therefore when k is
ods, yielding a multilevel version of that method.                        large, the coarsening process could be arbitrarily slow, takes
                                                                          O(k) steps instead of O(log k) steps.
Graph Coarsening                                                             One observation on the star structure is that there are
In Algorithm 2, we develop a hybrid graph coarsening                      strong second-order similarities between the peripheral
scheme which preserves global graph structural information                nodes since they share the same neighborhood. This leads
at different scales. Its two key parts, namely edge collaps-              to our star collapsing scheme, which merges nodes with the
ing and star collapsing, preserve first-order proximity and               same neighbors into supernodes since they are similar to
second-order proximity (Tang et al. 2015) respectively. First-            each other. As shown in Figure 2c, (v1 , v2 ), (v3 , v4 ) and
order proximity is concerned with preserving the observed                 (v5 , v6 ) are merged into supernodes as they share the same
edges in the input graph, while second-order proximity is                 neighbors (v7 ), generating a coarsened graph with only k/2
nodes.                                                                     Name            DBLP          Blogcatalog        CiteSeer
   Hybrid Coarsening Scheme. By combining edge col-                        # Vertices      29,199           10,312            3,312
lapsing and star collapsing, we present a hybrid scheme for                # Edges        133,664          333,983            4,732
graph coarsening in Algorithm 2, which is adopted on all                   # Classes          4               39                6
test graphs. In each coarsening step, the hybrid coarsening                Task         Classification   Classification   Classification
scheme first decomposes the input graph with star collaps-
ing, then adopts the edge collapsing scheme to generate the               Table 1: Statistics of the graphs used in our experiments.
coalesced graph. We repeat this process until a small enough
graph (with less than 100 vertices) is obtained.
                                                                         • DBLP (Perozzi et al. 2017) – DBLP is a co-author graph
Embedding Prolongation                                                     of researchers in computer science. The labels indicate the
After the graph representation for Gi+1 is learned, we pro-                research areas a researcher publishes his work in. The 4
long it into the initial representation for Gi . We observe that           research areas included in this dataset are DB, DM, IR,
each node v ∈ Gi+1 is either a member of the finer represen-               and ML.
tation (v ∈ Gi ), or the result of a merger, (v1 , v2 , · · · , vk ) ∈   • BlogCatalog (Tang and Liu 2009) – BlogCatalog is a net-
Gi . In both cases, we can simply reuse the representation of              work of social relationships between users on the Blog-
the parent node v ∈ Gi - the children are quickly separated                Catalog website. The labels represent the categories a
by gradient updates.                                                       blogger publishes in.
                                                                         • CiteSeer (Sen et al. 2008) – CiteSeer is a citation network
Complexity Analysis                                                        between publications in computer science. The labels in-
In this section, we discuss the time complexity of                         dicate the research areas a paper belongs to. The papers
HARP(DW) and HARP(LINE) and compare with the                               are classified into 6 categories: Agents, AI, DB, IR, ML,
time complexity of DeepWalk and LINE respectively.                         and HCI.
HARP(N2V) has the same time complexity as HARP(DW),
thus it is not included in the discussion below.                         Baseline Methods
HARP(DW): Given the number of random walks γ, walk                       We compare our model with the following graph embedding
length t, window size w and representation size d, the                   methods:
time complexity of DeepWalk is dominated by the train-                   • DeepWalk — DeepWalk is a two-phase method for em-
ing time of the Skip-gram model, which is O(γ|V |tw(d +                    bedding graphs. Firstly, DeepWalk generates random
dlog|V |)). For HARP(DW), coarsening a graph with |V |                     walks of fixed length from all the vertices of a graph.
nodes produces a coarser graph with about |V |/2 nodes.                    Then, the walks are treated as sentences in a language
The total number of nodes in all levels is approximately                   model and the Skip-Gram model for learning word em-
     Plog |V |
|V | i=02 ( 21 )i = 2|V |. Therefore, the time complex-                    beddings is utilized to obtain graph embeddings. Deep-
ity of HARP(DW) is O(|V |) for copying binary tree and                     Walk uses hierarchical softmax for Skip-gram model op-
O(γ|V |tw(d+dlog|V |)) for model training. Thus, the over-                 timization.
all time complexity of HARP(DW) is also O(γ|V |tw(d +
dlog|V |)).                                                              • LINE — LINE is a method for embedding large-scale net-
                                                                           works. The objective function of LINE is designed for pre-
HARP(LINE): The time complexity of LINE is linear to the
                                                                           serving both first-order and second-order proximities, and
number of edges in the graph and the number of iterations
                                                                           we use first-order LINE for comparison. Skip-gram with
r over edges, which is O(r|E|). For HARP(LINE), coars-
                                                                           negative sampling is used to solve the objective function.
ening a graph with |E| nodes produces a coarsened graph
with about |E|/2 edges. The total number edges in all levels             • Node2vec — Node2vec proposes an improvement to the
                       Plog |E|
is approximately |E| i=02 ( 12 )i = 2|E|. Thus, the time                   random walk phase of DeepWalk. By introducing the re-
complexity of HARP(LINE) is also O(r|E|).                                  turn parameter p and the in-out parameter q, Node2vec
                                                                           combines DFS-like and BFS-like neighborhood explo-
                                                                           ration. Node2vec also uses negative sampling for optimiz-
                          Experiment                                       ing the Skip-gram model.
In this section, we provide an overview of the datasets and              For each baseline method, we combine it with HARP and
methods used for experiments and evaluate the effectiveness              compare their performance.
of our method on challenging multi-label classification tasks
in several real-life networks. We further illustrate the scala-          Parameter Settings
bility of our method and discuss its performance with regard
to several important parameters.                                         Here we discuss the parameter settings for our models and
                                                                         baseline models. Since DeepWalk, LINE and Node2vec are
                                                                         all sampling based algorithms, we always ensure that the
Datasets                                                                 total number of samples seen by the baseline algorithm is
Table 1 gives an overview of the datasets used in our exper-             the same as that of the corresponding HARP enhanced algo-
iments.                                                                  rithm.
 1.0                                            1.0                                            1.0
                          Relative # of Nodes                            Relative # of Nodes                             Relative # of Nodes
 0.8                      Relative # of Edges   0.8                      Relative # of Edges   0.8                       Relative # of Edges

 0.6                                            0.6                                            0.6

 0.4                                            0.4                                            0.4

 0.2                                            0.2                                            0.2

 0.0                                            0.0                                            0.0
       0   2        4        6         8              0   2        4        6         8              0    2        4        6         8
               Coarsening Level                               Coarsening Level                                Coarsening Level


                                                                                                                                                            (a) Level 7   (b) Level 6   (c) Level 5
           (a) DBLP                                   (b) BlogCatalog                                    (c) CiteSeer

Figure 3: The ratio of nodes/edges of the coarsened graphs to
that of the original test graphs. For disconnected graphs, the
graph coarsening result on the largest connected component
is shown.
                                                                                                                                                            (d) Level 4   (e) Level 3   (f) Level 2


   DeepWalk. For DeepWalk and HARP(DW), we need to
set the following parameters: the number of random walks γ,
walk length t, window size w for the Skip-gram model and
representation size d. In HARP(DW), the parameter setting                                                                                                   (g) Level 1   (h) Level 0    (i) Input
is γ = 40, t = 10, w = 10, d = 128. For DeepWalk, all the
parameters except γ are the same as in HARP(DW). Specifi-                                                                                      Figure 4: Two-dimensional embeddings generated with
cally, to ensure a fair comparison, we increase the value of γ                                                                                 HARP(LINE) on different coarsening levels on Poisson 2D.
for DeepWalk. This gives DeepWalk a larger training dataset                                                                                    Level 7 denotes the smallest graph, while level 0 denotes the
(as large as all of the levels of HARP(DW) combined). We                                                                                       original graph. The last subfigure is the graph layout gener-
note that failure to increase γ in this way resulted in substan-                                                                               ated by a force-direct graph drawing algorithm.
tially worse DeepWalk (and Node2vec) models.
   LINE. For HARP(LINE), we run 50 iterations on all graph
edges on all coarsening levels. For LINE, we increase the                                                                                      Multi-label Classification
number of iterations over graph edges accordingly, so that                                                                                     We evaluate our method using the same experimental pro-
the amount of training data for both models remain the same.                                                                                   cedure in (Perozzi, Al-Rfou, and Skiena 2014). Firstly, we
The representation size d is set to 64 for both LINE and                                                                                       obtain the graph embeddings of the input graph. Then, a por-
HARP(LINE).                                                                                                                                    tion (TR ) of nodes along with their labels are randomly sam-
   Node2vec. For HARP(N2V), the parameter setting is γ =                                                                                       pled from the graph as training data, and the task is to predict
40, t = 10, w = 10, d = 128. Similar to DeepWalk, we                                                                                           the labels for the remaining nodes. We train a one-vs-rest lo-
increase the value of γ in Node2vec to ensure a fair compar-                                                                                   gistic regression model with L2 regularization on the graph
ison. Both in-out and return hyperparameters are set to 1.0.                                                                                   embeddings for prediction. The logistic regression model is
For all models, the initial learning rate and final learning rate                                                                              implemented by LibLinear (Fan et al. 2008). To ensure the
are set to 0.025 and 0.001 respectively.                                                                                                       reliability of our experiment, the above process is repeated
                                                                                                                                               for 10 times, and the average Macro F1 score is reported.
Graph Coarsening                                                                                                                               The other evaluation metrics such as Micro F1 score and ac-
                                                                                                                                               curacy follow the same trend as Macro F1 score, thus are not
Figure 3 demonstrates the effect of our hybrid coarsening                                                                                      shown.
method on all test graphs. The first step of graph coarsen-                                                                                       Table 2 reports the Macro F1 scores achieved on DBLP,
ing for each graph eliminates about half the nodes, but the                                                                                    BlogCatalog, and CiteSeer with 5%, 50%, and 5% labeled
number of edges only reduce by about 10% for BlogCata-                                                                                         nodes respectively. The number of class labels of BlogCat-
log. This illustrates the difficulty of coarsening real-world                                                                                  alog is about 10 times that of the other two graphs, thus
graphs. However, as the graph coarsening process contin-                                                                                       we use a larger portion of labeled nodes. We can see that
ues, the scale of all graphs drastically decrease. At level 8,                                                                                 our method improves all existing neural embedding tech-
all graphs have less than 10% nodes and edges left.                                                                                            niques on all test graphs. In DBLP, the improvements in-
                                                                                                                                               troduced by HARP(DW), HARP(LINE) and HARP(N2V) are
Visualization                                                                                                                                  7.8%, 3.0% and 0.3% respectively. Given the scale-free na-
                                                                                                                                               ture of BlogCatalog, graph coarsening is much harder due to
To show the intuition of the HARP paradigm, we set                                                                                             a large amount of star-like structures in it. Still, HARP(DW),
d = 2, and visualize the graph representation generated by                                                                                     HARP(LINE) and HARP(N2V) achieve gains of 4.0%, 4.6%
HARP(LINE) at each level.                                                                                                                      and 4.7% over the corresponding baseline methods respec-
   Figure 4 shows the level-wise 2D graph embeddings ob-                                                                                       tively. For CiteSeer, the performance improvement is also
tained with HARP(LINE) on Poisson 2D. The graph layout                                                                                         striking: HARP(DW), HARP(LINE) and HARP(N2V) out-
of level 5 (which has only 21 nodes) already highly resem-                                                                                     performs the baseline methods by 4.8%, 13.6%, and 2.8%.
bles the layout of the original graph. The graph layout on                                                                                        To have a detailed comparison between HARP and the
each subsequent level is initialized with the prolongation of                                                                                  baseline methods, we vary the portion of labeled nodes
the previous graph layout, thus the global structure is kept.                                                                                  for classification, and present the macro F1 scores in Fig-
                    0.64                                                 0.64                                                        0.64




                                                                                                                                                                                       Macro F1 Score
                    0.62                                                 0.62                                                        0.62

                    0.60                                                 0.60                                                        0.60


      DBLP
                    0.58                                                 0.58                                                        0.58

                    0.56                                                 0.56                                                        0.56

                    0.54                                                 0.54                                                        0.54
                                              HARP(DeepWalk)                                                     HARP(LINE)                                         HARP(Node2vec)
                    0.52                      DeepWalk                   0.52                                    LINE                0.52                           Node2vec
                    0.50                                                 0.50                                                        0.50
                       0.00   0.02     0.04    0.06     0.08      0.10      0.00     0.02        0.04     0.06         0.08   0.10      0.00   0.02       0.04     0.06     0.08      0.10
                                 Fraction of Labeled Data                                   Fraction of Labeled Data                                 Fraction of Labeled Data

                    0.30                                                 0.30                                                        0.30

                    0.28                                                 0.28                                                        0.28




      BlogCatalog                                                                                                                                                                      Macro F1 Score
                    0.26                                                 0.26                                                        0.26

                    0.24                                                 0.24                                                        0.24

                    0.22                                                 0.22                                                        0.22

                    0.20                                                 0.20                                                        0.20

                    0.18                       HARP(DeepWalk)            0.18                                    HARP(LINE)          0.18                           HARP(Node2vec)
                    0.16
                                               DeepWalk                  0.16
                                                                                                                 LINE                0.16
                                                                                                                                                                    Node2vec

                       0.0    0.2       0.4     0.6         0.8    1.0      0.0       0.2         0.4      0.6          0.8    1.0      0.0    0.2         0.4      0.6         0.8   1.0
                                 Fraction of Labeled Data                                   Fraction of Labeled Data                                 Fraction of Labeled Data

                    0.50                                                 0.50                                                        0.50




                                                                                                                                                                                       Macro F1 Score
                    0.45                                                 0.45                                                        0.45




      CiteSeer
                    0.40                                                 0.40                                                        0.40


                    0.35                                                 0.35                                                        0.35


                    0.30                      HARP(DeepWalk)             0.30                                    HARP(LINE)          0.30                           HARP(Node2vec)
                                              DeepWalk                                                           LINE                                               Node2vec
                    0.25                                                 0.25                                                        0.25
                       0.00   0.02     0.04    0.06     0.08      0.10      0.00     0.02        0.04     0.06         0.08   0.10      0.00   0.02       0.04     0.06     0.08      0.10
                                 Fraction of Labeled Data                                   Fraction of Labeled Data                                 Fraction of Labeled Data



                                    Figure 5: Detailed multi-label classification result on DBLP, BlogCatalog, and CiteSeer.


     Algorithm                                           Dataset                                           Walk with 8% label data. HARP(LINE) also consistently
                                        DBLP           BlogCatalog                CiteSeer                 outperforms LINE given any amount of training data, with
                                                                                                           macro F1 score gain between 1% and 3%. HARP(N2V)
    DeepWalk                            57.29          24.88                      42.72                    and Node2vec have comparable performance with less than
   HARP(DW)                             61.76∗         25.90∗                     44.78∗                   5% labeled data, but as the ratio of labeled data increases,
 Gain of HARP[%]                        7.8            4.0                        4.8                      HARP(N2V) eventually distances itself to a 0.7% improve-
       LINE                             57.76          22.43                      37.11                    ment over Node2vec. We can also see that Node2vec gen-
   HARP(LINE)                           59.51∗         23.47∗                     42.95∗                   erally has better performance when compared to DeepWalk,
 Gain of HARP[%]                        3.0            4.6                        13.6                     and the same holds for HARP(N2V) and HARP(DW). The
                                                                                                           difference in optimization method for Skip-gram (negative
     Node2vec                           62.64          23.55                      44.84
                                                                                                           sampling for Node2vec and hierarchical softmax for Deep-
   HARP(N2V)                            62.80          24.66∗                     46.08∗
                                                                                                           Walk) may account for this difference.
 Gain of HARP[%]                        0.3            4.7                        2.8
                                                                                                              BlogCatalog. As a scale-free network with complex
Table 2: Macro F1 scores and performance gain of HARP                                                      structure, BlogCatalog is challenging for graph coarsening.
on DBLP, BlogCatalog, and CiteSeer in percentage. * indi-                                                  Still, by considering both first-order proximity and second-
cates statistically superior performance to the corresponding                                              order proximity, our hybrid coarsening algorithm generates
baseline method at level of 0.001 using a standard paired                                                  an appropriate hierarchy of coarsened graphs. With the same
t-test. Our method improves all existing neural embedding                                                  amount of training data, HARP(DW) always leads by at
techniques.                                                                                                least 3.0%. For HARP(LINE), it achieves a relative gain of
                                                                                                           4.8% with 80% labeled data. For HARP(N2V), its gain over
                                                                                                           Node2vec reaches 4.7% given 50% labeled nodes.
ure 5. We can observe that HARP(DW), HARP(LINE) and
HARP(N2V) consistently perform better than the corre-                                                         Citeseer. For CiteSeer, the lead of HARP(DW) on Macro
sponding baseline methods.                                                                                 F1 score varies between 5.7% and 7.8%. For HARP(LINE),
  DBLP. For DBLP, the relative gain of HARP(DW) is                                                         its improvement over LINE with 4% labeled data is an
over 9% with 4% labeled data. With only 2% labeled data,                                                   impressive 24.4%. HARP(N2V) also performs better than
HARP(DW) achieves higher macro F1 score than Deep-                                                         Node2vec on any ratio of labeled nodes.
               700                                    666
                                                                                                            213
               600
                                                649
                                                                            DeepWalk
                                                                            HARP(Deepwalk)                  211
                                                                                                                                                                      adjacency matrix. Node2vec (Grover and Leskovec 2016)
                                                                            LINE

               500                             488
                                                                            HARP(LINE)
                                                                                                             29
                                                                                                             27
                                                                                                                                                                      combines DFS-like and BFS-like exploration within the ran-
                                                                                             Run Time (s)
                                             466            463             Node2vec
                                                                            HARP(Node2vec)
                                                                                                                                                                      dom walk framework. Second, matrix factorization methods
Run Time (s)
                                                       410                                                   25
               400
                                                                                                             23                                LINE

               300
                     246
                           263
                                                                                                             21
                                                                                                            2−1
                                                                                                                                               HARP(LINE)
                                                                                                                                               Node2vec
                                                                                                                                                                      and deep neural networks have also been proposed (Cao, Lu,
               200               174
                                       190
                                                                                                            2−3
                                                                                                                                               HARP(Node2vec)
                                                                                                                                               DeepWalk               and Xu 2015; Ou et al. 2016; Wang, Cui, and Zhu 2016;
                                                                                                            2−5                                HARP(DeepWalk)
               100          86 91

                                                                  20 25                                           101   102      103     104       105          106
                                                                                                                                                                      Abu-El-Haija, Perozzi, and Al-Rfou 2017) as alternatives to
                                                                          5 5 15 19
                                                                                                                              Number of Nodes
                 0
                           DBLP              Blogcatalog            CiteSeer                                                                                          the Skip-gram model for learning the latent representations.
                           (a) Test graphs.                                                                   (b) Erdos-Renyi graphs.                                    Although these methods are highly scalable, they all rely
                                                                                                                                                                      on optimizing a non-convex objective function. With no
                                                        Figure 6: Runtime analysis.                                                                                   prior knowledge of the graph, the latent representations are
                                                                                                                                                                      usually initialized with random numbers or zero. With such
                                                                                                                                                                      an initialization scheme, these methods are at risk of con-
Scalability                                                                                                                                                           verging to a poor local minima. HARP overcomes this prob-
                                                                                                                                                                      lem by introducing a multilevel paradigm for graph repre-
We already shown that introducing HARP does not affect                                                                                                                sentation learning.
the time complexity of the underlying graph embedding al-
                                                                                                                                                                      Graph Drawing. Multilevel layout algorithms are popular
gorithms. Here, we compare the actual run time of HARP en-
                                                                                                                                                                      methods in the graph drawing community, where a hier-
hanced embedding algorithms with the corresponding base-
                                                                                                                                                                      archy of approximations is used to solve the original lay-
line methods on all test graphs. All models run on a sin-
                                                                                                                                                                      out problem (Fruchterman and Reingold 1991; Hu 2005;
gle machine with 128GB memory, 24 CPU cores at 2.0GHZ
                                                                                                                                                                      Walshaw 2003). Using an approximation of the original
with 20 threads. As shown in Figure 6a, applying HARP typ-
                                                                                                                                                                      graph has two advantages - not only is the approximation
ically only introduces an overhead of less than 10% total
                                                                                                                                                                      usually simpler to solve, it can also be extended as a good
running time. The time spent on sampling and training the
                                                                                                                                                                      initialization for solving the original problem. In addition
Skip-gram model dominates the overall running time.
                                                                                                                                                                      to force-directed graph drawing, the multilevel framework
   Additionally, we learn graph embeddings on Erdos-Renyi
                                                                                                                                                                      (Walshaw 2004) has been proved successful in various graph
graphs with node count ranging from 100 to 100,000 and
                                                                                                                                                                      theory problems, including the traveling salesman problem
constant average degree of 10. In Figure 6b, we can observe
                                                                                                                                                                      (Walshaw 2001), and graph partitioning (Karypis and Ku-
that the running time of HARP increases linearly with the
                                                                                                                                                                      mar 1998).
number of nodes in the graph. Also, when compared to the
corresponding baseline method, the overhead introduces by                                                                                                                HARP extends the idea of the multilevel layout to neural
the graph coarsening and prolongation process in HARP is                                                                                                              representation learning methods. We illustrate the utility of
negligible, especially on large-scale graphs.                                                                                                                         this paradigm by combining HARP with three state-of-the-
                                                                                                                                                                      art representation learning methods.
                                                                      Related Work
                                                                                                                                                                                             Conclusion
The related work is in the areas of graph representation
learning and graph drawing, which we briefly describe here.                                                                                                           Recent literature on graph representation learning aims at
Graph Representation Learning. Most early methods                                                                                                                     optimizing a non-convex function. With no prior knowledge
treated representation learning as performing dimension re-                                                                                                           of the graph, these methods could easily get stuck at a bad
duction on the Laplacian and adjacency matrices (Belkin                                                                                                               local minima as the result of poor initialization. Moreover,
and Niyogi 2001; Cox and Cox 2000; Tenenbaum, De Silva,                                                                                                               these methods mostly aim to preserve local proximities in
and Langford 2000). These methods work well on small                                                                                                                  a graph but neglect its global structure. In this paper, we
graphs, but the time complexity of these algorithms is too                                                                                                            propose a multilevel graph representation learning paradigm
high for the large-scale graphs commonly encountered to-                                                                                                              to address these issues. By recursively coalescing the in-
day.                                                                                                                                                                  put graph into smaller but structurally similar graphs, HARP
   Recently, neural network-based methods have been pro-                                                                                                              captures the global structure of the input graph. By learn-
posed for constructing node representation in large-scale                                                                                                             ing graph representation on these smaller graphs, a good
graphs. Deepwalk (Perozzi, Al-Rfou, and Skiena 2014)                                                                                                                  initialization scheme for the input graph is derived. This
presents a two-phase algorithm for graph representation                                                                                                               multilevel paradigm is further combined with the state-of-
learning. In the first phase, Deepwalk samples sequences                                                                                                              the-art graph embedding methods, namely DeepWalk, LINE,
of neighboring nodes of each node by random walking on                                                                                                                and Node2vec. Experimental results on various real-world
the graph. Then, the node representation is learned by train-                                                                                                         graphs show that introducing HARP yields graph embed-
ing a Skip-gram model (Mikolov et al. 2013) on the random                                                                                                             dings of higher quality for all these three methods.
walks. A number of methods have been proposed which ex-                                                                                                                  In the future, we would like to combine HARP with other
tend this idea. First, several methods use different strategies                                                                                                       graph representation learning methods. Specifically, as Skip-
for sampling neighboring nodes. LINE (Tang et al. 2015)                                                                                                               gram is a shallow method for representation learning, it
learns graph embeddings which preserve both the first-order                                                                                                           would be interesting to see if HARP also works well with
and second-order proximities in a graph. Walklets (Per-                                                                                                               deep representation learning methods. On the other hand,
ozzi et al. 2017) captures multiscale node representation on                                                                                                          our method could also be applied to language networks, pos-
graphs by sampling edges from higher powers of the graph                                                                                                              sibly yielding better word embeddings.
                  Acknowledgements                              [Perozzi et al. 2017] Perozzi, B.; Kulkarni, V.; Chen, H.; and
 This work is partially supported by NSF grants IIS-1546113      Skiena, S. 2017. Don’t walk, skip!: Online learning of multi-
 and DBI-1355990.                                                scale network embeddings. In Proceedings of the 2017
                                                                 IEEE/ACM International Conference on Advances in Social
                                                                 Networks Analysis and Mining 2017, ASONAM ’17, 258–
                        References                               265. New York, NY, USA: ACM.
[Abu-El-Haija, Perozzi, and Al-Rfou 2017] Abu-El-Haija,         [Roweis and Saul 2000] Roweis, S. T., and Saul, L. K. 2000.
 S.; Perozzi, B.; and Al-Rfou, R. 2017. Learning edge            Nonlinear dimensionality reduction by locally linear embed-
 representations via low-rank asymmetric projections. arXiv      ding. Science 290(5500):2323–2326.
 preprint arXiv:1705.05615.
                                                                [Sen et al. 2008] Sen, P.; Namata, G. M.; Bilgic, M.; Getoor,
[Belkin and Niyogi 2001] Belkin, M., and Niyogi, P. 2001.        L.; Gallagher, B.; and Eliassi-Rad, T. 2008. Collective clas-
 Laplacian eigenmaps and spectral techniques for embedding       sification in network data. AI Magazine 29(3):93–106.
 and clustering. In NIPS, volume 14, 585–591.
                                                                [Tang and Liu 2009] Tang, L., and Liu, H. 2009. Relational
[Cao, Lu, and Xu 2015] Cao, S.; Lu, W.; and Xu, Q. 2015.         learning via latent social dimensions. In Proceedings of the
 Grarep: Learning graph representations with global struc-       15th ACM SIGKDD international conference on Knowledge
 tural information. In Proceedings of the 24th ACM Interna-      discovery and data mining, 817–826. ACM.
 tional on Conference on Information and Knowledge Man-
                                                                [Tang et al. 2015] Tang, J.; Qu, M.; Wang, M.; Zhang, M.;
 agement, 891–900. ACM.
                                                                 Yan, J.; and Mei, Q. 2015. Line: Large-scale information
[Cox and Cox 2000] Cox, T. F., and Cox, M. A. 2000. Mul-         network embedding. In Proceedings of the 24th Interna-
 tidimensional scaling. CRC press.                               tional Conference on World Wide Web, 1067–1077. Interna-
[Fan et al. 2008] Fan, R.-E.; Chang, K.-W.; Hsieh, C.-J.;        tional World Wide Web Conferences Steering Committee.
 Wang, X.-R.; and Lin, C.-J. 2008. Liblinear: A library for     [Tenenbaum, De Silva, and Langford 2000] Tenenbaum,
 large linear classification. The Journal of Machine Learning    J. B.; De Silva, V.; and Langford, J. C. 2000. A global ge-
 Research 9:1871–1874.                                           ometric framework for nonlinear dimensionality reduction.
[Fruchterman and Reingold 1991] Fruchterman, T. M., and          Science 290(5500):2319–2323.
 Reingold, E. M. 1991. Graph drawing by force-directed          [Walshaw 2001] Walshaw, C. 2001. A multilevel Lin-
 placement. Software: Practice and experience 21(11):1129–       Kernighan-Helsgaun algorithm for the travelling salesman
 1164.                                                           problem. Citeseer.
[Goldberg and Levy 2014] Goldberg, Y., and Levy, O. 2014.       [Walshaw 2003] Walshaw, C. 2003. A multilevel algorithm
 word2vec explained: deriving mikolov et al.’s negative-         for force-directed graph-drawing. Journal of Graph Algo-
 sampling word-embedding method.              arXiv preprint     rithms Applications 7(3):253–285.
 arXiv:1402.3722.                                               [Walshaw 2004] Walshaw, C. 2004. Multilevel refinement
[Grover and Leskovec 2016] Grover, A., and Leskovec, J.          for combinatorial optimisation problems. Annals of Opera-
 2016. node2vec: Scalable feature learning for networks. In      tions Research 131(1-4):325–372.
 Proceedings of the 22nd ACM SIGKDD International Con-          [Wang, Cui, and Zhu 2016] Wang, D.; Cui, P.; and Zhu, W.
 ference on Knowledge Discovery and Data Mining.                 2016. Structural deep network embedding. In Proceed-
[Hu 2005] Hu, Y. 2005. Efficient, high-quality force-directed    ings of the 22nd ACM SIGKDD International Conference
 graph drawing. Mathematica Journal 10(1):37–71.                 on Knowledge Discovery and Data Mining.
[Karypis and Kumar 1998] Karypis, G., and Kumar, V. 1998.
 A parallel algorithm for multilevel graph partitioning and
 sparse matrix ordering. Journal of Parallel and Distributed
 Computing 48(1):71–95.
[Mikolov et al. 2013] Mikolov, T.; Sutskever, I.; Chen, K.;
 Corrado, G. S.; and Dean, J. 2013. Distributed represen-
 tations of words and phrases and their compositionality. In
 Advances in neural information processing systems, 3111–
 3119.
[Ou et al. 2016] Ou, M.; Cui, P.; Pei, J.; and Zhu, W. 2016.
 Asymmetric transitivity preserving graph embedding. In
 Proceedings of the 22nd ACM SIGKDD International Con-
 ference on Knowledge Discovery and Data Mining.
[Perozzi, Al-Rfou, and Skiena 2014] Perozzi, B.; Al-Rfou,
 R.; and Skiena, S. 2014. Deepwalk: Online learning of
 social representations. In Proceedings of the 20th ACM
 SIGKDD international conference on Knowledge discovery
 and data mining, 701–710. ACM.

