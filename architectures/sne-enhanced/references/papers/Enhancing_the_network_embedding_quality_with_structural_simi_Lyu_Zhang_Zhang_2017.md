# Enhancing the network embedding quality with structural simi Lyu Zhang Zhang 2017

> Source: `Enhancing_the_network_embedding_quality_with_structural_simi_Lyu_Zhang_Zhang_2017.pdf`

---

                              Enhancing the Network Embedding Quality
                                      with Structural Similarity
                     Tianshu Lyu                                                    Yuan Zhang                                    Yan Zhang
             Peking University                                             Peking University                                 Peking University
     Department of Machine Intelligence                            Department of Machine Intelligence                Department of Machine Intelligence
          lyutianshu@pku.edu.cn                                           yuan.z@pku.edu.cn                                 zhy@cis.pku.edu.cn

ABSTRACT                                                                                      1   INTRODUCTION
Neural network techniques are widely used in network embedding,                               Network (or graph) is a group of interconnected nodes and contains
boosting the result of node classification, link prediction, visualiza-                       a wealth of information on the relationships between every pair
tion and other tasks in both aspects of efficiency and quality. All the                       of nodes. The analysis of graph is required in almost every field,
state of art algorithms put effort on the neighborhood information                            for instance, online social network [4], biological research [23],
and try to make full use of it. However, it is hard to recognize core                         credit rating [9] and so on. Birds of a feather flock together and
periphery structures simply based on neighborhood.                                            people always have certain characteristics in common with their
    In this paper, we first discuss the influence brought by random-                          friends surrounded by. Therefore, researchers naturally utilize the
walk based sampling strategies to the embedding results. Theo-                                neighborhood information of one node to predict its category.
retical and experimental evidences show that random-walk based                                    Unsupervised network embedding algorithms try to preserve
sampling strategies fail to fully capture structural equivalence. We                          the local relationships of each node in the graph. IsoMap [22], LLE
present a new method, SNS, that performs network embeddings                                   [17] and Laplacian eigenmaps [1] are three classic dimensional-
using structural information (namely graphlets) to enhance its qual-                          ity reduction and data representation algorithms. Finding the k
ity. SNS effectively utilizes both neighbor information and local-                            nearest neighbors is the key step of these three algorithms. Clas-
subgraphs similarity to learn node embeddings. This is the first                              sic algorithms can hardly tackle real networks due to the huge
framework that combines these two aspects as far as we know, pos-                             computational complexity of eigen-decomposition.
itively merging two important areas in graph mining and machine                                   DeepWalk [15] first introduces deep learning techniques word2vec
learning. Moreover, we investigate what kinds of local-subgraph                               to the network embedding task. The authors pioneered the similar-
features matter the most on the node classification task, which                               ity between random walks and natural language. Later, more work
enables us to further improve the embedding quality. Experiments                              [2, 7, 19, 28] is presented to improve DeepWalk by extending the
show that our algorithm outperforms other unsupervised and semi-                              definition of neighborhood and capturing neighborhood informa-
supervised neural network embedding algorithms on several real-                               tion from different levels of scope, namely first-order proximity,
world datasets.                                                                               second-order proximity and higher-order proximity. Using random
                                                                                              walks to capture the local structure is truly a neat idea, which makes
CCS CONCEPTS                                                                                  it possible to build representations of big networks.
                                                                                                  Although neighbors have been proved to be very significant fea-
• Computing methodologies → Neural networks; Learning latent
                                                                                              tures in all the state-of-art network embedding algorithms, Struc-
representations; • Applied computing → Sociology;
                                                                                              tural Similarity plays an irreplaceable role in various tasks ranging
                                                                                              from node classification to visualization. Figure 1 shows two ex-
KEYWORDS                                                                                      treme cases when classifying nodes. Figure 1a is the classic way
Network Embedding, Graphlet, Latent Representation                                            which is adopted by most algorithms. Nodes will be predicted to
                                                                                              have same labels if they share many friends and connect with each
ACM Reference format:                                                                         other closely. The strategy focuses on neighbors. On the other hand,
Tianshu Lyu, Yuan Zhang, and Yan Zhang. 2017. Enhancing the Network                           Figure 1b shows another strategy, that is dividing nodes by their
Embedding Quality with Structural Similarity. In Proceedings of CIKM’17 ,                     structural similarity, namely structural equivalence defined in [7].
Singapore, Singapore, November 6–10, 2017, 10 pages.
                                                                                              There are three groups altogether, core nodes, peripheral nodes
https://doi.org/10.1145/3132847.3132900
                                                                                              and hub nodes under this criteria for network partition. These dif-
                                                                                              ferent starting points induce different definitions of similar nodes:
                                                                                              (1) Densely connected nodes or (2) Nodes with similar network
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
                                                                                              positions.
for profit or commercial advantage and that copies bear this notice and the full citation         Most state-of-art algorithms, unfortunately, only concern
on the first page. Copyrights for components of this work owned by others than ACM            about densely connected nodes. These two strategies are not in
must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
to post on servers or to redistribute to lists, requires prior specific permission and/or a   conflict but complementary, each with its own sphere of compe-
fee. Request permissions from permissions@acm.org.                                            tence. For instance, if the node labels are about customers’ interests,
CIKM’17 , November 6–10, 2017, Singapore, Singapore                                           although the structure knowledge would contain invaluable in-
© 2017 Association for Computing Machinery.
ACM ISBN 978-1-4503-4918-5/17/11. . . $15.00
                                                                                              formation, blindly relying on it is not a good idea. Neighbors can
https://doi.org/10.1145/3132847.3132900                                                       provide much more reliable indications than structure similarity
                                                                            (3) We conduct some experimental studies to gain insight
                                                                                about how to make a rational use of structural information.


                                                                        2 RELATED WORK AND BACKGROUND
                                                                        2.1 Network Embedding
                                                                        Networks contains rich information of which, however, we cannot
                                                                        make full use easily and quickly. Algorithms on graphs, even those
                                                                        as simple as calculating distances between any pairs of nodes, are
                                                                        usually costly. Representation learning can help to extract useful
                                                                        information and transform network data into an easy-used one. The
                (a)                                (b)
                                                                        obtained network embeddings can be used as input features in many
                                                                        downstream tasks such as classification and link prediction, which
                                                                        obviates the need for complicated and time-consuming methods
                                                                        directly applied on graphs. We refer the reader to [6] for more
                                                                        comprehensive details.
                                                                           Traditional methods [20, 21] mainly focus on dimension reduc-
                                                                        tion. By the techniques of matrix factorization, traditional methods
                                                                        project the adjacency matrix to a low-dimension space. As this kind
                                 (c)                                    of algorithms are first designed for general high-dimension data,
                                                                        they do not fit the typical network data. Networks are sparse at
Figure 1: In (a) and (b), if two nodes are connected by dot-            most time and node degree follows power-law distribution. The
ted line, it means that there are a lot of intermediate nodes           optimization from a global view cannot provide a satisfactory re-
between them. The color of the node indicates its group. (a)            sult for classification and other downstream applications. Just as
and (b) present two cases that classifying nodes from two dif-          discussed in Section 1, local information, namely neighborhood, is
ferent fundamental angles. In (a), tightly connected nodes              much more significant than global information.
are in the same group. In (b), nodes with similar structural               Neural network embedding algorithms attract many attentions
position are grouped into one partition. (c) is the Graphlet            as the popular model, word2vec, in natural language processing
Degree Vector (GDV) of node 0,2,8 in (b). We only show 14               seems to perfectly fit this task and achieves much better effects than
dimensions of each node. E (·, ·) is the Euclidean distance be-         classic methods do. The main reason is that the word frequency
tween two nodes.                                                        follows a power law distribution, just as the degree distribution in a
                                                                        graph. Besides homophily is pervasive in both languages and graphs,
                                                                        which means that both words and nodes can be predicted by their
                                                                        surrounding words and nodes. The difference between the state-
does. If the labels are about the social status, topological informa-   of-art neural network based algorithms lies in the neighborhood
tion does matter. The most sensible approach is to make a balance       sampling strategies. DeepWalk [15] uses depth-first search in order
between these two strategies.                                           to sample the neighborhood of the target node. The depth is set
   Structural equivalence is being discussed not only in social role    to 2 by default. GraRep [2] also uses DFS but the depth is larger.
mining task [3], but in recent network embedding algorithms as          LINE [19] uses breadth-first search as well as depth-first search.
well [7]. We support the motivation that both kinds of similar          The number of step are all constrained below two. node2vec [7] also
nodes should be taken into consideration. However, we argue that        uses both two kinds of search methods and finds the balance point
embedding algorithms simply relied on random-walk sam-                  by semi-supervised learning. It selects the optimal balance point
pling are not capable of capturing Structural equivalence ef-           with grid search method, which is time-consuming. Furthermore,
fectively. In this paper, we propose a framework SNS considering        sampling strategy based on biased random walks is slower than the
both Structural and Neighborhood Similarity. It can be used to          unbiased one. Obviously, none of these algorithms take structural
refine all the word2vec-based network embedding algorithms.             similarity into consideration. SDNE [24] is not based on word2vec
   We conclude three main contributions of our research:                framework but deep autoencoder instead. It has the same goal as
    (1) We investigate the relation between random-walk based           LINE does, considering both first-order proximity and second-order
        sampling strategies and network embedding results theo-         proximity.
        retically and experimentally and point out the weakness            GraRep [2] and node2vec [7] discuss local structure and struc-
        (overestimation) of random-walk based sampling.                 tural equivalence. However, it is questionable whether the structural
    (2) We propose a general and robust neural network frame-           similarity is actually used during the learning process. Instead, they
        work that can effectively utilize both neighbor information     control the sampling process and capture neighborhood informa-
        and local-subgraph similarity to learn node embeddings.         tion from different levels of scope. Details will be discussed in the
        This is the first framework that combines these two aspects     following sections. Our framework is the first to directly leverage
        as far as we know.                                              local-subgraph information to learn network embedding.
                                                                              Authors of [15] give an analogy for a word and its context in
                                                                          natural language to the target node along with its neighbors in a
                                                                          graph. The learning process leverages the co-occurrence probability
                                                                          of the nodes that appear within a window in a random walk. Node
                                                                          pairs with high co-occurrence probability are regarded as neighbors.
                                                                          As the size of window is usually no less than two, we call this kind
                                                                          of neighbors as higher-order proximity. We denote the probability
                                                                          that node i and j are successively visited at an interval of r steps as
                                                                          P (i, j, r ).
                                                                              Consider an undirected graph G with N nodes and m edges. The
                                                                          adjacency matrix A is symmetric and the entries equal 1 if there is
                                                                          an edge between two nodes and 0 otherwise. Vector d = A1, where
                                                                          1 is a N × 1 vector of ones and each entry of d is the node degree.
Figure 2: Graphlets with 2-5 nodes and automorphism orbits                Graph G has an associated random walk in which the probability
[8]. Nodes of the same color belong to the same orbit within              of leaving a node is split uniformly among the edges.
that graphlet.
                                                                                                    pt+1 = pt D−1 A ≡ pt M,                  (1)
                                                                          where p is the probability vector and D is the diagonal matrix of
2.2    Structural Similarity                                              degree: D = diaд(d). M is the transition matrix. Given by π = π M,
Computing similarities between structured objects (network) is a          the stationary distribution of this Markov chain is π = d⊤ /2m,
hot topic in recent years. Methods are mainly based on graphlets,         P (i) = di /2m.
subtree patterns and random walks [25]. In the network embedding              For a walk starting at node i, the probability that we find it at j
task, we focus on nodes rather than the whole network structure,          after r steps is given by
which means that we need a metric describing the structural char-
                                                                                                        P (j, r |i) = [Mr ]ij ,
acteristics of a single node. This topic has been given much con-
sideration in the real fields such as biology and social science [26].                                        di
We first introduce graphlets and graphlet-based network distance                                                 [Mr ]ij ∝ [AMr−1 ]ij .
                                                                                   P (i, j, r ) = P (j, r |i)P (i) =                        (2)
                                                                                                             2m
measures.
                                                                             Equation 2 demonstrates that the co-occurrence probability of
    Graphlets are small, connected, non-isomorphic, induced sub-
                                                                          two arbitrary nodes i and j is mainly decided by the degree of
graphs of a big graph. As shown in Figure 2, there are altogether
                                                                          the nodes in the path between i and j and has nothing to do with
30 graphlets with 2-5 nodes. Orbit indicates different node position
                                                                          the degree of i or j. Note that the path length is restricted to the
of the graphlets. Symmetrical nodes have the same orbit number
                                                                          window size. Apparently, among all the intermediate nodes in the
and there are 73 orbits of 30 graphlets. The Graphlet Degree Vector
                                                                          feasible path between i and j, nodes with smaller degree and in
(GDV) of a node generalizes the notion of a node’s degree into a
                                                                          shorter path contribute more to the co-occurrence probability. To
73-dimensional vector, of which each components represents the
                                                                          put it more generally, two nodes are regarded as neighbors if the
number of times node n is touched by a graphlet at orbit i (where
                                                                          connection between them is strong and direct. As shown in Figure
i ∈ [1, 73]). If i equals 0, the number equals its degree. The struc-
                                                                          3, the more short feasible paths between two nodes (strong) and
tural similarity between nodes is denoted as the Euclidean distance
                                                                          the lower degree the intermediate nodes are (direct), the higher the
of their GDV. In Figure 1c, we calculate the GDV of node 0,2,8 in
                                                                          co-occurrence probability is.
Figure 1b for example. We only show 14 dimensions of each node,
                                                                             In the learning process, nodes with similar neighborhood will
corresponding to two 3-node graphlets and six 4-node graphlets.
                                                                          have similar latent representations. The scope of neighborhood
E (·, ·) denotes the Euclidean distance between two nodes’ GDV,
                                                                          is decided by the size of window. Two nodes in the network sep-
which can describe the structural similarity clearly. Besides Eu-
                                                                          arated by distance longer than the window size have no chance
clidean distance, GDV distance can also be defined in other forms,
                                                                          to calculate the mutual similarity. In other words, random-walk
for instance, Spearman correlation.
                                                                          based sampling strategy only captures the higher-order proximity
    There are many graphlet counting algorithms that can provide
                                                                          within the neighborhood of the target node. Besides, the higher-
precise results on small graph and approximate results on big graph
                                                                          order proximity of two nodes is mainly determined by how strong
with a quick speed. In this paper, we use orca [8] to help us calculate
                                                                          and direct the connections are. It is not capable of capturing the
GDV of each node. The code is available on the authors’ website.
                                                                          structural equivalence over the whole graph.
                                                                             Grover et. al. propose a more general method, node2vec [7],
3     RANDOM-WALK BASED SAMPLING                                          which simulates biased random walks by introducing two param-
Network embedding algorithms based on random walk preserve                eters p and q and controlling the search procedure interpolating
higher-order proximity between nodes by maximizing the probabil-          between BFS and DFS. These two parameters actually modify the
ity of occurrence of subsequent nodes in fixed length random walks.       transition matrix M. As stated in the last paragraph, smaller return
In this section, we take a deeper look at the sampling strategy based     parameter p encourages stronger connections between neighbors.
on random walks and the proximity captured by random walks.               Smaller in-out parameter q, on the other hand, encourages more
                                                                            Table 1: The relative importance of the orbits RIo of the node
                                                                            classification task on BlogCatalog

                                                                               orbit   RI %    orbit    RI %     orbit   RI %    orbit   RI %    orbit   RI %
                                                                                 0      0.05    1      21.18      2      1.79      3      0.14    4      84.58
                                                                                 5     62.26    6      95.38      7      10.94     8     21.27    9      81.94
                                                                                10     63.48    11      8.50      12     41.98    13      6.15    14      3.71
                                                                                15     92.27    16      76.27     17     85.71    18     86.70    19     93.90
                                                                                20     82.76    21      62.11     22     86.08    23     20.55    24     86.28
                                                                                25     83.70    26      65.51     27     88.68    28     81.02    29     82.77
                                                                                30     63.89    31     100.00     32     76.21    33     20.14    34     75.33
                                                                                35     95.19    36      77.29     37     82.30    38     44.18    39     94.31
                                                                                40     77.64    41      65.50     42     18.75    43     80.58    44     11.52
                                                                                45     87.21    46      80.75     47     58.46    48     65.70    49     83.55
                                                                                50     24.42    51      76.65     52     80.50    53     43.72    54     85.05
Figure 3: Using random-walk based sampling strategy, two                        55     15.38    56      91.93     57     66.20    58     14.91    59     77.65
nodes, i and j, are more likely neighbors if the connections                    60     55.84    61      13.73     62     74.85    63     31.37    64     41.12
between them are (a) strong: a number of feasible paths                         65     83.61    66      59.60     67     16.18    68     39.53    69      7.20
                                                                                70     49.11    71      14.99     72     12.14
and (b) direct: the intermediate nodes have few connections.
Nodes in this figure also connect with other nodes that are
not shown in this figure. These kind of edges are denoted by
dotted lines.                                                                  We use random forest implemented by sklearn [14] and calcu-
                                                                            late the orbit importance Io as stated before. The experiments are
                                                                            repeated on several datasets. Table 1 shows the relative importance
                                                                            of each orbits RIo on BlogCatalog. We can get similar observations
direct connections. This work defines a flexible notion of neigh-           on the other datasets.
borhoods and we argue that (1) the strength and directness of con-
                                                                                                                   Io
nections should not be seen as structural equivalence and (2) any                                      RIo =               × 100%
topological information outside the window cannot be captured.                                                   max (Ii )
                                                                                                                0⩽i⩽72
   In Section 6.1, we visualize the embedding results provided by
DeepWalk, node2vec and our proposed algorithm, demonstrating                We observe that orbit 0 (node degree) is of little importance from
our point of view experimentally.                                           the perspective of node classification. A more general idea is that
                                                                            peripheral orbit (orbits with less degrees) is more important than
4    PRE-PROCESSING STEP: TO OBTAIN THE                                     the core orbit (orbits with more degrees) for any graphlet. For
                                                                            instance, orbit 1 is more important than orbit 2 in G 1 . In G 12 , orbit
     STRUCTURALLY SIMILAR NODES                                             24 has the greatest importance while orbit 26 has the smallest one.
Now that random-walk based sampling strategies cannot capture               This can be verified by the Table 1 and Figure 2.
structure equivalence, we turn to other graph mining techniques                One reason of this interesting phenomenon might be the depen-
for additional capabilities. In this section, we discuss the details that   dency between graphlet degrees. Take any node with more than
how we obtain the local-subgraph information in the pre-processing          two neighbors as an example, it can form either G 1 or G 2 with any
step and use it in the network embedding task.                              two of its neighbors. If we denote Ci as the graphlet degree of orbit
   Peripheral orbits are informative. In Section 2.2, we have               i, we then have
presented the use of graphlets and orbits. The number of orbits                                        C 
                                                                                                         2 = C2 + C3 .
                                                                                                          0
grows quickly as the size of graphlets increases. Too many features,
however, will result in over-fitting and not all orbits contribute          Core orbits are consequently more likely to be redundant, as it
equally to a certain task. Therefore, we evaluate the importance of         might be expressed by the sum of some peripheral orbits.
different orbits on the node classification task [5] with the help of          Because of the analysis above, peripheral orbits should weigh
Random Forest [14].                                                         more than core orbits when we calculate the similarity of two GDVs.
   Random Forest consists of a number of decision trees. In the                Limit the scope of similar nodes. In order to cooperate the
decision trees, every node is a condition on a single feature, designed     network embedding task, we try to leverage local-subgraph simi-
to split the dataset into two parts so that similar response values end     larity information. For a target node, its local neighbors and struc-
up in the same set. Impurity is to measure the homogeneity of the           turally similar nodes are involved in the learning process. Struc-
target variable within the subsets. For classification, it is typically     turally similar nodes can be chosen from the whole network or
either Gini impurity or information gain. Thus when training a              only the neighborhood of the target node. The scope of structurally
tree, it can be computed how much each feature decreases the                similar nodes to be considered depends on the real task. In certain
weighted impurity in a tree. For a forest, the impurity decreased           circumstance, nodes connected together in the network are more
from each feature can be averaged and we call this measure feature          likely to have the same labels than those without connections be-
importance. Note that features with more categories and with less           tween them. The maximum step between the similar node and the
correlated features are considered to be more important when using          target node is supposed to be small. However, there are also cases
the impurity based ranking.                                                 where structural similarity is the major factor and the maximum
                                                                          Figure 5: Combine the structural information with the net-
Figure 4: The framework of CBOW. w 1 , w 2 , · · ·, wW are the
                                                                          work embedding. Given space limitations, we only demon-
context words of the target word. The vocabulary size is V .
                                                                          strate the complete neural network architecture of w n . For
The window size is W . W is the weight matrix between the
                                                                          the other W − 1 context words, they also have new architec-
input layer and the hidden layer. Each row of W is the N -dim
                                                                          tures as w n does.
word vector vw . And similarly, v ′w is from W ′ .

                                                                          with skip-gram model, CBOW model performs better on dataset
step has to be bigger.                                                    with short sentences but high number of sample sets (larger dataset).
                                                                             In the field of NLP, similar words have similar contexts and
   In the pre-processing step, graphlet counting algorithms first         CBOW aims to predict a word by its context. The loss function is as
produce Graphlet Distribution Vector for every node. Next for each        follows. C is the vocabulary set and w is the word to be predicted.
node, find its K nearest neighbors in the GDV space, i.e. the K
                                                                                               X
                                                                                          L=            log p(w |Context (w )).
neighboring nodes which have the smallest cosine distance from it.                                  w ∈C
Note that searching structurally similar nodes has to be restricted       CBOW takes the one-hot encoder of the context words w I as input
in the higher order S t h degree neighborhood of the target node          and the average of the context word vectors as hidden layer, where
NvSi as:                                                                  W is the number of context words of word w and vw i is the vector
                                                                          of word w i .
         NvSi = {v j | Ai j = 1 ∨ Ai2j ⩾ 1 ∨ · · · ∨ AiSj ⩾ 1}.                                    1
                                                                                        h = vw I =    (vw 1 + vw 2 + . . . + vwW ).
                                                                                                   W
This is the set of all nodes at a distance no more than S from target
                                                                          wO is the output of CBOW and y j is the j-th unit of the output
node vi . Search results are preserved in a sparse matrix S, where si j
                                                                          layer. Note that vw and v ′w are the input vector and output vector
equals the similarity score if j is one of the K nearest neighbors of
                                                                          of the word w respectively.
i and 0 otherwise (si j ∈ [0, 1) and j si j = 1). There are altogether
                                      P
K non-zero numbers in each row of the matrix S. To sum up, the                                                    exp(v ′Tw j vw I )
pre-processing is controlled flexibly from the following five aspects:                p(w |Context (w )) = y j = PV                     .
                                                                                                                            ′T
                                                                                                                   j ′ =1 (v w j vw I )
  (i) The metric to evaluate the similarity of GDV.
                                                                             The softmax function is very difficult to optimize because of the
 (ii) O: The number of orbits to be considered.
                                                                          huge amount of calculation in the denominator. One of the effective
(iii) R: The weight of different orbits when calculating the similar-
                                                                          solutions is negative sampling, which approximates the softmax
      ity of GDV.
                                                                          function by performing logistic regression to k noise samples. As
(iv) K: The number of the most similar nodes.
                                                                          shown in [12], the probabilistic distribution for the negative sam-
 (v) S: The maximum step between the similar node and the target
                                                                          pling process is a unigram distribution raised to the 34 th power. We
      node.
                                                                          denote Pn (w ) as the noise distribution and σ is the logistic function.
                                                                          Wneд is the set of k samples sampled based on Pn (w ).
5     NETWORK EMBEDDING POWERED BY
                                                                                                                         3
      STRUCTURAL SIMILARITY                                                                                       U (w ) 4
                                                                                                     Pn (w ) =             ,
We propose a neural network architecture that leverages both local                                                   Z
                                                                                  L = log σ (v ′Tw O vw I ) +              log σ (−v ′Tw j vw I ).
                                                                                                                  X
connected neighbors and topological information to learn network
embedding.                                                                                                      w j ∈Wneд

5.1    CBOW Model with Negative Sampling                                  5.2    Enhanced by the Structural Information
Although our framework can be used to refine all the word2vec-                   Branch
based network embedding algorithms, we use CBOW [12] as the               We propose a new framework in order to boost the learning process
basis of our framework due to the space limitations. Comparing            with the structural similarity information. Figure 4 presents the
framework of CBOW. We take the dotted portion in Figure 4 as
an example and its corresponding new architecture is showed in                             (new )         (old )                     (new )      (old )
                                                                                          c1        = c1           − ηvw i ∆, c 2             = c2        − ηvsim-i ∆,
Figure 5. In this model, the target node is predicted by its neighbor-
hood (contextual information, referred to as neighborhood branch),
as well as the structurally similar nodes nearby (referred to as                               (new )       (old )                   (new )      (old )
                                                                                           vw i         = vw i          − ηc 1 ∆, vw j        = vw j      − ηc 2si j ∆,
structural information branch). As discussed in the last section, the
sparse similarity score matrix S is derived from the pre-processing
                                                                                                                   (new )       (old )
step. Each row of S contains K non-zero items and si j is referred to                                            si j       = si j       − ηvw j ∆.
as the similarity score of node i and node j. Neighborhood branch                      Note that matrix W is updated twice in one iteration, that is, one
and structural information branch share one embedding matrix W.                     round by the nodes in the neighborhood and the other round by
Each row of W represents the input embedding vector v of a node.                    the topologically similar nodes. When the learning process is over,
The corresponding vector of structural information is written as                    we take the matrix W as the learned node representations.
                                             V
                                             X
                              vsim-n =            snm vwm .                         5.4    Variants
                                            m=1                                     We briefly discuss some variants of our framework here. For differ-
The similarity score snm is regarded as the weight assigned to the                  ent tasks and networks, we can choose proper variants.
vector of wm , one of the top-K topologically similar nodes of w n .                   Whose similar nodes are they? In our framework, we lever-
  An aggregated representation of the context of the target input                   age the similar nodes of the context nodes rather than the target
node is                                                                             node. However, the latter plan is also feasible. The difference mainly
                 W                                                                  lies in the scope of similar nodes to be chosen from. If we want to
              1 X
      vw I =        (c 1 (deд(w n ))vw n + c 2 (deд(w n ))vsim-n ),                 choose the nodes from a bigger scope (the maximum step away
             W n=1
                                                                                    from the target node is 3 or bigger), the latter plan will lead to huge
where c 1 and c 2 are the parameters regulating the proportion of the               computational complexity of the pre-processing step. The current
two kinds of information (c 1 , c 2 ∈ (0, 1)). Usually, the neighborhood            plan, on the contrary, achieves the goal by making the window-size
of a high degree node can provide enough information for label                      W bigger. Larger window-size does not influence the efficiency
prediction. The node with few edges, however, depends much on                       significantly. Problem for the current plan is that the similarity is
structural information and less on its neighborhood. Thereby, the                   not transitive in some cases. Which plan is better might depend on
value of c 1 is larger than c 2 for high degree nodes and vise versa.               the assortativity of the network.
Nodes are sorted in descending order of node degree and divided                        Fixed value of c 1 , c 2 and similarity matrix. In our frame-
into C levels. Nodes at the same level share the same c 1 and c 2 .                 work, c 1 , c 2 , and S are updated in the learning process. Setting
   In conclusion, the model takes the weighted average of the node                  proper initial values and making them fixed can also achieve a good
vectors of both the target node’s neighbors and their similar nodes                 performance. Moreover, fixed parameters accelerate the learning
as input.                                                                           process. The trick is setting bigger c 1 and smaller c 2 for high de-
                                                                                    gree nodes and vise versa. The similarity scores seem to be less
5.3      Learning Process                                                           important. If node j is one of the top-K topologically similar nodes
Matrix S, W, W ′ and the value of c 1 , c 2 are updated in the back-                of node i, then we set si j = 1/K. Otherwise, si j = 0.
propagation process. The partial derivative of L with regard to the
net input of the output unit w j is as follows:                                     6     EXPERIMENTS
                                                                                    In this section we first visualize a small network using DeepWalk
              ∂L               σ (v ′Tw j vw I ) − 1,         if w j = wO ;
                          =                                                        [15], node2vec [7] and our proposed algorithm separately, which
                           
          ∂v ′Tw j vw I    
                                 σ (v ′Tw j vw I ),         if w j ∈ Wneд .        demonstrates the arguments in Section 3 visually. Moreover, we
                          = σ (v ′Tw O vw I ) − t j .                               show the performance of different network embedding algorithms
                                                                                    on multi-label classification task. The parameter sensitivity will also
where t j is the indicator. Using the chain rule, we further get:                   be discussed, helping us have a deeper insight of the topological
       ∂L           ∂L        ∂v ′Tw j vw I                                         information and the algorithm efficiency.
              =             ·               = (σ (v ′Tw O vw I ) − t j )vw I ,
      ∂v ′w j   ∂v ′Tw vw I
                          j
                               ∂v ′w j                                              6.1    Case Study: Visualization in 2-D space
                                                                                    Les Miserables co-appearance network consists of 77 nodes and
           ∂L                 X                   ∂L             ∂v ′Tw j vw I      254 edges, where node corresponds to characters showed up in Les
                =                                            ·
          ∂vw I
                       w j ∈ {w O }∪Wneд
                                             ∂v ′Tw j vw I          ∂vw I           Miserables and edge corresponds to the co-appearance relationship.
                                                                                    As the network is small, it is easy to visualize. Nodes in this network
                                            (σ (v ′Tw O vw I ) − t j )v ′w j ≡ ∆.
                              X
                  =                                                                 have various structural positions, making it easier to analyze dif-
                       w j ∈ {w O }∪Wneд                                            ferent sampling strategies thoroughly. We compare our algorithm
   The corresponding update equation is as followed:                                with DeepWalk [15] and node2vec [7]. Both of them are classic
                                                                                    network embedding algorithm with random-walk based sampling
                 (new )          (old )
              v ′w j       = v ′w j       − η(σ (v ′Tw O vw I ) − t j )vw I ,       strategy. Node2vec claims to be able to find structural equivalence
                                                                       Figure 8: We use node2vec [7] to learn the latent representa-
                                                                       tion of Les Miserables network (p = 1, q = 2, d = 16). Nodes
                                                                       are mapped to the 2-D space using the PCA package [13]
                                                                       with learned embeddings as input.


Figure 6: Visualization of Les Miserables co-appearance net-
work using a force-directed layout: ForceAtlas2 [10].




                                                                       Figure 9: We use SNS to learn the latent representation of Les
                                                                       Miserables network. Nodes are mapped to the 2-D space us-
                                                                       ing the PCA package [13] with learned embeddings as input.

Figure 7: We use DeepWalk [15] to learn the latent repre-
sentation of Les Miserables network (p = 1, q = 1, d = 16).
                                                                       DeepWalk, node2vec and SNS respectively. Note that d, the dimen-
Nodes are mapped to the 2-D space using the PCA package
                                                                       sion size of the latent space, is 16 and we use Principal Component
[13] with learned embeddings as input.
                                                                       Analysis to project the vectors to 2-dimensional space. Colors of
                                                                       nodes are uniform in Figure 6-9, making it easier for us to track
                                                                       them in different figures.
when the parameters p and q are properly set. We use the same             As stated in Section 3, random-walk based sampling strategies
settings as they reported in the paper (p = 1, q = 2). DeepWalk is     are sensitive to strong and direct connections. In Figure 6, both
a special case of node2vec with p = 1 and q = 1. For simplicity,       node 51 (light blue) and node 56 (dark blue) are in the neighborhood
the window size of these three algorithms are all set to be 4, which   of node 26 (orange). There are many short paths between node 26
means that neighbors are chosen from the nodes at a distance no        and 51, 26-51, 26-49-51, 26-54-51, 26-11-51 and etc. On the contrast,
more than 2 from the target node.                                      the only two short paths between node 26 and 56 are 26-49-56 and
   Figure 6 is the visualization of Les Miserables co-appearance       26-55-56. Therefore, node 26 has stronger connection with node 51
network using a force-directed layout: ForceAtlas2 [10]. Figure 7,     and as shown in Figure 7, the distance between node 26 and node
Figure 8 and Figure 9 are the learned latent vectors provided by       51 is much shorter than it between node 26 and node 56. In Figure 6,
node 11 (light purple), 16-22 (red) and 27 (light purple) are two-step
neighbors of node 30 (orange). The intermediate node 23 has much                                                                       BlogCatalog
                                                                                                45                                                           30
more connections than node 31 does. Therefore, in Figure 7, node                                40                                                           25




                                                                               Micro F1 Score                                               Macro F1 Score
16-22 (red) are far away from node 30 (orange).                                                 35
   We also discuss the influence of biased random walk in Section 3                                                                                          20
                                                                                                30
that smaller p encourages stronger connections between neighbors                                                                                             15
                                                                                                25
and smaller q encourages more direct connections. In our exper-                                                                                              10
                                                                                                20
iments, node2vec are set to p = 1, q = 2 and are therefore more
                                                                                                15                                                            5
sensitive to strong connections. Following the discussion of Deep-                                   0                 0.5             1                          0        0.5           1
                                                                                                               Sample Portion                                         Sample Portion
Walk, the relationship between node 26 (orange) and 51 (light blue)                                                                        PPI
is stronger compared with the one between node 26 and 56 (dark                                  30                                                           25

blue). In Figure 8, we can figure that node 26 and node 51 are quite


                                                                               Micro F1 Score                                               Macro F1 Score
                                                                                                25                                                           20
close. The distance is much shorter than it is in Figure 7.
   In Figure 6, node 1, 4-9 (light green) are structurally similar as                           20                                                           15

they all have 1 edge connected to node 0. Node 57-65, 76 (dark                                  15                                                           10
green) are also a group of similar nodes as they all connect to each
other and thus form a clique. In Figure 7 and Figure 8, we find that                            10
                                                                                                     0                 0.5             1
                                                                                                                                                              5
                                                                                                                                                                  0        0.5           1
the locations of these two groups of nodes are quite different. Node                                           Sample Portion                                         Sample Portion

57-65, 76 (dark green) are separated from the other nodes, while                                                                           POS
                                                                                                60                                                           25
node 1, 4-9 (light green) can hardly separate from the surrounding                              55




                                                                               Micro F1 Score                                               Macro F1 Score
nodes. This difference proves that whether the random-walk based                                50
                                                                                                                                                             20

sampling strategy is biased or not, it is not capable of finding out                            45                                                           15
the nodes with structural equivalence under any circumstances.                                  40
The reason why node 57-65, 76 are away from other nodes is that                                 35
                                                                                                                                                             10

the connections between them are much stronger and more direct                                  30                                                            5
than the rest. Other nodes can hardly be visited in the random                                       0                 0.5
                                                                                                               Sample Portion
                                                                                                                                       1                          0        0.5
                                                                                                                                                                      Sample Portion
                                                                                                                                                                                         1

walks started from node 57-65 and 67. Node 1, 4-9, on the contrary,
                                                                                                     Spectural Clustering       LINE            DeepWalk                 node2vec      SNS
do not have such local structure and the structural similarity of
this group of nodes can not be captured by random-walk based
sampling algorithms.                                                      Figure 10: The Macro-F1 Score and Micro-F1 Score of differ-
   In Figure 9, these two groups of nodes are all separated from          ent algorithms on varying the sampling portion used for
the rest of nodes. This improvement attributes to our consideration       training.
of graphlets distribution. Note that although nodes 10, 13-15 (dark
purple) and etc. also only have one edge as node 1, 4-9 do, their local
structural position are quite different in our opinion. For example,
node 67 (yellow) connects to node 57 in Figure 6 and node 57-65              BlogCatalog [27] is the social blog directory which manages the
and 76 form a clique as we stated before. We consider node 67 is          bloggers and their blogs. The network depicts the contact between
more similar with node 57-65 and 76. Node 13-15 all have one edge         users and the user labels represent their interests. Users tend to
connected to node 11, which is the 2-step neighbor of node 1, 4-9.        have same hobbies as their close friends. Strangers but with similar
We consider node 10, 13-15 are similar with node 1, 4-9. These kinds      structural positions are also possible. The network contains 10,312
of relationships are all reflected in the latent space as shown in        nodes, 333,983 edges and 39 labels.
Figure 9.                                                                    Protein-Protein Interactions [18] is a subgraph of the PPI
   Through all the demonstrations above, we try to prove that             network for Homo Sapiens. Labels stand for the protein biological
random-walk based sampling strategies are not good at mining              states. The network contains 3,890 nodes, 76,584 edges and 50 labels.
structural equivalence. The properties of random walks constrain             POS [11] is a co-occurrence network of words appearing in the
the capability of this kind of algorithms. They only capture the          first million bytes of the Wikipedia dump. A part of speech (POS)
strength and directness of the relationships between nodes, rather        is a category of words which have similar grammatical properties.
than the exact topological information of every node. SNS, with the       Words that always show together have a high probability to be
help of graphlet distribution vector, is capable to mine structural       similar meaning. And words with similar network structure also
equivalence.                                                              tend to be with the same part of speech. The network contains 4,777
                                                                          nodes, 184,812 edges and 40 labels.
                                                                             We choose a classic method aiming at dimension reduction and
6.2    Datasets and Baselines                                             three representative neural network embedding algorithms as base-
                                                                          lines.
In the following experiments, we choose three datasets that accord
                                                                             Spectral Clustering [21] is based on matrix factorization and
our starting-point. They all exist a mix of homophily and structural
                                                                          aims at minimizing Normalized Cut. We set d = 500 , the same
equivalences [7].
                                                                          setting in [21].
    DeepWalk [15] is the first embedding algorithm that brings the
deep learning technology. It is an unsupervised learning algorithm.                                        44                                                                       28
It captures the neighborhood information of each node by random                                            42
                                                                                                                                                                                    26




                                                                                          Micro F1 Score                                                           Macro F1 Score
walks and uses skip-gram model with negative sampling. We set                                                                                                                       24
                                                                                                           40
d = 128, r = 10, l = 80, k = 10, the same setting in [15].                                                                                                                          22

                                                                                                           38                                                                       20
    LINE [19] defines a loss function based on 1-hop and 2-hop                                                                                                                      18
relational information. It learns d/2 dimensions of the node vector                                        36
                                                                                                                                                                                    16
by these two parts of information respectively and combines them                                           34
                                                                                                                0                 0.5              1
                                                                                                                                                                                    14
                                                                                                                                                                                         0                  0.5             1
directly as the final output. In a supervised learning task, it finds                                                       Sample Portion                                                             Sample Portion

the weighting of dimensions based on training data and achieves a                                                                S=1,R=9,O=14      S=2                                       R=1         O=72

better performance. In our experiments, we try the unsupervised                                            41                                                            40.6
mode of LINE [19] and set d = 128, r = 10, l = 80, k = 10, the same                             40.5                                                                     40.4




                                                                               Micro F1 Score                                                           Micro F1 Score
                                                                                                                                                                         40.2
setting in [19].                                                                                           40
                                                                                                39.5                                                                                40
    node2vec [7] simulates biased random walks over the underly-                                                                                                         39.8
                                                                                                           39
ing network. It uses parameter p and q to balance the BFS strategy                              38.5                                                                     39.6
with DFS strategy.DeepWalk is a special case of node2vec where                                             38                                                            39.4

p = q = 1. node2vec is a semi-supervised algorithm and it need                                  37.5
                                                                                                                6       8         10         12   14
                                                                                                                                                                         39.2
                                                                                                                                                                                         6         8        10        12    14
10% labeled data of the network to decide the value of p and q. We                                                  Number of Walks per Node                                                            Window Size

use the value of p and q as showed in the authors’ paper [7].                                              41                                                                       41

    The settings of our algorithm SNS is as follows: in the pre-                                           40
                                                                                                                                                                                    40




                                                                                          Micro F1 Score                                                           Micro F1 Score
processing step, K = 5, S = 1, O = 14, R = 9. In the learning                                              39
                                                                                                                                                                                    39
                                                                                                                                                                                    38
step, we use the same parameters as DeepWalk does and C = 5. In                                            38                                                                       37
the pre-processing step, we use orca [8], a very efficient algorithm                                                                                                                36
                                                                                                           37
for graphlet enumeration.                                                                                                                                                           35
                                                                                                           36                                                                       34
    As reported in the paper [8], on a desktop computer (Intel Core                                          40             60          80        100                                    4         5         6          7   8
                                                                                                                            Length of Walk                                                         Dimensions (log2)
2, 2.67 GHz), orca only takes 2.5s to count the four-node graphlets
of a network with 25,368 nodes and 75,004 edges. SNS spends more
time than DeepWalk does, as more parameters are involved in              Figure 11: Parameter Sensitivity of SNS on the BlogCatalog
the learning process. The sampling step of node2vec, similarly, is       network.
very time-consuming, comparing with the normal random-walk
sampling used by DeepWalk and SNS.
                                                                            Note that node2vec has to utilize supervised data to decide the
                                                                         best parameter, while SNS is unsupervised and achieves better per-
                                                                         formance. This illustrates that our intuition, combining the struc-
                                                                         tural similarity with neighborhood information, is an important
6.3    Multi-label Classification                                        knowledge of multi-label classification task.
This task uses the exact same datasets and experimental procedure        6.4                               Parameter Sensitivity
as presented in [7]. The node vectors are the input to a one-vs-
rest logistic regression implemented by sklearn [14]. We sample a        We test our algorithm with different settings on BlogCatalog dataset.
portion of the labeled nodes as training data and use the rest nodes     The results are similar for the other datasets. The first two sub-
as test data. This process is repeated 10 times, and we show the         figures of Figure 11 focus on the settings of structural similarity.
average Micro-F1 Score and Macro-F1 Score in Figure 10.                  S stands for the maximum step between the similar node and the
   We can easily figure out that SNS has advantages over other           target node. R stands for the weighting factors between peripheral
algorithms on these three datasets. Spectural Clustering is a com-       orbits and core orbits when calculating the GDV distance. O stands
petitive algorithm only on BlogCatalog dataset. DeepWalk seems to        for the number of orbits to be considered (the dimension of GDV).
be the most stable algorithm among the baselines. This performance       If the node labels is relevant to the node distance in the network,
suggests that different neighborhood sampling strategies are not         bigger S will bring many nodes far from the target node and have a
so reliable in different fields. Designing special sampling strategies   negative effect on the label prediction task. As discussed in Section
will result in limited application scope. In BlogCatalog dataset, the    4, peripheral orbits are more informative. Thereby, they weigh more
advantage of SNS is not as evident as it in the other two datasets.      than core orbits. The number of orbits does not matter much. Using
Neural network based algorithms have close scores. The reason            more orbits to calculate the topological similarity between nodes
might be that node labels of BlogCatalog datasets depend more on         does not contribute to a better result. Fourteen orbits from graphlets
the node neighborhood. In PPI dataset and POS dataset, the obvious       with 2-4 nodes is enough.
advantages have to be attributed to the help of structural similarity        The rest sub-figures are about the parameters of random walks.
information.                                                             Window size k has little effect on the results. We attribute the
                                                                         good performance of small window size to the help of topological
                                                                         information. A big window size brings much noisy information
and makes the embedding quality worse. Number of walks per                             [3] Sarvenaz Choobdar, Pedro Ribeiro, Srinivasan Parthasarathy, and Fernando Silva.
node r and walk length l have relatively large impact. These two                           2015. Dynamic inference of social roles in information cascades. Data Mining
                                                                                           and Knowledge Discovery 29, 5 (2015), 1152–1177.
parameters control the scale of training data. Increasing dimension                    [4] Linton C Freeman. 2000. Visualizing social networks. Journal of social structure
d improves the result. When d is bigger than 100, the improvement                          1, 1 (2000), 4.
                                                                                       [5] Robin Genuer, Jean-Michel Poggi, and Christine Tuleau-Malot. 2010. Variable
is less obvious.                                                                           selection using random forests. Pattern Recognition Letters 31, 14 (2010), 2225–
                                                                                           2236.
6.5     Discussion: Local Versus Global                                                [6] P. Goyal and E. Ferrara. 2017. Graph Embedding Techniques, Applications, and
                                                                                           Performance: A Survey. ArXiv e-prints (May 2017). arXiv:1705.02801
The proposed model SNS is capable of capturing both local and                          [7] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for
                                                                                           networks. In Proceedings of the 22nd ACM SIGKDD International Conference on
global structural characteristics. We are going to discuss about                           Knowledge Discovery and Data Mining. ACM, 855–864.
their feasibility and practicality. Utilizing global information of the                [8] Tomaž Hočevar and Janez Demšar. 2014. A combinatorial approach to graphlet
network is a difficult field and we have to acknowledge that the                           counting. Bioinformatics 30, 4 (2014), 559–565.
                                                                                       [9] Shian-Chang Huang. 2009. Integrating nonlinear graph based dimensionality
computational cost of SNS is huge when S is a large value. Some                            reduction schemes with SVMs for credit rating forecasting. Expert Systems with
related work [2, 16] suffers from the same drawback. A possible                            Applications 36, 4 (2009), 7515–7518.
way to improve the performance might be designing heuristics and                      [10] Mathieu Jacomy, Tommaso Venturini, Sebastien Heymann, and Mathieu Bastian.
                                                                                           2014. ForceAtlas2, a continuous graph layout algorithm for handy network
limiting the search scope. On the other hand, it seems that in most                        visualization designed for the Gephi software. PloS one 9, 6 (2014), e98679.
real tasks global information is less useful than local information.                  [11] Matt Mahoney. 2009. Large text compression benchmark. URL: http://www.
                                                                                           mattmahoney. net/text/text. html (2009).
Long distance, in the network indicates the potential gaps of node                    [12] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.
features. Structural similarity is less discriminative than spatial                        Distributed representations of words and phrases and their compositionality. In
proximity to certain extent. In conclusion, local structural informa-                      Advances in neural information processing systems. 3111–3119.
                                                                                      [13] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel,
tion is easy to be captured and more commonly used in real tasks.                          Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss,
Meanwhile, we are looking forward to the new network embedding                             Vincent Dubourg, et al. 2011. Scikit-learn: Machine learning in Python. Journal
algorithms effectively capturing global structural information.                            of Machine Learning Research 12, Oct (2011), 2825–2830.
                                                                                      [14] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M.
                                                                                           Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cour-
7     CONCLUSION AND FUTURE WORK                                                           napeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine
                                                                                           Learning in Python. Journal of Machine Learning Research 12 (2011), 2825–2830.
In this paper, we analyze the characteristics and the downsides                       [15] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning
of random-walk based sampling strategies and propose a network                             of social representations. In Proceedings of the 20th ACM SIGKDD international
                                                                                           conference on Knowledge discovery and data mining. ACM, 701–710.
embedding framework combining both structural similarity and                          [16] Leonardo FR Ribeiro, Pedro HP Saverese, and Daniel R Figueiredo. 2017.
neighborhood information. We capture the structural information                            struc2vec: Learning Node Representations from Structural Identity. In Proceed-
                                                                                           ings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery
by graphlet degree vector and make full use of it in our learning                          and Data Mining. ACM, 385–394.
framework. This method alters the neural networks used by the                         [17] Sam T Roweis and Lawrence K Saul. 2000. Nonlinear dimensionality reduction
word2vec algorithms to contain a topological information factor,                           by locally linear embedding. Science 290, 5500 (2000), 2323–2326.
                                                                                      [18] Chris Stark, Bobby-Joe Breitkreutz, Teresa Reguly, Lorrie Boucher, Ashton Bre-
introducing new layers with several tunable parameters. SNS is                             itkreutz, and Mike Tyers. 2006. BioGRID: a general repository for interaction
the first framework that leverages network structure directly and                          datasets. Nucleic acids research 34, suppl 1 (2006), D535–D539.
makes it possible no matter in local area or in a global scope of the                 [19] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
                                                                                           2015. Line: Large-scale information network embedding. In Proceedings of the
network. The experiments show that structural similarity does help                         24th International Conference on World Wide Web. ACM, 1067–1077.
in the node classification task.                                                      [20] Lei Tang and Huan Liu. 2009. Scalable learning of collective behavior based on
                                                                                           sparse social dimensions. In Proceedings of the 18th ACM conference on Information
   Tasks, including node classification, link prediction and visualiza-                    and knowledge management. ACM, 1107–1116.
tion, are all relevant with local network structure. If there exists any              [21] Lei Tang and Huan Liu. 2011. Leveraging social media networks for classification.
task that has to use global information of the network, algorithms                         Data Mining and Knowledge Discovery 23, 3 (2011), 447–478.
                                                                                      [22] Joshua B Tenenbaum, Vin De Silva, and John C Langford. 2000. A global geometric
based on random walks are not suitable any more. In conclusion,                            framework for nonlinear dimensionality reduction. science 290, 5500 (2000), 2319–
future work includes finding new application scenarios and new                             2323.
framework supporting capturing global network information.                            [23] Athanasios Theocharidis, Stjin Van Dongen, Anton J Enright, and Tom C Freeman.
                                                                                           2009. Network visualization and analysis of gene expression data using BioLayout
                                                                                           Express3D. Nature protocols 4, 10 (2009), 1535–1550.
8     ACKNOWLEDGMENT                                                                  [24] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural deep network em-
                                                                                           bedding. In Proceedings of the 22nd ACM SIGKDD International Conference on
The authors would like to thank the anonymous reviewers for                                Knowledge Discovery and Data Mining. ACM, 1225–1234.
                                                                                      [25] Pinar Yanardag and S.V.N. Vishwanathan. 2015. Deep Graph Kernels. In Pro-
their valuable comments and helpful suggestions. This work is                              ceedings of the 21th ACM SIGKDD International Conference on Knowledge Dis-
supported by 973 Program under Grant No.2014CB340405, NSFC                                 covery and Data Mining (KDD ’15). ACM, New York, NY, USA, 1365–1374.
under Grant No.61532001 and No.61370054, and MOE-RCOE under                                https://doi.org/10.1145/2783258.2783417
                                                                                      [26] Ömer Nebil Yaveroğlu, Noël Malod-Dognin, Darren Davis, Zoran Levnajic, Vuk
Grant No.2016ZD201.                                                                        Janjic, Rasa Karapandza, Aleksandar Stojmirovic, and Nataša Pržulj. 2014. Re-
                                                                                           vealing the hidden language of complex networks. Scientific reports 4 (2014),
                                                                                           4547.
REFERENCES                                                                            [27] R. Zafarani and H. Liu. 2009. Social Computing Data Repository at ASU. (2009).
 [1] Mikhail Belkin and Partha Niyogi. 2003. Laplacian eigenmaps for dimensionality        http://socialcomputing.asu.edu
     reduction and data representation. Neural computation 15, 6 (2003), 1373–1396.   [28] Daokun Zhang, Jie Yin, Xingquan Zhu, and Chengqi Zhang. 2016. Homophily,
 [2] Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2015. Grarep: Learning graph repre-           Structure, and Content Augmented Network Representation Learning. In Data
     sentations with global structural information. In Proceedings of the 24th ACM         Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 609–618.
     International on Conference on Information and Knowledge Management. ACM,
     891–900.

