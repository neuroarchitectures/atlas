# Deep Inductive Network Representation Learning Rossi Zhou Ahmed 2018

> Source: `Deep_Inductive_Network_Representation_Learning_Rossi_Zhou_Ahmed_2018.pdf`

---

                     Deep Inductive Network Representation Learning
                    Ryan A. Rossi                                                  Rong Zhou                                        Nesreen K. Ahmed
                     Adobe Research                                                Google                                                Intel Labs
                    rrossi@adobe.com                                        rongzhou@google.com                                 nesreen.k.ahmed@intel.com

ABSTRACT                                                                                    engineering in terms of cost and effort. For a survey and taxonomy
This paper presents a general inductive graph representation learn-                         of relational representation learning, see [28].
ing framework called DeepGL for learning deep node and edge                                    Recent work has largely been based on the popular skip-gram
features that generalize across-networks.1 In particular, DeepGL be-                        model [18] originally introduced for learning vector representa-
gins by deriving a set of base features from the graph (e.g., graphlet                      tions of words in the natural language processing (NLP) domain.
features) and automatically learns a multi-layered hierarchical                             In particular, DeepWalk [23] applied the successful word embed-
graph representation where each successive layer leverages the                              ding framework from [19] (called word2vec) to embed the nodes
output from the previous layer to learn features of a higher-order.                         such that the co-occurrence frequencies of pairs in short random
Contrary to previous work, DeepGL learns relational functions (each                         walks are preserved. More recently, node2vec [13] introduced hy-
representing a feature) that naturally generalize across-networks                           perparameters to DeepWalk that tune the depth and breadth of the
and are therefore useful for graph-based transfer learning tasks.                           random walks. These approaches have been extremely successful
Moreover, DeepGL naturally supports attributed graphs, learns in-                           and have shown to outperform a number of existing methods on
terpretable inductive graph representations, and is space-efficient                         tasks such as node classification.
(by learning sparse feature vectors). In addition, DeepGL is expres-                           However, the past work has focused on learning only node fea-
sive, flexible with many interchangeable components, efficient with                         tures [13, 23, 32] for a specific graph. Features from these methods
a time complexity of O(|E|), and scalable for large networks via an                         do not generalize to other networks and thus are unable to be used
efficient parallel implementation. Compared with recent methods,                            for across-network transfer learning tasks.2 In contrast, DeepGL
DeepGL is (1) effective for across-network transfer learning tasks                          learns relational functions that generalize for computation on any
and large (attributed) graphs, (2) space-efficient requiring up to                          arbitrary graph, and therefore naturally supports across-network
6× less memory, (3) fast with up to 182× speedup in runtime per-                            transfer learning tasks such as across-network link classification,
formance, and (4) accurate with an average improvement in AUC                               network alignment, graph similarity, among others. Existing meth-
of 20% or more on many learning tasks and across a wide variety                             ods are also not space-efficient as the node feature vectors are
of networks.                                                                                completely dense. For large graphs, the space required to store
                                                                                            these dense features can easily become too large to fit in-memory.
KEYWORDS                                                                                    The features are also notoriously difficult to interpret and explain
                                                                                            which is becoming increasingly important in practice [9, 33]. Fur-
Inductive network representation learning, inductive learning, trans-
                                                                                            thermore, existing embedding methods are also unable to capture
fer learning, network embeddings, representation learning, attrib-
                                                                                            higher-order subgraph structures as well as learn a hierarchical
uted networks, function learning, network motifs, deep learning
                                                                                            graph representation from such higher-order structures. Finally,
                                                                                            these methods are also inefficient with runtimes that are orders of
1     INTRODUCTION                                                                          magnitude slower than the algorithms presented in this paper (as
Learning a useful graph representation lies at the heart and success                        shown later in Section 3). Other key differences and limitations are
of many machine learning tasks such as node and link classifica-                            discussed below.
tion [20, 34], anomaly detection [5], link prediction [6], dynamic net-                        In this work, we present a general, expressive, and flexible deep
work analysis [21], community detection [25], role discovery [27],                          graph representation learning framework called DeepGL that over-
visualization and sensemaking [24], network alignment [16], and                             comes many of the above limitations.3 Intuitively, DeepGL begins
many others. Indeed, the success of machine learning methods                                by deriving a set of base features using the graph structure and any
largely depends on data representation [12, 28]. Methods capable of                         attributes (if available). The base features are iteratively composed
learning such representations have many advantages over feature                             using a set of learned relational feature operators that operate over
                                                                                            the feature values of the (distance-ℓ) neighbors of a graph element
1 This manuscript first appeared in April 2017 as R. Rossi et al., “Deep Feature Learning
                                                                                            (node, edge; see Table 1) to derive higher-order features from lower-
for Graphs” [30].
                                                                                            order ones forming a hierarchical graph representation where each
                                                                                            layer consists of features of increasingly higher orders. At each
This paper is published under the Creative Commons Attribution-NonCommercial-
NoDerivs 4.0 International (CC BY-NC-ND 4.0) license. Authors reserve their rights to       feature layer, DeepGL searches over a space of relational functions
disseminate the work on their personal and corporate Web sites with the appropriate
attribution.
                                                                                            2 The terms transfer learning and inductive learning are used interchangeably.
WWW ’18 Companion, April 23–27, 2018, Lyon, France
                                                                                            3 Note a deep learning method as defined by Bengio et al. [7, 8] is one that learns multiple
© 2018 IW3C2 (International World Wide Web Conference Committee), published
under Creative Commons CC BY-NC-ND 4.0 License.                                             levels of representation with higher levels capturing more abstract concepts through a
ACM ISBN 978-1-4503-5640-4/18/04.                                                           deeper composition of computations [12, 17]. This definition includes neural network
https://doi.org/10.1145/3184558.3191524                                                     based approaches as well as DeepGL and many other deep learning paradigms.
WWW ’18 Companion, April 23–27, 2018, Lyon, France                                                                                              R. A. Rossi et al.


defined compositionally in terms of a set of relational feature opera-          Table 1: Summary of notation. Matrices are bold upright ro-
tors applied to each feature given as output in the previous layer.             man letters; vectors are bold lowercase letters.
Features (or relational functions) are retained if they are novel and                           G    (un)directed (attributed) graph
thus add important information that is not captured by any other                                A    sparse adjacency matrix of the graph G = (V , E)
feature in the set. See below for a summary of the advantages and                         N, M       number of nodes and edges in the graph
properties of DeepGL.                                                                        F, L    number of learned features and layers
                                                                                                G    set of graph elements {д1, д2, · · · } (nodes, edges)
1.1     Summary of Contributions                                                   dv+ , dv− , dv    outdegree, indegree, degree of vertex v
                                                                                Γ +(дi ), Γ −(дi )   out/in neighbors of graph element дi
The proposed approach, DeepGL, provides a general powerful                                 Γ(дi )    neighbors (adjacent graph elements) of дi
framework for learning deep graph representations from attrib-                            Γℓ (дi )   ℓ-neighborhood Γ(дi ) = {дj ∈ G | dist(дi , дj ) ≤ ℓ }
uted graphs that are naturally inductive for use in across-network                 dist(дi , дj )    shortest distance between дi and дj
learning tasks. DeepGL overcomes many limitations of existing                                    S   set of graph elements related to дi , e.g., S = Γ(дi )
work and has the following key properties:                                                      X    a feature matrix
    • Novel framework: This paper presents a deep hierarchical                                   x   an N or M -dimensional feature vector
                                                                                                xi   the i-th element of x for graph element дi
       inductive graph representation learning framework called
                                                                                               |X|   number of nonzeros in a matrix X
       DeepGL for large (attributed) networks that generalizes for                              F    set of feature definitions/functions from DeepGL
       discovering both node and edge features. DeepGL searches                                Fk    k-th feature layer (where k is the depth)
       a space of relational functions (representing features) that                             fi   relational function (definition) of xi
       are expressed as compositions of relational feature operators                            Φ    set of relational operators Φ = {Φ1, · · · , ΦK }
       applied to a set of base features. The framework is flexible                          K(·)    a feature score function
       with many interchangeable components, expressive, and                                     λ   tolerance/feature similarity threshold
       shown to be effective for a wide variety of applications.                                α    transformation hyperparameter (e.g., bin size in log binning 0 ≤
                                                                                                     α ≤ 1)
    • Inductive representation learning: Contrary to existing                      x′ = Φi ⟨x⟩       relational operator applied to each graph element
       node embedding methods, DeepGL is naturally inductive by
       learning relational functions that generalize for computa-
       tion on any arbitrary graph and therefore supports across-
                                                                                the feature matrix X contains only the attributes given as input by
       network transfer learning tasks.
                                                                                the user. If no attributes are provided, then X will consist of only the
    • Space efficiency: While most existing methods learn dense
                                                                                base features derived below. Note that DeepGL can use any arbitrary
       high-dimensional feature vectors that are often impractical
                                                                                set of base features, and thus it is not limited to the base features
       for large graphs (e.g., too large to fit in-memory), DeepGL
                                                                                discussed below. Given a graph G = (V , E), we first decompose G
       is space-efficient by learning a sparse graph representation
                                                                                into its smaller subgraph components called graphlets (network
       that requires up to 6x less space than existing work.
                                                                                motifs) [1] using local graphlet decomposition methods [3] and
    • Fast, parallel, and scalable: It is fast with a runtime that
                                                                                concatenate the graphlet count-based feature vectors to the feature
       is linear in the number of edges. It scales to large graphs via
                                                                                matrix X. This work derives such features by counting all node or
       a simple and efficient parallelization. Notably, strong scaling
                                                                                edge orbits with up to 4 and/or 5-vertex graphlets. Orbits (graphlet
       results are observed in Section 3.
                                                                                automorphisms) are counted for each node or edge in the graph
    • Hierarchical graph representation: DeepGL learns hier-
                                                                                based on whether a node or edge-based feature representation is
       archical graph representations with multiple layers where
                                                                                warranted (as our approach naturally generalizes to both). Note
       each successive layer uses the output from the previous layer
                                                                                there are 15 node and 12 edge orbits with 2-4 nodes; and 73 node
       as input to derive features of a higher-order.
                                                                                and 68 edge orbits with 2-5 nodes.
    • Interpretable and explainable: Unlike existing embed-
                                                                                   We also derive simple base features such as in/out/total/weighted
       ding methods, DeepGL learns interpretable and explainable
                                                                                degree and k-core numbers for each graph element (node, edge)
       features.
                                                                                in G. For edge feature learning we derive edge degree features for
                                                                                each edge (v, u) ∈ E and each ◦ ∈ {+, ×} as follows:
2     FRAMEWORK                                                                      +
                                                                                      dv ◦ du+ , dv− ◦ du− , dv− ◦ du+ , dv+ ◦ du− , dv ◦ du
                                                                                                                                                
This section presents the DeepGL framework. Since the framework                                                                                       (1)
                                                                                where dv = dv+ ◦ dv− and recall from Table 1 that dv+ , dv− , and dv
naturally generalizes for learning node and edge representations,
it is described generally for a set of graph elements (e.g., nodes or
                                                                                denote the out/in/total degree of v. In addition, egonet features
edges).4
                                                                                are also used [4]. Given a node v and an integer ℓ, the ℓ-egonet
                                                                                of v is defined as the set of graph elements ℓ-hops away from v
2.1     Base Graph Features                                                     (i.e., distance at most ℓ) and all edges and nodes between that set.
The first step of DeepGL (Alg. 1) is to derive a set of base graph fea-         The external and within-egonet features for nodes are provided
tures5 using the graph topology and attributes (if available). Initially,       in Figure 1 and used as base features in DeepGL-node. For all the
4 For convenience, DeepGL-edge and DeepGL-node are sometimes used to refer to
                                                                                above base features, we also derive variations based on direction
the edge and node representation learning variants of DeepGL, respectively.     (in/out/both) and weights (weighted/unweighted). Observe that
5 The term graph feature refers to an edge or node feature.                     DeepGL naturally supports many other graph properties including
Deep Inductive Network Representation Learning                                                                 WWW ’18 Companion, April 23–27, 2018, Lyon, France


                            within-­‐ego                                                          as across-network prediction, anomaly detection, graph similarity,
         ego-­‐center
                            external-­‐ego                                                        matching, among others.
                                                                                                      2.2.1 Composing Relational Functions. The space of relational
                                                                                                  functions searched via DeepGL is defined compositionally in terms
                                                                                                  of a set of relational feature operators Φ = {Φ1 , · · · , ΦK }. A few
          (a) External egonet features                       (b) Within egonet features
                                                                                                  relational feature operators are defined formally in Table 2; see [28]
                                                                                                  (pp. 404) for a wide variety of other useful relational feature op-
Figure 1: Egonet Features. The set of base (ℓ=1 hop)-egonet
                                                                                                  erators. The expressivity of DeepGL (space of relational functions
graph features. (a) the external egonet features; (b) the
                                                                                                  expressed by DeepGL) depends on a few flexible and interchange-
within egonet features. See the legend for the vertex types:
                                                                                                  able components including: (i) the initial base features (derived
ego-center (•), within-egonet vertex (•), and external egonet
                                                                                                  using the graph structure, initial attributes given as input, or both),
vertices (◦).
                                                                                                  (ii) a set of relational feature operators Φ = {Φ1 , · · · , ΦK }, (iii) the
                                                                                                  sets of “related graph elements” S ∈ S (e.g., the in/out/all neigh-
efficient/linear-time properties such as PageRank. Moreover, fast                                 bors within ℓ hops of a given node/edge) that are used with each
approximation methods with provable bounds can also be used to                                    relational feature operator Φp ∈ Φ, and finally, (iv) the number of
derive features such as the local coloring number and largest clique                              times each relational function is composed with another (i.e., the
centered at the neighborhood of each graph element (node, edge)                                   depth). Observe that under this formulation each feature vector
in G.                                                                                             x ′ from X (that is not a base feature) can be written as a composi-
   A key advantage of DeepGL lies in its ability to naturally handle                              tion of relational feature operators applied over a base feature. For
attributed graphs. In particular, any set of initial attributes given as                          instance, given an initial base feature x, by abuse of notation let
input can simply be concatenated with X and treated the same as                                   x ′ = Φk (Φj (Φi ⟨x⟩)) = (Φk ◦ Φj ◦ Φi )(x) be a feature vector given
the initial base features.                                                                        as output by applying the relational function constructed by com-
                                                                                                  posing the relational feature operators Φk ◦ Φj ◦ Φi to every graph
                                                                                                  element дi ∈ G and its set S of related elements.7 Obviously, more
                             x1"         ⋯"                      …"
                                                                         ⋯"                       complex relational functions are easily expressed such as those
                             x2"           xj          ⋯"                                         involving compositions of different relational feature operators
                           ⋯"                                   ⋯"                        X,"ℱ!
                                         ⋯" ⋯"
                                                 wjk   xk"                                        (and possibly different sets of related graph elements). Furthermore,
                                   wij
                             xi"                                                                  DeepGL is able to learn relational functions that often correspond
                                                       ⋯"                ⋯"                       to increasingly higher-order subgraph features based on a set of ini-
                           ⋯"                                                                     tial lower-order (base) subgraph features (Figure 2). Intuitively, just
        !input              ℱ1             ℱ2          ℱ3       ⋯"         ℱ/                     as filters are used in Convolutional Neural Networks (CNNs) [12],
                                                                                                  one can think of DeepGL in a similar way, but instead of simple
                                                                                                  filters, we have features derived from lower-order subgraphs be-
Figure 2: Overview of the DeepGL           architecture for graph rep-
                                                                                                  ing combined in various ways to capture higher-order subgraph
resentation learning. Let W = w i j be a matrix of feature
                                          
                                                                                                  patterns of increasingly complexity at each successive layer.
weights where w i j (or Wi j ) is the weight between the fea-
ture vectors xi and xj . Notice that W has the constraint that                                       2.2.2 Summation and Multiplication of Functions. We can also
i < j < k and xi , xj , and xk are increasingly deeper. Each                                      derive a wide variety of relational functions compositionally by
feature layer Fh ∈ F defines a set of unique relational func-                                     adding and multiplying relational functions (e.g., Φi + Φj , and Φi ×
tions Fh = { · · ·, fk , · · · } of order h (depth) and each fk ∈ Fh                              Φj ). A sum of relational functions is similar to an OR operation
denotes a relational function. Further, let F = F1 ∪F2 ∪· · ·∪Fτ                                  in that two instances are “close” if either has a large value, and
and |F | = |F1 | + |F2 | + · · · + |Fτ |. Moreover, the layers are or-                            similarly, a product of relational functions is analogous an AND
dered where F1 < F2 < · · · < Fτ such that if i < j then Fj is                                    operation as two instances are close if both relational functions
said to be a deeper layer w.r.t. Fi . See Table 1 for a summary                                   have large values.
of notation.
                                                                                                  2.3      Searching the Relational Function Space
                                                                                                  A general and flexible framework for DeepGL is given in Alg. 1.
2.2     Relational Function Space & Expressivity                                                  Recall that DeepGL begins by deriving a set of base features which
In this section, we formulate the space of relational functions6 that                             are used as a basis for learning deeper and more discriminative
can be expressed and searched over by DeepGL. A relational func-                                  features of increasing complexity (Line 2). The base feature vectors
tion (feature) in DeepGL is defined as a composition of relational                                are then transformed (Line 3). For instance, one may transform
feature operators applied to an initial base feature x. Recall that                               each feature vector xi using logarithmic binning as follows: sort xi
unlike recent node embedding methods [13, 23, 32], the proposed ap-                               in ascending order and set the αM graph elements (edges/nodes)
proach learns graph functions that are transferable across-networks
                                                                                                  7 For simplicity, we use Φ⟨x⟩ (whenever clear from context) to refer to the application
for a variety of important graph-based transfer learning tasks such
                                                                                                  of Φ to all sets S derived from each graph element дi ∈ G and thus the output of Φ⟨x⟩
6 The terms graph function and relational function are used interchangeably.                      in this case is a feature vector with a single feature-value for each graph element.
WWW ’18 Companion, April 23–27, 2018, Lyon, France                                                                                           R. A. Rossi et al.


                                                                                     deeper higher-order (edge/node) graph functions: F1 < F2 < · · · <
Algorithm 1 DeepGL: Deep Inductive Graph Representation                              Fτ s.t. if i < j then Fj is said to be deeper than Fi . In particular, the
Learning Framework                                                                   feature layers F2 , F3 , · · · , Fτ are derived as follows (Alg. 1 Lines 4-
Require:                                                                             11): First, we derive the feature layer Fτ by searching over the space
  a (un)directed graph G = (V , E); a set of relational feature operators            of graph functions that arise from applying the relational feature
  Φ = {Φ1, · · · , ΦK }, and a feature similarity threshold λ.                       operators Φ to each of the novel features fi ∈ Fτ −1 learned in the
                                                                                     previous layer (Alg. 1 Line 5). An algorithm for deriving a feature
 1: F1 ← ∅ and initialize X if not given as input
                                                                                     layer is provided in Alg. 2. Next, the feature vectors from layer Fτ
 2: Given G, construct base features (see text for further details) and
                                                                                     are transformed in Line 7 as discussed previously.
    concatenate the feature vectors to X and add the function definitions
    to F1 ; and set F ← F1 .                                                            The resulting features in layer τ are then evaluated. The feature
 3: Transform base feature vectors; Set τ ← 2
                                                                                     evaluation routine (in Alg. 1 Line 8) chooses the important features
                                                                                     (relational functions) at each layer τ from the space of novel rela-
 4: repeat                                   ▷ feature layers Fτ for τ = 2, ..., T
                                                                                     tional functions (at depth τ ) constructed by applying the relational
 5:    Search the space of features defined by applying   relational feature        feature operators to each feature (relational function) learned (and
       operators Φ = {Φ1, · · · , ΦK } to features · · · xi xi +1 · · ·
                                                   
                                                                                     given as output) in the previous layer τ − 1. Notice that DeepGL is
       given as output in the previous layer Fτ −1 (via Alg. 2). Add feature
       vectors to X and functions/def. to Fτ .
                                                                                     extremely flexible as the feature evaluation routine (Alg. 3) called
                                                                                     in Line 8 of Alg. 1 is completely interchangeable and can be fine-
 6:    Diffuse new feature vectors via a feature diffusion process (Eq. 3)           tuned for specific applications and/or data. This approach derives
 7:    Transform feature vectors of layer Fτ                                         a score between pairs of features. Pairs of features xi and xj that
 8:    Evaluate the features (functions) in layer Fτ using the criterion K           are strongly dependent as determined by the hyperparameter λ and
       to score feature pairs along with a feature selection method to select        evaluation criterion K are assigned Wi j = K(xi , xj ) and Wi j = 0
       a subset (Alg. 3).                                                            otherwise (Alg. 3 Line 2-6). More formally, let EF denote the set of
 9:    Discard features from X that were pruned (not in Fτ ) and set F ←             connections representing dependencies between features:
       F ∪ Fτ
                                                                                               E F = (i, j) | ∀(i, j) ∈ |F | × |F | s.t. K(xi , xj ) > λ
                                                                                                      
10:    Set τ ← τ + 1 and initialize Fτ to ∅ for next feature layer
                                                                                                                                                             (2)
11: until no new features emerge or the max number of layers is reached              The result is a weighted feature dependence graph GF . Now, GF is
12: return X and the set of relational functions (definitions) F
                                                                                     used to select a subset of important features from layer τ . Features
                                                                                     are selected as follows: First, the feature graph GF is partitioned
                                                                                     into groups of features {C1 , C2 , . . .} where each set Ck ∈ C repre-
                                                                                     sents features that are dependent (though not necessarily pairwise
with smallest values to 0 where 0 < α < 1, then set α fraction of                    dependent). To partition the feature graph GF , Alg. 3 uses con-
remaining graph elements with smallest value to 1, and so on. Many                   nected components, though other methods are also possible, e.g.,
other techniques exist for transforming the feature vectors and the                  a clustering or community detection method. Next, one or more
selected technique will largely depend on the graph structure.                       representative features are selected from each group (cluster) of
   The framework proceeds to learn a hierarchical graph representa-                  dependent features. Alternatively, it is also possible to derive a new
tion (Figure 2) where each successive layer represents increasingly                  feature from the group of dependent features, e.g., finding a low-
                                                                                     dimensional embedding of these features or taking the principal
                                                                                     eigenvector. In Alg. 3 the earliest feature in each connected com-
Table 2: Definitions of a few relational feature operators. Re-
                                                                                     ponent Ck = {..., fi , ..., f j , ...} ∈ C is selected and all others are
call the notation from Table 1. For generality, S is defined in
                                                                                     removed. After pruning the feature layer Fτ , the discarded features
Table 1 as a set of related graph elements (nodes, edges) of
                                                                                     are removed from X and DeepGL updates the set of features learned
дi and thus s j ∈ S may be an edge s j = e j or a node s j = v j ; in
                                                                                     thus far by setting F ← F ∪ Fτ (Alg. 1: Line 9). Next, Line 10
this work S ∈ Γℓ (дi ), Γℓ+(дi ), Γℓ−(дi ) . The relational operators
                                                                                     increments τ and sets Fτ ← ∅. Finally, we check for convergence,
generalize to ℓ-distanceneighborhoods (e.g.,Γℓ (дi ) where ℓ is
                                                                                     and if the stopping criterion is not satisfied, then DeepGL learns
the distance). Note x = x 1 x 2 · · · x i · · · ∈ RM where x i
                                                                                     an additional feature layer (Line 4-11).
is the i-th element of x for дi .
                                                                                        In contrast to node embedding methods that output only a node
                                                                                     feature matrix X, DeepGL also outputs the (hierarchical) relational
        Operator       Definition
                                                                                     functions F corresponding to the learned features. Maintaining the
                       Φ ⟨S, x⟩ =        xj
                                     Î
        Hadamard
                                  s j ∈S                                             relational functions are important for transferring the features to
             mean      Φ ⟨S, x⟩ = |S |
                                    1 Í
                                            xj                                       another arbitrary graph of interest, but also for interpreting them.
                                       s ∈S
                                    Í j                                              Moreover, DeepGL is an inductive representation learning approach
             sum       Φ ⟨S, x⟩ =            xj
                                    s j ∈S                                           as the relational functions can be used to derive embeddings for
        maximum        Φ ⟨S, x⟩ = max x j                                            new nodes or even graphs.
                                  s j ∈S
                                                  p
       Weight. Lp      Φ ⟨S, x⟩ =        xi − x j
                                    Í
                                    s j ∈S
                                                       2                          2.4    Feature Diffusion
                       Φ ⟨S, x⟩ = exp − 12     xi − x j
                                           Í 
              RBF
                                                  σ   s j ∈S                         We introduce the notion of feature diffusion where the feature
                                                                                     matrix at each layer can be smoothed using an arbitrary feature
Deep Inductive Network Representation Learning                                                   WWW ’18 Companion, April 23–27, 2018, Lyon, France


                                                                                      For node representation learning, the time complexity of DeepGL is:
Algorithm 2 Derive a feature layer using the features from the previous
                                                                                                                 O F (M + N F )
                                                                                                                                
                                                                                                                                                     (7)
layer and the set of relational feature operators Φ = {Φ1, · · · , ΦK }.
 1 procedure FeatureLayer(G , X, Φ, F , Fτ , λ )
                                                                                      Thus, in both cases, the runtime of representation learning in
 2    parallel for each graph element дi ∈ G do                                       DeepGL is linear in the number of edges. As an aside, the initial
 3       Set t ← | F |                                                                graphlet features are computed using fast and accurate estimation
 4       for each feature xk s.t. f k ∈ Fτ −1 in order do                            methods, see Ahmed et al. [3].
 5           for each S ∈ Γℓ+(дi ), Γℓ−(дi ), Γℓ (дi ) do
 6               for each relational operator Φ ∈ Φ do                                   2.5.2 Inductive relational functions. We now state the computa-
 7                  X i t = Φ(S, xk ) and t ← t + 1
                                                                                      tional complexity of directly computing the set of inductive rela-
 8     Add feature definitions to Fτ
 9     return feature matrix X and Fτ
                                                                                      tional functions (feature definitions) which were previously learned
                                                                                      on another arbitrary graph. Computation of the relational functions
Algorithm 3 Score and prune the feature layer                                         F on another arbitrary graph is important for inductive across-
                                                                                      network learning tasks. Given the set of learned relational functions
 1 procedure EvaluateFeatureLayer(G , X, F , Fτ )
                                                                                      F , the total computational complexity of the edge relational func-
 2    Let GF = (VF , E F , W)
                                                                                      tions is:
       parallel for each feature f i ∈ Fτ do
                                                                                                                     O FM
 3                                                                                                                           
 4        for each feature f j ∈ (Fτ −1 ∪ · · · ∪ F1 ) do
                                                                                                                                                         (8)
              if K xi , x j > λ then
                           
                                                                                              the time complexity of the node relational functions is also
                                                                                      Similarly,
 5
 6                E F = E F ∪ {(i, j)}                                               O F M . Thus, the runtime of deriving the edge and node relational
 7                Wi j = K xi , x j
                                                                                      functions in DeepGL is linear in the number of edges. Computing
 8     Partition GF using conn. components C = { C1, C2, . . . }
                                                                                      the set of inductive relational functions on another arbitrary graph
 9     parallel for each Ck ∈ C do                                ▷ Remove features
10        Find f i s.t. ∀f j ∈ Ck : i < j .                                           obviously requires less work than learning the actual set of induc-
11        Remove Ck from Fτ and set Fτ ← Fτ ∪ {f i }                                  tive relational functions (Section 2.5.1). The key difference is that
                                                                                      features are not evaluated when deriving the relational functions
                                                                                      directly. In contrast, representation learning in DeepGL scores the
diffusion process. As an example, suppose X is the resulting feature                  features at each layer.
matrix from layer τ , then we can set X̄(0) ← X and solve
                                                                                      3     EXPERIMENTS
                             X̄(t ) = D−1 AX̄(t −1)                             (3)
                                                                                      This section demonstrates the effectiveness of the proposed frame-
where D is the diagonal degree matrix and A is the adjacency                          work.
matrix of G. The diffusion process above is repeated for a fixed
number of iterations t = 1, 2, ...,T or until convergence; and X̄(t ) =               3.1    Experimental settings
D−1 AX̄(t −1) corresponds to a simple feature propagation. More                       In these experiments, we use the following instantiation of DeepGL:
complex feature diffusion processes can also be used in DeepGL                        Features are transformed using logarithmic binning and evaluated
such as the normalized Laplacian feature diffusion defined as                         using a simple agreement score function where K(xi , xj ) = fraction
            X̄(t ) = (1 − θ )LX̄(t −1) + θ X,          for t = 1, 2, ...        (4)   of graph elements that agree. More formally, agreement scoring is
                                                                                      defined as:
where L is the normalized Laplacian:
                                                                                                              (x ik , x jk ), ∀k = 1, . . . ,N | x ik = x jk
                                                                                                            
                               L = I − D /2 AD /2
                                           1       1                                          K(xi , xj ) =                                                     (9)
                                                                                (5)                                                N
The resulting diffused feature vectors X̄ =                                           where x ik and x jk are the k-th feature value of the N -dimensional
                                                                                 
                                                  x̄1 x̄2 · · ·
are effectively smoothed by the features of related graph elements                    vectors xi and xj , respectively. Unless otherwise mentioned, we
(nodes/edges) governed by the particular diffusion process. Notice                    set α = 0.5 (bin size of logarithmic binning) and perform              a grid
                                                                                      search over λ ∈ {0.01, 0.05, 0.1, 0.2, 0.3} and Φ ∈ Φmean , Φsum ,
                                                                                                                                                      
that feature vectors given as output at each layer can be diffused
(e.g., after Line 5 or 9 of Alg. 1). Note X̄ can be leveraged in a                    Φprod , {Φmean , Φsum }, {Φprod , Φsum }, {Φprod , Φmean } . See Table 2.
variety of ways:
                  X ← X̄ (replacing previous) or concatenated by                     Note Φprod refers to the Hadamard relational operator defined for-
X ← X X̄ . Feature diffusion can be viewed as a form of graph                         mally in Table 2. As an aside, DeepGL has fewer hyperparameters
regularization as it can improve the generalizability of a model                      than node2vec, DeepWalk, and LINE used in the comparison below.
learned using the graph embedding.                                                    The specific model defined by the above instantiation of DeepGL is
                                                                                      selected using 10-fold cross-validation on 10% of the labeled data.
2.5     Computational Complexity                                                      Experiments are repeated for 10 random seed initializations. All
Recall that M is the number of edges, N is the number of nodes,                       results are statistically significant with p-value < 0.01.
and F is the number of features.                                                         We evaluate the proposed framework against node2vec [13],
                                                                                      DeepWalk [23], and LINE [32]. For node2vec, we use the hyper-
   2.5.1 Learning. The total computational complexity of the edge                     parameters and grid search over p, q ∈ {0.25, 0.50, 1, 2, 4} as men-
representation learning from the DeepGL framework is:                                 tioned in [13]. The experimental setup mentioned in [13] is used for
                          O F (M + MF )
                                        
                                                               (6)                    DeepWalk and LINE. Unless otherwise mentioned, we use logistic
WWW ’18 Companion, April 23–27, 2018, Lyon, France                                                                                                                                         R. A. Rossi et al.


regression with an L2 penalty and one-vs-rest for multiclass prob-                                               1
lems. Data has been made available at NetworkRepository [26].8                                                  0.9




                                                                                            Embedding Density
                                                                                                                0.8
Table 3: AUC scores for Within-network Link Classification                                                      0.7
                                                                                                                                                                                DeepGL-edge
                                                                                                                0.6
                                                                                                                                                                                DeepGL-node
                                                                                                                0.5                                                            node2vec
                                             escorts           yahoo-msg
                                                                                                                0.4
                           DeepGL            0.6891            0.9410                                           0.3
          mean            node2vec           0.6426            0.9397                                           0.2
       xi + x j 2
               
                         DeepWalk            0.6308            0.9317                                           0.1
                             LINE            0.6550            0.7967                                            0
                                                                                                                                   fb-MIT        yahoo-msg       enron        fb-PU         DD21
                           DeepGL            0.6339            0.9324
         product          node2vec           0.5445            0.8633               Figure 3: DeepGL requires up to 6x less space than node2vec
          xi ⊙ x j       DeepWalk            0.5366            0.8522               and other methods that learn dense embeddings.
                             LINE            0.5735            0.7384
                           DeepGL            0.6857            0.9247
       weighted l1        node2vec           0.5050            0.7644
         xi − x j        DeepWalk            0.5040            0.7609
                             LINE            0.6443            0.7492               state-of-the-art methods [13, 23]. We focus first on node represen-
                           DeepGL            0.6817            0.9160               tations since existing methods are limited to only node features.
       weighted l2        node2vec           0.4950            0.7623               Results are shown in Figure 3. In all cases, the node representations
        (xi − x j )◦2    DeepWalk            0.4936            0.7529               learned by DeepGL are extremely sparse and significantly more
                             LINE            0.6466            0.5346
                                                                                    space-efficient than node2vec [13] as observed in Figure 3. Deep-
                                                                                    Walk and LINE use nearly the same space as node2vec, and thus
                                                                                    are omitted for brevity. Strikingly, DeepGL uses only a fraction of
3.2     Within-Network Link Classification                                          the space required by existing methods (Figure 3). Moreover, the
We first evaluate the effectiveness of DeepGL for link classifica-                  density
                                                                                             of node
                                                                                                   and edge representations
                                                                                                                               from DeepGL is between
tion. To be able to compare DeepGL to node2vec and the other                          0.162, 0.334 for nodes and 0.164, 0.318 for edges and up to 6×
methods, we focus in this section on within-network link classifica-                more space-efficient than existing methods.
tion. For comparison, we use the same set of binary operators to                       Notably, recent node embedding methods not only output dense
construct features for the edges indirectly using the learned node                  node features, but are also real-valued and often negative (e.g., [13,
representations: Given the feature vectors xi and xj for node i and j,              23, 32]). Thus, they require 8 bytes per feature-value, whereas
(xi +xj ) 2 is the mean; xi ⊙xj is the (Hadamard) product; xi − xj
         
                                                                                    DeepGL requires only 2 bytes and can sometimes be reduced to
and (xi − xj )◦2 is the weighted-l1 and weighted-l2 binary opera-                   even 1 byte if needed by adjusting α (i.e., the bin size of the log
tors, respectively.9 Note that these binary operators (used to create               binning transformation). To understand the impact of this, assume
edge features) are not to be confused with the relational feature                   both approaches learn a node representation with 128 dimensions
operators defined in Table 2. In Table 3, we observe that DeepGL                    (features) for a graph with 10,000,000 nodes. In this case, node2vec,
outperforms node2vec, DeepWalk, and LINE with an average gain                       DeepWalk, and LINE require 10.2GB, whereas DeepGL uses only
between 18.09% and 20.80% across all graphs and binary operators.                   0.768GB (assuming a modest 0.3 density) — a significant reduction
   Notice that node2vec, DeepWalk, and LINE all require that the                    in space by a factor of 13.
training graph contain at least one edge among each node in G.
However, DeepGL overcomes this fundamental limitation and can
actually predict the class label of edges that are not in the training
graph as well as the class labels of edges in an entirely different                                                                    10 5        DeepGL
network.                                                                                                                               10 4
                                                                                                                                                   node2vec




                                                                                                                      time (seconds)
                                                                                                                                       10 3
3.3     Analysis of Space-Efficiency
                                                                                                                                       10 2
Learning sparse space-efficient node and edge feature representa-
                                                                                                                                       10 1
tions is of vital importance for large networks where storing even
                                                                                                                                       10 0
a modest number of dense features is impractical (especially when
stored in-memory). Despite the importance of learning a sparse                                                                         10 -1

space-efficient representation, existing work has been limited to                                                                      10 -2

discovering completely dense (node) features [13, 23, 32]. To un-                                                                         10 1   10 2   10 3   10 4   10 5   10 6   10 7   10 8
derstand the effectiveness of the proposed framework for learning                                                                                               nodes
sparse graph representations, we measure the density of each rep-
resentation learned from DeepGL and compare these against the                       Figure 4: Runtime comparison on Erdös-Rényi graphs with
                                                                                    an average degree of 10. The proposed approach is shown to
8 See http://networkrepository.com/ for data description and statistics
9 Note x◦2 is the element-wise Hadamard power; x ⊙ x is the element-wise product.
                                                                                    be orders of magnitude faster than node2vec [13].
                                                i   j
Deep Inductive Network Representation Learning                                       WWW ’18 Companion, April 23–27, 2018, Lyon, France

          Table 4: AUC scores for node classification                     indirect approaches such as LR and SVM. In particular, rsm assigns
                                                                          a test vector xi to the class that is most similar w.r.t. the train-
                       graph     |C|    DeepGL       node2vec
                                                                          ing vectors (i.e., feature vectors of the nodes with known labels);
              DD242               20      0.730          0.673            see [29] for further details. Similarity is measured using the RBF
              DD497               20      0.696          0.660            kernel and RBF’s hyperparameter σ is set using cross-validation
               DD68               20      0.730          0.713            with a grid search over σ ∈ {0.001, 0.01, 0.1, 1}. Results are shown
         ENZYMES118               2       0.779          0.610            in Table 4. In all cases, we observe that DeepGL significantly outper-
         ENZYMES295               2       0.872          0.588            forms node2vec across all graphs and node classification problems
         ENZYMES296               2       0.823          0.610            including both binary and multiclass problems. Further, DeepGL
                                                                          achieves the best improvement in AUC on ENZYMES295 of 48%. As
                                                                          an aside, results for DeepWalk and LINE were removed for brevity
3.4    Runtime & Scalability                                              since node2vec outperformed them in all cases.
To evaluate the performance and scalability of the proposed frame-
work, we learn node representations for Erdös-Rényi graphs of in-         4   RELATED WORK
creasing size (from 100 to 10,000,000 nodes) such that each graph has
                                                                          Related research is categorized below.
an average degree of 10. We compare the performance of DeepGL
against node2vec [13] which is designed specifically to be scalable       Node embedding methods: There has been a lot of interest re-
for large graphs and shown to be faster than DeepWalk and LINE.           cently in learning a set of useful node features from large-scale
Default parameters are used for each method. In Figure 4, we ob-          networks automatically [13, 22, 23, 32]. In particular, recent meth-
serve that DeepGL is significantly faster and more scalable than          ods that apply the popular word2vec framework to learn node
node2vec. In particular, node2vec takes 1.8 days (45.3 hours) for         embeddings [13, 23, 32]. The proposed DeepGL framework differs
10 million nodes, whereas DeepGL finishes in only 15 minutes; see         from these methods in six fundamental ways: (1) DeepGL learns
Figure 4. Strikingly, this is 182 times faster than node2vec.             complex relational functions that generalize for across-network
                                                                          transfer learning. Features learned from DeepGL on one graph can
                       12
                                                                          be extracted from another graph for transfer learning tasks such
                                                                          as network alignment, graph similarity, role discovery, temporal
                       10                                                 graph modeling, among others. (2) DeepGL learns sparse features
                        8
                                                                          and thus is extremely space-efficient for large networks. (3) DeepGL

             Speedup
                                                                          learns important and useful edge and node representations whereas
                        6                                                 existing work is limited to node features [13, 23, 32]. (4) DeepGL nat-
                                                                          urally supports attributed graphs. (5) DeepGL is fast and efficient
                        4                      DeepGL-Node
                                               DeepGL-Node+Attr
                                                                          with a runtime that is linear in the number of edges. (6) DeepGL is
                        2                      DeepGL-Edge                also completely parallel and shown in Section 3 to scale strongly.
                                               DeepGL-Edge+Attr               There is also another related body of work focused on attributed
                        0
                            1     4        8        12           16       graphs. Recently, Huang et al. [14] proposed a label informed em-
                                Number of processing units                bedding method for attributed networks. This approach assumes
Figure 5: Parallel speedup of different variants from the                 the graph is labeled and uses this information to improve predictive
DeepGL framework. See text for discussion.                                performance. However, this work is significantly different. First
                                                                          and foremost, while DeepGL is able to naturally support attrib-
                                                                          uted graphs, this work does not focus on such graphs. Moreover,
                                                                          DeepGL does not require attributes or class labels on the nodes.
3.5    Parallel Scaling
                                                                          Another important fundamental difference is that DeepGL learns
This section investigates the parallel performance of DeepGL. To          features representing relational functions that generalize for ex-
evaluate the effectiveness of the parallel algorithm we measure           traction on any other arbitrary graph. The relational functions
speedup defined as Sp = TTp1 where T1 and Tp are the execution time       naturally represent higher-order structures when based on lower-
of the sequential and parallel algorithms (w/ p processing units),        order subgraph features (Figure 2). DeepGL also learns features that
respectively. In Figure 5, we observe strong parallel scaling for all     are sparse and therefore space-efficient for large graphs. Moreover,
DeepGL variants with the edge representation learning variants            it is fast with a runtime that is linear in the number of edges and is
performing slightly better than the node representation learning          completely parallel with strong scaling. There has also been some
methods from DeepGL. Results are reported for soc–gowalla on a            recent work on heterogeneous network embeddings [10, 11, 37],
machine with 4 Intel Xeon E5-4627 v2 3.3GHz CPUs. Other graphs            semi-supervised network embeddings [15, 38], and methods for
and machines gave similar results.                                        improving the learned representations [31, 35, 36]. This work in-
                                                                          vestigates entirely different problems than the one discussed in this
3.6    Node Classification                                                paper.
For node classification, we use the i.i.d. variant of rsm [29] since it       We can also use the inferred embeddings for graph-based transfer
is able to handle multiclass problems in a direct fashion (as opposed     learning. This is possible since DeepGL learns relational functions
to indirectly, e.g., one-vs-rest) and consistently outperformed other     that generalize across-networks and therefore are easily extracted
WWW ’18 Companion, April 23–27, 2018, Lyon, France                                                                                                        R. A. Rossi et al.


on another arbitrary graph. Other key differences were summarized                       [8] Yoshua Bengio. 2013. Deep learning of representations: Looking forward. In SLSP.
previously in Section 1.                                                                    Springer, 1–37.
                                                                                        [9] Adrien Bibal and Benoît Frénay. 2016. Interpretability of machine learning models
Higher-order network analysis: Other methods use high-order                                 and representations: an introduction. In Proc. ESANN. 77–82.
                                                                                       [10] Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, Charu C Aggarwal, and
network properties (such as graphlet frequencies) as features for                           Thomas S Huang. 2015. Heterogeneous network embedding via deep archi-
graph classification [34]. Graphlets (network motifs) are small in-                         tectures. In SIGKDD. 119–128.
duced subgraphs and have been used for graph classification [34],                      [11] Ting Chen and Yizhou Sun. 2017. Task-Guided and Path-Augmented Heteroge-
                                                                                            neous Network Embedding for Author Identification. In WSDM. 295–304.
role discovery [2], and visualization and exploratory analysis [1].                    [12] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep learning. MIT
However, our work focuses on using graphlet frequencies as base                             Press.
                                                                                       [13] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for
features for learning node and edge representations from large                              networks. In KDD. 855–864.
networks. To the best of our knowledge, this paper is the first to                     [14] Xiao Huang, Jundong Li, and Xia Hu. 2017. Label informed attributed network
use network motifs (including all motifs of size 3, 4, and 5 vertices)                      embedding. In WSDM. 731–739.
                                                                                       [15] Thomas N Kipf and Max Welling. 2017. Semi-supervised classification with graph
as base features for graph representation learning.                                         convolutional networks. In ICLR.
                                                                                       [16] Mehmet Koyutürk, Yohan Kim, Umut Topkara, Shankar Subramaniam, Wojciech
Sparse graph feature learning: This work proposes the first prac-                           Szpankowski, and Ananth Grama. 2006. Pairwise alignment of protein interaction
tical space-efficient approach that learns sparse node/edge feature                         networks. JCB 13, 2 (2006), 182–199.
vectors. Notably, DeepGL requires significantly less space than                        [17] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. 2015. Deep learning. Nature
                                                                                            521, 7553 (2015), 436–444.
existing node embedding methods [13, 23, 32] (see Section 3). In                       [18] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
contrast, existing embedding methods store completely dense fea-                            estimation of word representations in vector space. In ICLR Workshop.
                                                                                       [19] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013.
ture vectors which is impractical for any relatively large network,                         Distributed representations of words and phrases and their compositionality. In
e.g., they require more than 3TB of memory for a 750 million node                           NIPS.
graph with 1K features.                                                                [20] Jennifer Neville and David Jensen. 2000. Iterative classification in relational data.
                                                                                            In AAAI Workshop on Learning Statistical Models from Relational Data. 13–20.
                                                                                       [21] Vincenzo Nicosia, John Tang, Cecilia Mascolo, Mirco Musolesi, Giovanni Russo,
5    CONCLUSION                                                                             and Vito Latora. 2013. Graph metrics for temporal networks. In Temporal
                                                                                            Networks. Springer, 15–40.
We proposed DeepGL, a general, flexible, and highly expressive                         [22] Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. 2016. Learning
framework for learning deep node and edge features that general-                            Convolutional Neural Networks for Graphs. In arXiv:1605.05273.
                                                                                       [23] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning
ize for across-network transfer learning tasks. Each feature learned                        of social representations. In KDD. 701–710.
by DeepGL corresponds to a composition of relational feature oper-                     [24] Robert Pienta, James Abello, Minsuk Kahng, and Duen Horng Chau. 2015. Scalable
ators applied over a base feature. Thus, features learned by DeepGL                         graph exploration and visualization: Sensemaking challenges and opportunities.
                                                                                            In BigComp.
are interpretable and naturally generalize for across-network trans-                   [25] F. Radicchi, C. Castellano, F. Cecconi, V. Loreto, and D. Parisi. 2004. Defining and
fer learning tasks as they can be derived on any arbitrary graph. The                       identifying communities in networks. PNAS 101, 9 (2004), 2658–2663.
                                                                                       [26] Ryan A. Rossi and Nesreen K. Ahmed. 2015. The Network Data Repository with
framework is flexible with many interchangeable components, ex-                             Interactive Graph Analytics and Visualization. In AAAI. http://networkrepository.
pressive, interpretable, parallel, and is both space- and time-efficient                    com
for large graphs with runtime that is linear in the number of edges.                   [27] Ryan A. Rossi and Nesreen K. Ahmed. 2015. Role Discovery in Networks. TKDE
                                                                                            27, 4 (2015), 1112–1131.
DeepGL has all the following desired properties:                                       [28] Ryan A. Rossi, Luke K. McDowell, David W. Aha, and Jennifer Neville. 2012.
      • Effective for learning functions (features) that generalize for                     Transforming graph data for statistical relational learning. JAIR 45, 1 (2012),
        graph-based transfer learning and large (attributed) graphs                         363–441.
                                                                                       [29] Ryan A. Rossi, Rong Zhou, and Nesreen K. Ahmed. 2016. Relational Similarity
      • Space-efficient requiring up to 6× less memory                                      Machines. In KDD MLG. 1–8.
      • Fast with up to 182× speedup in runtime performance                            [30] Ryan A. Rossi, Rong Zhou, and Nesreen K. Ahmed. 2017. Deep Feature Learning
                                                                                            for Graphs. In arXiv:1704.08829. 11.
      • Accurate with a mean improvement in AUC of 20% or more                         [31] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele
        on many applications                                                                Monfardini. 2009. The graph neural network model. IEEE Transactions on Neural
      • Expressive and flexible with many interchangeable com-                              Networks 20, 1 (2009), 61–80.
                                                                                       [32] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
        ponents making it useful for a range of applications, graph                         2015. LINE: Large-scale Information Network Embedding.. In WWW.
        types, and learning scenarios.                                                 [33] Alfredo Vellido, José David Martín-Guerrero, and Paulo JG Lisboa. 2012. Making
      • Parallel with strong scaling results.                                               machine learning models interpretable.. In ESANN, Vol. 12. 163–172.
                                                                                       [34] S Vichy N Vishwanathan, Nicol N Schraudolph, Risi Kondor, and Karsten M
                                                                                            Borgwardt. 2010. Graph kernels. JMLR 11 (2010), 1201–1242.
REFERENCES                                                                             [35] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural deep network embed-
                                                                                            ding. In SIGKDD. 1225–1234.
 [1] Nesreen K. Ahmed, Jennifer Neville, Ryan A. Rossi, and Nick Duffield. 2015.       [36] Jason Weston, Frédéric Ratle, and Ronan Collobert. 2008. Deep learning via
     Efficient Graphlet Counting for Large Networks. In ICDM. 10.                           semi-supervised embedding. In ICML. 1168–1175.
 [2] Nesreen K. Ahmed, Ryan A. Rossi, Theodore L. Willke, and Rong Zhou. 2017.         [37] Linchuan Xu, Xiaokai Wei, Jiannong Cao, and Philip S Yu. 2017. Embedding of
     Edge Role Discovery via Higher-order Structures. In PAKDD. Springer.                   Embedding (EOE): Joint Embedding for Coupled Heterogeneous Networks. In
 [3] Nesreen K. Ahmed, Theodore L. Willke, and Ryan A. Rossi. 2016. Estimation of           WSDM. ACM, 741–749.
     Local Subgraph Counts. In IEEE BigData. 586–595.                                  [38] Zhilin Yang, William W Cohen, and Ruslan Salakhutdinov. 2016. Revisiting semi-
 [4] Leman Akoglu, Mary McGlohon, and Christos Faloutsos. 2010. Oddball: Spotting           supervised learning with graph embeddings. arXiv preprint arXiv:1603.08861
     anomalies in weighted graphs. PAKDD (2010), 410–421.                                   (2016).
 [5] Leman Akoglu, Hanghang Tong, and Danai Koutra. 2015. Graph based anomaly
     detection and description: a survey. DMKD 29, 3 (2015), 626–688.
 [6] Mohammad Al Hasan and Mohammed J Zaki. 2011. A survey of link prediction
     in social networks. In Social Network Data Analytics. Springer, 243–275.
 [7] Yoshua Bengio. 2009. Learning deep architectures for AI. Foundations and Trends
     in Machine Learning 2, 1 (2009), 1–127.

