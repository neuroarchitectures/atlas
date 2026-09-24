# struc2vec Learning Node Representations from Structural Iden Identity Ribeiro Saverese etal 2017

> Source: `struc2vec_Learning_Node_Representations_from_Structural_Iden_Identity_Ribeiro_Saverese_etal_2017.pdf`

---

                                                 struc2vec: Learning Node Representations from Structural
                                                                         Identity
                                                     Leonardo F. R. Ribeiro                                         Pedro H. P. Saverese                              Daniel R. Figueiredo
                                             Federal University of Rio de Janeiro                          Federal University of Rio de Janeiro                  Federal University of Rio de Janeiro
                                            Systems Eng. and Comp. Science Dep.                           Systems Eng. and Comp. Science Dep.                   Systems Eng. and Comp. Science Dep.
                                                      leo@land.ufrj.br                                           savarese@land.ufrj.br                                   daniel@land.ufrj.br
                                        ABSTRACT                                                                                                 c                                                y
                                        Structural identity is a concept of symmetry in which network                                       b           d                                   x             t
                                        nodes are identified according to the network structure and their                                                             network                      v
                                        relationship to other nodes. Structural identity has been studied                                       u
                                                                                                                                                                                        w




arXiv:1704.03165v3 [cs.SI] 3 Jul 2017
                                        in theory and practice over the past decades, but only recently                                 a                   e                                                 z
                                        has it been addressed with representational learning techniques.
                                        This work presents struc2vec, a novel and flexible framework for
                                        learning latent representations for the structural identity of nodes.                         Figure 1: An example of two nodes (u and v) that are struc-
                                        struc2vec uses a hierarchy to measure node similarity at differ-                              turally similar (degrees 5 and 4, connected to 3 and 2 trian-
                                        ent scales, and constructs a multilayer graph to encode structural                            gles, connected to the rest of the network by two nodes), but
                                        similarities and generate structural context for nodes. Numerical                             very far apart in the network.
                                        experiments indicate that state-of-the-art techniques for learning
                                        node representations fail in capturing stronger notions of structural                         emerges when node function is defined solely by the network struc-
                                        identity, while struc2vec exhibits much superior performance in                               ture. In this context, not even the labels of the nodes matter but
                                        this task, as it overcomes limitations of prior approaches. As a con-                         just their relationship to other nodes (edges). Indeed, mathematical
                                        sequence, numerical experiments indicate that struc2vec improves                              sociologists have worked on this problem since the 1970s, defin-
                                        performance on classification tasks that depend more on structural                            ing and computing structural identity of individuals in social net-
                                        identity.                                                                                     works [11, 17, 19]. Beyond sociology, the role of webpages in the
                                                                                                                                      webgraph is another example of identity (in this case, hubs and
                                        CCS CONCEPTS                                                                                  authorities) emerging from the network structure, as defined by
                                        •Computing methodologies → Unsupervised learning; Learn-                                      the celebrated work of Kleinberg [8].
                                        ing latent representations; •Artificial Intelligence → Learning;                                 The most common practical approaches to determine the struc-
                                                                                                                                      tural identity of nodes are based on distances or recursions. In
                                        KEYWORDS                                                                                      the former, a distance function that leverages the neighborhood of
                                        feature learning; node embeddings; structural identity                                        the nodes is used to measure the distance between all node pairs,
                                                                                                                                      with clustering or matching then performed to place nodes into
                                                                                                                                      equivalent classes [5, 9]. In the later, a recursion with respect to
                                        1     INTRODUCTION                                                                            neighboring nodes is constructed and then iteratively unfolded
                                        In almost all networks, nodes tend to have one or more functions                              until convergence, with final values used to determine the equiva-
                                        that greatly determine their role in the system. For example, individ-                        lent classes [3, 8, 26]. While such approaches have advantages and
                                        uals in a social network have a social role or social position [11, 19],                      disadvantages, we provide an alternative methodology, one based
                                        while proteins in a protein-protein interaction (PPI) network exert                           on unsupervised learning of representations for the structural iden-
                                        specific functions [1, 22]. Intuitively, different nodes in such net-                         tity of nodes (to be presented). Recent efforts in learning latent
                                        works may perform similar functions, such as interns in the social                            representations for nodes in networks have been quite successful
                                        network of a corporation or catalysts in the PPI network of a cell.                           in performing classification and prediction tasks [6, 14, 16, 23]. In
                                        Thus, nodes can often be partitioned into equivalent classes with                             particular, these efforts encode nodes using as context a generalized
                                        respect to their function in the network.                                                     notion of their neighborhood (e.g., w steps of a random walk, or
                                           Although identification of such functions often leverage node                              nodes with neighbors in common). In a nutshell, nodes that have
                                        and edge attributes, a more challenging and interesting scenario                              neighborhoods with similar sets of nodes should have similar latent
                                        Permission to make digital or hard copies of all or part of this work for personal or         representations. But neighborhood is a local concept defined by
                                        classroom use is granted without fee provided that copies are not made or distributed         some notion of proximity in the network. Thus, two nodes with
                                        for profit or commercial advantage and that copies bear this notice and the full citation     neighborhoods that are structurally similar but that are far apart
                                        on the first page. Copyrights for components of this work owned by others than ACM
                                        must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,       will not have similar latent representations. Figure 1 illustrates the
                                        to post on servers or to redistribute to lists, requires prior specific permission and/or a   problem, where nodes u and v play similar roles (i.e., have similar
                                        fee. Request permissions from permissions@acm.org.
                                                                                                                                      local structures) but are very far apart in the network. Since their
                                        KDD ’17, August 13-17, 2017, Halifax, NS, Canada
                                        © 2017 ACM. 978-1-4503-4887-4/17/08. . . $15.00                                               neighborhoods have no common nodes, recent approaches cannot
                                        DOI: 10.1145/3097983.3098061                                                                  capture their structural similarity (as we soon show).
    It is worth noting why recent approaches for learning node                The remainder of this paper is organized as follows. Section 2
representations such as DeepWalk [16] and node2vec [6] succeed in          briefly overviews the recent related work on learning latent repre-
classification tasks but tend to fail in structural equivalence tasks.     sentations of nodes in networks. Section 3 presents the struc2vec
The key point is that many node features in most real networks             framework in detail. Experimental evaluation and comparison to
exhibit a strong homophily (e.g., two blogs with the same political        other methods are shown in Section 4. Finally, Section 5 concludes
inclination are much more likely to be connected than at random).          the paper with a brief discussion.
Neighbors of nodes with a given feature are more likely to have the
same feature. Thus, nodes that are “close” in the network and in
the latent representation will tend to share features. Likewise, two
nodes that are “far” in the network will tend to be separated in the       2   RELATED WORK
latent representation, independent of their local structure. Thus,         Embedding network nodes in (Euclidean) space has received much
structural equivalence will not properly be captured in the latent         attention over the past decades from different communities. The
representation. However, if classification is performed on features        technique is instrumental for Machine Learning applications that
that depend more on structural identity and less on homophily,             leverage network data, as node embeddings can be directly used in
then such recent approaches are likely to be outperformed by latent        tasks such as classification and clustering.
representations that better capture structural equivalence (as we              In Natural Language Processing [2], generating dense embed-
soon show).                                                                dings for sparse data has a long history. Recently, Skip-Gram [12,
    Our main contribution is a flexible framework for learning latent      13] was proposed as an efficient technique to learn embeddings
representations for the structural identity of nodes, called struc2vec.    for text data (e.g., sentences). Among other properties, the learned
It is an alternative and powerful tool to the study of structural          language model places semantically similar words near each other
identity through latent representations. The key ideas of struc2vec        in space.
are:                                                                           Learning a language model from a network was first proposed
                                                                           by DeepWalk [16]. It uses random walks to generate sequences
      • Assess structural similarity between nodes independently
                                                                           of nodes from the network, which are then treated as sentences
        of node and edge attributes as well as their position in
                                                                           by Skip-Gram. Intuitively, nodes close in the network will tend to
        the network. Thus, two nodes that have a similar local
                                                                           have similar contexts (sequences) and thus have embeddings that
        structure will be considered so, independent of network
                                                                           are near one another. This idea was later extended by node2vec [6].
        position and node labels in their neighborhoods. Our ap-
                                                                           By proposing a biased second order random walk model, node2vec
        proach also does not require the network to be connected,
                                                                           provides more flexibility when generating the context of a vertex.
        and identifies structurally similar nodes in different con-
                                                                           In particular, the edge weights driving the biased random walks
        nected components.
                                                                           can be designed in an attempt to capture both vertex homophily
      • Establish a hierarchy to measure structural similarity, al-
                                                                           and structural equivalence. However, a fundamental limitation is
        lowing progressively more stringent notions of what it
                                                                           that structurally similar nodes will never share the same context if
        means to be structurally similar. In particular, at the bot-
                                                                           their distance (hop count) is larger than the Skip-Gram window.
        tom of the hierarchy, structural similarity between nodes
                                                                               subgraph2vec [14] is another recent approach for learning em-
        depend only on their degrees, while at the top of the hier-
                                                                           beddings for rooted subgraphs, and unlike the previous techniques
        archy similarity depends on the entire network (from the
                                                                           it does not use random walks to generate context. Alternatively, the
        viewpoint of the node).
                                                                           context of a node is simply defined by its neighbors. Additionally,
      • Generates random contexts for nodes, which are sequences
                                                                           subgraph2vec captures structural equivalence by embedding nodes
        of structurally similar nodes as observed by a weighted
                                                                           with the same local structure to the same point in space. Nonethe-
        random walk traversing a multilayer graph (and not the
                                                                           less, the notion of structural equivalence is very rigid since it is
        original network). Thus, two nodes that frequently appear
                                                                           defined as a binary property dictated by the Weisfeiler-Lehman
        with similar contexts will likely have similar structure.
                                                                           isomorphism test [21]. Thus, two nodes that are structurally very
        Such context can be leveraged by language models to learn
                                                                           similar (but fail the test) and have non-overlapping neighbors may
        latent representation for the nodes.
                                                                           not be close in space.
We implement an instance of struc2vec and show its potential                   Similarly to subgraph2vec, considerable effort has recently been
through numerical experiments on toy examples and real networks,           made on learning richer representations for network nodes [4, 24].
comparing its performance with DeepWalk [16] and node2vec [6] –            However, building representations that explicitly capture structural
two state-of-the-art techniques for learning latent representations        identity is a relative orthogonal problem that has not received much
for nodes, and with RolX [7] – a recent approach to identify roles         attention. This is the focus of struc2vec.
of nodes. Our results indicate that while DeepWalk and node2vec                A recent approach to explicitly identify the role of nodes using
fail to capture the notion of structural identity, struc2vec excels on     just the network structure is RolX [7]. This unsupervised approach
this task – even when the original network is subject to strong ran-       is based on enumerating various structural features for nodes, find-
dom noise (random edge removal). We also show that struc2vec is            ing the more suited basis vector for this joint feature space, and
superior in a classification task where node labels depends more on        then assigning for every node a distribution over the identified roles
structural identity (i.e., air-traffic networks with labels representing   (basis), allowing for mixed membership across the roles. Without
airport activity).                                                         explicitly considering node similarity or node context (in terms of
structure), RolX is likely to miss node pairs that are structurally      denotes the number of nodes in the network and k ∗ its diameter.
equivalent (to be shown).                                                Let Rk (u) denote the set of nodes at distance (hop count) exactly
                                                                         k ≥ 0 from u in G. Note that R 1 (u) denotes the set of neighbors of
3     STRUC2VEC                                                          u and in general, Rk (u) denotes the ring of nodes at distance k. Let
Consider the problem of learning representations that capture the        s(S) denote the ordered degree sequence of a set S ⊂ V of nodes.
structural identity of nodes in the network. A successful approach          By comparing the ordered degree sequences of the rings at dis-
should exhibit two desired properties:                                   tance k from both u and v we can impose a hierarchy to measure
       • The distance between the latent representation of nodes         structural similarity. In particular, let fk (u, v) denote the structural
         should be strongly correlated to their structural similarity.   distance between u and v when considering their k-hop neighbor-
         Thus, two nodes that have identical local network struc-        hoods (all nodes at distance less than or equal to k and all edges
         tures should have the same latent representation, while         among them). In particular, we define:
         nodes with different structural identities should be far                   fk (u, v) = fk −1 (u, v) + д(s(Rk (u)), s(Rk (v))),
         apart.                                                                                                                               (1)
                                                                                                  k ≥ 0 and |Rk (u)|, |Rk (v)| > 0
       • The latent representation should not depend on any node or
                                                                         where д(D 1 , D 2 ) ≥ 0 measures the distance between the ordered
         edge attribute, including the node labels. Thus, structurally
                                                                         degree sequences D 1 and D 2 and f −1 = 0. Note that by definition
         similar nodes should have close latent representation, inde-
                                                                         fk (u, v) is non-decreasing in k and is defined only when both u or
         pendent of node and edge attributes in their neighborhood.
                                                                         v have nodes at distance k. Moreover, using the ring at distance
         The structural identity of nodes must be independent of its
                                                                         k in the definition of fk (u, v) forces the comparison between the
         “position” in the network.
                                                                         degree sequences of nodes that are at the same distance from u
Given these two properties, we propose struct2vec, a general frame-      and v. Finally, note that if the k-hop neighborhoods of u and v are
work for learning latent representations for nodes composed of           isomorphic, and maps u onto v, then fk −1 (u, v) = 0.
four main steps, as follows:                                                 A final step is determining the function that compares two degree
     (1) Determine the structural similarity between each vertex         sequences. Note that s(Rk (u)) and s(Rk (v)) can be of different sizes
         pair in the graph for different neighborhood sizes. This        and its elements are arbitrary integers in the range [0, n − 1] with
         induces a hierarchy in the measure for structural similar-      possible repetitions. We adopt Dynamic Time Warping (DTW) to
         ity between nodes, providing more information to assess         measure the distance between two ordered degree sequences, a
         structural similarity at each level of the hierarchy.           technique that can cope better with sequences of different sizes and
     (2) Construct a weighted multilayer graph where all nodes           loosely compares sequence patterns [18, 20].
         in the network are present in every layer, and each layer           Informally, DTW finds the optimal alignment between two se-
         corresponds to a level of the hierarchy in measuring struc-     quences A and B. Given a distance function d(a, b) for the elements
         tural similarity. Moreover, edge weights among every node       of the sequence, DTW matches each element a ∈ A to b ∈ B, such
         pair within each layer are inversely proportional to their      that the sum of the distances between matched elements is mini-
         structural similarity.                                          mized. Since elements of sequence A and B are degrees of nodes,
     (3) Use the multilayer graph to generate context for each node.     we adopt the following distance function:
         In particular, a biased random walk on the multilayer graph                                    max(a, b)
         is used to generate node sequences. These sequences are                                d(a, b) =          −1                    (2)
                                                                                                        min(a, b)
         likely to include nodes that are more structurally similar.
     (4) Apply a technique to learn latent representation from a         Note that when a = b then d(a, b) = 0. Thus, two identical ordered
         context given by the sequence of nodes, for example, Skip-      degree sequences will have zero distance. Also note that by taking
         Gram.                                                           the ratio between the maximum and the minimum, the degrees
                                                                         1 and 2 are much more different than degrees 101 and 102, a de-
Note that struct2vec is quite flexible as it does not mandates any
                                                                         sired property when measuring the distance between node degrees.
particular structural similarity measure or representational learning
                                                                         Finally, while we use DTW to assess the similarity between two
framework. In what follows, we explain in detail each step of
                                                                         ordered degree sequences, any other cost function could be adopted
struct2vec and provide a rigorous approach to a hierarchical measure
                                                                         by our framework.
of structural similarity.

3.1    Measuring structural similarity                                   3.2    Constructing the context graph
                                                                         We construct a multilayer weighted graph that encodes the struc-
The first step of struct2vec is to determine a structural similarity
                                                                         tural similarity between nodes. Recall that G = (V , E) denotes the
between two nodes without using any node or edge attributes.
                                                                         original network (possibly not connected) and k ∗ its diameter. Let
Moreover, this similarity metric should be hierarchical and cope
                                                                         M denote the multilayer graph where layer k is defined using the
with increasing neighborhood sizes, capturing more refined notions
                                                                         k-hop neighborhoods of the nodes.
of structural similarity. Intuitively, two nodes that have the same
                                                                            Each layer k = 0, . . . , k ∗ is formed by a weighted undirected
degree are structurally similar, but if their neighbors also have the
                                                                         complete graph with node set V , and thus, n2 edges. The edge
                                                                                                                           
same degree, then they are even more structurally similar.
                                                                         weight between two nodes in a layer is given by:
   Let G = (V , E) denote the undirected, unweighted network under
consideration with vertex set V and edge set E, where n = |V |                           w k (u, v) = e −fk (u,v) , k = 0, . . . , k ∗        (3)
Note that edges are defined only if fk (u, v) is defined and that                With probability 1 −q, the random walk decides to change layers,
weights are inversely proportional to structural distance, and as-            and moves to the corresponding node either in layer k + 1 or layer
sume values smaller than or equal to 1, being equal to 1 only if              k − 1 (layer permitting) with probability proportional to the edge
fk (u, v) = 0. Note that nodes that are structurally similar to u will        weights. In particular:
have larger weights across various layers of M.                                                                         w(uk , uk +1 )
    We connect the layers using directed edges as follows. Each                              pk (uk , uk +1 ) =
                                                                                                                w(uk , uk +1 ) + w(uk , uk −1 )                (8)
vertex is connected to its corresponding vertex in the layer above
                                                                                             pk (uk , uk −1 ) = 1 − pk (uk , uk +1 )
and below (layer permitting). Thus, every vertex u ∈ V in layer k is
connected to the corresponding vertex u in layer k + 1 and k − 1.             Note that every time the walker steps within a layer it includes the
The edge weight between layers are as follows:                                current vertex as part of its context, independent of the layer. Thus,
                                                                              a vertex u may have a given context in layer k (determined by the
         w(uk , uk +1 ) = log(Γk (u) + e), k = 0, . . . , k ∗ − 1
                                                                        (4)   structural similarity of this layer), but have a subset of this context
         w(uk , uk −1 ) = 1, k = 1, . . . , k ∗                               at layer k + 1, as the structural similarity cannot increase as we
                                                                              move to higher layers. This notion of a hierarchical context across
where Γk (u) is number of edges incident to u that have weight
                                                                              the layers is a fundamental aspect of the proposed methodology.
larger than the average edge weight of the complete graph in layer
                                                                                  Finally, for each node u ∈ V , we start a random walk in its
k. In particular:
                                                                              corresponding vertex in layer 0. Random walks have a fixed and
                                                                              relatively short length (number of steps), and the process is repeated
                           Õ
                  Γk (u) =    1(w k (u, v) > w k )             (5)
                                 v ∈V                                         a certain number of times, giving rise to multiple independent walks
                                                                              (i.e., the multiple contexts of node u).
where w k =                      w k (u, v)/ n2 . Thus, Γk (u) measures the
               Í                               
                   (u,v)∈(V2 )
similarity of node u to other nodes in layer k. Note that if u has            3.4     Learning a language model
many similar nodes in the current layer, then it should change
                                                                              Recent language modeling techniques have been extensively used
layers to obtain a more refined context. Note that by moving up
                                                                              to learn word embeddings, and only require sets of sentences in
one layer the number of similar nodes can only decrease. Last, the
                                                                              order to generate meaningful representations. Informally, the task
log function simply reduces the magnitude of the potentially large
                                                                              can be defined as learning word probabilities given a context. In
number of nodes that are similar to u in a given layer. 
                                                                              particular, Skip-Gram [12] has proven to be effective at learning
   Finally, note that M has nk ∗ vertices and at most k ∗ n2 + 2n(k ∗ −
                                                                              meaningful representations for a variety of data. In order to apply it
1) weighted edges. In Section 3.5 we discuss how to reduce the
                                                                              to networks, it suffices to use artificially generated node sequences
complexity of generating and storing M.
                                                                              instead of word sentences. In our framework, these sequences
                                                                              are generated by biased random walks on the multilayer graph M.
3.3    Generating context for nodes                                           Given a node, Skip-Gram aims to maximize the likelihood of its
The multilayer graph M is used to generate structural context for             context in a sequence, where a node’s context is given by a window
each node u ∈ V . Note that M captures the structure of structural            of size w centered on it.
similarities between nodes in G using absolutely no label infor-                 For this work we use Hierarchical Softmax, where conditional
mation. As in previous works, struct2vec uses random walks to                 symbol probabilities are calculated using a tree of binary classi-
generate sequence of nodes to determine the context of a given                fiers. For each node v j ∈ V , Hierarchical Softmax assigns a spe-
node. In particular, we consider a biased random walk that moves              cific path in the classification tree, defined by a set of tree nodes
around M making random choices according to the weights of M.                 n(v j , 1), n(v j , 2), . . . , n(v j , h), where n(v j , h) = v j . In this setting,
Before each step, the random walk first decides if it will change             we have:
layers or walk on the current layer (with probability q > 0 the                                                           Öh
random walk stays in the current layer).                                                                  P(v j |vi ) =      C(n(v j , k), vi )                 (9)
   Given that it will stay in the current layer, the probability of                                                 k =1
stepping from node u to node v in layer k is given by:                        where C is a binary classifier present in every node in the tree. Note
                                                                              that since Hierarchical Softmax operates on a binary tree, we have
                                           e −fk (u,v)                        that h = O(log |V |).
                           pk (u, v) =                                  (6)
                                             Z k (u)                             We train Skip-Gram according to its optimization problem given
where Z k (u) is the normalization factor for vertex u in layer k,            by equation (9). Note that while we use Skip-Gram to learn node
simply given by:                                                              embeddings, any other technique to learn latent representations
                                Õ                                             for text data could be used in our framework.
                      Z k (u) =   e −fk (u,v)                  (7)
                                        v ∈V
                                        v,u                                   3.5     Complexity and optimizations
Note that the random walk will prefer to step onto nodes that are             In order to construct M, the structural distance between every
structurally more similar to the current vertex, avoiding nodes that          node pair for every layer must be computed, namely, fk (u, v) for
have very little structural similarity with it. Thus, the context of a        u, v ∈ V , and 0 ≤ k ≤ k ∗ . However, fk (u, v) uses the result of
node u ∈ V is likely to have structurally similar nodes, independent          a DTW calculation between two degree sequences. While classic
of their labels and position in the original network G.                       implementation of DTW has complexity O(` 2 ), fast techniques have
complexity O(`), where ` is the size of the largest sequence [20].            Reducing the number of layers (OPT3). The number of layers in
Let d max denote the largest degree in the network. Then, the size of      M is given by the diameter of the network, k ∗ . However, for many
the degree sequence |s(Rk (u))| ≤ min(d k
                                         max , n), for any node u and      networks the diameter can be much larger than the average distance.
layer k. Since in each layer there are n2 pairs, the complexity of         Moreover, the importance of assessing the structural similarity
computing all distances for layer k is O(n 2 min(d max
                                                     k , n)). The final    between two nodes diminishes with arbitrarily large values for k.
                          3
complexity is then O(k n ). In what follows we describe a series of
                        ∗                                                  In particular, when k is near k ∗ the length of the degree sequences
optimizations that will significantly reduce the computation and           of the rings become relatively short, and thus fk (u, v) is not much
memory requirements of the framework.                                      different from fk −1 (u, v). Therefore, we cap the number the layers
                                                                           in M to a fixed constant k 0 < k ∗ , capturing the most important
   Reducing the length of degree sequences (OPT1). Although de-            layers for assessing structural similarity. This significantly reduces
gree sequences at layer k have lengths bounded by min(d max    k , n),
                                                                           the computational and memory requirements for constructing M.
for some networks this can be quite large even for small k (e.g.,             Although the combination of the above optimizations affects the
for k = 3 the sequences are already O(n)). To reduce the cost of           capacity of the framework in generating good representations for
comparing large sequences, we propose compressing the ordered              nodes that are structurally similar, we will show that their impact
degree sequence as follows. For each degree in the sequence, we            is marginal and sometimes even beneficial. Thus, the benefits in re-
count the number of occurrences of that degree. The compressed             ducing computational and memory requirements of the framework
ordered degree sequence is a tuple with the degree and the number          greatly outweighs its drawbacks. Last, we make struc2vec available
of occurrences. Since many nodes in a network tend to have the             at: https://github.com/leoribeiro/struc2vec
same degree, in practice the compressed ordered degree sequence
can be an order of magnitude smaller than the original.
   Let A0 and B 0 denote the compressed degree sequences of A and
                                                                           4     EXPERIMENTAL EVALUATION
B, respectively. Since the elements of A0 and B 0 are tuples, we adapt     In what follows we evaluate struct2vec in different scenarios in
the DTW pairwise distance function as follows:                             order to illustrate its potential in capturing the structural identity
                                                                           of nodes, also in light of state-of-the-art techniques for learning
                            max(a 0 , b0 )
                                             
             dist(a, b) =                  − 1 max(a 1 , b1 )     (10)     node representations.
                            min(a 0 , b0 )
where a = (a 0 , a 1 ) and b = (b0 , b1 ) are tuples in A0 and B 0 , re-   4.1    Barbell graph
spectively; a 0 and b0 are the degrees; a 1 and b1 are the number of
                                                                           We denote B(h, k) as the (h, k)-barbell graph which consists of two
occurrences. Note that using the compressed degree sequence leads
                                                                           complete graphs K 1 and K 2 (each having h nodes) connected by a
to comparisons between pieces of the original sequences that have
                                                                           path graph P of length k. Two nodes b1 ∈ V (K 1 ) and b2 ∈ V (K 2 )
the same degree (as opposed to comparing every degree). Thus,
                                                                           act as the bridges. Using {p1 , . . . , pk } to denote V (P), we connect
equation (10) leads to an approximation of the DTW on the orig-
                                                                           b1 to p1 and b2 to pk , thus connecting the three graphs.
inal degree sequences, as given by equation (2). However, DTW
                                                                               The barbell graph has a significant number of nodes with the
now operates on A0 and B 0 , which are much shorter than A and B,
                                                                           same structural identity. Let C 1 = V (K 1 ) \ {b1 } and C 2 = V (K 2 ) \
respectively.
                                                                           {b2 }. Note that all nodes v ∈ {C 1 ∪ C 2 } are structurally equivalent,
   Reducing the number of pairwise similarity calculations (OPT2).         in the strong sense that there exists an automorphism between
While the original framework assesses the similarity between every         any pair of such nodes. Additionally, we also have that all node
node pair at every layer k, clearly this seems unnecessary. Consider       pairs {pi , pk −i }, for 1 ≤ i ≤ k − 1, along with the pair {b1 , b2 },
two nodes with very different degrees (eg., 2 and 20). Their struc-        are structurally equivalent in the same strong sense. Figure 2a
tural distance even for k = 0 will be large, and consequently the          illustrates a B(10, 10) graph, where structurally equivalent nodes
edge between them in M will have a very small weight. Thus, when           have the same color.
generating context for these nodes, the random walk is unlikely to             Thus, we expect struct2vec to learn vertex representations that
traverse this edge. Consequently, not having this edge in M will           capture the structural equivalence mentioned above. Every node
not significantly change the model.                                        pair that is structurally equivalent should have similar latent rep-
   We limit the number of pairwise similarity calculations to Θ(log n)     resentation. Moreover, the learned representations should also
per node, for every level k. Let Ju denote the set of nodes that will      capture structural hierarchies: while the node p1 is not equivalent
be neighbors of u in M, which will be the same for every level. Ju         to neither nodes p2 or b1 , we can clearly see that from a structural
should have the nodes most structurally similar to u. In order to          point of view it is more similar to p2 (it suffices to compare their
determine Ju , we take the nodes that have degrees most similar to         degrees).
u. This can be computed efficiently by performing a binary search              Figure 2 shows the latent representations learned by DeepWalk,
on the ordered degree sequence of all nodes in the network (for the        node2vec and struct2vec for B(10, 10). DeepWalk fails to capture
degree of node u), and taking log n consecutive nodes on each direc-       structural equivalences, which is expected since it was not designed
tion after the search completes. Thus, computing Ju has complexity         to consider structural identities. As illustrated, node2vec does not
Θ(log n). Computing Ju for all nodes has complexity Θ(n log n)             capture structural identities even with different variations of its
which is also needed for sorting the degrees of the network. As for        parameters p and q. In fact, it learns mostly graph distances, placing
memory requirements, each layer of M will now have Θ(n log n)              closer in the latent space nodes that are closer (in hops) in the graph.
edges as opposed to Θ(n 2 ).                                               Another limitation of node2vec is that Skip-Gram’s window size
Figure 2: (a) Barbell graph B(10, 10). (b) Roles identified by RolX. Latent representations in R2 learned by (c) DeepWalk,
(d) node2vec and (e,f,g,h) struc2vec. Parameters used for all methods: number of walks per node: 20, walk length: 80, skip-
gram window size: 5. For node2vec: p = 1 and q = 2.


makes it impossible for nodes in K 1 and K 2 to appear in the same       of structural equivalence when assigning roles to nodes, struct2vec
context.                                                                 better identifies and separates structural equivalence.
   struct2vec, on the other hand, learns representations that prop-
erly separate the equivalent classes, placing structurally equivalent
nodes near one another in the latent space. Note that nodes of the
same color are tightly grouped together. Moreover, p1 and p10 are        4.2   Karate network
placed close to representations for nodes in K 1 and K 2 , as they are   The Zachary’s Karate Club [25] is a network composed of 34 nodes
the bridges. Finally, note that none of the three optimizations have     and 78 edges, where each node represents a club member and
any significant effect on the quality of the representations. In fact,   edges denote if two members have interacted outside the club. In
structurally equivalent nodes are even closer to one another in the      this network, edges are commonly interpreted as indications of
latent representations under OPT1.                                       friendship between members.
   Last, we apply RolX to the barbell graph (results in Figure 2(b)).       We construct a network composed of two copies G 1 and G 2 of
A total of six roles were identified and some roles indeed precisely     the Karate Club network, where each node v ∈ V (G 1 ) has a mirror
captured structural equivalence (roles 1 and 3). However, struc-         node u ∈ V (G 2 ). We also connect the two networks by adding
turally equivalent nodes (in K 1 and K 2 ) were placed in three dif-     an edge between mirrored node pairs 1 and 37. Although this is
ferent roles (role 0, 2, and 5) while role 4 contains all remaining      not necessary for our framework, DeepWalk and node2vec cannot
nodes in the path. Thus, although RolX does capture some notion          place in the same context nodes in different connected components
                                                                         of the graph. Thus, we add the edge for a more fair comparison
                                                                Figure 4: (a) Mirrored Karate network. Identical colors cor-
                                                                respond to mirrored nodes. (b) Roles identified by RolX



                                                                with the two baselines. Figure 4a shows mirrored network with
                                                                corresponding pairs having the same color.
                                                                    Figure 3 shows the representations learned by DeepWalk, node2vec
                                                                and struct2vec. Clearly, Deepwalk and node2vec fail to group in
                                                                the latent space structurally equivalent nodes, including mirrored
                                                                nodes.
                                                                    Once again, struct2vec manages to learn features that properly
                                                                capture the structural identity of nodes. Mirrored pairs – that
                                                                is, nodes with the same color – stay close together in the latent
                                                                space, and there is a complex structural hierarchy in the way the
                                                                representations are grouped together.
                                                                    As an example, note that nodes 1, 34 and their correspondent
                                                                mirrors (37 and 42) are in a separate cluster in the latent space.
                                                                Interestingly, these are exactly the nodes that represent the club
                                                                instructor Mr. Hi and his administrator John A. The network was
                                                                gathered after a conflict between the two split the members of
                                                                the club into two groups – centered on either Mr. Hi or John A.
                                                                Therefore, nodes 1 and 34 have the specific and similar role of
                                                                leaders in the network. Note that struct2vec captures their function
                                                                even though there is no edge between them.
                                                                    Another visible cluster in the latent space is composed of nodes
Figure 3: Node representations for the mirrored Karate net-
                                                                2, 3, 4 and 33, also along with their mirrors. These nodes also have
work created by (a) DeepWalk, (b) node2vec and (c) struc2vec.
                                                                a specific structural identity in the network: all of them have high
Parameters used for all methods: number of walks per node:
5, walk length: 15, Skip-Gram window size: 3. For node2vec:
p = 1 and q = 2.
degrees and are also connected to at least one of the leaders. Lastly,
nodes 26 and 25 (far right in the latent space) have extremely close
representations, which agrees with their structural role: both have
low degree and are 2 hops away from leader 34.
   struct2vec also captures non-trivial structural equivalences. Note
that nodes 7 and 50 (pink and yellow) are mapped to close points
in the latent space. Surprisingly, these two nodes are structurally
equivalent – there exists an automorphism in the graph that maps
one into the other. This can be more easily seen once we note that
nodes 6 and 7 are also structurally equivalent, and 50 is the mirrored
version of node 6 (therefore also structurally equivalent).
   Last, Figure 4b shows the roles identified by RolX in the mirrored
Karate network (28 roles were identified). Note that leaders 1 and
34 were placed in different roles. The mirror for 1 (node 37) was
also placed in a different role, while the mirror for 34 (node 42) was
placed in the same role as 34. A total of 7 corresponding pairs (out
of 34) were placed in the same role. However, some other structural
similarities were also identified – e.g., nodes 6 and 7 are structurally
equivalent and were assigned the same role. Again, RolX seems              Figure 5: Distance distributions between node pairs (mir-
to capture some notion of structural similarities among network            rored pairs and all pairs) in the latent space, for the mirrored
nodes but struct2vec can better identify and separate structural           Karate network learned by node2vec and struc2vec (as shown
equivalences using latent representations.                                 in Figure 3). Curves marked with × correspond to distances
   Consider the distance between the latent representation for             between mirrored pairs while + corresponds to all pairs; cor-
nodes. We measure the distance distribution between pairs cor-             responding averages indicated by vertical lines.
responding to mirrored nodes and among all node pairs (using
the representation shown in Figure 3). Figure 5 shows the two              Table 1: Pearson and Spearman correlation coefficients be-
distance distributions for the representations learned by node2vec         tween structural distance and euclidean distance in latent
and struc2vec. For node2vec the two distributions are practically          space for all node pairs in the mirrored Karate network.
identical, indicating that distances between mirrored pairs blend
well with all pairs. In contrast, struc2vec exhibits two very different           Layer       Pearson (p-value)       Spearman (p-value)
distributions: 94% of mirrored node pairs have distance smaller
than 0.25 while 68% of all node pairs have distance larger than 0.25.               0              0.83 (0.0)               0.74 (0.0)
Moreover, the average distance between all node pairs is 5.6 times                  2              0.71 (0.0)               0.65 (0.0)
larger than that of mirrored pairs, while this ratio is about slightly              4              0.70 (0.0)               0.57 (0.0)
smaller than 1 for node2vec.                                                        6              0.74 (0.0)              0.57 (2.37)
   To better characterize the relationship between structural dis-
tance and distances in the latent representation learned by struc2vec,
we compute the correlation between the two distances for all node          independently. Thus, each edge of G is present in G 1 with probabil-
pairs. In particular, for each layer k we compute the Spearman             ity s. Repeat the process again using G to generate another graph
and Pearson correlation coefficients between fk (u, v), as given by        G 2 . Thus, G 1 and G 2 are structurally correlated through G, and s
equation (1), and the euclidean distance between u and v in the            controls the amount of structural correlation. Note that when s = 1,
learned representation. Results shown in Table 1 for the mirrored          G 1 and G 2 are isomorphic, while when s = 0 all structural identity
Karate network indeed corroborate that there is a very strong cor-         is lost.
relation between the two distances, for every layer, captured by               We apply the edge sampling model to an egonet extracted from
both coefficients. This suggests that struc2vec indeed captures in         Facebook (224 nodes, 3192 edges, max degree 99, min degree 1) [10]
the latent space the measure for structural similarity adopted by          to generate G 1 and G 2 with different values for s. We relabel the
the methodology.                                                           nodes in G 2 (to avoid identical labels) and consider the union of the
                                                                           two graphs as the input network to our framework. Note that this
                                                                           graph has at least two connected components (corresponding to G 1
4.3    Robustness to edge removal                                          and G 2 ) and every node (in G 1 ) has a corresponding pair (in G 2 ).
We illustrate the potential of the framework in effectively repre-             Figure 6 shows the distance (latent space with 2 dimensions) dis-
senting structural identity in the presence of noise. In particular,       tribution between corresponding node pairs and all node pairs for
we randomly remove edges from the network, directly changing               various values for s (corresponding averages are shown in Table 2).
its structure. We adopt the parsimonious edge sampling model to            For s = 1, the two distance distributions are strikingly different,
instantiate two structurally correlated networks [15].                     with the average distance for all pairs being 21 times larger than
   The model works by taking a fixed graph G = (V , E) and gener-          that for corresponding pairs. More interestingly, when s = 0.9 the
ating a graph G 1 by sampling each edge e ∈ E with probability s,          two distributions are still very different. Note that while further
                                                                           Figure 7: Average accuracy for multi-class node classifica-
Figure 6: Distance distribution between node pairs in la-                  tion in air-traffic networks of Brazil, USA and Europe for
tent space representation (2 dimensions) under the edge                    different node features used in supervised learning.
sampling model (different values for s). Bottom curves
(marked with ×) are for corresponding node pairs; top
curves (marked with +) are for all node pairs.                                    • Brazilian air-traffic network: Data collected from the Na-
                                                                                    tional Civil Aviation Agency (ANAC)1 from January to
                                                                                    December 2016. The network has 131 nodes, 1,038 edges
decreasing s does not significantly affect the distance distribution                (diameter is 5). Airport activity is measured by the total
of all pairs, it slowly increases the distribution of corresponding                 number of landings plus takeoffs in the corresponding year.
pairs. However, even when s = 0.3 (which means that the proba-                    • American air-traffic network: Data collected from the Bu-
bility that an original edge appears both in G 1 and G 2 is 0.09, s 2 ),            reau of Transportation Statistics2 from January to October,
the framework still places corresponding nodes closer in the latent                 2016. The network has 1,190 nodes, 13,599 edges (diameter
space.                                                                              is 8). Airport activity is measured by the total number of
   This experiment indicates the robustness of the framework in                     people that passed (arrived plus departed) the airport in
uncovering the structural identity of nodes even in the presence of                 the corresponding period.
structural noise, modeled here through edge removals.                             • European air-traffic network: Data collected from the Statis-
                                                                                    tical Office of the European Union (Eurostat)3 from January
Table 2: Average and standard deviation for distances be-                           to November 2016. The network has 399 nodes, 5,995 edges
tween node pairs in the latent space representation (see cor-                       (diameter is 5). Airport activity is measured by the to-
responding distributions in Figure 6).                                              tal number of landings plus takeoffs in the corresponding
                                                                                    period.
                                                                           For each airport, we assign one of four possible labels corresponding
       s    Corresponding - avg (std)       All nodes - avg (std)          to their activity. In particular, for each dataset, we use the quartiles
      1.0          0.083 (0.05)                 1.780 (1.354)              obtained from the empirical activity distribution to split the dataset
      0.9         0.117 (0.142)                 1.769 (1.395)              in four groups, assigning a different label for each group. Thus, label
      0.3         0.674 (0.662)                 1.962 (1.445)              1 is given to the 25% less active airports, and so on. Note that all
                                                                           classes (labels) have the same size (number of airports). Moreover,
                                                                           classes are related more to the role played by the airport.
                                                                              We learn latent representations for nodes of each air-traffic net-
4.4     Classification                                                     work using struc2vec and node2vec using a grid search to select the
A common application of latent representations for network nodes           best hyperparameters for each case. Note that this step does not use
is classification. struc2vec can be leveraged for this task when labels    any node label information. The latent representation for each node
for nodes are more related to their structural identity than to the        becomes the feature that is then used to train a supervised classifier
labels of their neighbors. To illustrate this potential, we consider       (one-vs-rest logistic regression with L2 regularization). We also
air-traffic networks: unweighted, undirected networks where nodes          consider just the node degree as a feature since it captures a very
correspond to airports and edges indicate the existence of commer-         basic notion of structural identity. Last, since classes have identical
cial flights. Airports will be assigned a label corresponding to their     1 http://www.anac.gov.br/
level of activity, measured in flights or people (discussed below).        2 https://transtats.bts.gov/

We consider the following datasets (collected for this study):             3 http://ec.europa.eu/
sizes, we use just the accuracy to assess performance. Experiments          network. struc2vec assesses the structural similarity of node pairs
are repeated 10 times using random samples to train the classifier          by considering a hierarchical metric defined by the ordered degree
(80% of the nodes used for training) and we report on the average           sequence of nodes and uses a weighted multilayer graph to generate
performance.                                                                context.
   Figure 7 shows the classification performance of the different              We have shown that struc2vec excels in capturing the structural
features for all air-traffic networks. Clearly, struc2vec outperforms       identity of nodes, in comparison to state-of-the-art techniques such
the other approaches, and its optimizations have little influence. For      as DeepWalk, node2vec and RolX. It overcomes their limitation by
the Brazilian network, struc2vec improves classification accuracy           focusing explicitly on structural identity. Not surprising, we also
by 50% in comparison to node2vec. Interestingly, for this network           show that struc2vec is superior in classification tasks where node
node2vec has average performance (slightly) inferior to node degree,        labels are more dependent on their role or structural identity. Last,
indicating the importance played by the structural identity of the          different models to generate representations tend to capture dif-
nodes in classification.                                                    ferent properties, and we argue that structural identity is clearly
                                                                            important when considering possible node representations.
4.5    Scalability
In order to illustrate its scalability, we apply struc2vec with the first   REFERENCES
two optimizations to instances of the Erdös-Rényi random graph             [1] Nir Atias and Roded Sharan. 2012. Comparative analysis of protein networks:
                                                                                 hard problems, practical solutions. Commun. ACM 55 (2012).
model (using 128 dimensions, 10 walks per node, walk length 80,              [2] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. 2003. A
Skip-Gram window 10). We compute the average execution time                      Neural Probabilistic Language Model. JMLR (2003).
                                                                             [3] V Blondel, A Gajardo, M Heymans, P Senellart, and P Van Dooren. 2004. A mea-
for 10 independent runs on graphs with sizes from 100 to 1,000,000               sure of similarity between graph vertices: Applications to synonym extraction
nodes and average degree of 10. In order to speed up training the                and web searching. SIAM review (2004).
language model, we use Skip-Gram with Negative Sampling [13].                [4] Shaosheng Cao, Wei Lu, and Qiongkai Xu. 2016. Deep Neural Networks for
                                                                                 Learning Graph Representations. In AAAI.
Figure 8 shows the execution time (in log-log scale) indicating              [5] F Fouss, A Pirotte, J Renders, and M Saerens. 2007. Random-Walk Computation
that struc2vec scales super-linearly but closer to linear than to n 1.5          of Similarities Between Nodes of a Graph with Application to Collaborative
(dashed lines). Thus, despite its unfavorable worst case time and                Recommendation. IEEE Trans. on Knowl. and Data Eng. (2007).
                                                                             [6] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable Feature Learning for
space complexity, in practice struc2vec can be applied to very large             Networks. In ACM SIGKDD.
networks.                                                                    [7] K Henderson, B Gallagher, T Eliassi-Rad, H Tong, S Basu, L Akoglu, D Koutra,
                                                                                 C Faloutsos, and L Li. 2012. Rolx: structural role extraction & mining in large
                                                                                 graphs. In ACM SIGKDD.
                                                                             [8] Jon M Kleinberg. 1999. Authoritative sources in a hyperlinked environment.
                                                                                 Journal of the ACM (JACM) (1999).
                                                                             [9] Elizabeth A Leicht, Petter Holme, and Mark EJ Newman. 2006. Vertex similarity
                                                                                 in networks. Physical Review E 73 (2006).
                                                                            [10] Jure Leskovec and Julian J Mcauley. 2012. Learning to discover social circles in
                                                                                 ego networks. In NIPS.
                                                                            [11] Francois Lorrain and Harrison C White. 1971. Structural equivalence of individ-
                                                                                 uals in social networks. The Journal of mathematical sociology 1 (1971).
                                                                            [12] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient
                                                                                 Estimation of Word Representations in Vector Space. In ICLR Workshop.
                                                                            [13] T Mikolov, I Sutskever, K Chen, G Corrado, and J Dean. 2013. Distributed
                                                                                 Representations of Words and Phrases and their Compositionality. In NIPS.
                                                                            [14] A Narayanan, M Chandramohan, L Chen, Y Liu, and S Saminathan. 2016. sub-
                                                                                 graph2vec: Learning Distributed Representations of Rooted Sub-graphs from
                                                                                 Large Graphs. In Workshop on Mining and Learning with Graphs.
                                                                            [15] Pedram Pedarsani and Matthias Grossglauser. 2011. On the privacy of
                                                                                 anonymized networks. In ACM SIGKDD.
                                                                            [16] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. DeepWalk: Online
                                                                                 Learning of Social Representations. In ACM SIGKDD.
                                                                            [17] Narciso Pizarro. 2007. Structural Identity and Equivalence of Individuals in Social
                                                                                 Networks Beyond Duality. International Sociology 22 (2007).
                                                                            [18] T Rakthanmanon, B Campana, A Mueen, G Batista, B Westover, Q Zhu, J Zakaria,
                                                                                 and E Keogh. 2013. Addressing big data time series: Mining trillions of time
                                                                                 series subsequences under dynamic time warping. ACM TKDD (2013).
Figure 8: Average execution time of struc2vec on Erdös-                    [19] Lee Douglas Sailer. 1978. Structural equivalence: Meaning and definition, com-
Rényi graphs with average degree of 10. Training time refers                    putation and application. Social Networks (1978).
                                                                            [20] S Salvador and P Chan. 2004. FastDTW: Toward accurate dynamic time warping
to the additional time required by Skip-Gram.                                    in linear time and space. In Workshop on Min. Temp. and Seq. Data, ACM SIGKDD.
                                                                            [21] N Shervashidze, P Schweitzer, E van Leeuwen, K Mehlhorn, and K Borgwardt.
                                                                                 2011. Weisfeiler-Lehman Graph Kernels. JMLR (2011).
                                                                            [22] R Singh, J Xu, and B Berger. 2008. Global alignment of multiple protein interaction
5     CONCLUSION                                                                 networks with application to functional orthology detection. PNAS (2008).
                                                                            [23] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei.
Structural identity is a concept of symmetry in networks in which                2015. LINE: Large-scale Information Network Embedding. In WWW.
nodes are identified based on the network structure. The concept            [24] Daixin Wang, Peng Cui, and Wenwu Zhu. 2016. Structural Deep Network
                                                                                 Embedding. In ACM SIGKDD.
is strongly related to functions or roles played by nodes in the            [25] Wayne W Zachary. 1977. An information flow model for conflict and fission in
network, an important problem in social and hard sciences.                       small groups. Journal of anthropological research (1977).
   We propose struc2vec, a novel and flexible framework to learn            [26] Laura A Zager and George C Verghese. 2008. Graph similarity scoring and
                                                                                 matching. Applied mathematics letters (2008).
representations that capture the structural identity of nodes in a

