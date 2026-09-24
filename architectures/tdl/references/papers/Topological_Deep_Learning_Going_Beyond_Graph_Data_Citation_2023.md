# Topological Deep Learning Going Beyond Graph Data Citation 2023

> Source: `Topological_Deep_Learning_Going_Beyond_Graph_Data_Citation_2023.pdf`

---

See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/370134352



Topological Deep Learning: Going Beyond Graph Data

Article · April 2023



CITATION                                                                                                  READS
1                                                                                                         3,747


14 authors, including:

           Mustafa Hajij                                                                                             Ghada Zamzmi
           University of San Francisco                                                                               National Institutes of Health
           79 PUBLICATIONS 460 CITATIONS                                                                             73 PUBLICATIONS 492 CITATIONS

              SEE PROFILE                                                                                                 SEE PROFILE



           Theodore Papamarkou                                                                                       Nina Miolane
           The University of Manchester                                                                              Stanford University
           61 PUBLICATIONS 3,991 CITATIONS                                                                           47 PUBLICATIONS 167 CITATIONS

              SEE PROFILE                                                                                                 SEE PROFILE




Some of the authors of this publication are also working on these related projects:


                Sinogram-based Interpolation Strategies for Incomplete Trajectories in X-ray Computed Tomography View project



                Topological Deep Learning View project




 All content following this page was uploaded by Mustafa Hajij on 21 April 2023.

 The user has requested enhancement of the downloaded file.
Contents                                                                                     Topological deep learning




Topological Deep Learning: Going Beyond Graph Data

Mustafa Hajij                                                                                        mhajij@usfca.edu
                    ∗
Ghada Zamzmi                                                                                      ghadh@mail.usf.edu
Theodore Papamarkou∗                                                              theo.papamarkou@manchester.ac.uk
Nina Miolane                                                                                   ninamiolane@ucsb.edu
Aldo Guzmán-Sáenz                                                                        aldo.guzman.saenz@ibm.com
Karthikeyan Natesan Ramamurthy                                                                   knatesa@us.ibm.com
Tolga Birdal                                                                                    tbirdal@imperial.ac.uk
Tamal K. Dey                                                                                    tamaldey@purdue.edu
Soham Mukherjee                                                                                 mukher26@purdue.edu
Shreyas N. Samaga                                                                                ssamaga@purdue.edu
Neal Livesay                                                                               n.livesay@northeastern.edu
Robin Walters                                                                              r.walters@northeastern.edu
Paul Rosen                                                                                        prosen@sci.utah.edu
Michael T. Schaub                                                                           schaub@cs.rwth-aachen.de




Contents

1 Introduction                                                                                                       4

2 Motivation: what does topology offer to machine learning?                                                          8
   2.1     Modeling and learning from data on topological spaces . . . . . . . . . . . . . . . . . . . . . .          8
   2.2 The utility of topology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       10
   2.3 A unifying perspective on deep learning and structured computations . . . . . . . . . . . . . .               12

3 Preliminaries: from graphs to higher-order networks                                                                12
   3.1     Neighborhood functions and topological spaces . . . . . . . . . . . . . . . . . . . . . . . . . .         13
   3.2     Bridging the gap among higher-order networks . . . . . . . . . . . . . . . . . . . . . . . . . .          14
   3.3     Hierarchical structure and set-type relations . . . . . . . . . . . . . . . . . . . . . . . . . . . .     16

4 Combinatorial complexes (CCs)                                                                                      16
   4.1     CC definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   17
   4.2     CC-homomorphisms and sub-CCs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          18
   4.3     Motivation for CCs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    19
           4.3.1   Pooling operations on CCs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       19
  * These authors contributed equally to this work




                                                           1
Topological deep learning                                                                                    Contents




         4.3.2   Structural advantages of CCs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       20
   4.4   Neighborhood functions on CCs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        21
         4.4.1   Incidence in a CC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      22
         4.4.2   Adjacency in a CC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      22
   4.5   Data on CCs: cochain spaces and maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .           24

5 Combinatorial complex neural networks (CCNNs)                                                                     25
   5.1   Building CCNNs: tensor diagrams . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          26
   5.2   Push-forward operator and merge node         . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   28
   5.3 Three tensor operations for the construction of CCNNs . . . . . . . . . . . . . . . . . . . . . .            29
   5.4   Combinatorial complex convolutional networks (CCCNNs) . . . . . . . . . . . . . . . . . . . .              31
   5.5   Combinatorial complex attention neural networks (CCANNs) . . . . . . . . . . . . . . . . . .               31

6 Higher-order message passing                                                                                      34
   6.1   Definition of higher-order message passing . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       34
   6.2   Higher-order message-passing neural networks are CCNNs . . . . . . . . . . . . . . . . . . . .             35
   6.3   Merge nodes and higher-order message passing: a qualitative comparison . . . . . . . . . . . .             36
   6.4 Attention higher-order message passing and CCANNs . . . . . . . . . . . . . . . . . . . . . .                38

7 Push-forward and higher-order (un)pooling                                                                         38
   7.1   CC-(un)pooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     39
   7.2   Formulating common pooling operations as CC-pooling . . . . . . . . . . . . . . . . . . . . .              39
         7.2.1   Graph pooling as CC-pooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        40
         7.2.2   Image pooling as CC-pooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        40
   7.3 (Un)pooling CCNNs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          41
   7.4   Mapper and the CC-pooling operation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          42

8 Hasse graph interpretation and equivariances of CCNNs                                                             42
   8.1   Hasse graph interpretation of CCNNs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          43
         8.1.1   CCs as Hasse graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      43
         8.1.2   Augmented Hasse graphs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         45
         8.1.3   Reducibility of CCNNs to graph-based models . . . . . . . . . . . . . . . . . . . . . .            45
         8.1.4   Augmented Hasse graphs and CC-pooling . . . . . . . . . . . . . . . . . . . . . . . . .            46
         8.1.5   Augmented Hasse diagrams, message passing, and merge nodes . . . . . . . . . . . . .               46
         8.1.6   Higher-order representation learning . . . . . . . . . . . . . . . . . . . . . . . . . . . .       47
   8.2   On the equivariance of CCNNs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         48
         8.2.1   Permutation equivariance of CCNNs . . . . . . . . . . . . . . . . . . . . . . . . . . . .          48
         8.2.2   Orientation equivariance of CCNNs        . . . . . . . . . . . . . . . . . . . . . . . . . . . .   50


                                                         2
Contents                                                                                      Topological deep learning




9 Implementation and numerical experiments                                                                             51
   9.1     Software: TopoNetX, TopoEmbedX, and TopoModelX . . . . . . . . . . . . . . . . . . . . . .                  51
   9.2     Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    52
   9.3     Shape analysis: mesh segmentation and classification . . . . . . . . . . . . . . . . . . . . . . .          52
           9.3.1   Mesh segmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       52
           9.3.2   Mesh and point cloud classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       54
           9.3.3   Graph classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      54
   9.4     Pooling with mapper on graphs and data classification . . . . . . . . . . . . . . . . . . . . . .           55
           9.4.1   Mesh classification: CC-pooling with input vertex and edge features . . . . . . . . . .             56
           9.4.2   Mesh classification: CC-pooling with input vertex features only . . . . . . . . . . . . .           56
           9.4.3   Point cloud classification: CC-pooling with input vertex features only . . . . . . . . .            56
   9.5 Ablation studies        . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   57

10 Related work                                                                                                        57
   10.1 Graph-based models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .           57
   10.2 Higher-order deep learning models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .            58
   10.3 Attention-based models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .           58
   10.4 Graph-based pooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          59
   10.5 Applied algebraic topology . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         59

11 Conclusions                                                                                                         59

References                                                                                                             61

A Glossary                                                                                                             73

B Lifting maps                                                                                                         74
   B.1 n-hop CC of a graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .           74
   B.2 Path-based and subgraph-based CC of a graph . . . . . . . . . . . . . . . . . . . . . . . . . .                 74
   B.3 Loop-based CC of a graph . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .            74
   B.4 Coface CC of a simplicial complex/CC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .              75
   B.5 Augmentation of CCs by higher-rank cells . . . . . . . . . . . . . . . . . . . . . . . . . . . . .              75

C CCNN architecture search and topological quantum field theories                                                      76

D Learning discrete exterior calculus operators with CCANNs                                                            77

E A mapper-induced topology-preserving CC-pooling operation                                                            78




                                                            3
Topological deep learning                                                                          1   Introduction




                                                    Abstract

        Topological deep learning is a rapidly growing field that pertains to the development of deep
        learning models for data supported on topological domains such as simplicial complexes,
        cell complexes, and hypergraphs, which generalize many domains encountered in scientific
        computations. In this paper, we present a unifying deep learning framework built upon a
        richer data structure that includes widely adopted topological domains.
        Specifically, we first introduce combinatorial complexes, a novel type of topological domain.
        Combinatorial complexes can be seen as generalizations of graphs that maintain certain
        desirable properties. Similar to hypergraphs, combinatorial complexes impose no constraints
        on the set of relations. In addition, combinatorial complexes permit the construction of
        hierarchical higher-order relations, analogous to those found in simplicial and cell complexes.
        Thus, combinatorial complexes generalize and combine useful traits of both hypergraphs
        and cell complexes, which have emerged as two promising abstractions that facilitate the
        generalization of graph neural networks to topological spaces.
        Second, building upon combinatorial complexes and their rich combinatorial and algebraic
        structure, we develop a general class of message-passing combinatorial complex neural networks
        (CCNNs), focusing primarily on attention-based CCNNs. We characterize permutation and
        orientation equivariances of CCNNs, and discuss pooling and unpooling operations within
        CCNNs in detail.
        Third, we evaluate the performance of CCNNs on tasks related to mesh shape analysis and
        graph learning. Our experiments demonstrate that CCNNs have competitive performance
        as compared to state-of-the-art deep learning models specifically tailored to the same tasks.
        Our findings demonstrate the advantages of incorporating higher-order relations into deep
        learning models in different applications.



1   Introduction

In recent years, there has been an exponential growth in the amount of data available for computational
analysis, including scientific data as well as common data types such as text, images, and audio. This
abundance of data has empowered various fields, including physics, chemistry, computational social sciences,
and biology, to make significant progress using machine learning techniques, primarily deep neural networks.
As deep neural networks can effectively summarize and extract patterns from large data sets, they are suitable
for many complex tasks. Initially, deep neural networks have been developed to learn from data supported
on regular (Euclidean) domains, such as grids in images, sequences of text and time-series. These models,
including convolutional neural networks (CNNs) [156, 162, 243], recurrent neural networks (RNNs) [249,
13] and transformers [256], have proven highly effective in processing such Euclidean data [117], resulting
in unprecedented performance in various applications, most recently in chat-bots (e.g., ChatGPT [2]) and
text-controlled image synthesis [223].
However, scientific data in various fields are often structured differently and are not supported on regular
Euclidean domains. As a result, adapting deep neural networks to process this type of data has been a
challenge. Against this backdrop, geometric deep learning (GDL) [50, 284, 268] has emerged as an extension of
deep learning models to non-Euclidean domains. To achieve this, GDL restricts the performed computations
via principles of geometric regularity, such as symmetries, invariances, and equivariances. The GDL perspective
allows for appropriate inductive biases to be imposed when working with arbitrary data domains, including
sets [215, 217, 81, 283, 138], grids [45, 187, 46, 154, 242, 267, 196], manifolds [45, 187, 46, 154, 242, 267, 196],
and graphs [232, 101, 284, 268, 46, 196, 49, 150]. Graphs, in particular, have garnered interest due to their
applicability in numerous scientific studies and their ability to generalize conventional grids. Accordingly, the
development of graph neural networks (GNNs) [49, 150] has remarkably enhanced our ability to model and
analyze several types of data in which graphs naturally appear.


                                                         4
1   Introduction                                                                               Topological deep learning




                   Geometric deep learning                          Higher-order domains
                          domains




                   Set of entities     Graph       Simplicial complex         Cell complex      Hypergraph
            (a)
                        Rank-0 cells

                        Rank-1 cells


                        Rank-2 cells
                                                     Combinatorial complex




            (b)




                        Feature vector supported on 0-cells          Push-forward operations
                        Feature vector supported on 1-cells       Unpooling     Pooling Message passing
                        Feature vector supported on 2-cells


Figure 1: A graphical abstract that visualizes our main contributions. (a): Different mathematical structures
can be used to represent relations between abstract entities. Sets have entities with no connections, graphs
encode binary relations between vertices, simplicial and cell complexes model hierarchical higher-order
relations, and hypergraphs accommodate arbitrary set-type relations with no hierarchy. We introduce
combinatorial complexes (CCs), which generalize graphs, simplicial and cell complexes, and hypergraphs.
CCs are equipped with set-type relations as well as with a hierarchy of these relations. (b): By utilizing
the hierarchical and topological structure of CCs, we introduce the push-forward operation, a fundamental
building block for higher-order message-passing protocols and for (un)pooling operations on CCs. Our
push-forward operations on CCs enable us to construct combinatorial complex neural networks (CCNNs),
which provide a general conceptual framework for topological deep learning on higher-order domains.


Despite the success of GDL and GNNs, seeing graphs through a purely geometric viewpoint yields a solely
local abstraction and falls short of capturing non-local properties and dependencies in data. Topological
data, including interactions of edges (in graphs), triangles (in meshes) or cliques, arise naturally in an array
of novel applications in complex physical systems [30, 161], traffic forecasting [144], social influence [285],
protein interaction [200], molecular design [237], visual enhancement [95], recommendation systems [160],
and epidemiology [82]. To natively and effectively model such data, we are bound to go beyond graphs and


                                                              5
Topological deep learning                                                                                       1   Introduction




consider qualitative spatial properties remaining unchanged under some geometric transformations. In other
words, we need to consider the topology of data [58] to formulate neural network architectures capable of
extracting semantic meaning from complex data.
One approach to extract more global information from data is to go beyond graph-based abstractions and
consider extensions of graphs, such as simplicial complexes, cell complexes, and hypergraphs, generalizing
most data domains encountered in scientific computations [41, 29, 32, 253]. The development of machine
learning models to learn from data supported on these topological domains [97, 53, 222, 234, 42, 121, 123, 91,
235, 221, 112, 272] is a rapidly growing new frontier, to which we refer hereafter as topological deep learning
(TDL). TDL intertwines several research areas, including topological data analysis (TDA) [93, 58, 86, 178,
108], topological signal processing [233, 273, 236, 222, 21, 219, 229], network science [245, 161, 20, 29, 41, 39,
33, 80, 19, 203], and geometric deep learning [278, 56, 99, 177, 27, 197, 26].
Despite the growing interest in TDL, a broader synthesis of the foundational principles of these ideas has
not been established so far. We argue that this is a deficiency that inhibits progress in TDL, as it makes
it challenging to draw connections between different concepts, impedes comparisons, and makes it difficult
for researchers of other fields to find an entry point to TDL. Hence, in this paper, we aim to provide a
foundational overview over the principles of TDL, serving not only as a unifying framework for the many
exciting ideas that have emerged in the literature over recent years, but also as a conceptual starting point to
promote the exploration of new ideas. Ultimately, we hope that this work will contribute to the accelerated
growth in TDL, which we believe would be a key enabler of transferring deep learning successes to an enlarged
range of application scenarios.
By drawing inspiration from traditional topological notions in algebraic topology [108, 133] and recent
advancements in higher-order networks [29, 253, 41, 30], we first introduce combinatorial complexes (CCs) as
the major building blocks of our TDL framework. CCs constitute a novel topological domain that unifies
graphs, simplicial and cell complexes, and hypergraphs as special cases, as illustrated in Figure 11 . Similar to
hypergraphs, CCs can encode arbitrary set-like relations between collections of abstract entities. Moreover,
CCs permit the construction of hierarchical higher-order relations, analogous to those found in simplicial and
cell complexes. Hence, CCs generalize and combine the desirable traits of hypergraphs and cell complexes.
In addition, we introduce the necessary operators to construct deep neural networks for learning features and
abstract summaries of input anchored on combinatorial complexes. Such operators provide convolutions, atten-
tion mechanisms, message-passing schemes as well as the means for incorporating invariances, equivariances,
or other geometric regularities. Specifically, our novel push-forward operation allows pushing data across
different dimensions, thus forming an elementary building block upon which higher-order message-passing
protocols and (un)pooling operations are defined on CCs. The resulting learning machines, which we call
combinatorial complex neural networks (CCNNs), are capable of learning abstract higher-order data structures,
as clearly demonstrated in our experimental evaluation.
We envisage our contributions to be a platform encouraging researchers and practitioners to extend our
CCNNs, and invite the community to build upon our work to expand TDL on higher-order domains. Our
contributions, which are also visually summarized in Figure 1, are the following:
     • First, we introduce CCs as domains for TDL. We characterize CCs and their properties, and
       explain how they generalize major existing domains, such as graphs, hypergraphs, simplicial and cell
       complexes. CCs can thus serve as a unifying starting point that enables the learning of expressive
       representations of topological data.
     • Second, using CCs as domains, we construct CCNNs, an abstract class of higher-order message-
       passing neural networks that provide a unifying blueprint for TDL models based on hypergraphs and
       cell complexes.
           – Based upon a push-forward operator defined on CCs, we introduce convolution, attention,
             pooling and unpooling operators for CCNNs.
           – We formalize and investigate permutation and orientation equivariance of CCNNs, paving the
             way to future work on geometrization of CCNNs.
  1 All figures in this paper should be displayed in color, as different colors communicate different pieces of information.




                                                               6
1   Introduction                                                                        Topological deep learning




           – We show how CCNNs can be intuitively constructed via graphical notation.
      • Third, we evaluate our ideas in practical scenarios.
           – We release the source code of our framework as three supporting Python libraries: TopoNetX,
             TopoEmbedX and TopoModelX.
           – We show that CCNNs attain competing predictive performance against state-of-the-art task-
             specific neural networks in various applications, including shape analysis and graph learning.
           – We establish connections between our work and classical constructions in TDA, such as the
             mapper [244]. Particularly, we realize the mapper construction in terms of our TDL framework
             and demonstrate how it can be utilized in higher-order (un)pooling on CCs.
           – We demonstrate the reducibility of any CC to a special graph called the Hasse graph. This enables
             the characterization of certain aspects of CCNNs in terms of graph-based models, allowing us to
             reduce higher-order representation learning to graph representation learning (using an enlarged
             computational graph).

Glossary. Prior to delving into more details, we present the elementary terminologies of already established
concepts used throughout the paper. Some of these terms are revisited formally in Section 3. Appendix A
provides notation and terminology glossaries for novel ideas put forward by the paper.

    Cell complex: a topological space obtained as a disjoint union of topological disks (cells), with each
    of these cells being homeomorphic to the interior of a Euclidean ball. These cells are attached together
    via attaching maps in a locally reasonable manner.
    Domain: the underlying space where data is typically supported.
    Entity or vertex: an abstract point; it can be thought as an element of a set.
    Graph or network: a set of entities (vertices) together with a set of edges representing binary
    relations between the vertices.
    Hierarchical structure on a topological domain: an integer-valued function that assigns a positive
    integer (rank) to every relation in the domain such that higher-order relations are assigned higher rank
    values. For instance, a simplicial complex admits a hierarchical structure induced by the cardinality
    of its simplices.
    Higher-order network/topological domain: generalization of a graph that captures (binary and)
    higher-order relations between entities. Simplicial/cell complexes and hypergraphs are examples of
    higher-order networks.
    Hypergraph: a set of entities (vertices) and a set of hyperedges representing binary or higher-order
    relations between the vertices.
    Message passing: a computational framework that involves passing data, ‘messages’, among neighbor
    entities in a domain to update the representation of each entity based on the messages received from
    its neighbors.
    Relation or cell: a subset of a set of entities (vertices). A relation is called binary if its cardinality
    is equal to two. A relation is called higher-order if its cardinality is greater than two.
    Set-type relation: a higher-order network is said to have set-type relations if the existence of a
    relation is not implied by another relation in the network; e.g., hypergraphs admit set-type relations.
    Simplex: a generalization of a triangle or tetrahedron to arbitrary dimensions; e.g., a simplex of
    dimension zero, one, two, or three is a point, line segment, triangle, or tetrahedron, respectively.
    Simplicial complex: a collection of simplices such that every face of each simplex in the collection is
    also in the collection, and the intersection of any two simplices in the collection is either empty or a
    face of both simplices.
    Topological data: feature vectors supported on relations in a topological domain.
    Topological deep learning: the study of topological domains using deep learning techniques, and
    the use of topological domains to represent data in deep learning models.



                                                        7
Topological deep learning                          2   Motivation: what does topology offer to machine learning?




                   (a)                                 (b)                              (c)

Figure 2: Data might be supported naturally on higher-order relations. (a): An edge-based vector field. (b):
A face-based vector field. Both vector fields in (a) and (b) are defined on a cell complex torus. An interactive
visualization of (a–b) is provided here. (c): Class-labeled topological data might naturally be supported on
higher-order relations. For instance, mesh segmentation labels for 2-faces are depicted by different colors
(blue, green, turquoise, pink, brown) to represent different parts (head, neck, body, legs, tail) of the horse.

2     Motivation: what does topology offer to machine learning?

Combinatorial complex neural networks (CCNNs), as presented in this work, generalize graph neural networks
(GNNs) and their higher-order analogues via topological constructions. In this section, we provide the
motivation for the use of topological constructions in machine learning, and specifically in deep learning from
three angles. First, data modeling: how topological abstraction can help us to reason and compute with
various data types supported on topological spaces. Second, utility: how accounting for topology improves
performance. Third, unification: how topology can be used to synthesize and abstract disparate concepts.

2.1   Modeling and learning from data on topological spaces

In the context of machine learning, the domain on which the data is supported is typically a (linear) vector
space. This underlying vector space facilitates many computations and is often implicitly assumed. However,
as has been recognized within geometric deep learning, it is often essential to consider data supported on
different domains. Indeed, explicitly considering and modeling the domain on which the data is supported
can be crucial for various reasons.
First, different domains can have distinct characteristics and properties that require different types of deep
learning models to effectively process and analyze the data. For example, a graph domain may require a
graph convolutional network [150] to account for the non-Euclidean structure, while a point cloud domain
may require a PointNet-like architecture [215, 276] to handle unordered sets of points.
Second, the specific data within each domain can vary widely in terms of size, complexity and noise. By
understanding specific data properties within a given domain, researchers can develop models that are tailored
to those particular properties. For example, a model designed for point clouds with high levels of noise may
incorporate techniques such as outlier removal or local feature aggregation.
Third, accurately distinguishing between domain and data is important for developing models that generalize
well to new, unseen data. By identifying the underlying structure of the domain, researchers can develop
models that are robust to variations in the data while still being able to capture relevant geometric features.
To summarize, by considering both the domain and data together, researchers can develop models that are
better suited to handle the specific challenges and complexities of different geometric learning tasks.

Different perspectives on data: domains and relations. When referring to topological data or
topological data analysis in the literature, there are different views on what the actual data consists of and
what it is supposed to model. Here, we follow [41] and distinguish between different types of data and goals
of our learning procedures, even though the distinction between these different types of data can sometimes
be more blurry than what is presented here.


                                                       8
2   Motivation: what does topology offer to machine learning?                           Topological deep learning




                          (a)                                        (b)

Figure 3: Examples of processing data supported on graphs or on higher-order networks. (a): Graphs can be
used to model particle interactions in fluid dynamics. Vertices represent particles, whereas particle-to-particle
interactions are modeled via message passing among vertices [226, 241]. (b): When modeling springs and
self-collision, it is natural to work with edges rather than vertices. This is because the behavior of the cloth
is determined by the tension and compression forces acting along the edges, and not only by the position of
individual particles. To model the interactions among multiple edges, polygonal faces can be used to represent
the local geometry of the cloth. The polygonal faces provide a way to compute higher-order message passing
among the edges.

Relational data. Relational data describes the relations between different entities or objects. This fundamental
concept can manifest in various ways, such as connections between users in social networks, where friend
relationships or follower-ship serve as examples of such relations. Traditionally, these relations are understood
through graphs, with vertices and edges representing entities and their pairwise connections, respectively.
Stated differently, the edges within the graph are considered to be units of measurement; i.e., every piece of
relational data involves two vertices.
However, many real-world phenomena naturally involve complex, multi-party interactions, where more than
two entities interact with each other in intricate ways. For instance, groups of individuals within a society,
sets of co-authors, and genes or proteins that interact with each other are all examples of such higher-order
relations. Such dependencies, which go beyond pairwise interactions, can be modeled by higher-order relations
and aptly abstracted as hypergraphs, cell complexes, or CCs. Clearly, topological ideas can play an important
role for understanding this data, and crucially enable us to move from graphs modeling pairwise relations to
richer representations.
Data as an instance of a domain. A related conceptualization of what our observed data is supposed to
represent is adopted in TDA. Namely, all of observed data is supposed to correspond to a noisy instance of a
topological object itself; i.e., we aim to learn the ‘topological shape’ of the data. Note that in this context,
we typically do not directly observe relational data, but instead we construct relations from observed data,
which we then use to classify the observed dataset. Persistent homology performed on point cloud data is a
mainstream example for this viewpoint.
Data supported on a topological domain. Topological ideas also play an important role when considering data
that is defined on top of a domain such as a graph. This could be particular dynamics, such as epidemic
spreading, or it could be any type of other data or signals supported on the vertices (cases of edge-signals
on a graph are not considered often, though exceptions exist [233, 235]). In the language of graph signal
processing, a variety of functions or signals can be defined on a graph domain, and these are said to be
supported by the domain. Crucially, the graph itself can be arbitrarily complex but is typically considered to
be fixed; the observed data is not relational, but supported on the vertices of a graph.
Moving beyond graphs, more general topological constructions, such as CCs, can support data not just on
vertices and edges, but also on other ‘higher-order’ entities, as demonstrated in Figure 2(a–b). For example,
vector fields defined on meshes in computer graphics are often data supported on edges and faces, and can be
conveniently modeled as data supported on a higher-order domain [114]. Similarly, class-labeled data can be
provided on the edges and faces of a given mesh; see Figure 2(c) for an example. To process such data it is
again relevant to take into account the structure of the underlying topological domain.


                                                        9
Topological deep learning                            2   Motivation: what does topology offer to machine learning?




            (a)                        (b)                (c)                      (d)                   (e)

Figure 4: An illustration of representing hierarchical data via higher-order networks. Black arrows indicate
graph augmentation by higher-order relations, whereas orange arrows indicate coarsened graph extraction.
(a): A graph encodes binary relations (pink edges) among abstract entities (yellow vertices). (b): Higher-order
relations, represented by blue cells, can be thought as relations among vertices or edges of the original graph.
(c): Extraction of a coarsened version of the original graph. In the coarsened graph, vertices represent the
higher-order relations (blue cells) of the original graph, and edges represent the intersections of these blue
cells. (d–e): The same process repeats to obtain a more coarsened version of the original graph. The entire
process corresponds to hierarchical higher-order relations, that is relations among relations, which extract
meaning and content (including the ‘shape of data’), a common task in topological data analysis [58, 86].


Modeling and processing data beyond graphs: illustrative examples. Graphs are well-suited for
modeling systems that exhibit pairwise interactions. For instance, particle-based fluid simulations can be
effectively represented using graphs as shown in Figure 3(a), with message passing used to update the physical
state of fluid molecules. In this approach, each molecule is represented as a vertex containing its physical
state, and the edges connecting them are utilized to calculate their interactions [226, 241].
Graphs may not be adequate for modeling more complex systems such as cloth simulation, where state
variables are associated with relations like edges or triangular faces rather than vertices. In such cases,
higher-order message passing is required to compute and update physical states, as depicted in Figure 3(b).
A similar challenge arises in natural language processing, where language exhibits multiple layers of syntax,
semantics and context. Although GNNs may capture basic syntactic and semantic relations between words,
more complex relations such as negation, irony or sarcasm may be difficult to represent [110]. Including
higher-order and hierarchical relations can provide a better model for these relations, and can enable a more
nuanced and accurate understanding of language.

2.2   The utility of topology

In addition to providing a versatile framework for modeling complex systems, TDL models have broad utility:
they can induce meaningful inductive biases for learning, facilitate efficient message propagation over longer
ranges in the underlying domain, enhance the expressive power of existing graph-based models, and have the
potential to unveil crucial characteristics of deep networks themselves, as described next.

Building hierarchical representations of data. While leveraging higher-order information is important
for learning representations, it is also crucial to preserve the complex higher-order relations between entities
for achieving powerful and versatile representations. For instance, humans can form abstractions and analogies
by building relations among relations hierarchically. However, graph-based models, which are commonly used
for building relational reasoning among various entities [227, 278, 69, 238], are limited in their ability to build
higher-order relations among these entities.
The need for hierarchical relational reasoning [269] demands methods that can capture deeper abstract
relations among relations, beyond modeling relations between raw entities. Higher-order network models
provide a promising solution for addressing the challenge of higher-order relational reasoning, as depicted in
Figure 4.


                                                         10
2   Motivation: what does topology offer to machine learning?                         Topological deep learning




Figure 5: The graph in the middle can be augmented with higher-order relations to improve a learning task.
In (a), cells have been added to the missing faces; a similar inductive bias has been considered in [44] to
improve classification on molecular data. In (b), one-hop neighborhoods have been added to some of the
vertices in the graph; [97] use such an inductive bias, based on lifting a graph to its corresponding one-hop
neighborhood hypergraph, to improve performance on vertex-based tasks.




Figure 6: Illustration of message-passing among binary or higher-order cells. (a): Using a graph-based
message-passing scheme, some information that starts at vertex x needs to travel a long distance, that is
a long edge path, before reaching vertex y. (b): Using a higher-order cell structure, indicated by the the
blue cell, the signal can be lifted from the vertices to a higher-order cell and thus propagates back and
forth between vertices x and y in fewer steps. This shortcut allows to share information efficiently, in fewer
computational steps, among all vertices that belong to the blue cell [126].

Improving performance through inductive bias. A topological deep learning model can be leveraged
to enhance the predictive performance on graph-based learning tasks by providing a well-defined procedure
to lift a graph to a given higher-order network. The incorporation of multi-vertex relations via this approach
can be viewed as an inductive bias, and is utilized to improve predictive performance. Inductive bias allows a
learning algorithm to prioritize one solution over another, based on factors beyond the observed data [194].
Figure 5 illustrates two forms of augmentation on graphs using higher-order cells.
Construction of efficient graph-based message passing over long-range. The hierarchical structure
of TDL networks facilitates the construction of long-range interactions in an efficient manner. By leveraging
such structures, signals defined on the vertices of a domain can propagate efficiently distant dependencies
among vertices connected by long edge paths. Figure 6 illustrates the process of lifting a signal defined on
graph vertices by augmenting the graph with additional topological relations and unpooling the signal back
to the vertices. [126] employ such pooling and unpooling operations to construct models capable of capturing
distant dependencies, leading to more effective and adaptive learning of both the local and global structure of
the underlying domain. For related work on graph-based pooling, we refer the reader to [141, 247].

Improved expressive power. TDL can capture intricate and subtle dependencies that elude traditional
graph-based models. By allowing richer higher-order interactions, higher-order networks can have improved
expressive power and lead to superior performance in numerous tasks [44, 197].


                                                       11
Topological deep learning                                 3   Preliminaries: from graphs to higher-order networks




Figure 7: This work introduces combinatorial complexes, which are higher-order networks that generalize
most discrete domains typically encountered in scientific computing, including (a) sequences and images, (b)
graphs, (c) 3D shapes and simplicial complexes, (d) cubical and cellular complexes, (e) discrete manifolds,
and (f) hypergraphs.

2.3   A unifying perspective on deep learning and structured computations

Topology provides a framework to generalize numerous discrete domains that are typically encountered in
scientific computations. Examples of these domains include graphs, simplicial complexes, and cell complexes
(see Figure 7). Computational models supported on such domains can be seen as a generalization of various
deep learning architectures; i.e., can provide a unifying framework to compare and design TDL architectures.
Further, understanding the connections between different types of TDL via a unifying theory can provide
valuable insights into the underlying structure and behavior of complex systems, where higher-order relations
naturally occur [123, 132, 41, 182].

Understanding why deep networks work. A remarkable aspect of contemporary deep neural networks is
their capacity to generalize very well to unseen data, even though they have an enormous number of parameters.
This contradicts our previous understanding of statistical learning theory, prompting researchers to seek
novel approaches in comprehending the underlying workings of deep neural networks [201]. Recent studies
have revealed that the topology of graphs induced by deep neural networks or their corresponding training
trajectories, modeled as Markov chains, exhibit strong correlations with generalization performance [43, 90].
An open question is how far such correlations between topological structure, computational function and
generalization properties can be found in learning architectures supported on higher-order domains. We
believe that by studying these questions using our novel topological constructs, such as CCs and CCNNs, we
can gain deeper insights not only into TDL but also into deep learning more generally.

3     Preliminaries: from graphs to higher-order networks

The notion of proximity among entities in a set S holds significant relevance in various machine learning
applications as it facilitates the comprehension of inter-entity relationships in S. For example, clustering
algorithms aim to group points that are proximal to each other. In recommendation systems, the objective is


                                                     12
3   Preliminaries: from graphs to higher-order networks                                   Topological deep learning




          (a)                    (b)                   (c)                   (d)                      (e)

Figure 8: Demonstration of the notion of proximity among the entities of a set S. (a): A finite set S of
abstract entities. (b): A neighborhood (in yellow) of an entity x (in red) in S. The neighborhood is defined
to be the set of entities in S that are adjacent to x via an edge. (c): A neighborhood of x consisting of all
yellow entities in S that have distance at most two from red entity x. (d): A neighborhood of x consisting
of all yellow entities in S that form triangles (in blue) incident to red entity x. (d): A neighborhood of x
consisting of all yellow entities in S that form trapezoids (in blue) incident to red entity x.


to suggest items that exhibit similarity to those for which a user has already expressed interest. However, the
question is how do we precisely quantify the notion of proximity?
 Consider a set S consisting of a collection of abstract entities, as depicted in Figure 8(a). Consider the red
 entity (vertex) x in the same figure. We wish to identify the entities (vertices) in S that are ‘closely related’ or
‘in close proximity’ to x. However, a set has no inherent notion of proximity or relations between its entities.
Defining binary relations (edges) among the entities of set S provides one way of introducing the concept
of proximity, which results in a graph whose vertex set is S, as shown in Figure 8(b). Using this ‘auxiliary
structure’ of edges defined on top of set S, one may declare the ‘local neighborhood’ of x, denoted by N (x),
to be the subset of S consisting of all entities that are adjacent to x via an edge. In Figure 8(b), the
neighborhood of vertex x (in red) in S consists of all vertices colored in yellow.
The choice of neighborhood N (x) in Figure 8(b) is arbitrary. For example, an alternative valid notion of
neighborhood is given by defining N (x) to contain all vertices whose distance from the red vertex x is at
most two, represented by all vertices colored in yellow in Figure 8(c). In Figure 8(d), the neighborhood
N (x) is chosen to consist of all yellow vertices that form triangles incident to red vertex x. In Figure 8(e),
the neighborhood N (x) is chosen to consist of all yellow vertices that form trapezoids incident to the red
vertex x. It becomes clear from Figures 8(d) and (e) that additional auxiliary structures, such as triangles
and trapezoids, can be used to define the notion of neighborhood. In practice, the choice of neighborhood
typically depends on the application. Recent research has explored using graph geometry to obtain richer
neighborhood notions, as seen in [197, 282, 123]. However, the fundamental concept remains the same: one
starts by introducing an auxiliary structure defined on a vertex set, and then utilizes the auxiliary structure
to induce a well-defined notion of proximity.
Upon generalizing graphs to higher-order networks, it is natural to also generalize the notion of neighborhood
for graphs to higher-order networks. The precise notion of neighborhood or proximity among entities of a
set S has been studied in topology [199]. A topology defined on S allows us to meaningfully describe the
proximity of the elements in S to each other. This section introduces topological notions and definitions with
the aim of generalizing graphs to higher-order networks.


3.1   Neighborhood functions and topological spaces

There are several equivalent ways to define a topological space. For example, topological spaces are commonly
defined in terms of ‘open’ or ‘closed’ sets (e.g., [199]). In this work, we opt to define topological spaces in
terms of ‘neighborhoods’. This definition is more compatible with the message-passing paradigm that is
usually defined on graphs [109], and it can be generalized to higher-order networks. We refer the reader
to [51, Section 2.2] for an explanation of why the definition in terms of neighborhoods is equivalent to the
definition in terms of open sets.


                                                          13
Topological deep learning                                         3   Preliminaries: from graphs to higher-order networks




      Definition 1 (Neighborhood function). Let S be a nonempty set. A neighborhood function on S
      is a function N : S → P(S) that assigns to each point x in S a nonempty collection N (x) of subsets
      of S. The elements of N (x) are called neighborhoods of x with respect to N .


      Definition 2 (Neighborhood topology). Let N be a neighborhood function on a set S. N is called a
      neighborhood topology on S if it satisfies the following axioms:

           1. If N is a neighborhood of x, then x ∈ N .
           2. If N is a subset of S containing a neighborhood of x, then N is a neighborhood of x.
           3. The intersection of two neighborhoods of a point x in S is a neighborhood of x.
           4. Any neighborhood N of a point x in S contains a neighborhood M of x such that N is a
              neighborhood of each point of M .


      Definition 3 (Topological space). A pair (S, N ) consisting of a nonempty set S and a neighborhood
      topology N on S is called a topological space.

Hence, a topological space is a set S equipped with a neighborhood function N , which satisfies the properties
specified in Definition 2. In Section 4.4, we introduce a similar notion of proximity in the context of higher-
order networks. Further, the choice of neighborhood function N is the first and most fundamental step in the
construction of a deep learning model supported on a higher-order domain (see Section 5).

3.2     Bridging the gap among higher-order networks

Given a finite set S of abstract entities, a neighborhood function N on S can be induced by equipping S
with an auxiliary structure, such as edges, as demonstrated in Figure 8(b). Edges provide one way of defining
relations among the entities of S 2 . Specifically, each edge defines a binary relation (i.e., a relation between
two entities) in S. In many applications, it is desirable to permit relations that incorporate more than two
entities. The idea of using relations that involve more than two entities is central to higher-order networks.
Such higher-order relations allow for a broader range of neighborhood functions to be defined on S to capture
multi-way interactions among entities of S.
To describe more intricate multi-way interactions, it is necessary to employ more intricate neighborhood func-
tions and topologies. With an eye towards defining a general higher-order network in Section 4 (as motivated
in Section 2), this section reviews the definitions, advantages, and disadvantages of some commonly studied
higher-order networks, including (abstract) simplicial complexes, regular cell complexes, and hypergraphs. In
Section 4, we introduce combinatorial complexes, which generalize and bridge the gaps between all of these
commonly studied topological domains.
Simplicial complexes are one of the simplest higher-order domains with many desirable properties, extending
the corresponding properties of graphs. For instance, Hodge theory is naturally defined on simplicial complexes,
extending similar notions on graphs [234, 21, 235].

      Definition 4 (Simplicial complex). An abstract simplicial complex on a nonempty set S is a pair
      (S, X ), where X is a subset of P(S) \ {∅} such that x ∈ X and y ⊆ x imply y ∈ X . Elements of X are
      called simplices.

Figure 7(c) displays examples of triangular meshes, which are special cases of simplicial complexes with many
applications in computer graphics. We refer the reader to [235, 75] for a relevant introduction to simplicial
complexes. As it is evident from Definition 4, each relation x on S must contain all relations y with y ⊆ x.
  2 Recall that a relation on S is a nonempty subset of S.




                                                             14
3   Preliminaries: from graphs to higher-order networks                                  Topological deep learning




Hence, a simplicial complex may encode a relatively large amount of data, taking up a lot of memory [222].
Further, real-world higher-order data (e.g., traffic-flows on a rectangular street network) may not admit a
meaningful simplicial complex structure due to the inherent lack of available simplices on the underlying data
space. To address this limitation, cell complexes [133, 132] can generalize simplicial complexes, and overcome
many of their drawbacks.



    Definition 5 (Regular cell complex). A regular cell complex is a topological space S with a partition
    into subspaces (cells) {xα }α∈PS , where PS is an index set, satisfying the following conditions:
         1. S = ∪α∈PS int(xα ), where int(x) denotes the interior of cell x.

         2. For each α ∈ PS , there exists a homeomorphism ψα , called an attaching map, from xα to
            Rnα for some nα ∈ N, called the dimension nα of cell xα .
         3. For each cell xα , the boundary ∂xα is a union of finitely many cells, each having dimension
            less than the dimension of xα .


For the sake of brevity, we henceforth refer to a ‘regular cell complex’ as a ‘cell complex’. Cell complexes
encompass a wide range of higher-order networks. Several types of higher-order networks can be viewed as
instances of cell complexes. For instance, cell complexes are a natural generalization of graphs, simplicial
complexes, and cubical complexes [123]. Figure 7(d) shows examples of cell complexes. Intuitively, a
cell complex is a disjoint union of cells, with each of these cells being homeomorphic to the interior of a
k-dimensional Euclidean ball for some k. These cells are attached together via attaching maps in a locally
suitable manner. The information of attaching maps of a regular cell-complex can be stored combinatorially
in a sequence of matrices, called the incidence matrices [133]. We describe these matrices in detail in
Section 4.4.1.
Condition 3 in Definition 5 is known as the regularity condition for regular cell complexes. The regularity
condition implies that the topological information of a cell complex can be realized combinatorially by
equipping the index set PS with a poset structure given by α ≤ β if and only if xα ⊆ xβ , where x denotes
the closure of a cell x. This poset structure is typically called the face poset [132]. It can be shown that the
topological information encoded in a cell complex is completely determined by the face poset structure [132],
which allows cell complexes to be represented combinatorially in practice via a poset [8, 25, 231, 152].
Definition 5 implies that the boundary cells of each cell in a cell complex are also cells in the cell complex.
Hence, one may think about a cell complex as a collection of cells of varying dimensions, which are related
via their boundaries. In terms of relations, this implies that the boundary of a cell in a cell complex must
also be a cell in the cell complex. While cell complexes form a general class of higher-order networks, this
property sets a constraint on the relations of a cell complex. Such a constraint may not be desirable in certain
applications if the data do not satisfy it. To remove all constraints on the relations among entities of a set,
hypergraphs are typically considered.



    Definition 6 (Hypergraph). A hypergraph on a nonempty set S is a pair (S, X ), where X is a
    subset of P(S) \ {∅}. Elements of X are called hyperedges.



A hyperedge of cardinality two is called an edge. Hypergraphs can be considered as a generalization of
simplicial and cell complexes. However, hypergraphs do not directly entail the notion of the dimension of a
cell (or of a relation), which is explicitly encoded in the definition of cell complex and is also indicated by the
cardinality of a relation in a simplicial complex. As we demonstrate in Section 4.3, the dimensionality of
cells and relations in simplicial and cell complexes can be used to endow these complexes with hierarchical
structures, which can be utilized for (un)pooling type computations on these structures.


                                                          15
Topological deep learning                                                         4   Combinatorial complexes (CCs)




         No relations        Binary relations            Hierarchical relations        Set-type relations




Figure 9: Illustration of how CCs generalize various domains. (a): A set S consists of abstract entities
(vertices) with no relations. (b): A graph models binary relations between its vertices (i.e., elements of S).
(c): A simplicial complex models a hierarchy of higher-order relations (i.e., relations between the relations)
but with rigid constraints on the ‘shape’ of its relations. (d): Similar to simplicial complexes, a cell complex
models a hierarchy of higher-order relations, but with more flexibility in the shape of the relations (i.e., ‘cells’).
(f): A hypergraph models arbitrary set-type relations between elements of S, but these relations do not have
a hierarchy among them. (e): A CC combines features of cell complexes (hierarchy among its relations) and
of hypergraphs (arbitrary set-type relations), generalizing both domains.

3.3     Hierarchical structure and set-type relations

The properties of simplicial complexes, cell complexes and hypergraphs, as outlined in Section 3.2, give rise to
two main features of relations on higher-order domains, namely hierarchies of relations and set-type relations.
In the present subsection, we formalize these two features.

      Definition 7 (Rank function). A rank function on a higher-order domain X is an order-preserving
      function rk : X → Z≥0 ; i.e., x ⊆ y implies rk(x) ≤ rk(y) for all x, y ∈ X .

Intuitively, a rank function rk on a higher-order domain X attaches a ranking, represented by a non-negative
integer value, to every relation in X such that set inclusion in X is preserved via rk. Effectively, a rank
function induces a hierarchical structure on X . Cell and simplicial complexes are common examples of
higher-order domains equipped with rank functions and therefore with hierarchies of relations.

      Definition 8 (Set-type relations). Relations in a higher-order domain are called set-type relations
      if the existence of a relation is not implied by another relation in the domain.

Hypergraphs constitute examples of higher-order domains equipped with set-type relations. Given the
modeling limitations of simplicial complexes, cell complexes, and hypergraphs, we develop the combinatorial
complex in Section 4, a higher-order domain that features both hierarchies of relations and set-type relations.

4      Combinatorial complexes (CCs)

This section introduces combinatorial complexes (CCs), a novel class of higher-order domains that generalizes
graphs, simplicial complexes, cell complexes, and hypergraphs. Figure 9 shows a first illustration of the
generalization that CCs provide over such domains. Moreover, Table 1 enlists the relation-related features of
higher-order domains and graphs, thus summarizing the relation-related generalization attained by CCs.
In Section 4.1, we introduce the definition of CC and provide examples of CCs. In Section 4.2, we define
the notion of CC-homomorphism and present relevant examples. In Section 4.3, we present the motivation
behind the CC structure from a practical point of view. In Section 4.4, we demonstrate the computational
version of neighborhood functions on neighborhood matrices. Finally, in Section 4.5, we introduce the notion
of CC-cochain.


                                                         16
4   Combinatorial complexes (CCs)                                                        Topological deep learning




                                                        Topological domains
             Relation-related features                                                           Graph
                                         CC   Hypergraph Cell complex Simplicial complex
             Hierarchy of relations      ✓                      ✓             ✓
             Set-type relations          ✓        ✓
             Multi-relation coupling     ✓                      ✓             ✓
             Rank ⇏ cardinality          ✓        ✓             ✓

Table 1: Tabular summary of relation-related features of topological domains and of graphs. Recall that a
 relation is an element of a domain. A domain is specified via its relations and via the way these relations are
 related to each other. Desirable relation-related features are indicated in the first column. A hierarchy of
relations implies that relations of the higher-order domain can have different rankings. Set-type relations are
 free from constraints among relations or their lengths. Multi-relation coupling implies that each relation can
 have other neighboring relations via multiple neighborhood functions defined on the higher-order domain.
‘Rank ⇏ cardinality’ indicates that relations with the same ranking in a given hierarchy on the higher-order
 domain do not need to have the same cardinality.


4.1     CC definition

We seek to define a structure that bridges the gap between simplicial/cell complexes and hypergraphs, as
discussed in Section 3.3. To this end, we introduce the combinatorial complex (CC), a higher-order domain
that can be viewed from three perspectives: as a simplicial complex whose cells and simplices are allowed to
be missing; as a generalized cell complex with relaxed structure; or as a hypergraph enriched through the
inclusion of a rank function.

      Definition 9 (CC). A combinatorial complex (CC) is a triple (S, X , rk) consisting of a set S, a
      subset X of P(S) \ {∅}, and a function rk : X → Z≥0 with the following properties:
           1. for all s ∈ S, {s} ∈ X , and
           2. the function rk is order-preserving, which means that if x, y ∈ X satisfy x ⊆ y, then rk(x) ≤
              rk(y).

      Elements of S are called entities or vertices, elements of X are called relations or cells, and rk is
      called the rank function of the CC.

For brevity, X is used as shorthand notation for a CC (S, X , rk). Definition 9 sets up a framework to construct
higher-order networks upon which we can define general purpose higher-order deep learning architectures.
Observe that CCs exhibit both hierarchical and set-type relations. Particularly, the rank function rk of a
CC induces a hierarchy of relations in the CC. Further, CCs encompass set-type relations as there are no
relation constraints in Definition 9. Thus, CCs subsume cell complexes and hypergraphs in the sense that they
combine the relation-related features of both. Table 1 provides a comparative summary of relation-related
features among CCs and common higher-order networks and graphs.
Remark 4.1. We typically require that rk({s}) = 0 for each singleton cell {s} in a CC. Such a convention
aligns CCs naturally with simplicial and cellular complexes.

The rank of a cell x ∈ X is the value rk(x) of the rank function rk at x. The dimension dim(X ) of a CC X is
the maximal rank among its cells. A cell of rank k is called a k-cell and is denoted by xk . The k-skeleton of
a CC X , denoted X (k) , is the set of cells of rank at most k in X . The set of cells of rank exactly k is denoted
by X k . Note that this set corresponds to X k = rk−1 ({k}). The 1-cells are called the edges of X . In general,
an edge of a CC may contain more than two nodes. CCs whose edges have exactly two nodes are called
graph-based CCs. In this paper, we primarily work with graph-based CCs.
Example 4.2 (CCs of dimension two and three). Figure 10 shows four examples of CCs. For instance,
Figure 10(a) shows a 2-dimensional CC on a vertex set S = {s0 , s1 , s2 }, consisting of 0-cells {s0 }, {s1 }, and
{s2 } (shown in orange), 1-cell {s1 , s2 } (purple), and 2-cell {s0 , s1 , s2 } = S (blue).


                                                        17
Topological deep learning                                                      4   Combinatorial complexes (CCs)




Figure 10: Examples of CCs. Orange circles represent vertices. Pink, blue, and green colors represent cells
of rank one, two, and three, respectively. Each of the CCs in (a), (b), and (d) has dimension equal to two,
whereas the CC in (c) has dimension equal to three.


4.2     CC-homomorphisms and sub-CCs

CC-homomorphisms are maps relating CCs to one another. CC-homomorphisms play an important role in
delineating the process of lifting graphs or other higher-order domains to CCs. Intuitively, a lifting map is a
well-defined procedure that converts a domain of certain type, such as a graph, to a domain of another type,
such as a CC. The usefulness of lifting maps lie in their ability to enable the application of deep learning
models defined on CCs to more common domains, such as graphs, cell complexes or simplicial complexes. We
study lifting maps in detail and provide examples of them in Appendix B. Definition 10 formalizes the notion
of CC-homomorphism.

      Definition 10 (CC-homomorphism). A homomorphism from a CC (S1 , X1 , rk1 ) to a CC (S2 , X2 , rk2 ),
      also called a CC-homomorphism, is a function f : X1 → X2 that satisfies the following conditions:

           1. If x, y ∈ X1 satisfy x ⊆ y, then f (x) ⊆ f (y).
           2. If x ∈ X1 , then rk1 (x) ≥ rk2 (f (x)).

The second condition in Definition 10 assures that a CC-homomorphism may only map a k-cell in X1 to a cell
in X2 with rank no greater than k. When rk1 (x) = rk2 (f (x)) for all x ∈ X1 and f is injective, then we call
the homomorphism f a CC-embedding. CC-embeddings are useful in practice as they can be used to ‘lift’ a
domain, such as a graph, to a CC by augmenting that domain with higher-order cells. Example 4.3 displays
three CC-embeddings, while Example 4.4 presents a CC-homomorphism that is not a CC-embedding.
Example 4.3 (CC-embeddings). Figures 11(a) and (b) show two CC-embeddings. In Figure 11(a), each cell
in the graph on the left-hand side is sent to its corresponding cell in the CC on the right-hand side. Similarly,
in Figure 11(b), each cell in the cell complex on the left is sent to its corresponding cell in the CC on the
right. It is easy to verify that each of these two maps is a CC-embedding.

Example 4.4 (A CC-homomorphism). We present an example of a CC-homomorphism that is not a CC-
embedding. Consider the sets S1 = {1, 2, 3, 4} and S2 = {a, b, c}. Let X1 denote the CC on S1 consisting of
one 3-cell {1, 2, 3, 4}, one 2-cell {1, 2, 3}, and four 0-cells corresponding to the elements of S1 . Likewise, let
X2 denote the CC on S2 consisting of one 3-cell {a, b, c}, one 2-cell {a, b}, and three 0-cells corresponding to
the elements of S2 . CCs X1 and X2 are visualized in Figure 11(c). Consider the function f : S1 → S2 defined
as f (1) = f (2) = a, f (3) = b and f (4) = c. It is easy to verify that f induces a CC-homomorphism from X1
to X2 .

      Definition 11 (Sub-CC). Let (S, X , rk) be a CC. A sub-combinatorial complex (sub-CC) of a
      CC (S, X , rk) is a CC (A, Y, rk′ ) such that A ⊆ S, Y ⊆ X and rk′ = rk |Y is the restriction of rk on
      Y.

For brevity, we refer to the sub-CC (A, Y, rk′ ) as Y. Any subset of A ⊆ S can be used to induce a sub-CC as
follows. Consider the set XA = {x ∈ X | x ⊆ A} together with the restriction rk |XA . It is easy to see that


                                                         18
4   Combinatorial complexes (CCs)                                                          Topological deep learning




Figure 11: Examples of CC-homomorphisms. Pink, blue, and green colors represent cells of rank one, two, and
three, respectively. (a): An embedding of a 1-dimensional CC into a 2-dimensional CC. (b): An embedding
of a 2-dimensional CC into a 3-dimensional CC. (c) An example of a CC-homomorphism, as defined in
Example 4.4. The CC X1 is on the left, X2 is on the right, and the homomorphism f is represented by the
black arrow. Intuitively, the CC-homomorphism f can be viewed as a combinatorial analogue to a continuous
function between S1 = {1, 2, 3, 4} and S2 = {a, b, c} that ‘collapses’ the cell {1, 2, 3} to the cell {a, b}, and
the cell {1, 2, 3, 4} to the cell {a, b, c}. Observe that CC-homomorphisms generalize simplicial maps [198]
from this perspective.


the triple (A, XA , rk |XA ) constitutes a CC, which we call the sub-CC of X induced by A. Note that any cell
in the set X induces a sub-CC obtained by considering all the cells that are contained in it. Finally, for any
k, it is easy to see that the skeleton X k of a CC X is a sub-CC.
Example 4.5 (Sub-CC). Recall the CC X = {{s0 }, {s1 }, {s2 }, {s1 , s2 }, {s0 , s1 , s2 }} displayed in Figure 10(a).
The set A = {s1 , s2 } induces the sub-CC XA = {{s1 }, {s2 }, {s1 , s2 }} of X .


4.3     Motivation for CCs

Definition 9 for CCs aims to fulfill all motivational aspects of higher-order modeling, as outlined in Section 2.
To further motivate Definition 9, we consider the pooling operations on CCs along with several structural
advantages of CCs.

4.3.1    Pooling operations on CCs

We first consider the general characteristics of pooling on graphs, and then demonstrate how graph-based
pooling can be realized in a unified manner via CCs. A general pooling function on a graph G is a function
POOL : G → G ′ , where G ′ is a pooled graph that represents a coarsened version of G. A vertex of G ′
corresponds to a cluster of vertices (a super-vertex) in the original graph G, while edges in G ′ indicate the
presence or absence of a connection among such clusters. See Figure 12 for an example.
The formalism proposed via CCs encompasses graph-based pooling as a special case. Specifically, the super-
vertices (cluster of vertices in G) defined by the vertices in G ′ can be realized as higher-order ranked cells in a
CC obtained by augmenting G with the cells representing these super-vertices. The hierarchical structure
consisting of the original graph G as well as augmented higher-order cells induces a CC. This realization of
graph-based pooling in terms of CC-based higher-ranked cells is hierarchical because the new cells can be
grouped together in a recursive manner to obtain a coarser version of the underlying space. Given such a
notion of higher-order pooling operations on CCs, we demonstrate how to express graph/mesh pooling and
image-based pooling in Sections 7.2.1 and 7.2.2, respectively. From this perspective, CCs provide a general
framework for defining pooling operations on higher-order networks, which include graphs and images as
special cases.
In addition, two main features of CCs are exploited via higher-order pooling operations. First, the ability of
CCs to model set-type cells provides flexibility in the shape of the clusters that define the pooling operation.
This flexibility is needed to realize novel user-defined pooling operations, in which the shape of the clusters
might be task-specific. Second, higher-order ranked cells in a CC correspond to a coarser version of the
underlying space. The type of higher-order pooling proposed here enables the construction of coarser
representations. See Figure 12 for an example. Note that less general structures, such as hypergraphs and


                                                         19
Topological deep learning                                                     4   Combinatorial complexes (CCs)




                                                       (a)




                            (b)                      (c)                          (d)




Figure 12: Graph-based pooling formulated via CCs. (a): Pooling on graphs is expressed here in terms of a
pooling function that maps a graph along with the data defined on it to a coarser version of the graph. The
super-vertices in the pooled graph on the right correspond to a clustering of vertices in the original graph on
the left, whereas edges in the graph on the right indicate the presence or absence of a connection among these
clusters. (b-d): The super-vertices in the pooled graph on the right can be realized as augmented higher-order
cells (blue cells) in a CC obtained from the original graph. The edges in the pooled graph on the right can be
realized in terms of an incidence matrix of the CC between the vertices and the higher-order augmented cells.



cell complexes, do not accommodate simultaneous flexibility in cluster shaping and generation of coarser
representations of the underlying space.


4.3.2   Structural advantages of CCs

In addition to the realization of graph-based pooling, CCs offer several structural advantages. Specifically, CCs
unify numerous commonly used higher-order networks, enable fine-grained analysis of topological properties,
facilitate message passing of topological features in deep learning, and accommodate flexible modeling of
relations among relations.


Flexible higher-order structure and fine-grained message passing. Message-passing graph models
pass messages between vertices in a graph to learn the representations of the graph, with messages computed
based on the vertex and edge features. These messages update the vertex and edge features, and gather
information from the graph’s local neighborhood. The rank functions of CCs render CCs more versatile than
other higher-order networks and than graphs in two fronts. First, rank functions make CCs more flexible in
terms of higher-order structure representation. Second, in the context of deep learning, rank functions equip
CCs with more fine-grained message-passing capabilities. For instance, each hyperedge in a hypergraph is
treated as a set without any notion of a rank, and consequently all hyperedges are treated uniformly without
any distinction. Further details are provided in Section 5.


Flexible modeling of relations among relations. In the process of populating a topological domain
with topological data, constructing meaningful relations can be challenging due to the inherent lack of data
that can be naturally supported on all cells in the domain. This is particularly true when working with
simplicial or cell complexes. For instance, any relation between k entities of a simplicial complex must be
built from the relations of all corresponding subsets of k − 1 entities. Real-world data may contain a subset
of these relations, and not all of them. While cell complexes offer more flexibility in modeling relations, the
boundary conditions that cell complexes must satisfy constraints the types of permissible relations. In order
to remove all restrictions among relations, hypergraphs come to aid as they allow arbitrary set-type relations.
However, hypergraphs do not offer hierarchical features, which can be disadvantageous in applications that
require considering local and global features simultaneously.


                                                       20
4   Combinatorial complexes (CCs)                                                         Topological deep learning




                       (a)




                       (b)




Figure 13: A visual comparison between neighborhood functions with continuous domains and CC-
neighborhood functions, adhering to Definitions 1 and 12, respectively. (a): A neighborhood function
with a continuous domain S assigns to x ∈ S a set N (x) of subsets of S that are in the local vicinity of x.
(b): Similarly, a CC-neighborhood function on a CC (S, X , rk) assigns to x ∈ S a set N (x) of subsets of X
that are in the local vicinity of x.

4.4     Neighborhood functions on CCs

We introduce the notion of a CC-neighborhood function on a CC as a mechanism of exploiting the topological
information stored in the CC. In practice, crafting a neighborhood function is usually a part of the learning
task. For our purposes, we restrict the discussion to two types of generalized neighborhood functions, namely
those specifying adjacency and incidence. From a deep learning perspective, CC-neighborhood functions set
the foundations to extend the general message-passing schemes of deep learning models, thus subsuming
several state-of-the-art GNNs [99, 177, 27, 197, 26, 150].
Given a CC, we aim to describe cells in the local proximity of a sub-CC of the CC. To this end, we define the
CC-neighborhood function, an analogue of Definition 1 in the context of CCs.

      Definition 12 (CC-neighborhood function). A CC-neighborhood function on a CC (S, X , rk) is a
      function N that assigns to every sub-CC (A, Y, rk′ ) of the CC a nonempty collection N (Y) of subsets
      of S.

Without loss of generality, we assume that the elements of the neighborhood N (Y) are cells or sub-CCs of X .
Intuitively, the neighborhood N (Y) of the sub-CC Y is a set of subsets of S that are in the ‘local vicinity’ of
Y. The term ‘local vicinity’ is generally stated here, since it is typically context-specific.
Definition 12 is a discrete analogue of the well-known Definition 1. The correspondence between these
two definitions is demonstrated in Figure 13. For the rest of the paper, CC-neighborhood functions are
succinctly called neighborhood functions. In practice, the information encoded in CC-neighborhood functions
is represented in terms of matrices as described next.

Neighborhood matrix induced by a neighborhood function. For the purposes of computation, it is
convenient to represent neighborhood functions as matrices. Incidence, adjacency, and coadjacency matrices
are well-known examples of matrices that encode respective neighborhood functions. In Definition 13, we
introduce a generalization of these matrices called the ‘neighborhood matrix’. In this definition and henceforth,
we denote the cardinality of a set S by |S|.

      Definition 13 (Neighborhood matrix). Let N be a neighborhood function defined on a CC X .
      Moreover, let Y = {y1 , . . . , yn } and Z = {z1 , . . . , zm } be two collections of cells in X such that
      N (yi ) ⊆ Z for all 1 ≤ i ≤ n. The neighborhood matrix of N with respect to Y and Z is the
      |Z| × |Y| binary matrix G whose (i, j)-th entry [G]ij has value 1 if zi ∈ N (yj ) and 0 otherwise.



                                                         21
Topological deep learning                                                      4   Combinatorial complexes (CCs)




Remark 4.6. In Definition 13, N (yj ) is stored in the j-th column of the associated neighborhood matrix
G. For this reason, we denote by NG (j) the neighborhood function of the cell yj when we work with the
neighborhood matrix G.

There are many ways to define useful neighborhood functions on a CC. In this work, we constrain ourselves
to the most immediate neighborhood functions: the incidence and adjacency neighborhood functions.

4.4.1   Incidence in a CC

We define three notions of incidence to capture different facets of incidence structures of cells in a CC. First,
in Definition 14, we introduce the down-incidence and up-incidence neighborhood functions to describe the
incidence structure of a cell via cells of arbitrary rank. Second, in Definition 15, we introduce the k-down and
k-up incidence neighborhood functions to describe the incidence structure of a cell via cells of a particular
rank k. Third, in Definition 16, we introduce (r, k)-incidence matrices to describe the incidence structures of
cells of particular ranks r and k. In what follows, we assume that the cells in the set X of a CC (S, X , rk) are
given a fixed order.


   Definition 14 (Down/up-incidence neighborhood functions).          Let (S, X , rk) be a CC. Two cells
   x, y ∈ X of the CC are called incident if either x ⊊ y or y ⊊ x. In particular, the down-incidence
   neighborhood function N↘ (x) of a cell x ∈ X is defined to be the set {y ∈ X | y ⊊ x}, while the
   up-incidence neighborhood function N↗ (x) of x is defined to be the set {y ∈ X | x ⊊ y}.


Definition 15 provides a more granular specification of incidence than Definition 14. In particular, Definitions 15
and 14 describe the incidence structure of a cell with respect to cells of a particular rank or of arbitrary rank,
respectively.


   Definition 15 (k-down/up incidence neighborhood functions). Let (S, X , rk) be a CC. For any k ∈ N,
   the k-down incidence neighborhood function N↘,k (x) of a cell x ∈ X is defined to be the set
   {y ∈ X | y ⊊ x, rk(y) = rk(x) − k}. The k-up incidence neighborhood function N↗,k (x) of x is
   defined to be the set {y ∈ X | y ⊊ x, rk(y) = rk(x) + k}.

Clearly, N↘ (x) = k∈N N↘,k (x) and N↗ (x) = k∈N N↗,k (x). Immediate incidence is particularly important.
                  S                             S
To this end, the set of faces of a cell x ∈ X is defined to be N↘,1 (x), and the set of cofaces of a cell x is
defined to be N↗,1 (x). An illustration of k-down and k-up incidence neighborhood functions is given in
Figure 14.


   Definition 16 (Incidence matrix). Let (S, X , rk) be a CC. For any r, k ∈ Z≥0 with 0 ≤ r < k ≤
   dim(X ), the (r, k)-incidence matrix Br,k between X r and X k is defined to be the |X r | × |X k | binary
   matrix whose (i, j)-th entry [Br,k ]ij equals one if xri is incident to xkj and zero otherwise.


The incidence matrices Br,k of Definition 16 determine a neighborhood function on the underlying CC.
Neighborhood functions induced by incidence matrices are utilized to construct a higher-order message
passing scheme on CCs, as described in Section 5.

4.4.2   Adjacency in a CC

CCs admit incidence as well as other neighborhood functions. For instance, for a CC that is reduced to
a graph, a more natural neighborhood function is based on the notion of adjacency relation. Incidence
defines relations among cells of different ranks, while adjacency defines relations among cells of similar
ranks. Definitions 17, 18 and 19/20 introduce the (co)adjacency analogues of the respective incidence-related
Definitions 14, 15, and 16.


                                                        22
4   Combinatorial complexes (CCs)                                                            Topological deep learning




                                 k-down incidence




                                 k-up incidence




Figure 14: Illustration of k-down and k-up incidence neighborhood functions on a CC of dimension three.
(a): Illustration of k-down incidence neighborhood functions. The target orange cell x has rank three. From
left to right, the red cells represent N↘,1 (x), N↘,2 (x) and N↘,3 (x). (b): Illustration of k-up incidence
neighborhood functions. The target orange cell x has rank zero. From left to right, the red cells represent
N↗,1 (x), N↗,2 (x) and N↗,3 (x).



    Definition 17 ((Co)adjacency neighborhood functions). Let (S, X , rk) be a CC. The adjacency
    neighborhood function Na (x) of a cell x ∈ X is defined to be the set

                    {y ∈ X | rk(y) = rk(x), ∃z ∈ X with rk(z) > rk(x) such that x, y ⊊ z}.

    The coadjacency neighborhood function Nco (x) of x is defined to be the set

                {y ∈ X | rk(y) = rk(x), ∃z ∈ X with rk(z) < rk(x) such that z ⊊ y and z ⊊ x}.

    A cell z satisfying the conditions of either Na (x) or Nco (x) is called a bridge cell.


    Definition 18 (k-(co)adjacency neighborhood functions). Let (S, X , rk) be a CC. For any k ∈ N,
    The k-adjacency neighborhood function Na,k (x) of a cell x ∈ X is defined to be the set

                  {y ∈ X | rk(y) = rk(x), ∃z ∈ X with rk(z) = rk(x) + k such that x, y ⊊ z}.

    The k-coadjacency neighborhood function Nco,k (x) of x is defined to be the set

              {y ∈ X | rk(y) = rk(x), ∃z ∈ X with rk(z) = rk(x) − k such that z ⊊ y and z ⊊ x}.


    Definition 19 (Adjacency matrix). For any r ∈ Z≥0 and k ∈ Z>0 with 0 ≤ r < r + k ≤ dim(X ), the
    (r, k)-adjacency matrix Ar,k among the cells of X r with respect to the cells of X k is defined to be
    the |X r | × |X r | binary matrix whose (i, j)-th entry [Ar,k ]ij equals one if xri is k-adjacent to xrj and
    zero otherwise.


    Definition 20 (Coadjacency matrix). For any r ∈ Z≥0 and k ∈ N with 0 ≤ r − k < r ≤ dim(X ), the
    (r, k)-coadjacency matrix coAr,k among the cells of X r with respect to the cells of X k is defined to
    be the |X r | × |X r | binary matrix whose (i, j)-th entry [coAr,k ]ij has value 1 if xri is k-coadjacent to xrj
    and 0 otherwise.

Clearly, Na (x) = ∪k∈N Na,k (x) and Nco (x) = ∪k∈N Nco,k (x). An illustration of k-adjacency and k-coadjacency
neighborhood functions is given in Figure 15.


                                                          23
Topological deep learning                                                        4   Combinatorial complexes (CCs)




                                k-adjacency




                                k-coadjacency




Figure 15: Illustration of k-(co)adjacency neighborhood functions on a CC of dimension three. (a): Illustration
of k-adjacency neighborhood functions. The target orange cell x has rank zero. From left to right, the red
cells represent Na,1 (x), Na,2 (x) and Na,3 (x). (b): Illustration of k-coadjacency neighborhood functions. The
target orange cell x has rank two. From left to right, the red cells represent Nco,1 (x), Nco,2 (x) and Nco,3 (x).


4.5     Data on CCs: cochain spaces and maps

As we are interested in processing the data defined over a CC (S, X , rk), we introduce k-cochain spaces,
k-cochains and cochain maps.

      Definition 21 (k-cochain spaces). Let C k (X , Rd ) be the R-vector space of functions Hk : X k → Rd for
      a rank k ∈ Z≥0 and dimension d. d is called the data dimension. C k (X , Rd ) is called the k-cochain
      space. Elements Hk in C k (X , Rd ) are called k-cochains or k-signals.

We use the notation C k (X ) or C k when the underlying CC is clear. Moreover, we say that a k-cochain space
C k (X ) is defined on X . Intuitively, a k-cochain can be interpreted as a signal defined on the k-cells of X [119].
Figure 16(a) shows an example of a cochain supported on 0, 1, and 2-rank cells of a simplicial complex.
When X is a graph, 0-cochains correspond to graph signals [204]. Ordering the cells in X k , we canon-
                                                                     k
ically identify C k (X , Rd ) with the Euclidean vector space R|X |×d and explicitly write Hk as the vector
[hxk1 , . . . , hxk ], where hxkj ∈ R is a feature vector associated with the cell xkj . The notation Hk,j refers to
                                     d
              |X k |

the feature vector hxkj to avoid explicit reference to cell xkj . We also work with maps between cochain spaces,
which we call cochain maps.

      Definition 22 (Cochain maps). For r < k, an incidence matrix Br,k induces a map

                                                Br,k : C k (X ) → C r (X ),
                                                          Hk → Br,k (Hk ),

      where Br,k (Hk ) denotes the usual product Br,k Hk of matrix Br,k with vector Hk . Similarly, an
      (r, k)-adjacency matrix Ar,k induces a map

                                                Ar,k : C r (X ) → C r (X ),
                                                          Hr → Ar,k (Hr ).

      These two types of maps between cochain spaces are called cochain maps.

Cochain maps serve as operators that ‘shuffle’ and ‘redistribute’ the data on the underlying CC. In the context
of TDL, cochain maps are the main tools for defining higher-order message passing (Section 6.1) as well as
(un)pooling operations (Section 7.1). Each adjacency matrix Ar,k or coadjacency matrix coAr,k defines a


                                                             24
5   Combinatorial complex neural networks (CCNNs)                                     Topological deep learning




Figure 16: Examples of k-cochains (left) and cochain maps (right) supported on a CC of dimension four.
Left: a k-cochain can be interpreted as a signal or a feature vector defined on the k-cells. In the figure, a
3-dimensional cochain is attached to the vertices, 2-dimensional cochains are attached to the 1-cells, and
4-dimensional cochains to the 2-cells. Right: each of coAr,k and Ar,k defines a cochain map between cochain
spaces of equal dimension, whereas each Br,k defines a cochain map between different dimensions.

cochain map between cochain spaces of equal dimension, while each incidence matrix Br,k defines a cochain
map between different dimensions. Figure 16(b) shows examples of cochain maps on a CC of dimension 4.


5    Combinatorial complex neural networks (CCNNs)

The modelling flexibility of CCs enables the exploration and analysis of a wide spectrum of CC-based neural
network architectures. A CC-based neural network can exploit all neighborhood matrices or a subset of them,
thus accounting for multi-way interactions among various cells in the CC to solve a learning task. In this
section, we introduce the blueprint for TDL by developing the general principles of CC-based TDL models.
We utilize our TDL blueprint framework for examining current approaches and offer directives for designing
novel models.
The learning tasks in TDL can be broadly classified into three categories: cell classification, complex
classification, and cell prediction. Our numerical experiments in Section 9 provide examples on cell and
complex classification. In more detail, the learning tasks of the three categories are the following:
      • Cell classification: the goal is to predict targets for each cell in a complex. To accomplish this, we
        can utilize a TDL classifier that takes into account the topological neighbors of the target cell and
        their associated features. An example of cell classification is triangular mesh segmentation, in which
        the task is to predict the class of each face or edge in a given mesh.
      • Complex classification: the aim is to predict targets for an entire complex. To achieve this, we can
        reduce the topology of the complex into a common representation using higher-order cells, such
        as pooling, and then learn a TDL classifier over the resulting flat vector. An example of complex
        classification is class prediction for each input mesh.
      • Cell prediction: the objective is to predict properties of cell-cell interactions in a complex, and, in
        some cases, to predict whether a cell exists in the complex. This can be achieved by utilizing the
        topology and associated features of the cells. A relevant example is the prediction of linkages among
        entities in hyperedges of a hypergraph.
Figure 17 outlines our general setup for TDL. Initially, a higher-order domain, represented by a CC, is
constructed on a set S. A set of neighborhood functions defined on the domain is then selected. The
neighborhood functions are usually selected based on the learning problem at hand and they are used to
build a topological neural network. To develop our general TDL framework, we introduce combinatorial
complex neural networks (CCNNs), an abstract class of neural networks supported on CCs that effectively
captures the pipeline of Figure 17. CCNNs can be thought of as a template that generalizes many popular
architectures, such as convolutional and attention-based neural networks. The abstraction of CCNNs offers
many advantages. First, any result that holds for CCNNs is immediately applicable to any particular instance


                                                      25
Topological deep learning                                         5   Combinatorial complex neural networks (CCNNs)




            (a)                    (b)                             (c)                            (d)

Figure 17: A TDL blueprint. (a): A set of abstract entities. (b): A CC (S, X , rk) is defined on S. (c): For
an element x ∈ X , we select a collection of neighborhood functions defined on the CC. (d): We build a
neural network on the CC using the neighborhood functions selected in (c). The neural network exploits the
neighborhood functions selected in (c) to update the data supported on x.


of CCNN architecture. Indeed, the theoretical analysis and results in this paper are applicable to any
CC-based neural network as long as it satisfies the CCNN definition. Second, working with a particular
parametrization might be cumbersome if the neural network has a complicated architecture. In Section 5.1,
we elaborate on the intricate architectures of parameterized TDL models. The more abstract high-level
representation of CCNNs simplifies the notation and the general purpose of the learning process, thereby
making TDL modelling more intuitive to handle.


      Definition 23 (CCNN). Let X be a CC. Let C i1 × C i2 × . . . × C im and C j1 × C j2 × . . . × C jn be a
      Cartesian product of m and n cochain spaces defined on X . A combinatorial complex neural
      network (CCNN) is a function of the form

                             CCNN : C i1 × C i2 × . . . × C im −→ C j1 × C j2 × . . . × C jn .

Intuitively, a CCNN takes a vector of cochains (Hi1 , . . . , Him ) as input and returns a vector of cochains
(Kj1 , . . . , Kjn ) as output. In Section 5.1, we show how neighborhood functions play a central role in the
construction of a general CCNN. Definition 23 does not show how a CCNN can be computed in general.
Sections 6 and 7 formalize the computational workflow in CCNNs.

5.1     Building CCNNs: tensor diagrams

Unlike graphs that involve vertex or edge signals, higher-order networks entail a higher number of signals (see
Figure 16). Thus, constructing a CCNN requires building a non-trivial amount of interacting sub-networks.
Due to the relatively large number of cochains in a CCNN, we introduce tensor diagrams, a diagrammatic
notation for higher-order networks.
Remark 5.1. Diagrammatic notation is common in the geometric topology literature [133, 255], and it is
typically used to construct functions built from simpler building blocks. See Appendix C for further discussion.
See also [221] for related constructions on simplicial neural networks.


      Definition 24 (Tensor diagram). A tensor diagram represents a CCNN via a directed graph. The
      signal on a tensor diagram flows from the source nodes to the target nodes. The source and target
      nodes correspond to the domain and codomain of the CCNN.

Figure 18 depicts an example of a tensor diagram. On the left, a CC of dimension three is shown. Consider a
0-cochain C 0 , a 1-cochain C 1 and a 2-cochain C 2 . The middle figure displays a CCNN that maps a cochain
vector in C 0 × C 1 × C 2 to a cochain vector in C 0 × C 1 × C 2 . On the right, a tensor diagram representation of the
CCNN is shown. We label each edge on the tensor diagram by a cochain map or by its matrix representation.


                                                            26
5   Combinatorial complex neural networks (CCNNs)                                              Topological deep learning




          Rank-0 cells
          Rank-1 cells

          Rank-2 cells




Figure 18: A tensor diagram is a diagrammatic representation of a CCNN that captures the flow of signals
on the CCNN.




Figure 19: Examples of tensor diagrams. (a): Tensor diagram of a CCNNcoA1,1 : C 1 → C 1 . (b): Tensor
diagram of a CCNN{B1,2 ,B1,2
                           T }: C × C
                                 1    2
                                        → C 1 × C 2 . (c): A merge node that merges three cochains. (d): A
tensor diagram generated by vertical concatenation of the tensor diagrams in (c) and (b). The edge label Id
denotes the identity matrix.


The edge labels on the tensor diagram of Figure 18 are A0,1 , B0,1
                                                               T
                                                                   , A1,1 , B1,2 and coA2,1 . Thus, the tensor
diagram specifies the flow of cochains on the CC.
The labels on the arrows of a tensor diagram form a sequence G = (Gi )li=1 of cochain maps defined on
the underlying CC. In Figure 18 for example, G = (Gi )5i=1 = (A0,1 , B0,1
                                                                      T
                                                                          , A1,1 , B1,2 , coA2,1 ). When a tensor
diagram is used to represent a CCNN, we use the notation CCNNG for the tensor diagram and for its
corresponding CCNN. The cochain maps (Gi )li=1 reflect the structure of the CC and are used to determine
the flow of signals on the CC. Any of the neighborhood matrices mentioned in Section 4.4 can be used as
cochain maps. The choice of cochain maps depends on the learning task.
Figure 19 visualizes additional examples of tensor diagrams. The height of a tensor diagram is the number of
edges on a longest path from a source node to a target node. For instance, the heights of the tensor diagrams
in Figures 19(a) and 19(d) are one and two, respectively. The vertical concatenation of two tensor diagrams
represents the composition of their corresponding CCNNs. For example, the tensor diagram in Figure 19(d)
is the vertical concatenation of the tensor diagrams in Figures 19(c) and (b).
If a node in a tensor diagram receives one or more signals, we call it a merge node. Mathematically, a merge
node is a function MG1 ,...,Gm : C i1 × C i2 × . . . × C im → C j given by
                                                M
                             (Hi1 , . . . , Him ) −−→ Kj = MG1 ,...,Gm (Hi1 , . . . , Him ),                         (1)

where Gk : C ik (X ) → C j (X ), k = 1, . . . , m, are cochain maps. We think of M as a message-passing function
that takes into account the messages outputted by maps G1 , . . . , Gm , which collectively act on a cochain vector
(Hi1 , . . . , Him ), to obtain an updated cochain Kj . See Sections 5.2 and 6.2 for more details. Figure 19(c)
shows a merge node example.


                                                           27
Topological deep learning                                               5        Combinatorial complex neural networks (CCNNs)




                                              (a)                                                          (b)


Figure 20: Examples of push-forward operators. (a): Let G1 : C 1 → C 2 be a cochain map. A push-forward FG1
induced by G1 takes as input a 1-cochain H1 defined on the edges of the underlying CC X and ‘pushes-forward’
this cochain to a 2-cochain K2 defined on X 2 . The cochain K2 is formed by aggregating the information in
H1 using the neighborhood function NGT1 . In this case, the neighbors of the 2-rank (blue) cell with respect to
G1 are the four (pink) edges on the boundary of this cell. (b): Similarly, G2 : C 0 → C 2 induces a push-forward
map FG2 : C 0 → C 2 that sends a 0-cochain H0 to a 2-cochain K2 . The cochain K2 is defined by aggregating
the information in H0 using the neighborhood function NGT2 .



5.2     Push-forward operator and merge node

We introduce the push-forward operation, a computational scheme that enables sending a cochain supported
on i-cells to j-cells. The push-forward operation is a computational building block used to formalize the
definition of the merge nodes given in Equation 1, the higher-order message passing introduced in Section 6,
and the (un)pooling operations introduced in Section 7.


      Definition 25 (Cochain push-forward). Consider a CC X , a cochain map G : C i (X ) → C j (X ), and a
      cochain Hi in C i (X ). A (cochain) push-forward induced by G is an operator FG : C i (X ) → C j (X )
      defined via
                                     Hi → Kj = [kyj , . . . , kyj ] = FG (Hi ),                         (2)
                                                       1                |X j |


      such that for k = 1, . . . , |X |,
                                     j
                                                           M
                                             kyj =                      αG (hxi ),                                     (3)
                                                k                                   l
                                                     xil ∈N        j
                                                              GT (y )
                                                                   k
              L
      where       is a permutation-invariant aggregation function and αG is a differentiable function.


The operator FG pushes forward an i-cochain Hi supported on X i to a j-cochain FG (Hi ) supported on X j .
For every cell y ∈ X j , Equation 3 constructs the vector ky by aggregating all vectors hx attached to the
neighbors x ∈ X i of y with respect to the neighborhood function NGT , and by then applying a differentiable
function αG on the set of aggregated vectors {hx |x ∈ NGT (y)}.
Figure 20 visualizes two examples of push-forward operators. Example 5.2 provides a push-forward function
induced by an indicence matrix. The push-forward function in Example 5.2 does not contain any parameters,
therefore it is not trainable. In Section 5.4, we give examples of parameterized push-forward operations,
whose parameters can be learnt.
Example 5.2. Consider a CC X of dimension 2. Let B0,2 : C 2 (X ) → C 0 (X ) be an incidence matrix. The
function FBm
            0,2
                : C 2 (X ) → C 0 (X ) defined by FB
                                                  m
                                                   0,2
                                                       (H2 ) = B0,2 (H2 ) is a push-forward induced by B0,2 . FB
                                                                                                               m
                                                                                                                 0,2

pushes forward the cochain H2 ∈ C 2 to cochain B0,2 (H2 ) ∈ C 0 .

In Definition 26, we formulate the notion of merge node using push-forward operators. Figure 21 visualizes
Definition 26 of merge node via a tensor diagram.


                                                              28
5   Combinatorial complex neural networks (CCNNs)                                            Topological deep learning




                       Figure 21: A tensor diagram depicting Definition 26 of merge node.



      Definition 26 (Merge node). Let X be a CC. Moreover, let G1 : C i1 (X ) → C j (X ) and G2 : C i2 (X ) →
      C j (X ) be two cochain maps. Given a cochain vector (Hi1 , Hi2 ) ∈ C i1 × C i2 , a merge node
      MG1 ,G2 : C i1 × C i2 → C j is defined as
                                                                     O           
                                  MG1 ,G2 (Hi1 , Hi2 ) = β FG1 (Hi1 )   FG2 (Hi2 ) ,                       (4)

              : C × C j → C j is an aggregation function, FG1 and FG2 are push-forward operators induced
            N j
      where
      by G1 and G2 , and β is an activation function.


5.3     Three tensor operations for the construction of CCNNs

Any tensor diagram representation of a CCNN can be built from two elementary operations: the push-forward
operator and the merge node. In practice, it is convenient to introduce other operations that facilitate
building involved neural network architectures more effectively. For example, one useful operation is the dual
operation of the merge node, which we call the split node.

      Definition 27 (Split node). Let X be a CC. Moreover, let G1 : C j (X ) → C i1 (X ) and G2 : C j (X ) →
      C i2 (X ) be two cochain maps. Given a cochain Hj ∈ C j , a split node SG1 ,G2 : C j → C i1 × C i2 is defined
      as
                                    SG1 ,G2 (Hj ) = (β1 (FG1 (Hj )), β2 (FG2 (Hj ))) ,                           (5)
      where FGi is a push-forward operator induced by Gi , and βi is an activation function for i = 1, 2.

While it is clear from Definition 27 that split nodes are simply tuples of push-forward operations, using split
nodes allows us to build neural networks more effectively and intuitively. Definition 28 puts forward a set of
elementary tensor operations, including split nodes, to facilitate the formulation of CCNNs in terms of tensor
diagrams.

      Definition 28 (Elementary tensor operations). We refer collectively to push-forward operations,
      merge nodes and split nodes as elementary tensor operations.

Figure 22 displays tensor diagrams of elementary tensor operations, and Figure 23 exemplifies how existing
topological neural networks are expressed via tensor diagrams based on elementary tensor operations. For
example, the simplicial complex net (SCoNe), a Hodge decomposition-based neural network proposed by [221],
can be effectively realized in terms of split and merge nodes, as shown in Figure 23(a).


                                                           29
Topological deep learning                                   5   Combinatorial complex neural networks (CCNNs)




                    Push-forward    Merge Node                   Split Node




Figure 22: Tensor diagrams of the elementary tensor operations, namely of push-forward operations, merge
nodes and split nodes. These three elementary tensor operations are building blocks for constructing tensor
diagrams of general CCNNs. A general tensor diagram can be formed using compositions and horizontal
concatenations of the three elementary tensor operations. (a): A tensor diagram of a push-forward operation
induced by a cochain map G : C i → C j . (b): A merge node induced by two cochain maps G1 : C i1 → C j and
G2 : C i2 → C j . (c): A split node induced by two cochain maps G1 : C j → C i1 and G2 : C j → C i2 . In this
illustration, the function ∆ : C j → C j × C j is defined as ∆(Hj ) = (Hj , Hj ).




Figure 23: Examples of existing neural networks that can be realized in terms of the elementary operations
given in Figure 22. Edge labels are dropped to simplify exposition. (a): The simplicial complex net (SCoNe),
proposed by [221], can be realized as a composition of a split node that splits an input 1-cochain to three
cochains of dimensions zero, one, and two, followed by a merge node that merges these cochains into a 1-
cochain. (b): The simplicial neural network (SCN), proposed by [91], can be realized in terms of push-forward
operations. (c)–(e): Examples of cell complex neural networks (CXNs); see [123]. Note that (e) can be
realized in terms of a single merge node that merges the 0 and 2-cochains to a 1-cochain as well as a single
split node that splits the 1-cochain to 0- and 2-cochains.


Remark 5.3. The elementary tensor operations constitute the only framework needed to define any pa-
rameterized topological neural network. Indeed, since tensor diagrams can be built via the three elementary
tensor operations, then it suffices to define the push-forward and the merge operators in order to fully define
a parameterized class of CCNNs (recall that the split node is completely determined by the push-forward
operator). In Sections 5.4 and 5.5, we build two parameterized classes of CCNNs: the convolutional and
attention classes. In both cases, we only define their corresponding parameterized elementary tensor operations.
Beyond convolutional and attention versions of CCNNs, the three elementary tensor operations allow us to
build arbitrary parameterized tensor diagrams, therefore providing scope to discover novel topological neural
network architectures on which our theory remains applicable.

An alternative way of constructing CCNNs draws ideas from topological quantum field theory (TQFT). In
Appendix C, we briefly discuss this relationship in more depth.


                                                      30
5   Combinatorial complex neural networks (CCNNs)                                        Topological deep learning




5.4     Combinatorial complex convolutional networks (CCCNNs)

One of the fundamental computational requirements for deep learning on higher-order domains is the ability
to define and compute convolutional operations. Here, we introduce CCNNs equipped with convolutional
operators, which we call combinatorial complex convolutional neural networks (CCCNNs). In particular,
we put forward two convolutional operators for CCCNNs: CC-convolutional push-forward operators and
CC-convolutional merge nodes.
We demonstrate how CCCNNs can be introduced from the two basic blocks: the push-forward and the
merge operations which have been defined abstractly in Section 5.2. In its simplest form, a CC-convolutional
push-forward, as conceived in Definition 29, is a generalization of the convolutional graph neural network
introduced in [150].

      Definition 29 (CC-convolutional push-forward). Consider a CC X , a cochain map G : C i (X ) →
      C j (X ), and a cochain Hi ∈ C i (X , Rsin ). A CC-convolutional push-forward is a cochain map
         conv
      FG;W    : C i (X , Rsin ) → C j (X , Rtout ) defined as

                                                Hi → Kj = GHi W,                                           (6)

      where W ∈ Rdsin ×dsout are trainable parameters.

Having defined the CC-convolutional push-forward, the CC-convolutional merge node (Definition 30) is a
straightforward application of Definition 26. Variants of Definition 30 have appeared in recent works on
higher-order networks [53, 234, 235, 221, 123, 222, 91, 127, 55, 272].

      Definition 30 (CC-convolutional merge node). Let a X be a CC. Moreover, let G1 : C i1 (X ) → C j (X )
      and G2 : C i2 (X ) → C j (X ) be two cochain maps. Given a cochain vector (Hi1 , Hi2 ) ∈ C i1 × C i2 , a
      CC-convolutional merge node Mconv       G;W : C × C
                                                     i1   i2
                                                             → C j is defined as

                                G;W (Hi1 , Hi2 ) = β FG1 ;W1 (Hi1 ) + FG2 ;W2 (Hi2 )
                               Mconv                  conv             conv
                                                                                    
                                                                                                           (7)
                                                 = β(G1 Hi1 W1 + G2 Hi2 W2 ),

      where G = (G1 , G2 ), W = (W1 , W2 ) is a tuple of trainable parameters, and β is an activation function.

In practice, the matrix representation of the cochain map G in Definition 29 might require problem-specific
normalization during training. For various types of normalization in the context of higher-order convolutional
operators, we refer the reader to [150, 53, 234].
We have used the notation CCNNG for a tensor diagram and its corresponding CCNN. Our notation indicates
that the CCNN is composed of elementary tensor operations based on a sequence G = (Gi )li=1 of cochain
maps defined on the underlying CC. When the elementary tensor operations that make up the CCNN are
parameterized by a sequence W = (Wi )ki=1 of trainable parameters, we denote the CCNN and its tensor
diagram representation by CCNNG;W .

5.5     Combinatorial complex attention neural networks (CCANNs)

The majority of higher-order deep learning models focus on layers that use isotropic aggregation, which means
that neighbors in the vicinity of an element contribute equally to an update of the element’s representation.
As information is aggregated in a diffusive manner, such isotropic aggregation can limit the expressiveness
of these learning models, leading to phenomena such as oversmoothing [31]. In contrast, attention-based
learning [71] allows deep learning models to assign a probability distribution to neighbors in the local
vicinity of elements in the underlying domain, thus highlighting components with the most task-relevant
information [258]. Attention-based models are successful in practice as they ignore noise in the domain,
thereby improving the signal-to-noise ratio [168, 195]. Accordingly, attention-based models have achieved


                                                         31
Topological deep learning                                         5   Combinatorial complex neural networks (CCNNs)




remarkable success on traditional machine learning tasks on graphs, including node classification and link
prediction [172], node ranking [248], and attention-based embeddings [71, 167].
After introducing the CC-convolutional push-forward in Section 5.4, our second example of a push-forward
operation is the CC-attention push-forward, which is introduced in the present section. Thus, it becomes
possible to use CCNNs equipped with CC-attention push-forward operators, which we call combinatorial
complex attention neural networks (CCANNs). We first provide the general notion of attention of a sub-CC
Y0 of a CC with respect to other sub-CCs in the CC.




   Definition 31 (Higher-order attention). Let X be a CC, N a neighborhood function defined on X ,
   and Y0 a sub-CC of X . Let N (Y0 ) = {Y1 , . . . , Y|N (Y0 )| } be a set of sub-CCs that are in the vicinity of
   Y0 with respect to the neighborhood function N . A higher-order attention of Y0 with respect to N
   is a function a : Y0 × N (Y0 ) → [0, 1] that assigns a weight a(Y0 , Yi ) to each element Yi ∈ N (Y0 ) such
        P|N (Y )|
   that i=1 0 a(Y0 , Yi ) = 1.




As seen from Definition 31, a higher-order attention of a sub-CC Y0 with respect to a neighborhood function
N assigns a discrete distribution to the neighbors of Y0 . Attention-based learning typically aims to learn
the function a. Observe that the function a relies on the neighborhood function N . In our context, we aim
to learn the function a whose neighborhood function is either an incidence or a (co)adjacency function, as
introduced in Section 4.4.
Recall from Definition 31 that a weight a(Y0 , Yi ) requires both a source sub-CC Y0 and a target sub-CC Yi as
inputs. Thus, a CC-attention push-forward operation requires two cochain spaces. Definition 32 introduces a
notion of CC-attention push-forward in which the two underlying cochain spaces contain cochains supported
on cells of equal rank.




   Definition 32 (CC-attention push-forward for cells of equal rank). Let G : C s (X ) → C s (X )
   be a neighborhood matrix. A CC-attention push-forward induced by G is a cochain map
   FGatt
         : C s (X , Rdsin ) → C s (X , Rdsout ) defined as

                                          Hs → Ks = (G ⊙ att)Hs Ws ,                                          (8)

   where ⊙ is the Hadamard product, Ws ∈ Rdsin ×dsout are trainable parameters, and att : C s (X ) → C s (X )
   is a higher-order attention matrix that has the same dimension as matrix G. The (i, j)-th entry
   of matrix att is defined as
                                                         eij
                                        att(i, j) = P            ,                                        (9)
                                                               k∈NG (i)eik

   where eij = ϕ(a [Ws Hs,i ||Ws Hs,j ]), a ∈ R
                    T
                                                     is a trainable vector, [a||b] denotes the concatenation
                                                 2×sout

   of a and b, ϕ is an activation function, and NG (i) is the neighborhood of cell i with respect to matrix
   G.




Definition 33 treats a more general case than Definition 32. Specifically, Definition 33 introduces a notion of
CC-attention push-forward in which the two underlying cochain spaces contain cochains supported on cells of
different ranks.


                                                          32
5   Combinatorial complex neural networks (CCNNs)                                               Topological deep learning




    Definition 33 (CC-attention push-forward for cells of unequal ranks). For s ̸= t, let G : C s (X ) →
    C t (X ) be a neighborhood matrix. A CC-attention block induced by G is a cochain map
    FGatt
          A : C s (X , Rdsin ) × C t (X , Rdtin ) → C t (X , Rdtout ) × C s (X , Rdsout ) defined as

                                                  (Hs , Ht ) → (Kt , Ks ),                                       (10)

    with
                                 Kt = (G ⊙ atts→t )Hs Ws , Ks = (GT ⊙ attt→s )Ht Wt ,                            (11)
    where Ws ∈ Rdsin ×dtout , Wt ∈ Rdtin ×dsout are trainable parameters, and attks→t : C s (X ) →
    C t (X ), attkt→s : C t (X ) → C s (X ) are higher-order attention matrices that have the same di-
    mensions as matrices G and GT , respectively. The (i, j)-th entries of matrices atts→t and attt→s are
    defined as
                                                  eij                     fij
                                 (atts→t )ij = P        , (attt→s )ij = P         ,                  (12)
                                                 k∈NG (i)eik                    k∈NGT (i)fik

    with
                          eij = ϕ((a)T [Ws Hs,i ||Wt Ht,j ]), fij = ϕ(rev(a)T [Wt Ht,i ||Ws Hs,j ]),             (13)
    where a ∈ Rtout +sout is a trainable vector, and rev(a) = [al [: tout ]||al [tout :]].


The incidence matrices of Definition 16 can be employed as neighborhood matrices in Definition 33. In
Figure 24, we illustrate the notion of CC-attention for incidence neighborhood matrices. Figure 24(c) shows
the non-squared incidence matrix B1,2 associated with the CC displayed in Figure 24(a). The attention block
HBB1,2 learns two incidence matrices atts→t and attt→s . The matrix atts→t has the same shape as B1,2 ,
and non-zero elements exactly where B1,2 has elements equal to one. Each column i in atts→t represents a
probability distribution that defines the attention of the i-th 2-cell to its incident 1-cells. The matrix attt→s
has the same shape as B1,2T
                            , and similarly represents the attention of 1-cells to 2-cells.



                      3

                                       4
            1


                                2




Figure 24: Illustration of notion of CC-attention for cells of unequal ranks. (a): A CC. Each 2-cell (blue face)
of the CC attends to its incident 1-cells (pink edges). (b): The attention weights reside on a graph constructed
from the cells and their incidence relations; see Section 8.1 for details. (c): Incidence matrix B1,2 of the
CC given in (a). The non-zero elements in column [1, 2, 3] correspond to the neighborhood NB1,2 ([1, 2, 3]) of
[1, 2, 3] with respect to B1,2 .


Remark 5.4. The computation of a push-forward cochain Kt requires two cochains, namely Hs and Ht .
While Kt depends on Hs directly in Equation 11, it depends on Ht only indirectly via atts→t as seen from
Equations 11–13. Moreover, the cochain Ht is only needed for atts→t during training, and not during
inference. In other words, the computation of a CC-attention push-forward block requires only a single cochain,
Hs , during inference, in agreement with the computation of a general cochain push-forward as specified in
Definition 25.

The operators G ⊙ att and G ⊙ atts→t in Equations 8 and 11 can be viewed as learnt attention versions of G.
This perspective allows to employ CCANNs to learn arbitrary types of discrete exterior calculus operators, as
elaborated in Appendix D.


                                                               33
Topological deep learning                                                      6   Higher-order message passing




Figure 25: An illustration of higher-order message passing (Definition 34). Left-hand side: a collection
of neighborhood functions N1 , . . . , Nk are selected. The selection typically depends on the learningLtask.
Right-hand side: for each Nk , the N   messages are aggregated using an intra-neighborhood (function) Nk .
The inter-neighborhood (function)        aggregates the final messages obtained from all neighborhoods.


6      Higher-order message passing

In this section, we explain the relation between the notion of the merge node introduced in Section 5.2 and
higher-order message passing. In particular, we prove that higher-order message passing on CCs can be
realized in terms of the elementary tensor operations introduced in Section 5.3. Further, we demonstrate the
connection between CCANNs (Section 5.5) and higher-order message passing, and introduce an attention
version of higher-order message passing. We first define higher-order message passing on CCs, generalizing
notions introduced in [123].
We remark that many of the constructions discussed here are presented in their most basic form, but can be
extended further. An important aspect in this direction is the construction of message-passing protocols that
are invariant or equivariant with respect to the action of a specific group.

6.1     Definition of higher-order message passing

Higher-order message passing refers to a computational framework that involves exchanging messages among
entities and cells in a higher-order domain using a set of neighborhood functions. In Definition 34, we
formalize the notion of higher-order message passing for CCs. Figure 25 illustrates Definition 34.


      Definition 34 (Higher-order message passing on a CC). Let X be a CC. Let N = {N1 , . . . , Nn }
      be a set of neighborhood functions defined on X . Let x be a cell and y ∈ Nk (x) for some Nk ∈ N .
      A message mx,y between cells x and y is a computation that depends on these two cells or on the
                                                                                                (l)
      data supported on them. Denote by N (x) the multi-set {{N1 (x), . . . , Nn (x)}}, and by hx some data
      supported on the cell x at layer l. Higher-order message passing on X , induced by N , is defined
      via the following four update rules:

                                        mx,y = αNk (hx(l) , hy(l) ),                                   (14)
                                                M
                                         mkx =          mx,y , 1 ≤ k ≤ n,                              (15)
                                                 y∈Nk (x)
                                                   O
                                          mx =           mkx ,                                         (16)
                                                 Nk ∈N

                                       hx(l+1) = β(hx(l) , mx ).                                       (17)
            L                                                                                        N
      Here,    is a permutation-invariant aggregation function called the intra-neighborhood of x,       is
      an aggregation function called the inter-neighborhood of x, and αNk , β are differentiable functions.



                                                         34
6   Higher-order message passing                                                                 Topological deep learning




Some remarks on Definition 34 are as follows. First, the message mx,y in Equation 14 does not depend only
               (l)   (l)
on the data hx , hy supported on the cells x, y; it also depends on the cells themselves. For instance, if X is
a cell complex, the orientation of both x and y factors into the computation of message mx,y . Alternatively,
x ∪ y or x ∩ y might be cells in X and it might be useful to include their data in the computation of message
mx,y . This unique characteristic only manifests in higher-order domains, and does not occur in graphs-based
message-passing frameworks [50, 109]3 . Second, higher-order message passing relies on the choice of a set N
of neighborhood functions. This is also a unique characteristic that only occurs in a higher-order domain,
where a neighborhood function is necessarily described by a set of neighborhood relations rather than graph
adjacency as in graph-based message passing. Third, in Equation 14, since y is implicitly defined with
respect to a neighborhood  Nrelation Nk ∈ N , the function αNk and the message mx,y depend on Nk . Fourth,
the inter-neighborhood        does not necessarily have to be a permutation-invariant aggregation function.
For instance, it is possible to set an order on the multi-set N (x) and compute mx with respect to this
order. Finally, higher-order message passing relies on two aggregation functions, the intra-neighborhood and
inter-neighborhood, whereas graph-based message passing relies on a single aggregation function. The choice
of set N , as illustrated in Section 4, enables the use of a variety of neighborhood functions in higher-order
message passing.
Remark 6.1. The push-forward operator given in Definition 25 is related to the update rule of Equation 14.
                                                      (l)           (l)       (l)   (l)           (l)
On one hand, Equation 14 requires two cochains Xi = [hxi , . . . , hxi ] and Yj = [hyj , . . . , hyj ] to compute
                                                                   1        |X i |               1       |X j |
 (l+1)     (l+1)         (l+1)
Xi     = [hxi , . . . , hxi ], so signals on both C j and C i must be present in order to execute Equation 14.
             1              |X i |
From this perspective, it is natural and customary to think about this operation as an update rule. On the
other hand, the push-forward operator of Definition 25 computes a cochain Kj ∈ C j given a cochain Hi ∈ C i .
As a single cochain Hi is required to perform this computation, it is natural to think about Equation 3 as a
function. See Section 6.3 for more details.

The higher-order message-passing framework given in Definition 34 can be used to construct novel neural
network architectures on a CC, as we have also alluded in Figure 17. First, a CC X and cochains Hi1 . . . , Him
supported on X are given. Second, a collection of neighborhood functions are chosen, taking into account the
desired learning task. Third, the update rules of Definition 34 are executed on the input cochains Hi1 . . . , Him
using the chosen neighborhood functions. The second and the third steps are repeated to obtain the final
computations.

      Definition 35 (Higher-order message-passing neural network). We refer to to any neural network
      constructed using Definition 34 as a higher-order message-passing neural network.


6.2     Higher-order message-passing neural networks are CCNNs

In this section, we show that higher-order message-passing computations can be realized in terms of merge
node computations, and therefore that higher-order message-passing neural networks are CCNNs. As a
consequence, higher-order message passing unifies message passing on simplicial complexes, cell complexes
and hypergraphs through a coherent set of update rules and, alternatively, through the expressive language
of tensor diagrams.
Theorem 6.2. The higher-order message-passing computations of Definition 34 can be realized in terms of
merge node computations.

Proof. Let X be a CC. Let N = {N1 , . . . , Nn } be a set of neighborhood functions as specified in Definition 34.
Let Gk be the matrix induced by the neighborhood function Nk . We assume that the cell x given in
Definition 34 is a j-cell and the neighbors y ∈ Nk (x) are ik -cells. We will show that Equations 14–17 can
be realized as applications of merge nodes. In what follows, we define the neighborhood function to be
NId (x) = {x} for x ∈ X . Moreover, we denote the associated neighborhood matrix of NId by Id : C j → C j ,
as it is the identity matrix.
    3 The message ‘direction’ in m
                                  x,y is from y to x. In general, mx,y and my,x are not equal.



                                                              35
Topological deep learning                                                                                      6        Higher-order message passing




Computing message mx,y of Equation 14 involves two cochains:
                                   (l)         (l)           (l)             (l)       (l)           (l)
                                 Xj = [hxj , . . . , hxj               ], Yik = [h ik , . . . , h ik               ].
                                                   1          |X j |
                                                                                       y1            y
                                                                                                         |X ik |



Every message mxj ,yik corresponds to the entry [Gk ]st of matrix Gk . In other words, there is a one-to-one
                 t  s
correspondence between non-zero entries of matrix Gk and messages mxj ,yik .
                                                                                                     t     s


It follows from Section 5.2 that computing {mkx }nk=1 corresponds to a merge node MIdj ,Gk : C j × C ik → C j
that performs the computations determined via αk and                               , and yields
                                                         L

                                                                                             (l)   (l)
                                mjk = [mkxj , . . . , mkxj            ] = MIdj ,Gk (Xj , Yik ) ∈ C j .
                                               1             |X j |



At this stage, we have n j-cochains {mjk }nk=1 . Equations 16 and 17 merge these cochains with the input
             (l)
j-cochain Xj . Specifically, computing mx in Equation 16 corresponds to n − 1 applications of merge nodes of
the form MIdk ,Idk : C j × C j → C j on the cochains {mjk }nk=1 . Explicitly, we first merge mj1 and mj2 to obtain
nj1 = MIdj ,Idj (mj1 , mj2 ). Next, we merge the j-cochain nj1 with the j-cochain mj3 , and so on. The final merge
node in this stage performs the merge njn−1 = MIdj ,Idj (njn−2 , mjn ), which is mj = [mxj , . . . , mxj ]4 . Finally,
                                                                                                                            1        |X j |
           (l+1)                                                 (l)
computing Xj     is realized by a merge node M(Idj ,Idj ) (mj , Xj ) whose computations are determined by
function β of Equation 17.

Theorem 6.2 shows that higher-order message-passing networks defined on CCs can be constructed from the
elementary tensor operations, and hence they are special cases of CCNNs. We state this result formally in
Theorem 6.3.
Theorem 6.3. A higher-order message-passing neural network is a CCNN.

Proof. The conclusion follows immediately from Definition 35 and Theorem 6.2.

It follows from Theorem 6.3 that higher-order message-passing neural networks defined on higher-order
domains that are less general than CCs (such as simplicial complexes, cell complexes and hypergraphs)
are also special cases of CCNNs. Thus, tensor diagrams, as introduced in Definition 24, form a general
diagrammatic method for expressing neural networks defined on commonly studied higher-order domains.
Theorem 6.4. Message-passing neural networks defined on simplicial complexes, cell complexes or hypergraphs
can be expressed in terms of tensor diagrams and their computations can be realized in terms of the three
elementary tensor operators.

Proof. The conclusion follows from Theorem 6.3 and from the fact that simplicial complexes, cell complexes
and hypergraphs can be realized as special cases of CCs.

Theorems 6.3 and 6.4 put forward a unifying TDL framework based on tensor diagrams, thus providing
scope for future developments. For instance, [206] have already used our framework to express existing TDL
architectures for simplicial complexes, cell complexes and hypergraphs in terms of tensor diagrams.

6.3   Merge nodes and higher-order message passing: a qualitative comparison

Higher-order message passing, as given in Definition 34, provides an update rule to obtain the vector hxl+1
from vector hxl using a set of neighborhood vectors hyl determined by N (x). Clearly, this computational
                                         (l)           (l)
framework assumes that vectors hx and hy are provided as inputs. In other words, performing higher-order
                                                                 (l)
message passing according to Definition 34 requires the cochain Xj ∈ C j in the target domain as well as the
  4 Recall that while we use the same notation M
                                                Idk ,Idk for all merge nodes, these nodes have in general different parameters.



                                                                        36
6   Higher-order message passing                                                                Topological deep learning




Figure 26: The depicted neural network can be realized as a composition of two merge nodes. More specifically,
the input is a cochain vector (H0 , H2 ). The first merge node computes the 1-chain H1 = MB0,1   T ,B
                                                                                                     1,2
                                                                                                         (H0 , H2 ).
Similarly, the second merge node MB1,3 T ,B
                                            2,3
                                                : C × C → C computes the 3-cochain H3 = MB1,3
                                                   1   2   3
                                                                                                 T ,B
                                                                                                     2,3
                                                                                                         (H1 , H2 ).
The merge node perspective allows to compute a cochain supported on the 1-cells and 3-cells without having
initial cochains of these cells. On the other hand, the higher-order message-passing framework has two major
constraints: it assumes as input initial cochains supported on all cells of all dimensions of the domain, and it
updates all cochains supported on all cells of all dimensions of the input domain at every iteration.



             (l)                                                             (l+1)
cochains Yik ∈ C ik in order to compute the updated j-cochain Xj         . On the other hand, performing a
merge node computation requires a cochain vector (Hi1 , Hi2 ), as seen from Equation 1 and Definition 26.
The difference between these two computational frameworks might seem notational and the message passing
perspective might seem more intuitive, especially when working with graph-based models. However, we argue
that the merge node framework is more natural and flexible computationally in the presence of a custom
higher-order network architecture. To illustrate this, we consider the example visualized in Figure 26.
In Figure 26, the displayed neural network has a cochain input vector (H0 , H2 ) ∈ C 0 × C 2 . In the first layer,
the neural network computes the cochain H1 ∈ C 1 , while in the second layer it computes the cochain H3 ∈ C 3 .
To obtain cochain H1 in the first layer, we need to consider the neighborhood functions induced by B0,1   T
                                                                                                              and
B1,2 . However, if we employ Equations 14 and 15 to perform the computations determined by the first layer
of the tensor diagram in Figure 26, then we notice that no cochain is provided on C 1 as part of the input.
Hence, when applying Equations 14 and 15, a special treatment is required since the vectors hx1j have not
been computed yet. Note that such an artifact is not present in GNNs, since they often update node features,
which are typically provided as part of the input. To be specific, in GNNs, the first two arguments in the
update rule of Equation 14 are cochains that are supported on the 0-cells of the underlying graph.
Similarly, to compute the cochain H3 ∈ C 3 in the second layer of Figure 26, we must consider the neighborhood
functions induced by B1,3T
                            and B2,3 , and we must use the cochain vector (H1 , H2 ). This means that the
cochains H1 and H3 resulting from the computation of the neural network given in Figure 26 are not obtained
from an iterative process. Further, the input vectors H0 and H2 are never updated at any step of the
procedure. Finally, the cochains H1 and H3 are never updated. From the perspective of update rules such as
the ones appearing in the higher-order message passing framework (Definition 34), this setting is unnatural
in the sense that it assumes initial cochains supported on all cells of all dimensions as input, and in the sense
that it updates all cochains supported on all cells in the complex of the input domain at every iteration.
In practice, such difficulties in using the higher-order message passing framework can be overcome with ad
hoc engineering solutions based on turning on and off iterations on certain cochains or based on introducing
auxiliary cochains. The merge node is designed to overcome these limitations. Specifically, from the merge
node perspective, we can think of the first layer of Figure 26 as a function MB0,1         T ,B
                                                                                                 1,2
                                                                                                     : C 0 × C 1 → C 1 ; see
Equation 3. The function MB0,1       T ,B
                                          1,2
                                              takes as input the cochain vector (H0 , H2 ), and computes the 1-chain
H1 = MB0,1 T ,B
                1,2
                    (H  0 , H 2 ). Similarly,  we compute the 3-cochain H3 = MB1,3    T ,B
                                                                                           2,3
                                                                                               (H1 , H2 ) using a merge
node MB1,3T ,B
               2,3
                   : C ×C →C .
                      1       2      3



                                                            37
Topological deep learning                                              7   Push-forward and higher-order (un)pooling




6.4     Attention higher-order message passing and CCANNs

Here, we demonstrate the connection between higher-order message passing (Definition 34) and CCANNs
(Section 5.5). Initially, we introduce an attention version of Definition 34.


      Definition 36 (Attention higher-order message passing on a CC). Let X be a CC. Let N =
      {N1 , . . . , Nn } be a set of neighborhood functions defined on X . Let x be a cell and y ∈ Nk (x) for some
      Nk ∈ N . A message mx,y between cells x and y is a computation that depends on these two cells or
                                                                                                           (l)
      on the data supported on them. Denote by N (x) the multi-set {{N1 (x), . . . , Nn (x)}}, and by hx some
      data supported on the cell x at layer l. Attention higher-order message passing on X , induced
      by N , is defined via the following four update rules:

                                       mx,y = αNk (hx(l) , hy(l) ),                                          (18)
                                               M
                                        mkx =          ak (x, y)mx,y , 1 ≤ k ≤ n,                            (19)
                                                y∈Nk (x)
                                                  O
                                         mx =           bk mkx ,                                             (20)
                                                Nk ∈N

                                      hx(l+1) = β(hx(l) , mx ).                                              (21)

      Here, ak : {x} × Nk (x) → [0,P
                                   1] is a higher-order attention function (Definition 31), bk are trainable
                                     n
      attention weights satisfying k=1 b = 1,
                                          k
                                                 L                                                     N
                                                    is a permutation-invariant aggregation function,      is
      an aggregation function, αNk and β are differentiable functions.


Definition 36 distinguishes two types of attention weights. The first type is determined by the function ak .
The attention weight ak (x, y) of Equation 19 depends on the neighborhood function Nk and on cells x and y.
Further, ak (x, y) determines the attention a cell x pays to its surrounding neighbors y ∈ Nk , as determined
by the neighborhood function Nk . The CC-attention push-forward operations defined in Section 5.5 are a
particular parameterized realization of these weights. On the other hand, the weights bk of Equation 20
are only a function of the neighborhood Nk , and therefore determine the attention that cell x pays to the
information obtained from each neighborhood function Nk . In our CC-attention push-forward operations
given in Section 5.5, we set bk equal to one. However, the notion of merge node (Definition 26) can be easily
extended to introduce a corresponding notion of attention merge node, which in turn can be used to realize
Equation 20 in practice. Note that the attention determined by weights bk is unique to higher-order domains,
and does not arise in graph-based attention models.


7      Push-forward and higher-order (un)pooling

This section shows how the push-forward operation of Definition 25 can be used to realize (un)pooling
operations on CCs, and subsequently introduces (un)pooling operations for CCNNs. Further, this section
demonstrates how CC-based pooling provides a unifying framework for image and graph-based pooling, and
how shape-preserving pooling on CCs is related to the mapper on graphs.
In particular, we establish yet another unifying mathematical principle: pooling, as message passing, can be
fundamentally built from push-forward operations. Thus, push-forward operations form the main fundamental
building block from which all higher-order computations can be realized. This realization is important because
it establishes a mathematical foundation for a unifying deep learning application programming interface (API)
on complexes that combines pooling as well as message passing-based computations as a single operation.
Indeed, in TopoModelX5 , one of our contributed Python packages, higher-order message passing and the
pooling/unpooling operations are implemented as a single function across various topological domains.

    5 https://github.com/pyt-team/TopoModelX




                                                              38
7   Push-forward and higher-order (un)pooling                                           Topological deep learning




Figure 27: An example of successive CC-(un)pooling operations. A CC-pooling operation exploits the
hierarchical structure of the underlying CC to coarsen a lower-rank cochain by pushing it forward to higher-
order cells. CC-pooling operations improve invariance to certain distortions. Pink, blue, and green cells have
ranks one, two, and three, respectively. A CC X of dimension three is shown in (a). On X , we consider three
cochain operators: B0,1 : C 1 → C 0 , B1,2 : C 2 → C 1 and B2,3 : C 3 → C 2 . For the top row in Figures (b), (c),
and (d), we assume that we are initially given a 0-cochain H0 , while for the bottom row in the same figure,
we assume that we are initially given a 3-cochain H3 . For instance, at the top row of Figure (b), the input
0-cochain H0 gets pushed forward via the functional FB0,1   T : C
                                                                  0
                                                                     → C 1 to a 1-cochain H1 . This push-forward
operator induced by B0,1
                      T
                         is a CC-pooling operator, since it sends a 0-cochain to a cochain of higher rank. At
the bottom row of Figure (b), the push-forward operator FB0,1 : C 1 → C 0 induced by B0,1 is an unpooling
operator that sends a 1-cochain to a 0-cochain. Figures (c) and (d) are similar to (b), demonstrating the
pooling operators induced by B1,2
                                T
                                   and B2,3
                                         T
                                            (top), and the unpooling operators induced by B1,2 and B2,3
(bottom).


7.1     CC-(un)pooling

We define a CC-based pooling operation that extends the main characteristics of image-based and graph-based
pooling operations. Specifically, we build a pooling operation that ‘downscales’ the size of a signal supported
on a CC X . To this end, we exploit the hierarchical nature of CCs and define the pooling operation as a
push-forward operation induced by a cochain map G : C i (X ) → C j (X ) that pushes an i-cochain to a j-cochain.
To obtain a useful pooling operation that downscales the size of its input i-cochain, we impose the constraint
j > i. Definition 37 realizes our idea of CC-pooling. Figure 27 visualizes the intuition behind Definition 37. In
particular, Figure 27 shows an example of successive applications of pooling operations on cochains supported
on a CC of dimension three.


      Definition 37 (CC-pooling operation). Let X be a CC and G : C i (X ) → C j (X ) a cochain map. The
      push-forward operation induced by G is called a CC-pooling operation if j > i.


In Definition 38, we introduce an unpooling operation on CCs that pushes forward a cochain to a lower-rank
cochain. Figure 27 shows as example of unpooling on CCs.


      Definition 38 (CC-unpooling operation). Let X be a CC and G : C i (X ) → C j (X ) a cochain map.
      The push-forward operation induced by G is called a CC-unpooling operation if j < i.



7.2     Formulating common pooling operations as CC-pooling

In this section, we formulate common pooling operations in terms of CC-pooling. In particular, we demonstrate
that graph and image pooling can be cast as CC-pooling.


                                                       39
Topological deep learning                                           7   Push-forward and higher-order (un)pooling




Figure 28: Realizing image pooling in terms of CC-pooling. (a): An image of size 3 × 3. (b): The lattice
graph that corresponds to the image given in (a). (c): Augmenting the lattice graph with 2-cells. Choosing
these particular cells shown in (c) is equivalent to choosing the image-pooling window size to be 2 × 2 and
the pooling stride to be one. (d): Performing the image-pooling computation is equivalent to performing a
CC-pooling operation induced by the cochain map B0,2    T
                                                           : C 0 → C 2 , which pushes forward the image signal
(the 0-cochain supported on X ) to a signal supported on X 2 .
                               2



7.2.1   Graph pooling as CC-pooling

Here, we briefly demonstrate that the CC-pooling operation (Definition 37) is consistent with a graph-based
pooling algorithm. Let H0 be a cochain defined on the vertices and edges of a graph G. Moreover, let H′0
be a cochain defined on the vertices of a coarsened version G ′ of G. Under such a setup, H′0 represents a
coarsened version of H0 . A graph pooling function supported on the pair (G, H0 ) is a function of the form
POOL : (G, H0 ) → (G ′ , H′0 ) that sends every vertex in G to a vertex in G ′ , which corresponds to a cluster of
vertices in G. We now elucidate how the function POOL can be realized in terms of CC-pooling.
Proposition 7.1. The function POOL can be realized in terms of CC-pooling operations.

Proof. Each vertex in the graph G ′ represents a cluster of vertices in the original graph G. Using the
membership of these clusters, we construct a CC by augmenting G by a collection of 2-cells, so that each
of these cells corresponds to a supernode of G ′ . We denote the resulting CC structure by XG , consisting
of G augmented by the 2-cells. Hence, any 0-cochain H′0 defined on G ′ can be written as a 2-cochain
H2 ∈ C 2 (XG ). The relation between the vertices of the original graph G and the vertices of the pooled graph
G ′ , or equivalently the CC XG , is described via the incidence matrix B0,2
                                                                         T
                                                                             . Hence, learning the signal H2 can
be realized in terms of a map B0,2 : C (XG ) → C (XG ) that pushes forward the cochain H0 to H2 .
                                   T     2          0



The 2-cells defined on XG can be practically constructed using the mapper on graphs [125], a classification
tool in TDA. See Section 7.4 for more details of such a construction.

7.2.2   Image pooling as CC-pooling

Since images can be realized as lattice graphs, a signal stored on an image grid can be realized as a 0-cochain
of the lattice graph that corresponds to the image. See Figures 28(a–b) for an example. Here, we demonstrate
that the CC-pooling operation (Definition 37) is consistent with the known image-pooling definition. Indeed,
one may augment the lattice graph of Figure 28(b) by 2-cells, as shown in Figure 28(c), to perform the image
pooling operation. Usually, these cells have a regular window size. In Figure 28(c), we have chosen the
pooling window size, or equivalently the size of the 2-cell, to be 2 × 2, and the pooling stride to be 1. The
image pooling operation in this case can be realized as a CC-pooling operation induced by the cochain map
  T
B0,2 : C 0 → C 2 , as visualized in Figure 28(d). We formally record this in the following proposition.
Proposition 7.2. An image pooling operator can be realized in terms of a push-forward operator from the
underlying image domain to a 2-dimensional CC obtained by augmenting the image by appropriate 2-cells
where image pooling computations occur.

Proof. The proof is a straightforward conclusion from the definition of image pooling.


                                                       40
7   Push-forward and higher-order (un)pooling                                                  Topological deep learning




                                                                                                           3



                Rank-0 cells
                Rank-1 cells
                Rank-2 cells
                Rank-3 cells
                                                          (a)                                       (b)

Figure 29: Examples of pooling operations. In this example, a CC of dimension three is displayed. A rank
cell of rank 3 (shown in green) has the highest rank among all cells in the CC. (a): A pooling CCNN of height
one pools a cochain vector (H0 , H1 ) to a 2-cochain H2 . (b): A readout operation can be realized as a pooling
CCNN of height one by encapsulating the entire CC by a single (green) cell of rank higher than all other cells
in the CC, and by pooling (reading out) all signals of lower-order cells to the encapsulating (green) cell.


7.3     (Un)pooling CCNNs

The pooling operator of Definition 37 considers only the special case in which the tensor diagram of a CCNN
has a single edge. In what follows, we generalize the notion of pooling by identifying the defining properties
that characterize a CCNN as a pooling CCNN. To this end, we start with a CCNN whose tensor diagram has
a height of one.



      Definition 39 (Pooling CCNN of height one). Consider a CCNN represented by a tensor diagram
      CCNNG;W of height one. Let C i1 × C i2 × · · · × C im be the domain and let C j1 × C j2 × · · · × C jn be
      the codomain of the CCNN. Let imin = min(i1 , . . . , im ) and jmin = min(j1 , . . . , jn ). We say that the
      CCNN is a pooling CCNN of height one if
            1. imin < jmin , and
            2. the tensor diagram CCNNG;W has an edge labeled by a cochain operator G : C imin → C k for
               some k ≥ jmin .



Intuitively, a CCNN represented by a tensor diagram of height one is a pooling CCNN of height one if it
pushes forward its lowest-rank signals to higher-rank cells. Observe that a readout operation can be realized
as a pooling CCNN of height one; see Figure 29 for an illustration.
A CCNN may not perform a pooling operation at every layer, and it may preserve the dimensionality
of the lowest-rank signal. Before we give the general definition of pooling CCNNs, we first define lowest
rank-preserving CCNNs of height one.



      Definition 40 (Lowest rank-preserving CCNN of height one). Consider a CCNN represented by a
      tensor diagram of height one. Let C i1 × C i2 × · · · × C im be the domain and let C j1 × C j2 × · · · × C jn be
      the codomain of the CCNN. Let imin = min(i1 , . . . , im ) and jmin = min(j1 , . . . , in ). We say that the
      CCNN is a lowest rank-preserving CCNN of height one if imin = jmin .



Every CCNN is a composition of CCNNs that are represented by tensor diagrams of height one. Hence,
pooling CCNNs can be characterized in terms of tensor diagrams of height one, as elaborated in Definition 41.


                                                            41
Topological deep learning                                   8    Hasse graph interpretation and equivariances of CCNNs




      Definition 41 (Pooling CCNN). Let CCNNG;W be a tensor diagram representation of a CCNN. We
      decompose the CCNN as

                                  CCNNG;W = CCNNGN ;WN ◦ · · · ◦ CCNNG1 ;W1 ,

      where CCNNGi ;Wi , i = 1, . . . , N , is a tensor diagram of height one representing the i-th layer of the
      CCNN, and Gi ⊆ G. We call the CCNN represented by CCNNG;W a pooling CCNN if
           1. every CCNNGi ;Wi is either a pooling CCNN of height one or a lowest rank-preserving CCNN
              of height one, and
           2. at least one of the layers CCNNGi ;Wi is a pooling CCNN of height one.

Intuitively, a pooling CCNN is a CCNN whose tensor diagram forms a ‘ladder’ that pushes signals to
higher-rank cells at every layer. Figure 19(d) gives an example of a pooling CCNN of height two.
An unpooling CCNN of height one is defined similarly to a pooling CCNN of height one (Definition 39),
with the only difference being that the inequality imin < jmin becomes imin > jmin . Moreover, an unpooling
CCNN (Definition 42) is defined analogously to a pooling CCNN (Definition 41).

    Definition 42 (Unpooling CCNN). Let CCNNG;W be a tensor diagram representation of a CCNN.
    We decompose the CCNN as

                                  CCNNG;W = CCNNGN ;WN ◦ · · · ◦ CCNNG1 ;W1 ,

      where CCNNGi ;Wi , i = 1, . . . , N , is a tensor diagram of height one representing the i-th layer of the
      CCNN, and Gi ⊆ G. We call the CCNN represented by CCNNG;W an unpooling CCNN if
           1. every CCNNGi ;Wi is either an unpooling CCNN of height one or a lowest rank-preserving
              CCNN of height one, and
           2. at least one of the layers CCNNGi ;Wi is an unpooling CCNN of height one.


7.4     Mapper and the CC-pooling operation

In practice, constructing a useful CC-pooling operation on a higher-order domain is determined by the
higher-rank cells in the input CC. Similar to image-based models, CC-pooling operations can be applied
sequentially at the end of a higher-order network to provide a summary representation of the input domain;
see Figure 27 for an example. Such a hierarchical summary might not be readily available in the input CC.
For instance, if X is a graph, then a CC-pooling operation, as given in Definition 37, can only push forward
an input node-signal to an edge-signal, which may not always provide a compact summary of the input signal.
In such situations, one may choose to augment the input CC X with a collection of new cells of dimension
dim(X ) + 1 so that the new cells approximate the shape of the input CC X . Figure 30 displays an example
of augmenting a graph X , which is a CC of dimension one, with new cells of dimension two using the mapper
on graphs (MOG) construction suggested in [125, 85, 244]6 . The augmented higher-rank cells obtained from
the MOG construction summarize the shape features of the underlying graph, which is a desirable pooling
characteristic (e.g., in shape analysis). We refer to Appendix E for details about the MOG construction of
topology-preserving CC-pooling operations.

8      Hasse graph interpretation and equivariances of CCNNs

In this section, we scrutinize the properties of our topological learning machines by establishing connections
to the pre-existing findings regarding GNNs. We begin our inquiry by elucidating the interpretation of
   6While graph skeletonization using the mapper algorithm [244] has been studied in [85], our implementation and discussion

here relies on the notions suggested in [125].


                                                            42
8   Hasse graph interpretation and equivariances of CCNNs                                                 Topological deep learning




Figure 30: An example of constructing a shape-preserving pooling operation on a CC X . Here, we demonstrate
the case in which X is a graph. We utilize the mapper on graphs (MOG) construction [125, 244], a graph
skeletonization algorithm that can be used to augment X with topology-preserving cells of rank 2. (a): An
input graph X , which is a CC of dimension one. (b): The MOG algorithm receives three elements as input,
namely the graph X , a feature-preserving scalar function g : X 0 → [a, b], and a cover U of range [a, b], that is
a collection of open sets that covers the closed interval [a, b]. The scalar function g is used to pull back a
covering U on the range [a, b] to a covering on X . The colors of nodes in Figure (b) indicate the scalar values
of g. In Figure (b), X is split into four segments so that each segment corresponds to a cover element of U.
(c): The figure shows the connected components in pull-back cover elements. Each connected component is
enclosed by a blue cell. Each of these blue cells is considered as a cell of rank 2. We augment X with the blue
cells to form a CC of dimension two, consisting of the original 0-cells and 1-cells in X as well as the augmented
2-cells. This augmented CC is denoted by Xg,U . (d): The MOG algorithm constructs a graph, whose nodes are
the connected components contained in each cover element in the pull-back of the cover U, and whose edges
are formed by the intersection between these connected components. In other words, the MOG-generated
graph summarizes the connectivity between the augmented cells of rank 2 added via the MOG algorithm.
Observe that the adjacency matrix of the MOG-generated graph given in Figure (d) is equivalent to the
adjacency matrix A2,2 of Xg,U , since 2-cells in X are 2-adjacent if and only if they intersect on a node (and
the latter occurs if and only if there is an edge between the nodes of the MOG-generated graph). Given
this CC structure, the cochain map B0,2  T
                                            : C 0 (Xg,U ) → C 2 (Xg,U ) can be used to induce a shape-preserving
CC-pooling operation. Moreover, a signal H0 supported on the nodes of X can be push-forwarded and pooled
to a signal H2 supported on the augmented 2-cells. This figure is inspired from [125].


CCs as specialized graphs, known as Hasse graphs, followed by characterizing their equivariance properties
against the actions of permutations and orientations. We further link our definitions of equivariances to the
conventional ones under the Hasse graph representation.

8.1     Hasse graph interpretation of CCNNs

We first demonstrate that every CC can be reduced to a unique and special graph known as the Hasse graph.
This reduction enables us to analyze and understand various computational and conceptual aspects of CCNNs
in terms of graph-based models.

8.1.1     CCs as Hasse graphs

Definition 10 implies that a CC is a poset, that is a partially ordered set whose partial order relation is the
set inclusion relation. It also implies that two CCs are equivalent if and only if their posets are equivalent7 .
Definition 43 introduces the Hasse graph [1, 259] of a CC, which is a directed graph associated with a finite
poset.
    7 For related structures (e.g., simplicial/cell complexes), this poset is typically called the face poset [259].




                                                                  43
Topological deep learning                                                     8         Hasse graph interpretation and equivariances of CCNNs




                                                                        1 2                 3   4                   4 2
                                                                        3 5                 5   6                   6       1




                                Hasse graph
                                                        1   2   1 3   2 5         3 5       3       4   5       6       4 6     4 2    6    1




                                                                       1      2         3           4       5           6
                  2         1


                  5         3


                  6         4

                                                                       1 2                  3   4                   4 2
                  1         2
                                                                       3 5                  5   6                   6   1




                                Augmented Hasse graph
                                                        1   2   1 3   2 5     3 5           3   4       5   6           4 6     4 2    6    1




                                                                       1      2         3       4           5           6




Figure 31: Example of the Hasse graph of a CC. (a): CC of a Möbius strip. (b): Hasse graph of the CC,
describing the poset structure between cells. (c): Hasse graph augmented with the edges defined via A0,1 and
coA2,1 .



   Definition 43 (Hasse graph). The Hasse graph of a CC (S, X , rk) is a directed graph HX =
   (V (HX ), E(HX )) with vertices V (HX ) = X and edges E(HX ) = {(x, y) : x ⊊ y, rk(x) = rk(y) − 1}.

The vertices of the Hasse graph HX of a CC (S, X , rk) are the cells of X , while the edges of HX are determined
by the immediate incidence among these cells. Figure 31 shows an example of the Hasse graph of a CC.
The CC structure class is the set of CCs determined up to isomorphism, according to Definition 10. Proposi-
tion 8.1 provides sufficient criteria for determining CC structure classes. The proof of Proposition 8.1 relies on
the observations that CC structure classes are determined by the underlying Hasse graph representation and
                                                                                            dim(X )−1
that the Hasse graph provides the same information as the incidence matrices {Bk,k+1 }k=0             . Figure 32
supports visually the proofs of parts 2 and 3 in Proposition 8.1.
Proposition 8.1. Let (S, X , rk) be a CC. For the CC structure class indicated by (S, X , rk), the following
sufficient conditions hold:

                                                                                                                                           dim(X )−1
     1. The CC structure class is determined by the incidence matrices {Bk,k+1 }k=0                                                                       .
                                                                                                                                      dim(X )−1
     2. The CC structure class is determined by the adjacency matrices {Ak,1 }k=0                                                                    .
                                                                                                                                                dim(X )
     3. The CC structure class is determined by the coadjacency matrices {coAk,1 }k=1                                                                     .

Proof. The proof of the three parts of the proposition follows by noting that the structure of a CC is
determined completely by its Hasse graph representation. The first part of the proposition follows from the
                                                                                               dim(X −1)
fact that the edges in the Hasse graph are precisely the non-zero entries of matrices {Bk,k+1 }k=0       . The


                                                                                  44
8   Hasse graph interpretation and equivariances of CCNNs                                 Topological deep learning




Figure 32: Relation between immediate incidence and (co)adjacency on the Hasse graph of a CC. This figure
supports visually the proofs of parts 2 and 3 in Proposition 8.1. (a): Two (k − 1) cells xk−1 and y k−1 (orange
vertices) are 1-adjacent if and only if there exists a k-cell z k (pink vertex) that is incident to xk−1 and y k−1 .
(b): Two (k + 1) cells xk+1 and y k+1 (blue vertices) are 1-coadjacent if and only if there exists a k-cell z k
(pink vertex) that is incident to xk+1 and y k+1 .


second part follows by observing that two (k − 1) cells xk−1 and y k−1 are 1-adjacent if and only if there exists
a k-cell z k that is incident to xk−1 and y k−1 . The third part is confirmed by noting that two (k + 1)-cells
xk+1 and y k+1 are 1-coadjacent if and only if there exists a k-cell z k that is incident to xk+1 and y k+1 .



8.1.2   Augmented Hasse graphs

The Hasse graph of a CC is useful because it shows that computations for a higher-order deep learning model
can be reduced to computations for a graph-based model. Particularly, a k-cochain (signal) being processed
on a CC X can be thought as a signal on the corresponding vertices of the associated Hasse graph HX .
The edges specified by the matrices Bk,k+1 determine the message-passing structure of a given higher-order
model defined on X . However, the message-passing structure determined via the matrices Ar,k is not directly
supported on the corresponding edges of HX . Thus, it is sometimes desirable to augment the Hasse graph
with additional edges other than the ones specified by the poset partial order relation of the CC. Along these
lines, Definition 44 introduces the notion of augmented Hasse graph.

    Definition 44 (Augmented Hasse graph). Let X be a CC, and let HX be its Hasse graph with vertex
    set V (HX ) and edge set E(HX ). Let N = {N1 , . . . , Nn } be a set of neighborhood functions defined
    on X . We say that HX has an augmented edge ex,y induced by N if there exist Ni ∈ N such that
    x ∈ Ni (y) or y ∈ Ni (x). Denote by EN the set of all augmented edges induced by N . The augmented
    Hasse graph of X induced by N is defined to be the graph HX (N ) = (V (HX ), E(HX ) ∪ EN ).

It is easier to think of the augmented Hasse graph in Definition 44 in terms of the matrices G = {G1 , . . . , Gn }
associated with the neighborhood functions N = {N1 , . . . , Nn }. Each augmented edge in HX (N ) corresponds
to a non-zero entry in some Gi ∈ G. Since N and G store equivalent information, we use HX (G) to
denote the augmented Hasse graph induced by the edges determined by G. For instance, the graph given in
Figure 31(c) is denoted by HX (A0,1 , coA2,1 ).

8.1.3   Reducibility of CCNNs to graph-based models

In this section, we show that any CCNN-based computational model can be realized as a message-passing
scheme over a subgraph of the augmented Hasse graph of the underlying CC. Every CCNN is determined
via a computational tensor diagram, which can be built using the elementary tensor operations, namely
push-forward operations, merge nodes and split nodes. Thus, the reducibility of CCNN-based computations
to message-passing schemes over graphs can be achieved by proving that these three tensor operations can be
executed on an augmented Hasse graph. Proposition 8.2 states that push-forward operations are executable
on augmented Hasse graphs.
Proposition 8.2. Let X be a CC and let FG : C i (X ) → C j (X ) be a push-forward operator induced by
a cochain map G : C i (X ) → C j (X ). Any computation executed via FG can be reduced to a corresponding
computation over the augmented Hassed graph HX (G) of X .


                                                        45
Topological deep learning                           8     Hasse graph interpretation and equivariances of CCNNs




Proof. Let X be a CC. Let HX (G) be the augmented Hasse graph of X determined by G. The definition of
the augmented Hasse graph implies that there is a one-to-one correspondence between the vertices HX (G)
and the cells in X . Given a cell x ∈ X , let x′ be the corresponding vertex in HX (G). Let y be a cell in X
with a feature vector hy computed via the push-forward operation specified by Equation 3. Recall that the
vector hy is computed by aggregating all vectors hx attached to the neighbors x ∈ X i of y with respect
to the neighborhood function NGT . Let mx,y be a computation (message) that is executed between two
cells x and y of X as a part of the computation of push-forward FG . It follows from the augmented Hasse
graph definition that the cells x and y must have a corresponding non-zero entry in matrix G. Moreover, this
non-zero entry corresponds to an edge in HX (G) between x′ and y ′ . Thus, the computation mx,y between
the cells x and y of X can be carried out as the computation (message) mx′ ,y′ between the corresponding
vertices x′ and y ′ of HX (G).

Similarly, computations on an arbitrary merge node can be characterized in terms of computations on a
subgraph of the augmented Hasse graph of the underlying CC. Proposition 8.3 formalizes this statement.
Proposition 8.3. Any computation executed via a merge node MG,W as given in Equation 1 can be reduced
to a corresponding computation over the augmented Hasse graph HX (G) of the underlying CC.

Proof. Let X be a CC. Let G = {G1 , . . . , Gn } be a sequence of cochain operators defined on X . Let HX (G)
be the augmented Hasse graph determined by G. By the augmented Hasse graph definition, there is a
one-to-one correspondence between the vertices of HX (G) and the cells of X . For each cell x ∈ X , let x′ be
the corresponding vertex in HX (G). Let mx,y be a computation (message) that is executed between two
cells x and y of X as part of the evaluation of function MG,W . Hence, the two cells x and y must have a
corresponding non-zero entry in a matrix Gi ∈ G. By the augmented Hasse graph definition, this non-zero
entry corresponds to an edge in HX (G) between x′ and y ′ . Thus, the computation mx,y between the cells x
and y of X can be carried out as the computation (message) mx′ ,y′ between the corresponding vertices x′
and y ′ of HX (G).

Propositions 8.2 and 8.3 ensure that push-forward and merge node computations can be realized on augmented
Hasse graphs. Theorem 8.4 generalizes Propositions 8.2 and 8.3, stating that any computation on tensor
diagrams is realizable on augmented Hasse graphs.
Theorem 8.4. Any computation executed via a tensor diagram CCNNG;W can be reduced to a corresponding
computation on the augmented Hasse graph HX (G).

Proof. The conclusion follows directly from Propositions 8.1.1 and 8.2, along with the fact that any tensor
diagram can be realized in terms of the three elementary tensor operations.

According to Theorem 8.4, a tensor diagram and its corresponding augmented Hasse graph encode the
same computations in alternative forms. Figure 33 illustrates that the augmented Hasse graph provides a
computational summary of the associated tensor diagram representation of a CCNN.

8.1.4   Augmented Hasse graphs and CC-pooling

The Hasse graph and its augmented version are graph representations of the poset structure of the underlying
CC. It is instructive to interpret the (un)pooling operations (Definitions 37 and 38) with respect to these
graphs. The CC-pooling operation of Definition 37 maps a signal in the poset structure from lower-rank
cells to higher-rank ones. On the other hand, the CC-unpooling operation of Definition 38 maps a signal
in the opposite direction. Figure 34 presents an example of CC (un)-pooling operations visualized over the
augmented Hasse graph of the underlying CC.

8.1.5   Augmented Hasse diagrams, message passing, and merge nodes

The difference between constructing a CCNN using the higher-order message passing paradigm given in
Section 6.1 versus using the three elementary tensor operations given in Section 5.3 has been demonstrated
in Section 6.3. In particular, Section 6.3 mentions that merge nodes naturally allow for a more flexible


                                                     46
8   Hasse graph interpretation and equivariances of CCNNs                           Topological deep learning




                              (a)                                          (b)

Figure 33: Tensor diagrams of two CCNNs and their corresponding augmented Hasse graphs. Edge labels are
dropped from the tensor diagrams to avoid clutter, as they can be inferred from the corresponding augmented
Hasse graphs. (a): A tensor diagram obtained from a higher-order message-passing scheme. (b): A tensor
diagram obtained by using the three elementary tensor operations introduced in Section 5.3.




Figure 34: CC (un)-pooling operations viewed on the augmented Hasse graph of a CC. The vertices in the
figure represent skeletons in the underlying CC. Black edges represent edges in the Hasse graph between these
vertices, whereas red edges represent edges obtained from the augmented Hasse graph structure. CC-pooling
corresponds to pushing a signal in the poset structure from lower-rank to higher-rank vertices, whereas
CC-unpooling corresponds to pushing a signal in the poset structure from higher-rank to lower-rank vertices.

computational framework in comparison to the higher-order message-passing paradigm. This flexibility
manifests in terms of the underlying tensor diagram as well as the input for the network under consideration.
The difference between tensor operations and higher-order message passing can also be highlighted with
augmented Hasse graphs, as demonstrated in Figure 33. Figure 33(a) shows a tensor diagram obtained
from a higher-order message-passing scheme on a CCNN. We observe two key properties of this CCNN: the
initial input cochains are supported on all cells of all dimensions of the domain, and the CCNN updates
all cochains supported on all cells of all dimensions of the domain at every iteration given a predetermined
set of neighborhood functions. As a consequence, the corresponding augmented Hasse graph exhibits a
uniform topological structure. In contrast, Figure 33(b) shows a tensor diagram constructed using the three
elementary tensor operations. As the higher-order message-passing rules do not impose constraints, the
resulting augmented Hasse graph exhibits a more flexible structure.

8.1.6   Higher-order representation learning

The relation between augmented Hasse graphs and CCs given by Theorem 8.4 suggests that many graph-based
deep learning constructions have analogous constructions for CCs. In this section, we demonstrate how
higher-order representation learning can be reduced to graph representation learning [130], as an application
of certain CC computations as augmented Hasse graph computations.


                                                      47
Topological deep learning                               8     Hasse graph interpretation and equivariances of CCNNs




The goal of graph representation is to learn a mapping that embeds the vertices, edges or subgraphs of a
graph into a Euclidean space, so that the resulting embedding captures useful information about the graph.
Similarly, higher-order representation learning [123] involves learning an embedding of various cells in a
given topological domain into a Euclidean space, preserving the main structural properties of the topological
domain. More precisely, given a complex X , higher-order representation learning refers to learning a pair
(enc, dec) of functions, consisting of the encoder map enc : X k → Rd and the decoder map dec : Rd × Rd → R.
The encoder function associates to every k-cell xk in X a feature vector enc(xk ), which encodes the structure
of xk with respect to the structures of other cells in X . On the other hand, the decoder function associates
to every pair of cell embeddings a measure of similarity, which quantifies some notion of relation between
the corresponding cells. We optimize the trainable functions (enc, dec) using a context-specific similarity
measure sim : X k × X k → R and an objective function
                                        X
                                Lk =         l(dec(enc(xk ), enc(y k )), sim(xk , y k )),                 (22)
                                       xk ∈X k

where l : R × R → R is a loss function. The precise relation between higher-order and graph representation
learning is given by Proposition 8.5.
Proposition 8.5. Higher-order representation learning can be reduced to graph representation learning.

Proof. Let sim : X k × X k → R be a similarity measure. The graph GX k is defined as the graph whose vertex
set corresponds to cells in X k and whose edges correspond to cell pairs in X k × X k mapped to non-zero values
by the function sim. Thus, the pair (enc, dec) corresponds to a pair (encG , decG ) of the form encG : GX k → R
and decG : Rd × Rd → R. Thereby, learning the pair (enc, dec) is reduced to learning the pair (encG , decG ).
TopoEmbedX8 , one of our three contributed software packages, supports higher-order representation learning
on cell complexes, simplicial complexes, and CCs. The main computational principle underlying TopoEmbedX
is Proposition 8.5. Specifically, TopoEmbedX converts a given higher-order domain into a subgraph of the
corresponding augmented Hasse graph, and then utilizes existing graph representation learning algorithms to
compute the embedding of elements of this subgraph. Given the correspondence between the elements of the
augmented Hasse graph and the original higher-order domain, this results in obtaining embeddings for the
higher-order domain.
Remark 8.6. Following our discussion on Hasse graphs, and particularly the ability to transform computations
on a CCNN to computations on a (Hasse) graph, one may argue that GNNs are sufficient and that there is no
need for CCNNs. However, this is a misleading clue, in the sense that any computation can be represented by
a computational graph. Applying a standard GNN over the augmented Hasse graph of a CC is not equivalent
to applying a CCNN. This point will become clearer in Section 8.2, where we introduce CCNN equivariances.

8.2     On the equivariance of CCNNs

Analogous to their graph counterparts, higher-order deep learning models, and CCNNs in particular, should
always be considered in conjunction with their underlying equivariance [50]. We now provide novel defini-
tions for permutation and orientation equivariance for CCNNs and draw attention to their relations with
conventional notions of equivariance defined for GNNs.

8.2.1     Permutation equivariance of CCNNs

Motivated by Proposition 8.1, which characterizes the structure of a CC, this section introduces permutation-
equivariant CCNNs. We first define the action of the permutation group on the space of cochain maps.

      Definition 45 (Permutation action on space of cochain maps). Let X be a CC. Define Sym(X ) =
      Qdim(X )
        i=0    Sym(X k ) the group of rank-preserving permutations of the cells of X . Let G = {Gk } be a se-
                                                                                                      dim(X )
      quence of cochain maps defined on X with Gk : C ik → C jk , 0 ≤ ik , jk ≤ dim(X ). Let P = (Pi )i=0     ∈
                                                                                               T dim(X )
      Sym(X ). Define the permutation (group) action of P on G by P(G) = (Pjk Gk Pik )i=0 .

  8 https://github.com/pyt-team/TopoEmbedX




                                                         48
8   Hasse graph interpretation and equivariances of CCNNs                                          Topological deep learning




We introduce permutation-equivariant CCNNs in Definition 46, using the group action given in Definition 45.
Definition 46 generalizes the relevant definitions in [221, 235]. We refer the reader to [257, 145] for a related
discussion. Hereafter, we use Projk : C 1 × · · · × C m → C k to denote the standard k-th projection for 1 ≤ k ≤ m,
defined via Projk (H1 , . . . , Hk , . . . , Hm ) = Hk .

    Definition 46 (Permutation-equivariant CCNN). Let X be a CC and let G = {Gk } be a finite
                                                         dim(X )
    sequence of cochain maps defined on X . Let P = (Pi )i=0     ∈ Sym(X ). A CCNN of the form

                              CCNNG;W : C i1 × C i2 × · · · × C im → C j1 × C j2 × · · · × C jn

    is called a permutation-equivariant CCNN if

              Projk ◦CCNNG;W (Hi1 , . . . , Him ) = Pk Projk ◦CCNNP(G);W (Pi1 Hi1 , . . . , Pim Him )               (23)

    for all 1 ≤ k ≤ m and for any (Hi1 , . . . , Him ) ∈ C i1 × C i2 × · · · × C im .

Equation 23 generalizes the corresponding notion of permutation equivariance of GNNs. Consider a graph
with n vertices and adjacency matrix A. Denote a GNN on this graph by GNNA;W . Let H ∈ Rn×k be
vertex features. Then GNNA;W is permutation equivariant in the sense that for P ∈ Sym(n) we have
P GNNA;W (H) = GNNP AP T ;W (P H).
In general, working with Definition 46 may be cumbersome. It is easier to characterize the equivariance in
terms of merge nodes. To this end, recall that the height of a tensor diagram is the longest path from any
source node to any target node. Proposition 8.7 allows us to express tensor diagrams of height one in terms
of merge nodes.
Proposition 8.7. Let CCNNG;W : C i1 × C i2 × · · · × C im → C j1 × C j2 × · · · × C jn be a CCNN with a tensor
diagram of height one. Then

                                        CCNNG;W = (MGj1 ;W1 , . . . , MGjn ;Wn ),                                      (24)

where Gk ⊆ G.

Proof. Let CCNNG;W : C i1 × C i2 × · · · × C im → C j1 × C j2 × · · · × C jn be a CCNN with a tensor diagram
of height one. Since the codomain of the function CCNNG;W is C j1 × C j2 × . . . × C jn , then CCNNG;W is
determined by n functions Fk : C i1 × C i2 × · · · × C im → C jk for 1 ≤ k ≤ n. Since the height of the tensor
diagram of CCNNG;W is one, then each function Fk is also of height one and it is thus a merge node by
definition. The result follows.

Proposition 8.7 states that every target node jk in a tensor diagram of height one is a merge node specified
by the operators Gjk formed by the labels of the edges with target jk . Definition 46 introduces the general
notion of permutation equivariance of CCNNs. Definition 47 introduces the notion of permutation-equivariant
merge node. Since a merge node is a CCNN, Definition 47 is a special case of Definition 46.

    Definition 47 (Permutation-equivariant merge node). Let X be a CC and let G = {Gk } be a finite
                                                                                            dim(X )
    sequence of cochain operators defined on X with Gk : C ik (X ) → C j (X ). Let P = (Pi )i=0     ∈ Sym(X ).
    We say that the merge node given in Equation 1 is a permutation-equivariant merge node if

                             MG;W (Hi1 , . . . , Him ) = Pj MP(G);W (Pi1 Hi1 , . . . , Pi1 Him )                    (25)

    for any (Hi1 , . . . , Him ) ∈ C i1 × C i2 × · · · × C im .

Proposition 8.8. Let CCNNG;W : C i1 × C i2 × · · · × C im → C j1 × C j2 × · · · × C jn be a CCNN with a tensor
diagram of height one. Then CCNNG;W is permutation equivariant if and only the merge nodes MGjk ;Wk
given in Equation 24 are permutation equivariant for 1 ≤ k ≤ n.


                                                                  49
Topological deep learning                             8     Hasse graph interpretation and equivariances of CCNNs




Proof. If a CCNN is of height one, then by Proposition 8.7, Projk ◦ CCNNG;W (Hi1 , . . . , Him ) = MGjk ;Wk .
Hence, the result follows from the definition of merge node permutation equivariance (Definition 47) and the
definition of CCNN permutation equivariance (Definition 46).


Finally, Theorem 8.9 characterizes the permutation equivariance of CCNNs in terms of merge nodes. From
this point of view, Theorem 8.9 provides a practical version of permutation equivariance for CCNNs.
Theorem 8.9. A CCNNG;W is permutation equivariant if and only if every merge node in CCNNG;W is
permutation equivariant.



Proof. Proposition 8.8 proves this fact for CCNNs of height one. For CCNNs of height n, it is enough to
observe that a CCNN of height n is a composition of n CCNNs of height one and that the composition of
two permutation-equivariant networks is a permutation-equivariant network.


Remark 8.10. Our permutation equivariance assumes that all cells in each dimension are independently
labeled with indices. However, if we label the cells in a CC with subsets of the powerset P(S) rather than with
indices, then we only need to consider permutations of the powerset that are induced by permutations of the
0-cells in order to ensure permutation equivariance.
Remark 8.11. A GNN is equivariant in that a permutation of the vertex set of the graph and the input
signal over the vertex set yields the same permutation of the GNN output. Applying a standard GNN over
the augmented Hasse graph of the underlying CC is thus not equivalent to applying a CCNN. Although the
message-passing structures are the same, the weight-sharing and permutation equivariance of the standard
GNN and CCNN are different. In particular, Definition 10 gives additional structure, which is not preserved
by an arbitrary permutation of the vertices in the augmented Hasse graph. Thus, care is required in order
to reduce message passing over a CCNN to message passing over the associated augmented Hasse graph.
Specifically, one need only consider the subgroup of permutations of vertex labels in the augmented Hasse graph
which are induced by permutations of 0-cells in the corresponding CC. Thus, there is merit in adopting the rich
notions of topology to think about distributed, structured learning architectures, as topological constructions
facilitate reasoning about computation in ways that are not within the scope of graph-based approaches.
Remark 8.12. Note that Proposition 8.5 does not contradict Remark 8.11. In fact, the computations
described in Proposition 8.5 are conducted on a particular subgraph of the Hasse graph whose vertices are
the k-cells of the underlying complex. Differences between graph-based networks and TDL networks start to
emerge particularly once different dimensions are considered simultaneously during computations.


8.2.2   Orientation equivariance of CCNNs

When CC are reduced to regular cell complexes, then orientation equivariance can also be introduced to
CCNNs. Analogous to Definition 45, we introduce the following definition of orientation actions on CCs.


   Definition 48 (Orientation action on space of diagonal-cochain maps). Let X be a CC. Let G = {Gk }
   be a sequence of cochain operators defined on X with Gk : C ik → C jk , 0 ≤ ik , jk ≤ dim(X ). Let O(X )
                                   dim(X )
   be the group of tuples D = (Di )i=0     of diagonal matrices with diagonals ±1 of size |X k | × |X k | such
                                                                                                    dim(X )
   that D0 = I. Define the orientation (group) action of D on G by D(G) = (Djk Gk Dik )i=0 .


We introduce orientation-equivariant CCNNs in Definition 49, using the group action given in Definition 48.
Orientation equivariance for CCNNs (Definition 49) is put forward in analogous way to permutation equivari-
ance for CCNNs (Definition 46).


                                                       50
9    Implementation and numerical experiments                                                     Topological deep learning




      Definition 49 (Orientation-equivariant CCNN). Let X be a CC and let G = {Gk } be a finite sequence
      of cochain operators defined on X . Let D ∈ O(X ). A CCNN of the form

                              CCNNG;W : C i1 × C i2 × · · · × C im → C j1 × C j2 × · · · × C jn

      is called an orientation-equivariant CCNN if

              Projk ◦ CCNNG;W (Hi1 , . . . , Him ) = Dk Projk ◦CCNND(G);W ((Di1 Hi1 , . . . , Di1 Him ))           (26)

      for all 1 ≤ k ≤ m and for any (Hi1 , . . . , Him ) ∈ C i1 × C i2 × · · · × C im .

Propositions 8.7 and 8.8 can be stated analogously for the orientation equivariance case. We skip stating
these facts here, and only state the main theorem that characterizes the orientation equivariance of CCNNs
in terms of merge nodes.
Theorem 8.13. A CCNNG;W is orientation equivariant if and only if every merge node in CCNNG;W is
orientation equivariant.


Proof. The proof of Theorem 8.13 is similar to the proof of Theorem 8.9.


9      Implementation and numerical experiments

The proposed CCNNs can be used to construct different neural network architectures for diverse learning
tasks. In this section, we demonstrate the generality and the efficacy of CCNNs by evaluating their predictive
performance on shape analysis and graph learning tasks. In our geometric processing experiments, we compare
CCNNs against state-of-the-art methods, which are highly engineered and trained towards specific tasks.
Furthermore, we conduct our experiments on various data modalities commonly studied in geometric data
processing, namely on point clouds and 3D meshes. We also perform experiments on graph data. In our
experiments, we tune three main components: the choice of CCNN architecture, the learning rate, and the
number of replications in data augmentation. We justify the choice of CCNN architecture for each learning
task. We have implemented our pipeline in PyTorch, and ran the experiments on a single GPU NVIDIA
GeForce RTX 3060 Ti using a Microsoft Windows back-end.

9.1     Software: TopoNetX, TopoEmbedX, and TopoModelX

All our software development and experimental analysis have been carried out using Python. We have devel-
oped three Python packages, which we have also used to run our experiments: TopoNetX9 ,TopoEmbedX10
and TopoModelX11 . TopoNetX supports the construction of several topological structures, including cell
complex, simplicial complex, and combinatorial complex classes. These classes provide methods for computing
boundary operators, Hodge Laplacians and higher-order adjacency operators on cell, simplicial, and combina-
torial complexes, respectively. TopoEmbedX supports representation learning of higher-order relations on
cell complexes, simplicial complexes, and combinatorial complexes. TopoModelX supports computing deep
learning models defined on these topological domains.
In addition to our software package implementations, we have utilized PyTorch [207] for training the neural
networks reported in this section. Also, we have utilized Scikit-learn [208] to compute the eigenvectors of the
1-Hodge Laplacians. The normal vectors of point clouds have been computed using the Point Cloud Utils
package [265]. Finally, we have used NetworkX [122] and HyperNetX [146], both in the development of our
software packages as well as in our computations.

    9 https://github.com/pyt-team/TopoNetX
    10 https://github.com/pyt-team/TopoEmbedX
    11 https://github.com/pyt-team/TopoModelX




                                                              51
Topological deep learning                                          9   Implementation and numerical experiments




9.2     Datasets

In our evaluation of CCNNs, we use four datasets: the Human Body, the COSEG, the SHREC11, and a
benchmark dataset for graph classification taken from [37]. A summary of these datasets follows.

Human Body segmentation dataset. The original Human Body segmentation dataset presented in [11]
contains relatively large meshes with a size up to 12,000 vertices. The segmentation labels provided in this
dataset are set per-face, and the segmentation accuracy is defined to be the ratio of the correctly classified
faces over the total number of faces in the entire dataset. In this work, we use a simplified version of
the original Human Body dataset, as provided by [131], in which meshes have less than 1,000 nodes and
segmentation labels are remapped to edges. We use this simplified version of the Human Body dataset for
the shape analysis (i.e., mesh segmentation) task in Section 9.3.1.

COSEG segmentation dataset. The original COSEG dataset [262] contains eleven sets of shapes with
ground-truth segmentation. In this work, we use a subset of the original COSEG dataset that contains
relatively large sets of aliens, vases, and chairs. These three sets consist of 200, 300, and 400 shapes,
respectively. We use this custom subset of the COSEG dataset for the shape analysis (i.e., mesh segmentation)
task in Section 9.3.1.

SHREC11 classification dataset. SHREC 2011 [173], abbreviated as SHREC11, is a large-scale dataset
that contains 600 nonrigid deformed shapes (watertight triangle meshes) from 30 categories, where each
category contains an equal number of objects. Examples of these categories include hand, lamp, woman, man,
flamingo, and rabbit. The dataset is available as a training and test set consisting of 480 and 120 shapes,
respectively. We use the SHREC11 dataset for the shape analysis tasks in Sections 9.3.2 and 9.4.

Benchmark dataset for graph classification. This dataset has graphs belonging to three different
classes [37]. For each graph, the feature vector on each vertex (the 0-cochain) is a one-hot vector of size five,
and it stores the relative position of the vertices on the graph. This dataset has easy and hard versions. The
easy version has highly connected graphs, while the hard version has sparse graphs. We use this dataset for
the graph classification task in Section 9.3.3.

9.3     Shape analysis: mesh segmentation and classification

The CC structure used for the shape analysis experiments (mesh segmentation and classification) is simply
induced by the triangulation of the meshes. Specifically, the 0-, 1-, and 2-cells are the vertices, edges, and
faces of the mesh, respectively. The matrices used for the CCNNs are B0,1 , B0,2 , their transpose matrices,
and the (co)adjacency matrices A1,1 , coA1,1 , and coA2,1 .
A CCNN takes a vector of cochains as input features. For shape analysis tasks, we consider cochains, whose
features are built directly from the vertex coordinates of the underlying mesh. We note that other choices
(e.g., spectral-based cochains as in [188]) can also be included. Our shape analysis tasks have three input
cochains: the vertex, edge and face cochains. Each vertex cochain has two input features: the position and
the normal vectors associated with the vertex. Similar to [131], each edge cochain consists of five features:
the length of the edge, the dihedral angle, two inner angles, and two edge-length ratios for each face. Finally,
each input face cochain consists of three input features: the face area, face normal, and the three face angles.

9.3.1    Mesh segmentation

For the Human Body dataset [185], we built a CCNN that produces an edge class. The tensor diagram of the
architecture is shown in Figure 35(a). For the COSEG dataset [262], we built a CCNN that combines our
proposed feature vectors defined on vertices, edges, and faces to learn the final face class. The architecture
uses incidence matrices as well as (co)adjacency matrices to construct a signal flow as demonstrated in
Figure 35(b). Specifically, the tensor diagram displays three non-squared attention-blocks and three squared
attention blocks. The depth of the model is chosen to be two, as indicated in Figure 35(b).


                                                       52
9                   Implementation and numerical experiments                                                                                                                                   Topological deep learning




                                                                                                                                                                           Pooling with MOG
                    CCNNCOSEG




Mesh segmentation                         Mesh classification                                       Graph classification
                                                                CCNNSHREC                                                  CCNNGraph
                                                                                                                                                                                                        CCNNMOG1         CCNNMOG2



                                                                                                                                                                      mesh/point cloud classification
                    CCNNHB




                                (a)                                            (b)                                                                (c)                                                              (d)
                                                                Rank-1 cells         Rank-2 cells                                 Rank-3 cells   Rank-4 cells   Rank-5 cells



Figure 35: The tensor diagrams of the CCNNs used in our experiments. (a): The CCNNs used in the mesh
segmentation tasks. In particular, CCNNHB and CCNNCOSEG are the architectures used on the Human
Body dataset [11] and on the COSEG dataset [262], respectively. (b): The mesh classification CCNN used on
the SHREC11 dataset [173]. (c): The graph classification CCNN used on the dataset provided in [37]. (d):
The mesh/point cloud classification CCNNs used in conjunction with the MOG algorithm on the SHREC11
dataset.

Table 2: Predictive accuracy on test sets related to shape analysis, namely on Human Body and COSEG
(vase, chair, alien) datasets. Red and blue colors indicate best and second best results, respectively. The
results reported here are based on the CCNNCOSEG and CCNNHB architectures, which are visualized in
Figure 35(a). In particular, the result for CCNNHB is reported in the first column, whereas the results for
CCNNCOSEG are reported in the second, third and forth columns.
                                                                                                            Segmentation tasks
                                        Method
                                                                            Human Body                   COSEG vase COSEG chair                                   COSEG alien
                                      HodgeNet                                 85.03                       90.30          95.68                                      96.03
                                      PD-MeshNet                               85.61                       95.36          97.23                                      98.18
                                      MeshCNN                                  85.39                       92.36          92.99                                      96.26
                                      CCNN                                     87.30                       93.40          98.30                                      93.70


Note that the architectures chosen for the COSEG and for the Human Body datasets have the same number
and types of building blocks; compare Figures 35(a) and (b). We use a random 85% − 15% train-test split.
For both of these architectures, a softmax activation is applied to the output tensor. All our segmentation
models are trained for 600 epochs using a learning rate of 0.0001 and the standard cross-entropy loss. These
results are consistent across Human Body and Shape COSEG datasets
We test the proposed CCNNs on mesh segmentation using the Human Body [185] and the Shape COSEG
(vase, chair, and alien) [262] datasets. For each mesh in these datasets, the utilized CC structure is the one
induced by the triangulation of the meshes, although other variations in the CC structure yield comparable
results. Further, three k-cochains are constructed for 0 ≤ k ≤ 2 and are utilized in CCNN training. As
shown in Table 2, CCNNs outperform three neural networks tailored to mesh analysis (HodgeNet [246],
PD-MeshNet [192] and MeshCCN [131]) on two out of four datasets, and are among the best two neural
networks on all four datasets.

Architecture of CCNNCOSEG and CCNNHB . In CCNNCOSEG , as shown in Figure 35(a), we choose a
CCNN pooling architecture as given in Definition 41, which pushes signals from vertices, edges and faces, and
aggregates their information towards the final face prediction class. We choose CCNNHB similarly, except
that the predicted signal is an edge class. The reason for this choice is that the Human Body dataset [11]
encodes the segmentation information on edges.


                                                                                                                                       53
Topological deep learning                                           9   Implementation and numerical experiments




Table 3: Predictive accuracy on the SHREC11 test dataset. The left and right column report the mesh and
point cloud classification results, respectively. The CCNN for mesh classification is CCNNSHREC , shown in
Figure 35(b), while the CCNN for point cloud classification is CCNNM OG2 , shown in Figure 35(d).
                                                       Classification tasks
                                           Method
                                                       Mesh Point cloud
                                         HodgeNet      99.10       94.70
                                         PD-MeshNet    99.70       99.10
                                         MeshCNN       98.60       91.00
                                         CCNN          99.17       95.20


Table 4: Predictive accuracy on the test set of [37] related to graph classification; red and blue colors indicate
best and second best results, respectively. All results are reported using the CCNNGraph architecture shown
in Figure 35(c).
                                                           Method
            Dataset
                       Graclus   NDP      DiffPool    Top-K SAGPool       MinCutPool     CCNNGraph
              Easy      97.81    97.93     98.64      82.47    84.23         99.02         98.90
              Hard      69.08    72.67     69.98      42.80    37.71         73.80         75.79


9.3.2   Mesh and point cloud classification

We evaluate our method on mesh classification using the SHREC11 dataset [173] based on the same cochains
and CC structure used in the segmentation experiment of Section 9.3.1. The CCNN architecture for
our mesh classification task, denoted by CCNNSHREC , is demonstrated in Figure 35(b). The final layer of
CCNNSHREC , depicted as a grey node in Figure 35(b), is a simple pooling operation that sums all embeddings
of the CC after mapping them to the same Euclidean space. The CCNNSHREC is trained for 40 epochs with
both tanh and identity activation functions using a learning rate of 0.005 and the standard cross-entropy loss.
We use anisotropic scaling and random rotations for data augmentation. Each mesh is augmented 30 times,
is centered around the vertex center of the mass, and is rescaled to fit inside the unit cube.
The CCNNSHREC with identity activations and tanh activations achieve predictive accuracies of 96.67%
and 99.17%, respectively. Table 3 shows that CCNNs outperform two neural networks tailored to mesh
analysis (HodgeNet and MeshCCN), being the second best model behind PD-MeshNet in mesh and point
cloud classification. It is worth mentioning that the mesh classification CCNN requires a significantly lower
number of epochs to train (40 epochs) as compared to the mesh segmentation CCNNs (600 epochs).

Architecture of CCNNSHREC . The CCNNSHREC has two layers and is chosen as a pooling CCNN in
the sense of Definition 41, similar to CCNNCOSEG and CCNNHB . The main difference is that the final layer
of CCNNSHREC , represented by the grey point in Figure 35(b), is a global pooling function that sums all
embeddings of all dimensions (zero, one and two) of the underlying CC after mapping them to the same
Euclidean space.

9.3.3   Graph classification

For the graph classification task, we use the graph classification benchmark provided in [37]; the dataset
consists of graphs with three different labels. For each graph, the feature vector on each vertex (the 0-cochain)
is a one-hot vector of size five, and it stores the relative position of the vertex on the graph. To construct
the CC structure, we use the 2-clique complex of the input graph. We then proceed to build the CCNN for
graph classification, denoted by CCNNGraph , which is visualized in Figure 35(c). The matrices used for the
construction of CCNNGraph are B0,1 , B1,2 , B0,2 , their transpose matrices, and the (co)adjacency matrices
A0,1 , A1,1 , coA2,1 . The cochains of CCNNGraph are constructed as follows. For each graph in the dataset, we
set the 0-cochain to be the one-hot vector of size 5 provided by the dataset. This one-hot vector stores the
relative position of the vertex on the graph. We also construct the 1-cochain and 2-cochain on the 2-clique
complex of the graph by considering the coordinate-wise max value of the one-hot vectors attached to the
vertices of each cell. The input to CCNNgraoh consists of the 0-cochain provided as a part of the dataset as


                                                        54
9   Implementation and numerical experiments                                                          Topological deep learning




Figure 36: Examples of applying the MOG algorithm on the SHREC11 dataset [173]. In each figure, we
show the original mesh graph on the left and the mapper graph on the right. The scalar function chosen for
the MOG algorithm is the average geodesic distance (AGD). We observe that the pooled mapper graph has
similar overall shape to the original graphs.


well as the constructed 1 and 2-cochains. The grey node in Figure 35(c) indicates a simple mean pooling
operation. We train this network with a learning rate of 0.005 and no data augmentation.
Table 4 reports the results on the easy and the hard versions of the datasets12 , and compares them to six
state-of-the-art GNNs. As shown in Table 4, CCNNs outperform all six GNNs on the hard dataset, and five
of the GNNs on the easy dataset. The proposed CCNN outperforms MinCutPool on the hard dataset, while
it attains comparable performance to MinCutPool on the easy dataset.

Architecture of CCNNGraph . In the CCNNGraph displayed in Figure 35(c) we choose a CCNN pooling
architecture as given in Definition 41 that pushes signals from vertices, edges and faces, and aggregate their
information towards the higher-order cells before making making the final prediction. For the dataset of [37],
we experiment with two architectures; the first one is identical to the CCNNSHREC shown in Figure 35(b), and
the second one is the CCNNGraph shown in Figure 35(c). We report the results for CCNNGraph , as it provides
superior performance. Note that when this neural network is conducted on an underlying simplicial complex,
the neighborhood matrices B0,1 and B1,3 are typically not considered, hence the CC-structure equipped with
these additional incidence matrices improves the generalization performance of the CCNNGraph .

9.4    Pooling with mapper on graphs and data classification

We perform experiments to measure the effectiveness of the MOG pooling strategy discussed in Section 7.4.
Recall that the MOG algorithm requires two pieces of input: the 1-skeleton of a CC X , and a scalar function
on the vertices of X . Our choice for the input scalar function is the average geodesic distance (AGD) [149],
which is suitable for shape detection as it is invariant to reflection and rotation. For two entities u and v on a
graph, the geodesic distance between u and v, denoted by d(v, u), is computed using Dijkstra’s shortest path
algorithm. The AGD is given by the following equation:
                                                                1 X
                                                 AGD(v) =           d(v, u).                                                   (27)
                                                               |V |
                                                                     u∈V


From Equation 27, it is immediate that the vertices near the center of the graph are likely to have low
function values, while points on the periphery are likely to have high values. This observation has been
utilized to study graph symmetry [149], and it provides a justification for selecting the AGD for the MOG
pooling strategy. Figure 36 presents a few examples of applying the MOG pooling strategy using AGD on
the SHREC11 dataset.
In order to demonstrate the effectiveness of our MOG pooling approach, we conduct three experiments
on the SHREC11 dataset: mesh classification based on CC-pooling with input vertex and edge features
(Section 9.4.1), mesh classification based on CC-pooling with input vertex features only (Section 9.4.2),
and point cloud classification based on CC-pooling with input vertex features only (Section 9.4.3). The
experiments in Sections 9.4.1 and 9.4.2 utilize the mesh structure in the SHREC11 dataset, whereas the
 12 The difficulty in these datasets is controlled by the compactness degree of the graph clusters; clusters in the ‘easy’ data have

more in-between cluster connections, while clusters in the ‘hard’ data are more isolated [37].


                                                                55
Topological deep learning                                        9   Implementation and numerical experiments




experiment in Section 9.4.3 utilizes its own point cloud version. In particular, we choose two simple CCNN
architectures shown in Figure 35(d), denoted by CCNNM OG1 and CCNNM OG2 , as opposed to the more
complicated architecture of CCNNSHREC in Figure 35(b). The main difference between CCNNM OG1 and
CCNNM OG2 is the choice of the input feature vectors as described next.

9.4.1   Mesh classification: CC-pooling with input vertex and edge features

In this experiment, we consider the vertex feature vector to be the position concatenated with the normal
vectors for each vertex in the underlying mesh. For the edge features, we compute the first ten eigenvectors
of the 1-Hodge Laplacian [92, 89] and attach a 10-dimensional feature vector to the edges of the underlying
mesh. The CC that we consider here is 3-dimensional, as it consists of the triangular mesh (vertices, edges
and faces) and of 3-cells. The 3-cells are obtained using the MOG algorithm, and are used for augmenting
each mesh. We calculate the 3-cells via the MOG algorithm using the AGD scalar function as input. We
conduct this experiment using the CCNN defined via the tensor diagram CCNNM OG1 given in Figure 35(d).
During training, we augment each mesh with ten additional meshes, with each of these additional meshes
being obtained by a random rotation as well as 0.1% noise perturbation to the vertex positions. We train
CCNNM OG1 for 100 epochs using a learning rate of 0.0002 and the standard cross-entropy loss, and obtain
an accuracy of 98.1%. While the accuracy of CCNNM OG1 is lower than the one we report for CCNNSHREC
(99.17%) in Table 3, we note that CCNNM OG1 requires a significantly smaller number of replications for
mesh augmentation to achieve a similar accuracy (CCNNM OG1 requires 10, whereas CCNNSHREC required
30 replications).

Architecture of CCNNM OG1 . The tensor diagram CCNNM OG1 of Figure 35(d) corresponds to a pooling
CCNN. In particular, CCNNM OG1 pushes forward the signal towards two different higher-order cells: the
faces of the mesh as well as the 3-cells obtained from the MOG algorithm.

9.4.2   Mesh classification: CC-pooling with input vertex features only

In this experiment, we consider the position and the normal vectors of the input vertices. The CC structure
that we consider is the underlying graph structure obtained from each mesh; i.e., we only use the vertices
and the edges, and ignore the faces. We augment this structure by 2-cells obtained via the MOG algorithm
using the AGD scalar function as input. We choose the network architecture to be relatively simpler than
CCNNM OG1 , and report it in Figure 35(d) as CCNNM OG2 . During training we augment each mesh with
10 additional meshes, with each of these additional meshes being obtained by a random rotation as well as
0.05% noise perturbation to the vertex positions. We train CCNNM OG2 for 100 epochs using a learning rate
of 0.0003 and the standard cross-entropy loss, and obtain an accuracy of 97.1%.

Architecture of CCNNM OG2 for mesh classification. The tensor diagram CCNNM OG2 of Figure 35(d)
corresponds to a pooling CCNN. In particular, CCNNM OG2 pushes forward the signal towards a single
2-cell obtained from the MOG algorithm. Observe that the overall architecture of CCNNM OG2 is similar in
principle to AlexNet [155], where convolutional layers are followed by pooling layers.

9.4.3   Point cloud classification: CC-pooling with input vertex features only

In this experiment, we consider point cloud classification on the SHREC11 dataset. The setup is similar in
principle to the one studied in Section 9.4.2 where we consider only the features supported on the vertices of
the point cloud as input. Specifically, for each mesh in the SHREC11 dataset, we sample 1, 000 points from
the surface of the mesh. Additionally, we estimate the normal vectors of the resulting point clouds using the
Point Cloud Utils package [265]. To build the CC structure, we first consider the k-nearest neighborhood
graph obtained from each point cloud using k = 7. We then augment this graph by 2-cells obtained via the
MOG algorithm using the AGD scalar function as input. We train the CCNNM OG2 shown in Figure 35(d).
During training, we augment each point cloud with 12 additional instances, each one of these instances being
obtained by random rotation. We train CCNNM OG2 for 100 epochs using a learning rate of 0.0003 and the
standard cross-entropy loss, and obtain an accuracy of 95.2% (see Table 3).


                                                     56
10    Related work                                                                     Topological deep learning




9.5    Ablation studies

In this section, we perform two ablation studies. The first ablation study reveals that pooling strategies
in CCNNs have a crucial effect on predictive performance. The second ablation study demonstrates that
CCNNs have better predictive capacity than GNNs; the advantage of CCNNs arises from their topological
pooling operations and from their ability to learn from topological features.

Pooling strategies in CCNNs. To evaluate the impact of the choice of pooling strategy on predictive
performance, we experiment with two pooling strategies using the SHREC11 classification dataset. The first
pooling strategy is the MOG algorithm described in Section 9.4; the results of this pooling strategy based
on CCNNM OG2 are discussed in Section 9.4.2 (97.1%). The second pooling strategy is briefly described as
follows. For each mesh, we consider the 2-dimensional CC obtained by considering each 1-hop neighborhood
to be the 1-cells in the CC and each 2-hop neighborhood to be the 2-cells in the CC. We train CCNNM OG2 ,
and obtain an accuracy of 89.2%, which is lower than 97.1%. These experiments suggest that the choice of
pooling strategy has a crucial effect on predictive performance.

Comparing CCNNs to GNNs in terms of predictive performance. Observe that CCNNSHREC has
topological features of dimension one and two as inputs. On the other hand, CCNNM OG2 has only vertex
features as input, but it learns the higher-order cell latent features by using the push-forward operation
that pushes the signal from 0-cells to the 2-cells obtained from the MOG algorithm. In both cases, using a
higher-order structure is essential for improving predictive performance, even though two different strategies
towards exploiting the higher-order structures are utilized. To support our claim, we run an experiment
in which we replace the pooling layer in CCNNM OG2 by the cochain operator induced by A0,1 , effectively
rendering the neural network as a GNN. In this setting, using the same setup as in experiment 9.4.2, we
obtain an accuracy of 84.56%. This experiment reveals the performance advantages of employing higher-order
structures, either by utilizing the input topological features supported on higher-order cells or via pooling
strategies that augment higher-order cells.


10     Related work

Topological deep learning (TDL) has recently emerged as a new research frontier that lies at the intersection
of several areas, including geometric and topological machine learning, and network science. To demonstrate
where TDL fits in the existing literature, we review a broad spectrum of prior works, and categorize them
into graph-based models, higher-order deep learning models, graph-based pooling, attention-based models,
and applied algebraic topology.

10.1    Graph-based models

Graph-based models have been widely used for modeling pairwise interactions (edges) between elements
(vertices) of different systems, including social systems (e.g., social network analysis) and biological systems
(e.g., protein-protein interactions), see [153, 142]. Based on their edge or vertex properties, graphs can
be classified as unweighted graphs (unweighted edges), weighted graphs (weighted edges), signed graphs
(signed edges), undirected or directed graphs (undirected or directed edges), and spatio-temporal graphs
(spatio-temporal vertices), as discussed in [268, 118]. Each of these graph types can be combined with neural
networks to form graph neural networks and model different interactions in various systems [268, 118]. For
example, unweighted and undirected graph-based models have been used for omic data mapping [3] and
mutual friendship detection in social networks [250]; weighted graph-based models have been widely used with
systems related to traffic forecasting [277, 129] and epidemiological modeling/forecasting [175, 184]; signed
graph-based models are suitable for tasks such as segmentation [16] and clustering [157, 102]; spatio-temporal
graph-based models can describe systems that are spatio-temporal in nature, such as human activity and
different types of motion [271, 213, 36].
As graph-based approaches that utilize single-layer or monolayer graphs cannot model multiple types of
relations between vertices in a network [268, 118], multilayer or multiplex networks have been proposed [67,


                                                      57
Topological deep learning                                                                       10   Related work




279, 151]. Similar to monolayer graphs, multiplex networks contain vertices and edges, but the edges exist in
separate layers, where each layer represents a specific type of interaction or relation. Multiplex networks
have been used in various applications, including multilayer modeling of the human brain [79, 4] and online
gaming [67]. All these types of networks can only model pairwise relations between vertices, motivating the
need for higher-order networks, as discussed in Section 2.2.

10.2   Higher-order deep learning models

In recent years, there has been an increasing interest in higher-order networks [189, 29, 41] due to the ability
of these networks to adequately capture higher-order interactions. Hodge-theoretic approaches, message
passing schemes, and skip connections have been developed for higher-order networks in the signal processing
and deep learning literature.
A Hodge-theoretic approach [174] over simplicial complexes has been introduced by [235, 21]. This effort
has been extended to hypergraphs by [24, 235] and to cell complexes by [230, 222]. The work of [220] has
defined an edge-based convolutional neural network by exploiting the 1-Hodge Laplacian operator for linear
filtering [23, 233, 21, 22, 235].
Convolutional operators and message-passing algorithms have been developed for higher-order neural networks.
For example, a convolutional operator on hypergraphs has been proposed by [143, 97, 7] and has been
investigated further by [266, 14, 143, 15, 106, 116, 112]. A unifying framework for learning on graphs and
hypergraphs has been proposed recently in [139]. The authors in [105] have introduced the so-called general
hypergraph neural networks, which constitute a multi-modal/multi-type data correlation modeling framework.
As for message passing on complexes, the work of [123] has introduced a higher-order message-passing
framework that encompasses those proposed by [91, 53, 109, 134] and has utilized various local neighborhood
aggregation schemes. In [193], recurrent simplicial neural networks have been proposed and applied to
trajectory prediction. The authors in [55] have addressed the challenge of processing signals supported
on multiple cell dimensions concurrently, by introducing a coupling multi-signal approach on higher-order
networks that utilizes the Dirac operator. Several simplicial and cellular neural networks have been introduced
recently, including [54, 230, 44, 222, 229, 28, 272]. For more details, the reader is referred to the recent survey
of [206] on TDL.
A generalization of skip connections [135, 224] to simplicial complexes has been introduced by [126], which
allows the training of higher-order deep neural networks. The authors in [197] have proposed a higher-order
graph neural network that takes into account higher-order graph structures at multiple scales. While these
methods allow for multi-way hierarchical coupling, the coupling is isotropic and weight differences within a
particular multi-way connection can not be learned. These limitations can be alleviated by attention-based
models.
Higher-order models have achieved promising performance in several real-world applications, including link
prediction [70, 126, 212], action recognition [260], visual classification [240], optimal homology generator
detection [147], time series [228], dynamical systems [182], spectral clustering [216], node classification [126],
and trajectory prediction [221, 34].

10.3   Attention-based models

Real-world relational data is large, unstructured, sparse and noisy. As a result, graph neural networks (GNNs)
may learn suboptimal data representations, and therefore may exhibit compromised performance [268, 78, 9].
To address these issues, various attention mechanisms [68] have been incorporated in GNNs, which allow
to learn neural architectures that detect the most relevant parts of a given graph while ignoring irrelevant
parts. Based on the used attention mechanism, existing graph attention approaches can be divided into
weight-based attention, similarity-based attention, and attention-guided walk [168].
The majority of attention-based mechanisms, with the exception of [112, 115, 113, 15, 148, 107], are designed
for graphs. For example, the attention model proposed by [115] is a generalization of the graph attention
model of [258]. In [113], the authors have utilized a model based on Hodge decomposition, similar to the
one suggested in [221], to introduce an attention model for simplicial complexes. The hypergraph attention


                                                        58
11   Conclusions                                                                           Topological deep learning




models introduced in [15, 148] provide alternative generalizations of the graph attention model of [258].
The aforementioned attention models neither allow nor combine higher-order attention blocks of entities of
different dimensions. This limits the space of neural architectures and the scope of applications of existing
attention models.

10.4   Graph-based pooling
Several attempts have been made to emulate the success of image-based pooling layers in the context of graphs.
Some of the early work employs popular graph clustering algorithms [158, 88] to achieve graph-based pooling
architectures [52]. Coarsening operations have been applied to graphs to attain the invariance properties
needed in learning tasks [275, 191, 104, 191]. The current state-of-the-art graph-based pooling approaches
mostly rely on dynamically learning the pooling needed for the learning task [120]. This includes spectral
methods [181], clustering methods such as DiffPool [275] and MinCut [38], top-K methods [103, 169, 281],
and hierarchical graph pooling [280, 140, 169, 281, 171, 205]. Pooling on higher-order networks remains
unstudied, with the exception of a general simplicial complex pooling strategy developed by [72] along the
lines of the proposal made by [120].

10.5   Applied algebraic topology
Although algebraic topology [133] is a relatively old field, applications of this field have only recently started to
crystallize [93, 58]. Indeed, topological constructions have been found to be natural tools for the formulation
of longstanding problems in many fields. For instance, persistent homology [93] has been successful at finding
solutions to various complex data problems [10, 17, 48, 64, 76, 77, 111, 159, 163, 164, 165, 166, 180, 202,
225]. Recent years have witnessed increased interest in the role of topology in machine learning and data
science [86]. Topology-based machine learning models have been applied in many areas, including topological
signatures of data [40, 63, 218], neuroscience [76, 77, 111, 163, 164, 165, 166], bioscience [66, 84, 176, 202,
251, 252], the study of graphs [18, 65, 263, 137, 210, 211, 124], and time-varying setups [94, 183, 209].
Topological data analysis (TDA) [93, 58, 86, 178, 108] has emerged as a scientific area that harnesses
topological tools to analyze data and develop machine learning algorithms. TDA has mostly focused on
enhancing existing machine learning models [136, 100, 261, 170, 35, 73] or on improving the explainablity of
deep learning models [96, 59, 179]. Our work introduces combinatorial complexes (CCs) as a generalized
higher-order network on which deep learning models can be defined and studied in a unifying manner.
Hence, our work expands TDA by formalizing deep learning notions in topological terms and by realizing
constructions in TDA (e.g., mapper [244]) in terms of our TDL framework. The construction of CCs and
of combinatorial complex neural networks (CCNNs), which are neural networks defined on CCs, is inspired
by classical notions in algebraic topology [133] and in topological quantum field theory [255], and by recent
advances in TDA [57, 58, 62, 60, 61, 63, 74] as applied to machine learning [214, 86].

11     Conclusions
We have established a topological deep learning (TDL) framework that enables the learning of representations
for data supported on topological domains. To this end, we have introduced combinatorial complexes (CCs)
as a new topological domain to model and characterize the main components of TDL. Our framework provides
a unification for many concepts that may be perceived as separate in the current literature. Specifically, we
can reduce most deep learning architectures presented thus far in the literature to particular instances of
combinatorial complex neural networks (CCNNs), based on computational operations defined on CCs. Our
framework thus provides a platform for a more systematic exploration and comparison of the large space of
deep learning protocols on topological spaces.

Limitations. This work has laid out the foundations of a novel TDL framework. While TDL has great
potential, similar to other novel learning frameworks, there are limitations, many of which are still not
well-understood. Specifically, some known limitations involve:
• Computational complexity: The primary challenge for moving from graphs to richer topological domains is
  the combinatorial increase in the complexity to store and process data defined on such domains. Training


                                                         59
Topological deep learning                                                                                        11   Conclusions




  a TDL network can be a computationally intensive task, requiring careful consideration of neighborhood
  functions and generalization performance. TDL networks also require a large amount of memory, especially
  when working with a large number of matrices during network construction. The topology of the network
  can also increase the computational complexity of training.
• The choice of the neural network architecture: Choosing the appropriate neural network architecture for a
  given dataset and a given learning task can be challenging. The performance of a TDL network can be
  highly dependent on the choice of architecture and its hyperparameters.
• Interpretability and explainability: The architecture of a TDL network can make it difficult to interpret
  the learnt representations and understand how the network is making predictions.
• Limited availability of datasets: TDL networks require topological data, which we have found to be of
  limited availability. Ad hoc conversions of existing data to include higher-order relations may not always
  be ideal.

Future work. Aforementioned limitations leave ample room for future studies, making the realization of
the full potential of TDL an interesting endeavour. While the flexible definition of CCs, as compared to, e.g.,
simplicial complexes, already provides some mitigation to the associated computational challenges, improving
the scaling of CCNNs even further will require the exploration of sparsification techniques, randomization and
other algorithmic improvements. Besides addressing the aforementioned limitations, promising directions not
treated within this paper include explorations of directed [12], weighted [28], multilayer [190], and time-varying
dynamic topological domains [253, 5, 274]. There are also several issues related to the selection of the most
appropriate topological domain for a given dataset in the first place, which need further exploration in
the future. Additionally, there is a need, but also a research opportunity, to better understand CCNN
architectures from a theoretical perspective. This could in turn lead to better architectures. To illustrate
this point, consider graph neural networks (GNNs) based on message passing [109], which have recently
been shown to be as powerful as the Weisfeiler–Lehman isomorphism test13 [270, 186, 197]. This connection
between GNNs and classical graph-based combinatorial invariants has driven theoretical developments of the
graph isomorphism problem and has inspired new architectures [270, 186, 6, 47]. We expect that connecting
similar developments will also be important for TDL.
The topological viewpoint we adopt brings about many interesting properties. For example, we are able to
model other types of topological inductive biases in our computational architectures, such as the properties
that do not change under different discretizations of the underlying domain, e.g., the Euler characteristic
that is commonly used to distinguish topological spaces. While isomorphisms are the primary equivalence
relation in graph theory, homeomorphisms and topological equivalence are more relevant for data defined on
topological spaces, and invariants under homeomorphisms have different machine learning applications14 .
Homeomorphism equivalence is more relevant in various applications in which domain discretization is an
artifact of data processing, and not an intrinsic part of the data [239]. Further, homeomorphism equivalence
translates to a similarity question between two structures. Indeed, topological data analysis has been
extensively utilized towards addressing the problem of similarity between meshes [87, 128]. In geometric
data processing, neural network architectures that are agnostic to mesh discretization are often desirable
and perform better in practice [239]. We anticipate that the development of TDL models will open up new
avenues to explore topological invariants across topological domains.

Acknowledgments

M. H. acknowledges support from the National Science Foundation, award DMS-2134231. G. Z. is currently
affiliated with National Institutes of Health (NIH), but the core of this research was done while being
associated with the University of South Florida (USF). This article reports contributions of the authors and
does not represent the views of NIH, or the United States Government. N. M. acknowledges support from
the National Science Foundation, Award DMS-2134241. T. B. acknowledges support from the Engineering
and Physical Sciences Research Council [grant EP/X011364/1]. T. K. D. acknowledges support from the
National Science Foundation, Award CCF 2049010. N. L. acknowledges support from the Roux Institute
  13 The Weisfeiler–Lehman isomorphism test [264] is a widely-used graph isomorphism algorithm that provides a coloring of a

graph’s vertices, and this coloring gives a necessary condition for two graphs to be isomorphic.
  14 Intuitively, two topological spaces are equivalent if one of them can be deformed to the other via a continuous transformation.




                                                                60
References                                                                         Topological deep learning




and the Harold Alfond Foundation. R. W. acknowledges support from the National Science Foundation,
Award DMS-2134178. P. R. acknowledges support from the National Science Foundation, Award IIS-2316496.
M. T. S. acknowledges funding by the Ministry of Culture and Science (MKW) of the German State of North
Rhine-Westphalia (NRW Rückkehrprogramm) and the European Union (ERC, HIGH-HOPeS, 101039827).
Views and opinions expressed are however those of M. T. S. only and do not necessarily reflect those of the
European Union or the European Research Council Executive Agency; neither the European Union nor the
granting authority can be held responsible for them.
The authors would like to thank Mathilde Papillon and Sophia Sanborn for helping improve Figure 9 and for
the insightful discussions on the development of tensor diagrams.

References
  [1] Peter Abramenko and Kenneth S. Brown. Buildings: theory and applications. Vol. 248. Springer Science
      & Business Media, 2008.
  [2] Gerardo Adesso. “GPT4: the ultimate brain”. In: Authorea Preprints (2022).
  [3] David Amar and Ron Shamir. “Constructing module maps for integrated analysis of heterogeneous
      biological networks”. In: Nucleic Acids Research 42.7 (2014), pp. 4208–4219.
  [4] D. V. Anand and Moo K. Chung. “Hodge Laplacian of brain networks”. In: IEEE Transactions on
      Medical Imaging (2023).
  [5] Md Sayeed Anwar and Dibakar Ghosh. “Stability of synchronization in simplicial complexes with
      multiple interaction layers”. In: Phys. Rev. E 106 (3 Sept. 2022), p. 034314.
  [6] Vikraman Arvind et al. “On Weisfeiler-Leman invariance: subgraph counts and related graph proper-
      ties”. In: Journal of Computer and System Sciences 113 (2020), pp. 42–59.
  [7] Devanshu Arya and Marcel Worring. “Exploiting relational information in social networks using
      geometric deep learning on hypergraphs”. In: Proceedings of the 2018 ACM on International Conference
      on Multimedia Retrieval. 2018, pp. 117–125.
  [8] Michael Aschbacher. “Combinatorial cell complexes”. In: Progress in Algebraic Combinatorics. Mathe-
      matical Society of Japan, 1996, pp. 1–80.
  [9] Nurul A. Asif et al. “Graph neural network: a comprehensive review on non-Euclidean space”. In:
      IEEE Access 9 (2021), pp. 60588–60606.
 [10] Marco Attene, Silvia Biasotti, and Michela Spagnuolo. “Shape understanding by contour-driven
      retiling”. In: The Visual Computer 19.2 (2003), pp. 127–138.
 [11] Matan Atzmon, Haggai Maron, and Yaron Lipman. “Point convolutional neural networks by extension
      operators”. In: ACM Trans. Graph. 37.4 (July 2018).
 [12] Giorgio Ausiello and Luigi Laura. “Directed hypergraphs: introduction and fundamental algorithms—a
      survey”. In: Theoretical Computer Science 658 (2017), pp. 293–306.
 [13] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. “Neural machine translation by jointly
      learning to align and translate”. In: arXiv preprint arXiv:1409.0473 (2014).
 [14] Junjie Bai et al. “Multi-scale representation learning on hypergraph for 3D shape retrieval and
      recognition”. In: IEEE Transactions on Image Processing 30 (2021), pp. 5327–5338.
 [15] Song Bai, Feihu Zhang, and Philip H. S. Torr. “Hypergraph convolution and hypergraph attention”.
      In: Pattern Recognition 110 (2021), p. 107637.
 [16] Alberto Bailoni et al. “GASP, a generalized framework for agglomerative clustering of signed graphs
      and its application to instance segmentation”. In: IEEE Conf. Comput. Vis. Pattern Recog. 2022,
      pp. 11645–11655.
 [17] Chandrajit L. Bajaj, Valerio Pascucci, and Daniel R. Schikore. “The contour spectrum”. In: Proceedings
      of the 8th Conference on Visualization ’97. IEEE Computer Society Press. 1997, 167–ff.
 [18] Maria Bampasidou and Thanos Gentimis. “Modeling collaborations with persistent homology”. In:
      arXiv preprint arXiv:1403.5346 abs/1403.5346 (2014).
 [19] Xiaoge Bao et al. “Impact of basic network motifs on the collective response to perturbations”. In:
      Nature Communications 13.1 (2022), p. 5301.


                                                    61
Topological deep learning                                                                         References




 [20] Albert-László Barabási. “Network science”. In: Philosophical Transactions of the Royal Society A:
      Mathematical, Physical and Engineering Sciences 371.1987 (2013), p. 20120375.
 [21] Sergio Barbarossa and Stefania Sardellitti. “Topological signal processing over simplicial complexes”.
      In: IEEE Transactions on Signal Processing 68 (2020), pp. 2992–3007.
 [22] Sergio Barbarossa and Stefania Sardellitti. “Topological signal processing: making sense of data
      building on multiway relations”. In: IEEE Signal Processing Magazine 37.6 (2020), pp. 174–183.
 [23] Sergio Barbarossa, Stefania Sardellitti, and Elena Ceci. “Learning from signals defined over simplicial
      complexes”. In: 2018 IEEE Data Science Workshop (DSW). IEEE. 2018, pp. 51–55.
 [24] Sergio Barbarossa and Mikhail Tsitsvero. “An introduction to hypergraph signal processing”. In: 2016
      IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE. 2016,
      pp. 6425–6429.
 [25] Tathagata Basak. “Combinatorial cell complexes and poincaré duality”. In: Geometriae Dedicata 147.1
      (2010), pp. 357–387.
 [26] Peter Battaglia et al. “Interaction Networks for Learning about Objects, Relations and Physics”. In:
      Proceedings of the 30th International Conference on Neural Information Processing Systems. NIPS’16.
      Barcelona, Spain: Curran Associates Inc., 2016, pp. 4509–4517.
 [27] Peter W. Battaglia et al. “Relational inductive biases, deep learning, and graph networks”. In: arXiv
      preprint arXiv:1806.01261 (2018).
 [28] Claudio Battiloro et al. “Topological signal processing over weighted simplicial complexes”. In: arXiv
      preprint arXiv:2302.08561 (2023).
 [29] Federico Battiston et al. “Networks beyond pairwise interactions: structure and dynamics”. In: Physics
      Reports 874 (2020), pp. 1–92.
 [30] Federico Battiston et al. “The physics of higher-order interactions in complex systems”. In: Nature
      Physics 17.10 (2021), pp. 1093–1098.
 [31] Dominique Beaini et al. “Directional graph networks”. In: Int. Conf. Mach. Learn. PMLR. 2021,
      pp. 748–758.
 [32] Austin R. Benson, David F. Gleich, and Desmond J. Higham. “Higher-order network analysis takes
      off, fueled by classical ideas and new data”. In: arXiv preprint arXiv:2103.05031 (2021).
 [33] Austin R. Benson, David F. Gleich, and Jure Leskovec. “Higher-order organization of complex networks”.
      In: Science 353.6295 (2016), pp. 163–166.
 [34] Austin R. Benson et al. “Simplicial closure and higher-order link prediction”. In: Proceedings of the
      National Academy of Sciences 115.48 (2018), E11221–E11230.
 [35] Aicha BenTaieb and Ghassan Hamarneh. “Topology aware fully convolutional networks for histology
      gland segmentation”. In: Medical Image Computing and Computer-Assisted Intervention–MICCAI
      2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19.
      Springer. 2016, pp. 460–468.
 [36] Uttaran Bhattacharya et al. “STEP: spatial temporal graph convolutional networks for emotion
      perception from Gaits”. In: Proceedings of the AAAI Conference on Artificial Intelligence 34.02 (Apr.
      2020), pp. 1342–1350.
 [37] Filippo Maria Bianchi, Claudio Gallicchio, and Alessio Micheli. “Pyramidal reservoir graph neural
      network”. In: Neurocomputing 470 (2022), pp. 389–404.
 [38] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. “Spectral clustering with graph neural
      networks for graph pooling”. In: Int. Conf. Mach. Learn. PMLR. 2020, pp. 874–883.
 [39] Ginestra Bianconi. Higher-order networks. Cambridge University Press, 2021.
 [40] Silvia Biasotti et al. “Describing shapes by geometrical-topological properties of real functions”. In:
      ACM Computing Surveys (CSUR) 40.4 (2008), p. 12.
 [41] Christian Bick et al. “What are higher-order networks?” In: arXiv preprint arXiv:2104.11329 (2021).
 [42] Jacob Charles Wright Billings et al. “Simplex2Vec embeddings for community detection in simplicial
      complexes”. In: arXiv preprint arXiv:1906.09068 (2019).
 [43] Tolga Birdal et al. “Intrinsic dimension, persistent homology and generalization in neural networks”.
      In: Advances in Neural Information Processing Systems 34 (2021), pp. 6776–6789.


                                                     62
References                                                                         Topological deep learning




 [44] Cristian Bodnar et al. “Weisfeiler and Lehman go cellular: CW networks”. In: Advances in Neural
      Information Processing Systems 34 (2021), pp. 2625–2640.
 [45] Davide Boscaini et al. “Learning class-specific descriptors for deformable shapes using localized
      spectral convolutional networks”. In: Computer Graphics Forum. Vol. 34. 5. Wiley Online Library.
      2015, pp. 13–23.
 [46] Davide Boscaini et al. “Learning shape correspondence with anisotropic convolutional neural networks”.
      In: Advances in Neural Information Processing Systems. 2016, pp. 3189–3197.
 [47] Giorgos Bouritsas et al. “Improving graph neural network expressivity via subgraph isomorphism
      counting”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence 45.1 (2023), pp. 657–
      668.
 [48] Roger L. Boyell and Henry Ruston. “Hybrid techniques for real-time radar simulation”. In: Proceedings
      of the November 12-14, 1963, Fall Joint Computer Conference. ACM. 1963, pp. 445–458.
 [49] Michael M. Bronstein et al. “Geometric deep learning: going beyond Euclidean data”. In: IEEE Signal
      Processing Magazine 34.4 (2017), pp. 18–42.
 [50] Michael M. Bronstein et al. “Geometric deep learning: grids, groups, graphs, geodesics, and gauges”.
      In: arXiv preprint arXiv:2104.13478 (2021).
 [51] Ronald Brown. Topology and groupoids. 2006.
 [52] Joan Bruna et al. “Spectral networks and locally connected networks on graphs”. In: Proceedings of the
      2nd International Conference on Learning Representations. Ed. by Yoshua Bengio and Yann LeCun.
      ICLR 2014. Banff, AB, Canada, 2014.
 [53] Eric Bunch et al. “Simplicial 2-complex convolutional neural nets”. In: NeurIPS Workshop on Topolog-
      ical Data Analysis and Beyond (2020).
 [54] Thomas F. Burns and Tomoki Fukai. “Simplicial Hopfield networks”. In: The Eleventh International
      Conference on Learning Representations.
 [55] Lucille Calmon, Michael T. Schaub, and Ginestra Bianconi. “Higher-order signal processing with the
      Dirac operator”. In: 56th Asilomar Conference on Signals, Systems, and Computers. Pacific Grove,
      CA, USA: IEEE, 2022, pp. 925–929.
 [56] Wenming Cao et al. “A comprehensive survey on geometric deep learning”. In: IEEE Access 8 (2020),
      pp. 35929–35949.
 [57] Erik Carlsson, Gunnar Carlsson, and Vin De Silva. “An algebraic topological method for feature
      identification”. In: International Journal of Computational Geometry & Applications 16.04 (2006),
      pp. 291–314.
 [58] Gunnar Carlsson. “Topology and data”. In: Bulletin of the American Mathematical Society 46.2 (2009),
      pp. 255–308.
 [59] Gunnar Carlsson and Rickard Brüel Gabrielsson. “Topological approaches to deep learning”. In:
      Topological Data Analysis: The Abel Symposium 2018. Springer. Springer, 2020, pp. 119–146.
 [60] Gunnar Carlsson and Facundo Mémoli. “Persistent clustering and a theorem of J. Kleinberg”. In:
      arXiv preprint arXiv:0808.2241 (2008).
 [61] Gunnar Carlsson and Afra Zomorodian. “The theory of multidimensional persistence”. In: Discrete &
      Computational Geometry 42.1 (2009), pp. 71–93.
 [62] Gunnar Carlsson et al. “On the local behavior of spaces of natural images”. In: Int. J. Comput. Vis.
      76.1 (2008), pp. 1–12.
 [63] Gunnar Carlsson et al. “Persistence barcodes for shapes”. In: International Journal of Shape Modeling
      11.02 (2005), pp. 149–187.
 [64] Hamish Carr, Jack Snoeyink, and Michiel van de Panne. “Simplifying flexible isosurfaces using local
      geometric measures”. In: IEEE Visualization. IEEE. 2004, pp. 497–504.
 [65] C. J. Carstens and K. J. Horadam. “Persistent homology of collaboration networks”. In: Mathematical
      Problems in Engineering 2013 (2013).
 [66] Joseph Minhow Chan, Gunnar Carlsson, and Raul Rabadan. “Topology of viral evolution”. In:
      Proceedings of the National Academy of Sciences 110.46 (2013), pp. 18566–18571.


                                                    63
Topological deep learning                                                                       References




 [67] Yaomin Chang et al. “GraphRR: a multiplex graph based reciprocal friend recommender system with
      applications on online gaming service”. In: Knowledge-Based Systems 251 (2022), p. 109187.
 [68] Sneha Chaudhari et al. “An attentive survey of attention models”. In: ACM Transactions on Intelligent
      Systems and Technology (TIST) 12.5 (2021), pp. 1–32.
 [69] Yunpeng Chen et al. “Graph-based global reasoning networks”. In: IEEE Conf. Comput. Vis. Pattern
      Recog. 2019, pp. 433–442.
 [70] Yuzhou Chen, Yulia R. Gel, and H. Vincent Poor. “BScNets: block simplicial complex neural networks”.
      In: Proceedings of the AAAI Conference on Artificial Intelligence 36.6 (June 2022), pp. 6333–6341.
 [71] Edward Choi et al. “GRAM: graph-based attention model for healthcare representation learning”. In:
      Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data
      Mining. 2017, pp. 787–795.
 [72] Domenico Mattia Cinque, Claudio Battiloro, and Paolo Di Lorenzo. “Pooling strategies for simplicial
      convolutional networks”. In: arXiv preprint arXiv:2210.05490 (2022).
 [73] James R. Clough et al. “Explicit topological priors for deep-learning based image segmentation using
      persistent homology”. In: Information Processing in Medical Imaging: 26th International Conference,
      IPMI 2019, Hong Kong, China, June 2–7, 2019, Proceedings 26. Springer. 2019, pp. 16–28.
 [74] Anne Collins et al. “A barcode shape descriptor for curve point cloud data”. In: Computers & Graphics
      28.6 (2004), pp. 881–894.
 [75] Keenan Crane et al. “Digital geometry processing with discrete exterior calculus”. In: ACM SIGGRAPH
      2013 Courses. Association for Computing Machinery, 2013, pp. 1–126.
 [76] Carina Curto. “What can topology tell us about the neural code?” In: Bulletin of the American
      Mathematical Society 54.1 (2017), pp. 63–78.
 [77] Y. Dabaghian et al. “A topological paradigm for hippocampal spatial map formation using persistent
      homology”. In: PLoS Computational Biology 8.8 (2012), e1002581.
 [78] Enyan Dai, Charu Aggarwal, and Suhang Wang. “NRGNN: learning a label noise resistant graph
      neural network on sparsely and noisily labeled graphs”. In: Proceedings of the 27th ACM SIGKDD
      Conference on Knowledge Discovery & Data Mining. 2021, pp. 227–236.
 [79] Manlio De Domenico. “Multilayer modeling and analysis of human brain networks”. In: GigaScience
      6.5 (Feb. 2017).
 [80] Manlio De Domenico et al. “The physics of spreading processes in multilayer networks”. In: Nature
      Physics 12.10 (2016), pp. 901–906.
 [81] Haowen Deng, Tolga Birdal, and Slobodan Ilic. “PPFNet: global context aware local features for
      robust 3D point matching”. In: IEEE Conf. Comput. Vis. Pattern Recog. 2018, pp. 195–205.
 [82] Songgaojun Deng et al. “Cola-GNN: cross-location attention based graph neural networks for long-
      term ILI prediction”. In: Proceedings of the 29th ACM International Conference on Information &
      Knowledge Management. 2020, pp. 245–254.
 [83] Mathieu Desbrun, Eva Kanso, and Yiying Tong. “Discrete differential forms for computational
      modeling”. In: Discrete Differential Geometry. Springer, 2008, pp. 287–324.
 [84] D. DeWoskin et al. “Applications of computational homology to the analysis of treatment response in
      breast cancer patients”. In: Topology and its Applications 157.1 (2010), pp. 157–164.
 [85] Tamal K. Dey, Facundo Mémoli, and Yusu Wang. “Multiscale mapper: Topological summarization via
      codomain covers”. In: Proceedings of the Twenty-Seventh Annual ACM-SIAM Symposium on Discrete
      Algorithms. SIAM. 2016, pp. 997–1013.
 [86] Tamal K. Dey and Yusu Wang. Computational topology for data analysis. Cambridge University Press,
      2022.
 [87] Tamal K. Dey et al. “Persistent heat signature for pose-oblivious matching of incomplete models”. In:
      Comput. Graph. Forum. Vol. 29. 5. Wiley Online Library. 2010, pp. 1545–1554.
 [88] Inderjit S. Dhillon, Yuqiang Guan, and Brian Kulis. “Weighted graph cuts without eigenvectors a
      multilevel approach”. In: IEEE Trans. Pattern Anal. Mach. Intell. 29.11 (2007), pp. 1944–1957.
 [89] Jozef Dodziuk. “Finite-difference approach to the Hodge theory of harmonic forms”. In: American
      Journal of Mathematics 98.1 (1976), pp. 79–104.


                                                    64
References                                                                         Topological deep learning




 [90] Benjamin Dupuis, George Deligiannidis, and Umut Şimşekli. “Generalization bounds with data-
      dependent fractal dimensions”. In: arXiv preprint arXiv:2302.02766 (2023).
 [91] Stefania Ebli, Michaël Defferrard, and Gard Spreemann. “Simplicial neural networks”. In: NeurIPS
      Workshop on Topological Data Analysis and Beyond (2020).
 [92] Beno Eckmann. “Harmonische funktionen und randwertaufgaben in einem komplex”. In: Commentarii
      Mathematici Helvetici 17.1 (1944), pp. 240–255.
 [93] Herbert Edelsbrunner and John Harer. Computational topology: an introduction. American Mathemat-
      ical Soc., 2010.
 [94] Herbert Edelsbrunner et al. “Time-varying Reeb graphs for continuous space-time data”. In: Proceedings
      of the Twentieth Annual Symposium on Computational Geometry. ACM. 2004, pp. 366–372.
 [95] Athanasios Efthymiou et al. “Graph neural networks for knowledge enhanced visual representation of
      paintings”. In: arXiv preprint arXiv:2105.08190 (2021).
 [96] Hamza Elhamdadi, Shaun Canavan, and Paul Rosen. “AffectiveTDA: using topological data analysis
      to improve analysis and explainability in affective computing”. In: IEEE Transactions on Visualization
      and Computer Graphics 28.1 (2021), pp. 769–779.
 [97] Yifan Feng et al. “Hypergraph neural networks”. In: Proceedings of the AAAI Conference on Artificial
      Intelligence 33.01 (2019), pp. 3558–3565.
 [98] Massimo Ferri, Dott Mattia G. Bergomi, and Lorenzo Zu. “Simplicial complexes from graphs towards
      graph persistence”. In: arXiv preprint arXiv:1805.10716 (2018).
 [99] Matthias Fey and Jan Eric Lenssen. “Fast graph representation learning with PyTorch Geometric”.
      In: arXiv preprint arXiv:1903.02428 (2019).
[100] Rickard Brüel Gabrielsson et al. “A Topology Layer for Machine Learning”. In: Proceedings of the
      Twenty Third International Conference on Artificial Intelligence and Statistics. Ed. by Silvia Chiappa
      and Roberto Calandra. Vol. 108. Proc. of Mach. Learn Res. PMLR, 2020, pp. 1553–1563.
[101] Claudio Gallicchio and Alessio Micheli. “Graph echo state networks”. In: The 2010 International Joint
      Conference on Neural Networks (IJCNN). IEEE. 2010, pp. 1–8.
[102] Jean Gallier. “Spectral theory of unsigned and signed graphs. Applications to graph clustering: a
      survey”. In: arXiv preprint arXiv:1601.04692 (2016).
[103] Hongyang Gao and Shuiwang Ji. “Graph U-Nets”. In: Int. Conf. Mach. Learn. PMLR. 2019, pp. 2083–
      2092.
[104] Hongyang Gao, Yi Liu, and Shuiwang Ji. “Topology-aware graph pooling networks”. In: IEEE Trans.
      Pattern Anal. Mach. Intell. 43.12 (2021), pp. 4512–4518.
[105] Yue Gao et al. “HGNN+: general hypergraph neural networks”. In: IEEE Transactions on Pattern
      Analysis and Machine Intelligence (2022).
[106] Yue Gao et al. “Hypergraph learning: methods and practices”. In: IEEE Transactions on Pattern
      Analysis and Machine Intelligence (2020).
[107] Dobrik Georgiev Georgiev, Marc Brockschmidt, and Miltiadis Allamanis. “HEAT: hyperedge attention
      networks”. In: Transactions on Machine Learning Research (2022).
[108] Robert W. Ghrist. Elementary applied topology. Vol. 1. Createspace Seattle, 2014.
[109] Justin Gilmer et al. “Neural message passing for quantum chemistry”. In: Int. Conf. Mach. Learn.
      PMLR. 2017, pp. 1263–1272.
[110] Benjamin Girault, Shrikanth S. Narayanan, and Antonio Ortega. “Towards a definition of local
      stationarity for graph signals”. In: 2017 IEEE International Conference on Acoustics, Speech and
      Signal Processing (ICASSP). IEEE. 2017, pp. 4139–4143.
[111] Chad Giusti, Robert Ghrist, and Danielle S. Bassett. “Two’s company, three (or more) is a simplex:
      algebraic-topological tools for understanding higher-order structure in neural data”. In: Journal of
      Computational Neuroscience 41 (2016), p. 1.
[112] Lorenzo Giusti et al. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).
[113] Lorenzo Giusti et al. “Simplicial attention networks”. In: arXiv preprint arXiv:2203.07485 (2022).
[114] Fernando de Goes, Mathieu Desbrun, and Yiying Tong. “Vector field processing on triangle meshes”.
      In: ACM SIGGRAPH 2016 Courses. 2016, pp. 1–49.


                                                    65
Topological deep learning                                                                          References




[115] Christopher Wei Jin Goh, Cristian Bodnar, and Pietro Lio. “Simplicial attention networks”. In: ICLR
      2022 Workshop on Geometrical and Topological Representation Learning. 2022.
[116] Xue Gong, Desmond J. Higham, and Konstantinos Zygalakis. “Generative hypergraph models and
      spectral embedding”. In: Scientific Reports 13.1 (2023), p. 540.
[117] Ian Goodfellow et al. Deep learning. Vol. 1. MIT press Cambridge, 2016.
[118] Palash Goyal and Emilio Ferrara. “Graph embedding techniques, applications, and performance: a
      survey”. In: Knowledge-Based Systems 151 (2018), pp. 78–94.
[119] Leo J. Grady and Jonathan R. Polimeni. Discrete calculus: applied analysis on graphs for computational
      science. Vol. 3. Springer, 2010.
[120] Daniele Grattarola et al. “Understanding pooling in graph neural networks”. In: IEEE Transactions
      on Neural Networks and Learning Systems (2022).
[121] Celia Hacker. “K-simplex2vec: a simplicial extension of node2vec”. In: NeurIPS workshop on Topological
      Data Analysis and Beyond (2020).
[122] Aric Hagberg, Pieter Swart, and Daniel S Chult. Exploring network structure, dynamics, and function
      using NetworkX. Tech. rep. Los Alamos National Lab.(LANL), Los Alamos, NM (United States), 2008.
[123] Mustafa Hajij, Kyle Istvan, and Ghada Zamzmi. “Cell complex neural networks”. In: NeurIPS 2020
      Workshop TDA and Beyond (2020).
[124] Mustafa Hajij and Paul Rosen. “An efficient data retrieval parallel Reeb graph algorithm”. In:
      Algorithms 13.10 (2020), p. 258.
[125] Mustafa Hajij, Bei Wang, and Paul Rosen. “MOG: mapper on graphs for relationship preserving
      clustering”. In: arXiv preprint arXiv:1804.11242 (2018).
[126] Mustafa Hajij et al. “High skip networks: a higher order generalization of skip connections”. In: ICLR
      2022 Workshop on Geometrical and Topological Representation Learning. 2022.
[127] Mustafa Hajij et al. “Simplicial complex representation learning”. In: Machine Learning on Graphs
      (MLoG) Workshop at 15th ACM International WSD Conference (2022).
[128] Mustafa Hajij et al. “Visual detection of structural changes in time-varying graphs using persistent
      homology”. In: 2018 IEEE Pacific Visualization Symposium (PacificVis). IEEE. 2018, pp. 125–134.
[129] Hatem F. Halaoui. “Smart traffic online system (STOS): presenting road networks with time-weighted
      graphs”. In: 2010 International Conference on Information Society. IEEE. 2010, pp. 349–356.
[130] William L. Hamilton, Rex Ying, and Jure Leskovec. “Representation Learning on Graphs: Methods
      and Applications”. In: IEEE Data Eng. Bull. 40.3 (2017), pp. 52–74.
[131] Rana Hanocka et al. “MeshCNN: a network with an edge”. In: ACM Trans. Graph. 38.4 (2019),
      pp. 1–12.
[132] Jakob Hansen and Robert Ghrist. “Toward a spectral theory of cellular sheaves”. In: Journal of Applied
      and Computational Topology 3.4 (2019), pp. 315–358.
[133] Allen Hatcher. Algebraic topology. Cambridge University Press, 2005.
[134] Mikhail Hayhoe et al. “Stable and transferable hyper-graph neural networks”. In: arXiv preprint
      arXiv:2211.06513 (2022).
[135] Kaiming He et al. “Deep Residual Learning for Image Recognition”. In: 2016 IEEE Conference on
      Computer Vision and Pattern Recognition (CVPR). 2016, pp. 770–778.
[136] Christoph Hofer et al. “Deep learning with topological signatures”. In: Adv. Neural Inform. Process.
      Syst. 2017, pp. 1634–1644.
[137] Danijela Horak, Slobodan Maletić, and Milan Rajković. “Persistent homology of complex networks”.
      In: Journal of Statistical Mechanics: Theory and Experiment (2009), P03034.
[138] Jiahui Huang et al. “Multiway non-rigid point cloud registration via learned functional map synchro-
      nization”. In: IEEE Trans. Pattern Anal. Mach. Intell. (2022).
[139] Jing Huang and Jie Yang. “UniGNN: a unified framework for graph and hypergraph neural networks”.
      In: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI. 2021.
[140] Jingjia Huang et al. “AttPool: towards hierarchical feature representation in graph convolutional
      networks via attention mechanism”. In: Int. Conf. Comput. Vis. 2019, pp. 6480–6489.


                                                     66
References                                                                           Topological deep learning




[141] Takeshi D. Itoh, Takatomi Kubo, and Kazushi Ikeda. “Multi-level attention pooling for graph neural
      networks: unifying graph representations with multiple localities”. In: Neural Networks 145 (2022),
      pp. 356–373.
[142] Kanchan Jha, Sriparna Saha, and Hiteshi Singh. “Prediction of protein–protein interaction using graph
      neural networks”. In: Scientific Reports 12.1 (2022), pp. 1–12.
[143] Jianwen Jiang et al. “Dynamic hypergraph neural networks.” In: IJCAI. 2019, pp. 2635–2641.
[144] Weiwei Jiang and Jiayun Luo. “Graph neural network for traffic forecasting: a survey”. In: Expert
      Systems with Applications (2022), p. 117921.
[145] Fabian Jogl. “Do we need to improve message passing? Improving graph neural networks with graph
      transformations”. PhD thesis. Vienna University of Technology, 2022.
[146] Cliff A Joslyn et al. “Hypernetwork science: from multidimensional networks to computational topology”.
      In: Unifying Themes in Complex Systems X: Proceedings of the Tenth International Conference on
      Complex Systems. Springer. 2021, pp. 377–392.
[147] Alexandros D. Keros, Vidit Nanda, and Kartic Subr. “Dist2Cycle: a simplicial neural network for
      homology localization”. In: Proceedings of the AAAI Conference on Artificial Intelligence 36.7 (June
      2022), pp. 7133–7142.
[148] Eun-Sol Kim et al. “Hypergraph attention networks for multimodal learning”. In: IEEE Conf. Comput.
      Vis. Pattern Recog. 2020, pp. 14581–14590.
[149] Vladimir G Kim et al. “Möbius transformations for global intrinsic symmetry analysis”. In: Computer
      Graphics Forum 29.5 (2010), pp. 1689–1700.
[150] Thomas N. Kipf and Max Welling. “Semi-supervised classification with graph convolutional networks”.
      In: arXiv preprint arXiv:1609.02907 (2016).
[151] Mikko Kivelä et al. “Multilayer networks”. In: Journal of complex networks 2.3 (2014), pp. 203–271.
[152] Reinhard Klette. “Cell complexes through time”. In: Vision Geometry IX. Vol. 4117. SPIE. 2000,
      pp. 134–145.
[153] David Knoke and Song Yang. Social network analysis. SAGE publications, 2019.
[154] Iasonas Kokkinos et al. “Intrinsic shape context descriptors for deformable shapes”. In: IEEE Conf.
      Comput. Vis. Pattern Recog. IEEE. 2012, pp. 159–166.
[155] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. “Imagenet classification with deep convolu-
      tional neural networks”. In: Communications of the ACM 60.6 (2017), pp. 84–90.
[156] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. “ImageNet classification with deep convo-
      lutional neural networks”. In: Advances in Neural Information Processing Systems. 2012, pp. 1097–
      1105.
[157] Jérôme Kunegis et al. “Spectral analysis of signed graphs for clustering, prediction and visualization”.
      In: Proceedings of the 2010 SIAM international conference on data mining. SIAM. 2010, pp. 559–570.
[158] Dan Kushnir, Meirav Galun, and Achi Brandt. “Fast multiscale clustering and manifold identification”.
      In: Pattern Recognition 39.10 (2006), pp. 1876–1891.
[159] In So Kweon and Takeo Kanade. “Extracting topographic terrain features from elevation maps”. In:
      CVGIP: image understanding 59.2 (1994), pp. 171–182.
[160] Valerio La Gatta et al. “Music recommendation via hypergraph embedding”. In: IEEE Transactions
      on Neural Networks and Learning Systems (2022).
[161] Renaud Lambiotte, Martin Rosvall, and Ingo Scholtes. “From networks to optimal higher-order models
      of complex systems”. In: Nature physics 15.4 (2019), pp. 313–320.
[162] Yann LeCun et al. “Gradient-based learning applied to document recognition”. In: Proceedings of the
      IEEE 86.11 (1998), pp. 2278–2324.
[163] Hyekyoung Lee et al. “Computing the shape of brain networks using graph filtration and gromov-
      hausdorff metric”. In: International Conference on Medical Image Computing and Computer Assisted
      Intervention (2011), pp. 302–309.
[164] Hyekyoung Lee et al. “Discriminative persistent homology of brain networks”. In: IEEE International
      Symposium on Biomedical Imaging: From Nano to Macro (2011), pp. 841–844.


                                                     67
Topological deep learning                                                                           References




[165] Hyekyoung Lee et al. “Persistent brain network homology from the perspective of dendrogram”. In:
      IEEE Transactions on Medical Imaging 31.12 (2012), pp. 2267–2277.
[166] Hyekyoung Lee et al. “Weighted functional brain network modeling via network filtration”. In: NIPS
      Workshop on Algebraic Topology and Machine Learning (2012).
[167] John Boaz Lee, Ryan Rossi, and Xiangnan Kong. “Graph classification using structural attention”.
      In: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data
      Mining. 2018, pp. 1666–1674.
[168] John Boaz Lee et al. “Attention models in graphs: a survey”. In: ACM Transactions on Knowledge
      Discovery from Data (TKDD) 13.6 (2019), pp. 1–25.
[169] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. “Self-attention graph pooling”. In: Int. Conf. Mach.
      Learn. PMLR. 2019, pp. 3734–3743.
[170] Samuel Leventhal et al. “Exploring classification of topological priors with machine learning for feature
      extraction”. In: IEEE Transactions on Visualization and Computer Graphics (2023).
[171] Juanhui Li et al. “Graph pooling with representativeness”. In: 2020 IEEE International Conference
      on Data Mining (ICDM). IEEE. 2020, pp. 302–311.
[172] Zhifei Li et al. “Learning knowledge graph embedding with heterogeneous relation attention networks”.
      In: IEEE Transactions on Neural Networks and Learning Systems (2021).
[173] Z. Lian et al. “Shape retrieval on non-rigid 3D watertight meshes”. In: Eurographics workshop on 3d
      object retrieval (3DOR). Citeseer. 2011.
[174] Lek-Heng Lim. “Hodge Laplacians on graphs”. In: SIAM Review 62.3 (2020), pp. 685–715.
[175] Kevin Linka et al. “Outbreak dynamics of COVID-19 in Europe and the effect of travel restrictions”.
      In: Computer methods in biomechanics and biomedical engineering 23.11 (2020), pp. 710–717.
[176] Derek Lo and Briton Park. “Modeling the spread of the Zika virus using topological data analysis”.
      In: arXiv preprint arXiv:1612.03554 (2016).
[177] Andreas Loukas. “What graph neural networks cannot learn: depth vs width”. In: arXiv preprint
      arXiv:1907.03199 (2019).
[178] Ephy R. Love et al. “Topological convolutional layers for deep learning”. In: J. of Mach. Learn Res.
      24.59 (2023), pp. 1–35.
[179] Ephy R. Love et al. “Topological deep learning”. In: arXiv preprint arXiv:2101.05778 (2021).
[180] P. Y. Lum et al. “Extracting insights from the shape of complex data using topology”. In: Scientific
      reports 3 (2013), p. 1236.
[181] Yao Ma et al. “Graph convolutional networks with eigenpooling”. In: Proceedings of the 25th ACM
      SIGKDD international conference on knowledge discovery & data mining. 2019, pp. 723–731.
[182] Soumen Majhi, Matjaž Perc, and Dibakar Ghosh. “Dynamics on higher-order networks: a review”. In:
      Journal of the Royal Society Interface 19.188 (2022), p. 20220043.
[183] Slobodan Maletić, Yi Zhao, and Milan Rajković. “Persistent topological features of dynamical systems”.
      In: Chaos: An Interdisciplinary Journal of Nonlinear Science 26.5 (2016), p. 053105.
[184] Ronald Manrıéquez, Camilo Guerrero-Nancuante, and Carla Taramasco. “Protection strategy against
      an epidemic disease on edge-weighted graphs applied to a COVID-19 case”. In: Biology 10.7 (2021),
      p. 667.
[185] Haggai Maron et al. “Convolutional neural networks on surfaces via seamless toric covers.” In: ACM
      Trans. Graph. 36.4 (2017), pp. 71–1.
[186] Haggai Maron et al. “Provably powerful graph networks”. In: arXiv preprint arXiv:1905.11136 (2019).
[187] Jonathan Masci et al. “Geodesic convolutional neural networks on Riemannian manifolds”. In: IEEE
      Conf. Comput. Vis. Pattern Recog. 2015, pp. 37–45.
[188] Daniel Mejia, Oscar Ruiz-Salguero, and Carlos A. Cadavid. “Spectral-based mesh segmentation”. In:
      International Journal on Interactive Design and Manufacturing (IJIDeM) 11.3 (2017), pp. 503–514.
[189] Jerry M. Mendel. “Tutorial on higher-order statistics (spectra) in signal processing and system theory:
      theoretical results and some applications”. In: Proceedings of the IEEE 79.3 (1991), pp. 278–305.



                                                      68
References                                                                           Topological deep learning




[190] Giulia Menichetti, Luca Dall’Asta, and Ginestra Bianconi. “Control of multilayer networks”. In:
      Scientific reports 6.1 (2016), pp. 1–8.
[191] Diego Mesquita, Amauri Souza, and Samuel Kaski. “Rethinking pooling in graph neural networks”.
      In: Adv. Neural Inform. Process. Syst. 33 (2020), pp. 2220–2231.
[192] Francesco Milano et al. “Primal-dual mesh convolutional neural networks”. In: Adv. Neural Inform.
      Process. Syst. 33 (2020), pp. 952–963.
[193] Edward C. Mitchell et al. “A topological deep learning framework for neural spike decoding”. In: arXiv
      preprint arXiv:2212.05037 (2022).
[194] Tom M. Mitchell. The need for biases in learning generalizations. Citeseer, 1980.
[195] Volodymyr Mnih, Nicolas Heess, Alex Graves, et al. “Recurrent models of visual attention”. In:
      Advances in neural information processing systems 27 (2014).
[196] Federico Monti et al. “Geometric deep learning on graphs and manifolds using mixture model CNNs”.
      In: IEEE Conf. Comput. Vis. Pattern Recog. 2017, pp. 5115–5124.
[197] Christopher Morris et al. “Weisfeiler and Leman go neural: higher-order graph neural networks”. In:
      Proceedings of the AAAI Conference on Artificial Intelligence 33.01 (2019), pp. 4602–4609.
[198] James R. Munkres. Elements of algebraic topology. CRC press, 2018.
[199] James R. Munkres. Topology; a first course. 1974.
[200] Kevin A. Murgas, Emil Saucan, and Romeil Sandhu. “Hypergraph geometry reflects higher-order
      dynamics in protein interaction networks”. In: Scientific Reports 12.1 (2022), p. 20879.
[201] Behnam Neyshabur et al. “The role of over-parametrization in generalization of neural networks”. In:
      7th International Conference on Learning Representations, ICLR 2019. 2019.
[202] Monica Nicolau, Arnold J. Levine, and Gunnar Carlsson. “Topology based data analysis identifies a
      subgroup of breast cancers with a unique mutational profile and excellent survival”. In: Proceedings of
      the National Academy of Sciences 108.17 (2011), pp. 7265–7270.
[203] Christopher Oballe et al. “Bayesian topological signal processing”. In: Discrete & Continuous Dynamical
      Systems-S (2021).
[204] Antonio Ortega et al. “Graph signal processing: overview, challenges, and applications”. In: Proceedings
      of the IEEE 106.5 (2018), pp. 808–828.
[205] Yunsheng Pang, Yunxiang Zhao, and Dongsheng Li. “Graph pooling via coarsened graph infomax”.
      In: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in
      Information Retrieval. 2021, pp. 2177–2181.
[206] Mathilde Papillon et al. “Architectures of topological deep learning: a survey on topological neural
      networks”. In: (2023).
[207] Adam Paszke et al. “Automatic differentiation in PyTorch”. In: (2017).
[208] F. Pedregosa et al. “Scikit-learn: machine learning in Python”. In: J. of Mach. Learn Res. 12 (2011),
      pp. 2825–2830.
[209] Jose A. Perea et al. “SW1PerS: sliding windows and 1-persistence scoring; discovering periodicity in
      gene expression time series data”. In: BMC bioinformatics 16.1 (2015), p. 257.
[210] Giovanni Petri et al. “Networks and cycles: a persistent homology approach to complex networks”.
      In: Proceedings European Conference on Complex Systems 2012, Springer Proceedings in Complexity
      (2013), pp. 93–99.
[211] Giovanni Petri et al. “Topological strata of weighted complex networks”. In: PLoS ONE 8.6 (2013).
[212] Simone Piaggesi, André Panisson, and Giovanni Petri. “Effective higher-order link prediction and
      reconstruction from simplicial complex embeddings”. In: Learning on Graphs Conference. PMLR. 2022,
      pp. 55–1.
[213] Chiara Plizzari, Marco Cannici, and Matteo Matteucci. “Spatial temporal transformer network for
      skeleton-based action recognition”. In: Int. Conf. Pattern Recog. Springer. 2021, pp. 694–701.
[214] Chi Seng Pun, Kelin Xia, and Si Xian Lee. “Persistent-homology-based machine learning and its
      applications–a survey”. In: arXiv preprint arXiv:1811.00252 (2018).



                                                     69
Topological deep learning                                                                          References




[215] Charles R. Qi et al. “PointNet: deep learning on point sets for 3D classification and segmentation”. In:
      IEEE Conf. Comput. Vis. Pattern Recog. 2017, pp. 652–660.
[216] Thummaluru Siddartha Reddy, Sundeep Prabhakar Chepuri, and Pierre Borgnat. “Clustering with
      simplicial complexes”. In: arXiv preprint arXiv:2303.07646 (2023).
[217] Davis Rempe et al. “CASPR: learning canonical spatiotemporal point cloud representations”. In: Adv.
      Neural Inform. Process. Syst. 33 (2020), pp. 13688–13701.
[218] Bastian Rieck and Heike Leitte. “Persistent homology for the evaluation of dimensionality reduction
      schemes”. In: Computer Graphics Forum. Vol. 34. 3. Wiley Online Library. 2015, pp. 431–440.
[219] Michael Robinson. Topological signal processing. Vol. 81. Springer, 2014.
[220] T Mitchell Roddenberry and Santiago Segarra. “HodgeNet: graph neural networks for edge data”. In:
      2019 53rd Asilomar Conference on Signals, Systems, and Computers. IEEE. 2019, pp. 220–224.
[221] T. Mitchell Roddenberry, Nicholas Glaze, and Santiago Segarra. “Principled simplicial neural networks
      for trajectory prediction”. In: Int. Conf. Mach. Learn. PMLR. 2021, pp. 9020–9029.
[222] T. Mitchell Roddenberry, Michael T. Schaub, and Mustafa Hajij. “Signal processing on cell complexes”.
      In: ICASSP (2022).
[223] Robin Rombach et al. “High-resolution image synthesis with latent diffusion models”. In: IEEE Conf.
      Comput. Vis. Pattern Recog. 2022, pp. 10684–10695.
[224] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. “U-Net: convolutional networks for biomedical
      image segmentation”. In: International Conference on Medical image computing and computer-assisted
      intervention. Springer. 2015, pp. 234–241.
[225] Paul Rosen et al. “Using contour trees in the analysis and visualization of radio astronomy data cubes”.
      In: arXiv preprint arXiv:1704.04561 (2017), pp. 1–7.
[226] Alvaro Sanchez-Gonzalez et al. “Learning to simulate complex physics with graph networks”. In: Int.
      Conf. Mach. Learn. PMLR. 2020, pp. 8459–8468.
[227] Adam Santoro et al. “A simple neural network module for relational reasoning”. In: Advances in neural
      information processing systems. 2017, pp. 4967–4976.
[228] Andrea Santoro et al. “Higher-order organization of multivariate time series”. In: Nature Physics
      (2023), pp. 1–9.
[229] Stefania Sardellitti and Sergio Barbarossa. “Topological signal representation and processing over cell
      complexes”. In: arXiv preprint arXiv:2201.08993 (2022).
[230] Stefania Sardellitti, Sergio Barbarossa, and Lucia Testa. “Topological signal processing over cell
      complexes”. In: Proceeding IEEE Asilomar Conference. Signals, Systems and Computers (2021).
[231] Maxime Savoy. “Combinatorial cell complexes: duality, reconstruction and causal cobordisms”. In:
      arXiv preprint arXiv:2201.12846 (2022).
[232] Franco Scarselli et al. “The graph neural network model”. In: IEEE transactions on neural networks
      20.1 (2008), pp. 61–80.
[233] Michael T. Schaub and Santiago Segarra. “Flow smoothing and denoising: graph signal processing in
      the edge-space”. In: 2018 IEEE Global Conference on Signal and Information Processing (GlobalSIP).
      2018, pp. 735–739.
[234] Michael T. Schaub et al. “Random walks on simplicial complexes and the normalized Hodge 1-
      Laplacian”. In: SIAM Review 62.2 (2020), pp. 353–391.
[235] Michael T. Schaub et al. “Signal processing on higher-order networks: livin’on the edge... and beyond”.
      In: Signal Processing 187 (2021), p. 108149.
[236] Michael T. Schaub et al. “Signal processing on simplicial complexes”. In: Higher-Order Systems.
      Springer, 2022, pp. 301–328.
[237] Yair Schiff et al. “Characterizing the latent space of molecular deep generative models with persistent
      homology metrics”. In: arXiv preprint arXiv:2010.08548 (2020).
[238] Michael Schlichtkrull et al. “Modeling relational data with graph convolutional networks”. In: European
      Semantic Web Conference. Springer. 2018, pp. 593–607.
[239] Nicholas Sharp et al. “DiffusionNet: discretization agnostic learning on surfaces”. In: ACM Trans.
      Graph. 41.3 (2022), pp. 1–16.


                                                     70
References                                                                            Topological deep learning




[240] Heyuan Shi et al. “Hypergraph-induced convolutional networks for visual classification”. In: IEEE
      transactions on neural networks and learning systems 30.10 (2018), pp. 2963–2972.
[241] Jonathan Shlomi, Peter Battaglia, and Jean-Roch Vlimant. “Graph neural networks in particle physics”.
      In: Machine Learning: Science and Technology 2.2 (2020), p. 021001.
[242] David I. Shuman, Benjamin Ricaud, and Pierre Vandergheynst. “Vertex-frequency analysis on graphs”.
      In: Applied and Computational Harmonic Analysis 40.2 (2016), pp. 260–291.
[243] Karen Simonyan and Andrew Zisserman. “Very deep convolutional networks for large-scale image
      recognition”. In: arXiv preprint arXiv:1409.1556 (2014).
[244] Gurjeet Singh, Facundo Mémoli, Gunnar E Carlsson, et al. “Topological methods for the analysis of
      high dimensional data sets and 3d object recognition.” In: PBG@ Eurographics 2 (2007), pp. 091–100.
[245] Per Sebastian Skardal et al. “Higher-order interactions improve optimal collective dynamics on
      networks”. In: arXiv preprint arXiv:2108.08190 (2021).
[246] Dmitriy Smirnov and Justin Solomon. “HodgeNet: learning spectral geometry on triangle meshes”. In:
      ACM Trans. Graph. 40.4 (2021), pp. 1–11.
[247] Zidong Su, Zehui Hu, and Yangding Li. “Hierarchical graph representation learning with local capsule
      pooling”. In: ACM Multimedia Asia. 2021, pp. 1–7.
[248] Yizhou Sun et al. “RankClus: integrating clustering with ranking for heterogeneous information
      network analysis”. In: Proceedings of the 12th international conference on extending database technology:
      advances in database technology. 2009, pp. 565–576.
[249] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. “Sequence to sequence learning with neural networks”.
      In: Advances in neural information processing systems. 2014, pp. 3104–3112.
[250] Shazia Tabassum et al. “Social network analysis: an overview”. In: Wiley Interdisciplinary Reviews:
      Data Mining and Knowledge Discovery 8.5 (2018), e1256.
[251] Dane Taylor et al. “Topological data analysis of contagion maps for examining spreading processes on
      networks”. In: Nature communications 6 (2015), p. 7723.
[252] Chad M Topaz, Lori Ziegelmeier, and Tom Halverson. “Topological data analysis of biological aggrega-
      tion models”. In: PloS one 10.5 (2015), e0126383.
[253] Leo Torres et al. “The why, how, and when of representations for complex systems”. In: SIAM Review
      63.3 (2021), pp. 435–485.
[254] Nathaniel Trask, Andy Huang, and Xiaozhe Hu. “Enforcing exact physics in scientific machine learning:
      a data-driven exterior calculus on graphs”. In: Journal of Computational Physics (2022), p. 110969.
[255] Vladimir G. Turaev. Quantum invariants of knots and 3-manifolds. Vol. 18. Walter de Gruyter GmbH
      & Co KG, 2016.
[256] Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural information processing
      systems 30 (2017).
[257] Petar Veličković. “Message passing all the way up”. In: ICLR 2022 Workshop on Geometrical and
      Topological Representation Learning (2022).
[258] Petar Veličković et al. “Graph attention networks”. In: International Conference on Learning Repre-
      sentations. 2018.
[259] Michelle L. Wachs. “Poset topology: tools and applications”. In: arXiv preprint math/0602226 (2006).
[260] Cheng Wang et al. “Survey of hypergraph neural networks and its application to action recognition”.
      In: Artificial Intelligence: Second CAAI International Conference, CICAI 2022, Beijing, China, August
      27–28, 2022, Revised Selected Papers, Part II. Springer. 2023, pp. 387–398.
[261] Fan Wang et al. “Topogan: A topology-aware generative adversarial network”. In: Eur. Conf. Comput.
      Vis. Springer. 2020, pp. 118–136.
[262] Yunhai Wang et al. “Active co-analysis of a set of shapes”. In: ACM Trans. Graph. 31.6 (2012),
      pp. 1–10.
[263] E. Weinan, Luan Jianfeng, and Yao Yuan. “The landscape of complex networks: critical nodes and a
      hierarchical decomposition”. In: Methods and applications of analysis 20 (2013), pp. 383–404.
[264] Boris Weisfeiler and Andrei Leman. “The reduction of a graph to canonical form and the algebra
      which appears therein”. In: NTI, Series 2.9 (1968), pp. 12–16.


                                                      71
Topological deep learning                                                                         References




[265] Francis Williams. Point Cloud Utils. 2022.
[266] Hanrui Wu and Michael K. Ng. “Hypergraph convolution on nodes-hyperedges network for semi-
      supervised node classification”. In: ACM Transactions on Knowledge Discovery from Data (TKDD)
      16.4 (2022), pp. 1–19.
[267] Zhirong Wu et al. “3D ShapeNets: a deep representation for volumetric shapes”. In: IEEE Conf.
      Comput. Vis. Pattern Recog. 2015, pp. 1912–1920.
[268] Zonghan Wu et al. “A comprehensive survey on graph neural networks”. In: IEEE transactions on
      neural networks and learning systems 32.1 (2020), pp. 4–24.
[269] Chenxin Xu et al. “GroupNet: multiscale hypergraph neural networks for trajectory prediction with
      relational reasoning”. In: IEEE Conf. Comput. Vis. Pattern Recog. 2022, pp. 6498–6507.
[270] Keyulu Xu et al. “How powerful are graph neural networks?” In: arXiv preprint arXiv:1810.00826
      (2018).
[271] Sijie Yan, Yuanjun Xiong, and Dahua Lin. “Spatial temporal graph convolutional networks for
      skeleton-based action recognition”. In: Thirty-second AAAI conference on artificial intelligence. 2018.
[272] Maosheng Yang and Elvin Isufi. “Convolutional learning on simplicial complexes”. In: arXiv preprint
      arXiv:2301.11163 (2023).
[273] Maosheng Yang et al. “Finite impulse response filters for simplicial complexes”. In: 2021 29th European
      Signal Processing Conference (EUSIPCO). IEEE. 2021, pp. 2005–2009.
[274] Nan Yin et al. “Dynamic hypergraph convolutional network”. In: 2022 IEEE 38th International
      Conference on Data Engineering (ICDE). IEEE. 2022, pp. 1621–1634.
[275] Zhitao Ying et al. “Hierarchical graph representation learning with differentiable pooling”. In: Adv.
      Neural Inform. Process. Syst. 31 (2018).
[276] Manzil Zaheer et al. “Deep sets”. In: arXiv preprint arXiv:1703.06114 (2017).
[277] Qi Zhang et al. “Kernel-weighted graph convolutional network: a deep learning approach for traffic
      forecasting”. In: 2018 24th International Conference on Pattern Recognition (ICPR). IEEE. 2018,
      pp. 1018–1023.
[278] Shi-Xue Zhang et al. “Deep relational reasoning graph network for arbitrary shape text detection”. In:
      IEEE Conf. Comput. Vis. Pattern Recog. 2020, pp. 9699–9708.
[279] Weifeng Zhang et al. “Multiplex graph neural networks for multi-behavior recommendation”. In:
      Proceedings of the 29th ACM International Conference on Information & Knowledge Management.
      2020, pp. 2313–2316.
[280] Zhen Zhang et al. “Hierarchical graph pooling with structure learning”. In: arXiv preprint
      arXiv:1911.05954 (2019).
[281] Zhen Zhang et al. “Hierarchical multi-view graph pooling with structure learning”. In: IEEE Transac-
      tions on Knowledge and Data Engineering 35.1 (2021), pp. 545–559.
[282] Lingxiao Zhao et al. “From stars to subgraphs: uplifting any GNN with local structure awareness”. In:
      arXiv preprint arXiv:2110.03753 (2021).
[283] Yongheng Zhao et al. “3DPointCaps++: learning 3D representations with capsule networks”. In: Int.
      J. Comput. Vis. 130.9 (2022), pp. 2321–2336.
[284] Jie Zhou et al. “Graph neural networks: a review of methods and applications”. In: AI Open 1 (2020),
      pp. 57–81.
[285] Jianming Zhu et al. “Social influence maximization in hypergraph in social networks”. In: IEEE
      Transactions on Network Science and Engineering 6.4 (2018), pp. 801–811.




                                                     72
A   Glossary                                                                                   Topological deep learning




A    Glossary

Tables 5 and 6 summarize the paper’s notations and acronyms, respectively.


                                 Table 5: Tabulation of the notations used in this paper.

      Notation                        Description
                                                          Set notations
      S                               Non-empty finite set of abstract entities
      PS                              Index set
      P(S)                            Power set of a set S
      (S, N )                         Topological space of a nonempty set S and a neighborhood topology N
      Na (x)                          Adjacency set of a cell x
      Nco (x)                         Coadjacency set of a cell x
      N{G1 ,...,Gn } (x)              Neighbors of x specified by the neighborhood matrices {G1 , . . . , Gn }
      N↘ (x)                          Set of down-incidence of a cell x
      N↗ (x)                          Set of up-incidence of a cell x
      N↘,k (x)                        Set of k-down incidence of a cell x
      N↗,k (x)                        Set of k-up incidence of a cell x
      N and Z≥0                       Set of positive integers and non-negative integers, respectively
                                                            Domains
      G                               Graph
      xk                              Cell x of rank k
      rk                              Rank function
      (S, X , rk)                     CC, consisting of a set S, a subset X of P(S) \ {∅}, and a rank function rk
      dim(X )                         Dimension of a CC X
      {cα }α∈I                        Partition into subspaces (cells) indexed by an index set I
      int(x)                          Interior of a cell x in a regular cell complex
      nα ∈ N                          Dimension of a cell in a regular cell complex
      0-cells                         Vertices of a CC
      1-cells                         Edges of a CC
      k-cells                         Cells with rank k
      X (k)                           k-skeleton of X , formed by i-cells in X with i ≤ k
      Xk                              Set of k-cells of X
      |X k |                          Cardinality of X k , that is number of k-cells of X
      CCn−hop (G)                     n-hop CC of a graph G
      CCp (G)                         Path-based CC of a graph G
      CCloop (G)                      Loop-based CC of a graph G
      CCSC (Y)                        Coface CC of a simplicial complex/CC Y
                                                        Matrix notations
      Br,k                            Incidence matrices between r-cells and k-cells
      Ar,k                            Adjacency matrices among the cells of X r with respect to the cells of X k
      coAr,k                          Coadjacency matrices among the cells of X r with respect to the cells of X k
                                                             CCNNs
      W                               Trainable parameter
      C k (X , Rd )                   k-cochain space with features in Rd
      Ck                              k-cochain space with features in some Euclidean space
      G = {G1 , . . . , Gm }          Set of cochain maps Gi defined on defined on a complex
      MG;W                            Merge node
      G : C s (X ) → C t (X )         Cochain map
      (xi1 , . . . , xim )            Vector of cochains
      attl : C s (X ) → C s (X )      Higher-order attention matrix
      NY0 = {Y1 , . . . , σ|NY0 | }   Set of a complex object in the vicinity of Y0
      a : Y0 × NY0 → [0, 1]           Higher-order attention function
      CCNNG;W                         CCNN or its tensor diagram representation
      HX = (V (HX ), E(HX ))          Hasse graph with vertices V (HX ) and edges E(HX ); see Definition 43




                                                             73
Topological deep learning                                                                          B   Lifting maps




                             Table 6: Tabulation of the acronyms used in this paper.

                        Acronym       Description
                        AGD           Average geodesic distance
                        CC            Combinatorial complex
                        CCANN         Combinatorial complex attention neural network
                        CCCNN         Combinatorial complex convolutional neural network
                        CCNN          Combinatorial complex neural network
                        CNN           Convolutional neural network
                        DEC           Discrete exterior calculus
                        GDL           Geometric deep learning
                        GNN           Graph neural network
                        MOG           Mapper on graphs
                        RNN           Recurrent neural network
                        SCoNe         Simplicial complex network
                        sub-CC        sub-combinatorial complex
                        TDA           Topological data analysis
                        TDL           Topological deep learning
                        TQFT          Topological quantum field theory


B     Lifting maps

Lifting refers to the process of mapping a featured domain to another featured domain via a well-defined
procedure. This section shows how we can lift a given domain to a CC or cell complex. Such lifting is
useful as it allows CCNNs to be applied to common topological domains, including graphs and cell/simplicial
complexes. This section only scratches the surface, as there remain many lifting constructions to be explored.
We refer the reader to [98] for examples of lifting graphs to simplicial complexes.

B.1   n-hop CC of a graph

Let G = (V (G), E(G)) be a graph and n ≥ 2 an integer. The n-hop CC of G, denoted by CCn-hop (G), is the
CC whose 0-cells, 1-cells, and n-cells are the nodes of G, edges of G, and set nodes in n-hop neighborhoods
of the nodes in G, respectively. It is easy to verify that CCn-hop (G) is a CC of dimension n. Figure 37(a)
visualizes the 1-hop CC of a graph.

B.2   Path-based and subgraph-based CC of a graph

Let G = (V (G), E(G)) be a graph. A natural CC structure on G considers paths of G. We define a path-based
CC of G, denoted by CCp (G), to be a CC consisting of 0-cells, 1-cells and 2-cells specified as follows. First,
X 0 and X 1 in CCp (G) are the sets of nodes and edges of G, respectively. We now explain how to construct a
2-cell in CCp (G). Let P be a path in G with length larger than or equal to two (i.e., with two or more edges).
An element xP in X 2 induced by P is defined to be xP = ∪v∈P {v}. The set X 2 in CCp (G) is a non-empty
collection of elements xP . It is easy to verify that CCp (G) is a CC with dim(CCp (G)) = 2. Note that we
may replace the path P by a tree/subgraph of graph G and obtain a similar CC structure induced by the
tree/subgraph of G. Figure 37(b) shows an example of a path-based CC of a graph.

B.3   Loop-based CC of a graph

Let G = (V (G), E(G)) be a graph. We associate a CC structure with G that considers loops in G. We define
a loop-based CC of G, denoted by CCloop (G), to be a CC consisting of 0-cells, 1-cells and 2-cells specified
as follows. First, we set X 0 and X 1 in CCloop (G) to be the nodes and edges of G, respectively. We now
explain how to construct a 2-cell in CCloop (G). A 2-cell in CCloop (G) is a set C = {x01 , . . . , x0k } ⊂ X 0 such
that {x0i , x0i+1 }, 1 ≤ i ≤ k − 1, and {x0k , x01 } are the only edges in X 1 ∩ C. The set X 2 in CCloop (G) is a
nonempty collection of elements C. It is easy to verify that CCloop (G) is a CC with dim(CCloop (G)) = 2.
Note that the sequence (x01 , . . . , x0k ) defines a loop in G. This loop is called the loop that characterizes the


                                                        74
B   Lifting maps                                                                              Topological deep learning




Figure 37: Examples of lifting domains to CCs and cell complexes. (a): The 1-hop neighborhood of the red
node can be considered as a 2-cell that we can augment to the graph. Adding such 2-cells to a graph yields a
CC called the 1-hop neighborhood of the graph (see Section B.1). (b): A path on a graph of length more
than two can be considered as a 2-cell that we can augment to the graph. Adding such 2-cells to a graph
yields a CC called a path-based CC of the graph (see Section B.2). (c): A loop in a graph (i.e., a closed path
with no repeating edges) can be considered as a 2-cell that we can augment to the graph. Adding such 2-cells
to a graph yields a CC called a loop-based CC of the graph (see Section B.3). (d): For every blue 2-cell of a
simplicial complex, we introduce a green 3-cell obtained by considering the 1-coface of the 2-cell. Adding
such 3-cells to a simplicial complex yields a CC of dimension three called the coface CC of the simplicial
complex (see Section B.4).


2-cell C = {x01 , . . . , x0k }. Similar constructions are suggested in [8, 25, 231, 222]. In fact, it is easy to confirm
that every 2-dimensional regular cell complex can be constructed in this manner [222]. Figure 37(c) shows an
example of a loop-based CC of a graph.

B.4   Coface CC of a simplicial complex/CC

Here, we describe a method to lift a simplicial complex of dimension two to a CC of dimension three. This
method can be easily generalized to other dimensions. For a simplicial complex Y of dimension two, the
coface CC of Y, denoted by CCSC (Y), is defined as follows. Y 0 , Y 1 , and Y 2 in CCSC (Y) are the nodes, the
edges, and the triangles in Y, respectively. We now explain how to construct a 3-cell in CCSC (Y). Let x2 be
a 2-cell in Y. The 3-cell in CCSC (Y) associated with x2 is the union of all 0-cells in Nco,1 (x2 ) ∪ x2 . The set
Y 3 in CCSC (Y) is defined as the set of all 3-cells associated with all 2-cells x2 in Y. It is easy to verify that
CCSC (Y) is a CC with dim(CCSC (Y)) = 3. A similar lifting construction can be defined to augment any CC
of dimension n with (n + 1)-cells in order to obtain a CC of dimension n + 1.

B.5   Augmentation of CCs by higher-rank cells

The lifting methods proposed in Sections B.2, B.3, and B.4 can be described abstractly under a single general
lifting construction. Specifically, the essence of all these lifting methods is to augment the underlying CC X
with new cells that have a rank of dim(X ) + 1. Proposition B.1 formalizes the general lifting construction.
Proposition B.1. Let S be a nonempty set and (S, X , rk) a CC of dimension n defined on S. Consider
a set X n+1 ⊂ P(S) such that if x ∈ X and y ∈ X n+1 with x ⊆ y, then x ⊊ y. Further, consider a map
 ˆ : X ∪ X n+1 → Z≥0 that satisfies rk(x)
rk                                    ˆ                                 ˆ
                                           = rk(x) for all x ∈ X and rk(x)   = n + 1 for all x ∈ X n+1 . For
X n+1       ˆ
       and rk satisfying such conditions, (S, X ∪ X n+1 ˆ
                                                       , rk) is a CC of dimension n + 1.


Proof. The proof follows directly from Definition 9.


                                                           75
Topological deep learning                                             C   CCNN architecture search and topological quantum field theories




                                             Marked trivalent graph
                                                                                                     Tensor diagram




                            (a)                                                              (b)

Figure 38: A sketch of the main idea of a TQFT construction in (a) and a functor Ftri for constructing
tensor diagrams in (b). (a): In the depicted example, the goal is to construct a map f : A → B between two
topological spaces A and B. We first decompose A and B into simpler sub-spaces, say A = A1 ∪ A2 and
B = B1 ∪ B2 , so that A1 , A2 and B1 , B2 are ‘more elementary spaces’ than A and B, respectively. We then
construct two maps f1 : A1 → B1 and f2 : A2 → B2 , and use them to construct f . (b): A visual example of a
functor Ftri that pairs each marked trivalent graph with a corresponding tensor diagram. This example sends
a marked trivalent graph with marked points {0, 1, 2} at the bottom and marked points {1, 2} at the top to a
tensor diagram with domain C 0 × C 1 × C 2 and codomain C 1 × C 2 . Once the push-forward, the merge and the
split nodes are defined, the functor Ftri attaches a well-defined tensor diagram to each trivalent graph. A
CCNN can be then constructed from the tensor diagram.


                                                         ˆ as constructed in Proposition B.1, a highest-rank
Given a CC X , we call a CC of the form (S, X ∪ X n+1 , rk),
augmented CC of X . Note that Proposition B.1 provides a constructive and iterative method to build a CC
of arbitrary dimension from a nonempty set S of abstract points.


C    CCNN architecture search and topological quantum field theories

The problem of CCNN architecture search for a given TDL task can be cast as a hyperparameter optimization
problem over the space of CCNNs. More precisely, consider the query of searching for an optimal CCNN
between two fixed Cartesian products C i1 × C i2 × · · · × C im and C j1 × C j2 × · · · × C jn of cochain spaces. In
practice, this query poses a challenging problem. Rather than performing a computationally expensive CCNN
search directly, an alternative approach is to conduct a search in the simpler space of marked trivalent graphs
between marked points {i1 , . . . , im } and {j1 , . . . , jn }, and then map the resulting marked trivalent graph to
a corresponding CCNN architecture. In the present section, we briefly sketch such a graph-based search
method using tools from topological quantum field theory (TQFT) [255], and accordingly we assume some
familiarity with the basics of category theory.
Tensor diagrams draw their inspiration from TQFT, in which arbitrary maps between topological spaces are
constructed from simpler and more manageable building blocks. The main workflow in TQFT constructions
involves breaking down the topological spaces under consideration into simpler subspaces, and subsequently
utilizing maps between the subspaces to construct maps between the initial topological spaces; see Figure 38(a).
Thus, the topological properties of maps between topological spaces are better understood via the topological
properties of simpler constituent maps between respective subspaces. This can be especially useful in
applications such as knot theory or the study of three-dimensional manifolds, where understanding the
topological properties of involved maps is the main interest [255].
Before we introduce the relation between tensor diagrams and TQFTs, we sketch two required preliminaries,
namely marked trivalent graphs and TQFTs. First, we define the objects and morphisms of the category of
marked trivalent graphs T ri. A marked trivalent graph is intuitively described via its layout. Consider the
2d-disk [0, 1] × [0, 1] with a collection of marked points {i1 , . . . , im } at the bottom and a collection of marked
points {j1 , . . . , jn } at the top. A marked trivalent graph is a graph that is drawn inside the disk in such a
way that all edges flow within the disk, connecting the marked points at the top with the marked points at
the bottom, and meeting at each vertex with exactly three edges. In the category T ri, a trivalent graph with


                                                                               76
D   Learning discrete exterior calculus operators with CCANNs                                 Topological deep learning




marked points {i1 , . . . , im } and {j1 , . . . , jn } at the bottom and top, respectively, plays the role of a morphism
between the object {i1 , . . . , im } and the object {j1 , . . . , jn }. The composition of two such morphisms, when
admissible, is defined to be the vertical concatenation of their corresponding marked trivalent graphs. The
vertical concatenation of marked trivalent graphs yields a marked trivalent graph. Finally, any trivalent
graph can be built by concatenating horizontally and vertically three elementary trivalent graphs similar to
the graph shown in Figure 38(b), ignoring the labels in the figure.
At a high level, a TQFT is a functor F : Bordn → V ecK that assigns a K-vector space F (x) ∈ V ecK to each
n-manifold x ∈ Bordn , and a linear map F (f ) to each (n + 1)-cobordism f ∈ Bordn . It is noted that a
(n + 1)-cobordism is an (n + 1)-manifold representing a morphism between two n-manifolds. The functor F
is typically defined on a few elementary morphisms in Bordn , but can be extended to arbitrary morphisms in
Bordn by preserving the structure of the elementary maps when mapped via F . Such a functor enables the
study of n-manifolds and (n + 1)-cobordisms defined between them by mapping n-manifolds to their simpler
counterparts in V ecK .
To establish the relation between marked trivalent graphs and tensor diagrams, we introduce a new TQFT
Ftri that sends each morphism (marked trivalent graph) in T ri to a corresponding tensor diagram by labeling
marked trivalent graphs with appropriate cochain maps. Figure 38(b) shows an illustration of such a functor
Ftri . Through the lens of Ftri , the relation between tensor diagrams and marked trivalent graphs becomes
clear; a tensor diagram is a marked trivalent graph whose edges are labeled via cochain maps. The functor
Ftri allows us to search the space of marked trivalent graphs as an equivalent way of conducting CCNN
architecture search, since CCNNs are represented by tensor diagrams.


D    Learning discrete exterior calculus operators with CCANNs

The operator Gtr = G ⊙ att of Equations 8 and 11 has an advantageous cross-cutting interpretation. First,
recall that Gtr has the same shape as the original operator G. More importantly, Gtr can be viewed as
a learnt version of G. For instance, if G is the k-Hodge Laplacian Lk , then the learnt attention version
Gtr of it represents a k-Hodge Laplacian that is adapted to the domain X for the learning task at hand.
This perspective converts our attention framework to a tool for learning discrete exterior calculus (DEC)
operators [83]. We refer the interested reader to recent works along these lines [246, 254], where neural
networks are used to learn Laplacian operators in various shape analysis tasks.
Concretely, one of the main building blocks of DEC is a collection of linear operators of the form A : C i (X ) →
C j (X ) that act on a cochain H to produce another cochain A(H). An example of an operator A is the
graph Laplacian. There are seven primitive DEC operators, including the discrete exterior derivative, the
hodge star and the wedge product. These seven primitive operators can be combined together to form other
operators. In our setting, the discrete exterior derivatives are precisely a signed version of the incidence
matrices defined in the context of cell/simplicial complexes. We denote the k-signed incidence matrix defined
on a cell/simplicial complex by Bk . It is common in the context of discrete exterior calculus [83] to refer
to BTk as the k th discrete exterior derivative dk . So, from a DEC point of view, the matrices BT0 , BT1 and
BT2 are regarded as the discrete exterior derivatives d0 (H), d1 (H), and d2 (H) of some 0-, 1-, and 2-cochains
defined on X , which in turn are the discrete analogs of the gradient ∇H, curl ∇ × H and divergence ∇ · H
of a smooth function defined on a smooth surface. We refer the reader to [83] for a coherent list of DEC
operators and their interpretation. Together, cochains and the operators that act on them provide a concrete
framework that facilitates computing a cochain of interest, such as a cochain obtained by solving a partial
differential equation on a discrete surface.
Our attention framework can be viewed as a non-linear version of the DEC based on linear operators A,
and can be used to learn the DEC operators on a domain X for a particular learning task. Specifically, a
linear operator A, as it appears in classical DEC, can be considered as a special case of Equations 8 and 11.
Unlike existing work [246, 254], our DEC learning approach based on CCANNs generalizes and applies to all
domains in which DEC is typically applicable; examples of such domains include triangular and polygonal
meshes [75]. In contrast, existing operator learning methods are defined only for particular types of DEC
operators, and therefore cannot be used to learn arbitrary types of DEC operators.


                                                           77
Topological deep learning                          E   A mapper-induced topology-preserving CC-pooling operation




Figure 39: An illustration of the MOG algorithm. The input to the MOG algorithm is a triplet (X , g, U),
where X is a graph, g : X 0 → [0, 1] is a scalar function defined on the vertex set of X , and U is a cover of [0, 1].
(a): An input graph X . (b): The scalar function g : X → [0, 1] is visualized by color-mapping its scalar values
according to the displayed color bar. Figure (b) also shows the covering U = {U1 , U2 , U3 , U4 } depicted in red,
orange, yellow and blue colors. (c): We pull back via g each cover element Ui in U, and we compute the
connected components in g −1 (Ui ). (d): The vertex set in the graph generated by the MOG algorithm consists
of the connected components induced by g −1 (Ui ), while the edge set is formed by considering the intersection
among the connected components. Note that the graph generated by the MOG algorithm approximates the
shape of the input graph.


E    A mapper-induced topology-preserving CC-pooling operation

In this section, we give an example of constructing a shape-preserving pooling operation on a CC X .
Specifically, we demonstrate the case in which X is a graph, and utilize the mapper on graphs (MOG)
construction [125], which is a graph skeletonization algorithm that can be used to augment X with topology-
preserving higher-rank cells, as demonstrated in Figure 4. Although we only demonstrate the shape-preserving
pooling construction on graphs, the method suggested herein can be easily extended to CCs.
Let X be a connected graph and g : X 0 → [0, 1] a scalar function. Let U = {Uα }α∈I be a finite collection of
open sets that covers the interval [0, 1]. The MOG construction of a graph M OG(VM OG , EM OG ) based on
the triplet (X , g, U) consists of the following steps:

      1. We first use the cover U to construct the pull-back cover g ⋆ (U) = {g −1 (Uα )}α∈I .
      2. The vertex set VM OG is formed by considering the connected components (i.e., maximal connected
         subgraphs) induced by g −1 (Uα ) for each α.
      3. The edge set EM OG is formed by considering the intersection among the connected components
         computed in step 2.

Figure 39 shows an illustrative example of applying the MOG algorithm to a graph. Figure 39(a) shows the
graph on which the MOG algorithm is applied, while Figure 39(b) visualizes the scalar function g and the
covering U, which consists of four covering elements depicted in red, orange, yellow and blue colors.
The connected components obtained from the MOG algorithm can be used to augment a graph X with a
skeleton X 2 of dimension 2, thus constructing a CC, as described in Proposition B.1. We denote the resulting
CC by Xg,U . Figure 40 demonstrates the construction of a CC using this augmentation process based on the
MOG algorithm.
MOG is a topology-preserving graph skeletonization algorithm, which can be used to coarsen the size of an
input graph X . Such coarsening typically occurs when constructing a pooling layer. So, the skeleton X 2
generated by the MOG algorithm can be utilized to obtain a feature-preserving pooling operation. Figure 41


                                                         78
E   A mapper-induced topology-preserving CC-pooling operation                           Topological deep learning




                                                    +




Figure 40: A visual example of obtaining a CC from a graph via the MOG algorithm. (a): An input graph X .
(b): The graph X is augmented by the 2-cells formed via the connected components obtained from applying
the MOG algorithm to X , as described in Figure 30. (c): The CC Xg,U obtained by augmenting X with
these 2-cells.




Figure 41: Illustration of coarsening a graph via the MOG algorithm. (a): Visualization of a graph and
of a scalar function defined on it, which are used as inputs to the MOG algorithm. The scalar function
is visualized by color-mapping its scalar values. (b): The resulting MOG graph. Notice how the MOG
construction preserves the overall shape of the original graph.


illustrates that the MOG algorithm coarsens an input graph to produce a graph that preserves topological
features of the input graph.
To recap, the MOG algorithm receives an input graph, and outputs a coarsened graph along with connected
components. The connected components yield in turn a CC. The coarsened graph and the adjacency relation
between the 2-cells of the associated CC are related, as elaborated in Proposition E.1. Figure 42 conveys
visually Proposition E.1.
Proposition E.1. Let (X , g, U) be a triplet consisting of a graph X , a scalar function g : X 0 → [0, 1] defined
on the vertex set of X , and a cover U of [0, 1]. Let M OG(X , g, U) be the graph generated by the MOG
algorithm upon receiving the triplet (X , g, U) as input. Furthermore, let Xg,U be the CC constructed from the
connected components that are generated by the MOG algorithm. The adjacency matrix of the graph M OG(X ,
g, U) is equivalent to the adjacency matrix A2,2 of the CC Xg,U .

Proof. The proof follows by observing that the 2-cells, which are augmented to X to generate the CC Xg,U ,
are 2-adjacent if and only if they intersect on a vertex. Moreover, two cells intersect on a vertex if and only if
there is an edge between them in the graph M OG(X , g, U) outputted by the MOG algorithm.


                                                       79
            Topological deep learning                       E   A mapper-induced topology-preserving CC-pooling operation




            Figure 42: Visual demonstration of Proposition E.1. (a): Visualization of a CC Xg,U , which highlights
            the intersection between blue 2-cells. (b): The corresponding M OG(X , g, U) graph. There is a one-to-one
            correspondence between the blue 2-cells of Xg,U on the left-hand side and the blue vertices of M OG(X , g, U)
            on the right-hand side. Further, two blue 2-cells of Xg,U (left) intersect on an orange vertex if and only if
            there is an orange edge (right) that connects the corresponding two vertices of M OG(X , g, U).


            Given a CC Xg,U obtained from a graph X via the MOG algorithm, the map B0,2     T
                                                                                               : C 0 (Xg,U ) → C 2 (Xg,U )
            can be used to induce a shape-preserving CC-pooling operation (Definition 37). More specifically, a signal
            H0 supported on the vertices of X can be push-forwarded and pooled to a signal H2 supported on the 2-cells
            of Xg,U .




                                                                 80

View publication stats

