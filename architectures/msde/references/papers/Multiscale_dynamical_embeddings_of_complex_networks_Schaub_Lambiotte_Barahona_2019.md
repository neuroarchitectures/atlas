# Multiscale dynamical embeddings of complex networks Schaub Lambiotte Barahona 2019

> Source: `Multiscale_dynamical_embeddings_of_complex_networks_Schaub_Lambiotte_Barahona_2019.pdf`

---

                                                                 Multiscale dynamical embeddings of complex networks

                                                   Michael T. Schaub,1, 2, ∗ Jean-Charles Delvenne,3, 4 Renaud Lambiotte,5 and Mauricio Barahona6, †
                                               1
                                                   Institute for Data, Systems and Society, Massachusetts Institute of Technology, Cambridge, MA 02139, USA
                                                                      2
                                                                        Department of Engineering Science, University of Oxford, Oxford, UK
                                                                 3
                                                                   ICTEAM, Université catholique de Louvain, B-1348 Louvain-la-Neuve, Belgium
                                                                   4
                                                                     CORE, Université catholique de Louvain, B-1348 Louvain-la-Neuve, Belgium
                                                                            5
                                                                              Mathematical Institute, University of Oxford, Oxford, UK
                                                                  6
                                                                    Department of Mathematics, Imperial College London, London SW7 2AZ, UK
                                                                                               (Dated: May 14, 2019)
                                                           Complex systems and relational data are often abstracted as dynamical processes on networks. To
                                                        understand, predict and control their behavior, a crucial step is to extract reduced descriptions of
                                                        such networks. Inspired by notions from Control Theory, we propose a time-dependent dynamical
                                                        similarity measure between nodes, which quantifies the effect a node-input has on the network.




arXiv:1804.03733v2 [cs.SI] 13 May 2019
                                                        This dynamical similarity induces an embedding that can be employed for several analysis tasks.
                                                        Here we focus on (i) dimensionality reduction, i.e., projecting nodes onto a low dimensional space
                                                        that captures dynamic similarity at different time scales, and (ii) how to exploit our embeddings
                                                        to uncover functional modules. We exemplify our ideas through case studies focusing on directed
                                                        networks without strong connectivity, and signed networks. We further highlight how certain ideas
                                                        from community detection can be generalized and linked to Control Theory, by using the here
                                                        developed dynamical perspective.


                                                            I.   INTRODUCTION                              that a coarse-grained description at the aggregated level
                                                                                                           of social circles may be sufficient to describe the process.
                                            Complex systems comprising a large number of inter-               A classical source for dimensionality reduction is the
                                         acting dynamical elements commonly display a rich reper-          presence of symmetries in the system [4, 5], or the presence
                                         toire of behaviors across different time and length scales.       of homogeneously connected blocks of nodes. Yet, strict,
                                         Viewed as collections of coupled dynamical entities, the          global symmetries are rare in real complex systems, and
                                         dynamical trajectories of such systems reflect how the            while a statistical approach can be used to interpret the
                                         topology of the underlying graph constrains and moulds            irregularities as random fluctuations from an ideal model,
                                         the local dynamics. Even for networks without an in-              e.g., stochastic blockmodels [6, 7], such models often
                                         trinsically defined dynamics, such as networks derived            posit strong locality assumptions such as i.i.d. edges. In
                                         from relational data, a dynamics is often associated to           particular, many global features, such as cyclic structures
                                         the network data to serve as a proxy for a process of             and higher order dynamical couplings, cannot be captured
                                         functional interest, e.g., in the form of a diffusion process.    within such a block structure paradigm [8, 9].
                                         Comprehending how the network connectivity influences                Embedding techniques, which define an (often low-
                                         a dynamics is thus a task arising across many different           dimensional) representation of the network and its nodes
                                         scientific domains [1–3].                                         in a metric vector-space, have thus gained prominence
                                            However, it is often impractical to keep a full description    recently [10, 11], as they allow us to use a plethora of
                                         of a dynamics and the network for system analysis. In             computational techniques that have been developed for
                                         many cases it may be unclear how such an exhaustive               analysing data in vector spaces. Thus far most of these
                                         description could be interpreted, or whether such finely          techniques have focussed squarely on representing topo-
                                         detailed data is necessary to understand the phenomena            logical information such as communities, e.g., by using a
                                         of interest. Accordingly, many studies aim to reduce the          geometry induced by diffusion processes. However, net-
                                         complexity of the system by extracting lower dimensional          works often come equipped with a more general dynamics
                                         descriptions, which explain the behavior of interest in a         than diffusion, or contain signed and directed edges, for
                                         simpler manner with fewer, aggregated variables.                  which it is not clear how to define an appropriate diffusion.
                                            This reductionist paradigm may be illustrated with the            Inspired by notions from control theory, here we pro-
                                         process of opinion formation in a social network. In gen-         pose a dynamical embedding of networks that can account
                                         eral, there will be many actors in the network, organized         for such cases. Our embedding associates to each node
                                         in different social circles and influenced by various agents,     the trajectory of its (zero-state) impulse response. As
                                         media, etc. While the full dynamics is highly complex             illustrated in Figure 1 we construct, for each time t, a
                                         and variable, the globally emerging dynamics may still            representation of the nodes in signal (vector)-space, which
                                         evolve on an effective subspace of low dimensionality, such       provides us with a dynamics-based, geometric representa-
                                                                                                           tion of the system, and associated similarity and distance
                                                                                                           measures. Nodes that are close in this embedding induce
                                         ∗ mschaub@mit.edu
                                                                                                           a similar state in the network at a particular time scale t
                                         † m.barahona@imperial.ac.uk
                                                                                                           following the application of an impulse.
                                                                                                                                             2


                                                                                                             Example Tasks:
                                                              High similarity          Similarity
                                                                                                               * Dimensionality reduction

                                                                                                               * Communities / Clustering

                                                                                                               * Node importance / centrality
                                                              Low similarity
                                                                                       Distance                * Network comparison

                    ...
 Node impulse     Node trajectories in     Time-dependent          Time-dependent dynamical                      Learning tasks in
  responses          state space             node vectors           node similarity / distance              vector space representation

Figure 1. Schematic of constructing dynamical similarity measures. Impulses are applied as inputs to different nodes
of the network. The responses in time are interpreted as node vectors evolving in state space, and can be compared, e.g., via an
inner product from which we construct the similarity matrix Ψ(t), or alternatively, its associated distance D(2) . Nodes that drive
the system similarly (differently) within the projected subspace are assigned a high (low) similarity score. The thus derived
vector space representation of the nodes can be used for a number of different learning tasks.


  We can exploit this vector space representation and                   II. DYNAMICAL EMBEDDINGS OF
the associated similarity and dual distance measures for             NETWORKS AND NODE DISTANCE METRICS
various analysis tasks. While such representations are
amenable for general learning tasks, in this work we focus              A.       An illustrative example of dynamical node
on two examples that highlight particular features of in-                                       similarity
terest for dynamical network analysis. First, we illustrate
how low-dimensional embeddings of the system can be                   To fix ideas, let us envision our system in the form of a
constructed — providing a dimensionality reduction of               discrete time random walk dynamics on a network of n
the system in continuous space. Second, we illustrate               nodes:
how these ideas can be exploited to uncover dynamical
modules in the system, i.e., groups of nodes that act                                               yt+1 = M > yt ,                         (1)
approximately as a dynamical unit over a given time                 where M = K −1 A is the transition matrix of an unbiased
scale, and discuss how these modules can be related to              random walker, A is the (weighted) adjacency matrix,
notions from Control Theory — an important topic that               and K = diag(A1) is the diagonal (weighted) out-degree
has gained prominence recently in network theory. We                matrix.
further show how our embeddings provide links between                  The entries of vector y(t) correspond to the the proba-
certain ideas from model order reduction and control the-           bilities of the random walker to be present at each node
ory on the one hand, and notions from network analysis              at time t. As each variable is identified with a node of
and low-dimensional embeddings on the other hand.                   a graph, we can assess whether two nodes play a similar
                                                                    dynamical role as follows. Let us inject an impulse at
                                                                    node i at time t = 0 and observe the response of the
                                                                    system yi (t) ∈ Rn . In the context of our diffusion system
                                                                    this means fixing all the probability mass at node i at
                                                                    time t = 0 and observing its temporal evolution over time.
  The paper is structured as follows. In Section II we              We define the mapping i 7→ yi (t), which associates to
introduce dynamical similarity and distance measures,               each node its zero-state impulse response. This mapping
their theoretical underpinnings and discuss various in-             embeds the nodes into a space of signals, and we can thus
terpretations of the measures we derive. The similarity             use any suitable similarity measure between the signals
measures and the associated distances can be utilized               yi (t) and yj (t) to define a node similarity.
in different ways for system analysis as we illustrate as              To quantify whether the impact of node i in the network
we illustrate in Sections III to V. Section III focusses on         is aligned with the impact of node j at a particular time
applications of our embeddings for ranking, as illustrated          t, a wide variety of similarity functions between yi (t)
by the analysis of an academic hiring network. Section IV           are possible, including nonlinear kernels [12]. However,
highlights how our framework can be used for (dynamical)            we find it convenient to use the standard bilinear inner
dimensionality reduction for signed social networks, using          product:
a network of tribal interactions as example. Section V                                             
then discusses how we can detect functional modules in                               Ψ(t) = ψij (t) i,j=1,...,n              (2)
a signed network of neurons. We conclude with a brief                                                                       >
discussion in Section VI.                                                       with   ψij (t) = hyi (t), yj (t)i = yi (t) yj (t).
                                                                                                                              3

                                                                                                           Adjacency matrix
Note that Ψ does in general not indicate the presence of
regions in which the flow is trapped; instead, the similarity
between two nodes i, j is defined by how aligned the
influence of an impulse emanating from nodes i, j is after
a time t. Accordingly, a high dynamical similarity does
not necessitate direct proximity in the underlying graph.
   Figure 2 illustrates some of the key aspects of the
                                                                    Example
dynamical similarity measure defined in those terms for
an example graph equipped with a diffusion dynamics.
As can been seen, e.g., nodes 5 and 6 in the pink group
behave similarly over short time-scales (t = 1), whereas
the other nodes behave more distinctly. Over intermediate
time scales (t = 8), the similarity of the nodes converges
into four blocks: the pink group, the cyan group, and two
subgroups within the green cycle subgraph (1–2, 3–4). At
longer time scales (t = 16) the similarity of the nodes,
may be approximated by three dynamical blocks (green,
pink, cyan).
   While the above example hints at how our similarity
measure may be employed for the detection of dynamically
cohesive modules, note that in contrast to many methods
used to detect graph communities based on diffusion [13–        Figure 2. Constructing dynamical similarity measures.
16] or on the propagation of a perturbation [17, 18], the       A Visualization of an asymmetric directed network (not
above formulation in terms of response dynamics does            strongly connected) and its adjacency matrix. Note that
not require the graph to be strongly connected. Further,        the which contains a bipartite (disassortative) substructure.
                                                                                                     >
there is also no notion of ‘assortative’ network structure      B Similarity matrix Ψ(t) = M t [M t ] for times t = {1, 8, 16}.
built into the similarity measure: as seen in Figure 2,
cyclic and bipartite structures are identified in a naturally
interpretable manner over particular time scales. However,      correspond to a degree weighting W = diag(d), where d
let us emphasize here that the purpose of the embedding         is the vector of node degrees, such that the influence on
is not to detect topological meaningful communities, but        nodes with a higher-degree will be weighted more strongly.
to quantify in how far nodes behave dynamically similar,           Instead of an inner product, other measures of similarity
which is a different objective [19].                            between the responses yi could be considered, such as
                                                                different correlations, or information theoretic measures.
                                                                However, defining the similarity via an inner product is
   B.    General dynamical similarity and distance              conceptually appealing as there is an associated distance
                     measures                                   matrix D(2) (t), whose entries correspond to a squared
                                                                Euclidean distance of the form:
  Let us now formalize the above ideas in more general             (2)          1

terms and consider the following linear dynamics:                Dij (t) = kW 2 (yi (t)− yj (t)) k2 = ψii + ψjj − 2ψij . (5)

        ẋ = Ax + Bu A ∈ Rm×m ,       B ∈ Rm×p          (3a)       The time parameter inherent to both Ψ(t) and D(2) (t)
                                                                may be understood as a sampling of the network dynam-
        y = Cx          C ∈ Rn×m ,                      (3b)    ics at a particular time-scale, which enables us to focus
where x ∈ Rm , y ∈ Rn , u ∈ Rp are the state, the ob-           on different time scales of interest. For instance, we can
served state, and the input vectors, respectively. While        ignore fast paced transients τ and consider only long-time
discrete-time systems are also of interest (Fig. 2), we will    behaviors for t ≥ τ . As a concrete example, consider
in the following primarily stick to the continuous time         again a diffusion dynamics in discrete time as shown in
formulation for simplicity. All our results can however be      Figure 2B-C, where different time-scales provide different
naturally translated to discrete time.                          meaningful descriptions. Setting t = 1 amounts effectively
   Based on eq. (3), we collect the (zero state) impulse        to a structural analysis in which merely the direct cou-
responses yi for every node i and assemble them into the        pling is considered; setting t > 1 amounts to integrating
matrix Y (t) = [y1 , . . . , yn ] = C exp(At)B. We now define   information over multi-step pathways [8, 20]. If we are
the similarity matrix Ψ(t) as:                                  interested in features persistent over a range of times, we
                                                                may also integrate over t. This integration eliminates the
                                  >                             time-dependence, and we recover an interesting connec-
  Ψ(t) = Y > WY = B > exp (At) C > W C exp(At)B. (4)
                                                                tion to Gramian matrices considered in Control Theory
where we allowed for a weighted inner-product by includ-        (see next subsection).
ing the matrix W. For instance, we may choose W to                 Instead of using the matrix W as a weighting, we may
                                                                                                                                4

                                                                      (2)
alternatively employ it akin to a ‘null model’ term. For         D[0,t] is simply proportional to the resistance distance κij
instance, we can chose W to project out the average of y         between nodes i and j [23], i.e.,
or certain other components (see also Appendix B). If the
system (3) corresponds to a diffusion processes, by chosing
                                                                           h
                                                                              (2)
                                                                                  i      κij  1          >
                                                                     lim D[0,t]        =     = (ei − ej ) L† (ei − ej ), (10)
an appropriate projection, we can recover concepts such as          t→∞             ij    2   2
the modularity matrix and its generalisations as specific        where L† is the Moore-Penrose pseudoinverse of the Lapla-
cases (see Appendices B and C). However, the above               cian, and ei is the i-th unit vector.
formulation can equally be applied for other dynamics,              The integrated Gramian Ψ[0,t] in (8) can also be in-
as we will showcase in Sections III to V.                        terpreted in terms of an observability / controllability
                                                                 Gramian considered in Control Theory. Specifically, con-
                                                                 sider the Gram matrix based on the L2 inner product:
  C.    Interpretations of dynamical similarites and                                              Z t
                       distances
                                                                                   hfi , fj iL2 =     fi> fj dt
                                                                                                    0
   Before discussing specific applications of the above
                                                                 between the vector functions fi : [0, t] → Rn defined via
dynamical similarity measures, let us examine some prop-
                                                                 the mapping fi : t 7→ C eAt ei . This is precisely the finite-
erties of the above measures in more detail.
                                                                 time observability Gramian GO (t) of the linear system (3),
   First, note that the definition of the dynamical similar-
                                                                 which is defined as:
ity Equation (4) can be written in the form:                                            Z t
                                                                                                >

                       Ψ(t) = B > Ξ(t)B,                   (6)                 GO (t) =     eA t C > C eAt dt.            (11)
                                                                                              0

where Ξ(t) is governed by the following Lyapunov matrix          [GO ]ij quantifies how inferable the initial state at node i
differential equation [21]:                                      is from output j. Hence, a high value of the entry [GO ]ij
                                                                 signifies that node j is highly observable from node i,
        dΞ                                                       when C = I. More precisely, each entry reflects how
           = A> Ξ + ΞA,        with   Ξ(0) = C > WC.       (7)
        dt                                                       the energy of the initial states (localized on the nodes)
Thus, Ψ(t) is a dynamically evolving positive semi-definite      spread to the outputs [24]. From our discussion above we
Gram matrix, or a dynamic kernel matrix. The same type           may alternatively say that the observability Gramian (11)
of Lyapunov equation also governs the evolution of the           measures the similarity between two nodes in terms of
covariance matrix of the system (3) driven by white Gaus-        their dynamical response over the interval [0, t].
sian noise [21, 22], which yields another interpretation of         Accordingly, Ψ(t) can be interpreted as an instanta-
the above similarity measure.                                    neous Gramian corresponding to a particular time in-
                                                                 stance t, i.e., Ψ(t) may be understood as computing inner
                                                                 products between sampled zero-state impulse response
  1.   Integrated dynamical similarity and control-theoretic     trajectories t 7→ C eAt Bei at a particular time t. Indeed,
                    interpretations of Ψ(t)                      our measure Ψ(t) can be rewritten as
                                                                                                 dGO (t)
   Instead of selecting a particular time t in our similarity                          Ψ(t) = B >         B.
                                                                                                     dt
measure, we may integrate over time and thus define the
                                                                 As GO has the interpretation of an energy, the entries
integrated similarity measure
                                                                 of Ψ(t) may thus be interpreted as a power transferred
                               Z t                               between the nodes.
                     Ψ[0,t] :=     Ψ(t)dt.                (8)       It is well known that there exists a duality between the
                               0                                 observability of a system and the controllability of the sys-
Analogously, we define the associated integrated squared         tem governed by transposed matrices. We may thus also
distance matrix:                                                 view Ψ(t) as assessing the instantaneous controllability of
                                                                 a dual system to (3) obtained by making the transforma-
                 (2)
               D[0,t] = 1z> + z1> − 2Ψ[0,t] ,              (9)   tion (A, B, C) → (A> , C > , B > ). In a similar vein, we can
                                                                 explore the dual controllability measure in that system.
where z = diag(Ψ[0,t] ) is the column vector containing          A more detailed investigation of these directions will be
the diagonal entries of Ψ[0,t] (cf. Equation (5)). Only          the object of future work.
longer lived features will contribute significantly to the
this integral, and thus short-lived features are integrated
out. If we are mostly interested in features that are            2.     Relations to time-scale separation, low-rank structure and
dominant over a certain range of time-scales we may thus                                   model reduction.
employ the integrated similarity.
  Note that for a diffusion dynamics on an undirected              Asymptotically, the dynamics of many networked sys-
graph with A = −L, C = I −11> /n, the distance measure           tems converges to a lower dimensional manifold. Think,
                                                                                                                                   5

for instance, of synchronisation processes. In structured       have different types of dynamical modules, depending on
networks, however, one typically observes that the state        the dynamics acting on top of it. In section Section V, we
transition matrix, and therefore the similarity matrix          will see an example in which the structural grouping and
Ψ(t), becomes numerically low-rank at much early times.         the dynamical grouping of the nodes are indeed different.
Stated differently, in many structured networks we observe
time scale separation linked to low dimensional subspaces
of slowly decaying metastable states. Therefore, the sys-        III.   DIMENSIONALITY REDUCTION USING
tem can be effectively described by a small set of slow                    DYNAMICAL DISTANCES
modes that govern the dynamics over some time scale.
A feature specific to networked systems is the fact that           In this section we outline how the above dynamical
these slow modes can be localized on the space of nodes.        similarity measures may be employed for dimensionality
It then follows, that instead of having to account for the      reduction.
whole system, we may just keep track of a few aggregated           Consider the spectral decomposition of Ψ(t) into its
‘metanodes’, whose state is governed by the slow modes,         eigenvectors v1 (t), v2 (t), . . . , vn (t) with associated eigen-
thereby reducing the complexity of the dynamics.                values µ1 (t) ≥ µ2 (t) ≥ · · · ≥ µn (t). We define the map-
   For a Laplacian diffusion dynamics (A = −L) this             ping i 7→ φi (t):
idea can be made more precise using so-called externally
equitable partitions [25], which explicitly relate our simi-                   √         √                 √         >
                                                                     φi (t) = [ µ1 v1,i , µ2 v2,i , . . . , µn vn,i ] .          (14)
larity measure to model reduction. Consider an external
equitable partition (EEP) characterized by the relation         Using simple algebraic manipulations, it can now be shown
                                                                that our dynamical distance measure (5) can be written
                      LHEE = HEE L,
                                 b                      (12)    as:
where HEE is an indicator matrix encoding the EEP, and                              (2)
                                                                                 Dij (t) = kφi (t) − φj (t)k2 .                  (15)
                      −1
     b = (H > HEE )
     L                      >
                           HEE         +
                               LHEE = HEE LHEE          (13)
           EE                                                   Hence, the vectors φi map the data into a Euclidean space,
is the Laplacian of the quotient graph, the graph in which      in which the (Euclidean) distance is aligned with the
each group of the partition becomes a ‘metanode’.               dynamical impacts of the nodes at time t. The entry-wise
   It can be shown that if we observe such a system             squared distance matrix D(2) (t) can thus be approximated
through its projection onto this external equitable par-        by keeping only the first c coordinates in each mapping
                           +
tition (i.e., we set C = HEE  ), then every node within a       φi (t), thereby producing a low dimensional embedding of
group will have exactly the same influence on the observed      the original system.
output trajectories. The similarity matrix Ψ(t) can be             For a diffusion dynamics with either −A = L (the com-
written in terms of the quotient graph as:                      binatorial Laplacian) or −A> = Lrw (the random walk
                                                                Laplacian matrix), it can be shown that D(2) (t) corre-
                          + >     +>                            sponds precisely to the distance induced by diffusion maps
        Ψ(t) = exp(−Lt) (HEE ) HEE  exp(−Lt)
               h            i>                                  if the weighting matrix W is chosen appropriately [28, 29].
             = exp(−Lt)H +
                               exp(−Lt)H +                      To see this, note P that from the orthogonal spectral de-
                                         EE ,
                     b               b
                         EE
                                                                composition L = i λi vi vi> , it follows that
which shows that Ψ will be block-structured. Consequently
                                                                                                                             >
the dynamics of the full system within the subspace                          φi (t) = [e−λ1 t v1,i , . . . , e−λn t vn,i ]
spanned by the partition can be described exactly by
a reduced model [25, 26], which is governed here by             are a time-dependent diffusion map embedding [28, 29].
A = L,  e C = I and has only a single input per group,
equal to the average input within the original group.
   It is instructive to compare the above dynamical block-              Analysing academic hiring networks via
structure to the notions like stochastic block-models [6, 7],                low-dimensional embeddings
in which each node in a group has statistically the same
(static) connection profile. Here we are interested in nodes       Which universities are the most prestigious in North
that have dynamically the same effect, and define nodes         America? In a recent study, Clauset et al. [27] provided a
accordingly. Note however, that the connections formed          data-driven assessment of this question by examining the
by each node do not have to be the same, but simply lead        hiring patterns US-universities by means of a minimum
to similar dynamical effects: our measures assess the node      violation ranking. This ranking aims to order universi-
similarity with respect to some observable y and not with       ties such that the fewest number of directed links, corre-
respect to the connections formed. In other words our           sponding to faculty hirings, move from lower-ranked to
objective is to obtain a joint low-dimensional description      higher-ranked universities. Stated differently, universities
of the dynamics of the system and localized features of         with a higher prestige are assumed to act as sources of
the network structure. For the same network we may              faculty for lower-ranked universities.
                                                                                                                                  6

     A                                                                  B            Ranking




                                            Harvard
                                              Urbana Champaign




Figure 3. Analysing academic influence using low-dimensional embeddings A We develop a low-dimensional embed-
dings based on the influence dynamics in the hiring network, as described in the text. The first two dimensions of this embedding
                                     [0,1]
are plotted. The first coordinate φi,1 is strongly correlated with the prestige ranking of Clauset et al. [27], highlighting the
influential role played by the universities at the top. Interestingly, the second dimension distinguishes the Canadian universities
from the US universities, showing that although these universities are well integrated within the faculty hiring market [27], they
play a different role and exert a different type of influence on the system. B The ranking obtained when projecting onto the
first coordinate only and the associated subgraph of faculty hirings. The numbers in parenthesis correspond to the rankings
obtained by Clauset et al [27]. The arrows are proportional to the number of faculty moving between the institutions. Arrows
pointing downwards in the ranking are plotted on the left, arrows point upward in terms of the ranking are plotted on the right.
The number of hirings from inside the institution itself (self-loops) are indicated by size of the core (darker color) of each node.
The area of the core is proportional to the number of self-loops compared to the total out-degree within this subnetwork.


   The dataset was released by Clauset et al.[30] and               laxation dynamics of the recently proposed ‘spring rank’
consists of the placement of nearly 19,000 tenure track or          formalism, whose long-term behavior would then corre-
tenured faculty among 461 North American departmental               spond to the spring rank [31].
or school level academic units. The hiring data was                    As the hiring graph is not strongly connected, the
collected for the disciplines business (112 institutions),          long-term behavior will be dominated by a few modes
history (144 institutions), and computer science (205               depending on the initial condition. We thus concentrate
institutions). In contrast to the history and business data,        here on short time-scales for which paths of shorter lengths
the computer science hiring data included 23 Canadian               will be more important. To avoid having to choose a
institutions.                                                       particular time parameter, we integrate with respect to
   Here we reconsider the ranking question using our above          t ∈ [0, 1]. Note that while the underlying network is
defined distance measure. To illustrate our procedure, let          not strongly connected, there is no need to introduce a
us focus on the computer science (CS) data first. Consider          teleportation into the dynamics as is commonly the case
the adjacency matrix A of the graph of hiring patterns              in diffusion based methods.
for CS, where Aij denotes the number of faculty moving                 To derive a low-dimensional embedding we approximate
from university i to university j. This provides us with a                                                                   (2)
                                                                    the resulting (squared) dynamical distance matrix D[0,1]
directed, weighted network with 205 nodes corresponding             via a low-rank spectral decomposition of Ψ[0,1] = V ΛV T .
to CS units at the departmental or school level, where we                                          [0,1]
ignore movements of faculty to/from entities outside this           To this end we define φi               via the relation
set of 205 units.                                                                [0,1]
                                                                               [φ1       , . . . , φ[0,1]
                                                                                                    n ]=Λ
                                                                                                          1/2 T
                                                                                                             V =: Φ[0,1] .    (16)
   Let us denote the influence of university by the state
variable xi . We posit that a university i exerts an influence      The vectors φi
                                                                                     [0,1]
                                                                                       define a new coordinate system, whose
on another university j by sending faculty members to               coordinates are ranked according to their importance to
it. To normalize for the size of the universities we divide         the dynamics. We note that
this influence by the in-degree of each university, i.e., an                     h        i                    2
influence of size 1 may be exerted on each university. This                         (2)
                                                                                   D[0,1]
                                                                                                  [0,1]
                                                                                            = φi − φj
                                                                                                         [0,1]
                                                                                                                 ,      (17)
leads us to consider an influence dynamics of the form                                        ij

                            −1 T
                     ẋ = [Kin A − I]x                              and thus our dynamical distance can be approximated by
                                                                    truncating our coordinate system to the first few compo-
among the universities, where Kin = diag(A> 1) is the               nents of the vectors φi (t).
diagonal matrix of in-degrees. Note that one could also                Figure 3 shows the results of this procedure when ap-
consider alternative artificial dynamics here, e.g., the re-        plied to the CS dataset of Clauset et al. [27]. We find
                                                                                                                                      7

                                                               History
      A                                                                  B         Ranking
                                                                                        1. Harvard University (1)
                                                                                        2. Yale University (2)
                                                                                        3. Columbia University (7)
                                                                                        4. UC Berkeley (3)
                                                                                        5. University of Chicago (6)
                                                                                        6. Princeton University (4)
                                                                                        7. Stanford University (5)
                                                                                        8. University of Michigan (12)
                                                                                        9. University of Wisconsin, Madison (11)
                                                                                        10. UCLA (13)
                                                                                        11. University of Pennsylvania (10)
                                                                                        12. Johns Hopkins University (9)
                                                                                        13. Cornell University (15)
                                                                                        14. Brown University (16)
                                                                                        15. University of Virginia (25)



                                                            Business
      C                                                                  D        Ranking




Figure 4. Analysing academic influence using low-dimensional embeddings A Low-dimensional embeddings based on
the influence dynamics in the hiring network, as described in the text (see Figure 3) for the discipline History. B The ranking
obtained when projecting onto the first coordinate only and the associated subgraph of faculty hirings (see Figure 3). The
Spearman rank correlation to the results obtained by Clauset et al is ρ ≈ 0.92. C Low-dimensional embeddings based on the
influence dynamics in the hiring network, as described in the text (see Figure 3) for the discipline Business. D The ranking
obtained when projecting onto the first coordinate only and the associated subgraph of faculty hirings (see Figure 3). The
Spearman rank correlation to the results obtained by Clauset et al is ρ ≈ 0.96.


                               [0,1]
that the first coordinates φi,1 are strongly correlated              With Canadian institutions absent from the data for
with the previously obtained ranking [27] (Spearman rank-         History and Business, the second dimension of the embed-
correlation ρ ≈ 0.90), i.e., our dimensionality reduction         ding appears to not correlate clearly with a geographical
maintains the essential features of the identified prestige       feature. For the History dataset, the Southern Baptist
hierarchy. In addition, our embedding reveals that the            Theological Seminary is singled out in our second projec-
Canadian universities play a somewhat different role in the       tion coordinate. One of the main differences of this unit
                                         [0,1]
system. Indeed the second coordinate φi,2 is singling out         is its relatively large number of self-loops in the hiring
Canadian universities, highlighting that not all features of      data (11 hirings come from the same institution), leading
the influence dynamics are captured well by a unidimen-           to a highly localized influence of this institution. Indeed,
sional ranking (see Figure 3). When symmetrizing the              the coordinate of all other institutions is essentially zero
network, the Spearman correlation of the first dimension          in this second embedding dimension.
with the minimum violation ranking of Clauset et al. [27]           For the business data there is a slight separation along
drops markedly to ρ ≈ 0.80, emphasizing again that the            the second dimension. More coastal regions (West, North-
directionality in this network is an essential feature.                                         [0,1]
                                                                  east) tend to have a higher φi,2 projection. South and
                                                                                                                                   [0,1]
   In Figure 4 we show the corresponding analyses for             Midwest institutions tend to have a lower coordinate φi,2 .
the disciplines history and business. As shown, from the          However, the separation of the 3 top institutions may be
first dimension of the embedding we can again derive an           better explained by their relative position in the network.
influence ranking that is strongly correlated to the results      First, these are the only 3 institutions that placed more
obtained by Clauset et al.                                        than 300 faculty members (outdegree 412 Stanfard, 364
                                                                                                                           8

 MIT, and 344 Harvard). The next largest institution in          in Fig. 5) or antagonistic (‘rova’; blue edges in Fig. 5). A
 terms of this placement is the University of Michigan          ‘hina’ edge signifies political alignment and limited feuds.
(outdegree 282). This large direct influence is further         A ‘rova’ edges denote relationships in which warfare is
 boosted as not only are there strong ties from these top 3      commonplace.
 institutions to most lower ranked universities, but also a
 relatively strong circular influence among these three top
 institutions. A substantial fraction of the hirings of each      Spectral partitioning and dynamical embeddings
 of these 3 institutions comes from within their own small
‘rich club’.
                                                                   To illustrate how our dynamical embedding can provide
    While we focussed here on the first two embedding di-
                                                                further insight into such a system with signed interac-
 mensions, there is no reason to focus only on 2 dimensional
                                                                tions, let us initially focus on a discrete categorization of
 projections a priori. Indeed the very same procedure can
                                                                the nodes into clusters, instead of finding a continuous
 be applied to more dimensions, which could lead to a
                                                                embedding for our system. Many methods have been
 more nuanced appraisal of the relative influence of these
                                                                proposed to cluster signed networks [34, 37, 38] that can
 institutions in the hiring network. Our focus here was
                                                                be used to find the groupings in the here considered set-
 on the conceptual aspects of these embeddings, but a
                                                                ting. All of them follow a combinatorial approach and
 more detailed investigation, potentially linking these re-
                                                                aim to find dense groupings in the network containing
 sults to the relaxation dynamics of the recently proposed
                                                                a maximum number of positive links within the groups
 SpringRank method [31], would be an interesting subject
                                                                and most negative links across groups. Perhaps the most
 of future investigations.
                                                                straightforward way to split the nodes of the network into
                                                                blocks is an approach based on spectral clustering [39].
 IV.   DYNAMICAL EMBEDDINGS OF SIGNED
                                                                For signed networks, such a spectral clustering based on
       SOCIAL INTERACTION NETWORK                               the signed Laplacian may be interpreted as optimizing a
                                                                signed ratio cut [40], which provides a principled way to
                                                                detect groups in a signed network.
   Many networked systems contain both attractive and
                                                                   To split a system into k groups, we assemble the matrix
repulsive interactions. Examples include social systems,
                                                                Vc containing the c eigenvectors corresponding to the
in which people may be friends or foes, or genetic net-
                                                                smallest eigenvalues of Ls . (Note that Vc also corresponds
works, in which inhibitory and excitatory interactions are
                                                                to the c dominant eigenvectors of Ψ(t) for t > 0.) The
commonplace. Such systems can be represented as signed
                                                                rows of Vc are then taken as new c-dimensional coordinate
graphs, with positive and negative edge weights. A simple
                                                                vectors for each node on which a k-means clustering is
model for opinion formation on signed networks is given
                                                                run to obtain the k modules. Though, in general, the
by [32, 33]:
                                                                dimension of the coordinate space c and the number of
                      ẋ = −Ls x + u,                   (18)    modules k need not be the same, one typically chooses
                                                                c = k or c = k − 1 [39]. To showcase the utility of this
where the signed Laplacian matrix is defined as                 procedure, we applied this form based on the 2 dominant
Ls = Ds − As and the state vector x describes the ‘opin-        eigenvectors (v1 and v2 ) to split the network into k = 2, 3
ion’ of each node.                                              groups (Fig. 5A). The blocks obtained are characterized
   Here As is the adjacency matrix of the network, with         by high internal density of positive links with negative
positive and negative edge weights, and Ds is the matrix        links placed across groups. Interestingly, if we aim to
containing the weightedPabsolute strengths of the nodes on      cluster the network into k = 2 groups using only c = 1
the diagonal, [Ds ]ii = k |(As )ik | and [Ds ]ij = 0 for i 6=   eigenmodes of Ls , we obtain a grouping in which the
j. The signed Laplacian is positive semidefinite [32, 34]       Seu’ve tribe is place together with the Gama, Kotuni,
and reduces to the standard combinatorial Laplacian if As       Gaveve, and Nagamidzuha tribe, and the remaining tribes
contains only positive weights. Clearly, this dynamics is       form a second group (see Figure 5).
of the form (3) discussed in the main text, with A = −Ls           To gain additional insight, we study the dynamical
and B = C = I. In this case, the dynamic similarity             coordinates φi (t) defined in Eq. (14), which can be seen
                                                                as feature vectors that combine the information of the
                                >                               eigenvectors and eigenvalues. The time evolution of these
             Ψ(t) = exp(−Ls t) exp(−Ls t),              (19)
                                                                feature vectors provides a dynamical embedding of the
has time-independent eigenvectors vi and associated eigen-      signed opinion network, reflecting the relative position of
values µi (t) = e−λi t , where the vi and λi are eigenvectors   the nodes (tribes) in the state-space of ‘opinions’. Instead
and eigenvalues of Ls .                                         of providing a discrete categorization, the continuous na-
   Let us consider the network of relationships between         ture of the embedding provides us with a more nuanced
16 tribal groups in New Guinea chartered by Read [35]           view on how closely aligned individual tribes are to each
and first examined in the social network literature by          other over time (Figure 5A). Note that the spectral clus-
Hage and Harari [36]. The relationships between the             tering with c = 1 discussed above corresponds essentially
different tribes are either sympathetic (‘hina’; red edges      to the long-term behaviour of this dynamics. Our dynam-
                                                                                                                                   9




                                          alternative
                                          split




Figure 5. Analysis of a signed social network: the highland tribes in New Guinea. A The network of 16 tribes with
positive interactions (‘hina’) in red and negative interactions (‘rova’) in blue. Spectral clustering using c = 2 eigenvectors of the
signed Laplacian Ls . The top 2 eigenvectors of Ψ(t) reveals partitions into k = 3 and k = 2 groups with positive interactions
mostly concentrated within groups, and antagonistic interactions across groups. If we instead try to split the signed network
into k = 2 groups based on c = 1 eigenvector, we obtain an alternative split as indicated by the dashed gray line. B The time
evolution of the tribes in state space under the consensus dynamics (18) is represented through the dynamical embeddings φi (t).
Here we plot only the first two dominant coordinates. As time grows, the Seu’ve tribe switches from a marginal allegiance to the
pink/green groupings to be grouped with the blue block. This is the result of an ‘enemy of my enemy is my friend’ effect.


ical embedding shows that the Seu’ve tribe has effectively           Seu’ve tribe to be ‘turned around’.
a zero, but slightly negative coordinate within direction               Indeed, instead of using the eigenvector of Ls , we may
φi,1 . Hence, if we concentrate only on c = 1 eigenvec-              alternatively use the dynamical φi coordinates for clus-
tor the obtained split will be commensurate with the φi,1            tering, thereby taking into account the eigenvalues of
coordinate, which is exactly the partition obtained before.          the dynamics as well. For k = 3 groups the resulting
   To understand why the impact of the Seu’ve tribe on               clustering is the same as the one we obtain from spectral
the network in terms of the φi,1 coordinate is indeed neg-           clustering based on the eigenvectors of Ls alone. However,
ative, it is instructive to examine the position of Seu’ve           for k = 2 the split is somewhat different. For all but the
in the network in a bit more detail. Note that the Seu’ve            largest time-scales the green and pink groups are merged,
tribe has 2 direct positive links with the Nagamiza and              and only for very large time-scales does the Seu’ve tribe
the Uheto tribe. Seu’ve has also negative with the Uku-             ‘flip’ and become part of the group containing the Gama
rudzuha, Asarodzuha and the Gama tribe. The split into               tribe.
3 groups is exactly aligned with these positive and nega-
tive relationships. The spectral split into 2 groups based
on the first dominant vector appears to be at odds with                V. FINDING FUNCTIONAL MODULES IN
these relationships, though.                                           NEURONAL NETWORKS VIA DYNAMICAL
   The reason for this at first sight non-intuitive split is                  SIMILARITY MEASURES
a behavior of the type “the enemy of my enemy is my
friend”, which is inherent to signed interaction dynamics.             As a final example for the utility of our embedding
This effect plays a more important role for larger time             framework, we now consider the analysis of networks of
scales and is thus reflected in the sign patters of the             spiking neurons. Specifically we will consider the dynam-
dominant eigenvector. In this case, the mutual antipathy            ics of a network of leaky-integrate-and-fire (LIF) neu-
of all three Seu’ve, Gama and Nagadmidzuha clans against            rons. Due to their computational simplicity yet complex
the Asarodzuha tribe implies that Gama, Nagadmidzuha                dynamics, networks of LIF neurons are widely used as
and Seu’ve behave in the long run similar and thus have             scalable prototypes of neural activity. Recently, it has
a negative φi,1 coordinate.                                         been shown that LIF networks can display “slow switch-
   Following structural balance theory [41], one may con-           ing activity” [42, 43], sustained in-group spiking that
jecture that the Gama-Seu’ve relationship could cease to            switches from group to group across the network. Impor-
be of ‘rova’ type in a future observation of the network.           tantly, the cell assemblies of coherently spiking neurons
In his socio-ethnographic characterization of this tribal           in this context can include both excitatory neurons and
system, Read indeed remarked that the system was “rela-             inhibitory neurons, and dense clusters of connections are
tive and dynamic” [35]. Our analysis highlights that there          not necessary to give rise to such dynamics (see Figure 6).
is additional information to be gained when adopting a              These cell assembles are thus an interesting example for
dynamical point of view, as shown by potential of the               a functional module, that cannot be discerned from the
                                                                                                                                                                                                                                                                         10

   A                       LIF neural network                           B                                     Synaptic weight matrix                                                Dynamical similarity: linear rate model
                                                                                                                           Neuron index
                                                                                                                     200     400    600   800    1000
                                                                                                                                                                                    Neuron index                           Neuron index
                                                                                                                                                                                  200    400 600     800             200    400 600   800
                                                                                                             200
                   excitatory


                                                                                   Neuron index
                                                                                                                                                                            200                                200
                                                                                                             400



                                                                                                                                                             Neuron index
                                                                                                                                                                            400                                400
                                                                                                             600
                                                                                                                                                                            600                                600
                   inhibitory                                                                                800
                                                                                                                                                                            800                                800
                                                                                                            1000
                                                                                                                       excitatory         inhibitory
                                   10 planted dynamical                                                                      Weight                                                     Similarity                           Similarity
                                        assemblies
                                                                                                                   -0.04   -0.02    0     0.02                                                                       -


                            Nonlinear spiking dynamics
                                                                        C                                          Reordered neuron index
                                                                                                                                                                                               Dynamical blocks
                                                                                                                     200     400    600   800    1000
                                                                                                                                                         1




                                                                                   Reordered neuron index                                                                                                                                            Reordered neuron index
                                                                                                                                                         2
                                                              200                                           200                                                                                                                                200
                                                                                                                                                         3



                                                                    Neuron index
  excitatory
                                                                                                                                                         4
                                                              400                                           400                                                                                                                                400
                                                                                                                                                         5
                                                                                                                                                         6
                                                              600                                           600                                                                                                                                600
                                                                                                                                                         7
                                                                                                                                                         8
                                                                                                            800                                                                                                                                800

  inhibitory
                                                              800
                                                                                                                                                         9
                                                                                                        1000
                                                                                                                                                        10                                                                                    1000
                                                          1000
               0                         1                2                                                                  Weight                             0                                          1                              2
                                     Time (s)                                                                      -0.04    -0.02   0     0.02
                                                                                                                                                                                                      Time (s)

Figure 6. Finding dynamical groups in a neural network description. A Schematic of the connectivity of the leaky
integrate and fire neuronal network, which shows its disassortative feedback structure between inhibitory and excitatory
neurons. The exemplar raster plot illustrating its spiking dynamics shows that the system is characterized by slow switching
between coherent spiking activity of 10 groups of neurons (each containing both inhibitory and excitatory units). B Left: The
weighted, signed and directed synaptic connectivity matrix (WN ) of the network does not contain groups of nodes with high
internal connection density. However, the analysis of the linear rate model governed by this connectivity matrix (20) using the
dynamical similarity (21) in conjunction with a Louvain-type optimization reveals the presence of 10 dynamical modules. Right:
Visualization of the centered similarity (21). For visualization purposes only the diagonal of Ψ⊥ has been removed. Note that
how after an initial short transient period the block structure into 10 groups becomes apparent, in comparison to the original
weight matrix. C The blocks revealed from the linear rate model coincide with the dynamically co-activated groups of neurons
in the full LIF dynamics, as shown by reordering the neuron indices. On the original weight matrix, they correspond however to
a mixture of the blocks inside the weight-matrix WN .


network structure alone.                                                                                                            blocks are homogeneous in terms of the probability of ob-
  As has been shown previously, key insights into the                                                                               serving a connection and their connection link-strengths.
nonlinear LIF dynamics can be obtained from linear rate                                                                             If we were to partition this coupling matrix into homoge-
models of the following form [43], which are amenable to                                                                            neously connected blocks in terms of weights and number
the methodology developed above:                                                                                                    of connections, we would find these 20 structural blocks.
                                                                                                                                       It turns out that this arrangement corresponds however
                                  ẋ = (−I + WN )x + u,                                                            (20)             to only 10 planted dynamical cell assemblies, each con-
where x describes the n-dimensional firing rate vector                                                                              sisting of a mixture of inhibitory and excitatory neurons.
relative to baseline; u is the input; and WN is the asym-                                                                           Thus, while from an inspection of WN we may conclude
metric synaptic connectivity matrix containing excitatory                                                                           that there should be 20 groups, we know from our design
(positive) and inhibitory (negative) connections between                                                                            that there only 10 dynamically relevant groupings [43]. In
the neurons. The asymmetry of WN follows from Dale’s                                                                                order to assess which dynamical role is played by the dif-
principle [44], which states that each neuron acts either                                                                           ferent neurons, we thus consider our dynamical similarity
completely inhibitory or completely excitatory on its ef-                                                                           measure, this time however not with a focus on deriving
ferent neighbours. Clearly, the rate dynamics (20) is of                                                                            an embedding, but with an eye towards identifying the
the form (3).                                                                                                                       planted functional groups in the (nonlinear) dynamics.
   In this example we consider a LIF network, whose                                                                                    Since cell assemblies are characterized by a relative
coupling matrix WN is shown by the signed network in                                                                                firing increase/decrease with respect to the population
Figure 6B. The structure of the network can be described                                                                            mean, we use a centered similarity matrix by chosing a
by a block-partition into 20 blocks: 10 groups of excita-                                                                           weighting matrix of the form W = I − 11> /n.
tory neurons, and 10 groups of inhibitory neurons, whose
                                                                                                                                                                        11>
                                                                                                                                                                           
ordering is consistent with with the network drawing                                                                                                                                          >
                                                                                                                                                   Ψ⊥ (t) = exp(At)  I−       exp(At),                                                               (21)
in Figure 6B. The connectivity patterns between these                                                                                                                    n
                                                                                                                          11

where A = (−I + WN ). As discussed in Appendix B,                                 VI.   DISCUSSION
this can be interpreted as a choice of a null model, or as
introducing a relaxation on the distance matrix different          Building on ideas from systems and control theory, we
to the low-rank approximation discussed in the previous         have presented a framework that provides dynamical em-
section. These type of relaxations of our dynamical simi-       beddings of complex networks, including signed, weighted
larity measures enable us to draw further connections to        and directed networks. These embeddings can be used
quality functions more commonly employed in network             in a variety of analysis tasks for network data. We have
analysis, as we discuss in the next section.                    focused here on applications to dimensionality reduction
                                                                and the detection of dynamical modules to highlight im-
                                                                portant features of our embedding framework. However,
  Revealing dynamical modules with a Louvain-like
                                                                the dynamical similarity measures Ψ(t) and D(2) (t) may
             combinatorial optimization
                                                                also be used in the context of other problem formulations
                                                                not considered here. For instance, we could consider the
  Let us consider a general similarity matrix Ψ defined        (functional) networks induced by our dynamical similarity
via an orthogonal projection W⊥ = I − νν > as weighting         measures, and employ generative models [6, 45, 46] for
matrix:                                                         their analysis. One way to approach this would be to
                           >
      Ψ⊥ (t) = B > exp(At) C > W⊥ C exp(At)B.          (22)     define a (negative) Hamiltonian based on our similarity
                                                                matrix, e.g., in a form similar to (23), and a Boltzmann
 Note that the centered similarity (21) considered for our      distribution of the corresponding form. In this view the
 neuronal network is precisely of this form.                    state-variables of the node (or other labels defined on
    Clearly, the weighted inner product hyi (t), yj (t)iW       the nodes) would correspond to latent variables that are
 projects out particular properties associated with ν. The      coupled via the Hamiltonian.
 choice of W⊥ can thus be interpreted as selecting a type of       One may further consider the extension to kernels com-
‘null model’ for the nodes. Alternatively, we can think of      puted directly from nonlinear dynamical systems, akin
 this operation as projecting out uninformative dimensions      to the perturbation modularity recently introduced by
 of the data, thereby providing a geometric perspective on      Kolchinsky et al. [18], or consider linearisations around a
 the selection of a null-model √(see Appendices B and C).       particular state of interest. Alternatively, Ψ(t) could be
 For instance, choosing ν = 1/ n as done in (21) is equiv-      extended to represent nonlinear systems through an inner
 alent to centering the data by subtracting the mean of         product in a higher dimensional space, e.g, by using the
 each of the vectors yi (t).                                   ‘kernel trick’ [12]. Our measures also provides links with
    Having defined a similarity matrix Ψ⊥ (t) as above, we      other notions of similarity in networks including struc-
 can obtain dynamical blocks with respect to the null           tural, equivalence, diffusion-based [47, 48] and iterative
 model ν as follows. Let us define the quality function         node similarity in networks [49, 50]. Such connections are
                                                                interesting for machine learning, where a good measure
              rν (t, H) = trace H > Ψ⊥ (t) H,          (23)
                                                                of similarity is central to solving problems such as link
where H is a partition indicator matrix with entries Hij =      prediction [51] and node classification [23].
1 if state i is in group j and Hij = 0 otherwise. The              For simplicity, we assumed in the examples above the
combinatorial optimization of rν (t, H) over the space of       number of state variables equals the number of nodes in
partitions can be performed efficiently for different values    the network. Nevertheless, our derivations remain valid
of the time t through an augmented version of the Louvain      when there is more than one state variable per node. For
heuristic.                                                      instance, our ideas may be readily translated to multiplex
   We applied this optimization procedure for the quality       networks [38] or networks with temporal memory [52, 53],
function induced by the similarity matrix (21) to search        that feature expanded state space descriptions and have
for possible functional modules within the neuronal net-        gained considerable interest recently.
work. As can be seen in Figure 6, optimizing (23) reveals          We remark that the measures presented here are differ-
precisely the mixed groups of excitatory and inhibitory         ent from correlation analysis of time-series data, as consid-
neurons that exhibit synchronized firing in the fully non-      ered, e.g., by MacMahon and Garlaschelli [54]. Instead of
linear LIF network simulations. Again, these groups do          interpreting a correlation matrix as a functional network
not correspond to tightly knit groups in the topology (see      from n scalar valued time-series and then analysing this
Figure 6) but rather reflect dynamical similarity.              correlation matrix, we start with the joint description of
   As our example highlights, if we are interested in some      a network and a dynamics.
kind of process on a network, rather than the network              Conceptually, the similarity measure Ψ(t) has strong
structure itself, using a dynamical similarity measure can      theoretical links to model reduction and controllabil-
lead to a more meaningful analysis. The specific example        ity, which provide meaningful interpretations of dynamic
here is however not meant to suggest a particular null          blocks in terms of coarse-grained representations. Classic
model, or a generic optimization method. Indeed, similar        model reduction [55–57] aims to find reduced models that
results can be obtained, e.g., by directly analysing Ψ (or      approximate the input-output behavior of the system;
D(2) ) using spectral techniques as outlined above.            yet the states of the reduced model do not usually have
                                                                                                                              12

a sparse support in terms of the states of the original           the Belgian Network DYSCO (Dynamical Systems, Con-
system. In contrast, the dynamical blocks found using             trol and Optimisation) funded by the Interuniversity At-
Ψ are directly associated with particular sets of nodes           traction Poles Programme initiated by the Belgian State
and can thus be localized on the original graph, an im-           Science Policy Office; and the ARC (Action de Recherche
portant requirement for many applications. Future work            Concerte) on Mining and Optimization of Big Data Mod-
will investigate alternative measures to Ψ(t) based on the        els funded by the Wallonia-Brussels Federation. MTS
duality between controllability and observability Grami-          received funding from the European Union’s Horizon
ans from control, as well as measuring the quality of the         2020 research and innovation programme under the Marie
dynamical blocks in a model reduction sense.                      Sklodowska-Curie grant agreement No 702410. MB ac-
                                                                  knowledges funding from the EPSRC (EP/N014529/1).
                                                                  The funders had no role in the design of this study; the
                ACKNOWLEDGMENTS                                   results presented here reflect solely the authors’ views.
                                                                  We thank Leto Peel, Mauro Faccin, and Nima Dehmamy
  JCD, and RL acknowledge support from: FRS-FNRS;                 for interesting discussions.




 [1] M. E. J. Newman, Networks: An Introduction (Oxford           [20] J.-C. Delvenne, M. T. Schaub, S. N. Yaliraki, and
     University Press, USA, 2010).                                     M. Barahona, in Dynamics On and Of Complex Net-
 [2] A. Arenas, A. Dı́az-Guilera, J. Kurths, Y. Moreno, and            works, Volume 2 , Modeling and Simulation in Science,
     C. Zhou, Physics Reports 469, 93 (2008).                          Engineering and Technology, edited by A. Mukherjee,
 [3] E. Bullmore and O. Sporns, Nature Reviews Neuroscience            M. Choudhury, F. Peruani, N. Ganguly, and B. Mitra
     10, 186 (2009).                                                   (Springer New York, 2013) pp. 221–242.
 [4] L. M. Pecora, F. Sorrentino, A. M. Hagerstrom, T. E.         [21] H. Abou-Kandil, G. Freiling, V. Ionescu, and G. Jank,
     Murphy, and R. Roy, Nature communications 5, 4079                 Matrix Riccati equations in control and systems theory
     (2014).                                                           (Birkhäuser, 2012).
 [5] F. Sorrentino, L. M. Pecora, A. M. Hagerstrom, T. E. Mur-    [22] R. E. Skelton, T. Iwasaki, and D. E. Grigoriadis, A
     phy, and R. Roy, Science advances 2, e1501737 (2016).             unified algebraic approach to control design (CRC Press,
 [6] P. W. Holland, K. B. Laskey, and S. Leinhardt, Social             1997).
     networks 5, 109 (1983).                                      [23] F. Fouss, A. Pirotte, J. Renders, and M. Saerens, IEEE
 [7] T. A. Snijders and K. Nowicki, Journal of classification          Transactions on knowledge and data engineering (2007).
     14, 75 (1997).                                               [24] E. I. Verriest, “Time Variant Balancing and Nonlinear Bal-
 [8] M. T. Schaub, J.-C. Delvenne, S. N. Yaliraki, and                 anced Realizations,” in Model Order Reduction: Theory,
     M. Barahona, PloS one 7, e32210 (2012).                           Research Aspects and Applications, edited by W. H. A.
 [9] R. Banisch and N. D. Conrad, EPL (Europhysics Letters)            Schilders, H. A. van der Vorst, and J. Rommes (Springer
     108, 68008 (2015).                                                Berlin Heidelberg, Berlin, Heidelberg, 2008) pp. 213–250.
[10] B. Perozzi, R. Al-Rfou, and S. Skiena, in Proceedings        [25] N. O’Clery, Y. Yuan, G.-B. Stan, and M. Barahona,
     of the 20th ACM SIGKDD international conference on                Physical Review E 88, 042805 (2013).
     Knowledge discovery and data mining (ACM, 2014) pp.          [26] N. Monshizadeh, H. L. Trentelman, and M. K. Camlibel,
     701–710.                                                          Control of Network Systems, IEEE Transactions on 1,
[11] A. Grover and J. Leskovec, in Proceedings of the 22nd             145 (2014).
     ACM SIGKDD international conference on Knowledge             [27] A. Clauset, S. Arbesman, and D. B. Larremore, Science
     discovery and data mining (ACM, 2016) pp. 855–864.                Advances 1 (2015), 10.1126/sciadv.1400005.
[12] B. Schölkopf and A. J. Smola, Learning with kernels:        [28] R. R. Coifman, S. Lafon, A. B. Lee, M. Maggioni,
     support vector machines, regularization, optimization, and        B. Nadler, F. Warner, and S. W. Zucker, Proceedings of
     beyond (MIT press, 2002).                                         the National Academy of Sciences of the United States of
[13] J.-C. Delvenne, S. N. Yaliraki, and M. Barahona, Pro-             America 102, 7426 (2005).
     ceedings of the National Academy of Sciences 107, 12755      [29] S. Lafon and A. Lee, Pattern Analysis and Machine Intel-
     (2010).                                                           ligence, IEEE Transactions on 28, 1393 (2006).
[14] M. Rosvall and C. T. Bergstrom, Proceedings of the           [30] http://tuvalu.santafe.edu/~aaronc/
     National Academy of Sciences 105, 1118 (2008).                    facultyhiring/.
[15] P. Pons and M. Latapy, in Computer and Information           [31] C. De Bacco, D. B. Larremore, and C. Moore, arXiv
     Sciences-ISCIS 2005 (Springer, 2005) pp. 284–293.                 preprint arXiv:1709.09002 (2017).
[16] M. De Domenico, Physical Review Letters 118, 168301          [32] C. Altafini, Automatic Control, IEEE Transactions on
     (2017).                                                           58, 935 (2013).
[17] A. Arenas, A. Fernández, and S. Gómez, New Journal of      [33] C. Altafini and G. Lini, Automatic Control, IEEE Trans-
     Physics 10, 053039 (2008).                                        actions on 60, 342 (2015).
[18] A. Kolchinsky, A. J. Gates, and L. M. Rocha, Physical        [34] E. W. D. Luca, S. Albayrak, J. Kunegis, A. Lommatzsch,
     Review E 92, 060801 (2015).                                       S. Schmidt, and J. Lerner, in Proceedings of the 2010
[19] M. T. Schaub, J.-C. Delvenne, M. Rosvall, and R. Lam-             SIAM International Conference on Data Mining (2010)
     biotte, Applied Network Science 2, 4 (2017).                      Chap. 48, pp. 559–570.
                                                                                                                              13

[35] K. E. Read, Southwestern Journal of Anthropology 10,         [65] A. Lancichinetti and S. Fortunato, Phys. Rev. E 84,
     pp. 1 (1954).                                                     066122 (2011).
[36] P. Hage and F. Harary, Structural Models in Anthropology     [66] R. Lambiotte, in Modeling and Optimization in Mobile,
     (Cambridge University Press, 1983).                               Ad Hoc and Wireless Networks (WiOpt), 2010 Proceedings
[37] V. A. Traag and J. Bruggeman, Phys. Rev. E 80, 036115             of the 8th International Symposium on (IEEE, 2010) pp.
     (2009).                                                           546–553.
[38] P. J. Mucha, T. Richardson, K. Macon, M. A. Porter,          [67] A. Delmotte, E. W. Tate, S. N. Yaliraki, and M. Bara-
     and J.-P. Onnela, science 328, 876 (2010).                        hona, Physical biology 8, 055010 (2011).
[39] U. Von Luxburg, Statistics and computing 17, 395 (2007).     [68] M. Meila, Journal of Multivariate Analysis 98, 873 (2007).
[40] J. Kunegis, S. Schmidt, A. Lommatzsch, J. Lerner,            [69] K. Cooper and M. Barahona, ArXiv , arXiv:1103.5582
     E. W. D. Luca, and S. Albayrak, in Proceedings of the             (2011).
     2010 SIAM International Conference on Data Mining            [70] K. Cooper and M. Barahona, “Role-based similarity in
     (2010) pp. 559–570.                                               directed networks,” (2010).
[41] D. Cartwright and F. Harary, Psychological review 63,        [71] M. Faccin, M. T. Schaub, and J.-C. Delvenne, Journal
     277 (1956).                                                       of Complex Networks , cnx055 (2017).
[42] A. Litwin-Kumar and B. Doiron, Nature Neuroscince 15,        [72] R. J. Sanchez-Garcia, arXiv preprint arXiv:1803.06915
     1498 (2012).                                                      (2018).
[43] M. T. Schaub, Y. Billeh, C. A. Anastassiou, C. Koch, and     [73] M. Schaub, N. O’Clery, Y. N. Billeh, J.-C. Delvenne,
     M. Barahona, PLoS Computational Biology 11, e1004196              R. Lambiotte, and M. Barahona, Chaos 26, 094821
     (2015).                                                           (2016).
[44] P. Strata and R. Harvey, Brain Research Bulletin 50, 349     [74] R. Lambiotte, J. Delvenne, and M. Barahona, Network
     (1999).                                                           Science and Engineering, IEEE Transactions on 1, 76
[45] M. E. Newman and A. Clauset, Nature communications                (2014).
     7 (2016).                                                    [75] P. Bremaud, Markov Chains: Gibbs fields, Monte Carlo
[46] T. P. Peixoto, Physical review letters 110, 148701 (2013).        simulation, and queues, corrected edition ed. (Springer,
[47] R. I. Kondor and J. D. Lafferty, in Proceedings of the            1999).
     Nineteenth International Conference on Machine Learn-        [76] R. Gallager, Stochastic Processes: Theory for Appli-
     ing, ICML ’02 (Morgan Kaufmann Publishers Inc., San               cations, Stochastic Processes: Theory for Applications
     Francisco, CA, USA, 2002) pp. 315–322.                            (Cambridge University Press, 2013).
[48] A. J. Smola and R. Kondor, in Learning theory and kernel     [77] J. Reichardt and S. Bornholdt, Phys. Rev. Lett. 93,
     machines (Springer, 2003) pp. 144–158.                            218701 (2004).
[49] V. Blondel, A. Gajardo, M. Heymans, P. Senellart, and        [78] V. A. Traag, P. Van Dooren, and Y. Nesterov, Phys. Rev.
     P. Van Dooren, SIAM Review 46, 647 (2004).                        E 84, 016114 (2011).
[50] E. Leicht, P. Holme, and M. Newman, Phys. Rev. E 73,         [79] J. Shi and J. Malik, Pattern Analysis and Machine Intel-
     026120 (2006).                                                    ligence, IEEE Transactions on 22, 888 (2000).
[51] L. Lü and T. Zhou, Physica A: Statistical Mechanics and     [80] L. Page, S. Brin, R. Motwani, and T. Winograd, (1999).
     its Applications 390, 1150 (2011).                           [81] V. Satuluri and S. Parthasarathy, in Proceedings of the
[52] M. Rosvall, A. V. Esquivel, A. Lancichinetti, J. D. West,         14th International Conference on Extending Database
     and R. Lambiotte, Nature communications 5 (2014).                 Technology (ACM, 2011) pp. 343–354.
[53] J.-C. Delvenne, R. Lambiotte, and L. E. Rocha, Nature        [82] J. M. Kleinberg, Journal of the ACM (JACM) 46, 604
     communications 6 (2015).                                          (1999).
[54] M. MacMahon and D. Garlaschelli, Phys. Rev. X 5,             [83] R. Lambiotte and M. Rosvall, Phys. Rev. E 85, 056107
     021006 (2015).                                                    (2012).
[55] G. E. Dullerud and F. Paganini, A course in robust control   [84] M. T. Schaub, Unraveling complex networks under the
     theory, Vol. 6 (Springer New York, 2000).                         prism of dynamical processes: relations between structure
[56] U. Baur, P. Benner, and L. Feng, Archives of Computa-             and dynamics, Ph.D. thesis, Imperial College London
     tional Methods in Engineering 21, 331 (2014).                     (2014).
[57] W. H. Schilders, H. A. Van der Vorst, and J. Rommes,
     Model order reduction: theory, research aspects and appli-
     cations, Vol. 13 (Springer, 2008).
[58] S. Fortunato, Physics Reports 486, 75 (2010).
[59] J. Reichardt and S. Bornholdt, Phys. Rev. E 74, 016110
     (2006).
[60] R. Campigotto, P. C. Céspedes, and J.-L. Guillaume,
     arXiv:1406.2518 (2014).
[61] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and
     E. Lefebvre, Journal of Statistical Mechanics: Theory
     and Experiment 2008, P10008 (2008).
[62] M. E. Newman and M. Girvan, Physical review E 69,
     026113 (2004).
[63] M. T. Schaub, R. Lambiotte, and M. Barahona, Phys.
     Rev. E 86, 026112 (2012).
[64] S. Fortunato and M. Barthélemy, Proceedings of the Na-
     tional Academy of Sciences 104, 36 (2007).
                                                                                                                              14

    Appendix A: Leaky-integrate-and-fire neural                      neurons connected as follows. The excitatory neurons
        networks with functional modules                             are statistically biased to target the inhibitory neurons
                                                                     in their own assembly with probability pin  IE = 0.90 and
                                                                                in
   Due to their computational simplicity yet complex dy-             weight WIE    = 0.0263, compared to pIE = 0.4545 and
namics, networks of LIF neurons are widely used as scal-             WIE = 0.0087 otherwise. Inhibitory neurons connect to
able prototypes of neural activity. The non-linear dy-               all excitatory neurons with probability pEI = 0.5263 and
namics of LIF models reproduce Poisson-like neuronal                 weight WEI = 0.045, apart from the excitatory neurons in
firing with refractory periods, among other features. Here,          their own assembly which are connected with probability
we employ that LIF networks display structured behav-                pin                   in
                                                                      EI = 0.2632 and WEI = 0.015. Note that, while from a
ior [42, 43], in which sustained in-group spiking switches           purely structural point of view we may split this network
from group to group across the network. Importantly,                 into 20 groups (10 groups of excitatory neurons, 10 groups
these cell assemblies of coherently spiking neurons include          of inhibitory neurons; see also Figure 6), it can be shown
both excitatory neurons (which exhibit a positive influ-             that this configuration gives rise to 10 functional groups
ence on their neighbours) and inhibitory neurons (whose              of neurons firing in synchrony with respect to the rest of
influence is negative), which have different connection              the network [43].
profiles. Moreover, we remark that these groups are not
densely connected clusters which are only weakly con-
nected to other clusters, but the behavior emerges from                 Appendix B: Relations between dimensionality
the connections between the various groups. Stated dif-                       reduction and module detection
ferently, the observed grouping is dynamical (functional)
rather than structural.                                                In this section we elaborate on the relationship between
   We simulated leaky-integrate-and-fire (LIF) networks              dimensionality reduction and the detection of dynamical
with n = 1000 neurons (800 excitatory, 200 inhibitory).              modules as discussed in the main text.
Using a time step of 0.1ms, we numerically integrated the              Let us initially consider the problem from the point
non-dimensionalized membrane potential of each neuron,               of view of the squared distance matrix D(2) , where we
                                                                     omit writing the time-dependence to emphasize that the
  dVi (t)   1                   X          E/I                                                                            (2)
          = E/I (ui − Vi (t)) +   [WN ]ij gj (t), (A1)               derivations below apply to both the integrated D[0,t] as
    dt     τm                   j                                    well as the instantaneous distance matrix D(2) (t).
with a firing threshold of 1 and a reset potential of 0.               A naive idea to derive a clustering measure would be to
The input terms ui were chosen uniformly at random in                simply try and place all nodes into the same group such
the interval [1.1, 1.2] for excitatory neurons, and in the           that the sum of the distances in each group is minimized,
interval [1, 1.05] for inhibitory neurons. The membrane              which would lead to the following optimization procedure:
time constants for excitatory and inhibitory neurons were
        E                  I                                                           min trace H > D(2) H,
set to τm  = 15 ms and τm    = 10 ms, respectively, and the                              H
refractory period was fixed at 5 ms for both excitatory and
                                                                                      n×k
inhibitory neurons. Note that although the constant input            where H ∈ {0, 1}     is a partition indicator matrix with
term is supra-threshold, balanced inputs guarantee an                Hij = 1 if node i is in group j and Hij = 0 otherwise.
average sub-threshold membrane potential [42]. The net-              We can rewrite the above using the definition of D(2) as
work dynamics is captured by the sum in (A1), which de-
                                                                               min trace H > 1z> + z1> − 2Ψ H,
                                                                                                               
scribes the input to neuron i from all other neurons in the
                                                                                 H
network and [WN ]ij denotes the weight of the connection
from neuron j to neuron i. Synaptic inputs are modelled              where z = diag(Ψ) is the vector containing the diagonal
     E/I
by gj (t), which is increased step-wise instantaneously              entries of Ψ. It is easy to see that if k is not constrained
                                                  E/I   E/I          in the above optimization problem, then the best choice
after a presynaptic spike of neuron j (gj → gj                + 1)
and then decays exponentially according to:                          will be to trivially put each node in its own group (k = n).
                                                                     Stated differently, if we are free to choose any number
                           E/I
                         dgj         E/I
                                                                     of groups k, then we can make the distance within each
                 τsE/I           = −gj     (t),               (A2)   group zero, thus minimizing the above objective.
                           dt
                                                                        One potential remedy to fix the above shortcoming
with time constants τsE = 3 ms for an excitatory in-                 would be to fix the number of groups, a priori, and then
teraction, and τsI = 2 ms if the presynaptic neuron is               perform some kind of selection procedure afterwards to
inhibitory. Excitatory and inhibitory neurons were con-              pick the number of groups. Another option is to intro-
nected uniformly with probabilities pEE = 0.2, pII = 0.5,            duce some ‘slack’ in the distance measurements, thus
and weight parameters WEE = 0.022 and WII = 0.042,                   permitting nodes whose distance is comparably small to
respectively.                                                        contribute negative to the cost function (which is here to
  The network comprised 10 functional groups of neurons,             be minimized). As we will show in the following this natu-
each of which consists of 80 excitatory and 20 inhibitory            rally leads to a problem formulation akin to many network
                                                                                                                                  15

partitioning procedures which have been proposed in the                    community detection scheme [20], where the last term can
literature.                                                                be identified with an Erdős-Rényi null model. Following
   LetP us      consider    the    spectral     expansion                  exactly the same procedure, similar expressions may also
        n
Ψ = i=1 λi vi vi> , where we assume the eigenval-                          be derived for various other null models, such as the
ues to be ordered, such that λ1 > · · · > λn ≥ 0. Then we                  configuration model.
              2
can rewrite Dij  as:                                                          Further, as discussed in the next section, the above
                                                                           expression can be rewritten in the form rν (t, H) (see
         n
         X                  2             2                                Equation (23)), and can thus be interpreted as a quality
  2
 Dij =         λk (vk> ei ) + λk (vk> ei ) − 2λk (vk> ej )(vk> ei ),       function that we can optimize using the Louvain opti-
         k=1                                                               mization scheme. This emphasizes how the Louvain op-
where ei is the i-th unit vector. Let us now introduce                     timization scheme can be interpreted as operating with
some slack variables (multipliers) γk for all the modes in                 an approximation of the distance matrix / similarity ma-
the first two terms and rewrite the above expression in                    trix Ψ. This result may be used in several ways to derive
terms of the dynamical coordinates φk :                                    some more general null models by choosing an appropriate
                                                                           weighting scheme.
        n
        X                       2             2
 2
Dij =         γk λk (vk> ei ) + γk λk (vk> ei ) − 2λk (vk> ej )(vk> ei )
        k=1                                                                  1.   Null models, projections and time-parameter
        Xn                                                                                        choices
                        2           2
    =         γk [(φi,k ) + (φj,k ) ] − 2φ>
                                          i φj ,
        k=1                                                                   As observed above, the Louvain algorithm might be
                                                                           seen as solving a closely related (‘dual’) problem to the
which shows that γk , may be seen as weighting functions
                                                                           distance minimization. However, instead of trying to find
for the first k coordinates in the φ coordinate space for
                                                                           groups with minimal distance according to some crite-
the first 2 (norm) terms in the distance.
                                                                           rion, the optimization operates in terms of the associated
  Using the above derivation, let us rewrite the previously
                                                                           similarity measure (inner product). An interesting ques-
considered minimization as an equivalent maximization
                                                                           tion that we will not pursue in the following would thus
problem.
                                                                           be to investigate equivalent distance based optimization
                          
                                1
                                                                          problems. Instead, in the following we will discuss how
                        >             >      >
         max 2 trace H Ψ − (z̃1 − 1z̃ ) H                                  the here derived formulation ties in with many quality
           H                    2
                                                                           function commonly considered in networks analysis.
         with z̃ = diag Φ> ΓΦ ∈ Rn ,                                          Within network science many quality functions for
                Γ = diag(γ1 , . . . , γn ) ∈ Rn×n                          community detection can effectively be written in the
                                                                           form [20, 58–60]:
   While different weighting schemes {γk } are of potential
interest here, let us now consider the specific choice γ1 = 1,                          r(H) = trace H > [G − αN ] H,          (B1)
γk = 0, (k > 1), which corresponds to making a simple
                                                                           where G is a term customarily related to the network
low rank-correction of Ψ. This specific scheme is akin to
                                                                           structure, and N is a ‘null-model’ term, which custom-
choosing a type of null model in our optimization scheme
                                                                           arily includes a scalar multiplier α as a free resolution
as we will illustrate next.
                                                                           parameter. It is insightful to rewrite the above as:
   For concreteness, let us consider the familiar case of
Laplacian dynamics for a symmetric graph, i.e., Ψ =                                     r(H) = hH, GHi − αhH, N Hi.            (B2)
          >
exp(−Lt) exp(−Lt). In this case the eigenvectors of Ψ
are constant over time, and the first eigenvalue of Ψ is                   In particular, if the null model term is a positive semi-
one with √ an associated constant eigenvector and thus                     definite matrix, we can further simplify this to:
φi,1 = 1/ n, ∀i. This leads to an optimization of the
form:                                                                                  r(H) = hH, GHi − αkN 1/2 Hk2F ,         (B3)
                      
                                      1
                                                                          where k · kF is the Frobenius norm. This highlights how
                    >                       >       >
   max 2 trace H exp(−2Lt) −            (11 − 11 ) H,                      the null model term acts effectively as a regularization
     H                               2n
                                                                           term (similar to regression type-problems) with a weight-
which can be simplified to the equivalent optimization:                    ing factor (Lagrange multiplier) given by the resolution
                                                                         parameter α.
                                       1                                     Let us now consider how the above formulations apply
        max trace H > exp(−2Lt) − 11> H                                    to the similarity measures derived above. To link with our
          H                           n
                                                                           above discussion let us concentrate on the case where we
Note that this is just the (rescaled) Markov stability                     (partially) project out a rank-1 term via W = I − ανν > .
at time 2t (see also Section C). Linearising the above                     Note that the same effect can also be achieved by choos-
expression thus leads to recovering a Potts-model like                     ing C appropriately, providing additional interpretations
                                                                                                                          16

which we will not explore in the following. This would            3. Repeat steps 1–2, until no further improvement is
lead to a similarity matrix Ψ⊥ of the form:                          possible.
                 Ψ⊥ = Y > Y − αY > νν > Y,             (B4)    As outlined in Ref. [60], this generic procedure can be
                                                               used to optimize any quality function of the form:
where Y = C exp(At)B. This kind of similarity lead to a
                                                                                 trace H > [F − ab> ]H,                (B7)
quality function of the form:
                                                               where H is the partition indicator matrix, F is a general
       rν (t, α, H) = kY (t)Hk2 − αkν > Y (t)Hk2       (B5)    matrix derived from the network, and a, b are two n
                   = kHk2Ψ(t) − αkν > Y (t)Hk2         (B6)    dimensional vectors. The quality function (23) is clearly
                                                               of this form. An inherent problem of many community
where we explicitly have written out the dependency            detection measures is the choice of the relevant resolution,
on t and α and have defined the semi-norms kXkY :=             or scale, of the partitioning. In many methods this choice
trace X > Y X, which is possible in our formulation as all     has to be made explicitly a priori, by declaring how many
the relevant matrices are positive semi-definite.              groups are to be found by the method. If this is not
   From the above we can make the following observations.      the case, then there there is either a free (’resolution’)
First, as already alluded to before, the influence of the      parameter or a regularization scheme, with which the
resolution parameter α is akin to a Lagrange multiplier,       size of the groups found can be controlled explicitly or
which linearly scales the regularization term. In contrast     implicitly, or there is an implicit scale associated with the
the influence of the time-parameter is more subtle as it       method, which will determine an upper and lower limit
changes the eigenvalues / eigenvectors of Ψ and thus acts      of size the communities to be found [8, 63–65].
in a nonlinear fashion.                                           Instead of choosing and fixing a particular scale, we
   Second, by choosing a particular ν in the projection        here identify significant partitions according to the criteria
term such that a time-independent component is picked          outlined in Refs. [8, 66, 67]. We advocate to look at the
out from Y , we can recover classical some classical null      trajectories for all times t and let thereby the dynamical
model terms. As an example we can again consider the           process reveal the important scales of the problem. These
symmetric Laplacian dynamics Y = exp(−Lt), for which           scales should be associated with robust partitions over
the centering operation W⊥ = I − 11> /n corresponds to         time and relative to the optimization. Thus we are inter-
projecting out the stationary eigenvector (see also Sec-       ested in identifying robust partitions as indicated by: (i)
tion C, where the case Y = exp(−D−1 Lt) and the config-        a persistence to (small) time-variations, which translates
uration null model is discussed as well). Note, however,       into long plateaux in the number of communities plotted
that in general the second term remains time-dependent         against time; (ii) consistency of partitions obtained from
and the null model may vary with time, too. The choice of      the generalized Louvain algorithm over random initialisa-
a projection may thus be guided either by simple consider-     tion conditions as measured by the mean distance between
ations on what aspect of the dynamics we are interested in     partitions using the normalized Variation of Information
(e.g., the relative difference of the influence which would    (VI) metric [68]. A VI of zero results when all iterations of
lead to a type of centering operation), or by suppressing      the Louvain algorithm return exactly the same clustering.
some type of mode which we know might be irrelevant            By computing the matrix V I(t, t0 ), containing the mean
for our considerations (e.g., the stationary distribution in   variation of information between any two sets of partitions
a diffusion process [8, 20]).                                  at different times, we can easily identify time-epochs over
                                                               which we always obtain very similar, robust partitions.
2.    Optimizing the quality measure rν (t, H) using an
              adapted Louvain algorithm                            3.   Dynamical roles, modules and symmetries

   The optimization of the quality measure rν (t, H) given        As many notions of dynamical or functional role have
in (23) can be achieved by various means, e.g., via MCMC       been presented in the literature so far, we provide here a
or spectral techniques. Here we propose to use an aug-         short conceptual clarification on what we would consider
mented version of the Louvain algorithm [61], which was        a ‘dynamical’ module in this work, in the sense that our
initially proposed as an efficient algorithm to optimize the   similarity measures would assign a high similarity score
Newman-Girvan modularity [62]. The algorithm operates          between each node in a module. To this end consider the
as follows:                                                    example network depicted in Figure 7. For simplicity of
     1. Loop over all nodes in a random order, and assign      our exposition we will consider here a simple consensus
        each node greedily to the community for which the      dynamics of the form ẋ = −Lx, where L is the standard
        increase in quality is maximal until no further move   graph Laplacian. In this case it is essentially the structure
        is possible.                                           of the network that dictates how our similarity measures
                                                               evolves through the spectral properties of the Laplacian.
     2. Build a coarse-grained network, in which each node        We consider two possible partitions in Figure 7A. Both
        represents a community in the previous network.        partition may be seen to correspond to a type of ‘role’ of
                                                                                                                                                           17

A           Configuration I                                 Configuration II                     future work. However, see for instance Refs [69–73] for
                   4                                               4
              6        7                                      6        7                         related discussions.
               4       4                                       4        4
                   3                                               3                             Appendix C: The (centered) dynamic similarity Ψ(t)
                                                                                                              for diffusive processes
    4   4                      4   8               4    4                       4        8
4             1            2           4       4              1             2                4
                                                                                                    In this section we comment on how specific measures
    5   4                      4   9               5    4                       4        9
                                                                                                 can be recovered and extended within the here presented
B                                  C       Partition II: up to symmetry same influence           framework, when focussing on diffusion processes. Note
                                                                                                 however, as discussed in the main text, the dynamical
                                                                                                 similarity Ψ is applicable to general linear dynamics (in-
                                                                                                 cluding signed networks). Below we consider first the case
                                                                                                 of a diffusion process on undirected network, before we
                                                                                                 comment on the diffusion processes on directed networks.


                                                                                                                 1.   The undirected case

                                                                                                   To put our approach in the context of diffusion pro-
                                                                                                 cesses, let us first consider an undirected dynamics of the
                                                                                                 form

                                                                                                                         ṗ = −p L,                     (C1)

                                                                                                 where p is the 1 × n row vector describing the probability
Figure 7. Dynamical similarities, modules and roles. A                                           of a particle to be present at any node. Note that this
A small network with 2 possible partitions with nodes may                                        diffusion is the dual of the consensus process:
be described as similar. B If we were to group the nodes
according to the dynamical similarity measure as defined in                                                               ẋ = −Lx,                     (C2)
the text we would pick out configuration A, as the nodes in
each group have a similar impulse response after time t. C                                       and indeed, in this case these two dynamics are in fact
However, if we consider the impulse responses up to the action                                   identical, as transposing Equation (C1) corresponds to a
of a permutation Ω (isometric mapping), we could also infer                                      dynamics of the form (3) with B = C = I and A = −L =
the role’ partition of configuration B.                                                          −L> , the combinatorial graph Laplacian.
                                                                                                    As it is customary in the context of diffusion processes
                                                                                                 to deal with row vectors, and accordingly many results
the nodes in the network. In this case, as can been seen in                                      in the literature are presented in this form we will adopt
Figure 7B, our notion of similarity is commensurate with                                         this convention throughout this section. All these results
partition I, as the impulse responses of the nodes in the                                        can be readily transformed into a column vector setup
colored group influence the same parts of the network in                                         (or in the directed case, may also be interpreted in the
essentially the same way. Stated differently, if we denote                                       light of the dual consensus process).
by yi (t) the impulse response of node i after time t, then                                         Consider the random walk associated with the dynam-
after a brief transient period an impulse given to any node                                      ics (C1) described by the row indicator vector N(t) ∈
                                                                                                        n
in the same group results in approximately the same state                                        {0, 1} , where Ni (t) = 1 if the walker is present at node
vector of the network (e.g. y1 (t) ≈ y4 (t) ≈ y5 (t), in case                                    i at time t and zero otherwise, and the 1 × n-dimensional
of the red group).                                                                               vector p(t) describes the probability of the walker to be
   However, the nodes in partition II are indeed similar                                         at each node at time t. It is well known that if the process
in the following sense. If we consider the vectors yi (t) up                                     takes places on an undirected (i.e., L = L> ) connected
to the action of a symmetry group (permutation), then                                            graph, it is guaranteed to be wide sense stationary (in
we can see that indeed partition II groups nodes together                                        fact ergodic) and p(t) converges to the unique stationary
which are similar in this sense. More precisely, call Ω a                                        distribution
permutation matrix corresponding to the orbit partition                                                                        1>
II indicated in Figure 7A. Then for any two nodes i, j in                                                                 π=      ,
the same group there exist a permutation matrix Ω such                                                                          n
that yi = Ωyj (see Figure 7C) for an illustration.                                               irrespective of the initial condition.
   One could thus try to search for these kinds of par-                                             To derive further results, it is insightful to compute the
titions as well from the perspective of our dynamical                                            auto-covariance matrix of this process. Let us assume
framework. We postpone a exploration of these tasks for                                          that we prepare the system at stationarity, p(0) ∼ π (i.e.,
                                                                                                                           18

the walker is equally likely to start at any node at time      Hence for the case of diffusion on undirected graphs, the
t = 0). The expectation of N(t) remains constant over          centered dynamic similarity Ψ⊥ (t) is proportional to the
time and we have                                               autocovariance of the diffusion on a rescaled time.
                                                                  Although we have exemplified this connection with
          E[N(t)] = E[N0 ] P (t) = πP (t) = π,
                                                               a particular example, this result applies to any time-
where the transition matrix for this process is given by:      reversible dynamics. This includes all customary de-
                                                               fined diffusion dynamics on undirected graphs like the
                     P (t) = exp(−Lt).
                                                               continuous-time unbiased random walk, the combinatorial
The auto-covariance matrix of the process is then:             Laplacian random walk, or the maximum entropy random
                    h
                          >
                                  i                            walk [74].
          Σ(t) = cov N(0) , N(t)                   (C3)           This result follows from the reversibility condition for a
                                                               Markov process [75, 76], also known as detailed balance:
                  = E[N0 > N(t)] − E[N>
                                      0 ] E[N0 ]      (C4)
                              >
                  = ΠP (t) − π π,                     (C5)                     π (i) pi→j = π (j) pj→i   ∀ i, j,       (C11)

where Π = diag(π). Defining Σ0 = Π − π > π, we get that        i.e., at stationarity, the probability to transition from state
                                                               i to state j is the same as the probability to transitions
             Σ(t) = Σ0 P (t) = Σ0 exp(−Lt),           (C6)     from j to i (for any i, j). In matrix terms, the detailed
and it becomes apparent that Σ(t) is governed by the           balance condition is:
matrix differential equation:                                                                    >
                                                                                 ΠP (t) = P (t) Π,       ∀t.           (C12)
              dΣ
                  = −ΣL with Σ(0) = Σ0 .                (C7)      Let us consider a centered dynamical similarity measure,
               dt
                                                               in which we choose the weighting matrix
This autocovariance matrix Σ(t) has been used as a dy-
namic similarity matrix in the Markov Stability framework                            WΠ = Π − π > π,
for community detection [8, 13, 20].
   Now, let us compare the autocovariance Σ(t) with the        which can be throught of as a generalization of the stan-
dynamic similarity Ψ(t). As shown in (7), when B = I           dard projection matrix W⊥ . It is now easy to see that
the dynamic similarity Ψ(t) obeys a Lyapunov matrix            the analogous relationship between ΨΠ (t) and the auto-
differential equation, which in this diffusive case is:        covariance Σ(t) holds also under detailed balance:
      dΨ                                                                               >
                                                                         ΨΠ (t) = P (t) Π − π > π P (t)
                                                                                                   
          = −L> Ψ − ΨL with Ψ(0) = Ψ0 .            (C8)
       dt
                                                                                = ΠP (t) − π > πP (t) P (t)
                                                                                                     
Without loss of generality we may pick Ψ0 = Σ0 as initial
                                                                                = Π − π > π P (2t) = Σ(2t).
                                                                                            
condition, so that the solution is given by
              >
  Ψ(t) = P (t) Σ0 P (t) = exp(−L> t)Σ0 exp(−Lt), (C9)             In the case of diffusive dynamics on undirected net-
                                                               works with detailed balance, we have shown that the
which is to be compared to (C6)                                dynamical similarity ΨΠ (t) is equivalent to the autoco-
  For the case of undirected graphs, we have L = L> and        variance Σ(t) up to a rescaling. Therefore the Louvain-
L1 = 0, hence                                                  like analysis of ΨΠ (t) on the quality function rΠ (t, H) =
                               1
                                 
                                      11>
                                                              trace H > ΨΠ (t)H can be seen as a proper generalization
                        >
            Σ0 = Π − π π =         I−        .                 of the Markov Stability framework, which optimizes the
                               n        n
                                                               quality function s(t, H) = trace H > Σ(t)H, and thus en-
Therefore, we can interpret W⊥ = nΣ0 as a (scaled)             compasses a wide array of notions of community detection
projection matrix in Eq. (C9), and (C9) as a centered          including the classical Newman-Girvan Modularity [62],
dynamic similarity:                                            the self-loop adjusted modularity version of Arenas et
   1                   1
                         
                              11>
                                                              al. [17], the Potts model heuristics of Reichardt and Born-
     Ψ⊥ (t) = exp(−Lt)     I−       exp(−Lt). (C10)            holdt [77], as well as Traag et al. [78], and classical spec-
   n                   n       n
                                                               tral clustering [79]. For details, we refer the reader to the
  We can then rewrite Ψ⊥ (t) to show the equivalence           derivations given in Refs. [20, 74] in terms of the Markov
with Σ(t) up to a simple rescaling:                            Stability measure.
                                                                  Note, however, that Markov Stability only deals with
                                11>
                                   
   1          1                                                diffusion dynamics, whereas both the centered dynamical
     Ψ⊥ (t) = exp(−Lt) I −            exp(−Lt)
   n          n                  n                             similarity ΨΠ (related to the quality function rΠ (t, H)),
              11>                                              and the kernel Ψ can be applied to general linear models
                  
      1
   =      I−         exp(−Lt) exp(−Lt)                         (including signed networks), as discussed in the main
      n         n
                                                               text. An interesting case occurs when considering diffusive
              11>
                  
      1                                                        processes on directed graphs, as discussed in the next
   =      I−         exp(−L(2t)) = Σ0 P (2t) = Σ(2t),
      n         n                                              section.
                                                                                                                                                      19

A                                                    B                                       node similarities cannot be based on autocovariances at
                                                                                             stationarity.
                                                                                                In order to study node similarities based on the diffu-
                                                                                             sive dynamics, the original process is usually modified by
       Adjacency matrix                                         Adjacency matrix             adding a small ‘teleportation’ term (e.g., allowing for the
                   Node                                                     Node             process to diffuse to any node on the graph with a small
            1 2 3 4 5 6 7 8 9 10                                     1 2 3 4 5 6 7 8 9 10
        1                                                        1                           probability)[13, 14]. This approach is also known as the
        2                                                        2
        3                                                        3                          ‘Google trick,’ as it was popularized through its use in the
        4                                                        4
                                                                                             original computation of Pagerank [80]. In its original form,
    Node                                                     Node
        5                                                        5
        6                                                        6
        7                                                        7                           the introduction of teleportation creates a related strongly
        8                                                        8
        9                                                        9                           connected graph by combining the original graph (with
       10                                                       10
                  Weight                                                    Weight           Laplacian L) together with the complete graph. This
                                                                                             creates a related (yet different) ergodic process on this
                                                                                             surrogate graph which can then be analyzed [74] via dy-
                                                                                             namic similarities based on autocovariances, as discussed
                                                                                             in Section C 1. Specifically, the surrogate, ergodic system
                                         Map equation                                        is defined by an adjusted Laplacian operator
                                      (with teleportation)
                                          Map equation
                                     (without teleportation)
                                                                                                                 L
                                                                                                                 e = L + Lteleport .

                                                                                              Following an analogous calculation as above, the auto-
Figure 8. Unpredictable influence of teleportation on
clustering of directed graphs (A 6= A> ). When study-                                       covariance Σ(t) of this process:
ing a directed diffusive dynamics (on a directed graph), a                                                                            
teleportation component is usually added to make the pro-                                     Σ(t)
                                                                                              e    =Π         e −π
                                                                                                     e exp(−Lt)      e >π
                                                                                                                        e= Π      e >π
                                                                                                                               e −π  e exp(−Lt),
                                                                                                                                              e
cess ergodic. The addition of teleportation can lead to unex-
pected effects when clustering the network, since teleportation                             where πe is the stationary distribution of the surrogate er-
also influences the cut (i.e., the probability flow across group                            godic process (e.g., Page Rank). However, this autocovari-
boundaries). For concreteness, we illustrate this effect here                               ance is now asymmetric, in general, and its interpretation
the map-equation [14] although this effect is general. A A                                  as a similarity matrix is problematic.
directed network of two cliques. The weights between nodes
                                                                                              In contrast, we can use the Lyapunov equation (C8) to
inside each group are drawn from a uniform distribution with
mean 1 ± 0.1; the weights across different groups are drawn
                                                                                            define the dynamic similarity (C9) of the ergodic system
from a uniform distribution with mean 0.2±0.02. The network                                                                      
is directional but the asymmetry of A is so weak as to appear                                   Ψ
                                                                                                e e (t) = exp(−L e > t) Πe −π >
                                                                                                                                π   exp(−Lt),
                                                                                                                                          e      (C13)
                                                                                                  Π
                                                                                                                            e   e
visually virtually undirected. The introduction of teleportation
leads to a resolution limit effect in which the groups cannot                               where we have chosen the initial condition Ψ0 = e Π−π  e >πe.
be resolved. B A directed cycle network with equal weights.                                 Note that Eq. (C13) may alternatively be constructed
Without teleportation, a split of the ring into multiple groups
is found, whereas the introduction of teleportation improves                                from the dual process ẋ = Ax, with the operator A = − e   L.
the result in this case so that the whole cycle is detected. Us-                            The similarity Ψ ee
                                                                                                                Π (t) may now be exploited directly to
ing an analysis based on Ψ without teleportation (C14) finds                                carry out embeddings, spectral clusterings, or Louvain-
the dynamical blocks directly in both of these examples.                                    like block detection, as described in the Methods section.
                                                                                            Note that the analysis based on the dynamic similarity
                                                                                            Ψe e (t) remains distinct to the symmetrized autocovariance
                                                                                               Π
                           2.      The directed case                                        (Σ
                                                                                             e+Σ    e > )/2 which is optimized when using a Louvain-
                                                                                            like algorithm [74]. Other symmetrizations have been
   For undirected diffusions, the autocovariance can be                                     introduced in the context of transition matrices in directed
used as a dynamic similarity between nodes. However,                                        graphs [81] generalizing Kleinberg’s HITS scores [82].
there are important differences for directed (asymmetric)                                      The definition of the dynamic similarity of the asso-
diffusive processes, as such processes are not guaranteed                                   ciated ergodic process (C13) renders it consistent with
to be ergodic.                                                                              our generic framework. However, the introduction of tele-
   Let us consider a diffusion on a directed graph:                                         portation to create the surrogate process has conceptual
                                                                                            disadvantages.
                                     ṗ = −pL                                                  First, teleportation perturbs the dynamics in a non-
                                                                                            local manner and induces uncontrolled effects when find-
so that B = C = I and −A = L 6= L> , an asymmetric                                          ing node similarities based on dynamics. In particular,
Laplacian. This process is only ergodic if the graph is                                     it can reduce overclustering (a positive effect), but can
strongly connected. In many scenarios this is not the case,                                 also lead to (unwanted) resolution limits when finding
however, and the process asymptotically concentrates the                                    dynamic blocks (Fig. 8). These issues are only at best
probability on sink nodes with no outgoing links. Hence                                     mitigated by recent teleportation schemes [83].
                                                                                                                        20

  Second, teleportation creates an ergodic, stationary         applied to non-ergodic, directed graphs without the need
dynamics when key features of the original system might        to add teleportation (i.e., without creating the associated,
be fundamentally linked to non-stationary data and non-        but distinct stationary process). In this case, the dynamic
ergodic processes. This can have a bearing on the con-         similarity is
clusions drawn from the surrogate ergodic process with
added teleportation.                                                        Ψ(t) = exp(−L> t)Σ0 exp(−Lt),           (C14)

   We illustrate some of these problems in Figure 8 for the    which fulfils the Lyapunov equation (C8). The initial
particular example of the map-equation. However, these         condition Σ0 can be chosen to be any (covariance) matrix
issues are generic and effect other diffusion based cluster-   which serves as the null model for the process. The
ing measures that are based on a notion of persistence of      analysis of this dynamic similarity can reveal dynamic
the flow within a region over time, and require a type of      blocks based on directed flows from the original diffusive
ergodicity assumption. For instance, similar effects will      process, as shown in (Fig. 8). This directed case is another
affect the Markov stability measure [13, 84].                  instance where the notion of ‘dynamic block’ generalizes
                                                               the idea of modules, originally conceived from a structural
  Importantly, our dynamic similarity Ψ(t) can be directly     perspective.

