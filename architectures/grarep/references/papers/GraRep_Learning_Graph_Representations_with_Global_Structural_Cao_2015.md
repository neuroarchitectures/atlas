# GraRep Learning Graph Representations with Global Structural Cao 2015

> Source: `GraRep_Learning_Graph_Representations_with_Global_Structural_Cao_2015.pdf`

---

See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/301417811



GraRep

Conference Paper · October 2015
DOI: 10.1145/2806416.2806512




CITATIONS                                                                                                 READS
80                                                                                                        7,010


3 authors:

            Shaosheng Cao                                                                                            Wei Lu
            Singapore University of Technology and Design                                                            Singapore University of Technology and Design
            2 PUBLICATIONS 180 CITATIONS                                                                             47 PUBLICATIONS 438 CITATIONS

               SEE PROFILE                                                                                                SEE PROFILE



            Qiongkai Xu
            Australian National University
            6 PUBLICATIONS 116 CITATIONS

               SEE PROFILE




Some of the authors of this publication are also working on these related projects:


              Named Entity Recognition With Syntactic Dependency Tree Information View project



              Weak Semi-Markov CRFs for Noun Phrase Chunking in Informal Text View project




 All content following this page was uploaded by Qiongkai Xu on 07 December 2016.

 The user has requested enhancement of the downloaded file.
                    GraRep: Learning Graph Representations with
                            Global Structural Information

                       Shaosheng Cao                                                    Wei Lu                           Qiongkai Xu
                         Xidian University                                 Singapore University of                 IBM Research - China
               shelsoncao@gmail.com                                        Technology and Design                Australian National University
                                                                            luwei@sutd.edu.sg                      xuqkai@cn.ibm.com


ABSTRACT                                                                                      information from the graphs. One strategy is to learn the
In this paper, we present GraRep, a novel model for learn-                                    graph representations of a graph: each vertex of the graph
ing vertex representations of weighted graphs. This model                                     is represented with a low-dimensional vector in which mean-
learns low dimensional vectors to represent vertices appear-                                  ingful semantic, relational and structural information con-
ing in a graph and, unlike existing work, integrates global                                   veyed by the graph can be accurately captured.
structural information of the graph into the learning process.                                   Recently, there has been a surge of interest in learning
We also formally analyze the connections between our work                                     graph representations from data. For example, DeepWalk
and several previous research efforts, including the Deep-                                    [20], one recent model, transforms a graph structure into a
Walk model of Perozzi et al. [20] as well as the skip-gram                                    sample collection of linear sequences consisting of vertices
model with negative sampling of Mikolov et al. [18]                                           using uniform sampling (which is also called truncated ran-
   We conduct experiments on a language network, a social                                     dom walk). The skip-gram model [18], originally designed
network as well as a citation network and show that our                                       for learning word representations from linear sequences, can
learned global representations can be effectively used as fea-                                also be used to learn the representations of vertices from such
tures in tasks such as clustering, classification and visualiza-                              samples. Although this method is empirically effective, it is
tion. Empirical results demonstrate that our representation                                   not well understood what is the exact loss function defined
significantly outperforms other state-of-the-art methods in                                   over the graph involved in their learning process.
such tasks.                                                                                      In this work, we first present an explicit loss function of
                                                                                              the skip-gram model defined over the graph. We show that
                                                                                              essentially we can use the skip-gram model to capture the
                                                                                              k-step (k = 1, 2, 3, . . . ) relationship between each vertex and
Categories and Subject Descriptors                                                            its k-step neighbors in the graph with different values of k.
I.2.6 [Artificial Intelligence]: Learning; H.2.8 [Database                                    One limitation of the skip-gram model is it projects all such
Management]: Database Applications - Data Mining                                              k-step relational information into a common subspace. We
                                                                                              argue that such a simple treatment can lead to potential
General Terms                                                                                 issues. The above limitation is overcome in our proposed
                                                                                              model through the preservation of different k-step relational
Algorithms, Experimentation
                                                                                              information in distinct subspaces.
                                                                                                 Another recently proposed work is LINE [25], which has
Keywords                                                                                      a loss function to capture both 1-step and 2-step local re-
Graph Representation, Matrix Factorization, Feature Learn-                                    lational information. To capture certain complex relations
ing, Dimension Reduction                                                                      in such local information, they also learn non-linear trans-
                                                                                              formations from such data. While their model can not be
1.     INTRODUCTION                                                                           easily extended to capture k-step (with k > 2) relational
                                                                                              information for learning their graph representation, one im-
   In many real-world problems, information is often orga-
                                                                                              portant strategy used to enhance the effectiveness of their
nized using graphs. For example, in social network research,
                                                                                              model is to consider higher-order neighbors for vertices with
classification of users into meaningful social groups based on
                                                                                              small degrees. This strategy implicitly captures certain k-
social graphs can lead to many useful practical applications
                                                                                              step information into their model to some extent. We believe
such as user search, targeted advertising and recommenda-
                                                                                              k-step relational information between different vertices, with
tions. Therefore, it is essential to accurately learn useful
                                                                                              different values of k, reveals the useful global structural in-
Permission to make digital or hard copies of all or part of this work for personal or
                                                                                              formation associated with the graph, and it is essential to
classroom use is granted without fee provided that copies are not made or distributed         explicitly take full advantage of this when learning a good
for profit or commercial advantage and that copies bear this notice and the full cita-        graph representation.
tion on the first page. Copyrights for components of this work owned by others than              In this paper, we propose GraRep, a novel model for learn-
ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or re-          ing graph representations for knowledge management. The
publish, to post on servers or to redistribute to lists, requires prior specific permission   model captures the different k-step relational information
and/or a fee. Request permissions from Permissions@acm.org.
CIKM’15, October 19–23, 2015, Melbourne, Australia.
                                                                                              with different values of k amongst vertices from the graph
 c 2015 ACM. ISBN 978-1-4503-3794-6/15/10 ...$15.00.                                          directly by manipulating different global transition matrices
DOI: http://dx.doi.org/10.1145/2806416.2806512.
defined over the graph, without involving slow and complex        can utilize global statistics [5]. Previous work include La-
sampling processes. Unlike existing work, our model defines       tent Semantic Analysis (LSA) [15], which decomposes term-
different loss functions for capturing the different k-step lo-   document matrix and yields latent semantic representations.
cal relational information (i.e., a different k). We optimize     Lund et al. [17] put forward Hyperspace Analogue to Lan-
each model with matrix factorization techniques, and con-         guage (HAL), factorizing a word-word co-occurrence counts
struct the global representations for each vertex by combin-      matrix to generate word representations. Levy et al. [4]
ing different representations learned from different models.      presented matrix factorization over shifted positive Point-
Such learned global representations can be used as features       wise Mutual Information (PMI) matrix for learning word
for further processing.                                           representations and showed that the Skip-Gram model with
   We give a formal treatment of this model, showing the          Negative Sampling (SGNS) can be regarded as a model that
connections between our model and several previous mod-           implicitly such a matrix [16].
els. We also demonstrate the empirical effectiveness of the
learned representations in solving several real-world prob-       2.2    Graph Representation Approaches
lems. Specifically, we conducted experiments on a language           There exist several classical approaches to learning low di-
network clustering task, a social network multi-label clas-       mensional graph representations, such as multidimensional
sification task, as well as a citation network visualization      scaling (MDS) [8], IsoMap [28], LLE [21], and Laplacian
task. In all such tasks, GraRep outperforms other graph           Eigenmaps [3]. Recently, Tang et al. [27] presented meth-
representation methods, and is trivially parallelizable.          ods for learning latent representational vectors of the graphs
   Our contributions are as follows:                              which can then be applied to social network classification.
                                                                  Ahmed et al. [1] proposed a graph factorization method,
     • We introduce a novel model to learn latent represen-       which used stochastic gradient descent to optimize matrices
       tations of vertices on graphs, which can capture global    from large graphs. Perozzi et al. [20] presented an approach,
       structural information associated with the graph.          which transformed graph structure into several linear vertex
     • We provide from a probabilistic prospective an un-         sequences by using a truncated random walk algorithm and
       derstanding of the uniform sampling method used in         generated vertex representations by using skip-gram model.
       DeepWalk for learning graph representations, which         This is considered as an equally weighted linear combina-
       translates a graph structure into linear sequences. Fur-   tion of k-step information. Tang et al. [25] later proposed a
       thermore, we explicitly define their loss function over    large-scale information network embedding, which optimizes
       graphs and extend it to support weighted graphs.           a loss function where both 1-step and 2-step relational in-
                                                                  formation can be captured in the learning process.
     • We formally analyze the deficiency associated with the
       skip-gram model with negative sampling. Our model
       defines a more accurate loss function that allows non-
                                                                  3.    GRAREP MODEL
       linear combinations of different local relational infor-      In this section, we define our task and present our loss
       mation to be integrated.                                   function for our task, which is then optimized with the ma-
                                                                  trix factorization method.
   The organization of this paper is described below. Sec-
tion 2 discusses related work. Section 3 proposes our loss        3.1    Graphs and Their Representations
function and states the optimization method using matrix
factorization. Section 4 presents the overall algorithm. Sec-        Definition 1. (Graph) A graph is defined as G = (V, E).
tion 5 gives a mathematical explanation to elucidate the          V = {v1 , v2 , . . . , vn } is the set of vertices with each V indi-
rationality of the proposed work and shows its connection         cating one object while E = {ei,j } is the set of edges with
to previous work. Section 6 discusses the evaluation data         each E indicating the relationship between two vertices. A
and introduces baseline algorithms. Section 7 presents ex-        path is a sequence of edges which connect a sequence of ver-
periments as well as analysis on the parameter sensitivity.       tices.
Finally, we conclude in Section 8.
                                                                    We first define adjacency matrix S for a graph. For an
2.    RELATED WORK                                                unweighted graph, we have Si,j = 1 if and only if there exists
                                                                  an edge from vi to vj , and Si,j = 0 otherwise. For a weighted
2.1     Linear Sequence Representation Methods                    graph, Si,j is a real number called the weight of the edge ei,j ,
   Natural language corpora, consisting of streams of words,      which indicates the significance of edge. Although weights
can be regarded as special graph structures, that is, linear      can be negative, we only consider non-negative weights in
chains. Currently, there are two mainstream methods for           this paper. For notational convenience, we also use w and c
learning word representations: neural embedding methods           to denote vertices throughout this paper.
and matrix factorization based approaches.                          The following diagonal matrix D is known as the degree
   Neural embedding methods employ a fixed slide window           matrix for a graph with adjacency matrix S:
capturing context words of current word. Models like skip-                                  P
                                                                                                p Sip , if i = j
gram [18] are proposed, which provide an efficient approach                         Dij =
                                                                                             0,         if i 6= j
to learning word representations. While these methods may
yield good performances on some tasks, they can poorly cap-          Assume we would like to capture the transitions from one
ture useful information since they use separate local context     vertex to another, and assume in the adjacency matrix Si,j
windows, instead of global co-occurrence counts [19]. On          is proportional to the transition probability from vi to vj , we
the other hand, the family of matrix factorization methods        can define the following (1-step) probability transition matrix
                                                                                        C1                                A1         A2
                                                   A1         A2
                     A1         A2                                                      C2 A2
                                                                        A1
                                                                                                                     B1                   B2
                                                                              B         C3

                                              B1    B2    B3       B4                   C4
                                                                                                           4    C1         C2        C3         C4
                                              A1
                     A1 (a) A2                          (b) A2                (c)                                         A1 (d) A2
                                                                                                                                C3             C4
                                              A1              A2
                     A1         A2                                      A1    B     C        A2                           A1         A2



                                                          B                                                          B1                   B2


                          (e)                           (f)                   (g)                                              (h)

Figure 1: The importance of capturing different k-step information in the graph representations. Here we
give examples for k = 1, 2, 3, 4.

A:                                                                           their relation. Clearly such 3-step information is essential to
                                     −1                                      be captured when learning a good graph representation with
                            A=D           S
                                                                             global structural information. Similarly, the 4-step informa-
where Ai,j is the probability of a transition from vi to vertex              tion can also be crucial in revealing the global structural
vj within one step. It can be observed that the A matrix                     properties of the graph, as illustrated in (d) and (h). Here,
can be regarded as a re-scaled S matrix whose rows are                       in (d), the relation between A1 and A2 is clearly strong,
normalized.                                                                  while in (h) the two vertices are unrelated since there does
                                                                             not exist a path from one vertex to the other. In the absence
   Definition 2. (Graph Representations with Global Struc-                   of 4-step relational information, such important distinctions
tural Information) Given a graph G, the task of Learning                     can not be properly captured.
                                                                                                                                                      B
Graph Representations with Global Structural Information
                                                                                                       A
aims to learn a global representation matrix W ∈ R|V |×d for
the complete graph, whose i-th row Wi is a d-dimensional                                                                                       C1          A   C4
                                                                                                           B
vector representing the vertex vi in the graph G where the
global structural information of the graph can be captured
in such vectors.                                                                                                                               C2              C3
                                                                                        C1        C2       C3   C4

                                                                                                       (a)                                           (b)
   In this paper, global structural information serves two
functions: 1) the capture of long distance relationship be-                  Figure 2: The importance of maintaining different
tween two different vertices and 2) the consideration of dis-                k-step information separately in the graph represen-
tinct connections in terms of different transitional steps.                  tations.
This would be further illustrated later.
   As we have discussed earlier, we believe the k-step (with                    We also additionally argue that it is essential to treat
varying k) relational information from the graph needs to                    different k-step information differently when learning graph
be captured when constructing such global graph representa-                  representations. We present one simple graph in (a) of Fig-
tions. To validate this point, Figure 1 gives some illustrative              ure 2. Let us focus on learning the representation for the
examples showing the importance of k-step (for k=1,2,3,4)                    vertex A in the graph. We can see that A receives two types
relational information that needs to be captured between                     of information when learning its representation: 1-step in-
two vertices A1 and A2 . In the figure, a thick line indicates               formation from B, as well as 2-step information from all C
a strong relation between two vertices, while a thin line in-                vertices. We note that if we do not distinguish these two dif-
dicates a weaker relation. Here, (a) and (e) show the impor-                 ferent types of information, we can construct an alternative
tance of capturing the simple 1-step information between                     graph as shown in (b) of Figure 2, where A receives exactly
the two vertices which are directly connected to each other,                 the same information as (a), but has a completely different
where one has a stronger relation and the other has a weaker                 structure.
relation. In (b) and (f), 2-step information is shown, where                    In this paper, we propose a novel framework for learning
in (b) both vertices share many common neighbors, and in                     accurate graph representations, integrating various k-step
(f) only one neighbor is shared between them. Clearly, 2-                    information which together captures the global structural
step information is important in capturing how strong the                    information associated with the graph.
connection between the two vertices is – the more common
neighbors they share, the stronger the relation between them                 3.2        Loss Function On Graph
is. In (c) and (g), the importance of 3-step information is                     We discuss our loss function used for learning the graph
illustrated. Specifically, in (g), despite the strong relation               representations with global structural information in this
between A1 and B, the relation between A1 and A2 can be                      section. Assume we are now given a graph with a collec-
weakened due to the two weaker edges connecting B and C,                     tion of vertices and edges. Consider a vertex w together
as well as C and A2 . In contrast, in (c), the relation between              with another vertex c. In order for us to learn global repre-
A1 and A2 remains strong because of the large number of                      sentations to capture their relations, we need to understand
common neighbors between B and A2 which strengthened                         how strongly these two vertices are connected to each other.
   Let us begin with a few questions. Does there exist a path                      Note that here N is the number of vertices in graph G,
from w to c? If so, if we randomly sample a path starting                       and q(w0 ) is the probability of selecting w0 as the first vertex
with w, how likely is it for us to reach c (possibly within                     in the path, which we assume follow a uniform distribution,
a fixed number of steps)? To answer such questions, we                          i.e., q(w0 ) = 1/N . This leads to:
first use the term pk (c|w) to denote the probability for a
                                                                                                                     λ X k
transition from w to c in exactly k steps. We already know                       Lk (w, c) = Akw,c · log σ(w
                                                                                                           ~ · ~c) +      A 0 · log σ(−w   ~ · ~c)
what are the 1-step transition probabilities, to compute the                                                         N 0 w ,c
                                                                                                                        w
k-step transition probabilities we introduce the following k-
step probability transition matrix                                               Following [16], we define e = w  ~ · ~c, and setting ∂L
                                                                                                                                       ∂e
                                                                                                                                         k
                                                                                                                                           = 0.
                                                                                This yields the following:
                            Ak = A
                                 | ·{z
                                     · · A}                                                                            !
                                          k                                                                  Akw,c
                                                                                             ~ · ~c = log
                                                                                             w             P     k
                                                                                                                          − log(β)
  We can observe that Aki,j exactly refers to the transition                                                 w0 Aw0 ,c

probability from vertex i to vertex j where the transition                      where β = λ/N .
consists of exactly k step(s). This directly leads to:                            This concludes that we essentially need to factorize the
                           pk (c|w) = Akw,c                                     matrix Y into two matrices W and C, where each row of W
                                                                                and each row of C consists of a vector representation for the
where Akw,c is the element from w-th row and c-th column                        vertex w and c respectively, and the entries of Y are:
of the matrix Ak .                                                                                                        !
   Let us consider a particular k first. Given a graph G,                                 k      k    k           Aki,j
                                                                                        Yi,j = Wi · Cj = log     P     k
                                                                                                                            − log(β)
consider the collection of all paths consisting of k steps that                                                    t At,j
can be sampled from the graph which start with w and end
with c (here we call w “current vertex”, and c “context ver-                      Now we have defined our loss function and showed that
tex”). Our objective aims to maximize: 1) the probability                       optimizing the proposed loss essentially involves a matrix
that these pairs come from the graph, and 2) the probability                    factorization problem.
that all other pairs do not come from the graph.
   Motivated by the skip-gram model by Mikolov et al. [18],
                                                                                3.3     Optimization with Matrix Factorization
we employ noise contrastive estimation (NCE), which is pro-                       Following the work of Levy et al. [16], to reduce noise,
posed by Gutmann et al. [11], to define our objective func-                     we replace all negative entries in Y k with 0. This gives us a
tion. Following a similar discussion presented in [16], we first                positive k-step log probabilistic matrix X k , where
introduce our k-step loss function defined over the complete                                           k
                                                                                                      Xi,j         k
                                                                                                           = max(Yi,j , 0)
graph as follows:
                              X                                                    While various techniques for matrix factorization exist, in
                       Lk =       Lk (w)                                        this work we focus on the popular singular value decompo-
                                    w∈V
                                                                                sition (SVD) method due to its simplicity. SVD has been
where                                                                           shown successful in several matrix factorization tasks [9, 14],
                                                                                and is regarded as one of the important methods that can be
                                              !
            X
Lk (w) =                                                        w·c~0 )]
                                 ~ · ~c) +λEc0 ∼pk (V ) [log σ(− ~
                  pk (c|w) log σ(w                                              used for dimensionality reduction. It was also used in [16].
            c∈V                                                                    For the matrix X k , SVD factorizes it as:
   Here pk (c|w) describes the k-step relationship between w                                           X k = U k Σk (V k )T
and c (the k-step transition probability from w to c), σ(·)
is sigmoid function defined as σ(x) = (1 + e−x )−1 , λ is a                     where U and V are orthonormal matrices and Σ is a diagonal
hyper-parameter indicating the number of negative samples,                      matrix consisting of an ordered list of singular values.
and pk (V ) is the distribution over the vertices in the graph.                  We can approximate the original matrix X k with Xdk :
The term Ec0 ∼pk (V ) [·] is the expectation when c0 follows the                                   X k ≈ Xdk = Udk Σkd (Vdk )T
distribution pk (V ), where c0 is an instance obtained from
negative sampling. This term can be explicitly expressed as:                    where Σkd is the matrix composed by the top d singular val-
                                                                                ues, and Udk and Vdk are first d columns of U k and V k , re-
                       ~ · c~0 )]
  Ec0 ∼pk (V ) [log σ(−w
                                                                                spectively (which are the first d eigenvector of XX T and
                                                                                X T X respectively).
                                      X
   = pk (c) · log σ(−w
                     ~ · ~c) +                    pk (c0 ) · log σ(−w
                                                                    ~ · c~0 )
                                    c0 ∈V \{c}                                    This way, we can factorize our matrix X k as:
  This leads to a local loss defined over a specific (w,c):                                           X k ≈ Xdk = W k C k
 Lk (w, c) = pk (c|w) · log σ(w
                              ~ · ~c) + λ · pk (c) · log σ(−w
                                                            ~ · ~c)             where
                                                                                                            1                    1      T
  In this work, we set a maximal length for each path we                                   W k = Udk (Σkd ) 2 ,    C k = (Σkd ) 2 Vdk
consider. In other words, we assume 1 ≤ k ≤ K. In fact,
when k is large enough, the transition probabilities converge                     The resulting W k gives representations of current vertices
to certain fixed values. The distribution pk (c) can be com-                    as its column vectors, and C k gives the representations of
puted as follows:                                                               context vertices as its column vectors [6, 5, 30]. The fi-
                                                                                nal matrix W k is returned from the algorithm as the low-d
                   X                      1 X k
          pk (c) =     q(w0 )pk (c|w0 ) =      Aw0 ,c                           representations of the vertices which capture k-step global
                     0
                                          N  0                                  structural information in the graph.
                      w                                  w
   In our algorithm we consider all k-step transitions with
all k = 1, 2, . . . , K, where K is a pre-selected constant. In               Table 1: Overall Algorithm
our algorithm, we integrate all such k-step information when       GraRep Algorithm
learning our graph representation, as to be discussed next.        Input
   Note that here we are essentially finding a projection from     Adjacency matrix S on graph
the row space of X k to the row space of W k with a lower          Maximum transition step K
rank. Thus alternative approaches other than the popular           Log shifted factor β
SVD can also be exploited. Examples include incremental            Dimension of representation vector d
SVD [22], independent component analysis (ICA) [7, 13],
                                                                   1. Get k-step transition probability matrix Ak
and deep neural networks [12]. Our focus in this work is on
                                                                   Compute A = D−1 S
the novel model for learning graph representations, so we
do not pursue any alternative methods. In fact, if alterna-        Calculate A1 , A2 , . . . , AK , respectively
tive method such as sparse auto-encoder is used at this step,      2. Get each k-step representations
it becomes relatively harder to justify whether the empiri-        For k = 1 to K
cal effectiveness of our representations is due to our novel           2.1 Get positive log probabilityP       matrix
model, or comes from any non-linearity introduced in this              calculate Γk1 , Γk2 , . . . , ΓkN (Γkj = p Akp,j ) respectively
                                                                                      k
dimensionality reduction step. To maintain the consistency             calculate {Xi,j    }              
with Levy et al. [16], we only employed SVD in this work.                           k             Ak
                                                                                                   i,j
                                                                                   Xi,j = log      Γk
                                                                                                          − log(β)
                                                                                                    j

4.    ALGORITHM                                                        assign negative entries of X k to 0
                                                                       2.2 Construct the representation vector W k
   We detail our learning algorithm in this section. In gen-
                                                                       [U k , Σk , (V k )T ] = SV D(X k )
eral, graph representations are extracted for other applica-                               1
tions as features, such as classification and clustering. An           W k = Udk (Σkd ) 2
effective way to encode k-step representation in practice is to    End for
concatenate the k-step representation as a global feature for      3. Concatenate all the k-step representations
each vertex, since each different step representation reflects     W = [W 1 , W 2 , . . . , W K ]
different local information. Table 1 shows the overall algo-       Output
rithm, and we explain the essential steps of our algorithm         Matrix of the graph representation W
here.
   Step 1. Get k-step transition probability matrix
Ak for each k = 1, 2, . . . , K.
   As shown in Section 3, given a graph G, we can calcu-          (truncated random walk). This method first samples uni-
late the k-step transition probability matrix Ak through the      formly a random vertex from the graph, then walks ran-
product of inverse of degree matrix D and adjacent matrix         domly to one of its neighbors and repeats this process. If the
S. For a weighted graph, S is a real matrix, while for an         length of the vertex sequence reaches a certain preset value,
unweighted graph, S is a binary matrix. Our algorithm is          then stop and start generating a new sequence. This pro-
applicable to both cases.                                         cedure can be used to produce a large number of sequences
   Step 2. Get each k-step representation                         from the graph.
   We get k-step log probability matrix X k , then subtract          Essentially, for an unweighted graph, this strategy of uni-
each entry by log (β), and replace the negative entries by        form sampling works, while for a weighted graph, a proba-
zeros. After that, we construct the representational vectors      bilistic sampling method based on the weights of the edges is
as rows of W k , where we introduce a solution to factorize the   needed, which is not employed in DeepWalk. In this paper,
positive log probability matrix X k using SVD. Finally, we        we propose an Enhanced SGNS (E-SGNS) method suitable
get all k-step representations for each vertex on the graph.      for weighted graphs. We also note that DeepWalk optimizes
   Step 3. Concatenate all k-step representations                 an alternative loss function (hierarchical softmax) that is
   We concatenate all k-step representations to form a global     different from negative sampling.
representation, which can be used in other tasks as features.        First, we consider total K-step loss L on whole graph

                                                                                       L = f (L1 , L2 , . . . , LK )
5.    SKIP-GRAM MODEL AS A SPECIAL CASE
      OF GRAREP                         where f (·) is a linear combination of its arguments defined
                                                                  as follows:
  GraRep aims to learn representations for graphs where
we optimize the loss function based on matrix factorization.                f (ϕ1 , ϕ2 , · · · , ϕK ) = ϕ1 + ϕ2 + · · · + ϕK
On the other hand, SGNS has been shown to be successful
in handling linear structures such as natural language sen-         We focus on the loss of a specific pair (w,c) which are i-
tences. Is there any intrinsic relationship between them? In      th and j-th vertex in the graph. Similar to Section 3.2, we
this section, we provide a view of SGNS as a special case of      assign partial derivative to 0, and get
the GraRep model.                                                                                         
                                                                               E−SGN S             Mi,j
5.1    Explicit Loss of Skip-gram Model on Graph                             Yi,j        = log P             − log(β)
                                                                                                    t Mt,j
  SGNS aims at representing words in linear sequences, so
we need to translate a graph structure into linear structures.    where M is transition probability matrix within K step(s),
Deepwalk reveals an effective way with uniform sampling           and Mi,j refers to transition probability from vertex i to
            E−SGN S
vertex j. Yi,j      is a factorized matrix for E-SGNS, and            Finally, we plug these expected counts into the equation
                                                                        E−SGN S
                                                                   of Yi,j       , and we arrive at:
                  M = A1 + A2 + · · · + AK                                                                   
                                                                               E−SGN S          #(w, c) · |D|
The difference between E-SGNS and GraRep model is on the                     Yw,c       = log                   − log(λ)
                                                                                                #(w) · #(c)
definition of f (·). E-SGNS can be considered as a linear com-
bination of K-step loss, and each loss has an equal weight.        where D is the collection of all observed pairs in sequences.
Our GraRep model does not make such a strong assump-               Here |D| = γKN where N is the total number of vertices
tion, but allows their (potentially non-linear) relationship       in the graph. This matrix Y E−SGN S becomes exactly the
to be learned from data in practice. Intuitively, different        same as that of SGNS as described in [16].
k-step transition probabilities should have different weights,       This shows SGNS is essentially a special version of our
and linear combination of these may not achieve desirable          GraRep model that deals with linear sequences which can
results for heterogeneous network data.                            be sampled from graphs. Our approach has several advan-
                                                                   tages over the slow and expensive sampling process, which
5.2    Intrinsic Relation Between Sampling and                     typically involves several parameters to tune, such as maxi-
       Transition Probabilities                                    mum length of linear sequence, sampling frequency for each
  In our approach, we used transition probabilities to mea-        vertex, and so on.
sure relationship between vertices. Is this reasonable? In
this subsection, we articulate the intrinsic relation between      6.     EXPERIMENTAL DESIGN
sampling and transition probabilities.                               In this section, we assess the effectiveness of our GraRep
  Among the sequences generated by random walk, we as-             model through experiments. We conduct experiments on
sume vertex w occurs a total of T = γ · K times:                   several real-word datasets for several different tasks, and
                           #(w) = γ · K                            make comparisons with baseline algorithms.

where γ is a variable related to #(w). We regard vertex w          6.1     Datasets and Tasks
as the current vertex, then the expected number of times              In order to demonstrate the performance of GraRep, we
that we see c1 as its direct neighbor (1-step away from w)         conducted the experiments across three different types of
is:                                                                graphs – a social network, a language network and a cita-
                  #(w, c1 ) = γ · K · p1 (c|w)                     tion network, which include both weighted and unweighted
                                                                   graphs. We conducted experiments across three different
This holds for both uniform sampling for unweighted graphs         types of tasks, including clustering, classification, and vi-
or our proposed probabilistic sampling for weighted graphs.        sualization. As we mentioned in earlier sections, this work
  Further, we analyze the expected number of times of co-          focuses on proposing a novel framework for learning good
occurrence for w and c of context window size 2,                   representation of graph with global structural information,
                     X                                             and aims to validate the effectiveness of our proposed model.
   #(w, c2 ) = γ · K    p(c2 |c0 ) · p(c0 |w) = γ · K · p2 (c|w)   Thus we do not employ alternative more efficient matrix
                      c0
                                                                   factorization methods apart from SVD, and focused on the
where c can be any vertex bridging w and c2 . That is, c0 is
        0                                                          following three real-world datasets in this section.
shared neighbor between w and c2 . Similarly, we can derive           1. 20-Newsgroup1 is a language network, which has ap-
the equations for k = 3, 4, · · · , K:                             proximately 20,000 newsgroup documents and is partitioned
                                                                   by 20 different groups. In this network, each document is
                  #(w, c3 ) = γ · K · p3 (c|w)                     represented by a vector with tf-idf scores of each word, and
                          ..                                       the cosine similarity is used to calculate the similarity be-
                           .                                       tween two documents. Based on these similarity scores of
                 #(w, cK ) = γ · K · pK (c|w)                      each pair of documents, a language network is built. Fol-
                                                                   lowing [29], in order to show the robustness of our model,
then we add them up and divide both sides by K, leading            we also construct the following 3 graphs built from 3, 6 and
to:                                                                9 different newsgroups respectively (note that NG refers to
                                  K
                                  X                                “Newsgroups”):
                  #(w, c) = γ ·         pk (c|w)                      3-NG:comp.graphics, comp.graphics and talk.politics.guns;
                                  k=1                              6-NG: alt.atheism, comp.sys.mac.hardware, rec.motorcycles,
where #(w, c) is the expected co-occurrence count between          rec.sport.hockey, soc.religion.christian and talk.religion.misc;
w and c within K step(s).                                          9-NG: talk.politics.mideast, talk.politics.misc, comp.os.ms-
  According to definition of Mw,c , we can get                     windows.misc, sci.crypt, sci.med, sci.space, sci.electronics,
                                                                   misc.forsale, and comp.sys.ibm.pc.hardware
                      #(w, c) = γ · Mw,c                              Besides randomly sampling 200 documents from a topic
  Now, we can also compute the expected number of times            as described in [29], we also conduct experiment on all doc-
we see c as the context vertex, #(c):                              uments as comparison. The topic label on each document is
                                X                                  considered to be true.
                    #(c) = γ ·     Mt,c                               This language network is a fully connected and weighted
                                   t                               graph, and we will demonstrate the results of clustering,
                                                                   using graph representation as features.
where we consider the transitions from all possible vertices
                                                                   1
to c.                                                                  qwone.com/˜jason/20Newsgroups/
   2. Blogcatalog2 is a social network, where each vertex            4. Spectral Clustering [23]. Spectral clustering is a
indicates one blogger author, and each edge corresponds to              reasonable baseline algorithm, which aims at minimiz-
the relationship between authors. 39 different types of topic           ing Normalized Cut (NCut). Like our method, Spec-
categories are presented by authors as labels.                          tral clustering also factorize a matrix, but it focuses on
   Blogcatalog is an unweighted graphs, and we test the per-            a different matrix of the graphs – the Laplacian Matrix.
formance of the learned representations on the multi-label              Essentially, the difference between spectral clustering
classification task, where we classify each author vertex into          and E-SGNS lies on their different loss function.
a set of labels. The graph representations generated from
our model and each baseline algorithm are considered as            6.3     Parameter Settings
features.                                                             As suggested in [25], for LINE, we set the mini-batch size
   3. DBLP Network3 is a citation network. We extract              of stochastic gradient descent (SGD) as 1, learning rate of
author citation network from DBLP, where each vertex in-           starting value as 0.025, the number of negative samples as
dicates one author and the number of references from one           5, and the total number of samples as 10 billion. We also
author to the other is recorded by the weight of edge be-          concatenate both 1-step and 2-step relational information
tween these two authors. Following [26], we totally select 6       to form the representations and employ the reconstruction
different popular conferences and assign them into 3 groups,       strategy for vertices with small degrees to achieve the opti-
where WWW and KDD are grouped as data mining, NIPS                 mal performance. As mentioned in [20], for DeepWalk and
and ICML as machine learning, and CVPR and ICCV as                 E-SGNS, we set window size as 10, walk length as 40, walks
computer vision. We visualize the learned representations          per vertex as 80. According to [25], LINE yielded better
from all systems using a visualisation tool t-SNE [31], which      results when the learned graph representations are L2 nor-
provides both qualitative and quantitative results for the         malized, while DeepWalk and E-SGNS can achieve optimal
learned representations. We give more details in Section           performance without normalization. For GraRep, we found
6.4.3.                                                             the L2 normalization yielded better results. We reported
   To summarize, in this paper we conduct experiments on           all the results based on these findings for each system ac-
both weighted and unweighted graphs, and both sparse and           cordingly. For a fair comparison, the dimension d of repre-
dense graphs, where three different types of learning tasks        sentations is set as 128 for Blogcatalog network and DBLP
are carried out. More details of the graph we used are shown       network as used in [25] and is set as 64 for 20-NewsGroup
in Table 2.                                                        network as used in [29]. For GraRep, we set β = N1 and max-
                                                                   imum matrix transition step K=6 for Blogcatalog network
6.2      Baseline Algorithms                                       and DBLP network and K=3 for 20-NewsGroup network.
  We use the following methods of graph representation as          To demonstrate the advantage of our model which captures
baseline algorithms.                                               global information of the graph, we also conducted exper-
                                                                   iments under various parameter settings for each baseline
     1. LINE [25]. LINE is a recently proposed method for          systems, which are discussed in detail in the next section.
        learning graph representations on large-scale informa-
        tion networks. LINE defines a loss function based on       6.4     Experimental Results
        1-step and 2-step relational information between ver-         In this section, we present empirical justifications that our
        tices. One strategy to improve the performance of          GraRep model can integrate different k-step local relational
        vertices with small degrees is to make graph denser        information into a global graph representation for different
        by expanding their neighbors. LINE will get the best       types of graphs, which can then be effectively used for dif-
        performance, if concatenating the representation of 1-     ferent tasks. We make the source code of GraRep available
        step and 2-step relational information and tuning the      at http://shelson.top/.
        threshold of maximum number of vertices.
                                                                   6.4.1    20-Newsgroup Network
     2. DeepWalk [20]. DeepWalk is a method that learns               We first conduct an experiment on a language network
        the representation of social networks. The original        through a clustering task by employing the learned repre-
        model only works for unweighted graph. For each ver-       sentations in a k-means algorithm [2].
        tex, truncated random walk is used to translate graph         To assess the quality of the results, we report the aver-
        structure into linear sequences. The skip-gram model       aged Normalized Mutual Information (NMI) score [24] over
        with hierarchical softmax is used as the loss function.    10 different runs for each system. To understand the effect
                                                                   of different dimensionality d in the end results, we also show
     3. E-SGNS. Skip-gram is an efficient model that learns        the results when dimension d is set to 192 for DeepWalk,
        the representation of each word in large corpus [18].      E-SGNS and Spectral Clustering. For LINE, we employ the
        For this enhanced version, we first utilize uniform sam-   reconstruction strategy proposed by their work by adding
        pling for unweighted graph and probabilistic sampling      neighbors of neighbors as additional neighbors to improve
        proportional to weight of edges for weighted graph, to     performance. We set k-max=0, 200, 500, 1000 for experi-
        generate linear vertex sequences, and then introduce       ments, where k-max is a parameter that is used to control
        SGNS to optimize. This method can be regarded as a         how the higher-order neighbors are added to each vertex in
        special case of our model, where different representa-     the graph.
        tional vector of each k-step information is averaged.         As shown in Table 3, the highest results are highlighted
                                                                   in bold for each column. We can see that GraRep con-
2
    leitang.net/code/social-dimension/data/blogcatalog.mat         sistently outperforms other baseline methods for this task.
3
    aminer.org/billboard/citation                                  For DeepWalk, E-SGNS and Spectral Clustering, increasing
                                            Table 2: Statistics of the real-world graphs
                                                         Language Network                      Social Network        Citation Network
                                               20-NewsGroup         20-NewsGroup                                           DBLP
                                 Name                                                            Blogcatalog
                                                (200 samples)          (all data)                                     (author citation)
                                Type               weighted            weighted                   unweighted              weighted
                                #(V)         600, 1200 and 1800 1,720, 3,224 and 5,141               10,312                 7,314
                                #(E)           Fully connected     Fully connected                  333,983                72,927
                              Avg. degree             —                    —                          64.78                 19.94
                               #Labels            3, 6 and 9          3, 6 and 9                       39                     3
                                 Task             Clustering          Clustering                 Classification         Visualization

                                                     Table 3: Results on 20-NewsGroup
                                                                           200 samples                               all data
                                         Algorithm              3NG(200)    6NG(200) 9NG(200)            3NG(all)   6NG(all)    9NG(all)
                                        GraRep                   81.12          67.53      59.43          81.44       71.54         60.38
                                   LINE (k-max=0)                80.36          64.88      51.58          80.58       68.35         52.30
                                  LINE (k-max=200)               78.69          66.06      54.14          80.68       68.83         53.53
                                       DeepWalk                  65.58          63.66      48.86          65.67       68.38         49.19
                                  DeepWalk (192dim)              60.89          59.89      47.16          59.93       65.68         48.61
                                        E-SGNS                   69.98          65.06      48.47          69.04       67.65         50.59
                                   E-SGNS (192dim)               63.55          64.85      48.65          66.64       66.57         49.78
                                  Spectral Clustering            49.04          51.02      46.92          62.41       59.32         51.91
                              Spectral Clustering (192dim)       28.44          27.80      36.05          44.47       36.98         47.36
                              Spectral Clustering (16dim)        69.91          60.54      47.39          78.12       68.78         57.87


                                                      Table 4: Results on Blogcatalog
                               Metric         Algorithm         10%      20%     30%     40%      50%      60%      70%       80%     90%
                                               GraRep           38.24   40.31    41.34   41.87   42.60     43.02    43.43   43.55    44.24
                                                LINE            37.19   39.82    40.88   41.47   42.19     42.72    43.15   43.36    43.88
                              Micro-F1
                                              DeepWalk          35.93   38.38    39.50   40.39   40.79     41.28    41.60   41.93    42.17
                                               E-SGNS           35.71   38.34    39.64   40.39   41.23     41.66    42.01   42.16    42.25
                                          Spectral Clustering   37.16   39.45    40.22   40.87   41.27     41.50    41.48   41.62    42.12
                                               GraRep           23.20   25.55    26.69   27.53   28.35     28.78    29.67   29.96    30.93
                                                LINE            19.63   23.04    24.52   25.70   26.65     27.26    27.94   28.68    29.38
                              Macro-F1
                                              DeepWalk          21.02   23.81    25.39   26.27   26.85     27.36    27.67   27.96    28.41
                                               E-SGNS           21.01   24.09    25.61   26.59   27.64     28.08    28.33   28.34    29.26
                                          Spectral Clustering   19.26   22.24    23.51   24.33   24.83     25.19    25.36   25.52    26.21




 (a) Spectral Clustering       (b) DeepWalk                              (c) E-SGNS                                  (d) LINE                (e) GraRep
Figure 3: Visualization of author citation network. Each point indicates one author. Green: Data Mining,
magenta: computer vision and blue: Machine Learning.

the dimension d of representations does not appear to be                                  6.4.2          Blogcatalog Network
effective in improving the performance. We believe this is                                  In this experiment, we focus on a supervised task on a so-
because a higher dimension does not provide different com-                               cial network. We evaluate the effectiveness of different graph
plementary information to the representations. For LINE,                                 representations through a multi-label classification task by
the reconstruction strategy does help, since it can capture                              regarding the learned representations as features.
additional structural information of the graph beyond 1-step                                Following [20, 27], we use the LibLinear package [10] to
and 2-step local information.                                                            train one-vs-rest logistic regression classifiers, and we run
   It is worth mentioning that GraRep and LINE can achieve                               this process for 10 rounds and report the averaged Micro-
good performance with a small graph. We believe this is be-                              F1 and Macro-F1 measures. For each round, we randomly
cause both approaches can capture rich local relational in-                              sample 10% to 90% of the vertices and use these samples
formation even when the graph is small. Besides, for tasks                               for training, and use the remaining vertices for evaluation.
with more labels, such as 9NG, GraRep and LINE can pro-                                  As suggested in [25], we set k-max as 0, 200, 500 and 1000,
vide better performances than other methods, with GraRep                                 respectively, and we report the best performance with k-
providing the best. An interesting finding is that Spectral                              max=500.
Clustering arrives at its best performance when setting di-                                 Table 4 reports the results of GraRep and each baseline al-
mension d to 16. However, for all other algorithms, their                                gorithm. The highest performance is highlighted in bold for
best results are obtained when d is set to 64. We show these                             each column which corresponds to one particular train/test
results in Figure 5 when we assess the sensitivity of param-                             split. Overall, GraRep significantly outperforms other meth-
eters.                                                                                   ods, especially under the setting where only 10% of the ver-
   Due to limited space, we do not report the detailed results                           tices are used for training. This indicates different types
with d=128 and d=256 for all baseline methods, as well as                                of rich local structural information learned by the GraRep
k-max=500, 1000 and without reconstruction strategy for                                  model can be used to complement each other to capture
LINE, since these results are no better than the results pre-                            the global structural properties of the graph, which serves
sented in Table 3. In Section 6.5, we will further discuss the
issue of parameter sensitivity.
                                                                                                                            Empirically we can observe that the performance of K=7
Table 5: Final KL divergence for the DBLP dataset                                                                           is no better than K=6. In this figure, for readability and
  Algorithm     GraRep LINE DeepWalk E-SGNS                                                                                 clarity we only present the results of K=1,2,3,6 and 7. We
 KL divergence  1.0070  1.0816    1.1115  1.1009                                                                            found the performance of K = 4 is slightly better than K=3,
                                                                                                                            while the results for K=5 is comparable to K=4.
as a distinctive advantage over existing baseline approaches,
especially when the data is relatively scarce.                                                                                                     0.9                                                                                        0.6



                                                                                                                                                   0.8

6.4.3                   DBLP Network                                                                                                                                                                                                  0.55


                                                                                                                                                   0.7

   In this experiment, we focus on visualizing the learned rep-                                                                  NMI                                                                                  NMI                     0.5


resentations by examining a real citation network – DBLP.                                                                                          0.6



We feed the learned graph representations into the standard                                                                                        0.5                                                 GraRep
                                                                                                                                                                                                       DeepWalk
                                                                                                                                                                                                                                      0.45
                                                                                                                                                                                                                                                                                         GraRep
                                                                                                                                                                                                                                                                                         DeepWalk
                                                                                                                                                                                                       E-SGNS                                                                            E-SGNS
t-SNE tool [31] to lay out the graph, where the authors from                                                                                       0.4
                                                                                                                                                                                                       LINE
                                                                                                                                                                                                                                              0.4
                                                                                                                                                                                                                                                                                         LINE

                                                                                                                                                          32    64              128                           256                                   32     64              128                 256
the same research area (one of data mining, machine learn-                                                                                                                 Dimensions                                                                                    Dimensions
ing or computer vision, see Section 6.1) share the same color.                                                                                                    (a) 3NG, NMI                                                                               (b) 9NG, NMI
The graphs are visualized on a 2-dimensional space and the
Kullback-Leibler divergence is reported [31], which captures                                                                                                   Figure 5: Performance over dimensions, d
the errors between the input pairwise similarities and their                                                                   Figure 5 shows the NMI scores of each algorithm over dif-
projections in the 2-dimensional map as displayed (a lower                                                                  ferent settings of the dimension d on 3NG and 9NG data.
KL divergence score indicates a better performance).                                                                        We can observe that GraRep consistently outperforms other
   Under the help of t-SNE toolkit, we set the same param-                                                                  baseline algorithms which learn representations with the same
eter configuration and import representations generated by                                                                  dimension. This set of experiments serves as an additional
each algorithm. From Figure 3, we can see that the lay-                                                                     supplement to Table 3. Interestingly, all algorithms can ob-
out using Spectral Clustering is not very informative, since                                                                tain the optimal performance with d = 64. As we increase d
vertices of different colors are mixed with each other. For                                                                 from 64 to larger values, it appears that the performances of
DeepWalk and E-SGNS, results look much better, as most                                                                      all the algorithms start to drop. Nevertheless, our GraRep
vertices of the same color appear to form groups. However,                                                                  algorithm is still superior to the baseline systems across dif-
vertices still do not appear in clearly separable regions with                                                              ferent d values.
clear boundaries. For LINE and GraRep, the boundaries                                                                                              5500                                                                                       300

of each group become much clearer, with vertices of differ-                                                                                        5000
                                                                                                                                                                                                                                              250




                                                                                                                            Cost of time/seconds                                                                       Cost of time/seconds
ent colors appearing in clearly distinguishable regions. The                                                                                       4500

                                                                                                                                                   4000

results for GraRep appear to be better, with clearer bound-                                                                                        3500
                                                                                                                                                                                                                                              200




aries for each regions as compared to LINE.                                                                                                        3000

                                                                                                                                                   2500
                                                                                                                                                                                                                                              150



   Table 5 reports the KL divergence at the end of iterations.                                                                                     2000
                                                                                                                                                                                                                                              100



Under the same parameter setting, a lower KL divergence                                                                                            1500
                                                                                                                                                                                                                                               50
                                                                                                                                                   1000

indicates a better graph representation. From this result, we                                                                                       500
                                                                                                                                                      128       128*2   128*3         128*4   128*5   128*6   128*7
                                                                                                                                                                                                                                                0
                                                                                                                                                                                                                                                    600   1200   1800            3224       5141

also can see that GraRep yields better representations than                                                                                                                Dimensions                                                                                   Nodes of graph

all other baseline methods.                                                                                                 (a) Blogcatalog, running time                                                                                     (b) 20NG, running time

6.5                Parameter Sensitivity                                                                                    Figure 6: Running time as a function of a) dimension
                                                                                                                            and b) size of graph.
   We discuss the parameter sensitivity in this section. Specif-
ically, we assess the how the different choices of the maximal                                                                Figure 6 shows the running time over different dimensions
k-step size K, as well as the dimension d for our represen-                                                                 as well as over different graph sizes. In Figure 6(a), we set
tations can affect our results.                                                                                             K from 1 to 7 where each complementary feature vector has
           0.46                                                          0.32                                               a dimension of 128. This set of experiments is conducted
                                                                          0.3
           0.44
                                                                         0.28
                                                                                                                            on the Blogcatalog dataset which contains around 10,000
           0.42                                                          0.26                                               vertices. It can be seen that the running time increases ap-
Micro-F1                                                      Macro-F1
            0.4
                                                                         0.24
                                                                                                                            proximately linearly as the dimension increases. In Figure
                                                                         0.22

           0.38                                                           0.2
                                                                                                                            6(b), we analyze running time with respect to graph sizes.
                                                       K=7
                                                       K=6               0.18
                                                                                                                     K=7
                                                                                                                     K=6    This set of experiment is conducted on different size of the
           0.36                                        K=5                                                           K=5
                                                       K=2
                                                       K=1
                                                                         0.16                                        K=2
                                                                                                                     K=1    20-NewsGroup dataset. The result shows a significant in-
           0.34                                                          0.14
              1%   2%   3%    4%   5%   6%   7%   8%     9%                 1%   2%   3%    4%   5%   6%   7%   8%     9%   crease of running time as graph size increases. The reason
                             Labeled Nodes                                                 Labeled Nodes
                                                                                                                            resulting in the large increase in running time is mainly due
           (a) Blogcatalog, Micro-F1                               (b) Blogcatalog, Macro-F1
                                                                                                                            to high time complexity involved in the computation of the
                        Figure 4: Performance over steps, K                                                                 power of a matrix and the SVD procedure.
   Figure 4 shows the Micro-F1 and Macro-F1 scores over
different choices of K on the Blogcatalog data. We can ob-                                                                  7.                                 CONCLUSIONS
serve that the setting K=2 has a significant improvement                                                                      In this paper, GraRep, a novel model for learning better
over the setting K=1, and K=3 further outperforms K=2.                                                                      graph representations is proposed. Our model, with our k-
This confirms that different k-step can learn complementary                                                                 step loss functions defined on graphs which integrate rich
local information. In addition, as mentioned in Section 3,                                                                  local structural information associated with the graph, cap-
when k is large enough, learned k-step relational informa-                                                                  tures the global structural properties of the graph. We also
tion becomes weak and shifts towards a steady distribution.                                                                 provide mathematical derivations justifying the model and
  establish the connections to previous research efforts. Em-      [13] C. Jutten and J. Herault. Blind separation of sources,
  pirically, the learned representations can be effectively used        part i: An adaptive algorithm based on neuromimetic
  as features in other learning problems such as clustering             architecture. Signal processing, 24(1):1–10, 1991.
  and classification. This model comes with one limitation:        [14] V. Klema and A. J. Laub. The singular value
  the expensive computation of the power of a matrix and                decomposition: Its computation and some
  SVD involved in the learning process. Future work would               applications. Automatic Control, 25(2):164–176, 1980.
  include the investigation of efficient and online methods to     [15] T. K. Landauer, P. W. Foltz, and D. Laham. An
  approximate matrix algebraic manipulations, as well as in-            introduction to latent semantic analysis. Discourse
  vestigation of alternative methods by employing deep ar-              processes, 25(2-3):259–284, 1998.
  chitectures for learning low-dimensional representations in      [16] O. Levy and Y. Goldberg. Neural word embedding as
  place of SVD.                                                         implicit matrix factorization. In NIPS, pages
                                                                        2177–2185, 2014.
  8.          ACKNOWLEDGMENTS                                      [17] K. Lund and C. Burgess. Producing high-dimensional
                                                                        semantic spaces from lexical co-occurrence. BRMIC,
    The authors would like to thank the anonymous reviewers             28(2):203–208, 1996.
  for their helpful comments. The authors would also like to
                                                                   [18] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and
  thank Fei Tian, Qingbiao Miao and Jie Liu for their sug-
                                                                        J. Dean. Distributed representations of words and
  gestions or help with this work. This paper was done when
                                                                        phrases and their compositionality. In NIPS, pages
  the first author was an intern at SUTD. This work was sup-
                                                                        3111–3119, 2013.
  ported by SUTD grant ISTD 2013 064 and Temasek Lab
  project IGDST1403013.                                            [19] J. Pennington, R. Socher, and C. D. Manning. Glove:
                                                                        Global vectors for word representation. EMNLP, 12,
                                                                        2014.
  9.          REFERENCES                                           [20] B. Perozzi, R. Al-Rfou, and S. Skiena. Deepwalk:
                                                                        Online learning of social representations. In SIGKDD,
   [1] A. Ahmed, N. Shervashidze, S. Narayanamurthy,                    pages 701–710. ACM, 2014.
       V. Josifovski, and A. J. Smola. Distributed large-scale     [21] S. T. Roweis and L. K. Saul. Nonlinear dimensionality
       natural graph factorization. In WWW, pages 37–48.                reduction by locally linear embedding. Science,
       International World Wide Web Conferences Steering                290(5500):2323–2326, 2000.
       Committee, 2013.                                            [22] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl.
   [2] D. Arthur and S. Vassilvitskii. k-means++: The                   Incremental singular value decomposition algorithms
       advantages of careful seeding. In SODA, pages                    for highly scalable recommender systems. In ICIS,
       1027–1035. Society for Industrial and Applied                    pages 27–28. Citeseer, 2002.
       Mathematics, 2007.
                                                                   [23] J. Shi and J. Malik. Normalized cuts and image
   [3] M. Belkin and P. Niyogi. Laplacian eigenmaps and                 segmentation. PAMI, 22(8):888–905, 2000.
       spectral techniques for embedding and clustering. In        [24] A. Strehl, J. Ghosh, and R. Mooney. Impact of
       NIPS, volume 14, pages 585–591, 2001.                            similarity measures on web-page clustering. In
   [4] J. A. Bullinaria and J. P. Levy. Extracting semantic             Workshop on Artificial Intelligence for Web Search
       representations from word co-occurrence statistics: A            (AAAI 2000), pages 58–64, 2000.
       computational study. BRM, 39(3):510–526, 2007.
                                                                   [25] J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and
   [5] J. A. Bullinaria and J. P. Levy. Extracting semantic             Q. Mei. Line: Large-scale information network
       representations from word co-occurrence statistics:              embedding. In WWW. ACM, 2015.
       stop-lists, stemming, and svd. BRM, 44(3):890–907,          [26] J. Tang, J. Zhang, L. Yao, J. Li, L. Zhang, and Z. Su.
       2012.                                                            Arnetminer: extraction and mining of academic social
   [6] J. Caron. Experiments with lsa scoring: Optimal rank             networks. In SIGKDD, pages 990–998. ACM, 2008.
       and basis. In CIR, pages 157–169, 2001.                     [27] L. Tang and H. Liu. Relational learning via latent
   [7] P. Comon. Independent component analysis, a new                  social dimensions. In SIGKDD, pages 817–826. ACM,
       concept? Signal processing, 36(3):287–314, 1994.                 2009.
   [8] T. F. Cox and M. A. Cox. Multidimensional scaling.          [28] J. B. Tenenbaum, V. De Silva, and J. C. Langford. A
       CRC Press, 2000.                                                 global geometric framework for nonlinear
   [9] C. Eckart and G. Young. The approximation of one                 dimensionality reduction. Science,
       matrix by another of lower rank. Psychometrika,                  290(5500):2319–2323, 2000.
       1(3):211–218, 1936.                                         [29] F. Tian, B. Gao, Q. Cui, E. Chen, and T.-Y. Liu.
  [10] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang,                 Learning deep representations for graph clustering. In
       and C.-J. Lin. Liblinear: A library for large linear             AAAI, 2014.
       classification. JMLR, 9:1871–1874, 2008.                    [30] P. D. Turney. Domain and function: A dual-space
  [11] M. U. Gutmann and A. Hyvärinen. Noise-contrastive               model of semantic relations and compositions. JAIR,
       estimation of unnormalized statistical models, with              pages 533–585, 2012.
       applications to natural image statistics. JMLR,             [31] L. Van der Maaten and G. Hinton. Visualizing data
       13(1):307–361, 2012.                                             using t-sne. JMLR, 9(2579-2605):85, 2008.
  [12] G. E. Hinton and R. R. Salakhutdinov. Reducing the
       dimensionality of data with neural networks. Science,
       313(5786):504–507, 2006.




View publication stats

