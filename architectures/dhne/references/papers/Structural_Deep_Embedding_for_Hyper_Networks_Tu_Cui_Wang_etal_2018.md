# Structural Deep Embedding for Hyper Networks Tu Cui Wang etal 2018

> Source: `Structural_Deep_Embedding_for_Hyper_Networks_Tu_Cui_Wang_etal_2018.pdf`

---

                                                                     Structural Deep Embedding for Hyper-Networks

                                                                      Ke Tu1 , Peng Cui1 , Xiao Wang1 , Fei Wang2 , Wenwu Zhu1
                                                                      1
                                                                      Department of Computer Science and Technology, Tsinghua University
                                                                       2
                                                                         Department of Healthcare Policy and Research, Cornell University
                                                                tuke1993@gmail.com, cuip@tsinghua.edu.cn, wangxiao007@mail.tsinghua.edu.cn
                                                                                feiwang03@gmail.com,wwzhu@tsinghua.edu.cn



                                                                    Abstract                                order node relationships is usually referred to as a hyper-




arXiv:1711.10146v2 [cs.SI] 31 Jan 2018
                                                                                                            network.
                                           Network embedding has recently attracted lots of attentions
                                           in data mining. Existing network embedding methods mainly           A typical way to analyze hyper-network is to expand them
                                           focus on networks with pairwise relationships. In real world,    into conventional pairwise networks and then apply the an-
                                           however, the relationships among data points could go be-        alytical algorithms developed on pairwise networks. Clique
                                           yond pairwise, i.e., three or more objects are involved in       expansion (Sun, Ji, and Ye 2008) (Figure 1 (c)) and star ex-
                                           each relationship represented by a hyperedge, thus forming       pansion (Agarwal, Branson, and Belongie 2006) (Figure 1
                                           hyper-networks. These hyper-networks pose great challenges       (d)) are two representative techniques to achieve such a goal.
                                           to existing network embedding methods when the hyperedges        In clique expansion, each hyperedge is expanded as a clique.
                                           are indecomposable, that is to say, any subset of nodes in       In star expansion, a hypergraph is transformed into a bipar-
                                           a hyperedge cannot form another hyperedge. These inde-           tite graph where each hyperedge is represented by an in-
                                           composable hyperedges are especially common in hetero-
                                                                                                            stance node which links to the original nodes it contains.
                                           geneous networks. In this paper, we propose a novel Deep
                                           Hyper-Network Embedding (DHNE) model to embed hyper-             These methods assume that the hyperedges are decompos-
                                           networks with indecomposable hyperedges. More specifi-           able either explicitly or implicitly. That is to say, if we treat
                                           cally, we theoretically prove that any linear similarity met-    a hyperedge as a set of nodes, then any subset of nodes in
                                           ric in embedding space commonly used in existing meth-           this hyperedge can form another hyperedge. In a homoge-
                                           ods cannot maintain the indecomposibility property in hyper-     neous hyper-network, this assumption is reasonable, as the
                                           networks, and thus propose a new deep model to realize           formation of hyperedges are, in most cases, caused by the la-
                                           a non-linear tuplewise similarity function while preserving      tent similarity among the involved objects such as common
                                           both local and global proximities in the formed embedding        labels. However, when learning the heterogeneous hyper-
                                           space. We conduct extensive experiments on four different        network embedding, we need to address the following new
                                           types of hyper-networks, including a GPS network, an on-
                                                                                                            requirements.
                                           line social network, a drug network and a semantic network.
                                           The empirical results demonstrate that our method can signif-    1. Indecomposablity: The hyperedges in heterogeneous
                                           icantly and consistently outperform the state-of-the-art algo-      hyper-networks are usually indecomposable. In this case,
                                           rithms.                                                             a set of nodes in a hyperedge has a strong relationship,
                                                                                                               while the nodes in its subset does not necessarily have a
                                                                Introduction                                   strong relationship. For example, in the recommendation
                                         Nowadays, networks are widely used to represent the rich              system with huser, movie, tagi relationships, the huser,
                                         relationships of data objects in various domains, forming so-         tagi relationships are not typically strong. This means that
                                         cial networks, biology networks, brain networks, etc. Many            we cannot simply decompose hyperedges using those tra-
                                         methods are proposed for network analysis, among which                ditional expansion methods.
                                         network embedding methods (Tang et al. 2015; Wang, Cui,            2. Structure Preserving: The local structures are preserved
                                         and Zhu 2016; Deng et al. 2016; Ou et al. 2015) arouse                by the observed relationships in network embedding.
                                         more and more interests in recent years. Most of the existing         However, due to the sparsity of networks, many exist-
                                         network embedding methods are designed for conventional               ing relationships are not observed. It is not sufficient for
                                         pairwise networks, where each edge links only a pair of               preserving hyper-network structure using only local struc-
                                         nodes. However, in real world applications, the relationships         tures. And global structures, e.g. the neighborhood struc-
                                         among data objects are much more complicated and they                 ture, are required to address the sparsity problem. How to
                                         typically go beyond pairwise. For example, John purchas-              capture and preserve both local and global structures si-
                                         ing a shirt with cotton material forms a high-order relation-         multaneously in a hyper-network is still an unsolved prob-
                                         ship hJohn, shirt, cottoni. The network capturing those high-         lem.
                                         Copyright c 2018, Association for the Advancement of Artificial      To address Indecomposablity issue, we design an inde-
                                         Intelligence (www.aaai.org). All rights reserved.                  composable tuplewise similarity function. The function is
                                                                                                              4, we introduce the proposed model in details. Experimental
           𝒆𝟏
                       𝒆𝟐
                                 (𝒃)     𝐴2    U2    𝐿1        𝐴2   U2    𝐿2
                                                                                 …       𝐴1   U1     𝐿2
                                                                                                              results are presented in section 5. Finally, we conclude in
                 𝐴2
                                                                                     tuplewise similarity     Section 6.
                                          𝐴2        𝐴2    U2             𝐴1     𝐴1     U1
      𝐿2              U2           (𝒄)
                                                                …
                 𝐴1
                            𝐿1            U2        𝐿1    𝐿1             U1     𝐿2      𝐿2                                          Related work
                                                                                        explicit similarity
      𝑈1                           (𝒅)
                                          𝑒1        𝑒1    𝑒1             𝑒4     𝑒4      𝑒4                    Our work is related to network embedding which aims to
 𝒆𝟒
                            𝒆𝟑
                                          𝐴2        U2    𝐿1
                                                               …         𝐴1     U1      𝐿2
                                                                                                              learn low-dimension representations for networks. Earlier
                (𝒂)                                                                                           works, such as Local Linear Embedding (LLE) (Roweis and
                                                                               implicit similarity            Saul 2000), Laplacian eigenmaps (Belkin and Niyogi 2001)
                                                                                                              and IsoMap (Tenenbaum, De Silva, and Langford 2000), are
Figure 1: (a) An example of a hyper-network. (b) Our method. (c)                                              based on matrix factorization. They express a network as
The clique expansion. (d) The star expansion. Our method mod-                                                 a matrix where the entries represent relationships and cal-
els the hyperedge as a whole and the tuplewise similarity is pre-                                             culate the leading eigenvectors as network representations.
served. In clique expansion, each hyperedge is expanded into a                                                Eigendecomposition is a very expensive operation so these
clique. Each pair of nodes has explicit similarity. As for the star                                           methods cannot efficiently scale to large real world net-
expansion, each node in one hyperedge links to a new node which
stands for the origin hyperedge. Each pair of nodes in the origin
                                                                                                              works. Recently, DeepWalk (Perozzi, Al-Rfou, and Skiena
hyperedge has implicit similarity for the reason that they link to                                            2014) learns latent representations of nodes in a network by
the same node.                                                                                                modeling a stream of short random walks. LINE (Tang et
                                                                                                              al. 2015) optimizes an objective function which aims to pre-
                                                                                                              serve both the first-order and second-order proximities of
directly defined over all the nodes in a hyperedge, ensur-                                                    networks. HOPE (Ou et al. 2016) extends the work to utilize
ing that the subsets of a hyperedge are not incorporated in                                                   higher-order information and M-NMF (Wang et al. 2017)
network embedding. We theoretically prove that the inde-                                                      incorporates the community structure into network embed-
composable tuplewise similarity function can not be a lin-                                                    ding. Furthermore, due to the powerful representation abil-
ear function. Therefore, we realize the tuplewise similarity                                                  ity of deep learning (Niepert, Ahmed, and Kutzkov 2016),
function with a deep neural network and add a non-linear                                                      several network embedding methods based on deep learn-
activation function to make it highly non-linear. To address                                                  ing (Chang et al. 2015; Wang, Cui, and Zhu 2016) have
Structure Preserving issue, we design a deep autoencoder                                                      been proposed. (Wang, Cui, and Zhu 2016) proposes a deep
to learn node representations by reconstructing neighbor-                                                     model with a semi-supervised architecture, which simulta-
hood structures, ensuring that the nodes with similar neigh-                                                  neously optimizes the first-order and second-order proxim-
borhood structures will have similar embeddings. The tu-                                                      ity. (Chang et al. 2015) employs deep model to transfer dif-
plewise similarity function and deep autoencoder are jointly                                                  ferent objects in heterogeneous networks to unified vector
optimized to simultaneously address the two issues.                                                           representations.
   It is worthwhile to highlight the following contributions                                                     However, all of the above methods assume pairwise rela-
of this paper:                                                                                                tionships among objects in real world networks. In view of
                                                                                                              the aforementioned facts, a series of methods (Zhou, Huang,
• We investigate the problem of indecomposable hyper-
                                                                                                              and Schölkopf 2006; Liu et al. 2013; Wu, Han, and Zhuang
   network embedding, where indecomposibility of hyper-
                                                                                                              2010) is proposed by generalizing spectral clustering tech-
   edges is a common property in hyper-networks but largely
                                                                                                              niques (Ng et al. 2001) to hypergraphs. Nevertheless, these
   ignored in literature. We propose a novel deep model,
                                                                                                              methods focus on homogeneous hypergraph. They construct
   named Deep Hyper-Network Embedding (DHNE), to
                                                                                                              hyperedge by latent similarity like common label and pre-
   learn embeddings for the nodes in heterogeneous hyper-
                                                                                                              serve hyperedge implicitly. Therefore, they cannot preserve
   networks, which can simultaneously address indecompos-
                                                                                                              the structure of indecomposable hyperedges. For heteroge-
   able hyperedges while preserving rich structural informa-
                                                                                                              neous hyper-network, the tensor decomposition (Kolda and
   tion. The complexity of this method is linear to the num-
                                                                                                              Bader 2009; Rendle and Schmidt-Thieme 2010; Symeonidis
   ber of node and it can be used in large scale networks
                                                                                                              2016) may be directly applied to learn the embedding. Un-
• We theoretically prove that any linear similarity metric                                                    fortunately, the time cost of tensor decomposition is usually
   in embedding space cannot maintain the indecomposibil-                                                     very expensive so it cannot scale efficiently to large network.
   ity property in hyper-networks, and thus propose a novel                                                   Besides, HyperEdge Based Embedding (HEBE) (Gui et al.
   deep model to simultaneously maintain the indecomposi-                                                     2016) is proposed to model the proximity among partici-
   bility as well as the local and global structural information                                              pating objects in each heterogeneous event as a hyperedge
   in hyper-networks.                                                                                         based on prediction. It does not take high-order network
• We conduct experiments on four real-world information                                                       structure and high degree of sparsity into account which af-
   networks. The results demonstrate the effectiveness and                                                    fects the predictive performance in our task.
   efficiency of the proposed model.
   The remainder of this paper is organized as follows. In the                                                              Notations and Definitions
next section, we review the related work. Section 3 gives the                                                 In this section, we define the problem of hyper-network em-
preliminaries and formally defines our problem. In Section                                                    bedding. The key notations used in this paper are shown in
                           Table 1: Notations.                                                    Unsupervised Heterogeneous Component
                                                                                                Node Type 𝑎   Node Type 𝑏      Node Type 𝑐
          Symbols                              Meaning                        𝐴𝑎i , 𝐴𝑗𝑏 , 𝐴𝑐𝑘       …              …                  …

            T                             number of node types               First Layer        autoencoder    autoencoder     autoencoder   second-order
     V = {Vt }Tt=1                              node set
  E = {(v1 , v2 , ..., vni )}                 hyperedge set                   𝑋i𝑎 , 𝑋𝑗𝑏 , 𝑋𝑘𝑐      …              …                   …

            A                     adjacency matrix of hyper-network                                                                           Non-linear
                                                                            Second Layer
           Xji                     embedding of node i with type j                                                                             mapping
   S(X1 , X2 , ..., XN )            N -tuplewise similarity function              𝐿𝑖𝑗𝑘                            .……
              (i)
          Wj                    the i-th layer weight matrix with type j
                                                                             Third Layer                                       tuple-wise
             (i)                                                                                                                              first-order
          bj                        the i-th layer biases with type j                                             +1   -1
                                                                                                                                similarity
                                                                                  𝑆𝑖𝑗𝑘
                                                                                                        Supervised Binary Component


Table 1. First we give the definition of a hyper-network.                     Figure 2: Framework of Deep Hyper-Network Embedding.
Definition 1 (Hyper-network). A hyper-network is defined
as a hypergraph G = (V, E) with the set of nodes V be-
longing to T types V = {Vt }Tt=1 and the set of edges                      Loss function
E which may have more than two nodes E = {Ei =                             To preserve the first-order proximity of a hyper-network, an
(v1 , v2 , ..., vni )}(ni ≥ 2). If the number of nodes is 2 for            N -tuplewise similarity measure in embedding space is re-
each hyperedge, the hyper-network degenerates to a net-                    quired. If there exists a hyperedge among N vertexes, the
work. The type of edge Ei is defined as the combination of                 N -tuplewise similarity of these vertexes should be large, and
types of nodes belonging to the edge. If T ≥ 2, the hyper-                 small otherwise.
network is defined as a heterogeneous hyper-network.
                                                                           Property 1. We mark Xi as the embedding of node vi and
  To obtain the embeddings in a hyper-network, the inde-                   S as N -tuplewise similarity function.
composable tuplewise relationships need be preserved. We
define the indecomposable structures as the first-order prox-              • if (v1 , v2 , ..., vN ) ∈ E, S(X1 , X2 , .., XN ) should be
imity of hyper-network:                                                      large (without loss of generality, large than a threshold
                                                                             l).
Definition 2 (The First-order Proximity of Hyper-network).                 • if (v1 , v2 , ..., vN ) ∈
                                                                                                     / E, S(X1 , X2 , .., XN ) should be
The first-order proximity of hyper-network measures the                      small (without loss of generality, smaller than a thresh-
N-tuplewise similarity between nodes. For any N vertexes                     old s).
v1 , v2 , ..., vN , if there exists a hyperedge among these N
vertexes, the first-order proximity of these N vertexes is de-               In our model, we propose a data-dependent N -tuplewise
fined as 1, but this implies no first-order proximity for any              similarity function. In this paper, we mainly focus on hyper-
subsets of these N vertexes.                                               edges with uniform length N = 3, but it is easy to extend to
                                                                           N > 3.
   The first-order proximity implies the indecomposable                      Here we provide the theorem to demonstrate that a linear
similarity of several entities in real world. Meanwhile, real              tuplewise similarity function cannot satisfy Property 1.
world networks are always incomplete and sparse. Only con-
sidering first-order proximity is not sufficient for learning              Theorem
                                                                           P           1. Linear function S(X1 , X2 , ..., XN )                             =
node embeddings. Higher order proximity needs to be con-                     i W i X i cannot satisfy Property 1.
sidered to fix this issue. We then introduce the second-order              Proof. To prove it by contradiction, we assume that theo-
proximity of hyper-network to capture the global structure.                rem 1 is false, i.e., the linear function S satisfies Property 1.
Definition 3 (The Second-order Proximity of Hyper-net-                     We suggest the following counter example. Assume we have
work). The second-order Proximity of hyper-network mea-                    3 types of nodes, and each type of node has 2 clusters (0 and
                                                                           1). There is a hyperedge if and only if 3 nodes from differ-
sures the proximity of two nodes with respect to their
neighborhood structures. For any node vi ∈ Ei , Ei /vi                     ent types have the same cluster id. We use Yij to represent
is defined as a neighborhood of vi . If vi ’s neighborhoods                embeddings of nodes with type j in cluster i. By Property 1,
                                                                           we have
{Ei /vi f or any vi ∈ Ei } are similar to vj ’s, then vi ’s em-
bedding xi should be similar to vj ’s embedding xj .                                             W1 Y01 + W2 Y02 + W3 Y03 > l                          (1)
   For example, in Figure 1(a), A1 ’s neighborhoods set is                                       W1 Y11 + W2 Y02 + W3 Y03 < s                          (2)
{(L2 , U1 ), (L1 , U2 )}. A1 and A2 have second-order similar-                                   W1 Y11 + W2 Y12 + W3 Y13 > l                          (3)
ity since they have common neighborhood, (L1 , U2 ).
                                                                                                 W1 Y01 + W2 Y12 + W3 Y13 < s.                         (4)
          Deep Hyper-Network Embedding                                      By combining Equation (1)(2)(3)(4), we get W1 ∗ (Y01 −
In this section, we introduce the proposed Deep Hyper-                     Y11 ) > l − s and W1 ∗ (Y11 − Y01 ) > l − s, which is a
Network Embedding (DHNE). The framework is shown in                        contradiction.
Figure 2.                                                                    This completes the proof.
   Above theorem shows that N-tuplewise similarity func-            The goal of autoencoder is to minimize the reconstruc-
tion S should be in a non-linear form. This motivates us to      tion error between the input and the output. The autoen-
model it by a multilayer perceptron. The multilayer percep-      coder’s reconstruction process will make the nodes with
tron is composed of two parts, which are shown separately        similar neighborhoods have similar latent representations,
in the second layer and third layer of Figure 2. The second      and thus the second-order proximity is preserved. It is note-
layer is a fully connected layer with non-linear activation      worthy that the input feature is the adjacency matrix of the
functions. With the input of the embeddings (Xai , Xbj , Xck )   hyper-network, and the adjacency matrix is often extremely
of 3 nodes (vi , vj , vk ), we concatenate them and map them     sparse. To speed up our model, we only reconstruct non-zero
non-linearly to a common latent space L. Their joint repre-      element in the adjacency matrix. The reconstruction error is
sentation in latent space is shown as follows:                   shown as follows:

                            (2)                                                  ||sign(Ai )     (Ai − Âi )||2F ,           (10)
 Lijk = σ(Wa(2) ∗Xai +Wb ∗Xbj +Wc(2) ∗Xck +b(2) ), (5)
where σ is the sigmoid function.                                 where sign is the sign function.
  After obtaining the latent representation Lijk , we finally       Furthermore, in hyper-networks, the vertexes often have
map it to a probability space in the third layer to get the      various types, forming heterogeneous hyper-networks. Con-
similarity:                                                      sidering the special characteristics of different types of
                                                                 nodes, it is required to learn unique latent spaces for dif-
                                                                 ferent node types. In our model, each heterogeneous type of
   Sijk ≡ S(Xai , Xbj , Xck ) = σ(W(3) ∗ Lijk + b(3) ).   (6)    entities have their own autoencoder model as shown in Fig-
  Combining the aforementioned two layers, we obtain a           ure 2. Then for all types of nodes, the loss function is defined
non-linear tuplewise similarity measure function S. In or-       as:
der to make this similarity function satisfy Property 1, we                        X
present the objective function as follows:                                  L2 =        ||sign(Ati )   (Ati − Âti )||2F ,   (11)
                                                                                    t

  L1 = −(Rijk log Sijk + (1 − Rijk ) log(1 − Sijk )), (7)        where t is the index for node types.
where Rijk is defined as 1 if there is a hyperedge between         To preserve both first-order proximity and second-order
vi , vj and vk and 0 otherwise. From the objective function,     proximity of heterogeneous hyper-networks, we jointly min-
it is easy to check that if Rijk equals to 1, the similarity     imize the objective function by combining Equation 7 and
Sijk should be large, and otherwise the similarity should be     Equation 11:
small. In other words, the first-order proximity is preserved.
    Next, we consider to preserve the second-order proxim-                               L = L1 + αL2 .                      (12)
ity. The first layer of Figure 2 is designed to preserve the
second-order proximity. Second-order proximity measures          Optimization
neighborhood structure similarity. Here, we define the adja-
cency matrix of hyper-network to capture the neighborhood        We use stochastic gradient descent (SGD) to optimize the
structure. First, we give some basic definitions of hyper-       model. The key step is to calculate the partial deriva-
graph. For a hypergraph G = (V, E), a |V| ∗ |E| incidence        tive of the parameters θ = {W(i) , b(i) , Ŵ(i) , b̂(i) }3i=1 .
matrix H with entries h(v, e) = 1 if v ∈ e and 0 otherwise,      These derivatives can be easily estimated by using back-
is defined to represent the hypergraph. ForPa vertex v ∈ V,      propagation algorithm (LeCun, Bengio, and Hinton 2015).
the degree of vertex is defined by d(v) = e∈E h(v, e). Let       Notice that there is only positive relationship in most real
Dv denote the diagonal matrix containing the vertex degree.      world network, so this algorithm may converge to a trivial
Then the adjacency matrix A of hypergraph G can be de-           solution where all tuplewise relationships are similar. To ad-
fined as A = HHT − Dv , where HT is the transpose of             dress this problem, we sample multiple negative edges based
H. The entries of adjacency matrix A denote the concurrent       on noisy distribution for each edge, as in (Mikolov et al.
times between two nodes, and the i-th row of adjacency ma-       2013). The whole algorithm is shown in Algorithm 1.
trix A shows the neighborhood structure of vertex vi . We
use an adjacency matrix A as our input feature and an au-        Analysis
toencoder (LeCun, Bengio, and Hinton 2015) as the model
to preserve the neighborhood structure. The autoencoder is       In this section, we present the out-of-sample extension and
composed by an encoder and a decoder. The encoder is a           complexity analysis.
non-linear mapping from feature space A to latent represen-      Out-of-sample extension For a newly arrived vertex v, we
tation space X and the decoder is a non-linear mapping from      can easily obtain its adjacency vector by its connections to
latent representation X space back to origin feature space Â,   existing vertexes. We feed its adjacency vector into the spe-
which is shown as follows:                                       cific autoencoder corresponding to its type, and apply Equa-
                Xi = σ(W(1) ∗ Ai + b(1) )                 (8)    tion 8 to get representation for vertex v. The complexity for
                             (1)           (1)
                                                                 such steps is O(dv d), where dv is the degree of vertex v and
                Âi = σ(Ŵ         ∗ Xi + b̂     ).       (9)    d is the dimensionality of the embedding space..
Algorithm 1 The Deep Hyper-Network Embedding (DHNE)                              Table 2: Statistics of the datasets.
Require: the hyper-network G = (V, E) with adjacency                datasets          node type                      #(V)            #(E)
    matrix A, the parameter α                                        GPS       user   location    activity     146    70       5     1436
Ensure: Hyper-network Embeddings E and updated Pa-                 MovieLens   user    movie         tag      2113   5908    9079   47957
                                                                     drug      user     drug      reaction     12    1076    6398   171756
    rameters θ = {W(i) , b(i) , Ŵ(i) , b̂(i) }3i=1                 wordnet    head   relation      tail     40504    18    40551   145966
 1: initial parameters θ by random process
 2: while the value of objective function do not converge
    do
 3:    generate next batch from the hyperedge set E               • wordnet (Bordes et al. 2013): This dataset consists of a
 4:    sample negative hyperedge randomly                           collection of triplets (synset, relation type, synset) ex-
 5:    calculate partial derivative ∂L/∂θ with back-                tracted from WordNet 3.0. We can construct the hyper-
       propagation algorithm to update θ.                           network by regarding head entity, relation, tail entity as
 6: end while
                                                                    three types of nodes and the triplet relationships as hyper-
                                                                    edges.
                                                                  The detailed statistics of the datasets are summarized in Ta-
Complexity analysis During the training procedure, the            ble 2.
time complexity of calculating gradients and updating pa-
rameters is O((nd + dl + l)bI), where n is the number of          Parameter Settings
nodes, d is the dimension of embedding vectors, l is the size     We compared DHNE against the following six widely-used
of latent layer, b is the batch size and I is the number of it-   algorithms: DeepWalk (Perozzi, Al-Rfou, and Skiena 2014),
erations. Parameter l is usually related to the dimension of      LINE (Tang et al. 2015), node2vec (Grover and Leskovec
embedding vectors d but independent with the number of            2016), Spectral Hypergraph Embedding (SHE) (Zhou,
vertexes n. The batch size is normally a small number. The        Huang, and Schölkopf 2006), Tensor decomposition (Kolda
number of iterations is also not related with the number of       and Bader 2009) and HyperEdge Based Embedding
vertexes n. Therefore, the complexity of training procedure       (HEBE) (Gui et al. 2016).
is linear to the number of vertexes.                                 In summary, DeepWalk, LINE and node2vec are conven-
                                                                  tional pairwise network embedding methods. In our experi-
                         Experiment                               ment, we use clique expansion in Figure 1(c) to transform a
                                                                  hyper-network in a conventional network, and then use these
In this section, we evaluate our proposed method on several       three methods to learn node embeddings from the conven-
real world datasets and multiple application scenarios.           tional networks. SHE is designed for homogeneous hyper-
                                                                  network embeddings. Tensor method is a direct way for
Datasets                                                          preserving high-order relationship in heterogeneous hyper-
In order to comprehensively evaluate the effectiveness of our     network. HEBE learns node embeddings for heterogeneous
proposed method, we use four different types of datasets,         event data. Note that DeepWalk, LINE , node2vec and SHE
including a GPS network, a social network, a medicine net-        can only measure pairwise relationship. In order to make
work and a semantic network. The detailed information is          them applicable to network reconstruction and link predic-
shown as follows.                                                 tion in hyper-networks, without loss of generality, we use
                                                                  the mean or minimum value among all pairwise similarities
• GPS (Zheng et al. 2010): The dataset describes a user           in a candidate hyperedge to represent the tuplewise similar-
  joins in an activity in certain location. The (user, loca-      ity of the hyperedge. For DeepWalk and node2vec, we set
  tion, activity) relations are used for building the hyper-      window size as 10, walk length as 40, walks per vertex as
  network.                                                        10. For LINE, we set the number of negative samples as 5.
• MovieLens (Harper and Konstan 2016): This dataset de-              We uniformly set the representation size as 64 for all
  scribes personal tagging activity from MovieLens1 . Each        methods. Specifically, for DeepWalk and node2vec, we set
  movie is labeled by at least one genres. The (user, movie,      window size as 10, walk length as 40, walks per vertex as
  tag) relations are considered as the hyperedges to form a       10. For LINE, we set the number of negative samples as 5.
  hyper-network.                                                     For our model, we use one-layer autoencoder to preserve
                                                                  hyper-network structure and one-layer fully connected layer
• drug2 : This dataset is obtained from FDA Adverse Event         to learn tuplewise similarity function. The size of hidden
  Reporting System (FAERS). It contains information on            layer of autoencoder is set as 64 which is also the repre-
  adverse event and medication error reports submitted to         sentation size. The size of fully connect layer is set as sum
  FDA. We construct hyper-network by (user, drug, reac-           of the embedding length from all types, 192. We do grid
  tion) relationships, i.e., a user who has certain reaction      search from {0.01, 0.1, 1, 2, 5, 10} to tune the parameter α
  and takes some drugs will lead to adverse event.                which is shown in Parameter Sensitivity section. Similar to
                                                                  LINE (Tang et al. 2015), the learning rate is set with the
   1
       https://movielens.org/                                     starting value ρ0 = 0.025 and decreased linearly with the
   2
       http://www.fda.gov/Drugs/                                  times of iterations.
         Table 3: AUC value for network reconstruction.                                   DHNE                                       line(mean)                node2vec(mean)              SHE(mean)           tensor
                                                                                          deepwalk(mean)                             line(min)                 node2vec(min)               SHE(min)            HEBE
                                                                                          deepwalk(min)
                                                                                          1.0
        methods        GPS     MovieLens      drug    wordnet                             0.9
                                                                                                                                                                          0.9



         DHNE         0.9598     0.9344      0.9356   0.9073                              0.8




                                                                     True Positive Rate
                                                                                          0.7                                                                             0.8

           deepwalk   0.6714     0.8233      0.5750   0.8176                              0.6



  mean
             line     0.8058     0.8431      0.6908   0.8365                              0.5
                                                                                                                                                                    AUC   0.7


           node2vec   0.6715     0.9142      0.6694   0.8609                              0.4


             SHE      0.8596     0.7530      0.5486   0.5618                              0.3                                                                             0.6

                                                                                          0.2

           deepwalk   0.6034     0.7117      0.5321   0.7423                              0.1
                                                                                                                                                                          0.5

  min        line     0.7369     0.7910      0.7625   0.7751                              0.0
                                                                                                0.0   0.1   0.2    0.3   0.4   0.5    0.6   0.7   0.8   0.9   1.0           0.0   0.2    0.4     0.6     0.8       1.0

           node2vec   0.6578     0.9100      0.6557   0.8387                                                      False Positive Rate                                                   Percentage
             SHE      0.7981     0.7972      0.6236   0.5918
         tensor       0.9229     0.8640      0.7025   0.7771     Figure 3: left: ROC curve on GPS; right: Performance for link
         HEBE         0.9337     0.8772      0.8236   0.7391     prediction on networks of different sparsity.


                                                                                                             Table 4: AUC value for link prediction.
Network Reconstruction
A good network embedding method should preserve the                                         methods                                         GPS               MovieLens                  drug          wordnet
original network structure well in the embedding space.                                         DHNE                                  0.9166                        0.8676              0.9254         0.8268
We first evaluate our proposed algorithm on network re-                                                deepwalk                       0.6593                        0.7151              0.5822         0.5952
construction task. We use the learned embeddings to pre-                                                 line                         0.7795                        0.7170              0.7057         0.6819
dict the links of origin networks. The AUC (Area Under the         mean
                                                                                                       node2vec                       0.5835                        0.8211              0.6573         0.8003
Curve) (Fawcett 2006) is used as the evaluation metric. The                                              SHE                          0.8687                        0.7459              0.5899         0.5426
results are show in Table 3.                                                                           deepwalk                       0.5715                        0.6307              0.5493         0.5542
   From the results, we have the following observations:                                                 line                         0.7219                        0.6265              0.7651         0.6225
                                                                    min
• Our method achieves significant improvements on AUC                                                  node2vec                       0.5869                        0.7675              0.6546         0.7985
  values over the baselines on all four datasets. It demon-                                              SHE                          0.8078                        0.8012              0.6508         0.5507
  strates that our method is able to preserve the origin net-                                    tensor                               0.8646                        0.7201              0.6470         0.6516
  work structure well.                                                                           HEBE                                 0.8355                        0.7740              0.8191         0.6364
• Compared with the baselines, our method achieves higher
  improvements in sparse drug and wordnet datasets than
  those on GPS and MovieLens datasets. It indicates the ro-      • Our method achieves significant improvements over the
  bustness of proposed methods on sparse datasets.                 baselines on all the datasets. It demonstrates the learned
• The results of DHNE perform better than DeepWalk,                embeddings of our method have strong predictive power
  LINE and SHE which assume that high-order rela-                  for unseen links.
  tionships are decomposable. It demonstrates the impor-         • By comparing the performance of LINE, DeepWalk, SHE
  tance of preserving indecomposable hyperedge in hyper-           and DHNE, we can observe that transforming the in-
  network.                                                         decomposable high-order relationship into multiple pair-
                                                                   wise relationship will damage the predictive power of the
Link Prediction                                                    learned embeddings.
Link prediction is a widely-used application in real world
especially in recommendation systems. In this section, we        • As Tensor and HEBE can somewhat address the indecom-
fulfil two link prediction tasks on all the four datasets. The     posibility of hyperedges, the large improvement margin of
two tasks evaluate the overall performance and the perfor-         our method over these two methods clearly demonstrates
mance with different sparsity of the networks, respectively.       the importance of second-order proximities in hyper-
We calculate AUC value as in network reconstruction task           network embedding.
to evaluate the performance.
   For the first task, we randomly hide 20 percentage of ex-        For the second task, we change the sparsity of network by
isting edges and use the left network to train hyper-network     randomly hiding different ratios of existing edges and repeat
embedding models. After training, we obtain embedding for        the previous task. Particularly, we conduct this task on the
each node and similarity function for N nodes and apply the      drug dataset as it has the most hyperedges. The ratio of re-
similarity function to predict the held-out links. For GPS       mained edges is selected from 10% to 90%. The results are
dataset which is a small and dense dataset, we can draw          shown in Figure 3 right.
ROC curve for this task to observe the performance at vari-         We can observe that DHNE has a significant improve-
ous threshold settings, as shown in Figure 3 left. The AUC       ments over the best baselines on all sparsity of networks.
results on all datasets are shown in Table 4. The observations   It demonstrates the effectiveness of DHNE on sparse net-
are illustrated as follows:                                      works.
                             DHNE                   line                               SHE               HEBE                   0.95
                                                                                                                                                                     0.94
                                                                                                                                                                                                                         1.4


                                                                                                                                0.94                                                                                     1.2
                             deepwalk               node2vec                           tensor                                                                        0.92
                                                                                                                                0.93                                                                                     1.0

            0.48                                                      0.26                                                                                           0.90




                                                                                                                                                                                                               time(s)
                                                                                                                                0.92

                                                                                                                          AUC                                  AUC
                                                                                                                                                                                                                         0.8
                                                                                                                                                                     0.88
            0.47                                                      0.24                                                      0.91                                                                                     0.6
                                                                                                                                                                     0.86
                                                                                                                                0.90                                                                                     0.4
            0.46                                                      0.22
                                                                                                                                                                     0.84
                                                                                                                                0.89                                                                                     0.2




 Micro-F1                                                  Macro-F1
            0.45                                                      0.20                                                                                           0.82
                                                                                                                                0.88                                                                                     0.0
                                                                                                                                       0   1   2   3   4   5                 20   40     60   80   100   120                   0     20000   40000   60000   80000
            0.44                                                      0.18                                                                     alpha                                   dimension                                   number of nodes(*104)

            0.43                                                      0.16

            0.42                                                      0.14
                                                                                                                         Figure 5: left, middle: Parameter w.r.t. embedding dimensions d,
            0.41                                                      0.12
                                                                                                                         the value of α; right: training time per batch w.r.t. embedding di-
            0.40                                                      0.10
                0.0    0.2    0.4     0.6     0.8      1.0                0.0    0.2     0.4      0.6     0.8      1.0   mensions.
            0.30                                                      0.14

            0.28
                                                                      0.12
            0.26

            0.24                                                      0.10
                                                                                                                         when the number of embedding dimension increases. This is
 Micro-F1                                                  Macro-F1
            0.22

            0.20
                                                                      0.08
                                                                                                                         reasonable because higher embedding dimensions can em-
            0.18
                                                                      0.06
                                                                                                                         body more information of a hyper-network. After the em-
            0.16                                                      0.04
            0.14
                                                                                                                         bedding dimension is larger than 32, the curve is relatively
                                                                      0.02
            0.12                                                                                                         stable, demonstrating that our algorithm is not very sensitive
            0.10                                                      0.00
               0.00   0.02   0.04   0.06    0.08    0.10                 0.00   0.02    0.04    0.06    0.08    0.10     to embedding dimension.
                             Percentage                                                Percentage
                                                                                                                         Effect of parameter α The parameter α measures the
Figure 4: top: multi-label classification on MovieLens dataset;                                                          trade-off of the first-order proximity and second-order prox-
bottom: multi-class classification on wordnet dataset.                                                                   imity of a hyper-network. We show how the values of α af-
                                                                                                                         fect the performance in Figure 5 middle. When α equals 0,
                                                                                                                         only the first-order proximity is taken into account in our
Classification                                                                                                           method. The performance with α between 0.1 and 2 is bet-
In this section, we conduct multi-label classification (Bha-                                                             ter than that of α = 0, demonstrating the importance of
tia et al. 2015) on MovieLens dataset and multi-class clas-                                                              second-order proximity. The fact that the performance with
sification in wordnet, because only these two datasets have                                                              α between 0.1 and 2 is better than that of α = 5 can demon-
label or category information. After deriving the node em-                                                               strate the importance of the first-order proximity. In sum-
beddings from different methods, we choose SVM as the                                                                    mary, both the first-order proximity and the second-order
classifier. For MovieLens dataset, we randomly sample 10%                                                                proximity are necessary for hyper-network embedding.
to 90% of the vertexes as the training samples and use the                                                               Training time analysis To testify the scalability, we test
left vertexes to test the performance. For wordnet dataset, the                                                          training time per batch, as shown in Figure 5 right. We can
portion of training data is selected from 1% to 10%. Besides,                                                            observe that the training time scales linearly with the num-
we remove the nodes without labels on these two datasets.                                                                ber of nodes. This results conform to the above complexity
Averaged Macro-F1 and Micro-F1 are used to evaluate the                                                                  analysis and indicate the scalability of our model.
performance. The results are shown in Figure 4.
   From the results, we have following observations:                                                                                                                        Conclusion
• In both Micro-F1 and Macro-F1 curves, our method per-                                                                  In this paper, we propose a novel deep model named DHNE
  forms consistently better than baselines. It demonstrates                                                              to learn the low-dimensional representation for hyper-
  the effectiveness of our proposed method in classification                                                             networks with indecomposible hyperedges. More specifi-
  tasks.                                                                                                                 cally, we theoretically prove that any linear similarity met-
• When the labelled data becomes richer, the relative im-                                                                ric in embedding space commonly used in existing methods
  provement of our method is more obvious than baselines.                                                                cannot maintain the indecomposibility property in hyper-
  Besides, as shown in Figure 4 bottom, when the labeled                                                                 networks, and thus propose a new deep model to realize
  data is quite sparse, our method still outperforms the base-                                                           a non-linear tuplewise similarity function while preserving
  lines. This demonstrates the robustness of our method.                                                                 both local and global proximities in the formed embedding
                                                                                                                         space. We conduct extensive experiments on four different
Parameter Sensitivity                                                                                                    types of hyper-networks, including a GPS network, an on-
                                                                                                                         line social network, a drug network and a semantic network.
In this section, we investigate how parameters influence the                                                             The empirical results demonstrate that our method can sig-
performance and training time. Especially, we evaluate the                                                               nificantly and consistently outperform the state-of-the-art al-
effect of the ratio of first-order proximity loss and second-                                                            gorithms.
order proximity loss α and the embedding dimension d. For
brevity, we report the results by link prediction task with                                                                                                Acknowledgments
drug dataset.
                                                                                                                         This work is supported by National Program on Key Basic
Effect of embedding dimension We show how the di-                                                                        Research Project No. 2015CB352300, National Natural Sci-
mension of embedding space affects the performance in Fig-                                                               ence Foundation of China Major Project No. U1611461; Na-
ure 5 left. We can see that the performance raises firstly                                                               tional Natural Science Foundation of China No. 61772304,
 61521002, 61531006, 61702296; NSF IIS-1650723 and IIS-            retrieval with heterogeneous social contexts. Neurocomput-
 1716432. Thanks for the research fund of Tsinghua-Tencent         ing 119:49–58.
 Joint Laboratory for Internet Innovation Technology, and the     [Mikolov et al. 2013] Mikolov, T.; Sutskever, I.; Chen, K.;
 Young Elite Scientist Sponsorship Program by CAST. Peng           Corrado, G. S.; and Dean, J. 2013. Distributed represen-
 Cui, Xiao Wang and Wenwu Zhu are the corresponding au-            tations of words and phrases and their compositionality. In
 thors. All opinions, findings, conclusions and recommenda-        Advances in neural information processing systems, 3111–
 tions in this paper are those of the authors and do not neces-    3119.
 sarily reflect the views of the funding agencies.
                                                                  [Ng et al. 2001] Ng, A. Y.; Jordan, M. I.; Weiss, Y.; et al.
                        References                                 2001. On spectral clustering: Analysis and an algorithm.
                                                                   In NIPS, volume 14, 849–856.
[Agarwal, Branson, and Belongie 2006] Agarwal, S.; Bran-
 son, K.; and Belongie, S. 2006. Higher order learning with       [Niepert, Ahmed, and Kutzkov 2016] Niepert, M.; Ahmed,
 graphs. In Proceedings of the 23rd international conference       M.; and Kutzkov, K. 2016. Learning convolutional neu-
 on Machine learning, 17–24. ACM.                                  ral networks for graphs. In Proceedings of the 33rd annual
                                                                   international conference on machine learning. ACM.
[Belkin and Niyogi 2001] Belkin, M., and Niyogi, P. 2001.
 Laplacian eigenmaps and spectral techniques for embedding        [Ou et al. 2015] Ou, M.; Cui, P.; Wang, F.; Wang, J.; and
 and clustering. In NIPS, volume 14, 585–591.                      Zhu, W. 2015. Non-transitive hashing with latent similarity
                                                                   components. In Proceedings of the 21th ACM SIGKDD In-
[Bhatia et al. 2015] Bhatia, K.; Jain, H.; Kar, P.; Varma, M.;
                                                                   ternational Conference on Knowledge Discovery and Data
 and Jain, P. 2015. Sparse local embeddings for extreme
                                                                   Mining, 895–904. ACM.
 multi-label classification. In Advances in Neural Informa-
 tion Processing Systems, 730–738.                                [Ou et al. 2016] Ou, M.; Cui, P.; Pei, J.; Zhang, Z.; and Zhu,
[Bordes et al. 2013] Bordes, A.; Usunier, N.; Garcia-Duran,        W. 2016. Asymmetric transitivity preserving graph embed-
 A.; Weston, J.; and Yakhnenko, O. 2013. Translating em-           ding. In KDD, 1105–1114.
 beddings for modeling multi-relational data. In Advances in      [Perozzi, Al-Rfou, and Skiena 2014] Perozzi, B.; Al-Rfou,
 neural information processing systems, 2787–2795.                 R.; and Skiena, S. 2014. Deepwalk: Online learning of
[Chang et al. 2015] Chang, S.; Han, W.; Tang, J.; Qi, G.-J.;       social representations. In Proceedings of the 20th ACM
 Aggarwal, C. C.; and Huang, T. S. 2015. Heterogeneous net-        SIGKDD international conference on Knowledge discovery
 work embedding via deep architectures. In Proceedings of          and data mining, 701–710. ACM.
 the 21th ACM SIGKDD International Conference on Knowl-           [Rendle and Schmidt-Thieme 2010] Rendle,           S.,     and
 edge Discovery and Data Mining, 119–128. ACM.                     Schmidt-Thieme, L. 2010. Pairwise interaction tensor
[Deng et al. 2016] Deng, C.; Tang, X.; Yan, J.; Liu, W.; and       factorization for personalized tag recommendation. In
 Gao, X. 2016. Discriminative dictionary learning with com-        Proceedings of the third ACM international conference on
 mon label alignment for cross-modal retrieval. IEEE Trans-        Web search and data mining, 81–90. ACM.
 actions on Multimedia 18(2):208–218.                             [Roweis and Saul 2000] Roweis, S. T., and Saul, L. K. 2000.
[Fawcett 2006] Fawcett, T. 2006. An introduction to roc            Nonlinear dimensionality reduction by locally linear embed-
 analysis. Pattern recognition letters 27(8):861–874.              ding. science 290(5500):2323–2326.
[Grover and Leskovec 2016] Grover, A., and Leskovec, J.           [Sun, Ji, and Ye 2008] Sun, L.; Ji, S.; and Ye, J. 2008. Hy-
 2016. node2vec: Scalable feature learning for networks. In        pergraph spectral learning for multi-label classification. In
 Proceedings of the 22nd ACM SIGKDD international con-             Proceedings of the 14th ACM SIGKDD international con-
 ference on Knowledge discovery and data mining, 855–864.          ference on Knowledge discovery and data mining, 668–676.
 ACM.                                                              ACM.
[Gui et al. 2016] Gui, H.; Liu, J.; Tao, F.; Jiang, M.; Norick,   [Symeonidis 2016] Symeonidis, P. 2016. Matrix and tensor
 B.; and Han, J. 2016. Large-scale embedding learning in           decomposition in recommender systems. In Proceedings of
 heterogeneous event data. In ICDM.                                the 10th ACM Conference on Recommender Systems, 429–
[Harper and Konstan 2016] Harper, F. M., and Konstan, J. A.        430. ACM.
 2016.       The movielens datasets: History and context.         [Tang et al. 2015] Tang, J.; Qu, M.; Wang, M.; Zhang, M.;
 ACM Transactions on Interactive Intelligent Systems (TiiS)        Yan, J.; and Mei, Q. 2015. Line: Large-scale information
 5(4):19.                                                          network embedding. In Proceedings of the 24th Interna-
[Kolda and Bader 2009] Kolda, T. G., and Bader, B. W.              tional Conference on World Wide Web, 1067–1077. ACM.
 2009. Tensor decompositions and applications. SIAM re-           [Tenenbaum, De Silva, and Langford 2000] Tenenbaum,
 view 51(3):455–500.                                               J. B.; De Silva, V.; and Langford, J. C. 2000. A global ge-
[LeCun, Bengio, and Hinton 2015] LeCun, Y.; Bengio, Y.;            ometric framework for nonlinear dimensionality reduction.
 and Hinton, G.         2015.      Deep learning.       Nature     science 290(5500):2319–2323.
 521(7553):436–444.                                               [Wang et al. 2017] Wang, X.; Cui, P.; Wang, J.; Pei, J.; Zhu,
[Liu et al. 2013] Liu, Y.; Shao, J.; Xiao, J.; Wu, F.; and         W.; and Yang, S. 2017. Community preserving network
 Zhuang, Y. 2013. Hypergraph spectral hashing for image            embedding.
[Wang, Cui, and Zhu 2016] Wang, D.; Cui, P.; and Zhu, W.
 2016. Structural deep network embedding. In Proceed-
 ings of the 22nd ACM SIGKDD International Conference on
 Knowledge Discovery and Data Mining, 1225–1234. ACM.
[Wu, Han, and Zhuang 2010] Wu, F.; Han, Y.-H.; and
 Zhuang, Y.-T. 2010. Multiple hypergraph clustering of web
 images by mining word2image correlations. Journal of
 Computer Science and Technology 25(4):750–760.
[Zheng et al. 2010] Zheng, V. W.; Cao, B.; Zheng, Y.; Xie,
 X.; and Yang, Q. 2010. Collaborative filtering meets mo-
 bile recommendation: A user-centered approach. In AAAI,
 volume 10, 236–241.
[Zhou, Huang, and Schölkopf 2006] Zhou, D.; Huang, J.;
 and Schölkopf, B. 2006. Learning with hypergraphs: Clus-
 tering, classification, and embedding. In NIPS, volume 19,
 1633–1640.

