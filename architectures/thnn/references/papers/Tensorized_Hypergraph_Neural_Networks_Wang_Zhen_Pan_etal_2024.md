# Tensorized Hypergraph Neural Networks Wang Zhen Pan etal 2024

> Source: `Tensorized_Hypergraph_Neural_Networks_Wang_Zhen_Pan_etal_2024.pdf`

---

                                                                  Tensorized Hypergraph Neural Networks
                                               Maolin Wang †‡           Yaoming Zhen†            Yu Pan§          Yao Zhao‡              Chenyi Zhuang‡
                                                                Zenglin Xu§ △             Ruocheng Guo⋄               Xiangyu Zhao†⋆


                                         Abstract                                                     have pairwise interaction. However, in many real-world
                                         Hypergraph neural networks (HGNN) have recently              applications, the interactions among objects can go be-




arXiv:2306.02560v2 [cs.AI] 10 Jan 2024
                                         become attractive and received significant attention         yond pairwise interactions and involve higher-order re-
                                         due to their excellent performance in various do-            lationships. For example, in brain connectivity net-
                                         mains. However, most existing HGNNs rely on first-           works [20], multiple brain regions often work together in
                                         order approximations of hypergraph connectivity pat-         a neurological manner to accomplish certain functional
                                         terns, which ignores important high-order information.       tasks. To faithfully characterize such connections, pair-
                                         To address this issue, we propose a novel adjacency-         wise modeling in graph structure is inadequate, and it is
                                         tensor-based Tensorized Hypergraph Neural Network            necessary to incorporate high-order interacting informa-
                                         (THNN). THNN is a faithful hypergraph modeling               tion across brain regions. To articulate the correlation
                                         framework through high-order outer product feature           among multiple regions, hypergraph structures [1,10,20]
                                         message passing and is a natural tensor extension of         can be created with each vertex as a brain region and
                                         the adjacency-matrix-based graph neural networks. The        hyperedges representing the interactions among regions.
                                         proposed THNN is equivalent to a high-order polyno-               As discussed in [27], there is a clear difference be-
                                         mial regression scheme, which enables THNN with the          tween the pairwise relationship and the high order re-
                                         ability to efficiently extract high-order information from   lationship of multiple objects. The capacity of graph
                                         uniform hypergraphs. Moreover, in consideration of the       structures is limited as they can only describe pair-
                                         exponential complexity of directly processing high-order     wise relationships. Compared to graphs, a hypergraph
                                         outer product features, we propose using a partially         provides significant advantages in modeling the high-
                                         symmetric CP decomposition approach to reduce model          order relationships among multiple objects in real-world
                                         complexity to a linear degree. Additionally, we propose      data [27]. For example, in the case of multi-agent trajec-
                                         two simple yet effective extensions of our method for        tory prediction [34], adopting the multiscale hypergraph
                                         non-uniform hypergraphs commonly found in real-world         can extract the interactions among groups of varying
                                         applications. Results from experiments on two widely         sizes and performs much better than prior graph-based
                                         used hypergraph datasets for 3-D visual object classifi-     methods which can solely describe pairwise interactions.
                                         cation show the model’s promising performance.               Hypergraphs have also recently been used in a variety of
                                              Keywords: Hypergraph, graph neural networks,            other data mining tasks such as classification [7], com-
                                         tensorial neural networks, tensor decomposition              munity detection [38], and item matching [18].
                                                                                                           In these applications, the majority of hypergraph
                                         1   Introduction                                             neural network architectures are based on the Cheby-
                                                                                                      shev formula for hypergraph Laplacians proposed by
                                         The rapid development of graph neural networks
                                                                                                      HGNN [7]. These neural hypergraph operators can be
                                         (GNNs, [15, 17, 30]) greatly benefits various crucial re-
                                                                                                      seen as constructing a weighted graph and thus can uti-
                                         search areas due to their extraordinary performance.
                                                                                                      lize the off-the-shelf graph learning models (e.g., GCN).
                                         Generally, a conventional GNN only allows objects to
                                                                                                      However, most of these methods are incapable
                                                                                                      of learning higher-order information since they
                                            † City University of Hong Kong                            only make use of the first-order approximation,
                                         {morin.wang@my., yzhen8-c@my., xianzhao@}cityu.edu.hk        e.g., the clique expansion [10] of a hypergraph. In order
                                            ‡ Antgroup
                                                                                                      to better characterize high-order information in hyper-
                                         {nanxiao.zy, chenyi.zcy}@antgroup.com
                                            § Harbin Institute of Technology Shenzhen                 graphs, it is natural to model the high-order interaction
                                         {iperryuu, Zenglin}@gmail.com                                information of a hypergraph through high-order repre-
                                           △ Pengcheng Laboratory                                     sentation (like outer product).
                                            ⋄ ByteDance Research rguo.asu@gmail.com
                                            ⋆ Xiangyu Zhao is the corresponding author                     Some studies [14,38] have revealed the great success


                                                                                                                                          Copyright © 2024 by SIAM
                                                                                                                  Unauthorized reproduction of this article is prohibited
of tensor representation (like adjacency tensor [38]) in       represented by its adjacency matrix A ∈ {0, 1}|V |×|V | ,
hypergraph modeling. However, a general tensor                 where | · | denotes the set cardinality. The matrix A en-
based hypergraph neural network, which con-                    tries indicate whether two vertices in the graph are ad-
ducts a high-order information message passing                 jacent. More specifically, Ai,j = 1 if {vi , vj } ∈ E and 0
procedure, has not yet been developed until now.               otherwise. A hypergraph G = (V, E) is a generalization
    Motivated by the two aforementioned observations,          of a graph in which any number of vertices can be joined
in this work, we propose a tensor based Tensorized             in one edge. V is the set of vertices, and E is a set of ver-
Hypergraph Neural Network (THNN) to extend the                 tex sets, a.k.a hyperedges. A hypergraph (undirected)
adjacency matrix based graph neural networks into an           is always described by an incidence H ∈ {0, 1}|V|×|E| ,
adjacency tensor based framework. THNN has high ex-            where |V| is the number of vertices and |E| is the num-
pressiveness due to its intrinsic similarity to a high-order   ber of hyperedges. Specifically, Hi,j = 1, if vi ∈ ej and
outer product feature aggregation scheme [12, 19, 37],         Hi,j = 0 if vi ∈/ ej .
which can capture intra-feature and inter-feature dy-
namics in multilinear interaction information modeling.
In other words, the intrinsic multilinear mathematical
architecture of THNN is effective and natural in model-
ing high-order information, resulting in a more accurate
extraction of high-order interactions.
    Furthermore, because adjacency tensors can only be
used to represent uniform hypergraphs, the straightfor-
ward THNN is incapable of handling the widely existing
non-uniform hypergraphs. Therefore, we propose two                        3-Uniform
                                                                                                    Adjacency Tensor
novel solutions: (1) adding a global node and (2) multi-                 Hypergraph
uniform processing. To evaluate the performance of the
proposed THNN framework, experiments on two 3-D vi-            Figure 1: An example of adjacency tensor of a 3-uniform
sual object recognition datasets are performed. The ex-        hypergraph. In this example, the adjacency tensor of
perimental results show that the proposed THNN model           hypergraph G is defined as the 3-order tensor A ∈
achieves state-of-the-art performance. In summary, our         {0, 1}7×7×7 with the entry Avi ,vj ,vk = 1 if {vi , vj , vk } ∈
major contributions are as follows:                            E and 0 otherwise.

                                                                    Hypergraphs can be approximated by graphs via
• We propose, to the best of our knowledge, the first          its clique expansion [10]. The clique expansion ap-
  hypergraph neural network based on adjacency tensor          proximates the original hypergraph G = (V, E) via a
  that comes with a message passing mechanism cap-             graph Gclique = (V, Eclique ), which reduces each hy-
  turing high-order interactions in hypergraphs. Previ-        peredge e ∈ E into a clique in Gclique . However, the
  ous hypergraph neural networks are mostly based on           clique expansion will lead to information loss [36]. The
  the first-order approximation.                               original hypergraph can not be recovered according to
• Given the fact that the naive outer product based            the adjacency matrix of clique expansion, as the hyper-
  model of high-order information suffers from expo-           dependency and high order relationship collapses into
  nential time/space complexity, we propose to uti-            linearity [36].
  lize partially symmetric CP decomposition to reduce               Another important way to represent hypergraphs is
  time/space complexity from exponential to linear.            by Adjacency Tensor [38]. As shown in Figure 1, ad-
• To handle non-uniform hypergraphs, we propose two            jacency tensor can represent uniform hypergraph where
  simple yet effective solutions, i.e., adding a global        all hyperedges share the same size. The m-uniform hy-
  node and multi-uniform processing, to overcome the           pergraph means that the sizes of all hyperedges are m.
  limitation that straightforward THNN of adjacency            For an m-uniform hypergraph, the adjacency tensor is
  tensor methods can only be used to model and process         defined as the m-order tensor A ∈ {0, 1}n×...×n with the
  uniform hypergraphs.                                         entry Ai1 ,...,im = 1 if {vi1 , . . . , vim } ∈ E and 0 otherwise.

                                                        2.2 Tensor Contraction Tensor contraction [21,31]
2   Preliminaries and Background                        means that two tensors are contracted into one tensor
2.1 Graph and Hypergraph A graph can be de- along their associated pairs of indices. Given two
noted by G = (V, E), where V is the set of vertices and tensors A ∈ RI1 ×I2 ×···×IN and B ∈ RJ1 ×J2 ×···×JM ,
E is a set of paired vertices, or edges. A graph can be with some common modes, In1 = Jm1 , · · · InS =


                                                                                                    Copyright © 2024 by SIAM
                                                                            Unauthorized reproduction of this article is prohibited
                                                              graph that all the edges in a same clique sharing the
                                                              same learnable weight to approximate the hypergraph.
                                                              Each hyperedge of size s is approximated by a weighted
           1                       =                          s-clique. By analyzing the computation of Eq. (3.1), we
       1       1
                                                              denote the embedding of node vi in the l + 1-th layer by
                                                                (l+1)
                                                              xvi . The computation can be demonstrated as the
                                                              following aggregation function form,
                                                              (3.2)
Figure 2: Illustration of concatenating 1 and high-order
                                                                                                                          
                                                                (l+1)       1      X        1         X        1   l   (l)
fusion [37]. Every circle corresponds to an element in a      x vk = σ p                       Wjj            p x vi Θ       .
                                                                            dvk ej ,vk ∈ej dej     vi ,vi ∈ej
                                                                                                               dvi
vector or tensor and ◦ indicates the outer product. We
can concatenate a 1 to each vector, and then the outer            This approach tackles information aggregation by
product of vectors will introduce lower order dynamics. a weighted summation of the linearly processed (via
                                                              Θ(l) ) node embeddings of neighbors in the weighted
                                   (n ,n ··· ,nS )            clique expansion graph. However, it is insufficient for
JmS , the tensor contraction A ×(m11 ,m2 2 ··· ,m S)
                                                     B yields
                                                              higher-order information extraction as only the first-
a (N + M − 2S)-order tensor C. Tensor contraction can order linear information is considered in Eq. (3.2).
be formulated as:
                            (in1 ,in2 ,...in )
           C = A ×(jm ,jm ,...jSm ) B                         3.1 Tensorized Hypergraph Neural Network
                X
                     1    2        S                          Eq. (3.2) has revealed that classical Hypergraph Neural
            =        Ai1 ,i2 ,···inS ,∗ B∗,i1 ,i2 ,···inS .   Networks approximate high-order information via the
                   i1 ,i2 ,···iN                              first-order summation. However, higher-order informa-
                                                              tion better characterizes co-occurrence relationship in
    The well known Mode-N Product is a special case of        hypergraph.
Tensor Contraction. Given a tensor A ∈ RI1 ×I2 ×···×IN             In order to characterize the influence of other nodes
and a matrix B ∈ RJ1 ×J2 . If J2 = In , then                  in the same hyperedge on its high-order interaction
                                       (n)                    information, for a node in a hypergraph, the most
                         C = A ×(2) B = A ×n B.               intuitive method is to use the outer product pooling [37]
                                                              of the feature vectors of its neighbors. For example, for
                                                              a node vi of a hyperedge {vi , vj , vk } ∈ E in a third-order
3   Methods
                                                              hypergraph G, the message of the hyperedge {vi , vj , vk }
In this section, we first analyze the widely adopted          to node vi is xvj ◦ xvk ∈ RIin ×Iin . Similar to other
hypergraph neural network – HGNN [7], which uses              graph neural networks, a trainable weight tensor can
first-order information for hypergraph representation         process and align features, followed by aggregating all
learning. Next, we propose and analyze tensorized             hyperedges. Then, the information aggregation of node
hypergraph neural network based on adjacency tensors.         vi embedding can be represented as:
Since the straightforward THNN cannot handle more
                                                                                                        (2,3)
                                                                                    X
common non-uniform hypergraphs, we introduce two              (3.3)      x(l+1)
                                                                           vi   =          xlvj ◦ xlvk ×(1,2) W,
simple yet effective solutions: global node adding and                                  (j,k)∈Ni
multi-uniform processing.
     Feng et al. [7] develop the classical Hypergraph Neu-    where W ∈ RIin ×Iin ×Iout is the weight tensor. , where
ral Networks which use truncated Chebyshev formula            Ni is the set of neighbor pairs of the node vi and ◦
as hypergraph Laplacians. Given the incidence matrix          indicates the outer product. The general N -th order
H ∈ R|V|×|E| of the hypergraph G, its operator can be         form. Then, the outer product information aggregation
written as                                                    of node vi embedding can be represented as follows:
(3.1)                                                         (3.4)
                                                                                                            (2,3,··· ,N )
                                                                           X
                                                                                     (xvj1 ◦· · ·◦xvjN −1 )×(1,2,··· ,N −1) W
                                                    
    X(l+1) = σ D(v) HWD−1   ⊤−1/2  (l) (l)       −1/2         xvj1 =
                       (e) H D(v) X Θ      ,
                                                                      (j1 ,j2 ,··· ,jN −1 )∈Ni

where W ∈ R|E|×|E| is a diagonal matrix to be learned         where W ∈ RIin ×Iin ···×Iout is the weight tensor.
and Θ(l) is a learnable matrix in layer l. D(v)ii =               Eq.(3.3) and (3.4) formulate the basic framework of
P|E|                         P|V|
  j=1 Wjj Hij and D(e)jj =      i=1 Hij are diagonal de-      the hypergraph neural network that utilizes high-order
gree matrices of vertices and edges, respectively. These      polynomial information directly. Such formulation also
methods can be viewed as applying clique expansion            reminds us of the polynomial regression scheme [13],


                                                                                                    Copyright © 2024 by SIAM
                                                                            Unauthorized reproduction of this article is prohibited
                                                                                        Low Rank weight




                                                                                        Low Rank weight




                                    Hypergraph Data       HIgh Order Fusion           Tensor Feature Process   Summation and Activation


Figure 3: Illustration of THNN. THNN tries to pass the high-order interactions of neighbors in different
hyperedges. The information of interactions is computed and processed via tensor operations.


which are highly recognized successful techniques for                             Using partially symmetric constraints is motivated
high-order interaction information extracting (such as                        by the assumption that the same combination of nodes
multi-modality analysis [12, 19, 37]).                                        in undirected hypergraph should result in equal out-
    Similar to the representation equivalence between                         put features after outer product fusion. For example,
the aggregation scheme and adjacency matrix formula-                          as for node pair vj and vk , the weight should hold
tion for Graph Convolution Neural Networks (GCN),                             W ×1 xvj ×2 xvk = W ×1 xvk ×2 xvj . Therefore, partially
the adjacency tensor can also be used to describe                             symmetry constraints can address this assumption and
Eq.(3.4). The outer product feature aggregation can                           reduce the number of parameters. The final low-rank
be reformulated as,                                                           aggregation scheme can be represented as follows

(3.5)                                                  
                                       (2,3,··· ,N )
X(l+1) = σ (A ×2 X(l) · · · ×N X(l) ) ×(1,2,··· ,N −1) W ,
                                                                                                                                      
                                                                                 X(l+1) = σ(( Ã ×2 (X(l) Θ(l) ) · · · ×N (X(l) Θ(l) )
                                                                                   (2,3,··· ,N )
     Similar to GCN, simple graph convolution layer                              ×(1,2,··· ,N −1) I)Q(l)T )
without feature normalization can result in numerical
instabilities because directly applying convolution layer
changes the scale of feature vectors. As a consequence                            where Ã ∈ R|V|×|V|···×|V| , X ∈ R|V|×Iin , I ∈
of this, there is a need for an appropriate level of degree-                R  R×R···×R
                                                                                         is the identity tensor, Θ(l) ∈ RIin ×R , and
                                                                              (l)     Iout ×R
normalization [15]. Similar to the degree-normalized                        Q ∈R              . Θ(l) and Q(l) are the learnable weights
adjacency matrix in GCN, we adopt the well-known                            in the l-th layer. Iin is the input feature dimension
normalizing adjacency tensor extension [24],                                number, Iout is the output dimensionality and R is
                                                                            the number of rank. We define the family of such
                     1              √ 1
               (            Q
                   (k−1)!   1⩽j⩽k   k dij
                                               if {vi1 , . . . , vik } ∈ E hypergraph neural networks as Tensorized Hypergraph
Ãi1 ...ik =                                                              . Neural Networks (THNN).
               0                               otherwise
                                                                              3.2 Architecture Analysis and Model Details
   Thus, Eq.(3.5) can be then reformulated as,
                                                                              Traditional GCN speeds up their computation via
          
                                        (2,3,··· ,N )
                                                                             sparse matrix operation in Pytorch1 or Tensorflow2 .
X(l+1) = σ (Ã ×2 X(l) · · · ×N X(l) ) ×(1,2,··· ,N −1) W ,
                                                                              However, sparse tensor operations have not been sup-
                                                                              ported well in common differential programming li-
Since the size of the parameter tensor will grow expo-
                                                                              braries. The above extension would suffer from high
nentially with the order number, such extremely large
                                                                              computational space cost of huge adjacency tensor, es-
storage and computational complexity is unacceptable.
                                                                              pecially when the order is high. After fully optimizing
This phenomenon is known as the curse of dimen-
sions [5], and proper tensor decomposition format can
effectively solve this problem. So, we decompose the
weight tensor W ∈ RIin ×Iin ···×Iout into the following
partially symmetric CP decomposition [16,22] structure
with the rank R:
                                                                                1 https://pytorch.org/

    W = I ×1 Θ(l) ×2 Θ(l) · · · ×N −1 Θ(l) ×N Q(l)T .                           2 https://www.tensorflow.org/




                                                                                                                    Copyright © 2024 by SIAM
                                                                                            Unauthorized reproduction of this article is prohibited
the order of calculations, we can rewrite the THNN in
                       X                   1        1
  xvi (l+1) =                   (Q(l) (             p                                                                                    Process Like a
                                        (N − 1)! Πl N djl                          Adding a global point                               Uniform Hypergraph
              (j ,j ,··· ,j
               1   2        )∈N
                          N −1      i

(3.6)
           l                l       
            xvj1                 x vj
     Θ(l)⊤        ⋆ · · · Θ(l)⊤      N −1    )).
             1                       1
                                                               Figure 4: Inspired by [38], we can add a global node
      where Ni is the set of neighbor pairs of the node vi ,   vg to a non-uniform hypergraph in order to make
and ⋆ is element-wise dot product. And considering             it uniform. The global node can be added many
that low order information can also be very important in       times in one hyperedge. Following this procedure, the
some cases, we concatenate a scalar 1 in feature vectors       hypergraph can well be represented as an adjacency
to generate lower-order dynamics. Such a strategy              tensor, allowing it to be directly processed in uniform-
could help THNN with low-order information modeling.           hypergraph form models.
In detail, Eq. (3.6) considers more on the 2nd-order
interactions and ignores some 1st-order information. As
for the 4-uniform situation, the 3rd-order interactions
would be considered more. As shown in Figure 2, such
preference can be alleviated if we concatenate original                                                      THNN Model 1

feature vector with a scalar 1.                                                     Or
                                                                                      de
                                                                                          r2



      As dot product of many vectors would lead the nu-
merical insatiability empirically, we add a new activa-                             Order 3                    THNN Model 2

                ′
tion function σ in the original architectures. We evalu-
                                                                                     Or
ated common activation functions and used T anh(·) in                                    de
                                                                                           r4                THNN Model 3


experiments. The final expression of THNN for uniform
Hyper-Graph is represented as follows                             Separate Into Different Order   Processing in Different Orders   Concatenation Merge   Processing Merged Vector


                     X                         1               Figure 5: We can also process hypergraphs in layers
xvi (l+1) = σ(                  (Q(l) Tanh(
                                            (N − 1)!           and utilize distinct models to process sub-hypergraphs
             (j1 ,j2 ,··· ,jN −1 )∈Ni
                                                               of different orders. The resultant embedding vectors of
(3.7)                                                          distinct layers are therefore concatenated and integrated
                 l               l       
    1       (l)⊤  xvj1          (l)⊤  x vj                     through a fully-connected layer.
    p     Θ             ⋆ ··· Θ           N −1    ))).
Πl N djl           1                      1

The whole procedure of THNN is shown in Figure 3.

3.3 Non-Uniform Generalization One of the                      linked neighbors of the informative global node will tend
most critical issues of using adjacency tensor in hy-          to converge to the same value, which may exacerbate the
pergraph analysis is that only uniform hypergraph can          oversmoothing in message passing procedure.
be processed. In addition, the proposed THNN in                Multi-Uniform Processing. So in order to mit-
Eq. (3.7) only considered the uniform hypergraph. But          igate the problem of oversmoothing of global node
in many real-world situations, non-uniform hypergraphs         adding strategies, as shown in Figure 5, we also pro-
are needed. Hence, we aim to extend the proposed               posed a multi-uniform processing scheme that decom-
model for non-uniform hypergraphs. Motivated by [38]           poses a non-uniform hypergraph into several uniform
and [24], we propose two methods to extend the uniform         sub-hypergraphs. One sub-hypergraph contains all the
hypergraph models for general hypergraphs.                     hyperedges with the same number of orders. Then, we
Global Node. As shown in Figure 4, we can add a                could process uniform sub-hypergraphs separately. Fi-
global node to the hypergraph. The non-informative             nally, we can concatenate feature vectors with different
global node would be added many times in one hyper-            orders and process them via a trainable weight matrix.
edge until the order of the hyperedge is equal to the              In this paper, we mainly focus on node classification
max-order number. The feature vector of the global             tasks. Therefore, when we get the node representation
point will be a trainable vector with the same size as         through several layers of THNN, we directly use a fully
other node features. Since the non-informative global          connected layer to obtain the predicted label and use
node has too many neighbors, the representation of the         the cross-entropy loss to optimize parameters.


                                                                                                            Copyright © 2024 by SIAM
                                                                                    Unauthorized reproduction of this article is prohibited
4   Experiments                                               datasets are reported in Tables 1.
In this section, we conduct experiments on two differ-             We have the following observations: First, THNN
ent 3-D visual object classification datasets under both      maintains the best performance in the majority of cases
uniform and non-uniform hypergraph construction set-          because the proposed models are compelling in ex-
tings. Experimental results verify the effectiveness of       tracting high-order information of hypergraph struc-
the proposed model. In addition, we also performed            tures. For example, compared with HyperGCN, THNN
ablation analysis and hyperparameter experiments in           achieves gains of 0.74% and 1.03% on average of 7 set-
order to acquire a better knowledge of the model.             tings on the ModelNet40 and the NTU datasets, re-
                                                              spectively. Second, the graph-based model performs
4.1 Datasets In experiments, two public bench-                worse than the hypergraph in most cases because the hy-
marks, the Princeton ModelNet40 dataset [32] and the          pergraph structure can convey complicated high-order
National Taiwan University (NTU) 3D model dataset [3]         correlations among data, whereas the clique expan-
are used. The ModelNet40 dataset includes 12,311 ob-          sion introduces some information loss. Furthermore,
jects from 40 popular categories, while the NTU dataset       as a result of information loss, the graph-based mod-
has 2,012 3D items from 67 categories. We apply the           els achieve poor results in some settings. For example,
similar data split settings in [7, 28], and we extract the    GCN achieves 77.72%, and GIN achieves 88.72% in the
features of 3D objects through the Multi-view Convolu-        C: MvGv, T: Mv setting of ModelNet40, while the
tional Neural Network (MVCNN) [29] and Group-view             mean accuracy of other hypergraph models under such
Convolutional Neural Network (GVCNN) [8]. 12 vir-             setting is all above 90%. The representation informa-
tual cameras are used to collect images with a 30-degree      tion under this setting can be better characterized by
interval angle, and MVCNN features and GVCNN fea-             a model considering higher-order information. These
tures are then extracted accordingly.                         cases indicate that simple graph-based models are un-
                                                              stable when higher-order information dominates.
4.2 Hypergraph Generation After obtaining the
vector embedding representation of the 3D object in           4.5 Results on the non-Uniform Setting We also
Euclidean space, we implement a distance-based hyper-         create non-uniform hypergraph structures to validate
graph generation [7, 9, 10]. Such a distance-generation       the efficacy of the two proposed extensions of THNN.
approach connects a group of k similar vertices to the        Because each element of the probability incidence ma-
same centroid and exploits the correlations among ver-        trix Ĥ takes value in [0, 1], and the closer the value
tices. Distance-based hyperedges can represent node           is to 1, the more similar with centroid node the value
connection in feature space [10].                             is, it is possible to perform Bernoulli sampling on Ĥ.
                                                              We employ the two-layer THNN extension models with
4.3 Baseline Models After constructing hyper-                 adding a global node (THNN-AdG) and Multi-Uniform
graphs, we compare THNN with multiple graph and               Processing (THNN-Multi) a rank setting of 128. The
hypergraph neural network baselines. All results about        two extensions are also trained with Adam optimizers
baseline are reproduced by ourselves. For GNN base-           whose learning rates are set to be 0.005.
lines, we feed the clique-expansion of constructed hy-             Detailed results are reported in Tables 2. In
pergraphs. The baselines are listed and decribed as fol-      these two tables, the results remain consistent with
lows: Graph convolutional network (GCN) [15], Graph           those in the uniform settings. The two proposed ex-
attention network (GAT) [30], Graph Isomorphism Net-          tensions of THNN perform the best in the majority
work (GIN) [17],HyperGCN [35], Hypergraph Networks            of scenarios. Specifically, THNN-Multi performs bet-
with Hyperedge Neurons (HNHN) [6], Hypergraph Neu-            ter than THNN-AdG. Compared with THNN-AdG,
ral Networks (HGNN) [7].                                      THNN-Multi achieves gains of 1.18% and 0.76% on av-
                                                              erage of the 7 settings on the ModelNet40 and the NTU
4.4 Results on the Uniform Setting First, we                  datasets, respectively. The possible reason is that in-
evaluate THNN along with the baselines in a uniform           teractions of different orders are learned via hypergraph
hypergraph setting with K = 4. We employ a two-layer          separation processing procedure and aligned via merged
THNN model with a rank setting of 128. THNN is                layer in THNN-Multi, while artificially introduced ex-
trained with the Adam optimizer whose learning rate is        ternal global node might disturb the interaction rep-
set to be 0.001 initially. Similar to the evaluation proce-   resentation of different orders due to the oversmooth-
dure in [7], we create multiple hypergraph structures for     ing issue in THNN-AdG. But THNN-Multi requires
comparison using either the single features or the con-       several THNN models to process the hypergraph mes-
catenation multi-features. Detailed results on the two        sage passing procedure of different orders. As a result,


                                                                                                 Copyright © 2024 by SIAM
                                                                         Unauthorized reproduction of this article is prohibited
Table 1: Experiment result of uniform generation setting. “C” means the feature type used in hypergraph
construction. ”T” the feature type used in model training (as the node feature vector). “Mv” means the MVCNN
feature, “Gv” means the GVCNN feature and “MvGv” means the concatenation of MVCNN feature and GVCNN
feature. We run 5 times in each setting and the results are displayed in (mean ± std) form.

 Dataset                                                 NUT2012                                                                                           ModelNet40
             C: Mv          C: Mv         C: Gv         C: Gv         C: MvGv       C: MvGv       C: MvGv       C: Mv         C: Mv         C: Gv         C: Gv         C: MvGv       C: MvGv       C: MvGv
 Setting
             T: Mv          T: Gv         T: Mv         T: Gv         T: Mv         T: Gv         T: MvGv       T: Mv         T: Gv         T: Mv         T: Gv         T: Mv         T: Gv         T: MvGv
  GCN        71.31±2.54%    75.23±2.11%   74.23±2.07%   78.76±2.14%   80.37±2.28%   80.84±1.93%   79.57±1.05%   85.12±1.75%   89.57±1.38%   83.09±1.23%   89.91±1.61%   86.80±3.05%   91.97±2.89%   88.21±3.03%
  GAT        74.45±0.69%    77.81±0.92%   81.31±0.91%   82.94±0.58%   82.57±0.28%   83.11±0.13%   80.11±0.67%   86.41±0.84%   90.79±0.81%   88.99±0.98%   91.55±0.21%   89.86±0.53%   95.03±0.55%   93.50±0.18%
  GIN        75.71±1.13%    78.38±0.78%   78.39±0.73%   80.77±0.85%   80.33±0.98%   82.47±0.28%   78.75±2.03%   86.74±0.57%   91.57±0.43%   89.97±0.43%   91.39±0.18%   88.79±1.38%   91.86±0.76%   91.95±0.32%
HyperGCN     77.21±0.35%    78.81±0.22%   83.95±0.28%   84.03±0.17%   80.09±0.12%   81.37±0.45%   80.32±0.76%   88.52±0.43%   91.04±0.98%   91.01±0.28%   91.87±0.24%   90.02±0.34%   95.08±0.16%   92.01±0.17%
 HNHH        77.33±0.15%    78.88±0.92%   84.03±0.41%   81.91±0.32%   81.94±0.73%   82.72±0.37%   82.24±0.33%   86.56±0.34%   91.15±0.18%   90.05±0.97%   91.93±0.58%   91.35±0.27%   94.36±0.22%   93.01±0.36%
 HGNN        77.27±0.12%    78.02±1.06%   81.65±0.23%   83.98±0.21%   81.79±0.38%   82.67±0.19%   82.87±0.28%   86.79±0.34%   91.96±0.42%   91.91±0.18%   91.76±0.23%   91.72±0.11%   95.75±0.13%   92.37±0.27%
 THNN        77.33±0.20%    78.21±0.27%   84.21±0.15%   83.98±0.15%   82.71±0.32%   82.77±0.24%   83.71±0.22%   87.55±0.25%   92.18±0.32%   91.91±0.21%   92.02±0.18%   92.53±0.19%   96.11±0.07%   93.65±0.15%




Table 2: Experiment result of non-uniform hypergraph. Compared with uniform hypergraph results in Table 1,
the hypergraph structure is non-uniform in this table. The original THNN is incapable of processing non-uniform
hypergraph structures. So, we report the result of two non-uniform extensions of THNN in this table. The non-
uniform hypergraph is generated via Bernoulli sampling on the probabilistic incidence matrix and the hypergraph
structure is the same in one setting.

 Dataset                                                 NUT2012                                                                                          ModelNet40
              C: Mv         C: Mv         C: Gv         C: Gv         C: MvGv       C: MvGv       C: MvGv       C: Mv         C: Mv         C: Gv         C: Gv         C: MvGv       C: MvGv       C: MvGv
 Setting
              T: Mv         T: Gv         T: Mv         T: Gv         T: Mv         T: Gv         T: MvGv       T: Mv         T: Gv         T: Mv         T: Gv         T: Mv         T: Gv         T: MvGv
   GCN        71.09±1.04%   74.01±1.01%   73.51±3.59%   77.73±3.63%   80.23±0.57%   80.01±0.91%   81.17±0.83%   84.73±0.98%   91.77±0.67%   77.95±3.21%   90.10±1.35%   77.72±1.53%   94.47±1.09%   88.79±2.29%
   GAT        75.04±0.87%   76.45±0.76%   80.56±0.82%   80.83±0.37%   79.67±0.62%   81.34±0.54%   77.62±0.78%   84.62±2.14%   89.65±1.29%   88.04±0.92%   90.17±0.63%   88.24±0.49%   94.52±0.57%   92.45±0.72%
   GIN        75.53±0.43%   75.37±0.99%   79.17±0.84%   80.07±0.66%   77.88±1.83%   82.32±0.24%   77.57±1.08%   87.23±0.59%   91.42±0.97%   87.03±1.32%   91.17±0.43%   88.72±1.50%   92.01±1.16%   94.29±0.65%
 HyperGCN     75.88±0.91%   76.72±0.32%   80.67±0.65%   82.23±1.07%   79.68±0.79%   81.63±0.91%   80.57±1.06%   87.59±0.95%   93.53±0.14%   91.91±0.13%   91.26±0.89%   91.19±1.08%   95.43±0.53%   93.72±0.61%
  HNHH        75.06±0.69%   76.41±0.36%   81.29±0.69%   81.70±1.32%   82.79±0.63%   82.76±0.65%   81.54±0.76%   88.28±0.34%   90.25±1.54%   91.76±0.33%   91.05±0.97%   90.79±0.82%   94.99±0.32%   93.75±0.99%
  HGNN        76.06±0.87%   77.25±0.15%   83.01±0.53%   81.57±0.47%   82.04±0.61%   83.01±0.96%   82.09±0.83%   89.33±0.78%   93.01±0.92%   91.13±0.54%   91.05±0.37%   90.32±1.03%   95.71±0.16%   94.91±0.12%
THNN-AdG      74.34±0.92%   77.01±0.19%   83.38±0.41%   82.01±0.11%   83.08±0.76%   81.50±0.87%   82.04±0.35%   87.44±1.01%   92.16±0.83%   91.31±0.56%   92.12±0.32%   93.01±0.81%   96.05±0.17%   93.33±0.32%
THNN-Multi    77.05±0.26%   77.42±0.14%   82.07±0.82%   82.84±0.06%   83.79±0.21%   82.09±0.27%   82.27±0.34%   90.01±0.31%   94.23±0.28%   91.42±0.37%   91.61±0.93%   93.19±0.73%   95.93±0.12%   95.07±0.13%


                                                          Table 3: Ablation Experiment Results. Results show
the number of parameters of THNN-Multi increases lin- that it is crucial to use the activation function like T anh,
early with the order number, but the parameter num- and other techniques can also improve the performance.
ber of the THNN-AdG is not influenced by order num-
                                                                   Dataset               NUT2012      ModelNet40
ber. Thus, the well-performing THNN-Multi generally                                 C: Mv C: MvGv C: Mv C: MvGv
has a larger number of parameters than THNN-AdG                    Setting
                                                                                    T: Gv T: MvGv T: Gv T: MvGv
when the order number is large. In addition, the graph-            No Tanh          69.43% 68.36% 82.74% 83.95%
                                                             No Concatenating Ones  77.47% 82.84% 92.14% 92.26%
based model still performs worse than the hypergraph        No Degree Normalization 77.47% 82.57% 92.38% 92.38%
in most cases and suffers from large variance across hy-            THNN            78.55% 83.91% 92.38% 94.25%
pergraph construction and training feature selection set-
tings. In summary, hypergraph-based models can bet-
ter model complex correlations among data than graph- which makes training neural networks extremely chal-
based models. Compared with other hypergraph mod- lenging. Small changes in the x can cause large distur-
els that adopt first-order approximations, our proposed bances in xn function.
adjacency-tensor-based THNN enjoys higher-order in-
formation modeling, leading to better performance.        4.7 Hyper-parameter Analysis:                Rank and
                                                          Number of Layers As shown in Fig 6a, we evaluate
4.6 Ablation Study We also conducted some abla- the impact of rank R on model performance by gen-
tion experiments to verify the effectiveness of the pro- erating hypergraphs with varying values of K (from 2
posed techniques. We choose two uniform hypergraph to 6) in the NUT2012 experiment and by adjusting the
generation settings. For these two datasets, we evalu- rank setting in the THNN. We choose the ”C: Mv T:
ate the proposed three techniques. We use T anh in σ ′ Gv” setting in NUT2012. We found that Rank = 128
to make the element-wise product in CP decomposition is a proper setting, as the accuracy does not change
stable. Secondly, we use the techniques of the concate- much when Rank ∈ [75, 150]. We also examine how the
nating ones to improve the expressiveness of the model. number of layers in THNN affects performance. Under
We also evaluate the normalized adjacency tensor strat- the setting of ”C: Mv T: Gv” in NUT2012, we discover
egy, which is inspired by the normalized Laplacian adja- that the model’s expressiveness cannot be fully explored
cency matrix to make the training stable. Results show with a single layer. Still, the model’s performance de-
that it is crucial to use T anh. This is because higher- grades dramatically when there are too many layers.
order tensor operations introduce operations such as xn , We conject that this may be due to numerical instabil-


                                                                                                                                                       Copyright © 2024 by SIAM
                                                                                                                               Unauthorized reproduction of this article is prohibited
  ity (for example, 10.00110 is much larger than 1010 ) of                 multiple modalities. Nevertheless, the TFL model faces
  higher order operations or over-smoothing problems in                    challenges related to computational complexity and an
  graph learning [2]. The experiment results in Fig 6b has                 increase in the number of parameters, particularly when
  demonstrated that stacking two layers is optimal.                        the number of modalities grows. In response to the scal-
                                                                           ability concerns of TFL, the Low-rank Multimodal Fu-
          80                                                               sion (LMF) [19] method was proposed. LMF employs
          75                                     85
          70                                                               low-rank tensor decomposition techniques to reduce the
          65                                     80



                                       Accuracy(%)
          60                                     75                        computational burden associated with TFL. This type


Accuracy(%)
          55                                     70                        of fusion operation can be regarded as a specialized form
          50                 K=2                 65
          45                 K=3                 60                        of Higher-order Polynomial Regression [4, 13]. The pro-
          40                                     55
          35                 K=4                 50                        posed THNN represents the first Tensorized Neural Net-
          30                 K=5                 45                        work tailored to model high-order information interac-
          25                                     400
          20                 K=6                       1   2   3   4   5   tions within hypergraphs.
          150   25 50 75 100 125 150
                                                       Number of Layers
                     Rank                                                  6   Conclusion and Discussion
                                         (b) Influence of number lay-
        (a) Rank analysis result.        ers.                              In this paper, we discover that existing hypergraph neu-
                                                                           ral networks are mostly based on the message passing
  Figure 6: Hyper-parameter analysis. (a) Rank = 128 is
                                                                           of first-order approximations of the original hypergraph
  a proper setting, as the accuracy does not change much
                                                                           structure and ignore higher-order interactions encoded
  when Rank ∈ [75, 150]. (b) The performance of THNN
                                                                           in hypergraphs. In order to better model the higher-
  degrades dramatically when there are too many layers.
                                                                           order information of the hypergraph structure, we pro-
                                                                           pose a novel hypergraph neural network, THNN, which
                                                                           is based on adjacency tensors of hypergraphs and is
  5           Related Works                                                a high-order extension of traditional Graph Convolu-
  5.1 Hypergraph Learning with Adjacency Ten-                              tion Neural Networks. We also find that the proposed
  sor In hypergraph learning, most existing methods                        models are well connected with tensor fusion, which is
  focus on converting the hypergraph into a weighted                       a highly recognized successful technique for high-order
  graph [6,7,35], thereby allowing the application of tradi-               interaction modeling. Therefore, the proposed models
  tional graph methods. For instance, spectral clustering                  are compelling in extracting high-order information en-
  can be performed on the weighted graph Laplacian [11]                    coded in hypergraphs. We show that our framework can
  to detect communities within hypergraphs. However,                       achieve the promising performance in 3-D visual object
  as highlighted in [14], this conversion process can lead                 classification tasks under both uniform and non-uniform
  to significant information loss and consequently, sub-                   hypergraph settings. In the future, we plan to explore
  optimal performance in detecting hypergraph communi-                     more applications of THNN to verify its advantages.
  ties. Given that tensors naturally represent high-order
  relationships [24], several methods based on adjacency                   7   Acknowledge
  tensors have been proposed. These methods aim to         This research was partially supported by Research Im-
  achieve optimal community detection [14] in both uni-    pact Fund (No.R1015-23), APRC - CityU New Re-
  form and non-uniform hypergraphs. Despite the major-     search Initiatives (No.9610565, Start-up Grant for New
  ity of adjacency tensor-based research focusing on the   Faculty of City University of Hong Kong), CityU -
  spectral properties [25, 26, 33] of hypergraphs and ten- HKIDS Early Career Research Grant (No.9360163),
  sor decomposition modelings [23, 38], there appears to   Hong Kong ITC Innovation and Technology Fund
  be a lack of hypergraph neural networks based on ad-     Midstream Research Programme for Universities
  jacency tensors. This gap in the literature presents an  Project (No.ITS/034/22MS), Hong Kong Environmen-
  opportunity for significant advancements in this field.  tal and Conservation Fund (No.88/2022), and SIRG
                                                           - CityU Strategic Interdisciplinary Research Grant
  5.2 Tensor Fusion Models and Tensorized Neu- (No.7020046, No.7020074), Ant Group (CCF-Ant Re-
  ral Networks Zadeh et al. [37] introduced a pioneering search Fund, Ant Group Research Fund), Huawei
  concept known as the Tensor Fusion Layer (TFL), which (Huawei Innovation Research Program), Tencent (CCF-
  utilizes a tensorized outer product to serve as a deep Tencent Open Fund, Tencent Rhino-Bird Focused Re-
  information fusion layer. The TFL framework is de- search Program), CCF-BaiChuan-Ebtech Foundation
  signed to learn both intra-modality and inter-modality Model Fund, and Kuaishou.
  dynamics while efficiently aggregating interactions from


                                                                                                              Copyright © 2024 by SIAM
                                                                                      Unauthorized reproduction of this article is prohibited
References                                                      [20] M. V. Martinet, L-E Kramer and Others, Ro-
                                                                     bust dynamic community detection with applications to
 [1] A. Bretto, Hypergraph theory, An introduction.                  human brain functional networks, Nature communica-
     Mathematical Engineering. Cham: Springer, (2013).               tions, (2020).
 [2] D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, and X. Sun,        [21] D. A. Matthews, High-performance tensor contrac-
     Measuring and relieving the over-smoothing problem              tion without transposition, SIAM Journal on Scientific
     for graph neural networks from the topological view, in         Computing, 40 (2018), pp. C1–C24.
     Proc. of AAAI, 2020.                                       [22] G. Ni and Y. Li, A semidefinite relaxation method for
 [3] D.-Y. Chen, X.-P. Tian, Y.-T. Shen, and                         partially symmetric tensor decomposition, Mathemat-
     M. Ouhyoung, On visual similarity based 3d model                ics of Operations Research, 47 (2022), pp. 2931–2949.
     retrieval, in Computer graphics forum, 2003.               [23] M. Nickel and V. Tresp, Tensor factorization for
 [4] C.-L. Cheng and H. Schneeweiss, Polynomial re-                  multi-relational learning, in Proc. of ECML, 2013.
     gression with errors in the variables, Journal of the      [24] X. Ouvrard, J.-M. Le Goff, and S. Marchand-
     Royal Statistical Society: Series B (Statistical Method-        Maillet, On adjacency and e-adjacency in general
     ology), 60 (1998), pp. 189–199.                                 hypergraphs: Towards a new e-adjacency tensor, Elec-
 [5] A. Cichocki, Era of big data processing: A new ap-              tronic Notes in Discrete Mathematics, (2018).
     proach via tensor networks and tensor decompositions,      [25] K. Pearson and T. Zhang, Eigenvalues on the
     arXiv preprint arXiv:1403.2048, (2014).                         adjacency tensor of products of hypergraphs, Int. J.
 [6] Y. Dong, W. Sawin, and Y. Bengio, Hnhn: hy-                     Contemp. Math. Sci, (2013).
     pergraph networks with hyperedge neurons, ICML 2020        [26] K. J. Pearson and T. Zhang, On spectral hypergraph
     Workshop, (2020).                                               theory of the adjacency tensor, Graphs and Combina-
 [7] Y. Feng, H. You, Z. Zhang, R. Ji, and Y. Gao,                   torics, (2014).
     Hypergraph neural networks, in Proc. of AAAI, 2019.        [27] L. Pu, Relational learning with hypergraphs, Ph.D.
 [8] Y. Feng, Z. Zhang, X. Zhao, R. Ji, and Y. Gao,                  Thesis in EPFL, (2013).
     Gvcnn: Group-view convolutional neural networks for        [28] H. Ran, J. Liu, and C. Wang, Surface representation
     3d shape recognition, in Proc. of CVPR, 2018.                   for point clouds, in Proc. of CVPR, 2022.
 [9] Y. Gao, M. Wang, D. Tao, R. Ji, and Q. Dai, 3-d            [29] H. Su, S. Maji, E. Kalogerakis, and E. Learned-
     object retrieval and recognition with hypergraph analy-         Miller, Multi-view convolutional neural networks for
     sis, IEEE Transactions on Image Processing, (2012).             3d shape recognition, in Proc. of ICCV, 2015.
[10] Y. Gao, Z. Zhang, H. Lin, X. Zhao, S. Du, and              [30] P. Velickovic, G. Cucurull, A. Casanova,
     C. Zou, Hypergraph learning: Methods and practices,             A. Romero, P. Lio, and Y. Bengio, Graph atten-
     IEEE Transactions on PAMI, (2020).                              tion networks, stat, (2017).
[11] D. Ghoshdastidar and A. Dukkipati, Consistency             [31] M. Wang, C. Zhang, Y. Pan, J. Xu, and Z. Xu,
     of spectral hypergraph partitioning under planted parti-        Tensor ring restricted boltzmann machines, in IJCNN
     tion model, The Annals of Statistics, (2017).                   2019, IEEE, 2019, pp. 1–8.
[12] M. Hou, J. Tang, J. Zhang, W. Kong, and                    [32] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang,
     Q. Zhao, Deep multimodal multilinear fusion with                X. Tang, and J. Xiao, 3d shapenets: A deep rep-
     high-order polynomial pooling, Proc. of NeurIPS,                resentation for volumetric shapes, in Proc. of CVPR,
     (2019).                                                         2015.
[13] J.-L. Hsu, W. Van Hecke, C.-H. Bai, C.-H. Lee,             [33] J. Xie and A. Chang, On the z-eigenvalues of the
     et al., Microstructural white matter changes in nor-            adjacency tensors for uniform hypergraphs, Linear Al-
     mal aging: a diffusion tensor imaging study with                gebra and its Applications, (2013).
     higher-order polynomial regression models, Neuroim-        [34] C. Xu, M. Li, Z. Ni, Y. Zhang, and S. Chen,
     age, (2010).                                                    Groupnet: Multiscale hypergraph neural networks for
[14] Z. T. Ke, F. Shi, and D. Xia, Community detection               trajectory prediction with relational reasoning, in Proc.
     for hypergraph networks via regularized tensor power            of CVPR, 2022.
     iteration, arXiv preprint arXiv:1909.06503, (2019).        [35] N. Yadati, M. Nimishakavi, and Others, Hyper-
[15] T. N. Kipf and M. Welling, Semi-supervised classi-              gcn: A new method for training graph convolutional
     fication with graph convolutional networks, in Proc. of         networks on hypergraphs, Proc. of NeurIPS, (2019).
     ICLR, 2016.                                                [36] C. Yang, R. Wang, S. Yao, and T. Abdelza-
[16] T. G. Kolda and B. W. Bader, Tensor decomposi-                  her, Hypergraph learning with line expansion, arXiv
     tions and applications, SIAM review, (2009).                    preprint arXiv:2005.04843, (2020).
[17] K. X. W. H. J. Leskovec and S. Jegelka, How                [37] A. Zadeh, M. Chen, S. Poria, E. Cambria, and L.-
     powerful are graph neural networks, ICLR., (2019).              P. Morency, Tensor fusion network for multimodal
[18] X. Liao, Y. Xu, and H. Ling, Hypergraph neural                  sentiment analysis, in Proc. of EMNLP, 2017.
     networks for hypergraph matching, in ICCV, 2021.           [38] Y. Zhen and J. Wang, Community detection in
[19] Z. Liu and Y. Shen, Efficient low-rank multimodal               general hypergraph via graph embedding, Journal of the
     fusion with modality-specific factors, in ACL, 2018.            American Statistical Association, (2022).



                                                                                                    Copyright © 2024 by SIAM
                                                                            Unauthorized reproduction of this article is prohibited

