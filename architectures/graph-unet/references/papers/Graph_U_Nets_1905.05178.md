# Paper (Gao et al. 2019)

> Source: `https://arxiv.org/abs/1905.05178`

---

Graph U-Nets
Hongyang Gao 1 Shuiwang Ji 1
Abstract
We consider the problem of representation learn-
ing for graph data. Convolutional neural networks
can naturally operate on images, but have sig-
niﬁcant challenges in dealing with graph data.
Given images are special cases of graphs with
nodes lie on 2D lattices, graph embedding tasks
have a natural correspondence with image pixel-
wise prediction tasks such as segmentation. While
encoder-decoder architectures like U-Nets have
been successfully applied on many image pixel-
wise prediction tasks, similar methods are lack-
ing for graph data. This is due to the fact that
pooling and up-sampling operations are not nat-
ural on graph data. To address these challenges,
we propose novel graph pooling (gPool) and un-
pooling (gUnpool) operations in this work. The
gPool layer adaptively selects some nodes to form
a smaller graph based on their scalar projection
values on a trainable projection vector. We fur-
ther propose the gUnpool layer as the inverse op-
eration of the gPool layer. The gUnpool layer
restores the graph into its original structure us-
ing the position information of nodes selected
in the corresponding gPool layer. Based on our
proposed gPool and gUnpool layers, we develop
an encoder-decoder model on graph, known as
the graph U-Nets. Our experimental results on
node classiﬁcation and graph classiﬁcation tasks
demonstrate that our methods achieve consistently
better performance than previous models.
1. Introduction
Convolutional neural networks (CNNs) (LeCun et al., 2012)
have demonstrated great capability in various challenging
artiﬁcial intelligence tasks, especially in ﬁelds of computer
1Department of Computer Science & Engineering, Texas
A&M University, TX, USA. Correspondence to: Hongyang Gao
<hongyang.gao@tamu.edu>, Shuiwang Ji <sji@tamu.edu>.
Proceedings of the 36 th International Conference on Machine
Learning, Long Beach, California, PMLR 97, 2019. Copyright
2019 by the author(s).
vision (He et al., 2017; Huang et al., 2017) and natural lan-
guage processing (Bahdanau et al., 2015). One common
property behind these tasks is that both images and texts
have grid-like structures. Elements on feature maps have lo-
cality and order information, which enables the application
of convolutional operations (Defferrard et al., 2016).
In practice, many real-world data can be naturally repre-
sented as graphs such as social and biological networks.
Due to the great success of CNNs on grid-like data, apply-
ing them on graph data (Gori et al., 2005; Scarselli et al.,
2009) is particularly appealing. Recently, there have been
many attempts to extend convolutions to graph data (GNNs)
(Kipf & Welling, 2017; Veliˇckovi´c et al., 2017; Gao et al.,
2018). One common use of convolutions on graphs is to
compute node representations (Hamilton et al., 2017; Ying
et al., 2018). With learned node representations, we can
perform various tasks on graphs such as node classiﬁcation
and link prediction.
Images can be considered as special cases of graphs, in
which nodes lie on regular 2D lattices. It is this special
structure that enables the use of convolution and pooling op-
erations on images. Based on this relationship, node classi-
ﬁcation and embedding tasks have a natural correspondence
with pixel-wise prediction tasks such as image segmenta-
tion (Noh et al., 2015; Gao & Ji, 2017; J´egou et al., 2017). In
particular, both tasks aim to make predictions for each input
unit, corresponding to a pixel on images or a node in graphs.
In the computer vision ﬁeld, pixel-wise prediction tasks
have achieved major advances recently. Encoder-decoder
architectures like the U-Net (Ronneberger et al., 2015) are
state-of-the-art methods for these tasks. It is thus highly
interesting to develop U-Net-like architectures for graph
data. In addition to convolutions, pooling and up-sampling
operations are essential building blocks in these architec-
tures. However, extending these operations to graph data
is highly challenging. Unlike grid-like data such as images
and texts, nodes in graphs have no spatial locality and order
information as required by regular pooling operations.
To bridge the above gap, we propose novel graph pool-
ing (gPool) and unpooling (gUnpool) operations in this
work. Based on these two operations, we propose U-Net-
like architectures for graph data. The gPool operation sam-
ples some nodes to form a smaller graph based on their
arXiv:1905.05178v1  [cs.LG]  11 May 2019

Graph U-Nets
scalar projection values on a trainable projection vector. As
an inverse operation of gPool, we propose a corresponding
graph unpooling (gUnpool) operation, which restores the
graph to its original structure with the help of locations of
nodes selected in the corresponding gPool layer. Based on
the gPool and gUnpool layers, we develop graph U-Nets,
which allow high-level feature encoding and decoding for
network embedding. Experimental results on node classiﬁ-
cation and graph classiﬁcation tasks demonstrate the effec-
tiveness of our proposed methods as compared to previous
methods.
2. Related Work
Recently, there has been a rich line of research on graph
neural networks (Gilmer et al., 2017). Inspired by the ﬁrst
order graph Laplacian methods, (Kipf & Welling, 2017)
proposed graph convolutional networks (GCNs), which
achieved promising performance on graph node classiﬁ-
cation tasks. The layer-wise forward-propagation operation
of GCNs is deﬁned as:
Xℓ+1 = σ( ˆD−1
2 ˆA ˆD−1
2 XℓWℓ),
(1)
where ˆA = A + I is used to add self-loops in the input
adjacency matrix A, Xℓis the feature matrix of layer ℓ.
The GCN layer uses the diagonal node degree matrix ˆD to
normalize ˆA. Wℓis a trainable weight matrix that applies a
linear transformation to feature vectors. GCNs essentially
perform aggregation and transformation on node features
without learning trainable ﬁlters. (Hamilton et al., 2017)
tried to sample a ﬁxed number of neighboring nodes to
keep the computational footprint consistent. (Veliˇckovi´c
et al., 2017) proposed to use attention mechanisms to enable
different weights for neighboring nodes. (Schlichtkrull et al.,
2018) used relational graph convolutional networks for link
prediction and entity classiﬁcation. Some studies applied
GNNs to graph classiﬁcation tasks (Duvenaud et al., 2015;
Dai et al., 2016; Zhang et al., 2018). (Bronstein et al., 2017)
discussed possible ways of applying deep learning on graph
data. (Henaff et al., 2015) and (Bruna et al., 2014) proposed
to use spectral networks for large-scale graph classiﬁcation
tasks. Some studies also applied graph kernels on traditional
computer vision tasks (Gama et al., 2019; Fey et al., 2018;
Monti et al., 2017).
In addition to convolution, some studies tried to extend pool-
ing operations to graphs. (Defferrard et al., 2016) proposed
to use binary tree indexing for graph coarsening, which ﬁxes
indices of nodes before applying 1-D pooling operations.
(Simonovsky & Komodakis, 2017) used deterministic graph
clustering algorithm to determine pooling patterns. (Ying
et al., 2018) used an assignment matrix to achieve pooling
by assigning nodes to different clusters of the next layer.
3. Graph U-Nets
In this section, we introduce the graph pooling (gPool) layer
and graph unpooling (gUnpool) layer. Based on these two
new layers, we develop the graph U-Nets for node classiﬁ-
cation tasks.
3.1. Graph Pooling Layer
Pooling layers play important roles in CNNs on grid-like
data. They can reduce sizes of feature maps and enlarge
receptive ﬁelds, thereby giving rise to better generaliza-
tion and performance (Yu & Koltun, 2016). On grid-like
data such as images, feature maps are partitioned into non-
overlapping rectangles, on which non-linear down-sampling
functions like maximum are applied. In addition to local
pooling, global pooling layers (Zhao et al., 2015a) perform
down-sampling operations on all input units, thereby re-
ducing each feature map to a single number. In contrast,
k-max pooling layers (Kalchbrenner et al., 2014) select the
k-largest units out of each feature map.
However, we cannot directly apply these pooling operations
to graphs. In particular, there is no locality information
among nodes in graphs. Thus the partition operation is not
applicable on graphs. The global pooling operation will
reduce all nodes to one single node, which restricts the
ﬂexibility of networks. The k-max pooling operation out-
puts the k-largest units that may come from different nodes
in graphs, resulting in inconsistency in the connectivity of
selected nodes.
In this section, we propose the graph pooling (gPool) layer
to enable down-sampling on graph data. In this layer, we
adaptively select a subset of nodes to form a new but smaller
graph. To this end, we employ a trainable projection vector
p. By projecting all node features to 1D, we can perform
k-max pooling for node selection. Since the selection is
based on 1D footprint of each node, the connectivity in
the new graph is consistent across nodes. Given a node i
with its feature vector xi, the scalar projection of xi on p is
yi = xip/∥p∥. Here, yi measures how much information
of node i can be retained when projected onto the direction
of p. By sampling nodes, we wish to preserve as much
information as possible from the original graph. To achieve
this, we select nodes with the largest scalar projection values
on p to form a new graph.
Suppose there are N nodes in a graph G and each of which
contains C features. The graph can be represented by two
matrices; those are the adjacency matrix Aℓ∈RN×N and
the feature matrix Xℓ∈RN×C. Each non-zero entry in
the adjacency matrix A represents an edge between two
nodes in the graph. Each row vector xℓ
i in the feature matrix
Xℓdenotes the feature vector of node i in the graph. The
layer-wise propagation rule of the graph pooling layer ℓis

Graph U-Nets
Projection
Top k Node Selection
Gate
!
"#
#
$%
%ℓ'(
%ℓ
×
*ℓ'(
idx
*ℓ
Outputs
top k
sigmoid
⨀
Inputs
1
!
Figure 1. An illustration of the proposed graph pooling layer with k = 2. × and ⊙denote matrix multiplication and element-wise product,
respectively. We consider a graph with 4 nodes, and each node has 5 features. By processing this graph, we obtain the adjacency matrix
Aℓ∈R4×4 and the input feature matrix Xℓ∈R4×5 of layer ℓ. In the projection stage, p ∈R5 is a trainable projection vector. By matrix
multiplication and sigmoid(·), we obtain y that are scores estimating scalar projection values of each node to the projection vector. By
using k = 2, we select two nodes with the highest scores and record their indices in the top-k-node selection stage. We use the indices to
extract the corresponding nodes to form a new graph, resulting in the pooled feature map ˜
Xℓand new corresponding adjacency matrix
Aℓ+1. At the gate stage, we perform element-wise multiplication between ˜
Xℓand the selected node scores vector ˜y, resulting in Xℓ+1.
This graph pooling layer outputs Aℓ+1 and Xℓ+1.
deﬁned as:
y = Xℓpℓ/∥pℓ∥,
idx = rank(y, k),
˜y = sigmoid(y(idx)),
˜Xℓ= Xℓ(idx, :),
Aℓ+1 = Aℓ(idx, idx),
Xℓ+1 = ˜Xℓ⊙
 ˜y1T
C

,
(2)
where k is the number of nodes selected in the new graph.
rank(y, k) is the operation of node ranking, which returns
indices of the k-largest values in y. The idx returned by
rank(y, k) contains the indices of nodes selected for the new
graph. Aℓ(idx, idx) and Xℓ(idx, :) perform the row and/or
column extraction to form the adjacency matrix and the
feature matrix for the new graph. y(idx) extracts values in y
with indices idx followed by a sigmoid operation. 1C ∈RC
is a vector of size C with all components being 1, and ⊙
represents the element-wise matrix multiplication.
Xℓis the feature matrix with row vectors xℓ
1, xℓ
2, · · · , xℓ
N,
each of which corresponds to a node in the graph. We ﬁrst
compute the scalar projection of Xℓon pℓ, resulting in
y = [y1, y2, · · · , yN]T with each yi measuring the scalar
projection value of each node on the projection vector pℓ.
Based on the scalar projection vector y, rank(·) operation
ranks values and returns the k-largest values in y. Sup-
pose the k-selected indices are i1, i2, · · · , ik with im < in
and 1 ≤m < n ≤k. Note that the index selection pro-
cess preserves the position order information in the original
graph. With indices idx, we extract the adjacency matrix
Aℓ∈Rk×k and the feature matrix ˜Xℓ∈Rk×C for the new
graph. Finally, we employ a gate operation to control infor-
mation ﬂow. With selected indices idx, we obtain the gate
vector ˜y ∈Rk by applying sigmoid to each element in the
extracted scalar projection vector. Using element-wise ma-
trix product of ˜Xℓand ˜y1T
C, information of selected nodes
is controlled. The ith row vector in Xℓ+1 is the product of
the ith row vector in Xℓand the ith scalar value in ˜y.
Notably, the gate operation makes the projection vector p
trainable by back-propagation (LeCun et al., 2012). Without
the gate operation, the projection vector p produces discrete
outputs, which makes it not trainable by back-propagation.
Figure 1 provides an illustration of our proposed graph
pooling layer. Compared to pooling operations used in grid-
like data, our graph pooling layer employs extra training
parameters in projection vector p. We will show that the
extra parameters are negligible but can boost performance.
3.2. Graph Unpooling Layer
Up-sampling operations are important for encoder-decoder
networks such as U-Net. The encoders of networks usually
employ pooling operations to reduce feature map size and
increase receptive ﬁeld. While in decoders, feature maps

Graph U-Nets
gPool
gUnpool
GCN
Figure 2. An illustration of the proposed graph unpooling (gUnpool) layer. In this example, a graph with 7 nodes is down-sampled using a
gPool layer, resulting in a coarsened graph with 4 nodes and position information of selected nodes. The corresponding gUnpool layer
uses the position information to reconstruct the original graph structure by using empty feature vectors for unselected nodes.
need to be up-sampled to restore their original resolutions.
On grid-like data like images, there are several up-sampling
operations such as the deconvolution (Isola et al., 2017;
Zhao et al., 2015b) and unpooling layers (Long et al., 2015).
However, such operations are not currently available on
graph data.
To enable up-sampling operations on graph data, we propose
the graph unpooling (gUnpool) layer, which performs the
inverse operation of the gPool layer and restores the graph
into its original structure. To achieve this, we record the
locations of nodes selected in the corresponding gPool layer
and use this information to place nodes back to their original
positions in the graph. Formally, we propose the layer-wise
propagation rule of graph unpooling layer as
Xℓ+1 = distribute(0N×C, Xℓ, idx),
(3)
where idx ∈Z∗k contains indices of selected nodes in the
corresponding gPool layer that reduces the graph size from
N nodes to k nodes. Xℓ∈Rk×C are the feature matrix of
the current graph, and 0N×C are the initially empty feature
matrix for the new graph. distribute(0N×C, Xℓ, idx) is the
operation that distributes row vectors in Xℓinto 0N×C fea-
ture matrix according to their corresponding indices stored
in idx. In Xℓ+1, row vectors with indices in idx are updated
by row vectors in Xℓ, while other row vectors remain zero.
3.3. Graph U-Nets Architecture
It is well-known that encoder-decoder networks like U-Net
achieve promising performance on pixel-wise prediction
tasks, since they can encode and decode high-level features
while maintaining local spatial information. Similar to pixel-
wise prediction tasks (Gong et al., 2014; Ronneberger et al.,
2015), node classiﬁcation tasks aim to make a prediction for
each input unit. Based on our proposed gPool and gUnpool
layers, we propose our graph U-Nets (g-U-Nets) architecture
for node classiﬁcation tasks.
In our graph U-Nets (g-U-Nets), we ﬁrst apply a graph
embedding layer to convert nodes into low-dimensional
representations, since original inputs of some dataset like
Cora (Sen et al., 2008) usually have very high-dimensional
feature vectors. After the graph embedding layer, we build
the encoder by stacking several encoding blocks, each of
which contains a gPool layer followed by a GCN layer.
gPool layers reduce the size of graph to encode higher-order
features, while GCN layers are responsible for aggregating
information from each node’s ﬁrst-order information. In the
decoder part, we stack the same number of decoding blocks
as in the encoder part. Each decoder block is composed of a
gUnpool layer and a GCN layer. The gUnpool layer restores
the graph into its higher resolution structure, and the GCN
layer aggregates information from the neighborhood. There
are skip-connections between corresponding blocks of en-
coder and decoder layers, which transmit spatial information
to decoders for better performance. The skip-connection
can be either feature map addition or concatenation. Finally,
we employ a GCN layer for ﬁnal predictions before the soft-
max function. Figure 3 provides an illustration of a sample
g-U-Nets with two blocks in encoder and decoder. Notably,
there is a GCN layer before each gPool layer, thereby en-
abling gPool layers to capture the topological information
in graphs implicitly.
3.4. Graph Connectivity Augmentation via Graph
Power
In our proposed gPool layer, we sample some important
nodes to form a new graph for high-level feature encoding.
Since related edges are removed when removing nodes in
gPool, the nodes in the pooled graph might become iso-

Graph U-Nets
GCN
Inputs
GCN
GCN
GCN
GCN
gPool
gPool
gUnpool
gUnpool
Network Embedding
Figure 3. An illustration of the proposed graph U-Nets (g-U-Nets). In this example, each node in the input graph has two features. The
input feature vectors are transformed into low-dimensional representations using a GCN layer. After that, we stack two encoder blocks,
each of which contains a gPool layer and a GCN layer. In the decoder part, there are also two decoder blocks. Each block consists of a
gUnpool layer and a GCN layer. For blocks in the same level, encoder block uses skip connection to fuse the low-level spatial features
from the encoder block. The output feature vectors of nodes in the last layer are network embedding, which can be used for various tasks
such as node classiﬁcation and link prediction.
lated. This may inﬂuence the information propagation in
subsequent layers, especially when GCN layers are used to
aggregate information from neighboring nodes. We need
to increase connectivity among nodes in the pooled graph.
To address this problem, we propose to use the kth graph
power Gk to increase the graph connectivity. This operation
builds links between nodes whose distances are at most k
hops (Chepuri & Leus, 2016). In this work, we employ
k = 2 since there is a GCN layer before each gPool layer
to aggregate information from its ﬁrst-order neighboring
nodes. Formally, we replace the ﬁfth equation in Eq 2 by:
A2 = AℓAℓ,
Aℓ+1 = A2(idx, idx),
(4)
where A2 ∈RN×N is the 2nd graph power. Now, the graph
sampling is performed on the augmented graph with better
connectivity.
3.5. Improved GCN Layer
In Eq. 1, the adjacency matrix before normalization is com-
puted as ˆA = A + I in which a self-loop is added to each
node in the graph. When performing information aggrega-
tion, the same weight is given to node’s own feature vector
and its neighboring nodes. In this work, we wish to give a
higher weight to node’s own feature vector, since its own
feature should be more important for prediction. To this
end, we change the calculation to ˆA = A + 2I by imposing
larger weights on self loops in the graph, which is common
in graph processing. All experiments in this work use this
modiﬁed version of GCN layer for better performance.
4. Experimental Study
In this section, we evaluate our gPool and gUnpool layers
based on the g-U-Nets proposed in Section 3.3. We com-
pare our networks with previous state-of-the-art models on
node classiﬁcation and graph classiﬁcation tasks. Exper-
imental results show that our methods achieve new state-
of-the-art results in terms of node classiﬁcation accuracy
and graph classiﬁcation accuracy. Some ablation studies
are performed to examine the contributions of the proposed
gPool layer, gUnpool layer, and graph connectivity augmen-
tation to performance improvements. We conduct studies
on the relationship between network depth and node classiﬁ-
cation performance. We investigate if additional parameters
involved in gPool layers can increase the risk of over-ﬁtting.
4.1. Datasets
In experiments, we evaluate our networks on node classiﬁ-
cation tasks under transductive learning settings and graph
classiﬁcation tasks under inductive learning settings.
Under transductive learning settings, unlabeled data are
accessible for training, which enables the network to learn
about the graph structure. To be speciﬁc, only part of nodes
are labeled while labels of other nodes in the same graph
remain unknown. We employ three benchmark datasets for
this setting; those are Cora, Citeseer, and Pubmed (Kipf &
Welling, 2017), which are summarized in Table 1. These
datasets are citation networks, with each node and each edge

Graph U-Nets
Table 1. Summary of datasets used in our node classiﬁcation experiments (Yang et al., 2016; Zitnik & Leskovec, 2017). The Cora, Citeseer,
and Pubmed datasets are used for transductive learning experiments.
Dataset
Nodes
Features
Classes
Training
Validation
Testing
Degree
Cora
2708
1433
7
140
500
1000
4
Citeseer
3327
3703
6
120
500
1000
5
Pubmed
19717
500
3
60
500
1000
6
Table 2. Summary of datasets used in our inductive learning experiments. The D&D (Dobson & Doig, 2003), PROTEINS (Borgwardt
et al., 2005), and COLLAB (Yanardag & Vishwanathan, 2015) datasets are used for inductive learning experiments.
Dataset
Graphs
Nodes (max)
Nodes (avg)
Classes
D&D
1178
5748
284.32
2
PROTEINS
1113
620
39.06
2
COLLAB
5000
492
74.49
3
representing a document and a citation, respectively. The
feature vector of each node is the bag-of-word representation
whose dimension is determined by the dictionary size. We
follow the same experimental settings in (Kipf & Welling,
2017). For each class, there are 20 nodes for training, 500
nodes for validation, and 1000 nodes for testing.
Under inductive learning settings, testing data are not avail-
able during training, which means the training process
does not use graph structures of testing data. We evalu-
ate our methods on relatively large graph datasets selected
from common benchmarks used in graph classiﬁcation
tasks (Ying et al., 2018; Niepert et al., 2016; Zhang et al.,
2018). We use protein datasets including D&D (Dobson &
Doig, 2003) and PROTEINS (Borgwardt et al., 2005), the
scientiﬁc collaboration dataset COLLAB (Yanardag & Vish-
wanathan, 2015). These data are summarized in Table 2.
4.2. Experimental Setup
We describe the experimental setup for both transductive
and inductive learning settings. For transductive learning
tasks, we employ our proposed g-U-Nets proposed in Sec-
tion 3.3. Since nodes in the three datasets are associated
with high-dimensional features, we employ a GCN layer
to reduce them into low-dimensional representations. In
the encoder part, we stack four blocks, each of which con-
sists of a gPool layer and a GCN layer. We sample 2000,
1000, 500, 200 nodes in the four gPool layers, respectively.
Correspondingly, the decoder part also contains four blocks.
Each decoder block is composed of a gUnpool layer and a
GCN layer. We use addition operation in skip connections
between blocks of encoder and decoder parts. Finally, we
apply a GCN layer for ﬁnal prediction. For all layers in the
model, we use identity activation function (Gao et al., 2018)
after each GCN layer. To avoid over-ﬁtting, we apply L2
regularization on weights with λ = 0.001. Dropout (Sri-
vastava et al., 2014) is applied to both adjacency matrices
and feature matrices with keep rates of 0.8 and 0.08, respec-
tively.
For inductive learning tasks, we follow the same experi-
mental setups in (Zhang et al., 2018) using our g-U-Nets
architecture as described in transductive learning settings for
feature extraction. Since the sizes of graphs vary in graph
classiﬁcation tasks, we sample proportions of nodes in four
gPool layers; those are 90%, 70%, 60%, and 50%, respec-
tively. The dropout keep rate imposed on feature matrices is
0.3.
4.3. Performance Study
Under transductive learning settings, we compare our pro-
posed g-U-Nets with other state-of-the-art models in terms
of node classiﬁcation accuracy. We report node classiﬁca-
tion accuracies on datasets Cora, Citeseer, and Pubmed, and
the results are summarized in Table 3. We can observe from
the results that our g-U-Nets achieves consistently better
performance than other networks. For baseline values listed
for node classiﬁcation tasks, they are the state-of-the-art
on these datasets. Our proposed model is composed of
GCN, gPool, and gUnpool layers without involving more
advanced graph convolution layers like GAT. When com-
pared to GCN directly, our g-U-Nets signiﬁcantly improves
performance on all three datasets by margins of 2.9%, 2.9%,
and 0.6%, respectively. Note that the only difference be-
tween our g-U-Nets and GCN is the use of encoder-decoder
architecture containing gPool and gUnpool layers. These
results demonstrate the effectiveness of g-U-Nets in network
embedding.
Under inductive learning settings, we compared our meth-
ods with other state-of-the-art models on graph classiﬁcation
tasks with datasets D&D, PROTEINS, and COLLAB, and
the results are summarized in Table 4. We can observe from
the results that our proposed gPool method outperforms

Graph U-Nets
Table 3. Results of transductive learning experiments in terms of node classiﬁcation accuracies on Cora, Citeseer, and Pubmed datasets.
g-U-Nets denotes our proposed graph U-Nets model.
Models
Cora
Citeseer
Pubmed
DeepWalk (Perozzi et al., 2014)
67.2%
43.2%
65.3%
Planetoid (Yang et al., 2016)
75.7%
64.7%
77.2%
Chebyshev (Defferrard et al., 2016)
81.2%
69.8%
74.4%
GCN (Kipf & Welling, 2017)
81.5%
70.3%
79.0%
GAT (Veliˇckovi´c et al., 2017)
83.0 ± 0.7%
72.5 ± 0.7%
79.0 ± 0.3%
g-U-Nets (Ours)
84.4 ± 0.6%
73.2 ± 0.5%
79.6 ± 0.2%
Table 4. Results of inductive learning experiments in terms of graph classiﬁcation accuracies on D&D, PROTEINS, and COLLAB datasets.
g-U-Nets denotes our proposed graph U-Nets model.
Models
D&D
PROTEINS
COLLAB
PSCN (Niepert et al., 2016)
76.27%
75.00%
72.60%
DGCNN (Zhang et al., 2018)
79.37%
76.26%
73.76%
DiffPool-DET (Ying et al., 2018)
75.47%
75.62%
82.13%
DiffPool-NOLP (Ying et al., 2018)
79.98%
76.22%
75.58%
DiffPool (Ying et al., 2018)
80.64%
76.25%
75.48%
g-U-Nets (Ours)
82.43%
77.68%
77.56%
DiffPool (Ying et al., 2018) by margins of 1.79% and 1.43%
on the D&D and PROTEINS datasets. Notably, the result ob-
tained by DiffPool-DET on COLLAB is signiﬁcantly higher
than all other methods and the other two DiffPool models.
On all three datasets, our model outperforms baseline mod-
els including DiffPool. In addition, DiffPool claimed that
their training utilized auxiliary task of link prediction to sta-
bilize model performance, which indicates the instability of
DiffPool model. But in our experiments, we only use graph
labels for training without any auxiliary tasks to stabilize
training.
4.4. Ablation Study of gPool and gUnpool layers
Although GCNs have been reported to have worse perfor-
mance when the network goes deeper (Kipf & Welling,
2017), it may also be argued that the performance improve-
ment over GCN in Table 3 is due to the use of a deeper
network architecture. In this section, we investigate the
contributions of gPool and gUnpool layers to the perfor-
mance of g-U-Nets. We conduct experiments by removing
all gPool and gUnpool layers from our g-U-Nets, leading
to a network with only GCN layers with skip connections.
Table 5 provides the comparison results between g-U-Nets
with and without gPool or gUnpool layers. The results show
that g-U-Nets have better performance over g-U-Nets with-
out gPool or gUnpool layers by margins of 2.3%, 1.6% and
0.5% on Cora, Citeseer, and Pubmed datasets, respectively.
These results demonstrate the contributions of gPool and
gUnpool layers to performance improvement. When con-
sidering the difference between the two models in terms of
architecture, g-U-Nets enable higher level feature encoding,
thereby resulting in better generalization and performance.
4.5. Graph Connectivity Augmentation Study
In the above experiments, we employ gPool layers with
graph connectivity augmentation by using the 2nd graph
power in Section 3.4. Here, we conduct experiments on node
classiﬁcation tasks to investigate the beneﬁts of graph con-
nectivity augmentation based on g-U-Nets. We remove the
graph connectivity augmentation from gPool layers while
keeping other settings the same for fairness of comparisons.
Table 6 provides comparison results between g-U-Nets with
and without graph connectivity augmentation. The results
show that the absence of graph connectivity augmentation
will cause consistent performance degradation on all of
three datasets. This demonstrates that graph connectivity
augmentation via 2nd graph power can help with the graph
connectivity and information transfer among nodes in sam-
pled graphs.
4.6. Network Depth Study of Graph U-Nets
Since the network depth in terms of the number of blocks in
encoder and decoder parts is an important hyper-parameter
in the g-U-Nets, we conduct experiments to investigate the
relationship between network depth and performance in
terms of node classiﬁcation accuracy. We use different
network depths on node classiﬁcation tasks and report the
classiﬁcation accuracies. The results are summarized in Ta-
ble 7. We can observe from the results that the performance
improves as network goes deeper until a depth of 4. The

Graph U-Nets
Table 5. Comparison of g-U-Nets with and without gPool or gUnpool layers in terms of node classiﬁcation accuracy on Cora, Citeseer,
and Pubmed datasets.
Models
Cora
Citeseer
Pubmed
g-U-Nets without gPool or gUnpool
82.1 ± 0.6%
71.6 ± 0.5%
79.1 ± 0.2%
g-U-Nets (Ours)
84.4 ± 0.6%
73.2 ± 0.5%
79.6 ± 0.2%
Table 6. Comparison of g-U-Nets with and without graph connectivity augmentation in terms of node classiﬁcation accuracy on Cora,
Citeseer, and Pubmed datasets.
Models
Cora
Citeseer
Pubmed
g-U-Nets without augmentation
83.7 ± 0.7%
72.5 ± 0.6%
79.0 ± 0.3%
g-U-Nets (Ours)
84.4 ± 0.6%
73.2 ± 0.5%
79.6 ± 0.2%
Table 7. Comparison of different network depths in terms of node classiﬁcation accuracy on Cora, Citeseer, and Pubmed datasets. Based
on g-U-Nets, we experiment with different network depths in terms of the number of blocks in encoder and decoder parts.
Depth
Cora
Citeseer
Pubmed
2
82.6 ± 0.6%
71.8 ± 0.5%
79.1 ± 0.3%
3
83.8 ± 0.7%
72.7 ± 0.7%
79.4 ± 0.4%
4
84.4 ± 0.6%
73.2 ± 0.5%
79.6 ± 0.2%
5
84.1 ± 0.5%
72.8 ± 0.6%
79.5 ± 0.3%
Table 8. Comparison of the g-U-Nets with and without gPool or gUnpool layers in terms of the node classiﬁcation accuracy and the
number of parameters on Cora dataset.
Models
Accuracy
#Params
Ratio of increase
g-U-Nets without gPool or gUnpool
82.1 ± 0.6%
75,643
0.00%
g-U-Nets (Ours)
84.4 ± 0.6%
75,737
0.12%
over-ﬁtting problem happens in deeper networks and pre-
vents networks from improving when the depth goes beyond
that. In image segmentation, U-Net models with depth 3 or
4 are commonly used (Badrinarayanan et al., 2017; C¸ ic¸ek
et al., 2016), which is consistent with our choice in exper-
iments. This indicates the capacity of gPool and gUnpool
layers in receptive ﬁeld enlargement and high-level feature
encoding even working with very shallow networks.
4.7. Parameter Study of Graph Pooling Layers
Since our proposed gPool layer involves extra parameters,
we compute the number of additional parameters based
on our g-U-Nets. The comparison results between g-U-
Nets with and without gPool or gUnpool layers on dataset
Cora are summarized in Table 8. From the results, we can
observe that gPool layers in U-Net model only adds 0.12%
additional parameters but can promote the performance by
a margin of 2.3%. We believe this negligible increase of
extra parameters will not increase the risk of over-ﬁtting.
Compared to g-U-Nets without gPool or gUnpool layers, the
encoder-decoder architecture with our gPool and gUnpool
layers yields signiﬁcant performance improvement.
5. Conclusion
In this work, we propose novel gPool and gUnpool layers in
g-U-Nets networks for network embedding. The gPool layer
implements the regular global k-max pooling operation on
graph data. It samples a subset of important nodes to en-
able high-level feature encoding and receptive ﬁeld enlarge-
ment. By employing a trainable projection vector, gPool
layers sample nodes based on their scalar projection values.
Furthermore, we propose the gUnpool layer which applies
unpooling operations on graph data. By using the position
information of nodes in the original graph, gUnpool layer
performs the inverse operation of the corresponding gPool
layer and restores the original graph structure. Based on our
gPool and gUnpool layers, we propose the graph U-Nets (g-
U-Nets) architecture which uses a similar encoder-decoder
architecture as regular U-Net on image data. Experimental
results demonstrate that our g-U-Nets achieve performance
improvements as compared to other GNNs on transductive
learning tasks. To avoid the isolated node problem that may
exist in sampled graphs, we employ the 2nd graph power
to improve graph connectivity. Ablation studies indicate
the contributions of our graph connectivity augmentation
approach.

Graph U-Nets
Acknowledgments
This work was supported in part by National Science Foun-
dation grants IIS-1908166 and IIS-1908198.
References
Badrinarayanan, V., Kendall, A., and Cipolla, R. Segnet: A
deep convolutional encoder-decoder architecture for im-
age segmentation. IEEE Transactions on Pattern Analysis
& Machine Intelligence, (12):2481–2495, 2017.
Bahdanau, D., Cho, K., and Bengio, Y. Neural machine
translation by jointly learning to align and translate. Inter-
national Conference on Learning Representations, 2015.
Borgwardt, K. M., Ong, C. S., Sch¨onauer, S., Vishwanathan,
S., Smola, A. J., and Kriegel, H.-P. Protein function
prediction via graph kernels. Bioinformatics, 21(suppl 1):
i47–i56, 2005.
Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Van-
dergheynst, P. Geometric deep learning: going beyond
euclidean data. IEEE Signal Processing Magazine, 34(4):
18–42, 2017.
Bruna, J., Zaremba, W., Szlam, A., and Lecun, Y. Spectral
networks and locally connected networks on graphs. In
International Conference on Learning Representations,
2014.
Chepuri, S. P. and Leus, G. Subsampling for graph power
spectrum estimation. In Sensor Array and Multichannel
Signal Processing Workshop, pp. 1–5. IEEE, 2016.
C¸ ic¸ek, ¨O., Abdulkadir, A., Lienkamp, S. S., Brox, T., and
Ronneberger, O. 3D U-Net: learning dense volumetric
segmentation from sparse annotation. In International
Conference on Medical Image Computing and Computer-
Assisted Intervention, pp. 424–432. Springer, 2016.
Dai, H., Dai, B., and Song, L. Discriminative embeddings of
latent variable models for structured data. In International
Conference on Machine Learning, pp. 2702–2711, 2016.
Defferrard, M., Bresson, X., and Vandergheynst, P. Con-
volutional neural networks on graphs with fast localized
spectral ﬁltering. In Advances in Neural Information
Processing Systems, pp. 3844–3852, 2016.
Dobson, P. D. and Doig, A. J. Distinguishing enzyme struc-
tures from non-enzymes without alignments. Journal of
molecular biology, 330(4):771–783, 2003.
Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell,
R., Hirzel, T., Aspuru-Guzik, A., and Adams, R. P. Con-
volutional networks on graphs for learning molecular ﬁn-
gerprints. In Advances in Neural Information Processing
Systems, pp. 2224–2232, 2015.
Fey, M., Eric Lenssen, J., Weichert, F., and M¨uller, H.
SplineCNN: Fast geometric deep learning with continu-
ous B-spline kernels. In Proceedings of the IEEE Con-
ference on Computer Vision and Pattern Recognition, pp.
869–877, 2018.
Gama, F., Marques, A. G., Leus, G., and Ribeiro, A. Con-
volutional neural network architectures for signals sup-
ported on graphs. IEEE Transactions on Signal Process-
ing, 67(4):1034–1049, 2019.
Gao, H. and Ji, S. Efﬁcient and invariant convolutional
neural networks for dense prediction.
In 2017 IEEE
International Conference on Data Mining, pp. 871–876.
IEEE, 2017.
Gao, H., Wang, Z., and Ji, S. Large-scale learnable graph
convolutional networks. In Proceedings of the 24th ACM
SIGKDD International Conference on Knowledge Dis-
covery & Data Mining, pp. 1416–1424. ACM, 2018.
Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and
Dahl, G. E. Neural message passing for quantum chem-
istry. arXiv preprint arXiv:1704.01212, 2017.
Gong, Y., Jia, Y., Leung, T., Toshev, A., and Ioffe, S. Deep
convolutional ranking for multilabel image annotation. In
Proceedings of the International Conference on Learning
Representations, 2014.
Gori, M., Monfardini, G., and Scarselli, F. A new model for
learning in graph domains. In 2005 IEEE International
Joint Conference on Neural Networks, volume 2, pp. 729–
734, 2005.
Hamilton, W., Ying, Z., and Leskovec, J. Inductive repre-
sentation learning on large graphs. In Advances in Neural
Information Processing Systems, pp. 1024–1034, 2017.
He, K., Gkioxari, G., Doll´ar, P., and Girshick, R. Mask r-
cnn. IEEE International Conference on Computer Vision,
2017.
Henaff, M., Bruna, J., and LeCun, Y.
Deep convolu-
tional networks on graph-structured data. arXiv preprint
arXiv:1506.05163, 2015.
Huang, G., Liu, Z., Weinberger, K. Q., and van der Maaten,
L. Densely connected convolutional networks. IEEE
Conference on Computer Vision and Pattern Recognition,
2017.
Isola, P., Zhu, J.-Y., Zhou, T., and Efros, A. A. Image-to-
image translation with conditional adversarial networks.
In 2017 IEEE Conference on Computer Vision and Pat-
tern Recognition, pp. 5967–5976. IEEE, 2017.

Graph U-Nets
J´egou, S., Drozdzal, M., Vazquez, D., Romero, A., and
Bengio, Y. The one hundred layers tiramisu: Fully con-
volutional densenets for semantic segmentation. In Pro-
ceedings of the IEEE Conference on Computer Vision
and Pattern Recognition Workshops, pp. 11–19, 2017.
Kalchbrenner, N., Grefenstette, E., and Blunsom, P. A
convolutional neural network for modelling sentences.
In Proceedings of the 52nd Annual Meeting of the As-
sociation for Computational Linguistics, volume 1, pp.
655–665, 2014.
Kipf, T. N. and Welling, M. Semi-supervised classiﬁca-
tion with graph convolutional networks. International
Conference on Learning Representations, 2017.
LeCun, Y., Bottou, L., Orr, G. B., and M¨uller, K.-R. Efﬁ-
cient backprop. In Neural networks: Tricks of the trade,
pp. 9–48. Springer, 2012.
Long, J., Shelhamer, E., and Darrell, T. Fully convolutional
networks for semantic segmentation. In Proceedings of
the IEEE Conference on Computer Vision and Pattern
Recognition, pp. 3431–3440, 2015.
Monti, F., Boscaini, D., Masci, J., Rodola, E., Svoboda, J.,
and Bronstein, M. M. Geometric deep learning on graphs
and manifolds using mixture model cnns. In Proceedings
of the IEEE Conference on Computer Vision and Pattern
Recognition, pp. 5115–5124, 2017.
Niepert, M., Ahmed, M., and Kutzkov, K. Learning con-
volutional neural networks for graphs. In International
Conference on Machine Learning, pp. 2014–2023, 2016.
Noh, H., Hong, S., and Han, B. Learning deconvolution
network for semantic segmentation. In Proceedings of
the IEEE International Conference on Computer Vision,
pp. 1520–1528, 2015.
Perozzi, B., Al-Rfou, R., and Skiena, S. Deepwalk: Online
learning of social representations. In Proceedings of the
20th ACM SIGKDD International Conference on Knowl-
edge Discovery and Data Mining, pp. 701–710. ACM,
2014.
Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolu-
tional networks for biomedical image segmentation. In In-
ternational Conference on Medical image computing and
computer-assisted intervention, pp. 234–241. Springer,
2015.
Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., and
Monfardini, G. The graph neural network model. IEEE
Transactions on Neural Networks, 20(1):61–80, 2009.
Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R.,
Titov, I., and Welling, M. Modeling relational data with
graph convolutional networks. In European Semantic
Web Conference, pp. 593–607. Springer, 2018.
Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B.,
and Eliassi-Rad, T. Collective classiﬁcation in network
data. AI magazine, 29(3):93, 2008.
Simonovsky, M. and Komodakis, N.
Dynamic edge-
conditioned ﬁlters in convolutional neural networks on
graphs. In Proceedings of the IEEE Conference on Com-
puter Vision and Pattern Recognition, pp. 3693–3702,
2017.
Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I.,
and Salakhutdinov, R. Dropout: A simple way to prevent
neural networks from overﬁtting. Journal of Machine
Learning Research, 15(1):1929–1958, 2014.
Veliˇckovi´c, P., Cucurull, G., Casanova, A., Romero, A.,
Li`o, P., and Bengio, Y. Graph attention networks. In
International Conference on Learning Representations,
2017.
Yanardag, P. and Vishwanathan, S. A structural smoothing
framework for robust graph comparison. In Advances in
Neural Information Processing Systems, pp. 2134–2142,
2015.
Yang, Z., Cohen, W., and Salakhudinov, R.
Revisiting
semi-supervised learning with graph embeddings. In
International Conference on Machine Learning, pp. 40–
48, 2016.
Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., and
Leskovec, J. Hierarchical graph representation learning
with differentiable pooling. In Advances in Neural Infor-
mation Processing Systems, pp. 4800–4810, 2018.
Yu, F. and Koltun, V. Multi-scale context aggregation by
dilated convolutions. In Proceedings of the International
Conference on Learning Representations, 2016.
Zhang, M., Cui, Z., Neumann, M., and Chen, Y. An end-to-
end deep learning architecture for graph classiﬁcation. In
Thirty-Second AAAI Conference on Artiﬁcial Intelligence,
pp. 4438–4445, 2018.
Zhao, H., Lu, Z., and Poupart, P. Self-adaptive hierarchical
sentence model. In Twenty-Fourth International Joint
Conference on Artiﬁcial Intelligence, 2015a.
Zhao, J. J., Mathieu, M., Goroshin, R., and LeCun,
Y.
Stacked what-where auto-encoders.
CoRR,
abs/1506.02351, 2015b.
Zitnik, M. and Leskovec, J. Predicting multicellular function
through multi-layer tissue networks. Bioinformatics, 33
(14):i190–i198, 2017.
