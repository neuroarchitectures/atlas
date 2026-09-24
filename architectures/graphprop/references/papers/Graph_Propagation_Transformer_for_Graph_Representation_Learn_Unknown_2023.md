# Graph Propagation Transformer for Graph Representation Learn Unknown 2023

> Source: `Graph_Propagation_Transformer_for_Graph_Representation_Learn_Unknown_2023.pdf`

---

                                                       Graph Propagation Transformer for Graph Representation Learning

                                                                  Zhe Chen1∗ , Hao Tan2∗ , Tao Wang1 , Tianrun Shen1 , Tong Lu1†
                                                                              Qiuying Peng2 , Cheng Cheng2 and Yue Qi2
                                                                   1
                                                                     State Key Lab for Novel Software Technology, Nanjing Univerisity
                                                                                        2
                                                                                          OPPO Research Institute
                                                                            chenzhe98@smail.nju.edu.cn, lutong@nju.edu.com




arXiv:2305.11424v2 [cs.LG] 15 Jun 2023
                                                                       Abstract
                                                  This paper presents a novel transformer architec-
                                                  ture for graph representation learning. The core
                                                  insight of our method is to fully consider the
                                                  information propagation among nodes and edges
                                                  in a graph when building the attention module
                                                  in the transformer blocks. Specifically, we pro-         (a) Node-to-Node        (b) Node-to-Edge        (c) Edge-to-Node

                                                  pose a new attention mechanism called Graph
                                                  Propagation Attention (GPA). It explicitly passes                 Node       Edge       Node Embedding      Edge Embedding

                                                  the information among nodes and edges in three
                                                  ways, i.e., node-to-node, node-to-edge, and edge-       Figure 1: Illustration of the three ways for graph information propa-
                                                  to-node, which is essential for learning graph-         gation. Circles and black lines indicate nodes and edges, and green
                                                  structured data. On this basis, we design an effec-     and pink cubes represent node embeddings and edge embeddings.
                                                  tive transformer architecture named Graph Propa-        Our GPTrans achieves better graph representation learning by ex-
                                                  gation Transformer (GPTrans) to further help learn      plicitly constructing three ways for information propagation in the
                                                  graph data. We verify the performance of GP-            proposed Graph Propagation Attention (GPA) module, including (a)
                                                  Trans in a wide range of graph learning experi-         node-to-node, (b) node-to-edge, and (c) edge-to-node.
                                                  ments on several benchmark datasets. These results
                                                  show that our method outperforms many state-of-
                                                  the-art transformer-based graph models with better
                                                                                                          The first category mainly focuses on performing Graph Neu-
                                                  performance. The code will be released at https:
                                                                                                          ral Networks (GNNs) on graph data. These methods follow
                                                  //github.com/czczup/GPTrans.
                                                                                                          the convolutional pattern to define the convolution operation
                                                                                                          in the graph data, and design effective neighborhood aggrega-
                                                                                                          tion schemes to learn node representations by fusing the node
                                         1       Introduction                                             and graph topology information. The representative method
                                         In many real-world scenarios, information is usually orga-       is Graph Convolutional Network (GCN) [Kipf and Welling,
                                         nized by graphs, and graph-structured data can be used in        2016], which learns the representation of a node in the graph
                                         many research areas, including communication networks and        by considering fusing its neighbors. After that, many GCN
                                         molecular property prediction, etc. For instance, based on so-   variants [Xu et al., 2018; Chen et al., 2020; Liu et al., 2021a;
                                         cial graphs, lots of algorithms are proposed to classify users   Bresson and Laurent, 2017] containing different neighbor-
                                         into meaningful social groups in the task of social network      hood aggregation schemes have been developed. The second
                                         research, which can produce many useful practical applica-       kind of method is to build graph models based on the trans-
                                         tions such as user search and recommendations. Therefore,        former architecture. For example, Cai and Lam [2020] uti-
                                         graph representation learning has become a hot topic in pat-     lized the explicit relation encoding between nodes and fused
                                         tern recognition and machine learning [Cai and Lam, 2020;        them into the encoder-decoder transformer network for ef-
                                         Ying et al., 2021; Brossard et al., 2020].                       fective graph-to-sequence learning. Graphormer [Ying et al.,
                                            With the development of deep learning, many methods           2021] established state-of-the-art performance on graph-level
                                         have been developed for graph representation learning [Per-      prediction tasks by transforming the structure and edge fea-
                                         ozzi et al., 2014; Zhang et al., 2019; Ying et al., 2021;        tures of the graph into attention biases.
                                         Hussain et al., 2021; Rampášek et al., 2022]. In general,         Although recent transformer-based methods report promis-
                                         these methods can be approximately divided into two parts.       ing performance for graph representation learning, they still
                                                                                                          suffer the following problems. (1) Not explicitly employ the
                                             ∗
                                                 Equal contribution. † Corresponding author.              relationship among nodes and edges in the graph data. Re-
cent transformer-based methods [Cai and Lam, 2020; Ying             Recently, the self-attention mechanism and transformer
et al., 2021] only simply fuse nodes and edges information       architecture have been gradually introduced into the graph
by using positional encodings. However, due to the complex-      representation learning tasks, such as graph-level prediction
ity of graph structure, how to fully employ the relationship     [Ying et al., 2021; Hussain et al., 2021], producing compet-
among nodes and edges for graph representation learning re-      itive performance compared to the traditional GNN models.
mains to be studied. (2) Inefficient dual-FFN structure in the   The early self-attention-based GNNs focused on adopting the
transformer block. Recent works resort to the dual-path struc-   attention mechanism to a local neighborhood of each node
ture in the transformer block to incorporate the edge informa-   in a graph, or directly on the whole graph. For example,
tion. For instance, the Edge-augmented Graph Transformer         Graph Attention Network (GAT) [Veličković et al., 2017] and
(EGT) [Hussain et al., 2021] adopted dual feed-forward net-      Graph Transformer (GT) [Dwivedi and Bresson, 2020] uti-
works (FFN) in the transformer block to update the edge em-      lized self-attention mechanisms as local constraints for the
beddings, and let the structural information evolve from layer   local neighborhood of each node. In contrast to employing
to layer. However, this paradigm learns the information of       local self-attention for graph learning, Graph-BERT [Zhang
edges and nodes separately, which introduces more calcula-       et al., 2020] introduced the global self-attention mechanism
tions and easily leads to the low efficiency of the model.       in a revised transformer network to predict one masked node
   To overcome these issues, we propose an efficient and         in a sampled subgraph.
powerful transformer architecture for graph learning, termed        In addition, several works have attempted to use the trans-
Graph Propagation Transformer (GPTrans). A key design el-        former architecture to tackle graph-related tasks directly. Two
ement of GPTrans is its Graph Propagation Attention (GPA)        notable examples are [Cai and Lam, 2020] and Graphormer
module. As illustrated in Figure 1, the GPA module propa-        [Ying et al., 2021]. The former method adopted explicit
gates the information among the node embeddings and edge         relation encoding between nodes and integrated them into
embeddings of the preceding layer by modeling three con-         the encoder-decoder transformer network, to enable graph-
nections, i.e., node-to-node, node-to-edge, and edge-to-node,    to-sequence learning. The latter mainly regarded the structure
which significantly enhances modeling capability (see Ta-        and edges of the graph as the attention biases, which were in-
ble 1). This design benefits us no longer the need to maintain   corporated into the transformer block. With the help of these
an FFN module specifically for edge embeddings, bringing         attention biases, Graphormer achieved leading performance
higher efficiency than previous dual-FFN methods.                on graph-level prediction tasks (e.g., classification and regres-
   The contributions of our work are as follows:                 sion on molecular graphs).
   (1) We propose an effective Graph Propagation Trans-
former (GPTrans), which can better model the relationship        2.2   Graph Convolutional Network
among nodes and edges and represent the graph.                   Graph Convolutional Network (GCN) is a kind of deep neu-
   (2) We introduce a novel attention mechanism in the trans-    ral network that extends the CNN from grid data (e.g., im-
former blocks, which explicitly passes the information among     age and video) to graph-structured data. Generally speak-
nodes and edges in three ways. These relationships play a        ing, GCN methods can be approximately divided into two
critical role in graph representation learning.                  types: spectral-based methods [Bruna et al., 2013; Defferrard
   (3) Extensive experiments show that the proposed GPTrans      et al., 2016; Henaff et al., 2015; Kipf and Welling, 2016] and
model outperforms many state-of-the-art transformer-based        non-spectral methods [Chen et al., 2018; Gilmer et al., 2017;
methods on benchmark datasets with better performance.           Scarselli et al., 2008; Veličković et al., 2017].
                                                                    Spectral GCN methods are designed under the theory of
2     Related Works                                              spectral graphs. For instance, Spectral GCN [Bruna et al.,
                                                                 2013] resorted to the Fourier basis of a graph to conduct con-
2.1    Transformer                                               volution operation in the spectral domain, which is the first
The past few years have seen many transformer-based mod-         work on spectral graph CNNs. Based on [Bruna et al., 2013],
els designed for various language [Vaswani et al., 2017;         Defferrard et al. [2016] designed a strict control over the lo-
Radford et al., 2019; Brown et al., 2020] and vision tasks       cal support of filters and avoided an explicit use of the Graph
[Parmar et al., 2018; Liu et al., 2021b]. For example, in the    Fourier basis in the convolution, which achieved better accu-
field of vision, Dosovitskiy et al. [2021] presented the Vi-     racy. Kipf and Welling [2016] adopted the first-order approx-
sion Transformer (ViT), which decomposed an image into           imation of the spectral graph convolution to simplify com-
a sequence of patches and captured their mutual relation-        monly used GNN.
ships. However, training ViT on large-scale datasets can            On the other hand, non-spectral methods directly define
be computationally expensive. To address this issue, DeiT        convolution operations on the graph data. GraphSage [Hamil-
[Touvron et al., 2021] proposed an efficient training strat-     ton et al., 2017] proposed learnable aggregator functions in
egy that enabled ViT to deliver exceptional performance even     the network to fuse neighbors’ information for effective graph
when trained on smaller datasets. Nevertheless, the complex-     representation learning. In GAT [Veličković et al., 2017], dif-
ity and performance of ViT remain challenging. To over-          ferent weight matrices are used for nodes with different de-
come these limitations, researchers further proposed many        grees for graph representation learning. In addition, another
well-designed models [Liu et al., 2021b; Wang et al., 2021;      line of GCN methods is mainly designed for specific graph-
Wang et al., 2022a; Chen et al., 2023; Ji et al., 2023;          level tasks. For example, some techniques such as subsam-
Chen et al., 2022; Wang et al., 2022b].                          pling [Chen et al., 2017] and inductive representation for a
                                                                                                                   ×𝐿


                           Graph             𝑥!"#$
                                                                       Graph Propagation          Feed-Forward                  Head
                         Embedding
                                                                           Attention                Network

      Graph                                  𝑥$#%$                               Transformer Blocks

Figure 2: Overall architecture of GPTrans. It contains a graph embedding layer, L transformer blocks, and a head. The graph embedding layer
transforms the graph data into node embeddings xnode and edge embeddings xedge , as the input of the transformer blocks. Each transformer
block comprises a Graph Propagation Attention (GPA) and a Feed-Forward Network (FFN). It is worth noting that we no longer need to
maintain an FFN module specifically for edge embeddings due to the proposed GPA module, which improves the efficiency of our method.
Finally, a head of two fully-connected layers is employed on the output embeddings for various graph tasks.



large graph [Hamilton et al., 2017] have been introduced for              in addition to the nodes, edges also have rich structural infor-
better graph representative learning.                                     mation in many types of graphs, e.g., molecular graphs [Hu et
                                                                          al., 2021] and social graphs [Huang et al., 2022]. Therefore,
3     GPTrans                                                             we encode both nodes and edges into embeddings to fully
                                                                          utilize the structure of the input graph.
3.1    Overall Architecture                                                  For nodes in the graph, we transform each node into node
An overview of the proposed GPTrans framework is depicted                 embedding. Specifically, we follow [Ying et al., 2021] to
in Figure 2. Specifically, it takes a graph G = (V, E) as input,          exploit the node attributes and the degree information, and
in which nodes V = {v1 , v2 , . . . , vn }, E indicates edges be-         add a virtual node [v0 ] into the graph to collect and prop-
tween nodes, and n means the number of nodes. The pipeline                agate graph-level features. Without loss of generality, tak-
of GPTrans can be divided into three parts: graph embedding,              ing a directed graph as an example, its node embeddings
transformer blocks, and prediction head.                                  xnode ∈ R(1+n)×d1 can be expressed as:
    In the graph embedding layer, for each given graph G, we
follow [Ying et al., 2021; Hussain et al., 2021] to add a virtual                      xnode = xnode attr + xdeg− + xdeg+ ,            (1)
node [v0 ] into V , to aggregate the information of the entire
graph. Thus, the newly-generated node set with the virtual                where xnode attr , xdeg− , and xdeg+ are embeddings encoded
node is represented as V ′ = {[v0 ], v1 , v2 , . . . , vn }, and the      from node attributes, indegree, and outdegree statistics, re-
number of nodes is updated to |V ′ | = 1 + n. For more ad-                spectively. d1 is the dimension of node embeddings.
equate information propagation across the whole graph, each                  For edges in the graph, we also transform them into
node and edge is treated as a token. In detail, we trans-                 edge embeddings to help the learning of graph representa-
form the input nodes into a sequence of node embeddings                   tion. In our implementation, the edge embeddings xedge ∈
xnode ∈ R(1+n)×d1 , and encode the input edges into a tensor              R(1+n)×(1+n)×d2 are defined as:
of edge embeddings xedge ∈ R(1+n)×(1+n)×d2 .
    Then, L transformer blocks with our re-designed self-                                  xedge = xedge attr + xrel pos ,             (2)
attention operation (i.e., Graph Propagation Attention) are ap-
plied to node embeddings and edge embeddings. Both these                  where xedge attr is encoded from the edge attributes, and
embeddings are fed throughout all transformer blocks. After               xrel pos is a relative positional encoding that embeds the spa-
that, GPTrans generates the representation of each node and               tial location of node pairs. d2 means the dimension of edge
edge, in which the output embedding of the virtual node takes             embeddings. We adopt the encoding of the shortest path dis-
along the representation of the whole graph.                              tance by default following [Ying et al., 2021]. In other words,
    Finally, the head of our GPTrans is composed of two fully-            for the position (i, j), xijedge ∈ R
                                                                                                                d2
                                                                                                                   represents the learned
connected (FC) layers. For graph-level tasks, we employ it                structural embedding of the edge (path) between node vi and
on top of the output embedding of the virtual node. For node-             node vj in the graph G.
level or edge(link)-level tasks, we apply it to the output node              It is worth noting that, unlike Graphormer [Ying et al.,
embeddings or edge embeddings. In summary, benefiting                     2021] that encodes edge attributes and spatial position as at-
from the proposed novel Graph Propagation Attention, our                  tention biases and shares them across all blocks, we optimize
GPTrans can better support various graph tasks with only a                the edge embeddings xedge in each transformer block by the
little additional computational cost compared to the previous             proposed Graph Propagation Attention. Then, the updated
method Graphormer [Ying et al., 2021].                                    edge embeddings are fed into the next block. Therefore, each
                                                                          block of our model could adaptively learn different ways to
3.2    Graph Embedding                                                    exploit edge features and propagate information. This more
The role of the graph embedding layer is to transform the                 flexible way is beneficial for graph representation learning,
graph data as the input of transformer blocks. As we know,                which we will show in later experiments.
                              ..
                             𝑥,-&%
                                                  ...
                                                 𝑥,-&%        .
                                                             𝑥%&/%               Unlike Graphormer [Ying et al., 2021] that used shared at-
                                                                                 tention biases in all blocks, we employ a parameter matrix
              Sum & FC                       𝑊-
                                                                                 Wreduce ∈ Rd2 ×nhead to predict layer-specific attention bi-
                                                                                 ases ϕ ∈ R(1+n)×(1+n)×nhead from the edge embeddings
                                                                                 xedge , which can be written as:
                 Softmax
                                                                                                      ϕ = xedge Wreduce .                   (4)
                      .
                     𝑥%&/%                        (c)
                                                                                 Then, we add ϕ to the attention map of the query-key dot
                 𝑊%)*+,&
                                                                                 product and compute the output node embeddings x′node .
                                                                                 This process can be formulated as:
                                                                     (b)

                 Softmax                  .                                                 QK T
                                         𝑥,-&%                                           A= √       + ϕ,       x′node = softmax(A)V,        (5)
                                                                                              dhead
                     𝐴
                                                                                 where A ∈ R(1+n)×(1+n)×nhead represents the output atten-
                                                                                 tion map, and dhead refers to the dimension of each head. In
            Product & Scale                                                      summary, since ϕ are projected from higher dimensional edge
             𝑄       𝐾               𝑉                       ∅                   embeddings by the learnable matrix, our attention map A will
              𝑊"      𝑊!        𝑊#                       𝑊$%&'(%                 have more flexible patterns to aggregate node features.
                                                                     (a)         Node-to-Edge
                                                                                 To propagate node features to edges, we make some addi-
              Node Embeddings                     Edge Embeddings                tional use of the attention map A. According to the defini-
                                                                                 tion of self-attention [Vaswani et al., 2017], attention map A
Figure 3: Illustration of Graph Propagation Attention. It explicitly             captures the similarity between node embeddings. Therefore,
builds three paths for information propagation among node embed-                 considering both local and global connections, we add A with
dings and edge embeddings, including (a) node-to-node, (b) node-                 its softmax confidence, and expand its dimension to as same
to-edge, and (c) edge-to-node.                                                   as xedge through the learnable matrix Wexpand ∈ Rnhead ×d2 .
                                                                                 This operation is designed to perform explicit high-order spa-
                                                                                 tial interactions, which can be written as follows:
3.3     Graph Propagation Attention                                                          x′edge = (A + softmax(A))Wexpand .             (6)
In recent years, many works [Ying et al., 2021; Shi et al.,                      In this way, we achieve node-to-edge propagation without re-
2022; Hussain et al., 2021] show global self-attention could                     lying on an additional FFN module like GT [Dwivedi and
serve as a flexible alternative to graph convolution and help                    Bresson, 2020] and EGT [Hussain et al., 2021].
better graph representation learning. However, most of them
                                                                                 Edge-to-Node
only consider part of the information propagation paths in
graph, or introduce a lot of extra computational overhead to                     In this part, we delve into the following question: How to
utilize edge information. For instance, Graphormer [Ying                         generate dynamic weights for edge embeddings xedge and
et al., 2021] only used edge features as shared bias terms                       fuse them into node embeddings xnode ? Due to the compu-
to refine the attention weights of nodes. GT [Dwivedi and                        tational efficiency, we do not additionally perform attention
Bresson, 2020] and EGT [Hussain et al., 2021] designed                           operation, but directly apply the softmax function to the just
dual-FFN networks to fuse edge features. Inspired by this,                       generated x′edge ∈ R(1+n)×(1+n)×d2 in Eqn. 6 and calculate
we introduce Graph Propagation Attention (GPA), as an ef-                        element-wise product with itself:
ficient replacement for vanilla self-attention in graph trans-
                                                                                  x′′node = FC(sum(x′edge · softmax(x′edge ), dim = 1)), (7)
formers. With an affordable cost, it could support three types
of propagation paths, including node-to-node, node-to-edge,                      in which the fully-connected (FC) layer is used to align the
and edge-to-node. For simplicity of description, we consider                     dimension of edge embeddings and node embeddings. This
single-head self-attention in the following formulas.                            process again explicitly introduces high-order spatial interac-
                                                                                 tions. Finally, we add these two types of node embeddings,
Node-to-Node                                                                     and employ a learnable matrix WO ∈ Rd1 ×d1 to fuse them.
Following common practices [Ying et al., 2021; Shi et al.,                       Then we have the updated node embeddings:
2022; Hussain et al., 2021], we adopt global self-attention
[Vaswani et al., 2017] to perform node-to-node propaga-                                         x′′′      ′       ′′
                                                                                                 node = (xnode + xnode )WO .                (8)
tion. First, we use parameter matrices WQ , WK , and WV ∈                        GPA in Transformer Blocks
Rd1 ×d1 to project the node embeddings xnode to queries Q,                       Equipped with our proposed GPA module, the block of our
keys K, and values V :                                                           GPTrans can be calculated as follows:
      Q = xnode WQ , K = xnode WK , V = xnode WV .                         (3)            x̂lnode , xledge += GPA(LN(xl−1      l−1
                                                                                                                      node ), xedge ),      (9)
                             PCQM4M↓                PCQM4Mv2↓              Model                                     #Param Test AP(%)↑
Model                #Param Validate Test          Validate Test-dev
                                                                           Non-transformer-based Methods
Non-transformer-based Methods                                              DeeperGCN-VN-FLAG [Li et al., 2020]        5.6M     28.42 ± 0.43
GCN                2.0M   0.1684          0.1838    0.1379    0.1398       PNA [Corso et al., 2020]                   6.5M     28.38 ± 0.35
GIN                3.8M   0.1536          0.1678    0.1195    0.1218       DGN [Beaini et al., 2021]                  6.7M     28.85 ± 0.30
GCN-VN             4.9M   0.1510          0.1579    0.1153    0.1152       GINE-VN [Brossard et al., 2020]            6.1M     29.17 ± 0.15
GIN-VN             6.7M   0.1396          0.1487    0.1083    0.1084       PHC-GNN [Le et al., 2021]                  1.7M     29.47 ± 0.26
GINE-VN           13.2M 0.1430              −         −         −          GIN-VN† [Xu et al., 2018]                  3.4M     29.02 ± 0.17
DeeperGCN-VN 25.5M 0.1398                   −         −         −
                                                                           Transformer-based Methods
Transformer-based Methods                                                  GRPE-Standard† [Park et al., 2022]         46.2M    30.77 ± 0.07
GPS-Small         6.2M      −               −       0.0938       −         GPTrans-B† (ours)                          45.7M    31.15 ± 0.16
GPTrans-T (ours) 6.6M     0.1179            −       0.0833       −
                                                                           GRPE-Large† [Park et al., 2022]           118.3M    31.50 ± 0.10
Graphormer-S          12.5M     0.1264      −       0.0910       −
                                                                           Graphormer-L† [Ying et al., 2021]         119.5M    31.39 ± 0.32
EGT-Small             11.5M     0.1260      −       0.0899       −
GPS-Medium            19.4M       −         −       0.0858       −         EGT-Larger† [Hussain et al., 2021]        110.8M    29.61 ± 0.24
GPTrans-S (ours)      13.6M     0.1162      −       0.0823       −         GPTrans-L† (ours)                         86.0M     32.43 ± 0.22

TokenGT               48.5M       −         −       0.0910      −
                                                                           Table 2: Results on MolPCBA. † indicates the model is pre-trained
Graphormer-B          47.1M     0.1234      −       0.0906      −
                                                                           on PCQM4M or PCQM4Mv2. The higher the better. Highlighted
GRPE-Standard         46.2M     0.1225      −       0.0890    0.0898
                                                                           are the best results for each model size.
EGT-Medium            47.4M     0.1224      −       0.0881      −
GPTrans-B (ours)      45.7M     0.1153      −       0.0813      −
GT-Wide          83.2M          0.1408      −         −         −
GraphormerV2-L 159.3M           0.1228      −       0.0883      −           Model                                 #Param Test AUC(%)↑
EGT-Large        89.3M            −         −       0.0869    0.0872        Non-transformer-based Methods
EGT-Larger       110.8M           −         −       0.0859      −           DeeperGCN-FLAG [Li et al., 2020]       532K       79.42 ± 1.20
GRPE-Large       118.3M           −         −       0.0867    0.0876        PNA [Corso et al., 2020]               326K       79.05 ± 1.32
GPS-Deep         138.1M           −         −       0.0852    0.0862        DGN [Beaini et al., 2021]              110K       79.70 ± 0.97
GPTrans-L (ours) 86.0M          0.1151      −       0.0809    0.0821        PHC-GNN [Le et al., 2021]              114K       79.34 ± 1.16
                                                                            GIN-VN† [Xu et al., 2018]              3.3M       77.80 ± 1.82
Table 1: Results on PCQM4M and PCQM4Mv2. The metric is the
                                                                            Transformer-based Methods
Mean Absolute Error (MAE), and the lower the better. “−” denotes
results are not available since the labels of test and test-dev sets are    Graphormer-B† [Ying et al., 2021]     47.0M       80.51 ± 0.53
not public. Highlighted are the best results for each model size.           EGT-Larger† [Hussain et al., 2021]    110.8M      80.60 ± 0.65
                                                                            GRPE-Standard† [Park et al., 2022]    46.2M       81.39 ± 0.49
                                                                            GPTrans-B† (ours)                     45.7M       81.26 ± 0.32

              xlnode = FFN(LN(x̂lnode )) + x̂lnode ,               (10)    Table 3: Results on MolHIV. † indicates the model is pre-trained on
where LN(·) means layer normalization [Ba et al., 2016].                   PCQM4M or PCQM4Mv2. The higher the better. Highlighted are
                                                                           the best results.
x̂lnode and xledge denote the output node embeddings and edge
embeddings of the GPA module for block l. And xlnode repre-
sents the output node embeddings of the FFN module. Over-
all, our GPA module effectively extends the ability of our                     • GPTrans-Small: d1 = 384, d2 = 48, layer number = 12
GPTrans to various graph tasks, but only introduces a small
                                                                               • GPTrans-Base: d1 = 608, d2 = 76, layer number = 18
amount of extra overhead compared with previous methods
[Hussain et al., 2021; Ying et al., 2021].                                     • GPTrans-Large: d1 = 736, d2 = 92, layer number = 24
                                                                           The model size and performance of the model variants on
3.4    Architecture Configurations
                                                                           the large-scale PCQM4M and PCQM4Mv2 benchmarks [Hu
We build five variants of the proposed model with different                et al., 2021] are listed in Table 1, and the analysis of model
model sizes, namely GPTrans-Nano, Tiny, Small, Base, and                   efficiency is provided in Table 6. More detailed model con-
Large. Note that the number of parameters of our GPTrans                   figurations are presented in the appendix.
is similar to Graphormer [Ying et al., 2021] and EGT [Hus-
sain et al., 2021]. The dimension of each head is set to 10
for our nano model, and 32 for others. Following common
                                                                           4     Experiments
practices, the expansion ratio of the FFN module is α = 1                  4.1    Graph-Level Tasks
for all model variants. The architecture hyper-parameters of
these five models are as follows:                                          Datasets
                                                                           We verify the following graph-level tasks:
   • GPTrans-Nano: d1 = 80, d2 = 40, layer number = 12                     (1) PCQM4M [Hu et al., 2021] is a quantum chemistry
   • GPTrans-Tiny: d1 = 256, d2 = 32, layer number = 12                    dataset that includes 3.8 million molecular graphs and a total
                                                               ZINC             PATTERN             CLUSTER                TSP
  Model                                       #Param         Test MAE↓         Accuracy(%)↑        Accuracy(%)↑          F1-Score↑
  Non-transformer-based Methods
  GCN [Kipf and Welling, 2016]                  505K        0.367 ± 0.011      71.892 ± 0.334      68.498 ± 0.976            −
  GraphSage [Hamilton et al., 2017]             505K        0.398 ± 0.002      50.492 ± 0.001      63.884 ± 0.110            −
  GIN [Xu et al., 2018]                         510K        0.526 ± 0.051      85.387 ± 0.136      64.716 ± 1.553            −
  GAT [Veličković et al., 2017]               531K        0.384 ± 0.007      78.271 ± 0.186      70.587 ± 0.447            −
  GatedGCN [Bresson and Laurent, 2017]          505K        0.214 ± 0.013      86.508 ± 0.085      76.082 ± 0.196      0.838 ± 0.002
  PNA [Corso et al., 2020]                      387K        0.142 ± 0.010            −                   −                   −
  Transformer-based Methods
  GT [Dwivedi and Bresson, 2020]                589K        0.226 ± 0.014      84.808 ± 0.068      73.169 ± 0.622            −
  SAN [Kreuzer et al., 2021]                    509K        0.139 ± 0.006      86.581 ± 0.037      76.691 ± 0.650            −
  Graphormer-Slim [Ying et al., 2021]           489K        0.122 ± 0.006      86.650 ± 0.033      74.660 ± 0.236      0.698 ± 0.007
  EGT [Hussain et al., 2021]                    500K        0.108 ± 0.009      86.821 ± 0.020      79.232 ± 0.348      0.853 ± 0.001
  GPS [Rampášek et al., 2022]                 424K        0.070 ± 0.004      86.685 ± 0.059      78.016 ± 0.180            −
  GPTrans-Nano (ours)                           554K        0.077 ± 0.009      86.731 ± 0.085      78.069 ± 0.154      0.832 ± 0.004

Table 4: Results on four benchmarking datasets, including graph regression (ZINC), node classification (PATTERN and CLUSTER), and
edge classification (TSP) tasks. The arrow next to the metric means higher or lower is better. “−” denotes the results are not available.
Highlighted are the top first and second results.



of 53 million nodes. The task is to regress a DFT-calculated             Further, we take the PCQM4Mv2 pre-trained weights as
quantum chemical property, e.g., HOMO-LUMO energy gap.                the initialization and fine-tune our models on the OGB molec-
(2) PCQM4Mv2 [Hu et al., 2021] is an updated version of               ular datasets MolPCBA and MolHIV, to verify the transfer
PCQM4M, in which the number of molecules slightly de-                 learning capability of GPTrans. All experiments are per-
creased, and some of the graphs are revised.                          formed five times with different random seeds, and we report
(3) MolHIV [Hu et al., 2020] is a small-scale molecular prop-         the mean and standard deviation of the results. From Table 2
erty prediction dataset. It has 41, 127 graphs with a total of        and 3, we can see that GPTrans outperforms many strong
1, 048, 738 nodes and 1, 130, 993 edges.                              counterparts, such as GRPE [Park et al., 2022], EGT [Hus-
(4) MolPCBA [Hu et al., 2020] is another property prediction          sain et al., 2021], and Graphormer [Ying et al., 2021].
dataset, which is larger than MolHIV. It contains 437, 929               Moreover, we follow previous methods [Park et al., 2022;
graphs with 11, 386, 154 nodes and 12, 305, 805 edges.                Ying et al., 2021] to train the GPTrans-Nano model with
(5) ZINC [Dwivedi et al., 2020] is a popular real-world               about 500K parameters on the ZINC subset from scratch. As
molecular dataset for graph property regression. It has               demonstrated in Table 4, our model achieves a promising test
10, 000 train, 1, 000 validation, and 1, 000 test graphs.             MAE of 0.077 ± 0.009, bringing 36.9% relative MAE de-
Settings                                                              cline compared to Graphormer [Ying et al., 2021]. The above
                                                                      inspiring results show that the proposed GPTrans performs
For the large-scale PCQM4M and PCQM4Mv2 datasets, we
                                                                      well on graph-level tasks.
use AdamW [Loshchilov and Hutter, 2018] with an initial
learning rate of 1e-3 as the optimizer. Following common              4.2    Node-Level Tasks
practice, we adopt a cosine decay learning rate scheduler with
a 20-epoch warmup. All models are trained for 300 epochs              Datasets
with a total batch size of 1024. When fine-tuning the MolHIV          PATTERN and CLUSTER [Dwivedi et al., 2020] are both
and MolPCBA datasets, we load the PCQM4Mv2 pre-trained                synthetic datasets for node classification. Specifically, PAT-
weights as initialization. For the ZINC dataset, we train our         TERN has 10, 000 training, 2, 000 validation, and 2, 000 test
GPTrans-Nano model from scratch. More detailed training               graphs, and CLUSTER contains 10, 000 training, 1, 000 vali-
strategies are provided in the appendix.                              dation, and 1, 000 test graphs.
Results                                                               Settings
First, we benchmark our GPTrans method on PCQM4M and                  For the PATTERN and CLUSTER datasets, we train our
PCQM4Mv2, two datasets from OGB large-scale challenge                 GPTrans-Nano up to 1000 epochs with a batch size of 256.
[Hu et al., 2021]. We mainly compare our GPTrans against              We employ the AdamW [Loshchilov and Hutter, 2018] op-
a set of representative transformer-based methods, including          timizer with a 20-epoch warmup. The learning rate is ini-
GT [Dwivedi and Bresson, 2020], Graphormer [Ying et al.,              tialized to 5e-4, and is declined by a cosine scheduler. More
2021], GRPE [Park et al., 2022], EGT [Hussain et al., 2021],          training details can be found in the appendix.
GPS [Rampášek et al., 2022], and TokenGT [Kim et al.,
2022]. As reported in Table 1, our method yields the state-of-        Results
the-art validate MAE score on both datasets across different          In this part, we compare our GPTrans-Nano with various
model complexities.                                                   GCN variants and recent graph transformers. As shown in
 Model                      #Param FLOPs Validate MAE↓                                        Train      Inference PCQM4Mv2
                                                                      Model          #Param (min / ep.) (graph / s) Validate MAE↓
 Baseline (Graphormer-S)     12.5M     0.399G        0.0928
 + Node-to-Node              13.3M     0.402G        0.0874           EGT-Small       11.5M       7.6      10291.8        0.0899
 ++ Node-to-Edge             13.3M     0.405G        0.0865           GPTrans-S       13.6M       5.5      11391.2        0.0823
 +++ Edge-to-Node            13.5M     0.417G        0.0854
                                                                      EGT-Medium 47.4M           11.3       4840.8        0.0881
 GPTrans-Swider              13.5M     0.417G        0.0854           GPTrans-B  45.7M           7.7        6670.6        0.0813
 GPTrans-Sdeeper (ours)      13.6M     0.472G        0.0835
                                                                      EGT-Large       89.3M      15.5       3759.4        0.0869
                                                                      GPTrans-L       86.0M      9.6        4193.4        0.0809
Table 5: Ablation studies of GPTrans. FLOPs is calculated using a
graph with 20 nodes. We build our baseline based on Graphormer-S
                                                                      Table 6: Efficiency analysis of GPTrans. These experiments are
with a shorter schedule of 100 epochs, and decline its validate MAE
                                                                      conducted with PyTorch1.12 and CUDA11.3. Training time is mea-
on the PCQM4Mv2 dataset from 0.0928 to 0.0854 by gradually in-
                                                                      sured on 8 A100 GPUs with half-precision training, and the infer-
troducing our GPA module. Moreover, we find that the deeper model
                                                                      ence throughput is tested on a single A100 GPU.
outperforms the wider model with a similar number of parameters.



Table 4, our GPTrans-Nano produces the promising accuracy             with a shorter schedule of 100 epochs. Other settings are the
of 86.731 ± 0.085% and 78.069 ± 0.154% on the PATTERN                 same as described in Section 4.1.
and CLUSTER datasets, respectively. These results out-
perform many Convolutional/Message-Passing Graph Neu-                 Graph Propagation Attention
ral Networks by large margins, showing that the proposed              To investigate the contribution of each key design in our GPA
GPTrans can serve as an alternative to traditional GCNs for           module, we gradually extend the Graphormer baseline [Ying
node-level tasks. Moreover, we find that our method exceeds           et al., 2021] to our GPTrans. As shown in Table 5, the model
Graphormer [Ying et al., 2021] on the CLUSTER dataset                 gives the best performance when all three information prop-
by significant gaps of 3.4% accuracy, which suggests that             agation paths are introduced. It is worth noting that the im-
the three propagation ways explicitly constructed in the GPA          provement from our node-to-node propagation is most signif-
module are also helpful for node-level tasks.                         icant, thanks to learning the attention biases for a particular
                                                                      layer rather than sharing them across all layers. In summary,
4.3   Edge-Level Tasks                                                our proposed GPA module collectively brings a large gain to
Datasets                                                              Graphormer, i.e., 8.0% relative validate MAE decline on the
TSP [Dwivedi et al., 2020] is a dataset for the Traveling             PCQM4Mv2 dataset.
Salesman Problem, which is an NP-hard combinatorial op-
timization problem. The problem is reduced to a binary edge           Deeper vs. Wider
classification task, where edges in the TSP tour have positive        Here we explore the question of whether the transformers for
labels. TSP dataset has 10, 000 training, 1, 000 validation,          graph representation learning should go deeper or wider. For
and 1, 000 test graphs.                                               fair comparisons, we build a deeper but thinner model un-
Settings                                                              der comparable parameter numbers, by increasing the depth
                                                                      from 6 to 12 layers and decreasing the width from 512 to 384
We experiment on the TSP dataset in a similar setting to that
                                                                      dimensions. As reported in Table 5, the validate MAE of the
used in the PATTERN and CLUSTER datasets. Details are
                                                                      PCQM4Mv2 dataset is declined from 0.0854 to 0.0835 by the
shown in the appendix.
                                                                      deeper model, which shows that depth is more important than
Results                                                               width for graph transformers. Based on this observation, we
Table 4 compares the edge classification performance of               prefer to develop GPTrans with a large model depth.
our GPTrans-Nano model and previously transformer-based
methods on the TSP dataset. We observe GPTrans can outper-            Efficiency Analysis
form Graphormer [Ying et al., 2021] with a large margin and           As shown in Table 6, we benchmark the training time and
is comparable with EGT [Hussain et al., 2021], showing that           inference throughputs of our GPTrans and EGT [Hussain
the proposed GPA module design is competitive when used               et al., 2021]. Specifically, we employ PyTorch1.12 and
for edge-level tasks. By applying the GPA module, we avoid            CUDA11.3 to perform these experiments. For a fair com-
designing an inefficient dual-FFN network, which boosts the           parison, the training time of these two methods is measured
efficiency of our method. We will analyze the efficiency of           using 8 Nvidia A100 GPUs with half-precision training and
GPTrans in detail in Section 4.4.                                     a total batch size of 1024. The inference throughputs of
                                                                      PCQM4Mv2 models in Table 6 are tested using an A100 GPU
4.4   Ablation Study                                                  with a batch size of 128, where our GPTrans is slightly faster
We conduct several ablation studies on the PCQM4Mv2 [Hu               in inference than EGT under a similar number of parameters.
et al., 2021] dataset, to validate the effectiveness of each key      This preliminary study shows a good signal that the proposed
design in our GPTrans. Due to the limited computational re-           GPTrans, equipped with the GPA module, could be an effi-
sources, we adopt GPTrans-S as the base model, and train it           cient model for graph representation learning.
5    Conclusion                                                        [Chen et al., 2022] Guo Chen, Sen Xing, Zhe Chen, Yi Wang, Kun-
                                                                          chang Li, Yizhuo Li, Yi Liu, Jiahao Wang, Yin-Dong Zheng,
This paper aims for graph representation learning with a
                                                                          Bingkun Huang, et al. Internvideo-ego4d: A pack of champion
Graph Propagation Transformer (GPTrans), which explores                   solutions to ego4d challenges. arXiv preprint arXiv:2211.09529,
the information propagation among nodes and edges in a                    2022. 2
graph when establishing the self-attention mechanism in the
                                                                       [Chen et al., 2023] Zhe Chen, Yuchen Duan, Wenhai Wang, Jun-
transformer block. Especially in the GPTrans, we propose a
                                                                          jun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer
Graph Propagation Attention (GPA) mechanism to explicitly                 adapter for dense predictions. In International Conference on
pass the information among nodes and edges in three ways,                 Learning Representations, 2023. 2
i.e., node-to-node, node-to-edge, and edge-to-node, which is
                                                                       [Corso et al., 2020] Gabriele Corso, Luca Cavalleri, Dominique
essential for learning graph-structured data. Extensive com-
                                                                          Beaini, Pietro Liò, and Petar Veličković. Principal neighbour-
parisons with state-of-the-art methods on several benchmark               hood aggregation for graph nets. Proceedings of Advances in
datasets demonstrate the superior capability of the proposed              Neural Information Processing Systems, 33:13260–13271, 2020.
GPTrans with better performance.                                          5, 6
                                                                       [Defferrard et al., 2016] Michaël Defferrard, Xavier Bresson, and
Contribution Statement                                                    Pierre Vandergheynst. Convolutional neural networks on graphs
Zhe Chen and Hao Tan contributed equally to this work.                    with fast localized spectral filtering. In Proceedings of Advances
                                                                          in Neural Information Processing Systems, 2016. 2
Acknowledgements                                                       [Dosovitskiy et al., 2021] Alexey Dosovitskiy, Lucas Beyer,
                                                                          Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
This work is supported by the Natural Science Foundation of               Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
China under Grant 61672273 and Grant 61832008.                            Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16
                                                                          words: Transformers for image recognition at scale. In Proceed-
References                                                                ings of International Conference on Machine Learning, 2021.
[Ba et al., 2016] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E          2
   Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,       [Dwivedi and Bresson, 2020] Vijay Prakash Dwivedi and Xavier
   2016. 5                                                                Bresson. A generalization of transformer networks to graphs.
[Beaini et al., 2021] Dominique Beaini, Saro Passaro, Vincent             arXiv preprint arXiv:2012.09699, 2020. 2, 4, 6
   Létourneau, Will Hamilton, Gabriele Corso, and Pietro Liò. Di-    [Dwivedi et al., 2020] Vijay Prakash Dwivedi, Chaitanya K Joshi,
   rectional graph networks. In Proceedings of International Con-         Thomas Laurent, Yoshua Bengio, and Xavier Bresson.
   ference on Machine Learning, pages 748–758. PMLR, 2021. 5              Benchmarking graph neural networks.                arXiv preprint
[Bresson and Laurent, 2017] Xavier Bresson and Thomas Lau-                arXiv:2003.00982, 2020. 6, 7, 10, 11
   rent.      Residual gated graph convnets.        arXiv preprint     [Gilmer et al., 2017] Justin Gilmer, Samuel S Schoenholz,
   arXiv:1711.07553, 2017. 1, 6                                           Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural
[Brossard et al., 2020] Rémy Brossard, Oriel Frigo, and David De-        message passing for quantum chemistry. In Proceedings of In-
   haene. Graph convolutions that can finally model local structure.      ternational Conference on Machine Learning, pages 1263–1272,
   arXiv preprint arXiv:2011.15069, 2020. 1, 5                            2017. 2
[Brown et al., 2020] Tom Brown, Benjamin Mann, Nick Ryder,             [Hamilton et al., 2017] Will Hamilton, Zhitao Ying, and Jure
   Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind             Leskovec. Inductive representation learning on large graphs. In
   Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.        Proceedings of Advances in Neural Information Processing Sys-
   Language models are few-shot learners. Advances in Neural In-          tems, 2017. 2, 3, 6
   formation Processing Systems, 33:1877–1901, 2020. 2                 [Henaff et al., 2015] Mikael Henaff, Joan Bruna, and Yann LeCun.
[Bruna et al., 2013] Joan Bruna, Wojciech Zaremba, Arthur Szlam,          Deep convolutional networks on graph-structured data. arXiv
   and Yann LeCun. Spectral networks and locally connected net-           preprint arXiv:1506.05163, 2015. 2
   works on graphs. arXiv preprint arXiv:1312.6203, 2013. 2            [Hu et al., 2020] Weihua Hu, Matthias Fey, Marinka Zitnik, Yux-
[Cai and Lam, 2020] Deng Cai and Wai Lam. Graph transformer               iao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure
   for graph-to-sequence learning. In Proceedings of the AAAI Con-        Leskovec. Open graph benchmark: Datasets for machine learn-
   ference on Artificial Intelligence, pages 7464–7471, 2020. 1, 2        ing on graphs. In Proceedings of Advances in Neural Information
[Chen et al., 2017] Jianfei Chen, Jun Zhu, and Le Song. Stochastic        Processing Systems, 2020. 6, 10
   training of graph convolutional networks with variance reduction.   [Hu et al., 2021] Weihua Hu, Matthias Fey, Hongyu Ren, Maho
   arXiv preprint arXiv:1710.10568, 2017. 2                               Nakata, Yuxiao Dong, and Jure Leskovec. Ogb-lsc: A large-scale
[Chen et al., 2018] Jie Chen, Tengfei Ma, and Cao Xiao. Fastgcn:          challenge for machine learning on graphs. In Proceedings of
   fast learning with graph convolutional networks via importance         Conference on Neural Information Processing Systems Datasets
   sampling. In Proceedings of International Conference on Learn-         and Benchmarks Track, 2021. 3, 5, 6, 7, 10
   ing Representations, 2018. 2                                        [Huang et al., 2022] Xuanwen Huang, Yang Yang, Yang Wang,
[Chen et al., 2020] Deli Chen, Yankai Lin, Wei Li, Peng Li, Jie           Chunping Wang, Zhisheng Zhang, Jiarong Xu, and Lei Chen.
   Zhou, and Xu Sun. Measuring and relieving the over-smoothing           Dgraph: A large-scale financial dataset for graph anomaly detec-
   problem for graph neural networks from the topological view. In        tion. arXiv preprint arXiv:2207.03579, 2022. 3
   Proceedings of the AAAI Conference on Artificial Intelligence,      [Hussain et al., 2021] Md Shamim Hussain, Mohammed J Zaki,
   pages 3438–3445, 2020. 1                                               and Dharmashankar Subramanian.                Global self-attention
   as a replacement for graph convolution.           arXiv preprint    [Scarselli et al., 2008] Franco Scarselli, Marco Gori, Ah Chung
   arXiv:2108.03348, 2021. 1, 2, 3, 4, 5, 6, 7, 10                        Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The
[Ji et al., 2023] Yuanfeng Ji, Zhe Chen, Enze Xie, Lanqing Hong,          graph neural network model. IEEE Transactions on Neural Net-
                                                                          works, 20(1):61–80, 2008. 2
    Xihui Liu, Zhaoqiang Liu, Tong Lu, Zhenguo Li, and Ping Luo.
    Ddp: Diffusion model for dense visual prediction. arXiv preprint   [Shi et al., 2022] Yu Shi, Shuxin Zheng, Guolin Ke, Yifei Shen, Ji-
    arXiv:2303.17559, 2023. 2                                             acheng You, Jiyan He, Shengjie Luo, Chang Liu, Di He, and Tie-
                                                                          Yan Liu. Benchmarking graphormer on large-scale molecular
[Kim et al., 2022] Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min,
                                                                          modeling datasets. arXiv preprint arXiv:2203.04810, 2022. 4
   Sungjun Cho, Moontae Lee, Honglak Lee, and Seunghoon Hong.
   Pure transformers are powerful graph learners. arXiv preprint       [Touvron et al., 2021] Hugo Touvron, Matthieu Cord, Matthijs
   arXiv:2207.02505, 2022. 6                                              Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé
                                                                          Jégou. Training data-efficient image transformers & distillation
[Kipf and Welling, 2016] Thomas N Kipf and Max Welling. Semi-             through attention. In Proceedings of International Conference on
   supervised classification with graph convolutional networks.           Machine Learning, pages 10347–10357, 2021. 2
   arXiv preprint arXiv:1609.02907, 2016. 1, 2, 6
                                                                       [Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Par-
[Kreuzer et al., 2021] Devin Kreuzer, Dominique Beaini, Will              mar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
   Hamilton, Vincent Létourneau, and Prudencio Tossou. Rethink-          Kaiser, and Illia Polosukhin. Attention is all you need. In Pro-
   ing graph transformers with spectral attention. In Proceedings         ceedings of Advances in Neural Information Processing Systems,
   of Advances in Neural Information Processing Systems, pages            2017. 2, 4
   21618–21629, 2021. 6
                                                                       [Veličković et al., 2017] Petar Veličković, Guillem Cucurull, Aran-
[Le et al., 2021] Tuan Le, Marco Bertolini, Frank Noé, and Djork-        txa Casanova, Adriana Romero, Pietro Lio, and Yoshua Ben-
   Arné Clevert. Parameterized hypercomplex graph neural net-            gio. Graph attention networks. arXiv preprint arXiv:1710.10903,
   works for graph classification. In Proceedings of International        2017. 2, 6
   Conference on Artificial Neural Networks, pages 204–216, 2021.      [Wang et al., 2021] Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping
   5
                                                                          Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling
[Li et al., 2020] Guohao Li, Chenxin Xiong, Ali Thabet, and               Shao. Pyramid vision transformer: A versatile backbone for
   Bernard Ghanem. Deepergcn: All you need to train deeper gcns.          dense prediction without convolutions. In Proceedings of the
   arXiv preprint arXiv:2006.07739, 2020. 5                               IEEE International Conference on Computer Vision, pages 568–
[Liu et al., 2021a] Meng Liu, Zhengyang Wang, and Shuiwang Ji.            578, 2021. 2
   Non-local graph neural networks. IEEE Transactions on Pattern       [Wang et al., 2022a] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhen-
   Analysis and Machine Intelligence, 2021. 1                             hang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei
                                                                          Lu, Hongsheng Li, et al. Internimage: Exploring large-scale
[Liu et al., 2021b] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan
                                                                          vision foundation models with deformable convolutions. arXiv
   Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin trans-            preprint arXiv:2211.05778, 2022. 2
   former: Hierarchical vision transformer using shifted windows.
   In Proceedings of the IEEE International Conference on Com-         [Wang et al., 2022b] Yi Wang, Kunchang Li, Yizhuo Li, Yinan He,
   puter Vision, pages 10012–10022, 2021. 2                               Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu,
                                                                          Zun Wang, et al. Internvideo: General video foundation mod-
[Loshchilov and Hutter, 2018] Ilya Loshchilov and Frank Hutter.           els via generative and discriminative learning. arXiv preprint
   Decoupled weight decay regularization. In Proceedings of In-           arXiv:2212.03191, 2022. 2
   ternational Conference on Learning Representations, 2018. 6,
   11                                                                  [Xu et al., 2018] Keyulu Xu, Weihua Hu, Jure Leskovec, and Ste-
                                                                          fanie Jegelka. How powerful are graph neural networks? arXiv
[Park et al., 2022] Wonpyo Park, Woong-Gi Chang, Donggeon                 preprint arXiv:1810.00826, 2018. 1, 5, 6
   Lee, Juntae Kim, et al. Grpe: Relative positional encoding for
                                                                       [Ying et al., 2021] Chengxuan Ying, Tianle Cai, Shengjie Luo,
   graph transformer. In Proceedings of ICLR2022 Machine Learn-
   ing for Drug Discovery, 2022. 5, 6                                     Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan
                                                                          Liu. Do transformers really perform badly for graph represen-
[Parmar et al., 2018] Niki Parmar, Ashish Vaswani, Jakob Uszko-           tation? In Proceedings of Advances in Neural Information Pro-
   reit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin            cessing Systems, pages 28877–28888, 2021. 1, 2, 3, 4, 5, 6, 7,
   Tran. Image transformer. In Proceedings of the International           11
   Conference on Machine Learning, pages 4055–4064, 2018. 2            [Zhang et al., 2019] Chuxu Zhang, Ananthram Swami, and
[Perozzi et al., 2014] Bryan Perozzi, Rami Al-Rfou, and Steven            Nitesh V Chawla. Shne: Representation learning for semantic-
   Skiena. Deepwalk: Online learning of social representations.           associated heterogeneous networks. In Proceedings of ACM
   In Proceedings of ACM SIGKDD International Conference on               International Conference on Web Search and Data Mining,
   Knowledge Discovery and Data Mining, pages 701–710, 2014. 1            pages 690–698, 2019. 1
[Radford et al., 2019] Alec Radford, Jeffrey Wu, Rewon Child,          [Zhang et al., 2020] Jiawei Zhang, Haopeng Zhang, Congying Xia,
  David Luan, Dario Amodei, Ilya Sutskever, et al. Language               and Li Sun. Graph-bert: Only attention is needed for learning
  models are unsupervised multitask learners. OpenAI blog, 1(8):9,        graph representations. arXiv preprint arXiv:2001.05140, 2020. 2
  2019. 2
[Rampášek et al., 2022] Ladislav Rampášek, Mikhail Galkin, Vi-
  jay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique
  Beaini. Recipe for a general, powerful, scalable graph trans-
  former. arXiv preprint arXiv:2205.12454, 2022. 1, 6
    Dataset                                #Graphs       Avg. Nodes       Avg. Edges           Task Type                    Metric
    PCQM4M [Hu et al., 2021]              3,803,453          14.1             14.6         Graph Regression          Mean Absolute Error
    PCQM4Mv2 [Hu et al., 2021]            3,746,619          14.1             14.6         Graph Regression          Mean Absolute Error
    MolHIV [Hu et al., 2021]               41,127            25.5             27.5        Graph Classification           ROC-AUC
    MolPCBA [Hu et al., 2021]              437,929           26.0             28.1        Graph Classification        Average Precision
    ZINC [Dwivedi et al., 2020]            12,000            23.2             49.8         Graph Regression          Mean Absolute Error
    PATTERN [Dwivedi et al., 2020]          14,000          117.5            4749.2        Node Classification             Accuracy
    CLUSTER [Dwivedi et al., 2020]          12,000          117.2            4301.7        Node Classification             Accuracy
    TSP [Dwivedi et al., 2020]              12,000          275.6            6894.0        Edge Classification             F1 Score

              Table 7: Overview of the datasets used in our experiments, covering graph-level, node-level, and edge-level tasks.



A     Datasets                                                               Hyper-parameter                Nano/Tiny/Small/Base/Large
In Table 7, we list the information of the datasets used for                 #Layers                                12/12/12/18/24
training and evaluation. Next, we describe them in detail.                   Dimension d1                         80/256/384/608/736
                                                                             Dimension d2                           40/32/48/76/92
PCQM4M [Hu et al., 2021] is a large-scale molecular dataset                  FFN Ratio α                                  1.0
that includes 3.8 million molecular graphs and a total of                    #Attention Head                         8/8/12/19/23
53 million nodes. The task is to regress a DFT-calculated                    Dimension of Each Head                 10/32/32/32/32
quantum chemical property, e.g., the HOMO-LUMO energy                        Layer Scale                              ✗/✓/✓/✓/✓
gap of a given molecule. The HOMO-LUMO gap is one of
the most practically relevant quantum chemical properties of             Table 8: Model configurations of GPTrans. We build 5 variants of
molecules. Using efficient and accurate deep learning mod-               GPTrans with different model sizes, namely GPTrans-Nano, Tiny,
els to approximate DFT enables diverse downstream applica-               Small, Base, and Large.
tions, e.g., drug discovery.
PCQM4Mv2 [Hu et al., 2021] is an updated version of
PCQM4M. In PCQM4Mv2, the number of molecules slightly                          Hyper-parameter                   Tiny/Small/Base/Large
decreased, and some of the graphs were revised.                                FFN Dropout                          0.1/0.1/0.1/0.2
                                                                               Embedding Dropout                    0.1/0.1/0.1/0.2
MolHIV [Hu et al., 2020] is a small-scale molecular prop-                      Attention Dropout                    0.1/0.1/0.1/0.2
erty prediction dataset, and is part of the Open Graph Bench-                  Drop Path Rate                       0.1/0.1/0.2/0.4
mark [Hu et al., 2021]. Specifically, the MolHIV dataset is a                  Max Epochs                                 300
molecular tree-like dataset consisting of 41, 127 graphs, with                 Warm-up Epochs                             20
an average number of 25.5 nodes and 27.5 edges per graph.                      Peak Learning Rate                        1e-3
The task is to predict whether a molecule inhibits HIV virus                   Batch Size                                1024
replication.                                                                   Learning Rate Decay                      Cosine
                                                                               Adam ϵ                                    1e-8
MolPCBA [Hu et al., 2020] is a medium-scale molecu-                            Adam (β2 , β2 )                       (0.9, 0.999)
lar property prediction dataset featuring 128 imbalanced bi-                   Weight Decay                              0.05
nary classification tasks. It contains 437, 929 graphs with                    EMA                                         ✓
11, 386, 154 nodes and 12, 305, 805 edges.
                                                                             Table 9: Hyper-parameters on PCQM4M and PCQM4Mv2.
ZINC [Dwivedi et al., 2020] is the most popular real-world
molecular dataset to predict graph property regression for
constrained solubility, which is an important chemical prop-
erty for designing generative GNNs for molecules. To be spe-             where edges in graphs have binary labels corresponding to
cific, ZINC has 12,000 graphs, usually used as a benchmark               the TSP tour of that graph. Specifically, the label of an edge
for evaluating GNN performances.                                         is set to 1 if it belongs to the TSP tour and is set to 0 other-
                                                                         wise. The TSP problem is one of the NP-hard combinatorial
PATTERN and CLUSTER [Dwivedi et al., 2020] are node
                                                                         optimization problems, and the utilization of machine learn-
classification datasets synthesized with Stochastic Block
                                                                         ing methods to solve them has been intensively researched in
Model. In PATTERN, the task is to tell if a node belongs to
                                                                         recent years.
one of the randomly generated 100 patterns in a large graph.
In CLUSTER, every graph comprises 6 SBM clusters, and
each graph only contains one labeled node with a feature                 B     Model Configurations
value set to the cluster ID. The task is to predict the cluster
                                                                         We build 5 variants of GPTrans with different model sizes,
ID of every node.
                                                                         called GPTrans-Nano, Tiny (T), Small (S), Base (B), and
TSP [Dwivedi et al., 2020] is an edge classification dataset,            Large (L). Specifically, we follow EGT [Hussain et al., 2021]
    Dataset                     MolHIV           MolPCBA         times with 5 different random seeds, and the results are used
    Model                        Base            Base/Large      to calculate the mean and standard deviations of the metric.
    Initialization            PCQM4Mv2          PCQM4Mv2
    FFN Dropout                     0.1            0.1/0.2
                                                                 C.3   ZINC/PATTERN/CLUSTER/TSP
    Embedding Dropout               0.1            0.1/0.2       For the 4 benchmarking datasets from [Dwivedi et al., 2020],
    Attention Dropout               0.1            0.1/0.2       i.e., ZINC, PATTERN, CLUSTER, and TSP, we employ the
    Drop Path Rate                  0.1            0.1/0.4       AdamW optimizer [Loshchilov and Hutter, 2018] with a 20-
    Max Epochs                      10                50         epoch warmup, and reduce the learning rate by a cosine learn-
    Peak Learning Rate             2e-4              4e-4        ing rate scheduler. The dropout ratios of FFN, embedding,
    Min Learning Rete              1e-4              1e-9        and attention are set to 0.3, 0.3, and 0.5, respectively. The
    Batch Size                     128               128
    Warm-up Epochs                   1                 1
                                                                 weight decay is set to 0.05. Each experiment is run 5 times
    Learning Rate Decay          Cosine            Cosine        with 5 different random seeds, and the results are used to cal-
    Adam ϵ                         1e-8              1e-8        culate the mean and standard deviations of the metric.
    Adam (β2 , β2 )            (0.9, 0.999)      (0.9, 0.999)
    Weight Decay                    0.0               0.0
    EMA                             ✓                 ✓

      Table 10: Hyper-parameters on MolHIV and MolPCBA.


    Dataset                 ZINC/PATTERN/CLUSTER/TSP
    FFN Dropout                           0.3
    Embedding Dropout                     0.3
    Attention Dropout                     0.5
    Drop Path Rate                        0.3
    Max Epochs                   10000/1000/1000/1000
    Peak Learning Rate                   5e-4
    Batch Size                      256/256/256/32
    Warm-up Epochs                        20
    Learning Rate Decay                 Cosine
    Weight Decay                         0.05
    EMA                                   ✓

      Table 11: Hyper-parameters on 4 benchmarking datasets.



and Graphormer [Ying et al., 2021] to scale up our GPTrans.
The dimension of each head is set to 10 for our nano model,
and 32 for others. The expansion ratio of the FFN modules
is α = 1 for all model variants. Other hyper-parameters of
these models can be found in Table 8.

C     Training Strategies
C.1     PCQM4M and PCQM4Mv2
We first report the hyper-parameters of the experiments on
PCQM4M and PCQM4Mv2 datasets in Table 9. Empirically,
for our GPTrans-T/S/B models, the dropout ratios of FFN,
embedding, and attention are set to 0.1 by default, while for
the GPTrans-L model, it is set to 0.2. Besides, the drop path
rates are set to 0.1/0.1/0.2/0.4 for these variants along with
the model scaling up.
C.2     MolHIV and MolPCBA
We fine-tune our GPTrans-B on MolHIV and GPTrans-B/L
on MolPCBA datasets, and load the PCQM4Mv2 pre-trained
weights as initialization. Most of the hyper-parameters are
consistent with the pre-training stage, see Table 10 for de-
tails. In addition, each experiment in these datasets is run 5

