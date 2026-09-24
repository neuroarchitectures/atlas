# A Generalization of Transformer Networks to Graphs Dwivedi Bresson 2020

> Source: `A_Generalization_of_Transformer_Networks_to_Graphs_Dwivedi_Bresson_2020.pdf`

---

                                                                   A Generalization of Transformer Networks to Graphs
                                                                                   Vijay Prakash Dwivedi,¶ Xavier Bresson¶
                                                          ¶
                                                              School of Computer Science and Engineering, Nanyang Technological University, Singapore
                                                                         vijaypra001@e.ntu.edu.sg, xbresson@ntu.edu.sg




arXiv:2012.09699v2 [cs.LG] 24 Jan 2021
                                                                     Abstract                                et al. 2017) in graph neural networks (GNNs), this process
                                                                                                             of learning word feature representations by combining fea-
                                           We propose a generalization of transformer neural network         ture information from other words in a sentence can alter-
                                           architecture for arbitrary graphs. The original transformer
                                           was designed for Natural Language Processing (NLP), which
                                                                                                             natively be viewed as a case of a GNN applied on a fully
                                           operates on fully connected graphs representing all connec-       connected graph of words (Joshi 2020). Transformers based
                                           tions between the words in a sequence. Such architecture does     models have led to state-of-the-art performance on several
                                           not leverage the graph connectivity inductive bias, and can       NLP applications (Devlin et al. 2018; Radford et al. 2018;
                                           perform poorly when the graph topology is important and           Brown et al. 2020). On the other hand, graph neural net-
                                           has not been encoded into the node features. We introduce a       works (GNNs) are shown to be the most effective neural
                                           graph transformer with four new properties compared to the        network architectures on graph datasets and have achieved
                                           standard model. First, the attention mechanism is a function      significant success on a wide range of applications, such
                                           of the neighborhood connectivity for each node in the graph.      as in knowledge graphs (Schlichtkrull et al. 2018; Chami
                                           Second, the positional encoding is represented by the Lapla-      et al. 2020), in social sciences (Monti et al. 2019), in physics
                                           cian eigenvectors, which naturally generalize the sinusoidal
                                           positional encodings often used in NLP. Third, the layer
                                                                                                             (Cranmer et al. 2019; Sanchez-Gonzalez et al. 2020), etc.
                                           normalization is replaced by a batch normalization layer,         In particular, GNNs exploit the given arbitrary graph struc-
                                           which provides faster training and better generalization per-     ture while learning the feature representations for nodes and
                                           formance. Finally, the architecture is extended to edge feature   edges and eventually the learned representations are used for
                                           representation, which can be critical to tasks s.a. chemistry     downstream tasks. In this work, we explore inductive biases
                                           (bond type) or link prediction (entity relationship in knowl-     at the convergence of these two active research areas in deep
                                           edge graphs). Numerical experiments on a graph benchmark          learning towards presenting an improved version of Graph
                                           demonstrate the performance of the proposed graph trans-          Transformer (see Figure 1) which extends the key design
                                           former architecture. This work closes the gap between the         components of the NLP transformers to arbitrary graphs.
                                           original transformer, which was designed for the limited case
                                           of line graphs, and graph neural networks, that can work with
                                           arbitrary graphs. As our architecture is simple and generic,      1.1   Related Work
                                           we believe it can be used as a black box for future applica-      As a preliminary, we highlight the most recent research
                                           tions that wish to consider transformer and graphs.1              works which attempt to develop graph transformers (Li et al.
                                                                                                             2019; Nguyen, Nguyen, and Phung 2019; Zhang et al. 2020)
                                                                                                             with few focused on specialized cases such as on heteroge-
                                                               1    Introduction                             neous graphs, temporal networks, generative modeling, etc.
                                         There has been a tremendous success in the field of nat-            (Yun et al. 2019; Xu, Joshi, and Bresson 2019; Hu et al.
                                         ural language processing (NLP) since the development of             2020; Zhou et al. 2020).
                                         Transformers (Vaswani et al. 2017) which are currently the             The model proposed in Li et al. (2019) employs attention
                                         best performing neural network architectures for handling           to all graph nodes instead of a node’s local neighbors for
                                         long-term sequential datasets such as sentences in NLP.             the purpose of capturing global information. This limits the
                                         This is achieved by the use of attention mechanism (Bah-            efficient exploitation of sparsity which we show is a good in-
                                         danau, Cho, and Bengio 2014) where a word in a sentence             ductive bias for learning on graph datasets. For the purpose
                                         attends to each other word and combines the received in-            of global information, we argue that there are other ways to
                                         formation to generate its abstract feature representations.         incorporate the same instead of letting go sparsity and local
                                         From a perspective of message-passing paradigm (Gilmer              contexts. For example, the use of graph-specific positional
                                         AAAI’21 Workshop on Deep Learning on Graphs: Methods and            features (Zhang et al. 2020), or node Laplacian position
                                         Applications (DLG-AAAI’21). Copyright © 2021, Association for       eigenvectors (Belkin and Niyogi 2003; Dwivedi et al. 2020),
                                         the Advancement of Artificial Intelligence (www.aaai.org). All      or relative learnable positional information (You, Ying, and
                                         rights reserved.                                                    Leskovec 2019), virtual nodes (Li et al. 2015), etc. Zhang
                                             1
                                               https://github.com/graphdeeplearning/graphtransformer.        et al. (2020) propose Graph-BERT with an emphasis on pre-
                     Add&Norm                                                    Add&Norm                     Add&Norm




                     Add&Norm                                                    Add&Norm                     Add&Norm




             +                          +                                    +                   +




                                                      Laplacian EigVecs as          Graph Transformer Layer
              Graph Transformer Layer                 Positional Encoding              with edge features


Figure 1: Block Diagram of Graph Transformer with Laplacian Eigvectors (λ) used as positional encoding (LapPE). LapPE
is added to input node embeddings before passing the features to the first layer. Left: Graph Transformer operating on node
embeddings only to compute attention scores; Right: Graph Transformer with edge features with designated feature pipeline to
maintain layer wise edge representations. In this extension, the available edge attributes in a graph is used to explicitly modify
the corresponding pairwise attention scores.


training and parallelized learning using a subgraph batch-             preting the generated meta-paths. There is another trans-
ing scheme that creates fixed-size linkless subgraphs to be            former based approach developed for heterogeneous in-
passed to the model instead of the original graph. Graph-              formation networks, namely Heterogeneous Graph Trans-
BERT employs a combination of several positional encod-                former (HGT) by Hu et al. (2020). Apart from its ability
ing schemes to capture absolute node structural and rela-              of handling arbitrary number of node and edge types, HGT
tive node positional information. Since the original graph is          also captures the dynamics of information flow in the hetero-
not used directly in Graph-BERT and the subgraphs do not               geneous graphs in the form of relative temporal positional
have edges between the nodes (i.e., linkless), the proposed            encoding which is based on the timestamp differences of the
combination of positional encodings attempts at retaining              central node and the message-passing nodes. Furthermore,
the original graph structure information in the nodes. We              Zhou et al. (2020) proposed a transformer based generative
perform detailed analysis of Graph-BERT positional encod-              model which generates temporal graphs by directly learn-
ing schemes, along with experimental comparison with the               ing from dynamic information in networks. The architecture
model we present in this paper in Section 4.1.                         presented in Nguyen, Nguyen, and Phung (2019) somewhat
                                                                       proceeds along our goal to develop graph transformer for ar-
   Yun et al. (2019) developed Graph Transformer Networks              bitrary homogeneous graphs with a coordinate embedding
(GTN) to learn on heterogeneous graphs with a target to                based positional encoding scheme. However, their experi-
transform a given heterogeneous graph into a meta-path                 ments show that the coordinate embeddings are not universal
based graph and then perform convolution. Notably, their               in performance and only helps in a couple of unsupervised
focus behind the use of attention framework is for inter-
learning experiments among all evaluations.                          feasibility and large transformer models can be trained on
                                                                     such fully connected graphs of words.
1.2   Contributions                                                     In case of actual graph datasets, graphs have arbitrary con-
                                                                     nectivity structure available depending on the domain and
Overall, we find that the most fruitful ideas from the trans-
                                                                     target of application, and have node sizes in ranges of up
formers literature in NLP can be applied in a more efficient
                                                                     to millions, or billions. The available structure presents us
way and posit that sparsity and positional encodings are two
                                                                     with a rich source of information to exploit as an inductive
key aspects in the development of a Graph Transformer. As
                                                                     bias in a neural network, whereas the node sizes practically
opposed to designing a best performing model for specific
                                                                     makes it impossible to have a fully connected graph for such
graph tasks, our work attempts for a generic, competitive
                                                                     datasets. On these accounts, it is ideal and practical to have
transformer model which draws ideas together from the do-
                                                                     a Graph Transformer where a node attends to local node
mains of NLP and GNNs. For an overview, this paper brings
                                                                     neighbors, same as in GNNs (Defferrard, Bresson, and Van-
the following contributions:
                                                                     dergheynst 2016; Kipf and Welling 2017; Monti et al. 2017;
• We put forward a generalization of transformer networks            Gilmer et al. 2017; Veličković et al. 2018; Bresson and Lau-
  to homogeneous graphs of arbitrary structure, namely               rent 2017; Xu et al. 2019).
  Graph Transformer, and an extended version of Graph
  Transformer with edge features that allows the usage of            2.2   On Positional Encodings
  explicit domain information as edge features.
                                                                     In NLP, transformer based models are, in most cases, sup-
• Our method includes an elegant way to fuse node po-                plied with a positional encoding for each word. This is criti-
  sitional features using Laplacian eigenvectors for graph           cal to ensure unique representation for each word, and even-
  datasets, inspired from the heavy usage of positional en-          tually preserve distance information. For graphs, the design
  codings in NLP transformer models and recent research              of unique node positions is challenging as there are sym-
  on node positional features in GNNs. The comparison                metries which prevent canonical node positional informa-
  with literature shows Laplacian eigenvectors to be well-           tion (Murphy et al. 2019). In fact, most of the GNNs which
  placed than any existing approaches to encode node posi-           are trained on graph datasets learn structural node informa-
  tional information for arbitrary homogeneous graphs.               tion that are invariant to the node position (Srinivasan and
• Our experiments demonstrate that the proposed model                Ribeiro 2020). This is a critical reason why simple attention
  surpasses baseline isotropic and anisotropic GNNs. The             based models, such as GAT (Veličković et al. 2018), where
  architecture simultaneously emerges as a better attention          the attention is a function of local neighborhood connectiv-
  based GNN baseline as well as a simple and effective               ity, instead full-graph connectivity, do not seem to achieve
  Transformer network baseline for graph datasets for fu-            competitive performance on graph datasets. The issue of po-
  ture research at the intersection of attention and graphs.         sitional embeddings has been explored in recent GNN works
                                                                     (Murphy et al. 2019; You, Ying, and Leskovec 2019; Srini-
                                                                     vasan and Ribeiro 2020; Dwivedi et al. 2020; Li et al. 2020)
              2    Proposed Architecture                             with a goal to learn both structural and positional features.
As stated earlier, we take into account two key aspects to           In particular, Dwivedi et al. (2020) make the use of avail-
develop Graph Transformers – sparsity and positional en-             able graph structure to pre-compute Laplacian eigenvectors
codings which should ideally be used in the best possible            (Belkin and Niyogi 2003) and use them as node positional
way for learning on graph datasets. We first discuss the mo-         information. Since Laplacian PEs are generalization of the
tivations behind these using a transition from NLP to graphs,        PE used in the original transformers (Vaswani et al. 2017)
and then introduce the architecture proposed.                        to graphs and these better help encode distance-aware in-
                                                                     formation (i.e., nearby nodes have similar positional fea-
2.1   On Graph Sparsity                                              tures and farther nodes have dissimilar positional features),
                                                                     we use Laplacian eigenvectors as PE in Graph Transformer.
In NLP transformers, a sentence is treated as a fully con-           Although these eigenvectors have multiplicity occuring due
nected graph and this choice can be justified for two reasons        to the arbitrary sign of eigenvectors, we randomly flip the
– a) First, it is difficult to find meaningful sparse interactions   sign of the eigenvectors during training, following Dwivedi
or connections among the words in a sentence. For instance,          et al. (2020).We pre-compute the Laplacian eigenvectors of
the dependency of a word in a sentence on another word can           all graphs in the dataset. Eigenvectors are defined via the
vary with context, perspective of a user and specific applica-       factorization of the graph Laplacian matrix;
tion. There can be numerous plausible ground truth connec-
tions among words in a sentence and therefore, text datasets                     ∆ = I − D−1/2 AD−1/2 = U T ΛU,                 (1)
of sentences do not have explicit word interactions available.
It thereby makes sense to have each word attending to each           where A is the n × n adjacency matrix, D is the degree
other word in a sentence, as followed by the Transformer             matrix, and Λ, U correspond to the eigenvalues and eigen-
architecture (Vaswani et al. 2017). – b) Next, the so-called         vectors respectively. We use the k smallest non-trivial eigen-
graph considered in an NLP transformer often has less than           vectors of a node as its positional encoding and denote by λi
tens or hundreds of nodes (i.e. sentences are often less than        for node i. Finally, we refer to Section 4.1 for a comparison
tens or hundreds of words). This makes for computationally           of Laplacian PE with existing Graph-BERT PEs.
2.3   Graph Transformer Architecture                                         Graph Transformer Layer with edge features The
We now introduce the Graph Transformer Layer and Graph                       Graph Transformer with edge features is designed for bet-
Transformer Layer with edge features. The layer archi-                       ter utilization of rich feature information available in several
tecture is illustrated in Figure 1. The first model is de-                   graph datasets in the form of edge attributes. See Figure 1
signed for graphs which do not have explicit edge attributes,                (Right) for a reference to the building block of a layer. Since
whereas the second model maintains a designated edge fea-                    our objective remains to better use the edge features which
ture pipeline to incorporate the available edge information                  are pairwise scores corresponding to a node pair, we tie these
and maintain their abstract representations at every layer.                  available edge features to implicit edge scores computed by
                                                                             pairwise attention. In other words, say an intermediate at-
Input First of all, we prepare the input node and edge em-                   tention score before softmax, ŵij , is computed when a node
beddings to be passed to the Graph Transformer Layer. For                    i attends to node j after the multiplication of query and key
a graph G with node features αi ∈ Rdn ×1 for each node i                     feature projections, see the expression inside the brackets in
and edge features βij ∈ Rde ×1 for each edge between node                    Equation 5. Let us treat this score ŵij as implicit information
i and node j, the input node features αi and edge features                   about the edge < i, j >. We now try to inject the available
βij are passed via a linear projection to embed these to d-                  edge information for the edge < i, j > and improve the al-
dimensional hidden features h0i and e0ij .                                   ready computed implicit attention score ŵij . It is done by
             ĥ0i = A0 αi + a0 ; e0ij = B 0 βij + b0 ,                 (2)   simply multiplying the two values ŵij and eij , see Equa-
         0           d×dn       0        d×de       0      0       d         tion 12. This kind of information injection is not seen to be
where A ∈ R          ,B ∈R          and a , b ∈ R are the                    explored much, or applied in NLP Transformers as there is
parameters of the linear projection layers. We now embed                     usually no available feature information between two words.
the pre-computed node positional encodings of dim k via a                    However, in graph datasets such as molecular graphs, or
linear projection and add to the node features ĥ0i .                        social media graphs, there is often some feature informa-
             λ0i = C 0 λi + c0 ; h0i = ĥ0i + λ0i ,       (3)                tion available on the edge interactions and it becomes nat-
where C 0 ∈ Rd×k and c0 ∈ Rd . Note that the Laplacian po-                   ural to design an architecture to use this information while
sitional encodings are only added to the node features at the                learning. For the edges, we also maintain a designated node-
input layer and not during intermediate Graph Transformer                    symmetric edge feature representation pipeline for propagat-
layers.                                                                      ing edge attributes from one layer to another, see Figure 1.
                                                                             We now proceed to define the layer update equations for a
Graph Transformer Layer The Graph Transformer is                             layer `.
closely the same transformer architecture initially proposed
in (Vaswani et al. 2017), see Figure 1 (Left). We now pro-
                                                                                                             H
ceed to define the node update equations for a layer `.                                                      n     X
                                                                                                                             k,` k,` `
                                                                                                                                       
                                                                                      ĥ`+1
                                                                                        i         =    Oh`                  wij V hj ,         (9)
                                H
                                n     X                                                                    k=1     j∈Ni
                                                k,` k,` `
             ĥ`+1
               i   = Oh`                       wij V hj ,              (4)                                   H
                                                                                                             n           
                                                                                                                      k,`
                                k=1     j∈Ni                                           ê`+1
                                                                                         ij       =    Oe`          ŵij    , where,          (10)
                                    Qk,` h` · K k,` h`                                                     k=1
               k,`                         i           j
       where, wij  = softmaxj               √             ,   (5)
                                               dk
                                                                                      k,`                         k,`
and Qk,` , K k,` , V k,` ∈ Rdk ×d , Oh` ∈ Rd×d , k = 1 to H de-                      wij       =      softmaxj (ŵij  ),                      (11)
notes the number of attention heads, and k denotes concate-                                            Qk,` h` · K k,` h` 
nation. For numerical stability, the outputs after taking expo-                        k,`                    i          j
                                                                                     ŵij      =               √             · E k,` e`ij ,   (12)
nents of the terms inside softmax is clamped to a value be-                                                      dk
tween −5 to +5. The attention outputs ĥ`+1       are then passed
                                             i                               and Qk,` , K k,` , V k,` , E k,` ∈ Rdk ×d , Oh` , Oe` ∈ Rd×d ,
to a Feed Forward Network (FFN) preceded and succeeded
                                                                             k = 1 to H denotes the number of attention head, and k de-
by residual connections and normalization layers, as:
                                                                             notes concatenation. For numerical stability, the outputs af-
                                                                             ter taking exponents of the terms inside softmax is clamped
               ˆ `+1
                                                    
               ĥi          =       Norm h`i + ĥ`+1
                                                 i     ,               (6)   to a value between −5 to +5. The outputs ĥ`+1i     and ê`+1
                                                                                                                                       ij are
                 `+1
                                                                             then passed to separate Feed Forward Networks preceded
               ˆ
               ˆ                                 ˆ `+1                       and succeeded by residual connections and normalization
               ĥi          =       W2` ReLU(W1` ĥi ),                (7)
                                                                             layers, as:
                                          `+1 ˆ `+1 
                                           ˆ      ˆ
               h`+1
                i           =       Norm ĥi + ĥi      ,              (8)
                                                                                         ˆ `+1
                                                                                                                          
                                            ˆ `+1 `+1                                    ĥi          =   Norm h`i + ĥ`+1
                                                                                                                       i     ,                (13)
                                      ˆ     ˆ
where W1` , ∈ R2d×d , W2` , ∈ Rd×2d , ĥi , ĥi denote in-
termediate representations, and Norm can either be Layer-                                ˆˆ `+1           `        ` ˆ
                                                                                                                         `+1

Norm(Ba, Kiros, and Hinton 2016) or BatchNorm (Ioffe and                                 ĥi          = Wh,2 ReLU(Wh,1 ĥi ),                 (14)
Szegedy 2015). The bias terms are omitted for clarity of pre-                                                 `+1 ˆ `+1 
                                                                                                              ˆ     ˆ
sentation.                                                                               h`+1
                                                                                          i           = Norm ĥi + ĥi     ,                  (15)
                                                   `+1
         `                                ˆ `+1 ˆ
                                                ˆ                 the learning rate reaches to a value of 1 × 10−6 . We run each
where Wh,1   , ∈ R2d×d , Wh,2
                           `
                              , ∈ Rd×2d , ĥi , ĥi      denote   experiment with 4 different seeds and report the mean and
intermediate representations,                                     average performance measure of the 4 runs. The results are
                                                                  reported in Table 1 and comparison in Table 2.
            ˆê`+1 = Norm e` + ê`+1 ,
                                          
               ij                ij   ij                   (16)
            ˆ  `+1                                                             4    Analysis and Discussion
            ˆê           `          ` ˆ`+1
               ij     = We,2 ReLU(We,1   êij ),           (17)   We now present the analysis of our experiments on the pro-
                              `+1
                                     ˆ  `+1                      posed Graph Transformer Architecture, see Tables 1 and 2.
            e`+1
             ij       = Norm ˆêij + ˆêij ,               (18)
                                                                  • The generalization of transformer network on graphs is
                                             `+1 ˆ`+1
                                                                    best when Laplacian PE are used for node positions and
         `
where We,1  , ∈ R2d×d , We,2
                           `
                              , ∈ Rd×2d , ˆêij , ˆêij denote      Batch Normalization is selected instead of Layer Normal-
intermediate representations.                                       ization. For all three benchmark datasets, the experiments
                                                                    score the highest performance in this setting, see Table 1.
Task based MLP Layers The node representations ob-
tained at the final layer of Graph Transformer are passed         • The proposed architecture performs significantly better
to a task based MLP network for computing task-dependent            than baseline isotropic and anisotropic GNNs (GCN and
outputs, which are then fed to a loss function to train the         GAT respectively), and helps close the gap between the
parameters of the model. The formal definitions of the task         original transformer and transformer for graphs. Notably,
based layers that we use can be found in Appendix A.1.              our architecture emerges as a fresh and improved attention
                                                                    based GNN baseline surpassing GAT (see Table 2), which
            3        Numerical Experiments                          employs multi-headed attention inspired by the original
                                                                    transformer (Vaswani et al. 2017) and have been often
We evaluate the performance of proposed Graph Trans-                used in the literature as a baseline for attention-based
former on three benchmark graph datasets– ZINC (Irwin               GNN models.
et al. 2012), PATTERN and CLUSTER (Abbe 2017) from
a recent GNN benchmark (Dwivedi et al. 2020).                     • As expected, sparse graph connectivity is a critical in-
                                                                    ductive bias for datasets with arbitrary graph structure, as
ZINC, Graph Regression ZINC (Irwin et al. 2012) is a                demonstrated by comparing sparse vs. full graph experi-
molecular dataset with the task of graph property regres-           ments.
sion for constrained solubility. Each ZINC molecule is rep-       • Our proposed extension of Graph Transformer with edge
resented as a graph of atoms as nodes and bonds as edges.           features reaches close to the best performing GNN,
Since this dataset have rich feature information in terms of        i.e., GatedGCN, on ZINC. This architecture specifically
bonds as edge attributes, we use the ‘Graph Transformer             brings exciting promise to datasets where domain infor-
with edge features’ for this task. We use the 12K subset of         mation along pairwise interactions can be leveraged for
the data as in Dwivedi et al. (2020).                               maximum learning performance.
PATTERN, Node Classification PATTERN is a node
classification dataset generated using the Stochastic Block       4.1   Comparison to PEs used in Graph-BERT
Models (SBM) (Abbe 2017). The task is classify the nodes          In addition to the reasons underscored in Sections 1.1 and
into 2 communities. PATTERN graphs do not have explicit           2.2, we demonstrate the usefulness of Laplacian eigenvec-
edge features and hence we use the simple ‘Graph Trans-           tors as a suitable candidate PE for Graph Transformer in
former’ for this task. The size of this dataset is 14K graphs.    this section, by its comparison with different PE schemes ap-
                                                                  plied in Graph-BERT (Zhang et al. 2020).2 In Graph-BERT,
CLUSTER, Node Classification CLUSTER is also a                    which operates on fixed size sampled subgraphs, a node at-
synthetically generated dataset using SBM model. The task         tends to every other node in a subgraph. For a given graph
is to assign a cluster label to each node. There are total 6      G = (V, E) with V nodes and E edges, a subgraph gi of size
cluster labels. Similar to PATTERN, CLUSTER graphs do             k + 1 is created for every node i in the graph, which means
not have explicit edge features and hence we use the simple       the original single graph G is converted to V subgraphs. For
‘Graph Transformer’ for this task. The size of this dataset is    a subgraph gi corresponding to node ui , the k other nodes
12K graphs. We refer the readers to (Dwivedi et al. 2020)         are the ones which have the top k intimacy scores with node
for additional information, inlcuding preparation, of these       ui based on a pre-computed intimacy matrix that maps every
datasets.                                                         edge in the graph G to an intimacy score. While the sampling
Model Configurations For experiments, we follow the               is great for parallelization and efficiency, the original graph
benchmarking protocol introduced in Dwivedi et al. (2020)         structure is not directly used in the layers. Graph-BERT uses
based on PyTorch (Paszke et al. 2019) and DGL (Wang et al.            2
                                                                        Note that we do not perform empirical comparison with other
2019). We use 10 layers of Graph Transformer layers with          PEs in Graph Transformer literature except Graph-BERT, because
each layer having 8 attention heads and arbitrary hidden di-      of two reasons: i) Some existing Graph Transformer methods do
mensions such that the total number of trainable parameters       not use PEs, ii) If PEs are used, they are usually specialised; for
is in the range of 500k. We use learning rate decay strategy      instance, Relative Temporal Encoding (RTE) for encoding dynamic
to train the models where the training stops at a point when      information in heterogeneous graphs in (Hu et al. 2020).
                                                             Sparse Graph                                                        Full Graph
    Dataset   LapPE   L    #Param    Test Perf.±s.d.   Train Perf.±s.d. #Epoch    Epoch/Total         Test Perf.±s.d.     Train Perf.±s.d. #Epoch               Epoch/Total
                                                             Batch Norm: False; Layer Norm: True
               x      10   588353     0.278±0.018        0.027±0.004    274.75    26.87s/2.06hr        0.741±0.008          0.431±0.013            196.75       37.64s/2.09hr
     ZINC
               X      10   588929     0.284±0.012        0.031±0.006    263.00    26.64s/1.98hr        0.735±0.006          0.442±0.031            196.75       31.50s/1.77hr
               x      10   523146    70.879±0.295       86.174±0.365    128.50   202.68s/7.32hr       19.596±2.071          19.570±2.053           103.00      512.34s/15.15hr
 CLUSTER
               X      10   524026    70.649±0.250       86.395±0.528    130.75   200.55s/7.43hr       27.091±3.920          26.916±3.764           139.50      565.13s/22.37hr
               x      10   522742    73.140±13.633      73.070±13.589   184.25   276.66s/13.75hr      50.854±0.111          50.906±0.005           108.00      540.85s/16.77hr
 PATTERN
               X      10   522982    71.005±11.831      71.125±11.977   192.50   294.91s/14.79hr      56.482±3.549          56.565±3.546           124.50      637.55s/22.69hr
                                                             Batch Norm: True; Layer Norm: False
               x      10   588353     0.264±0.008        0.048±0.006    321.50    28.01s/2.52hr        0.724±0.013          0.518±0.013            192.25       50.27s/2.72hr
     ZINC
               X      10   588929     0.226±0.014        0.059±0.011    287.50    27.78s/2.25hr        0.598±0.049          0.339±0.123            273.50       45.26s/3.50hr
               x      10   523146    72.139±0.405       85.857±0.555    121.75   200.85s/6.88hr       21.092±0.134          21.071±0.037           100.25      595.24s/17.10hr
 CLUSTER
               X      10   524026    73.169±0.622       86.585±0.905    126.50   201.06s/7.20hr       27.121±8.471          27.192±8.485           133.75      552.06s/20.72hr
               x      10   522742    83.949±0.303       83.864±0.489    236.50   299.54s/19.71hr      50.889±0.069          50.873±0.039           104.50      621.33s/17.53hr
 PATTERN
               X      10   522982    84.808±0.068       86.559±0.116    145.25   309.95s/12.67hr      54.941±3.739          54.915±3.769           117.75      683.53s/22.77hr


Table 1: Results of GraphTransformer (GT) on all datasets. Performance Measure for ZINC is MAE, for PATTERN and CLUS-
TER is Acc. Results (higher is better for all except ZINC) are averaged over 4 runs with 4 different seeds. Bold: the best
performing model for each dataset. We perform each experiment with given graphs (Sparse Graph) and (Full Graph) in
which we create full connections among all nodes; For ZINC full graphs, edge features are discarded given our motive of the
full graph experiments without any sparse structure information.


         Model        ZINC           CLUSTER            PATTERN                                                                          Sparse Graph
                                                                                      Dataset   PE    #Param     Test Perf.±s.d.   Train Perf.±s.d. #Epoch       Epoch/Total
        GNN BASELINE SCORES from (Dwivedi et al. 2020)                                                   Batch Norm: True; Layer Norm: False; L = 10
         GCN       0.367±0.011      68.498±0.976       71.892±0.334                               x    588353     0.264±0.008       0.048±0.006       321.50     28.01s/2.52hr
                                                                                       ZINC       L    588929     0.226±0.014       0.059±0.011       287.50     27.78s/2.25hr
         GAT       0.384±0.007      70.587±0.447       78.271±0.186                               W    590721     0.267±0.012       0.059±0.010       263.25     27.04s/2.00hr
    GatedGCN       0.214±0.013      76.082±0.196       86.508±0.085                               x    523146    72.139±0.405       85.857±0.555      121.75     200.85s/6.88hr
                                                                                   CLUSTER        L    524026    73.169±0.622       86.585±0.905      126.50     201.06s/7.20hr
                           OUR RESULTS                                                            W    531146    70.790±0.537       86.829±0.745      119.00     196.41s/6.69hr
     GT (Ours)     0.226±0.014      73.169±0.622       84.808±0.068                               x    522742    83.949±0.303       83.864±0.489      236.50    299.54s/19.71hr
                                                                                   PATTERN        L    522982    84.808±0.068       86.559±0.116      145.25    309.95s/12.67hr
                                                                                                  W    530742    75.489±0.216       97.028±0.104      109.25     310.11s/9.73hr
Table 2: Comparison of our best performing scores (from
Table 1) on each dataset against the GNN baselines (GCN                           Table 3: Analysis of GraphTransformer (GT) using different
(Kipf and Welling 2017), GAT (Veličković et al. 2018), Gat-                     PE schemes. Notations x: No PE; L: LapPE (ours); W: WL-
edGCN(Bresson and Laurent 2017)) of 500k model param-                             PE (Zhang et al. 2020). Bold: the best performing model for
eters. Note: Only GatedGCN and GT models use the avail-                           each dataset.
able edge attributes in ZINC.

                                                                                                                 5       Conclusion
                                                                                  This work presented a simple yet effective approach to gen-
a combination of node PE schemes to inform the model on
                                                                                  eralize transformer networks on arbitrary graphs and intro-
node structural, positional, and distance information from
                                                                                  duced the corresponding architecture. Our experiments con-
original graph– i) Intimacy based relative PE, ii) Hop based
                                                                                  sistently showed that the presence of – i) Laplacian eigen-
relative distance encoding, and iii) Weisfeiler Lehman based
                                                                                  vectors as node positional encodings and – ii) batch normal-
absolute PE (WL-PE). The intimacy based PE and the hop
                                                                                  ization, in place of layer normalization, around the trans-
based PE are variant to the sampled subgraphs, i.e., these
                                                                                  former feed forward layers enhanced the transformer univer-
PEs for a node in a subgraph gi depends on the node ui w.r.t
                                                                                  sally on all experiments. Given the simple and generic na-
which it is sampled, and cannot be directly used in other
                                                                                  ture of our architecture and competitive performance against
cases unless we use similar sampling strategy. The WL-PE
                                                                                  standard GNNs, we believe the proposed model can be used
which are absolute structural roles of nodes in the original
                                                                                  as baseline for further improvement across graph applica-
graph computed using WL algorithm (Zhang et al. 2020;
                                                                                  tions employing node attention. In future works, we are in-
Niepert, Ahmed, and Kutzkov 2016), are not variant to the
                                                                                  terested in building upon the graph transformer along as-
subgraphs and can be easily used as a generic PE mecha-
                                                                                  pects such as efficient training on single large graphs, ap-
nism. On that account, we swap Laplacian PE in our experi-
                                                                                  plicability on heterogeneous domains, etc., and perform ef-
ments for an ablation analysis and use WL-PE from Graph-
                                                                                  ficient graph representation learning keeping in account the
BERT, see Table 3. As Laplacian PE capture better struc-
                                                                                  recent innovations in graph inductive biases.
tural and positional information about the nodes, which es-
sentially is the objective behind using the three Graph-BERT
PEs, they outperform the WL-PE. Besides, WL-PEs tend to                                                         Acknowledgments
overfit SBM datasets and lead to poor generalization.                             XB is supported by NRF Fellowship NRFF2017-10.
                       References                                Li, P.; Wang, Y.; Wang, H.; and Leskovec, J. 2020. Dis-
Abbe, E. 2017. Community detection and stochastic block          tance Encoding–Design Provably More Powerful GNNs
models: recent developments. The Journal of Machine              for Structural Representation Learning. arXiv preprint
Learning Research 18(1): 6446–6531.                              arXiv:2009.00142 .
Ba, J. L.; Kiros, J. R.; and Hinton, G. E. 2016. Layer nor-      Li, Y.; Liang, X.; Hu, Z.; Chen, Y.; and Xing, E. P. 2019.
malization. NeurIPS workshop on Deep Learning .                  Graph Transformer. URL https://openreview.net/forum?id=
Bahdanau, D.; Cho, K.; and Bengio, Y. 2014. Neural ma-           HJei-2RcK7.
chine translation by jointly learning to align and translate.    Li, Y.; Tarlow, D.; Brockschmidt, M.; and Zemel, R. 2015.
arXiv preprint arXiv:1409.0473 .                                 Gated graph sequence neural networks. arXiv preprint
Belkin, M.; and Niyogi, P. 2003. Laplacian eigenmaps for         arXiv:1511.05493 .
dimensionality reduction and data representation. Neural         Monti, F.; Boscaini, D.; Masci, J.; Rodola, E.; Svoboda, J.;
computation 15(6): 1373–1396.                                    and Bronstein, M. M. 2017. Geometric Deep Learning on
Bresson, X.; and Laurent, T. 2017. Residual gated graph          Graphs and Manifolds Using Mixture Model CNNs. 2017
convnets. arXiv preprint arXiv:1711.07553 .                      IEEE Conference on Computer Vision and Pattern Recogni-
Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.;      tion (CVPR) doi:10.1109/cvpr.2017.576.
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,    Monti, F.; Frasca, F.; Eynard, D.; Mannion, D.; and Bron-
A.; et al. 2020. Language models are few-shot learners.          stein, M. M. 2019.        Fake news detection on social
arXiv preprint arXiv:2005.14165 .                                media using geometric deep learning. arXiv preprint
Chami, I.; Wolf, A.; Juan, D.-C.; Sala, F.; Ravi, S.; and Ré,   arXiv:1902.06673 .
C. 2020. Low-Dimensional Hyperbolic Knowledge Graph              Murphy, R.; Srinivasan, B.; Rao, V.; and Ribeiro, B. 2019.
Embeddings. arXiv preprint arXiv:2005.00545 .                    Relational Pooling for Graph Representations. In Interna-
Cranmer, M. D.; Xu, R.; Battaglia, P.; and Ho, S. 2019.          tional Conference on Machine Learning, 4663–4673.
Learning Symbolic Physics with Graph Networks. arXiv
                                                                 Nguyen, D. Q.; Nguyen, T. D.; and Phung, D. 2019. Univer-
preprint arXiv:1909.05862 .
                                                                 sal Self-Attention Network for Graph Classification. arXiv
Defferrard, M.; Bresson, X.; and Vandergheynst, P. 2016.         preprint arXiv:1909.11855 .
Convolutional Neural Networks on Graphs with Fast Local-
ized Spectral Filtering. In Advances in Neural Information       Niepert, M.; Ahmed, M.; and Kutzkov, K. 2016. Learning
Processing Systems 29, 3844–3852.                                convolutional neural networks for graphs. In International
                                                                 conference on machine learning, 2014–2023.
Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018.
Bert: Pre-training of deep bidirectional transformers for lan-   Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.;
guage understanding. arXiv preprint arXiv:1810.04805 .           Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.;
                                                                 Desmaison, A.; Köpf, A.; Yang, E.; DeVito, Z.; Raison, M.;
Dwivedi, V. P.; Joshi, C. K.; Laurent, T.; Bengio, Y.; and
                                                                 Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.;
Bresson, X. 2020. Benchmarking graph neural networks.
                                                                 and Chintala, S. 2019. PyTorch: An Imperative Style, High-
arXiv preprint arXiv:2003.00982 .
                                                                 Performance Deep Learning Library.
Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; and
Dahl, G. E. 2017. Neural message passing for quantum             Radford, A.; Narasimhan, K.; Salimans, T.; and Sutskever,
chemistry. In Proceedings of the 34th International Confer-      I. 2018. Improving language understanding by generative
ence on Machine Learning-Volume 70, 1263–1272. JMLR.             pre-training.
org.                                                             Sanchez-Gonzalez, A.; Godwin, J.; Pfaff, T.; Ying, R.;
Hu, Z.; Dong, Y.; Wang, K.; and Sun, Y. 2020. Heteroge-          Leskovec, J.; and Battaglia, P. W. 2020. Learning to sim-
neous graph transformer. In Proceedings of The Web Con-          ulate complex physics with graph networks. arXiv preprint
ference 2020, 2704–2710.                                         arXiv:2002.09405 .
Ioffe, S.; and Szegedy, C. 2015. Batch normalization: Accel-     Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; Van Den Berg, R.;
erating deep network training by reducing internal covariate     Titov, I.; and Welling, M. 2018. Modeling relational data
shift. arXiv preprint arXiv:1502.03167 .                         with graph convolutional networks. In European Semantic
Irwin, J. J.; Sterling, T.; Mysinger, M. M.; Bolstad, E. S.;     Web Conference, 593–607. Springer.
and Coleman, R. G. 2012. ZINC: a free tool to discover           Srinivasan, B.; and Ribeiro, B. 2020. On the Equivalence
chemistry for biology. Journal of chemical information and       between Node Embeddings and Structural Graph Represen-
modeling 52(7): 1757–1768.                                       tations. International Conference on Learning Representa-
Joshi, C. 2020. Transformers are Graph Neural Networks.          tions .
The Gradient .                                                   Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
Kipf, T. N.; and Welling, M. 2017. Semi-Supervised Clas-         L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. At-
sification with Graph Convolutional Networks. In Interna-        tention is all you need. In Advances in neural information
tional Conference on Learning Representations (ICLR).            processing systems, 5998–6008.
Veličković, P.; Cucurull, G.; Casanova, A.; Romero, A.; Liò,                        A     Appendix
P.; and Bengio, Y. 2018. Graph Attention Networks. Inter-         A.1    Task based MLP layer equations
national Conference on Learning Representations .
                                                                  Graph prediction layer For graph prediction task, the fi-
Wang, M.; Yu, L.; Zheng, D.; Gan, Q.; Gai, Y.; Ye, Z.; Li,        nal layer node features of a graph is averaged to get a d-
M.; Zhou, J.; Huang, Q.; Ma, C.; Huang, Z.; Guo, Q.; Zhang,       dimensional graph-level feature vector yG .
H.; Lin, H.; Zhao, J.; Li, J.; Smola, A. J.; and Zhang, Z.
2019. Deep Graph Library: Towards Efficient and Scalable                                          V
                                                                                               1X L
Deep Learning on Graphs. ICLR Workshop on Representa-                                   yG =        h ,                    (19)
tion Learning on Graphs and Manifolds .                                                        V i=0 i
Xu, K.; Hu, W.; Leskovec, J.; and Jegelka, S. 2019. How           The graph feature vector is then passed to a MLP to obtain
Powerful are Graph Neural Networks? In International Con-         the un-normalized prediction score for each class, ypred ∈
ference on Learning Representations.                              RC for each class:
Xu, K.; Li, C.; Tian, Y.; Sonobe, T.; Kawarabayashi, K.-
                                                                                   ypred = P ReLU (Q yG ) ,                (20)
i.; and Jegelka, S. 2018. Representation learning on
graphs with jumping knowledge networks. arXiv preprint
                                                                  where P ∈ Rd×C , Q ∈ Rd×d , C is the number of task la-
arXiv:1806.03536 .
                                                                  bels (classes) to be predicted. Since we perform single-target
Xu, P.; Joshi, C. K.; and Bresson, X. 2019. Multi-graph           graph regression in ZINC, C = 1, and the L1-loss between
transformer for free-hand sketch recognition. arXiv preprint      the predicted and groundtruth values is minimized during
arXiv:1912.11258 .                                                training.
You, J.; Ying, R.; and Leskovec, J. 2019. Position-aware
graph neural networks. International Conference on Ma-            Node prediction layer For node prediction task, each
chine Learning .                                                  node’s feature vector is passed to a MLP for computing
Yun, S.; Jeong, M.; Kim, R.; Kang, J.; and Kim, H. J. 2019.       the un-normalized prediction scores yi,pred ∈ RC for each
Graph transformer networks. In Advances in Neural Infor-          class:
                                                                                 yi,pred = P ReLU Q hL
                                                                                                           
mation Processing Systems, 11983–11993.                                                                   i ,          (21)
Zhang, J.; Zhang, H.; Sun, L.; and Xia, C. 2020. Graph-Bert:      where P ∈ Rd×C , Q ∈ Rd×d . During training, the cross-
Only Attention is Needed for Learning Graph Representa-           entropy loss weighted inversely by the class size is used.
tions. arXiv preprint arXiv:2001.05140 .
Zhou, D.; Zheng, L.; Han, J.; and He, J. 2020. A Data-               As a note, these task based layers can be modified as
Driven Graph Generative Model for Temporal Interaction            per the requirements of the dataset, and or the prediction to
Networks. In Proceedings of the 26th ACM SIGKDD Inter-            be done. For example, the Graph Transformer edge outputs
national Conference on Knowledge Discovery & Data Min-            (Figure 1 (Right)) can be used for edge prediction tasks and
ing, 401–411.                                                     the task based MLP layers can be defined in similar fashion
                                                                  as we do for node prediction. Besides, different styles of us-
                                                                  ing final and/or intermediate Graph Transformer layers can
                                                                  be used as inputs to the task based MLP layers, such as JK
                                                                  Readout (Jumping Knowledge) (Xu et al. 2018), etc. used
                                                                  often in GNNs.

                                                                  A.2    Hardware Information
                                                                  All experiments are run on Intel Xeon CPU E5-2690 v4
                                                                  server with 4 Nvidia 1080Ti GPUs. At a given time, 4 exper-
                                                                  iments were run on the server with each single GPU running
                                                                  1 experiment. The maximum training time for an experiment
                                                                  is limited to 24 hours.

