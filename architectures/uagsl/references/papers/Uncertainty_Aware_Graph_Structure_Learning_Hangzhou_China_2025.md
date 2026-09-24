# Uncertainty Aware Graph Structure Learning Hangzhou China 2025

> Source: `Uncertainty_Aware_Graph_Structure_Learning_Hangzhou_China_2025.pdf`

---

                                                                      Uncertainty-Aware Graph Structure Learning
                                                                 Shen Han†                                                   Zhiyao Zhou†                                  Jiawei Chen∗†‡
                                                    The State Key Laboratory of                                    The State Key Laboratory of                      The State Key Laboratory of
                                                    Blockchain and Data Security,                                  Blockchain and Data Security,                    Blockchain and Data Security,
                                                        Zhejiang University                                            Zhejiang University                               Zhejiang University
                                                         Hangzhou, China                                                 Hangzhou, China                                  Hangzhou, China
                                                       drhanshen@zju.edu.cn                                            zjucszzy@zju.edu.cn                             sleepyhunt@zju.edu.cn

                                                             Zhezheng Hao                                                      Sheng Zhou                                     Gang Wang
                                                     Northwestern Polytechnical                                         Zhejiang University                             Bangsun Technology




arXiv:2502.12618v2 [cs.LG] 19 Feb 2025
                                                            University                                                   Hangzhou, China                                 Hangzhou, China
                                                           Xi’an, China                                              zhousheng_zju@zju.edu.cn                          wanggang@bsfit.com.cn
                                                     haozhezheng@outlook.com

                                                                 Yan Feng†                                                    Chun Chen†                                      Can Wang‡
                                                   The State Key Laboratory of                                     The State Key Laboratory of                      The State Key Laboratory of
                                                   Blockchain and Data Security,                                   Blockchain and Data Security,                    Blockchain and Data Security,
                                                       Zhejiang University                                             Zhejiang University                              Zhejiang University
                                                         Hangzhou, China                                                Hangzhou, China                                  Hangzhou, China
                                                       fengyan@zju.edu.cn                                               chenc@zju.edu.cn                                 wcan@zju.edu.cn

                                         Abstract                                                                                        into existing GSL methods with minimal additional computational
                                         Graph Neural Networks (GNNs) have become a prominent approach                                   cost. In our experiments, we implement UnGSL into six representa-
                                         for learning from graph-structured data. However, their effective-                              tive GSL methods, demonstrating consistent performance improve-
                                         ness can be significantly compromised when the graph structure is                               ments. The code is available at https://github.com/UnHans/UnGSL.
                                         suboptimal. To address this issue, Graph Structure Learning (GSL)
                                         has emerged as a promising technique that refines node connec-                                  CCS Concepts
                                         tions adaptively. Nevertheless, we identify two key limitations in                              • Computing methodologies → Neural networks; • Informa-
                                         existing GSL methods: 1) Most methods primarily focus on node                                   tion systems → Data mining.
                                         similarity to construct relationships, while overlooking the quality
                                         of node information. Blindly connecting low-quality nodes and                                   Keywords
                                         aggregating their ambiguous information can degrade the perfor-                                 Graph Structure Learning; Graph Neural Network; Uncertainty
                                         mance of other nodes. 2) The constructed graph structures are often
                                         constrained to be symmetric, which may limit the model’s flexibility                            ACM Reference Format:
                                         and effectiveness.                                                                              Shen Han, Zhiyao Zhou, Jiawei Chen, Zhezheng Hao, Sheng Zhou, Gang
                                            To overcome these limitations, we propose an Uncertainty-
                                                                                                                                         Wang, Yan Feng, Chun Chen, and Can Wang. 2025. Uncertainty-Aware
                                                                                                                                         Graph Structure Learning. In Proceedings of the ACM Web Conference 2025
                                         aware Graph Structure Learning (UnGSL) strategy. UnGSL esti-                                    (WWW ’25), April 28–May 2, 2025, Sydney, NSW, Australia. ACM, New York,
                                         mates the uncertainty of node information and utilizes it to adjust                             NY, USA, 12 pages. https://doi.org/10.1145/3696410.3714927
                                         the strength of directional connections, where the influence of
                                                                                                                                         1   Introduction
                                         nodes with high uncertainty is adaptively reduced. Importantly,
                                         UnGSL serves as a plug-in module that can be seamlessly integrated
                                                                                                                                         Graph neural networks (GNNs) [17, 24, 44] have demonstrated re-
                                         ∗ Corresponding author.                                                                         markable performance in tackling graph-structured data. To date,
                                         † College of Computer Science and Technology, Zhejiang University.
                                                                                                                                         GNNs have evolved with increasingly sophisticated model architec-
                                         ‡ Hangzhou High-Tech Zone (Binjiang) Institute of Blockchain and Data Security.
                                                                                                                                         tures [3, 5, 9, 11, 27, 39] to enhance their capabilities. However, these
                                                                                                                                         model-centric methods often neglect potential flaws in the under-
                                         Permission to make digital or hard copies of all or part of this work for personal or
                                         classroom use is granted without fee provided that copies are not made or distributed           lying graph structure, which can lead to suboptimal performance.
                                         for profit or commercial advantage and that copies bear this notice and the full citation       In practice, graph data frequently exhibit suboptimal characteris-
                                         on the first page. Copyrights for components of this work owned by others than the              tics, such as noisy connections and incomplete information, due
                                         author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
                                         republish, to post on servers or to redistribute to lists, requires prior specific permission   to the inherent complexities and inconsistencies in data collection
                                         and/or a fee. Request permissions from permissions@acm.org.                                     [37, 51].
                                         WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                             To address these issues, Graph Structure Learning (GSL) [6, 13, 23,
                                         © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
                                         ACM ISBN 979-8-4007-1274-6/25/04                                                                30, 49], a data-centric approach, has garnered increasing attention.
                                         https://doi.org/10.1145/3696410.3714927                                                         Beyond learning node representations with GNNs, GSL learns to
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                                                         Shen Han, et al.



       85                                               72                                             Embedding-based                        Uncertainty-aware
                                                                                                    Graph Structure Learning               Graph Structure Learning
       83
Accuracy                                         Accuracy
                                                                                                                    Poison
                                                        70
       81
             Eliminate Low-quality Neighbors                  Eliminate Low-quality Neighbors
             Eliminate Random Neighbors                       Eliminate Random Neighbors                            Enhance                                 Enhance
       79 0.0                                           68 0.0     0.1      0.2     0.3     0.4
                  0.1      0.2     0.3     0.4
             Ratios of Eliminated Neighbors                   Ratios of Eliminated Neighbors
                      (a) Cora                                      (b) Citeseer
                                                                                                     : High-quality Node       : Low-quality Node         : Embedding Vector
Figure 1: Performance varies with the ratios of eliminated                                                                                     : Directed Edge
                                                                                                                       : Undirected Edge
neighbors on Cora and Citeseer datasets. Here we first con-
structed a graph based on GRCN [47]. We then eliminate a                                          Figure 2: Illustration of how our UnGSL differs from existing
certain ratio of neighbors with the highest entropy for each                                      embedding-based GSL methods. Existing GSL learns sym-
node, and evaluate the performance of GCN on such a pruned                                        metric relationships, leading to a dilemma when managing
graph. For comparison, we also report the performance un-                                         connections between high-quality and low-quality nodes. In
der random elimination.                                                                           contrast, UnGSL learns asymmetric relationships, allowing
                                                                                                  low-quality nodes to benefit from high-quality nodes while
refine node connections and edge weights. This approach has been                                  mitigating the negative influence of low-quality nodes.
shown to effectively enhance the accuracy of GNNs on downstream
tasks while improving their resilience to topological perturbations
                                                                                                    a negative impact on the high-quality node. Symmetric relation-
[26, 51]. Early work on GSL directly treated the graph structure (i.e.,
                                                                                                    ships fail to account for this disparity, leading to a dilemma in the
the adjacency matrix) as learnable parameters. However, due to
                                                                                                    learning process. This inspires us to explore asymmetric struc-
the large parameter space, these strategies often incur substantial
                                                                                                    ture learning. By modeling directional relationships separately,
computational overhead and are difficult to train effectively [14, 23].
                                                                                                    we allow the low-quality node to benefit from the high-quality
More recently, research has shifted towards embedding-based GSL
                                                                                                    node’s information while reducing the adverse influence in the
[6, 22, 28, 41], which constructs the adjacency matrix based on the
                                                                                                    opposite direction, thus protecting the high-quality node from
similarity of node embeddings. Various similarity metrics, such
                                                                                                    negative effects. Although a few studies [30, 34, 35] have begun
as cosine similarity [22, 41] or neural networks [28], have been
                                                                                                    to explore asymmetric graph structure learning, they typically
employed. These methods aim to increase graph homophily and
                                                                                                    restrict the asymmetry to relations between labeled and unla-
typically achieve state-of-the-art performance, as nodes with similar
                                                                                                    beled nodes, overlooking the richer relationships between the
features (or embeddings) are more likely to be connected.
                                                                                                    vast majority of unlabeled nodes.
   Despite their success, we identify two key limitations in these
embedding-based GSL methods:                                                                         To overcome these limitations, we propose an uncertainty-aware
                                                                                                  graph structure learning (UnGSL) method that considers nodes’
 • These methods mainly rely on embedding similarity for
                                                                                                  information quality to learn an asymmetric graph structure. UnGSL
   graph construction while neglecting the quality of node in-
                                                                                                  directly utilizes the uncertainty (Shannon Entropy [1]) of the node
   formation. Given the critical role of edges in GNNs as conduits
                                                                                                  in classification to indicate the node’s information quality, and
   for information propagation, it is essential to evaluate the quality
                                                                                                  conducts theoretical analyses to demonstrate that blindly aggre-
   of the information being propagated. Aggregating unclear or
                                                                                                  gating information from high-uncertainty nodes would lift the
   ambiguous information from neighbor nodes can disrupt the em-
                                                                                                  lower bound of uncertainty for the target node. Building on this,
   bedding learning of the target node. Constructing connections
                                                                                                  UnGSL leverages a learnable node-wise threshold to differentiate
   based solely on node similarity, without assessing the quality of
                                                                                                  low-quality neighbors from high-quality ones, and adaptively re-
   the node’s information, may lead to suboptimal performance. To
                                                                                                  duces directional edge weights from those low-quality neighbors.
   empirically validate this point, we conduct a simple experiment
                                                                                                  Notably, our UnGSL is simple and can be easily incorporated into
   using a representative GSL method (GRCN [47]). As shown in
                                                                                                  various embedding-based GSL methods, boosting their performance
   Figure 1, removing a certain proportion of neighbors with the
                                                                                                  with minor extra computational overhead.
   highest entropy results in a significant performance gain.
                                                                                                     In summary, this work makes the following contributions:
 • These methods often generate symmetric graph structures,
   which can hinder their effectiveness. Current embedding-                                       • We highlight the necessity of modeling a node’s uncertainty in
   based methods tend to construct symmetric relationships be-                                      graph structure learning, and theoretically demonstrate that the
   tween nodes, implying that both nodes exert an equal and bidirec-                                uncertainty of a node after GNN layer is positively correlated
   tional influence during the GNN learning process. This imposed                                   with those of its neighbors.
   symmetry can constrain the model’s flexibility and capacity, par-                              • We propose a simple yet novel uncertainty-aware graph structure
   ticularly when the connected nodes differ in quality. For example,                               learning strategy (UnGSL), which can be seamlessly integrated
   consider a scenario where a high-quality node is linked to a low-                                with various embedding-based GSL models to mitigate the direc-
   quality node (refer to Figure 2). While the high-quality node can                                tional impact of high-uncertainty nodes.
   provide valuable information that greatly benefits the low-quality                             • We conduct extensive experiments to demonstrate that UnGSL
   node, the reverse influence from the low-quality node may have                                   can consistently boost existing embedding-based GSL models
Uncertainty-Aware Graph Structure Learning                                                      WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


    across seven benchmark datasets, with an average performance           encoder and the adjacency matrix S. The regularization term LReg
    increase of 2.07%.                                                     are introduced by existing GSL methods to achieve diverse desirable
                                                                           properties in the learned adjacency matrix S, such as sparsity [23],
2     Preliminaries                                                        feature smoothness [35], and connectivity [47] in the graph struc-
In this section, we introduce basic notations and background on            ture. For example, PROGNN [23] employs 𝑙 1 norm penalization on
GNNs and GSL methods.                                                      the learned adjacency matrix to enhance the sparsity of the graph
   Consider the graph G = (V, E, A, X), where V represents a               structure. 𝜆 is a trade-off hyperparameter.
set of 𝑛 nodes {𝑣 1, ..., 𝑣𝑛 } and E represents the set of edges. Let A       Traditional GSL methods [14, 23] that treat each edge S𝑖 𝑗 as a
be the initial adjacency matrix of the graph, where A𝑖 𝑗 = 1 if an         learnable parameter often suffer from significant computational
edge exists between node 𝑣𝑖 and 𝑣 𝑗 ; otherwise, A𝑖 𝑗 = 0. The matrix      overhead and are challenging to train efficiently. Recent research
X = [𝑥 1, ..., 𝑥𝑛 ] ∈ R𝑛×𝑑 represents the node feature matrix, where       has focused on embedding-based GSL methods [6, 22, 28, 41] , which
each column 𝑥𝑖 corresponds to the feature vector of node 𝑣𝑖 . Let          employ an auxiliary GNN encoder to extract node embeddings
                                                               Í
D denote the diagonal degree matrix defined as D𝑖𝑖 = 1 + 𝑗 A𝑖 𝑗 ;          from the initial graph and estimate embedding similarities as edge
                                                                           weights:
and Â denotes the normalized adjacency matrix with self-loop, i.e.,
         1             1
Â = D − 2 (A+I)D − 2 for symmetric normalization or Â = D −1 (A+I)                                 E = GNN(A, X),                                (3)
for row normalization.                                                                            S𝑖 𝑗 = S 𝑗𝑖 = 𝜙 (E𝑖 , E 𝑗 ).                     (4)

2.1     Graph Neural Networks                                              Here E is the node embedding matrix, which differs from Z as
                                                                           it is specifically used for constructing the graph structure. Note
Graph Neural Networks (GNNs) have become a prominent ap-
                                                                           that existing works generally utilize different GNN architectures
proach for learning from graph-structured data, where node repre-
                                                                           and input structure compared with those used to compute the
sentations are learned by iteratively aggregating and transforming
                                                                           embedding Z when constructing the embedding E. 𝜙 (·) is a metric
information from neighboring nodes. In recent years, various de-
                                                                           function (i.e., cosine similarity) used to calculate the similarity
signs for the aggregation and transformation processes have given
                                                                           between nodes.
rise to different GNN models[24, 38, 44]. Among these, the Graph
                                                                               Although the structure modeling paradigm in Equation (4) is
Convolutional Network (GCN) [24] stands out as one of the most
                                                                           widely adopted in GSL models, it suffers from two key limitations:
widely adopted and influential architectures. The operation at the
𝑙-th layer in GCN can be formulated as:                                    • This paradigm relies on embedding similarity while ne-
                                                                             glecting the quality of node information. Only semantic
                        Z (𝑙 ) = 𝜎 ( ÂZ (𝑙 −1) W (𝑙 ) ),           (1)      similarities between embeddings E𝑖 and E 𝑗 are considered in this
where 𝜎 (·) denotes the activation function, W (𝑙 ) is a learnable           structure modeling paradigm, while the varying uncertainties of
parameter matrix used to transform the node features.                        them, which reflect their information quality, are neglected. This
   Given the critical role of the graph structure in GNNs, which             may undermine the quality of embeddings of target nodes when
determines the sources of information aggregation, ensuring the              aggregating inferior information from low-quality neighbors (as
quality of graph structure is of paramount importance. Recent                validated by the preliminary experiment in the Introduction) .
work demonstrates that suboptimal graph structures, even with the          • This paradigm constraining the graph to be symmetric,
introduction of a small percentage of noisy edges or topological             which potentially hinder effectiveness of GSL models. The
perturbations (e.g., 10%), can significantly degrade the performance         pair-wise similarity constrains the learned edge between 𝑣𝑖 and
of GNNs (e.g., 25%) [26, 51].                                                𝑣 𝑗 to be bidirectional, overlooking their unequal influence due to
                                                                             varying information quality. For example, if E𝑖 contains higher-
2.2     Graph Structure Learning                                             quality information than E 𝑗 , the constructed edge S 𝑗𝑖 can provide
                                                                             valuable information that greatly benefits the low-quality node
Graph Structure Learning (GSL) aims to enhance the accuracy and
                                                                             𝑣 𝑗 while the edge S𝑖 𝑗 propagates inferior information to poi-
robustness of GNNs by learning a refined adjacency matrix S, where
                                                                             son the embedding of 𝑣𝑖 . By modeling directional relationships
S𝑖 𝑗 denotes the edge weight between node 𝑣𝑖 and node 𝑣 𝑗 . For
                                                                             separately, we enable the low-quality node to benefit from the
convenience, we also use S𝑖 𝑗 to indicate whether there exists a edge
                                                                             high-quality node’s information while mitigating the negative
between node 𝑣𝑖 and 𝑣 𝑗 . When S𝑖 𝑗 > 0, it indicates the presence
                                                                             impact in the reverse direction.
of a directed edge from 𝑣 𝑗 to 𝑣𝑖 , with the edge weight equal to S𝑖 𝑗 .
Conversely, S𝑖 𝑗 = 0 indicates that no edge exists.                           Although a few works [30, 34, 35] have been proposed to learn
    The quality of the learned adjacency matrix is often examined          asymmetric graphs by generating directed edges S𝑖 𝑗 from the la-
on the downstream tasks. Conventionally, S is fed into a GNN               beled node 𝑣 𝑗 to unlabeled node 𝑣𝑖 and constraining S 𝑗𝑖 = 0, which
encoder to obtain node representations, which are used to evaluate         facilitates the propagation of label information and avoids intro-
performance on downstream tasks. The objective function of most            ducing inconsistency to labeled nodes. However, these methods
GSL methods can be generally formulated as:                                completely rely on annotated labels and fail to learn reasonable
                                                                           asymmetric connections between the vast majority of unlabeled
                   L = L𝑇 𝑎𝑠𝑘 (Z, Y) + 𝜆L𝑅𝑒𝑔 (Z, S),                (2)
                                                                           nodes.
where Z = GNN(S, X) is the node embedding matrix, LTask aims to               Given the flaws of existing methods, we argue for the necessity
utilize the labels Y as a supervised signal to optimize both the GNN       of incorporating node uncertainty into graph structure learning to
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                                 Shen Han, et al.



                        1.25                                                 Uncertainty Estimation via Entropy. Entropy measures the


    Average Entropy
                                                                          information uncertainty within a probability distribution [1]. For
                        1.00                                              node classification tasks, the entropy of the classification proba-


     of Its Neighbors
                                                                          bilities reflects the classifier’s certainty in assigning the node rep-
                        0.75                                              resentation to a specific class. A high-entropy node indicates that
                        0.50                                              its representation carries significant uncertainty, making it chal-
                                                                          lenging for the classifier to reach a confident decision. Aggregating
                                  0.5         1.0        1.5              information from such nodes can poison the target node’s repre-
                               Node Entropy After GNN Aggregation         sentation, hindering the generation of accurate predictions. Given
Figure 3: Visualization of node entropy after GNN aggrega-                probabilities 𝑝𝑖 of node 𝑣𝑖 , its entropy is defined as:
tion (i.e., 𝑢𝑖 ) alongside the average entropy of its neighbors                                             𝐾
                                                                                                           ∑︁
(i.e., 𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 𝑢 ′𝑗 ) on Cora dataset.
      Í
                                                                                                  𝑢𝑖 = −         𝑝𝑖𝑘 log(𝑝𝑖𝑘 ).                    (7)
learn an optimal asymmetric structure. We propose the uncertainty-                                         𝑘=1
aware graph structure learning (UnGSL) method to enhance GSL              We further discuss the uncertainty metric for unsupervised learning
models, which leverage learnable node-wise thresholds to identify         scenario at the end of this section.
high-uncertainty neighbors and adaptively reduce directional edge            Formally, to demonstrate the impact of the uncertainty of neigh-
weights from those low-quality neighbors.                                 bors along GNNs, we have the following proposition, with detailed
                                                                          proof provided in Appendix A:
3         Methodology                                                        Proposition 1. Define the logits of the initial node feature matrix
In this section, we first conduct theoretical and empirical analyses      as O′ = XW. For a given node 𝑣𝑖 , let 𝑢𝑖 denote its entropy after
to demonstrate the detrimental impact of neighbors with high un-          GNN aggregation. ∀𝑣 𝑗 ∈ N (𝑣𝑖 ), let 𝑢 ′𝑗 denote the entropy of its initial
certainty levels on GNN learning (Subsection 3.1). We then present        features. Then the entropy of 𝑣𝑖 and the entropy of 𝑣 𝑗 ∈ N (𝑣𝑖 ) satisfy
the proposed uncertainty-aware graph structure learning method            the following inequality:
in detail (Subsection 3.2).                                                                                 ∑︁
                                                                                                   𝑢𝑖 ≥             𝜂 𝑗 𝑢 ′𝑗 ,                    (8)
3.1             Analyses on the Impact of Neighbor                                                         𝑣 𝑗 ∈ N (𝑣𝑖 )

                Uncertainty                                               where
                                                                                                                 Â𝑖 𝑗 (𝑂 ′𝑗𝑘 + 1)
                                                                                                       Í𝐾
Aggregating unclear or ambiguous information can intuitively dis-                                        𝑘=1
                                                                                           𝜂𝑗 = Í                                      .           (9)
                                                                                                                 Í𝐾              ′
rupt the learning process of target nodes, thereby negatively af-                                   𝑣 𝑗 ∈ N (𝑣𝑖 ) 𝑘=1 Â𝑖 𝑗 (𝑂 𝑗𝑘 + 1)
fecting the performance of Graph Neural Networks (GNNs). In this          For coefficient 𝜂 𝑗 , we have 0 < 𝜂 𝑗 < 1 and 𝑣 𝑗 ∈ N (𝑣𝑖 ) 𝜂 𝑗 = 1.
                                                                                                                               Í
section, we aim to conduct both theoretical and empirical analyses
to substantiate this claim. To begin, we introduce several formal            Discussion. Proposition 1 establishes the bounded relationship
concepts to facilitate these analyses.                                    between the uncertainty of a targe node after aggregation and the
   Semi-supervised Node Classification Task. For convenience,             uncertainty of its neighbors. It suggests blindly aggregating infor-
we refer to recent analytical work on GNNs [31] and focus our             mation from high-uncertainty nodes would lift the lower bound
theoretical analysis on the semi-supervised node classification task,     of the uncertainty for the target node. Removing or mitigating the
which is the most common and widely studied scenario. Neverthe-           impact of these high-uncertainty nodes, may significantly decrease
less, at the end of this section, we will also discuss how our method     the uncertainty of the target node, leading to performance gains.
can be adapted to the unsupervised learning scenario. Following              Empirical Analyses. We conduct simple experiments to em-
the definitions in [31], we consider a 𝐾-class classification problem     pirically demonstrate that aggregating neighbors with higher un-
and employ a linear classification model. Formally, the classification    certainty would result in a high-uncertainty node. Specifically, we
logits can be expressed as:                                               train the model with 1-layer GCN and linear classifier on given
                                                                          datasets. Then we visualize the entropy of a node after aggrega-
                                   O = D −1 (A + I)XW,              (5)
                                                                          tion (i.e., 𝑢𝑖 ) alongside the average entropy of its neighbors (i.e.,
where W ∈ R𝑑 ×𝐾 is the linear classification matrix. Assume the                                   ′
                                                                          Í
                                                                            𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 𝑢 𝑗 ). As shown in Figure 3, we observe a strong posi-
logit in matrix O are bounded by the scalar 1, i.e., max |O𝑖 𝑗 | <        tive correlation between the entropy of a node after aggregation
1. From a local perspective for node 𝑣𝑖 , its probabilities can be        and the average entropy of its neighbors, which substantiates our
formulated as :                                                           proposition (results on more datasets are provided in Appendix
                              O𝑖 + 1𝐾
                      𝑝𝑖 = Í𝐾               ,                   (6)       D.1).
                             𝑗=1 (O𝑖 𝑗 + 1)                                  The above analyses clearly illustrates the impact of node un-
where 1𝐾 is a 𝐾-dimensional all-ones vector. Here we simply omit          certainty in aggregation process in GNNs. Blindly connecting and
the nonlinear exponential function in the softmax, given that our         aggregating information from nodes with high uncertainty may
primary focus is on the impact of aggregation on node uncertainty         undermine the performance of the nodes themselves. It is therefore
and the exponential function mainly serves to generate positive           important to consider node uncertainty in graph structure learning
values. This simplification has been adopted in prior work [48, 50],      to learn a reasonable asymmetric structure. Specifically, one can
and it can also be considered a first-order Taylor approximation.         prevent a node falling into a high-uncertainty region by weakening
Uncertainty-Aware Graph Structure Learning                                                        WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


its connections to neighbors with high uncertainty. Meanwhile, it           learning of an optimal graph that mitigates the negative impact of
can receive more stable information via strengthened connections            high-uncertainty neighbors in GNN learning.
with low-uncertainty nodes.                                                    Model-agnostic. UnGSL refines the learned adjacency matrix
                                                                            S by utilizing uncertainty-aware weights. In essence, UnGSL just
3.2     Uncertainty-aware Graph Structure                                   replaces the calculation of the adjacency matrix from Equation
        Learning                                                            (4) to Equation (10) for GSL methods. Therefore, UnGSL can be
Given the importance of considering uncertainty in graph struc-             seamlessly integrated into existing GSL methods to further enhance
ture learning, we propose the simple yet novel uncertainty-aware            their ability to learn graphs, improving the performance of GNNs
graph structure learning (UnGSL) method that leverages learnable            on downstream tasks. Comprehensive experiments supporting this
node-wise thresholds to distinguish low-quality neighbors from              can be found in Subsection 4.2 and Subsection 4.4.
high-quality ones and adaptively refines edges based on their un-              Asymmetric Graph Structure. UnGSL proposes learning an
certainty levels. Beyond similarity, we introduce uncertainty to            asymmetric graph, where the edge weights between nodes differ
guide the learning of the refined adjacency matrix. Specifically, the       based on their uncertainty levels. Specifically, UnGSL weakens
uncertainty estimated through entropy is transformed into con-              edges from uncertain nodes to confident nodes, mitigating the
fidence scores. Afterwards, UnGSL applies learnable node-wise               impact of inferior information. Meanwhile it enhances the edge in
thresholds to split neighbors with different levels of confidence into      the opposite direction to improve the representations of uncertain
two groups (i.e., high-confidence and low confidence) for each node.        nodes.
Finally, it amplifies edge weights from confident neighbors while              Notably, several methods that construct directed edges from la-
reducing edge weights from uncertain ones. In summary, we refine            beled nodes to unlabeled nodes [30, 35] can be viewed as specific
the generation of adjacent matrix of existing GSL from Equation             cases of UnGSL, as the embeddings of labeled nodes are directly opti-
(4) into Equation (10):                                                     mized during training and therefore more likely to exhibit lower un-
                                                                            certainty. In contrast, UnGSL models the asymmetric relationships
                         Ŝ𝑖 𝑗 = S𝑖 𝑗 · 𝜓 (𝑒 −𝑢 𝑗 − 𝜀𝑖 ),           (10)    among the predominantly unlabeled nodes, leading to superior
 Here 𝑢 𝑗 is entropy of node 𝑣 𝑗 estimated during pretraining stage, 𝜀𝑖     performance. We further empirically validate this in Subsection 4.2.
is the learnable node-wise threshold of node 𝑣𝑖 , 𝜓 (·) is an activate         Efficiency. UnGSL contains only 𝑛 learnable parameters and
function, S is the generated adjacency matrix of the GSL model.             refines the existing edges of the given graph without generating
    Equation (10) first normalizes node uncertainty into confidence         new ones. The additional operations introduced by UnGSL have
within the interval (0, 1) , and then leverages node-wise learnable         a complexity of 𝑂 (𝑛 + 𝑚), where 𝑛 and 𝑚 denote the number of
thresholds 𝜺 to distinguish low-confidence neighbors (i.e., 𝑒 −𝑢 𝑗 < 𝜀𝑖 )   nodes and edges in the graph. Therefore, UnGSL imposes minimal
from high-confidence ones (i.e., 𝑒 −𝑢 𝑗 ≥ 𝜀𝑖 ) and adaptively refines       computational costs on the base GSL models. We provide empirical
the corresponding edges using the following activation function:            evidence to demonstrate the efficiency of UnGSL in Subsection 4.6.
                                                                              Adaptive to Unsupervised Scenarios. Despite the mainstream
                                𝜏 · 𝑠 (𝑥), 𝑥 ≥ 0,
                     𝜓 (𝑥) =                                        (11)    focus of GSL research on supervised learning, our approach can be
                                 𝛽,        𝑥 < 0,
                                                                            generalized to unsupervised GSL scenarios. The main challenge lies
where 𝑠 (·) is the sigmoid function, 𝜏 is a hyperparameter ampli-           in determining an appropriate uncertainty metric for unsupervised
fies the edge weights from high-confidence neighbors, 𝛽 is a hy-            GSL method. We suggest employing the self-supervised structure
perparameter that controls the reduction of edge weights from               learning loss (i.e., node contrastive learning loss [29]) as a proxy
low-confidence neighbors.                                                   for uncertainty, which can be formulated as:
    For the training of UnGSL, we employ the same loss function                                              1
(i.e., Equation (2)) and the same training procedure as the original                                 𝑢𝑖 = (𝑙 (𝑧𝑖 , 𝑧˜𝑖 ) + 𝑙 (𝑧˜𝑖 , 𝑧𝑖 ),       (12)
                                                                                                             2
GSL model, except we modify the generation of the adjacency
                                                                            where
matrix from Equation (4) to Equation (10), i.e., using Ŝ to replace the
                                                                                                                        𝑒 sim(z𝑖 ,z̃𝑖 )/𝑡
generated adjacency matrix S in existing GSL methods. Therefore,                              𝑙 (𝑧𝑖 , 𝑧˜𝑖 ) = − log Í𝑛                        . (13)
UnGSL can be seamlessly integrated into existing GSL methods                                                         𝑘=1
                                                                                                                            𝑒 sim(z𝑖 ,z̃𝑘 )/𝑡
with minimal additional computational cost.                                 Here sim(·) is the cosine similarity function. z𝑖 and z̃𝑖 are the embed-
    Implementation Details. We begin by selecting a specific GSL            dings of node 𝑣𝑖 learned by GNN from target graph and augmented
                                                                            graph, respectively. 𝑛𝑘=1 𝑒 sim(z𝑖 ,z̃𝑘 )/𝑡 is cumulative similarity be-
                                                                                                  Í
method that UnGSL aims to enhance , pretraining it to obtain the
classifier and estimating uncertainty using entropy (Equation (7)).         tween 𝑧𝑖 and embeddings of different nodes in the augmented graph.
Next, the chosen GSL method is re-trained, during which Equation            𝑡 is the temperature parameter. The GCL loss can measure the invari-
(10) is employed to generate the adjacency matrix. Finally, the             ance of node representations to feature or structure perturbations,
learned graph structure is leveraged for downstream applications.           where this invariance can be interpreted as the uncertainty of nodes
                                                                            with respect to their original features and structure.
3.3     Discussion
Based on the above formulation, we next discuss several key ad-             4    Experiments
vantages of UnGSL:                                                          In this section, we conduct experiments to evaluate the effectiveness
   Uncertainty-aware. UnGSL considers the information quality               of the proposed UnGSL strategy. Our experiments aim to answer
of nodes during the structure modeling process, facilitating the            the following research questions:
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                                Shen Han, et al.


Table 1: Node classification accuracy±std comparsion(%). Each experiment is repeated 10 times with different random seeds.
"OOM" denotes out of memory. The top-performing results are marked in bold.
        Model              Cora       Citeseer     Pubmed       BlogCatalog   Roman-empire       Flickr       Ogbn-arxiv
         GRCN                84.70±0.31         72.49±0.77      78.94±0.16   76.17±0.23         44.29±0.28        59.55±0.31          OOM
      GRCN+UnGSL             85.84±0.51         73.88±0.55      79.59±0.48   76.78±0.11         52.54±0.31        63.85±0.22          OOM
      PROGNN                 80.39±0.41         67.94±0.52            OOM    76.17±0.22            OOM            61.74±0.24          OOM
   PROGNN+UnGSL              81.86±0.55         69.66±0.27            OOM    76.82±0.19            OOM            61.92±0.43          OOM
         PROSE                81.1±0.45          72.3±0.37       83.3±0.71   75.31±0.17         55.61±0.34        59.77±1.13       71.22±0.31
      PROSE+UnGSL            81.90±0.31         73.10 ±0.32     83.86±0.30   75.77±0.36         56.17±0.31        64.72±0.97       71.46±0.12
         IDGL                 84.50±0.5         72.49±0.67      82.83±0.33   89.66±0.28         46.67±0.56        85.77±0.16       70.45 ± 0.36
      IDGL+UnGSL             84.90±0.42         73.74±1.01      83.33±0.32   92.13±0.18         47.05±0.59        86.42±0.35       71.02±0.12
         SLAPS               72.89±1.02         70.05±0.83      70.96±0.99   91.62±0.39         65.35±0.45        83.89±0.70          OOM
      SLAPS+UnGSL            74.28±0.95         72.08±0.94      72.25±1.48   91.89±0.41         66.20±0.33        84.92±0.68          OOM
      SUBLIME                83.30±1.04         72.36±0.68      80.41±0.69   95.20±0.21         63.48±0.53        88.68±0.23       71.37±0.13
   SUBLIME+UnGSL             84.24±0.91         74.34±0.73      80.84±0.92   96.18±0.38         65.45±0.32        89.23±0.20       71.82±0.15


Table 2: Performance comparison between the CUR decom-                       SLAPS [13]), and a self-supervised model (SUBLIME [29]). Detailed
position and UnGSL modules.                                                  information of these baselines is provided in Appendix C. For all
         Model       Cora    Citeseer Roman-empire                           models, we report the average performance and standard deviations
       GRCN+CUR 84.89±0.22 73.35±0.46                    44.04±0.28          of 10 runs with different random seeds.
      GRCN+UnGSL 85.84±0.51 73.88±0.55                   52.54±0.31          4.1.3 Configuration. With regard to hyperparameter settings, only
       IDGL+CUR 84.73±0.23 72.93±0.85                       OOM              the hyperparameter 𝛽 and the learning rate of the threshold 𝜀
      IDGL+UnGSL 84.90±0.42 73.74±1.01                   47.05±0.59          require fine-tuning. The threshold 𝜀 is consistently initialized as
                                                                             a random number in the range [0, 1] across various datasets. The
• RQ1: Does the integration of UnGSL into existing GSL methods               hyperparameter 𝜏 is simply fixed as a constant value of 2. The
  lead to performance improvements?                                          hyperparameter 𝛽 can be quickly tuned using parameter search
• RQ2: What is the impact of key configurations (e.g., node-wise             tools like Bayesian optimization within the range of [0.001,1]. We
  thresholds, asymmetric graphs, hyperparameters 𝛽 and 𝜏) on                 optimized the UnGSL using Adam optimizer, with the learning rate
  UnGSL performance?                                                         selected from range [0.0001, 0.01]. For detailed hyperparameter
• RQ3: Can UnGSL enhance the robustness of GNNs against struc-               settings please refer to Appendix D.2.
  tural noise, feature noise and lable noise?                                   For all baselines, we strictly adhere to their original settings for
• RQ4: How well does UnGSL generalize across different GNN                   hyperparameter tuning to ensure that they attain best performance.
  backbones?                                                                 All GSL methods are evaluated based on the performance of GNNs
• RQ5: Does UnGSL introduce significant additional computa-                  on downstream tasks when using the learned structure. We also
  tional overhead?                                                           consider cross-architecture scenarios in Subsection 4.5, where GSL
                                                                             training and downstream tasks use different GNN architectures.
4.1     Experiment Settings
                                                                             4.2    Main Results (RQ1)
4.1.1 Datasets. To comprehensively evaluate UnGSL’s performance
                                                                             4.2.1 Comparison to Vanilla GSL Models. Table 1 presents the ex-
on node classification, we follow previous works [47, 51] and select
                                                                             perimental results of the UnGSL module applied to various GSL
7 commonly used datasets, including four homophilous citation
                                                                             models. We can observe that: 1) UnGSL significantly improves the
datasets [20, 33] (Cora, Citeseer, Pubmed and Ogbn-arxiv) and three
                                                                             node classification accuracy for all GSL models across all datasets
heterophilous datasets (Blogcatalog [21] , Roman-Empire [32] and
                                                                             with an average increase of 2.07%, achieving new state-of-the-art
Flickr [21]). The chosen datasets cover a wide range of homophily
                                                                             performance in the GSL literature. The improvement brought by
levels and graph sizes, allowing us to demonstrate UnGSL’s effec-
                                                                             UnGSL is particularly pronounced on GRCN [47], achieving an
tiveness under various conditions. For a fair comparison, we strictly
                                                                             average improvement of 5.12%. These results empirically demon-
follow the data split settings used in the newly proposed benchmark
                                                                             strate UnGSL’s effectiveness in further denoising the learned graph
for GSL [51]. Detailed statistics of these datasets are provided in
                                                                             structure from the perspective of uncertainty, resulting in more ac-
Appendix B.
                                                                             curate predictions. 2) For the self-supervised GSL model SUBLIME
4.1.2 Baselines. To demonstrate UnGSL’s generalizability across              [29], UnGSL continues to achieve higher accuracy by utilizing con-
different GSL models, we select 6 state-of-the-art GSL algorithms            trastive loss in Equation (12) as a proxy of uncertainty estimation
corresponding to the chosen datasets as baselines, including super-          to guide structure refinement, resulting in an average improvement
vised models (GRCN [42], PROGNN [23], IDGL [6], PROSE [41],                  of 1.40%.
Uncertainty-Aware Graph Structure Learning                                                                                            WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


Table 3: Ablation study on the UnGSL when integrating with                                                  Table 4: Comparsion of different 𝜏 on Cora and Citeseer
GRCN model [47].                                                                                            datasets.
       Method         Cora      Citeseer Roman-empire                                                                 Method  𝜏 =1       𝜏 =2      𝜏 =3
        Fixed 𝜺    85.23±0.15 73.2±0.96    44.72±0.14                                                                       Cora 85.12±0.47 85.84±0.51 84.87±0.55
    Symmetrize Ŝ 85.03±0.40 73.74±0.49 52.28±0.025
                                                                                                                           Citeseer 72.77±0.90 73.88±0.55 73.09±0.42
        UnGSL      85.84±0.51 73.88±0.55 52.54±0.31
                                                                                                            Table 5: Robustness analysis with random structural noise
                              Cora                                         Citeseer                                     Cora                      Citeseer
              86.0                                           74.0                                           injection on Cora dataset.   74
              85.5                                           73.5

Accuracy(%)                                    Accuracy(%)                                    Accuracy(%)                                       Accuracy(%)
                                                                                                                                    Edge Deletion Level                     Edge Addition Level
                                                                                                            84                                                72
              85.0                                           73.0                                                Methods       0%     20% 40% 60% 80%                  0%     20% 40% 60% 80%
                                                                                                                        84.70 83.00 79.87 78.47075.17 84.70 78.03 75.13 73.20 69.43
              84.5                                           72.5                                           82 GRCN
                                                                                                            GRCN+UnGSL 85.84 84.80 81.83 80.20 76.32 85.84 79.17 76.27 74.42 70.60
              84.0                                           72.0                                            Improve(%) 1.34% 2.16% 2.45% 2.21%68
                                                                                                                                                1.53% 1.34% 1.46% 1.65% 1.66% 1.69%
                     0.0 0.2 0.4 0.6 0.8 1.0                        0.0 0.2 0.4 0.6 0.8 1.0                      0.0 0.2 0.4 0.6 0.8 1.0                           0.0 0.2 0.4 0.6 0.8 1.0
                              value                                          value                             IDGL     value                                 value77.24 74.43 73.11
                                                                                                                         84.50 81.60 80.31 76.57 72.18 84.50 78.59
Figure 4: Comparsion of different 𝛽 on Cora and Citeseer                                                    IDGL+UnGSL 84.90 82.88 81.41 77.83 73.24 84.90 80.02 78.77 76.25 75.31
datasets.                                                                                                    Improve(%) 0.47% 1.57% 1.38% 1.65% 1.47% 0.47% 1.82% 1.98% 2.45% 3.01%

4.2.2 Comparison to the Label-oriented Directed GSL Module. Here
                                                                                                            UnGSL with 𝛽 = 0. The results indicate that high-uncertainty neigh-
we compare the performance of UnGSL with CUR decomposition
                                                                                                            bors still contain valuable information that enhances the node’s
[30], another GSL module proposed recently to learn asymmetric
                                                                                                            representation, and removing connections from high-uncertainty
graph structure by constructing directed edges from labeled nodes
                                                                                                            nodes blindly may lead to information loss. Results for additional
to unlabeled nodes. Table 2 shows the experiment results of UnGSL
                                                                                                            GSL models are provided in Appendix D.3.
and CUR decomposition. We can observe that: 1) UnGSL consis-
tently outperforms CUR decomposition on supervised GSL methods.                                             4.3.4 Analysis on Hyperparameter 𝜏. We conduct additional exper-
2) UnGSL demonstrates better scalability with large-scale graphs                                            iments to evaluate the model’s performance with different values
compared to CUR decomposition. For example, IDGL+UnGSL is                                                   of 𝜏 on Cora and Citeseer datasets using the base GSL model GRCN
applicable to the Roman-Empire dataset and further enhances ac-                                             [47]. The results are presented in Table 4. As can be seen, as 𝜏
curacy, while IDGL+CUR encounters an out-of-memory problem.                                                 increases, the performance initially improves and then declines.
                                                                                                            Hyperparameter 𝜏 controls influence of high-confidence neighbors.
4.3             Ablation Studies (RQ2)                                                                      While fine-tuning 𝜏 could potentially enhance model performance,
4.3.1 Effects of Adaptive Threshold 𝜺. To validate the effectiveness                                        we find that simply setting 𝜏 = 2 is sufficient to achieve good results.
of the learnable threshold, we consider a variant where we fixed
the 𝜺 in UnGSL during training phase. Specifically, for each node,                                          4.4     Robustness Analysis (RQ3)
we select a fixed proportion of the most uncertain neighbors, as                                            To evaluate the robustness of UnGSL under structural noise, we
high-uncertainty neighbors. This proportion is consistent across all                                        compare the accuracy of the base GSL model and GSL+UnGSL un-
nodes. Next, we reweight edges by Equation (10). As shown in Table                                          der different noise levels and calculate the relative improvement
3, the learnable 𝜺 outperforms the fixed 𝜺 across 3 datasets. The                                           achieved by GSL+UnGSL. Specifically, we randomly add or remove
results indicate that the learnable threshold effectively differentiates                                    edges from the original dataset. We conduct experiments on the
high-uncertainty neighbors from low-uncertainty neighbors.                                                  Cora dataset, and the results are presented in Table 5. We can ob-
                                                                                                            serve that: 1) UnGSL consistently enhances the performance of GSL
4.3.2 Superiority of Asymmetric Graph. To demonstrate the supe-
                                                                                                            models across different noise levels. 2) With increasing levels of
riority of the asymmetric edges constructed by UnGSL, we sym-
                                                                                                            noise, UnGSL achieves greater relative improvements. For example,
metrize the graph refined by UnGSL during training phase, generat-
                                                                                                            in the random edge addition scenario, IDGL+UnGSL achieves a
ing an symmetric graph structure. As shown in Table 3, symmetriz-
                                                                                                            3.01% improvement under 80% edge perturbation, which is signifi-
ing graph Ŝ of UnGSL degrades its performance on node classifica-                                          cantly higher than the 0.47% improvement observed at the 0% noise
tion. The underlying reason is that the symmetrization operation                                            level. In summary, the results suggest that UnGSL enhances robust-
may perturb the graph by weakening edges from low-uncertainty                                               ness against structural noise. We also validate UnGSL’s robustness
neighbors and strengthening edge weights from high-uncertainty                                              against feature noise and label noise, and the results are presented
neighbors, which violates the core mechanism of UnGSL.                                                      in Appendix D.4.
4.3.3 Analysis on Hyperparameter 𝛽. To explore the role of 𝛽 in
UnGSL’s activation function 𝜓 (·), we assign different values to                                            4.5     Generalizability on GNN Models (RQ4)
𝛽 from the interval [0, 1] and evaluate the corresponding perfor-                                           We further consider the scenario where GSL training and down-
mance on the GRCN model [47]. As shown in Figure 4, we can                                                  stream tasks use different GNN backbones. We evaluate the gener-
observe that: (1) UnGSL enhances the accuracy of the GSL model                                              alizability of the learned structures generated by the GSL+UnGSL
by reducing edge weights with low-uncertainty neighbors using an                                            on several other GNN models, including SGC [44], APPNP [15],
appropriate 𝛽. (2) UnGSL with positive 𝛽 consistently outperform                                            GAT [38], and JKNet [46]. The results are presented in Table 6.
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                              Shen Han, et al.


Table 6: Generalizability of UnGSL with different back-                    substantial computational overhead and optimization challenge.
bones on Cora dataset. The value in bold signifies the top-                Mainstream GSL methods [6, 22, 41, 47] learn edge weights based
performing result.                                                         on node-pair embedding similarities, employing various metrics
      Methods           SGC          APPNP           GAT        JKNet      such as cosine similarity [22, 41], inner product [47] or neural net-
     GRCN           84.40±0.00     84.00±0.15    81.45±1.04   83.73±1.33   works [28]. These methods aim to increase graph homophily and
  GRCN+UnGSL        84.80±0.10     84.40±0.61    82.45±1.04   84.53±1.38   typically achieve state-of-the-art performance, as these metrics can
     IDGL           78.00±0.00     82.13±0.32    79.87±1.01   76.23±1.19
                                                                           capture nodes with similar semantics. However, these embedding-
  IDGL+UnGSL        78.20±0.12     82.50±0.61    80.23±0.49   77.10±0.50   based GSL methods suffer from two limitations: First, they construct
                                                                           edges solely based on embedding similarities while neglecting node
Table 7: Convergence time and GPU memory consumption                       uncertainty, which may introduce inferior information that poison
of different models on Ogbn-arxiv dataset.                                 the target node’s embedding. Although some works [12, 28] incorpo-
                Model      Time(s) Memory(MB)                              rate predictive uncertainty in structure learning, they only use it for
                 IDGL        105       14,566                              cross-view structure fusion without distinguishing nodes of varying
             IDGL+UnGSL      113       15,338                              uncertainty levels within a graph. Second, embedding-based meth-
                                                                           ods impose bidirectional edges between nodes, disregarding their
               SUBLIME       807       14,586
                                                                           unequal influence due to varying levels of uncertainty. Recently,
           SUBLIME+UnGSL 886           14,678
                                                                           several methods propose constructing directed edges from labeled
                                                                           to unlabeled nodes to facilitate the propagation of supervision sig-
We observe that the graphs produced by GSL+UnGSL improve the               nals [30, 34, 35]. However, these methods fail to learn reasonable
prediction of various GNN models compared to those generated by            asymmetric connections between the vast majority of unlabeled
vanilla GSL. Overall, UnGSL demonstrates its generalizability in           nodes. Different from the above works, we propose an uncertainty-
enhancing performance across different GNN architectures. Results          aware structure learning (UnGSL) strategy that learns node-wise
for additional datasets are provided in Appendix D.5.                      thresholds to differentiate low-uncertainty from high-uncertainty
                                                                           neighbors and adaptively refine the corresponding edges.
4.6     Efficiency Analysis (RQ5)
We analyze the time and memory efficiency of UnGSL on the Ogbn-            5.3    Uncertainty in GNNs
Arxiv dataset, a large-scale graph with 169,343 nodes and 1,157,799
                                                                           GNNs inevitably present uncertainty towards their predictions,
edges. To assess time efficiency, we evaluate the algorithms by mea-
                                                                           leading to unstable and erroneous prediction results [40]. In re-
suring the time taken to converge, i.e., to reach optimal performance
                                                                           cent years, uncertainty in GNNs has been widely researched to
on the validation set. As shown in Table 7, UnGSL slightly increases
                                                                           adapt various tasks, including out-of-distribution (OOD) detection
convergence time (8.71% on average) and training memory (2.97%
                                                                           [36, 48, 50], trustworthy GNN learning [19, 43], and GNN model-
on average) compared to the base GSL models. The results demon-
                                                                           ing [37]. However, these uncertainty-based GNNs assume that the
strate UnGSL’s efficiency when applied to GSL models.
                                                                           input graph structure is clean and primarily focus on incorporating
5 Related Work                                                             uncertainty into the model architecture. In contrast, our proposed
                                                                           uncertainty-aware graph structure learning aims to refine the edge
5.1 Graph Neural Networks                                                  based on node uncertainty, assisting in improving graph quality.
Graph neural networks (GNNs) are powerful models for learning
node representations from graph data. Existing GNNs can be cate-           6     Conclusion
gorized as spectral GNNs and spatial GNNs. Spectral GNNs leverage
                                                                           In this paper, we first conduct theoretical and empirical analyses to
eigenvectors and eigenvalues within the graph Laplacian matrix
                                                                           demonstrate the detrimental impact of neighbors with high uncer-
to design graph signal filters in the spectral domain [2, 8, 18, 25].
                                                                           tainty on GNN learning. Building on this, we propose the UnGSL
Spatial GNNs [10, 17, 24, 38, 44, 45] simplified spectral graph filter
                                                                           strategy, a lightweight plug-in module that integrates seamlessly
[8] using first-order approximation, which aggregates features from
                                                                           with state-of-the-art GSL models and boosts performance with mini-
neighboring nodes in the spatial graph to generate node embed-
                                                                           mal extra computational overhead. UnGSL learns node-wise thresh-
dings. However, existing GNNs assume that the input graph struc-
                                                                           olds to differentiate between low-uncertainty and high-uncertainty
ture is sufficiently clean for learning, whereas real-world graphs
                                                                           neighbors, and adaptively refines the graph based on each node’s
are often noisy and incomplete, which limits GNN performance on
                                                                           uncertainty level. Experiments demonstrate that UnGSL consis-
downstream tasks [4, 16]. In this paper, we propose uncertainty-
                                                                           tently enhances the performance and robustness of GSL models. In
aware graph structure learning, which effectively denoises the
                                                                           the future, we plan to explore more effective uncertainty metrics
graph structure to alleviate the above limitation.
                                                                           to accurately identify uncertain nodes in graph structure learning.
5.2     Graph Structure Learning
Graph Structure Learning (GSL) aims to learn an optimal graph that         Acknowledgments
improves the accuracy and robustness of Graph Neural Networks              This work is supported by the National Natural Science Founda-
(GNNs) on downstream tasks. Early GSL methods [14, 23] directly            tion of China (62476244,62372399,62476245), Zhejiang Provincial
treat the target adjacency matrix as learnable parameters, incurring       Natural Science Foundation of China (Grant No: LTGG23F030005).
Uncertainty-Aware Graph Structure Learning                                                                          WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


References                                                                                     hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html
 [1] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mo-         [21] Xiao Huang, Jundong Li, and Xia Hu. 2017. Label informed attributed network
     hammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra                embedding. In Proceedings of the tenth ACM International Conference on Web
     Acharya, et al. 2021. A review of uncertainty quantification in deep learning:            Search and Data Mining. Association for Computing Machinery, New York, NY,
     Techniques, applications and challenges. Information Fusion 76 (2021), 243–297.           USA, 731–739.
 [2] Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. 2013. Spectral           [22] Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, and Chanyoung Park.
     networks and locally connected networks on graphs. In 2nd International Confer-           2024. Self-Guided Robust Graph Structure Refinement. In Proceedings of the ACM
     ence on Learning Representations, ICLR 2014. Curran Associates Inc., Red Hook;            on Web Conference 2024. Association for Computing Machinery, New York, NY,
     NY; United States.                                                                        USA, 697–708.
 [3] Hao Chen, Yuanchen Bei, Qijie Shen, Yue Xu, Sheng Zhou, Wenbing Huang,               [23] Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, and Jiliang Tang.
     Feiran Huang, Senzhang Wang, and Xiao Huang. 2024. Macro graph neural                     2020. Graph structure learning for robust graph neural networks. In Proceedings
     networks for online billion-scale recommender systems. In Proceedings of the              of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data
     ACM on Web Conference 2024. Association for Computing Machinery, New York,                mining. Association for Computing Machinery, New York, NY, USA, 66–74.
     NY, USA, 3598–3608.                                                                  [24] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with
 [4] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan                   Graph Convolutional Networks. In International Conference on Learning Rep-
     He. 2023. Bias and debias in recommender system: A survey and future directions.          resentations. Curran Associates, Inc., Red Hook; NY; United States. https:
     ACM Transactions on Information Systems 41, 3 (2023), 1–39.                               //openreview.net/forum?id=SJU4ayYgl
 [5] Sirui Chen, Jiawei Chen, Sheng Zhou, Bohao Wang, Shen Han, Chanfei Su, Yuqing        [25] Qimai Li, Xiaotong Zhang, Han Liu, Quanyu Dai, and Xiao-Ming Wu. 2021.
     Yuan, and Can Wang. 2024. SIGformer: Sign-aware Graph Transformer for                     Dimensionwise separable 2-D graph convolution for unsupervised and semi-
     Recommendation. In Proceedings of the 47th International ACM SIGIR Conference             supervised learning on graphs. In Proceedings of the 27th ACM SIGKDD Conference
     on Research and Development in Information Retrieval. Association for Computing           on Knowledge Discovery & Data Mining. Association for Computing Machinery,
     Machinery, New York, NY, USA, 1274–1284.                                                  New York, NY, USA, 953–963.
 [6] Yu Chen, Lingfei Wu, and Mohammed Zaki. 2020. Iterative deep graph learning          [26] Zhixun Li, Liang Wang, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao
     for graph neural networks: Better and robust node embeddings. Advances in                 Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, and Jeffrey Xu Yu. 2023.
     Neural Information Processing Systems 33 (2020), 19314–19326.                             GSLB: The Graph Structure Learning Benchmark. In Thirty-seventh Conference
 [7] Imre Csiszár, Paul C Shields, et al. 2004. Information theory and statistics: A           on Neural Information Processing Systems Datasets and Benchmarks Track. Curran
     tutorial. Foundations and Trends® in Communications and Information Theory 1,             Associates Inc., Red Hook; NY; United States. https://openreview.net/forum?id=
     4 (2004), 417–528.                                                                        xT3i5GS3zU
 [8] Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. 2016. Convolu-         [27] Juncheng Liu, Bryan Hooi, Kenji Kawaguchi, Yiwei Wang, Chaosheng Dong, and
     tional neural networks on graphs with fast localized spectral filtering. Advances         Xiaokui Xiao. 2024. Scalable and Effective Implicit Graph Neural Networks on
     in Neural Information Processing Systems 29 (2016).                                       Large Graphs. In The Twelfth International Conference on Learning Representations.
 [9] Chenhui Deng, Zichao Yue, and Zhiru Zhang. 2024. Polynormer: Polynomial-                  Curran Associates, Inc., Red Hook; NY; United States. https://openreview.net/
     Expressive Graph Transformer in Linear Time. In The Twelfth International Con-            forum?id=QcMdPYBwTu
     ference on Learning Representations. Curran Associates Inc., Red Hook; NY; United    [28] Nian Liu, Xiao Wang, Lingfei Wu, Yu Chen, Xiaojie Guo, and Chuan Shi. 2022.
     States. https://openreview.net/forum?id=hmv1LpNfXa                                        Compact graph structure learning via mutual information compression. In Pro-
[10] Mucong Ding, Tahseen Rabbani, Bang An, Evan Wang, and Furong Huang. 2022.                 ceedings of the ACM Web Conference 2022. Association for Computing Machinery,
     Sketch-GNN: Scalable graph neural networks with sublinear training complexity.            New York, NY, USA, 1601–1610.
     Advances in Neural Information Processing Systems 35 (2022), 2930–2943.              [29] Yixin Liu, Yu Zheng, Daokun Zhang, Hongxu Chen, Hao Peng, and Shirui Pan.
[11] Hande Dong, Jiawei Chen, Fuli Feng, Xiangnan He, Shuxian Bi, Zhaolin Ding,                2022. Towards unsupervised deep graph structure learning. In Proceedings of the
     and Peng Cui. 2021. On the equivalence of decoupled graph convolution network             ACM Web Conference 2022. Association for Computing Machinery, New York,
     and label propagation. In Proceedings of the Web Conference 2021. Association for         NY, USA, 1392–1403.
     Computing Machinery, New York,NY,United States, 3651–3662.                           [30] Jianglin Lu, Yi Xu, Huan Wang, Yue Bai, and Yun Fu. 2024. Latent graph inference
[12] Liang Duan, Xiang Chen, Wenjie Liu, Daliang Liu, Kun Yue, and Angsheng Li.                with limited supervision. Advances in Neural Information Processing Systems 36
     2024. Structural Entropy Based Graph Structure Learning for Node Classification.          (2024).
     In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. AAAI      [31] Yao Ma, Xiaorui Liu, Neil Shah, and Jiliang Tang. 2022. Is Homophily a Ne-
     Press, Washington DC,USA, 8372–8379.                                                      cessity for Graph Neural Networks?. In International Conference on Learning
[13] Bahare Fatemi, Layla El Asri, and Seyed Mehran Kazemi. 2021. Slaps: Self-                 Representations. Curran Associates, Inc., Austria. https://openreview.net/forum?
     supervision improves structure learning for graph neural networks. Advances in            id=ucASPPD9GKN
     Neural Information Processing Systems 34 (2021), 22667–22681.                        [32] Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila
[14] Luca Franceschi, Mathias Niepert, Massimiliano Pontil, and Xiao He. 2019. Learn-          Prokhorenkova. 2023. A critical look at the evaluation of GNNs under heterophily:
     ing discrete structures for graph neural networks. In International Conference on         Are we really making progress?. In The Eleventh International Conference on
     Machine Learning. PMLR, Curran Associates Inc., Long Beach, California, USA,              Learning Representations. Curran Associates, Inc., Kigali, Rwanda.
     1972–1982.                                                                           [33] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and
[15] Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. 2019. Com-              Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29,
     bining Neural Networks with Personalized PageRank for Classification on Graphs.           3 (2008), 93–93.
     In International Conference on Learning Representations. Curran Associates Inc.,     [34] Zixing Song, Yifei Zhang, and Irwin King. 2022. Towards an optimal asymmetric
     Red Hook; NY; United States. https://openreview.net/forum?id=H1gL-2A9Ym                   graph structure for robust semi-supervised node classification. In Proceedings
[16] Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bo-                 of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Dining.
     jchevski, and Stephan Günnemann. 2021. Robustness of graph neural networks                Association for Computing Machinery, New York, NY, USA, 1656–1665.
     at scale. Advances in Neural Information Processing Systems 34 (2021), 7637–7649.    [35] Zixing Song, Yifei Zhang, and Irwin King. 2024. Optimal block-wise asymmetric
[17] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E          graph construction for graph-based semi-supervised learning. Advances in Neural
     Dahl. 2017. Neural message passing for quantum chemistry. In International                Information Processing Systems 36 (2024).
     Conference on Machine Learning. PMLR, JMLR.org, Sydney NSW Australia, 1263–          [36] Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner, and
     1272.                                                                                     Stephan Günnemann. 2021. Graph posterior network: Bayesian predictive uncer-
[18] Mingguo He, Zhewei Wei, and Ji-Rong Wen. 2022. Convolutional neural net-                  tainty for node classification. Advances in Neural Information Processing Systems
     works on graphs with chebyshev approximation, revisited. Advances in Neural               34 (2021), 18033–18048.
     Information Processing Systems 35 (2022), 7264–7276.                                 [37] Daeho Um, Jiwoong Park, Seulki Park, and Jin young Choi. 2023. Confidence-
[19] Hans Hao-Hsun Hsu, Yuesong Shen, Christian Tomani, and Daniel Cremers.                    Based Feature Imputation for Graphs with Partially Known Features. In The
     2022. What makes graph neural networks miscalibrated? Advances in Neural                  Eleventh International Conference on Learning Representations. Curran Associates,
     Information Processing Systems 35 (2022), 13775–13786.                                    Inc., Red Hook; NY; United States. https://openreview.net/forum?id=YPKBIILy-
[20] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen                   Kt
     Liu, Michele Catasta, and Jure Leskovec. 2020. Open Graph Benchmark: Datasets        [38] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
     for Machine Learning on Graphs. In Advances in Neural Information Processing              Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In International Confer-
     Systems 33: Annual Conference on Neural Information Processing Systems 2020,              ence on Learning Representations. Curran Associates, Inc., Red Hook; NY; United
     NeurIPS 2020, December 6-12, 2020, virtual, Hugo Larochelle, Marc’Aurelio Ranzato,        States. https://openreview.net/forum?id=rJXMpikCZ
     Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (Eds.). Curran Associates     [39] Bohao Wang, Jiawei Chen, Changdong Li, Sheng Zhou, Qihao Shi, Yang Gao, Yan
     Inc., Red Hook; NY; United States. https://proceedings.neurips.cc/paper/2020/             Feng, Chun Chen, and Can Wang. 2024. Distributionally Robust Graph-based
                                                                                               Recommendation System. In Proceedings of the ACM on Web Conference 2024.
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                                                          Shen Han, et al.


     Association for Computing Machinery, New York, NY, USA, 3777–3788.                     [46] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi
[40] Fangxin Wang, Yuqing Liu, Kay Liu, Yibo Wang, Sourav Medya, and Philip S. Yu.               Kawarabayashi, and Stefanie Jegelka. 2018. Representation learning on graphs
     2024. Uncertainty in Graph Neural Networks: A Survey. Transactions on Machine               with jumping knowledge networks. In International Conference on Machine Learn-
     Learning Research (2024). https://openreview.net/forum?id=0e1Kn76HM1                        ing. PMLR, Curran Associates, Inc., Stockholm, Sweden, 5453–5462.
[41] Huizhao Wang, Yao Fu, Tao Yu, Linghui Hu, Weihao Jiang, and Shiliang Pu. 2023.         [47] Donghan Yu, Ruohong Zhang, Zhengbao Jiang, Yuexin Wu, and Yiming Yang.
     Prose: Graph structure learning via progressive strategy. In Proceedings of the 29th        2021. Graph-revised convolutional network. In Machine Learning and Knowledge
     ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Association                   Discovery in Databases: European Conference, ECML PKDD 2020, Ghent, Belgium,
     for Computing Machinery, New York, NY, USA, 2337–2348.                                      September 14–18, 2020, Proceedings, Part III. Springer, Springer-Verlag, Berlin,
[42] Ruijia Wang, Shuai Mou, Xiao Wang, Wanpeng Xiao, Qi Ju, Chuan Shi, and Xing                 Heidelberg, 378–393.
     Xie. 2021. Graph structure estimation neural networks. In Proceedings of the Web       [48] Linlin Yu, Yifei Lou, and Feng Chen. 2023. Uncertainty-aware Graph-based
     Conference 2021. Association for Computing Machinery, New York, NY, USA,                    Hyperspectral Image Classification. In The Twelfth International Conference on
     342–353.                                                                                    Learning Representations. OpenReview.net, Vienna, Austria.
[43] Xiao Wang, Hongrui Liu, Chuan Shi, and Cheng Yang. 2021. Be confident!                 [49] Zhen Zhang, Jiajun Bu, Martin Ester, Jianfeng Zhang, Zhao Li, Chengwei Yao,
     towards trustworthy graph neural networks via confidence calibration. Advances              Huifen Dai, Zhi Yu, and Can Wang. 2021. Hierarchical multi-view graph pooling
     in Neural Information Processing Systems 34 (2021), 23768–23779.                            with structure learning. IEEE Transactions on Knowledge and Data Engineering
[44] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian                 35, 1 (2021), 545–559.
     Weinberger. 2019. Simplifying graph convolutional networks. In International           [50] Xujiang Zhao, Feng Chen, Shu Hu, and Jin-Hee Cho. 2020. Uncertainty aware
     Conference on Machine Learning. PMLR, Curran Associates, Inc., Long Beach,                  semi-supervised learning on graph data. Advances in Neural Information Process-
     California, USA, 6861–6871.                                                                 ing Systems 33 (2020), 12827–12836.
[45] Junkang Wu, Wentao Shi, Xuezhi Cao, Jiawei Chen, Wenqiang Lei, Fuzheng                 [51] Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu
     Zhang, Wei Wu, and Xiangnan He. 2021. DisenKGAT: knowledge graph embed-                     Tan, Daochen Zha, Yan Feng, Chun Chen, and Can Wang. 2023. OpenGSL:
     ding with disentangled graph attention network. In Proceedings of the 30th ACM              A Comprehensive Benchmark for Graph Structure Learning. In Thirty-seventh
     International Conference on Information & Knowledge Management. Association                 Conference on Neural Information Processing Systems Datasets and Benchmarks
     for Computing Machinery, New York,NY,United States, 2140–2149.                              Track. Curran Associates Inc., Red Hook, NY, USA. https://openreview.net/
                                                                                                 forum?id=yXLyhKvK4D
Uncertainty-Aware Graph Structure Learning                                                                                                   WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


A     Proof of Proposition 1                                                                                 • ProGNN [23] treats the graph as a learnable adjacency matrix and
To prove Proposition 1, we first introduce the log-sum inequality                                              optimizes the sparsity, low-rankness, and feature smoothness of
[7] below.                                                                                                     the graph structure.
                                                                                                             • PROSE [41] identifies influential nodes using PageRank scores
   Lemma 1 (Log-sum Ineqality). Let 𝑎 1, ..., 𝑎𝑛 and 𝑏 1, ..., 𝑏𝑛 be
                                                                                                               and reconstructs the graph structure by connecting these influ-
non-negative numbers. Denote the sum of all 𝑎𝑖 s by a and the sum of
                                                                                                               ential nodes.
all 𝑏𝑖 s by b. Then log-sum inequality states that
                                                                                                             • IDGL [6] iteratively learns the graph structure and node embed-
                           𝑛
                         ∑︁         𝑎𝑖          𝑎                                                              dings, and introduces a node-anchor message-passing paradigm
                              𝑎𝑖 log ≥ 𝑎log ,                        (14)
                                    𝑏 𝑖        𝑏                                                               to scale IDGL to large graphs.
                          𝑖=1
                                                                                                             • SLAPS [13] proposes learning a denoising autoencoder to filter
   Proof. ∀𝑣𝑖 ∈ V, by define the logit 𝑜𝑖 = ( 𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 𝑥 𝑗 )W
                                                     Í
                                                                                                               noisy edges in the graph structure.
and logit 𝑜𝑖′ = 𝑥𝑖 W, where 𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 = 1, we have:
                               Í
                                                                                                             • CUR [30] is a model-agnostic structure learning module, which
                                   ∑︁                                                                          proposes constructing unidirectional edges from unlabeled nodes
                          𝑜𝑖 =          Â𝑖 𝑗 𝑜 ′𝑗 ,                 (15)                                      to labeled nodes via CUR decomposition to facilitate the propa-
                                        𝑣 𝑗 ∈ N (𝑣𝑖 )
                                                                                                               gation of supervision signals to unlabeled nodes.
then for predictive probability vector 𝑝𝑖 of 𝑜𝑖 , ∀𝑣 𝑗 ∈ N (𝑣𝑖 ), let 𝑝 ′𝑗                                   • SUBLIME [29] is an unsupervised GSL model that employs con-
denote the classification probabilities of its initial features. We have:                                      trastive learning between the learned graph and an augmented
                        𝑜𝑖 + 1𝐾                                                                                graph to enhance the robustness of the graph structure.
              𝑝𝑖 = Í𝐾
                             (𝑜 + 1)
                      𝑘=1 𝑖𝑘                                                                                 D Additional Experimental Results
                       Í                         ′
                          𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 (𝑜 𝑗 + 1𝐾 )
                 = Í𝐾 Í                                                                                      D.1 Visualization of Node Entropy and Average
                      𝑘=1
                             ( 𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 𝑜 ′𝑗𝑘 + 1)                                                    Neighbor Entropy
                    Í                   ′      Í𝐾          ′
                      𝑣 𝑗 ∈ N (𝑣𝑖 ) 𝑝 𝑗 ( Â𝑖 𝑗 𝑘=1 (𝑜 𝑗𝑘 + 1))                                              We also conduct experiments on more datasets with different ho-
                 = Í                                            ,                                            mophily characteristics. Figure 5 presents the node entropy after
                                              Í𝐾         ′
                          𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 𝑘=1 (𝑜 𝑗𝑘 + 1)                                                 GNN aggregation, along with the average entropy of its neighbors,
                       ∑︁
                 =               𝜂 𝑗 𝑝 ′𝑗 ,                          (16)                                    on the Citeseer and Flickr datasets. We observe a strong positive cor-
                       𝑣 𝑗 ∈ N (𝑣𝑖 )
                                                                                                             relation between the entropy of nodes and that of their neighbors
        Í                                                                                                    across different datasets.
where, 𝑣 𝑗 ∈ N (𝑣𝑖 ) 𝜂 𝑗 = 1. Now we focus the 𝑙-th element in the
probability vector:
                              ∑︁                                                                                                   1.5
                      𝑝𝑖𝑙 =        𝜂 𝑗 𝑝 ′𝑗𝑙

                                                                                                               Average Entropy
                                 𝑣 𝑗 ∈ N (𝑣𝑖 )
                                         ∑︁                                  ∑︁                                                    1.0

                                                                                                                of Its Neighbors
            =⇒ 𝑝𝑖𝑙 log𝑝𝑖𝑙 = (                      𝜂 𝑗 𝑝 ′𝑗𝑙 )log(                      𝜂 𝑗 𝑝 ′𝑗𝑙 )
                                  𝑣 𝑗 ∈ N (𝑣𝑖 )                         𝑣 𝑗 ∈ N (𝑣𝑖 )
                                                                                                 ′                                 0.5
                                                                         Í
                                         ∑︁                                  𝑣 𝑗 ∈ N (𝑣𝑖 ) 𝜂 𝑗 𝑝 𝑗𝑙
                             =(                    𝜂 𝑗 𝑝 ′𝑗𝑙 )log(                                    )
                                                                                                                                             0.5         1.0        1.5
                                                                           Í
                                  𝑣 𝑗 ∈ N (𝑣𝑖 )                                𝑣 𝑗 ∈ N (𝑣𝑖 ) 𝜂 𝑗
                                       ∑︁                                                                                                 Node Entropy After GNN Aggregation
                             ≤                   𝜂 𝑗 𝑝 ′𝑗𝑙 log𝑝 ′𝑗𝑙                                                                                (a) Citeseer dataset
                                 𝑣 𝑗 ∈ N (𝑣𝑖 )
             𝐾                                               𝐾                                                                     1.75

                                                                                                               Average Entropy
            ∑︁                             ∑︁               ∑︁
    =⇒ −          𝑝𝑖𝑙 log𝑝𝑖𝑙 ≥ −                                    𝑝 ′𝑗𝑙 log𝑝 ′𝑗𝑙
                                                                                                                                   1.50
                                                       𝜂𝑗
            𝑙=1                        𝑣 𝑗 ∈ N (𝑣𝑖 )          𝑙=1
                                                                                                                                   1.25
                                                                                                                of Its Neighbors
                                       ∑︁
                             =                   𝜂 𝑗 𝑢 ′𝑗 ,                                           (17)
                                 𝑣 𝑗 ∈ N (𝑣𝑖 )                                                                                     1.00
which completes the proof.                                                                                                         0.75
                                                                                                                                          1.0     1.5      2.0    2.5     3.0
B    Datasets                                                                                                                             Node Entropy After GNN Aggregation
                                                                                                                                                    (b) Flickr dataset
Table 8 shows the statistics of 7 datasets.

C     Baselines                                                                                              Figure 5: Visualization of node entropy after GNN aggrega-
we introduce all GSL models used in the experiment in this section.                                          tion (i.e., 𝑢𝑖 ) alongside the average entropy of its neighbors
                                                                                                             (i.e., 𝑣 𝑗 ∈ N (𝑣𝑖 ) Â𝑖 𝑗 𝑢 ′𝑗 ) on Citeseer and Flickr datasets.
                                                                                                                   Í
• GRCN [47] uses a GCN to extract topological features and com-
  pute the similarity between nodes as the edge weights of the
  learned graph.
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                                                                                  Shen Han, et al.


   Table 8: Detailed statistics of node classification datasets.                                                D.2              Hyperparameter Settings
  Dataset        #Nodes #Edges #Feat. #Avg.degree #Homophily
                                                                                                                Table 9 presents the hyperparameter settings of learning rate and
  Cora             2,708    5,278 1,433         3.9         0.81                                                𝛽 in UnGSL.
  Citeseer         3,327    4,552 3,703         2.7         0.74
  Pubmed          19,717 44,324      500        4.5         0.80                                                D.3              Additional Results on the Hyperparameter
  BlogCatalog      5,196 171,743 8,189         66.1         0.40                                                                 𝛽
  Roman-Empire 22,662 32,927         300        2.9         0.05                                               We assign different values to 𝛽 from the interval [0,1] and evaluate
  Flickr           7,575 239,738 12,047        63.3         0.24                                               the corresponding performance on the IDGL model [6]. The results
  Ogbn-arxiv     169,343 1,157,799 767         13.67        0.65                                               are presented in Figure 6.

Table 9: Hyperparameter settings of learning rate, 𝛽 and ini-
tial 86.0        Cora
     value in UnGSL.                           Citeseer                                                                                   Cora                                       Citeseer
                            Setting
                                   74.0 GRCN     PROGNN        PROSE      IDGL       SLAPS    SUBLIME
                                                                                                                                                                         74
              85.5                                                   73.50.001

Accuracy(%)                                                Accuracy(%)                                        Accuracy(%)                                  Accuracy(%)
                                                                                                                            84                                           72
                               lr       0.0005    0.0001          0.01          0.0001         0.001
                Cora           𝛽         0.40      0.82           0.01    0.95   0.75          0.98
              85.0             lr        0.006     0.03              73.00.0005 0.03
                                                                 0.001                          0.03

              84.5
               Citeseer        𝛽         0.56      0.57          0.75
                                                                     72.50.001 0.02
                                                                          0.78   0.19           0.89
                                                                                                                            82                                           70
                               lr        0.09       -            0.01                          0.004

              84.0
               Pubmed          𝛽         0.35       -            0.22
                                                                     72.00.16    0.65          0.46
                                                                                                                                                                         68
                      0.0 0.2lr𝛽 0.4 0.0056
              BlogCatalog             0.6   0.80.0002
                                                  1.0             0.06    0.01
                                                                                0.0 0.2
                                                                                0.007
                                                                                          0.4 0.0001
                                                                                                0.6 0.8 1.0                      0.0 0.2 0.4 0.6 0.8 1.0                      0.0 0.2 0.4 0.6 0.8 1.0
                                      value                                                  value                                        value                                        value
                                      0.65      0.87              0.36        0.90   0.18      0.47
                               lr       0.0008      -            0.001    0.002       0.02     0.007
          Roman-Empire         𝛽         0.001      -            0.001    0.82        0.83      0.54
                               lr        0.008    0.0001          0.03    0.002      0.036     0.0001           Figure 6: Comparsion of different 𝛽 on Cora and Citeseer
                 Flickr        𝛽         0.08      0.96           0.71    0.70       0.23       0.82
                               lr          -        -            0.001    0.003        -       0.024
                                                                                                                datasets.
              Ogbn-arxiv       𝛽           -        -             0.2      0.59        -        0.48

                                                                                                                D.4 Additional Results for Robustness Analysis
Table 10: Robustness analysis with random noise injection on
Cora dataset. We evaluate the performance of UnGSL under                                                       We evaluate the robustness of UnGSL against label noise and fea-
feature noise and label noise.                                                                                 ture noise on the Cora dataset. The results are shown in Table 10.
                                                                                                               For feature noise, we randomly mask a proportion of node fea-
                                 Feature Noise Level                            Label Noise Level
                                                                                                               tures by replacing their values with zeros. This approach allows
        Methods             0%      20% 40% 60% 80%                      0%      20% 40% 60% 80%               us to investigate the performance of UnGSL when node features
    GRCN     84.70 82.47 81.60 79.40 72.40 84.70 81.33 73.32 65.21 58.13                                       are subjected to varying degrees of damage. For label noise, we
 GRCN+UnGSL 85.84 83.70 82.80 80.62 73.70 85.84 83.43 74.52 67.94 62.17                                        randomly alter node labels. The ratio of modifications varies from
  Improve(%) 1.34% 1.49% 1.47% 1.53% 1.80% 1.34% 2.58% 1.63% 4.18% 6.95%                                       0 to 0.8 to simulate different levels of noise. We can observe that:
     IDGL     84.50 82.10 79.25 75.23 65.16 84.50 77.60 70.80 54.81 43.65                                      1) UnGSL consistently enhances the performance of GSL models
  IDGL+UnGSL 84.90 83.29 80.52 77.43 66.50 84.90 79.35 72.03 56.13 44.51                                       across different noise levels. 2) With increasing levels of noise,
   Improve(%) 0.47% 1.45% 1.60% 2.92% 2.04% 0.47% 2.26% 1.74% 2.41% 1.97%                                      UnGSL achieves greater relative improvements. For example, with
                                                                                                               80% wrong labels, GRCN+UnGSL achieves a 6.95% improvement,
                                                                                                               which is significantly higher than the improvement under 0% wrong
Table 11: Generalizability of UnGSL with different back-
                                                                                                               labels scenario (1.34%). In summary, the results suggest that UnGSL
bones on Citeseer dataset. The value in bold signifies the
                                                                                                               enhances robustness to label noise and feature noise.
top-performing result.
              Methods                 SGC           APPNP                      GAT             JKNet
                                                                                                                D.5              Additional Results on Generalizability of
        GRCN                   72.00±0.00         72.73±0.96             70.77±1.06          71.83±0.72                          UnGSL
     GRCN+UnGSL                72.37±0.06         73.50±0.30             70.88±1.70          72.32±1.75
                                                                                                                Table 11 illustrates the generalizability of UnGSL using various
         IDGL                  71.90±0.00         72.20±0.75             63.83±0.97          71.00±1.15
                                                                                                                backbones on the Citeseer dataset.
      IDGL+UnGSL               73.40±0.00         73.37±0.61             67.00±0.69          72.33±0.9

