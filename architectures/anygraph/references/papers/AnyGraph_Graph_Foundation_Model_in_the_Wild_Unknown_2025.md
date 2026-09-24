# AnyGraph Graph Foundation Model in the Wild Unknown 2025

> Source: `AnyGraph_Graph_Foundation_Model_in_the_Wild_Unknown_2025.pdf`

---

                                                            AnyGraph: Graph Foundation Model in the Wild
                                                                                            Lianghao Xia and Chao Huang∗
                                                                                              The University of Hong Kong
                                                                                               {lhaoxia, chuang7}@hku.hk
                                                                                      Github: https://github.com/HKUDS/AnyGraph

                                         Abstract                                                                  fast adaptation capabilities can unlock transformative opportunities
                                         The growing ubiquity of relational data structured as graphs has un-      for leveraging the rich insights encoded within graph data.
                                         derscored the need for graph learning models with exceptional gen-            The field of graph learning has seen significant advancements in
                                         eralization capabilities. However, current approaches often struggle      recent years, largely driven by the power of Graph Neural Networks




arXiv:2408.10700v1 [cs.LG] 20 Aug 2024
                                         to effectively extract generalizable insights, frequently requiring ex-   (GNNs) [15, 18, 31]. However, the current state-of-the-art models
                                         tensive fine-tuning and limiting their versatility. Graph foundation      often fall short when it comes to truly generalizable performance.
                                         models offer a transformative solution, with the potential to learn       Existing approaches tend to be heavily reliant on arduous fine-
                                         robust, generalizable representations from graph data. This enables       tuning processes, making them ill-equipped to handle the diverse
                                         more effective and adaptable applications across a wide spectrum          array of graph structures and distributions encountered in real-
                                         of tasks and domains. In this work, we investigate a unified graph        world applications. This inability to adapt swiftly and seamlessly
                                         model, AnyGraph, designed to handle key challenges: i) Structure          to novel graph domains poses a critical barrier to the widespread
                                         Heterogenity. Addressing distribution shift in graph structural           adoption of graph learning technologies. Therefore, addressing this
                                         information; ii) Feature Heterogenity. Handling diverse feature           challenge is of paramount importance if we are to fully harness the
                                         representation spaces across graph datasets; iii) Fast Adaptation.        transformative potential of graph-based insights.
                                         Efficiently adapting the model to new graph domains; iv) Scaling              Inspired by the principles that have driven the development of
                                         Law Emergence. Enabling the model to exhibit scaling law be-              successful foundation models in understanding vision and language
                                         havior, where its performance scales favorably with the amount            data [26, 27], the concept of a versatile graph foundation model
                                         of data and parameter sizes. To tackle these critical challenges, we      holds immense potential to unlock new frontiers in graph learning.
                                         build the AnyGraph upon a Graph Mixture-of-Experts (MoE) archi-           By learning rich, transferable representations from diverse graph-
                                         tecture. This approach empowers the model to effectively manage           structured data, such a model can be efficiently adapted to a wide
                                         both the in-domain and cross-domain distribution shift concern-           array of graph domains and tasks. However, building an effective
                                         ing structure-level and feature-level heterogeneity. Furthermore, a       and adaptive graph foundation model is not a trivial endeavor.
                                         lightweight graph expert routing mechanism is proposed to facili-         Several key challenges must be overcome, including:
                                         tate AnyGraph’s fast adaptability to new data and domains. Our            (i) Structure Heterogeneity. The development of versatile graph
                                         extensive experiments on diverse 38 graph datasets have demon-            models faces the challenge of accommodating diverse structural
                                         strated the strong zero-shot learning performance of AnyGraph             properties and data distributions in various graph datasets. For
                                         across diverse graph domains with significant distribution shift.         instance, graphs can exhibit substantial heterogeneity in node de-
                                         Furthermore, we have validated the model’s fast adaptation ability        gree distributions, ranging from homogeneous to highly skewed
                                         and scaling law emergence, showcasing its versatility.                    patterns. Similarly, graph structures can vary greatly in complex-
                                         ACM Reference Format:
                                                                                                                   ity, from simple topologies to intricate, hierarchical arrangements.
                                         Lianghao Xia and Chao Huang∗ . 2024. AnyGraph: Graph Foundation Model     These structural variations can significantly impact the perfor-
                                         in the Wild. In Proceedings of (ACM KDD). ACM, New York, NY, USA,         mance and generalization of graph learning algorithms. Effectively
                                         11 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn                         addressing this diversity is critical for developing unified models
                                                                                                                   that can thrive across a wide range of graph-structured data.
                                         1    Introduction                                                         (ii) Feature Heterogeneity. Graphs exhibit substantial hetero-
                                         The growing ubiquity of relational data in the form of graphs has         geneity in their node and edge features, which can span categorical
                                         underscored the pressing need for advanced graph learning models          attributes, continuous numerical data, and multi-modal content.
                                         that excel at generalization [7, 13]. As real-world applications of       Furthermore, the dimensionality and semantics of these features of-
                                         graph-structured data continue to proliferate across diverse do-          ten vary dramatically across different graph domains. For instance,
                                         mains, including social networks, academic networks, transporta-          a social interaction graph may include textual content and demo-
                                         tion systems, and biological networks, the ability of graph learning      graphic information associated with its nodes, while a molecular
                                         models to effectively handle distribution shifts and adapt to new         graph may feature atomic compositions and bond types. Effectively
                                         graph domains has become increasingly crucial [20, 29, 37, 38]. De-       handling this feature heterogeneity is crucial for building a versatile
                                         veloping models with robust zero-shot learning performance and            graph model capable of generalizing across diverse graph domains.
                                                                                                                   (iii) Fast Adaptation for Broad Applicability. A key capability
                                         ∗ Chao Huang is the Corresponding Author.                                 for effective graph foundation models is the ability to efficiently
                                         ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY
                                                                                                                   adapt to new graph dataset and domains. Rather than requiring
                                         2024. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM                                    extensive retraining or fine-tuning, the ideal model should be able to
                                         https://doi.org/10.1145/nnnnnnn.nnnnnnn
ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY                                                                                  Lianghao Xia and Chao Huang∗


                                                                                 heterogeneity. These fixed-capacity models may encounter in-
                           AnyGraph         0.12                                 terference between different types of graph datasets, and can
                           GNN-N                                                 sometimes overfit to new data, leading to catastrophic forgetting.
                                            0.11
                                       NDCG@20
                           GNN-L                                                 To address these challenges, the proposed Mixture-of-Experts
                                            0.10                                 (MoE) architecture for graph models was designed with a focus
                           GraphGPT         0.09                                 on adaptability. This new paradigm empowers the model to flexi-
                                                                   Zero-shot     bly adjust to the nuances of diverse graph datasets, dynamically
                                            0.08
                           OpenGraph                                             selecting the most appropriate experts to learn distinct patterns.
                                                      4 5 6 7 8
       *Test on 14 datasets                            log # Model Params      • Stronger Gernealiation Capacities of AnyGraph. Through
   (a) Generalizability of AnyGraph.             (b) Scaling law of AnyGraph     our extensive experiments, the proposed AnyGraph model with
                                                                                 the graph MoE framework has demonstrated strong generaliza-
Figure 1: AnyGraph’s generalizability and scaling law reveals                    tion capacities across a wide range of graph tasks and domains.
its exceptional capabilities. Compared to baseline methods,                      The experimental results showcase the AnyGraph’s ability to
the superior performance of AnyGraph can be observed in                          outperform existing graph models in terms of both predictive
its exceptional cross-domain generalization ability.                             performance and robustness to distribution shift.
                                                                               • Fast Adapability of AnyGraph. Our innovative dynamic ex-
quickly adjust its parameters and learning strategies to handle the              pert selection mechanism enhances AnyGraph’s ability to swiftly
structural and distributional characteristics of previously unseen               adapt to new graph domains. By dynamically routing inputs
graph datasets. By seamlessly generalizing and performing well                   through relevant experts, AnyGraph can quickly activate the spe-
across a diverse range of real-world scenarios – from user behavior              cialized networks best suited for the task. This strong adaptation
graphs to transportation networks and biological systems – these                 sets AnyGraph apart from baselines. Evaluation shows its supe-
adaptable models can unlock transformative insights across an ever-              riority through rapid convergence and exceptional performance,
expanding universe of graph-structured data.                                     further justifying its cross-domain versatility.
(iv) Scaling Laws for Transformative Graph Capabilities. A                     • The Scaling Law of AnyGraph. Our experiments reveal that
key characteristic of successful foundation models in domains like               AnyGraph’s performance follows the scaling law, where the
CV [5] and NLP [21] is their ability to exhibit scaling laws - where             model continues to improve as model size and training data
performance systematically improves as the model size or training                increase. Additionally, AnyGraph exhibits emergent abilities,
dataset increases. By harnessing this emergent scaling phenome-                  where its generalization capabilities see sudden significant im-
non, graph foundation models can unlock unprecedented levels                     provements with further scaling. This critical scaling law prop-
of capability and generalization, far surpassing the limitations of              erty has been largely overlooked in prior investigations, but it
fixed-capacity architectures. As the size of graph datasets and model            underscores the immense value that AnyGraph derives from its
complexity grow, these scaling-aware designs can continue deliver-               scaling-driven enhancements to generalization performance.
ing transformative performance gains.
The Presented Work. To tackle the above challenges, our Any-
                                                                               2   Preliminaries
Graph model is built upon a Mixture-of-Experts (MoE) architecture,
which allows for effective handling of both the in-domain and                  Graph-Structured Data. A graph G consists of a set of nodes
cross-domain distribution shift in structure-level and feature-level           V = {𝑣𝑖 } and a set of edges E = {(𝑣𝑖 , 𝑣 𝑗 )}. In many cases, each
heterogeneity. The proposed graph MoE paradigm empowers Any-                   node 𝑣𝑖 is associated with a feature vector f𝑖 ∈ R𝑑0 . To efficiently
Graph to learn a diverse ensemble of graph experts, each tailored              utilize such graph-structured data, the link information is typically
to specific structural characteristics. This enables the model to ef-          recorded using an adjacency matrix A ∈ R | V | × | V | . Each element
fectively manage the distribution shift in graph topologies, ranging           𝑎𝑖,𝑗 of A is either 1 or 0, inddicating whether there is an edge from
from homogeneous to highly skewed degree distributions, as well                node 𝑣𝑖 to 𝑣 𝑗 . Additionally, the feature vectors of the nodes are
as handle graphs with varying levels of complexity. Furthermore,               usually represented by a feature matrix F ∈ R | V | ×𝑑0 , where each
the MoE architecture of AnyGraph facilitates fast adaptation of                row corresponds to a node’s feature vector. The primary goal of
the graph model. Rather than relying on a single, fixed-capacity               learning from such graph-structured data is to generate embeddings
model, the Graph MoE learns an ensemble of specialized expert net-             for the graph elements, typically nodes, that effectively capture both
works, each tailored to capture distinct structural and feature-level          the structural and feature-based information of the graph.
characteristics of graph data. The lightweight graph expert routing
                                                                               Graph Foundation Models (GFMs). The essence of GFMs lies in
mechanism allows AnyGraph to quickly identify and activate the
                                                                               their strong generalization capabilities. Specifically, a graph founda-
most relevant experts for a given input graph, without requiring
                                                                               tion model should be able to handle unseen graph data that exhibits
extensive retraining or fine-tuning across the entire model. The
                                                                               significant discrepancies from its training graph datasets. These
key findings of this work can be summarized as:
                                                                               discrepancies may include differences in feature spaces, as well as
• Methodology Design Motivations of AnyGraph. Current                          variations in node and edge semantics across datasets. Formally,
  large graph models [4, 16, 17] often struggle when faced with                let’s denote the training graphs as S = G𝑠 , where each graph G𝑠
  the substantial heterogeneity found in real-world graph data.                is associated with a label set Y𝑠 . Similarly, the set of test graphs is
  This is especially challenging when it comes to feature-level                denoted as T = G𝑡 , with labels Y𝑡 . With a differentiable training
AnyGraph: Graph Foundation Model in the Wild                                                                       ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY


objective L and an evaluation criterion C to measure the prediction               model regarding the input graph G:
accuracy of downstream tasks, building a graph foundation model
𝑓Θ with trainable parameters Θ can be formalized as follows:                                                  𝑆
                                                                                                           1 ∑︁
                                                                                                    𝜑𝑘 =    ·    𝜎 ( ê𝑐⊤𝑠 ê𝑝𝑠 − ê𝑐⊤𝑠 ê𝑛𝑠 )           (3)
              ∑︁                                     ∑︁                                                    𝑆 𝑠=1
    arg max        C (𝑓Θ (G𝑡 ), Y𝑡 ) , Θ = arg min        L (𝑓Θ (G𝑠 ), Y𝑠 ) (1)
      𝑓 ,L    G𝑡                              Θ      G𝑠
                                                                                  where 𝜎 (·) represents the sigmoid activation function, which con-
The above formulation reveals that the key to building graph foun-                strains the competence score to the range of (0, 1). This prevents
dation models are: i) the model architecture design (𝑓 ), which must              the few outlier cases where the non-activated score difference is
have the capacity to encode diverse feature spaces and structural                 excessively large or small, which could otherwise distort the results.
patterns, and ii) the model training process (L), which must effec-               Training Frequency Regularization. Although being empirically
tively traverse such diverse data to find an optimal solution Θ for               accurate in measuring models’ competence using the aforemen-
the model 𝑓 . In light of this, our AnyGraph employs a mixture-of-                tioned self-supervised task loss, this routing mechanism tends to
experts architecture with an automated expert routing method, to                  result in a winner-takes-all sub-optimal situation. In this scenario,
seamlessly integrate powerful prediction models for highly diverse                a single model, or very few models, is predominantly selected as
graph data. AnyGraph is extensively trained on graphs from vari-                  the most competent expert and is used to handle almost all input
ous applications using multiple featuring methods, with a graph                   graphs. These expert models generally receive more or better train-
augmentation technique to further enhance data diversity.                         ing samples in the early training stages, giving them an advantage
                                                                                  over other experts. Consequently, subsequent training samples are
3     Methodology                                                                 also mostly assigned to them due to their performance advantages,
                                                                                  ultimately causing other experts to remain largely untrained.
The proposed AnyGraph framework aims to address both cross-
                                                                                     This situation contradicts our motivation of using different ex-
domain and in-domain heterogeneity in graph structures and node
                                                                                  pert models to learn different subsets of graph modeling knowledge.
features, while enabling fast adaptation to new data. The overall
                                                                                  To address this, we propose a training frequency regularization ap-
framework of AnyGraph is depicted in Fig. 2.
                                                                                  proach that recalibrates the competence score as follows:

3.1      MoE Architecture of AnyGraph                                                                    
                                                                                                                𝑚                𝜌
                                                                                                                                   
3.1.1 Addressing Cross-domain Graph Heterogeneity. To                                          𝜑𝑘′ = 𝜑𝑘 · (1 − Í 𝑘 ) · 𝜌 + 1.0 −                         (4)
                                                                                                                𝑘 ′ 𝑚𝑘 ′         2
model heterogeneous graph patterns across different domains, Any-
Graph employs a MoE architecture that consists of multiple graph                  where 𝜑 ′𝑘 represents the recalibrated routing score for the 𝑘-th
expert models, each responsible for handling graphs with specific                 expert model 𝑓 Θ𝑘 , based on the number of previously assigned
characteristics. An automated routing algorithm is designed to                    training steps 𝑚𝑘 for 𝑘 = 1, · · · , 𝐾. The notation 𝜌 refers to a hy-
assign input graph data to the most competent expert model for                    perparameter for the recalibration scale. A larger 𝜌 results in a
training and prediction. Specifically, the AnyGraph framework can                 greater adjustment to the competence score 𝜑𝑘 . With this addi-
be denoted as M = (𝑓Θ1 , 𝑓Θ2 , · · · , 𝑓Θ𝐾 ,𝜓 ), where 𝐾 denotes the              tional step, the expert routing mechanism will assign more training
number of experts. For an input graph G, the routing algorithm 𝜓                  instances to the less trained expert models, thereby preventing the
firstly identifies the most competent expert model, and the corre-                aforementioned winner-takes-all situation.
sponding model is then used for predicting the graph data, as:
                                                                                  3.1.3 Fast Adaptation Capabilities of AnyGraph. With the
                   𝑦ˆ𝑖,𝑗 = ê𝑖⊤ ê 𝑗 , Ê = 𝑓Θ𝑘 (G), 𝑘 = 𝜓 (G)              (2)   aforementioned MoE architecture and routing mechanism, the train-
                                                                                  ing and inference process of AnyGraph is conducted by only one ex-
where each expert model 𝑓Θ𝑘 can be viewed as a projection from                    pert model. This approach consumes only 1/𝐾 of the computational
the graph space to a node embedding space with uniquely trained                   and memory resources required for predictions and optimization,
parameters Θ𝑘 . And 𝑦ˆ𝑖,𝑗 represents the dot-product-based predic-                compared to other non-MoE graph foundation models based on
tion of whether the entity 𝑣𝑖 should be related to the entity 𝑣 𝑗 . Here,         complex networks like transformers. This endows AnyGraph with
𝑣𝑖 and 𝑣 𝑗 could be vanilla graph nodes, class labels, or graph labels.           the advantage of fast adaptation when dealing with new datasets.
3.1.2 Graph Expert Routing Mechanism. Inspired by the ef-
fectiveness of graph self-supervised learning tasks [12], we pro-                 3.2    Adaptive and Efficient Graph Experts
pose measuring the competence of expert models on specific graph                  3.2.1 Addressing In-domain Graph Heterogeneity. To handle
datasets using the models’ self-supervised learning loss values.                  graph data with different adjacency and feature dimensionalities,
Specifically, for an input graph G = (V, E), the routing mech-                    the expert models of our AnyGraph employ a structure and feature
anism 𝜓 calculates the dot-product-based relatedness scores for                   unification process. Adjacency matrices and node features of vary-
some positive edges (𝑣𝑐 1 , 𝑣 𝑝 1 ), · · · , (𝑣𝑐𝑆 , 𝑣 𝑝𝑆 ) ∈ E and analogously    ing sizes are both mapped into initial node embeddings of fixed
calculates the relatedness scores for some sampled negative node                  dimensionality using a unified mapping function. Inspired by the
pairs (𝑣𝑐 1 , 𝑣𝑛1 ), · · · , (𝑣𝑐𝑆 , 𝑣𝑛𝑆 ) ∉ E. The following score difference     effectiveness of singular value decomposition (SVD) in extracting
is then calculated as the competence indicator 𝜑𝑘 for the 𝑘-th expert             important latent features, we utilize SVD for this unified mapping
ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY                                                                                                                 Lianghao Xia and Chao Huang∗


       ?              ?            ?                                                             ② High-order           ③ Feat. Encoder     𝒢′8 𝒢′3 𝒢′6 … Expert1 Expert2
                                                  Expert1             Expert2        Expert3
                                                                …                …                          𝑙=1
                                                  2183 stps           567 stps       1860 stps                𝑙=2                           Training Steps Routed Experts
                                                           Frequency Regularization                                 ∑

               …                   …                                                                            1                                  𝒢1′′                     𝒢2′′
     Expert          Expert             Expert                  Positive                                        𝑑𝑖 𝑑𝑗

                                                  +                      -             -                                                   Edges          Initial   Edges          Initial
                                                                                                       ① ℝ𝑁×𝑑               ℝ𝑁×𝑑                          Embs                     Embs
                                                                          Negative
              Routing      Mechanism                                Sample Edges                       ℝ𝑁×𝑁
                                                                                                                    Unify
                                                                                                                         ℝ𝑁×𝑑0                      𝒢1′                     𝒢2′
                                                      𝒢1                     𝒢2                    Structure & Feature Heterogeneity          Augment                    Augment
               …                   …
                                                                                                                                                    𝒢1                      𝒢2
      Acad.          Ecom.              Othrs.
      MoE Architecture of AnyGraph                          Graph Experts Routing                   Graph Expert Model Design                      Cross-domain Training

Figure 2: The proposed graph Mixture-of-Experts (MoE) paradigm enables AnyGraph to learn a diverse ensemble of graph
experts, each tailored to specific structural characteristics. The lightweight expert routing mechanism allows AnyGraph to
quickly identify and activate the most relevant experts for a given input graph, without extensive retraining or fine-tuning.


process as follows:                                                                              MLP module comprises a linear transformation W ∈ R𝑑 ×𝑑 and bias
                                                                                                 b ∈ R𝑑 , followed by a ReLU non-linear activation, a dropout layer,
          UA, ΛA, VA = SVD( Ã) UF, ΛF, VF = SVD(F)
                      √︁                                                                       a residual connection, and layer normalization.
      E0 =LayerNorm UA ΛA + VA ΛA + Flip(UF ΛF )
                                √︁             √︁
                                                                                        (5)      Multiple Simple Experts as Strong Encoder. It is worth noting
                                                                                                 that each graph expert in AnyGraph adopts a very simple learnable
Here, UA, UA ∈ R | V | ×𝑑 and UF ∈ R | V | ×𝑑 , VF ∈ R𝑑0 ×𝑑 refer to                             network, foregoing the capacity to mine complex hidden relations
the 𝑑-dimensional features obtained through SVD of the Laplacian-                                like those in heavy graph neural networks such as GATs [25] and
normalized adjacency matrix Ã [14] and the node feature matrix                                  GraphTransformers [10]. This is because AnyGraph employs a MoE
F, respectively. If the dimensionality of Ã or F is less than 𝑑, SVD                            architecture, where each expert is expected to handle only a sub-
uses a smaller rank 𝑑 ′ equal to the smallest dimensionality of Ã/F,                            domain of all graph data through simple feature transformations.
and the remaining dimensions are padded with zeros up to 𝑑.                                      Therefore, no complex models are needed to accommodate differ-
   Due to the nature of SVD, the dimensions of these features                                    ent types of graphs within a single network. Compared to other
(U∗, V∗ ) are ranked from the most important to the least important,                             graph foundation models that rely on a single heavy network, this
corresponding to the descending eigenvalues in the diagonal matri-                               approach further accelerates the training and inference processes.
ces ΛA and ΛF . In light of this characteristic, we propose to better
preserve the most important feature dimensions for both Ã and F.                                3.3     Efficient Cross-domain Model Training
In particular, the function Flip(·) reverses the 𝑑 dimensions of each                            To maximize the cross-graph generalizability of AnyGraph, the
row for the SVD features of F, such that the important features of                               training samples from different datasets are mixed together and
Ã are aligned with the less important features of F, and vice versa.                            randomly shuffled for model training. Each batch of training sam-
High-order Connectivity Injection. A non-trainable layer nor-                                    ples contains the following information:
malization LayerNorm(·) is applied for numerical stability. The                                                               {(𝑣𝑐𝑏 , 𝑣 𝑝𝑏 )|𝑏 ∈ 𝐵} ⊂ E G𝑠 ,
initialized embeddings, denoted as E0 ∈ R | V | ×𝑑 , have consistent                                                        ©                                        ª
representation dimensionality and relatively stable semantics across                                                    S = ­ E1 = InitialEmbed(G𝑠 ),
                                                                                                                            ­                                        ®
                                                                                                                                                                     ®                       (8)
datasets. To better preserve the multi-hop connection information                                                           « 𝑓Θ𝑘 where 𝑘 = 𝜓 (G𝑠 )                  ¬
into the initial embeddings, AnyGraph adopts a simplified GCN                                    Inspired by the effectiveness of link-wise graph pre-training tasks [12],
without parameters [28] for E0 as follows:                                                       we utilize link prediction as the training task. Here, (𝑣𝑐𝑏 , 𝑣 𝑝𝑏 ) de-
                      𝐿                                                                          notes the positive edges for link prediction, and 𝐵 denotes the batch
                            (𝑙 )       (𝑙 )      (𝑙 −1)         (0)
                     ∑︁
              E1 =         E0 , E0 = Ã · E0               , E0       = E0              (6)      size. To facilitate batch training, each training batch involves only
                     𝑙=1                                                                         one training graph G𝑠 . The initial node embeddings E1 and the
3.2.2 Efficient and Strong Feature Encoder. To achieve effi-                                     most competent expert model 𝑓Θ𝑘 are preprocessed in advance to
ciency while retaining the capacity to encode graph features, our                                accelerate the training. Specifically, the loss function used by our
graph experts are configured by deep multi-layer perceptron (MLP)                                AnyGraph training is as follows:
networks. Specifically, the final node embeddings given by an expert                                             ∑︁ ∑︁ 1               exp(𝑦ˆ𝑐𝑏 ,𝑝𝑏 − 𝑦ˆmax )
model is calculated iteratively as follows:                                                                L=            − log Í                                      (9)
                                                                                                                           𝐵
                                                                                                                        S 𝑏 ∈𝐵      𝑣𝑛 ∈ VG𝑠 exp(𝑦ˆ𝑐𝑏 ,𝑛 − 𝑦ˆmax )
                                                              
      (𝑙+1)                                 (𝑙 )            (𝑙 )
   Ē       = LayerNorm Dropout ReLU( Ē W + b) + Ē               (7)                           This training objective maximizes the prediction scores for positive
                                                                                                 samples (𝑣𝑐𝑏 , 𝑣 𝑝𝑏 ) and minimizes the predictions for all possible
                                                           (𝐿 ′ )
The final embeddings are denoted as Ê = Ē      ∈ R | V | ×𝑑 , where                            node pairs between 𝑣𝑐𝑏 and all nodes 𝑣𝑛 . To avoid numerical insta-
  ′                                                              (0)
𝐿 represents the number of fully-connected layers. And Ē is                                     bility, we employ a technique where the maximum prediction score
initialized by the aforementioned embeddings E1 . Each layer of our                              of the batch, 𝑦ˆmax , is subtracted from all prediction scores.
AnyGraph: Graph Foundation Model in the Wild                                                                     ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY


3.3.1 Feature and Structure Augmentation. To further enrich                       product-wise relations), academic graphs (e.g. citation and collab-
the training data, the training of AnyGraph undergoes periodic                    oration networks), biological information networks (e.g. relations
reprocessing of, firstly, the initial graph embeddings E1 , and sec-              among drugs and proteins), and other domains like email networks,
ondly, the graph routing results. We demonstrate that such repro-                 website networks, trust networks, and road networks.
cessing augments the features and structures of the original graph                   We set up different dataset groups and conduct cross-dataset
data, thereby training AnyGraph using more diversified input data.                evaluations on these groups. Specifically, all datasets are divided
   For the initial graph embeddings, we periodically reconduct                    into two cross-domain groups, Link1 and Link2, which have a
the SVD and simplified GCN processes after a certain number of                    similar number of total edges and a similar number of domain-
training steps. This helps generate different embedding spaces for                specific edges. Specifically, the Link1 and Link2 groups contain
the same data, thereby greatly improving the generalizability of                  15 and 18 datasets, respectively. For the node classification task,
AnyGraph regarding representation heterogeneity [16]. To prevent                  we use 5 datasets gathered from e-commerce and academic in-
this process from consuming excessive computational time, we                      formation scenarios. Additionally, we have three domain-specific
propose adopting different augmentation frequencies adaptive to                   groups: Ecommerce, Academic, and Others. The Others group
the size of different datasets. Specifically, each dataset undergoes              is primarily composed of biological networks, combined with other
this representation augmentation after |E |/(10𝐵) training steps.                 small domains that have fewer datasets. See Appendix A.1 for more
   For the graph routing results, we also periodically recalculate the            information of our experimental datasets.
recalibrated competence scores. Specifically, the positive sample
                                                                                  4.1.2 Experimental Settings. We follow previous works [9, 14]
pairs (𝑣𝑐𝑠 , 𝑣 𝑝𝑠 ) for 𝑠 = 1, · · · , 𝑆, as well as the negative samples 𝑣𝑛𝑠 ,
                                                                                  for dataset splitting and evaluation metrics. Our AnyGraph model
are randomly sampled. This essentially performs structure augmen-
                                                                                  and the graph foundation models are evaluated on a cross-graph
tation by using a random subset to evaluate the performance of
                                                                                  zero-shot prediction task. For baselines that cannot handle cross-
graph experts on the input graph, thereby enhancing the model’s
                                                                                  dataset transfer, we evaluate their few-shot performance. Details of
robustness against structural noise.
                                                                                  the evaluation protocols are provided in Appendix A.2. The Hyper-
3.3.2 Complexity Analysis. The training and inference process                     parameter Settings of AnyGraph are provided in Appendix A.3.
of AnyGraph is conducted by only one expert model, which has a                    The compared Baseline Methods are introduced in Appendix A.4.
complexity of O (𝐵 ×𝑑 2 ×𝐿 ′ ) for each batch. Since we preprocess the
initial embeddings and the expert routing, these two processes do                 4.2    AnyGraph’s Zero-Shot Prediction (RQ1)
not increase the batch-wise computational complexity. As a result,                To assess the zero-shot prediction capabilities of the AnyGraph
the complexity of the forward and backward steps for AnyGraph is                  model, we conducted an extensive evaluation across 38 graph
much lower than that of other graph foundation models that involve                datasets from various domains. We independently trained two ver-
complex GNNs and graph transformers. Additionally, the expert                    sions of the AnyGraph model - one on the Link1 dataset and the
routing performs O G𝑠 |E𝑠 | × 𝑑 × 𝐾 + G𝑠 |V𝑠 | × 𝑑 2 × 𝐿 ′ × 𝐾
                      Í                      Í
                                                                                  other on the Link2 dataset. Each trained model was then used to
computations, where the latter term empirically has a larger scale                make zero-shot predictions on datasets it was not originally trained
compared to the former term. This dominant term is similar to a                   with. It is important to note that the Link1 and Link2 datasets do not
simple GCN network of a comparable model size. Overall, Any-                      share the same feature spaces or sources of data collection, which
Graph is more efficient than existing methods in both training                    adds to the complexity and challenges of the zero-shot evaluation.
and inference, and the additional computations for routing have a                 We also compare our AnyGraph with existing graph foundation
complexity comparable to simple GNNs.                                             models. And in this comparison we add another AnyGraph-F ver-
                                                                                  sion, which removes the utilization of node features. The outcomes
4     Evaluation                                                                  of this evaluation are detailed in Table 1 and Table 2, and our key
Our experiments aim to answer the following Research Questions:                   observations are listed as follows:
• RQ1: How does the zero-shot prediction performance of Any-                      i) Superior Generalizability across Diverse Datasets. • Supe-
  Graph compare to different baseline methods?                                    rior Prediction Accuracy. Compared to the few-shot capabilities
• RQ2: How do AnyGraph’s various modules impact its overall                       of existing GNN models, pre-training techniques, and foundation
  performance with the contribution of each component?                            models, AnyGraph demonstrates exceptional zero-shot prediction
• RQ3: How does the model size and the amount of training data                    accuracy across various domains. This superior performance spans
  influence the performance of AnyGraph?                                          both link prediction and node classification tasks. • Effectively
• RQ4: How interpretable is the expert routing mechanism within                   Handling Heterogeneity. The enhanced generalizability can be at-
  AnyGraph’s graph Mixture-of-Experts (MoE) architecture?                         tributed to the effective handling of structure-level and feature-level
• RQ5: How is the scalability and efficiency of AnyGraph compare                  data heterogeneity through unified structure and feature represen-
  to fine-tuning methods when adapting to new datasets?                           tations in the expert models. This approach enables AnyGraph to
                                                                                  develop comprehensive modeling functions that are universally
4.1     Experimental Settings                                                     applicable across different graph data scenarios. • Comprehen-
4.1.1 Experimental Datasets. To conduct a comprehensive eval-                     sive Training. Additionally, the extensive training regimen, which
uation of the cross-domain generalizability of graph models, we                   incorporates a variety of large-scale datasets, equips AnyGraph
employ a total of 38 graph datasets. These datasets span a wide                   with a deep and broad expertise in graph modeling and prediction.
range of domains, including e-commerce (e.g. user interactions and
ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY                                                                                   Lianghao Xia and Chao Huang∗


Table 1: We evaluate the AnyGraph model (in zero-shot settings) and baseline models (with 5% and 10% training data) on link
prediction (Recall@20, NDCG@20), node classification (Accuracy, Macro F1), and graph classification (Accuracy, Macro F1).
                     GIN                       GAT                         GPF               GraphPrompt                  GraphCL           AnyGraph
 Data
          Train 5%      Train 10%   Train 5%      Train 10%      Tune 5%      Tune 10%    Tune 5%   Tune 10%        Tune 5%    Tune 10%       0-shot
Metric Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG Rec NDCG
Link1 6.46 3.06 11.80 5.45 13.52 6.65 13.45 6.78 6.04 2.92 6.80 3.27 4.33 2.24 5.42 3.11 17.23 9.00 20.55 10.76 23.94 12.68
Link2 6.72 4.50 21.62 13.41 9.83 5.91 15.30 8.84 7.44 4.25 16.58 9.84 6.06 3.36 6.10 3.62 29.18 17.62 31.42 19.91 46.42 27.21
Ecom. 3.36 2.58 13.41 8.06 3.79 2.94 9.64 5.78 7.25 3.84 18.72 10.94 4.90 2.59 6.06 3.36 22.13 13.19 26.05 14.59 26.92 15.05
Acad. 10.82 4.70 20.61 9.04 14.95 6.29 11.17 4.67 13.22 5.80 14.83 6.41 6.73 3.05 7.72 3.40 24.86 12.50 28.69 14.31 32.74 15.31
Othrs. 6.92 4.46 18.43 11.85 16.34 9.22 16.17 20.88 2.40 2.12 4.51 3.44 2.93 2.36 3.42 2.72 24.54 14.93 24.62 15.90 46.83 28.97
Metric Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1 Acc MacF1
Node 20.79 19.46 36.04 30.60 53.76 40.14 54.83 41.61 12.77 11.45 16.29 16.00 18.01 20.59 23.15 22.89 43.70 33.72 48.75 36.15 64.31 43.24

Table 2: Comparing AnyGraph framework to existing graph                             In-domain generalization can be more straightforward, leading
foundation models in zero-shot prediction capabilities.                             to a plateau in performance improvements. This insight into the
 Method                    GraphGPT                      OpenGraph                  scaling law for graph data encourages further exploration of
 Data               Pubmed           Cora               Ecom. w/o GR                larger models on more complex graph learning tasks.
 Metric           Acc   MacF1    Acc    MacF1           Recall NDCG              • MoE Architecture. The integration of the Mixture of Experts
 Baseline       0.1813     0.1272   0.7011     0.6491   0.1444    0.1099            (MoE) architecture allows AnyGraph to effectively manage and
 AnyGraph-F     0.5852     0.5325   0.7134     0.6003   0.2281    0.1600            utilize a broader spectrum of knowledge, particularly in this zero-
 AnyGraph       0.6088     0.5492   0.7809     0.7591   0.2382    0.1552            shot scenario characterized by significant distribution disparities.
                                                                                 ii) Emergent Abilities of AnyGraph. The overall zero-shot per-
ii) Limitation of existing pre-training GNNs. • Challenges of                    formance curve illustrates that as the model size increases, the
Cross-Domain Transfer. Existing pre-training and tuning meth-                    performance sometimes experiences periodic stagnation. With fur-
ods, like GPF, GraphPrompt, and GraphCL, employ self-supervised                  ther increments in parameters, AnyGraph’s performance undergoes
learning and are pre-trained on half the datasets, then fine-tuned               a sudden significant improvement. This phenomenon indicates the
on the remaining datasets using few-shot data. However, this pre-                emergent abilities of AnyGraph, demonstrating the effectiveness
training often fails to yield significant improvements due to sub-               of scaling up in enhancing its generalization capabilities.
stantial distribution disparities across data domains. For instance,             iii) Insufficient training data may bring bias. In the initial stages
datasets may exhibit vastly different link densities or utilize distinct         of increasing the training data, the introduction of new datasets
node features, which significantly challenges the transfer of useful             might negatively impact performance due to their differences from
knowledge from divergent pre-training datasets during fine-tuning                the test graphs. However, this issue can be mitigated by further
and prediction. • AnyGraph’s Robust Adaptability To address this                 expanding the training data. By providing the model with a more
challenge, the AnyGraph model incorporates multiple graph expert                 comprehensive set of training samples, it helps prevent overfitting
models tailored to various sub-domains of graph data. This MoE                   and reduces bias stemming from dataset disparities.
architecture effectively manages datasets from distinctly different
domains, such as e-commerce user behaviors, academic networks,                   4.4     Ablation Study (RQ3)
and road networks, demonstrating its robust adaptability.
                                                                                 This section evaluates the effectiveness of AnyGraph’s sub-modules
                                                                                 by comparing ablated variants in terms of their zero-shot and full-
4.3     Scaling Law of AnyGraph Framework (RQ2)                                  shot performance across both cross-domain datasets and domain-
In this section, we explore the applicability of the scaling law to              specific datasets (specifically Academic data). The results are in
AnyGraph. We conduct experiments using 18 different versions                     Figure 4. We make the following observations:
of AnyGraph, each differing in model size and quantity of train-                 • MoE Significantly Enhances Zero-Shot Performance. The
ing data. Specific configurations of these variants are discussed                  -MoE variant, which employs a single expert model without the
in Appendix A.5. The evaluation results are depicted in Figure 3,                   MoE architecture, demonstrates decent performance on datasets
which includes overall and domain-specific performance, as well as                  on which it was trained, as shown in parts (b) and (c). However,
zero-shot and full-shot outcomes. Our key findings are as follows:                  this variant exhibits a substantial decline in zero-shot prediction
i) Generalizability of AnyGraph Follows the Scaling Law.                            capabilities. This underscores the critical role of the MoE archi-
As the model size and the volume of training data increase, we                      tecture in enhancing AnyGraph’s generalization abilities. The
notice a saturation point in AnyGraph’s full-shot performance. In                   use of multiple expert models significantly expands AnyGraph’s
contrast, the zero-shot prediction accuracy continues to improve.                   modeling capacity, effectively managing the large disparities be-
This pattern supports the scaling law of graph foundation models,                   tween various domains using multiple seperated models.
illustrating that scaling up can significantly enhance the capabilities          • Feature Modeling is Crucial in AnyGraph. In the -Feat vari-
of graph models. Two key factors contribute to this phenomenon:                     ant, node features are omitted, leading to the most significant
• Task Difficulty. The saturation in full-shot performance is partly                degradation in both zero-shot and full-shot performance. This
   because the evaluation tasks might not be challenging enough.                    underscores the effectiveness of AnyGraph’s unified structure
        AnyGraph: Graph Foundation Model in the Wild                                                                                                                           ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY


                                                                                                                                 0.15                                                                  0.15
                0.12                               0.34                              0.12                                                                            0.20
                                                                                                                                 0.14                                                                  0.12
                0.11
       NDCG@20                                NDCG@20                           NDCG@20                                     NDCG@20                             NDCG@20                           NDCG@20
                                                   0.32                              0.10                                                                            0.19
                                                                                                                                 0.13                                                                  0.09
                0.10
                0.09                               0.30                                                                          0.12                                0.18                           0.06
                                                                                     0.08
                0.08                 Zero-shot                          Full-shot                             Zero-shot          0.11                    Zero-shot                        Full-shot                    Zero-shot
                                               0.28                                                                                                                                                 0.03
                        4 5 6 7 8                         4 5 6 7 8                          12       14       16      18                   4 5 6 7 8 0.17                  4 5 6 7 8                      13       15      17
                         log # Model Params                log # Model Params                   log # Training Samples                       log # Model Params              log # Model Params          log # Training Samples
                       (a) Cross-domain performance on Link1 (zero-shot) and Link2 (full-shot).                                                               (b) Performance on academic data.

                                                   0.28                              0.12           Zero-shot                                                        0.50                                       Zero-shot
            0.120                                                                                                                0.10
                                                   0.26                                                                                                                                                0.10
                                                                                     0.11                                                                            0.49
       NDCG@20                                NDCG@20                           NDCG@20                                     NDCG@20                             NDCG@20                           NDCG@20
            0.115                                  0.24                                                                          0.08
                                                   0.22                                                                                                                                                0.08
            0.110                                                                    0.10                                                                            0.48
                                                   0.20                                                                          0.06
                                                                                                                                                                                                       0.06
            0.105                   Zero-shot      0.18                 Full-shot 0.09                                                                   Zero-shot 0.47                   Full-shot
                                                                                                                                 0.04
                         4 5 6 7 8                        4 5 6 7 8                      13       15     17                                 4 5 6 7 8                       4 5 6 7 8                           13       15     17
                         log # Model Params                log # Model Params          log # Training Samples                                log # Model Params              log # Model Params               log # Training Samples
                                           (c) Performance on ecommerce data.                                                                                      (d) Performance on others.

            Figure 3: Zero-shot and full-shot performance w.r.t. the number of model parameters and the amount of training samples.

   5
                          -MoE             -FreqReg                   -Aug          -Feat                 Ours                               Cora                                       Yelp
   4
                                       0.13                                                  0.36                                              CS                                      YelpT
   3
NDCG                                                                                                                                     arxiv-ta                                    SteamT
            0.24
                                       0.12                       0.55                       0.34
   2        0.22                       0.11                                                                                                Photo                                      AmazT
                                                                                             0.32
   1   Recall
            0.20                    NDCG
                                       0.10                  Recall
                                                                  0.50                    NDCG                                           GReads                                        Amaz
            0.18                       0.09                       0.45
                                                                                             0.30                                         Fitness                                        Soc
                                                                                             0.28                                          ML1M                                        Email
            0.16                       0.08
                                                                                                                                         ML10M                                          p2p
         (a) Multi-domain zero-shot performance. (b) Multi-domain full-shot performance.                                                             0 1 2 3 4 5 6 7                             0 1 2 3 4 5 6 7
                                                                                                                                                          Expert ID                                   Expert ID
            0.35                                                  0.45                       0.30                                     Figure 5: Competence score between datasets and expert mod-
                                       0.16
                                                                                             0.25                                     els, given by the routing mechanism of AnyGraph.
            0.30                       0.14                       0.40
       Recall                       NDCG                     Recall                       NDCG
                                                                                             0.20                                     4.5      Investigation on Expert Routing (RQ4)
            0.25                       0.12                       0.35
                                       0.10                                                  0.15                                     This section delves into the expert routing mechanism of AnyGraph.
            0.20                                                  0.30
                                                                                                                                      Figure 5 displays the competence scores of various expert models for
         (c) Zero-shot performance on Academic. (d) Full-shot performance on Academic.                                                the input datasets, as determined by AnyGraph’s routing algorithm
        Figure 4: Impact of different sub-modules on the zero-shot                                                                    based on the self-supervised loss. The figure illustrates that datasets
        and full-shot prediction capabilities of AnyGraph.                                                                            sharing common characteristics—such as source of collection or
                                                                                                                                      feature construction method—are often routed to the same expert
                                                                                                                                      models by AnyGraph. For instance, datasets like arxiv-ta, Photo,
          and feature representation method in successfully learning fea-                                                             GoodReads, and Fitness, which utilize a common text-embedding-
          tures. This component is crucial for tackling in-domain graph                                                               based feature space, are assigned to highly similar experts (expert
          data heterogeneity. Additionally, this outcome highlights the fea-                                                          0, 2, 4, 5). Additionally, ML1M and ML-10M, both sourced from the
          sibility of unifying different feature spaces created by various                                                            movie-rating platform Movielens, are predominantly associated
          methods into a single model for general use.                                                                                with expert 1. It is also notable that this routing pattern extends
        • Effectiveness of Frequency Regularization and Graph Aug-                                                                    to zero-shot datasets, as shown on the right part of Figure 5. Here,
          mentation. In the -FreqReg and -Aug variants of AnyGraph,                                                                   YelpT, SteamT, and AmazonT, which share the same feature space,
          the routing adjustment based on the training frequency of ex-                                                               are assigned to very similar expert models. This outcome under-
          perts and the feature and structure augmentation are individually                                                           scores the efficacy of AnyGraph’s routing mechanism in identifying
          removed. Frequency regularization specifically prevents subopti-                                                            the appropriate expert models for various datasets, and also show-
          mal training outcomes by ensuring that all experts are utilized,                                                            cases its explainability in revealing graph-wise relatedness.
          avoiding scenarios where the majority are overlooked. Mean-
          while, the graph augmentation module plays a crucial role in                                                                4.6      Efficiency Study (RQ5)
          enriching the training graph data, thus enhancing the model’s                                                               Tuning Curve Comparison. To evaluate the efficiency of Any-
          robustness. The outcomes from these modifications affirm the                                                                Graph, we compare its fine-tuning process with that of GraphCL
          beneficial impact of these two components within AnyGraph.                                                                  and the training from scratch process of a GCN model. As depicted
          Omitting them can lead to biased model training, which under-                                                               in Figure 6, when fine-tuned on a new dataset, the pre-trained Any-
          mines the robustness of AnyGraph in handling diverse datasets.                                                              Graph rapidly achieves a high performance saturation point. In
ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY                                                                                                 Lianghao Xia and Chao Huang∗


     0.175                                                                                     both node-specific information and higher-order topological struc-
                                                      0.25
     0.150                                                                                     tures. Notable techniques include Graph Convolutional Networks
                                                      0.20                                     (GCNs) [11], Graph Attention Networks (GATs) [1], Graph Isomor-
     0.125                                                         GCN Training
                                                      0.15
Recall                                           Recall            GraphCL Fine-tuning
     0.100                                                                                     phism Network (GIN) [33], and Graph Transformer [10], which
                                                      0.10         AnyGraph Fine-tuning        improves the encoding function for better graph modeling. Despite
     0.075              GCN Training
                        GraphCL Fine-tuning           0.05                                     these advancements, these methods still require high-quality train-
     0.050              AnyGraph Fine-tuning
                                                      0.00                                     ing data and often struggle with generalization capabilities.
              0         20        40        60               0         20        40       60
                   Training steps (x100)                          Training steps (x100)        Self-Supervised Graph Learning. Given the challenges with the
         (a) Performance on Citation-2019.                   (b) Performance on PPA.           generalizability of GNNs, considerable research efforts [32] have
                                                                                               focused on enhancing GNNs through self-supervised learning ob-
             Figure 6: Performance v.s. training/tuning steps.
                                                                                               jectives, aiming to capture invariant graph features. Specifically,
                                                                                               GraphCL [36] introduced a contrastive pre-training approach for
 Table 3: Training time for each 100 steps on different data.
                                                                                               graph data, designed to learn authentic graph characteristics that
   Dataset CS ML1M Yelp Email Cite19 roadNet PPA
                                                                                               are robust to structural and feature perturbations. Building on this,
    GCN 1.5s 4.2s 6.0s 2.5s 19.2s            27.8s 101.1s                                      JOAO [35] and GCA [39] have developed adaptive augmentation
  GraphCL 1.1s 4.9s 9.4s 2.8s 43.1s          57.1s 130.8s                                      strategies for self-supervised tasks, effectively mitigating the ad-
    Ours    1.5s 3.5s 6.1s 3.0s 31.6s        37.3s   41.1s                                     verse effects of random augmentations. Subsequent works have
                                                                                               sought to quickly adapt these pre-trained models to downstream
some instances, such as with the PPA dataset, GraphCL and the                                  tasks and evolving graph data, as demonstrated by GPF [6] and
end-to-end trained GCN struggle to attain comparable performance                               GraphPrompt [19]. Despite these advancements, the generalizabil-
levels. This advantage is based on i) the strong cross-domain gen-                             ity of these methods remains confined to graph data with similar
eralization capabilities of AnyGraph, which bring a high starting                              structural patterns and feature spaces, thus not addressing the cross-
point for the new dataset, and ii) the efficiency of AnyGraph’s MoE                            domain generalization challenges highlighted in this paper.
architecture, which requires only one MLP network for efficient                                Large-scale Graph Pre-training. Recent advances in graph mod-
but effective modeling and parameter tuning.                                                   eling have seen efforts to pre-train large-scale graph models across
   In addition, it is observed that pre-training GraphCL does not                              multiple datasets to improve their generalization abilities, drawing
consistently benefit its fine-tuning on new datasets, as evidenced                             inspiration from the strong generalization capabilities of large lan-
by GraphCL’s underperformance relative to GCN in Figure 6(b).                                  guage models (LLMs) [30]. For instance, OFA [17] and ZeroG [16]
This should be ascribed to the large distribution gap between the                              utilize text embeddings to standardize the feature spaces across
pre-training data Link2 and the test data PPA.                                                 various graph datasets and tasks, facilitating cross-dataset training
Training Time Comparison. To evaluate the efficiency of the                                    of graph models. Models like InstructGLM [34] GraphGPT [23] and
models under consideration, we compared the training times of the                              LLaGA [4] synchronize graph representation spaces with the hid-
three models. As indicated in Table 3, AnyGraph, despite having                                den spaces of LLMs, thus enabling the application of general LLMs
significantly more parameters, has training times that are compara-                            for graph prediction tasks. Furthermore, HiGPT [24] expands the
ble to, or even less than, the other two models. This underscores                              capabilities of LLMs to accommodate heterogeneous graph data.
the efficiency of our model design, and demonstrates the efficiency                               Despite these advancements, most generalized graph models
of AnyGraph to adapt to new data through model tuning.                                         require substantial access to and integration of text features, which
   Specifically, AnyGraph avoids the cumbersome process of full-                               confines their use primarily to text-abundant environments such
graph propagation at each training step. Instead, it utilizes structure-                       as academic networks. Additionally, these methods are typically
aware embeddings derived through a non-trainable pre-processing                                trained within specific application realms, failing to address the
method. This approach significantly reduces both the time and                                  significant variances between datasets from diverse domains.
memory requirements for AnyGraph. Furthermore, the MoE archi-
tecture equips AnyGraph with the capability to use only 1/𝐾 of                                 6   Conclusion
the computational resources for most prediction and optimization                               In this work, the presented AnyGraph framework, an effective and
processes, thereby greatly reducing overall computational costs.                               efficient graph foundation model designed to address the multi-
                                                                                               faceted challenges of structure and feature heterogeneity across
5          Related Works                                                                       diverse graph datasets. AnyGraph’s innovative Mixture-of-Experts
Graph Neural Models. Graph learning has garnered significant                                   (MoE) architecture, coupled with its dynamic expert routing mech-
interest for its broad applicability across various fields such as                             anism, positions it at the state-of-the-art of cross-domain gener-
user behavior modeling, social analysis, and studies in biology and                            alization capabilities. Extensive experiments on 38 varied graph
chemistry [2, 8]. Graph neural networks (GNNs) learn node repre-                               datasets have not only underscored AnyGraph’s superior zero-shot
sentation vectors for downstream tasks like node classification and                            learning performance but also its robustness to distribution shifts
link prediction. The core mechanism involves iterative message                                 and its adherence to scaling laws, thereby enhancing its predictive
passing, refining node embeddings to capture both node-specific                                accuracy with increased model size and data volume. The model’s
information and higher-order topological structures. This process                              efficiency in training and inference, validated through comparison
ensures that the final node embeddings effectively encapsulate                                 with existing methods, further cements its practical applicability.
AnyGraph: Graph Foundation Model in the Wild                                                                                           ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY


References                                                                                           volume 36, 2024.
 [1] S. Brody, U. Alon, and E. Yahav. How attentive are graph attention networks? In            [21] N. Muennighoff, A. Rush, B. Barak, T. Le Scao, N. Tazi, A. Piktus, S. Pyysalo,
     ICLR, 2022.                                                                                     T. Wolf, and C. A. Raffel. Scaling data-constrained language models. In NeurIPS,
 [2] J. Chang, C. Gao, Y. Zheng, Y. Hui, Y. Niu, Y. Song, D. Jin, and Y. Li. Sequential              volume 36, 2024.
     recommendation with graph neural networks. In SIGIR, pages 378–387, 2021.                  [22] M. Sun, K. Zhou, X. He, Y. Wang, and X. Wang. Gppt: Graph pre-training and
 [3] M. Chen, Z. Zhang, T. Wang, M. Backes, M. Humbert, and Y. Zhang. Graph                          prompt tuning to generalize graph neural networks. In KDD, pages 1717–1727,
     unlearning. In SIGSAC, pages 499–513, 2022.                                                     2022.
 [4] R. Chen, T. Zhao, A. Jaiswal, N. Shah, and Z. Wang. Llaga: Large language and              [23] J. Tang, Y. Yang, W. Wei, L. Shi, L. Su, S. Cheng, D. Yin, and C. Huang. Graphgpt:
     graph assistant. In ICML, 2024.                                                                 Graph instruction tuning for large language models. In SIGIR, pages 491–500,
 [5] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon,                        2024.
     C. Schuhmann, L. Schmidt, and J. Jitsev. Reproducible scaling laws for contrastive         [24] J. Tang, Y. Yang, W. Wei, L. Shi, L. Xia, D. Yin, and C. Huang. Higpt: Heterogeneous
     language-image learning. In CVPR, pages 2818–2829, 2023.                                        graph language model. In KDD, 2024.
 [6] T. Fang, Y. Zhang, Y. Yang, C. Wang, and L. Chen. Universal prompt tuning for              [25] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio. Graph
     graph neural networks. NeurIPS, 2023.                                                           attention networks. In ICLR, 2018.
 [7] M. Fey, W. Hu, K. Huang, J. E. Lenssen, R. Ranjan, J. Robinson, R. Ying, J. You, and       [26] J. Wang, D. Chen, Z. Wu, C. Luo, L. Zhou, Y. Zhao, Y. Xie, C. Liu, Y.-G. Jiang, and
     J. Leskovec. Position: Relational deep learning-graph representation learning on                L. Yuan. Omnivl: One foundation model for image-language and video-language
     relational databases. In ICML, 2024.                                                            tasks. NeurIPS, 35:5696–5710, 2022.
 [8] Z. Hao, C. Lu, Z. Huang, H. Wang, Z. Hu, Q. Liu, E. Chen, and C. Lee. Asgn: An             [27] W. Wang, J. Dai, Z. Chen, Z. Huang, Z. Li, X. Zhu, X. Hu, T. Lu, L. Lu, H. Li, et al.
     active semi-supervised graph neural network for molecular property prediction.                  Internimage: Exploring large-scale vision foundation models with deformable
     In KDD, pages 731–752, 2020.                                                                    convolutions. In CVPR, pages 14408–14419, 2023.
 [9] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang. Lightgcn: Simplifying               [28] F. Wu, A. Souza, T. Zhang, C. Fifty, T. Yu, and K. Weinberger. Simplifying graph
     and powering graph convolution network for recommendation. In SIGIR, pages                      convolutional networks. In ICLR, pages 6861–6871, 2019.
     639–648, 2020.                                                                             [29] L. Xia, C. Huang, C. Huang, K. Lin, T. Yu, and B. Kao. Automated self-supervised
[10] Z. Hu, Y. Dong, K. Wang, and Y. Sun. Heterogeneous graph transformer. In                        learning for recommendation. In WWW, pages 992–1002, 2023.
     WWW, pages 2704–2710, 2020.                                                                [30] L. Xia, B. Kao, and C. Huang. Opengraph: Towards open graph foundation models.
[11] D. Jin, Z. Yu, C. Huo, R. Wang, X. Wang, D. He, and J. Han. Universal graph                     arXiv preprint arXiv:2403.01121, 2024.
     convolutional networks. In NeurIPS, pages 10654–10664, 2021.                               [31] T. Xiao, Z. Chen, D. Wang, and S. Wang. Learning how to propagate messages in
[12] W. Jin, X. Liu, X. Zhao, Y. Ma, N. Shah, and J. Tang. Automated self-supervised                 graph neural networks. In KDD, pages 1894–1903, 2021.
     learning for graphs. In ICLR, 2022.                                                        [32] Y. Xie, Z. Xu, J. Zhang, Z. Wang, and S. Ji. Self-supervised learning of graph
[13] W. Jin, Y. Ma, X. Liu, X. Tang, S. Wang, and J. Tang. Graph structure learning for              neural networks: A unified review. Transactions on Pattern Analysis and Machine
     robust graph neural networks. In KDD, pages 66–74, 2020.                                        Intelligence (TPAMI), 45(2):2412–2429, 2022.
                                                                                                [33] K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerful are graph neural
[14] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolu-
                                                                                                     networks? In ICLR, 2018.
     tional networks. In ICLR, 2017.
                                                                                                [34] R. Ye, C. Zhang, R. Wang, S. Xu, and Y. Zhang. Language is all a graph needs. In
[15] G. Li, M. Müller, B. Ghanem, and V. Koltun. Training graph neural networks
                                                                                                     EACL, pages 1955–1973, 2024.
     with 1000 layers. In ICML, pages 6437–6449, 2021.
                                                                                                [35] Y. You, T. Chen, Y. Shen, and Z. Wang. Graph contrastive learning automated. In
[16] Y. Li, P. Wang, Z. Li, J. X. Yu, and J. Li. Zerog: Investigating cross-dataset zero-shot
                                                                                                     ICML, pages 12121–12132. PMLR, 2021.
     transferability in graphs. In KDD, 2024.
                                                                                                [36] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen. Graph contrastive learning
[17] H. Liu, J. Feng, L. Kong, N. Liang, D. Tao, Y. Chen, and M. Zhang. One for all:
                                                                                                     with augmentations. NeurIPS, 33:5812–5823, 2020.
     Towards training one graph model for all classification tasks. In ICLR, 2024.
                                                                                                [37] Q. Zhang, S. Pei, Q. Yang, C. Zhang, N. V. Chawla, and X. Zhang. Cross-domain
[18] Y. Liu, M. Jin, S. Pan, C. Zhou, Y. Zheng, F. Xia, and S. Y. Philip. Graph self-
                                                                                                     few-shot graph classification with a reinforced task coordinator. In AAAI, pages
     supervised learning: A survey. Transactions on Knowledge and Data Engineering
                                                                                                     4893–4901, 2023.
     (TKDE), pages 5879–5900, 2022.
                                                                                                [38] H. Zhao, A. Chen, X. Sun, H. Cheng, and J. Li. All in one and one for all: A simple
[19] Z. Liu, X. Yu, Y. Fang, and X. Zhang. Graphprompt: Unifying pre-training and
                                                                                                     yet effective method towards cross-domain graph pretraining. In KDD, 2024.
     downstream tasks for graph neural networks. In WWW, pages 417–428, 2023.
                                                                                                [39] Y. Zhu, Y. Xu, F. Yu, Q. Liu, S. Wu, and L. Wang. Graph contrastive learning with
[20] H. Mao, Z. Chen, W. Jin, H. Han, Y. Ma, T. Zhao, N. Shah, and J. Tang. Demystifying
                                                                                                     adaptive augmentation. In WWW, pages 2069–2080, 2021.
     structural disparity in graph neural networks: Can one size fit all? In NeurIPS,
ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY                                                                           Lianghao Xia and Chao Huang∗


A Appendix                                                                • Others: This group contains all the biological datasets, and other
                                                                            datasets including Email-Enron, Web-Stanford, RroadNet-PA,
A.1 Experimental Datasets
                                                                            P2P-Gnutella06, Soc-Epinions1.
We utilize a total of 38 graph datasets across various domains. The
entire dataset contains 14,437,372 nodes, and 199,265,688 edges. The      A.2    Evaluation Protocols
dataset specifics are detailed below:
                                                                          All datasets used in this study are sourced from previous research as
E-commerce Datasets. This category includes 15 datasets from
                                                                          referenced [16, 23]. We adhere to the original data splits from these
various e-commerce contexts such as user rating platforms and on-
                                                                          sources to delineate our training and testing sets. Given that many
line retail services. These datasets vary in terms of the presence and
                                                                          baseline methods are not equipped to manage zero-shot prediction
type of node features. For instance, datasets such as Amazon-book,
                                                                          across datasets, we instead assess their few-shot capabilities. This
Yelp2018, Gowalla, Yelp-text, Amazon-text, Steam-text, Goodreads,
                                                                          allows for a comparative analysis against the zero-shot performance
Amazon-Fitness, Amazon-Photo, Movielens-1M, Movielens-10M,
                                                                          of AnyGraph. We employ specific evaluation settings tailored to
Products-home, Products-tech, Home-node, Tech-node are included.
                                                                          each method, detailed as follows:
Notably, Amazon-text, Steam-text, and Yelp-text utilize the same
method for feature generation, while Fitness, Photo, and Goodreads        • Zero-shot Setting for AnyGraph, GraphGPT, and Open-
employ a different consistent method.                                        Graph. In our study, AnyGraph and two comparative graph
Academic Network Datasets. We use 13 datasets focused on                     foundation models, GraphGPT and OpenGraph, undergo evalua-
academic networks, which include citation and collaboration rela-            tions for zero-shot prediction capabilities. We pre-train two in-
tions among scholars and papers. These datasets represent various            stances of AnyGraph using Link1 and Link2 datasets. The model
research fields and employ diverse feature generation methods,               pre-trained on Link1 is then tested for zero-shot performance
such as NLP embeddings, bag-of-words, and different versions of              on the Link2 group datasets, and vice versa. Results labeled as
large language models. The specific datasets are Cora, Pubmed,               "zero-shot" for AnyGraph are derived using this cross-evaluation
Arxiv, Cora-link, Pubmed-link, Citeseer, CS, Arxiv-link, Arxiv-t             method. Conversely, results marked as "full-shot" pertain to
(with features derived using an alternative method), Citation-2019,          supervised learning outcomes, where, for example, the model
Citation-20Century, OGB-Collab.                                              trained on Link1 is tested on the test sets of Link1 group datasets.
Biological Information Networks. Our collection includes 6                   For GraphGPT and OpenGraph, we utilize the models as released
datasets related to biological entities like proteins, drugs, and dis-       in their respective original studies, which were pre-trained on
eases. This category features networks such as OGB-DDI, OGB-PPA              specified datasets.
and four protein relation networks for different species, denoted as      • Zero-shot Node Classification for AnyGraph. Drawing from
Proteins-0, Proteins-1, Proteins-2, Proteins-3.                              insights in prior research [22], our approach to zero-shot node
Other Datasets. In addition to the categories mentioned above,               classification involves a novel method where label classes are
we include 5 datasets from various other fields: an email network            represented as distinct nodes. We then connect existing nodes
Email-Enron, a website network Web-Stanford, a road network                  that have training labels directly to these new class nodes. This
dataset RoadNet-PA, a P2P web network dataset P2P-Gnutella06,                technique eliminates the need for learning specific parameters for
and a trust network dataset Soc-Epinions1.                                   each class within the zero-shot learning framework, streamlining
Dataset Groups. For conveinience of performance evaluation, we               the process. We have integrated this innovative approach into
split the many datasets using different grouping methods. Firstly,           baseline methods as well, enhancing their capability to handle
two big data groups Link1 and Link2 are made using all the link              unseen node labels effectively.
prediction datasets. Notably, datasets from the same source of col-       • Few-shot Training for GIN and GAT. The GIN and GAT mod-
lection, such as Movielens-1M and Movielens-10M, or uses the same            els, employed as end-to-end training baselines, undergo training
method to generate features, such as Fitness, and Photo, are put             from scratch on few-shot subsets of the evaluation datasets. This
into the same group, to avoid information leakage when evaluating            approach is necessary because these models are not well-suited
zero-shot performance on the other group. Apart from these two               for cross-dataset transfer, particularly when dealing with datasets
datasets, we also conduct evaluations on domain-specific groups,             that have varying feature dimensionalities.
including E-commerce, Acadmic, and Others. Specifically, these            • Pre-training and Few-shot Tuning for GraphCL, GPF and
data groups contain the following datasets, respectively:                    GraphPrompt. These category of baselien methods follow the
• Link1: Products-tech, Yelp2018, Yelp-textfeat, Products-home,              pre-training-and-fine-tuning mode. In our evaluations, they are
   Steam-text, Amazon-text, Amazon-book, Citation-2019, Citation-            firstly pre-trained using the same pre-training datasets as our
   20Century, Pubmed-link, Citeseer, OGB-PPA, P2P-Gnutella06,                AnyGraph. Then, they experience an additional fine-tuning pro-
   Soc-Epinions1, Email-Enron.                                               cess using the few-shot subsets of the evaluation datasets.
• Link2: Photo, Goodreads, Fitness, Movielens-1M, Movielens10M,           Evaluation Metrics. For link prediction, we follow previous works [9]
   Gowalla, Arxiv, Arxiv-t, Cora, CS, OGB-Collab, Proteins-0, Proteins-   and utilize Recall@20 and NDCG@20 as the evaluation metrics.
   1, Proteins-2, Proteins-3, OGB-DDI, Web-Stanford, RoadNet-PA.          Note that we typically use the summary results of the evaluation
• Ecommerce and Academic: These two datasets contain all the              results across multiple datasets. Results for fifferent datasets are
   domain-specific datasets as mentioned above.                           averaged according to their number of test samples. For the node
                                                                          classification task, we employ the widely-used Accuracy and Macro-
                                                                          F1 score as our metrics [3, 23].
AnyGraph: Graph Foundation Model in the Wild                                                                   ACM KDD, Aug 3–Aug 7, 2025, Woodstock, NY


                                                   Table 4: Statistics of the experimental datasets.
 Dataset       DDI         Collab        ML1m           ML10m      Amazon-book      PPA           Yelp2018      Gowalla Cora        Pubmed Citeseer
 # Nodes      4,267        235,868        9,746          80,555      144,242      576,289          69,716        70,839   2,708      19,717 3,327
 # Edges    1,334,889     1,285,465      920,193       9,200,050    2,984,108    45,495,642       1,561,406     1,027,370 10,556     88,648 9,104
 𝑑 Feats        0            128            0              0            0            58               0             0      1433       500   3703
 Datasets Proteins-0      Proteins-1   Proteins-2     Proteins-3 Products-home Products-tech        Yelp-t      Amazon-t Steam-t Goodreads Fitness
 # Nodes    25,449          6,568        18,108         13,015       9,790        47,428           22,101        20,332 28,547 676,084 173,055
 # Edges 11,660,646       1,845,960    7,418,688      3,962,930     131,843      2,077,241         277,535       200,860 525,922 8,582,306 1,773,500
 𝑑 Feats       0              0             0             0           100           100              1536         1536     1536     768       768
 Datasets Soc-Epinions1 Email-Enron Web-Stanford RoadNet-PA P2P-Gnutella06 Citation-2019 Citation-20Century Arxiv Arxiv-t Photo                 CS
 # Nodes      75,879      36,692      281,903     1,088,092     8,717         765,658         1,016,241     169,343 169343     48,362         18,333
 # Edges     508,837      183,831    2,312,497    1,541,898     31,525       1,917,381        5,565,798    1,166,243 1,166,243 500,939        163,788
  𝑑 Fets         0           0           0            0          128            128              128          128       768      768           6805


A.3     Hyperparameter Settings                                                    distinguishing between non-isomorphic graphs.
Optimization. Our model, AnyGraph, is implemented using Py-                      Graph Pre-training Models.
Torch. The optimization process employs the Adam optimizer with                  • GraphCL [39]. It enhances the pre-training of graph models
a learning rate of 1 × 10 −4 and a training batch size of 4096. We use             via self-discriminative contrastive learning, which is applied to
cross-entropy loss with a sampled negative set [29]. The learnable                 learned node embeddings. The method employs various graph
parameters of AnyGraph are initialized using the Xavier uniform                    augmentation techniques such as node dropping, edge permuta-
initializer. Network Configurations. The standard configuration                    tion, random walks, and feature masking to improve robustness.
of our AnyGraph includes 512 hidden units and 8 graph expert mod-                Graph Prompt Tuning Methods.
els. Each expert model comprises 8 fully-connected layers. These
                                                                                 • GraphPrompt [19]. GraphPrompt proposes a unified approach
layers utilize a ReLU activation function and incorporate a dropout
                                                                                   that integrates pre-training and prompt tuning for graph models.
layer with a dropout probability of 0.1. Algorithm Hyperparam-
                                                                                   It features a learnable prompt layer designed to automatically ex-
eters. The frequency regularization of our routing mechanism is
                                                                                   tract crucial information from the pre-trained model to enhance
set with an adjustment range of 𝜌 = 0.2. The SVD decomposition is
                                                                                   performance on downstream tasks.
performed using 2 iterations. For structural and feature augmenta-
tion, each dataset is reprojected after using 1/10 of its samples for            • GPF [6]. The Graph Prompt Framework (GPF) is a versatile graph
optimization. A minimum of 100 training steps should be executed                   prompt tuning framework compatible with various graph pre-
for each dataset before its initial representations are reprojected.               training methods. It offers two variants of a learnable graph
The reassignment of experts occurs after all training datasets have                prompt layer, tailored to different application needs.
undergone one cycle of re-projection.
   The baseline methods are evaluated using theeir original code                 A.5    Details of the Scaling Law Experiment
or released model. We closely follow the original code to adapt to               For the scaling law experiment (RQ2), we elaborate the configura-
our experiments. Grid search is conducted to search for the best                 tions of the developed instances of AnyGraph. For AnyGraph with
hyperparameter settings for each baseline method.                                different model sizes, we begin with the smallest model which has
                                                                                 64 hidden units, 1 fully-connected layer, and 1 expert model. The
A.4     Baseline Methods                                                         subsequent 3 model instances increases in their hidden dimension-
This section provides detailed descriptions of the baseline models               ality, from 64 to 128, 256, and 512. Then 3 larger models with more
used in our analysis. We employ seven different baseline models                  fully-connected layers are utilized, respectively containing 2, 4, and
across four distinct categories.                                                 8 MLP layers. Then we have MoE versions of AnyGraph, with 2, 4,
Training-from-scratch Graph Neural Networks.                                     and 8 experts, respectively. The final largest instance of AnyGraph
                                                                                 has a larger latent dimensionality of 1024.
• GAT [25]. Graph Attention Networks (GAT) leverage an atten-
                                                                                    For the increase of training data, we begin with a subset of
  tion mechanism to dynamically weight node-to-node connec-
                                                                                 Link2 data including Cora and CS. The next version additionally in-
  tions, enhancing the model’s ability to adaptively propagate and
                                                                                 cludes Photo. The thir one includes ML1M. The fourth one includes
  aggregate information across the graph.
                                                                                 Gowalla. The fifth one additionally include Arxiv and Arxiv-t. The
• GIN [33]. The Graph Isomorphism Network (GIN) significantly                    sixth one adds the following datasets: collab, ddi, Yelp2018, Fitness,
  boosts the expressive power of Graph Neural Networks by intro-                 proteins-spec1, web-Stanford, proteins-spec3. The seventh one is
  ducing a unique graph encoding technique aimed at effectively                  trained with proteins-2, roadNet-PA, and Fitness additionally. And
                                                                                 the final one is trained with all datasets from Link2.

