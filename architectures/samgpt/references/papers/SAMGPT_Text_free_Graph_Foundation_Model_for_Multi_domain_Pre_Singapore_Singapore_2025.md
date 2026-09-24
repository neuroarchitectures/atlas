# SAMGPT Text free Graph Foundation Model for Multi domain Pre Singapore Singapore 2025

> Source: `SAMGPT_Text_free_Graph_Foundation_Model_for_Multi_domain_Pre_Singapore_Singapore_2025.pdf`

---

                                          SAMGPT: Text-free Graph Foundation Model for Multi-domain
                                                  Pre-training and Cross-domain Adaptation
                                                              Xingtong Yu                                                  Zechuan Gong                                      Chang Zhou
                                               Singapore Management University                              University of Science and Technology                University of Science and Technology
                                                     Singapore, Singapore                                                  of China                                            of China
                                                   xingtongyu@smu.edu.sg                                                Hefei, China                                        Hefei, China
                                                                                                              gongzechuan@mail.ustc.edu.cn                       zhouchang21sy@mail.ustc.edu.cn

                                                                                              Yuan Fang∗                                             Hui Zhang∗
                                                                              Singapore Management University                           University of Science and Technology




arXiv:2502.05424v1 [cs.CL] 8 Feb 2025
                                                                                    Singapore, Singapore                                               of China
                                                                                     yfang@smu.edu.sg                                                Hefei, China
                                                                                                                                                  fzhh@ustc.edu.cn
                                        Abstract                                                                                         Keywords
                                        Graphs are able to model interconnected entities in many online                                  Graph learning, foundation models, multi-domain pre-training,
                                        services, supporting a wide range of applications on the Web. This                               prompt learning, few-shot learning.
                                        raises an important question: How can we train a graph founda-
                                                                                                                                         ACM Reference Format:
                                        tional model on multiple source domains and adapt to an unseen
                                                                                                                                         Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang∗ , and Hui Zhang∗ .
                                        target domain? A major obstacle is that graphs from different do-
                                                                                                                                         2025. SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-
                                        mains often exhibit divergent characteristics. Some studies leverage                             training and Cross-domain Adaptation. In Proceedings of the ACM Web
                                        large language models to align multiple domains based on textual                                 Conference 2025 (WWW ’25), April 28–May 2, 2025, Sydney, NSW, Australia.
                                        descriptions associated with the graphs, limiting their applicability                            ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3696410.3714828
                                        to text-attributed graphs. For text-free graphs, a few recent works at-
                                        tempt to align different feature distributions across domains, while
                                        generally neglecting structural differences. In this work, we propose                            1    Introduction
                                        a novel Structure Alignment framework for text-free Multi-domain                                 How to build foundation models has emerged as an important ques-
                                        Graph Pre-Training and cross-domain adaptation (SAMGPT). It                                      tion, paving a plausible path toward artificial general intelligence.
                                        is designed to learn multi-domain knowledge from graphs origi-                                   In natural language processing, recent works [1, 42] have demon-
                                        nating in multiple source domains, which can then be adapted to                                  strated the capabilities of universal foundation models. They are
                                        address applications in an unseen target domain. Specifically, we                                trained on a wide variety of data from multiple domains, and can
                                        introduce a set of structure tokens to harmonize structure-based                                 be further adapted to solve a diverse range of tasks. Other than
                                        aggregation across source domains during the pre-training phase.                                 natural languages, the World Wide Web has become a vast knowl-
                                        Next, for cross-domain adaptation, we design dual prompts, namely,                               edge repository, connecting an enormous amount of entities to
                                        holistic prompts and specific prompts, which adapt unified multi-                                form extensive and complex graphs. These graphs enable diverse
                                        domain structural knowledge and fine-grained, domain-specific                                    Web applications, including social network analysis [13, 27], Web
                                        information, respectively, to a target domain. Finally, we conduct                               mining [2, 5], and recommendation systems [23, 28]. Given the
                                        comprehensive experiments on seven public datasets to evaluate                                   rich graph data on the Web, can we build a universal graph model
                                        and analyze the effectiveness of SAMGPT.                                                         based on multi-domain graphs, to address various downstream
                                                                                                                                         graph-centric applications [19]?
                                        CCS Concepts                                                                                        Traditional supervised graph learning struggles to build uni-
                                        • Information systems → Web mining; Data mining; • Com-                                          versal models. These approaches require retraining a new graph
                                        puting methodologies → Learning latent representations.                                          neural network (GNN) [6, 15, 60] or graph transformer [32, 56, 66]
                                                                                                                                         for each new task, relying on abundant task-specific labeled data.
                                        ∗ Corresponding authors.
                                                                                                                                         In contrast, more recent graph pre-training methods [11, 30, 44]
                                                                                                                                         attempt to learn universal properties from unlabeled graphs in
                                        Permission to make digital or hard copies of all or part of this work for personal or
                                        classroom use is granted without fee provided that copies are not made or distributed            a self-supervised manner, which can be subsequently adapted to
                                        for profit or commercial advantage and that copies bear this notice and the full citation        a downstream task with some task-specific labels through fine-
                                        on the first page. Copyrights for components of this work owned by others than the               tuning [14, 30, 44] or prompt learning [21, 40, 62]. However, in
                                        author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
                                        republish, to post on servers or to redistribute to lists, requires prior specific permission    most existing graph pre-training approaches, the pre-training and
                                        and/or a fee. Request permissions from permissions@acm.org.                                      downstream graphs originate from the same dataset [21, 40, 44, 57],
                                        WWW ’25, Sydney, NSW, Australia.                                                                 a practice we refer to as single-domain methods, which fall short
                                        © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
                                        ACM ISBN 979-8-4007-1274-6/25/04                                                                 of building a universal, multi-domain graph model from multiple
                                        https://doi.org/10.1145/3696410.3714828                                                          graph datasets.
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                               Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang


                Source                      Target                           Second, how do we adapt multi-domain prior structural knowledge
  Domain1     Domain2    … Domain      K   DomainK+1     Tuned   Frozen
                                                                          to cross-domain downstream tasks? Multi-domain prior knowledge
                                                                          includes not only holistic knowledge across source domains, but
                                                                          also domain-specific knowledge from each domain. Therefore, in
                                                                          SAMGPT, we propose dual structural prompts, comprising a set
      Dimension           Feature          Dimension        Feature
                         alignment                         adaptation
                                                                          of holistic prompts and a set of specific prompts, thus facilitating
      alignment                            alignment
                                                                          the adaptation of both holistic and domain-specific knowledge to
                                                                          downstream tasks, as illustrated in Fig. 1(b). On one hand, the
                           Graph                          Pre-trained     holistic prompts consist of learnable vectors that holistically align
Tokens1
                          Encoder            Holistic    Graph Encoder    the target domain’s structural characteristics with the unified pre-
Tokens2
                                             prompts                      trained knowledge from all source domains. On the other hand,
            …           Pre-training                                      specific prompts integrate multi-domain structure tokens in a learn-
TokensK                     loss                          Downstream
                                             Specific        loss         able mixture to align the target domain with knowledge from each
                             Fuse            prompts                      source domain, capturing domain-specific structural information
 Structure tokens
 (a) Multi-domain pre-training             (b) Cross-domain adaptation    for finer-grained adaptation.
                                                                          Contributions. In summary, we make the following contributions
                 Figure 1: Motivation of SAMGPT.                          in this work. (1) We propose SAMGPT, a text-free graph foun-
                                                                          dation model with structure alignment for multi-domain graph
                                                                          pre-training and cross-domain adaptation. (2) For pre-training, we
Research problem. Thus, it is crucial to pre-train a graph model
                                                                          propose structure tokens to align structural distributions across
on a wide range of multi-domain (i.e., multi-dataset) graphs and
                                                                          domains, training a universal foundation model with multi-domain
achieve cross-domain adaptation. However, graph structures from
                                                                          graphs. (3) For downstream adaptation, we propose a dual-prompt
different datasets often exhibit markedly distinct characteristics.
                                                                          strategy, using holistic prompts to leverage holistic prior structural
For instance, the structural patterns in a social network might not
                                                                          knowledge and specific prompts to facilitate finer-grained, domain-
be directly applicable to a citation or e-commerce graph. Such diver-
                                                                          specific structural adaptation. (4) We conduct extensive experiments
sity poses significant challenges in integrating graphs from multiple
                                                                          on seven benchmark datasets, demonstrating the superior perfor-
domains and adapting prior knowledge to different domains. Al-
                                                                          mance of SAMGPT compared to state-of-the-art methods.
though some studies have explored cross-domain adaptation from a
single source domain [4, 9, 45, 47, 55], they do not exploit multiple     2   Related Work
source domains. Another line of work [18, 41, 53] employs large
                                                                          We review related literature on pre-training, cross-domain transfer
language models to extract and utilize multi-domain knowledge
                                                                          learning, and multi-domain pre-training for graph data.
based on textual descriptions associated with graphs, using text as
a universal medium to bridge different domains. However, this lim-        Graph pre-training. Graph pre-training methods aim to extract
its their applicability to text-attributed graphs [49, 70] and cannot     inherent properties of graphs, often utilizing self-supervised learn-
be extended to general graphs without textual descriptions. Few           ing approaches, which can be either generative [10–12, 16] or
recent studies [65, 69] have explored multi-domain pre-training on        contrastive [17, 44, 52, 54]. The pre-trained model is then em-
text-free graphs, but they focus on aligning the divergent feature        ployed to address downstream tasks through fine-tuning [30, 44, 57]
spaces and homophily patterns across multi-domain graphs, while           or parameter-efficient adaptation methods, notably prompt-based
overlooking the structural differences across domains.                    learning [7, 21, 39, 59]. However, these methods typically assume
                                                                          that the pre-training and downstream graphs originate from the
Challenges and insights. In this paper, we propose SAMGPT, a
                                                                          same domain, such as different subgraphs of a large graph [57, 61]
graph foundation model with Structural Alignment for text-free
                                                                          or collections of similar graphs within the same dataset [11, 30],
Multi-domain Graph Pre-Training, to facilitate cross-domain adap-
                                                                          failing to account for multiple domains in either pre-training or
tation. This is non-trivial due to two key challenges.
                                                                          downstream graphs.
    First, how do we harmonize structural variance across multiple
domains during pre-training? Graphs from different domains often          Graph cross-domain transfer. This line of work aims to transfer
exhibit distinct structural and topological characteristics, such as      single-source domain knowledge to a different target domain by
average node degree, shortest path length and clustering coeffi-          leveraging domain-invariant properties across domains [4, 9, 45, 47].
cient, as depicted in Table 1. Thus, merging multi-domain graphs          However, they rely exclusively on a single source domain, failing to
without structure alignment during pre-training may cause inter-          harness the extensive knowledge available across multiple domains.
ference rather than synergy, leading to suboptimal performance. In        Additionally, these approaches are often tailored to specific tasks
SAMGPT, we propose structure tokens to align structural distribu-         or domains [4, 9, 45, 47], limiting their generalization.
tions across multiple domains, as shown in Fig. 1(a). Each domain         Multi-domain graph pre-training. In the context of graphs from
is equipped with a series of structure tokens, which modify the           multiple domains, recent works [18, 41, 53] utilize large language
structure-based aggregation in each layer of the graph encoder.           models to align node features from different domains through
These tokens are learnable vectors that capture domain-specific           textual descriptions, thereby limiting their applicability to text-
structural patterns, enabling the model to accommodate the unique         attributed graphs [50, 67, 70]. For graphs without textual attributes,
structural characteristics of each domain during pre-training.            GraphControl [72] applies ControlNet [68] to incorporate target
SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation       WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


                                                                                                                        ˜
domain node features with the pre-trained model, while neglect-                     where DAL𝑆𝑖 : R |𝑉 | ×𝑑𝑆𝑖 → R |𝑉 | ×𝑑 is the dimension alignment func-
ing the alignment among multiple source domains. Another recent                     tion for domain 𝐷𝑆𝑖 , transforming the original dimension 𝑑𝑆𝑖 to a
study proposes GCOPE [69], which employs domain-specific virtual                    common dimension 𝑑˜ across domains. We implement DAL as sin-
nodes that interconnect nodes across domains, facilitating the align-               gular value decomposition [37] following prior art [58, 69]. Next,
ment of feature distribution and homophily patterns. Meanwhile,                     given the source-domain graphs G𝑆 with their dimension-aligned
MDGPT [65] pre-trains domain-specific tokens to align feature                       features X̃𝑆 = {X̃𝑖 : 𝐺𝑖 ∈ G𝑆 }, we further align the features to unify
semantics across various domains. However, these studies do not                     their semantic space across various domains. Letting FAL denote
account for structural variance across different domains, hindering                 the feature alignment procedure, we pre-train a graph encoder with
their effectiveness in integrating multi-domain knowledge. On a                     feature alignment:
related front, multi-task pre-training techniques [48, 64] employ
pretext tokens for each pre-training task. It is important to note                                    HFAL = GE(FAL(G𝑆 , X̃𝑆 ; Ψ); Θ),                      (4)
that they address a distinct problem, aiming to overcome potential                  where Ψ denotes the learnable parameters in FAL, and HFAL is the
interference among multiple tasks within a single domain, rather                    output node embedding matrix with feature alignment. While any
than interference across multiple domains.                                          feature alignment model can be employed [58, 69], we follow the
                                                                                    work of Yu et al. [58] due to its superior performance.
3    Preliminaries                                                                  Cross-domain task with feature adaptation. For each down-
In this section, we provide technical background, and outline the                   stream task, consider a set of graphs G𝑇 belonging to a target
scope of our work.                                                                  domain 𝐷𝑇 . The task is cross-domain if the target domain is unseen
Graph encoder. A graph is defined as 𝐺 = (𝑉 , 𝐸, X), where 𝑉 is                     during pre-training, i.e., ∀𝑖 𝐷𝑇 ≠ 𝐷𝑆𝑖 . Again, since the target do-
the set of nodes, 𝐸 is the set of edges, and X ∈ R |𝑉 | ×𝑑 is the node              main may exhibit different feature characteristics from the source
feature matrix with each row x𝑖 representing the feature vector of                  domains, previous works [65, 69] have proposed feature adaptation
node 𝑣𝑖 ∈ 𝑉 . A collection of graphs is denoted as G.                               strategies to transfer prior multi-domain knowledge to the target
   Message-passing GNNs are a common choice for encoding graph                      domain, which can be directly integrated into our work. Specifi-
representations [51]. Specifically, each node updates its embedding                 cally, we first employ the same dimension alignment method used
by receiving and aggregating features or embeddings from its neigh-                 in the pre-training phase, transforming the feature matrix of a
bors. By stacking such message-passing layers, information can                      downstream graph 𝐺 = (𝑉 , 𝐸, X) ∈ G𝑇 to X̃ = DAL𝑇 (X). We then
propagate recursively throughout the graph. Therefore, the node                     employ a feature adaptation technique FAD to adapt the pre-trained
embeddings are encoded based on both input features and graph                       model to the target domain, as follow.
structure. Let us denote the embedding of node 𝑣 at the 𝑙-th layer
                                                                                                      HFAD = GE(FAD(𝐺, X̃; Γ); Θpre ),                      (5)
as h𝑙𝑣 , which is derived from the features or embeddings in the
preceding layer as follows.                                                         where Γ denotes the learnable parameters in FAD, and Θpre is the
                                                                                    pre-trained weights in graph encoder GE. Here we implement FAD
                h𝑙𝑣 = Aggr(h𝑙𝑣−1, {h𝑢𝑙 −1 : 𝑢 ∈ N𝑣 }; 𝜃 𝑙 ),             (1)        following Yu et al. [58], which is paired with the feature alignment
                                                                                    method in pre-training.
where N𝑣 denotes the set of neighboring nodes of 𝑣, 𝜃 𝑙 represents
                                                                                    Our scope: Few-shot classification. For the downstream applica-
the learnable parameters in layer 𝑙, and Aggr(·) stands for the
                                                                                    tions, we aim to solve few-shot node and graph classification tasks.
neighborhood aggregation function. In the first layer, the node
                                                                                    For node classification, given a graph 𝐺 = (𝑉 , 𝐸, X) ∈ G𝑇 , each
embedding h0𝑣 is initialized as the input feature vector x𝑣 . We denote
                                                                                    node 𝑣 ∈ 𝑉 is associated with a label 𝑦 ∈ 𝑌 , where 𝑌 denotes the
the output node embedding after the last layer as h𝑣 , which is a row
                                                                                    set of node classes. For graph classification over a set of graphs
in the node embedding matrix H. Overall, the multi-layer message-
                                                                                    G𝑇 , each graph 𝐺 ∈ G𝑇 is associated with a label 𝑦 ∈ 𝑌 , where 𝑌
passing process can be abstracted as a graph encoder, as follows.
                                                                                    denotes the set of graph classes. An 𝑚-shot classification task con-
                            H = GE(𝐺, X; Θ),                             (2)        sists of only 𝑚 labeled examples per class, along with an arbitrary
                                                                                    number of unlabeled examples for testing.
where GE denotes a graph encoder, Θ = {𝜃 1, 𝜃 2, . . .} is the full set                In particular, we focus on low-shot settings, where 𝑚 is a small
of trainable parameters for the graph encoder.                                      number (e.g., 𝑚 ≤ 5), reflecting real-world scenarios where labeled
Multi-domain pre-training with feature alignment. Consider                          data are expensive or difficult to obtain. Due to the parameter-
a set of unlabeled graphs G𝑆 = {𝐺 1, 𝐺 2, . . . , 𝐺𝐾 } for pre-training,            efficient nature of prompt learning, many previous methods for
where each graph 𝐺𝑖 belongs to a specific source domain 𝐷𝑆𝑖 ∈ D𝑆 .                  prompt learning on graphs [21, 40, 59, 65, 69] also emphasize this
Thus, we have graph-domain pairs {(𝐺𝑖 , 𝐷𝑆𝑖 ) : 𝑖 ∈ {1, 2, . . . , 𝐾 }}.            setting. It is expected that, as more task-specific labeled data become
   As different domains exhibit distinct feature distributions, pre-                available, conventional fine-tuning or supervised approaches may
vious works [58, 69] have proposed solutions to align feature di-                   become sufficient.
mensions and semantics, which can be directly employed in our
work. Given a graph 𝐺𝑖 = (𝑉𝑖 , 𝐸𝑖 , X𝑖 ) from the source domain 𝐷𝑆𝑖 ,               4    Proposed Approach: SAMGPT
we first align the dimensions of its feature matrix:                                In this section, we present SAMGPT, beginning with an overview
                                                                                    and then delving into the details of multi-domain pre-training and
                             X̃𝑖 = DAL𝑆𝑖 (X𝑖 ),                          (3)        cross-domain adaptation.
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                             Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang



                            Dimension                                                        Dimension
                            alignment                                                        alignment

                                                                         Feature                                                                      Feature
                                   Layer 1 (l1)   Layer 2 (l2)          alignment                                   Layer 1      Layer 2
                                                                                                                                                     adaptation

                                                                            ⊕                                                               …
                                                                                            Target domain
                                                                        Pre-training
            …
                                                                 …          loss
                                                                                                                    Layer 1      Layer 2
                                                                                                                                                  ⊕ ⊕
   Source domains                                                                Fuse           l 1 l2
                                                                                                         …
                                                                                           Specific prompts                                          Downstream
               l1 l 2                                                          l1 l2                                                         …
   Domain1              …                                            Domain1           …                                                                loss
   Domain2              …               …                …           Domain2           …        l1 l2
                                                                                                         …                                               Tuned
   Domain3              …                 Graph encoder              Domain3           …                           Pre-trained graph encoder
       …                                                              …                     Holistic prompts                                             Frozen
      Structure tokens                  Structure alignment               Pre-trained
                                                                       structure tokens                              Structure adaptation
                 (a) Multi-domain pre-training                                                                   (b) Cross-domain adaptation

                                                         Figure 2: Overall framework of SAMGPT.


4.1      Overall Framework                                                             For example, the first layer aggregates one-hop neighborhood in-
SAMGPT consists of two phases: multi-domain pre-training, and                          formation, while the second layer incorporates a broader two-hop
cross-domain adaptation, as shown in Fig. 2.                                           neighborhoods. These layer-wise structural patterns may vary sig-
    In the pre-training phase, as depicted in Fig. 2(a), we first align                nificantly across domains.
the feature distributions from multiple source domains following                          Therefore, to unify the structural characteristics in multiple
previous work [65, 69]. Next, we introduce a set of structure tokens                   source domains, we introduce learnable structure tokens. For each
designed to align the structural distributions across diverse domains.                 domain 𝐷𝑆𝑖 , we inject a series of structure tokens T𝑆𝑖 = {t𝑙𝑆 : 𝑙 ∈
                                                                                                                                                     𝑖
These tokens are domain-specific and are integrated into each layer                    {1, . . . , 𝐿}} into the graph encoder, where 𝐿 denotes the number
of the graph encoder, modifying the structure-based aggregation at                     of layers. Specifically, when encoding the graph 𝐺𝑖 = (𝑉𝑖 , 𝐸𝑖 , X̃𝑖 )
each layer. Finally, the structure token-enhanced graph encoder is                     in 𝐷𝑆𝑖 , we assign structure token t𝑙𝑆 to the 𝑙-th layer, guiding
                                                                                                                                𝑖
pre-trained using a self-supervised loss [21].                                         structure-based aggregation:
    In the adaptation phase, as shown in Fig. 2(b), we first align the
                                                                                              h𝑙𝑣 = Aggr(h𝑙𝑣−1, {t𝑙𝑆𝑖 ⊙ h𝑢𝑙 −1 : 𝑢 ∈ N𝑣 }; 𝜃 𝑙 ), ∀𝑣 ∈ 𝑉𝑖 ,    (6)
feature dimension of the target domain with that of the source
domains. Then, we introduce dual prompts. The first type, holis-                       where ⊙ represents element-wise multiplication. Note that the
tic prompts, are learnable vectors that integrate the target domain                    graph encoders for feature alignment and structure alignment on all
with the holistic structural knowledge from all source domains.                        graphs share the same parameters Θ. Let H𝑖SAL denote the structure-
The second type, specific prompts, comprise learnable mixtures of                      aligned output node embedding matrix for 𝐺𝑖 in 𝐷𝑆𝑖 , following the
pre-trained structure tokens that incorporate domain-specific topo-                    aggregation in Eq. (6). In general, each source domain is attached
logical information tailored to the target domain. These prompts are                   with its own set of structure tokens, which are applied to modify
applied to each layer of the graph encoder to adjust the structure-                    the aggregation on the graph in the corresponding domain. By
based aggregation, while keeping the pre-trained weights of the                        stacking the structure-aligned output matrix across graphs in all
graph encoder frozen.                                                                  domains, we obtain the overall structure-aligned embedding matrix,
                                                                                       HSAL = Stack(HSAL             SAL
                                                                                                        1 , . . . , H𝐾 ).
                                                                                          Finally, we fuse H SAL    with HFAL in Eq. (4) to obtain the multi-
4.2      Multi-domain Graph Pre-training with
                                                                                       domain node embedding matrix H, incorporating both feature and
         Structure Alignment                                                           structure alignment, as shown below.
As defined in Sect. 3, we are given a set of pre-training graphs from
multiple source domains, G𝑆 . As both the features and structures of                                            HAL = HFAL + 𝛼HSAL,                            (7)
these domains can exhibit divergent distributions, effective integra-                  where 𝛼 > 0 is a hyperparameter.
tion of these multi-domain graphs requires aligning both. As our                       Pre-training loss. We leverage a universal task template based
work focuses on structure alignment, we follow previous feature                        on subgraph similarity calculation [21, 59], which ensures compati-
alignment methods [58, 69], as outlined in the preliminaries.                          bility across different tasks such as node classification and graph
Structure alignment. Recall that in the graph encoder, node repre-                     classification. As demonstrated in GraphPrompt+ [59], prevailing
sentations are updated layer-wise through a structure-based aggre-                     contrastive pre-training objectives can be unified under this tem-
gation. Each layer captures different levels of structural information.                plate, making them suitable choices for the pre-training loss in
SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation         WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


SAMGPT. In general, we can adopt the following form of con-                         tokens in the corresponding layer across all source domains 𝐷𝑆𝑖 ∈
trastive loss in pre-training.                                                      D𝑆 . Formally, we define
                                      Í                                                                            Í𝐾 𝑙 𝑙
                                               exp(sim(h ,h )/𝜏 )                                           p𝑙spe = 𝑖=1 𝜆𝑖 t𝑆 ,                  (10)
      Lpre (O; Θ, T , Ψ) = − 𝑜 ∈ O ln Í 𝑎∈Pos𝑜 exp(sim(h𝑎 ,h𝑜 )/𝜏 ) , (8)
                             Í
                                                                                                                                   𝑖
                                              𝑏 ∈Neg𝑜        𝑏   𝑜
                                                                                    where Λ𝑙 = {𝜆𝑙1, . . . , 𝜆𝑙𝐾 } are learnable coefficients. Thus, the full set
where O denotes the set of observed graph element in pre-training,
𝑎 ∈ pos𝑜 , 𝑏 ∈ neg𝑜 represent the positive or negative instance                     of learnable parameters for the specific prompts is Λ = {Λ1, . . . , Λ𝐿 }.
of 𝑜, respectively, and h𝑜 , h𝑎 , h𝑏 are their corresponding embed-                 Subsequently, specific prompts modify the structure-based aggre-
dings. Furthermore, sim(·, ·) is a similarity function, such as cosine              gation in the same way as in Eq. (9), while freezing the pre-trained
similarity [31] in our implementation, and 𝜏 > 0 is a temperature hy-               weights of the graph encoder. Similarly, we denote the output node
perparameter. Note that SAMGPT is flexible in the materialization                   embedding matrix based on the specific prompts as Hspe .
of 𝑜, 𝑎, 𝑏 to realize different contrastive losses [59]. Our experiments            Prompt tuning. To leverage both holistic multi-domain and domain-
adopt GraphCL [57], where 𝑜 is the original graph 𝐺, and 𝑎, 𝑏 rep-                  specific structural knowledge from the pre-trained model, we fuse
resent two different augmentations of 𝐺. Hence, h𝑜 , h𝑎 , h𝑏 are the                the output embedding matrices obtained via holistic prompts and
corresponding graph embeddings, which can be obtained through                       specific prompts as follows.
a readout operation [21] on the aligned node embeddings in HAL .                                              HSAD = Hhol + 𝛽Hspe,                           (11)
   The pre-training loss is optimized by updating the weights of
graph encoder Θ, structure tokens across all source domains T =                     where 𝛽 > 0 is a hyperparameter. Further incorporating feature
{T𝑆 1 , . . . , T𝑆𝐾 }, and feature alignment parameters Ψ.                          adaptation in Eq. (5), we obtain the overall node embedding matrix
                                                                                    with both feature and structure adaptations, given by
4.3    Cross-domain Structure Adaptation                                                                      HAD = HFAD + 𝛼HSAD .                           (12)
Beyond multi-domain pre-training, another challenge lies in cross-                  Here, 𝛼 is the same hyperparameter used in Eq. (7), as both share
domain adaptation. Given a model pre-trained on graphs G𝑆 from                      the objective of integrating the feature and structure counterparts.
source domains D𝑆 , we aim to adapt it to a downstream task on                         For downstream node and graph classification tasks, the loss
graphs G𝑇 from a target domain 𝐷𝑇 ∉ D𝑆 . As this work focuses                       function Ldown is formulated based on the same task template
on structure adaptation, we directly apply previous work [58] for                   with subgraph similarity [21], akin to the pre-training loss Lpre .
feature adaptation, as outlined in Sect. 3.                                         Let Ω = {(𝑥 1, 𝑦1 ), (𝑥 2, 𝑦2 ), . . .} represent the labeled training set,
   For structure adaptation, we propose dual prompts, consisting                    where each 𝑥𝑖 is either a node or graph instance, and 𝑦𝑖 ∈ 𝑌 is
of holistic prompts and specific prompts. On one hand, the holistic                 its respective class from the set 𝑌 . Subsequently, we optimize the
prompts are designed to holistically utilize the pre-trained struc-                 following cross-domain adaptation loss:
tural knowledge from all source domains. On the other hand, the                                                                         exp(sim(h𝑥𝑖 ,h𝑦𝑖 )/𝜏 )
                                                                                      Ldown (Ω; Phol, Λ, Γ) = − (𝑥𝑖 ,𝑦𝑖 ) ∈Ω ln Í exp(sim(h
                                                                                                                     Í
specific prompts combine multi-domain structure tokens through a                                                                       𝑦 ∈𝑌         𝑥𝑖  ,h )/𝜏 ) .
                                                                                                                                                         𝑦
learnable mixture, adapting fine-grained, domain-specific structural                                                                                         (13)
knowledge to the target domain.
                                                                                    Here, h𝑥𝑖 represents the adapted embedding of the node or graph
Holistic prompts. To transfer the holistic multi-domain struc-                      𝑥𝑖 based on HAD , where a readout operation on HAD is required if
tural knowledge to a downstream task, we propose a set of holistic                  𝑥𝑖 is a graph. Additionally, h𝑦 denotes the prototype embedding
prompts designed to align the target domain 𝐷𝑇 with the model                       for class 𝑦, which is calculated as the average embeddings of all
pre-trained on the source domains D𝑆 . Like any pre-training frame-                 training instances of class 𝑦.
work, we encode a downstream graph 𝐺 = (𝑉 , 𝐸, X̃) using the                           We outline the key steps for prompt tuning in Algorithm 1,
pre-trained graph encoder with frozen layer-wise weights Θpre =                     Appendix A and assess its complexity in Appendix B.
   1 , . . . , 𝜃 𝐿 }. However, the key difference is that we inject a se-
{𝜃 pre           pre
ries of learnable vectors Phol = {p1hol, . . . , p𝐿hol } as holistic prompts        5     Experiments
into the downstream structure-based aggregation:                                    In this section, we conduct experiments to assess the performance
                                                                                    of SAMGPT and analyze its empirical results.
      h𝑙𝑣 = Aggr(h𝑙𝑣−1, {p𝑙hol ⊙ h𝑢𝑙 −1 : 𝑢 ∈ N𝑣 }; 𝜃 pre
                                                      𝑙
                                                          ), ∀𝑣 ∈ 𝑉 .    (9)
The final layer outputs a holistic node embedding matrix for the                    5.1     Experimental Setup
downstream graph 𝐺, denoted as Hhol .                                               Datasets. We conduct experiments on seven benchmark datasets.
Specific prompts. In contrast to the holistic prompts, specific                     (1) Cora [26], (2) Citeseer [35] and (3) Pubmed [35] are scientific
prompts are designed to adapt structural knowledge specific to each                 paper citation networks from different fields, including computer
source domain. Since knowledge from related source domains is                       science and biomedical research. Nodes represent academic publi-
likely to be more applicable, it is essential to align the target domain            cations and edges denote citation relationships. (4) Photo [36] and
with different source domains to varying extents, prioritizing the                  (5) Computers [25] are both e-commerce networks from Amazon
most relevant ones. Consequently, we define specific prompts as                     in different categories, namely, photography and computer related
Pspe = {p1spe, . . . , p𝐿spe }, which will also be injected into different          products. Nodes represent products and edges signify frequent
layers of the pre-trained graph encoder. Specifically, in the 𝑙-th layer,           co-purchases between products. (6) Facebook [33] is a Web graph,
p𝑙spe is a combination of {t𝑙𝑆 , . . . , t𝑙𝑆 }, the pre-trained structure           where nodes represent official Facebook pages while the links are
                                 1        𝐾
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                  Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang


                     Table 1: Summary of datasets.                                           to unify self-supervised pre-training and downstream tasks, and
                                                                                             tune a single prompt on downstream tasks.
                                        Feature         Node      Avg.     Avg.     Avg.        (3) Graph cross-domain models: Hassani [9] pre-trains a GNN
                Nodes       Edges                                                            on a single source domain by incorporating both contextual and
                                       dimension       classes     nd       spl      cc
   Cora          2,708      10,556           1,433           7     3.89     6.30    0.24     topological views, which facilitates cross-domain adaptation for
  Citeseer       3,327       9,104           3,703           6     2.73     9.31    0.14     downstream tasks.
  Pubmed        19,717      88,648             500           3     4.49     6.33    0.06        (4) Multi-domain pre-training models: GCOPE [69] performs
   Photo         7,650     238,162             745           8    31.13     4.05    0.40     multi-domain pre-training, and subsequently adapts to cross-domain
Computers       13,752     491,722             767          10    35.75     3.38    0.34     tasks through either fine-tuning a classification head or prompt
 Facebook       22,470     342,004             128           4    15.22     4.97    0.35     tuning. We opt for fine-tuning as it yields superior performance.
  LastFM         7,624      55,612             128          18     7.29     5.23    0.21
                                                                                                Note that the above graph pre-training and cross-domain ap-
nd: node degree, spl: shortest path length [3], cc: clustering coefficient [8].
                                                                                             proaches are originally designed for pre-training on a single source
                                                                                             domain. For a fair comparison, we directly merge the multi-domain
mutual likes between these pages. (7) LastFM [34] is a social net-                           graphs and apply dimension alignment for them, as in SAMGPT.
work, where nodes denote users and edges represent interactions                              Further descriptions of the baselines are provided in Appendix D,
such as follower relationships. Note that each domain comprises a                            with implementation details in Appendix E.
single graph. We summary these datasets in Table 1 and present
additional details in Appendix C.
                                                                                             5.2    Performance Evaluation
Setup of pre-training and downstream tasks. Following pre-
vious work [65, 69], we treat each dataset as a distinct domain.                             We first compare SAMGPT and the baseline methods on one-shot
Among the seven datasets (or domains), we use each of them as                                node and graph classification tasks, and then investigate the effect
the target domain while leveraging the remaining six as source                               of increasing the number of shots.
domains for pre-training.                                                                    One-shot performance. Tables 2 and 3 show the results of one-
   On each target domain, we conduct 𝑚-shot node classification                              shot node and graph classification tasks. We observe that, first,
and graph classification, where 𝑚 labeled nodes or graphs per class                          SAMGPT achieves outstanding performance in both node and graph
are randomly selected for downstream prompt tuning. Given that                               classification across various target domains, demonstrating the ef-
each dataset comprises a single graph, performing graph classifica-                          fectiveness of our proposed structure tokens in multi-domain pre-
tion on whole graphs is not feasible. Therefore, following previous                          training and dual prompts in cross-domain adaptation. We defer
works [22, 61, 63], we generate a series of graphs by constructing                           analysis of the quantitative contributions of these components to
ego-networks centered on the labeled nodes within each dataset,                              the ablation studies in Sect. 5.3. Second, another text-free multi-
and set up graph classification on these ego-networks, with each                             domain pre-training method, GCOPE, significantly lags behind
network labeled according to its central node. Note that the graph                           SAMGPT because it only performs alignment and adaptation on
encoder is pre-trained only once for each set of source domains,                             feature and homophily patterns, without accounting for structural
and subsequently utilized across all downstream tasks. We generate                           differences across domains. This further emphasizes the importance
100 𝑚-shot tasks for both node classification and graph classifica-                          of our structure tokens and dual prompts. Third, graph pre-training
tion by repeatedly sampling 𝑚 labeled nodes or graphs per class                              methods generally outperform the end-to-end GCN and GAT, show-
for 100 times. Each task is executed with five different random                              casing the benefits of pre-training on unlabeled graphs.
seeds, leading to a total of 500 outcomes for each classification                            Few-shot performance. To evaluate the performance of SAMGPT
type. We use accuracy as the evaluation metric, as each task is                              with more labeled data, we vary the number of shots, 𝑚, in both
class-balanced [20, 21, 46, 59], and report the average accuracy and                         node and graph classification tasks. We compare SAMGPT to two
standard deviation over these 500 outcomes.                                                  competitive baselines, GraphPrompt and GCOPE, with results
Baselines. We compare the performance of SAMGPT against state-                               reported in Fig. 3, where error bars represent the standard deviation.
of-the-art methods in four broad groups, as follows.                                         We observe that SAMGPT consistently outperforms the baselines in
   (1) End-to-end graph neural networks: GCN [15] and GAT [43]                               low-shot settings (e.g., 𝑚 ≤ 5). When further increasing the number
aggregate information from neighboring nodes to update node                                  of shots, SAMGPT still performs best in general, although it may
representations. For each task, they are trained from scratch in a                           be on par with GCOPE in some cases when 𝑚 approaches 10. This
supervised fashion without pre-training.                                                     is not surprising, since the advantage of SAMGPT may diminish as
   (2) Graph pre-training models: DGI [44], InfoGraph [38]1 and                              more supervision becomes available.
GraphCL [57] first pre-train a graph encoder to capture the inherent
properties of the graphs, and then fine-tune a classifier on the
downstream task while freezing the pre-trained model. GPPT2 [39],                            5.3    Ablation Studies
GPF [7] and GraphPrompt [21] employ a universal task template                                To understand the impact of each component in SAMGPT, we
                                                                                             perform two ablation studies.
1 Original DGI only operates at the node level, while InfoGraph extends it to the graph
                                                                                             Data ablation. We evaluate the impact of incorporating more
level. We apply DGI to node classification, and InfoGraph to graph classification.
2 GPPT is tailored for node classification task and is not applicable to graph classifica-   source domains by incrementally adding Citeseer, LastFM, Photo,
tion. Thus, in our experiments, we only use GPPT for node classification.                    and Facebook, in this order, to the pre-training, while fixing Cora as
SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation          WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


Table 2: Accuracy (%) of one-shot node classification with standard deviations. Each column represents a target domain, using
other columns as source domains. The best method in each column is bolded, and the runner-up is underlined.

  Method \ Target domain             Cora             Citeseer           Pubmed               Photo         Computers           Facebook            LastFM

            GCN                 29.53 ± 7.56       26.29 ± 6.50       23.32 ± 11.56       26.96 ± 12.94    24.40 ± 5.62       20.45 ± 5.62        9.21 ± 3.11
            GAT                 24.27 ± 9.26       21.56 ± 8.09       22.28 ± 9.76        17.85 ± 10.22    23.03 ± 12.12      29.27 ± 6.47        9.01 ± 2.61
           DGI                  33.40 ± 10.48      25.80 ±   8.27     47.22 ±   9.50      30.89 ± 10.54    25.75 ± 12.45      34.36 ±   9.57     14.14 ±   6.31
         GraphCL                27.72 ± 9.37       35.02 ±   8.46     48.89 ±   9.03      34.78 ± 11.56    23.79 ± 12.28      34.85 ±   7.07     18.93 ±   7.32
          GPPT                  27.18 ± 4.88       25.90 ±   4.68     39.82 ±   8.79      31.58 ± 10.27    19.94 ± 9.61       34.73 ±   3.99     20.98 ±   3.98
       GraphPrompt              28.26 ± 12.68      32.51 ±   8.73     47.47 ±   9.15      48.11 ± 9.89     42.82 ± 11.67      40.44 ±   9.68     19.84 ±   7.23
           GPF                  32.17 ± 6.56       36.79 ±   7.70     41.28 ±   8.14      47.47 ± 8.19     35.75 ± 7.12       40.45 ±   6.34     27.26 ±   5.50
          Hassani               33.35 ± 6.93       33.66 ± 7.24       39.87 ± 8.16        48.48 ± 7.07     39.99 ± 7.91       37.70 ± 5.79       27.16 ± 4.94
           GCOPE                35.62 ± 11.93      38.33 ± 9.28      45.38 ± 9.87         52.87 ± 9.19     45.65 ± 10.69     40.63 ± 8.50       28.84 ± 7.59
          SAMGPT                47.80 ± 11.88      36.38 ± 9.10      50.25 ± 10.43        58.71 ± 8.69     48.22 ± 8.17      42.70 ± 8.73       33.36 ± 8.11


Table 3: Accuracy (%) of one-shot graph classification with standard deviations. Each column represents a target domain, using
other columns as source domains. The best method in each column is bolded, and the runner-up is underlined.

  Method \ Target domain             Cora             Citeseer           Pubmed               Photo         Computers           Facebook            LastFM

            GCN                 30.64 ± 10.31      26.90 ± 7.15       38.84 ± 11.82       15.60 ± 8.77     21.94 ± 14.51      31.33 ± 9.47       28.83 ± 9.60
            GAT                 27.80 ± 7.85       27.50 ± 7.13       21.66 ± 8.70        15.74 ± 7.62     16.02 ± 13.46      21.20 ± 7.31       27.80 ± 7.85
        InfoGraph               34.98 ± 10.15      35.87 ±   9.84    48.67 ± 12.29        25.70 ± 11.73    19.02 ± 14.09      31.26 ± 9.65       23.29 ± 7.99
         GraphCL                42.70 ± 10.64      36.66 ±   8.67    47.53 ± 11.52        33.07 ± 12.31    16.02 ± 13.47      21.99 ± 13.00      21.30 ± 10.45
       GraphPrompt              37.38 ± 14.03      36.66 ±   9.19    49.55 ± 10.25        50.79 ± 12.31    43.09 ± 11.45      41.71 ± 10.61      32.62 ± 8.54
           GPF                  39.62 ± 8.52       36.73 ±   7.66    45.08 ± 10.36        47.57 ± 10.16    35.70 ± 8.71       34.84 ± 5.14       34.31 ± 7.05
          Hassani               36.86 ± 10.74      35.78 ± 8.80       43.97 ± 13.27       41.55 ± 13.08    29.49 ± 13.86      35.57 ± 9.00       25.39 ± 8.14
           GCOPE                38.85 ± 10.99      39.93 ± 9.82       47.05 ± 11.74       53.93 ± 9.74     45.60 ± 10.96     40.26 ± 9.53       34.68 ± 7.70
          SAMGPT                55.35 ± 13.62      38.75 ± 9.40       48.69 ± 10.16       58.75 ± 11.67    48.72 ± 11.18     43.71 ± 9.54       48.28 ± 9.72


Table 4: Data ablation study with an increasing number of                              Model ablation. We analyze variants of SAMGPT by removing
source domains, while fixing Cora as the target domain.                                structure tokens, holistic prompts and specific prompts. We report
                                                                                       the results of these variants and SAMGPT in Table 5. Note that
                                Number of source domains                               Variant 1, which lacks our structural alignment design, is equivalent
 Method                                                                                to the feature alignment method MDGPT [65].
                          1         2             3                  4
                                                                                          The results confirm that each component plays an important
 GraphPrompt 35.53±12.06 37.13±11.79 36.90±11.23 38.54±11.84                           role. First, the use of structure tokens is essential. Notably, Vari-
 GCOPE       39.47±12.14 36.63± 9.46 35.28±11.99 38.61±12.74                           ant 3 consistently outperforms Variant 1 and 2, both of which do
 SAMGPT      40.43±11.00 41.97±11.01 42.30±11.56 45.95±12.96
                                                                                       not employ structure tokens, demonstrating the effectiveness of
                                                                                       structure tokens in aligning multi-domain structural knowledge.
                                                                                       Second, removing specific prompts leads to a drop in performance,
                                                                                       evident from the superior accuracy of Variants 2 over Variant 1,
the target domain. We present one-shot node classification perfor-
                                                                                       and SAMGPT over Variant 4. This indicates the significance of
mance of SAMGPT and two competitive baselines, namely, Graph-
                                                                                       leveraging source domain-specific structural knowledge for effec-
Prompt and GCOPE, in Table 4. Across the columns, 1 represents
                                                                                       tive cross-domain adaptation. Third, holistic prompts prove to be
using Citeseer as the single source domain, while 2 represents using
                                                                                       useful, as Variant 4 often outperforms Variant 3, highlighting the
Citeseer and LastFM as the source domains, etc.
                                                                                       significance of incorporating holistic multi-domain structural in-
   We make the following observations. First, SAMGPT is supe-
                                                                                       formation via holistic prompts. These key components together
rior across different numbers of source domains, demonstrating its
                                                                                       enable SAMGPT to achieve optimal performance.
robustness to varying configurations of the source domains. Sec-
ond, both GraphPrompt and GCOPE often perform worse as more
datasets are added due to the negligence of structural discrepancies                   5.4    Homophily Sensitivity
in various domains. In contrast, SAMGPT exhibits consistent im-                        Apart from feature and structural differences, graphs also exhibit
provement with the addition of more source domains, validating                         varying homophily and heterophily patterns based on whether
the effectiveness of our structure alignment and adaptation.                           linked nodes share the same attribute [24, 63, 71]. To further assess
               WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                               Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang


                                                                   Table 5: Model ablation study on key components of SAMGPT.

                                        Structure   Holistic                   Specific         Target domain for node classification                     Target domain for graph classification
                    Methods
                                         tokens     prompts                    prompts         Cora            Photo           Facebook                  Cora            Photo            Facebook
                 Variant 1                 ×            ×                         ×       36.36 ± 12.71       49.10 ± 9.94     35.36 ± 9.06         45.44 ± 13.47       52.45 ± 12.37       38.74 ± 10.26
                 Variant 2                 ×            ×                         ✓       40.62 ± 11.79       56.23 ± 9.04     39.80 ± 10.39        45.63 ± 13.52       57.78 ± 11.64       42.22 ± 10.95
                 Variant 3                 ✓            ×                         ×       44.26 ± 10.92       56.61 ± 10.14    41.11 ± 8.34         52.88 ± 12.25       58.14 ± 12.01       43.12 ± 9.76
                 Variant 4                 ✓            ✓                         ×       46.10 ± 12.02       57.76 ± 10.00    40.46 ± 8.89         54.52 ± 14.32       58.12 ± 12.30       43.15 ± 10.12
                 SAMGPT                    ✓            ✓                         ✓       47.80 ± 11.88       58.71 ± 8.69     42.70 ± 8.73         55.35 ± 13.62       58.75 ± 11.67       43.71 ± 9.54


                          80                                              90                                          Table 6: Analysis of one-shot node classification on ho-
                          70                                              80                                          mophilic and heterophilic graphs.
                          60                                              70


                Acc (%)                                         Acc (%)
                          50                                              60
                                                                                                                       Target                                          Accuracy (%)
                          40                                              50                                                          Source domains
                                                                                                                       domain                                 GraphPrompt GCOPE                  SAMGPT
                          30                                              40
                          20
                                                                                                                        Squi.       Cham., Corn., Cora           18.98±4.89        18.98±4.75    20.43±4.75
                                                                          30
                                                                                                                        Corn.       Squi., Cham., Cora           29.67±8.36        27.19±8.51    32.57±8.68
                          10                                              20
                               10   8     6    4    2       0                   10    8   6     4    2    0            Cham.        Squi., Corn., Cora           23.28±4.63        23.24±4.50    23.89±4.91
                           Shot 𝑚 for node
                                       Shotclassification
                                            m                Shot 𝑚 for graph
                                                                         Shot classification
                                                                              m                                       Facebook      Squi., Cora, Photo           32.22±6.91        35.81±7.89    41.10±9.38
                                             (a) Target domain: Cora                                                   Squi., Cham., Corn. are short for Squirrel, Chameleon, and Cornell, respectively.
                                                                          90
                          90
                                                                          80                                          the robustness of SAMGPT across domains with varying homophily
                          80
                                                                                                                      patterns, we conduct one-shot node classification on homophilic

                Acc (%)                                         Acc (%)
                                                                          70
                          70
                                                                                                                      (Cora, Photo, Facebook) and heterophilic (Chameleon, Cornell and
                          60                                              60
                                                                                                                      Squirrel) graphs. Details about the heterophilic datasets are pre-
                          50                                              50                                          sented in Appendix F.
                          40                                              40                                             We report the results in Table 6 and observe that SAMGPT consis-
                               10   8     6    4    2       0                   10    8    6    4    2    0           tently surpasses GraphPrompt and GCOPE, regardless of whether
                           Shot 𝑚 for node
                                       Shotclassification
                                            m                Shot 𝑚 for graph
                                                                         Shot classification
                                                                              m                                       the source or target domains are homophilic or heterophilic. These
                                             (b) Target domain: Photo                                                 results further validate the efficacy of SAMGPT, demonstrating its
                          80                                                                                          ability to leverage multi-domain knowledge across a wide variety
                                                                          80                                          of graph domains. Note that we focus on the node classification
                          70
                                                                          70                                          task here, as homophily is defined based on node attributes, which

                Acc (%)                                         Acc (%)
                          60
                                                                          60                                          directly impacts node-level tasks.
                          50                                              50
                          40                                              40
                                                                                                                      6       Conclusions
                          30                                              30                                          In this paper, we proposed SAMGPT, a graph foundation model with
                               10   8     6    4    2       0                   10    8    6    4    2    0
                                                                                                                      structure alignment for text-free graphs, supporting both multi-
                           Shot 𝑚 for node
                                       Shotclassification
                                            m                  Shot 𝑚 for Shot
                                                                          graphmclassification                        domain graph pre-training and cross-domain adaptation. In the
                                           (c) Target domain: Facebook                                                pre-training phase, SAMGPT utilizes a series of structure tokens to
                                                                                                                      harmonize the structural distributions across multiple source do-
                          70                                              80                                          mains and to extract multi-domain structural knowledge. For down-
                          60                                              70                                          stream cross-domain adaptation, SAMGPT employs dual prompts
                          50                                                                                          to tailor pre-trained holistic and domain-specific structural knowl-

                Acc (%)
                                                                          60

                                                                Acc (%)
                          40
                                                                          50                                          edge to the target domain. We conducted extensive experiments
                          30                                                                                          on seven benchmark datasets, demonstrating that SAMGPT signifi-
                                                                          40
                          20                                                                                          cantly outperforms various state-of-the-art baseline methods.
                                                                          30
                          10
                                                                          20                                          Acknowledgments
                               10   8     6    4    2       0                   10    8    6    4    2    0
                           Shot 𝑚 for node
                                       Shotclassification
                                            m                  Shot 𝑚 for graph classification                        This research / project is supported by the Ministry of Education,
                                                                          Shot m
                                            (d) Target domain: LastFM                                                 Singapore under its Academic Research Fund (AcRF) Tier 1 grant
                                                                                                                      (22-SIS-SMU-054) and Tier 2 grant (Proposal ID: T2EP20122-0041).
                                          GraphPrompt               GCOPE                 SAMGPT
                                                                                                                      Any opinions, findings and conclusions or recommendations ex-
          90 Figure 3: Impact of number of shots on node and graph clas-                                              pressed in this material are those of the author(s) and do not reflect
               sification on four target domains.                                                                     the views of the Ministry of Education, Singapore.
          80
          70


Acc (%)
          60
          50
SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation                 WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


References                                                                               [28] Xuelian Ni, Fei Xiong, Yu Zheng, and Liang Wang. 2024. Graph Contrastive
 [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-            Learning with Kernel Dependence Maximization for Social Recommendation. In
     cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal                 WWW. 481–492.
     Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774       [29] Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, and Bo Yang. 2020.
     (2023).                                                                                  Geom-GCN: Geometric graph convolutional networks. In ICLR.
 [2] Vibhor Agarwal, Sagar Joglekar, Anthony P Young, and Nishanth Sastry. 2022.         [30] Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding,
     GraphNLI: A Graph-based Natural Language Inference Model for Polarity Pre-               Kuansan Wang, and Jie Tang. 2020. Gcc: Graph contrastive coding for graph
     diction in Online Debates. In WWW. 2729–2737.                                            neural network pre-training. In SIGKDD. 1150–1160.
 [3] Karsten M Borgwardt and Hans-Peter Kriegel. 2005. Shortest-path kernels on          [31] Faisal Rahutomo, Teruaki Kitasuka, Masayoshi Aritsugi, et al. 2012. Semantic
     graphs. In ICDM. IEEE, 8–pp.                                                             cosine similarity. In ICAST, Vol. 4. 1.
 [4] Kaize Ding, Kai Shu, Xuan Shan, Jundong Li, and Huan Liu. 2021. Cross-domain        [32] Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy
     graph anomaly detection. IEEE TNNLS 33, 6 (2021), 2406–2415.                             Wolf, and Dominique Beaini. 2022. Recipe for a general, powerful, scalable graph
 [5] Junfeng Fang, Xinglin Li, Yongduo Sui, Yuan Gao, Guibin Zhang, Kun Wang, Xi-             transformer. NeurIPS 35 (2022), 14501–14515.
     ang Wang, and Xiangnan He. 2024. EXGC: Bridging Efficiency and Explainability       [33] Benedek Rozemberczki, Carl Allen, and Rik Sarkar. 2021. Multi-scale attributed
     in Graph Condensation. In WWW. 721–732.                                                  node embedding. Journal of Complex Networks 9, 2 (2021), cnab014.
 [6] Junfeng Fang, Wei Liu, Yuan Gao, Zemin Liu, An Zhang, Xiang Wang, and               [34] Benedek Rozemberczki and Rik Sarkar. 2020. Characteristic functions on graphs:
     Xiangnan He. 2023. Evaluating Post-hoc Explanations for Graph Neural Networks            Birds of a feather, from statistical descriptors to parametric models. In CIKM.
     via Robustness Analysis. In Thirty-seventh Conference on Neural Information              1325–1334.
     Processing Systems.                                                                 [35] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and
 [7] Taoran Fang, Yunchao Zhang, Yang Yang, Chunping Wang, and Lei Chen. 2023.                Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29,
     Universal Prompt Tuning for Graph Neural Networks. In NeurIPS, Vol. 36.                  3 (2008), 93–93.
 [8] Christos Giatsidis, Fragkiskos Malliaros, Dimitrios Thilikos, and Michalis Vazir-   [36] Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, and Stephan
     giannis. 2014. CoreCluster: A degeneracy based graph clustering framework. In            Günnemann. 2018. Pitfalls of graph neural network evaluation. arXiv preprint
     AAAI, Vol. 28.                                                                           arXiv:1811.05868 (2018).
 [9] Kaveh Hassani. 2022. Cross-domain few-shot graph classification. In AAAI,           [37] Gilbert W Stewart. 1993. On the early history of the singular value decomposition.
     Vol. 36. 6856–6864.                                                                      SIAM review 35, 4 (1993), 551–566.
[10] Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang,           [38] Fan-Yun Sun, Jordan Hoffman, Vikas Verma, and Jian Tang. 2019. InfoGraph:
     and Jie Tang. 2022. Graphmae: Self-supervised masked graph autoencoders. In              Unsupervised and Semi-supervised Graph-Level Representation Learning via
     KDD. 594–604.                                                                            Mutual Information Maximization. In ICLR.
[11] Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, and Yizhou Sun. 2020.           [39] Mingchen Sun, Kaixiong Zhou, Xin He, Ying Wang, and Xin Wang. 2022. Gppt:
     Gpt-gnn: Generative pre-training of graph neural networks. SIGKDD (2020),                Graph pre-training and prompt tuning to generalize graph neural networks. In
     1857–1867.                                                                               SIGKDD. 1717–1727.
[12] Xinke Jiang, Zidi Qin, Jiarong Xu, and Xiang Ao. 2023. Incomplete graph learning    [40] Xiangguo Sun, Hong Cheng, Jia Li, Bo Liu, and Jihong Guan. 2023. All in One:
     via attribute-structure decoupled variational auto-encoder. In WSDM. 304–312.            Multi-Task Prompting for Graph Neural Networks. SIGKDD (2023), 2120–2131.
[13] Xinke Jiang, Rihong Qiu, Yongxin Xu, Wentao Zhang, Yichen Zhu, Ruizhe Zhang,        [41] Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Long Xia, Dawei Yin, and Chao Huang.
     Yuchen Fang, Xu Chu, Junfeng Zhao, and Yasha Wang. 2024. RAGraph: A General              2024. HiGPT: Heterogeneous Graph Language Model. In SIGKDD. 2842–2853.
     Retrieval-Augmented Graph Learning Framework. NeurIPS (2024).                       [42] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
[14] Thomas N Kipf and Max Welling. 2016. Variational graph auto-encoders. arXiv              Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
     preprint arXiv:1611.07308 (2016).                                                        Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv
[15] Thomas N Kipf and Max Welling. 2017. Semi-supervised classification with graph           preprint arXiv:2302.13971 (2023).
     convolutional networks. In ICLR.                                                    [43] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
[16] Jintang Li, Ruofan Wu, Wangbin Sun, Liang Chen, Sheng Tian, Liang Zhu,                   Lio, and Yoshua Bengio. 2018. Graph attention networks. In ICLR.
     Changhua Meng, Zibin Zheng, and Weiqiang Wang. 2023. What’s Behind the              [44] Petar Veličković, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio,
     Mask: Understanding Masked Graph Modeling for Graph Autoencoders. In KDD.                and R Devon Hjelm. 2018. Deep Graph Infomax. In ICLR.
     1268–1279.                                                                          [45] Chen Wang, Yueqing Liang, Zhiwei Liu, Tao Zhang, and S Yu Philip. 2021. Pre-
[17] Rongfan Li, Xinke Jiang, Ting Zhong, Goce Trajcevski, Jin Wu, and Fan Zhou.              training graph neural network for cross domain recommendation. In CogMI.
     2022. Mining spatio-temporal relations via self-paced graph contrastive learning.        140–145.
     In SIGKDD. 936–944.                                                                 [46] Ning Wang, Minnan Luo, Kaize Ding, Lingling Zhang, Jundong Li, and Qinghua
[18] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen,              Zheng. 2020. Graph Few-shot Learning with Attribute Matching. In CIKM. 1545–
     and Muhan Zhang. 2024. One for All: Towards Training One Graph Model for                 1554.
     All Classification Tasks. In ICLR.                                                  [47] Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, and Christopher
[19] Jiawei Liu, Cheng Yang, Zhiyuan Lu, Junze Chen, Yibo Li, Mengmei Zhang, Ting             Leckie. 2023. Cross-domain graph anomaly detection via anomaly-aware con-
     Bai, Yuan Fang, Lichao Sun, Philip S Yu, et al. 2023. Towards graph foundation           trastive alignment. In AAAI, Vol. 37. 4676–4684.
     models: A survey and beyond. arXiv preprint arXiv:2310.11829 (2023).                [48] Zeyuan Wang, Qiang Zhang, HU Shuang-Wei, Haoran Yu, Xurui Jin, Zhichen
[20] Zemin Liu, Yuan Fang, Chenghao Liu, and Steven CH Hoi. 2021. Relative and                Gong, and Huajun Chen. 2022. Multi-level Protein Structure Pre-training via
     absolute location embedding for few-shot node classification on graph. In AAAI,          Prompt Learning. In ICLR.
     Vol. 35. 4267–4275.                                                                 [49] Zhihao Wen and Yuan Fang. 2023. Augmenting Low-Resource Text Classification
[21] Zemin Liu, Xingtong Yu, Yuan Fang, and Xinming Zhang. 2023. GraphPrompt:                 with Graph-Grounded Pre-training and Prompting. In SIGIR. 506–516.
     Unifying pre-training and downstream tasks for graph neural networks. In WWW.       [50] Zhihao Wen and Yuan Fang. 2024. Prompt tuning on graph-augmented low-
     417–428.                                                                                 resource text classification. IEEE TKDE (2024).
[22] Yuanfu Lu, Xunqiang Jiang, Yuan Fang, and Chuan Shi. 2021. Learning to pre-train    [51] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and
     graph neural networks. In AAAI, Vol. 35. 4276–4284.                                      S Yu Philip. 2020. A comprehensive survey on graph neural networks. IEEE
[23] Chenglong Ma, Yongli Ren, Pablo Castells, and Mark Sanderson. 2024. Temporal             TNNLS 32, 1 (2020), 4–24.
     Conformity-aware Hawkes Graph Network for Recommendations. In WWW.                  [52] Jun Xia, Lirong Wu, Jintao Chen, Bozhen Hu, and Stan Z Li. 2022. Simgrace: A
     3185–3194.                                                                               simple framework for graph contrastive learning without data augmentation. In
[24] Yao Ma, Xiaorui Liu, Neil Shah, and Jiliang Tang. 2022. Is homophily a necessity         WWW. 1070–1079.
     for graph neural networks?. In ICLR.                                                [53] Lianghao Xia, Ben Kao, and Chao Huang. 2024. Opengraph: Towards open graph
[25] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel.              foundation models. arXiv preprint arXiv:2403.01121 (2024).
     2015. Image-based recommendations on styles and substitutes. In SIGIR. 43–52.       [54] Minghao Xu, Hang Wang, Bingbing Ni, Hongyu Guo, and Jian Tang. 2021. Self-
[26] Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore.                supervised graph-level representation learning with local and global structure.
     2000. Automating the construction of internet portals with machine learning.             In ICML. 11548–11558.
     Information Retrieval 3 (2000), 127–163.                                            [55] Baoyao Yang and Pong C Yuen. 2019. Cross-domain visual representations via
[27] Kentaro Miyake, Hiroyoshi Ito, Christos Faloutsos, Hirotomo Matsumoto, and               unsupervised graph alignment. In AAAI, Vol. 33. 5613–5620.
     Atsuyuki Morishima. 2024. NETEVOLVE: Social Network Forecasting using               [56] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He,
     Multi-Agent Reinforcement Learning with Interpretable Features. In WWW.                  Yanming Shen, and Tie-Yan Liu. 2021. Do transformers really perform badly for
     2542–2551.                                                                               graph representation? NeurIPS 34 (2021), 28877–28888.
                                                                                         [57] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and
                                                                                              Yang Shen. 2020. Graph contrastive learning with augmentations. NeurIPS 33
WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                              Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang


     (2020), 5812–5823.                                                                 embeddings (lines 20–21), and generate final embeddings by ag-
[58] Xingtong Yu, Yuan Fang, Zemin Liu, Yuxia Wu, Zhihao Wen, Jianyuan Bo, Xin-         gregating feature- and structure-level adapted embeddings (lines
     ming Zhang, and Steven CH Hoi. 2024. Few-Shot Learning on Graphs: from
     Meta-learning to Pre-training and Prompting. arXiv preprint arXiv:2402.01440       22–23). Finally, we update the embeddings for the prototypical
     (2024).                                                                            instances based on the labeled samples in the task and optimize
[59] Xingtong Yu, Zhenghao Liu, Yuan Fang, Zemin Liu, Sihong Chen, and Xinming
     Zhang. 2024. Generalized Graph Prompt: Toward a Unification of Pre-Training
                                                                                        holistic prompts, Λ and Γ (lines 24-28).
     and Downstream Tasks on Graphs. IEEE TKDE (2024).
[60] Xingtong Yu, Zemin Liu, Yuan Fang, and Xinming Zhang. 2023. Learning to            B     Complexity Analysis
     count isomorphisms with graph neural networks. AAAI 37, 4 (2023), 4845–4853.
[61] Xingtong Yu, Zemin Liu, Yuan Fang, and Xinming Zhang. 2024. HGPROMPT:              For a downstream graph 𝐺𝑇 = (𝑉𝑇 , 𝐸𝑇 , X𝑇 ) from the target domain
     Bridging Homogeneous and Heterogeneous Graphs for Few-shot Prompt Learn-           𝐷𝑇 , the computational process of structure adaptation involves
     ing. In AAAI. 16578–16586.
[62] Xingtong Yu, Zhenghao Liu, Yuan Fang, and Xinming Zhang. 2025. Node-Time           injecting holistic prompts and specific prompts into the pre-trained
     Conditional Prompt Learning in Dynamic Graphs. In ICLR.                            GNN to modify the encoding of nodes.
[63] Xingtong Yu, Jie Zhang, Yuan Fang, and Renhe Jiang. 2025. Non-Homophilic              In a standard GNN, each node aggregates messages from its
     Graph Pre-Training and Prompt Learning. In SIGKDD.
[64] Xingtong Yu, Chang Zhou, Yuan Fang, and Xinming Zhang. 2024. MultiGPrompt          neighbors in each layer. Assuming that the aggregation involves
     for Multi-Task Pre-Training and Prompting on Graphs. In WWW. 515–526.              at most 𝑛 neighbors, the complexity of calculating node embed-
[65] Xingtong Yu, Chang Zhou, Yuan Fang, and Xinming Zhang. 2024. Text-Free
     Multi-domain Graph Pre-training: Toward Graph Foundation Models. arXiv
                                                                                        dings over 𝐿 layers is 𝑂 (𝑛𝐿 · |𝑉𝑇 |). One one hand, holistic prompts
     preprint arXiv:2405.13934 (2024).                                                  are directly injected into each layer of the GNN, increasing the
[66] Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, and Hyunwoo J Kim.          complexity by 𝑂 (𝐿 · |𝑉𝑇 |). On the other hand, specific prompts are
     2019. Graph transformer networks. NeurIPS 32 (2019).
[67] Delvin Ce Zhang, Menglin Yang, Rex Ying, and Hady W Lauw. 2024. Text-              first generated by the pre-trained structure tokens from 𝐾 source
     attributed graph representation learning: Methods, applications, and challenges.   domains, with an additional complexity of 𝑂 (𝐿 · 𝐾). Then, specific
     In WWW. 1298–1301.                                                                 prompts modify the structure-base aggregation with a complexity
[68] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional
     control to text-to-image diffusion models. In ICCV. 3836–3847.                     of 𝑂 (𝐿 · |𝑉𝑇 |). Since holistic prompts and specific prompts modifies
[69] Haihong Zhao, Aochuan Chen, Xiangguo Sun, Hong Cheng, and Jia Li. 2024. All
     in one and one for all: A simple yet effective method towards cross-domain graph
     pretraining. In SIGKDD. 4443–4454.                                                 Algorithm 1 Cross-domain Adaptation for SAMGPT
[70] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and
     Jian Tang. 2023. Learning on Large-scale Text-attributed Graphs via Variational    Input: Pre-trained graph encoder GE with parameters Θpre , pre-trained
     Inference. In ICLR.                                                                    structure tokens Tpre , target domain dimension alignment function
[71] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai             DA𝑇 (·), and feature adaptation function FAD(·)
     Koutra. 2020. Beyond homophily in graph neural networks: Current limitations
     and effective designs. NeurIPS (2020), 7793–7804.
                                                                                        Output: Optimized holistic prompts Phol , coefficients Λ, and feature adap-
[72] Yun Zhu, Yaoke Wang, Haizhou Shi, Zhenshuo Zhang, Dian Jiao, and Siliang Tang.         tation parameters Γ
     2024. GraphControl: Adding Conditional Control to Universal Graph Pre-trained       1: while not converged do
     Models for Graph Domain Transfer Learning. In WWW. 539–550.                         2:     for each graph 𝐺𝑇 = (𝑉𝑇 , 𝐸𝑇 , X𝑇 ) in target domain 𝐷𝑇 do
                                                                                         3:         /* Target domain feature dimensions alignment by Eq. (3) */
Appendices                                                                               4:         X̃ ← DAL𝑇 (X)
                                                                                         5:         Phol , Λ, Γ ← initialization
A Algorithm                                                                              6:         /* Feature adaptation by Eq. (5) */
Our algorithm consists of two stages, multi-domain graph pre-                            7:         HFAD ← GE(FAD( G, X̃; Γ); Θpre )
training and downstream adaptation.                                                      8:         /* Structure alignment by dual prompts */
   In the multi-domain pre-training phase, we first apply the dimen-                     9:         /* Modification to GE via holistic prompts by Eq. (9) */
sion alignment function, DAL, to align feature dimensions from dif-                     10:         for each layer in GE do
ferent source domains by Eq. (3). Then, we use a feature alignment                      11:             h𝑙𝑣 ← Aggr(h𝑙𝑣−1 , {p𝑙hol ⊙ h𝑢𝑙 −1 : 𝑢 ∈ N𝑣 }; 𝜃 pre
                                                                                                                                                         𝑙 ), ∀𝑣 ∈ 𝐺
                                                                                                                                                                    𝑇

method to unify feature semantic spaces by Eq. (4). For structure                       12:        Hhol ← STACK( {h𝑣 : ∀𝑣 ∈ 𝐺𝑇 } )
alignment, we inject source domain-specific structure tokens into                       13:        /* Generation of specific prompts by Eq. (10) */
each layer of the graph encoder by Eq. (6). Finally, we fuse the                        14:        for p𝑙spe in Pspe do
                                                                                                                Í𝐾 𝑙 𝑙
feature aligned embedding and structure aligned embedding by                            15:            p𝑙spe ← 𝑖=1   𝜆𝑖 t𝑆
                                                                                                                          𝑖
Eq. (7), and optimize the pre-training loss by Eq. (8).                                 16:        /* Modification to GE via specific prompts*/
   We further present the key steps for cross-domain adaptation in                      17:        for Each layer in GE do
Algorithm 1. In lines 3–4, we align target domain feature dimen-                        18:            h̃𝑙𝑣 ← Aggr( h̃𝑙𝑣−1 , {p𝑙spe ⊙ h̃𝑢𝑙 −1 : 𝑢 ∈ N𝑣 }; 𝜃 pre
                                                                                                                                                            𝑙 ), ∀𝑣 ∈ 𝐺
                                                                                                                                                                       𝑇
sions with source domains. In lines 6–7, we integrate the feature                       19:        Hspe ← STACK( { h̃𝑣 : ∀𝑣 ∈ 𝐺𝑇 } )
adaptation method to generate feature-level adapted embeddings.                         20:        /* Fusion of dual-prompt tuned embeddings by Eq. (11) */
In lines 8–21, we employ dual prompts to adapt structural prior                         21:        HSAD ← Hhol + 𝛽Hspe
knowledge to the target domain. Specifically, we first inject holistic                  22:        /* Fusion of adapted embeddings by Eq. (12) */
prompts to modify the structure-based aggregation in each layer of                      23:        HAD ← HFAD + 𝛼HSAD
the graph encoder for holistic knowledge adaptation (lines 9–12).                       24:        /* Update prototypical nodes */
                                                                                        25:        for each class 𝑦 do
Then, we generate specific prompts by fusing the pre-trained struc-
                                                                                        26:            h𝑦 ← Mean( {h𝑥 : instance 𝑥 belongs to class 𝑦})
ture tokens (lines 13–15), and utilize specific prompts for domain-
specific knowledge adaptation (lines 17–19). We obtain structure-                       27:       /* Optimizing Phol , Λ, and Γ */
                                                                                        28:       Calculate Ldown (Ω; Phol , Λ, Γ) by Eq. (13)
level adapted embeddings by fusing holistic and domain-specific
                                                                                        29: return Phol , Λ, and Γ
SAMGPT: Text-free Graph Foundation Model for Multi-domain Pre-training and Cross-domain Adaptation       WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.


the node encoding phase separately, the overall complexity would                    (1) End-to-end GNNs:
increase by (𝐿 · |𝑉𝑇 | + 𝐿 · 𝐾).                                                    • GCN [15]: GCN employs a mean-pooling approach for neighbor-
   Thus, the encoding time by the pre-trained GNN still dominates                      hood aggregation, enabling the integration of information from
the overall complexity, as 𝑂 (𝑛𝐿 · |𝑉𝑇 |) typically exceeds 𝑂 (𝐿 · |𝑉𝑇 | +             adjacent nodes.
𝐿 ·𝐾) given that 𝑛𝐿 > 𝐿 and |𝑉𝑇 | > 𝐾 in general. In other words, our               • GAT [43]: GAT relies on neighborhood aggregation for node rep-
structural adaptation adds a marginal overhead to the pre-trained                      resentation learning, but distinguishes itself by assigning varying
graph encoder.                                                                         attention weights to neighbors, thus adjusting their influence on
                                                                                       the aggregation process.
C    Further Descriptions of Datasets                                               (2) Graph pre-training models:
In this section, we provide more comprehensive descriptions of the                  • DGI [43]: DGI is a self-supervised pre-training approach. It is
benchmark datasets used in our experiments, as summarized in                           based maximizing mutual information (MI), with the goal of
Table 1, in the following.                                                             strengthening the MI between local node representations and
• Cora [26] consists of 2,708 publications in the computing field,                     their global context.
  each categorized into one of seven classes. The citation network                  • InfoGraph [38]: Building on DGI, InfoGraph focuses on graph-
  comprises 5,429 edges. Each publication is represented by a binary                   level tasks, aiming to align node and graph embeddings by maxi-
  word vector indicating the presence or absence of words from a                       mizing the similarity between them.
  dictionary containing 1,433 unique words.                                         • GraphCL [57]: GraphCL applies various graph augmentations
• Citeseer[35] contains 3,312 computer science publications, each                      for self-supervised learning, leveraging structural patterns within
  belonging to one of six categories, distinct from those in Cora.                     graphs. Its main objective is to improve the similarity across
  The citation network consists of 4,732 edges. Each publication is                    different augmentations during pre-training.
  represented by a binary word vector, reflecting the presence or                   • GPPT [39]: GPPT pre-trains a GNN model via link prediction task.
  absence of words from a dictionary of 3,703 unique words.                            Its downstream prompt module is specifically designed for node
• PubMed [35] consists of 19,717 biomedical publications related to                    classification, unifying it with the pre-training link prediction
  diabetes, each classified into one of three categories. The citation                 task.
  network includes 44,338 edges. Each publication is represented                    • GPF [7]: GPF serves as a universal prompt-based tuning approach
  by a TF/IDF-weighted word vector, indicating the presence of                         for pre-trained graph models. It adapts the input graph’s feature
  500 unique words from the dictionary.                                                space to simulate the behavior of various prompting functions.
• Photo [36] contains 7,487 products related to photography, each                   • GraphPrompt [21]: GraphPrompt utilizes subgraph similarity
  assigned to one of eight categories. The co-purchase network                         calculations as a unified framework to bridge the gap between
  comprises 119,043 edges, representing products frequently bought                     pre-training and downstream tasks, supporting both node and
  together. Each product is described by a feature vector derived                      graph classification. During downstream adaptation, a learnable
  from its metadata and reviews, and is labeled according to its                       prompt is tuned to incorporate task-specific knowledge.
  category.                                                                         (3) Graph cross-domain models:
• Computers [36] includes 13,752 computer-related products, di-
                                                                                    • Hassani [9]: Hassani proposes an attention-based graph encoder
  vided into ten categories. The co-purchase network consists of
                                                                                       that leverages both contextual and topological views to capture
  245,861 edges, representing products that are frequently bought
                                                                                       task-specific information for quick adaptation, as well as task-
  together. Each product is characterized by a feature vector gen-
                                                                                       independent knowledge for efficient transfer across domains.
  erated from its metadata and reviews and is labeled according to
  its respective category.                                                          (4) Multi-domain graph pre-training models:
• Facebook [33] represents a page-to-page Web graph of verified                     • GCOPE [69]: GCOPE propose a multi-domain pre-training strat-
  Facebook pages. The nodes correspond to official Facebook pages,                     egy that integrates graph datasets from various domains using
  and the edges indicate mutual “likes” between these pages. Node                      domain-specific interconnecting virtual nodes, which link nodes
  features are derived from the descriptions provided by the page                      within the same domain. The main objective is to enhance down-
  owners that outline the purpose of their pages.                                      stream performance by harnessing knowledge from multiple
• LastFM [34] represents a social network of LastFM users, col-                        source domains.
  lected via the public API in March 2020. The nodes correspond
  to LastFM users from various Asian countries, and the edges                       E     Implementation Details
  represent mutual follower relationships. The node features are                    We outline key settings for the baselines and SAMGPT.
  extracted based on the artists that users have liked. The associ-                 Baseline settings. We utilized the official codes for all open-source
  ated task for this graph is multinomial node classification, where                baselines. Each model was tuned based on the settings recom-
  the objective is to predict each user’s location, derived from the                mended in their respective work to achieve optimal performance.
  country field in their profile.                                                      For the baseline GCN [15], we employ a 3-layer architecture, and
                                                                                    set the hidden dimensions to 256. For GAT [43], we employ a 2-layer
D     Further Descriptions of Baselines                                             architecture and set the hidden dimension to 64. Additionally, we
In this section, we provide additional details about the baselines                  apply 8 attention heads in the first GAT layer.
used in our experiments.
      WWW ’25, April 28–May 2, 2025, Sydney, NSW, Australia.                                                                    Xingtong Yu, Zechuan Gong, Chang Zhou, Yuan Fang, and Hui Zhang



                         Node Classification                                   Graph Classification                 and heterophilic datasets in Sect. 5.4. Details of the heterophilic
                                                                                                                    datasets are introduced as follows. (1) Chameleon [33] is a Wikipedia-
                70
                                                                      70                                            based network containing 2,277 pages, categorized into five groups
                60                                                                                                  based on their average monthly traffic. This dataset forms a net-
                                                                      60


                                                            Acc (%)
                                                                                                                    work with 36,101 edges, and the node features are derived from
      Acc (%)
                50
                                                                      50                                            key nouns extracted from the Wikipedia content. (2) Cornell [29]
                40
                                                                      40                                            is another webpage network consisting of 183 nodes, where each
                30
                                                                                                                    node represents a webpage, and 295 edges denoting hyperlinks
                                                                      30
                20                                                                                                  between them. The node features are derived from a bag-of-words
                     0.001 0.01   0.1        1   10   100                   0.001 0.01   0.1        1   10    100   representation of the webpages. These pages are manually classified
                                        𝛼α                                                     α𝛼                   into five categories: student, project, course, staff, and faculty. (3)
                70                                                                                                  Squirrel [33] consists of 5,201 Wikipedia pages discussing specific
                                                                       70
                60                                                                                                  topics. The pages are divided into five categories based on their
                                                                       60
                                                                                                                    average monthly traffic. This dataset forms a page-to-page network

      Acc (%)                                                Acc (%)
                50
                                                                       50                                           with 217,073 edges, and the node features are derived from various
                40                                                                                                  informative nouns present in the Wikipedia content.
                                                                       40
                30
                                                                       30                                           G     Hyperparameter Sensitivity
                20
                     0.001 0.01   0.1        1   10   100                   0.001 0.01   0.1        1   10    100   We investigate the impact of hyperparameters, 𝛼 and 𝛽, in SAMGPT.
                                  𝛽β                                                           β
                                                                                               𝛽                    𝛼 governs the fusion of feature and structure alignment, as well
                          Target domain:                Cora                   Photo               Facebook         as their adaptation, in Eqs. (7) and (12), whereas 𝛽 controls the
                                                                                                                    aggregation of holistic and domain-specific adaptation in Eq. (11).
                              Figure 4: Sensitivity study of 𝛼 and 𝛽.
                                                                                                                    We vary 𝛼 and 𝛽 and present 1-shot node and graph classification
          70                                                                                                        results on three target domain, Cora, Photo and Facebook, in Fig. 4,
         For DGI [43], we utilize a 1-layer GCN as the backbone and set                                             with error bars denoting the standard deviation.
      the                                                                                                              We observe that increasing 𝛼 from lower values initially en-
       60hidden dimensions to 256. Additionally, we employ prelu as the                                             hances performance as structure alignment and adaptation are



Acc (%)
      activation function. For InfoGraph [38], a 3-layer GCN is used as the
      backbone, with its hidden dimensions set to 256. For GraphCL [57],                                            emphasized. However, after reaching a peak (𝛼 = 1), accuracy be-
      a50
        1-layer GCN is also employed as its backbone, with the hidden                                               gins to decline as 𝛼 grows further, implying that both feature and
      dimensions set to 256. Specifically, we select edge dropping as the                                           structure alignment are essential. Moreover, 𝛽 exhibit a trend simi-
      augmentations, with a default augmentation ratio of 0.2. For GPPT                                             lar to that of 𝛼, demonstrating that incorporating both holistic and
       40
      [39], we utilize a 2-layer GraphSAGE as its backbone, setting the                                             domain-specific knowledge is vital for cross-domain adaptation.
      hidden dimensions to 256. We employ a mean aggregator for the                                                 Based on the above observations, we set 𝛼 = 1 in our experiments,
       30
      aggregation in the backbone. For GraphPrompt [21], a 3-layer GCN                                              indicating a balance between the feature and structure counter-
      is used as the backbone for all datasets, with the hidden dimensions                                          parts, and 𝛽 = 1, indicating a balance between holistic and specific
      set to 256. For GPF [7], we employ a 5-layer GCN as the backbone                                              prompts, both of which show robust empirical performance.
      for all0.001
              datasets,0.01     0.1
                        following the     1
                                      recommended  10settings.
                                                           100The hidden
      dimensions are set to 256.                                                                                    H     Data Ethics Statement
                                                       α
         For Hassani [9], a 3-layer GCN is used as the backbone for all                                             To evaluate the efficacy of SAMGPT, we conducted experiments
      datasets, with the hidden dimensions set to 256.                                                              with only publicly available datasets, including Cora3 , Citeseer4 ,
         For GCOPE [69], we employ a 2-layer GCN as the backbone                                                    Pubmed5 , Photo6 , Computers7 , Facebook8 , LastFM9 , Chameleon10 ,
      and set the hidden dimensions to 100. Downstream adaptation is                                                Cornell11 , and Squirrel12 in accordance to their usage terms and
      achieved through fine-tuning, as it is reported to yield the best                                             conditions, if any. We also confirm that no personally identifiable
      performance in their literature.                                                                              information was utilized, and this research did not involve any
         For all baselines except GCOPE, we set the unified feature di-                                             human or animal subjects.
      mensions to 50, matching our SAMGPT. For GCOPE, we adhere to                                                  3 https://github.com/shchur/gnn-benchmark/raw/master/data/npz/cora.npz
                                                                                                                    4 https://github.com/shchur/gnn-benchmark/raw/master/data/npz/citeseer.npz
      the recommended settings and set it to 100.
                                                                                                                    5 https://github.com/shchur/gnn-benchmark/raw/master/data/npz/pubmed.npz
      SAMGPT settings. For our proposed SAMGPT, we utilize a 3-layer                                                6 https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_

      GCN as the backbone for all datasets, with the hidden dimensions                                              electronics_photo.npz
                                                                                                                    7 https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_
      set to 256. We set the unified feature dimensions to 50.
                                                                                                                    electronics_computers.npz
                                                                                                                    8 https://graphmining.ai/datasets/ptg/facebook.npz
      F              Details about Heterophilic Datasets                                                            9 https://graphmining.ai/datasets/ptg/lastfm_asia.npz
                                                                                                                    10 https://github.com/SitaoLuan/ACM-GNN/tree/main/new_data/chameleon
      To evaluate the robustness of SAMGPT across graphs with varying                                               11 https://github.com/bingzhewei/geom-gcn/tree/master/new_data/cornell
      homophily ratios, we conducted experiments on both homophilic                                                 12 https://github.com/SitaoLuan/ACM-GNN/tree/main/new_data/squirrel

