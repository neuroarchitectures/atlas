# Graph Classification using Structural Attention Lee Rossi Kong 2018

> Source: `Graph_Classification_using_Structural_Attention_Lee_Rossi_Kong_2018.pdf`

---

                          Graph Classification using Structural Attention
                    John Boaz Lee                                                      Ryan Rossi                                                   Xiangnan Kong
         Worcester Polytechnic Institute                                            Adobe Research                                      Worcester Polytechnic Institute
             Massachusetts, USA                                                      California, USA                                        Massachusetts, USA
                jtlee@wpi.edu                                                      rrossi@adobe.com                                           xkong@wpi.edu

ABSTRACT                                                                                              active                                                      inactive
Graph classification is a problem with practical applications in many                                     D                 D                                         D         D
different domains. To solve this problem, one usually calculates
certain graph statistics (i.e., graph features) that help discriminate                            C                                 C   B       A           C                           C       E           F
between graphs of different classes. When calculating such features,
most existing approaches process the entire graph. In a graphlet-                                         D                 D                                         D         D
based approach, for instance, the entire graph is processed to get the
                                                                                                      D       D                             D       D                               D       D
total count of different graphlets or subgraphs. In many real-world                                                   0.1
                                                                                                                         0.8                                0.1       0.9
applications, however, graphs can be noisy with discriminative                                    C           ! C ∗
                                                                                                                                B   A   C               C         B         A   C               C   B       A
                                                                                                                      0.1
patterns confined to certain regions in the graph only. In this work,                                 D       D                             D       D                               D       D
we study the problem of attention-based graph classification. The
use of attention allows us to focus on small but informative parts                                        C                                     B                                       A
                                                                                                                                                                                                predicted
of the graph, avoiding noise in the rest of the graph. We present a                                                                                                                               label

novel RNN model, called the Graph Attention Model (GAM), that                                   Figure 1: Attention-based graph classification. Given a start-
processes only a portion of the graph by adaptively selecting a                                 ing node v ∗ and a budget T = 3 nodes to select for graph clas-
sequence of “informative” nodes. Experimental results on multiple                               sification, attention is used to guide the walk towards more
real-world datasets show that the proposed method is competitive                                informative parts of the graph.
against various well-known methods in graph classification even
though our method is limited to only a portion of the graph.                                    chemoinformatics [11], social network analysis [2], urban comput-
                                                                                                ing [3], and cyber-security [6] can all be naturally represented as
CCS CONCEPTS                                                                                    labeled graphs. In chemoinformatics, for instance, molecules can be
• Information systems → Data mining; • Mathematics of com-                                      represented as graphs where nodes correspond to atoms, and edges
puting → Graph algorithms; • Theory of computation → Rein-                                      signify the presence of chemical bonds between pairs of atoms. The
forcement learning;                                                                             task then is to predict the class label of each graph – for instance,
                                                                                                the anti-cancer activity, solubility, or toxicity of a molecule.
KEYWORDS                                                                                           To solve this problem, the usual strategy is to calculate certain
                                                                                                graph statistics (i.e., graph features) on the entire graph. A popular
Attentional processing, graph mining, reinforcement learning, deep
                                                                                                technique is the graphlet kernel [26] which counts the occurrences
learning
                                                                                                of various graphlets (i.e., subgraphs) on a graph. Graphs that share
ACM Reference Format:                                                                           a lot of common graphlets are then considered similar. The Morgan
John Boaz Lee, Ryan Rossi, and Xiangnan Kong. 2018. Graph Classification                        algorithm [23] is another method for calculating graph features. It
using Structural Attention. In KDD 2018: 24th ACM SIGKDD International                          uses an iterative process which updates each node’s attribute vector
Conference on Knowledge Discovery & Data Mining, August 19–23, 2018,                            by hashing a concatenation of all the attributes in the node’s local
London, United Kingdom. ACM, New York, NY, USA, 9 pages. https://doi.
                                                                                                neighborhood. The graph feature is then computed from the final
org/10.1145/3219819.3219980
                                                                                                attributes of all the nodes in the graph. In more recent years, the
                                                                                                focus has shifted towards learning data-driven graph features [11,
1     INTRODUCTION                                                                              19] which is to say task-relevant features are learned automatically
Graph classification, or the problem of identifying the class labels                            from the graphs in a given dataset. Since we can expect graphs
of graphs in a dataset, is an important problem with practical ap-                              belonging to a particular class to exhibit some common pattern
plications in a diverse set of fields. Data from bioinformatics [5],                            that is not typically observed among the other graphs, we can then
                                                                                                use the calculated graph features for classification.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
                                                                                                   However, graphs in the real-world can be both large and noisy
for profit or commercial advantage and that copies bear this notice and the full citation       [21, 35]; this introduces some challenges when the entire graph has
on the first page. Copyrights for components of this work owned by others than the              to be processed to calculate graph features. When a graph is noisy,
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission   the significant subgraph patterns can be sparse and confined only to
and/or a fee. Request permissions from permissions@acm.org.                                     small neighborhoods within the graph. For instance, when studying
KDD 2018, August 19–23, 2018, London, United Kingdom                                            the interaction networks of complex diseases, researchers have
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-5552-0/18/08. . . $15.00                                                    identified specific subnetworks that are associated with the disease
https://doi.org/10.1145/3219819.3219980                                                         [8]. In this case, processing the entire graph can inadvertently cause
                                                                   D       D
                                                                                                       to focus on relevant parts of the graph while ignoring the noise in
                                                                                                       the rest of the graph which results in graph features that provide
                                                       5       C               C       B       A       better predictive performance. Furthermore, our attention-guided
                                                                   D       D                           walk is designed to rely only on local information from the graph
                                                                                                       which has the added benefit of keeping computation costs (space
         (a) attention-based classification [17]               (b) graph classification [10, 25]       in particular, in this case) low since there is no need to load the
     D     D                        D      D                           D       D                       entire graph into memory. We provide an illustration of this in
                                                                                                       Figure 1. On the other hand, Figure 2 highlights the difference
 C             C    B    A      C              C   B       A       C               C       B       A
                                                                                                       between attention-based graph classification and the two related
     D     D                        D      D                           D       D                       problems of (1) graph classification and (2) attentional processing
                             (c) attention-based graph classification                                  on non-graph data.
                                                                                                          Inspired by the recent success of Recurrent Neural Networks
Figure 2: Comparison of different classification problems.
                                                                                                       (RNN) with attention on vision-related tasks [18], we propose an
Traditional glimpse-based attentional processing in (a)
                                                                                                       RNN model with a built-in attention mechanism for graphs. The
takes several glimpses of the input before classification. On
                                                                                                       attention mechanism in our model is trained using reinforcement
the other hand, graph classification without attention in (b)
                                                                                                       learning to actively select informative regions in the graph to pro-
takes the entire input graph and uses it for classification.
                                                                                                       cess. We also introduce a memory component which allows the
The problem we study in (c) uses attention to process a small
                                                                                                       model to integrate information gathered from different parts of a
but relevant part of the graph for classification while obey-
                                                                                                       graph. The main contributions of our paper can be summarized as
ing graph structure.
                                                                                                       follows:
noise to be introduced into the calculated feature as most of the                                          (1) We propose to study the problem of attention-based graph
graph does not contain anything informative. Furthermore, it is                                                classification and introduce GAM, a general framework that
usually costly if not infeasible, to compute representations of large                                          uses attentional processing to learn data-driven features for
real-world graphs [24].                                                                                        graphs. The model uses attention to selectively process infor-
   To address the issues mentioned above, we study the problem of                                              mative portions of the graph. To the best of our knowledge,
attention-based graph classification which we formally define below.                                           this is the first model that uses attention to learn representa-
We begin by giving the definition of traditional graph classification.                                         tions for general attributed graphs.
Definition 1. Graph Classification: Given a set of attributed graphs                                       (2) We introduce a model that does not require global informa-
D = {(G1 , ℓ1 ), (G2 , ℓ2 ), · · · , (Gn , ℓn )}, the goal is to learn a function                              tion of the graph. Instead, the attention mechanism processes
f : G → L, where G is the input space of graphs and L is the set of                                            only a node’s local neighborhood. In addition, the model is
graph labels. Here each graph Gi = (A Gi , D Gi ) is comprised of an                                           easily parallelizable.
adjacency matrix A Gi ∈ {0, 1} Ni ×Ni , and an attribute matrix D Gi ∈                                     (3) We show through empirical evaluation on multiple real-
RNi ×D , where Ni is the number of nodes in the i-th graph and D                                               world datasets that the model can outperform various estab-
is the number of attributes. Each graph also has a corresponding                                               lished baselines.
label ℓi .                                                                                                 (4) We demonstrate the effectiveness of attention by comparing
                                                                                                               against modified baselines that utilize random attention.
Definition 2. Attention-based Graph Classification: Given a set                                           The rest of the paper is organized as follows. We start by review-
of attributed graphs D = {(G1 , ℓ1 ), (G2 , ℓ2 ), · · · , (Gn , ℓn )}, and a                           ing related work in the next section. We then present the proposed
budget T , the goal is to learn a composite function f ′ ◦ д : G → L.                                  method in section 3. Experimental setup and discussion of results
Here, Gi , G, and L still retain the same definition while ◦ is the                                    are provided in section 4. Finally, we conclude the paper and give
function composition operator. The function д : G → RT ×D selects                                      some directions for future work in the last section.
T nodes and returns their corresponding attributes. In other words,
given a graph Gi , we are limited to selecting T nodes and can only
use the corresponding information (i.e, node attributes) to make a                                     2    RELATED WORK
prediction on the graph label.                                                                         Many different techniques have been proposed to solve the graph
   We summarize the two main challenges of this problem as fol-                                        classification problem. One popular approach is to use a graph
lows: (1) graph structure is important; and (2) graphs can be noisy                                    kernel to measure similarity between different graphs [20]. This
with task-relevant patterns confined to small regions within the                                       similarity can be measured by considering various structural prop-
graph. A useful model has to take these two factors into considera-                                    erties like the shortest paths between nodes [4], the occurrence of
tion which makes the problem challenging. For instance, we cannot                                      certain graphlets or subgraphs [26], and even the structure of the
just choose a random set of nodes since this does not capture the                                      graph at different scales [15].
graph’s structure and is unlikely to yield task-relevant information.                                     Recently, several new methods which generalize over previ-
   To address these challenges, we propose a solution based on                                         ous approaches, have been introduced. These methods use a deep
attention-guided walks. Instead of choosing T random nodes, we                                         learning framework to learn data-driven representations of graphs
use a walk on the graph to sample T nodes. A walk allows us to                                         [11, 19, 34]. In [11], a method is introduced that generalizes the
capture the structure of the traversed region. On the other hand,                                      Weisfeiler-Lehman (WL) algorithm by learning to encode only rel-
an attention mechanism is used to guide the walk. This allows us                                       evant features from a node’s neighborhood during each iteration.
Interestingly, [19] proposes a method that processes a section of          The ultimate goal of the agent is to collect enough information that
the input graph using a convolutional neural network. However,             will allow it to make a correct prediction on the label of the graph.
for this to work for graphs of arbitrary sizes the method relies on a          The agent will only explore a small portion of the graph with
labeling step that ranks all the nodes in the graph which means it         the attention mechanism guiding it in its exploration. If the graph
still processes the entire graph initially.                                is large, we can also initialize multiple agents at different nodes
    One thing common among all the above-mentioned approaches              in the graph and run them in parallel. Deploying multiple agents
is that the entire graph is processed to compute the graph’s final         can help improve the performance of the model since each agent
representation. In contrast, we study a model that only processes a        can explore a different part of the graph with attention helping to
portion of the graph with attention used to determine the parts of         steer each agent’s exploration along the local neighborhood. This
the graph to focus on.                                                     allows us to use the model on large graphs that may be difficult or
    Recent studies have shown that deep learning frameworks with           impossible to load into memory.
attentional processing can perform well on a variety of tasks. In [17],
attention was used to allow the model to attend to a subset of the
                                                                           3.1    Proposed Model
source words in the language translation task. Meanwhile, [33] used
attention to help a model fix its gaze on salient objects for image        Our proposed model has an RNN at its core, as shown in Figure 3.
captioning and [18] applied attention to the image classification          At each time step, the core network processes new information
task. The work in [7], on the other hand, used attention to guide a        from the step that was just taken and integrates this into its internal
CNN to focus on relevant objects for the visual question answering         representation together with information retained from previous
task. Although attentional processing has been applied successfully        steps. It uses this information to predict the label of the input graph
to many problems, most of the existing work lie in the computer            and to decide which areas of the graph to prioritize for further
vision or natural language processing domains. In this work, we            exploration in the next time step.
focus on graphs, which have less well-behaved structure when                  Step module: At each time step, the step module considers the
compared to images/videos (grids) or text (sequences). Because of          one-hop neighborhood of the current node c t −1 and picks a neigh-
this, traditional models for attention cannot be directly applied to       bor c t to take a step towards. The step module is biased towards
graph data.                                                                picking neighbors whose types or labels have higher rankings in
    A few work have also begun to investigate the use of attention         the rank vector rt −1 . The attribute vector of the chosen node is then
on graphs [9, 28]. Choi et al. studied a model which used attentional      extracted and fed together with rt −1 to produce the step represen-
processing on medical ontology graphs [9]. However, our work is            tation st = fs (dc t , rt −1 ; θ s ) (see Figure 3a). The step representation
significantly different from that of [9] as their model is specifically    st is the new information available to the core LSTM network at
designed for medical ontologies and work only on directed acyclic          each time step. The step algorithm is summarized in Algorithm 1.
graphs (DAG) while we explore an attention mechanism on more                  Node type: The way we label or assign types to nodes allows us
general (un)directed attributed graphs. In another work, attentional       to bias the exploration towards certain nodes at different stages of
processing was used to solve the problem of node representation            the exploration. Depending on the application, the node type can
learning [28] which aims to learn embeddings for nodes in a graph.         be a simple discrete value (e.g., type of atom in a molecular graph)
This is quite different from the task of learning representations for      or it can be something more elaborate like a category derived from
graphs – and not nodes – in a dataset which is the problem studied         log-binning several attributes that capture the local structure of the
here. To the best of our knowledge, this is the first work that utilizes   node [1]. We give a simple example of the latter case. Suppose the
attention to learn data-driven features for graphs.                        agent wants to visit one of two Carbon nodes adjacent to it, it cannot
    Finally, we also experiment with an architecture that has a simple     differentiate between the nodes under the first node typing strategy.
external memory to allow multiple agents to integrate information          In the second method, the node type may be calculated based on
from various parts of the graph. In a sense, this is conceptually          the statistics encoded in the k-hop neighborhood of each node and
similar to the memory networks of [22, 27].                                this allows us to differentiate between the two Carbon nodes. Using
                                                                           more complex node typing strategies [1] may, however, increase
                                                                           the number of node types substantially and one may have to look
                                                                           into reinforcement learning strategies that work well when the
3   GRAPH ATTENTION MODEL
                                                                           discrete action space is large [10].
To simplify the discussion, we begin by describing a basic attention          History: The core LSTM network maintains a history vector
model. In subsequent discussion, we introduce a variant with more          which is a summary of all the information obtained by the agent
refined attention and external memory.                                     in its exploration of the graph thus far. At each time step, as new
   In this work, we formulate the problem of applying attention            information becomes available in the form of st from the step we
on graph-structured data as a decision process of a goal-directed          just took, the history vector is updated via ht = fh (st , ht −1 ; θh ).
agent traversing along an input attributed graph. The agent starts         This allows the core network to integrate information over time.
at a random node on the graph and, at each time step, moves to a              We use an LSTM in our architecture as it is superior to simple
neighboring node. The information available to the agent is limited        RNNs in capturing long-range dependencies. Even though LSTMs
to the node it chooses to explore. Since global information about the      have a more sophisticated memory model when compared to simple
graph is unavailable, the agent needs to integrate information over        RNNs, it has been shown that they still have trouble remembering
time to help it determine the parts of the graph to explore further.       information that was inputted too far in the past [30]. Because of
                                                                                                                                     [1,0,0,1,1]                                                         [1,0,0,1,1]
                                                                                                                            ct-1
                                                                                                                  [1,1,0,1,0]                                                        [1,1,0,1,0]
                                                                                                                                             [0,0,0,0,1]                                                        [0,0,0,0,1]
                                                                                           rt-1                                                                  rt                          ct
                                                                                                                               [1,0,0,0,1]                                                         [1,0,0,0,1] = dct
                                                                                                                                                   [0,1,0,0,0]                                                         [0,1,0,0,0]
                                                                                                                 [0,0,0,1,0]                                                         [0,0,0,1,0]

            [1,0,0,1,1]
                                                                                                                            ! = {A!, D!}                                                          ! = {A!, D!}
                         ct-1
   [1,1,0,1,0]
                                [0,0,0,0,1]
                                                      Step
                 [1,0,0,0,1]
                                   [0,1,0,0,0]        Module
                                                                          '#2                            "#(. ; '#)                                                          "#(. ; '#)
   [0,0,0,1,0]

          ! = {A!, D!}
                                                                    dct         '#3                 st                                                                st+1
                                                                                                                                        ht                                                                  ht+1
                                                                          '#1         st
        rt-1
                                                                                             ht-1        "ℎ(. ; 'ℎ)                                                          "ℎ(. ; 'ℎ)


                                                 (a) step network
                                                                                                                "*(. ; '*)             "+(. ; '+)                                   "*(. ; '*)              "+(. ; '+)



                                                                                                                      ,-t                    rt                                           ,-t+1                 rt+1
                                                                                                                                                     (b) GAM architecture
Figure 3: (a) Step network: Given a labeled graph G (composed of the adjacency matrix A G , and the attribute matrix D G ), a
current node c t −1 , and a stochastic rank vector rt −1 , the step module takes a step from the current node c t −1 to one of its
neighbors c t , prioritizing those whose type (i.e., node label) have higher rank in rt −1 . The attribute vector of c t , dc t , is extracted
and mapped to a hidden space using the linear layer parameterized by θ s2 . Similarly, rt −1 is mapped using another linear
layer parameterized by θ s1 . Information from these two sources are then combined using a linear layer parameterized by θ s3
to produce st , or the step embedding vector which represents information captured from the current step we took. (b) GAM
architecture: We use an RNN as the core component of the model; in particular, we use the Long Short-Term Memory (LSTM)
variant [12]. At each time step, the core network fh (.; θh ) takes the step embedding st and the internal representation of the
model’s history from the previous step ht −1 as input, and produces the new history vector ht . The history vector ht can be
thought of as a representation or summary of the information we’ve aggregated from our exploration of the graph thus far.
The rank network fr (.; θ r ) uses this to decide which types of nodes are more “interesting" and should thus be prioritized in
future exploration. Likewise, the classification network fc (.; θc ) uses ht to make a prediction on the graph label.
this, on large graphs, it may be better to deploy multiple agents                                          1: procedure Step(rt −1 ∈ RR , A ∈ N N ×N , D ∈ R N ×D , c t −1 )
with each agent exploring a relatively small neighborhood rather                                           2:     a ← A[c t −1 , : ]
than having one agent traverse the graph for a long period. To                                             3:     T ← τ (D)               ▷ T ∈ RN ×R is a matrix of one-hot
integrate information, we can augment the architecture with a                                                 row vectors indicating node types; we assume that type can be
shared external memory [27]. Additionally, a network conditioned                                              derived from node attributes.
on the current history vector can be trained to allow the model                                            4:     p ← (T rt −1 )⊤
to selectively save information to memory. This will allow the                                             5:     p←p⊙a
model to store information that is useful for graph classification                                         6:
                                                                                                                       Í
                                                                                                                  d ← i pi
(e.g., discriminative subgraphs).                                                                          7:     p ← p ⊙ d1
   Actions: Given the new history vector that captures what the                                            8:    c t ∼ Multinomial(π = p)         ▷ Sample a neighbor from
agent has seen so far, the agent performs two actions at each time                                            multinomial distribution parameterized by p.
step. First, it predicts the label of the input graph lˆt = arg max P(y =                                  9:     return D[c t , : ], c t
                                                                                       i
i | fc (ht ; θc )) from the softmax output of the classification network                                  10: end procedure
conditioned on ht . Second, it uses the rank network to generate                                         Algorithm 1: Procedure to pick a neighbor to move to. The
the rank vector rt = fr (ht ; θ r ) that will help “steer" exploration in                                algorithm is biased towards picking neighbors whose types
the next step by ranking the importance of different types of nodes.                                     have higher ranks in rt −1 . Here, ⊙ represents element-wise
     Primarily, the rank vector’s job is to encode the importance of                                     multiplication and ▷ denotes the start of a comment.
different types of nodes. However, we can augment it to include
additional actions such as one for deciding when to stop further                                            Reward: In the typical reinforcement learning setting, the agent
exploration if the agent is confident it has enough information to                                       receives new information x t +1 from the environment and a reward
classify the graph correctly. Another possible action is the one that                                    signal r t +1 after taking an action at each time step t. The goal of
allows the model to transfer its current internal information to a                                       the agent is to maximize the reward it receives which is usually
                                                                                                         quite sparse and delayed: R = Tt=1 r t . In our setting, xt +1 = dc t +1
                                                                                                                                          Í
memory component.
and the reward is given only at the end, where rT = 1 if the model                      instead. This provides us with an estimate that is equal in expecta-
classified the graph correctly and rT = −1 otherwise. Hence R = rT .                    tion to the original formulation but with possibly lower variance
   Under this formulation, we have what can be considered a Par-                        [18]. Here bti = fb (s 1:t
                                                                                                               i ; θ ) = f (hi ; θ ) captures the cumulative re-
                                                                                                                    b     b t b
tially Observable Markov Decision Process (POMDP). In this setting,                     ward we can expect to receive for a state hit . The term (γ T −t R i −bti ),
we only obtain partial information about the graph or our envi-                         or the advantage of choosing an action, allows us to increase the
ronment through our interactions with it at each time step. As in                       log-probability of actions that resulted in a much larger expected
[18], our goal is to learn a policy π ((rt , lˆt |s 1:t ; θ )) with parameters          cumulative reward and to decrease the log-probability of actions
θ that maps the sequence of our past interactions with the envi-                        that resulted in the reverse. We can train the parameter θb of fb by
ronment s 1:t = x1 , r1 , lˆ1 , · · · , xt −1 , rt −1 , lˆt −1 , xt to a distribution   reducing the mean squared error of R i − bti .
over actions for the current time step t. In other words, given the                        Finally, we use cross entropy loss to train the classification net-
history of past interactions as summarized in the history vector ht ,                   work fc (.; θc ) by maximizing log π (lT |s 1:T ; θc ), where lT is the true
the classification network fc (.; θc ) and the rank network fr (.; θ r ) –              label of the input graph G. As in [18], we use this hybrid loss formu-
or our policy networks – learn to generate actions that maximize                        lation where the rank network fr is trained at each time step using
reward.                                                                                 REINFORCE and the classification network fc and the baseline net-
                                                                                        work fb are trained using the classical approach from supervised
3.2     Training                                                                        learning.
Together, the core LSTM network, the step network, and the rank
network work in conjunction with each other to form the policy                          3.3    Space Complexity
of the agent. We learn the parameters θ = {θh , θ s , θ r } of these                    Let △ G be the max node degree for graph G and D be the dimension
networks to maximize the total reward the agent can expect to                           of the node attribute vector. Since the agent only moves to one of
obtain. Since each specific policy for the agent induces a distribution                 the current node’s neighbors at each time step, we only need to
over the possible interaction sequences s 1:T , we want to train our                    store a △ G × D matrix containing the attributes of neighboring
policy to maximize the reward under the generated distribution:                         nodes at any given time. After taking a step to a new node, the
J (θ ) = EP (s1:T ;θ ) [R].                                                             attribute matrix for the new set of neighbors can be fetched from
    It is a non-trivial task to maximize J exactly as we are dealing                    disk. Ignoring the space needed to store r, s, h, c, and the parameters
with a very large, and possibly infinite, number of possible interac-                   of our model, which are constant and negligible, our model has a
tion sequences. However, since we frame the problem as a POMDP,                         space complexity of O(△ G D) which is quite small in practice.
we are able to obtain a sample approximation of the gradient of J
by using the technique introduced by [32] as shown in [18]. This is
                                                                                        3.4    Initialization
given by
                                                                                        For each new instance, we initialize the start vertex c 0 by selecting a
                       M T −1
                  1 ÕÕ                                                                  random node in the input graph and the rank vector r0 is initialized
         ∇θ J ≈             ∇ log π (rit [τ (c ti +1 )]|s 1:t
                                                          i
                                                              ; θ )γ T −t R i    (1)
                  M i=1 t =1 θ                                                          to the uniform distribution.

where the s i ’s are the interaction sequences from running the agent
under the current policy for i = 1, · · · , M episodes, γ ∈ (0, 1] is
                                                                                        3.5    Attention with Memory
a discount factor that allows us to attribute more significance to                      When predicting the label of an input graph, one may choose to
actions performed closer to time T or when the prediction was                           average the softmax output of several runs by initializing multiple
made, and τ (c ti +1 ) is a function that maps a node to its type. The                  agents at different starting locations in the graph. In this case, we
intuition behind equation 1, which is also known as the REINFORCE                       can view each agent as one classifier in an ensemble where we
rule, is as follows. We run the agent with the current policy to                        predict by voting. While averaging the predictions of several agents
obtain samples of interaction sequences. The parameters θ are                           can certainly improve classification performance, our model is still
then adjusted to increase the log-probability or rank of the type of                    at a disadvantage against methods that integrate information from
nodes that were frequently selected during episodes that resulted                       the entire graph. This is because each agent makes a prediction
in a correct prediction. Training the policy this way allows us to                      independently, using only the information it gathered from a local
increase the chance that the agent will choose to take a step towards                   area within the graph.
a particular type of node the next time it finds itself in a similar                       To remedy this, we introduce a variant of our model with a shared
state. To compute ∇θ log π (rit [τ (c ti +1 )]|s 1:t
                                                 i ; θ ), we simply compute             external memory component that can store information from mul-
the gradient of our network at each time step, this can be done                         tiple agents. In this architecture, each agent i for i = 1, · · · , n stores
using standard backpropagation [31]. Note that we only adjust the                       information in a local memory component pi , these are then com-
log-probabilities for t = 1, · · · ,T − 1 since the rank vector rt in the               bined to form the shared memory m that the classification network
last step is no longer used.                                                            uses to make a single prediction. In the simplest case, pi = hTi ,
   Since the gradient estimate in Equation 1 may exhibit high vari-                     which means we use the final history vector as each agent’s local
ance, one may choose to estimate ∇θ J via                                               memory. However, not all parts of an agent’s walk through the
              M T −1
                                                                                        graph may yield equally important information. To allow the model
         1 ÕÕ                                                                           to retain only information useful to the task we set pi = Tj=1 u ij hij ,
                                                                                                                                                        Í
                   ∇ log π (rit [τ (c ti +1 )]|s 1:t
                                                 i
                                                     ; θ )(γ T −t R i − bti )    (2)
         M i=1 t =1 θ                                                                   where the u ij ’s are the softmaxed output of fu (hij ; θu ) which decides
  agent 1                                                                                                                                                                                                                             agent n
                                               [1,0,0,1,1]                                                 [1,0,0,1,1]                                                            [1,0,0,1,1]                                                                                [1,0,0,1,1]                                                      [1,0,0,1,1]                                                            [1,0,0,1,1]
                                          c0
                            [1,1,0,1,0]                                                      [1,1,0,1,0]                                                            [1,1,0,1,0]                                                                                [1,1,0,1,0]                                                      [1,1,0,1,0]                                                            [1,1,0,1,0]
                                                       [0,0,0,0,1]                                                       [0,0,0,0,1]                                                            [0,0,0,0,1]                                                                                [0,0,0,0,1]                                                      [0,0,0,0,1]                                                            [0,0,0,0,1]
  r0                                                                       r1                       c1                                          rT-1                                                  [0,1,0,0,0]                     r0                             c0                                       r1                                                                   rT-1               cT-1
                                         [1,0,0,0,1]                                                       [1,0,0,0,1]                                                            [1,0,0,0,1]                                                                                [1,0,0,0,1]                                        c1            [1,0,0,0,1]                                                            [1,0,0,0,1]
                           [0,0,0,1,0]
                                                             [0,1,0,0,0]
                                                                                             [0,0,0,1,0]
                                                                                                                              [0,1,0,0,0]                                                               cT-1                                                                                    [0,1,0,0,0]                                                      [0,1,0,0,0]                                                           [0,1,0,0,0]
                                                                                                                                                                    [0,0,0,1,0]                                                                                [0,0,0,1,0]                                                      [0,0,0,1,0]                                                            [0,0,0,1,0]
                                         ! = {A!, D!}                                                       ! = {A!, D!}                                                                                                                                                      ! = {A!, D!}                                                      ! = {A!, D!}
                                                                                                                                                                 ! = {A!, D!}                                                                                                                                                                                                                                         ! = {A!, D!}



                 "#(. ; '#)                                                          "#(. ; '#)                                                             "#(. ; '#)                                                                               "#(. ; '#)                                                         "#(. ; '#)                                                             "#(. ; '#)


            s1                                                                  s2                                                                     sT                                                                                       s1                                                                 s2                                                                     sT
                                                  h1                                                                h2                                                                     hT                                                                                         h1                                                               h2                                                                     hT
   h0            "ℎ(. ; 'ℎ)                                                          "ℎ(. ; 'ℎ)                                             …               "ℎ(. ; 'ℎ)                                                   …             h0            "ℎ(. ; 'ℎ)                                                         "ℎ(. ; 'ℎ)                                             …               "ℎ(. ; 'ℎ)




                           "*(. ; '*)             "+(. ; '+)                                "*(. ; '*)             "+(. ; '+)                                      "*(. ; '*)             "+(. ; '+)                                                           "*(. ; '*)            "+(. ; '+)                                "*(. ; '*)             "+(. ; '+)                                      "*(. ; '*)             "+(. ; '+)


                                                       r1                                                                r2                                                                     rT                                                                                         r1                                                               r2                                                                     rT
                 softmax       u1                                                                 u2                                                                     uT                                                                          softmax       u1                                                                u2                                                                     uT



                                                                                                                                                                                                                          ,

                                                                                                                                                                                                                    m
                                                                                                                   average      ,                                                                                                                                                                                                                     average
                                                                                                                                                                                                                                                                                                                                                                   ,
                                                                                                                                                                                                                        "-(. ; '-)
                                                                                                                     p1                                                                                                                                                                                                                                 pn

                                                                                                                                                                                                                           ./!
Figure 4: The attention model with memory. Multiple agents can be initialized (in parallel) at different starting nodes allowing
each agent to explore a different part of the graph. We use weighted pooling to combine each agent’s history vectors, giving
us an agent’s local memory. The outputs of fu (.; θu ), or the ui ’s, are normalized using softmax and multiplied with their corre-
sponding history vectors. The intuition here is to train fu (.; θu ) to assign more importance to useful information. To integrate
information, we average the local memories of all the agents to get the shared memory: m = n1 ni=1 pi – which is used to make
                                                                                                   Í
a prediction on the graph label. For brevity, we omit the superscripts for h, u, r, s, and c.
how useful a particular “piece of memory" is. In other words, we do                                                                                                                                                                  initially, the rank network assigns more or less equal importance
weighted pooling to obtain our local memory. This can be viewed                                                                                                                                                                      to the five types of nodes. However, after some time, it learns
as another form of attention. Finally, to integrate information from                                                                                                                                                                 to prioritize the nodes of types C or E. This guarantees that the
multiple agents, we simply set m = n1 ni=1 pi . This modification
                                         Í
                                                                                                                                                                                                                                     agent will prioritize exploration in the right direction, giving the
allows us to integrate information from various regions in the                                                                                                                                                                       model enough information to classify the graphs correctly in a
graph and is especially helpful if the graph is large and we only                                                                                                                                                                    small number of steps.
take a small number of steps T . Note that each agent’s exploration
is still guided by the attention mechanism proposed earlier. The                                                                                                                                                                     4.2              Experimental Setup
architecture of attention with memory is shown in Figure 4.                                                                                                                                                                              4.2.1 Data. We evaluated our proposed method on the binary
   Various modifications can be made to this architecture. For in-                                                                                                                                                                   classification task using five real-world graph datasets: HIV, NCI-
stance, we can choose to condition the output of the rank network                                                                                                                                                                    1, NCI-33, NCI-83, and NCI-123 [16]. These datasets have been
on the local memory or even the shared external memory. Addi-                                                                                                                                                                        made publicly available by the National Cancer Institute (NCI)1
tional actions can also be introduced to allow the model to modify                                                                                                                                                                   [34]. Since the molecular structures in the datasets were encoded
or rewrite the shared memory; also, it isn’t difficult to imagine                                                                                                                                                                    using the SMILES format [29], we used the RDKit2 package to
including an action that allows the agent to stop exploration if it                                                                                                                                                                  convert each string into its corresponding graph. We used the
has already gained enough information to make a prediction. In                                                                                                                                                                       same package to extract the following information for each node
this work, however, we deliberately choose to test on the simplest                                                                                                                                                                   (i.e. atom) to use as node attributes: atom element, node degree,
version because our goal is to show the usefulness of attention.                                                                                                                                                                     total number of attached hydrogens, the implicit valence, and atom
                                                                                                                                                                                                                                     aromaticity. Atom element was used to label or assign types to
4 EXPERIMENTS                                                                                                                                                                                                                        the nodes. The graph class labels indicate the anti-cancer property
4.1 Motivating Example                                                                                                                                                                                                               (active or negative) of each molecule. The datasets are all highly
                                                                                                                                                                                                                                     imbalanced with far more negative samples than positive ones. We
Before we consider the details of our main experimental setup, we
                                                                                                                                                                                                                                     follow the methodology in previous work [16, 34] and randomly
introduce a simple motivating example that shows how attention
                                                                                                                                                                                                                                     extract a balanced subset (500) for each dataset.
can be used to guide an agent towards more relevant regions in
the graph. For this toy example, we generated a small dataset of                                                                                                                                                                        4.2.2 Compared Methods. In order to demonstrate the effec-
random graphs. We embedded several patterns or subgraphs in the                                                                                                                                                                      tiveness of our proposed approach, we compare it against several
generated graphs, two of which were the 3-paths A − B − C − D,                                                                                                                                                                       baseline methods, all of which utilize the entire graph for feature
and A − B − E − D. The former pattern was embedded primarily                                                                                                                                                                         extraction. To the best of our knowledge, this is the first work on
onto positive samples while the latter was included in negative                                                                                                                                                                      attention with graphs so we compare against baselines that observe
samples. In Figure 5, we show the output of the rank network,                                                                                                                                                                        the entire graph. We would like to emphasize that our proposed
over time, when it is given the history vector h1 capturing the                                                                                                                                                                      1 https://www.cancer.gov/
initial step onto the node of type B. It is interesting to note that,                                                                                                                                                                2 http://www.rdkit.org/
Table 1: Summary of experimental results: “average accuracy ± SD (rank)". The “ave. rank" column shows the average rank of
each method. The lower the average rank, the better the overall performance of the method.
                                                                                      dataset                                              ave.
                          method                                                                                                           rank
                                         HIV                        NCI-1             NCI-33             NCI-83             NCI-123
                      Agg-Attr     69.58 ± 0.03 (4)            64.79 ± 0.04 (4)   61.25 ± 0.03 (6)   58.75 ± 0.05 (6)   60.00 ± 0.02 (6)   5.2
                      Agg-WL       69.37 ± 0.03 (6)            62.71 ± 0.04 (6)   67.08 ± 0.04 (5)   60.62 ± 0.02 (4)   62.08 ± 0.03 (5)   5.2
                      Kernel-SP    69.58 ± 0.04 (4)            65.83 ± 0.05 (3)   71.46 ± 0.03 (1)   60.42 ± 0.04 (5)   62.92 ± 0.07 (4)   3.4
                      Kernel-Gr    71.88 ± 0.05 (3)            67.71 ± 0.06 (1)   69.17 ± 0.03 (3)   66.04 ± 0.03 (3)   65.21 ± 0.05 (2)   2.4
                      GAM          74.79 ± 0.02 (2)            64.17 ± 0.05 (5)   67.29 ± 0.02 (4)   67.71 ± 0.03 (2)   64.79 ± 0.02 (3)   3.2
                      GAM-mem      78.54 ± 0.04 (1)            67.71 ± 0.04 (1)   69.58 ± 0.02 (2)   70.42 ± 0.03 (1)   67.08 ± 0.03 (1)   1.2

                                                                                                  label the nodes in the graph by concatenating the categorical
                                                                                                  attributes.
                                                                                                • Kernel-Gr: As in [34], we also compare against the graphlet
                                                                                                  kernel which measures graph similarity by counting the
                                                                                                  number of different graphlets. Here, we evaluate against the
                                                                                                  3-graphlet kernel and nodes are labeled as above.
                                                                                                • GAM: Our proposed approach which uses attention to steer
                                                                                                  the walk of an agent on an input graph.
                                                                                                • GAM-mem: Proposed approach with external memory. Note
                                                                                                  that given a budget of T for GAM , GAM-mem with n agents
                                                                                                  has access to same amount of information if each agent is
                                                                                                  constrained to take Tn steps.

                                                                                             We used a logistic regression (LR) classifier with the first two
                                                                                          baselines. To reduce overfitting, we applied ℓ1 and ℓ2 regularization
    A             B          C     D     A            B            E        D
   positive sub-pattern
                                                                                          and used a grid search over {0.01, 0.1, 1.0} to select the ideal regu-
                                        negative sub-pattern
                                                                                          larization penalty. Furthermore, we also did a grid search over the
Figure 5: Rank values, over time, in the generated rank vec-                              number of iterations for the WL algorithm, we tested over {2, 3, 4}.
tor r1 when the rank network is given h1 encoding informa-                                For a fair comparison, we limited the classification network for
tion from an initial step onto node B. Higher rank value sig-                             both our methods to a single softmax layer to make it equivalent
nifies more importance.                                                                   to LR. We also limited the number of hidden layers in all other
model (GAM) uses attention to explore only a portion of the input                         networks of our model to a single layer, whenever possible. For the
graph, this puts our model at a disadvantage since it only has partial                    graph-kernel based approaches, we used an SVM classifier using
observability. Since the main goal is to show the viability of using                      the precomputed kernel generated by each approach. Here, we did
attention, we limit the architecture of our tested models to simple                       a grid search over C = {0.01, 0.1, 1.0}. We used a vector in R200 for
ones (more detail below). The compared methods are summarized                             Agg-WL and limited the size of the LSTM history vector to this size
below.                                                                                    as well. In particular, we tried size = {156, 200}. We also tried the
                                                                                          following sizes for the first and second hidden layers, respectively,
    • Agg-Attr: Given an attributed graph, one simple way to con-                         of the step network: (128, 164), and (64, 128).
      struct a feature vector is to get the component-wise average                           Since we did not find any noticeable change in the performance
      of the attribute vectors of all the nodes in the graph.                             of GAM when increasing the following parameters, we fixed their
    • Agg-WL: The first approach captures information from node                           values. We set the number of steps T = 12 and the number of
      attributes. However, it completely ignores the graph’s struc-                       samples M = 20. M is also the number of agents we run on each
      tural information. The second method uses the Weisfeiler-                           graph for prediction. For GAM-mem, we did a grid search over
      Lehman (WL) algorithm [25] to calculate new node attributes                         T = {12, 25}, and M = {5, 10}. We use the Adam algorithm for
      that capture the local neighborhood of each node. The al-                           optimization [14] and fix the initial and final learning rates to 10−3
      gorithm works by iteratively assigning a new attribute to                           and 10−6 , respectively. We also did not use discounted reward as
      each node by computing a hash of the attributes of neigh-                           there was no noticeable gain, setting γ = 1. Finally, we limit the
      boring nodes. We simply average the new attributes after                            training of our methods to 200 epochs and applied early stopping
      running the WL algorithm to use as feature vector used for                          using a validation set.
      prediction.
    • Kernel-SP: As in [34], we compare against the shortest path
      (SP) kernel which measures the similarity of a pair of graphs
                                                                                          4.3     Classification Results
      by comparing the distance of the shortest paths between                             Table 1 shows the average classification accuracy, over 5-fold cross-
      nodes in the graphs. Since we use attributed graphs, we                             validation, of the compared methods. From the results, we can see
Table 2: Performance of the baselines when we restrict their setting to that of GAM where they are given 20 randomly selected
partial snapshots of each graph and have to predict by voting. The column “full" indicates the performance when the entire
graph is seen and “partial" shows the performance when only parts of the graph is seen. “Diff." is the difference in performance,
a ↓ means that performance deteriorated when only partial information is available and ↑ shows increase in performance.
                                                                                 dataset
  method               HIV                          NCI-1                        NCI-33                         NCI-83                       NCI-123
              full   partial     diff.      full   partial     diff.      full   partial     diff.      full   partial     diff.      full   partial     diff.
  Agg-Attr   69.58   64.17     05.41 (↓)   64.79   59.58     05.21 (↓)   61.25   58.54     02.71 (↓)   58.75   62.71     03.96 (↑)   60.00   57.50     02.50 (↓)
  Agg-WL     69.37   56.04     13.33 (↓)   62.71   51.46     11.25 (↓)   67.08   49.79     17.29 (↓)   60.62   51.46     09.16 (↓)   62.08   52.29     09.79 (↓)
  GAM          -     74.79         -         -     64.17         -         -     67.29         -         -     67.71         -         -     64.79         -

that our proposed model is always among the top-2 in terms of per-
formance on all tested datasets. In particular, the attention model
with memory performs the best on four of the five datasets and
comes in at second on the fifth dataset (NCI-33). In every single
case, GAM-mem outperforms GAM which shows that adding an
external memory to integrate information from various locations
is beneficial. However, we find that GAM still performs respectably
against the compared baselines and in fact comes in second on two
of the tested datasets. We also find that GAM outperforms Agg-Attr
and Agg-WL in almost every single case, which is remarkable since
each agent in GAM only has access to a portion of the graph while
the latter two have access to the entire graph. In our experiments,
we find that the first two baselines perform the worst, almost al-
ways performing the worst on all the datasets. The kernel-based
approaches are better, with the graphlet-based approach being supe-                Figure 6: Average runtime when agents are run in parallel
rior. It is able to outperform GAM slightly. However, GAM-mem is                   versus sequentially. Here we show the runtime for doing pre-
consistently the best performer on all the datasets that were tested.              diction on a mini-batch of 32 graphs with T = 100.

   4.3.1 Applying Random Attention. Our experiments show that                      found that both models already performed relatively well when
the attention model is competitive against baselines that observe                  T ≥ 3, in some cases being only 5-6% worse than the best accuracy.
the entire graph while our model is limited to seeing a portion of                 This may be because molecular graphs are fairly small in size. We
the graph. To demonstrate the effectiveness of attention further, we               found that GAM-mem, in general, benefits more from an increase
ran another experiment where we restrict the first two baselines                   in the size of T which may be due to the fact that we are using
to the setting of GAM. It is a straightforward modification since                  weighted pooling of the history vectors so the model can support
the methods also use the graph attribute vectors. However, the                     longer walks.
baselines do not have a concept of attention, so we use random
attention where we sample 20 subgraphs from each graph using                       4.5      Parallel Execution of Agents
a random-walk of length 12. This limits the information available                  One advantage of our model is the ability to execute multiple agents
to the baselines to that which is available to GAM since we fixed                  in parallel during prediction time. This is particularly useful when
M = 20 and T = 12.                                                                 the graph is large and we need multiple agents to explore different
   Table 2 shows the result of the baselines when they only observe                parts of the graph. For instance, in the task of malware classification
a random portion of each graph. It is clear that the performance de-               on function-call graphs the graphs have been known to contain up
teriorates for both methods, with Agg-WL showing a more marked                     to ~37,000 nodes [13]. Also, recall that in the proposed model the
difference in performance. This is with the exception of Agg-Attr                  agents are not required to have access to the entire graph. In fact,
on NCI-83. In fact, we can see that the performance of Agg-WL                      at any time-point t, the model only needs access the current node
drops so drastically that it performs almost no better than random                 c t and its neighbors along with their attributes. If the graph is too
guessing on four of the five datasets (NCI-1, NCI-33, NCI-83, and                  large to load into memory, this information can be accessed on the
NCI-123). This shows that attention can help us examine parts of                   fly (e.g., from a database).
the graph that are relevant.                                                           Since each agent in GAM can make an independent prediction,
4.4    Parameter Study                                                             the agents can be run in parallel. The only step that needs to be
We study the effect of varying step sizes T on performance of                      done in the end is to average the predictions of all the agents. For
both GAM and GAM-mem. For each of the 5 datasets, we fixed                         GAM-mem, since the history vectors of all the agents are combined,
all other parameters to the ones that yielded the best results and                 we wait for all the agents to take T steps before we combine the
varied T = {1, 3, · · · , 15, 18}. In both cases, accuracy increased as            information. However, the individual agents in GAM-mem can
we increased the number of steps with T = 12 giving fairly good                    still explore the graph in parallel. Figure 6 shows the difference
performance on all datasets on both methods. Surprisingly, we                      in runtime of our method when multiple agents are executed in
parallel versus in a single process. The experiments are conducted                       [11] David K. Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Bom-
on a Linux machine with 48 CPU cores and 160GB of RAM. We see                                 barell, Timothy Hirzel, Alan Aspuru-Guzik, and Ryan P. Adams. 2015. Convolu-
                                                                                              tional networks on graphs for learning molecular fingerprints. In Proceedings of
here that the parallelized version of GAM-mem is slightly slower.                             the Twenty-Eight Annual Conference on Neural Information Processing Systems.
This can be expected since each agent’s local memory needs to                                 2224–2232.
                                                                                         [12] Felix Gers, Jurgen Schmidhuber, and Fred Cummins. 1999. Learning to forget: con-
be combined to form the shared memory. However, it is clear that                              tinual prediction with LSTM. In Proceedings of the Tenth International Conference
parallelization helps keep the runtime close to constant for both                             on Artificial Neural Networks. 850–855.
methods as the bottleneck is in the attention-guided walk.                               [13] Xin Hu, Tzi cker Chiueh, and Kang G. Shin. 2009. Large-scale malware indexing
                                                                                              using function-call graphs. In Proceedings of the Sixteenth ACM Conference on
                                                                                              Computer and Communications Security. 611–620.
5    CONCLUSION                                                                          [14] Diederik P. Kingma and Jimmy Lei Ba. 2015. Adam: A method for stochastic
                                                                                              optimization. In Proceedings of the Third International Conference on Learning
In this work, we propose to study the problem of attention-based                              Representations.
graph classification and introduced GAM, a general RNN-based                             [15] Risi Kondor and Horace Pan. 2016. The Multiscale Laplacian Graph Kernel.
                                                                                              In Proceedings of the Twenty-Ninth Annual Conference on Neural Information
framework that uses attention to learn data-driven features for at-                           Processing Systems. 2982–2990.
tributed graphs. The model uses attention to process task-relevant                       [16] Xiangnan Kong, Wei Fang, and Philip S. Yu. 2011. Dual active feature and sample
parts of the graph (partial observability) and has an external mem-                           selection for graph classification. In Proceedings of the Seventeenth ACM SigKDD
                                                                                              International Conference on Knowledge Discovery and Data Mining. 654–662.
ory to integrate information from various parts of the graph. The                        [17] Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective Ap-
attention mechanism in our model is easily parallelizable. Empirical                          proaches to Attention-based Neural Machine Translation. In Proceedings of the
results show that the method can outperform methods that observe                              Thirteenth Conference on Empirical Methods in Natural Language Processing. 1412–
                                                                                              1421.
the entire graph.                                                                        [18] Volodymyr Mnih, Nicolas Heess, Alex Graves, and Koray Kavukcuoglu. 2014.
   There are a lot of interesting directions for future work. We in-                          Recurrent models of visual attention. In Proceedings of the Twenty-Seventh Annual
                                                                                              Conference on Neural Information Processing Systems. 2204–2212.
tend to study the model using more expressive node typing strate-                        [19] Mathias Niepert, Mohamed Ahmed, and Konstantin Kutzkov. 2016. Learning
gies. We would also like to experiment with an extension of the                               Convolutional Neural Networks for Graphs. In Proceedings of the Thirty-Third
model with a more sophisticated external memory (e.g., making                                 International Conference on Machine Learning. 2014–2023.
                                                                                         [20] Giannis Nikolentzos, Polykarpos Meladianos, and Michalis Vazirgiannis. 2017.
memory rewritable, and using memory to condition the output                                   Matching Node Embeddings for Graph Similarity. In Proceedings of the Thirty-First
of the rank network). Finally, it would be interesting to test more                           AAAI Conference on Artificial Intelligence. 2429–2435.
flexible architectures for LSTM like Tree-LSTMs that seem more                           [21] Shirui Pan, Jia Wu, Xingquan Zhu, and Chengqi Zhang. 2015. Graph Ensemble
                                                                                              Boosting for Imbalanced Noisy Graph Stream Classification. IEEE Transactions
natural for graphs.                                                                           on Cybernetics 45, 5 (2015), 940–954.
                                                                                         [22] Aaditya Prakash, Siyuan Zhao, Sadid A. Hasan, Vivek Datla, Kathy Lee, Ashequl
                                                                                              Qadir, Joey Liu, and Oladimeji Farri. 2017. Condensed Memory Networks for
ACKNOWLEDGMENTS                                                                               Clinical Diagnostic Inferencing. In Proceedings of the Thirty-First AAAI Conference
This work is supported in part by the National Science Foundation                             on Artificial Intelligence. 3274–3280.
                                                                                         [23] David Rogers and Mathew Hahn. 2010. Extended-connectivity fingerprints.
through grants IIS-1718310, MRI-1626236.                                                      Journal of Chemical Information and Modeling 50, 5 (2010), 742–754.
                                                                                         [24] Ryan A. Rossi, Rong Zhou, and Nesreen K. Ahmed. 2017. Estimation of graphlet
                                                                                              statistics. In arXiv preprint arXiv:1701.01772.
REFERENCES                                                                               [25] Nino Shervashidze, Pascal Schweitzer, Erik Jan van Leeuwen, Kurt Mehlhorn,
 [1] Nesreen K. Ahmed, Ryan A. Rossi, Rong Zhou, John Boaz Lee, Xiangnan Kong,                and Karsten M. Borgwardt. 2011. Weisfeiler-Lehman Graph Kernels. Journal of
     Theodore L. Willke, and Hoda Eldardiry. 2018. Learning Role-based Graph                  Machine Learning Research 12 (2011), 2539–2561.
     Embeddings. In arXiv preprint arXiv:1802.02896.                                     [26] Nino Shervashidze, S. V. N. Vishwanathan, Tobias Petri, Kurt Mehlhorn, and
 [2] Lars Backstrom and Jure Leskovec. 2011. Supervised random walks: predict-                Karsten M. Borgwardt. 2009. Efficient graphlet kernels for large graph comparison.
     ing and recommending links in social networks. In Proceedings of the Fourth              In Proceedings of the Twelfth International Conference on Artificial Intelligence and
     International Conference on Web Search and Web Data Mining. 635–644.                     Statistics. 488–495.
 [3] Jie Bao, Tianfu He, Sijie Ruan, Yanhua Li, and Yu Zheng. 2017. Planning Bike        [27] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. 2015. End-
     Lanes based on Sharing-Bikes’ Trajectories. In Proceedings of the Twenty-Second          To-End memory networks. In Proceedings of the Twenty-Eight Annual Conference
     ACM SigKDD International Conference on Knowledge Discovery and Data Mining.              on Neural Information Processing Systems. 2440–2448.
 [4] Karsten M. Borgwardt and Hans-Peter Kriegel. 2005. Shortest-Path Kernels on         [28] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
     Graphs. In Proceedings of the Fifth IEEE International Conference on Data Mining.        Lio, and Yoshua Bengio. 2017. Graph Attention Networks. In arXiv preprint
     74–81.                                                                                   arXiv:1710.10903v2.
 [5] Karsten M. Borgwardt, Cheng Soon Ong, Stefan Schonauer, S. V. N. Vishwanathan,      [29] David Weininger. 1988. SMILES, a chemical language and information system.
     Alex J. Smola, and Hans-Peter Kriegel. 2005. Protein function prediction via             Journal of Chemical Information and Modeling 28 (1988), 31–36.
     graph kernels. Bioinformatics 21, 1 (2005), i47–i56.                                [30] Jason Weston, Sumit Chopra, and Antoine Bordes. 2014. Memory Networks. In
 [6] Duen Horng Chau, Carey Nachenberg, Jeffrey Wilhelm, Adam Wright, and                     Proceedings of the Second International Conference on Learning Representations.
     Christos Faloutsos. 2011. Polonium: Tera-Scale Graph Mining for Malware             [31] Daan Wierstra, Alexander Forster, Jan Peters, and Jurgen Schmidhuber. 2007.
     Detection. In Proceedings of the Eleventh SIAM International Conference on Data          Solving deep memory POMDPs with recurrent policy gradients. In Proceedings of
     Mining. 131–142.                                                                         the Seventeenth International Conference on Artificial Neural Networks. 697–706.
 [7] Kan Chen, Jiang Wang, Liang-Chieh Chen, Haoyuan Gao, Wei Xu, and Ram                [32] Ronald J. Williams. 1992. Simple statistical gradient-following algorithms for
     Nevatia. 2015. ABC-CNN: An Attention Based Convolutional Neural Network                  connectionist reinforcement learning. Machine Learning 8, 3 (1992), 229–256.
     for Visual Question Answering. In arXiv preprint arXiv:1511.05960v2.                [33] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan
 [8] Dong-Yeon Cho, Yoo-Ah Kim, and Teresa M. Przytycka. 2012. Chapter 5: Network             Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. 2015. Show, Attend and
     Biology Approach to Complex Diseases. PLOS Computational Biology 8 (2012),               Tell: Neural Image Caption Generation with Visual Attention. In Proceedings of
     e1002820.                                                                                the Thirty-Second International Conference on Machine Learning. 2048–2057.
 [9] Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F. Stewart, and Jimeng         [34] Pinar Yanardag and S. V. N. Vishwanathan. 2015. Deep Graph Kernels. In Pro-
     Sun. 2017. GRAM: Graph-based Attention Model for Healthcare Representa-                  ceedings of the Twenty-First ACM SigKDD International Conference on Knowledge
     tion Learning. In Proceedings of the Twenty-Third ACM SigKDD International               Discovery and Data Mining. 1365–1374.
     Conference on Knowledge Discovery and Data Mining. 787–795.                         [35] Jingyuan Zhang, Bokai Cao, Sihong Xie, Chun-Ta Lu, Philip S. Yu, and Ann B.
[10] Gabriel Dulac-Arnold, Richard Evans, Hado van Hasselt, Peter Sunehag, Timothy            Ragin. 2016. Identifying Connectivity Patterns for Brain Diseases via Multi-side-
     Lillicrap, Jonathan Hunt, Timothy Mann, Theophane Weber, Thomas Degris, and              view Guided Deep Architectures. In Proceedings of the Sixteenth SIAM Interna-
     Ben Coppin. 2015. Deep reinforcement learning in large discrete action spaces.           tional Conference on Data Mining. 36–44.
     In arXiv preprint arXiv:1512.07679.

