# Cooperative Graph Neural Networks Unknown 2025

> Source: `Cooperative_Graph_Neural_Networks_Unknown_2025.pdf`

---

                                                                           Cooperative Graph Neural Networks


                                                           Ben Finkelshtein 1 Xingyue Huang 1 Michael Bronstein 1 İsmail İlkan Ceylan 1


                                                                  Abstract                                      The message-passing paradigm has been very influential in
                                                                                                                graph ML, but it also comes with well-known limitations
                                             Graph neural networks are popular architectures                    related to the information flow on a graph, pertaining to
                                             for graph machine learning, based on iterative                     long-range dependencies (Dwivedi et al., 2022). In order to




arXiv:2310.01267v2 [cs.LG] 9 Jun 2024
                                             computation of node representations of an input                    receive information from k-hop neighbors, a network needs
                                             graph through a series of invariant transforma-                    at least k layers, which typically implies an exponential
                                             tions. A large class of graph neural networks                      growth of a node’s receptive field. The growing amount of
                                             follow a standard message-passing paradigm: at                     information needs then to be compressed into fixed-sized
                                             every layer, each node state is updated based on                   node embeddings, possibly leading to information loss, re-
                                             an aggregate of messages from its neighborhood.                    ferred to as over-squashing (Alon & Yahav, 2021). Another
                                             In this work, we propose a novel framework for                     well-known limitation related the information flow is over-
                                             training graph neural networks, where every node                   smoothing (Li et al., 2018): the node features can become
                                             is viewed as a player that can choose to either ‘lis-              increasingly similar as the number of layers increases.
                                             ten’, ‘broadcast’, ‘listen and broadcast’, or to ‘iso-
                                             late’. The standard message propagation scheme                     Motivation. Our goal is to generalize the message-passing
                                             can then be viewed as a special case of this frame-                scheme by allowing each node to decide how to propagate
                                             work where every node ‘listens and broadcasts’ to                  information from or to its neighbors, thus enabling a more
                                             all neighbors. Our approach offers a more flexible                 flexible flow of information. Consider the example depicted
                                             and dynamic message-passing paradigm, where                        in Figure 1, where the top row shows the information flow
                                             each node can determine its own strategy based on                  relative to the node u across three layers, and the bottom
                                             their state, effectively exploring the graph topol-                row shows the information flow relative to the node v across
                                             ogy while learning. We provide a theoretical anal-                 three layers. Node u listens to every neighbor in the first
                                             ysis of the new message-passing scheme which is                    layer, only to v in the second layer, and to nodes s and r
                                             further supported by an extensive empirical anal-                  in the last layer. On the other hand, node v listens to node
                                             ysis on synthetic and real-world data.                             w for the first two layers, and to node u in the last layer.
                                                                                                                To realize this scenario, each node should be able to decide
                                                                                                                whether or not to listen to a particular node at each layer: a
                                        1. Introduction                                                         dynamic and asynchronous message-passing scheme, which
                                                                                                                clearly falls outside of standard message-passing.
                                        Graph neural networks (GNNs) (Scarselli et al., 2009; Gori
                                                                                                                Approach. To achieve this goal, we regard each node as a
                                        et al., 2005) are a class of architectures for learning on graph-
                                                                                                                player that can take the following actions in each layer:
                                        structured data. Their success in various graph machine
                                        learning (ML) tasks (Shlomi et al., 2021; Duvenaud et al.,              • S TANDARD (S): Broadcast to neighbors that listen and
                                        2015; Zitnik et al., 2018) has led to a surge of different archi-         listen to neighbors that broadcast.
                                        tectures (Kipf & Welling, 2017; Xu et al., 2019; Veličković           • L ISTEN (L): Listen to neighbors that broadcast.
                                        et al., 2018; Hamilton et al., 2017). The vast majority of              • B ROADCAST (B): Broadcast to neighbors that listen.
                                        GNNs can be implemented through message-passing, where                  • I SOLATE (I): Neither listen nor broadcast.
                                        the fundamental idea is to update each node’s representation
                                        based on an aggregate of messages flowing from the node’s               When all nodes perform the action S TANDARD, we recover
                                        neighbors (Gilmer et al., 2017).                                        the standard message-passing. Conversely, having all the
                                                                                                                nodes I SOLATE corresponds to removing all the edges from
                                           1
                                             Department of Computer Science, University of Oxford. Cor-         the graph implying node-wise predictions. The interplay
                                        respondence to: name surname <{name.surname}@cs.ox.ac.uk>.              between these actions and the ability to change them dy-
                                        Proceedings of the 41 st International Conference on Machine            namically makes the overall approach richer and allows to
                                        Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by            decouple the input graph from the computational one and
                                        the author(s).                                                          incorporate directionality into message-passing: a node can

                                                                                                            1
                                             Cooperative Graph Neural Networks

          ℓ=0                                                 ℓ=1                                                    ℓ=2
  s                                                   s                                                      s
      u   v   w                                           u   v       w                                          u   v   w

  r                                                   r                                                      r

  s                                                   s                                                      s
      u   v   w                                           u   v       w                                          u   v   w

  r                                                   r                                                      r

Figure 1: Example information flow for nodes u, v. Top: information flow relative to u across three layers. Node u listens to
every neighbor in the first layer, but only to v in the second layer, and only to s and r in the last layer. Bottom: information
flow relative to v across three layers. The node v listens only to w in the first two layers, and only to u in the last layer.


only listen to those neighbors that are currently broadcast-          2. Background
ing, and vice versa. We can emulate the example from
Figure 1 by making u choose the actions ⟨L, L, S⟩, v and w            Graph Neural Networks. We consider simple, undirected
the actions ⟨S, S, L⟩, and s and r the actions ⟨S, I, S⟩.             attributed graphs G = (V, E, X), where X ∈ R|V |×d is a
                                                                      matrix of (input) node features, and xv ∈ Rd denotes the
Contributions. We develop a new class of architectures,               feature of a node v ∈ V . We focus on message-passing
dubbed cooperative graph neural networks (C O -GNNs),                 neural networks (MPNNs) (Gilmer et al., 2017) that encap-
where every node in the graph is viewed as a player that              sulate the vast majority of GNNs. An MPNN updates the
can perform one of the aforementioned actions. C O -GNNs                                              (0)
                                                                      initial node representations hv = xv of each node v for
comprise two jointly trained “cooperating” message-passing            0 ≤ ℓ ≤ L − 1 iterations based on its own state and the state
neural networks: an environment network η (for solving the            of its neighbors Nv as:
given task), and an action network π (for choosing the best                                                               
actions). Our contributions can be summarized as follows:                h(ℓ+1)
                                                                           v     = ϕ(ℓ) h(ℓ)
                                                                                          v ,ψ
                                                                                                (ℓ)
                                                                                                      h(ℓ)    (ℓ)
                                                                                                       v , {{hu | u ∈ Nv }}     ,
• We propose a novel message-passing mechanism, which
                                                                      where {{·}} denotes a multiset and ϕ(ℓ) and ψ (ℓ) are differ-
  leads to C O -GNN architectures that effectively explore
                                                                      entiable update and aggregation functions, respectively. We
  the graph topology while learning (Section 4).
                                                                      denote by d(ℓ) the dimension of the node embeddings at
• We provide a detailed discussion on the properties of C O -                                                         (L)
                                                                      iteration (layer) ℓ. The final representations hv of each
  GNNs (Section 5.1) and show that they are more expres-
                                                                      node v can be used for predicting node-level properties or
  sive than 1-dimensional Weisfeiler-Leman algorithm (1-                                                                       (L)
  WL) (Section 5.2), and better suited for long-range tasks           they can be pooled to form a graph embedding vector zG ,
  due to their adaptive nature (Section 5.3).                         which can be used for predicting graph-level properties. The
                                                                      pooling often takes the form of simple averaging, summa-
• Empirically, we focus on C O -GNNs with basic action
                                                                      tion, or element-wise maximum. Of particular interest to us
  and environment networks to carefully assess the virtue of
                                                                      are the basic MPNNs:
  the new message-passing paradigm. We first validate the                                                                  
  strength of our approach on a synthetic task (Section 6.1).          h(ℓ+1)
                                                                         v     = σ Ws(ℓ) h(ℓ)         (ℓ)      (ℓ)
                                                                                            v + Wn ψ {{hu | u ∈ Nv }}            ,
  Then, we conduct experiments on real-world datasets, and
  observe that C O -GNNs always improve compared to their                      (ℓ)        (ℓ)
                                                                      where Ws and Wn are d(ℓ) ×d(ℓ+1) learnable parameter
  baseline models, and yield multiple state-of-the-art results        matrices acting on the node’s self-representation and on the
  (Section 6.2 and Appendix C.3).                                     aggregated representation of its neighbors, respectively, σ
• We compare the trend of the actions on homophilic and               is a non-linearity, and ψ is either mean or sum aggregation
  heterophilic graphs (Section 7.1); visualize the actions            function. We refer to the architecture with mean aggregation
  on a heterophilic graph (Section 7.2); and ablate on the            as M EAN GNNs and to the architecture with sum aggrega-
  choices of action and environment networks (Section 7.3).           tion as S UM GNNs (Hamilton, 2020). We also consider
  We complement these with experiments related to ex-                 prominent models such as GCN (Kipf & Welling, 2017),
  pressive power (Appendix C.1), long-range tasks (Ap-                GIN (Xu et al., 2019) and GAT (Veličković et al., 2018).
  pendix C.2), and over-smoothing (Appendix C.5).
                                                                      Straight-through Gumbel-softmax Estimator. In our ap-
Additional details can be found in the appendix of this paper.        proach, we rely on an action network for predicting cat-
                                                                      egorical actions for the nodes in the graph, which is not

                                                                  2
                                            Cooperative Graph Neural Networks

differentiable and poses a challenge for gradient-based opti-       Finally, classical message passing updates the nodes in a
mization. One prominent approach to address this is given           fixed and synchronous manner, which does not allow the
by the Gumbel-softmax estimator (Jang et al., 2017; Mad-            nodes to react to messages from their neighbors individually.
dison et al., 2017) which effectively provides a differen-          This has been recently argued as yet another limitation of
tiable, continuous approximation of discrete action sam-            classical message passing from the perspective of algorith-
pling. Consider a finite set Ω of actions. We are interested        mic alignment (Faber & Wattenhofer, 2022).
in learning a categorical distribution over Ω, which can be
                                                                    Our approach presents new perspectives on these limitations
represented in terms of a probability vector p ∈ R|Ω| whose
                                                                    via a dynamic and asynchronous information flow (see Sec-
elements store the probabilities of different actions. Let us
                                                                    tion 5). This is related to the work of Lai et al. (2020), where
denote by p(a) the probability of an action a ∈ Ω. Gumbel-
                                                                    the goal is to update each node using a different number
softmax is a special reparametrization trick that estimates
                                                                    of layers (over a fixed topology). This is also related to
the categorical distribution p ∈ R|Ω| with the help of a
                                                                    the work of (Dai et al., 2022), where the idea is to apply
Gumbel-distributed vector g ∈ R|Ω| , which stores an i.i.d.
                                                                    message passing on a learned topology but one that is the
sample g(a) ∼ G UMBEL(0, 1) for each action a. Given a
                                                                    same at every layer. C O -GNNs diverge from these studies
categorical distribution p and a temperature parameter τ ,
                                                                    in terms of the objectives and the approach.
Gumbel-softmax (GS) scores can be computed as follows:
                          exp ((log(p) + g)/τ )                     4. Cooperative Graph Neural Networks
     GS (p; τ ) = P
                      a∈Ω exp ((log(p(a)) + g(a))/τ )
                                                                    C O -GNNs view each node in a graph as a player of a multi-
As the softmax temperature τ decreases, the resulting vector        player environment, where the state of each player is given
tends to a one-hot vector. Straight-through GS estimator            in terms of the representation (or state) of its corresponding
utilizes the GS estimator during the backward pass only (for        node. Every node is updated following a two-stage process.
a differentiable update), while during the forward pass, it         In the first stage, each node chooses an action from the set
employs an ordinary sampling.                                       of actions given their current state and the states of their
                                                                    neighboring nodes. In the second stage, every node state
3. Related Work                                                     gets updated based on their current state and the states of a
                                                                    subset of the neighboring nodes, as determined by the ac-
Most of GNNs operate by message-passing (Gilmer et al.,             tions in the first stage. As a result, every node can determine
2017), including architectures such as GCNs (Kipf &                 how to propagate information from or to its neighbors.
Welling, 2017), GIN (Xu et al., 2019), GAT (Veličković
et al., 2018), and GraphSAGE (Hamilton et al., 2017). De-           A C O -GNN (π, η) architecture is given in terms of two
spite their success, MPNNs have some known limitations.             cooperating GNNs: (i) an action network π for choosing the
                                                                    best actions, and (ii) an environment network η for updating
First, the expressive power of MPNNs is upper bounded by            the node representations. A C O -GNN layer updates the
1-WL (Xu et al., 2019; Morris et al., 2019). This motivated                            (ℓ)
                                                                    representations hv of each node v as follows. First, an
the study of more expressive architectures, based on higher-        action network π predicts, for each node v, a probability
order structures (Morris et al., 2019; Maron et al., 2019;                         (ℓ)
                                                                    distribution pv ∈ R4 over the actions {S, L, B, I} that v
Keriven & Peyré, 2019), subgraph (Bevilacqua et al., 2022;         can take, given its state and the state of its neighbors Nv :
Thiede et al., 2021) or homomorphism counting (Barceló
et al., 2021; Jin et al., 2024), node features with unique                                                 
identifiers (Loukas, 2020), or random features (Abboud                          p(ℓ)    (ℓ)    (ℓ)
                                                                                 v = π hv , {{hu | u ∈ Nv }} .                  (1)
et al., 2021; Sato et al., 2021).
                                                                                                                        (ℓ)     (ℓ)
Second, MPNNs perform poorly on long-range tasks due                Then, for each node v, an action is sampled av ∼ pv
to their information propagation bottlenecks such as over-          using Straight-through GS, and an environment network η
squashing (Alon & Yahav, 2021) and over-smoothing (Li               is utilized to update the state of each node in accordance
et al., 2018). The former limitation motivated approaches           with the sampled actions:
based on rewiring the graph (Klicpera et al., 2019; Topping                            (
                                                                                               (ℓ)        (ℓ)
et al., 2022; Karhadkar et al., 2023) by connecting relevant                            η (ℓ) hv , {{}} , av = I ∨ B
                                                                            h(ℓ+1)
                                                                             v     =           (ℓ)        (ℓ)                  (2)
nodes and shortening propagation distances, or designing                                η (ℓ) hv , M , av = L ∨ S
new message-passing architectures that act on distant nodes
directly, e.g., using shortest-path distances (Abboud et al.,                        (ℓ)             (ℓ)
                                                                    where M = {{hu | u ∈ Nv , au = S ∨ B}}.
2022; Ying et al., 2021). The over-smoothing problem has
also motivated a body of work to avoid the collapse of node         This is a single layer update, and by stacking L ≥ 1 layers,
                                                                                                     (L)
features (Zhao & Akoglu, 2019; Chen et al., 2020).                  we obtain the representations hv for each node v.

                                                                3
                                                 Cooperative Graph Neural Networks

            H                                H (0)                              H (1)                             H (2)
    s                                s                                  s                                 s
        u   v   w                        u   v     w                        u   v   w                         u   v   w

    r                                r                                  r                                 r
     (a) Input graph H.           (b) Computation at ℓ = 0.           (c) Computation at ℓ = 1.        (d) Computation at ℓ = 2.

Figure 2: The input graph H and its computation graphs H (0) , H (1) , H (2) that are a result of applying the actions: ⟨L, L, S⟩
for the node u; ⟨S, S, L⟩ for the nodes v and w; ⟨S, I, S⟩ for the nodes s and r; ⟨S, S, S⟩ for all other nodes.


In its full generality, a C O -GNN (π, η) architecture can           be seen as operating on a potentially different directed graph
operate on (un)directed graphs and use any GNN archi-                induced by the choice of actions at every layer (illustrated
tecture in place of the action network π and the environ-            in Figure 2). Formally, given a graph G = (V, E), let
ment network η. To carefully assess the virtue of this new           us denote by G(ℓ) = (V, E (ℓ) ) the directed computational
message-passing paradigm, we consider relatively simple ar-          graphs induced by the actions chosen at layer ℓ, where E (ℓ)
chitectures such as S UM GNNs, M EAN GNNs, P   GCN, GIN              is the set of directed edges at layer ℓ. We can rewrite the
and GAT, which are respectively denoted as , µ, ∗, ϵ,                update given in Equation (2) concisely as follows:
and α. For example, we write C O -GNN(Σ, µ) to denote a                                                                    
C O -GNN architecture which uses S UM GNN as its action                     h(ℓ+1)
                                                                             v      = η (ℓ) h(ℓ)     (ℓ)
                                                                                              v , {{hu | (u, v) ∈ E
                                                                                                                      (ℓ)
                                                                                                                          }} .
network and M EAN GNN as its environment network.
                                                                     Consider the input graph H from Figure 2: u gets messages
Fundamentally, C O -GNNs update the node states in a fine-           from v only in the first two layers, and v gets messages from
grained manner: if a node v chooses to I SOLATE or to                u only in the last layer, illustrating a directional message-
B ROADCAST then it gets updated only based on its previous           passing between these nodes. This abstraction allows for a
state, which corresponds to a node-wise update function.             direct implementation of C O -GNNs by simply considering
On the other hand, if a node v chooses the action L ISTEN            the induced graph adjacency matrix at every layer.
or S TANDARD then it gets updated based on its previous
state as well as the state of its neighbors which perform the        Dynamic: In C O -GNNs, each
actions B ROADCAST or S TANDARD at this layer.                       node interacts with the ‘rele-
                                                                     vant’ neighbors and does so                            r
                                                                                                                     s
                                                                     only as long as they remain rel-           u    r      u
5. Model Properties                                                  evant. C O -GNNs do not op-           v         v      s
We analyze C O -GNNs, focusing on conceptual novelty,                erate on a fixed computational             w
expressive power, and suitability to long-range tasks.               graph, but rather on a learned
                                                                                                                            w
                                                                     computational graph, which is
5.1. Conceptual Properties                                           dynamic across layers. In our
                                                                     running example, the computa-      ℓ=3     2    1    0
Task-specific: Standard message-passing updates nodes                tional graph is a different one
based on their neighbors, which is completely task-agnostic.         at every layer (depicted on the right hand side): This is
By allowing each node to listen to the information from              advantageous for the information flow (see Section 5.3).
‘relevant’ neighbors only, C O -GNNs can determine a com-
putation graph which is best suited for the target task. For         Feature and Structure Based: Standard message-passing
example, if the task requires information only from the              is determined by the structure of the graph: two nodes with
neighbors with a certain degree then the action network can          the same neighborhood get the same aggregated message.
learn to listen only to these nodes (see Section 6.1).               This is not necessarily the case in our setup, since the action
                                                                     network can learn different actions for two nodes with dif-
Directed: The outcome of the actions that the nodes can take         ferent node features, e.g., by choosing different actions for
amounts to a special form of ‘directed rewiring’ of the input        a red node and a blue node. This enables different messages
graph: an edge can be dropped (e.g., if two neighbors listen         for different nodes even if their neighborhoods are identical.
without broadcasting); an edge can remain undirected (e.g.,
if both neighbors apply the standard action); or, an edge            Asynchronous: Standard message-passing updates all
can become directed implying directional information flow            nodes synchronously, which is not always optimal as ar-
(e.g., if one neighbor listens while its neighbor broadcasts).       gued by Faber & Wattenhofer (2022), especially when the
Taking this perspective, the proposed message-passing can            task requires to treat the nodes non-uniformly. By design,
                                                                     C O -GNNs enable asynchronous updates across nodes.

                                                                 4
                                             Cooperative Graph Neural Networks

Conditional Aggregation: The action network of C O -                  The explanation for this result is the following: C O -GNN
GNNs can be viewed as a look-ahead function that makes                architectures learn, at every layer, and for each node u, a
decisions after applying k layers. Specifically, at layer ℓ, an       probability distribution over the actions. These learned dis-
action network of depth k computes node representations on            tributions are identical for two isomorphic nodes. However,
the original graph topology, which are (k+ℓ)-layer represen-          the process relies on sampling actions from these distribu-
tations. Based on these representations, the action network           tions, and clearly, the samples from identical distributions
determines an action for each node, which induces a new               can differ. This makes C O -GNN models invariant in expec-
graph topology for the environment network to operate on.             tation, and the variance introduced by the sampling process
In this sense, the aggregation of environment network at              helps to discriminate nodes that are 1-WL indistinguishable.
layer ℓ is determined by (k + ℓ)-layer representations of the         Thus, for two nodes indistinguishable by 1-WL, there is a
action network, which can be viewed as a “look-ahead” ca-             non-trivial probability of sampling a different action for the
pability and the aggregation mechanism of the environment             respective nodes, which in turn makes their direct neigh-
network is conditioned on this look-ahead capability.                 borhood differ. This yields unique node identifiers (Loukas
                                                                      (2020)) with high probability and allows us to distinguish
Orthogonal to Attention: The (soft) attention mechanism
                                                                      any pair of graphs assuming an injective graph pooling func-
on graphs allows for aggregating — based on learnable at-
                                                                      tion (Xu et al., 2019). This is analogous to GNNs with ran-
tention coefficients — a weighted mean of the features of
                                                                      dom node features (Abboud et al., 2021; Sato et al., 2021),
neighboring nodes. While these architectures can weigh the
                                                                      which are more expressive than their classical counterparts.
contribution of different neighbors, they have certain limita-
                                                                      We validate the stated expressiveness gain in Appendix C.1.
tions, e.g., weighted mean aggregation cannot count node
degrees. Moreover, the conditional aggregation mechanism              It is important to note that C O -GNNs are not designed for
of C O -GNNs goes beyond the capabilities of attention-               expressiveness, and our result relies merely on variations
based architectures. The contribution of Co-GNNs is hence             in the sampling process, which is unstable and should be
orthogonal to that of attention-based architectures, such             noted as a limitation. Clearly, C O -GNNs can also use more
as GATs, and these architectures can be used as base ac-              expressive architectures as base architectures.
tion/environment architectures in C O -GNNs. In Section 6.1,
we empirically validate this via a task that GAT cannot solve,        5.3. Dynamic Message-passing for Long-range Tasks
but C O -GNNs with a GAT environment network can.
                                                                      Long-range tasks necessitate to propagate information be-
Mitigates Over-smoothing: In principle, the action net-               tween distant nodes: C O -GNNs are effective for such tasks
work of C O -GNNs can choose the action I SOLATE for                  since they can propagate only relevant task-specific informa-
a node if the features of the neighbours of this node are             tion. Suppose that we are interested in transmitting informa-
not informative. As a result, C O -GNNs can mitigate over-            tion from a source node to a distant target node: C O -GNNs
smoothing. We validate this empirically in Appendix C.5,              can efficiently filter irrelevant information by learning to
but we also note that the optimisation becomes increasingly           focus on a path connecting these two nodes, hence max-
difficult once the number of layers gets too large.                   imizing the information flow to the target node. We can
Efficient: While being more sophisticated, our approach is            generalize this observation towards receiving information
efficient in terms of runtime, as we detail in Appendix D.            from multiple distant nodes and prove the following:
C O -GNNs are also parameter-efficient: they share the same           Proposition 5.2. Let G = (V, E, X) be a connected graph
action network across layers and as a result a comparable             with node features. For some k > 0, for any target node
number of parameters to their baseline models.                        v ∈ V , for any k source nodes u1 , . . . , uk ∈ V , and for any
                                                                                                                  (0)            (0)
                                                                      compact, differentiable function f : Rd × . . . × Rd →
5.2. Expressive Power of C O -GNNs                                    Rd , there exists an L-layer C O -GNN computing final node
                                                                      representations such that for any ϵ, δ > 0 it holds that
The environment and action networks of C O -GNN archi-                      (L)
                                                                      P(|hv − f (xu1 , . . . xuk )| < ϵ) ≥ 1 − δ.
tectures are parameterized by standard MPNNs. This raises
an obvious question regarding the expressive power of                 This means that if a property of a node v is a function of k
C O -GNN architectures: are C O -GNNs also bounded by                 distant nodes then C O -GNNs can approximate this function.
1-WL in terms of distingushing graphs?                                This follows from two findings: (i) the features of k nodes
                                                                      can be transmitted to the source node without loss of infor-
Proposition 5.1. Let G1 = (V1 , E1 , X1 ) and G2 =                    mation and (ii) the final layer of a C O -GNN architecture,
(V2 , E2 , X2 ) be two non-isomorphic graphs. Then, for any           e.g., an MLP, can approximate any differentiable function
threshold 0 < δ < 1, there exists a parametrization of                over k node features (Hornik, 1991; Cybenko, 1989). We
a C O -GNN architecture using sufficiently many layers L,             validate these findings empirically on long-range interac-
                 (L)    (L)
satisfying P(zG1 ̸= zG2 ) ≥ 1 − δ.                                    tions datasets (Dwivedi et al., 2022) in Appendix C.2.

                                                                  5
                                              Cooperative Graph Neural Networks


                                                                                                       ⟨L⟩
                                 r

                   u             v                                                 ⟨B⟩           ⟨B⟩                 ⟨L⟩



Figure 3: ROOT N EIGHBORS examples. Left: Example tree for ROOT N EIGHBORS. Right: Example of an optimal directed
subgraph over the input tree, where the nodes with a degree of 6 (u and v) B ROADCAST, while other nodes L ISTEN.


6. Experimental Results                                                Table 1: Results on ROOT N EIGHBORS. Top three models
                                                                       are colored by First, Second, Third.
We evaluate C O -GNNs on a synthetic experiment, and
on real-world node classification datasets (Platonov et al.,                             Model               MAE
2023). We also report a synthetic expressiveness experiment,
an experiment on long-range interactions datasets (Dwivedi                               Random              0.474
et al., 2022), and graph classification datasets (Morris et al.,                         GAT                 0.442
2020) in Appendix C. Our codebase is available at https:                                 S UM GNN            0.370
//github.com/benfinkelshtein/CoGNN.                                                      M EAN GNN           0.329
                                                                                         C O -GNN(Σ, Σ)      0.196
6.1. Synthetic Experiment on ROOT N EIGHBORS                                             C O -GNN(µ, µ)      0.339
                                                                                         C O -GNN(Σ, α)      0.085
Task. In this experiment, we compare C O -GNNs to
                                                                                         C O -GNN(Σ, µ)      0.079
MPNNs on a new dataset: ROOT N EIGHBORS. We consider
the following regression task: given a rooted tree, predict
the average of the features of root-neighbors of degree 6.
This task requires to first identify the neighbors of the root
node with degree 6 and then to return the average feature of           Results for C O -GNNs. The ideal mode of operation for
these nodes. ROOT N EIGHBORS consists of trees of depth 2              C O -GNNs would be as follows:
with random features of dimension 5. The generation (Ap-
                                                                       1. The action network chooses either L ISTEN or S TAN -
pendix E.3) ensures each tree root has at least one degree-6
                                                                          DARD for the root node, and B ROADCAST or S TANDARD
neighbor. An example is shown on the left of Figure 3: the
                                                                          for the root-neighbors which have a degree 6.
root node r has only two neighbors with degree 6 (u and v)
                                                                       2. The action network chooses either L ISTEN or I SOLATE
and the target prediction value is (xu + xv )/2.
                                                                          for all the remaining root-neighbors.
Setup. We consider GCN, GAT, S UM GNN, M EAN GNN,                      3. The environment network updates the root node by aver-
as baselines, and compare to C O -GNN(Σ, Σ),                              aging features from its broadcasting neighbors.
C O -GNN(µ, µ), C O -GNN(Σ, α) and C O -GNN(Σ, µ).
We report the Mean Average Error (MAE), use the                        C O -GNN(Σ, µ): The best result is achieved by this model,
Adam optimizer and present all details including the                   because S UM GNN as the action network can accomplish
hyperparameters in Appendix E.4.                                       (1) and (2), and M EAN GNN as the environment network
                                                                       can accomplish (3). Therefore, this model leverages the
Results for MPNNs. The results are presented in Table 1,               strengths of S UM GNN and M EAN GNN to cater to the dif-
which includes the random baseline (i.e., MAE obtained via             ferent roles of the action and environment networks, making
a random prediction). All MPNNs perform poorly: GCN,                   it the most natural C O -GNN model for the regression task.
GAT, and M EAN GNN fail to identify node degrees, making
it impossible to detect nodes with a specific degree, which            C O -GNN(Σ, α): We observe a very similar phenomenon
is crucial for the task. GCN and GAT are only marginally               here to that of C O -GNN(Σ, µ). The action network allows
better than the random baseline, whereas M EAN GNN per-                GAT to determine the right topology, and GAT only needs
forms substantially better than the random baseline. The               to learn to average the features. This shows the contribution
latter can be explained by the fact that M EAN GNN employs             of C O -GNNs is orthogonal to that of attention aggregation.
a different transformation on the source node rather than              C O -GNN(Σ, Σ): This model also performs well, primarily
treating it as a neighbor (unlike the self-loop in GCN/GAT).           because it uses S UM GNN as the action network, accom-
S UM GNN uses sum aggregation and can identify the node                plishing (1) and (2). However, it uses another S UM GNN
degrees, but struggles in averaging the node features, which           as the environment network which cannot easily mimic the
yields comparable MAE results to that of M EAN GNN.                    averaging of the neighbor’s features.

                                                                   6
                                               Cooperative Graph Neural Networks

              Table 2: Results on node classification. Top three models are colored by First, Second, Third.

                               roman-empire       amazon-ratings      minesweeper        tolokers       questions
             GCN                73.69 ± 0.74        48.70 ± 0.63       89.75 ± 0.52    83.64 ± 0.67    76.09 ± 1.27
             SAGE               85.74 ± 0.67        53.63 ± 0.39       93.51 ± 0.57    82.43 ± 0.44    76.44 ± 0.62
             GAT                80.87 ± 0.30        49.09 ± 0.63       92.01 ± 0.68    83.70 ± 0.47    77.43 ± 1.20
             GAT-sep            88.75 ± 0.41        52.70 ± 0.62       93.91 ± 0.35    83.78 ± 0.43    76.79 ± 0.71
             GT                 86.51 ± 0.73        51.17 ± 0.66       91.85 ± 0.76    83.23 ± 0.64    77.95 ± 0.68
             GT-sep             87.32 ± 0.39        52.18 ± 0.80       92.29 ± 0.47    82.52 ± 0.92    78.05 ± 0.93
             C O -GNN(Σ, Σ) 91.57 ± 0.32            51.28 ± 0.56       95.09 ± 1.18    83.36 ± 0.89    80.02 ± 0.86
             C O -GNN(µ, µ) 91.37 ± 0.35            54.17 ± 0.37       97.31 ± 0.41    84.45 ± 1.17    76.54 ± 0.95


C O -GNN(µ, µ): This model clearly performs weakly,                 and environment networks. Importantly, C O -GNNs demon-
since M EAN GNN as an action network cannot achieve                 strate an average accuracy improvement of 2.23% compared
(1) hindering the performance of the whole task. Indeed,            to all baseline methods, across all datasets, surpassing the
C O -GNN(µ, µ) performs comparably to M EAN GNN sug-                performance of more complex models such as GT. In our
gesting that the action network is not useful in this case.         main finding we observe a consistent trend: enhancing stan-
                                                                    dard models with action networks of C O -GNNs results
To shed light on the performance of C O -GNN models, we
                                                                    in improvements in performance. For example, we report
computed the percentage of edges which are accurately
                                                                    3.19% improvement in accuracy on the roman-empire and
retained or removed by the action network in a single layer
                                                                    3.62% improvement in ROC AUC on minesweeper com-
C O -GNN model. We observe an accuracy of 99.71% for
                                                                    pared to the best performing baseline. This shows that
C O -GNN(Σ, µ), 99.55% for C O -GNN(Σ, Σ), and 57.20%
                                                                    C O -GNNs are flexible and effective on different datasets
for C O -GNN(µ, µ). This empirically confirms the expected
                                                                    and tasks. These results are reassuring as they establish
behavior of C O -GNNs. In fact, the example tree is shown
                                                                    C O -GNNs as a strong method in the heterophilic setting
on the right of Figure 3 is taken from the experiment with
                                                                    due to its unique ability to manipulate information flow.
C O -GNN(Σ, µ): reassuringly, this model learns precisely
the actions that induce the shown optimal subgraph.
                                                                    7. Empirical Insights for the Actions
6.2. Node Classification with Heterophilic Graphs                   The action network of C O -GNNs is the key model compo-
One of the strengths of C O -GNNs is their capability to uti-       nent. The purpose of this section is to provide additional
lize task-specific information propagation, which raises an         insights regarding the actions being learned by C O -GNNs.
obvious question: could C O -GNNs outperform the base-
lines on heterophilious graphs, where standard message              7.1. Actions on Heterophilic vs Homophilic Graphs
passing is known to suffer? To answer this question, we
                                                                    We aim to compare the actions learned on a homophilic task
assess the performance of C O -GNNs on heterophilic node
                                                                    to the actions learned on a heterophilic task. One idea would
classification datasets from (Platonov et al., 2023).
                                                                    be to inspect the learned action distributions, but they alone
Setup.      We evaluate S UM GNN, M EAN GNN and                     may not provide a clear picture of the graph’s topology.
their C O -GNN counterparts, C O -GNN(Σ, Σ) and                     For example, two connected nodes that choose to I SOLATE
C O -GNN(µ, µ) on the 5 heterophilic graphs, following              achieve the same topology as nodes that choose both to
the 10 data splits and the methodology of Platonov et al.           B ROADCAST or L ISTEN. This is a result of the immense
(2023). We report the accuracy and standard deviation for           number of action configurations and their interactions.
roman-empire and amazon-ratings. We also report the ROC
                                                                    To better understand the learned graph topology, we inspect
AUC and standard deviation for minesweeper, tolokers, and
                                                                    the induced directed graphs at every layer. Specifically, we
questions. The classical baselines GCN, GraphSAGE, GAT,
                                                                    present the ratio of the directed edges that are kept across
GAT-sep, GT (Shi et al., 2021) and GT-sep are taken from
                                                                    the different layers in Figure 4. We record the directed edge
Platonov et al. (2023). We use the Adam optimizer and
                                                                    ratio over the 10 different layers of our best, fully trained
report all hyperparameters in Appendix E.4.
                                                                    10 C O -GNN(µ, µ) models on the roman-empire (Platonov
Results. All results are reported in Table 2. Observe that          et al., 2023) and cora datasets (Pei et al., 2020). We follow
C O -GNNs achieve state-of-the-art results across the board,        the 10 data splits and the methodology of Platonov et al.
despite using relatively simple architectures as their action       (2023) and Yang et al. (2016), respectively.

                                                                7
                                              Cooperative Graph Neural Networks




Figure 4: The ratio of directed edges that are kept on cora
(as a homophilic dataset) and on roman-empire (as a het-
erophilic dataset) for each layer 0 ≤ ℓ < 10.


This experiment serves as a strong evidence for the adaptive
nature of C O -GNNs this statement. Indeed, by inspecting
Figure 4, we observe completely opposite trends between                   Figure 5: The 10-hop neighborhood at layer ℓ = 4
the two datasets.
On the homophilic dataset cora, the ratio of edges that are
                                                                      showing the number of adjacent mines. A randomly chosen
kept gradually decreases as we go to the deeper layers. In
                                                                      50% of the nodes have an unknown feature, indicated by a
fact, 100% of the edges are kept at layer ℓ = 0 while all
                                                                      separate binary feature. The task is to identify whether the
edges are dropped at layer ℓ = 9. This is very insight-
                                                                      querying node is a mine.
ful because homophilic datasets are known to not benefit
from using many layers, and the trained C O -GNN model                Setup. We train a 10-layer C O -GNN(µ, µ) model and
recognizes this by eventually isolating all the nodes. This           present the graph topology at layer ℓ = 4. The evolution
is particularly the case for cora, where classical MPNNs              of the graph topology from layer ℓ = 1 to layer ℓ = 8 is
typically achieve their best performance with 1-3 layers.             presented in Appendix F. We choose a node (black), and at
                                                                      every layer ℓ, we depict its neighbors up to distance 10. In
On the heterophilic dataset roman-empire, the ratio of edges
                                                                      this visualization, nodes which are mines are shown in red,
that are kept gradually increases after ℓ = 1 as we go to the
                                                                      and other nodes in blue. The features of non-mine nodes
deeper layers. Initially, ∼ 42% of the edges are kept at layer
                                                                      (indicating the number of neighboring mines) are shown
ℓ = 0 while eventually this reaches 99% at layer ℓ = 9. This
                                                                      explicitly whereas the nodes whose features are hidden are
is interesting, since in heterophilic graphs, edges tend to
                                                                      labeled with a question mark.
connect nodes of different classes and so classical MPNNs,
which aggregate information based on the homophily as-                Interpreting the Actions. The visualization of the actions
sumption perform poorly. Although, C O -GNN model uses                at layer ℓ = 4 is shown in Figure 5. In this game, every
these models it compensates by controlling information flow.          label is informative: non-0-labeled nodes are more informa-
The model manages to capture the heterophilous aspect of              tive in the earlier layers (ℓ = 1, 2, 3, 4), whereas 0-labeled
the dataset by restricting the flow of information in the early       nodes become more informative at later layers. This can be
layer and slowly enabling it the deeper the layer (the fur-           explained by considering two cases:
ther away the nodes), which might be a great benefit to its
                                                                      1. The target node has at least one 0-labeled neighbor: In
success over heterophilic benchmarks.
                                                                         this case, the prediction trivializes, since we know that
                                                                         the target node cannot be a mine.
7.2. What Actions are Performed on Minesweeper?
                                                                      2. The target node has no 0-labeled neighbors: In this case,
To better understand the topology learned by C O -GNNs,                  the model needs to make a sophisticated inference based
we visualize the topology at each layer in a C O -GNN model              on the surrounding mines within the k-hop distance. In
over the highly regular minesweepers dataset.                            this scenario, a node obtains more information by aggre-
                                                                         gating from non-0-label nodes. The model can still im-
Dataset. Minesweeper (Platonov et al., 2023) is a synthetic
                                                                         plicitly infer “no mine” from the lack of a signal/message.
dataset inspired by the Minesweeper game. It is a semi-
supervised node classification dataset with a regular 100 ×
100 grid where each node is connected to eight neighboring            The action network appears to diffuse the information of
nodes. Each node has an one-hot-encoded input feature                 non-0-labeled nodes in the early layers to capture nodes

                                                                  8
                                              Cooperative Graph Neural Networks




Figure 6: MAE and ROC AUC as a function of the choice of action (π) and environment (η) networks over ROOT N EIGHBORS
(left) and the minesweeper (right) experiments, respectively.


from case (2), while largely isolating the 0-labeled nodes            with different choices of action and environment networks.
until the last layers (avoiding mixing of signals and potential
                                                                      In terms of the environment network, we observe that
loss of information) which can later be used to capture the
                                                                      MeanGNN and GAT yield consistently robust results when
nodes from case (1). In the earlier layers, the action network
                                                                      used as environment networks regardless of the choice of
prioritizes the information flowing from the left sections of
                                                                      the action network. This makes sense in the context of the
the grid in Figure 5 where more mines are present. After
                                                                      minesweeper game. In order to make a good inference, a
identifying the most crucial information and propagating
                                                                      k-layer environment network can keep track of the average
this through the network, it then requires this information
                                                                      number of mines found in each hop distance. The task is
to also be communicated with the nodes that initially were
                                                                      hence well-suited to mean-style aggregation environment
labeled with 0. This leads to an almost fully connected grid
                                                                      networks, which manifests as empirical robustness. GCN
in the later layers (see ℓ = 7, 8 in Figures 16 and 17).
                                                                      performs worse as environment network, since it cannot
                                                                      distinguish a node from its neighbors.
7.3. Which Action or Environment Network?
                                                                      In terms of the action network, we observed earlier that the
We conduct an ablation study on the choice of action and              role of the action network is to mainly make a distinction
environment networks to quantify its affect.                          between 0-labeled nodes and non-0-labeled nodes, and such
Setup. We experiment with all 25 combinations of the                  an action network can be realized with all of the architec-
architectures M EAN GNNs, GAT, S UM GNN, GCN and                      tures considered. As a result, we do not observe dramatic
GIN on the heterophilic graph minesweeper and on the                  differences in the performance regarding to the choice of
synthetic dataset ROOT N EIGHBORS. We report MAE for                  the action network in this task.
ROOT N EIGHBORS and the ROC AUC for minesweeper.
ROOT N EIGHBORS Results. The results reported in Fig-                 8. Summary and Outlook
ure 6 (left) support our analysis from Section 6.1: an envi-          We introduced C O -GNN architectures which can dynam-
ronment network with mean-type aggregation (GCN, GAT,                 ically explore the graph topology while learning. These
or M EAN GNN) and an action network with sum-type ag-                 architectures have desirable properties which can inform
gregation (S UM GNN, or GIN) are best choices for the task.           future work. Looking forward, one potential future direc-
The choice of the action network is critical for this task:           tion is to adapt C O -GNNs to other types of graphs such
S UM GNN and GIN yield best results across the board when             as directed, and even multi-relational graphs. One possi-
they are used as action networks. In contrast, if we use GAT,         ble approach is by including actions that also consider the
GCN, or M EAN GNN as action networks then the results are             directionality. For example, for each node u, one can de-
poor and comparable to baseline results on this task. These           fine the actions L ISTEN -I NC (listen to nodes that have an
action networks cannot detect node cardinality which pre-             incoming edge to u) and L ISTEN -O UT (listen to nodes that
vents them from choosing the optimal actions as elaborated            have an incoming edge from u) and extend the other actions
in Section 6.1. The choice of the environment network is              analogously to incorporate directionality. It is also possible
relatively less important for this task.                              to consider other types of actions or extend our approach
Minesweeper Results. In Figure 6 (right), C O -GNNs                   to edge-wise actions (rather than node-wise), though these
achieve multiple state-of-the-art results on minesweeper              extensions will lead to a larger state space.


                                                                  9
                                             Cooperative Graph Neural Networks

Acknowledgments                                                       Dai, E., Jin, W., Liu, H., and Wang, S. Towards robust graph
                                                                        neural networks for noisy graphs with sparse labels. In
The authors would like to thank the anonymous reviewers                WSDM, 2022.
for their feedback which led to substantial improvements
in the presentation of the paper. The first author is funded          Di Giovanni, F., Giusti, L., Barbero, F., Luise, G., Lio, P.,
by the Clarendon scholarship. The authors would like to                 and Bronstein, M. M. On over-squashing in message
also acknowledge the use of the University of Oxford Ad-                passing neural networks: The impact of width, depth, and
vanced Research Computing (ARC) facility in carrying out                topology. In ICLR, 2023.
this work (http://dx.doi.org/10.5281/zenodo.22558). This
work was also partially funded by EPSRC Turing AI World-              Duvenaud, D., Maclaurin, D., Aguilera-Iparraguirre, J.,
Leading Research Fellowship No. EP/X040062/1.                           Gómez-Bombarelli, R., Hirzel, T., Aspuru-Guzik, A.,
                                                                        and Adams, R. P. Convolutional networks on graphs for
                                                                        learning molecular fingerprints. In NeurIPS, 2015.
Impact Statement
                                                                      Dwivedi, V. P., Rampášek, L., Galkin, M., Parviz, A., Wolf,
This paper presents a novel graph neural network paradigm              G., Luu, A. T., and Beaini, D. Long range graph bench-
whose goal is to explore the graph topology while learning.            mark. In NeurIPS Datasets and Benchmarks, 2022.
There are many potential societal consequences of our work,
none of which we feel must be specifically highlighted here.          Errica, F. and Niepert, M. Tractable probabilistic graph
                                                                        representation learning with graph-induced sum-product
                                                                        networks. In arXiv, 2023.
References
Abboud, R., Ceylan, İ. İ., Grohe, M., and Lukasiewicz,              Errica, F., Podda, M., Bacciu, D., and Micheli, A. A fair
 T. The surprising power of graph neural networks with                  comparison of graph neural networks for graph classifica-
  random node initialization. In IJCAI, 2021.                           tion. In ICLR, 2020.

Abboud, R., Dimitrov, R., and Ceylan, İ. İ. Shortest path           Faber, L. and Wattenhofer, R. Asynchronous neural net-
  networks for graph property prediction. In LoG, 2022.                 works for learning in graphs. In arXiv, 2022.

Alon, U. and Yahav, E. On the bottleneck of graph neural              Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and
  networks and its practical implications. In ICLR, 2021.               Dahl, G. E. Neural message passing for quantum chem-
                                                                        istry. In ICML, 2017.
Bacciu, D., Errica, F., and Micheli, A. Probabilistic learning
  on graphs via contextual architectures. In JMLR, 2020.              Gori, M., Monfardini, G., and Scarselli, F. A new model for
                                                                        learning in graph domains. In IJCNN, 2005.
Barceló, P., Geerts, F., Reutter, J. L., and Ryschkov, M.
  Graph neural networks with local graph parameters. In               Gutteridge, B., Dong, X., Bronstein, M. M., and Di Gio-
  NeurIPS, 2021.                                                       vanni, F. DRew: Dynamically rewired message passing
                                                                       with delay. In ICML, 2023.
Bevilacqua, B., Frasca, F., Lim, D., Srinivasan, B., Cai,
  C., Balamurugan, G., Bronstein, M. M., and Maron, H.                Hamilton, W., Ying, Z., and Leskovec, J. Inductive repre-
  Equivariant subgraph aggregation networks. In ICLR,                   sentation learning on large graphs. In NeurIPS, 2017.
  2022.
                                                                      Hamilton, W. L. Graph representation learning. In Synthesis
Bodnar, C., Giovanni, F. D., Chamberlain, B. P., Liò, P., and          Lectures on Artifical Intelligence and Machine Learning,
  Bronstein, M. M. Neural sheaf diffusion: A topological                2020.
  perspective on heterophily and oversmoothing in GNNs.
                                                                      He, X., Hooi, B., Laurent, T., Perold, A., Lecun, Y., and
  In NeurIPS, 2023.
                                                                        Bresson, X. A generalization of ViT/MLP-mixer to
Bresson, X. and Laurent, T. Residual gated graph convnets.              graphs. In ICML, 2023.
  In arXiv, 2018.
                                                                      Hornik, K. Approximation capabilities of multilayer feed-
Castellana, D., Errica, F., Bacciu, D., and Micheli, A. The             forward networks. In Neural Networks, 1991.
  infinite contextual graph Markov model. In ICML, 2022.
                                                                      Jang, E., Gu, S., and Poole, B. Categorical reparameteriza-
Chen, M., Wei, Z., Huang, Z., Ding, B., and Li, Y. Simple               tion with gumbel-softmax. In ICLR, 2017.
  and deep graph convolutional networks. In ICML, 2020.
                                                                      Jin, E., Bronstein, M., Ceylan, İ. İ., and Lanzinger, M. Ho-
Cybenko, G. Approximation by superpositions of a sig-                    momorphism counts for graph neural networks: All about
  moidal function. In MCSS, 1989.                                        that basis. In ICML, 2024.

                                                                 10
                                              Cooperative Graph Neural Networks

Karhadkar, K., Banerjee, P. K., and Montufar, G. FoSR:                 Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., and
  First-order spectral rewiring for addressing oversquashing             Monfardini, G. The graph neural network model. In
  in GNNs. In ICLR, 2023.                                                IEEE Transactions on Neural Networks, 2009.

Keriven, N. and Peyré, G. Universal invariant and equivari-           Sen, P., Namata, G., Bilgic, M., Getoor, L., Galligher, B.,
  ant graph neural networks. In NeurIPS, 2019.                           and Eliassi-Rad, T. Collective classification in network
                                                                         data. In AI Magazine, 2008.
Kipf, T. and Welling, M. Semi-supervised classification
  with graph convolutional networks. In ICLR, 2017.                    Shi, Y., Huang, Z., Feng, S., Zhong, H., Wang, W., and Sun,
                                                                         Y. Masked label prediction: Unified message passing
Klicpera, J., Weißenberger, S., and Günnemann, S. Diffu-                model for semi-supervised classification. In IJCAI, 2021.
  sion improves graph learning. In NeurIPS, 2019.
                                                                       Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland,
Lai, K.-H., Zha, D., Zhou, K., and Hu, X. Policy-GNN:                    D. J., and Sinop, A. K. Exphormer: Sparse transformers
  Aggregation optimization for graph neural networks. In                 for graphs. In ICML, 2023.
  KDD, 2020.
                                                                       Shlomi, J., Battaglia, P., and Vlimant, J.-R. Graph neural
Li, Q., Han, Z., and Wu, X. Deeper insights into graph                   networks in particle physics. In MLST, 2021.
  convolutional networks for semi-supervised learning. In
  AAAI, 2018.                                                          Simonovsky, M. and Komodakis, N. Dynamic edge-
                                                                         conditioned filters in convolutional neural networks on
Loukas, A. What graph neural networks cannot learn: depth                graphs. In CVPR, 2017.
  vs width. In ICLR, 2020.
                                                                       Thiede, E. H., Zhou, W., and Kondor, R. Autobahn:
Ma, L., Lin, C., Lim, D., Romero-Soriano, A., Dokania, K.,               Automorphism-based graph neural nets. In NeurIPS,
 Coates, M., H.S. Torr, P., and Lim, S.-N. Graph Inductive               2021.
 Biases in Transformers without Message Passing. In
                                                                       Tönshoff, J., Ritzert, M., Wolf, H., and Grohe, M. Walking
 ICML, 2023.
                                                                          out of the weisfeiler leman hierarchy: Graph learning
Maddison, C. J., Mnih, A., and Teh, Y. W. The concrete                    beyond message passing. In TMLR, 2023.
 distribution: A continuous relaxation of discrete random
                                                                       Topping, J., Giovanni, F. D., Chamberlain, B. P., Dong, X.,
 variables. In ICLR, 2017.
                                                                         and Bronstein, M. M. Understanding over-squashing and
Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman, Y.                  bottlenecks on graphs via curvature. In ICLR, 2022.
 Provably powerful graph networks. In NeurIPS, 2019.                   Tönshoff, J., Ritzert, M., Rosenbluth, E., and Grohe, M.
                                                                         Where did the gap go? reassessing the long-range graph
Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen,
                                                                         benchmark. In arXiv, 2023.
 J. E., Rattan, G., and Grohe, M. Weisfeiler and Leman
 go neural: Higher-order graph neural networks. In AAAI,               Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò,
 2019.                                                                   P., and Bengio, Y. Graph attention networks. In ICLR,
                                                                         2018.
Morris, C., Kriege, N. M., Bause, F., Kersting, K., Mutzel, P.,
 and Neumann, M. Tudataset: A collection of benchmark                  Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M.,
 datasets for learning with graphs. In ICML workshop on                 and Solomon, J. M. Dynamic graph CNN for learning on
 Graph Representation Learning and Beyond, 2020.                        point clouds. In TOG, 2019.
Pei, H., Wei, B., Chang, K. C.-C., Lei, Y., and Yang, B.               Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., and Yu,
  Geom-GCN: Geometric graph convolutional networks.                     P. S. A comprehensive survey on graph neural networks.
  In ICLR, 2020.                                                        In IEEE Transactions on Neural Networks and Learning
                                                                        Systems, 2019.
Platonov, O., Kuznedelev, D., Diskin, M., Babenko, A., and
  Prokhorenkova, L. A critical look at the evaluation of               Xu, K., Hu, W., Leskovec, J., and Jegelka, S. How powerful
  GNNs under heterophily: Are we really making progress?                 are graph neural networks? In ICLR, 2019.
  In ICLR, 2023.
                                                                       Yang, Z., Cohen, W. W., and Salakhutdinov, R. Revisiting
Sato, R., Yamada, M., and Kashima, H. Random features                    semi-supervised learning with graph embeddings. In
  strengthen graph neural networks. In SDM, 2021.                        ICML, 2016.

                                                                  11
                                            Cooperative Graph Neural Networks

Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., Shen,
  Y., and Liu, T. Do transformers really perform badly for
  graph representation? In NeurIPS, 2021.
Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W. L., and
  Leskovec, J. Hierarchical graph representation learning
  with differentiable pooling. In NeurIPS, 2018.

Zhao, L. and Akoglu, L. Pairnorm: Tackling oversmoothing
  in gnns. arXiv, 2019.
Zitnik, M., Agrawal, M., and Leskovec, J. Modeling
  polypharmacy side effects with graph convolutional net-
  works. In Bioinformatics, 2018.




                                                               12
                                            Cooperative Graph Neural Networks

A. Proofs of Technical Results
A.1. Proof of Proposition 5.1
In order to prove Proposition 5.1, we first prove the following lemma, which shows that all non-isolated nodes of an input
graph can be individualized by C O -GNNs:
Lemma A.1. Let G = (V, E, X) be a graph with node features. For every pair of non-isolated nodes u, v ∈ V and for all
                                                                                               (L)   (L)
δ > 0, there exists a C O -GNN architecture with sufficiently many layers L which satisfies P(hu ̸= hv ) ≥ 1 − δ.

Proof. We consider an L-layer C O -GNN(η, π) architecture satisfying the following:

 (i) the environment network η is composed of L injective layers,

(ii) the action network π is composed of a single layer, and it is shared across C O -GNN layers.

Item (i) can be satisfied by a large class of GNN architectures, including S UM GNN (Morris et al., 2019) and GIN (Xu et al.,
                                  (0)     (0)
2019). We start by assuming hu = hv . These representations can be differentiated if the model can jointly realize the
following actions at some layer ℓ using the action network π:

      (ℓ)
  1. au = L ∨ S,
      (ℓ)
  2. av = I ∨ B, and
                                (ℓ)
  3. ∃ a neighbor w of u s.t. aw = S ∨ B.

The key point is to ensure an update for the state of u via an aggregated message from its neighbors (at least one), while
isolating v. In what follows, we assume the worst-case for the degree of u and consider a node w to be the only neighbor of
u. Let us denote the joint probability of realizing these actions 1-3 for the nodes u, v, w at layer ℓ as:
                                                                                              
                              p(ℓ)          (ℓ)                (ℓ)               (ℓ)
                               u,v = P (au = L ∨ S) ∧ (av = I ∨ B) ∧ (aw = S ∨ B) .


The probability of taking each action is non-zero (since it is a result of applying softmax) and u has at least one neighbor
                           (ℓ)
(non-isolated), therefore pu,v > 0. For example, if we assume a constant action network that outputs a uniform distribution
                                                                (ℓ)
over the possible actions (each action probability 0.25) then pu,v = 0.125.
                                                                                                                    (ℓ)
This means that the environment network η applies the following updates to the states of u and v with probability pu,v > 0:
                                                                                      
                                h(ℓ+1)
                                 u     = η (ℓ) h(ℓ)         (ℓ)        (ℓ)
                                                  u , {{hw | w ∈ Nu , aw = S ∨ B}} ,
                                                           
                                h(ℓ+1)
                                 v     = η (ℓ)
                                                 h(ℓ)
                                                  v   , {
                                                        {}}   .

The inputs to the environment network layer η (ℓ) for these updates are clearly different, and since the environment layer is
                             (ℓ+1)     (ℓ+1)
injective, we conclude that hu     ̸= hv     .
Thus, the probability of having different final representations for the nodes u and v is lower bounded by the probability of
the events 1-3 jointly occurring at least once in one of the C O -GNN layers, which, by applying the union bound, yields:
                                                     Lu,v             
                                                     Y                                   Lu,v
                       P(h(L
                          u
                            u,v )
                                  ̸= h(L
                                      v
                                        u,v )
                                              )≥1−            1 − p(ℓ)
                                                                   u,v ≥ 1 − (1 − γu,v )      ≥1−δ
                                                     ℓ=0
                                    
                               (ℓ)
where γu,v = maxℓ∈[Lu,v ] pu,v and Lu,v = log1−γu,v (δ).

We repeat this process for all pairs of non-isolated nodes u, v ∈ V . Due to the injectivity of η (ℓ) for all ℓ ∈ [L], once
the nodes are distinguished, they cannot remain so in deeper layers of the architecture, which ensures that all nodes

                                                               13
                                            Cooperative Graph Neural Networks

                                              (ℓ)     (ℓ)
u, v ∈ V differ in their final representations hu ̸= hv after this process completes. The number of layers required for this
construction is then given by:                                     X
                                       L = |V \ I| log1−α (δ) ≥          log1−γu,v (δ) ,
                                                                  u,v∈V \I

where I is the set of all isolated nodes in V and
                                                                                                
                                    α = max (γu,v ) = max                   max            p(ℓ)
                                                                                            u,v        .
                                         u,v∈V \I           u,v∈V \I       ℓ∈[Lu,v ]

Having shown a C O -GNN construction with the number of layers bounded as above, we conclude the proof.
Proposition 5.1. Let G1 = (V1 , E1 , X1 ) and G2 = (V2 , E2 , X2 ) be two non-isomorphic graphs. Then, for any threshold
0 < δ < 1, there exists a parametrization of a C O -GNN architecture using sufficiently many layers L, satisfying
   (L)     (L)
P(zG1 ̸= zG2 ) ≥ 1 − δ.

Proof. Let δ > 0 be any value and consider the graph G = (V, E, X) which has G1 and G2 as its components:

                                     V = V1 ∪ V2 ,    E = E1 ∪ E2 ,          X = X1 ||X2 ,

where || is the matrix horizontal concatenation. By Lemma A.1, for every pair of non-isolated nodes u, v ∈ V and for all
δ > 0, there exists a C O -GNN architecture with sufficiently many layers L = |V \ I| log1−α (δ) which satisfies:
                                                                                           
                           P(h(L
                              u
                                 u,v )
                                       ̸
                                       = h(Lu,v )
                                          v       ) ≥ 1 − δ, with α = max      max      p(ℓ)
                                                                                         u,v    ,
                                                                       u,v∈V \I            ℓ∈[Lu,v ]

        (ℓ)
where pu,v represents a lower bound on the probability for the representations of nodes u, v ∈ V at layer ℓ being different.
We use the same C O -GNN construction given in Lemma A.1 on G, which ensures that all non-isolated nodes have different
representations in G. When applying this C O -GNN to G1 and G2 separately, we get that every non-isolated node from
either graph has a different representation with probability 1 − δ as a result. Hence, the multiset M1 of node features for G1
and the multiset M2 of node features of G2 must differ. Assuming an injective pooling function from these multisets to
graph-level representations, we get:
                                                       (L)      (L)
                                                   P(zG1 ̸= zG2 ) ≥ 1 − δ
for L = |V \ I| log1−α (δ).

A.2. Proof of Proposition 5.2
Proposition 5.2. Let G = (V, E, X) be a connected graph with node features. For some k > 0, for any target node
                                                                                                        (0)          (0)
v ∈ V , for any k source nodes u1 , . . . , uk ∈ V , and for any compact, differentiable function f : Rd × . . . × Rd →
Rd , there exists an L-layer C O -GNN computing final node representations such that for any ϵ, δ > 0 it holds that
     (L)
P(|hv − f (xu1 , . . . xuk )| < ϵ) ≥ 1 − δ.

                                                                                                   (0)              (0)
Proof. For arbitrary ϵ, δ > 0, we start by constructing a feature encoder ENC : Rd                         → R2(k+1)d     which encodes the
                                (0)
initial representations xw ∈ Rd of each node w as follows:
                                                                            ⊤ ⊤
                                            ENC(xw ) = x̃⊤
                                                                             
                                                              w ⊕ . . . ⊕ x̃w   ,
                                                            |     {z        }
                                                                    k+1

where x̃w = [ReLU(x⊤                    ⊤ ⊤
                      w ) ⊕ ReLU(−xw )] . Observe that this encoder can be parametrized using a 2-layer MLP, and that
x̃w can be decoded using a single linear layer to get back to the initial features:
                                                                                   
                                                                                  ⊤ ⊤
                              DEC(x̃w ) = DEC ReLU(x⊤           w ) ⊕ ReLU(−xw )        = xw


Individualizing the Graph. Importantly, we encode the features using 2(k + 1)d(0) dimensions in order to be able to
preserve the original node features. Using the construction from Lemma A.1, we can ensure that every pair of nodes in the

                                                             14
                                                Cooperative Graph Neural Networks

connected graph have different features with probability 1 − δ1 . However, if we do this naı̈vely, then the node features will be
changed before we can transmit them to the target node. We therefore make sure that the width of the C O -GNN architecture
from Lemma A.1 is increased to 2(k + 1)d(0) dimensions such that it applies the identity mapping on all features beyond the
first 2d(0) components. This way we make sure that all feature components beyond the first 2d(0) components are preserved.
The existence of such a C O -GNN is straightforward since we can always do an identity mapping using base environment
models such S UM GNNs. We use L1 C O -GNN layers for this part of the construction.
In order for our architecture to retain a positive representation for all nodes, we now construct 2 additional layers which
                            (L)              (0)
encode the representation hw ∈ R2(k+1)d of each node w as follows:
                                               ⊤            ⊤
                                        [ReLU(qw ) ⊕ ReLU(−qw ) ⊕ x̃⊤             ⊤ ⊤
                                                                    w ⊕ . . . ⊕ x̃w ]
                 (0)                                                   (L )
where qw ∈ R2d         denotes a vector of the first 2d(0) entries of hw 1 .
Transmitting Information. Consider a shortest path u1 = w0 → w1 → · · · → wr → wr+1 = v of length r1 from node u1
to node v. We use exactly r1 C O -GNN layers in this part of the construction. For the first these layers, the action network
assigns the following actions to these nodes:

  • w0 performs the action B ROADCAST,
  • w1 performs the action L ISTEN, and
  • all other nodes are perform the action I SOLATE.

This is then repeated in the remaining layers, for all consecutive pairs wi , wi+1 , 0 ≤ i ≤ r until the whole path is traversed.
That is, at every layer, all graph edges are removed except the one between wi and wi+1 , for each 0 ≤ i ≤ r. By construction
each element in the node representations is positive and so we can ignore the ReLU.
We apply the former construction such that it acts on entries 2d(0) to 3d(0) of the node representations, resulting in the
following representation for node v:
                                             ⊤            ⊤
                                      [ReLU(qw ) ⊕ ReLU(−qw ) ⊕ x̃u1 ⊕ x̃v ⊕ . . . ⊕ x̃v ]
                 (0)                                                   (L )
where qw ∈ R2d         denotes a vector of the first 2d(0) entries of hw 1 .
                                                                                                               (t)
We denote the probability in which node y does not follow the construction at stage 1 ≤ t ≤ r by βy such that the
                                                                                                                      |V |
probability that all graph edges are removed except the one between wi and wi+1 at stage t is lower bounded by (1 − β) ,
                                                                                                  |V |r1
where β = maxy∈V (βy ). Thus, the probability that the construction holds is bounded by (1 − β)          .
The same process is then repeated for nodes ui , 2 ≤ i ≤ k, acting on the entries (k + 1)d(0) to (k + 2)d(0) of the node
representations and resulting in the following representation for node v:
                                            ⊤            ⊤
                                     [ReLU(qw ) ⊕ ReLU(−qw ) ⊕ x̃u1 ⊕ x̃u2 ⊕ . . . ⊕ x̃uk ]
                                                                                                    (0)              (0)
In order to decode the positive features, we construct the feature decoder DEC′ : R2(k+2)d                → R(k+1)d , that for
1 ≤ i ≤ k applies DEC to entries 2(i + 1)d(0) to (i + 2)d(0) of its input as follows:
                                     [DEC(x̃u1 ) ⊕ . . . ⊕ DEC(x̃uk )] = [xu1 ⊕ . . . ⊕ xuk ]

Given ϵ, δ, we set:
                                                                  1−δ
                                            δ2 = 1 −                           Pk+1      > 0.
                                                                        |V |    i=1 ri
                                                       (1 − δ1 ) (1 − β)

Having transmitted and decoded all the required features into x = [x1 ⊕ . . . ⊕ xk ], where xi denotes the vector of entries
                                                                    (0)
id(0) to (i + 1)d(0) for 0 ≤ i ≤ k, we can now use an MLP : R(k+1)d → Rd and the universal approximation property to
                                             (L)
map this vector to the final representation hv such that:
                                                                                        Pk+1
                                                                                 |V |          r
                   P(|h(L)
                        v − f (xu1 , . . . xuk )| < ϵ) ≥ (1 − δ1 ) (1 − β)
                                                                           i=1 i
                                                                                 (1 − δ2 ) ≥ 1 − δ.
                                                  Pk     
The construction hence requires L = L1 + 2 + i=0 ri C O -GNN layers.

                                                                 15
                                             Cooperative Graph Neural Networks

B. Relation to Over-squashing
Over-squashing refers to the failure of message passing to propagate information on the graph. Topping et al. (2022) and
Di Giovanni et al. (2023) formalized over-squashing as the insensitivity of an r-layer MPNN output at node u to the input
                                                                              (r)
features of a distant node v, expressed through a bound on the Jacobian ∥∂hv /∂xu ∥ ≤ C r (Âr )vu , where C encapsulated
architecture-related constants (e.g., width, smoothness of the activation function, etc.) and the normalized adjacency matrix
Â captures the effect of the graph. Graph rewiring techniques amount to modifying Â so as to increase the upper bound and
thereby reduce the effect of over-squashing.
Observe that the actions of every node in C O -GNNs result in an effective graph rewiring (different at every layer). As
a result, the action network can choose actions that transmit the features of node u ∈ V to node v ∈ V as shown in
Proposition 5.2, resulting in the maximization of the bound on the Jacobian between a pair of nodes or (k nodes, for some
fixed k).

C. Additional Experiments
C.1. Expressivity Experiment
In Proposition 5.1 we state that C O -GNNs can distinguish between pairs of graphs which are 1-WL indistinguishable. We
validate this with a simple synthetic dataset: C YCLES. C YCLES consists of 7 pairs of undirected graphs, where the first graph
is a k-cycle for k ∈ [6, 12] and the second graph is a disjoint union of a (k−3)-cycle and a triangle. The train/validation/test
set are the k ∈ [6, 7]/[8, 9]/[10, 12] pairs, correspondingly. The task is to correctly identify the cycle graphs. As the pairs
are 1-WL indistinguishable, solving this task implies a strictly higher expressive power than 1-WL.
Our main finding is that C O -GNN(Σ, Σ) and C O -GNN(µ, µ) achieve 100% accuracy, perfectly classifying the cycles,
whereas their corresponding classical S UM GNN and M EAN GNN achieve a random guess accuracy of 50%. These results
imply that C O -GNN can increase the expressive power of their classical counterparts. We find the model behaviour rather
volatile during training, which necessitated careful tuning of hyperparameters.

C.2. Long-range Interactions
To validate the performance of C O -GNNs on long-range tasks, we exper-
iment with the LRGB benchmark (Dwivedi et al., 2022).                              Table 3: Results on LRGB. Top three mod-
Setup. We train C O -GNN(∗, ∗) and C O -GNN(ϵ, ϵ) C O -GNN(ϵ, ϵ) on                els are colored by First, Second, Third.
LRGB and report the unweighted mean Average Precision (AP) for Peptides-
func. All experiments are run 4 times with 4 different seeds and follow the                                 Peptides-func
data splits provided by Dwivedi et al. (2022). Following the methodology               GCN                 0.6860 ± 0.0050
of Tönshoff et al. (2023), we used AdamW as optimizer and cosine-with-                GINE                0.6621 ± 0.0067
warmup scheduler. We also use the provided results for GCN, GCNII                      GatedGCN            0.6765 ± 0.0047
(Chen et al., 2020), GINE, GatedGCN (Bresson & Laurent, 2018), CRaWl                   CRaWl               0.7074 ± 0.0032
(Tönshoff et al., 2023), DRew (Gutteridge et al., 2023), Exphormer (Shirzad           DRew                0.7150 ± 0.0044
et al., 2023), GRIT (Ma et al., 2023), Graph-ViT / G-MLPMixer (He et al.,              Exphormer           0.6527 ± 0.0043
2023).                                                                                 GRIT                0.6988 ± 0.0082
Results. We follow Tönshoff et al. (2023) who identified that the previously          Graph-ViT           0.6942 ± 0.0075
reported large performance gaps between classical MPNNs and transformer-               G-MLPMixer          0.6921 ± 0.0054
based models can be closed by a more extensive tuning of MPNNs. In light               C O -GNN(∗, ∗) 0.6990 ± 0.0093
of this, we note that the performance gap between different models is not              C O -GNN(ϵ, ϵ) 0.6963 ± 0.0076
large. Classical MPNNs such as GCN, GINE, and GatedGCN surpass some
transformer-based approaches such as Exphormer. C O -GNN(∗, ∗) further
improves on the competitive GCN and is the third best performing model
after DRew and CRaWl. Similarly, C O -GNN(ϵ, ϵ) closely matches C O -GNN(∗, ∗) and is substantially better than its base
architecture GIN. This experiment further suggests that exploring different classes of C O -GNNs is a promising direction, as
C O -GNNs typically boost the performance of their underlying base architecture.




                                                              16
                                           Cooperative Graph Neural Networks

             Table 4: Results on graph classification. Top three models are colored by First, Second, Third.

                            IMDB-B IMDB-M REDDIT-B REDDIT-M                      NCI1       PROTEINS ENZYMES
          DGCNN             69.2 ± 3.0 45.6 ± 3.4   87.8 ± 2.5    49.2 ± 1.2   76.4 ± 1.7   72.9 ± 3.5   38.9 ± 5.7
          DiffPool          68.4 ± 3.3 45.6 ± 3.4   89.1 ± 1.6    53.8 ± 1.4   76.9 ± 1.9   73.7 ± 3.5   59.5 ± 5.6
          ECC               67.7 ± 2.8 43.5 ± 3.1     OOR           OOR        76.2 ± 1.4   72.3 ± 3.4   29.5 ± 8.2
          GIN               71.2 ± 3.9 48.5 ± 3.3   89.9 ± 1.9    56.1 ± 1.7   80.0 ± 1.4   73.3 ± 4.0   59.6 ± 4.5
          GraphSAGE         68.8 ± 4.5 47.6 ± 3.5   84.3 ± 1.9    50.0 ± 1.3   76.0 ± 1.8   73.0 ± 4.5   58.2 ± 6.0
          CGMM                  -          -        88.1 ± 1.9    52.4 ± 2.2   76.2 ± 2.0       -            -
          ICGMMf            71.8 ± 4.4 49.0 ± 3.8   91.6 ± 2.1    55.6 ± 1.7   76.4 ± 1.4   73.2 ± 3.9       -
          SPN(k = 5)            -          -            -             -        78.6 ± 1.7   74.2 ± 2.7   69.4 ± 6.2
          GSPN                  -          -        90.5 ± 1.1    55.3 ± 2.0   76.6 ± 1.9       -            -
          C O -GNN(Σ, Σ) 70.8 ± 3.3 48.5 ± 4.0 88.6 ± 2.2         53.6 ± 2.3 80.6 ± 1.1 73.1 ± 2.3       65.7 ± 4.9
          C O -GNN(µ, µ) 72.2 ± 4.1 49.9 ± 4.5 90.5 ± 1.9         56.3 ± 2.1 79.4 ± 0.7 71.3 ± 2.0       68.3 ± 5.7




C.3. Graph Classification
In this experiment, we evaluate C O -GNNs on the TUDataset (Morris et al., 2020) graph classification benchmark.
Setup. We evaluate C O -GNN(Σ, Σ) and C O -GNN(µ, µ) on the 7 graph classification benchmarks, following the risk
assessment protocol of Errica et al. (2020), and report the mean accuracy and standard deviation. The results for the
baselines DGCNN (Wang et al., 2019), DiffPool (Ying et al., 2018), Edge-Conditioned Convolution (ECC) (Simonovsky &
Komodakis, 2017), GIN, GraphSAGE are from Errica et al. (2020). We also include CGMM (Bacciu et al., 2020), ICGMMf
(Castellana et al., 2022), SPN(k = 5) (Abboud et al., 2022) and GSPN (Errica & Niepert, 2023) as more recent baselines.
OOR (Out of Resources) implies extremely long training time or GPU memory usage. We use Adam optimizer and StepLR
learn rate scheduler, and report all hyperparameters in the appendix (Table 13).
Results. C O -GNN models achieve the highest accuracy on three datasets in Table 4 and remain competitive on the other
datasets. C O -GNN yield these performance improvements, despite using relatively simple action and environment networks,
which is intriguing as C O -GNNs unlock a large design space which includes a large class of model variations.

C.4. Homophilic Node Classification
In this experiment, we evaluate C O -GNNs on the homophilic
node classification benchmarks cora and pubmed (Sen et al.,          Table 5: Results on homophilic datasets. Top three
2008).                                                               models are colored by First, Second, Third.
Setup.      We assess M EAN GNN, S UM GNN and their                                             pubmed           cora
corresponding C O -GNNs counterparts C O -GNN(µ, µ) and
C O -GNN(Σ, Σ) on the homophilic graphs and their 10 fixed                 MLP                87.16 ± 0.37   75.69 ± 2.00
splits provided by Pei et al. (2020), where we report the mean             GCN                88.42 ± 0.50   86.98 ± 1.27
accuracy, standard deviation and the accuracy gain due to the              GraphSAGE          88.45 ± 0.50   86.90 ± 1.04
application of C O -GNN. We also use the results provided by               GAT                87.30 ± 1.10   86.33 ± 0.48
Bodnar et al. (2023) for the classical baseline: GCN, Graph-               Geom-GCN           87.53 ± 0.44   85.35 ± 1.57
SAGE, GAT, Geom-GCN (Pei et al., 2020) and GCNII.                          GCNII              90.15 ± 0.43   88.37 ± 1.25

Results. Table 5 illustrates a modest performance increase of              S UM GNN           88.58 ± 0.57   84.80 ± 1.71
1-2% across all datasets when transitioning from S UM GNN,                 M EAN GNN          88.66 ± 0.44   84.50 ± 1.25
M EAN GNN, and GCN to their respective C O -GNN counter-                   C O -GNN(Σ, Σ) 89.39 ± 0.39       86.43 ± 1.28
parts. These datasets are highly homophilic, but C O -GNNs                 C O -GNN(µ, µ) 89.60 ± 0.42       86.53 ± 1.20
nonetheless show improvements on these datasets (even though,              C O -GNN(∗, ∗) 89.51 ± 0.88       87.44 ± 0.85
modest) compared to their environment/action network archi-
tectures.

                                                             17
                                           Cooperative Graph Neural Networks




Figure 7: The accuracy of C O -GNN(µ, µ) and M EAN GNN on cora (left) and on roman-empire (right) for an increasing
number of layers.


C.5. Over-smoothing Experiments
Section 5.1 explains that C O -GNNs can mitigate the over-smoothing phenomenon, through the choice of B ROADCAST or
I SOLATE actions. To validate this, we experiment with an increasing number of layers of C O -GNN(µ, µ) and M EAN GNN
over the cora and roman-empire datasets.
Setup. We evaluate C O -GNN(µ, µ) and M EAN GNN over the cora and roman-empire datasets, following the 10 data splits
of Pei et al. (2020) and Platonov et al. (2023), respectively. We report the accuracy and standard deviation.
Results. Figure 7 indicates that the performance is generally retained for deep models and that C O -GNNs are effective in
alleviating the over-smoothing phenomenon even though their base GNNs suffer from performance deterioration already
with a few layers.

D. Runtime Analysis
Consider a GCN model with L layers and a hidden dimension of d on an input graph G = (V, E, X). Wu et al. (2019) has
shown the time complexity of this model to be O(Ld(|E|d + |V |)). To extend this analysis to C O -GNNs, let us consider a
C O -GNN(∗, ∗) architecture composed of:

  • a GCN environment network η with Lη layers and hidden dimension of dη , and

  • a GCN action network π with Lπ layers and hidden dimension of dπ .

A single C O -GNN layer first computes the actions for each node by feeding node representations through the action network
π, which is then used in the aggregation performed by the environment layer. This means that the time complexity of a single
C O -GNN layer is O(Lπ dπ (|E|dπ + |V |) + dη (|E|dη + |V |)). The time complexity of the whole C O -GNN architecture is
then O(Lη Lπ dπ (|E|dπ + |V |) + Lη dη (|E|dη + |V |)).
Typically, the hidden dimensions of the environment network and action network match. In all of our experiments, the depth
of the action network Lπ is much smaller (typically ≤ 3) than that of the environment network Lη . Therefore, assuming
Lπ << Lη we get that a runtime complexity of O(Lη dη (|E|dη + |V |)), matching the runtime of a GCN model.
To empirically confirm the efficiency of C O -GNNs, we report in Figure 8 the duration of a forward pass of a C O -GNN(∗, ∗)
and GCN with matching hyperparameters across multiple datasets. From Figure 8, it is evident that the increase in runtime
is linearly related to its corresponding base model with R2 values higher or equal to 0.98 across 4 datasets from different
domains. Note that, for the datasets IMDB-B and PROTEINS, we report the average forward duration for a single graph in a
batch.

                                                            18
                                           Cooperative Graph Neural Networks




    Figure 8: Empirical runtimes: C O -GNN(∗, ∗) forward pass duration as a function of GCN forward pass duration.


E. Further Details of the Experiments Reported in the Paper
E.1. The Gumbel Distribution and the Gumbel-softmax Temperature
The Gumbel distribution is used to model the distribution of the maximum (or the minimum) of a set of random variables.
Its probability density function has a distinctive, skewed shape,
with heavy tails, making it a valuable tool for analyzing and
quantifying the likelihood of rare and extreme occurrences. By
applying the Gumbel distribution to the logits or scores associated
with discrete choices, the Gumbel-Softmax estimator transforms
them into a probability distribution over the discrete options.
The probability density function of a variable that follows X ∼
                                −x
Gumbel(0, 1) is f (x) = e−x+e (Figure 9).
The Straight-through Gumbel-softmax estimator is known to
benefit from learning an inverse-temperature before sampling
an action, which we use in our experimental setup. For a given
graph G = (V, E, X) the inverse-temperature of node v ∈ V is
estimated by applying a bias-free linear layer L : Rd → R to the
intermediate representation h ∈ Rd . To ensure the temperature
is positive, an approximation of the ReLU function with a bias                                       −x
hyperparameter τ ∈ R is subsequently applied:                      Figure 9: The pdf f (x) = e−x+e        of Gumbel(0, 1).
                1
                    = log 1 + exp ω T h + τ0
                                       
              τ (h)
where τ0 controls the maximum possible temperature value.

                                                           19
                                           Cooperative Graph Neural Networks

E.2. Dataset Statistics
The statistics of the real-world long-range, node-based, and graph-based benchmarks used can be found in Tables 6 to 9.

                          Table 6: Statistics of the heterophilic node classification benchmarks.

                                  roman-empire amazon-ratings minesweeper          tolokers     questions
                # nodes               22662             24492         10000         11758   48921
                # edges               32927             93050         39402        519000  153540
                # node features        300               300             7            10     301
                # classes               18                 5             2             2       2
                edge homophily         0.05              0.38          0.68          0.59    0.84
                metrics               ACC               ACC          AUC-ROC      AUC-ROC AUC-ROC


                            Table 7: Statistics of the long-range graph benchmarks (LRGB).

                                                                 Peptides-func
                                            # graphs                15535
                                            # average nodes         150.94
                                            # average edges         307.30
                                            # classes                 10
                                            metrics                  AP


                                Table 8: Statistics of the graph classification benchmarks.

                                  IMDB-B      IMDB-M      REDDIT-B       NCI1    PROTEINS         ENZYMES
              # graphs             1000         1500         2000        4110       1113             600
              # average nodes      19.77        13.00       429.63       29.87      39.06           32.63
              # average edges      96.53        65.94       497.75       32.30      72.82           64.14
              # classes              2            3            2           2          2               6
              metrics              ACC          ACC          ACC         ACC        ACC             ACC


                          Table 9: Statistics of the homophilic node classification benchmarks.

                                                                pubmed cora
                                              # nodes         18717 2708
                                              # edges         44327 5278
                                              # node features  500 1433
                                              # classes         3     6
                                              edge homophily 0.80 0.81
                                              metrics         ACC ACC


E.3. ROOT N EIGHBORS: Dataset Generation
In Section 6.1, we compare C O -GNNs to a class of MPNNs on a dedicated synthetic dataset ROOT N EIGHBORS in order to
assess the model’s ability to redirect the information flow. ROOT N EIGHBORS consists of 3000 trees of depth 2 with random
node features of dimension d = 5 which is generated as follows:

 • Features: Each feature is independently sampled from a uniform distribution U [−2, 2].

                                                            20
                                              Cooperative Graph Neural Networks

  • Level-1 Nodes: The number of nodes in the first level of each tree in the train, validation, and test set is sampled from a
    uniform distribution U [3, 10], U [5, 12], and U [5, 12] respectively. Then, the degrees of the level-1 nodes are sampled as
    follows:
       • The number of level-1 nodes with a degree of 6 is sampled independently from a uniform distribution
         U [1, 3], U [3, 5], U [3, 5] for the train, validation, and test set, respectively.
       • The degree of the remaining level-1 nodes are sampled from the uniform distribution U [2, 3].

We use a train, validation, and test split of equal size.

E.4. Hyperparameters for all Experiments
In Tables 10 to 14, we report the hyperparameters used in our experiments.

                           Table 10: Hyperparameters used for ROOT N EIGHBORS and C YCLES.

                                                            ROOT N EIGHBORS C YCLES
                                         η # layers                 1            2
                                         η dim                   16, 32         32
                                         π # layers                1, 2          6
                                         π dim                    8, 16         32
                                         learned temp               ✓            -
                                         temp                       -            1
                                         τ0                        0.1           -
                                         # epochs                10000         1000
                                         dropout                   0             0
                                         learn rate              10−3          10−3
                                         batch size                -            14
                                         pooling                   -           sum


                    Table 11: Hyperparameters used for the heterophilic node classification benchmarks.

                                    roman-empire        amazon-ratings     minesweeper       tolokers   questions
            η # layers                  5-12                   5-10            8-15           5-10       5-9
            η dim                    128,256,512              128,256       32,64,128         16,32     32,64
            π # layers                   1-3                    1-6             1-3            1-3       1-3
            π dim                      4,8,16                4,8,16,32     4,8,16,32,64     4,8,16,32 4,8,16,32
            learned temp                  ✓                      ✓              ✓               ✓         ✓
            τ0                          0,0.1                  0,0.1           0,0.1          0,0.1     0,0.1
            # epochs                    3000                   3000            3000           3000      3000
            dropout                    0.2                 0.2                 0.2             0.2      0.2
            learn rate          3 · 10−3 , 3 · 10−5 3 · 10−4 , 3 · 10−5 3 · 10−3 , 3 · 10−5 3 · 10−3 10−3 , 10−2
            activation function       GeLU                GeLU                GeLU           GeLU      GeLU
            skip connections            ✓                   ✓                   ✓               ✓        ✓
            layer normalization         ✓                   ✓                   ✓               ✓        ✓




                                                                 21
                                       Cooperative Graph Neural Networks

             Table 12: Hyperparameters used for the long-range graph benchmarks (LRGB).

                                                               Peptides-func
                                       η # layers                  5-9
                                       η dim                     200,300
                                       π # layers                  1-3
                                       π dim                     8,16,32
                                       learned temp                 ✓
                                       τ0                          0.5
                                       # epochs                    500
                                       dropout                    0
                                       learn rate          3 · 10−4 , 10−3
                                       # decoder layer           2,3
                                       # warmup epochs            5
                                       positional encoding LapPE, RWSE
                                       batch norm                ✓
                                       skip connections          ✓


                Table 13: Hyperparameters used for social networks and proteins datasets.

                       IMDB-B IMDB-M REDDIT-B REDDIT-M                         NCI1        PROTEINS ENZYMES
η # layers                 1           1          3,6             6            2,5            3,5     1,2
η dim                    32,64      64,256      128,256        64, 128     64,128,256         64    128,256
π # layers                 2           3          1,2             1             2             1,2      1
π dim                    16,32        16         16,32           16           8, 16            8       8
learned temp.             ✓           ✓           ✓              ✓             ✓              ✓       ✓
τ0                        0.1         0.1         0.1            0.1           0.5            0.5     0.5
# epochs                 5000        5000        5000           5000          3000           3000    3000
dropout                   0.5        0.5          0.5            0.5           0               0       0
learn rate               10−4       10−3         10−3           10−4       10−3 ,10−2        10−3    10−3
pooling                  mean       mean         mean           mean         mean            mean    mean
batch size                32         32           32             32           32              32      32
scheduler step size       50         50           50             50           50              50      50
scheduler gamma           0.5        0.5          0.5            0.5          0.5             0.5     0.5


           Table 14: Hyperparameters used for the homophilic node classification benchmarks.

                                                   pubmed                       citeseer
                      η # layers                      1-3                         1-3
                      η dim                       32,64,128                    32,64,128
                      π # layers                      1-3                         1-3
                      π dim                         4,8,16                      4,8,16
                      temperature                    0.01                        0.01
                      τ0                              0.1                         0.1
                      # epochs                       2000                        2000
                      dropout                         0.5                        0.5
                      learn rate          5 · 10−3 , 10−2 , 5 · 10−2 5 · 10−3 , 10−2 , 5 · 10−2
                      learn rate decay        5 · 10−6 , 5 · 10−4        5 · 10−6 , 5 · 10−4
                      activation function           ReLU                       ReLU


                                                          22
                                            Cooperative Graph Neural Networks

F. Visualizing the Actions
We extend the discussion about C O -GNNs dynamic topology over the Minesweeper dataset in Section 7.2 and present the
evolution of the graph topology from layer ℓ = 1 to layer ℓ = 8.
Setup. We train a 10-layered C O -GNN(µ, µ) model and present the evolution of the graph topology from layer ℓ = 1 to
layer ℓ = 8. We choose a node (black), and at every layer ℓ, we depict its neighbors up to distance 10. In this visualization,
nodes which are mines are shown in red, and other nodes in blue. The features of non-mine nodes (indicating the number of
neighboring mines) are shown explicitly whereas the nodes whose features are hidden are labeled with a question mark. For
each layer ℓ, we gray out the nodes whose information cannot reach the black node with the remaining layers available.




                                                             23
       Cooperative Graph Neural Networks




Figure 10: The 10-hop neighborhood at layer ℓ = 1.




Figure 11: The 10-hop neighborhood at layer ℓ = 2.


                       24
       Cooperative Graph Neural Networks




Figure 12: The 10-hop neighborhood at layer ℓ = 3.




Figure 13: The 10-hop neighborhood at layer ℓ = 4.


                       25
       Cooperative Graph Neural Networks




Figure 14: The 10-hop neighborhood at layer ℓ = 5.




Figure 15: The 10-hop neighborhood at layer ℓ = 6.


                       26
       Cooperative Graph Neural Networks




Figure 16: The 10-hop neighborhood at layer ℓ = 7.




Figure 17: The 10-hop neighborhood at layer ℓ = 8.


                       27

