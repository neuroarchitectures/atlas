# RouteNet RouteNet 2020

> Source: `RouteNet_RouteNet_2020.pdf`

---

                                        IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY                                                                    1




                                           RouteNet: Leveraging Graph Neural Networks for
                                             network modeling and optimization in SDN
                                                  Krzysztof Rusek, José Suárez-Varela, Paul Almasan, Pere Barlet-Ros and Albert Cabellos-Aparicio
                                              NOTE: This is an extended version of a paper published in IEEE JSAC. Please use the following reference to cite this work:
                                           K. Rusek, J. Suárez-Varela, P. Almasan, P. Barlet-Ros and A. Cabellos-Aparicio, ”RouteNet: Leveraging Graph Neural Networks for
                                         network modeling and optimization in SDN,” in IEEE Journal on Selected Areas in Communications, doi: 10.1109/JSAC.2020.3000405.

                                           Abstract—Network modeling is a key enabler to achieve effi-                achieved by combining two main elements: (i) a network
                                        cient network operation in future self-driving Software-Defined               model, and (ii) an optimization algorithm. In this well-known
                                        Networks. However, we still lack functional network models able               optimization architecture, the network model is tasked to
                                        to produce accurate predictions of Key Performance Indicators
                                                                                                                      predict the resulting performance (e.g, delay, packet loss)




arXiv:1910.01508v2 [cs.NI] 9 Jul 2020
                                        (KPI) such as delay, jitter or loss at limited cost. In this
                                        paper we propose RouteNet, a novel network model based on                     for specific configurations, and the optimization algorithm
                                        Graph Neural Network (GNN) that is able to understand the                     iteratively explores different configurations until it finds one
                                        complex relationship between topology, routing, and input traffic             that meets the optimization goals.
                                        to produce accurate estimates of the per-source/destination per-                 One fundamental issue of network optimization solutions is
                                        packet delay distribution and loss. RouteNet leverages the ability
                                        of GNNs to learn and model graph-structured information and as                that they can only optimize based on the performance metrics
                                        a result, our model is able to generalize over arbitrary topologies,          provided by the network model. Thus, in order to optimize
                                        routing schemes and traffic intensity. In our evaluation, we show             Key Performance Indicators (KPI) such as delay or packet
                                        that RouteNet is able to predict accurately the delay distribution            loss in networks, it is essential a network model able to
                                        (mean delay and jitter) and loss even in topologies, routing and              understand how these performance indicators are related to
                                        traffic unseen in the training (worst case MRE=15.4%). Also, we
                                        present several use cases where we leverage the KPI predictions               the network state metrics collected from the data plane, which
                                        of our GNN model to achieve efficient routing optimization and                often can provide only timely statistics of traffic volume (e.g.,
                                        network planning.                                                             traffic matrix) in real-world deployments. In this context, much
                                          Index Terms—Graph neural networks, network modeling,                        effort has been devoted in the past to build network models
                                        network optimization, Software-Defined Networks                               able to predict performance metrics, however nowadays we
                                                                                                                      still lack functional models providing accurate predictions
                                                                 I. I NTRODUCTION                                     of relevant KPI like delay, jitter or packet loss. Analytic
                                                                                                                      models, mainly based on Queuing Theory [2], assume some
                                           Network modeling is a fundamental component to achieve                     non-realistic properties of networks (e.g., traffic with Poisson
                                        efficient network optimization with special attention on future               distribution, probabilistic routing) and, as a result, they are not
                                        self-driving networks [1]. In the context of Software-Defined                 accurate to produce KPI predictions in large-scale networks
                                        Networks, networking tasks are orchestrated from a centralized                with realistic configurations such as multi-hop routing [3].
                                        control plane, which may leverage a global picture of the                     Conversely, packet-level network simulators showed to be very
                                        network state in order to operate networks efficiently and                    accurate for this purpose, but their high computational cost
                                        dynamically adapt to changes in the network. To this end,                     makes it unfeasible to leverage them to operate networks in
                                        network administrators typically define a target policy that                  short time scales.
                                        may include some optimization objectives (e.g., minimize                         In this context, Deep Learning [4] seems to be a well-
                                        end-to-end latency) and constraints (e.g., security policy).                  suited alternative to develop a new breed of network models
                                        Then, SDN controllers are tasked to find some changes in                      that can be both accurate and lightweight. Relevant research
                                        the network configuration (e.g., routing) to accomplish the                   efforts are being devoted to apply neural networks to model
                                        optimization objectives set by administrators. This is typically              computer networks [5] and using such models for network
                                          Krzysztof Rusek is with the Department of Telecommunications, AGH Uni-      optimization [6], [7], [3]. Existing proposals [8], [9] typi-
                                        versity of Science and Technology, Krakow, Poland, and with the Barcelona     cally used well-known Neural Networks (NN) architectures
                                        Neural Networking Center, Universitat Politècnica de Catalunya, Barcelona,   like fully-connected Neural Networks, Convolutional Neural
                                        Spain. (e-mail: krusek@agh.edu.pl).
                                          José Suárez-Varela, Paul Almasan, Pere Barlet-Ros and Albert Cabellos-    Networks, Recurrent Neural Networks or Variational Auto-
                                        Aparicio are with the Barcelona Neural Networking Center,                     Encoders. However, computer networks are fundamentally
                                        Universitat Politècnica de Catalunya, Barcelona, Spain (e-mail:              represented as graphs, and such types of NN are not designed
                                        {jsuarezv,almasan,pbarlet,acabello}@ac.upc.edu).
                                          This work was supported by the Polish Ministry of Science and Higher        to learn graph-structured information. As a result, the models
                                        Education with the subvention funds of the Faculty of Computer Sci-           trained result in limited accuracy and are unable to generalize
                                        ence, Electronics and Telecommunications of AGH University, the Spanish       in terms of topologies or routing configurations.
                                        MINECO under contract TEC2017-90034-C2-1-R (ALLIANCE), the Catalan
                                        Institution for Research and Advanced Studies (ICREA) and the FI-AGAUR           In this paper we present RouteNet, a novel network model
                                        grant by the Catalan Government. The research was also supported in part by   based on Graph Neural Networks (GNN) [10]. Our model is
                                        PL-Grid Infrastructure.                                                       able to understand the complex relationship between topol-
                                        10.1109/JSAC.2020.3000405 c 2020 IEEE                                         ogy, routing, and input traffic to accurately estimate the
2                                                     IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY



                                                                                     Data plane   Control plane
distribution of the per-packet delay and loss ratio on every
source-destination pair. GNNs are tailored to achieve relational                                                                                knowledge plane
                                                                                                                     Performance prediction
reasoning and combinatorial generalization over information
                                                                                                         Target policy


                                                                                                                                                          src-dst KPI predictions
structured as graphs [11] and as a result our model is
                                                                                                                               Topology   ]
able to generalize over arbitrary topologies, routing schemes                                                                   Routing   ]   RouteNet
                                                                                                             Optimizer

                                                                                                                                                          (delay, jitter and loss)
and variable traffic intensity. In particular, RouteNet captures                                                                 Traffic
                                                                                                                                 matrix   ]
meaningfully traffic routing over network topologies. This
is achieved by modeling the relationships of the links in                                                            Evaluate configuration

topologies with the source-destination paths resulting from the
routing schemes and the traffic flowing through them. One            Fig. 1. Architecture for network optimization in SDN
main contribution of the RouteNet architecture compared to
other GNN based models [12] is the representation of paths
                                                                        • New network optimization use cases incorporating packet
as ordered sequences of links. This makes RouteNet a new
                                                                          loss requirements.
GNN architecture designed especially for computer network
                                                                        • More diverse and larger network topologies (with 14, 17,
control and management.
   An earlier version of this paper was presented at [13]. In that        24 and 50 nodes)
version, two different models were used to predict the per-path
mean delay and jitter. In this paper we present an extended                 II. SDN- BASED M ODELING AND O PTIMIZATION
RouteNet model inspired by Generalized Linear Models that                                    S CENARIO
directly estimates the per-packet distribution of the delay on          Network modeling enables the control plane to further
each path. This enables to use a single model to predict any         exploit the potential of SDN to perform fine-grained manage-
metric associated to end-to-end per-packet delay (e.g., mean         ment. This permits to evaluate the resulting performance of
delay, jitter). Additionally, in this paper we adapted RouteNet      what-if scenarios without the necessity to modify the state of
to make also predictions of the per-source/destination packet        the data plane. It may be profitable for a number of network
loss ratio.                                                          control and management applications such as optimization,
   We evaluated the accuracy of our GNN model with a dataset         planning or fast failure recovery. For instance, in Fig. 1
generated using a packet-level simulator (Omnet++ [14]), and         we show an architecture for network optimization within
this resulted in high estimation accuracy of delay, jitter, and      the context of the knowledge-Defined Networking (KDN)
loss when testing it against topologies, routing and traffic not     paradigm [1]. In this case, we assume that the control plane
seen during training. More importantly, we verify that our           receives timely updates of the network state (e.g., traffic
model is able to generalize and, for instance, when training         matrix, delay measurements). This can be achieved by means
the model with samples of 14-node, 24-node and 50-node               of “conventional” SDN-based measurement techniques (e.g.,
topologies the model is able to provide accurate estimates in a      OpenFlow [16], OpenSketch [17]) or more novel telemetry
never-seen 17-node network (MRE=15.4% in the worst case).            proposals such as INT for P4 [18] or iOAM [19]. Likewise,
   Finally, and in order to showcase the potential of our GNN        in the knowledge plane there is an optimizer whose behavior
model we present a series of use cases applicable to a SDN           is defined by a given target policy. This policy, in line with
architecture. In contrast to the use cases presented in [13], in     intent-based networking, may be defined by a declarative
this paper we include network scenarios that leverage also           language such as NEMO [20] and finally being translated
the new RouteNet model that predicts the packet loss to              to a (multi-objective) network optimization problem. At this
perform a joint optimization of mean delay, jitter, and loss. We     point, an accurate network model can play a crucial role in
first show that RouteNet can be used to optimize the routing         the optimization process by leveraging it to run optimization
configuration in QoS-aware scenarios with delay, jitter and loss     algorithms (e.g., hill-climbing) that iteratively explore the
requirements, and benchmark it against traditional utilization-      performance of candidate solutions in order to find the optimal
aware models (e.g., OSPF) and the optimal solution using a           configuration. We intentionally leave out of the scope of this
packet-level simulator. Also, we leverage the predictions of         architecture the training phase.
RouteNet in a network planning use case to select the optimal           To be successful in scenarios like the one proposed above,
link placement.                                                      the network model should meet two main requirements:
   We summarize below the main contributions of this paper           (i) accurate performance prediction, and (ii) low computa-
compared to the previous one and the state-of-the-art:               tional cost to enable network operation in short time scales.
    • Probabilistic modeling inspired by Generalized Linear          Moreover, it is essential for optimizers to have enough flexi-
      Models                                                         bility to predict the resulting performance after changing the
    • Adaptation to predictions of the per-source/destination        network configuration, vary the input traffic, or modify the
      packet loss ratio                                              topology (e.g., link failure). To this end, we rely on the capabil-
    • Residual connection to facilitate the training (like in        ity of Graph Neural Networks (GNN) to efficiently operate and
      ResNet [15])                                                   generalize over graph-structured data. RouteNet, the GNN-
    • Computation cost improvement (∼10× faster)                     based model proposed in this paper, is able to propagate any
    • Additional input features (support for arbitrary link ca-      routing scheme throughout a network topology and abstract
      pacities)                                                      meaningful information of the current network state to produce
RUSEK AND SUÁREZ-VARELA et al.                                                                                                                     3



relevant performance estimates. More in detail, RouteNet                    as output end-to-end performance predictions. The main as-
(Fig. 1) takes as input (i) a given topology, (ii) a source-                sumption behind RouteNet is that information at the path level
destination routing scheme (i.e., list of end-to-end paths) and             (e.g., end-to-end metrics such as delays or packet loss) and the
(iii) a traffic matrix (defined as the bandwidth between each               link level (e.g., link delay, packet loss rate, link utilization)
node pair in the network), and produces as output performance               can be encoded in learnable vectors of real numbers (path and
metrics according to the current network state (per-path mean               link state vectors respectively). Note that the path abstraction
delay, jitter, and packet loss). To achieve it, RouteNet uses               may not necessarily correspond to a physical path. It could
fixed-dimension vectors that encode information about the                   be a generic end-to-end traffic flow. For instance, an MPLS
state of paths and links and propagate the information among                tunnel. Based on this assumption, RouteNet is built upon the
them according to the input topology and the routing scheme.                following principles:
                                                                              1) The state of a path depends on the state of all the links
            III. N ETWORK M ODELING WITH GNN                                      that lie on the path.
A. Notation                                                                   2) The state of a link depends on the state of all the paths
   A computer network can be represented by a set of links                        that traverse the link.
N = {li |i ∈ (1, . . . , nl )}, and the routing scheme in the               In a more formal description, let the state of a link be denoted
network by a set of paths R = {pk |k ∈ (1, . . . , np )}. Each              by hli , which is an unknown hidden vector. Similarly, the
path is defined as a sequence of links pk = (lk(1) , . . . , lk(|pk |) ),   state of a path is defined by hpi . These principles can be
where k(i) is the index of the i-th link in the path k. The                 mathematically formulated with the following expressions:
properties (features) of both links and paths are denoted by xli
and xpi . Measurable KPIs are modeled as random variables                           hli = f (hp1 , . . . , hpj ),    li ∈ pk , k = 1, . . . , j   (1)
Wi and Li , where the former is the end-to-end delay and the                        hpk = g(hlk(1) , . . . , hlk(|pk |) )                         (2)
later is the total number of packet drops during a period of
                                                                            where f and g are some unknown functions. It is well-
time for every source-destination pair in the network.
                                                                            known that neural networks can work as universal function
                                                                            approximators. However, a direct approximation of functions
B. Background on Graph Neural Networks
                                                                            f and g is not possible in this case given that: (i) Equations (1)
   Graph neural network is an artificial neural architecture                and (2) define an implicit function (a nonlinear system of
designed for graph-structured data, where the nodes, edges,                 equations with the states being hidden variables), (ii) these
and the whole graph can have associated feature vectors.                    functions depend on the input routing scheme, and (iii) the
The most important property of GNN is that it preserves the                 dimensionality of each function is very large. This would
basic topological relations between node adjacencies (graph                 require a vast set of training samples.
isomorphism), so it is well suited to be used for different                    RouteNet represents a GNN architecture that learns effi-
topologies without retraining.                                              ciently f and g. It is invariant to the topology and routing
   Multiple GNN architectures have been proposed at the time                scheme and makes the neural function approximation feasi-
of this writing. Recently, Message Passing Neural Network                   ble. Algorithm 1 describes the forward propagation (and the
(MPNN) was proposed as a general family of GNN architec-                    internal architecture) of RouteNet. In this process, this GNN
tures [21]. Most of the existing GNN models can be described                model receives as input the initial path and link features xp , xl
as special cases of the MPNN framework.                                     and the routing description R, and outputs inferred per-path
   The main assumption of MPNN is that the information                      metrics (ŷp ). Note that we simplified the notation by dropping
related to nodes, edges or the whole graph can be encoded in                sub-indexes of paths and links.
fixed-dimension vectors, also called embeddings. The forward                   RouteNet’s architecture enables dealing with the circular de-
pass in MPNN is a combination of three simple functions:                    pendencies described in equations (1) and (2), and supporting
(i) Message, (ii) Update, and (iii) Readout. The Message                    arbitrary routing schemes (which are inherently represented
function takes as input node/edge embeddings, and it outputs                within the architecture). In order to address the circular
an information vector (the message) to be sent to all the                   dependencies, RouteNet repeats the same message passing
neighbors in the graph. The Update function collects (sums)                 operations over the links’ and paths’ state vectors T times
messages from all the neighbors in the graph and updates                    (loop from line 3). These steps represent the convergence
the embeddings of the nodes/edges. This message exchange                    process to the fixed point of a function from the initial states
is repeated T times, and finally the Readout function takes                 h0p and h0l .
the resulting nodes/edges embeddings to produce the output                     Regarding the issue of routing invariance (more generically
of the GNN model. We built upon this concept to construct                   known as topology invariance in the context of graph-related
RouteNet – a message passing architecture specifically tai-                 problems), RouteNet requires the use of a structure able to
lored to produce accurate performance estimates in computer                 represent graphs of different topologies of variable size. In
networks.                                                                   our case, we aim at representing different routing schemes
                                                                            in a uniform way. One state-of-the-art solution for this prob-
C. Message Passing Architecture Of RouteNet                                 lem [22] proposes using neural message passing architectures
  RouteNet handles variable-size input network topologies                   that combine both: a representation of the topology as a graph,
and arbitrary source-destination routing schemes, and produces              and vectors to encode the link states. In this context, RouteNet
4                                                    IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY



    Input: xp , xl ,R                                              dimension arrays (i.e., hidden states). Note that the size of
    Output: hTp , hTl , ŷp                                        the hidden states of links and paths are configurable hyper-
                           0
 1 foreach p ∈ R do hp ← [xp , 0 . . . , 0];                       parameters. In the end, all the hidden states in RouteNet
                           0
 2 foreach l ∈ N do hl ← [xl , 0 . . . , 0];                       represent an explicit function containing information of the
 3 for t = 0 to T − 1 do                                           link and path states. This enables to leverage them to infer
 4      foreach p ∈ R do                                           various features at the same time. Given a set of hidden states
 5          foreach l ∈ p do                                       hTp and hTl , it is possible to connect readout neural networks
 6              htp ← RN Nt (htp , htl )                           to estimate some path and/or link-level metrics. This can be
 7              m̃t+1
                   p,l ← hp
                             t
                                                                   typically achieved by using ordinary fully-connected neural
 8          end                                                    networks with some layers and proper activation functions. In
 9          ht+1
              p   ← htp                                            Algorithm 1, the function Fp (line 15) represents a readout
10      end                                                        function that predicts some path-level features (ŷp ) using as
11      foreach l ∈ N   do                                       input the path hidden states hp . Similarly, it would be possible
12
              t+1
            hl ← Ut htl , p:k∈p m̃t+1
                               P                                   to infer some global properties and link-level features (ŷl )
                                         p,k
                                                                   using also the information in the link hidden states hl .
13      end
14 end
                                                                   D. Delay, Jitter, and Drops Models
15 ŷp ← Fp (hp )
  Algorithm 1: Internal architecture of RouteNet. Complex-            Delay and jitter models are the first published applications
  ity in number of nodes: ∼ O(n2 log(n)), < O(n3 ) [13].           of RouteNet showing the capability of this neural architecture
                                                                   to model various network performance metrics. In [13] these
                                                                   were modeled independently using two neural network models
can be interpreted as an extension of a vanilla message passing    trained to minimize the mean squared error. Although this
neural network that is specifically suited to represent the        approach gives accurate results, it doubles the training time
dependencies among links and paths given a routing scheme          and model parametrization. Also, it hides the fact that average
(Equations (1) and (2)).                                           delay and jitter are two statistics of the same random process
                                                                   – the per-packet delay. In this paper, we propose a generalized
   In Algorithm 1, the loop from line 3 to line 14 represents
                                                                   probabilistic delay model that can be extended to account for
the message-passing operations that exchange mutually the
                                                                   packet drops.
information encoded (hidden states) among links and paths.
                                                                      Formally, the per-path (i-th path) delay and jitter are defined
Likewise, lines 9 and 12 are update functions that encode the
                                                                   as EWi and D2 Wi respectively. From the simulation, we
new collected information into the hidden states respectively
                                                                   obtain sample mean w̄i and variance s2 (wi ) being their
for paths and links. The update of paths’ states (line 9) is a
                                                                   estimates. Instead of modeling w̄i and s2 (wi ) independently,
simple assignment, while the update of links (line 12) is a
                                                                   let us approximate the whole distribution of Wi (marginal
trainable neural network. In general, the path update could be
                                                                   distribution given the input features) by a probability distri-
also a trainable neural network.
                                                                   bution parameterized by our RouteNet neural network output
   This architecture provides flexibility to represent any         ŷi being a two-element vector representing delay and jitter.
source-destination routing scheme. This is achieved by the         Direct generalization of previous models is:
direct mapping of R (i.e., the set of end-to-end paths) to
specific message passing operations among link and path
entities that define the architecture of RouteNet. Thus, each       Wi ∼ N orm(µi , σi ),     µi = ŷi0 , σi = sof tplus(ŷi1 ). (3)
path collects messages from all the links included in it (loop     Such a model can be trained by maximizing the log-likelihood
from line 5) and, similarly, each link receives messages from      function of the normal distribution (the loss function is its
all the paths containing it (line 12). Given that the order        negative):
of paths traversing the same link does not matter, we used
                                                                                         s (wi ) (w̄i − µi )2
                                                                                         2                             
a simple summation for the path-level message aggregation.            `(µi , σi ) = −ni         +             + log(σi ) , (4)
However, in the case of links, the presence of packet loss may                            2σi2       2σi2
imply sequential dependence in the links that form every path.     where ni is the total number of received packets. Note that this
Consequently, we use a Recurrent Neural Network (RNN)              loss function is just a scaled squared delay error plus additional
to aggregate link states on paths. Note that RNNs are well         terms representing jitter error and it is used in heteroscedastic
suited to capture dependence in sequences of variable size         regression. Such a simple form of the loss function (negative
(e.g., text processing). This allows us to model the sequential    log likelihood) is possible because the sample mean and
dependence of links and propagate this information through         variance are the sufficient statistics of the normal distribution
all the paths.                                                     and mean-field approximation is used. The per-packet and per-
   Moreover, the use of these message aggregation functions        path delays are assumed to be independent and identically
(RNN and summation) enables to significantly limit the di-         distributed (iid) random variables, and the dependence between
mensionality of the problem. The purpose of these functions        paths comes from the expected values only. The term mean-
is to collect an arbitrary number of messages received in every    field is used because of the similarity of such a model to a
(link or path) entity, and compress this information into fixed-   mean-field posterior in variational inference.
RUSEK AND SUÁREZ-VARELA et al.                                                                                                         5



   In the case of using different distributions to model the          us to give analytical results for packet loss probability as
delay (e.g., Gamma distribution), we would need to collect            well as to handle overloaded links. In the baseline model we
different statistics in the training dataset (e.g., log(wi ) for      made the following assumptions: 1) arrival to each queue is
the Gamma distribution) – namely the sufficient statistics of         approximated by the Poisson process, 2) packet lengths are
the distribution. In the case of Gamma distribution the loss          approximated by an exponential distribution, and 3) queues
function would be given by:                                           are assumed to be independent. Under those assumptions, we
                                                                      can derive analytical results for queue load, delay distribution,
  `(αi , βi ) = ni log Γ(αi ) + ni βi wi + ni (1 − αi )log(wi )−      and blocking probability.
                                               − ni αi log(βi ),         Let λk,i be the amount of traffic from path k passing trough
                                                                      link i. For each path we have:
where α and β would be the RouteNet outputs.
   See more details on the derivation of the loss functions for                 
                                                                                  λk,i = 0 if li 6∈ pk
the normal and gamma distributions in Appendix A.                               
                                                                                
                                                                                   λk,k(1) = Ak
                                                                                
                                                                                
   Note that this approach is not limited to model only the                     
                                                                                
                                                                                                  Qj−1
delay. The model can be tuned to different performance char-                       λk,k(j) = Ak i=1 (1 − P bk(i) ), j > 1             (7)
                                                                                                    b
acteristics by changing the distribution. Exponential family                              (1−ρi )ρi i
                                                                                   P bi =       bi +1 ,
                                                                                
                                                                                
                                                                                         P 1−ρi
                                                                                
distributions are perfect tools for this. In particular, choosing a             
                                                                                
                                                                                           pk ∈R λk,i
                                                                                
                                                                                   ρi =
                                                                                
discrete distribution like Binomial (Poisson is another option)                               ci
allows us to model per-path packet loss:                                       P
                                                                      where        pk ∈R k,i is the total traffic on the i-th link, P b
                                                                                        λ
    Li ∼ Binomial(pi , ni + li ),      pi = sigmoid(ŷi ),     (5)    denotes the blocking probability, Ai is the demand on the i-th
                                                                      path, b is the buffer size, and c is the link capacity. The system
where pi is the packet loss ratio on path i (i.e., li /(ni + li ).    of equations (7) is derived from traffic balance on the lossy
The log-likelihood function in this case is given by:                 network and it is solved using the fixed-point method. In the
              `(pi ) = li log(pi ) + ni log(1 − pi ),          (6)    first iteration, we assume no packet loss to compute initial traf-
                                                                      fic intensities. Given the first approximation, we can compute
where li is the observed number of losses, which is a sufficient      the loss probability and update the intensities to account for
statistic for the Binomial distribution. Such a loss function is      the losses. After a few iterations, the algorithm converges to
also common in binary classification problems.                        a fixed point. After convergence, it is possible to compute the
   Apart from the introduction of generalized probabilistic           delay, jitter and drop probability for each link from standard
modeling to RouteNet, we did not make any other substantial           queuing theory [26]. Path statistics are computed assuming
modification with respect to the original implementation in           link independence. Notice how this approach is similar to
[13]. The most relevant design choices are: 1) the size of            our derivation of the neural architecture of RouteNet. In the
the hidden states for both paths (hp ) and links (hl ), 2) the        baseline, we use known relations from queuing theory while in
number of message passing iterations (T ), and 3) the neural          RouteNet those relations are approximated by a neural network
network architectures for RN N , U , and F p. In line with the        and learned from the data.
previous model, we continue using Gated Recurrent Units
(GRU) [23], for both U and RN N . The readout function
                                                                      V. E VALUATION OF THE ACCURACY OF THE GNN M ODEL
(F p) is a fully-connected neural network with two layers and
uses selu activation functions in order to achieve desirable             In this section, we evaluate the accuracy of RouteNet (Sec.
scaling properties [24]. Compared to the architecture in [13],        III) to estimate the per-source/destination mean delay/jitter
we added a residual connection from hp to the last hidden             and the number of packet drops in a wide variety of network
layer of the readout function (F p) to provide a direct path for      topologies, routing schemes and traffic intensities.
the gradient. This connection shortens the information path
from measurement to the model parameters in the message-              A. Simulation Setup
passing part of the network. This was inspired by the residual
                                                                        We built a ground truth for our GNN model with a custom-
connections used in ResNet [15].
                                                                      built packet-level simulator with queues using OMNeT++
   In the readout function, the hidden layers are interleaved
                                                                      v4.6 [14]. Each simulation, we compute the mean end-to-end
with two dropout layers. The dropout layers play two impor-
                                                                      delay and jitter, and the packets dropped for every source-
tant roles in the model. During training, they help to avoid
                                                                      destination pair along 16k time units. We model the traffic
overfitting, and during the inference, they can be used for
                                                                      exchanged by every src-dst pair with the following traffic
Bayesian posterior approximation [25], [13].
                                                                      matrix (T M):
                         IV. BASELINE                                                      U(0.1, 1) ∗ T I
                                                                       T M(Si , Dj ) =                        ∀ i, j ∈ nodes, i 6= j (8)
  To asses the accuracy of RouteNet for network perfor-                                       N −1
mance modeling, we compare it to a queuing theory baseline.              Where U(0.1, 1) is a uniform distribution in the range
Alternatively to the standard Jackson network model we                [0.1, 1], TI represents a tunable parameter of the overall traffic
developed a new approach where each link is modeled as                intensity in the simulation, and N is the number of nodes in the
a finite M/M/1/b system instead of M/M/1. This allows                 network topology. In each source-destination pair, inter-packet
6                                                          IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY



arrival times are modeled with an exponential distribution                                           TABLE I
whose mean is derived from the traffic defined in T M. Also,              S UMMARY OF THE EVALUATION RESULTS (MRE); RN – ROUTE N ET, QT –
                                                                                                 Q UEUING T HEORY
packet sizes follow a binomial distribution, where 50% of
the packets have a size of 300 bits and the rest of packets                               Delay           Jitter           Drops
contain 1700 bits. All the queues have a size of 32 packets.
                                                                                          RN      QT      RN       QT      RN      QT
We made simulations in 4 different topologies with variable
link capacity and traffic intensity. Link capacities range the                 Test set   0.022   0.124   0.061    0.615   0.180   0.713
                                                                               GBN*       0.025   0.126   0.078    0.720   0.154   0.556
following values: 10, 40 or 100 kbps.
                                                                          Geant2 and Germany50 networks, we evaluate the accuracy
B. Training and Evaluation
                                                                          over the same 112,000 samples used for testing during the
   We implemented both models (delay and loss) in Tensor-                 training process (test set). Additionally, we tested the accuracy
Flow. The source code and all the training/evaluation datasets            over a dataset simulated in the 17-node GBN topology [31]
used in this paper are publicly available at [27]. The current            with 87,000 samples. Note that the GBN network was never
implementation is heavily optimized in terms of performance.              included in the training. The model was only trained with
The training speed was improved by a factor of 10x compared               samples from the NSF (14 nodes), Geant2 (24 nodes), and
to the first version reported in [13]. This allowed us to train the       Germany50 (50 nodes) networks. The high accuracy on the
model on a considerably larger dataset consisting of samples              GBN network (17 nodes) reveals the ability of RouteNet to
from the 14-node NSF [28], 24-node Geant2 [29] and 50-node                well generalize even to new networks. In all cases, RouteNet
Germany50 [30] networks.                                                  outperforms the queuing theory baseline, what is rather re-
   In total all models (delay/jitter, and drops) were trained on          markable given the Poisson traffic distribution used in the
a collection of 260,000 samples. Despite this dataset contains            simulator. For this traffic, the queuing theory should be a
only samples from the three topologies mentioned above, it                quite accurate approximation, yet the machine learning model
includes over 200 different routing schemes and a wide variety            achieves better accuracy.
of traffic matrices with different traffic intensity. For testing            Statistics like MRE provide a good picture of the general
(during training), we use 112,000 samples.                                accuracy of the model. However, there are more elaborated
   In our experiments, we select a size of 32 for both the                methods that offer a more detailed description of the model
path’s hidden states (hp ) and the link’s hidden states (hl ).            behavior. Hence, we focus on the full distribution of residuals
The initial path features (xp ) are defined by the bandwidth              (i.e., the error of the model and the baseline). Particularly, we
that each source-destination path carries (extracted from the             present a CDF of the relative error (Fig. 2) over all the evalua-
traffic matrix T M) while the initial link features (xl ) are             tion samples. This allows us to provide a comprehensive view
the link capacities. Note that, for larger networks, it might be          of the whole evaluation in a single plot. In these results, we
necessary to use larger sizes for the hidden states. Moreover,            can observe that the prediction error in general is considerably
every forward propagation we execute T =8 iterations. The                 low. Moreover, we see that the drops model is more biased
dropout rate is equal to 0.5. This means that each training               compared to the delay model. Note that the plot contains only
step we randomly deactivate half of neurons in the readout                the cases where we observed one or more packets dropped,
neural network. This also allows us to make a probabilistic               otherwise we would face division by zero when computing
sampling of results and infer the confidence of the estimates.            the relative error. However, in our simulation datasets there are
   During the training we minimized the loss function of                  many cases with zero drops, and these cases are also accurately
each model (negative log likelihood of the target distribution)           predicted by the RouteNet model. To test this one can compute
between the predictions of RouteNet and the ground truth                  the correlation coefficient between predictions and true loss
plus the L2 regularization loss (weight decay 0.1). The loss              ratio (including cases without loss where the value is 0). In
was summed over all source/destination pairs. We introduce                all experiments it is ≥ 0.997 for both the test and evaluation
a minibatch of samples by using a single disconnected graph               sets, and also for the baseline. This number is high because
composed by the individual connected graphs in the batch.                 cases with zero drops were always correctly labeled with a
The total loss function is minimized using an Adam optimizer              small probability.
with an initial learning rate of 0.001.                                      The generalization to unknown routings and traffic ma-
   We executed the training over 260,000 batches of 16 sam-               trices in known topologies (NSF, Geant2 and Germany50)
ples randomly selected from the training set. In our testbed              is almost perfect. The model is also equally accurate for
with a GPU Nvidia GeForce GTX 1080, this took around 20                   the unknown topology (GBN). This reveals the possibility to
hours (≈70 samples per second)1 .                                         deploy RouteNet models in different network scenarios where
   Table I shows a summary of the delay, jitter, and loss                 they were not trained. Also, there is the possibility to fine-tune
experiments we made in 4 different network topologies. We                 only the readout part with samples of the new networks and
report the Mean Relative Error (MRE) for both the RouteNet                reuse the computationally intensive message-passing part.
(RN) and the queuing theory baseline (QT). For the NSF,
   1 Note that this time consuming operation is required only once. The   C. Generalization Capabilities
inference is many orders of magnitude faster. On the same hardware the
single inference for a network with 200 nodes and 39,800 paths takes         This section discusses the generalization capabilities and
99.2 ms ± 561 µs, while on CPU (i5-6400) it takes 1.26 s ± 9.57 ms.       limitations of RouteNet. As in all ML-based solutions,
RUSEK AND SUÁREZ-VARELA et al.                                                                                                                                      7




                   1.0                                                                              1.0
                                 Test delay
                   0.8           Test jitter                                                        0.8
                                                                                                                                                 Test delay


       P( y < ε)                                                                        P( y < ε)
                                 Test drops                                                                                                      Test jitter
                   0.6                                                                              0.6
                                 GBN delay                                                                                                       Test drops
     y − ŷ        0.4           GBN jitter                                           y − ŷ        0.4
                                                                                                                                                 GBN delay
                                 GBN drops                                                                                                       GBN jitter
                   0.2                                                                              0.2
                                                                                                                                                 GBN drops
                   0.0                                                                              0.0

                         −1.00 −0.75 −0.50 −0.25 0.00   0.25   0.50   0.75   1.00                         −1.00 −0.75 −0.50 −0.25 0.00   0.25   0.50   0.75   1.00
                                                  ε                                                                                ε

                                        (a) RouteNet                                                                     (b) Baseline
Fig. 2. Cumulative Distribution Function (CDF) of the relative error. Solid line for delay, dashed line for jitter, dotted line for packet loss ratio (only in the
cases with observed drops). y is the true value, while ŷ denotes the model prediction.

RouteNet is expected to provide more accurate inference as                          models trained on the NSF, Geant2 and Germany50 datasets
the distribution of the input data is closer to the distribution of                 (Sec. V-B).
training samples. In our case, it involves topologies with sim-
ilar number of nodes and distribution of connectivity, routing                      A. Delay, Jitter, and Loss-aware Routing Optimization
schemes with similar patterns (e.g., variations of shortest path),
and similar ranges of traffic intensities. We experimentally                           This use case represents a QoS-aware routing optimization
observe the capability of RouteNet to generalize to topologies                      scenario where the target policy is to make a joint optimization
of variable size (from 14 to 50 nodes) while still providing                        of multiple KPI. Particularly, we leverage the KPI predictions
accurate estimates. In order to expand the generalization                           of RouteNet to minimize the per-source/destination mean
capabilities of RouteNet, an extended training set must be used                     delay and guarantee at the same time that jitter and packet
including a wider range of distributions of the input elements.                     loss are below certain thresholds. We define the following
   RouteNet’s architecture is built to estimate path-level met-                     optimization objectives in decreasing order of priority:
rics using information from the output path-level hidden states.                      1) Maintain the mean packet loss   rate below 0.1%.
However, it is relatively easy to modify the architecture and                              i.e., mean(Li /ni ) < 10−3
use information encoded in the link hidden states to produce                          2) Maintain the mean per-source/destination jitter below
link-related metrics inference (e.g., congestion probability on                           20% of the mean delay
links).                                                                                   [i.e., mean(jitter/delay) < 0.2]
                                                                                      3) Minimize the mean per-source/destination delay in the
                         VI. U SE C ASES                                                  network
   This section shows two different use cases where we                                 We implemented a RouteNet-based optimizer that utilizes
leverage the predictions of RouteNet (Sec. III) to address                          the performance predictions made by RouteNet (i.e., mean
relevant network optimization tasks from the control plane.                         delay, jitter, and loss). After evaluating a given set of candidate
In these use cases we use the delay, jitter and drops models of                     routing schemes, this optimizer selects the one that better
RouteNet to evaluate the resulting performance after applying                       fulfills the optimization objectives according to the RouteNet
some modifications in the network configuration. Particularly,                      predictions. In particular, it selects the routing scheme that
we limit the optimization problem to : (i) generate a set of                        results in lower per-source/destination mean delay among
candidate configurations (e.g., routing schemes), (ii) evaluate                     those configurations that fulfill the loss and jitter constraints
the resulting performance for each of them, and (iii) select                        (1 and 2). In the case that no routing scheme satisfies the
the one that best fits the optimization objective. We compare                       packet loss restriction (1), the optimizer selects the routing that
the performance achieved by our optimizer based on RouteNet                         minimizes the mean packet loss regardless of the other metrics.
to the results obtained by classic optimizers based on link uti-                    Likewise, if there is no routing that satisfies the jitter restriction
lization, the widely deployed Shortest Path routing policy, and                     (2), then the optimizer selects the routing scheme with lower
the optimal solution using an accurate packet-level simulator.                      mean delay among those that still satisfy the loss constraint
   In this context, state-of-the-art models predicting Key Per-                     (1). The set of candidate routing configurations comprises 450
formance Indicators (KPI) such as delay, jitter or drops are not                    variants of the shortest path policy. To this end, we consider
suited to perform online network optimization at large scale,                       an initial scenario where all the links of the NSF topology
since they often result into inaccurate estimation (e.g., analytic                  have a weight equal to 1 and we run the Dijkstra algorithm to
models) and/or prohibitive processing cost (e.g., packet-level                      compute the shortest path configuration. Then, the remaining
simulators). All the evaluations in this section are performed                      449 routing variations are generated by adding 0.05 to the
in network scenarios of the NSF network topology [28]. For                          weight of 21 links randomly selected. Note that in this random
the RouteNet-based optimizer we use the delay/jitter and drops                      selection process a link can be chosen more than once.
 8                                                                                              IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY




                                                                                                                                                                    0.55
                                                                                                Loss requirement                                                             Maximum Jitter/Delay level
                      Shortest Path
             1.2                                                                                Shortest Path                                                                Shortest Path
                      Utilization                                                      10−2     Utilization                                                         0.50
                                                                                                                                                                             Utilization
                      RouteNet                                                                  RouteNet
                                                                                                                                                                             RouteNet
                                                                                                Optimal                                                             0.45
                      Optimal                                                                                                                                                Optimal
             1.0
                                                                                       10−3
                                                                                                                                                                    0.40




Mean delay                                                               Packet loss                                                                 Jitter/Delay
             0.8                                                                       10−4                                                                         0.35

                                                                                                                                                                    0.30
                                                                                       10−5
             0.6
                                                                                                                                                                    0.25

                                                                                       10−6                                                                         0.20
             0.4
                                                                                                                                                                    0.15
                                                                                       10−7

                   11.0      12.0      13.0        14.0    15.0   16.0                        11.0        12.0     13.0       14.0     15.0   16.0                         11.0       12.0         13.0      14.0     15.0   16.0
                                       Traffic Intensity                                                           Traffic Intensity                                                              Traffic Intensity

                                    (a) Mean delay                                                             (b) Packet loss                                                    (c) Mean jitter/mean delay
 Fig. 3. Evaluation of the delay,jitter, and loss-aware routing optimization use case.



    We compare the results obtained by our RouteNet-based                                                                  values (packet loss = 10−7 ∀ packet loss ≤ 10−7 ).
 optimizer with two traditional routing approaches: (i) Shortest                                                              Looking into Figures 3b and 3c, the most remarkable result
 Path routing (hereafter SP), and (ii) a more elaborated routing                                                           is that the optimizer based on RouteNet’s predictions was able
 optimizer based on link utilization. For the SP routing baseline,                                                         to maintain the loss and jitter constraints even in most of
 we compute the resulting performance after applying the 450                                                               the scenarios with medium-high traffic load (TI=14). Only
 routing candidate solutions, which – as mentioned before –                                                                in some cases with the highest traffic intensity (TI=15-16)
 are all variants of the shortest path. Thus, the evaluation                                                               it did not find any routing scheme meeting the loss re-
 results for this baseline represent the average performance                                                               quirement. In contrast, the traditional SP policy and the
 achieved over all the SP routing variants (i.e., avg. delay, jitter                                                       utilization-based optimizer start to exceed the loss threshold
 and loss over the 450 configurations). The utilization-based                                                              from TI=13 (medium load). Moreover, in Figure 3a we observe
 optimizer selects the routing configuration that results in a                                                             that the RouteNet-based optimizer clearly outperforms these
 more balanced link utilization (i.e., less variance over all the                                                          other optimizers also in terms of per-source/destination mean
 links’ utilization). Similarly to state-of-the-art utilization-based                                                      delay. Particularly, as the network scenarios become more
 routing strategies, in this case we consider a fluid model of                                                             challenging (i.e., higher traffic intensity) the difference in
 the network without considering loss to compute the resulting                                                             performance is more remarkable. Note that in the cases where
 link utilization. Note that alternative utilization-based criteria                                                        the loss requirement could not be fulfilled, the RouteNet-based
 could also be applied, such as minimizing the utilization of the                                                          optimizer selected the routing scheme that resulted in less
 most loaded link. Moreover, we compute the optimal solution                                                               packet loss regardless of the mean delay and jitter predictions.
 with an optimizer that relies on the accurate performance                                                                 However, in these cases the configuration with lower loss
 estimates produced by our packet-level simulator (Sec. V-A).                                                              resulted also in lower average delay and jitter compared to
 In other words, this latter optimizer evaluates all the possible                                                          traditional routing techniques.
 routing configurations with our packet-level simulator and                                                                   Additionally, we evaluated the performance achieved by
 selects the one that best fits the optimization targets. For a                                                            the optimizer that uses directly the delay, jitter, and loss
 fair comparison, all the optimizers consider the same set with                                                            metrics computed by our packet-level network simulator
 450 routing schemes.                                                                                                      (Sec. V-A). These results are labeled as “optimal” in Fig-
    We evaluate the performance achieved by all the routing                                                                ures 3a, 3b, and 3c. As we can observe, the resulting
 strategies in scenarios with variable traffic intensity (from                                                             performance using the accurate predictions of RouteNet is
 low to high load). Particularly, we consider 6 different traffic                                                          practically the same as when we use the network simulator
 intensity levels.                                                                                                         metrics. This illustrates the potential of RouteNet to be used
                                                                                                                           for network optimization offering similar performance than
    Figure 3 summarizes the average per-source/destination                                                                 computationally intensive optimizers based on packet-level
 mean delay (Fig. 3a), average packet loss (Fig. 3b), and                                                                  simulation.
 average jitter/delay ratios (Fig. 3c) obtained by the different
 optimizers with respect to the traffic intensity (x-axis). Note
 that each boxplot represents the results over 100 scenarios                                                               B. Budget-constrained Network Upgrade
 with different input traffic matrices of the same traffic intensity                                                          This use case addresses a well-known optimization problem
 (TI). To this end, we generated 100 traffic matrices (T M) for                                                            in networking: how to optimally upgrade the network by
 each TI (from 11 to 16) according to Equation (8). For all the                                                            adding new links in the topology.
 optimization strategies, we provide the resulting performance                                                                For this use case, we selected 8 different network scenar-
 metrics computed by our packet-level simulator after applying                                                             ios from the previous use case where the RouteNet-based
 the best routing configuration selected in each case. To display                                                          optimizer could not meet the loss requirement (0.1%) given
 packet loss (Fig. 3b), we use a logarithmic scale (y-axis). Since                                                         the high traffic load. In particular, these scenarios include
 there are some cases without any loss (mostly at lower traffic                                                            traffic matrices (TM) of the highest load (TI=16). For each
 intensities), we defined a lower limit to avoid minus infinity                                                            scenario, the RouteNet-based optimizer selects the optimal link
RUSEK AND SUÁREZ-VARELA et al.                                                                                                                                                      9



                                                                            TABLE II
                                                  E VALUATION RESULTS OF THE O PTIMAL LINK PLACEMENT USE CASE

                                                                                                                                                     Relative reduction
 Traffic    Original scenario                    RouteNet-based optimizer                                   Baseline
                                                                                                                                                RouteNet-based vs Baseline
 matrix
           Delay   Jitter/delay   Opt. link placement     New delay   New jitter/delay   Most loaded link   New delay   New jitter/delay   Rel. Delay (%)    Rel. jitter/delay (%)

 T M1      0.964      0.345             (11-1)              0.715           0.222              (3-8)          1.377          0.352             48.09%               36.77%
 T M2      0.909      0.335              (9-2)              0.498           0.223             (12-5)          1.065          0.310             53.22%               28.12%
 T M3      0.949      0.374             (11-2)              0.752           0.236              (3-8)          1.312          0.320             42.66%               26.10%
 T M4      1.130      0.339             (12-2)              0.684           0.205             (5-12)          1.387          0.293             50.70%               29.95%
 T M5      0.967      0.314             (10-0)              0.630           0.214             (5-12)          1.321          0.311             52.33%               31.08%
 T M6      1.007      0.362             (11-1)              0.764           0.206             (12-5)          1.192          0.298             35.90%               30.97%
 T M7      1.055      0.379              (9-2)              0.743           0.223             (12-5)          1.087          0.345             31.67%               35.41%
 T M8      0.955      0.360             (11-1)              0.749           0.208              (3-8)          1.281          0.310             41.52%               32.90%




placement in combination with the routing scheme that results                               approach to network optimization constructs an objective
in lower per-source/destination mean delay. Particularly, we                                function based on the linearization of well known queuing
limit the problem to add only one link of 10 kbps, which is the                             theory results [32]. Network calculus is used for the worst-
minimum link capacity considered in the NSF topology. Then,                                 case scenario in networks, so it cannot be directly compared
the optimizer evaluates all the possible link placements in the                             to RouteNet as those worst cases are rarely observed in oper-
NSF network. Moreover, we consider 450 different routing                                    ational environments. Although fluid models are efficient and
schemes for each new possible link placement. The routing                                   popular for congestion control they are by design approximate
configurations are generated using the same method as in the                                and may lead to inaccurate results due to hiding the inherent
previous use case, i.e., they are shortest path variants. We                                properties of the traffic [33].
compare these results with the use of a traditional strategy that                              Given that deep learning models can learn queuing theory
network operators often apply when they detect degradation                                  with high accuracy [22], we may want to leverage these
in network performance. Particularly, this strategy selects the                             models for network optimization tasks. The main advantage
most loaded link (i.e., with highest utilization) in the current                            of this kind of models is that they always benefit from new
scenario and replaces it with another link with more capacity.                              data. Every edge case can be used to improve the model with
In our evaluation setup, the NSF network originally contains                                minimal investment.
links of 10 and 40 kbps. Then, if the most loaded link has a                                   Network modeling with deep neural networks is a recent
capacity of 10 kbps, it is replaced by a 40 kbps link. Likewise,                            topic proposed in the literature [5], [1] with few pioneering
40 kbps links are substituted by 100 kbps links.                                            attempts. The closest works to our contribution are first
    Table II shows the optimal link placement in the NSF                                    Deep-Q [8], where the authors infer the QoS of a network
network topology under the 8 TMs of high traffic intensity                                  using the traffic matrix as an input using Deep Generative
(TI=16). For each TM, we show the average delay and the                                     Models. And second [9], where a fully-connected feed-forward
jitter/delay ratio before and after adding the optimal link with                            neural network is used to model the mean delay of a set of
the best routing configuration. Also, we include the results ap-                            networks using as input the traffic matrix. The main goal of
plying the approach that updates the link capacity of the most                              the authors is to understand how fundamental network char-
loaded link (labeled as “Baseline”). All these results show the                             acteristics (such as traffic intensity) relate with basic neural
performance metrics computed by our packet-level simulator.                                 network parameters (depth of the neural network). RouteNet
To this end, we simulate the new scenarios considering the                                  is also able to produce accurate estimates of performance
optimal link placement and routing scheme selected by the                                   metrics (delay, jitter and loss), but it does not assume a fixed
RouteNet-based optimizer and the link upgrade selected by                                   topology and/or routing, rather it is able to produce such
the baseline. Here, we can observe that the optimizer using                                 estimates with arbitrary topologies and routing schemes not
RouteNet achieves an important reduction on the mean delay.                                 seen during training. This enables RouteNet to be used for
In particular, it achieves on average ≈44.5% more reduction                                 network operation, optimization, and what-if analysis.
in delay than the baseline. Note that the optimization target is                               Finally, an early attempt to use Graph Neural Networks
only based on minimizing the mean delay. However, we also                                   for computer networks can be found in [34]. In this case the
provide the results for jitter. As we expected, there is also an                            authors use a GNN to learn shortest-path routing and max-min
important reduction on this metric. We can observe that the                                 routing using supervised learning. While this approach is
jitter/delay ratio is reduced ≈31.4% on average with respect                                able to generalize to different topologies it cannot generalize
to the baseline.                                                                            to different routing schemes beyond the ones for which it
                                                                                            has been specifically trained. In addition, the focus of the
                           VII. R ELATED W ORK                                              paper is not to estimate the performance of such routing
   The ultimate goal of network modeling can be summarized                                  schemes. Another paper of the same author is focused on
as to provide a cost function for optimization. Many attempts                               performance [12], however, the model is a standard Graph
have been done over the years to derive the perfect solution.                               Neural Network, which loses information about the order of
Those include discrete-state Markov models (queuing theory),                                queues on paths. In contrast, RouteNet is designed to explicitly
stochastic fluid models and network calculus. Among them,                                   use this information. The importance of order can be found in
queuing theory is the most popular. The most advanced                                       the derivation of the baseline in section IV.
10                                                               IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY



                     VIII. C ONCLUSIONS                                            [6] B. Mao, Z. M. Fadlullah, F. Tang, N. Kato, O. Akashi, T. Inoue, and
                                                                                       K. Mizutani, “Routing or computing? the paradigm shift towards intel-
   Software-Defined Networks offer an unprecedented degree                             ligent computer network packet transmission based on deep learning,”
of flexibility in network control and management that, com-                            IEEE Transactions on Computers, vol. 66, no. 11, pp. 1946–1960, 2017.
bined with timely network measurements collected from the                          [7] A. Valadarsky, M. Schapira, D. Shahaf, and A. Tamar, “Learning to
                                                                                       route,” in Proceedings of HotNets, 2017.
data plane, open the possibility to achieve efficient online                       [8] S. Xiao, D. He, and Z. Gong, “Deep-q: Traffic-driven qos inference
network optimization.                                                                  using deep generative network,” in Proceedings of the ACM SIGCOMM
   However, existing network modeling techniques based on                              workshop on Network Meets AI & ML, 2018, pp. 67–73.
analytic models (e.g., Queuing theory) cannot handle this                          [9] A. Mestres, E. Alarcón, Y. Ji, and A. Cabellos-Aparicio, “Understanding
                                                                                       the modeling of computer network delays using neural networks,” in
huge complexity. As a result, current optimization approaches                          Proceedings of the ACM SIGCOMM workshop on Big Data Analytics
are limited to improve a global performance metric, such as                            and Machine Learning for Data Communication Networks, 2018, pp.
network utilization or planning the network based on the worst                         46–52.
                                                                                  [10] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini,
case estimates of the latencies obtained from network calculus.                        “The graph neural network model,” IEEE Transactions on Neural
   In this context, Deep Learning is a promising solution                              Networks, vol. 20, no. 1, pp. 61–80, 2009.
to handle such complexity and to exploit the full potential                       [11] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zam-
                                                                                       baldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner
of the SDN paradigm. However, earlier attempts to apply                                et al., “Relational inductive biases, deep learning, and graph networks,”
Deep Learning to networking problems resulted in tailor-made                           arXiv preprint arXiv:1806.01261, 2018.
solutions that failed to generalize to other network scenarios.                   [12] F. Geyer, “DeepComNet: Performance evaluation of network topologies
   In this paper, we presented RouteNet, a custom architec-                            using graph-based deep learning,” Performance Evaluation, vol. 130, pp.
                                                                                       1–16, apr 2019.
ture based on Graph Neural Network (GNN) specifically de-                         [13] K. Rusek, J. Suárez-Varela, A. Mestres, P. Barlet-Ros, and A. Cabellos-
signed for computer network modeling. RouteNet uses a novel                            Aparicio, “Unveiling the potential of graph neural networks for network
message-passing function that allows the GNN to capture the                            modeling and optimization in SDN,” in Proceedings of the ACM
                                                                                       Symposium on SDN Research (SOSR), 2019, pp. 140–151.
complex relationships between the state of paths and links                        [14] A. Varga, “The omnet++ discrete event simulation system,” in Proceed-
resulting from network topologies and routing configurations                           ings of the European Simulation Multiconference (ESM), 2001.
in order to model the resulting network performance.                              [15] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
                                                                                       recognition,” in Proceedings of the IEEE conference on computer vision
   We designed and implemented an extended RouteNet model                              and pattern recognition, 2016, pp. 770–778.
based on Generalized Linear Models that predicts the dis-                         [16] N. McKeown, T. Anderson, H. Balakrishnan, G. Parulkar, L. Peterson,
tribution of the per-source/destination per-packet delay and                           J. Rexford, S. Shenker, and J. Turner, “OpenFlow: Enabling Innova-
loss in networks. From these output distributions we evaluate                          tion in Campus Networks,” ACM SIGCOMM Comput. Commun. Rev.,
                                                                                       vol. 38, no. 2, p. 69, 2008.
the accuracy of the mean per-packet delay, the jitter, and the                    [17] M. Yu, L. Jose, and R. Miao, “Software defined traffic measurement
mean packet loss predicted. Our evaluation results show that                           with opensketch,” USENIX Symposium on Networked Systems Design
RouteNet is able to generalize to other network topologies,                            and Implementation, (NSDI), vol. 13, pp. 29–42, 2013.
                                                                                  [18] C. Kim, A. Sivaraman, N. Katta, A. Bas, A. Dixit, and L. J. Wobker,
routing configurations and traffic matrices not seen in the                            “In-band network telemetry via programmable dataplanes,” in ACM
training.                                                                              SIGCOMM, demo session, 2015.
   Also, the modular architecture of RouteNet simplifies trans-                   [19] “In-band OAM (iOAM),” https://github.com/CiscoDevNet/iOAM, Ac-
fer learning, which consists of reusing neural network models                          cessed: 2018-08-11.
                                                                                  [20] “NeMo: an application’s interface to intent-based networks,”
trained for a particular task and retrain them to address other                        http://nemo-project.net/, Accessed: 2018-08-11.
problems in similar domains. In the context of RouteNet this                      [21] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl,
was proposed in [13], where the jitter model was bootstrapped                          “Neural message passing for quantum chemistry,” in Proceedings of
                                                                                       the International Conference on Machine Learning (ICML), Volume 70,
from an early stage of the delay model.                                                2017, p. 1263–1272.
   Lastly, we presented some optimization use cases where                         [22] K. Rusek and P. Chołda, “Message-passing neural networks learn little’s
we use the performance predictions of RouteNet for different                           law,” IEEE Communications Letters, vol. 23, no. 2, pp. 274–277, 2019.
network optimization purposes. In particular, we perform                          [23] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical Evaluation
                                                                                       of Gated Recurrent Neural Networks on Sequence Modeling,” in Pro-
QoS-aware routing optimization based on delay, jitter and                              ceedings of NIPS, 2014.
packet loss requirements, and also use RouteNet to find the                       [24] G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter, “Self-
optimal link placement in a network planning scenario.                                 Normalizing Neural Networks,” in Proceedings of NIPS, 2017.
                                                                                  [25] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation:
                                                                                       Representing model uncertainty in deep learning,” in Proceedings of
                              R EFERENCES                                              the International Conference on Machine Learning (ICML), 2016, pp.
 [1] A. Mestres, A. Rodriguez-Natal, J. Carner, P. Barlet-Ros, and E. Alarcón,        1050–1059.
     et al., “Knowledge-defined networking,” ACM SIGCOMM Comput.                  [26] F. P. Kelly, Reversibility and Stochastic Networks. Cambridge Univer-
     Commun. Rev., vol. 47, no. 3, pp. 2–10, Sep. 2017.                                sity Press, 2011.
 [2] F. Ciucu and J. Schmitt, “Perspectives on network calculus: no free          [27] “Knowledge-defined         networking       repository,”    https://github.
     lunch, but still good value,” ACM SIGCOMM Comput. Commun. Rev.,                   com/knowledgedefinednetworking/Papers/wiki/RouteNet:
     vol. 42, no. 4, pp. 311–322, 2012.                                                -Leveraging-GNN-for-network-modeling-and-optimization-in-SDN,
 [3] Z. Xu, J. Tang, J. Meng, W. Zhang, Y. Wang, C. H. Liu, and D. Yang,               2019.
     “Experience-driven networking: A deep reinforcement learning based           [28] X. Hei, J. Zhang, B. Bensaou, and C.-C. Cheung, “Wavelength converter
     approach,” arXiv preprint arXiv:1801.05757, 2018.                                 placement in least-load-routing-based optical networks using genetic
 [4] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” Nature, vol. 521,            algorithms,” Journal of Optical Networking, vol. 3, no. 5, pp. 363–378,
     no. 7553, p. 436, 2015.                                                           2004.
 [5] M. Wang, Y. Cui, X. Wang, S. Xiao, and J. Jiang, “Machine learning           [29] F. Barreto, E. C. Wille, and L. Nacamura Jr, “Fast emergency paths
     for networking: Workflow, advances and opportunities,” IEEE Network,              schema to overcome transient link failures in ospf routing,” arXiv
     vol. 32, no. 2, pp. 92–99, 2018.                                                  preprint arXiv:1204.2465, 2012.
RUSEK AND SUÁREZ-VARELA et al.                                                                                                                                11



[30] S. Orlowski, R. Wessäly, M. Pióro, and A. Tomaszewski, “SNDlib                                       Paul Almasan received his B.Sc. and M.Sc. in
     1.0—survivable network design library,” Networks: An International                                     Computer Science from the Universitat Politècnica
     Journal, vol. 55, no. 3, pp. 276–286, 2010.                                                            de Catalunya (UPC), Spain, in 2017 and 2019 re-
[31] J. Pedro, J. Santos, and J. Pires, “Performance evaluation of integrated                               spectively. He is currently pursuing his Ph.D. degree
     OTN/DWDM networks with single-stage multiplexing of optical channel                                    at the Barcelona Neural Networking Center (BNN-
     data units,” in Proceedings of ICTON, 2011, pp. 1–4.                                                   UPC). His research interests are focused on Graph
[32] M. Pióro and D. Medhi, Routing, flow, and capacity design in commu-                                   Neural Networks and Deep Reinforcement Learning
     nication and computer networks. Elsevier, 2004.                                                        applied to network optimization.
[33] D. Y. Eun, “On the limitation of fluid-based approach for internet
     congestion control,” Telecommunication Systems, vol. 34, pp. 3–11,
     2007.
[34] F. Geyer and G. Carle, “Learning and generating distributed routing
     protocols using graph-based deep learning,” in Proceedings of the ACM
     SIGCOMM workshop on Big Data Analytics and Machine Learning for
     Data Communication Networks, 2018, pp. 40–45.

                                                                                                              Pere Barlet-Ros is an associate professor at Univer-
                                                                                                              sitat Politècnica de Catalunya (UPC) and scientific
                                                                                                              director at the Barcelona Neural Networking Center
                          Krzysztof Rusek is an assistant professor at AGH                                    (BNN-UPC). From 2013 to 2018, he was co-founder
                          and data scientist at the Barcelona Neural Net-                                     and chairman of the machine learning startup Talaia
                          working Center. He defended his Ph.D. Thesis on                                     Networks. The company was acquired by Auvik
                          queuing theory in 2016 at AGH. Prior to that he                                     Networks in 2018. He was also a visiting researcher
                          has worked as a system administrator and machine                                    at Endace (New Zealand), Intel Research Cambridge
                          learning engineer in the research group focused on                                  (UK) and Intel Labs Berkeley (USA). His research
                          processing and protection of multimedia content. His                                interests are in machine learning technologies for
                          main research interests are performance evaluation                                  network management and optimization, traffic clas-
                          of telecommunications systems, machine learning          sification and network security. In 2014, he received the 2nd VALORTEC prize
                          and data mining. Currently, he is working on the         for the best business plan awarded by the Catalan Government (ACCIO) and in
                          applications of Graph Neural Networks and proba-         2015 the Fiber Entrepreneurs award as the best entrepreneur of the Barcelona
bilistic modeling for performance evaluation of communications systems and         School of Informatics (FIB).
data mining in Astronomy.




                          José Suárez-Varela received his B.Sc. and M.Sc.
                                                                                                            Albert Cabellos-Aparicio is an assistant profes-
                          degrees in Telecommunication engineering from the
                                                                                                            sor at Universitat Politècnica de Catalunya (UPC),
                          Universidad de Granada (UGR), in 2014 and 2017
                                                                                                            where he obtained his PhD in computer science
                          respectively. He is currently a Ph.D. candidate at
                                                                                                            engineering in 2008. He is director of the Barcelona
                          the Barcelona Neural Networking Center (BNN-
                                                                                                            Neural Networking Center (BNN-UPC) and sci-
                          UPC). During 2019, he was a visiting researcher at
                                                                                                            entific director of the NaNoNetworking Center in
                          the University of Siena. His main research interests
                                                                                                            Catalunya. He has been a visiting researcher at Cisco
                          are in the field of Artificial Intelligence applied to
                                                                                                            Systems and Agilent Technologies, and a visiting
                          networking, particularly on the application of Graph
                                                                                                            professor at the KTH, Sweden, and the MIT, USA.
                          Neural Networks for network modeling and opti-
                                                                                                            His research interests include the application of
                          mization. He is also interested in traffic measurement
                                                                                                            Machine Learning to networking and nanocommu-
and classification, and their application in Software-Defined Networking.
                                                                                   nications. His research achievements have been awarded by the Catalan
                                                                                   Government, his university, and INTEL. He also participates regularly in
                                                                                   standardization bodies such as the IETF.
12                                                     IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, VOL. X, NO. X, MONTH, YYYY



                            A PPENDIX
                         L OSS FUNCTIONS
A. Normal
  The pdf of the normal distributions is:
                              1     1 x−µ 2
                  f (x) = √ e− 2 ( σ )
                            σ 2π
  The log-likelihood of a set of observations is:
                     Y 1          1 xi −µ 2
            L = log        √ e− 2 ( σ )
                      i
                         σ 2π
                                   (xi − µ)2
                 X                          
              =         − log(σ) −
                  i
                                      2σ 2
                                 1 X
              = − n log(σ) − 2          (xi − µ)2 .
                                2σ i
     Given the biased sample variance estimator:
                      1X
                 s2 =      (x − xi )2 = x2 − x2
                      n i
and the formula:
           X              X
              (xi − µ)2 =   (x2i − 2xi µ + µ2 )
                i               i
                                X
                            =       x2i − 2nµx + nµ2
                                i
,
     we can simplify the log-likelihood:

                          1
       L = −n log(σ) −        (ns2 + nx2 − 2nµx + nµ2 )
                         2σ 2
                                  s2    (x − µ)2
                                                
            L = −n log(σ) + 2 +
                                 2σ       2σ 2
   Since the log-likelihood is to be maximized, the loss func-
tion is −L and we get the equation (4):
                                 s2    (x − µ)2
                                               
             ` = n log(σ) + 2 +
                                2σ       2σ 2

B. Gamma
  In the case of gamma regression, we have the pdf of the
delay given by:
                               β α α−1 −βx
                      f (x) =      x    e
                              Γ(α)
                            Y βα
                    L = log          xα−1
                                      i   e−βxi
                            i
                               Γ(α)
.
           X
      L=        α log(β) + (α − 1) log(xi ) − βxi − log Γ(α)
            i

        L = n(α log(β) + (α − 1)log(x) − βx − log Γ(α)

                                                    
       ` = n log Γ(α) + βx + (1 − α)log(x) − α log(β)
  So for gamma regression we need average delay x and
average log delay log(x)

