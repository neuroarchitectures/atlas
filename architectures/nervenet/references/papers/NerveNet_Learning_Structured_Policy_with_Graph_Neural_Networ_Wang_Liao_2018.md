# NerveNet: Learning Structured Policy with Graph Neural Networks (Wang et al., ICLR 2018)

> Source: `https://openreview.net/forum?id=S1sqHMZCb`

---

Published as a conference paper at ICLR 2018

NERVENET: LEARNING STRUCTURED POLICY WITH
GRAPH NEURAL NETWORKS

Tingwu Wang∗, Renjie Liao∗, Jimmy Ba & Sanja Fidler
Department of Computer Science
University of Toronto
Vector Institute
tingwuwang,rjliao
{
jimmy@psi.toronto.edu, fidler@cs.toronto.edu

@cs.toronto.edu,
}

ABSTRACT

We address the problem of learning structured policies for continuous control. In
traditional reinforcement learning, policies of agents are learned by multi-layer
perceptrons (MLPs) which take the concatenation of all observations from the en-
vironment as input for predicting actions. In this work, we propose NerveNet to
explicitly model the structure of an agent, which naturally takes the form of a
graph. Speciﬁcally, serving as the agent’s policy network, NerveNet ﬁrst propa-
gates information over the structure of the agent and then predict actions for differ-
ent parts of the agent. In the experiments, we ﬁrst show that our NerveNet is com-
parable to state-of-the-art methods on standard MuJoCo environments. We further
propose our customized reinforcement learning environments for benchmarking
two types of structure transfer learning tasks, i.e., size and disability transfer, as
well as multi-task learning. We demonstrate that policies learned by NerveNet are
signiﬁcantly more transferable and generalizable than policies learned by other
models and are able to transfer even in a zero-shot setting.


INTRODUCTION

Deep reinforcement learning (RL) has received increasing attention over the past few years, with the
recent success of applications such as playing Atari Games, Mnih et al. (2015), and Go, (Silver et al.,
2016; 2017). Signiﬁcant advances have also been made in robotics using the latest RL techniques,
e.g., Levine et al. (2016); Gu et al. (2017).

Many RL problems feature agents with multiple dependent controllers. For example, humanoid
robots consist of multiple physically linked joints. Action to be taken by each joint or the body
should thus not only depend on its own observations but also on actions of other joints.

Previous approaches in RL typically use MLP to learn the agent’s policy. In particular, MLP takes
the concatenation of observations from the environment as input, which may be measurements like
positions, velocities of body and joints in the current time instance. The MLP policy then predicts
actions to be taken by every joint and body. Thus the task of the MLP policy is to discover the latent
relationships between observations. This typically leads to longer training times, requiring more
exposure of the agent to the environment. In our work, we aim to exploit the body structure of an
agent, and physical dependencies that naturally exist in such agents.

We rely on the fact that bodies of most robots and animals have a discrete graph structure. Nodes of
the graph may represent the joints, and edges represent the (physical) dependencies between them.
In particular, we deﬁne the agent’s policy using a Graph Neural Network, Scarselli et al. (2009),
which is a neural network that operates over graph structures. We refer to our model as NerveNet
due to the resemblance of the neural nervous system to a graph. NerveNet propagates information
between different parts of the body based on the underlying graph structure before outputting the
action for each part. By doing so, NerveNet can leverage the structure information encoded by the
agent’s body which is advantageous in learning the correct inductive bias, and thus is less prone to

∗Two authors contribute equally.


Published as a conference paper at ICLR 2018

Figure 1: Visualization of the graph structure of CentipedeEight in our environment. We use
this agent for testing the ability of transfer learning of our model. Since for this agent, each body
node is paired with at least one joint node, we omit the body nodes and ﬁll up the position
with the corresponding joint nodes. By omitting the body nodes, a more compact graph is
constructed, the details of which are illustrated in the experimental section.

overﬁtting. Moreover, NerveNet is naturally suitable for structure transferring tasks as most of the
model weights are shared across the nodes and edges, respectively.

We ﬁrst evaluate our NerveNet on standard RL benchmarks such as the OpenAI Gym, Brockman
et al. (2016) which stem from MuJoCo. We show that our model achieves comparable results to
state-of-the-art MLP based methods. To verify our claim regarding the structure transfer, we further
introduce our customized RL environments which are based on the ones of Gym. Two types of
structure transfer tasks are designed, size transfer and disability transfer. In particular, size transfer
focuses on the scenario in which policies are learned for small-sized agents (simpler body structure)
and applied directly to large-sized agents which are composed by some repetitive components shared
with the small-sized agent. Secondly, disability transfer investigates scenarios in which policies
are learned for one agent and applied to the same agent with some components disabled. Our
experiments demonstrate that for structure transfer tasks our NerveNet is signiﬁcantly better than
all other competitors, and can even achieve zero-shot learning for some agents. For the multi-task
learning tasks, NerveNet is also able to learn policies that are more robust with better efﬁciency.

The main contribution of this paper is the following: We explore the problem of learning transfer-
able and generalized features by incorporating a prior on the structure via graph neural networks.
NerveNet permits powerful transfer learning from one structure to another, which goes well beyond
the ability of previous models. NerveNet is also more robust and has more potential in perform-
ing multi-task learning. The demo and code for this project are released, under the project page of
http://www.cs.toronto.edu/˜tingwuwang/nervenet.html.

2 NERVENET

In this section, we ﬁrst introduce the notation. We then explain how to construct the graph for each of
our agents, followed by the description of the NerveNet. Finally, we describe the learning algorithm
for our model.

We formulate the locomotion control problems as an inﬁnite-horizon discounted Markov decision
process (MDP). To fully describe the MDP for continuous control problems which include locomo-
. To interact
tion control, we deﬁne the state space or observation space as
A
with the environments, the agent generates its stochastic policy πθ(aτ
sτ ) based on the current state
|
sτ
is the action and θ are the parameters of the policy function. The environ-
ment on the other hand, produces a reward r(sτ , aτ ) for the agent, and the agent’s objective is to
ﬁnd a policy that maximizes the expected reward.

and action space as

, where aτ

∈ A

∈ S

S


AnkleAnkleLeftHipRightHipTorsoAnkleAnkleLeftHipRightHipTorsoAnkleAnkleLeftHipRightHipTorsoAnkleAnkleLeftHipRightHipRoot
Published as a conference paper at ICLR 2018

Figure 2: In this ﬁgure, we use Walker-Ostrich as an example of NerveNet. In the input model,
for each node, NerveNet fetches the corresponding elements from the observation vector. NerveNet
then computes the messages between neighbors in the graph, and updates the hidden state of each
node. This process is repeated for a certain number of propagation steps. In the output model, the
policy is produced by collecting the output from each controller.

2.1 GRAPH CONSTRUCTION

In real life, skeletons of most robots and animals have a discrete graph structure, and are most often
trees. Simulators such as the MuJoCo engine by Todorov et al. (2012), organize the agents using an
XML-based kinematic tree. In our experiments, we will use the tree graphs as per MuJoCo. Note
that our model can be directly applied to arbitrary graphs. In particular, we assume two types of
nodes in our tree: body and joint. The body nodes are abstract nodes used to construct the
kinematic tree via nesting, which is similar to the coordinate frame system used in robotics, Spong
et al. (2006). The joint node represents the degrees of freedom of motion between the two body
nodes. Take a simple humanoid as an example; the body nodes Thigh and Shin are connected via
the Knee, where Knee is a hinge joint. We further add a root node which observes additional
information about the agent. For example, in the Reacher agent in MuJoCo, the root node has
access to the target position of the agent. We build edges to form a tree graph. Fig. 1 illustrates the
graph structure of an example agent, CentipedeEight. We sketch the agent and its correspond-
ing graph in the left and right part of the ﬁgure, respectively. Note that for better visualization, we
omit the joint nodes and use edges to represent the physical connections of joint nodes. Dif-
ferent elements of the agent are parsed into nodes with different colors. Further details are provided
in the experimental section.

2.2 NERVENET AS POLICY NETWORK

We now turn to NerveNet which parametrizes the policy with a Graph Neural Network. Before
delving into details, we ﬁrst introduce our notation. We then specify the input model which helps
to initialize the hidden state of each node. We further introduce the propagation model that updates
these hidden states. Finally, we describe the output model.

N

We denote the graph structure of the agent as G = (V, E) where V and E are the sets of nodes and
edges, respectively. We focus on the directed graphs as the undirected case can be easily addressed
by splitting one undirected edge into two directed edges. We denote the out-going neighborhood
out(u) which contains all endpoints v with (u, v) being an edge in the graph. Simi-
of node u as
larly, we denote the in-coming neighborhood of node u as
in(u). Every node u has an associated
, which in our case corresponds to body, joint and root. We also
node type pu
}
. Node type can help in cap-
associate each edge (u, v) with an edge type c(u,v) ∈ {
}
turing different importances across nodes. Edge type can be used to describe different relationships
between nodes, and thus propagate information between them differently. One can also add more
than one edge type to the same edge which results in a multi-graph. We stick to simple graphs for
simplicity. One interesting fact is that we have two notions of “time” in our model. One is the time
step in the environment which is the typical time coordinate for RL problems. The other corresponds

1, 2, . . . , C

1, 2, . . . , P

∈ {

N


ShinRootHipToeShinHipToeTailNeckHead ... ...Shin RootHipToeShinHipToeTailNeckHeadJoint’s VelocityJoint’s Angle ThetaAgent’s Positional Information ...ObservationVector ...InputModelPropagationModel...... ......OutputModelController’sOutput VectorStateUpdate
Published as a conference paper at ICLR 2018

to the internal propagation step of NerveNet. These two coordinates work as follows. At each time
step of the environment, NerveNet receives observation from the environment and performs a few
internal propagation steps in order to decide on the action to be taken by each node. To avoid con-
fusion, throughout this paper, we use τ to describe the time step in the environment and t for the
propagation step.

2.2.1

INPUT MODEL

For each time step τ in the environment, the agent receives an observation sτ
. The observation
vector sτ is the concatenation of observations of each node. We denote the elements of observation
vector sτ corresponding to node u with xu. From now on, we drop the time step in the environment
to derive the model for simplicity. The observation vector goes through an input network to obtain
a ﬁxed-size state vector as follows:

∈ S

h0
u = Fin(xu),
where the subscript and superscript denote the node index and propagation step, respectively. Here,
Fin may be a MLP and h0
u is the state vector of node u at propagation step 0. Note that we may need
to pad zeros to the observation vectors if different nodes have observations of different sizes.

(1)

2.2.2 PROPAGATION MODEL

We now describe the propagation model of our NerveNet which mimics a synchronous message
passing system studied in distributed computing, Attiya & Welch (2004). We will show how the
state vector of each node is updated from one propagation step to the next. This update process is
recurrently applied during the whole propagation. We leave the details to the appendix.

Message Computation In particular, at propagation step t, for every node u, we have access to a
state vector ht

out(u), node u computes a message vector as below,

u. For every edge (u, v)

(u,v) = Mc(u,v) (ht
where Mc(u,v) is the message function which may be an identity mapping or a MLP. Note that the
subscript c(u,v) indicates that edges of the same edge type share the same instance of the message
function. For example, the second torso in Fig. 1 sends a message to the ﬁrst and third torso, as
well as the LeftHip and RightHip.

u),

(2)

∈ N
mt

Message Aggregation Once every node ﬁnishes computing messages, we aggregate messages
sent from all in-coming neighbors of each node. Speciﬁcally, for every node u, we perform the
following aggregation:

ht
v|
{
where A is the aggregation function which may be a summation, average or max-pooling function.
Here, ¯mt
u is the aggregated message vector which contains the information sent from the node’s
neighborhood.

u = A(

in(u)

),
}

∈ N

¯mt

(3)

v

States Update We now update every node’s state vector based on both the aggregated message
and its current state vector. In particular, for every node u, we perform the following update:
u = Upu (ht
ht+1

u, ¯mt

u),

(4)

where U is the update function which may be a gated recurrent unit (GRU), a long short term
memory (LSTM) unit or a MLP. From the subscript pu of U , we can see that nodes of the same
node type share the same instance of the update function. The above propagation model is then
recurrently applied for a ﬁxed number of time steps T to get the ﬁnal state vectors of all nodes, i.e.,
hT
u
u |
{

.
}

∈

V

2.2.3 OUTPUT MODEL

In RL, agents typically use a MLP policy, where the network outputs the mean of the Gaussian
distribution for each of the actions, while the standard deviation is a trainable vector, Schulman
et al. (2017). In our output model, we also treat standard deviation in the same way.


Published as a conference paper at ICLR 2018

. For each such node, a MLP takes its ﬁnal state vectors hT

However, instead of predicting the action distribution of all nodes by a single network, we make
predictions for each individual node. We denote the set of nodes which are assigned controllers
for the actuators as
u∈O as input and
O
produces the mean of the action of the Gaussian policy for the corresponding actuator. For each
, we deﬁne its output type as qu. Different sharing schemes are available for
output node u
the instance of MLP Oqu, for example, we can force the nodes with similar physical structure to
share the instance of MLP. For example, in Fig. 1, two LeftHip nodes have a shared controller.
Therefore, we have the following output model:

∈ O

µu∈O = Oqu (hT
(5)
where µu∈O is the mean value for action applied on each actuator. In practice, we found that we
can force controllers of different output types to share one uniﬁed controller, while not hurting the
performance. By integrating the produced Gaussian policy for each action, the probability density
of the stochastic policy is calculated as

u ),

πθ(aτ

sτ ) =

|

(cid:89)

u∈O

πθ,u(aτ
u|

sτ ) =

(cid:89)

u∈O

2πσ2
u

(cid:112)

e(aτ

u−µu)2/(2σ2

u),

(6)

where aτ
θ represents the parameters of the policy function.

∈ A

is the output action, and σu is the variable standard deviation for each action. Here,

2.3 LEARNING ALGORITHM

sτ ) after several
To interact with the environments, the agent generates its stochastic policy πθ(aτ
|
propagation steps. The environment on the other hand, produces a reward r(sτ , aτ ) for the agent,
and transits to the next state with transition probability P (sτ +1
sτ ). The target of the agent is to
maximize its cumulative return

|

J(θ) = Eπ

(cid:34) ∞
(cid:88)

(cid:35)
γτ r(sτ , aτ )

τ =0

.

(7)

To optimize the expected reward, we use the proximal policy optimization (PPO) by Schulman
et al. (2017). In PPO, the agents alternate between sampling trajectories with the latest policy and
performing optimization on surrogate objective using the sampled trajectories. The algorithm tries to
keep the KL-divergence of the new policy and the old policy within the trust region. To achieve that,
PPO clips the probability ratio and adds an KL-divergence penalty term to the loss. The likelihood
ratio is deﬁned as rτ (θ; θold) = πθ(aτ
sτ ). Following the notation and the algorithm
|
of PPO, our NerveNet tries to minimize the summation of the original loss in Eq. (7), KL-penalty
and the value function loss which is deﬁned as:
βLKL(θ)

sτ )/πθold (aτ

˜J(θ) =J(θ)

αLV (θ)

|

−

(cid:16)

min

ˆAτ rτ (θ), ˆAτ clip (rτ (θ), 1

(cid:35)

(cid:17)

(cid:15), 1 + (cid:15))

−

−
(cid:34) ∞
(cid:88)

=Eπθ

τ =0

βEπθ

−

(cid:34) ∞
(cid:88)

τ =0

KL [πθ(:

sτ )
|

πθold(:
|

(cid:35)
sτ )]
|

−

αEπθ

(cid:34) ∞
(cid:88)

τ =0

(cid:0)Vθ(sτ )

−

V (sτ )target(cid:1)2

(cid:35)

,

(8)

where ˆAt is the generalized advantage estimation (GAE) calculated using algorithm from Schulman
et al. (2015b), and (cid:15) is the clip value, which we choose to be 0.2. Here, β is a dynamical coefﬁcient
adjusted to keep the KL-divergence constraints, and α is used to balance the value loss. Note that
in Eq. (8), V (st)target is the target state value in accordance with the GAE method. To optimize the
˜J(θ), PPO make use of the policy gradient in Sutton et al. (2000) to do ﬁrst-order gradient descent
optimization.

Value Network To produce the state value Vθ(sτ ) for given observation sτ , we have several al-
ternatives: (1) using one GNN as the policy network and using one MLP as the value network
(NerveNet-MLP); (2) using one GNN as policy network and using another GNN as value network
(NerveNet-2) (without sharing the parameters of the two GNNs); (3) using one GNN as both policy
network and value network (NerveNet-1). The GNN for value network is very similar to the GNN
for policy network. The output for value GNN is a scalar instead of a vector of mean action. We
will compare these variants in the experimental section.


Published as a conference paper at ICLR 2018

3 RELATED WORK

Reinforcement Learning Reinforcement learning (RL) has recently achieved huge success in a
variety of applications. Powered by the progress of deep neural networks, Krizhevsky et al. (2012),
agents are now able to successfully play Atari Games and beat the world’s best (human) players
in the game of Go (Mnih et al., 2015; Silver et al., 2016; 2017). Based on simulation engines like
MuJoCo, Todorov et al. (2012), numerous algorithms have been proposed to train agents also in
continuous control problems (Schulman et al., 2017; 2015a; Heess et al., 2017; Metz et al., 2017).

Structure in RL Most approaches that exploit priors on structure of the problem fall in the do-
main of hierarchical RL, (Kulkarni et al., 2016; Vezhnevets et al., 2017), which mainly focus on
modeling intrinsic motivation of agents. In Hausknecht & Stone (2015), the authors extend the deep
RL algorithms to MDPS with parameterized action space by exploiting the structure of action space
and bounding the action space gradients. Graphs have been used in RL problems prior to our work.
In Metzen (2013); Mabu et al. (2007); Shoeleh & Asadpour (2017); Mahadevan & Maggioni (2007),
the authors use graphs to learn a representation of the environment. However, these methods are lim-
ited to problems with simple dynamical models like for example the task of 2d-navigation, and thus
these problems are usually solved via model-based RL. However, for complex multi-joint agents,
learning the dynamical model as well as predicting the transition of states is time consuming and
biased. For problems of training model-free multi-joint agents in complex physical environments,
relatively little attention has been devoted to modeling the physical structure of the agents.

Graph Neural Networks There have been many efforts to generalize neural networks to graph-
structured data. One line of work is based on convolutional neural networks (CNNs). In (Bruna
et al., 2014; Defferrard et al., 2016; Kipf & Welling, 2017), CNNs are employed in the spectral
domain relying on the graph Laplacian matrix. (Goller & Kuchler, 1996; Duvenaud et al., 2015)
used hash functions in order to apply CNNs to graphs.

Another popular direction is based on recurrent neural networks (RNNs) (Goller & Kuchler, 1996;
Gori et al., 2005; Scarselli et al., 2009; Socher et al., 2011; Li et al., 2015; Tai et al., 2015).
Among RNN based methods, many are only applicable to special structured graph, e.g., sequences
or trees, (Socher et al., 2011; Tai et al., 2015). One class of models which are applicable to general
graphs are so-called graph neural networks (GNNs), Scarselli et al. (2009). The inference procedure
is a forward pass that exploits a ﬁxed-length propagation process which resembles synchronous mes-
sage passing system in the theory of distributed computing, Attiya & Welch (2004). Nodes in the
graph have state vectors which are recurrently updated based on their history and received messages.
One of the representative work of GNNs, i.e., gated graph neural networks (GGNNs) by Li et al.
(2015), uses gated recurrent unit to update the state vectors. Learning such a model can be achieved
by the back-propagation through time (BPTT) algorithm or recurrent back-propagation, Chauvin &
Rumelhart (1995). It has been shown that GNNs, (Li et al., 2015; 2017; Qi et al., 2017; Gilmer et al.,
2017) have a high capacity and achieve state-of-the-art performance in many applications which in-
volve graph-structured data. In this paper, we model the structure of the reinforcement learning
agents using GNNs.

Transfer and Multi-task Learning in RL Recently, there has been increased interest in transfer
learning tasks for RL, Taylor & Stone (2009), which mainly focus on transferring the policy learned
from one environment to another.
In Rajeswaran et al. (2017b;a), the authors show that agents
in reinforcement learning are prone to over-ﬁtting, and that the learned policies generalize poorly
across environments. In model-based RL, traditional control has been well studied for generalization
properties, Coros et al. (2010). Gupta et al. (2017) try to increase the transferability via learning
invariant visual features. Efforts have also been made from the meta-learning perspective (Duan
et al., 2016; Finn et al., 2017b;a). In Wulfmeier et al. (2017), the authors propose a method of transfer
learning by using imitation learning. Transferability comes naturally in our model by exploiting the
(shared) graph structure of the agents.

Multi-task learning has also received a lot of attention, Wilson et al. (2007). In Teh et al. (2017),
the authors use a distilled policy that captures common behaviour across tasks. Wilson et al. (2007);
Oh et al. (2017); Andreas et al. (2016) use a hierarchical approach, where multiple sub-policies are
learnt. In Yang et al. (2017); Parisotto et al. (2015), the authors exploited shared visual features


Published as a conference paper at ICLR 2018

Figure 3: Results of MLP, TreeNet and NerveNet on 8 continuous control benchmarks from the
Gym.

to tackle multi-task learning.
In Ammar et al. (2014), a multi-task policy gradient is proposed,
while, Calandriello et al. (2014) propose multi-task extensions of the ﬁtted Q-iteration algorithm.
Successor features in Barreto et al. (2017) are used to boost the performance of multiple tasks.
Unlike ours, these methods do not exploit the physical graph structures of the agents.

4 EXPERIMENTS

In this section, we ﬁrst verify the effectiveness of NerveNet on standard MuJoCo environments
in OpenAI Gym. We then investigate the transfer abilities of NerveNet and other competitors by
customizing some of those environments, as well as the multi-task learning ability and robustness.

4.1 COMPARISON ON STANDARD BENCHMARKS OF MUJOCO

Baselines We compare NerveNet with the standard MLP models utilized by Schulman et al. (2017)
and another baseline which is constructed as follows. We ﬁrst remove the physical graph structure
and introduce an additional super node which connects to all nodes in the graph. This results in a
singly rooted depth-1 tree. We refer to this baseline as TreeNet. The propagation model of TreeNet
is similar to NerveNet where, however, the policy ﬁrst aggregates the information from all children
and then feeds the state vector of the root to the output model. This simpler model serves as a
baseline to verify the importance of the graph structure.

We run experiments on 8 simulated continuous control benchmarks from the Gym, Brock-
man et al. (2016), which is based on MuJoCo, Todorov et al. (2012).
In particular, we use
Reacher, InvertedPendulum, InvertedDoublePendulum, Swimmer, and four walk-
ing or running tasks: HalfCheetah, Hopper, Walker2d, Ant. We set the maximum
number of training steps to be 1 million for all environments as it is enough to solve them.
Note that for InvertedPendulum, different from the original one in Gym, we add the dis-
tance penalty of the cart and velocity penalty so that the reward is more consistent to the
InvertedDoublePendulum. This change of design also makes the task more challenging.


02000004000006000008000001000000NumberofTimesteps−500−2500250500750100012501500AverageRewardAntModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps0100020003000AverageRewardHalfCheetahModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps05001000150020002500300035004000AverageRewardWalker2dModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps050010001500200025003000AverageRewardHopperModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps02000400060008000AverageRewardInvertedPendulumModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps02000400060008000AverageRewardInvertedDoublePendulumModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps−80−60−40−20AverageRewardReacherModelTypeMLPTreeNetNerveNet02000004000006000008000001000000NumberofTimesteps20406080100120AverageRewardSwimmerModelTypeMLPTreeNetNerveNet
Published as a conference paper at ICLR 2018

Table 1: Performance of the Pre-trained models on CentipedeFour and CentipedeSix.

C Four

Reward Avg

Std

Max

C Six

Reward Avg

Std

Max

NerveNet
MLP
TreeNet

2799.9
2398.5
1429.6

1247.2
1390.4
957.7

3833.9 NerveNet
3936.3 MLP
3021.7

TreeNet

2507.1
2793.0
2229.7

738.4
1005.2
1109.4

2979.2
3529.5
3727.4

Results We do grid search to ﬁnd the best hyperparameters and leave the details in the Ap-
pendix 6.3. As the randomness might have a big impact on the performance, for each environment,
we run 3 experiments with different random seeds and plot the average curves and the standard de-
viations. We show the results in Figure 3. From the ﬁgures, we can see that MLP with the same
setup as in Schulman et al. (2017) works the best in most of tasks.1 NerveNet basically matches the
performance of MLP in terms of sample efﬁciency as well as the performance after it converges.
In most cases, the TreeNet is worse than NerveNet which highlights the importance of keeping the
physical graph structure.

4.2 STRUCTURE TRANSFER LEARNING

We now benchmark our model in the task of structure transfer learning by creating customized
environments based on the existing ones from MuJoCo. We mainly investigate two types of structure
transfer learning tasks. The ﬁrst one is to train a model with an agent of small size (small graph) and
apply the learned model to an agent with a larger size, i.e., size transfer. When increasing the size
of the agent, observation and action space also increase which makes learning more challenging.
Another type of structure transfer learning is disability transfer where we ﬁrst learn a model for
the original agent and then apply it to the same agent with some components disabled.
If one
model overﬁts the environment, disabling some components of the agent might bring catastrophic
performance degradation. Note that for both transfer tasks, all factors of environments do not change
except the structure of the agent.

Centipede We create the ﬁrst environment in which the agent has a similar structure to a centipede.
The goal of the agent is to run as fast as possible along the y-direction in the MuJoCo environment.
The agent consists of repetitive torso bodies where each one has two legs attached. For two con-
secutive bodies, we add two actuators which control the rotation between them. Furthermore, each
leg consists of a thigh and shin, which are controlled by two hinge actuators. By linking copies
of torso bodies and corresponding legs, we create agents with different lengths. Speciﬁcally, the
shortest Centipede is CentipedeFour and the longest one is CentipedeFourty due to the
limit of supported resource of MuJoCo. For each time step, the total reward is the speed reward
minus the energy cost and force feedback from the ground. Note that in practice, we found that
training a CentipedeEight from scratch is already very difﬁcult. For size transfer experiments,
we create many instances which are listed in Figure 4, like “4to06”, “6to10”. For disability trans-
fer, we create CrippleCentipede agents of which two back legs are disabled. In Figure 4,
CrippleCentipede is speciﬁed as “Cp”.

Snakes We also create a snake-like agent which is common in robotics, Crespi & Ijspeert (2008).
We design the Snake environment based on the Swimmer model in Gym. The goal of the agent is
to move as fast as possible. For details of the environment, please see the schematic ﬁgure 16.

4.2.1 EXPERIMENTAL SETTINGS

To fully investigate the performance of NerveNet, we build several baseline models for structure
transfer learning which are explained below.

NerveNet For the NerveNet, since all the weights are exactly the same for the small and the large-
agent models, we directly use the old weights trained on the small-agent model. When the large

1By applying the adaptive learning rate schedule from Schulman et al. (2017), we obtained better perfor-

mances than the ones reported in original paper.


Published as a conference paper at ICLR 2018

(a) Zero-shot average reward.

(b) Zero-shot average running-length.

Figure 4: Performance of zero-shot learning on centipedes. For each task, we run the policy for 100
episodes and record the average reward and average length the agent runs before falling down.

(a)

(c)

(b)

(d)

Figure 5: (a), (b): Results of ﬁne-tuning for size transfer experiments. (c), (d) Results of ﬁne-tuning
for disability transfer experiments.

agent has repetitive structure, we further re-use the weights of the corresponding joints from the
small-agent model.

MLP Pre-trained (MLPP) For the MLP based model, while transferring from one structure to
another, the size of the input layer changes since the size of the observation changes. One straight-


1.Random2.MLPAA3.MLPP4.TreeNet5.NerveNetModels4to064to084to104to124to144toCp064toCp084toCp106to086to106to126to146to206to306to406toCp086toCp106toCp126toCp14Tasks-31.6(32%)109.4(84%)-126.7(-2%)-16.5(38%)139.6(96%)-45.8(41%)18.2(76%)-190.9(-39%)10.2(72%)44.3(91%)-44.3(42%)11.4(73%)-195.6(-42%)-101.5(10%)39.8(88%)-48.7(39%)9.9(72%)-215.8(-53%)17.6(76%)38.6(88%)-52.0(37%)8.0(71%)-233.0(-62%)-79.6(22%)39.0(88%)-17.0(26%)-5.1(29%)-113.9(1%)249.5(94%)47.6(42%)-25.5(52%)5.1(69%)-132.2(-6%)33.3(85%)40.0(88%)-28.2(51%)-12.8(59%)-133.7(-7%)28.2(82%)40.2(88%)-45.8(4%)21.1(7%)-191.4(-3%)320.8(24%)1674.9(99%)-44.3(7%)-42.4(7%)-193.2(-6%)44.7(15%)940.5(98%)-49.1(13%)-14.4(20%)-227.0(-20%)19.5(27%)367.7(95%)-52.0(17%)-10.0(28%)-221.3(-25%)126.3(63%)247.8(94%)-72.4(14%)10.0(39%)-351.4(-70%)-233.6(-34%)198.5(96%)-67.1(22%)-28.5(38%)-263.3(-59%)17.6(57%)114.9(97%)-72.7(19%)4.3(51%)-320.2(-83%)21.1(58%)97.8(90%)-25.4(14%)36.5(23%)-141.9(-3%)398.9(78%)523.6(97%)-28.2(14%)12.8(21%)-155.2(-5%)277.8(63%)504.0(99%)-32.0(22%)12.1(33%)-177.5(-14%)-114.9(1%)255.9(96%)-41.0(21%)11.8(36%)-187.5(-18%)-67.7(14%)224.8(95%)−0.250.000.250.500.751.Random2.MLPAA3.MLPP4.TreeNet5.NerveNetModels4to064to084to104to124to144toCp064toCp084toCp106to086to106to126to146to206to306to406toCp086toCp106toCp126toCp14Tasks-75.6(6%)545.3(92%)-73.5(6%)-47.7(10%)577.3(96%)-91.0(10%)62.0(67%)-80.8(14%)-84.8(13%)146.9(98%)-76.5(16%)21.0(52%)-89.9(11%)-70.0(18%)128.4(92%)-81.7(14%)13.1(49%)-76.9(15%)-63.2(21%)126.3(91%)-89.0(11%)0.0(44%)-86.4(12%)-73.2(17%)125.7(90%)-77.3(17%)-22.5(40%)-79.9(16%)-72.2(19%)91.1(87%)-82.9(17%)-26.9(44%)-91.8(13%)-66.9(25%)80.1(95%)-82.9(17%)-36.6(39%)-88.8(14%)-83.7(17%)86.9(98%)-91.0(0%)87.8(1%)-88.6(0%)8.5(1%)10612.6(99%)-76.5(0%)-17.0(1%)-81.8(0%)-77.4(0%)6343.6(99%)-81.2(1%)-1.4(4%)-79.3(1%)-91.5(1%)2532.2(99%)-89.0(1%)-3.9(6%)-83.2(1%)-51.5(3%)1749.7(98%)-95.2(1%)-4.9(7%)-105.4(0%)-102.3(1%)1447.4(98%)-111.2(0%)-48.2(7%)-127.3(0%)-76.7(4%)867.5(99%)63.9(16%)140.5(23%)56.9(15%)101.8(19%)971.5(98%)-83.0(1%)138.3(7%)-90.7(0%)-63.9(1%)3117.3(99%)-82.9(1%)13.6(3%)-86.4(0%)-72.3(1%)3230.3(99%)-87.5(1%)7.3(7%)-91.4(1%)-90.8(1%)1675.5(99%)-96.5(1%)2.9(7%)-92.5(1%)-93.9(1%)1517.6(99%)0.00.20.40.60.80500000100000015000002000000NumberofTimesteps0500100015002000250030003500AverageRewardCentipedeSixtoCentipedeEightModelTypeSolvedMLP+PretrainMLPAAMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain0500000100000015000002000000NumberofTimesteps050010001500200025003000AverageRewardCentipedeFourtoCentipedeEightModelTypeSolvedMLP+PretrainMLPAAMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain0500000100000015000002000000NumberofTimesteps050010001500200025003000AverageRewardCentipedeSixtoCpCentipedeEightModelTypeSolvedMLP+PretrainMLPAAMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain0500000100000015000002000000NumberofTimesteps0500100015002000250030003500AverageRewardCentipedeFourtoCpCentipedeSixModelTypeSolvedMLP+PretrainMLPAAMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain
Published as a conference paper at ICLR 2018

Figure 6: Results on zero-shot transfer learning on snake agents. Each tasks are simulated for 100
episodes.

(a)

(c)

(b)

(d)

Figure 7: Results of ﬁnetuning on snake environments.

forward idea is to reuse the weights from the ﬁrst hidden layer to the output layer and randomly
initialize the weights of the new input layer.

MLP Activation Assigning (MLPAA) Another way of making MLP transferable is assigning the
weights of the small-agent model to the corresponding partial weights of the large-agent model and
setting the remaining weights to be zero. Note that we do not add or remove any layers from the
small-agent model to the large-agent except for changing the size of the layers. By doing so, we can
keep the output of the large-agent model to be same as the small-agent in the beginning, i.e., keeping
the same initial policy.

TreeNet TreeNet is similar as the model described before. We apply the same way of assigning
weights as MLPAA to TreeNet for the transfer learning task.

Random We also include the random policy which is uniformly sampled from the action space.


1.Random2.MLPAA3.MLPP4.TreeNet5.NerveNetModelsSnakeFour2SnakeFiveSnakeFour2SnakeSevenSnakeFour2SnakeSixSnakeThree2SnakeFiveSnakeThree2SnakeFourSnakeThree2SnakeSevenSnakeThree2SnakeSixTasks20.59(12%)89.82(30%)-16.54(3%)-15.97(3%)342.88(95%)-2.28(7%)115.12(40%)-22.81(1%)-22.38(2%)313.15(95%)7.6(9%)105.63(34%)-20.45(2%)-34.34(-1%)351.85(97%)20.59(14%)51.41(22%)-16.48(3%)-21.26(2%)314.92(95%)3.86(9%)-30.19(0%)-12.36(4%)-21.82(2%)325.48(98%)-2.28(9%)23.53(17%)-22.38(2%)-42.74(-4%)256.29(95%)7.6(11%)32.2(18%)-19.67(3%)-42.46(-3%)282.43(94%)0.00.20.40.60.802000004000006000008000001000000NumberofTimesteps0100200300400AverageRewardSnakeThreeToSnakeFourModelTypeMLP+ActivationAssignMLP+PretrainMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain02000004000006000008000001000000NumberofTimesteps0100200300400AverageRewardSnakeThreeToSnakeFiveModelTypeMLP+ActivationAssignMLP+PretrainMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain02000004000006000008000001000000NumberofTimesteps0100200300400AverageRewardSnakeFourToSnakeFiveModelTypeMLP+ActivationAssignMLP+PretrainMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain02000004000006000008000001000000NumberofTimesteps−1000100200300400500AverageRewardSnakeFourToSnakeSixModelTypeMLP+ActivationAssignMLP+PretrainMLPTreeNet+PretrainTreeNetNerveNetNerveNet+Pretrain
Published as a conference paper at ICLR 2018

4.2.2 RESULTS

Centipedes For the Centipedes environment, we ﬁrst run experiments of all models on
CentipedeSix and CentipedeFour to get the pre-trained models for transfer learning. We
train different models until these agents run equally well as possible, which is reported in Table 1.
Note that, in practice, we train TreeNet on CentipedeFour for more than 8 million time steps.
However, due to the difﬁculty of optimizing TreeNet on CentipedeFour, the performance is still
lower. But visually, the TreeNet agent is able to run in CentipedeFour.

We then examine the zero-shot performance where zero-shot means directly applying the model
trained with one setting to the other without any ﬁne-tuning. To better visualize the results, we
linearly normalize the performance to get a performance score, and color the results accordingly.
The normalization scheme is recorded in Appendix 11. The performance score is less than 1, and
is shown in the parentheses behind the original results. As we can see from Figure 4 (full chart in
Appendix 6.5), NerveNet outperforms all competitors on all settings, except in the 4toCp06 scenario.
Note that transferring from CentipedeFour is more difﬁcult than from CentipedeSix since
the situation where one torso connects to two neighboring torsos only happens beyond 4 bodies.

TreeNet has a surprisingly good performance on tasks from CentipedeFour. However, by check-
ing the videos, the learned agent is actually not able to “move” as good as other methods. The high
reward is mainly due to the fact that TreeNet policy is better at standing still and gaining alive bonus.
We argue that the average running-length in each episode is also a very important metric.

By including the results of running-length, we notice that NerveNet is the only model able to walk in
the zero-shot setting. In fact, the performance of NerveNet is orders-of-magnitude better, and most
of the time, agents from other methods cannot even move forward. We also notice that if transferred
from CentipedeSix, NerveNet is able to provide walkable pre-trained models on all new agents.

We ﬁne-tune for both size transfer and disability transfer experiments and show the training curves
in Figure 5. From the ﬁgure, we can see that by using the pre-trained model, NerveNet signiﬁcantly
decreases the number of episodes required to reach the level of reward which is considered as solved.
By looking at the videos, we notice that the bottleneck of learning for the agent is “how to stand”.
When training from scratch, it can be seen that almost 0.5 million time steps are spent on a very ﬂat
reward surface. Therefore, the MLPAA agents, which copy the learned policy, are able to stand and
bypass this time-consuming process and reach to a good performance in the end.

Moreover, by examining the result videos, we noticed that the “walk-cycle” behavior is observed
for NerveNet but is not common for others. Walk-cycle are adopted for many insects in the
world, Biewener (2003). For example, six-legged ants use a tripedal gait, where the legs are used
in two separate triangles alternatively touching the ground. We give more details of walk-cycle in
Section 4.5.

One possible reason is that the agent of MLP based method (MLPAA, MLPP) learns a policy that
does not utilize all legs. From CentipedeEight and up, we do not observe any MLP agents to be
able to coordinate all legs whereas almost all policies learned by NerveNet use all legs. Therefore,
NerveNet is better at utilizing structure information and not over-ﬁtting the environments.

Snakes The zero-shot performance for snakes is summarized in Figure 6. As we can see, NerveNet
has the best performance on all transfer learning tasks.
In most cases, NerveNet has a starting
reward value of more than 300, which is a pretty good policy since 350 is considered as solved for
snakeThree. By looking at the videos, we found that agents of other competitors are not able to
control the new actuators in the zero-shot setting. They either overﬁt to the original models, where
the policy is completely useless in the new setting (e.g., the MLPAA is worse than random policy in
SnakeThree2SnakeFour), or the new actuators are not able to coordinate with the old actuators
trained before. While for NerveNet, the actuators are able to coordinate to its neighbors, regardless
of whether they are new to the agents.

We also summarize the training curves of ﬁne-tuning in Fig. 7. We can observe that NerveNet has a
very good initialization with the pre-trained model, and the performance increases with ﬁne-tuning.
When training from scratch, NerveNet is less sample efﬁcient compared to the MLP model which
might be caused by the fact that optimizing our model is more challenging than MLP. Fine-tuning
helps to improve the sample efﬁciency of our model by a large margin. At the same time, although


Published as a conference paper at ICLR 2018

Figure 8: Results of Multi-task learning. We train the networks simultaneously on ﬁve different
tasks from Walker.

Model

HalfHumanoid Hopper

Ostrich

Wolf

Horse

Average

MLP

TreeNet

NerveNet

Reward
Ratio

Reward
Ratio

Reward
Ratio

1775.75
57.7%

237.81
79.3%

2536.52
96.3%

1369.59
62.0%

1198.88
48.2%

1249.23
54.5%

2084.07
69.7%

417.27
98.0%

224.07
223.34
247.03
57.4% 141.2% 99.2%

/
58.6%

/
94.8%

2054.54
2113.56
101.8% 98.8% 105.9% 106.4% 101.8%

2343.62

1714.63

/

Table 2: Results of Multi-task learning, with comparison to the single-task baselines. For three
models, the ﬁrst row is the mean reward of each model of the last 40 iterations. The second row
indicates the percentage of the performance of the multi-task model compared with the single-task
baseline of each model.

the MLPAA has a very good initialization, its performance progresses slowly with the increasing
number of episodes. In most experiments, the MLPAA and TreeNet did not match the performance
of its non-pretrained MLP baseline.

4.3 MULTI-TASK LEARNING

In this section, we show that NerveNet has a good potential of multi-task learning by incorporating
structure prior into the network structure. It is important to point out that multi-task learning repre-
sents a very difﬁcult, and more often the case, unsolved problem in RL. Most multi-task learning
algorithms, Teh et al. (2017); Andreas et al. (2016); Oh et al. (2017); Yang et al. (2017) have not
been applied to domains as difﬁcult as locomotion for complex physical models, not to mention
multi-task learning among different agents with different dynamics.

In this work, we constrain our problem domain, and design the Walker multi-task learning task-
set, which contains ﬁve 2d-walkers. We aim to test the model’s ability of multi-task learning,
in particular, the ability to control multiple agents using one uniﬁed network. The walkers are
very different in terms of their dynamics, since they have very distinct structures, different types
and numbers of controllers. Walker-HalfHumanoid and Walker-Hopper are variants of
Walker2d and Hopper from the original MuJoCo Benchmarks, respectively. Walker-Horse,
Walker-Ostrich, Walker-Wolf on the other hand, are agents mimicking the natural animals.
Just like the real animals, some of the agents have tails or necks to help them to balance. The detailed
schematic ﬁgures are shown in the Appendix 6.8.


010000002000000300000040000005000000NumberofTimesteps050010001500200025003000AverageRewardWalker-HopperModelTypeSingle-TaskBaselineMLP+AggregationMLP+GrowingTreeNetNerveNet010000002000000300000040000005000000NumberofTimesteps050010001500200025003000AverageRewardWalker-HalfHumanoidModelTypeSingle-TaskBaselineMLP+AggregationMLP+GrowingTreeNetNerveNet010000002000000300000040000005000000NumberofTimesteps050010001500200025003000AverageRewardWalker-HorseModelTypeSingle-TaskBaselineMLP+AggregationMLP+GrowingTreeNetNerveNet010000002000000300000040000005000000NumberofTimesteps05001000150020002500AverageRewardWalker-OstrichModelTypeSingle-TaskBaselineMLP+AggregationMLP+GrowingTreeNetNerveNet010000002000000300000040000005000000NumberofTimesteps05001000150020002500AverageRewardWalker-WolfModelTypeSingle-TaskBaselineMLP+AggregationMLP+GrowingTreeNetNerveNet
Published as a conference paper at ICLR 2018

Model

Halfhumanoid Hopper Wolf

Ostrich

Horse

Average

Mass

Strength

MLP
NerveNet

MLP
NerveNet

33.28%
95.87%

25.96%
31.11%

74.04% 94.68% 59.23% 40.61% 60.37%
93.24% 90.13% 80.2% 69.23% 85.73%

21.77% 27.32% 30.08% 19.80% 24.99%
42.20% 42.84% 31.41% 36.54% 36.82%

Table 3: Results of robustness evaluations. Note that we show the average results for each type
of parameters after perturbation. And the results are columned by the agent type. The ratio of the
average performance of perturbed agents and the original performance is shown in the ﬁgure. Details
are listed in 6.6.

4.3.1 EXPERIMENTAL SETTINGS

To show the ability of multi-task learning of NerveNet, we design several baselines. We use a
vanilla multi-task policy update for all models. More speciﬁcally, for each sub-task in the multi-task
learning task-set, we use an equal number of time steps for each policy’s update and calculate the
gradients separately. Gradients are then aggregated and the mean value of gradients is applied to
update the network. To compensate for the additional difﬁculty in training more agents and tasks,
we linearly increase the number of update epochs during each update in training, as well as the total
number of time steps generated before the training is terminated. The hyper-parameter setting is
summarized in Appendix 6.7.

NerveNet For NerveNet, the weights are naturally shared among different agents. More speciﬁ-
cally, for different agents, the weight matrices for propagation and output are shared.

MLP Sharing For the MLP method, we shared the weight matrices between hidden layers.

MLP Aggregation In the MLP Sharing approach, the total size of the weight matrices grows with
the number of tasks. For different agents, whose dimension of the observations are usually different,
weights from observation to the ﬁrst hidden layer cannot be reused in the MLP Sharing approach.
Therefore in the MLP Aggregation method, we multiply each element of the observation vector
separately by one matrix, and aggregate the resulting vectors from each element. The size of this
multiplying matrix is (1, dimension of the ﬁrst hidden layer).

TreeNet Similarly, TreeNet also has the beneﬁts that its weights are naturally shared among dif-
ferent agents. However, TreeNet has no knowledge of the agents’ physical structure, where the
information of each node is aggregated into the root node.

We also include the baseline of training single-task MLP for each agent. We train the single-task
MLP baselines for 1 million time steps per agent. In the Figure 8, we align the results of single-task
MLP baseline and the results of multi-task models by the number of episodes of one task.

As can be seen from Figure 8, NerveNet achieves the best performance in all the sub-tasks.
In
Walker-HalfHumanoid, Walker-Hopper, Walker-Ostrich, Walker-Wolf our Ner-
veNet is able to out-perform other agents by a large margin. In Walker-Horse, the performance
of NerveNet and MLP Sharing are relatively similar. For MLP Sharing, the performance on other
four agents are relatively limited, while for Walker-Hopper, the improvement of performance is
limited from half of the experiment onwards. The MLP Aggregation and TreeNet methods are not
able to solve the multi-task learning problem, with both of them stuck at a very low reward level. In
the vanilla optimization setting, we show that NerveNet has a bigger potential than the baselines.

From Table 2, one can observe that the performance of MLP drops drastically (42% performance
drop) when switching from single-task to multi-task learning, while for NerveNet, there is no obvi-
ous drop in performance. Our intuition is that NerveNet is better at learning generalized features,
and learning of different agents can help in training other agents, while for MLP methods, the per-
formance decreases due the competition of different agents.


Published as a conference paper at ICLR 2018

Figure 9: Diagram of the walk cycle.
In the left ﬁgure, legs within the same triangle are used
simultaneously. For each leg, we use the same color for their diagram on the left and their curves on
the right.

Figure 10: Results of visualization of feature distribution and trajectory density. As can be seen from
the ﬁgure, NerveNet agent is able to learn shareable features for its legs, and certain walk-cycle is
learnt during training.

4.4 ROBUSTNESS OF LEARNT POLICIES

In this section, we also report the robustness of our policy by perturbing the agent parameters.
In reality, the parameters simulated might be different from the actual parameters of the agents.
Therefore, it is important that the agent is robust to parameters perturbation. The model that has the
better ability to learn generalized features are likely more robust.

We perturb the mass of the geometries (rigid bodies) in MuJoCo as well as the scale of the forces of
the joints. We use the pre-trained models with similar performance on the original task for both the
MLP and NerveNet. The performance is tested in ﬁve agents from Walker task set. The average
performance is recorded in Table 3, and the speciﬁc details are summarized in Appendix 6.6. The
robustness of NerveNet’ policy is likely due to the structure prior of the agent instilled in the network,
which facilitates overﬁtting.


0.02.55.07.510.012.515.017.520.0Timesteps−0.3−0.2−0.10.00.10.20.30.40.50.6FeatureNodeNameRighthipRighthipRighthipLefthipLefthipLefthip−202FeatureDimTwo−1.5−1.0−0.50.00.51.01.5FeatureDimOneJointTypeLeftHipRightHip−202FeatureDimTwo−1.5−1.0−0.50.00.51.01.5FeatureDimOneJointTypeLeftHipRightHip−202FeatureDimTwo−1.5−1.0−0.50.00.51.01.5FeatureDimOneJointTypeLeftHipRightHip−0.250.000.250.500.751.001.25FeatureDimOne−1.5−1.0−0.50.00.5FeatureDimTwopearsonr=0.59;p=0−1.0−0.50.00.5FeatureDimOne−0.50.00.51.01.52.0FeatureDimTwopearsonr=0.57;p=0
Published as a conference paper at ICLR 2018

Figure 11: Results of several variants of NerveNet for the reinforcement learning agents.

4.5

INTERPRETING THE LEARNED REPRESENTATIONS

In this section, we try to visualize and interpret the learned representations. We extract the ﬁnal
state vectors of nodes of NerveNet trained on CentipedeEight. We then apply 1-D and 2-D
PCA on the node representations. In Figure 10, we notice that each pair of legs is able to learn
invariant representations, despite their different position in the agent. We further plot the trajectory
density map in the feature map. By recording the period of the walk-cycle, we plot the transformed
features of the 6 legs on Figure 9. As we can see, there is a clear periodic behavior of our hidden
representations learned by our model. Furthermore, the representations of adjacent left legs and the
adjacent right legs demonstrate a phase shift, which further proves that our agents are able to learn
the walk-cycle without any additional supervision.

4.6 COMPARISON OF MODEL VARIANTS

We have several variants of NerveNet, based on the type of network we use for the policy/value
representation. We here compare all variants. Again, we run experiments for each task three times.
The details of hyper-parameters are given in the Appendix. For each environment, we train the
network for one million time steps, with batch size 2050 for one update.

As we can see from Figure 11, the NerveNet-MLP and NerveNet-2 variants perform better than
NerveNet-1. One potential reason is that sharing the weights of the value and policy networks
makes the trust-region based optimization methods, like PPO, more sensitive to the weight α of the
value function in equation 8. Based on the ﬁgure, choosing α to be 1 is not giving good performance
on the tasks we experimented on.

5 CONCLUSION

In this paper, we aimed to exploit the body structure of Reinforcement Learning agents in the form
of graphs. We introduced a novel model called NerveNet which uses a Graph Neural Network to
represent the agent’s policy. At each time instance of the environment, NerveNet takes observations
for each of the body joints, and propagates information between them using non-linear messages
computed with a neural network. Propagation is done through the edges which represent natural
dependencies between joints, such as physical connectivity. We experimentally showed that our
NerveNet achieves comparable performance to state-of-the-art methods on standard MuJoCo envi-
ronments. We further propose our customized reinforcement learning environments for benchmark-
ing two types of structure transfer learning tasks, i.e., size and disability transfer. We demonstrate
that policies learned by NerveNet are signiﬁcantly better than policies learned by other models and
are able to transfer even in a zero-shot setting.

REFERENCES

Haitham B Ammar, Eric Eaton, Paul Ruvolo, and Matthew Taylor. Online multi-task learning for
policy gradient methods. In Proceedings of the 31st International Conference on Machine Learn-
ing (ICML-14), pp. 1206–1214, 2014.

Jacob Andreas, Dan Klein, and Sergey Levine. Modular multitask reinforcement learning with

policy sketches. arXiv preprint arXiv:1611.01796, 2016.


02000004000006000008000001000000NumberofTimesteps20406080100120AverageRewardSwimmerModelTypeNerveNet-MLPNerveNet-2NerveNet-102000004000006000008000001000000NumberofTimesteps−50−40−30−20−10AverageRewardReacherModelTypeNerveNet-MLPNerveNet-2NerveNet-102000004000006000008000001000000NumberofTimesteps−5000500100015002000250030003500AverageRewardHalfCheetahModelTypeNerveNet-MLPNerveNet-2NerveNet-1
Published as a conference paper at ICLR 2018

Hagit Attiya and Jennifer Welch. Distributed computing: fundamentals, simulations, and advanced

topics, volume 19. John Wiley & Sons, 2004.

Andr´e Barreto, Will Dabney, R´emi Munos, Jonathan J Hunt, Tom Schaul, David Silver, and Hado P
van Hasselt. Successor features for transfer in reinforcement learning. In Advances in Neural
Information Processing Systems, pp. 4056–4066, 2017.

Andrew A Biewener. Animal locomotion. Oxford University Press, 2003.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and

Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally

connected networks on graphs. ICLR, 2014.

Daniele Calandriello, Alessandro Lazaric, and Marcello Restelli. Sparse multi-task reinforcement

learning. In Advances in Neural Information Processing Systems, pp. 819–827, 2014.

Yves Chauvin and David E Rumelhart. Backpropagation: theory, architectures, and applications.

Psychology Press, 1995.

Stelian Coros, Philippe Beaudoin, and Michiel Van de Panne. Generalized biped walking control.

ACM Transactions on Graphics (TOG), 29(4):130, 2010.

Alessandro Crespi and Auke Jan Ijspeert. Online optimization of swimming and crawling in an

amphibious snake robot. IEEE Transactions on Robotics, 24(1):75–87, 2008.

Micha¨el Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on

graphs with fast localized spectral ﬁltering. In NIPS, 2016.

Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl 2: Fast
reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.

David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Al´an
Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular
ﬁngerprints. In NIPS, 2015.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation

of deep networks. arXiv preprint arXiv:1703.03400, 2017a.

Chelsea Finn, Tianhe Yu, Tianhao Zhang, Pieter Abbeel, and Sergey Levine. One-shot visual imita-

tion learning via meta-learning. arXiv preprint arXiv:1709.04905, 2017b.

Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural

message passing for quantum chemistry. arXiv preprint arXiv:1704.01212, 2017.

Christoph Goller and Andreas Kuchler. Learning task-dependent distributed representations by
backpropagation through structure. In Neural Networks, 1996., IEEE International Conference
on, volume 1, pp. 347–352. IEEE, 1996.

Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains.

In IJCNN, 2005.

Shixiang Gu, Ethan Holly, Timothy Lillicrap, and Sergey Levine. Deep reinforcement learning for

robotic manipulation with asynchronous off-policy updates. In ICRA, 2017.

Abhishek Gupta, Coline Devin, YuXuan Liu, Pieter Abbeel, and Sergey Levine. Learning invariant
feature spaces to transfer skills with reinforcement learning. arXiv preprint arXiv:1703.02949,
2017.

Matthew Hausknecht and Peter Stone. Deep reinforcement learning in parameterized action space.

arXiv preprint arXiv:1511.04143, 2015.


Published as a conference paper at ICLR 2018

Nicolas Heess, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, Yuval Tassa, Tom Erez,
Ziyu Wang, Ali Eslami, Martin Riedmiller, et al. Emergence of locomotion behaviours in rich
environments. arXiv preprint arXiv:1707.02286, 2017.

Thomas N Kipf and Max Welling. Semi-supervised classiﬁcation with graph convolutional net-

works. ICLR, 2017.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classiﬁcation with deep convo-
lutional neural networks. In Advances in neural information processing systems, pp. 1097–1105,
2012.

Tejas D Kulkarni, Karthik Narasimhan, Ardavan Saeedi, and Josh Tenenbaum. Hierarchical deep
reinforcement learning: Integrating temporal abstraction and intrinsic motivation. In Advances in
Neural Information Processing Systems, pp. 3675–3683, 2016.

Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuo-
motor policies. J. Mach. Learn. Res., 17(1):1334–1373, January 2016. ISSN 1532-4435. URL
http://dl.acm.org/citation.cfm?id=2946645.2946684.

Ruiyu Li, Makarand Tapaswi, Renjie Liao, Jiaya Jia, Raquel Urtasun, and Sanja Fidler. Situation

recognition with graph neural networks. 2017.

Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural

networks. arXiv preprint arXiv:1511.05493, 2015.

Shingo Mabu, Kotaro Hirasawa, and Jinglu Hu. A graph-based evolutionary algorithm: Genetic
network programming (gnp) and its extension using reinforcement learning. Evolutionary Com-
putation, 15(3):369–398, 2007.

Sridhar Mahadevan and Mauro Maggioni. Proto-value functions: A laplacian framework for learn-
ing representation and control in markov decision processes. Journal of Machine Learning Re-
search, 8(Oct):2169–2231, 2007.

Luke Metz, Julian Ibarz, Navdeep Jaitly, and James Davidson. Discrete sequential prediction of

continuous actions for deep rl. arXiv preprint arXiv:1705.05035, 2017.

Jan Hendrik Metzen. Learning graph-based representations for continuous reinforcement learn-
ing domains. In Joint European Conference on Machine Learning and Knowledge Discovery in
Databases, pp. 81–96. Springer, 2013.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Belle-
mare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level
control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Junhyuk Oh, Satinder Singh, Honglak Lee, and Pushmeet Kohli. Zero-shot task generalization with

multi-task deep reinforcement learning. arXiv preprint arXiv:1706.05064, 2017.

Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and

transfer reinforcement learning. arXiv preprint arXiv:1511.06342, 2015.

Xiaojuan Qi, Renjie Liao, Jiaya Jia, Sanja Fidler, and Raquel Urtasun. 3d graph neural networks
for rgbd semantic segmentation. In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pp. 5199–5208, 2017.

Aravind Rajeswaran, Kendall Lowrey, Emanuel Todorov, and Sham Kakade. Towards generalization

and simplicity in continuous control. arXiv preprint arXiv:1703.02660, 2017a.

Aravind Rajeswaran, Kendall Lowrey, Emanuel Todorov, and Sham Kakade. Towards generalization

and simplicity in continuous control. arXiv preprint arXiv:1703.02660, 2017b.

Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini.

The graph neural network model. IEEE TNN, 2009.


Published as a conference paper at ICLR 2018

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region
policy optimization. In Proceedings of the 32nd International Conference on Machine Learning
(ICML-15), pp. 1889–1897, 2015a.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
arXiv preprint

dimensional continuous control using generalized advantage estimation.
arXiv:1506.02438, 2015b.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Farzaneh Shoeleh and Masoud Asadpour. Graph based skill acquisition and transfer learning for
continuous reinforcement learning domains. Pattern Recognition Letters, 87:104–116, 2017.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche,
Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering
the game of go with deep neural networks and tree search. Nature, 529(7587):484–489, 2016.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez,
Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go
without human knowledge. Nature, 550(7676):354–359, 2017.

Richard Socher, Cliff C Lin, Chris Manning, and Andrew Y Ng. Parsing natural scenes and natural
language with recursive neural networks. In Proceedings of the 28th international conference on
machine learning (ICML-11), pp. 129–136, 2011.

Mark W Spong, Seth Hutchinson, Mathukumalli Vidyasagar, et al. Robot modeling and control,

volume 3. Wiley New York, 2006.

Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient
methods for reinforcement learning with function approximation. In Advances in neural informa-
tion processing systems, pp. 1057–1063, 2000.

Kai Sheng Tai, Richard Socher, and Christopher D Manning. Improved semantic representations

from tree-structured long short-term memory networks. ACL, 2015.

Matthew E Taylor and Peter Stone. Transfer learning for reinforcement learning domains: A survey.

Journal of Machine Learning Research, 10(Jul):1633–1685, 2009.

Yee Teh, Victor Bapst, Razvan Pascanu, Nicolas Heess, John Quan, James Kirkpatrick, Wojciech M
Czarnecki, and Raia Hadsell. Distral: Robust multitask reinforcement learning. In Advances in
Neural Information Processing Systems, pp. 4497–4507, 2017.

Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control.
In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on, pp. 5026–
5033. IEEE, 2012.

Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David
Silver, and Koray Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. arXiv
preprint arXiv:1703.01161, 2017.

Aaron Wilson, Alan Fern, Soumya Ray, and Prasad Tadepalli. Multi-task reinforcement learning: a
hierarchical bayesian approach. In Proceedings of the 24th international conference on Machine
learning, pp. 1015–1022. ACM, 2007.

Markus Wulfmeier, Ingmar Posner, and Pieter Abbeel. Mutual alignment transfer learning. arXiv

preprint arXiv:1707.07907, 2017.

Zhaoyang Yang, Kathryn Merrick, Hussein Abbass, and Lianwen Jin. Multi-task deep reinforcement
In Proceedings of the Twenty-Sixth International Joint

learning for continuous action control.
Conference on Artiﬁcial Intelligence, IJCAI-17, pp. 3301–3307, 2017.


Published as a conference paper at ICLR 2018

6 APPENDIX

6.1 DETAILS OF NERVENET

We use MLP to compute the messages which uses tanh nonlinearities as the activation function.
We do a grid search on the size of the MLP to compute the messages, the details of which are listed
in Table 5, 4.

Throughout all of our experiments, we use average aggregation and GRU as the update function.

Table 4: Parameters used during training.

Parameters

Value Set

Parameters

Value Set

Value Discount Factor γ
PPO Clip Value
Gradient Clip Value

.99
0.2
5.0

GAE λ
Starting Learning Rate
Target KL

.95
3e-4
0.01

6.2 GRAPH OF AGENT

In MuJoCo, we observe that most body nodes are paired with one and only one joint node. Thus,
we simply merge the two paired nodes into one. We point out that this model is very compact, and
is the standard graph we use in our experiments.

In the Gym environments, observation for the joint nodes normally includes the angular veloc-
ity, twist angle and optionally the torque for the hinge joint, and position information for the
positional joint. For the body nodes, velocity, inertia, and force are common observations. For
example in the centipede environment 1, the LeftHip node will receive the angular velocity (cid:36)j,
and the twist angle θj.

6.3 HYPERPARAMETER SEARCH

For MLP, we run grid search with the hidden size from two layers to three layers, and with hidden
size from 32 to 256. For NerveNet, to reduce the time spent on grid search, we constrain the
propagation network and output network to be the same shape. Similarly, we run grid search with
the network’s hidden size, and at the same time, we run a grid search on the size of node’s hidden
states from 32 to 64. For the TreeNet, we run similar grid search on the node’s hidden states and
output network’s shape.

For details of hyperparameter search, please see Table 5, 7, 6.

Table 5: Hyperparameter grid search options for MLP.

MLP

Value Tried

Network Shape
Number of Iteration Per Update
Use KL Penalty
Learning Rate Scheduler

[64, 64], [128,128], [256, 256], [64,64,64]
10, 20
Yes, No
Linear Decay, Adaptive, Constant

Table 6: Hyperparameter grid search options for TreeNet.

TreeNet

Value Tried

Network Shape
Number of Iteration Per Update
Use KL Penalty
Learning Rate Scheduler

[64, 64], [128,128], [256, 256]
10, 20
Yes, No
Linear Decay, Adaptive, Constant


Published as a conference paper at ICLR 2018

Table 7: Hyperparameter grid search options for NerveNet.

NerveNet

Value Tried

Network Shape
Number of Iteration Per Update
Use KL Penalty
Learning Rate Scheduler
Number of Propogation Steps
NerveNet Variants
Size of Nodes’ Hidden State
Merge joint and body Node
Output Network
Disable Edge Type
Add Skip-connection from / to root

[64, 64], [128,128], [256, 256]
10, 20
Yes, No
Linear Decay, Adaptive, Constant
3, 4, 5, 6
NerveNet-1, NerveNet-2, NerveNet-MLP
32, 64, 128
Yes, No
Shared, Separate
Yes, No
Yes, No

6.4 SCHEMATIC FIGURES OF THE PARSED AGENTS

In this section, we also plot the schematic ﬁgures of the agents for readers’ reference. Graph struc-
tures are automatically parsed from the MuJoCo XML conﬁguration ﬁles.

Figure 12: Schematic diagrams and auto-parsed graph structures of the InvertedPendulum and
InvertedDoublePendulum.

Figure 13: Schematic diagrams and auto-parsed graph structures of the Walker2d and Reacher.


RootSliderHingeRootSliderHingeHingeInvertedPendulumInvertedDoublePendulumRootThighLegFootRootRootJointHingeWalker2DReacher
Published as a conference paper at ICLR 2018

Figure 14: Schematic diagrams and auto-parsed graph structures of the SnakeSix and Swimmer.

Figure 15: Schematic diagrams and auto-parsed graph structures of the Ant.

Figure 16:
HalfCheetah.

Schematic diagrams and auto-parsed graph structures of

the Hopper and


RootSegSegSegSegRootSegSegSegSegSnakeSixSwimmerRootHipAnkleHipAnkleHipAnkleHipAnkleAntRootThighLegFootFrontThighFrontShinFrontFootBackThighBackShinBackFootRootHopperHalfCheetah
Published as a conference paper at ICLR 2018

6.5 DETAILS OF ZERO-SHOT LEARNING RESULTS

e d e p i t n e C f o

g n i n r a e l

t o h s - o r e z

e
h
t

f
o

s t l u s e R

:

e l b a T

e g a r e v A d r a w e R

d
t

S d r a w e R

x a M d r a w e R

t e N e v r e N

t e N e e r T

P P L M

A A P L M

m o d n a R

t e N e v r e N

t e N e e r T

P P L M

A A P L M

m o d n a R

t e N e v r e N

t e N e e r T

P P L M

A A P L M

m o d n a R

k s a T

e g a r e v A e c n a t s i D

d t S e c n a t s i D

x
a
M

e c n a t s i D

t e N e v r e N

t e N e e r T

P P L M

A A P L M

m o d n a R

t e N e v r e N

t e N e e r T

P P L M

A A P L M

m o d n a R

t e N e v r e N

t e N e e r T

P P L M

A A P L M

m o d n a R

k s a T

.

.

.

.


.

.

.

.

.

.

.

.

.


.

.

.

.


.

.

.

.
-

.

.
-

.

.
-

.


.
-

.

.

.

.

.
-

.

.

.

.

.

.

.

.

.

.
-

.
-

.
-

.
-

.
-

.
-

-

.
-

.
-

-

.
-

.
-

-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

-

-

.
-

.
-

.
-

.
-

.

.

.

.


.

.
-

.
-

.

.
-

.
-

-


.
-

.

.
-

.

.
-

.

.

.

.

.

.

.
-

.
-

.
-

.
-

-

.
-

.
-

.
-

.
-

.
-

.
-

-

.
-

.
-

.
-

-

.
-

.
-

.
-

-

.
-

.
-

-

-

.

.

.

.

.

.


.

.


.


.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.


.

.

.

.

.

.

.

.

.


.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.

.

.


.

.

.

.

.

.

.

.

.

.

.

.

.


.

.

.


.

.

.

.

.

.

.

.

.

.

.


.

.

.


.

.

.

.

.

.

.

.

.


.

.

.

.

.
-

.
-

-

.
-

.
-

.
-

.
-

.
-

-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

-

.
-

.
-

.
-

.
-

.
-

.

.

.

.

.

.

.


.

.

.

.

.

.


.

.

.

.

.


.

.

.


.
-

.
-

.
-

.
-

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

.
-

.

.
-

.

.

.
-

.
-

.
-

.

.
-

.
-

.
-

n e e t r u o F e d e p i t n e C r u o F e d e p i t n e C

y t n e w T e d e p i t n e C r u o F e d e p i t n e C

e v l e w T e d e p i t n e C r u o F e d e p i t n e C

y t r i h T e d e p i t n e C r u o F e d e p i t n e C

y t r o F e d e p i t n e C r u o F e d e p i t n e C

t h g i E e d e p i t n e C x i S e d e p i t n e C

n e T e d e p i t n e C x i S e d e p i t n e C

n e e t r u o F e d e p i t n e C x i S e d e p i t n e C

y t n e w T e d e p i t n e C x i S e d e p i t n e C

e v l e w T e d e p i t n e C x i S e d e p i t n e C

y t r i h T e d e p i t n e C x i S e d e p i t n e C

y t r o F e d e p i t n e C x i S e d e p i t n e C

t h g i E e d e p i t n e C p C r u o F e d e p i t n e C

n e T e d e p i t n e C p C r u o F e d e p i t n e C

x i S e d e p i t n e C p C r u o F e d e p i t n e C

t h g i E e d e p i t n e C r u o F e d e p i t n e C

n e T e d e p i t n e C r u o F e d e p i t n e C

x i S e d e p i t n e C r u o F e d e p i t n e C

n e e t r u o F e d e p i t n e C p C r u o F e d e p i t n e C

e v l e w T e d e p i t n e C p C r u o F e d e p i t n e C

n e e t r u o F e d e p i t n e C p C x i S e d e p i t n e C

e v l e w T e d e p i t n e C p C x i S e d e p i t n e C

t h g i E e d e p i t n e C p C x i S e d e p i t n e C

n e T e d e p i t n e C p C x i S e d e p i t n e C


.

.

.

.

.

.


.

.

.

.

.

.

.

.

.

.

.


.

.

.

.

.

.
-

.
-

-

.
-

.
-

.
-

.
-

.

.

.
-

.
-

.
-

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.
-

.


.


.
-

.
-

.

.

-

.
-

.
-

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

.

.

.

.

.
-

-

.
-

.
-

-

.
-

.
-


-

.
-

.
-

-

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

-

.
-

.
-

.
-

.

.


.

.

.

.

.

.

.

.

.

.

.

.


.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.


.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.

.


.

.

.

.

.

.

.

.

.

.

.

.


.

.

.

.


.

.

.

.

.

.

.

.

.

.


.

.

.

.

.


.

.


.

.

.


.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.

.

.

.

.

.


.
-

.

.
-

.

.
-

.
-

.

.

.
-

.
-

.
-

.

.
-

.

.

.
-

.
-

.

.
-

.
-

.
-

.

.

.

.

.

.

.

.

.
-

.

.

.

.

.

.

.
-

.

.

.


.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.

.

.


.

.

.

.

.


.


.

.

.

.

.

.

.

.

.
-

.

.

.

.

.

.
-

.

.

.

.

.
-

.

.

.

.
-

.

.

n e e t r u o F e d e p i t n e C r u o F e d e p i t n e C

y t n e w T e d e p i t n e C r u o F e d e p i t n e C

e v l e w T e d e p i t n e C r u o F e d e p i t n e C

y t r i h T e d e p i t n e C r u o F e d e p i t n e C

y t r o F e d e p i t n e C r u o F e d e p i t n e C

t h g i E e d e p i t n e C x i S e d e p i t n e C

n e T e d e p i t n e C x i S e d e p i t n e C

n e e t r u o F e d e p i t n e C x i S e d e p i t n e C

y t n e w T e d e p i t n e C x i S e d e p i t n e C

e v l e w T e d e p i t n e C x i S e d e p i t n e C

y t r i h T e d e p i t n e C x i S e d e p i t n e C

y t r o F e d e p i t n e C x i S e d e p i t n e C

t h g i E e d e p i t n e C p C r u o F e d e p i t n e C

n e T e d e p i t n e C p C r u o F e d e p i t n e C

x i S e d e p i t n e C p C r u o F e d e p i t n e C

t h g i E e d e p i t n e C r u o F e d e p i t n e C

n e T e d e p i t n e C r u o F e d e p i t n e C

x i S e d e p i t n e C r u o F e d e p i t n e C

n e e t r u o F e d e p i t n e C p C r u o F e d e p i t n e C

e v l e w T e d e p i t n e C p C r u o F e d e p i t n e C

n e e t r u o F e d e p i t n e C p C x i S e d e p i t n e C

e v l e w T e d e p i t n e C p C x i S e d e p i t n e C

t h g i E e d e p i t n e C p C x i S e d e p i t n e C

n e T e d e p i t n e C p C x i S e d e p i t n e C


Published as a conference paper at ICLR 2018

.

d o h t e m d n i B - P L M g n i s u e d e p i t n e C f o

g n i n r a e l

t o h s - o r e z

e
h
t

f
o

s t l u s e R

:

e l b a T

e g a r e v A d r a w e R

d t S d r a w e R

x a M d r a w e R

e g a r e v A e c n a t s i D

d t S e c n a t s i D

x
a
M

e c n a t s i D

k s a T

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

-

.
-

.
-

.
-

.
-

.
-

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.
-

.
-

.

.
-

.
-

.
-

.
-

.
-

.

.
-

.
-

.
-

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.


.

.

.

.

.

.

.

.

.

.

.

.

x
i

S e d e p i t n e C r u o F e d e p i t n e C

t h g i E e d e p i t n e C r u o F e d e p i t n e C

n e T e d e p i t n e C r u o F e d e p i t n e C

e v l e w T e d e p i t n e C r u o F e d e p i t n e C

n e e t r u o F e d e p i t n e C r u o F e d e p i t n e C

y t n e w T e d e p i t n e C r u o F e d e p i t n e C

y t r i h T e d e p i t n e C r u o F e d e p i t n e C

y t r o F e d e p i t n e C r u o F e d e p i t n e C

t h g i E e d e p i t n e C x i

S e d e p i t n e C

n e T e d e p i t n e C x i

S e d e p i t n e C

e v l e w T e d e p i t n e C x i

S e d e p i t n e C

n e e t r u o F e d e p i t n e C x i

S e d e p i t n e C

y t n e w T e d e p i t n e C x i

S e d e p i t n e C

y t r i h T e d e p i t n e C x i

S e d e p i t n e C

y t r o F e d e p i t n e C x i

S e d e p i t n e C

x
i

S e d e p i t n e C p C r u o F e d e p i t n e C

t h g i E e d e p i t n e C p C r u o F e d e p i t n e C

n e T e d e p i t n e C p C r u o F e d e p i t n e C

e v l e w T e d e p i t n e C p C r u o F e d e p i t n e C

n e e t r u o F e d e p i t n e C p C r u o F e d e p i t n e C

t h g i E e d e p i t n e C p C x i

S e d e p i t n e C

n e T e d e p i t n e C p C x i

S e d e p i t n e C

e v l e w T e d e p i t n e C p C x i

S e d e p i t n e C

n e e t r u o F e d e p i t n e C p C x i

S e d e p i t n e C

In the MLP-Bind method, we bind the weights of MLP. By doing this, the weights of the agent from
the similar structures will be shared. For example, in the centipede environment, the weights
from observation to action of all the LeftHips are constrained to be same.


Published as a conference paper at ICLR 2018

Table 10: Results of the zero-shot learning of Snake

Random

MLPAA

SnakeFour-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1

-31.719
-56.382
-44.888
-52.764
-56.382
-44.888
-52.764

-43.328
29.262
-5.306
-25.511
-30.466
-35.655
89.358

Reward Min

MLPP

-64.958
-55.562
-42.744
-49.765
-49.953
-52.18
-57.334

Reward Std

Random

MLPAA

SnakeFour-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1

9.544
13.318
11.515
11.21
13.318
11.515
11.21

5.792
6.191
13.535
12.131
52.71
64.549
10.683

MLPP

10.769
11.715
10.304
10.981
11.544
10.528
11.615

Reward Max

Random

MLPAA

SnakeFour-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1

3.863
20.585
7.6
-2.28
20.585
7.6
-2.28

-19.409
63.866
58.779
51.265
139.617
156.771
143.824

MLPP

10.118
3.136
3.51
1.325
5.305
2.316
2.596

Reward Average

Random

MLPAA

SnakeFour-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1
SnakeFive-v1
SnakeSix-v1
SnakeSeven-v1

-13.531
-17.829
-19.89
-22.099
-17.829
-19.89
-22.099

-30.185
51.411
32.2
23.533
89.823
105.632
115.123

MLPP

-12.363
-16.481
-19.674
-22.378
-16.54
-20.454
-22.81

TreeNet

NerveNet

-58.811
-39.346
-76.42
-59.52
-47.339
-90.837
-52.015

308.994
301.4
266.066
231.557
329.456
333.237
253.828

TreeNet

NerveNet

10.146
7.883
16.869
6.953
8.519
23.109
13.529

5.513
5.788
7.395
7.179
5.85
6.604
14.084

TreeNet

NerveNet

0.918
-2.329
-2.139
-28.087
1.487
-4.159
8.394

338.374
326.974
305.011
275.536
356.366
367.768
336.561

TreeNet

NerveNet

-21.824
-21.256
-42.461
-42.742
-15.975
-34.343
-22.383

325.476
314.919
282.426
256.293
342.881
351.85
313.149

6.5.1 LINEAR NORMALIZATION ZERO-SHOT RESULTS AND COLORIZATION.

As the scale of zero-shot results is very different, we normalize the results across different models
for each transfer learning task. For each task, we record the worst value of results from different
models and the pre-set worst value Vmin. we set the normalization minimun value as this worst
value. We calculate the normalization maximum value by

max(V )/IntLen

IntLen.

(cid:98)

(cid:99) ∗

Table 11: Parameters for linear normalization during zero-shot results. Results are shown in the
order of Centipedes’ reward, Centipedes’ running-length, Snakes’ reward

parameters

value

Vmin
-100, -100, -20

IntLen

30, 30, 30


Published as a conference paper at ICLR 2018

6.6 DETAILS OF ROBUSTNESS RESULTS

Model

Mass

NerveNet

MLP

Model

NerveNet

MLP

Hopper Halfhumanoid

Horse

Wolf

Ostrich

1981.59
2047.92
1968.87
2062.36
1852.76

2268.64
1932.98
1637.47
1591.51
1073.1

2519.6
2392.05
2282.84
2161.61
1804.52

421.8
2115.75
278.92
1224.98
353.56

2304.25
1900.43
1703.47
1239.72
946.08

1692.08
1083.62
690.22
538.14
443.92

1862.21
1955.16
1648.28
1576.21
1354.51

1932.64
1754.94
1459.93
1039.9
814.55

922.12
932.48
983.11
910.29
791.84

878.75
945.94
879.66
805.99
802.41

Strength

Hopper Halfhumanoid

Horse

Wolf

Ostrich

1961.62
1117.91
492.33
468.4
446.14

800.15
431.97
429.76
426.27
412.39

1528.89
940.96
594.53
308.08
249.6

1439.89
1040.34
765.7
117.05
65.29

2316.82
960.34
488.51
305.16
201.42

1236.88
209.54
306.08
228.26
187.72

1827.73
862.19
274.77
189.49
133.84

1730.88
1108.32
519.73
88.28
108.97

794.56
539.45
371.19
255.12
197.84

230.85
483.36
163.36
232.8
133.99

Table 12: Detail results of Robustness testing.

6.7 HYPERPARAMETERS FOR MULTI-TASK LEARNING

Table 13: Hyperparameter settings for multi-task learning.

MLP, TreeNet and NerveNet

Value Tried

Network Shape
Number of Iteration Per Update
Use KL Penalty
Learning Rate Scheduler

NerveNet

Number of Propogation Steps
NerveNet Variants
Size of Nodes’ Hidden State
Merge joint and body Node
Output Network
Disable Edge Type
Add Skip-connection from / to root

[64, 64]
No
Adaptive

Value Tried

NerveNet-1
Yes
Separate
Yes
No

6.8 SCHEMATIC FIGURES OF WA L K E R S AGENTS

We design Walker task-set, which contains ﬁve 2d walkers in the MuJoCo engine.


Published as a conference paper at ICLR 2018

Figure 17: Schematic diagrams and auto-parsed graph structures of Walker-Ostrich and
Walker-Wolf.

Figure 18: Schematic diagrams and auto-parsed graph structures of Walker-Hopper and
Walker-HalfHumanoid.

Figure 19: Schematic diagrams and auto-parsed graph structures of Walker-Horse.


Walker-Ostrich RootHipToeShinHipToeShinTailNeckHeadWalker-Wolf RootThighLegFootThighLegFootThighLegFootThighLegFootHeadRootThighLegFootWalker-HopperRootThighLegFootWalker-HalfHumanoidThighLegFootWalker-Horse RootHeadHoofLegThighHoofLegThigh
