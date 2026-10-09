# Paper (Espeholt et al. 2018)

> Source: `https://arxiv.org/abs/1802.01561`

---

IMPALA: Scalable Distributed Deep-RL with Importance Weighted
Actor-Learner Architectures
Lasse Espeholt * 1 Hubert Soyer * 1 Remi Munos * 1 Karen Simonyan 1 Volodymyr Mnih 1 Tom Ward 1
Yotam Doron 1 Vlad Firoiu 1 Tim Harley 1 Iain Dunning 1 Shane Legg 1 Koray Kavukcuoglu 1
Abstract
In this work we aim to solve a large collection
of tasks using a single reinforcement learning
agent with a single set of parameters. A key
challenge is to handle the increased amount of
data and extended training time. We have devel-
oped a new distributed agent IMPALA (Impor-
tance Weighted Actor-Learner Architecture) that
not only uses resources more efﬁciently in single-
machine training but also scales to thousands of
machines without sacriﬁcing data efﬁciency or
resource utilisation. We achieve stable learning at
high throughput by combining decoupled acting
and learning with a novel off-policy correction
method called V-trace. We demonstrate the effec-
tiveness of IMPALA for multi-task reinforcement
learning on DMLab-30 (a set of 30 tasks from
the DeepMind Lab environment (Beattie et al.,
2016)) and Atari-57 (all available Atari games in
Arcade Learning Environment (Bellemare et al.,
2013a)). Our results show that IMPALA is able to
achieve better performance than previous agents
with less data, and crucially exhibits positive trans-
fer between tasks as a result of its multi-task ap-
proach. The source code is publicly available at
github.com/deepmind/scalable agent.
1. Introduction
Deep reinforcement learning methods have recently mas-
tered a wide variety of domains through trial and error
learning (Mnih et al., 2015; Silver et al., 2017; 2016; Zoph
et al., 2017; Lillicrap et al., 2015; Barth-Maron et al., 2018).
While the improvements on tasks like the game of Go (Sil-
ver et al., 2017) and Atari games (Horgan et al., 2018) have
been dramatic, the progress has been primarily in single
task performance, where an agent is trained on each task
*Equal contribution
1DeepMind Technologies, London,
United Kingdom. Correspondence to: Lasse Espeholt <lespe-
holt@google.com>.
separately. We are interested in developing new methods
capable of mastering a diverse set of tasks simultaneously as
well as environments suitable for evaluating such methods.
One of the main challenges in training a single agent on
many tasks at once is scalability. Since the current state-of-
the-art methods like A3C (Mnih et al., 2016) or UNREAL
(Jaderberg et al., 2017b) can require as much as a billion
frames and multiple days to master a single domain, training
them on tens of domains at once is too slow to be practical.
We propose the Importance Weighted Actor-Learner
Architecture (IMPALA) shown in Figure 1. IMPALA is
capable of scaling to thousands of machines without sacri-
ﬁcing training stability or data efﬁciency. Unlike the popular
A3C-based agents, in which workers communicate gradi-
ents with respect to the parameters of the policy to a central
parameter server, IMPALA actors communicate trajectories
of experience (sequences of states, actions, and rewards) to a
centralised learner. Since the learner in IMPALA has access
to full trajectories of experience we use a GPU to perform
updates on mini-batches of trajectories while aggressively
parallelising all time independent operations. This type of
decoupled architecture can achieve very high throughput.
However, because the policy used to generate a trajectory
can lag behind the policy on the learner by several updates at
the time of gradient calculation, learning becomes off-policy.
Therefore, we introduce the V-trace off-policy actor-critic
algorithm to correct for this harmful discrepancy.
With the scalable architecture and V-trace combined, IM-
PALA achieves exceptionally high data throughput rates of
250,000 frames per second, making it over 30 times faster
than single-machine A3C. Crucially, IMPALA is also more
data efﬁcient than A3C based agents and more robust to
hyperparameter values and network architectures, allow-
ing it to make better use of deeper neural networks. We
demonstrate the effectiveness of IMPALA by training a sin-
gle agent on multi-task problems using DMLab-30, a new
challenge set which consists of 30 diverse cognitive tasks
in the 3D DeepMind Lab (Beattie et al., 2016) environment
and by training a single agent on all games in the Atari-57
set of tasks.
arXiv:1802.01561v3  [cs.LG]  28 Jun 2018

IMPALA: Importance Weighted Actor-Learner Architectures
Actor
Actor
Actor
Actor
Actor
Actor
Learner
Observations
Parameters
Actor
Actor
Observations
Observations
Parameters
Gradients
Learner
Worker
Master
Learner
Actor
Actor
Actor
Figure 1. Left: Single Learner. Each actor generates trajectories
and sends them via a queue to the learner. Before starting the next
trajectory, actor retrieves the latest policy parameters from learner.
Right: Multiple Synchronous Learners. Policy parameters are
distributed across multiple learners that work synchronously.
2. Related Work
The earliest attempts to scale up deep reinforcement learn-
ing relied on distributed asynchronous SGD (Dean et al.,
2012) with multiple workers. Examples include distributed
A3C (Mnih et al., 2016) and Gorila (Nair et al., 2015), a
distributed version of Deep Q-Networks (Mnih et al., 2015).
Recent alternatives to asynchronous SGD for RL include
using evolutionary processes (Salimans et al., 2017), dis-
tributed BA3C (Adamski et al., 2018) and Ape-X (Horgan
et al., 2018) which has a distributed replay but a synchronous
learner.
There have also been multiple efforts that scale up reinforce-
ment learning by utilising GPUs. One of the simplest of
such methods is batched A2C (Clemente et al., 2017). At
every step, batched A2C produces a batch of actions and
applies them to a batch of environments. Therefore, the
slowest environment in each batch determines the time it
takes to perform the entire batch step (see Figure 2a and
2b). In other words, high variance in environment speed
can severely limit performance. Batched A2C works partic-
ularly well on Atari environments, because rendering and
game logic are computationally very cheap in comparison to
the expensive tensor operations performed by reinforcement
learning agents. However, more visually or physically com-
plex environments can be slower to simulate and can have
high variance in the time required for each step. Environ-
ments may also have variable length (sub)episodes causing
a slowdown when initialising an episode.
The most similar architecture to IMPALA is GA3C
(Babaeizadeh et al., 2016), which also uses asynchronous
data collection to more effectively utilise GPUs. It de-
couples the acting/forward pass from the gradient calcu-
lation/backward pass by using dynamic batching. The ac-
tor/learner asynchrony in GA3C leads to instabilities during
learning, which (Babaeizadeh et al., 2016) only partially
mitigates by adding a small constant to action probabilities
Environment steps
Forward pass
Backward pass
Actor 2
Actor 3
Actor 1
Actor 0
4 time steps
(a) Batched A2C (sync step.)
Actor 2
Actor 3
Actor 1
Actor 0
4 time steps
(b) Batched A2C (sync traj.)
…
…
Actor 2
Actor 3
Actor 1
Actor 0
Actor 4
Actor 5
Actor 6
Actor 7
...next unroll
(c) IMPALA
Figure 2. Timeline for one unroll with 4 steps using different ar-
chitectures. Strategies shown in (a) and (b) can lead to low GPU
utilisation due to rendering time variance within a batch. In (a),
the actors are synchronised after every step. In (b) after every n
steps. IMPALA (c) decouples acting from learning.
during the estimation of the policy gradient. In contrast,
IMPALA uses the more principled V-trace algorithm.
Related previous work on off-policy RL include (Precup
et al., 2000; 2001; Wawrzynski, 2009; Geist & Scherrer,
2014; O’Donoghue et al., 2017) and (Harutyunyan et al.,
2016). The closest work to ours is the Retrace algorithm
(Munos et al., 2016) which introduced an off-policy correc-
tion for multi-step RL, and has been used in several agent
architectures (Wang et al., 2017; Gruslys et al., 2018). Re-
trace requires learning state-action-value functions Q in
order to make the off-policy correction. However, many
actor-critic methods such as A3C learn a state-value func-
tion V instead of a state-action-value function Q. V-trace is
based on the state-value function.
3. IMPALA
IMPALA (Figure 1) uses an actor-critic setup to learn a
policy π and a baseline function V π. The process of gener-
ating experiences is decoupled from learning the parameters
of π and V π. The architecture consists of a set of actors,
repeatedly generating trajectories of experience, and one or
more learners that use the experiences sent from actors to
learn π off-policy.
At the beginning of each trajectory, an actor updates its
own local policy µ to the latest learner policy π and runs
it for n steps in its environment. After n steps, the ac-
tor sends the trajectory of states, actions and rewards
x1, a1, r1, . . . , xn, an, rn together with the corresponding
policy distributions µ(at|xt) and initial LSTM state to the
learner through a queue. The learner then continuously
updates its policy π on batches of trajectories, each col-
lected from many actors. This simple architecture enables
the learner(s) to be accelerated using GPUs and actors to
be easily distributed across many machines. However, the
learner policy π is potentially several updates ahead of the
actor’s policy µ at the time of update, therefore there is a
policy-lag between the actors and learner(s). V-trace cor-

IMPALA: Importance Weighted Actor-Learner Architectures
rects for this lag to achieve extremely high data throughput
while maintaining data efﬁciency. Using an actor-learner ar-
chitecture, provides fault tolerance like distributed A3C but
often has lower communication overhead since the actors
send observations rather than parameters/gradients.
With the introduction of very deep model architectures, the
speed of a single GPU is often the limiting factor during
training. IMPALA can be used with distributed set of learn-
ers to train large neural networks efﬁciently as shown in
Figure 1. Parameters are distributed across the learners and
actors retrieve the parameters from all the learners in par-
allel while only sending observations to a single learner.
IMPALA use synchronised parameter update which is vital
to maintain data efﬁciency when scaling to many machines
(Chen et al., 2016).
3.1. Efﬁciency Optimisations
GPUs and many-core CPUs beneﬁt greatly from running
few large, parallelisable operations instead of many small
operations. Since the learner in IMPALA performs updates
on entire batches of trajectories, it is able to parallelise more
of its computations than an online agent like A3C. As an
example, a typical deep RL agent features a convolutional
network followed by a Long Short-Term Memory (LSTM)
(Hochreiter & Schmidhuber, 1997) and a fully connected
output layer after the LSTM. An IMPALA learner applies
the convolutional network to all inputs in parallel by folding
the time dimension into the batch dimension. Similarly, it
also applies the output layer to all time steps in parallel
once all LSTM states are computed. This optimisation
increases the effective batch size to thousands. LSTM-based
agents also obtain signiﬁcant speedups on the learner by
exploiting the network structure dependencies and operation
fusion (Appleyard et al., 2016).
Finally, we also make use of several off the shelf optimisa-
tions available in TensorFlow (Abadi et al., 2017) such as
preparing the next batch of data for the learner while still per-
forming computation, compiling parts of the computational
graph with XLA (a TensorFlow Just-In-Time compiler) and
optimising the data format to get the maximum performance
from the cuDNN framework (Chetlur et al., 2014).
4. V-trace
Off-policy learning is important in the decoupled distributed
actor-learner architecture because of the lag between when
actions are generated by the actors and when the learner
estimates the gradient. To this end, we introduce a novel off-
policy actor-critic algorithm for the learner, called V-trace.
First, let us introduce some notations. We consider the
problem of discounted inﬁnite-horizon RL in Markov De-
cision Processes (MDP), see (Puterman, 1994; Sutton &
Barto, 1998) where the goal is to ﬁnd a policy π that
maximises the expected sum of future discounted rewards:
V π(x)
def
= Eπ
 P
t≥0 γtrt

, where γ ∈[0, 1) is the dis-
count factor, rt = r(xt, at) is the reward at time t, xt is the
state at time t (initialised in x0 = x) and at ∼π(·|xt) is the
action generated by following some policy π.
The goal of an off-policy RL algorithm is to use trajectories
generated by some policy µ, called the behaviour policy, to
learn the value function V π of another policy π (possibly
different from µ), called the target policy.
4.1. V-trace target
Consider a trajectory (xt, at, rt)t=s+n
t=s
generated by the ac-
tor following some policy µ. We deﬁne the n-steps V-trace
target for V (xs), our value approximation at state xs, as:
vs
def
=
V (xs) + Ps+n−1
t=s
γt−s Qt−1
i=s ci

δtV , (1)
where δtV
def
= ρt
 rt + γV (xt+1) −V (xt)

is a temporal
difference for V , and ρt
def
= min
 ¯ρ, π(at|xt)
µ(at|xt)

and ci
def
=
min
 ¯c, π(ai|xi)
µ(ai|xi)

are truncated importance sampling (IS)
weights (we make use of the notation Qt−1
i=s ci = 1 for
s = t). In addition we assume that the truncation levels are
such that ¯ρ ≥¯c.
Notice that in the on-policy case (when π = µ), and as-
suming that ¯c ≥1, then all ci = 1 and ρt = 1, thus (1)
rewrites
vs = V (xs) + Ps+n−1
t=s
γt−s rt + γV (xt+1) −V (xt)

= Ps+n−1
t=s
γt−srt + γnV (xs+n),
(2)
which is the on-policy n-steps Bellman target. Thus in
the on-policy case, V-trace reduces to the on-policy n-steps
Bellman update. This property (which Retrace (Munos et al.,
2016) does not have) allows one to use the same algorithm
for off- and on-policy data.
Notice that the (truncated) IS weights ci and ρt play dif-
ferent roles. The weight ρt appears in the deﬁnition of the
temporal difference δtV and deﬁnes the ﬁxed point of this
update rule. In a tabular case, where functions can be per-
fectly represented, the ﬁxed point of this update (i.e., when
V (xs) = vs for all states), characterised by δtV being equal
to zero in expectation (under µ), is the value function V π¯
ρ
of some policy π¯ρ, deﬁned by
π¯ρ(a|x)
def
=
min
 ¯ρµ(a|x), π(a|x)

P
b∈A min
 ¯ρµ(b|x), π(b|x)
,
(3)
(see the analysis in Appendix A ). So when ¯ρ is inﬁnite
(i.e. no truncation of ρt), then this is the value function V π
of the target policy. However if we choose a truncation

IMPALA: Importance Weighted Actor-Learner Architectures
level ¯ρ < ∞, our ﬁxed point is the value function V π¯
ρ of
a policy π¯ρ which is somewhere between µ and π. At the
limit when ¯ρ is close to zero, we obtain the value function
of the behaviour policy V µ. In Appendix A we prove the
contraction of a related V-trace operator and the convergence
of the corresponding online V-trace algorithm.
The weights ci are similar to the “trace cutting” coefﬁcients
in Retrace. Their product cs . . . ct−1 measures how much
a temporal difference δtV observed at time t impacts the
update of the value function at a previous time s. The more
dissimilar π and µ are (the more off-policy we are), the
larger the variance of this product. We use the truncation
level ¯c as a variance reduction technique. However notice
that this truncation does not impact the solution to which
we converge (which is characterised by ¯ρ only).
Thus we see that the truncation levels ¯c and ¯ρ represent
different features of the algorithm: ¯ρ impacts the nature of
the value function we converge to, whereas ¯c impacts the
speed at which we converge to this function.
Remark 1. V-trace targets can be computed recursively:
vs = V (xs) + δsV + γcs
 vs+1 −V (xs+1)

.
Remark 2. Like in Retrace(λ), we can also consider an
additional discounting parameter λ ∈[0, 1] in the deﬁnition
of V-trace by setting ci = λ min
 ¯c, π(ai|xi)
µ(ai|xi)

. In the on-
policy case, when n = ∞, V-trace then reduces to TD(λ).
4.2. Actor-Critic algorithm
POLICY GRADIENT
In the on-policy case, the gradient of the value function
V µ(x0) with respect to some parameter of the policy µ is
∇V µ(x0) = Eµ
hP
s≥0γs∇log µ(as|xs)Qµ(xs, as)
i
,
where Qµ(xs, as)
def
=
Eµ
 P
t≥s γt−srt|xs, as

is the
state-action value of policy µ at (xs, as).
This is
usually implemented by a stochastic gradient ascent
that updates the policy parameters in the direction of
Eas∼µ(·|xs)
h
∇log µ(as|xs)qs
xs
i
, where qs is an estimate
of Qµ(xs, as), and averaged over the set of states xs that
are visited under some behaviour policy µ.
Now in the off-policy setting that we consider, we can use
an IS weight between the policy being evaluated π¯ρ and the
behaviour policy µ, to update our policy parameter in the
direction of
Eas∼µ(·|xs)
hπ¯ρ(as|xs)
µ(as|xs) ∇log π¯ρ(as|xs)qs
xs
i
(4)
where qs
def
= rs + γvs+1 is an estimate of Qπ¯
ρ(xs, as)
built from the V-trace estimate vs+1 at the next state xs+1.
The reason why we use qs instead of vs as the target for
our Q-value Qπ¯
ρ(xs, as) is that, assuming our value esti-
mate is correct at all states, i.e. V = V π¯
ρ, then we have
E[qs|xs, as] = Qπ¯
ρ(xs, as) (whereas we do not have this
property if we choose qt = vt). See Appendix A for analy-
sis and Appendix E.3 for a comparison of different ways to
estimate qs.
In order to reduce the variance of the policy gradient es-
timate (4), we usually subtract from qs a state-dependent
baseline, such as the current value approximation V (xs).
Finally notice that (4) estimates the policy gradient for π¯ρ
which is the policy evaluated by the V-trace algorithm when
using a truncation level ¯ρ. However assuming the bias
V π¯
ρ −V π is small (e.g. if ¯ρ is large enough) then we can
expect qs to provide us with a good estimate of Qπ(xs, as).
Taking into account these remarks, we derive the following
canonical V-trace actor-critic algorithm.
V-TRACE ACTOR-CRITIC ALGORITHM
Consider a parametric representation Vθ of the value func-
tion and the current policy πω. Trajectories have been gen-
erated by actors following some behaviour policy µ. The
V-trace targets vs are deﬁned by (1). At training time s, the
value parameters θ are updated by gradient descent on the
l2 loss to the target vs, i.e., in the direction of
 vs −Vθ(xs)

∇θVθ(xs),
and the policy parameters ω in the direction of the policy
gradient:
ρs∇ω log πω(as|xs)
 rs + γvs+1 −Vθ(xs)

.
In order to prevent premature convergence we may add an
entropy bonus, like in A3C, along the direction
−∇ω
X
a
πω(a|xs) log πω(a|xs).
The overall update is obtained by summing these three gra-
dients rescaled by appropriate coefﬁcients, which are hyper-
parameters of the algorithm.
5. Experiments
We investigate the performance of IMPALA under multiple
settings. For data efﬁciency, computational performance
and effectiveness of the off-policy correction we look at the
learning behaviour of IMPALA agents trained on individual
tasks. For multi-task learning we train agents—each with
one set of weights for all tasks—on a newly introduced
collection of 30 DeepMind Lab tasks and on all 57 games of
the Atari Learning Environment (Bellemare et al., 2013a).
For all the experiments we have used two different model
architectures: a shallow model similar to (Mnih et al., 2016)

IMPALA: Importance Weighted Actor-Learner Architectures
/255
Conv. 8 ⇥8, stride 4
ReLU
Conv. 4 ⇥4, stride 2
ReLU
FC 256
ReLU
Vt
rt−1 at−1
32
16
3
96 ⇥72
LSTM 256
ht−1
⇡(at)
LSTM 64
Embedding 20
blue ladder
+
Conv. 3 ⇥3, stride 1
ReLU
Conv. 3 ⇥3, stride 1
/255
Conv. 3 ⇥3, stride 1
FC 256
ReLU
Max 3 ⇥3, stride 2
Residual Block
Residual Block
⇥3
ReLU
[16, 32, 32] ch.
LSTM 256
Vt
rt−1 at−1
LSTM 256
ht−1
⇡(at)
96 ⇥72
3
ReLU
LSTM 64
Embedding 20
blue ladder
Figure 3. Model Architectures. Left: Small architecture, 2 convo-
lutional layers and 1.2 million parameters. Right: Large architec-
ture, 15 convolutional layers and 1.6 million parameters.
Architecture
CPUs GPUs1
FPS2
Single-Machine
Task 1
Task 2
A3C 32 workers
64
0
6.5K
9K
Batched A2C (sync step)
48
0
9K
5K
Batched A2C (sync step)
48
1
13K
5.5K
Batched A2C (sync traj.)
48
0
16K
17.5K
Batched A2C (dyn. batch)
48
1
16K
13K
IMPALA 48 actors
48
0
17K
20.5K
IMPALA (dyn. batch) 48 actors3
48
1
21K
24K
Distributed
A3C
200
0
46K
50K
IMPALA
150
1
80K
IMPALA (optimised)
375
1
200K
IMPALA (optimised) batch 128
500
1
250K
1 Nvidia P100 2 In frames/sec (4 times the agent steps due to action repeat). 3 Limited by
amount of rendering possible on a single machine.
Table 1. Throughput on seekavoid arena 01 (task 1) and
rooms keys doors puzzle (task 2) with the shallow model
in Figure 3. The latter has variable length episodes and slow
restarts. Batched A2C and IMPALA use batch size 32 if not other-
wise mentioned.
with an LSTM before the policy and value (shown in Fig-
ure 3 (left)) and a deeper residual model (He et al., 2016)
(shown in Figure 3 (right)). For tasks with a language chan-
nel we used an LSTM with text embeddings as input.
5.1. Computational Performance
High throughput, computational efﬁciency and scalability
are among the main design goals of IMPALA. To demon-
strate that IMPALA outperforms current algorithms in these
metrics we compare A3C (Mnih et al., 2016), batched A2C
variations and IMPALA variants with various optimisations.
For single-machine experiments using GPUs, we use dy-
namic batching in the forward pass to avoid several batch
size 1 forward passes. Our dynamic batching module is
implemented by specialised TensorFlow operations but is
conceptually similar to the queues used in GA3C. Table 1
details the results for single-machine and multi-machine ver-
sions with the shallow model from Figure 3. In the single-
machine case, IMPALA achieves the highest performance
on both tasks, ahead of all batched A2C variants and ahead
of A3C. However, the distributed, multi-machine setup is
where IMPALA can really demonstrate its scalability. With
the optimisations from Section 3.1 to speed up the GPU-
based learner, the IMPALA agent achieves a throughput rate
of 250,000 frames/sec or 21 billion frames/day. Note, to
reduce the number of actors needed per learner, one can
use auxiliary losses, data from experience replay or other
expensive learner-only computation.
5.2. Single-Task Training
To investigate IMPALA’s learning dynamics, we employ the
single-task scenario where we train agents individually on
5 different DeepMind Lab tasks. The task set consists of a
planning task, two maze navigation tasks, a laser tag task
with scripted bots and a simple fruit collection task.
We perform hyperparameter sweeps over the weighting of
entropy regularisation, the learning rate and the RMSProp
epsilon. For each experiment we use an identical set of 24
pre-sampled hyperparameter combinations from the ranges
in Appendix D.1 . The other hyperparameters were ﬁxed to
values speciﬁed in Appendix D.3 .
5.2.1. CONVERGENCE AND STABILITY
Figure 4 shows a comparison between IMPALA, A3C
and batched A2C with the shallow model in Figure 3.
In all of the 5 tasks, either batched A2C or IMPALA
reach the best ﬁnal average return and in all tasks but
seekavoid arena 01 they are ahead of A3C through-
out the entire course of training. IMPALA outperforms
the synchronous batched A2C on 2 out of 5 tasks while
achieving much higher throughput (see Table 1). We hy-
pothesise that this behaviour could stem from the V-trace
off-policy correction acting similarly to generalised advan-
tage estimation (Schulman et al., 2016) and asynchronous
data collection yielding more diverse batches of experience.
In addition to reaching better ﬁnal performance, IMPALA is
also more robust to the choice of hyperparameters than A3C.
Figure 4 compares the ﬁnal performance of the aforemen-
tioned methods across different hyperparameter combina-
tions, sorted by average ﬁnal return from high to low. Note
that IMPALA achieves higher scores over a larger number
of combinations than A3C.
5.2.2. V-TRACE ANALYSIS
To analyse V-trace we investigate four different algorithms:
1. No-correction - No off-policy correction.

IMPALA: Importance Weighted Actor-Learner Architectures
IMPALA - 1 GPU - 200 actors
Batched A2C - Single Machine - 32 workers
A3C - Single Machine - 32 workers
A3C - Distributed - 200 workers
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
10
15
20
25
30
35
40
45
50
55
Return
rooms_watermaze
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
0
5
10
15
20
25
30
rooms_keys_doors_puzzle
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
−5
0
5
10
15
20
25
30
35
lasertag_three_opponents_small
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
0
50
100
150
200
250
explore_goal_locations_small
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
5
10
15
20
25
30
35
40
45
seekavoid_arena_01
1
5
9
13
17
21
24
Hyperparameter Combination
0
10
20
30
40
50
60
Final Return
rooms_watermaze
1
5
9
13
17
21
24
Hyperparameter Combination
0
5
10
15
20
25
30
35
40
rooms_keys_doors_puzzle
1
5
9
13
17
21
24
Hyperparameter Combination
−5
0
5
10
15
20
25
30
35
40
lasertag_three_opponents_small
1
5
9
13
17
21
24
Hyperparameter Combination
0
50
100
150
200
250
300
explore_goal_locations_small
1
5
9
13
17
21
24
Hyperparameter Combination
0
10
20
30
40
50
seekavoid_arena_01
Figure 4. Top Row: Single task training on 5 DeepMind Lab tasks. Each curve is the mean of the best 3 runs based on ﬁnal return.
IMPALA achieves better performance than A3C. Bottom Row: Stability across hyperparameter combinations sorted by the ﬁnal
performance across different hyperparameter combinations. IMPALA is consistently more stable than A3C.
Task 1
Task 2
Task 3
Task 4
Task 5
Without Replay
V-trace
46.8
32.9
31.3
229.2
43.8
1-Step
51.8
35.9
25.4
215.8
43.7
ε-correction
44.2
27.3
4.3
107.7
41.5
No-correction
40.3
29.1
5.0
94.9
16.1
With Replay
V-trace
47.1
35.8
34.5
250.8
46.9
1-Step
54.7
34.4
26.4
204.8
41.6
ε-correction
30.4
30.2
3.9
101.5
37.6
No-correction
35.0
21.1
2.8
85.0
11.2
Tasks: rooms watermaze, rooms keys doors puzzle,
lasertag three opponents small,
explore goal locations small, seekavoid arena 01
Table 2. Average ﬁnal return over 3 best hyperparameters for differ-
ent off-policy correction methods on 5 DeepMind Lab tasks. When
the lag in policy is negligible both V-trace and 1-step importance
sampling perform similarly well and better than ε-correction/No-
correction. However, when the lag increases due to use of expe-
rience replay, V-trace performs better than all other methods in 4
out 5 tasks.
2. ε-correction - Add a small value (ε = 1e-6) during
gradient calculation to prevent log π(a) from becoming
very small and leading to numerical instabilities, similar
to (Babaeizadeh et al., 2016).
3. 1-step importance sampling - No off-policy correction
when optimising V (x). For the policy gradient, multiply
the advantage at each time step by the corresponding im-
portance weight. This variant is similar to V-trace without
“traces” and is included to investigate the importance of
“traces” in V-trace.
4. V-trace as described in Section 4.
For V-trace and 1-step importance sampling we clip each
importance weight ρt and ct at 1 (i.e. ¯c = ¯ρ = 1). This
reduces the variance of the gradient estimate but introduces
a bias. Out of ¯ρ ∈[1, 10, 100] we found that ¯ρ = 1 worked
best.
We evaluate all algorithms on the set of 5 DeepMind Lab
tasks from the previous section. We also add an experience
replay buffer on the learner to increase the off-policy gap
between π and µ. In the experience replay experiments we
draw 50% of the items in each batch uniformly at random
from the replay buffer. Table 2 shows the ﬁnal performance
for each algorithm with and without replay respectively. In
the no replay setting, V-trace performs best on 3 out of 5
tasks, followed by 1-step importance sampling, ε-correction
and No-correction. Although 1-step importance sampling
performs similarly to V-trace in the no-replay setting, the
gap widens on 4 out 5 tasks when using experience replay.
This suggests that the cruder 1-step importance sampling ap-
proximation becomes insufﬁcient as the target and behaviour
policies deviate from each other more strongly. Also note
that V-trace is the only variant that consistently beneﬁts
from adding experience replay. ε-correction improves sig-
niﬁcantly over No-correction on two tasks but lies far behind
the importance-sampling based methods, particularly in the
more off-policy setting with experience replay. Figure E.1
shows results of a more detailed analysis. Figure E.2 shows
that the importance-sampling based methods also perform
better across all hyperparameters and are typically more
robust.
5.3. Multi-Task Training
IMPALA’s high data throughput and data efﬁciency allow us
to train not only on one task but on multiple tasks in parallel
with only a minimal change to the training setup. Instead
of running the same task on all actors, we allocate a ﬁxed
number of actors to each task in the multi-task suite. Note,
the model does not know which task it is being trained or
evaluated on.

IMPALA: Importance Weighted Actor-Learner Architectures
Model
Test score
A3C, deep
23.8%
IMPALA, shallow
37.1%
IMPALA-Experts, deep
44.5%
IMPALA, deep
46.5%
IMPALA, deep, PBT
49.4%
IMPALA, deep, PBT, 8 learners
49.1%
Table 3. Mean capped human normalised scores on DMLab-30.
All models were evaluated on the test tasks with 500 episodes per
task. The table shows the best score for each architecture.
5.3.1. DMLAB-30
To test IMPALA’s performance in a multi-task setting we use
DMLab-30, a set of 30 diverse tasks built on DeepMind Lab.
Among the many task types in the suite are visually complex
environments with natural-looking terrain, instruction-based
tasks with grounded language (Hermann et al., 2017), navi-
gation tasks, cognitive (Leibo et al., 2018) and ﬁrst-person
tagging tasks featuring scripted bots as opponents. A de-
tailed description of DMLab-30 and the tasks are available
at github.com/deepmind/lab and deepmind.com/dm-lab-30.
We compare multiple variants of IMPALA with a distributed
A3C implementation. Except for agents using population-
based training (PBT) (Jaderberg et al., 2017a), all agents are
trained with hyperparameter sweeps across the same range
given in Appendix D.1 . We report mean capped human
normalised score where the score for each task is capped
at 100% (see Appendix B ). Using mean capped human
normalised score emphasises the need to solve multiple
tasks instead of focusing on becoming super human on
a single task. For PBT we use the mean capped human
normalised score as ﬁtness function and tune entropy cost,
learning rate and RMSProp ε. See Appendix F for the
speciﬁcs of the PBT setup.
In particular, we compare the following agent variants. A3C,
deep, a distributed implementation with 210 workers (7
per task) featuring the deep residual network architecture
(Figure 3 (Right)). IMPALA, shallow with 210 actors and
IMPALA, deep with 150 actors both with a single learner.
IMPALA, deep, PBT, the same as IMPALA, deep, but ad-
ditionally using the PBT (Jaderberg et al., 2017a) for hy-
perparameter optimisation. Finally IMPALA, deep, PBT, 8
learners, which utilises 8 learner GPUs to maximise learn-
ing speed. We also train IMPALA agents in an expert setting,
IMPALA-Experts, deep, where a separate agent is trained
per task. In this case we did not optimise hyperparameters
for each task separately but instead across all tasks on which
the 30 expert agents were trained.
Table 3 and Figure 5 show all variants of IMPALA perform-
ing much better than the deep distributed A3C. Moreover,
the deep variant of IMPALA performs better than the shal-
low network version not only in terms of ﬁnal performance
but throughout the entire training. Note in Table 3 that
IMPALA, deep, PBT, 8 learners, although providing much
higher throughput, reaches the same ﬁnal performance as
the 1 GPU IMPALA, deep, PBT in the same number of steps.
Of particular importance is the gap between the IMPALA-
Experts which were trained on each task individually and
IMPALA, deep, PBT which was trained on all tasks at once.
As Figure 5 shows, the multi-task version is outperforms
IMPALA-Experts throughout training and the breakdown
into individual scores in Appendix B shows positive transfer
on tasks such as language tasks and laser tag tasks.
Comparing A3C to IMPALA with respect to wall clock time
(Figure 6) further highlights the scalability gap between
the two approaches. IMPALA with 1 learner takes only
around 10 hours to reach the same performance that A3C
approaches after 7.5 days. Using 8 learner GPUs instead of
1 further speeds up training of the deep model by a factor of
7 to 210K frames/sec, up from 30K frames/sec.
5.3.2. ATARI
The Atari Learning Environment (ALE) (Bellemare et al.,
2013b) has been the testing ground of most recent deep
reinforcement agents. Its 57 tasks pose challenging rein-
forcement learning problems including exploration, plan-
ning, reactive play and complex visual input. Most games
feature very different visuals and game mechanics which
makes this domain particularly challenging for multi-task
learning.
We train IMPALA and A3C agents on each game individu-
ally and compare their performance using the deep network
(without the LSTM) introduced in Section 5. We also pro-
vide results using a shallow network that is equivalent to
the feed forward network used in (Mnih et al., 2016) which
features three convolutional layers. The network is provided
with a short term history by stacking the 4 most recent ob-
servations at each step. For details on pre-processing and
hyperparameter setup please refer to Appendix G .
In addition to individual per-game experts, trained for 200
million frames with a ﬁxed set of hyperparameters, we train
an IMPALA Atari-57 agent—one agent, one set of weights—
on all 57 Atari games at once for 200 million frames per
game or a total of 11.4 billion frames. For the Atari-57 agent,
we use population based training with a population size of
24 to adapt entropy regularisation, learning rate, RMSProp ε
and the global gradient norm clipping threshold throughout
training.
We compare all algorithms in terms of median human nor-
malised score across all 57 Atari games. Evaluation follows
a standard protocol, each game-score is the mean over 200
evaluation episodes, each episode was started with a random

IMPALA: Importance Weighted Actor-Learner Architectures
Figure 5. Performance of best agent in each sweep/population dur-
ing training on the DMLab-30 task-set wrt. data consumed across
all environments. IMPALA with multi-task training is not only
faster, it also converges at higher accuracy with better data efﬁ-
ciency across all 30 tasks. The x-axis is data consumed by one
agent out of a hyperparameter sweep/PBT population of 24 agents,
total data consumed across the whole population/sweep can be
obtained by multiplying with the population/sweep size.
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e10
0
10
20
30
40
50
60
Mean Capped Normalized Score
0
20
40
60
80
100
120
140
160
180
Wall Clock Time (hours)
0
10
20
30
40
50
60
Mean Capped Normalized Score
IMPALA, deep, PBT - 8 GPUs
IMPALA, deep, PBT
IMPALA, deep
IMPALA, shallow
IMPALA-Experts, deep
A3C, deep
Figure 6. Performance on DMLab-30 wrt. wall-clock time. All
models used the deep architecture (Figure 3). The high throughput
of IMPALA results in orders of magnitude faster learning.
number of no-op actions (uniformly chosen from [1, 30]) to
combat the determinism of the ALE environment.
As table 4 shows, IMPALA experts provide both better ﬁnal
performance and data efﬁciency than their A3C counterparts
in the deep and the shallow conﬁguration. As in our Deep-
Mind Lab experiments, the deep residual network leads
to higher scores than the shallow network, irrespective of
the reinforcement learning algorithm used. Note that the
shallow IMPALA experiment completes training over 200
million frames in less than one hour.
We want to particularly emphasise that IMPALA, deep, multi-
task, a single agent trained on all 57 ALE games at once,
reaches 59.7% median human normalised score. Despite
Human Normalised Return Median
Mean
A3C, shallow, experts
54.9% 285.9%
A3C, deep, experts
117.9% 503.6%
Reactor, experts
187%
N/A
IMPALA, shallow, experts
93.2% 466.4%
IMPALA, deep, experts
191.8% 957.6%
IMPALA, deep, multi-task
59.7% 176.9%
Table 4. Human normalised scores on Atari-57. Up to 30 no-ops
at the beginning of each episode. For a level-by-level comparison
to ACKTR (Wu et al., 2017) and Reactor see Appendix C.1 .
the high diversity in visual appearance and game mechanics
within the ALE suite, IMPALA multi-task still manages
to stay competitive to A3C, shallow, experts, commonly
used as a baseline in related work. ALE is typically con-
sidered a hard multi-task environment, often accompanied
by negative transfer between tasks (Rusu et al., 2016). To
our knowledge, IMPALA is the ﬁrst agent to be trained in a
multi-task setting on all 57 games of ALE that is competitive
with a standard expert baseline.
6. Conclusion
We have introduced a new highly scalable distributed agent,
IMPALA, and a new off-policy learning algorithm, V-trace.
With its simple but scalable distributed architecture, IM-
PALA can make efﬁcient use of available compute at small
and large scale.
This directly translates to very quick
turnaround for investigating new ideas and opens up un-
explored opportunities.
V-trace is a general off-policy learning algorithm that is
more stable and robust compared to other off-policy correc-
tion methods for actor critic agents. We have demonstrated
that IMPALA achieves better performance compared to
A3C variants in terms of data efﬁciency, stability and ﬁnal
performance. We have further evaluated IMPALA on the
new DMLab-30 set and the Atari-57 set. To the best of
our knowledge, IMPALA is the ﬁrst Deep-RL agent that
has been successfully tested in such large-scale multi-task
settings and it has shown superior performance compared
to A3C based agents (49.4% vs. 23.8% human normalised
score on DMLab-30). Most importantly, our experiments
on DMLab-30 show that, in the multi-task setting, positive
transfer between individual tasks lead IMPALA to achieve
better performance compared to the expert training setting.
We believe that IMPALA provides a simple yet scalable and
robust framework for building better Deep-RL agents and
has the potential to enable research on new challenges.

IMPALA: Importance Weighted Actor-Learner Architectures
Acknowledgements
We would like to thank Denis Teplyashin, Ricardo Barreira,
Manuel Sanchez for their work improving the performance
on DMLab-30 environments and Matteo Hessel, Jony Hud-
son, Igor Babuschkin, Max Jaderberg, Ivo Danihelka, Jacob
Menick and David Silver for their comments and insightful
discussions.
References
Abadi, M., Isard, M., and Murray, D. G. A computational
model for tensorﬂow: An introduction. In Proceedings
of the 1st ACM SIGPLAN International Workshop on
Machine Learning and Programming Languages, MAPL
2017, 2017. ISBN 978-1-4503-5071-6.
Adamski, I., Adamski, R., Grel, T., Jedrych, A., Kaczmarek,
K., and Michalewski, H. Distributed deep reinforcement
learning: Learn how to play atari games in 21 minutes.
CoRR, abs/1801.02852, 2018.
Appleyard, J., Kocisk´y, T., and Blunsom, P. Optimizing
performance of recurrent neural networks on gpus. CoRR,
abs/1604.01946, 2016.
Babaeizadeh, M., Frosio, I., Tyree, S., Clemons, J., and
Kautz, J. GA3C: GPU-based A3C for deep reinforcement
learning. NIPS Workshop, 2016.
Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney,
W., Horgan, D., Tirumala, D., Muldal, A., Heess, N., and
Lillicrap, T. Distributional policy gradients. ICLR, 2018.
Beattie, C., Leibo, J. Z., Teplyashin, D., Ward, T., Wain-
wright, M., Kuttler, H., Lefrancq, A., Green, S., Valdes,
V., Sadik, A., Schrittwieser, J., Anderson, K., York, S.,
Cant, M., Cain, A., Bolton, A., Gaffney, S., King, H.,
Hassabis, D., Legg, S., and Petersen, S. Deepmind lab.
CoRR, abs/1612.03801, 2016.
Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M.
The Arcade Learning Environment: An evaluation plat-
form for general agents. Journal of Artiﬁcial Intelligence
Research, 47:253–279, June 2013a.
Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M.
The arcade learning environment: An evaluation platform
for general agents. J. Artif. Intell. Res.(JAIR), 47:253–279,
2013b.
Chen, J., Monga, R., Bengio, S., and J´ozefowicz,
R.
Revisiting distributed synchronous SGD.
CoRR,
abs/1604.00981, 2016.
Chetlur, S., Woolley, C., Vandermersch, P., Cohen, J., Tran,
J., Catanzaro, B., and Shelhamer, E. cudnn: Efﬁcient
primitives for deep learning. CoRR, abs/1410.0759, 2014.
Clemente, A. V., Mart´ınez, H. N. C., and Chandra, A. Ef-
ﬁcient parallel methods for deep reinforcement learning.
CoRR, abs/1705.04862, 2017.
Dean, J., Corrado, G., Monga, R., Chen, K., Devin, M.,
Mao, M., Ranzato, M., Senior, A., Tucker, P., Yang, K.,
Le, Q. V., and Ng, A. Y. Large scale distributed deep
networks. In Advances in Neural Information Processing
Systems 25, pp. 1223–1231, 2012.
Geist, M. and Scherrer, B. Off-policy learning with eligibil-
ity traces: A survey. The Journal of Machine Learning
Research, 15(1):289–333, 2014.
Gruslys, A., Dabney, W., Azar, M. G., Piot, B., Belle-
mare, M. G., and Munos, R. The Reactor: A fast and
sample-efﬁcient actor-critic agent for reinforcement learn-
ing. ICLR, 2018.
Harutyunyan, A., Bellemare, M. G., Stepleton, T., and
Munos, R. Q(λ) with Off-Policy Corrections, pp. 305–
320. Springer International Publishing, Cham, 2016.
He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings
in deep residual networks. In European Conference on
Computer Vision, pp. 630–645. Springer, 2016.
Hermann, K. M., Hill, F., Green, S., Wang, F., Faulkner, R.,
Soyer, H., Szepesvari, D., Czarnecki, W., Jaderberg, M.,
Teplyashin, D., et al. Grounded language learning in a
simulated 3d world. arXiv preprint arXiv:1706.06551,
2017.
Hochreiter, S. and Schmidhuber, J. Long short-term memory.
Neural computation, 9(8):1735–1780, 1997.
Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel,
M., van Hasselt, H., and Silver, D. Distributed prioritized
experience replay. ICLR, 2018.
Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M.,
Donahue, J., Razavi, A., Vinyals, O., Green, T., Dunning,
I., Simonyan, K., Fernando, C., and Kavukcuoglu, K.
Population based training of neural networks. CoRR,
abs/1711.09846, 2017a.
Jaderberg, M., Mnih, V., Czarnecki, W. M., Schaul, T.,
Leibo, J. Z., Silver, D., and Kavukcuoglu, K. Reinforce-
ment learning with unsupervised auxiliary tasks. ICLR,
2017b.
Leibo, J. Z., d’Autume, C. d. M., Zoran, D., Amos, D.,
Beattie, C., Anderson, K., Casta˜neda, A. G., Sanchez, M.,
Green, S., Gruslys, A., et al. Psychlab: A psychology
laboratory for deep reinforcement learning agents. arXiv
preprint arXiv:1801.08116, 2018.

IMPALA: Importance Weighted Actor-Learner Architectures
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez,
T., Tassa, Y., Silver, D., and Wierstra, D. Continuous
control with deep reinforcement learning. arXiv preprint
arXiv:1509.02971, 2015.
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness,
J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidje-
land, A. K., Ostrovski, G., et al. Human-level control
through deep reinforcement learning. Nature, 518(7540):
529–533, 2015.
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lilli-
crap, T. P., Harley, T., Silver, D., and Kavukcuoglu, K.
Asynchronous methods for deep reinforcement learning.
ICML, 2016.
Munos, R., Stepleton, T., Harutyunyan, A., and Bellemare,
M. Safe and efﬁcient off-policy reinforcement learning.
In Advances in Neural Information Processing Systems,
pp. 1046–1054, 2016.
Nair, A., Srinivasan, P., Blackwell, S., Alcicek, C., Fearon,
R., Maria, A. D., Panneershelvam, V., Suleyman, M.,
Beattie, C., Petersen, S., Legg, S., Mnih, V., Kavukcuoglu,
K., and Silver, D. Massively parallel methods for deep
reinforcement learning. CoRR, abs/1507.04296, 2015.
O’Donoghue, B., Munos, R., Kavukcuoglu, K., and Mnih,
V. Combining policy gradient and Q-learning. In ICLR,
2017.
Precup, D., Sutton, R. S., and Singh, S. Eligibility traces for
off-policy policy evaluation. In Proceedings of the Sev-
enteenth International Conference on Machine Learning,
2000.
Precup, D., Sutton, R. S., and Dasgupta, S. Off-policy
temporal-difference learning with function approxima-
tion. In Proceedings of the 18th International Conference
on Machine Laerning, pp. 417–424, 2001.
Puterman, M. L. Markov Decision Processes: Discrete
Stochastic Dynamic Programming. John Wiley & Sons,
Inc., New York, NY, USA, 1st edition, 1994.
ISBN
0471619779.
Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H.,
Kirkpatrick, J., Kavukcuoglu, K., Pascanu, R., and Had-
sell, R. Progressive neural networks. arXiv preprint
arXiv:1606.04671, 2016.
Salimans, T., Ho, J., Chen, X., and Sutskever, I. Evolu-
tion strategies as a scalable alternative to reinforcement
learning. arXiv preprint arXiv:1703.03864, 2017.
Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel,
P. High-dimensional continuous control using generalized
advantage estimation. In ICLR, 2016.
Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L.,
van den Driessche, G., Schrittwieser, J., Antonoglou, I.,
Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe,
D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T.,
Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis,
D. Mastering the game of go with deep neural networks
and tree search. Nature, 529:484–503, 2016.
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou,
I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M.,
Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L.,
Driessche, G. v. d., Graepel, T., and Hassabis, D. Master-
ing the game of go without human knowledge. Nature,
550(7676):354–359, 10 2017. ISSN 0028-0836. doi:
10.1038/nature24270.
Sutton, R. and Barto, A. Reinforcement learning: An intro-
duction, volume 116. Cambridge Univ Press, 1998.
Wang, Z., Bapst, V., Heess, N., Mnih, V., Munos, R.,
Kavukcuoglu, K., and de Freitas, N. Sample efﬁcient
actor-critic with experience replay. In ICLR, 2017.
Wawrzynski, P. Real-time reinforcement learning by sequen-
tial actor-critics and experience replay. Neural Networks,
22(10):1484–1497, 2009.
Wu, Y., Mansimov, E., Liao, S., Grosse, R. B., and Ba,
J. Scalable trust-region method for deep reinforcement
learning using kronecker-factored approximation. CoRR,
abs/1708.05144, 2017.
Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning
transferable architectures for scalable image recognition.
arXiv preprint arXiv:1707.07012, 2017.

Supplementary Material
A. Analysis of V-trace
A.1. V-trace operator
Deﬁne the V-trace operator R:
RV (x)
def
=
V (x) + Eµ
h X
t≥0
γt c0 . . . ct−1

ρt
 rt + γV (xt+1) −V (xt)
x0 = x, µ
i
,
(5)
where the expectation Eµ is with respect to the policy µ which has generated the trajectory (xt)t≥0, i.e., x0 = x,
xt+1 ∼p(·|xt, at), at ∼µ(·|xt). Here we consider the inﬁnite-horizon operator but very similar results hold for the n-step
truncated operator.
Theorem 1. Let ρt = min
 ¯ρ, π(at|xt)
µ(at|xt)

and ct = min
 ¯c, π(at|xt)
µ(at|xt)

be truncated importance sampling weights, with ¯ρ ≥¯c.
Assume that there exists β ∈(0, 1] such that Eµρ0 ≥β. Then the operator R deﬁned by (5) has a unique ﬁxed point V π¯
ρ,
which is the value function of the policy π¯ρ deﬁned by
π¯ρ(a|x)
def
=
min
 ¯ρµ(a|x), π(a|x)

P
b∈A min
 ¯ρµ(b|x), π(b|x)
,
(6)
Furthermore, R is a η-contraction mapping in sup-norm, with
η
def
= γ−1 −(γ−1 −1)Eµ
 X
t≥0
γt  t−2
Y
i=0
ci

ρt−1

≤1 −(1 −γ)β < 1.
Remark 3. The truncation levels ¯c and ¯ρ play different roles in this operator:
• ¯ρ impacts the ﬁxed-point of the operator, thus the policy π¯ρ which is evaluated. For ¯ρ = ∞(untruncated ρt) we get the
value function of the target policy V π, whereas for ﬁnite ¯ρ, we evaluate a policy which is in between µ and π (and
when ρ is close to 0, then we evaluate V µ). So the larger ¯ρ the smaller the bias in off-policy learning. The variance
naturally grows with ¯ρ. However notice that we do not take the product of those ρt coefﬁcients (in contrast to the cs
coefﬁcients) so the variance does not explode with the time horizon.
• ¯c impacts the contraction modulus η of R (thus the speed at which an online-algorithm like V-trace will converge to its
ﬁxed point V π¯
ρ). In terms of variance reduction, here is it really important to truncate the importance sampling ratios
in ct because we take the product of those. Fortunately, our result says that for any level of truncation ¯c, the ﬁxed point
(the value function V π¯
ρ we converge to) is the same: it does not depend on ¯c but on ¯ρ only.
Proof. First notice that we can rewrite R as
RV (x) = (1 −Eµρ0)V (x) + Eµ

X
t≥0
γt t−1
Y
s=0
cs

ρtrt + γ[ρt −ctρt+1]V (xt+1)


.
Thus
RV1(x) −RV2(x) = (1 −Eµρ0)

V1(x) −V2(x)

+ Eµ

X
t≥0
γt+1 t−1
Y
s=0
cs

[ρt −ctρt+1]

V1(xt+1) −V2(xt+1)


.
= Eµ


X
t≥0
γt t−2
Y
s=0
cs

[ρt−1 −ct−1ρt
|
{z
}
αt
]

V1(xt) −V2(xt)


,

IMPALA: Importance Weighted Actor-Learner Architectures
with the notation that c−1 = ρ−1 = 1 and Qt−2
s=0 cs = 1 for t = 0 and 1. Now the coefﬁcients (αt)t≥0 are non-negative in
expectation. Indeed, since ¯ρ ≥¯c, we have
Eµαt = E

ρt−1 −ct−1ρt

≥Eµ

ct−1(1 −ρt)

≥0,
since Eµρt ≤Eµ
 π(at|xt)
µ(at|xt)

= 1. Thus V1(x) −V2(x) is a linear combination of the values V1 −V2 at other states, weighted
by non-negative coefﬁcients whose sum is
X
t≥0
γtEµ
" t−2
Y
s=0
cs

[ρt−1 −ct−1ρt]
#
=
X
t≥0
γtEµ
" t−2
Y
s=0
cs

ρt−1
#
−
X
t≥0
γtEµ
" t−1
Y
s=0
cs

ρt
#
=
X
t≥0
γtEµ
" t−2
Y
s=0
cs

ρt−1
#
−γ−1

X
t≥0
γtEµ
" t−2
Y
s=0
cs

ρt−1
#
−1


=
γ−1 −(γ−1 −1)
X
t≥0
γtEµ
" t−2
Y
s=0
cs

ρt−1
#
|
{z
}
≥1+γEµρ0
≤
1 −(1 −γ)Eµρ0
≤
1 −(1 −γ)β
<
1.
We deduce that ∥RV1(x) −RV2(x)∥≤η∥V1 −V2∥∞, with η = γ−1 −(γ−1 −1) P
t≥0 γtEµ
h Qt−2
s=0 cs

ρt−1
i
≤
1 −(1 −γ)β < 1, so R is a contraction mapping. Thus R possesses a unique ﬁxed point. Let us now prove that this ﬁxed
point is V π¯
ρ. We have:
Eµ

ρt
 rt + γV π¯
ρ(xt+1) −V π¯
ρ(xt)
xt
i
=
X
a
µ(a|xt) min
 ¯ρ, π(a|xt)
µ(a|xt)
h
r(xt, a) + γ
X
y
p(y|xt, a)V π¯
ρ(y) −V π¯
ρ(xt)
i
=
X
a
π¯ρ(a|xt)
h
r(xt, a) + γ
X
y
p(y|xt, a)V π¯
ρ(y) −V π¯
ρ(xt)
i
|
{z
}
=0
X
b
min
 ¯ρµ(b|xt), π(b|xt)

=
0,
since this is the Bellman equation for V π¯
ρ. We deduce that RV π¯
ρ = V π¯
ρ, thus V π¯
ρ is the unique ﬁxed point of R.
A.2. Online learning
Theorem 2. Assume a tabular representation, i.e. the state and action spaces are ﬁnite. Consider a set of trajectories, with
the kth trajectory x0, a0, r0, x1, a1, r1, . . . generated by following µ: at ∼µ(·|xt). For each state xs along this trajectory,
update
Vk+1(xs) = Vk(xs) + αk(xs)
X
t≥s
γt−s cs . . . ct−1

ρt
 rt + γVk(xt+1) −Vk(xt)

,
(7)
with ci = min
 ¯c, π(ai|xi)
µ(ai|xi)

, ρi = min
 ¯ρ, π(ai|xi)
µ(ai|xi)

, ¯ρ ≥¯c. Assume that (1) all states are visited inﬁnitely often, and (2) the
stepsizes obey the usual Robbins-Munro conditions: for each state x, P
k αk(x) = ∞, P
k α2
k(x) < ∞. Then Vk →V π¯
ρ
almost surely.
The proof is a straightforward application of the convergence result for stochastic approximation algorithms to the ﬁxed
point of a contraction operator, see e.g. Dayan & Sejnowski (1994); Bertsekas & Tsitsiklis (1996); Kushner & Yin (2003).

IMPALA: Importance Weighted Actor-Learner Architectures
A.3. On the choice of qs in policy gradient
The policy gradient update rule (4) makes use of the coefﬁcient qs = rs +γvs+1 as an estimate of Qπ¯
ρ(xs, as) built from the
V-trace estimate vs+1 at the next state xs+1. The reason why we use qs instead of vs as target for our Q-value Qπ¯
ρ(xs, as)
is to make sure our estimate of the Q-value is as unbiased as possible, and the ﬁrst requirement is that it is entirely unbiased
in the case of perfect representation of the V-values. Indeed, assuming our value function is correctly estimated at all states,
i.e. V = V π¯
ρ, then we have E[qs|xs, as] = Qπ¯
ρ(xs, as) (whereas we do not have this property for vt). Indeed,
E[qs|xs, as] = rs + γE

V π¯
ρ(xs+1) + δs+1V π¯
ρ + γcs+1δs+2V π¯
ρ + . . .

= rs + γE

V π¯
ρ(xs+1)

= Qπ¯
ρ(xs, as)
whereas
E[vs|xs, as] = V π¯
ρ(xs) + ρs
 rs + γE

V π¯
ρ(xs+1)

−V π¯
ρ(xs)

+ γcsδs+1V π¯
ρ + . . .
= V π¯
ρ(xs) + ρs
 rs + γE

V π¯
ρ(xs+1)

−V π¯
ρ(xs)

= V π¯
ρ(xs)(1 −ρs) + ρsQπ¯
ρ(xs, as),
which is different from Qπ¯
ρ(xs, as) when V π¯
ρ(xs)̸ = Qπ¯
ρ(xs, as).

IMPALA: Importance Weighted Actor-Learner Architectures
B. Reference Scores
Task t
Human h Random r Experts IMPALA
rooms collect good objects test
10.0
0.1
9.0
5.8
rooms exploit deferred effects test
85.7
8.5
15.6
11.0
rooms select nonmatching object
65.9
0.3
7.3
26.1
rooms watermaze
54.0
4.1
26.9
31.1
rooms keys doors puzzle
53.8
4.1
28.0
24.3
language select described object
389.5
-0.1
324.6
593.1
language select located object
280.7
1.9
189.0
301.7
language execute random task
254.1
-5.9
-49.9
66.8
language answer quantitative question
184.5
-0.3
219.4
264.0
lasertag one opponent large
12.7
-0.2
-0.2
0.3
lasertag three oponents large
18.6
-0.2
-0.1
4.1
lasertag one opponent small
18.6
-0.1
-0.1
2.5
lasertag three opponents small
31.5
-0.1
19.1
11.3
natlab fixed large map
36.9
2.2
34.7
12.2
natlab varying map regrowth
24.4
3.0
20.7
15.9
natlab varying map randomized
42.4
7.3
36.1
29.0
skymaze irreversible path hard
100.0
0.1
13.6
30.0
skymaze irreversible path varied
100.0
14.4
45.1
53.6
pyschlab arbitrary visuomotor mapping
58.8
0.2
16.4
14.3
pyschlab continuous recognition
58.3
0.2
29.9
29.9
pyschlab sequential comparison
39.5
0.1
0.0
0.0
pyschlab visual search
78.5
0.1
0.0
0.0
explore object locations small
74.5
3.6
57.8
62.6
explore object locations large
65.7
4.7
37.0
51.1
explore obstructed goals small
206.0
6.8
135.2
188.8
explore obstructed goals large
119.5
2.6
39.5
71.0
explore goal locations small
267.5
7.7
209.4
252.5
explore goal locations large
194.5
3.1
83.1
125.3
explore object rewards few
77.7
2.1
39.8
43.2
explore object rewards many
106.7
2.4
58.7
62.6
Mean Capped Normalised Score: (P
t min [1, (st −rt)/(ht −rt)]) /N
100%
0%
44.5%
49.4%
Table B.1. DMLab-30 test scores.

IMPALA: Importance Weighted Actor-Learner Architectures
B.1. Final training scores on DMLab-30
0
20
40
60
80
100
120
140
160
Human Normalised Score
pyschlab_sequential_comparison
pyschlab_visual_search
lasertag_one_opponent_large
lasertag_one_opponent_small
language_execute_random_task
lasertag_three_oponents_large
rooms_exploit_deferred_effects_train
pyschlab_arbitrary_visuomotor_mapping
skymaze_irreversible_path_hard
natlab_fixed_large_map
rooms_keys_doors_puzzle
skymaze_irreversible_path_varied
lasertag_three_opponents_small
pyschlab_continuous_recognition
explore_object_rewards_few
rooms_select_nonmatching_object
rooms_watermaze
explore_object_rewards_many
explore_obstructed_goals_large
explore_goal_locations_large
natlab_varying_map_regrowth
natlab_varying_map_randomized
explore_object_locations_large
explore_object_locations_small
explore_obstructed_goals_small
rooms_collect_good_objects_train
explore_goal_locations_small
language_select_located_object
language_answer_quantitative_question
language_select_described_object
A3C, deep
IMPALA-Experts, deep
IMPALA, deep, PBT
Figure B.1. Human normalised scores across all DMLab-30 tasks.

IMPALA: Importance Weighted Actor-Learner Architectures
C. Atari Scores
ACKTR
The Reactor
IMPALA (deep, multi-task)
IMPALA (shallow)
IMPALA (deep)
alien
3197.10
6482.10
2344.60
1536.05
15962.10
amidar
1059.40
833
136.82
497.62
1554.79
assault
10777.70
11013.50
2116.32
12086.86
19148.47
asterix
31583.00
36238.50
2609.00
29692.50
300732.00
asteroids
34171.60
2780.40
2011.05
3508.10
108590.05
atlantis
3433182.00
308258
460430.50
773355.50
849967.50
bank heist
1289.70
988.70
55.15
1200.35
1223.15
battle zone
8910.00
61220
7705.00
13015.00
20885.00
beam rider
13581.40
8566.50
698.36
8219.92
32463.47
berzerk
927.20
1641.40
647.80
888.30
1852.70
bowling
24.30
75.40
31.06
35.73
59.92
boxing
1.45
99.40
96.63
96.30
99.96
breakout
735.70
518.40
35.67
640.43
787.34
centipede
7125.28
3402.80
4916.84
5528.13
11049.75
chopper command
N/A
37568
5036.00
5012.00
28255.00
crazy climber
150444.00
194347
115384.00
136211.50
136950.00
defender
N/A
113128
16667.50
58718.25
185203.00
demon attack
274176.70
100189
10095.20
107264.73
132826.98
double dunk
-0.54
11.40
-1.92
-0.35
-0.33
enduro
0.00
2230.10
971.28
0.00
0.00
ﬁshing derby
33.73
23.20
35.27
32.08
44.85
freeway
0.00
31.40
21.41
0.00
0.00
frostbite
N/A
8042.10
2744.15
269.65
317.75
gopher
47730.80
69135.10
913.50
1002.40
66782.30
gravitar
N/A
1073.80
282.50
211.50
359.50
hero
N/A
35542.20
18818.90
33853.15
33730.55
ice hockey
-4.20
3.40
-13.55
-5.25
3.48
jamesbond
490.00
7869.20
284.00
440.00
601.50
kangaroo
3150.00
10484.50
8240.50
47.00
1632.00
krull
9686.90
9930.80
10807.80
9247.60
8147.40
kung fu master
34954.00
59799.50
41905.00
42259.00
43375.50
montezuma revenge
N/A
2643.50
0.00
0.00
0.00
ms pacman
N/A
2724.30
3415.05
6501.71
7342.32
name this game
N/A
9907.20
5719.30
6049.55
21537.20
phoenix
133433.70
40092.20
7486.50
33068.15
210996.45
pitfall
-1.10
-3.50
-1.22
-11.14
-1.66
pong
20.90
20.70
8.58
20.40
20.98
private eye
N/A
15177.10
0.00
92.42
98.50
qbert
23151.50
22956.50
10717.38
18901.25
351200.12
riverraid
17762.80
16608.30
2850.15
17401.90
29608.05
road runner
53446.00
71168
24435.50
37505.00
57121.00
robotank
16.50
68.50
9.94
2.30
12.96
seaquest
1776.00
8425.80
844.60
1716.90
1753.20
skiing
N/A
-10753.40
-8988.00
-29975.00
-10180.38
solaris
2368.60
2760
1160.40
2368.40
2365.00
space invaders
19723.00
2448.60
199.65
1726.28
43595.78
star gunner
82920.00
70038
1855.50
69139.00
200625.00
surround
N/A
6.70
-8.51
-8.13
7.56
tennis
N/A
23.30
-8.12
-1.89
0.55
time pilot
22286.00
19401
3747.50
6617.50
48481.50
tutankham
314.30
272.60
105.22
267.82
292.11
up n down
436665.80
64354.20
82155.30
273058.10
332546.75
venture
N/A
1597.50
1.00
0.00
0.00
video pinball
100496.60
469366
20125.14
228642.52
572898.27
wizard of wor
702.00
13170.50
2106.00
4203.00
9157.50
yars revenge
125169.00
102760
14739.41
80530.13
84231.14
zaxxon
17448.00
25215.50
6497.00
1148.50
32935.50
Table C.1. Atari scores after 200M steps environment steps of training. Up to 30 no-ops at the beginning of each episode.

IMPALA: Importance Weighted Actor-Learner Architectures
D. Parameters
In this section, the speciﬁc parameter settings that are used throughout our experiments are given in detail.
Hyperparameter
Range
Distribution
Entropy regularisation
[5e-5, 1e-2]
Log uniform
Learning rate
[5e-6, 5e-3]
Log uniform
RMSProp epsilon (ε) regularisation parameter [1e-1, 1e-3, 1e-5, 1e-7] Categorical
Table D.1. The ranges used in sampling hyperparameters across all experiments that used a sweep and for the initial hyperparameters for
PBT. Sweep size and population size are 24. Note, the loss is summed across the batch and time dimensions.
Action
Native DeepMind Lab Action
Forward
[
0, 0,
0,
1, 0, 0, 0]
Backward
[
0, 0,
0, -1, 0, 0, 0]
Strafe Left
[
0, 0, -1,
0, 0, 0, 0]
Strafe Right
[
0, 0,
1,
0, 0, 0, 0]
Look Left
[-20, 0,
0,
0, 0, 0, 0]
Look Right
[ 20, 0,
0,
0, 0, 0, 0]
Forward + Look Left
[-20, 0,
0,
1, 0, 0, 0]
Forward + Look Right [ 20, 0,
0,
1, 0, 0, 0]
Fire
[
0, 0,
0,
0, 1, 0, 0]
Table D.2. Action set used in all tasks from the DeepMind Lab environment, including the DMLab-30 experiments.
D.1. Fixed Model Hyperparameters
In this section, we list all the hyperparameters that were kept ﬁxed across all experiments in the paper which are mostly
concerned with observations speciﬁcations and optimisation. We ﬁrst show below the reward pre-processing function that is
used across all experiments using DeepMind Lab, followed by all ﬁxed numerical values.
−10
−5
0
5
10
Reward
−1
0
1
2
3
4
5
Clipped Reward
Figure D.1. Optimistic Asymmetric Clipping - 0.3 · min(tanh(reward), 0) + 5.0 · max(tanh(reward), 0)

IMPALA: Importance Weighted Actor-Learner Architectures
Parameter
Value
Image Width
96
Image Height
72
Action Repetitions
4
Unroll Length (n)
100
Reward Clipping
- Single tasks
[-1, 1]
- DMLab-30, including experts
See Figure D.1
Discount (γ)
0.99
Baseline loss scaling
0.5
RMSProp momentum
0.0
Experience Replay (in Section 5.2.2 )
- Capacity
10,000 trajectories
- Sampling
Uniform
- Removal
First-in-ﬁrst-out
Table D.3. Fixed model hyperparameters across all DeepMind Lab experiments.

IMPALA: Importance Weighted Actor-Learner Architectures
E. V-trace Analysis
E.1. Controlled Updates
Here we show how different algorithms (On-Policy, No-correction, ε-correction, V-trace) behave under varying levels of
policy-lag between the actors and the learner.
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
10
20
30
40
50
60
Return
0
1
10
100
500
ϵ-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
No-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
V-trace
rooms_watermaze
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
5
10
15
20
25
30
Return
0
1
10
100
500
ϵ-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
No-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
1
0
10
100
500
V-trace
rooms_keys_doors_puzzle
0.2
0.4
0.6
0.8
Environment Frames
1e9
5
0
5
10
15
20
25
30
35
Return
0
1
10
100
500
ϵ-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
No-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
10
1
100
0
500
V-trace
lasertag_three_opponents_small
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
50
100
150
200
250
Return
0
1
100
10
500
ϵ-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
No-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
10
1
0
100
500
V-trace
explore_goal_locations_small
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
5
10
15
20
25
30
35
40
45
Return
0
1
10
100
500
ϵ-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
No-correction
0.2
0.4
0.6
0.8
Environment Frames
1e9
0
1
10
100
500
V-trace
seekavoid_arena_01
Figure E.1. As the policy-lag (the number of update steps the actor policy is behind learner policy) increases, learning with V-trace is
more robust compared to ε-correction and pure on-policy learning.

IMPALA: Importance Weighted Actor-Learner Architectures
E.2. V-trace Stability Analysis
1
5
9
13
17
21
24
Hyperparameter Combination
0
10
20
30
40
50
60
Final Return
rooms_watermaze
1
5
9
13
17
21
24
Hyperparameter Combination
0
5
10
15
20
25
30
35
40
rooms_keys_doors_puzzle
1
5
9
13
17
21
24
Hyperparameter Combination
−5
0
5
10
15
20
25
30
35
lasertag_three_opponents_small
1
5
9
13
17
21
24
Hyperparameter Combination
0
50
100
150
200
250
300
explore_goal_locations_small
1
5
9
13
17
21
24
Hyperparameter Combination
0
10
20
30
40
50
seekavoid_arena_01
V−trace − min(ρ, 1)
1 Step Importance Sampling − min(ρ, 1)
ε−correction
No-correction
Figure E.2. Stability across hyper parameter combinations for different off-policy correction variants using replay. V-trace is much more
stable across a wide range of parameter combinations compared to ε-correction and pure on-policy learning.
E.3. Estimating the State Action Value for Policy Gradient
We investigated different ways of estimating the state action value function used to estimate advantages for the policy
gradient calculation. The variant presented in the main section of the paper uses the V-trace corrected value function vs+1 to
estimate qs = rs + γvs+1. Another possibility is to use the actor-critic baseline V (xs+1) to estimate qs = rs + γV (xs+1).
Note that the latter variant does not use any information from the current policy rollout to estimate the policy gradient and
relies on an accurate estimate of the value function. We found the latter variant to perform worse both when comparing the
top 3 runs and an average over all runs of the hyperparameter sweep as can be see in ﬁgures E.3 and E.4.
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
5
10
15
20
25
30
35
40
45
50
Return
rooms_watermaze
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
5
10
15
20
25
30
rooms_keys_doors_puzzle
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
−5
0
5
10
15
20
25
30
35
lasertag_three_opponents_small
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
0
50
100
150
200
250
explore_goal_locations_small
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
5
10
15
20
25
30
35
40
45
seekavoid_arena_01
qs =rs +γ⋅vs+1
qs =rs +γ⋅V(xs+1)
Figure E.3. Variants for estimation of state action value function - average over top 3 runs.
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
5
10
15
20
25
30
35
Return
rooms_watermaze
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
4
6
8
10
12
14
16
18
20
22
rooms_keys_doors_puzzle
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
−2
0
2
4
6
8
10
12
lasertag_three_opponents_small
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
0
20
40
60
80
100
120
140
160
explore_goal_locations_small
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e9
0
5
10
15
20
25
30
35
seekavoid_arena_01
qs =rs +γ⋅vs+1
qs =rs +γ⋅V(xs+1)
Figure E.4. Variants for estimation of state action value function - average over all runs.
F. Population Based Training
For Population Based Training we used a “burn-in” period of 20 million frames where no evolution is done. This is to
stabilise the process and to avoid very rapid initial adaptation which hinders diversity. After collecting 5,000 episode rewards
in total, the mean capped human normalised score is calculated and a random instance in the population is selected. If the
score of the selected instance is more than an absolute 5% higher, then the selected instance weights and parameters are
copied.
No matter if a copy happened or not, each parameter (RMSProp epsilon, learning rate and entropy cost) is permuted
with 33% probability by multiplying with either 1.2 or 1/1.2. This is different from Jaderberg et al. (2017) in that our
multiplication is unbiased where they use a multiplication of 1.2 or .8. We found that diversity is increased when the
parameters are permuted even if no copy happened.
We reconstruct the learning curves of the PBT runs in Figure 5 by backtracking through the ancestry of copied checkpoints
for selected instances.

IMPALA: Importance Weighted Actor-Learner Architectures
0.0
0.2
0.4
0.6
0.8
1.0
Environment Frames
1e10
0.0000
0.0001
0.0002
0.0003
0.0004
0.0005
0.0006
0.0007
Learning Rate
IMPALA - PBT - 8 GPUs
IMPALA - PBT - 1 GPU
Figure F.1. Learning rate schedule that is discovered by the PBT Jaderberg et al. (2017) method compared against the linear annealing
schedule of the best run from the parameter sweep (red line).
G. Atari Experiments
All agents trained on Atari are equipped only with a feed forward network and pre-process frames in the same way as
described in Mnih et al. (2016). When training experts agents, we use the same hyperparameters for each game for
both IMPALA and A3C. These hyperparameters are the result of tuning A3C with a shallow network on the following
games: breakout, pong, space invaders, seaquest, beam rider, qbert. Following related work, experts
use game-speciﬁc action sets.
The multi-task agent was equipped with a feed forward residual network (see Figure 3 ). The learning rate, entropy
regularisation, RMSProp ε and gradient clipping threshold were adapted through population based training. To be able to
use the same policy layer on all Atari games in the multi-task setting we train the multi-task agent on the full Atari action set
consisting of 18 actions.
Agents were trained using the following set of hyperparameters:

IMPALA: Importance Weighted Actor-Learner Architectures
Parameter
Value
Image Width
84
Image Height
84
Grayscaling
Yes
Action Repetitions
4
Max-pool over last N action repeat frames
2
Frame Stacking
4
End of episode when life lost
Yes
Reward Clipping
[-1, 1]
Unroll Length (n)
20
Batch size
32
Discount (γ)
0.99
Baseline loss scaling
0.5
Entropy Regularizer
0.01
RMSProp momentum
0.0
RMSProp ε
0.01
Learning rate
0.0006
Clip global gradient norm
40.0
Learning rate schedule
Anneal linearly to 0
From beginning to end of training.
Population based training (only multi-task agent)
- Population size
24
- Start parameters
Same as DMLab-30 sweep
- Fitness
Mean capped human normalised scores
(P
l min [1, (st −rt)/(ht −rt)]) /N
- Adapted parameters
Gradient clipping threshold
Entropy regularisation
Learning rate
RMSProp ε
Table G.1. Hyperparameters for Atari experiments.
References
Bertsekas, D. P. and Tsitsiklis, J. N. Neuro-Dynamic Programming. Athena Scientiﬁc, 1996.
Dayan, P. and Sejnowski, T. J. TD(λ) converges with probability 1. Machine Learning, 14(1):295–301, 1994. doi:
10.1023/A:1022657612745.
Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., Vinyals, O., Green, T., Dunning, I.,
Simonyan, K., Fernando, C., and Kavukcuoglu, K. Population based training of neural networks. CoRR, abs/1711.09846,
2017.
Kushner, H. and Yin, G. Stochastic Approximation and Recursive Algorithms and Applications. Stochastic Modelling and
Applied Probability. Springer New York, 2003. ISBN 9780387008943.
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous
methods for deep reinforcement learning. ICML, 2016.
