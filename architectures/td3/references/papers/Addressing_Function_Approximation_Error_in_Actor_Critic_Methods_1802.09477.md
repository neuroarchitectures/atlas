# Paper (Fujimoto et al. 2018)

> Source: `https://arxiv.org/abs/1802.09477`

---

Addressing Function Approximation Error in Actor-Critic Methods
Scott Fujimoto 1 Herke van Hoof 2 David Meger 1
Abstract
In value-based reinforcement learning methods
such as deep Q-learning, function approximation
errors are known to lead to overestimated value
estimates and suboptimal policies. We show that
this problem persists in an actor-critic setting and
propose novel mechanisms to minimize its effects
on both the actor and the critic. Our algorithm
builds on Double Q-learning, by taking the mini-
mum value between a pair of critics to limit over-
estimation. We draw the connection between tar-
get networks and overestimation bias, and suggest
delaying policy updates to reduce per-update error
and further improve performance. We evaluate
our method on the suite of OpenAI gym tasks,
outperforming the state of the art in every envi-
ronment tested.
1. Introduction
In reinforcement learning problems with discrete action
spaces, the issue of value overestimation as a result of func-
tion approximation errors is well-studied. However, similar
issues with actor-critic methods in continuous control do-
mains have been largely left untouched. In this paper, we
show overestimation bias and the accumulation of error in
temporal difference methods are present in an actor-critic
setting. Our proposed method addresses these issues, and
greatly outperforms the current state of the art.
Overestimation bias is a property of Q-learning in which the
maximization of a noisy value estimate induces a consistent
overestimation (Thrun & Schwartz, 1993). In a function
approximation setting, this noise is unavoidable given the
imprecision of the estimator. This inaccuracy is further
exaggerated by the nature of temporal difference learning
(Sutton, 1988), in which an estimate of the value function
is updated using the estimate of a subsequent state. This
1McGill University, Montreal, Canada 2University of Amster-
dam, Amsterdam, Netherlands. Correspondence to: Scott Fujimoto
<scott.fujimoto@mail.mcgill.ca>.
Proceedings of the 35 th International Conference on Machine
Learning, Stockholm, Sweden, PMLR 80, 2018. Copyright 2018
by the author(s).
means using an imprecise estimate within each update will
lead to an accumulation of error. Due to overestimation bias,
this accumulated error can cause arbitrarily bad states to
be estimated as high value, resulting in suboptimal policy
updates and divergent behavior.
This paper begins by establishing this overestimation prop-
erty is also present for deterministic policy gradients (Silver
et al., 2014), in the continuous control setting. Furthermore,
we ﬁnd the ubiquitous solution in the discrete action setting,
Double DQN (Van Hasselt et al., 2016), to be ineffective
in an actor-critic setting. During training, Double DQN
estimates the value of the current policy with a separate tar-
get value function, allowing actions to be evaluated without
maximization bias. Unfortunately, due to the slow-changing
policy in an actor-critic setting, the current and target value
estimates remain too similar to avoid maximization bias.
This can be dealt with by adapting an older variant, Double
Q-learning (Van Hasselt, 2010), to an actor-critic format
by using a pair of independently trained critics. While this
allows for a less biased value estimation, even an unbiased
estimate with high variance can still lead to future overes-
timations in local regions of state space, which in turn can
negatively affect the global policy. To address this concern,
we propose a clipped Double Q-learning variant which lever-
ages the notion that a value estimate suffering from overes-
timation bias can be used as an approximate upper-bound to
the true value estimate. This favors underestimations, which
do not tend to be propagated during learning, as actions with
low value estimates are avoided by the policy.
Given the connection of noise to overestimation bias, this
paper contains a number of components that address vari-
ance reduction. First, we show that target networks, a com-
mon approach in deep Q-learning methods, are critical for
variance reduction by reducing the accumulation of errors.
Second, to address the coupling of value and policy, we
propose delaying policy updates until the value estimate
has converged. Finally, we introduce a novel regularization
strategy, where a SARSA-style update bootstraps similar
action estimates to further reduce variance.
Our modiﬁcations are applied to the state of the art actor-
critic method for continuous control, Deep Deterministic
Policy Gradient algorithm (DDPG) (Lillicrap et al., 2015), to
form the Twin Delayed Deep Deterministic policy gradient
arXiv:1802.09477v3  [cs.AI]  22 Oct 2018

Addressing Function Approximation Error in Actor-Critic Methods
algorithm (TD3), an actor-critic algorithm which consid-
ers the interplay between function approximation error in
both policy and value updates. We evaluate our algorithm
on seven continuous control domains from OpenAI gym
(Brockman et al., 2016), where we outperform the state of
the art by a wide margin.
Given the recent concerns in reproducibility (Henderson
et al., 2017), we run our experiments across a large num-
ber of seeds with fair evaluation metrics, perform abla-
tion studies across each contribution, and open source both
our code and learning curves (https://github.com/
sfujim/TD3).
2. Related Work
Function approximation error and its effect on bias and
variance in reinforcement learning algorithms have been
studied in prior works (Pendrith et al., 1997; Mannor et al.,
2007). Our work focuses on two outcomes that occur as the
result of estimation error, namely overestimation bias and a
high variance build-up.
Several approaches exist to reduce the effects of overestima-
tion bias due to function approximation and policy optimiza-
tion in Q-learning. Double Q-learning uses two independent
estimators to make unbiased value estimates (Van Hasselt,
2010; Van Hasselt et al., 2016). Other approaches have
focused directly on reducing the variance (Anschel et al.,
2017), minimizing over-ﬁtting to early high variance esti-
mates (Fox et al., 2016), or through corrective terms (Lee
et al., 2013). Further, the variance of the value estimate
has been considered directly for risk-aversion (Mannor &
Tsitsiklis, 2011) and exploration (O’Donoghue et al., 2017),
but without connection to overestimation bias.
The concern of variance due to the accumulation of error in
temporal difference learning has been largely dealt with by
either minimizing the size of errors at each time step or mix-
ing off-policy and Monte-Carlo returns. Our work shows
the importance of a standard technique, target networks, for
the reduction of per-update error, and develops a regulariza-
tion technique for the variance reduction by averaging over
value estimates. Concurrently, Nachum et al. (2018) showed
smoothed value functions could be used to train stochastic
policies with reduced variance and improved performance.
Methods with multi-step returns offer a trade-off between
accumulated estimation bias and variance induced by the
policy and the environment. These methods have been
shown to be an effective approach, through importance sam-
pling (Precup et al., 2001; Munos et al., 2016), distributed
methods (Mnih et al., 2016; Espeholt et al., 2018), and ap-
proximate bounds (He et al., 2016). However, rather than
provide a direct solution to the accumulation of error, these
methods circumvent the problem by considering a longer
horizon. Another approach is a reduction in the discount
factor (Petrik & Scherrer, 2009), reducing the contribution
of each error.
Our method builds on the Deterministic Policy Gradient
algorithm (DPG) (Silver et al., 2014), an actor-critic method
which uses a learned value estimate to train a deterministic
policy. An extension of DPG to deep reinforcement learn-
ing, DDPG (Lillicrap et al., 2015), has shown to produce
state of the art results with an efﬁcient number of iterations.
Orthogonal to our approach, recent improvements to DDPG
include distributed methods (Popov et al., 2017), along with
multi-step returns and prioritized experience replay (Schaul
et al., 2016; Horgan et al., 2018), and distributional methods
(Bellemare et al., 2017; Barth-Maron et al., 2018).
3. Background
Reinforcement learning considers the paradigm of an agent
interacting with its environment with the aim of learning
reward-maximizing behavior. At each discrete time step
t, with a given state s ∈S, the agent selects actions
a ∈A with respect to its policy π : S →A, receiv-
ing a reward r and the new state of the environment s′.
The return is deﬁned as the discounted sum of rewards
Rt = PT
i=t γi−tr(si, ai), where γ is a discount factor de-
termining the priority of short-term rewards.
In reinforcement learning, the objective is to ﬁnd the op-
timal policy πφ, with parameters φ, which maximizes the
expected return J(φ) = Esi∼pπ,ai∼π [R0]. For continuous
control, parametrized policies πφ can be updated by taking
the gradient of the expected return ∇φJ(φ). In actor-critic
methods, the policy, known as the actor, can be updated
through the deterministic policy gradient algorithm (Silver
et al., 2014):
∇φJ(φ) = Es∼pπ

∇aQπ(s, a)|a=π(s)∇φπφ(s)

.
(1)
Qπ(s, a) = Esi∼pπ,ai∼π [Rt|s, a], the expected return
when performing action a in state s and following π af-
ter, is known as the critic or the value function.
In Q-learning, the value function can be learned using tem-
poral difference learning (Sutton, 1988; Watkins, 1989), an
update rule based on the Bellman equation (Bellman, 1957).
The Bellman equation is a fundamental relationship between
the value of a state-action pair (s, a) and the value of the
subsequent state-action pair (s′, a′):
Qπ(s, a) = r + γEs′,a′ [Qπ(s′, a′)] ,
a′ ∼π(s′).
(2)
For a large state space, the value can be estimated with a
differentiable function approximator Qθ(s, a), with param-
eters θ. In deep Q-learning (Mnih et al., 2015), the network
is updated by using temporal difference learning with a sec-
ondary frozen target network Qθ′(s, a) to maintain a ﬁxed

Addressing Function Approximation Error in Actor-Critic Methods
objective y over multiple updates:
y = r + γQθ′(s′, a′),
a′ ∼πφ′(s′),
(3)
where the actions are selected from a target actor network
πφ′. The weights of a target network are either updated
periodically to exactly match the weights of the current
network, or by some proportion τ at each time step θ′ ←
τθ + (1 −τ)θ′. This update can be applied in an off-policy
fashion, sampling random mini-batches of transitions from
an experience replay buffer (Lin, 1992).
4. Overestimation Bias
In Q-learning with discrete actions, the value estimate is
updated with a greedy target y = r + γ maxa′ Q(s′, a′),
however, if the target is susceptible to error ϵ, then the max-
imum over the value along with its error will generally be
greater than the true maximum, Eϵ[maxa′(Q(s′, a′)+ϵ)] ≥
maxa′ Q(s′, a′) (Thrun & Schwartz, 1993). As a result,
even initially zero-mean error can cause value updates to
result in a consistent overestimation bias, which is then prop-
agated through the Bellman equation. This is problematic as
errors induced by function approximation are unavoidable.
While in the discrete action setting overestimation bias is
an obvious artifact from the analytical maximization, the
presence and effects of overestimation bias is less clear in an
actor-critic setting where the policy is updated via gradient
descent. We begin by proving that the value estimate in de-
terministic policy gradients will be an overestimation under
some basic assumptions in Section 4.1 and then propose
a clipped variant of Double Q-learning in an actor-critic
setting to reduce overestimation bias in Section 4.2.
4.1. Overestimation Bias in Actor-Critic
In actor-critic methods the policy is updated with respect
to the value estimates of an approximate critic. In this
section we assume the policy is updated using the deter-
ministic policy gradient, and show that the update induces
overestimation in the value estimate. Given current policy
parameters φ, let φapprox deﬁne the parameters from the ac-
tor update induced by the maximization of the approximate
critic Qθ(s, a) and φtrue the parameters from the hypothet-
ical actor update with respect to the true underlying value
function Qπ(s, a) (which is not known during learning):
φapprox = φ + α
Z1
Es∼pπ

∇φπφ(s)∇aQθ(s, a)|a=πφ(s)

φtrue = φ + α
Z2
Es∼pπ

∇φπφ(s)∇aQπ(s, a)|a=πφ(s)

,
(4)
where we assume Z1 and Z2 are chosen to normalize the
gradient, i.e., such that Z−1||E[·]|| = 1. Without normal-
ized gradients, overestimation bias is still guaranteed to
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
100
200
300
400
Average Value
CDQ
DDPG
True CDQ
True DDPG
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
100
200
300
400
500
(a) Hopper-v1
(b) Walker2d-v1
Figure 1. Measuring overestimation bias in the value estimates
of DDPG and our proposed method, Clipped Double Q-learning
(CDQ), on MuJoCo environments over 1 million time steps.
occur with slightly stricter conditions. We examine this case
further in the supplementary material. We denote πapprox
and πtrue as the policy with parameters φapprox and φtrue re-
spectively.
As the gradient direction is a local maximizer, there exists ϵ1
sufﬁciently small such that if α ≤ϵ1 then the approximate
value of πapprox will be bounded below by the approximate
value of πtrue:
E [Qθ(s, πapprox(s))] ≥E [Qθ(s, πtrue(s))] .
(5)
Conversely, there exists ϵ2 sufﬁciently small such that if
α ≤ϵ2 then the true value of πapprox will be bounded above
by the true value of πtrue:
E [Qπ(s, πtrue(s))] ≥E [Qπ(s, πapprox(s))] .
(6)
If in expectation the value estimate is at least as large as
the true value with respect to φtrue, E [Qθ (s, πtrue(s))] ≥
E [Qπ (s, πtrue(s))], then Equations (5) and (6) imply that if
α < min(ϵ1, ϵ2), then the value estimate will be overesti-
mated:
E [Qθ(s, πapprox(s))] ≥E [Qπ(s, πapprox(s))] .
(7)
Although this overestimation may be minimal with each
update, the presence of error raises two concerns. Firstly, the
overestimation may develop into a more signiﬁcant bias over
many updates if left unchecked. Secondly, an inaccurate
value estimate may lead to poor policy updates. This is
particularly problematic because a feedback loop is created,
in which suboptimal actions might be highly rated by the
suboptimal critic, reinforcing the suboptimal action in the
next policy update.
Does this theoretical overestimation occur in practice
for state-of-the-art methods? We answer this question by
plotting the value estimate of DDPG (Lillicrap et al., 2015)
over time while it learns on the OpenAI gym environments
Hopper-v1 and Walker2d-v1 (Brockman et al., 2016). In
Figure 1, we graph the average value estimate over 10000
states and compare it to an estimate of the true value. The

Addressing Function Approximation Error in Actor-Critic Methods
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
100
200
300
400
Average Value
DQ-AC
DDQN-AC
True DQ-AC
True DDQN-AC
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
100
200
300
400
(a) Hopper-v1
(b) Walker2d-v1
Figure 2. Measuring overestimation bias in the value estimates of
actor critic variants of Double DQN (DDQN-AC) and Double Q-
learning (DQ-AC) on MuJoCo environments over 1 million time
steps.
true value is estimated using the average discounted return
over 1000 episodes following the current policy, starting
from states sampled from the replay buffer. A very clear
overestimation bias occurs from the learning procedure,
which contrasts with the novel method that we describe in
the following section, Clipped Double Q-learning, which
greatly reduces overestimation by the critic.
4.2. Clipped Double Q-Learning for Actor-Critic
While several approaches to reducing overestimation bias
have been proposed, we ﬁnd them ineffective in an actor-
critic setting. This section introduces a novel clipped variant
of Double Q-learning (Van Hasselt, 2010), which can re-
place the critic in any actor-critic method.
In Double Q-learning, the greedy update is disentangled
from the value function by maintaining two separate value
estimates, each of which is used to update the other. If the
value estimates are independent, they can be used to make
unbiased estimates of the actions selected using the opposite
value estimate. In Double DQN (Van Hasselt et al., 2016),
the authors propose using the target network as one of the
value estimates, and obtain a policy by greedy maximization
of the current value network rather than the target network.
In an actor-critic setting, an analogous update uses the cur-
rent policy rather than the target policy in the learning target:
y = r + γQθ′(s′, πφ(s′)).
(8)
In practice however, we found that with the slow-changing
policy in actor-critic, the current and target networks were
too similar to make an independent estimation, and offered
little improvement. Instead, the original Double Q-learning
formulation can be used, with a pair of actors (πφ1, πφ2)
and critics (Qθ1, Qθ2), where πφ1 is optimized with respect
to Qθ1 and πφ2 with respect to Qθ2:
y1 = r + γQθ′
2(s′, πφ1(s′))
y2 = r + γQθ′
1(s′, πφ2(s′)).
(9)
We measure the overestimation bias in Figure 2, which
demonstrates that the actor-critic Double DQN suffers from
a similar overestimation as DDPG (as shown in Figure 1).
While Double Q-learning is more effective, it does not en-
tirely eliminate the overestimation. We further show this
reduction is not sufﬁcient experimentally in Section 6.1.
As πφ1 optimizes with respect to Qθ1, using an indepen-
dent estimate in the target update of Qθ1 would avoid the
bias introduced by the policy update. However the critics
are not entirely independent, due to the use of the oppo-
site critic in the learning targets, as well as the same re-
play buffer. As a result, for some states s we will have
Qθ2(s, πφ1(s)) > Qθ1(s, πφ1(s)). This is problematic be-
cause Qθ1(s, πφ1(s)) will generally overestimate the true
value, and in certain areas of the state space the overestima-
tion will be further exaggerated. To address this problem,
we propose to simply upper-bound the less biased value
estimate Qθ2 by the biased estimate Qθ1. This results in
taking the minimum between the two estimates, to give the
target update of our Clipped Double Q-learning algorithm:
y1 = r + γ min
i=1,2 Qθ′
i(s′, πφ1(s′)).
(10)
With Clipped Double Q-learning, the value target cannot
introduce any additional overestimation over using the stan-
dard Q-learning target. While this update rule may induce
an underestimation bias, this is far preferable to overesti-
mation bias, as unlike overestimated actions, the value of
underestimated actions will not be explicitly propagated
through the policy update.
In implementation, computational costs can be reduced by
using a single actor optimized with respect to Qθ1. We then
use the same target y2 = y1 for Qθ2. If Qθ2 > Qθ1 then
the update is identical to the standard update and induces no
additional bias. If Qθ2 < Qθ1, this suggests overestimation
has occurred and the value is reduced similar to Double Q-
learning. A proof of convergence in the ﬁnite MDP setting
follows from this intuition. We provide formal details and
justiﬁcation in the supplementary material.
A secondary beneﬁt is that by treating the function approxi-
mation error as a random variable we can see that the min-
imum operator should provide higher value to states with
lower variance estimation error, as the expected minimum
of a set of random variables decreases as the variance of
the random variables increases. This effect means that the
minimization in Equation (10) will lead to a preference for
states with low-variance value estimates, leading to safer
policy updates with stable learning targets.
5. Addressing Variance
While Section 4 deals with the contribution of variance to
overestimation bias, we also argue that variance itself should
be directly addressed. Besides the impact on overestimation

Addressing Function Approximation Error in Actor-Critic Methods
bias, high variance estimates provide a noisy gradient for the
policy update. This is known to reduce learning speed (Sut-
ton & Barto, 1998) as well as hurt performance in practice.
In this section we emphasize the importance of minimizing
error at each update, build the connection between target
networks and estimation error and propose modiﬁcations to
the learning procedure of actor-critic for variance reduction.
5.1. Accumulating Error
Due to the temporal difference update, where an estimate of
the value function is built from an estimate of a subsequent
state, there is a build up of error. While it is reasonable to
expect small error for an individual update, these estimation
errors can accumulate, resulting in the potential for large
overestimation bias and suboptimal policy updates. This is
exacerbated in a function approximation setting where the
Bellman equation is never exactly satisﬁed, and each update
leaves some amount of residual TD-error δ(s, a):
Qθ(s, a) = r + γE[Qθ(s′, a′)] −δ(s, a).
(11)
It can then be shown that rather than learning an estimate
of the expected return, the value estimate approximates the
expected return minus the expected discounted sum of future
TD-errors:
Qθ(st, at) = rt + γE[Qθ(st+1, at+1)] −δt
= rt + γE [rt+1 + γE [Qθ(st+2, at+2) −δt+1]] −δt
= Esi∼pπ,ai∼π
" T
X
i=t
γi−t(ri −δi)
#
.
(12)
If the value estimate is a function of future reward and es-
timation error, it follows that the variance of the estimate
will be proportional to the variance of future reward and es-
timation error. Given a large discount factor γ, the variance
can grow rapidly with each update if the error from each
update is not tamed. Furthermore each gradient update only
reduces error with respect to a small mini-batch which gives
no guarantees about the size of errors in value estimates
outside the mini-batch.
5.2. Target Networks and Delayed Policy Updates
In this section we examine the relationship between target
networks and function approximation error, and show the
use of a stable target reduces the growth of error. This
insight allows us to consider the interplay between high
variance estimates and policy performance, when designing
reinforcement learning algorithms.
Target networks are a well-known tool to achieve stabil-
ity in deep reinforcement learning. As deep function ap-
proximators require multiple gradient updates to converge,
target networks provide a stable objective in the learning
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e5)
150
200
250
300
350
Average Value
τ = 1
τ = 0.1
τ = 0.01
True Value
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e5)
102
103
104
(a) Fixed Policy
(b) Learned Policy
Figure 3. Average estimated value of a randomly selected state
on Hopper-v1 without target networks, (τ = 1), and with slow-
updating target networks, (τ = 0.1, 0.01), with a ﬁxed and a
learned policy.
procedure, and allow a greater coverage of the training data.
Without a ﬁxed target, each update may leave residual error
which will begin to accumulate. While the accumulation of
error can be detrimental in itself, when paired with a policy
maximizing over the value estimate, it can result in wildly
divergent values.
To provide some intuition, we examine the learning behavior
with and without target networks on both the critic and actor
in Figure 3, where we graph the value, in a similar manner to
Figure 1, in the Hopper-v1 environment. In (a) we compare
the behavior with a ﬁxed policy and in (b) we examine the
value estimates with a policy that continues to learn, trained
with the current value estimate. The target networks use a
slow-moving update rate, parametrized by τ.
While updating the value estimate without target networks
(τ = 1) increases the volatility, all update rates result in sim-
ilar convergent behaviors when considering a ﬁxed policy.
However, when the policy is trained with the current value
estimate, the use of fast-updating target networks results in
highly divergent behavior.
When do actor-critic methods fail to learn? These results
suggest that the divergence that occurs without target net-
works is the result of policy updates with a high variance
value estimate. Figure 3, as well as Section 4, suggest failure
can occur due to the interplay between the actor and critic
updates. Value estimates diverge through overestimation
when the policy is poor, and the policy will become poor if
the value estimate itself is inaccurate.
If target networks can be used to reduce the error over mul-
tiple updates, and policy updates on high-error states cause
divergent behavior, then the policy network should be up-
dated at a lower frequency than the value network, to ﬁrst
minimize error before introducing a policy update. We pro-
pose delaying policy updates until the value error is as small
as possible. The modiﬁcation is to only update the policy
and target networks after a ﬁxed number of updates d to the
critic. To ensure the TD-error remains small, we update the

Addressing Function Approximation Error in Actor-Critic Methods
target networks slowly θ′ ←τθ + (1 −τ)θ′.
By sufﬁciently delaying the policy updates we limit the like-
lihood of repeating updates with respect to an unchanged
critic. The less frequent policy updates that do occur will
use a value estimate with lower variance, and in principle,
should result in higher quality policy updates. This creates a
two-timescale algorithm, as often required for convergence
in the linear setting (Konda & Tsitsiklis, 2003). The effec-
tiveness of this strategy is captured by our empirical results
presented in Section 6.1, which show an improvement in
performance while using fewer policy updates.
5.3. Target Policy Smoothing Regularization
A concern with deterministic policies is they can overﬁt
to narrow peaks in the value estimate. When updating the
critic, a learning target using a deterministic policy is highly
susceptible to inaccuracies induced by function approxima-
tion error, increasing the variance of the target. This induced
variance can be reduced through regularization. We intro-
duce a regularization strategy for deep value learning, target
policy smoothing, which mimics the learning update from
SARSA (Sutton & Barto, 1998). Our approach enforces
the notion that similar actions should have similar value.
While the function approximation does this implicitly, the
relationship between similar actions can be forced explicitly
by modifying the training procedure. We propose that ﬁtting
the value of a small area around the target action
y = r + Eϵ [Qθ′(s′, πφ′(s′) + ϵ)] ,
(13)
would have the beneﬁt of smoothing the value estimate by
bootstrapping off of similar state-action value estimates. In
practice, we can approximate this expectation over actions
by adding a small amount of random noise to the target
policy and averaging over mini-batches. This makes our
modiﬁed target update:
y = r + γQθ′(s′, πφ′(s′) + ϵ),
ϵ ∼clip(N(0, σ), −c, c),
(14)
where the added noise is clipped to keep the target close to
the original action. The outcome is an algorithm reminiscent
of Expected SARSA (Van Seijen et al., 2009), where the
value estimate is instead learned off-policy and the noise
added to the target policy is chosen independently of the ex-
ploration policy. The value estimate learned is with respect
to a noisy policy deﬁned by the parameter σ.
Intuitively, it is known that policies derived from SARSA
value estimates tend to be safer, as they provide higher value
to actions resistant to perturbations. Thus, this style of
update can additionally lead to improvement in stochastic
domains with failure cases. A similar idea was introduced
concurrently by Nachum et al. (2018), smoothing over Qθ,
rather than Qθ′.
Algorithm 1 TD3
Initialize critic networks Qθ1, Qθ2, and actor network πφ
with random parameters θ1, θ2, φ
Initialize target networks θ′
1 ←θ1, θ′
2 ←θ2, φ′ ←φ
Initialize replay buffer B
for t = 1 to T do
Select action with exploration noise a ∼πφ(s) + ϵ,
ϵ ∼N(0, σ) and observe reward r and new state s′
Store transition tuple (s, a, r, s′) in B
Sample mini-batch of N transitions (s, a, r, s′) from B
˜a ←πφ′(s′) + ϵ,
ϵ ∼clip(N(0, ˜σ), −c, c)
y ←r + γ mini=1,2 Qθ′
i(s′, ˜a)
Update critics θi ←argminθi N −1 P(y−Qθi(s, a))2
if t mod d then
Update φ by the deterministic policy gradient:
∇φJ(φ) = N −1 P ∇aQθ1(s, a)|a=πφ(s)∇φπφ(s)
Update target networks:
θ′
i ←τθi + (1 −τ)θ′
i
φ′ ←τφ + (1 −τ)φ′
end if
end for
(a)
(b)
(c)
(d)
Figure 4. Example MuJoCo environments (a) HalfCheetah-v1, (b)
Hopper-v1, (c) Walker2d-v1, (d) Ant-v1.
6. Experiments
We present the Twin Delayed Deep Deterministic policy
gradient algorithm (TD3), which builds on the Deep Deter-
ministic Policy Gradient algorithm (DDPG) (Lillicrap et al.,
2015) by applying the modiﬁcations described in Sections
4.2, 5.2 and 5.3 to increase the stability and performance
with consideration of function approximation error. TD3
maintains a pair of critics along with a single actor. For each
time step, we update the pair of critics towards the minimum
target value of actions selected by the target policy:
y = r + γ min
i=1,2 Qθ′
i(s′, πφ′(s′) + ϵ),
ϵ ∼clip(N(0, σ), −c, c).
(15)
Every d iterations, the policy is updated with respect to Qθ1
following the deterministic policy gradient algorithm (Silver
et al., 2014). TD3 is summarized in Algorithm 1.

Addressing Function Approximation Error in Actor-Critic Methods
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
2000
4000
6000
8000
10000
Average Return
TD3
DDPG
our DDPG
PPO
TRPO
ACKTR
SAC
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
500
1000
1500
2000
2500
3000
3500
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
1000
2000
3000
4000
5000
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
−1000
0
1000
2000
3000
4000
(a) HalfCheetah-v1
(b) Hopper-v1
(c) Walker2d-v1
(d) Ant-v1
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
−12
−10
−8
−6
−4
Average Return
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
400
500
600
700
800
900
1000
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
2000
4000
6000
8000
10000
(e) Reacher-v1
(f) InvertedPendulum-v1
(g) InvertedDoublePendulum-v1
Figure 5. Learning curves for the OpenAI gym continuous control tasks. The shaded region represents half a standard deviation of the
average evaluation over 10 trials. Curves are smoothed uniformly for visual clarity.
Table 1. Max Average Return over 10 trials of 1 million time steps. Maximum value for each task is bolded. ± corresponds to a single
standard deviation over trials.
Environment
TD3
DDPG
Our DDPG
PPO
TRPO
ACKTR
SAC
HalfCheetah
9636.95 ± 859.065
3305.60
8577.29
1795.43
-15.57
1450.46
2347.19
Hopper
3564.07 ± 114.74
2020.46
1860.02
2164.70
2471.30
2428.39
2996.66
Walker2d
4682.82 ± 539.64
1843.85
3098.11
3317.69
2321.47
1216.70
1283.67
Ant
4372.44 ± 1000.33
1005.30
888.77
1083.20
-75.85
1821.94
655.35
Reacher
-3.60 ± 0.56
-6.51
-4.01
-6.18
-111.43
-4.26
-4.44
InvPendulum
1000.00 ± 0.00
1000.00
1000.00
1000.00
985.40
1000.00
1000.00
InvDoublePendulum
9337.47 ± 14.96
9355.52
8369.95
8977.94
205.85
9081.92
8487.15
6.1. Evaluation
To evaluate our algorithm, we measure its performance on
the suite of MuJoCo continuous control tasks (Todorov et al.,
2012), interfaced through OpenAI Gym (Brockman et al.,
2016) (Figure 4). To allow for reproducible comparison, we
use the original set of tasks from Brockman et al. (2016)
with no modiﬁcations to the environment or reward.
For our implementation of DDPG (Lillicrap et al., 2015), we
use a two layer feedforward neural network of 400 and 300
hidden nodes respectively, with rectiﬁed linear units (ReLU)
between each layer for both the actor and critic, and a ﬁnal
tanh unit following the output of the actor. Unlike the orig-
inal DDPG, the critic receives both the state and action as
input to the ﬁrst layer. Both network parameters are updated
using Adam (Kingma & Ba, 2014) with a learning rate of
10−3. After each time step, the networks are trained with a
mini-batch of a 100 transitions, sampled uniformly from a
replay buffer containing the entire history of the agent.
The target policy smoothing is implemented by adding ϵ ∼
N(0, 0.2) to the actions chosen by the target actor network,
clipped to (−0.5, 0.5), delayed policy updates consists of
only updating the actor and target critic network every d
iterations, with d = 2. While a larger d would result in a
larger beneﬁt with respect to accumulating errors, for fair
comparison, the critics are only trained once per time step,
and training the actor for too few iterations would cripple
learning. Both target networks are updated with τ = 0.005.
To remove the dependency on the initial parameters of the
policy we use a purely exploratory policy for the ﬁrst 10000
time steps of stable length environments (HalfCheetah-v1
and Ant-v1) and the ﬁrst 1000 time steps for the remaining
environments. Afterwards, we use an off-policy exploration
strategy, adding Gaussian noise N(0, 0.1) to each action.
Unlike the original implementation of DDPG, we used un-
correlated noise for exploration as we found noise drawn
from the Ornstein-Uhlenbeck (Uhlenbeck & Ornstein, 1930)
process offered no performance beneﬁts.
Each task is run for 1 million time steps with evaluations
every 5000 time steps, where each evaluation reports the

Addressing Function Approximation Error in Actor-Critic Methods
average reward over 10 episodes with no exploration noise.
Our results are reported over 10 random seeds of the Gym
simulator and the network initialization.
We compare our algorithm against DDPG (Lillicrap et al.,
2015) as well as the state of art policy gradient algorithms:
PPO (Schulman et al., 2017), ACKTR (Wu et al., 2017)
and TRPO (Schulman et al., 2015), as implemented by
OpenAI’s baselines repository (Dhariwal et al., 2017), and
SAC (Haarnoja et al., 2018), as implemented by the author’s
GitHub1. Additionally, we compare our method with our
re-tuned version of DDPG, which includes all architecture
and hyper-parameter modiﬁcations to DDPG without any
of our proposed adjustments. A full comparison between
our re-tuned version and the baselines DDPG is provided in
the supplementary material.
Our results are presented in Table 1 and learning curves in
Figure 5. TD3 matches or outperforms all other algorithms
in both ﬁnal performance and learning speed across all tasks.
6.2. Ablation Studies
We perform ablation studies to understand the contribution
of each individual component: Clipped Double Q-learning
(Section 4.2), delayed policy updates (Section 5.2) and target
policy smoothing (Section 5.3). We present our results in
Table 2 in which we compare the performance of removing
each component from TD3 along with our modiﬁcations to
the architecture and hyper-parameters. Additional learning
curves can be found in the supplementary material.
The signiﬁcance of each component varies task to task.
While the addition of only a single component causes in-
signiﬁcant improvement in most cases, the addition of com-
binations performs at a much higher level. The full algo-
rithm outperforms every other combination in most tasks.
Although the actor is trained for only half the number of
iterations, the inclusion of delayed policy update generally
improves performance, while reducing training time.
We additionally compare the effectiveness of the actor-critic
variants of Double Q-learning (Van Hasselt, 2010) and Dou-
ble DQN (Van Hasselt et al., 2016), denoted DQ-AC and
DDQN-AC respectively, in Table 2. For fairness in com-
parison, these methods also beneﬁted from delayed policy
updates, target policy smoothing and use our architecture
and hyper-parameters. Both methods were shown to reduce
overestimation bias less than Clipped Double Q-learning in
Section 4. This is reﬂected empirically, as both methods
result in insigniﬁcant improvements over TD3 - CDQ, with
an exception in the Ant-v1 environment, which appears to
beneﬁt greatly from any overestimation reduction. As the
inclusion of Clipped Double Q-learning into our full method
1See the supplementary material for hyper-parameters and a
discussion on the discrepancy in the reported results of SAC.
Table 2. Average return over the last 10 evaluations over 10 trials
of 1 million time steps, comparing ablation over delayed policy
updates (DP), target policy smoothing (TPS), Clipped Double
Q-learning (CDQ) and our architecture, hyper-parameters and
exploration (AHE). Maximum value for each task is bolded.
Method
HCheetah
Hopper
Walker2d
Ant
TD3
9532.99
3304.75
4565.24
4185.06
DDPG
3162.50
1731.94
1520.90
816.35
AHE
8401.02
1061.77
2362.13
564.07
AHE + DP
7588.64
1465.11
2459.53
896.13
AHE + TPS
9023.40
907.56
2961.36
872.17
AHE + CDQ
6470.20
1134.14
3979.21
3818.71
TD3 - DP
9590.65
2407.42
4695.50
3754.26
TD3 - TPS
8987.69
2392.59
4033.67
4155.24
TD3 - CDQ
9792.80
1837.32
2579.39
849.75
DQ-AC
9433.87
1773.71
3100.45
2445.97
DDQN-AC
10306.90
2155.75
3116.81
1092.18
outperforms both prior methods, this suggests that subdu-
ing the overestimations from the unbiased estimator is an
effective measure to improve performance.
7. Conclusion
Overestimation has been identiﬁed as a key problem in
value-based methods. In this paper, we establish overesti-
mation bias is also problematic in actor-critic methods. We
ﬁnd the common solutions for reducing overestimation bias
in deep Q-learning with discrete actions are ineffective in an
actor-critic setting, and develop a novel variant of Double
Q-learning which limits possible overestimation. Our re-
sults demonstrate that mitigating overestimation can greatly
improve the performance of modern algorithms.
Due to the connection between noise and overestimation,
we examine the accumulation of errors from temporal dif-
ference learning. Our work investigates the importance of
a standard technique in deep reinforcement learning, target
networks, and examines their role in limiting errors from
imprecise function approximation and stochastic optimiza-
tion. Finally, we introduce a SARSA-style regularization
technique which modiﬁes the temporal difference target to
bootstrap off similar state-action pairs.
Taken together, these improvements deﬁne our proposed
approach, the Twin Delayed Deep Deterministic policy gra-
dient algorithm (TD3), which greatly improves both the
learning speed and performance of DDPG in a number of
challenging tasks in the continuous control setting. Our
algorithm exceeds the performance of numerous state of
the art algorithms. As our modiﬁcations are simple to im-
plement, they can be easily added to any other actor-critic
algorithm.

Addressing Function Approximation Error in Actor-Critic Methods
References
Anschel, O., Baram, N., and Shimkin, N. Averaged-dqn:
Variance reduction and stabilization for deep reinforce-
ment learning. In International Conference on Machine
Learning, pp. 176–185, 2017.
Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney,
W., Horgan, D., TB, D., Muldal, A., Heess, N., and Lil-
licrap, T. Distributional policy gradients. International
Conference on Learning Representations, 2018.
Bellemare, M. G., Dabney, W., and Munos, R. A distribu-
tional perspective on reinforcement learning. In Interna-
tional Conference on Machine Learning, pp. 449–458,
2017.
Bellman, R. Dynamic Programming. Princeton University
Press, 1957.
Bertsekas, D. P. Dynamic programming and optimal control,
volume 1. Athena scientiﬁc Belmont, MA, 1995.
Brockman, G., Cheung, V., Pettersson, L., Schneider, J.,
Schulman, J., Tang, J., and Zaremba, W. Openai gym,
2016.
Dhariwal, P., Hesse, C., Plappert, M., Radford, A., Schul-
man, J., Sidor, S., and Wu, Y. Openai baselines. https:
//github.com/openai/baselines, 2017.
Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih,
V., Ward, T., Doron, Y., Firoiu, V., Harley, T., Dunning,
I., et al. Impala: Scalable distributed deep-rl with impor-
tance weighted actor-learner architectures. arXiv preprint
arXiv:1802.01561, 2018.
Fox, R., Pakman, A., and Tishby, N. Taming the noise in
reinforcement learning via soft updates. In Proceedings of
the Thirty-Second Conference on Uncertainty in Artiﬁcial
Intelligence, pp. 202–211. AUAI Press, 2016.
Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft
actor-critic: Off-policy maximum entropy deep reinforce-
ment learning with a stochastic actor. arXiv preprint
arXiv:1801.01290, 2018.
He, F. S., Liu, Y., Schwing, A. G., and Peng, J. Learning
to play in a day: Faster deep reinforcement learning by
optimality tightening. arXiv preprint arXiv:1611.01606,
2016.
Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup,
D., and Meger, D. Deep Reinforcement Learning that
Matters. arXiv preprint arXiv:1709.06560, 2017.
Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel,
M., van Hasselt, H., and Silver, D. Distributed prioritized
experience replay. International Conference on Learning
Representations, 2018.
Kingma, D. and Ba, J. Adam: A method for stochastic
optimization. arXiv preprint arXiv:1412.6980, 2014.
Konda, V. R. and Tsitsiklis, J. N. On actor-critic algorithms.
SIAM journal on Control and Optimization, 42(4):1143–
1166, 2003.
Lee, D., Defourny, B., and Powell, W. B. Bias-corrected
q-learning to control max-operator bias in q-learning.
In Adaptive Dynamic Programming And Reinforcement
Learning (ADPRL), 2013 IEEE Symposium on, pp. 93–99.
IEEE, 2013.
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez,
T., Tassa, Y., Silver, D., and Wierstra, D. Continuous
control with deep reinforcement learning. arXiv preprint
arXiv:1509.02971, 2015.
Lin, L.-J. Self-improving reactive agents based on reinforce-
ment learning, planning and teaching. Machine learning,
8(3-4):293–321, 1992.
Mannor, S. and Tsitsiklis, J. N. Mean-variance optimization
in markov decision processes. In International Confer-
ence on Machine Learning, pp. 177–184, 2011.
Mannor, S., Simester, D., Sun, P., and Tsitsiklis, J. N. Bias
and variance approximation in value function estimates.
Management Science, 53(2):308–322, 2007.
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness,
J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidje-
land, A. K., Ostrovski, G., et al. Human-level control
through deep reinforcement learning. Nature, 518(7540):
529–533, 2015.
Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap,
T., Harley, T., Silver, D., and Kavukcuoglu, K. Asyn-
chronous methods for deep reinforcement learning. In
International Conference on Machine Learning, pp. 1928–
1937, 2016.
Munos, R., Stepleton, T., Harutyunyan, A., and Bellemare,
M. Safe and efﬁcient off-policy reinforcement learning.
In Advances in Neural Information Processing Systems,
pp. 1054–1062, 2016.
Nachum, O., Norouzi, M., Tucker, G., and Schuurmans, D.
Smoothed action value functions for learning gaussian
policies. arXiv preprint arXiv:1803.02348, 2018.
O’Donoghue, B., Osband, I., Munos, R., and Mnih, V. The
uncertainty bellman equation and exploration.
arXiv
preprint arXiv:1709.05380, 2017.
Pendrith, M. D., Ryan, M. R., et al. Estimator variance in
reinforcement learning: Theoretical problems and practi-
cal solutions. University of New South Wales, School of
Computer Science and Engineering, 1997.

Addressing Function Approximation Error in Actor-Critic Methods
Petrik, M. and Scherrer, B. Biasing approximate dynamic
programming with a lower discount factor. In Advances in
Neural Information Processing Systems, pp. 1265–1272,
2009.
Popov, I., Heess, N., Lillicrap, T., Hafner, R., Barth-
Maron, G., Vecerik, M., Lampe, T., Tassa, Y., Erez,
T., and Riedmiller, M.
Data-efﬁcient deep reinforce-
ment learning for dexterous manipulation. arXiv preprint
arXiv:1704.03073, 2017.
Precup, D., Sutton, R. S., and Dasgupta, S. Off-policy
temporal-difference learning with function approxima-
tion. In International Conference on Machine Learning,
pp. 417–424, 2001.
Schaul, T., Quan, J., Antonoglou, I., and Silver, D. Priori-
tized experience replay. In International Conference on
Learning Representations, Puerto Rico, 2016.
Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz,
P. Trust region policy optimization. In International
Conference on Machine Learning, pp. 1889–1897, 2015.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
arXiv preprint arXiv:1707.06347, 2017.
Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and
Riedmiller, M. Deterministic policy gradient algorithms.
In International Conference on Machine Learning, pp.
387–395, 2014.
Singh, S., Jaakkola, T., Littman, M. L., and Szepesv´ari,
C.
Convergence results for single-step on-policy
reinforcement-learning algorithms. Machine learning,
38(3):287–308, 2000.
Sutton, R. S. Learning to predict by the methods of temporal
differences. Machine learning, 3(1):9–44, 1988.
Sutton, R. S. and Barto, A. G. Reinforcement learning: An
introduction, volume 1. MIT press Cambridge, 1998.
Thrun, S. and Schwartz, A. Issues in using function approx-
imation for reinforcement learning. In Proceedings of the
1993 Connectionist Models Summer School Hillsdale, NJ.
Lawrence Erlbaum, 1993.
Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics
engine for model-based control. In Intelligent Robots
and Systems (IROS), 2012 IEEE/RSJ International Con-
ference on, pp. 5026–5033. IEEE, 2012.
Uhlenbeck, G. E. and Ornstein, L. S. On the theory of the
brownian motion. Physical review, 36(5):823, 1930.
Van Hasselt, H. Double q-learning. In Advances in Neural
Information Processing Systems, pp. 2613–2621, 2010.
Van Hasselt, H., Guez, A., and Silver, D. Deep reinforce-
ment learning with double q-learning. In AAAI, pp. 2094–
2100, 2016.
Van Seijen, H., Van Hasselt, H., Whiteson, S., and Wiering,
M. A theoretical and empirical analysis of expected sarsa.
In Adaptive Dynamic Programming and Reinforcement
Learning, 2009. ADPRL’09. IEEE Symposium on, pp.
177–184. IEEE, 2009.
Watkins, C. J. C. H. Learning from delayed rewards. PhD
thesis, King’s College, Cambridge, 1989.
Wu, Y., Mansimov, E., Grosse, R. B., Liao, S., and Ba,
J. Scalable trust-region method for deep reinforcement
learning using kronecker-factored approximation. In Ad-
vances in Neural Information Processing Systems, pp.
5285–5294, 2017.

Supplementary Material
A. Proof of Convergence of Clipped Double Q-Learning
In a version of Clipped Double Q-learning for a ﬁnite MDP setting, we maintain two tabular value estimates QA, QB. At
each time step we select actions a∗= argmaxa QA(s, a) and then perform an update by setting target y:
a∗= argmax
a
QA(s′, a)
y = r + γ min(QA(s′, a∗), QB(s′, a∗)),
(16)
and update the value estimates with respect to the target and learning rate αt(s, a):
QA(s, a) = QA(s, a) + αt(s, a)(y −QA(s, a))
QB(s, a) = QB(s, a) + αt(s, a)(y −QB(s, a)).
(17)
In a ﬁnite MDP setting, Double Q-learning is often used to deal with noise induced by random rewards or state transitions,
and so either QA or QB is updated randomly. However, in a function approximation setting, the interest may be more
towards the approximation error and thus we can update both QA and QB at each iteration. The proof extends naturally to
updating either randomly.
The proof borrows heavily from the proof of convergence of SARSA (Singh et al., 2000) as well as Double Q-learning
(Van Hasselt, 2010). The proof of lemma 1 can be found in Singh et al. (2000), building on a proposition from Bertsekas
(1995).
Lemma 1. Consider a stochastic process (ζt, ∆t, Ft), t ≥0 where ζt, ∆t, Ft : X →R satisfy the equation:
∆t+1(xt) = (1 −ζt(xt))∆t(xt) + ζt(xt)Ft(xt),
(18)
where xt ∈X and t = 0, 1, 2, .... Let Pt be a sequence of increasing σ-ﬁelds such that ζ0 and ∆0 are P0-measurable and
ζt, ∆t and Ft−1 are Pt-measurable, t = 1, 2, .... Assume that the following hold:
1. The set X is ﬁnite.
2. ζt(xt) ∈[0, 1], P
t ζt(xt) = ∞, P
t(ζt(xt))2 < ∞with probability 1 and ∀x̸ = xt : ζ(x) = 0.
3. ||E [Ft|Pt] || ≤κ||∆t|| + ct where κ ∈[0, 1) and ct converges to 0 with probability 1.
4. Var[Ft(xt)|Pt] ≤K(1 + κ||∆t||)2, where K is some constant
Where || · || denotes the maximum norm. Then ∆t converges to 0 with probability 1.
Theorem 1. Given the following conditions:
1. Each state action pair is sampled an inﬁnite number of times.
2. The MDP is ﬁnite.
3. γ ∈[0, 1).
4. Q values are stored in a lookup table.
5. Both QA and QB receive an inﬁnite number of updates.
6. The learning rates satisfy αt(s, a) ∈[0, 1], P
t αt(s, a) = ∞, P
t(αt(s, a))2 < ∞with probability 1 and αt(s, a) =
0, ∀(s, a)̸ = (st, at).

Addressing Function Approximation Error in Actor-Critic Methods
7. Var[r(s, a)] < ∞, ∀s, a.
Then Clipped Double Q-learning will converge to the optimal value function Q∗, as deﬁned by the Bellman optimality
equation, with probability 1.
Proof of Theorem 1. We apply Lemma 1 with Pt = {QA
0 , QB
0 , s0, a0, α0, r1, s1, ..., st, at}, X = S × A, ∆t = QA
t −
Q∗, ζt = αt.
First note that condition 1 and 4 of the lemma holds by the conditions 2 and 7 of the theorem respectively. Lemma condition
2 holds by the theorem condition 6 along with our selection of ζt = αt.
Deﬁning a∗= argmaxa QA(st+1, a) we have
∆t+1(st, at) = (1 −αt(st, at))(QA
t (st, at) −Q∗(st, at))
+ αt(st, at)(rt + γ min(QA
t (st+1, a∗), QB
t (st+1, a∗)) −Q∗(st, at))
= (1 −αt(st, at))∆t(st, at) + αt(st, at)Ft(st, at)),
(19)
where we have deﬁned Ft(st, at) as:
Ft(st, at) = rt + γ min(QA
t (st+1, a∗), QB
t (st+1, a∗)) −Q∗
t (st, at)
= rt + γ min(QA
t (st+1, a∗), QB
t (st+1, a∗)) −Q∗
t (st, at) + γQA
t (st+1, a∗) −γQA
t (st+1, a∗)
= F Q
t (st, at) + ct,
(20)
where F Q
t
= rt + γQA
t (st+1, a∗) −Q∗
t (st, at) denotes the value of Ft under standard Q-learning and ct
=
γ min(QA
t (st+1, a∗), QB
t (st+1, a∗)) −γQA
t (st+1, a∗). As E
h
F Q
t |Pt
i
≤γ||∆t|| is a well-known result, then condition 3
of lemma 1 holds if it can be shown that ct converges to 0 with probability 1.
Let y = rt + γ min(QB
t (st+1, a∗), QA
t (st+1, a∗)) and ∆BA
t
(st, at) = QB
t (st, at) −QA
t (st, at), where ct converges to 0
if ∆BA converges to 0. The update of ∆BA
t
at time t is the sum of updates of QA and QB:
∆BA
t+1(st, at) = ∆BA
t
(st, at) + αt(st, at)
 y −QB
t (st, at) −(y −QA
t (st, at))

= ∆BA
t
(st, at) + αt(st, at)
 QA
t (st, at) −QB
t (st, at)

= (1 −αt(st, at))∆BA
t
(st, at).
(21)
Clearly ∆BA
t
will converge to 0, which then shows we have satisﬁed condition 3 of lemma 1, implying that QA(st, at)
converges to Q∗
t (st, at). Similarly, we get convergence of QB(st, at) to the optimal vale function by choosing ∆t =
QB
t −Q∗and repeating the same arguments, thus proving theorem 1.
B. Overestimation Bias in Deterministic Policy Gradients
If the gradients from the deterministic policy gradient update are unnormalized, this overestimation is still guaranteed to
occur under a slightly stronger condition on the expectation of the value estimate. Assume the approximate value function is
equal to the true value function, in expectation over the steady-state distribution, with respect to policy parameters between
the original policy and in the direction of the true policy update:
Es∼π [Qθ(s, πnew(s))] = Es∼π [Qπ(s, πnew(s))]
∀φnew ∈[φ, φ + β(φtrue −φ)] such that β > 0.
(22)
Noting that φtrue maximizes the rate of change of the true value ∆π
true = Qπ(s, πtrue(s))−Qπ(s, πφ(s)), ∆π
true ≥∆π
approx. By
the given condition 22 the maximal rate of change of the approximate value must be at least as great ∆θ
approx ≥∆π
true. Given
Qθ(s, πφ) = Qπ(s, πφ) this implies Qθ(s, πapprox(s)) ≥Qπ(s, πtrue(s)) ≥Qπ(s, πapprox(s)), showing an overestimation of
the value function.

Addressing Function Approximation Error in Actor-Critic Methods
Table 3. A complete comparison of hyper-parameter choices between our DDPG and the OpenAI baselines implementation (Dhariwal
et al., 2017).
Hyper-parameter
Ours
DDPG
Critic Learning Rate
10−3
10−3
Critic Regularization
None
10−2 · ||θ||2
Actor Learning Rate
10−3
10−4
Actor Regularization
None
None
Optimizer
Adam
Adam
Target Update Rate (τ)
5 · 10−3
10−3
Batch Size
100
64
Iterations per time step
1
1
Discount Factor
0.99
0.99
Reward Scaling
1.0
1.0
Normalized Observations
False
True
Gradient Clipping
False
False
Exploration Policy
N(0, 0.1)
OU, θ = 0.15, µ = 0, σ = 0.2
C. DDPG Network and Hyper-parameter Comparison
DDPG Critic Architecture
(state dim, 400)
ReLU
(action dim + 400, 300)
ReLU
(300, 1)
DDPG Actor Architecture
(state dim, 400)
ReLU
(400, 300)
ReLU
(300, 1)
tanh
Our Critic Architecture
(state dim + action dim, 400)
ReLU
(action dim + 400, 300)
RelU
(300, 1)
Our Actor Architecture
(state dim, 400)
ReLU
(400, 300)
RelU
(300, 1)
tanh

Addressing Function Approximation Error in Actor-Critic Methods
D. Additional Implementation Details
For clarity in presentation, certain implementation details were omitted, which we describe here. For the most complete
possible description of the algorithm, code can be found on our GitHub (https://github.com/sfujim/TD3).
Our implementation of both DDPG and TD3 follows a standard practice in deep Q-learning, in which the update differs for
terminal transitions. For transitions where the episode terminates by reaching some failure state, and not due to the episode
running until the max horizon, the value of Q(s, ·) is set to 0 in the target y:
y =
(
r
if terminal s′ and t < max horizon
r + γQθ′(s′, πφ′(s′))
else
For target policy smoothing (Section 5.3), the added noise is clipped to the range of possible actions, to avoid error introduced
by using values of impossible actions:
y = r + γQθ′(s′, clip(πφ′(s′) + ϵ, min action, max action)),
ϵ ∼clip(N(0, σ), −c, c).
E. Soft Actor-Critic Implementation Details
For our implementation of Soft Actor-Critic (Haarnoja et al., 2018) we use the code provided by the author (https:
//github.com/haarnoja/sac), using the hyper-parameters described by the paper. We use a Gaussian mixture
policy with 4 Gaussian distributions, except for the Reacher-v1 task, where we use a single Gaussian distribution due to
numerical instability issues in the provided implementation. We use the environment-dependent reward scaling as described
by the authors, multiplying the rewards by 3 for Walker2d-v1 and Ant-v1, and 1 for all remaining environments.
For fair comparison with our method, we train for only 1 iteration per time step, rather than the 4 iterations used by the
results reported by the authors. This along with fewer total time steps should explain for the discrepancy in results on some
of the environments. Additionally, we note this comparison is against a prior version of Soft Actor-Critic, while the most
recent variant includes our Clipped Double Q-learning in the value update and produces competitive results to TD3 on most
tasks.

Addressing Function Approximation Error in Actor-Critic Methods
F. Additional Learning Curves
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
2000
4000
6000
8000
10000
Average Return
TD3
DDPG
AHE
AHE + TPS
AHE + DP
AHE + CDQ
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
500
1000
1500
2000
2500
3000
3500
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
1000
2000
3000
4000
5000
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
−1000
0
1000
2000
3000
4000
(a) HalfCheetah-v1
(b) Hopper-v1
(c) Walker2d-v1
(d) Ant-v1
Figure 6. Ablation over the varying modiﬁcations to our DDPG (AHE), comparing the subtraction of delayed policy updates (TD3 - DP),
target policy smoothing (TD3 - TPS) and Clipped Double Q-learning (TD3 - CDQ).
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
2000
4000
6000
8000
10000
Average Return
TD3
DDPG
AHE
TD3 - TPS
TD3 - DP
TD3 - CDQ
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
500
1000
1500
2000
2500
3000
3500
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
1000
2000
3000
4000
5000
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
−1000
0
1000
2000
3000
4000
(a) HalfCheetah-v1
(b) Hopper-v1
(c) Walker2d-v1
(d) Ant-v1
Figure 7. Ablation over the varying modiﬁcations to our DDPG (AHE), comparing the addition of delayed policy updates (AHE + DP),
target policy smoothing (AHE + TPS) and Clipped Double Q-learning (AHE + CDQ).
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
2000
4000
6000
8000
10000
Average Return
TD3
AHE
TD3 - CDQ
DQ-AC
DDQN-AC
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
500
1000
1500
2000
2500
3000
3500
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
0
1000
2000
3000
4000
5000
0.0
0.2
0.4
0.6
0.8
1.0
Time steps (1e6)
−1000
0
1000
2000
3000
4000
(a) HalfCheetah-v1
(b) Hopper-v1
(c) Walker2d-v1
(d) Ant-v1
Figure 8. Comparison of TD3 and the Double Q-learning (DQ-AC) and Double DQN (DDQN-AC) actor-critic variants, which also
leverage delayed policy updates and target policy smoothing.
