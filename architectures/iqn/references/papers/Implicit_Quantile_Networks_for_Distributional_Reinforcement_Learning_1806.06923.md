# Paper (Dabney et al. 2018)

> Source: `https://arxiv.org/abs/1806.06923`

---

Implicit Quantile Networks for Distributional Reinforcement Learning
Will Dabney * 1 Georg Ostrovski * 1 David Silver 1 R´emi Munos 1
Abstract
In this work, we build on recent advances in dis-
tributional reinforcement learning to give a gener-
ally applicable, ﬂexible, and state-of-the-art dis-
tributional variant of DQN. We achieve this by
using quantile regression to approximate the full
quantile function for the state-action return distri-
bution. By reparameterizing a distribution over
the sample space, this yields an implicitly deﬁned
return distribution and gives rise to a large class of
risk-sensitive policies. We demonstrate improved
performance on the 57 Atari 2600 games in the
ALE, and use our algorithm’s implicitly deﬁned
distributions to study the effects of risk-sensitive
policies in Atari games.
1. Introduction
Distributional reinforcement learning (Jaquette, 1973; Sobel,
1982; White, 1988; Morimura et al., 2010b; Bellemare et al.,
2017) focuses on the intrinsic randomness of returns within
the reinforcement learning (RL) framework. As the agent in-
teracts with the environment, irreducible randomness seeps
in through the stochasticity of these interactions, the ap-
proximations in the agent’s representation, and even the
inherently chaotic nature of physical interaction (Yu et al.,
2016). Distributional RL aims to model the distribution over
returns, whose mean is the traditional value function, and to
use these distributions to evaluate and optimize a policy.
Any distributional RL algorithm is characterized by two
aspects: the parameterization of the return distribution, and
the distance metric or loss function being optimized. To-
gether, these choices control assumptions about the random
returns and how approximations will be traded off. Cat-
egorical DQN (Bellemare et al., 2017, C51) combines a
categorical distribution and the cross-entropy loss with the
Cram´er-minimizing projection (Rowland et al., 2018). For
*Equal contribution 1DeepMind, London, UK. Correspondence
to: Will Dabney <wdabney@google.com>, Georg Ostrovski <os-
trovski@google.com>.
Proceedings of the 35 th International Conference on Machine
Learning, Stockholm, Sweden, PMLR 80, 2018. Copyright 2018
by the author(s).
this, it assumes returns are bounded in a known range and
trades off mean-preservation at the cost of overestimating
variance.
C51 outperformed all previous improvements to DQN on
a set of 57 Atari 2600 games in the Arcade Learning En-
vironment (Bellemare et al., 2013), which we refer to as
the Atari-57 benchmark. Subsequently, several papers have
built upon this successful combination to achieve signiﬁ-
cant improvements to the state-of-the-art in Atari-57 (Hessel
et al., 2018; Gruslys et al., 2018), and challenging continu-
ous control tasks (Barth-Maron et al., 2018).
These algorithms are restricted to assigning probabilities to
an a priori ﬁxed, discrete set of possible returns. Dabney
et al. (2018) propose an alternate pair of choices, parameter-
izing the distribution by a uniform mixture of Diracs whose
locations are adjusted using quantile regression. Their algo-
rithm, QR-DQN, while restricted to a discrete set of quan-
tiles, automatically adapts return quantiles to minimize the
Wasserstein distance between the Bellman updated and cur-
rent return distributions. This ﬂexibility allows QR-DQN to
signiﬁcantly improve on C51’s Atari-57 performance.
In this paper, we extend the approach of Dabney et al.
(2018), from learning a discrete set of quantiles to learning
the full quantile function, a continuous map from probabil-
ities to returns. When combined with a base distribution,
such as U([0, 1]), this forms an implicit distribution capable
of approximating any distribution over returns given suf-
ﬁcient network capacity. Our approach, implicit quantile
networks (IQN), is best viewed as a simple distributional
generalization of the DQN algorithm (Mnih et al., 2015),
and provides several beneﬁts over QR-DQN.
First, the approximation error for the distribution is no
longer controlled by the number of quantiles output by
the network, but by the size of the network itself, and the
amount of training. Second, IQN can be used with as few, or
as many, samples per update as desired, providing improved
data efﬁciency with increasing number of samples per train-
ing update. Third, the implicit representation of the return
distribution allows us to expand the class of policies to more
fully take advantage of the learned distribution. Speciﬁcally,
by taking the base distribution to be non-uniform, we ex-
pand the class of policies to ϵ-greedy policies on arbitrary
distortion risk measures (Yaari, 1987; Wang, 1996).
arXiv:1806.06923v1  [cs.LG]  14 Jun 2018

Implicit Quantile Networks for Distributional Reinforcement Learning
We begin by reviewing distributional reinforcement learn-
ing, related work, and introducing the concepts surround-
ing risk-sensitive RL. In subsequent sections, we introduce
our proposed algorithm, IQN, and present a series of ex-
periments using the Atari-57 benchmark, investigating the
robustness and performance of IQN. Despite being a simple
distributional extension to DQN, and forgoing any other im-
provements, IQN signiﬁcantly outperforms QR-DQN and
nearly matches the performance of Rainbow, which com-
bines many orthogonal advances. In fact, in human-starts
as well as in the hardest Atari games (where current RL
agents still underperform human players) IQN improves
over Rainbow.
2. Background / Related Work
We consider the standard RL setting, in which the interaction
of an agent and an environment is modeled as a Markov
Decision Process (X, A, R, P, γ) (Puterman, 1994), where
X and A denote the state and action spaces, R the (state- and
action-dependent) reward function, P(·|x, a) the transition
kernel, and γ ∈(0, 1) a discount factor. A policy π(·|x)
maps a state to a distribution over actions.
For an agent following policy π, the discounted sum
of future rewards is denoted by the random variable
Zπ(x, a) = P∞
t=0 γtR(xt, at), where x0 = x, a0 = a,
xt ∼P(·|xt−1, at−1), and at ∼π(·|xt). The action-value
function is deﬁned as Qπ(x, a) = E [Zπ(x, a)], and can be
characterized by the Bellman equation
Qπ(x, a) = E [R(x, a)] + γEP,π [Qπ(x′, a′)] .
The objective in RL is to ﬁnd an optimal policy π∗, which
maximizes E[Zπ], i.e. Qπ∗(x, a) ≥Qπ(x, a) for all π and
all x, a. One approach is to ﬁnd the unique ﬁxed point
Q∗= Qπ∗of the Bellman optimality operator (Bellman,
1957):
Q(x, a) = T Q(x, a) := E [R(x, a)]+γEP max
a′ Q(x′, a′).
To this end, Q-learning (Watkins, 1989) iteratively improves
an estimate, Qθ, of the optimal action-value function, Q∗,
by repeatedly applying the Bellman update:
Qθ(x, a) ←E [R(x, a)] + γEP
h
max
a′ Qθ(x′, a′)
i
.
The action-value function can be approximated by a param-
eterized function Qθ (e.g. a neural network), and trained by
minimizing the squared temporal difference (TD) error,
δ2
t =

rt + γ max
a′∈A Qθ(xt+1, a′) −Qθ(xt, at)
2
,
over samples (xt, at, rt, xt+1) observed while following an
ϵ-greedy policy over Qθ. This policy acts greedily with re-
spect to Qθ with probability 1 −ϵ and uniformly at random
otherwise. DQN (Mnih et al., 2015) uses a convolutional
neural network to parameterize Qθ and the Q-learning algo-
rithm to achieve human-level play on the Atari-57 bench-
mark.
2.1. Distributional RL
In distributional RL, the distribution over returns (the law
of Zπ) is considered instead of the scalar value function
Qπ that is its expectation. This change in perspective has
yielded new insights into the dynamics of RL (Azar et al.,
2012), and been a useful tool for analysis (Lattimore &
Hutter, 2012). Empirically, distributional RL algorithms
show improved sample complexity and ﬁnal performance,
as well as increased robustness to hyperparameter variation
(Barth-Maron et al., 2018).
An analogous distributional Bellman equation of the form
Zπ(x, a)
D= R(x, a) + γZπ(X′, A′)
can be derived, where A
D= B denotes that two random
variables A and B have equal probability laws, and the
random variables X′ and A′ are distributed according to
P(·|x, a) and π(·|x′), respectively.
Morimura et al. (2010a) deﬁned the distributional Bellman
operator explicitly in terms of conditional probabilities, pa-
rameterized by the mean and scale of a Gaussian or Laplace
distribution, and minimized the Kullback-Leibler (KL) di-
vergence between the Bellman target and the current esti-
mated return distribution. However, the distributional Bell-
man operator is not a contraction in the KL.
As with the scalar setting, a distributional Bellman optimal-
ity operator can be deﬁned by
T Z(x, a)
D
:= R(x, a) + γZ(X′, arg max
a′∈A
E Z(X′, a′)),
with X′ distributed according to P(·|x, a). While the distri-
butional Bellman operator for policy evaluation is a contrac-
tion in the p-Wasserstein distance (Bellemare et al., 2017),
this no longer holds for the control case. Convergence to the
optimal policy can still be established, but requires a more
involved argument.
Bellemare et al. (2017) parameterize the return distribution
as a categorical distribution over a ﬁxed set of equidistant
points and minimize the KL divergence to the projected
distributional Bellman target. Their algorithm, C51, outper-
formed previous DQN variants on the Atari-57 benchmark.
Subsequently, Hessel et al. (2018) combined C51 with en-
hancements such as prioritized experience replay (Schaul
et al., 2016), n-step updates (Sutton, 1988), and the dueling
architecture (Wang et al., 2016), leading to the Rainbow
agent, current state-of-the-art in Atari-57.

Implicit Quantile Networks for Distributional Reinforcement Learning
The categorical parameterization, using the projected KL
loss, has also been used in recent work to improve the critic
of a policy gradient algorithm, D4PG, achieving signiﬁ-
cantly improved robustness and state-of-the-art performance
across a variety of continuous control tasks (Barth-Maron
et al., 2018).
2.2. p-Wasserstein Metric
The p-Wasserstein metric, for p ∈[1, ∞], plays a key role
in recent results in distributional RL (Bellemare et al., 2017;
Dabney et al., 2018). It has also been a topic of increasing
interest in generative modeling (Arjovsky et al., 2017; Bous-
quet et al., 2017; Tolstikhin et al., 2017), because unlike the
KL divergence, the Wasserstein metric inherently trades off
approximate solutions with likelihoods.
The p-Wasserstein distance is the Lp metric on inverse cu-
mulative distribution functions (c.d.f.), also known as quan-
tile functions (M¨uller, 1997). For random variables U and
V with quantile functions F −1
U
and F −1
V , respectively, the
p-Wasserstein distance is given by
Wp(U, V ) =
Z 1
0
|F −1
U (ω) −F −1
V (ω)|pdω
1/p
.
The class of optimal transport metrics express distances
between distributions in terms of the minimal cost for trans-
porting mass to make the two distributions identical. This
cost is given in terms of some metric, c: X × X →R≥0,
on the underlying space X. The p-Wasserstein metric cor-
responds to c = Lp. We are particularly interested in the
Wasserstein metrics due to the predominant use of Lp spaces
in mean-value reinforcement learning.
2.3. Quantile Regression for Distributional RL
Bellemare et al. (2017) showed that the distributional Bell-
man operator is a contraction in the p-Wasserstein metric,
but as the proposed algorithm did not itself minimize the
Wasserstein metric, this left a theory-practice gap for distri-
butional RL. Recently, this gap was closed, in both direc-
tions. First and most relevant to this work, Dabney et al.
(2018) proposed the use of quantile regression for distribu-
tional RL and showed that by choosing the quantile targets
suitably the resulting projected distributional Bellman op-
erator is a contraction in the ∞-Wasserstein metric. Con-
currently, Rowland et al. (2018) showed the original class
of categorical algorithms are a contraction in the Cram´er
distance, the L2 metric on cumulative distribution functions.
By estimating the quantile function at precisely chosen
points, QR-DQN minimizes the Wasserstein distance to
the distributional Bellman target (Dabney et al., 2018). This
estimation uses quantile regression, which has been shown
to converge to the true quantile function value when mini-
mized using stochastic approximation (Koenker, 2005).
In QR-DQN, the random return is approximated by a uni-
form mixture of N Diracs,
Zθ(x, a) := 1
N
N
X
i=1
δθi(x,a),
with each θi assigned a ﬁxed quantile target, ˆτi = τi−1+τi
2
for 1 ≤i ≤N, where τi = i/N. These quantile estimates
are trained using the Huber (1964) quantile regression loss,
with threshold κ,
ρκ
τ(δij) = |τ −I{δij < 0}|Lκ(δij)
κ
,
with
Lκ(δij) =
(
1
2δ2
ij,
if |δij| ≤κ
κ(|δij| −1
2κ),
otherwise
,
on the pairwise TD-errors
δij = r + γθj(x′, π(x′)) −θi(x, a).
At the time of this writing, QR-DQN achieves the best per-
formance on Atari-57, human-normalized mean and median,
of all agents that do not combine distributional RL, priori-
tized replay, and n-step updates (Dabney et al., 2018; Hessel
et al., 2018; Gruslys et al., 2018).
2.4. Risk in Reinforcement Learning
Distributional RL algorithms have been theoretically jus-
tiﬁed for the Wasserstein and Cram´er metrics (Bellemare
et al., 2017; Rowland et al., 2018), and learning the dis-
tribution over returns, in and of itself, empirically results
in signiﬁcant improvements to data efﬁciency, ﬁnal perfor-
mance, and stability (Bellemare et al., 2017; Dabney et al.,
2018; Gruslys et al., 2018; Barth-Maron et al., 2018). How-
ever, in each of these recent works the policy used was
based entirely on the mean of the return distribution, just
as in standard reinforcement learning. A natural question
arises: can we expand the class of policies using information
provided by the distribution over returns (i.e. to the class
of risk-sensitive policies)? Furthermore, when would this
larger policy class be beneﬁcial?
Here, ‘risk’ refers to the uncertainty over possible outcomes,
and risk-sensitive policies are those which depend upon
more than the mean of the outcomes. At this point, it is
important to highlight the difference between intrinsic un-
certainty, captured by the distribution over returns, and
parametric uncertainty, the uncertainty over the value es-
timate typically associated with Bayesian approaches such
as PSRL (Osband et al., 2013) and Kalman TD (Geist &
Pietquin, 2010). Distributional RL seeks to capture the

Implicit Quantile Networks for Distributional Reinforcement Learning
former, which classic approaches to risk are built upon1.
Expected utility theory states that if a decision policy is
consistent with a particular set of four axioms regarding
its choices then the decision policy behaves as though it is
maximizing the expected value of some utility function U
(von Neumann & Morgenstern, 1947),
π(x) = arg max
a
E
Z(x,a)[U(z)].
This is perhaps the most pervasive notion of risk-sensitivity.
A policy maximizing a linear utility function is called risk-
neutral, whereas concave or convex utility functions give
rise to risk-averse or risk-seeking policies, respectively.
Many previous studies on risk-sensitive RL adopt the util-
ity function approach (Howard & Matheson, 1972; Marcus
et al., 1997; Maddison et al., 2017).
A crucial axiom of expected utility is independence: given
random variables X, Y and Z, such that X ≻Y (X pre-
ferred over Y ), any mixture between X and Z is preferred to
the same mixture between Y and Z (von Neumann & Mor-
genstern, 1947). Stated in terms of the cumulative probabil-
ity functions, αFX +(1−α)FZ ≥αFY +(1−α)FZ, ∀α ∈
[0, 1].
This axiom in particular has troubled many re-
searchers because it is consistently violated by human be-
havior (Tversky & Kahneman, 1992). The Allais paradox
is a frequently used example of a decision problem where
people violate the independence axiom of expected utility
theory (Allais, 1990).
However, as Yaari (1987) showed, this axiom can be re-
placed by one in terms of convex combinations of outcome
values, instead of mixtures of distributions. Speciﬁcally,
if as before X ≻Y , then for any α ∈[0, 1] and random
variable Z, αF −1
X + (1 −α)F −1
Z
≥αF −1
Y
+ (1 −α)F −1
Z .
This leads to an alternate, dual, theory of choice than that
of expected utility. Under these axioms the decision policy
behaves as though it is maximizing a distorted expectation,
for some continuous monotonic function h:
π(x) = arg max
a
Z ∞
−∞
z ∂
∂z (h ◦FZ(x,a))(z) dz.
Such a function h is known as a distortion risk measure,
as it distorts the cumulative probabilities of the random
variable (Wang, 1996). That is, we have two fundamentally
equivalent approaches to risk-sensitivity. Either, we choose
a utility function and follow the expectation of this utility.
Or, we choose a reweighting of the distribution and compute
expectation under this distortion measure. Indeed, Yaari
(1987) further showed that these two functions are inverses
of each other. The choice between them amounts to a choice
1One exception is the recent work (Moerland et al., 2017) to-
wards combining both forms of uncertainty to improve exploration.
 
f
Actions
Returns
 
f
cos
⌧⇠β(·)
φ
Actions
Returns
z0z1z2z3z4
 
Actions
Returns
 
Actions
Returns
DQN
C51
IQN
QR-DQN
Figure 1. Network architectures for DQN and recent distributional
RL algorithms.
over whether the behavior should be invariant to mixing
with random events or to convex combinations of outcomes.
Distortion risk measures include, as special cases, cumula-
tive probability weighting used in cumulative prospect the-
ory (Tversky & Kahneman, 1992), conditional value at risk
(Chow & Ghavamzadeh, 2014), and many other methods
(Morimura et al., 2010b). Recently Majumdar & Pavone
(2017) argued for the use of distortion risk measures in
robotics.
3. Implicit Quantile Networks
We now introduce the implicit quantile network (IQN), a
deterministic parametric function trained to reparameterize
samples from a base distribution, e.g. τ ∼U([0, 1]), to
the respective quantile values of a target distribution. IQN
provides an effective way to learn an implicit representa-
tion of the return distribution, yielding a powerful function
approximator for a new DQN-like agent.
Let F −1
Z (τ) be the quantile function at τ ∈[0, 1] for the
random variable Z. For notational simplicity we write
Zτ := F −1
Z (τ), thus for τ ∼U([0, 1]) the resulting state-
action return distribution sample is Zτ(x, a) ∼Z(x, a).
We propose to model the state-action quantile function as
a mapping from state-actions and samples from some base
distribution, typically τ ∼U([0, 1]), to Zτ(x, a), viewed as
samples from the implicitly deﬁned return distribution.
Let β : [0, 1] →[0, 1] be a distortion risk measure, with
identity corresponding to risk-neutrality. Then, the distorted
expectation of Z(x, a) under β is given by
Qβ(x, a) :=
E
τ∼U([0,1])

Zβ(τ)(x, a)

.
Notice that the distorted expectation is equal to the ex-
pected value of F −1
Z(x,a) weighted by β, that is, Qβ =
R 1
0 F −1
Z (τ)dβ(τ). The immediate implication of this is that
for any β, there exists a sampling distribution for τ such that
the mean of Zτ is equal to the distorted expectation of Z

Implicit Quantile Networks for Distributional Reinforcement Learning
under β, that is, any distorted expectation can be represented
as a weighted sum over the quantiles (Dhaene et al., 2012).
Denote by πβ the risk-sensitive greedy policy
πβ(x) = arg max
a∈A
Qβ(x, a).
(1)
For two samples τ, τ ′ ∼U([0, 1]), and policy πβ, the sam-
pled temporal difference (TD) error at step t is
δτ,τ ′
t
= rt + γZτ ′(xt+1, πβ(xt+1)) −Zτ(xt, at).
(2)
Then, the IQN loss function is given by
L(xt, at, rt, xt+1) = 1
N ′
N
X
i=1
N ′
X
j=1
ρκ
τi

δ
τi,τ ′
j
t

,
(3)
where N and N ′ denote the respective number of iid sam-
ples τi, τ ′
j ∼U([0, 1]) used to estimate the loss. A cor-
responding sample-based risk-sensitive policy is obtained
by approximating Qβ in Equation 1 by K samples of
˜τ ∼U([0, 1]):
˜πβ(x) = arg max
a∈A
1
K
K
X
k=1
Zβ(˜τk)(x, a).
Implicit quantile networks differ from the approach of Dab-
ney et al. (2018) in two ways. First, instead of approximat-
ing the quantile function at n ﬁxed values of τ we approxi-
mate it with Zτ(x, a) ≈f(ψ(x), φ(τ))a for some differen-
tiable functions f, ψ, and φ. If we ignore the distributional
interpretation for a moment and view each Zτ(x, a) as a
separate action-value function, this highlights that implicit
quantile networks are a type of universal value function
approximator (UVFA) (Schaul et al., 2015). There may
be additional beneﬁts to implicit quantile networks beyond
the obvious increase in representational ﬁdelity. As with
UVFAs, we might hope that training over many different
τ’s (goals in the case of the UVFA) leads to better gener-
alization between values and improved sample complexity
than attempting to train each separately.
Second, τ, τ ′, and ˜τ are sampled from continuous, inde-
pendent, distributions. Besides U([0, 1]), we also explore
risk-sentive policies πβ, with non-linear β. The independent
sampling of each τ, τ ′ results in the sample TD errors being
decorrelated, and the estimated action-values go from being
the true mean of a mixture of n Diracs to a sample mean
of the implicit distribution deﬁned by reparameterizing the
sampling distribution via the learned quantile function.
3.1. Implementation
Consider the neural network structure used by the DQN
agent (Mnih et al., 2015). Let ψ: X →Rd be the function
computed by the convolutional layers and f : Rd →R|A|
the subsequent fully-connected layers mapping ψ(x) to the
estimated action-values, such that Q(x, a) ≈f(ψ(x))a. For
our network we use the same functions ψ and f as in DQN,
but include an additional function φ: [0, 1] →Rd comput-
ing an embedding for the sample point τ. We combine these
to form the approximation Zτ(x, a) ≈f(ψ(x) ⊙φ(τ))a,
where ⊙denotes the element-wise (Hadamard) product.
As the network for f is not particularly deep, we use the
multiplicative form, ψ ⊙φ, to force interaction between the
convolutional features and the sample embedding. Alter-
native functional forms, e.g. concatenation or a ‘residual’
function ψ ⊙(1 + φ), are conceivable, and φ(τ) can be
parameterized in different ways. To investigate these, we
compared performance across a number of architectural
variants on six Atari 2600 games (ASTERIX, ASSAULT,
BREAKOUT, MS.PACMAN, QBERT, SPACE INVADERS).
Full results are given in the Appendix. Despite minor varia-
tion in performance, we found the general approach to be
robust to the various choices. Based upon the results we
used the following function in our later experiments, for
embedding dimension n = 64:
φj(τ) := ReLU(
n−1
X
i=0
cos(πiτ)wij + bj).
(4)
After settling on a network architecture, we study the effect
of the number of samples, N and N ′, used in the estimate
terms of Equation 3.
We hypothesized that N, the number of samples of τ ∼
U([0, 1]), would affect the sample complexity of IQN, with
larger values leading to faster learning, and that with N = 1
one would potentially approach the performance of DQN.
This would support the hypothesis that the improved perfor-
mance of many distributional RL algorithms rests on their
effect as auxiliary loss functions, which would vanish in
the case of N = 1. Furthermore, we believed that N ′, the
number of samples of τ ′ ∼U([0, 1]), would affect the vari-
ance of the gradient estimates much like a mini-batch size
hyperparameter. Our prediction was that N ′ would have the
greatest effect on variance of the long-term performance of
the agent.
We used the same set of six games as before, with our
chosen architecture, and varied N, N ′ ∈{1, 8, 32, 64}. In
Figure 2 we report the average human-normalized scores on
the six games for each conﬁguration. Figure 2 (left) shows
the average performance over the ﬁrst ten million frames,
while (right) shows the average performance over the last
ten million (from 190M to 200M).
As expected, we found that N has a dramatic effect on early
performance, shown by the continual improvement in score
as the value increases. Additionally, we observed that N ′

Implicit Quantile Networks for Distributional Reinforcement Learning
N
N 0
N
Figure 2. Effect of varying N and N ′, the number of samples used
in the loss function in Equation 3. Figures show human-normalized
agent performance, averaged over six Atari games, averaged over
ﬁrst 10M frames of training (left) and last 10M frames of training
(right). Corresponding values for baselines: DQN (32, 253) and
QR-DQN (144, 1243).
affected performance very differently than expected: it had
a strong effect on early performance, but minimal impact
on long-term performance past N ′ = 8.
Overall, while using more samples for both distributions is
generally favorable, N = N ′ = 8 appears to be sufﬁcient
to achieve the majority of improvements offered by IQN
for long-term performance, with variation past this point
largely insigniﬁcant. To our surprise we found that even for
N = N ′ = 1, which is comparable to DQN in the number
of loss components, the longer term performance is still
quite strong (≈3× DQN).
In an informal evaluation, we did not ﬁnd IQN to be sensi-
tive to K, the number of samples used for the policy, and
have ﬁxed it at K = 32 for all experiments.
4. Risk-Sensitive Reinforcement Learning
In this section, we explore the effects of varying the distor-
tion risk measure, β, away from identity. This only affects
the policy, πβ, used both in Equation 2 and for acting in the
environment. As we have argued, evaluating under different
distortion risk measures is equivalent to changing the sam-
pling distribution for τ, allowing us to achieve various forms
of risk-sensitive policies. We focus on a handful of sampling
distributions and their corresponding distortion measures.
The ﬁrst one is the cumulative probability weighting param-
eterization proposed in cumulative prospect theory (Tversky
& Kahneman, 1992; Gonzalez & Wu, 1999):
CPW(η, τ) =
τ η
(τ η + (1 −τ)η)
1
η .
In particular, we use the parameter value η = 0.71 found
by Wu & Gonzalez (1996) to most closely match human
subjects. This choice is interesting as, unlike the others we
consider, it is neither globally convex nor concave. For small
values of τ it is locally concave and for larger values of τ it
becomes locally convex. Recall that concavity corresponds
to risk-averse and convexity to risk-seeking policies.
Second, we consider the distortion risk measure proposed by
Wang (2000), where Φ and Φ−1 are taken to be the standard
Normal cumulative distribution function and its inverse:
Wang(η, τ) = Φ(Φ−1(τ) + η).
For η < 0, this produces risk-averse policies and we in-
clude it due to its simple interpretation and ability to switch
between risk-averse and risk-seeking distortions.
Third, we consider a simple power formula for risk-averse
(η < 0) or risk-seeking (η > 0) policies:
Pow(η, τ) =
(
τ
1
1+|η| ,
if η ≥0
1 −(1 −τ)
1
1+|η| ,
otherwise
.
Finally, we consider conditional value-at-risk (CVaR):
CVaR(η, τ) = ητ.
CVaR has been widely studied in and out of reinforcement
learning (Chow & Ghavamzadeh, 2014). Its implementation
as a modiﬁcation to the sampling distribution of τ is partic-
ularly simple, as it changes τ ∼U([0, 1]) to τ ∼U([0, η]).
Another interesting sampling distribution, not included in
our experiments, is denoted Norm(η) and corresponds to τ
sampled by averaging η samples from U([0, 1]).
In Figure 3 (right) we give an example of a distribution
(Neutral) and how each of these distortion measures affects
the implied distribution due to changing the sampling distri-
bution of τ. Norm(3) and CPW(.71) reduce the impact of
the tails of the distribution, while Wang and CVaR heavily
shift the distribution mass towards the tails, creating a risk-
averse or risk-seeking preference. Additionally, while CVaR
entirely ignores all values corresponding to τ > η, Wang
gives these non-zero, but vanishingly small, probability.
By using these sampling distributions we can induce various
risk-sensitive policies in IQN. We evaluate these on the same
set of six Atari 2600 games previously used. Our algorithm
simply changes the policy to maximize the distorted expec-
tations instead of the usual sample mean. Figure 3 (left)
shows our results in this experiment, with average scores
reported under the usual, risk-neutral, evaluation criterion.
Intuitively, we expected to see a qualitative effect from
risk-sensitive training, e.g. strengthened exploration from a
risk-seeking objective. Although we did see qualitative dif-
ferences, these did not always match our expectations. For
two of the games, ASTERIX and ASSAULT, there is a very
signiﬁcant advantage to the risk-averse policies. Although
CPW tends to perform almost identically to the standard
risk-neutral policy, and the risk-seeking Wang(1.5) per-
forms as well or worse than risk-neutral, we ﬁnd that both
risk-averse policies improve performance over standard IQN.
However, we also observe that the more risk-averse of the

Implicit Quantile Networks for Distributional Reinforcement Learning
Assault
MsPacman
Asterix
QBert
Breakout
Space Invaders
Training Frames (Million)
Average Score
Training Frames (Million)
Training Frames (Million)
Average Score
-
-
-
-
-
-
-
Figure 3. Effects of various changes to the sampling distribution, that is various cumulative probability weightings.
two, CVaR(0.1), suffers some loss in performance on two
other games (QBERT and SPACE INVADERS).
Additionally, we note that the risk-seeking policy signif-
icantly underperforms the risk-neutral policy on three of
the six games. It remains an open question as to exactly
why we see improved performance for risk-averse policies.
There are many possible explanations for this phenomenon,
e.g. that risk-aversion encodes a heuristic to stay alive longer,
which in many games is correlated with increased rewards.
5. Full Atari-57 Results
Finally, we evaluate IQN on the full Atari-57 benchmark,
comparing with the state-of-the-art performance of Rainbow,
a distributional RL agent that combines several advances in
deep RL (Hessel et al., 2018), the closely related algorithm
QR-DQN (Dabney et al., 2018), prioritized experience re-
play DQN (Schaul et al., 2016), and the original DQN agent
(Mnih et al., 2015). Note that in this section we use the
risk-neutral variant of the IQN, that is, the policy of the IQN
agent is the regular ϵ-greedy policy with respect to the mean
of the state-action return distribution.
It is important to remember that Rainbow builds upon the
distributional RL algorithm C51 (Bellemare et al., 2017),
but also includes prioritized experience replay (Schaul et al.,
2016), Double DQN (van Hasselt et al., 2016), Dueling
Network architecture (Wang et al., 2016), Noisy Networks
(Fortunato et al., 2017), and multi-step updates (Sutton,
1988). In particular, besides the distributional update, n-
step updates and prioritized experience replay were found to
have signiﬁcant impact on the performance of Rainbow. Our
other competitive baseline is QR-DQN, which is currently
state-of-the-art for agents that do not combine distributional
updates, n-step updates, and prioritized replay.
Thus, between QR-DQN and the much more complex Rain-
bow we compare to the two most closely related, and best
performing, agents in published work. In particular, we
would expect that IQN would beneﬁt from the additional
enhancements in Rainbow, just as Rainbow improved sig-
niﬁcantly over C51.
Figure 4 shows the mean (left) and median (right) human-
normalized scores during training over the Atari-57 bench-
mark. IQN dramatically improves over QR-DQN, which
itself improves on many previously published results. At
100 million frames IQN has reached the same level of perfor-
mance as QR-DQN at 200 million frames. Table 1 gives a
comparison between the same methods in terms of their best,
human-normalized, scores per game under the 30 random
no-op start condition. These are averages over the given
number of seeds. Additionally, using human-starts, IQN
achieves 162% median human-normalized score, whereas
Rainbow reaches 153% (Hessel et al., 2018), see Table 2.
Mean
Median
Human Gap
Seeds
DQN
228%
79%
0.334
1
PRIOR.
434%
124%
0.178
1
C51
701%
178%
0.152
1
RAINBOW
1189%
230%
0.144
2
QR-DQN
864%
193%
0.165
3
IQN
1019%
218%
0.141
5
Table 1. Mean and median of scores across 57 Atari 2600 games,
measured as percentages of human baseline (Nair et al., 2015).
Scores are averages over number of seeds.
Human-starts (median)
DQN
PRIOR.
A3C
C51
RAINBOW
IQN
68%
128%
116%
125%
153%
162%
Table 2. Median human-normalized scores for human-starts.

Implicit Quantile Networks for Distributional Reinforcement Learning
Training Frames (Million)
Mean
Median
Training Frames (Million)
DQN
IQN
Prioritized DQN
QR-DQN
Rainbow
Human-Normalized Score
Figure 4. Human-normalized mean (left) and median (right) scores on Atari-57 for IQN and various other algorithms. Random seeds
shown as traces, with IQN averaged over 5, QR-DQN over 3, and Rainbow over 2 random seeds.
Finally, we took a closer look at the games in which each al-
gorithm continues to underperform humans, and computed,
on average, how far below human-level they perform2. We
refer to this value as the human-gap3 metric and give results
in Table 1. Interestingly, C51 outperforms QR-DQN in this
metric, and IQN outperforms all others. This shows that the
remaining gap between Rainbow and IQN is entirely from
games on which both algorithms are already super-human.
The games where the most progress in RL is needed happen
to be the games where IQN shows the greatest improvement
over QR-DQN and Rainbow.
6. Discussion and Conclusions
We have proposed a generalization of recent work based
around using quantile regression to learn the distribution
over returns of the current policy. Our generalization leads
to a simple change to the DQN agent to enable distribu-
tional RL, the natural integration of risk-sensitive policies,
and signiﬁcantly improved performance over existing meth-
ods. The IQN algorithm provides, for the ﬁrst time, a fully
integrated distributional RL agent without prior assumptions
on the parameterization of the return distribution.
IQN can be trained with as little as a single sample from
each state-action value distribution, or as many as computa-
tional limits allow to improve the algorithm’s data efﬁciency.
Furthermore, IQN allows us to expand the class of control
policies to a large class of risk-sensitive policies connected
to distortion risk measures. Finally, we show substantial
gains on the Atari-57 benchmark over QR-DQN, and even
halving the distance between QR-DQN and Rainbow.
Despite the signiﬁcant empirical successes in this paper
2Details of how this is computed can be found in the Appendix.
3Thanks to Joseph Modayil for proposing this metric.
there are many areas in need of additional theoretical analy-
sis. We highlight a few particularly relevant open questions
we were unable to address in the present work. First, sample-
based convergence results have been recently shown for a
class of categorical distributional RL algorithms (Rowland
et al., 2018). Could existing sample-based RL convergence
results be extended to the QR-based algorithms?
Second, can the contraction mapping results for a ﬁxed grid
of quantiles given by Dabney et al. (2018) be extended to
the more general class of approximate quantile functions
studied in this work? Finally, and particularly salient to
our experiments with distortion risk measures, theoretical
guarantees for risk-sensitive RL have been building over
recent years, but have been largely limited to special cases
and restricted classes of risk-sensitive policies. Can the
convergence of the distribution of returns under the Bellman
operator be leveraged to show convergence to a ﬁxed-point
in distorted expectations? In particular, can the control
results of Bellemare et al. (2017) be expanded to cover
some class of risk-sensitive policies?
There remain many intriguing directions for future research
into distributional RL, even on purely empirical fronts. Hes-
sel et al. (2018) recently showed that distributional RL
agents can be signiﬁcantly improved, when combined with
other techniques. Creating a Rainbow-IQN agent could
yield even greater improvements on Atari-57. We also recall
the surprisingly rich return distributions found by Barth-
Maron et al. (2018), and hypothesize that the continuous
control setting may be a particularly fruitful area for the
application of distributional RL in general, and IQN in par-
ticular.

Implicit Quantile Networks for Distributional Reinforcement Learning
References
Allais, M. Allais paradox. In Utility and Probability, pp.
3–9. Springer, 1990.
Arjovsky, M., Chintala, S., and Bottou, L. Wasserstein
Generative Adversarial Networks.
In Proceedings of
the 34th International Conference on Machine Learning
(ICML), 2017.
Azar, M. G., Munos, R., and Kappen, H. J. On the sample
complexity of reinforcement learning with a generative
model. In Proceedings of the International Conference
on Machine Learning (ICML), 2012.
Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W.,
Horgan, D., TB, D., Muldal, A., Heess, N., and Lillicrap,
T. Distributional policy gradients. In Proceedings of the
International Conference on Learning Representations
(ICLR), 2018.
Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M.
The Arcade Learning Environment: an evaluation plat-
form for general agents. Journal of Artiﬁcial Intelligence
Research, 47:253–279, 2013.
Bellemare, M. G., Dabney, W., and Munos, R. A distribu-
tional perspective on reinforcement learning. Proceedings
of the 34th International Conference on Machine Learn-
ing (ICML), 2017.
Bellman, R. E. Dynamic Programming. Princeton Univer-
sity Press, Princeton, NJ, 1957.
Bousquet, O., Gelly, S., Tolstikhin, I., Simon-Gabriel, C.-J.,
and Schoelkopf, B. From optimal transport to gener-
ative modeling: the vegan cookbook. arXiv preprint
arXiv:1705.07642, 2017.
Chow, Y. and Ghavamzadeh, M. Algorithms for CVaR opti-
mization in MDPs. In Advances in Neural Information
Processing Systems, pp. 3509–3517, 2014.
Dabney, W., Rowland, M., Bellemare, M. G., and Munos,
R. Distributional reinforcement learning with quantile
regression. In Proceedings of the AAAI Conference on
Artiﬁcial Intelligence, 2018.
Dhaene, J., Kukush, A., Linders, D., and Tang, Q. Remarks
on quantiles and distortion risk measures.
European
Actuarial Journal, 2(2):319–328, 2012.
Fortunato, M., Azar, M. G., Piot, B., Menick, J., Osband, I.,
Graves, A., Mnih, V., Munos, R., Hassabis, D., Pietquin,
O., et al. Noisy networks for exploration. arXiv preprint
arXiv:1706.10295, 2017.
Geist, M. and Pietquin, O. Kalman temporal differences.
Journal of Artiﬁcial Intelligence Research, 39:483–532,
2010.
Gonzalez, R. and Wu, G. On the shape of the probability
weighting function. Cognitive Psychology, 38(1):129–
166, 1999.
Gruslys, A., Dabney, W., Azar, M. G., Piot, B., Bellemare,
M. G., and Munos, R. The Reactor: a fast and sample-
efﬁcient actor-critic agent for reinforcement learning. In
Proceedings of the International Conference on Learning
Representations (ICLR), 2018.
Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Os-
trovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M.,
and Silver, D. Rainbow: combining improvements in
deep reinforcement learning. In Proceedings of the AAAI
Conference on Artiﬁcial Intelligence, 2018.
Howard, R. A. and Matheson, J. E. Risk-sensitive markov
decision processes. Management Science, 18(7):356–369,
1972.
Huber, P. J. Robust estimation of a location parameter. The
Annals of Mathematical Statistics, 35(1):73–101, 1964.
Jaquette, S. C. Markov decision processes with a new opti-
mality criterion: discrete time. The Annals of Statistics, 1
(3):496–505, 1973.
Koenker, R. Quantile Regression. Cambridge University
Press, 2005.
Lattimore, T. and Hutter, M. PAC bounds for discounted
MDPs.
In International Conference on Algorithmic
Learning Theory, pp. 320–334. Springer, 2012.
Maddison, C. J., Lawson, D., Tucker, G., Heess, N., Doucet,
A., Mnih, A., and Teh, Y. W. Particle value functions.
arXiv preprint arXiv:1703.05820, 2017.
Majumdar, A. and Pavone, M. How should a robot assess
risk? Towards an axiomatic theory of risk in robotics.
arXiv preprint arXiv:1710.11040, 2017.
Marcus, S. I., Fern´andez-Gaucherand, E., Hern´andez-
Hernandez, D., Coraluppi, S., and Fard, P. Risk sensitive
markov decision processes. In Systems and Control in
the Twenty-First Century, pp. 263–279. Springer, 1997.
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness,
J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidje-
land, A. K., Ostrovski, G., et al. Human-level control
through deep reinforcement learning. Nature, 518(7540):
529–533, 2015.
Moerland, T. M., Broekens, J., and Jonker, C. M. Efﬁcient
exploration with double uncertain value networks. arXiv
preprint arXiv:1711.10789, 2017.

Implicit Quantile Networks for Distributional Reinforcement Learning
Morimura, T., Hachiya, H., Sugiyama, M., Tanaka, T., and
Kashima, H. Parametric return density estimation for
reinforcement learning. In Proceedings of the Conference
on Uncertainty in Artiﬁcial Intelligence (UAI), 2010a.
Morimura, T., Sugiyama, M., Kashima, H., Hachiya, H.,
and Tanaka, T. Nonparametric return distribution approx-
imation for reinforcement learning. In Proceedings of
the 27th International Conference on Machine Learning
(ICML), pp. 799–806, 2010b.
M¨uller, A. Integral probability metrics and their generating
classes of functions. Advances in Applied Probability, 29
(2):429–443, 1997.
Nair, A., Srinivasan, P., Blackwell, S., Alcicek, C., Fearon,
R., De Maria, A., Panneershelvam, V., Suleyman, M.,
Beattie, C., and Petersen, S. e. a. Massively parallel meth-
ods for deep reinforcement learning. In ICML Workshop
on Deep Learning, 2015.
Osband, I., Russo, D., and Van Roy, B. (more) efﬁcient
reinforcement learning via posterior sampling. In Ad-
vances in Neural Information Processing Systems, pp.
3003–3011, 2013.
Puterman, M. L. Markov Decision Processes: Discrete
Stochastic Dynamic Programming. John Wiley & Sons,
Inc., 1994.
Rowland, M., Bellemare, M. G., Dabney, W., Munos, R.,
and Teh, Y. W. An analysis of categorical distributional
reinforcement learning. In AISTATS, 2018.
Schaul, T., Horgan, D., Gregor, K., and Silver, D. Uni-
versal value function approximators. In International
Conference on Machine Learning, pp. 1312–1320, 2015.
Schaul, T., Quan, J., Antonoglou, I., and Silver, D. Prior-
itized experience replay. In Proceedings of the Interna-
tional Conference on Learning Representations (ICLR),
2016.
Sobel, M. J. The variance of discounted markov decision
processes. Journal of Applied Probability, 19(04):794–
802, 1982.
Sutton, R. S. Learning to predict by the methods of temporal
differences. Machine Learning, 3(1):9–44, 1988.
Tolstikhin, I., Bousquet, O., Gelly, S., and Schoelkopf,
B.
Wasserstein auto-encoders.
arXiv preprint
arXiv:1711.01558, 2017.
Tversky, A. and Kahneman, D. Advances in prospect theory:
cumulative representation of uncertainty. Journal of Risk
and Uncertainty, 5(4):297–323, 1992.
van Hasselt, H., Guez, A., and Silver, D. Deep reinforce-
ment learning with double Q-learning. In Proceedings of
the AAAI Conference on Artiﬁcial Intelligence, 2016.
von Neumann, J. and Morgenstern, O. Theory of Games and
Economic Behavior. Princeton University Press, 1947.
Wang, S. Premium calculation by transforming the layer
premium density. ASTIN Bulletin: The Journal of the
IAA, 26(1):71–92, 1996.
Wang, S. S. A class of distortion operators for pricing ﬁnan-
cial and insurance risks. Journal of Risk and Insurance,
pp. 15–36, 2000.
Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot,
M., and de Freitas, N. Dueling network architectures
for deep reinforcement learning. In Proceedings of the
International Conference on Machine Learning (ICML),
2016.
Watkins, C. J. C. H. Learning from delayed rewards. PhD
thesis, King’s College, Cambridge, 1989.
White, D. J. Mean, variance, and probabilistic criteria in
ﬁnite markov decision processes: a review. Journal of
Optimization Theory and Applications, 56(1):1–29, 1988.
Wu, G. and Gonzalez, R.
Curvature of the probability
weighting function. Management Science, 42(12):1676–
1690, 1996.
Yaari, M. E. The dual theory of choice under risk. Econo-
metrica: Journal of the Econometric Society, pp. 95–115,
1987.
Yu, K.-T., Bauza, M., Fazeli, N., and Rodriguez, A. More
than a million ways to be pushed. a high-ﬁdelity exper-
imental dataset of planar pushing. In Intelligent Robots
and Systems (IROS), 2016 IEEE/RSJ International Con-
ference on, pp. 30–37. IEEE, 2016.

Implicit Quantile Networks for Distributional Reinforcement Learning
Appendix
Architecture and Hyperparameters
We considered multiple architectural variants for parame-
terizing an IQN. All of these build on the Q-network of a
regular DQN (Mnih et al., 2015), which can be seen as the
composition of a convolutional stack ψ: X →Rd and an
MLP f : Rd →R|A|, and extend it by an embedding of
the sample point, φ: [0, 1] →Rd, and a merging function
m: Rd × Rd →Rd, resulting in the function
IQN(x, τ) = f(m(ψ(x), φ(τ))).
For the embedding φ, we considered a number of variants: a
learned linear embedding, a learned MLP embedding with a
single hidden layer of size n, and a learned linear function of
n cosine basis functions of the form cos(πiτ), i = 1, . . . , n.
Each of those was followed by either a ReLU or sigmoid
nonlinearity.
For the merging function m, the simplest choice would
be a simple vector concatenation of ψ(x) and φ(τ). Note
however, that the MLP f which takes in the output of m and
outputs the action-value quantiles, only has a single hidden
layer in the DQN network. Therefore, to force a sufﬁciently
early interaction between the two representations, we also
considered a multiplicative function m(ψ, φ) = ψ ⊙φ,
where ⊙denotes the element-wise (Hadamard) product of
two vectors, as well as a ‘residual’ function m(ψ, φ) =
ψ ⊙(1 + φ).
Early experiments showed that a simple linear embedding
of τ was insufﬁcient to achieve good performance, and the
residual version of m didn’t show any marked difference
to the multiplicative variant, so we do not include results
for these here. For the other conﬁgurations, Figure 5 shows
pairwise comparisons between 1) a cosine basis function
embedding and a completely learned MLP embedding, 2)
an embedding size (hidden layer size or number of cosine
basis elements) 32 and 64, 3) ReLU and sigmoid nonlinear-
ity following the embedding, and 4) concatenation and a
Figure 5. Comparison of architectural variants.
multiplicative interaction between ψ(x) and φ(τ).
Each comparison ‘violin plot’ can be understood as a
marginalization over the other variants of the architecture,
with the human-normalized performance at the end of train-
ing, averaged across six Atari 2600 games, on the y-axis.
Each white dot corresponds to a conﬁguration (each repre-
sented by two seeds), the black dots show the position of our
preferred conﬁguration. The width of the colored regions
corresponds to a kernel density estimate of the number of
conﬁgurations at each performance level.
Our ﬁnal choice is a multiplicative interaction with a linear
function of a cosine embedding, with n = 64 and a ReLU
nonlinearity (see Equation 4), as this conﬁguration yielded
the highest performance consistently over multiple seeds.
Also noteworthy is the overall robustness of the approach to
these variations: most of the conﬁgurations consistently out-
perform the QR-DQN baseline shown as a grey horizontal
line for comparison.
We give pseudo-code for the IQN loss in Algorithm 1. All
other hyperparameters for this agent correspond to the ones
used by Dabney et al. (2018). In particular, the Bellman tar-
get is computed using a target network. Notice that IQN will
generally be more computationally expensive per-sample
than QR-DQN. However, in practice IQN requires many
fewer samples per update than QR-DQN so that the actual
running times are comparable.
Algorithm 1 Implicit Quantile Network Loss
Require: N, N ′, K, κ and functions β, Z
input x, a, r, x′, γ ∈[0, 1)
# Compute greedy next action
a∗←arg maxa′ 1
K
PK
k Z˜τk(x′, a′),
˜τk ∼β(·)
# Sample quantile thresholds
τi, τ ′
j ∼U([0, 1]),
1 ≤i ≤N, 1 ≤j ≤N ′
# Compute distributional temporal differences
δij ←r + γZτ ′
j(x′, a∗) −Zτi(x, a),
∀i, j
# Compute Huber quantile loss
output PN
i=1 Eτ ′ 
ρκ
τi(δij)


Implicit Quantile Networks for Distributional Reinforcement Learning
Evaluation
The human-normalized scores reported in this paper are
given by the formula (van Hasselt et al., 2016; Dabney et al.,
2018)
score = agent −random
human −random,
where agent, human and random are the per-game raw
scores (undiscounted returns) for the given agent, a refer-
ence human player, and random agent baseline (Mnih et al.,
2015).
The ‘human-gap’ metric referred to at the end of Section 5
builds on the human-normalized score, but emphasizes the
remaining improvement for the agent to reach super-human
performance. It is given by gap = max(1 −score, 0), with
a value of 1 corresponding to random play, and a value of
0 corresponding to super-human level of performance. To
avoid degeneracies in the case of human < random, the
quantity is being clipped above at 1.

Implicit Quantile Networks for Distributional Reinforcement Learning
DQN
IQN
QR-DQN-1
Rainbow
Figure 6. Complete Atari-57 training curves.

Implicit Quantile Networks for Distributional Reinforcement Learning
GAMES
RANDOM
HUMAN
DQN
PRIOR. DUEL.
QR-DQN
IQN
Alien
227.8
7,127.7
1,620.0
3,941.0
4,871
7,022
Amidar
5.8
1,719.5
978.0
2,296.8
1,641
2,946
Assault
222.4
742.0
4,280.4
11,477.0
22,012
29,091
Asterix
210.0
8,503.3
4,359.0
375,080.0
261,025
342,016
Asteroids
719.1
47,388.7
1,364.5
1,192.7
4,226
2,898
Atlantis
12,850.0
29,028.1
279,987.0
395,762.0
971,850
978,200
Bank Heist
14.2
753.1
455.0
1,503.1
1,249
1,416
Battle Zone
2,360.0
37,187.5
29,900.0
35,520.0
39,268
42,244
Beam Rider
363.9
16,926.5
8,627.5
30,276.5
34,821
42,776
Berzerk
123.7
2,630.4
585.6
3,409.0
3,117
1,053
Bowling
23.1
160.7
50.4
46.7
77.2
86.5
Boxing
0.1
12.1
88.0
98.9
99.9
99.8
Breakout
1.7
30.5
385.5
366.0
742
734
Centipede
2,090.9
12,017.0
4,657.7
7,687.5
12,447
11,561
Chopper Command
811.0
7,387.8
6,126.0
13,185.0
14,667
16,836
Crazy Climber
10,780.5
35,829.4
110,763.0
162,224.0
161,196
179,082
Defender
2,874.5
18,688.9
23,633.0
41,324.5
47,887
53,537
Demon Attack
152.1
1,971.0
12,149.4
72,878.6
121,551
128,580
Double Dunk
-18.6
-16.4
-6.6
-12.5
21.9
5.6
Enduro
0.0
860.5
729.0
2,306.4
2,355
2,359
Fishing Derby
-91.7
-38.7
-4.9
41.3
39.0
33.8
Freeway
0.0
29.6
30.8
33.0
34.0
34.0
Frostbite
65.2
4,334.7
797.4
7,413.0
4,384
4,324
Gopher
257.6
2,412.5
8,777.4
104,368.2
113,585
118,365
Gravitar
173.0
3,351.4
473.0
238.0
995
911
H.E.R.O.
1,027.0
30,826.4
20,437.8
21,036.5
21,395
28,386
Ice Hockey
-11.2
0.9
-1.9
-0.4
-1.7
0.2
James Bond
29.0
302.8
768.5
812.0
4,703
35,108
Kangaroo
52.0
3,035.0
7,259.0
1,792.0
15,356
15,487
Krull
1,598.0
2,665.5
8,422.3
10,374.4
11,447
10,707
Kung-Fu Master
258.5
22,736.3
26,059.0
48,375.0
76,642
73,512
Montezumas Revenge
0.0
4,753.3
0.0
0.0
0.0
0.0
Ms. Pac-Man
307.3
6,951.6
3,085.6
3,327.3
5,821
6,349
Name This Game
2,292.3
8,049.0
8,207.8
15,572.5
21,890
22,682
Phoenix
761.4
7,242.6
8,485.2
70,324.3
16,585
56,599
Pitfall!
-229.4
6,463.7
-286.1
0.0
0.0
0.0
Pong
-20.7
14.6
19.5
20.9
21.0
21.0
Private Eye
24.9
69,571.3
146.7
206.0
350
200
Q*Bert
163.9
13,455.0
13,117.3
18,760.3
572,510
25,750
River Raid
1,338.5
17,118.0
7,377.6
20,607.6
17,571
17,765
Road Runner
11.5
7,845.0
39,544.0
62,151.0
64,262
57,900
Robotank
2.2
11.9
63.9
27.5
59.4
62.5
Seaquest
68.4
42,054.7
5,860.6
931.6
8,268
30,140
Skiing
-17,098.1
-4,336.9
-13,062.3
-19,949.9
-9,324
-9,289
Solaris
1,236.3
12,326.7
3,482.8
133.4
6,740
8,007
Space Invaders
148.0
1,668.7
1,692.3
15,311.5
20,972
28,888
Star Gunner
664.0
10,250.0
54,282.0
125,117.0
77,495
74,677
Surround
-10.0
6.5
-5.6
1.2
8.2
9.4
Tennis
-23.8
-8.3
12.2
0.0
23.6
23.6
Time Pilot
3,568.0
5,229.2
4,870.0
7,553.0
10,345
12,236
Tutankham
11.4
167.6
68.1
245.9
297
293
Up and Down
533.4
11,693.2
9,989.9
33,879.1
71,260
88,148
Venture
0.0
1,187.5
163.0
48.0
43.9
1,318
Video Pinball
16,256.9
17,667.9
196,760.4
479,197.0
705,662
698,045
Wizard Of Wor
563.5
4,756.5
2,704.0
12,352.0
25,061
31,190
Yars Revenge
3,092.9
54,576.9
18,098.9
69,618.1
26,447
28,379
Zaxxon
32.5
9,173.3
5,363.0
13,886.0
13,112
21,772
Figure 7. Raw scores for a single seed across all games, starting with 30 no-op actions. Reference values from (Wang et al., 2016).
