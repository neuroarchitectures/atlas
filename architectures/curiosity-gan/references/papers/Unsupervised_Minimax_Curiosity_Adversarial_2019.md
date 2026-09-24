# Unsupervised Minimax Curiosity Adversarial 2019

> Source: `Unsupervised_Minimax_Curiosity_Adversarial_2019.pdf`

---

                                                          Unsupervised Minimax:
                                                Adversarial Curiosity, Generative Adversarial
                                                 Networks, and Predictability Minimization


                                                                               Jürgen Schmidhuber
                                                                             The Swiss AI Lab, IDSIA




arXiv:1906.04493v1 [cs.NE] 11 Jun 2019
                                                                            USI & SUPSI, Manno-Lugano
                                                                                   Switzerland



                                                                                      Abstract

                                                  Generative Adversarial Networks (GANs) learn to model data distributions through
                                                  two unsupervised neural networks, each minimizing the objective function maxi-
                                                  mized by the other. We relate this game theoretic strategy to earlier neural networks
                                                  playing unsupervised minimax games. (i) GANs can be formulated as a special case
                                                  of Adversarial Curiosity (1990) based on a minimax duel between two networks,
                                                  one generating data through its probabilistic actions, the other predicting conse-
                                                  quences thereof. (ii) We correct a previously published claim that Predictability
                                                  Minimization (PM, 1990s) is not based on a minimax game. PM models data distri-
                                                  butions through a neural encoder that maximizes the objective function minimized
                                                  by a neural predictor of the code components.


                                         1   Introduction
                                         Computer science has a rich history of problem solving through computational procedures seeking
                                         to minimize an objective function maximized by another procedure. For example, chess programs
                                         date back to 1945 [87], and for many decades have successfully used a recursive minimax procedure
                                         with continually shrinking look-ahead, e.g., [83]. Game theory of adversarial players originated in
                                         1944 [35]. In the field of machine learning, early adversarial settings include reinforcement learners
                                         playing against themselves [45], or the evolution of parasites in predator-prey games, e.g., [23, 72].
                                         Since 1990, adversarial techniques have also been employed in the field of unsupervised artificial
                                         neural networks (NNs)—see Sec. 2. In such settings, a single agent has two separate learning NNs.
                                         Without a teacher, and without external reward for achieving user-defined goals, the first NN somehow
                                         generates data. The second NN learns to predict consequences or properties of the generated outputs,
                                         minimizing its error. The first NN maximizes the objective function minimized by the second NN,
                                         effectively trying to generate data from which the second NN can learn more.
                                         A recent example of such an unsupervised minimax game is embodied by the Generative Adversarial
                                         Network (GAN). The terminology was introduced at NIPS 2014 [19]. Basic ideas of certain GAN
                                         types were published (albeit without peer review) in 2010 [39].
                                         In what follows, let m, n, q, k denote positive integer constants. GANs learn to model (probability
                                         distributions over) data through two NNs that fight each other. A typical GAN samples codes z ∈ Rm
                                         from some given probability distribution, e.g., a Gaussian. A generator NN transforms the codes
                                         into patterns x ∈ Rn . Its goal is to create patterns similar to (or drawn from the same probability
                                         distribution as) those in a user-given set X of k training patterns X = {x1 , x2 , . . . , xk ∈ Rn }.
                                         A separate predictor NN (called the discriminator network) receives patterns as inputs, and is
                                         trained to predict via a single real-valued output unit (active in the interval [0, 1]) whether they
                                         stem from X (output target 1) or not (output target 0). The generator NN, however, is trained to
maximize the objective function minimized by the predictor. This motivates the generator to produce
more and more realistic patterns. GANs and related approaches are now widely used and studied,
e.g., [42, 12, 25, 40, 1] [17, 32, 6, 79].




2   GANs as special cases of Adversarial Curiosity (1990)


GANs are quite different from early adversarial machine learning settings of the 1950s [45] which
neither involved unsupervised NNs nor were about modeling user-given data (nor used gradient
descent).
GANs are quite related, however, to the first unsupervised adversarial NNs of 1990 used to implement
curiosity [46, 50] in the general context of exploration in Reinforcement Learning (RL) [28, 77, 84,
64]. We will refer to this approach as Adversarial Curiosity (AC) of 1990, or AC1990 for short.
In the AC context, the first NN is often called the controller C. C may interact with an environment
through sequences of interactions called trials or episodes. During the execution of a single interaction
of any given trial, C generates an output vector x ∈ Rn . This may influence an environment, which
produces a reaction to x in form of an observation y ∈ Rq . In turn, y may affect C’s inputs during the
next interaction if there is any.
In the first variant of AC1990 [46, 50], C is recurrent, and thus a general purpose computer. Some of
C’s adaptive recurrent units are mean and variance-generating Gaussian units, such that C can become
a generative model—see Section “Explicit Random Actions versus Imported Randomness” [46] (see
also [51, 85]). What these stochastic units do can be equivalently accomplished by having C perceive
pseudorandom numbers or noise, like the generator NNs of GANs [19].
To compute an output action during an interaction, C updates all its NN unit activations for several
discrete time steps in a row—see Section “More Network Ticks than Environmental Ticks” [46]. In
principle, this allows for computing highly nonlinear, stochastic mappings from environmental inputs
(if there are any) and/or from internal “noise” to outputs.
The second NN is called the world model M [46, 47, 51, 22]. In the first variant of AC1990 [46, 50],
M is also recurrent, for reasons of generality. M receives C’s outputs x ∈ Rn as inputs and predicts
their visible environmental effects or consequences y ∈ Rq .
According to AC1990, M minimizes its prediction errors, thus becoming a better predictor. In absence
of external reward, however, the adversarial C tries to find actions that maximize the errors of M: M’s
errors are the intrinsic rewards of C. Hence C maximizes the errors that M minimizes. The loss of M
is the gain of C.
Without external reward, C is thus intrinsically motivated to invent novel action sequences or
experiments that yield data that M still finds surprising, until the data becomes familiar and eventually
boring.
The 1990 paper [46] describes gradient-based learning methods for both C and M. In particular,
backpropagation [31] through the model M down into the controller C (whose outputs are inputs to
M) is used to compute weight changes for C [82, 81, 36, 27, 46]. This is closely related to how a
GAN generator NN can be trained by backpropagation through its discriminator NN. Furthermore,
the concept of backpropagation through random number generators [85] is used to derive error
signals even for those units of C that are stochastic [46].
However, the original AC1990 paper points out that the basic ideas of AC are not limited to particular
learning algorithms—see Section “Implementing Dynamic Curiosity and Boredom” [46]. Compare
more recent summaries and numerous later variants and extensions of AC1990’s simple but powerful
exploration principle [59, 62], which inspired much later work, e.g., [74, 41, 62, 7].
To summarize, unsupervised minimax-based neural networks of the previous millennium (now often
called CM systems [65]) were both adversarial and generative, stochastically generating actions or
experiments yielding data, not only for stationary patterns but also for pattern sequences, even for the
general case of RL, and even for recurrent NN-based RL in partially observable environments [46, 50].


                                                   2
2.1   Which environment makes AC1990 a GAN?

For illustrative purposes, let us now formulate a GAN variant of 2014 [19] as a special case of a
curious CM system as in Sec. 2 above, where each sequence of interactions of the CM system with
its environment (each trial) is limited to a single interaction, like in bandit problems [44, 18, 3, 2].
What kind of partially observable environment makes AC1990 an image-generating GAN? The
environment must contain a representation of the user-given training set of “real” images X =
{x1 , x2 , . . . , xk ∈ Rn } (see Sec. 1). X is not directly visible to C and M, but its properties are
probed by AC1990 through GAN-like actions or experiments.
In the beginning of any given trial, the activations of all units in C and M are reset. C is blind (there is
no input from the environment). Using its internal stochastic units [46, 51](Sec. 2), C then computes
a single output x ∈ Rn , which is interpreted as a “fake” image. In a pre-wired fraction of all cases, x
is replaced by a randomly selected “real” image from X (the simple default exploration policy of
traditional RL chooses a random action in a fixed fraction of all cases [28, 77, 84]). This ensures that
M will see both fake and real images.
The environment will react to output action x and return as its effect a binary observation y ∈ R,
where y = 1 if the image is real, and y = 0 otherwise.
As always in AC1990-like systems, M now takes C’s output x as an input, and predicts its environ-
mental effect y, in that case a single bit of information, 1 or 0. As always, M learns by minimizing its
prediction errors. However, as always in absence of external reward, the adversarial C is learning
to generate data that maximizes the error minimized by M. M’s loss is C’s negative loss. That is, M
behaves essentially like the discriminator of a GAN, and C like the generator.1
Unlike AC1990 [46] and the GAN of 2014 [19], the GAN of 2010 [39] (now known as a conditional
GAN or cGAN [34]) does not have an internal source of randomness. Instead, such cGANs depend
on sufficiently diverse inputs from the environment. cGANs also can be formulated as special cases
of AC1990: cGAN-like additional environmental inputs just mean that the controller C of AC1990 is
not blind any more like in the example above with the GAN of 2014 [19].
Like the first version of AC1990 [46], the cGAN of 2010 [39] minimaxed Least Squares errors. This
was later called LSGANs [33].
The first variant of AC1990 [46, 50] also generalized to the case of recurrent NNs a well-known
way [82, 81, 36, 38, 27, 68] of using a differentiable world model M to approximate gradients for
C’s parameters even when environmental rewards are non-differentiable functions of C’s actions. In
the simple differentiable GAN environment above, however, there are no such complications, since
the rewards of C (the 1-dimensional errors of M) are differentiable functions of C’s outputs. That is,
standard backpropagation [31] can directly compute the gradients of C’s parameters with respect to
C’s rewards.
It should be emphasized though that AC1990 has much broader applicability [74, 41, 62, 7] than
the GAN-like special cases above. In particular, C may sequentially interact with the environment
for a long time, producing a sequence of environment-manipulating outputs resulting in complex
environmental constructs. For example, C may trigger actions that generate brush strokes on a
canvas, incrementally refining a painting over time, e.g., [21, 16, 86, 24, 37]. Similarly, M may
sequentially predict many other aspects of the environment besides the single bit of information in the
GAN-like setup above. General AC1990 is about unsupervised RL agents that actively shape their
observation streams through their own actions, setting themselves their own goals through intrinsic
rewards, exploring the world by inventing their own action sequences or experiments, to discover
novel, previously unknown predictability in the data generated by the experiments.

    1
      In the GAN-like AC1990 setup above, real images are produced in a pre-wired fraction of all cases. However,
one could easily give C the freedom to decide by itself to focus on particular real images that M finds still
difficult to learn. For example, one could employ the following procedure: once C has generated a fake image
x̂ ∈ Rn , and the activation of a special hidden unit of C is above a given threshold, say, 0.5, then x̂ is replaced
by the pattern in X most similar to x̂, according to some similarity measure. In this case, C is not only motivated
to learn to generate almost realistic fake images that are still hard to classify by M, but also to address and focus
on those real images that are still hard on M. This may be useful as C sometimes may find it easier to fool M by
sending it a particular real image, rather than a fake image. To our knowledge, however, this is rarely done with
standard GANs.


                                                         3
Since the GAN-like environment above is restricted to a teacher-given set X of patterns and a
procedure deciding whether a given pattern is in X, the teacher will find it rather easy to evaluate the
quality of C’s X-imitating behavior. In this sense the GAN setting is “more” supervised than certain
other applications of AC1990, which may be “highly" unsupervised in the sense that C may have
much more freedom when it comes to selecting environment-affecting actions.


3   Improvements of AC1990

Numerous improvements of the original AC1990 [46, 50] are summarized in more recent surveys [59,
62]. Let us focus here on a first important improvement of 1991.
The errors of AC1990’s M (to be minimized) are the rewards of its C (to be maximized). This
makes for a fine exploration strategy in many deterministic environments. In stochastic environments,
however, this might fail. C might learn to focus on those parts of the environment where M can always
get high prediction errors due to randomness, or due to computational limitations of M. For example,
an agent controlled by C might get stuck in front of a TV screen showing highly unpredictable white
noise, e.g., [62] (see also [7]).
Therefore, as pointed out in 1991, in stochastic environments, C’s reward should not be the errors of
M, but (an approximation of) the first derivative of M’s errors across subsequent training iterations,
that is, M’s improvements [48, 60]. As a consequence, despite M’s high errors in front of the noisy
TV screen above, C won’t get rewarded for getting stuck there, simply because M’s errors won’t
improve. Both the totally predictable and the fundamentally unpredictable will get boring.
This insight led to lots of follow-up work [62]. For example, one particular RL approach for AC
in stochastic environments was published in 1995 [76]. A simple M learned to predict or estimate
the probabilities of the environment’s possible responses, given C’s actions. After each interaction
with the environment, C’s reward was the KL-Divergence [30] between M’s estimated probability
distributions before and after the resulting new experience (the information gain) [76]. (This was later
also called Bayesian Surprise [26]; compare earlier work on information gain and its maximization
without NNs [73, 15].)
AC1990’s above-mentioned limitations in probabilistic environments, however, are not an issue
in the simple GAN-like setup of Sec. 2.1, because there the environmental reactions are totally
deterministic: For each image-generating action of C, there is a unique deterministic binary response
from the environment stating whether the generated image is in X or not.
Hence it is not obvious that above-mentioned improvements of AC1990 hold promise also for GANs.


4   Adversarial brains bet on outcomes of probabilistic programs (AC1997)

Of particular interest in the context of the present paper is one more advanced adversarial approach
to curious exploration of 1997 [55, 56, 58], referred to as AC1997.
In AC1997, a single agent has two dueling, reward-maximizing policies called the left brain and the
right brain. Each policy is a modifiable probability distribution over programs running on a general
purpose computer. Experiments are programs sampled in a collaborative way that is influenced by
both brains. Each experiment specifies how to execute an instruction sequence (which may affect both
the environment and the agent’s internal state), and how to compute the outcome of the experiment
through instructions implementing a computable function (possibly resulting in an internal binary
yes/no classification) of the observation sequence triggered by the experiment. The modifiable
parameters of both brains are instruction probabilities. They can be accessed and manipulated through
programs that include subsequences of special self-referential policy-modifying instructions [54, 69].
Both brains may also trigger the execution of certain bet instructions whose effect is to predict
experimental outcomes before they are observed. If their predictions or hypotheses differ, they may
agree to execute the experiment to determine which brain was right, and the surprised loser will pay
an intrinsic reward (the real-valued bet, e.g., 1.0) to the winner in a zero sum game.
That is, each brain is intrinsically motivated to outwit or surprise the other by proposing an experiment
such that the other agrees on the experimental protocol but disagrees on the predicted outcome, which


                                                   4
is typically an internal computable abstraction of complex spatio-temporal events generated through
the execution the self-invented experiment.
This motivates the unsupervised two brain system to focus on "interesting" computational questions,
losing interest in "boring" computations (potentially involving the environment) whose outcomes
are consistently predictable by both brains, as well as computations whose outcomes are currently
still hard to predict by any brain. Again, in the absence of external reward, each brain maximizes the
value function minimised by the other.
Using the meta-learning Success-Story RL algorithm [54, 69], AC1997 learns when to learn and
what to learn [55, 56, 58]. AC1997 will also minimize the computational cost of learning new skills,
provided both brains receive a small negative reward for each computational step, which introduces a
bias towards simple still surprising experiments (reflecting simple still unsolved problems). This may
facilitate hierarchical construction of more and more complex experiments, including those yielding
external reward (if there is any). In fact, AC1997’s artificial creativity may not only drive artificial
scientists and artists, e.g., [61], but can also accelerate the intake of external reward, e.g., [55, 58],
intuitively because a better understanding of the world can help to solve certain problems faster.
Other RL or evolutionary algorithms could also be applied to such two-brain systems implemented
as two interacting (possibly recurrent) RL NNs or other computers. However, certain issues such
as catastrophic forgetting are presumably better addressed by the later P OWER P LAY framework
(2011) [63, 75], which offers an asymptotically optimal way of finding the simplest yet unsolved
problem in a (potentially infinite) set of formalizable problems with computable solutions, and adding
its solution to the repertoire of a more and more general, curious problem solver. Compare also the
One Big Net For Everything [66] which offers a simplified, less strict NN version of P OWER P LAY.
How does AC1997 relate to GANs? AC1997 is similar to standard GANs in the sense that both
are unsupervised generative adversarial minimax players and focus on experiments with a binary
outcome: 1 or 0, yes or no, hypothesis true or false. However, for GANs the experimental protocol
is prewired and always the same: It simply tests whether a recently generated pattern is in a given
training set or not (Sec. 2.1). One can restrict AC1997 to such simple settings by limiting its domain
and the nature of the instructions in its programming language, such that possible bets of both brains
are limited to binary yes/no outcomes of GAN-like experiments. In general, however, the adversarial
brains of AC1997 can invent essentially arbitrary computational questions or problems by themselves,
generating programs that interact with the environment in any computable way that will yield binary
results on which both brains can bet. A bit like a pure scientist deriving internal joy signals from
inventing experiments that yield discoveries of initially surprising but learnable and then reliably
repeatable predictabilities.



5   GANs and Predictability Minimization (PM)

An important NN task is to learn the statistics of given data such as images. To achieve this,
the principles of gradient descent/ascent were used in yet another type of unsupervised minimax
game where one NN minimizes the objective function maximized by another. This duel between two
unsupervised adversarial NNs was introduced in the 1990s in a series of papers [49, 52, 53, 67, 71, 57].
It was called Predictability Minimization (PM).
PM’s goal is to achieve an important goal of unsupervised learning, namely, an ideal, disentangled,
factorial code [5, 4] of given data, where the code components are statistically independent of each
other. That is, the codes are distributed like the data, and the probability of a given data pattern is
simply the product of the probabilities of its code components. Such codes may facilitate subsequent
downstream learning [67, 71, 57].
PM requires an encoder network with initially random weights. It maps data samples x ∈ Rn (such
as images) to codes y ∈ [0, 1]m represented across m so-called code units. In what follows, integer
indices i, j range over 1, . . . , m. The i-th component of y is called yi ∈ [0, 1]. A separate predictor
network is trained by gradient descent to predict each yi from the remaining components yj (j 6= i).
The encoder, however, is trained to maximize the same objective function (e.g., mean squared error)
minimized by the predictor. Compare the text near Equation 2 in the 1996 paper [67]: “The clue is:
the code units are trained (in our experiments by online backprop) to maximize essentially the same


                                                    5
 objective function the predictors try to minimize;" or Equation 3 in Sec. 4.1 of the 1999 paper [57]:
“But the code units try to maximize the same objective function the predictors try to minimize."
Why should the end result of this fight between predictor and encoder be a disentagled factorial
code? Using gradient descent, to maximize the prediction errors, the code unit activations yj run
away from their real-valued predictions in [0, 1], that is, they are forced towards the corners of the
unit interval, and tend to become binary, either 0 or 1. And according to a proof of 1992 [11, 53],2
the encoder’s objective function is maximized when the i-th code unit maximizes its variance (thus
maximizing the information it conveys about the input data) while simultaneously minimizing
the deviation between its (unconditional) expected activations E(yi ) and its predictor-modeled,
conditional expected activations E(yi | {yj , j 6= i}), given the other code units. See also conjecture
6.4.1 and Sec. 6.9.3 of the thesis [53]. That is, the code units are motivated to extract informative yet
mutually independent binary features from the data.
PM’s inherent class of probability distributions is the set of multivariate binomial distributions. In
the ideal case, PM has indeed learned to create a binary factorial code of the data. That is, in response
to some input pattern, each yi is either 0 or 1, and the predictor has learned the conditional expected
value E(yi | {yj , j 6= i}). Since the code is both binary and factorial, this value is equal to the code
unit’s unconditional probability P (yi = 1) of being on (e.g., [52], Equation in Sec. 2). E.g., if some
code unit’s prediction is 0.25, then the probability of this code unit being on is 1/4.
The first toy experiments with PM [49] were conducted nearly three decades ago when compute
was about a million times more expensive than today. When it had become about 10 times cheaper
5 years later, it was shown that simple semi-linear PM variants applied to images automatically
generate feature detectors well-known from neuroscience, such as on-center-off-surround detectors,
off-center-on-surround detectors, orientation-sensitive bar detectors, etc [67, 71].


5.1       Is it true that PM is NOT a minimax game?

The NIPS 2014 GAN paper [19] states that PM differs from GANs in the sense that PM is NOT based
on a minimax game with a value function that one agent seeks to maximize and the other seeks to
minimise. It states that for GANs "the competition between the networks is the sole training criterion,
and is sufficient on its own to train the network," while PM "is only a regularizer that encourages the
hidden units of a neural network to be statistically independent while they accomplish some other
task; it is not a primary training criterion" [19].
But this claim is incorrect, since PM is indeed a pure minimax game, too, e.g., [67], Equation 2. There
is no "other task." In particular, PM was also trained [49, 52, 53, 67, 71, 57] (also on images [67, 71])
such that "the competition between the networks is the sole training criterion, and is sufficient on its
own to train the network."


5.2       Learning generative models through PM variants (not done in previous PM work)

One of the variants in the first peer-reviewed PM paper ([52] e.g., Sec 4.3, 4.4) had an optional
decoder (called reconstructor) attached to the code such that data can be reconstructed from its
code. Let’s assume that PM has indeed found an ideal factorial code of the data. Since the codes
are distributed like the data, with the decoder, we could immediately use the system as a generative
model, by randomly activating each binary code unit according to its unconditional probability (which
for all training patterns is now equal to the activation of its prediction—see Sec. 5), and sampling



      2
     It should be mentioned that the above-mentioned proof [11, 53] is limited to binary factorial codes. There is
no proof that PM is a universal method for approximating all kinds of non-binary distributions (most of which
are incomputable anyway). Nevertheless, it is well-known that binary Bernoulli distributions can approximate at
least Gaussians and other distributions, that is, with enough binary code units one should get at least arbitrarily
close approximations of broad classes of distributions. In the PM papers of the 1990s, however, this was not
studied in detail.


                                                        6
output data through the decoder.3 With an accurate decoder, the sampled data must obey the statistics
of the original distribution, by definition of factorial codes.
However, to our knowledge, this straight-forward application as a generative model was never
explicitly mentioned in any PM paper, and the decoder (as well as additional, optional local variance
maximization for the code units) was actually omitted in several PM papers after 1993 [67, 71,
57] which focused on unsupervised learning of disentangled internal representations, to facilitate
subsequent downstream learning [67, 71, 57].
Nevertheless, generative models producing data through stochastic outputs of minimax-trained NNs
were described in 1990 [46, 50] (see Sec. 2 on Adversarial Curiosity) and 2014 [19] (Sec. 1).
Compare also the concept of Adversarial Autoencoders [32].

5.3   Learning factorial codes through GAN variants

PM variants could easily be used as GAN-like generative models (Sec. 5.2). In turn, GAN variants
could easily be used to learn factorial codes like PM. If we take a GAN generator network trained on
random input codes with independent components, and attach a traditional encoder network to its
output layer, and train this encoder to map the output patterns back to their original random codes,
then in the ideal case this encoder will become a factorial code generator that can also be applied to
the original data. This was not done by the GANs of 2014 [19]. However, compare InfoGANs [8]
and related work [32, 13, 14].

5.4   Relation between PM and GANs and their variants

Both PM and GANs are unsupervised learning techniques that model the statistics of given data.
Both employ gradient-based adversarial nets that play a minimax game to achieve their goals.
While PM tries to make easily decoded, random-looking, factorial codes of the data, GANs try to
make decoded data directly from random codes. In this sense, the inputs of PM’s encoders are like
the outputs of GAN’s decoders, while the outputs of PM’s encoders are like the inputs of GAN’s
decoders. In another sense, the outputs of PM’s encoders are like the outputs of GAN’s decoders
because both are shaped by the adversarial loss.
Effectively, GANs are trying to approximate the true data distribution through some other distribution
of a given type (e.g. Gaussian, binomial, etc). Likewise, PM is trying to approximate it through a
multivariate factorial binomial distribution, whose nature is also given in advance (see Footnote 2).
While other post-PM methods such as the Information Bottleneck Method [78] based on rate distortion
theory [10, 9], Variational Autoencoders [29, 43], Noise-Contrastive Estimation [20], and Self-
Supervised Boosting [80] also share certain relationships to PM, none of them employs gradient-based
adversarial NNs in a PM-like minimax game. GANs do.
A certain duality between PM variants with attached decoders (Sec. 5.2) and GAN variants with
attached encoders (Sec. 5.3) can be illustrated through the following work flow pipelines (view them
as very similar 4 step cycles by identifying their beginnings and ends—see Fig. 1):

       • Pipeline of PM variants with standard decoders:
         data → minimax-trained encoder → code → traditional decoder (often omitted) → data
       • Pipeline of GAN variants with standard encoders (compare InfoGANs):
         code → minimax-trained decoder → data → traditional encoder → code

It will be interesting to study experimentally whether the GAN pipeline above is easier to train than
PM to make factorial codes or useful approximations thereof.

    3
      Note that even one-dimensional data may have a complex distribution whose binary factorial code (Sec. 5)
may require many dimensions. PM’s goal is the discovery of such a code, with an a priori unknown number of
components. For example, if there are 8 input patterns, each represented by a single real-valued number between
0 and 1, each occurring with probability 1/8, then there is an ideal binary factorial code across 3 binary code
units, each active with probability 1/2. Through a decoder on top of the 3-dimensional code of the 1-dimensional
data we could resample the original data distribution, by randomly activating each of the 3 binary code units
with probability 50% (these probabilities are actually directly visible as predictor activations).


                                                       7
6   Conclusion

The notion of Unsupervised Minimax refers to unsupervised adaptive modules (typically neural
networks or NNs) playing a zero sum game. The first NN somehow learns to generate data. The
second NN learns to predict properties of the generated data, minimizing its error. The first NN
maximizes the objective function minimized by the second NN, trying to produce outputs that are
hard on the second NN. Examples are provided by Adversarial Curiosity (AC since 1990, Sec. 2),
Predictability Minimization (PM since 1991, Sec. 5), Generative Adversarial Networks (GANs since
2014; conditional GANs since 2010, Sec. 1). GANs and cGANs can be formulated as special cases
of AC (Sec. 2.1). GANs are also related to PM, because both GANs and PM model the statistics of
given data distributions through gradient-based adversarial nets that play a minimax game (Sec. 5).
(Unlike AC and GANs, however, PM apparently has never been decribed/used as a generative model
as suggested in Sec. 5.2.) The present paper clarifies some of the previously published confusion
surrounding these issues.
AC’s generality (see end of Sec. 2.1) extends GAN-like unsupervised minimax to sequential problems,
not only for plain pattern generation and classification, but even for RL problems in partially
observable environments. In turn, the large body of recent GAN-related insights might help to
improve training procedures of certain AC systems.




                   DATA                                                    DATA

     MINIMAX                                                                           MINIMAX
                                 Standard decoder       Standard encoder
     TRAINED
     ENCODER
                      PM          (often omitted)          (InfoGAN)       GAN         TRAINED
                                                                                       DECODER




                   CODE                                                    CODE


Figure 1: Symmetric work flows of PM and GAN variants. Both PM and GANs model given data
distributions in unsupervised fashion. PM uses gradient-based minimax or adversarial training to
learn an encoder of the data, such that the codes are distributed like the data, and the probability of
a given pattern can be read off its code as the product of the predictor-modeled probabilities of the
code components (Sec. 5). GANs, however, use gradient-based minimax or adversarial training to
directly learn a decoder of given codes (Sec. 1). In turn, to decode its codes again, PM can learn a
non-adversarial traditional decoder (omitted in most PM papers after 1992—see Sec. 5.2). Similarly,
to encode the data again, GAN variants can learn a non-adversarial traditional encoder (absent in
the 2014 GAN paper but compare InfoGANs—see Sec. 5.3). While PM’s minimax procedure starts
from the data and learns a factorial code in form of a multivariate binomial distribution, GAN’s
minimax procedure starts from the codes (distributed according to any user-given distribution), and
learns to make data distributed like the original data.


Acknowledgments

Thanks to Paulo Rauber, Joachim Buhmann, Sjoerd van Steenkiste, David Ha, Róbert Csordás, and
Louis Kirsch, for useful comments on a draft of this paper. This work was partially funded by a
European Research Council Advanced Grant (ERC no: 742870).


                                                    8
References
 [1] M. Arjovsky, S. Chintala, and L. Bottou. Wasserstein GAN. Preprint arXiv:1701.07875, 2017.
 [2] J.-Y. Audibert and S. Bubeck. Minimax policies for adversarial and stochastic bandits. In Proc.
     COLT, pages 217–226, 2009.
 [3] P. Auer, N. Cesa-Bianchi, Y. Freund, and R. E. Schapire. Gambling in a rigged casino: The
     adversarial multi-armed bandit problem. In Proc. IEEE 36th Annual Foundations of Computer
     Science, pages 322–331. IEEE, 1995.
 [4] H. B. Barlow. Unsupervised learning. Neural Computation, 1(3):295–311, 1989.
 [5] H. B. Barlow, T. P. Kaushal, and G. J. Mitchison. Finding minimum entropy codes. Neural
     Computation, 1(3):412–423, 1989.
 [6] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan. Domain separation
     networks. In Advances in Neural Information Processing Systems (NIPS), pages 343–351, 2016.
 [7] Y. Burda, H. Edwards, D. Pathak, A. Storkey, T. Darrell, and A. A. Efros. Large-scale study of
     curiosity-driven learning. Preprint arXiv:1808.04355, 2018.
 [8] X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. InfoGAN: Inter-
     pretable Representation Learning by Information Maximizing Generative Adversarial Nets. TR
     arXiv:1606.03657, 2016.
 [9] T. M. Cover and J. A. Thomas. Elements of information theory. John Wiley & Sons, 2012.
[10] L. D. Davisson. Rate-distortion theory and application. Proceedings of the IEEE, 60(7):800–808,
     1972.
[11] P. Dayan, R. Zemel, and A. Pouget, 1992. Personal Communication.
[12] E. L. Denton, S. Chintala, R. Fergus, et al. Deep generative image models using a Laplacian
     pyramid of adversarial networks. In Advances in Neural Information Processing Systems (NIPS),
     pages 1486–1494, 2015.
[13] J. Donahue, P. Krähenbühl, and T. Darrell.          Adversarial feature learning.     Preprint
     arXiv:1605.09782, 2016.
[14] V. Dumoulin, I. Belghazi, B. Poole, A. Lamb, M. Arjovsky, O. Mastropietro, and A. Courville.
     Adversarially learned inference. Preprint arXiv:1606.00704, 2016.
[15] V. V. Fedorov. Theory of optimal experiments. Academic Press, 1972.
[16] Y. Ganin, T. Kulkarni, I. Babuschkin, S. Eslami, and O. Vinyals. Synthesizing programs for
     images using reinforced adversarial learning. Preprint arXiv:1804.01118, 2018.
[17] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and
     V. Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning
     Research, 17(59):1–35, 2016.
[18] J. C. Gittins. Multi-armed Bandit Allocation Indices. Wiley, Chichester, NY, 1989.
[19] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and
     Y. Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems
     (NIPS), pages 2672–2680, Dec 2014.
[20] M. Gutmann and A. Hyvärinen. Noise-contrastive estimation: A new estimation principle
     for unnormalized statistical models. In Proceedings of the 13th International Conference on
     Artificial Intelligence and Statistics, pages 297–304, 2010.
[21] D. Ha and D. Eck. A neural representation of sketch drawings. Preprint arXiv:1704.03477,
     2017.
[22] D. Ha and J. Schmidhuber. World models. Preprint arXiv:1803.10122 (variant at NeurIPS
     2018), 2018.
[23] W. D. Hillis. Co-evolving parasites improve simulated evolution as an optimization procedure.
     Physica D: Nonlinear Phenomena, 42(1-3):228–234, 1990.
[24] Z. Huang, W. Heng, and S. Zhou. Learning to paint with model-based deep reinforcement
     learning. CoRR, abs/1903.04411, 2019.


                                                 9
[25] F. Huszár. How (not) to train your generative model: Scheduled sampling, likelihood, adversary?
     Preprint arXiv:1511.05101, 2015.
[26] L. Itti and P. F. Baldi. Bayesian surprise attracts human attention. In Advances in Neural
     Information Processing Systems (NIPS) 19, pages 547–554. MIT Press, Cambridge, MA, 2005.
[27] M. I. Jordan and D. E. Rumelhart. Supervised learning with a distal teacher. Technical Report
     Occasional Paper #40, Center for Cog. Sci., Massachusetts Institute of Technology, 1990.
[28] L. P. Kaelbling, M. L. Littman, and A. W. Moore. Reinforcement learning: a survey. Journal of
     AI research, 4:237–285, 1996.
[29] D. P. Kingma and M. Welling. Auto-encoding variational Bayes. Preprint arXiv:1312.6114,
     2013.
[30] S. Kullback and R. A. Leibler. On information and sufficiency. The Annals of Mathematical
     Statistics, pages 79–86, 1951.
[31] S. Linnainmaa. The representation of the cumulative rounding error of an algorithm as a Taylor
     expansion of the local rounding errors. Master’s thesis, Univ. Helsinki, 1970.
[32] A. Makhzani, J. Shlens, N. Jaitly, I. Goodfellow, and B. Frey. Adversarial autoencoders. Preprint
     arXiv:1511.05644, 2015.
[33] X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. Paul Smolley. Least squares generative
     adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision,
     pages 2794–2802, 2017.
[34] M. Mirza and S. Osindero. Conditional generative adversarial nets. arXiv preprint
     arXiv:1411.1784, 2014.
[35] O. Morgenstern and J. Von Neumann. Theory of games and economic behavior. Princeton
     University Press, 1944.
[36] P. W. Munro. A dual back-propagation scheme for scalar reinforcement learning. Proceedings
     of the Ninth Annual Conference of the Cognitive Science Society, Seattle, WA, pages 165–176,
     1987.
[37] R. Nakano. Neural painters: A learned differentiable constraint for generating brushstroke
     paintings. Preprint arXiv:1904.08410, 2019.
[38] N. Nguyen and B. Widrow. The truck backer-upper: An example of self learning in neural
     networks. In Proceedings of the International Joint Conference on Neural Networks, pages
     357–363. IEEE Press, 1989.
[39] O.      Niemitalo.             A       method     for     training   artificial    neural     net-
     works        to     generate      missing       data      within     a      variable      context.
     https://web.archive.org/web/20120312111546/http://yehar.com:80/blog/?p=167,              Internet
     Archive, 2010.
[40] S. Nowozin, B. Cseke, and R. Tomioka. f-GAN: training generative neural samplers using
     variational divergence minimization. In Advances in Neural Information Processing Systems
     (NIPS), pages 271–279, 2016.
[41] P.-Y. Oudeyer, A. Baranes, and F. Kaplan. Intrinsically motivated learning of real world
     sensorimotor skills with developmental constraints. In G. Baldassarre and M. Mirolli, editors,
     Intrinsically Motivated Learning in Natural and Artificial Systems. Springer, 2013.
[42] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolu-
     tional generative adversarial networks. Preprint arXiv:1511.06434, 2015.
[43] D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and approximate
     inference in deep generative models. Preprint arXiv:1401.4082, 2014.
[44] H. Robbins. Some aspects of the sequential design of experiments. Bulletin of the American
     Mathematical Society, 58(5):527–535, 1952.
[45] A. L. Samuel. Some studies in machine learning using the game of checkers. IBM Journal on
     Research and Development, 3:210–229, 1959.
[46] J. Schmidhuber. Making the world differentiable: On using fully recurrent self-supervised
     neural networks for dynamic reinforcement learning and planning in non-stationary environ-
     ments. Technical Report FKI-126-90, http://people.idsia.ch/~juergen/FKI-126-90_
     (revised)bw_ocr.pdf, Tech. Univ. Munich, 1990.


                                                 10
[47] J. Schmidhuber. An on-line algorithm for dynamic reinforcement learning and planning in
     reactive environments. In Proc. IEEE/INNS International Joint Conference on Neural Networks,
     San Diego, volume 2, pages 253–258, 1990.
[48] J. Schmidhuber. Curious model-building control systems. In Proceedings of the International
     Joint Conference on Neural Networks, Singapore, volume 2, pages 1458–1463. IEEE press,
     1991.
[49] J. Schmidhuber. Learning factorial codes by predictability minimization. Technical Report
     CU-CS-565-91, Dept. of Comp. Sci., University of Colorado at Boulder, Dec 1991.
[50] J. Schmidhuber. A possibility for implementing curiosity and boredom in model-building
     neural controllers. In J. A. Meyer and S. W. Wilson, editors, Proc. of the International
     Conference on Simulation of Adaptive Behavior: From Animals to Animats, pages 222–227.
     MIT Press/Bradford Books, 1991.
[51] J. Schmidhuber. Reinforcement learning in Markovian and non-Markovian environments. In
     D. S. Lippman, J. E. Moody, and D. S. Touretzky, editors, Advances in Neural Information
     Processing Systems 3 (NIPS 3), pages 500–506. Morgan Kaufmann, 1991.
[52] J. Schmidhuber. Learning factorial codes by predictability minimization. Neural Computation,
     4(6):863–879, 1992.
[53] J. Schmidhuber. Netzwerkarchitekturen, Zielfunktionen und Kettenregel. (Network architectures,
     objective functions, and chain rule.) Habilitation Thesis, Inst. f. Inf., Tech. Univ. Munich, 1993.
[54] J. Schmidhuber. On learning how to learn learning strategies. Technical Report FKI-198-94,
     Fakultät für Informatik, Technische Universität München, 1994. See [70, 69].
[55] J. Schmidhuber. What’s interesting?              Technical Report IDSIA-35-97, IDSIA, 1997.
     ftp://ftp.idsia.ch/pub/juergen/interest.ps.gz; extended abstract in Proc. Snowbird’98, Utah, 1998;
     see also [58].
[56] J. Schmidhuber. Artificial curiosity based on discovering novel algorithmic predictability
     through coevolution. In P. Angeline, Z. Michalewicz, M. Schoenauer, X. Yao, and Z. Zalzala,
     editors, Congress on Evolutionary Computation, pages 1612–1618. IEEE Press, 1999.
[57] J. Schmidhuber. Neural predictors for detecting and removing redundant information. In
     H. Cruse, J. Dean, and H. Ritter, editors, Adaptive Behavior and Learning. Kluwer, 1999.
[58] J. Schmidhuber. Exploring the predictable. In A. Ghosh and S. Tsuitsui, editors, Advances in
     Evolutionary Computing, pages 579–612. Springer, 2002.
[59] J. Schmidhuber. Developmental robotics, optimal artificial curiosity, creativity, music, and the
     fine arts. Connection Science, 18(2):173–187, 2006.
[60] J. Schmidhuber. Simple algorithmic principles of discovery, subjective beauty, selective atten-
     tion, curiosity & creativity. In Proc. 18th Intl. Conf. on Algorithmic Learning Theory (ALT
     2007), LNAI 4754, pages 32–33. Springer, 2007. Joint invited lecture for ALT 2007 and DS
     2007, Sendai, Japan, 2007.
[61] J. Schmidhuber. Art & science as by-products of the search for novel patterns, or data com-
     pressible in unknown yet learnable ways. In M. Botta, editor, Multiple ways to design research.
     Research cases that reshape the design discipline, Swiss Design Network - Et al. Edizioni,
     pages 98–112. Springer, 2009.
[62] J. Schmidhuber. Formal theory of creativity, fun, and intrinsic motivation (1990-2010). IEEE
     Transactions on Autonomous Mental Development, 2(3):230–247, 2010.
[63] J. Schmidhuber. P OWER P LAY: Training an Increasingly General Problem Solver by Continually
     Searching for the Simplest Still Unsolvable Problem. Frontiers in Psychology, 2013. (Based on
     arXiv:1112.5309v1 [cs.AI], 2011).
[64] J. Schmidhuber. Deep learning in neural networks: An overview. Neural Networks, 61:85–117,
     2015. Published online 2014; 888 references; based on TR arXiv:1404.7828 [cs.NE].
[65] J. Schmidhuber. On learning to think: Algorithmic information theory for novel combi-
     nations of reinforcement learning controllers and recurrent neural world models. Preprint
     arXiv:1511.09249, 2015.
[66] J. Schmidhuber. One big net for everything. Preprint arXiv:1802.08864 [cs.AI], February 2018.


                                                  11
[67] J. Schmidhuber, M. Eldracher, and B. Foltin. Semilinear predictability minimization produces
     well-known feature detectors. Neural Computation, 8(4):773–786, 1996.
[68] J. Schmidhuber and R. Huber. Learning to generate artificial fovea trajectories for target
     detection. International Journal of Neural Systems, 2(1 & 2):135–141, 1991. (Based on TR
     FKI-128-90, TUM, 1990).
[69] J. Schmidhuber, J. Zhao, and N. Schraudolph. Reinforcement learning with self-modifying
     policies. In S. Thrun and L. Pratt, editors, Learning to learn, pages 293–309. Kluwer, 1997.
[70] J. Schmidhuber, J. Zhao, and M. Wiering. Shifting inductive bias with success-story algorithm,
     adaptive Levin search, and incremental self-improvement. Machine Learning, 28:105–130,
     1997.
[71] N. N. Schraudolph, M. Eldracher, and J. Schmidhuber. Processing images by semi-linear
     predictability minimization. Network: Computation in Neural Systems, 10(2):133–169, 1999.
[72] J. Seger and W. Hamilton. Parasites and sex. In The evolution of sex: An examination of current
     ideas, pages 176–193. Sinauer Associates, Inc., 1988.
[73] C. E. Shannon. A mathematical theory of communication (parts I and II). Bell System Technical
     Journal, XXVII:379–423, 1948.
[74] S. Singh, A. G. Barto, and N. Chentanez. Intrinsically motivated reinforcement learning. In
     Advances in Neural Information Processing Systems 17 (NIPS). MIT Press, Cambridge, MA,
     2005.
[75] R. K. Srivastava, B. R. Steunebrink, and J. Schmidhuber. First experiments with PowerPlay.
     Neural Networks, 41(0):130 – 136, 2013. Special Issue on Autonomous Learning.
[76] J. Storck, S. Hochreiter, and J. Schmidhuber. Reinforcement driven information acquisition in
     non-deterministic environments. In Proceedings of the International Conference on Artificial
     Neural Networks, Paris, volume 2, pages 159–164. EC2 & Cie, 1995.
[77] R. Sutton and A. Barto. Reinforcement learning: An introduction. Cambridge, MA, MIT Press,
     1998.
[78] N. Tishby, F. C. Pereira, and W. Bialek. The information bottleneck method. arXiv preprint
     physics/0004057, 2000.
[79] T. Unterthiner, B. Nessler, G. Klambauer, M. Heusel, H. Ramsauer, and S. Hochreiter. Coulomb
     GANs: provably optimal Nash equilibria via potential fields. Preprint arXiv:1708.08819, 2017.
[80] M. Welling, R. S. Zemel, and G. E. Hinton. Self supervised boosting. In Advances in neural
     information processing systems (NIPS), pages 681–688, 2003.
[81] P. J. Werbos. Learning how the world works: Specifications for predictive networks in robots
     and brains. In Proceedings of IEEE International Conference on Systems, Man and Cybernetics,
     N.Y., 1987.
[82] P. J. Werbos. Neural networks for control and system identification. In Proceedings of
     IEEE/CDC Tampa, Florida, 1989.
[83] N. Wiener. Cybernetics or Control and Communication in the Animal and the Machine,
     volume 25. MIT press, 1965.
[84] M. Wiering and M. van Otterlo. Reinforcement Learning. Springer, 2012.
[85] R. J. Williams. On the use of backpropagation in associative reinforcement learning. In IEEE
     International Conference on Neural Networks, San Diego, volume 2, pages 263–270, 1988.
[86] N. Zheng, Y. Jiang, and D. Huang. Strokenet: A neural painting environment. In ICLR, 2019.
[87] K. Zuse. Chess programs, in The Plankalkuel. Rept. No. 106, Gesellschaft fuer Mathematik und
     Datenverarbeitung, pages 201–244, 1976 (Translation of German original, 1945).




                                                12

