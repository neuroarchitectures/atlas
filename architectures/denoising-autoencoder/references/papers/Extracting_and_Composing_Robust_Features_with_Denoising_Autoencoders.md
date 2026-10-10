# Extracting and Composing Robust Features with Denoising Autoencoders

> Source: `https://doi.org/10.1145/1390156.1390294`

---

Extracting and Composing Robust Features with Denoising
Autoencoders
Pascal Vincent
vincentp@iro.umontreal.ca
Hugo Larochelle
larocheh@iro.umontreal.ca
Yoshua Bengio
bengioy@iro.umontreal.ca
Pierre-Antoine Manzagol
manzagop@iro.umontreal.ca
Universit´e de Montr´eal, Dept. IRO, CP 6128, Succ. Centre-Ville, Montral, Qubec, H3C 3J7, Canada
Abstract
Previous work has shown that the diﬃcul-
ties in learning deep generative or discrim-
inative models can be overcome by an ini-
tial unsupervised learning step that maps in-
puts to useful intermediate representations.
We introduce and motivate a new training
principle for unsupervised learning of a rep-
resentation based on the idea of making the
learned representations robust to partial cor-
ruption of the input pattern. This approach
can be used to train autoencoders, and these
denoising autoencoders can be stacked to ini-
tialize deep architectures. The algorithm can
be motivated from a manifold learning and
information theoretic perspective or from a
generative model perspective. Comparative
experiments clearly show the surprising ad-
vantage of corrupting the input of autoen-
coders on a pattern classiﬁcation benchmark
suite.
1. Introduction
Recent theoretical studies indicate that deep architec-
tures (Bengio & Le Cun, 2007; Bengio, 2007) may be
needed to eﬃciently model complex distributions and
achieve better generalization performance on challeng-
ing recognition tasks. The belief that additional levels
of functional composition will yield increased repre-
sentational and modeling power is not new (McClel-
land et al., 1986; Hinton, 1989; Utgoﬀ& Stracuzzi,
2002). However, in practice, learning in deep archi-
tectures has proven to be diﬃcult.
One needs only
Appearing in Proceedings of the 25 th International Confer-
ence on Machine Learning, Helsinki, Finland, 2008. Copy-
right 2008 by the author(s)/owner(s).
to ponder the diﬃcult problem of inference in deep
directed graphical models, due to “explaining away”.
Also looking back at the history of multi-layer neural
networks, their diﬃcult optimization (Bengio et al.,
2007; Bengio, 2007) has long prevented reaping the ex-
pected beneﬁts of going beyond one or two hidden lay-
ers. However this situation has recently changed with
the successful approach of (Hinton et al., 2006; Hinton
& Salakhutdinov, 2006; Bengio et al., 2007; Ranzato
et al., 2007; Lee et al., 2008) for training Deep Belief
Networks and stacked autoencoders.
One key ingredient to this success appears to be the
use of an unsupervised training criterion to perform
a layer-by-layer initialization:
each layer is at ﬁrst
trained to produce a higher level (hidden) represen-
tation of the observed patterns, based on the repre-
sentation it receives as input from the layer below,
by optimizing a local unsupervised criterion.
Each
level produces a representation of the input pattern
that is more abstract than the previous level’s, be-
cause it is obtained by composing more operations.
This initialization yields a starting point, from which
a global ﬁne-tuning of the model’s parameters is then
performed using another training criterion appropriate
for the task at hand. This technique has been shown
empirically to avoid getting stuck in the kind of poor
solutions one typically reaches with random initializa-
tions. While unsupervised learning of a mapping that
produces “good” intermediate representations of the
input pattern seems to be key, little is understood re-
garding what constitutes “good” representations for
initializing deep architectures, or what explicit crite-
ria may guide learning such representations. We know
of only a few algorithms that seem to work well for
this purpose: Restricted Boltzmann Machines (RBMs)
trained with contrastive divergence on one hand, and
various types of autoencoders on the other.
The present research begins with the question of what

Extracting and Composing Robust Features with Denoising Autoencoders
explicit criteria a good intermediate representation
should satisfy. Obviously, it should at a minimum re-
tain a certain amount of “information” about its input,
while at the same time being constrained to a given
form (e.g. a real-valued vector of a given size in the
case of an autoencoder). A supplemental criterion that
has been proposed for such models is sparsity of the
representation (Ranzato et al., 2008; Lee et al., 2008).
Here we hypothesize and investigate an additional spe-
ciﬁc criterion: robustness to partial destruction
of the input, i.e., partially destroyed inputs should
yield almost the same representation. It is motivated
by the following informal reasoning: a good represen-
tation is expected to capture stable structures in the
form of dependencies and regularities characteristic of
the (unknown) distribution of its observed input. For
high dimensional redundant input (such as images) at
least, such structures are likely to depend on evidence
gathered from a combination of many input dimen-
sions. They should thus be recoverable from partial
observation only.
A hallmark of this is our human
ability to recognize partially occluded or corrupted im-
ages. Further evidence is our ability to form a high
level concept associated to multiple modalities (such
as image and sound) and recall it even when some of
the modalities are missing.
To validate our hypothesis and assess its usefulness as
one of the guiding principles in learning deep architec-
tures, we propose a modiﬁcation to the autoencoder
framework to explicitly integrate robustness to par-
tially destroyed inputs. Section 2 describes the algo-
rithm in details. Section 3 discusses links with other
approaches in the literature. Section 4 is devoted to
a closer inspection of the model from diﬀerent theo-
retical standpoints. In section 5 we verify empirically
if the algorithm leads to a diﬀerence in performance.
Section 6 concludes the study.
2. Description of the Algorithm
2.1. Notation and Setup
Let X and Y be two random variables with joint prob-
ability density p(X, Y ), with marginal distributions
p(X) and p(Y ).
Throughout the text, we will use
the following notation: Expectation: EEp(X)[f(X)] =
R
p(x)f(x)dx.
Entropy:
IH(X)
=
IH(p)
=
EEp(X)[−log p(X)].
Conditional entropy: IH(X|Y ) =
EEp(X,Y )[−log p(X|Y )].
Kullback-Leibler divergence:
IDKL(p∥q) = EEp(X)[log p(X)
q(X)]. Cross-entropy: IH(p∥q) =
EEp(X)[−log q(X)] = IH(p) + IDKL(p∥q). Mutual infor-
mation: I(X; Y ) = IH(X) −IH(X|Y ). Sigmoid: s(x) =
1
1+e−x and s(x) = (s(x1), . . . , s(xd))T . Bernoulli dis-
tribution with mean µ:
Bµ(x).
and by extension
Bµ(x) = (Bµ1(x1), . . . , Bµd(xd)).
The setup we consider is the typical supervised learn-
ing setup with a training set of n (input, target) pairs
Dn = {(x(1), t(1)) . . . , (x(n), t(n))}, that we suppose
to be an i.i.d. sample from an unknown distribution
q(X, T) with corresponding marginals q(X) and q(T).
2.2. The Basic Autoencoder
We begin by recalling the traditional autoencoder
model such as the one used in (Bengio et al., 2007)
to build deep networks.
An autoencoder takes an
input vector x ∈[0, 1]d, and ﬁrst maps it to a hid-
den representation y ∈[0, 1]d′ through a deterministic
mapping y = fθ(x) = s(Wx + b), parameterized by
θ = {W, b}.
W is a d′ × d weight matrix and b
is a bias vector. The resulting latent representation
y is then mapped back to a “reconstructed” vector
z ∈[0, 1]d in input space z = gθ′(y) = s(W′y + b′)
with θ′ = {W′, b′}.
The weight matrix W′ of the
reverse mapping may optionally be constrained by
W′ = WT , in which case the autoencoder is said to
have tied weights. Each training x(i) is thus mapped
to a corresponding y(i) and a reconstruction z(i). The
parameters of this model are optimized to minimize
the average reconstruction error:
θ⋆, θ′⋆=
arg min
θ,θ′
1
n
n
X
i=1
L

x(i), z(i)
=
arg min
θ,θ′
1
n
n
X
i=1
L

x(i), gθ′(fθ(x(i)))

(1)
where L is a loss function such as the traditional
squared error L(x, z) = ∥x −z∥2. An alternative loss,
suggested by the interpretation of x and z as either
bit vectors or vectors of bit probabilities (Bernoullis)
is the reconstruction cross-entropy:
LIH(x, z)= IH(Bx∥Bz)
= −
d
X
k=1
[xk log zk+(1 −xk) log(1 −zk)] (2)
Note that if x is a binary vector, LIH(x, z) is a negative
log-likelihood for the example x, given the Bernoulli
parameters z. Equation 1 with L = LIH can be written
θ⋆, θ′⋆= arg min
θ,θ′
EEq0(X) [LIH (X, gθ′(fθ(X)))]
(3)
where q0(X) denotes the empirical distribution asso-
ciated to our n training inputs. This optimization will
typically be carried out by stochastic gradient descent.

Extracting and Composing Robust Features with Denoising Autoencoders
2.3. The Denoising Autoencoder
To test our hypothesis and enforce robustness to par-
tially destroyed inputs we modify the basic autoen-
coder we just described. We will now train it to recon-
struct a clean “repaired” input from a corrupted, par-
tially destroyed one. This is done by ﬁrst corrupting
the initial input x to get a partially destroyed version
˜x by means of a stochastic mapping ˜x ∼qD(˜x|x). In
our experiments, we considered the following corrupt-
ing process, parameterized by the desired proportion ν
of “destruction”: for each input x, a ﬁxed number νd
of components are chosen at random, and their value
is forced to 0, while the others are left untouched. All
information about the chosen components is thus re-
moved from that particuler input pattern, and the au-
toencoder will be trained to “ﬁll-in” these artiﬁcially
introduced “blanks”. Note that alternative corrupting
noises could be considered1. The corrupted input ˜x is
then mapped, as with the basic autoencoder, to a hid-
den representation y = fθ(˜x) = s(W˜x+b) from which
we reconstruct a z = gθ′(y) = s(W′y + b′) (see ﬁgure
1 for a schematic representation of the process). As
before the parameters are trained to minimize the av-
erage reconstruction error LIH(x, z) = IH(Bx∥Bz) over
a training set, i.e. to have z as close as possible to the
uncorrupted input x. But the key diﬀerence is that z
is now a deterministic function of ˜x rather than x and
thus the result of a stochastic mapping of x.
fθ
x
˜x
qD
y
z
LH(x, z)
gθ′
Figure 1. An example x is corrupted to ˜x.
The autoen-
coder then maps it to y and attempts to reconstruct x.
Let us deﬁne the joint distribution
q0(X, e
X, Y ) = q0(X)qD( e
X|X)δfθ( e
X)(Y )
(4)
where δu(v) puts mass 0 when u̸ = v. Thus Y is a
deterministic function of e
X.
q0(X, e
X, Y ) is param-
eterized by θ. The objective function minimized by
stochastic gradient descent becomes:
arg min
θ,θ′
EEq0(X, e
X)
h
LIH

X, gθ′(fθ( e
X))
i
.
(5)
So from the point of view of the stochastic gradient de-
scent algorithm, in addition to picking an input sam-
ple from the training set, we will also produce a ran-
dom corrupted version of it, and take a gradient step
1The approach we describe and our analysis is not spe-
ciﬁc to a particular kind of corrupting noise.
towards reconstructing the uncorrupted version from
the corrupted version. Note that in this way, the au-
toencoder cannot learn the identity, unlike the basic
autoencoder, thus removing the constraint that d′ < d
or the need to regularize speciﬁcally to avoid such a
trivial solution.
2.4. Layer-wise Initialization and Fine Tuning
The basic autoencoder has been used as a building
block to train deep networks (Bengio et al., 2007), with
the representation of the k-th layer used as input for
the (k + 1)-th, and the (k + 1)-th layer trained after
the k-th has been trained. After a few layers have been
trained, the parameters are used as initialization for a
network optimized with respect to a supervised train-
ing criterion.
This greedy layer-wise procedure has
been shown to yield signiﬁcantly better local minima
than random initialization of deep networks , achieving
better generalization on a number of tasks (Larochelle
et al., 2007).
The procedure to train a deep network using the de-
noising autoencoder is similar. The only diﬀerence is
how each layer is trained, i.e., to minimize the crite-
rion in eq. 5 instead of eq. 3. Note that the corrup-
tion process qD is only used during training, but not
for propagating representations from the raw input to
higher-level representations. Note also that when layer
k is trained, it receives as input the uncorrupted out-
put of the previous layers.
3. Relationship to Other Approaches
Our training procedure for the denoising autoencoder
involves learning to recover a clean input from a cor-
rupted version, a task known as denoising. The prob-
lem of image denoising, in particular, has been exten-
sively studied in the image processing community and
many recent developments rely on machine learning
approaches (see e.g. Roth and Black (2005); Elad and
Aharon (2006); Hammond and Simoncelli (2007)). A
particular form of gated autoencoders has also been
used for denoising in Memisevic (2007). Denoising us-
ing autoencoders was actually introduced much ear-
lier (LeCun, 1987; Gallinari et al., 1987), as an alter-
native to Hopﬁeld models (Hopﬁeld, 1982). Our ob-
jective however is fundamentally diﬀerent from that of
developing a competitive image denoising algorithm.
We investigate explicit robustness to corrupting noise
as a novel criterion guiding the learning of suitable in-
termediate representations to initialize a deep network.
Thus our corruption+denoising procedure is applied
not only on the input, but also recursively to interme-
diate representations.

Extracting and Composing Robust Features with Denoising Autoencoders
The approach also bears some resemblance to the well
known technique of augmenting the training data with
stochastically “transformed” patterns. E.g. augment-
ing a training set by transforming original bitmaps
through small rotations, translations, and scalings is
known to improve ﬁnal classiﬁcation performance. In
contrast to this technique our approach does not use
any prior knowledge of image topology, nor does it pro-
duce extra labeled examples for supervised training.
We use corrupted patterns in a generic (i.e. not spe-
ciﬁc to images) unsupervised initialization step, while
the supervised training phase uses the unmodiﬁed orig-
inal data.
There is a well known link between “training with
noise” and regularization: they are equivalent for small
additive noise (Bishop, 1995). By contrast, our cor-
ruption process is a large, non-additive, destruction
of information. We train autoencoders to ”ﬁll in the
blanks”, not merely be smooth functions (regulariza-
tion). Also in our experience, regularized autoencoders
(i.e. with weight decay) do not yield the quantitative
jump in performance and the striking qualitative dif-
ference observed in the ﬁlters that we get with denois-
ing autoencoders.
There are also similarities with the work of (Doi et al.,
2006) on robust coding over noisy channels. In their
framework, a linear encoder is to encode a clean input
for optimal transmission over a noisy channel to a de-
coder that reconstructs the input. This work was later
extended to robustness to noise in the input, in a pro-
posal for a model of retinal coding (Doi & Lewicki,
2007).
Though some of the inspiration behind our
work comes from neural coding and computation, our
goal is not to account for experimental data of neu-
ronal activity as in (Doi & Lewicki, 2007). Also, the
non-linearity of our denoising autoencoder is crucial
for its use in initializing a deep neural network.
It may be objected that, if our goal is to handle missing
values correctly, we could have more naturally deﬁned
a proper latent variable generative model, and infer the
posterior over the latent (hidden) representation in the
presence of missing inputs. But this usually requires
a costly marginalization2 which has to be carried out
for each new example. By contrast, our approach tries
to learn a fast and robust deterministic mapping fθ
from examples of already corrupted inputs. The bur-
den is on learning such a constrained mapping during
training, rather than on unconstrained inference at use
time. We expect this may force the model to capture
implicit invariances in the data, and result in interest-
2as for RBMs, where it is exponential in the number of
missing values
ing features. Also note that in section 4.2 we will see
how our learning algorithm for the denoising autoen-
coder can be viewed as a form of variational inference
in a particular generative model.
4. Analysis of Denoising Autoencoders
The above intuitive motivation for the denoising au-
toencoder was given with the perspective of discover-
ing robust representations. In the following, which can
be skipped without hurting the remainder of the paper,
we propose alternative perspectives on the algorithm.
4.1. Manifold Learning Perspective
The process of mapping a corrupted example to an
uncorrupted one can be visualized in Figure 2, with
a low-dimensional manifold near which the data con-
centrate. We learn a stochastic operator p(X| e
X) that
maps an e
X to an X, p(X| e
X) = Bgθ′(fθ( e
X))(X). The
corrupted examples will be much more likely to be
outside and farther from the manifold than the uncor-
rupted ones. Hence the stochastic operator p(X| e
X)
learns a map that tends to go from lower probability
points e
X to high probability points X, generally on
or near the manifold.
Note that when e
X is farther
from the manifold, p(X| e
X) should learn to make big-
ger steps, to reach the manifold. At the limit we see
that the operator should map even far away points to
a small volume near the manifold.
The denoising autoencoder can thus be seen as a way
to deﬁne and learn a manifold. The intermediate rep-
resentation Y = f(X) can be interpreted as a coordi-
nate system for points on the manifold (this is most
clear if we force the dimension of Y to be smaller than
the dimension of X). More generally, one can think of
Y = f(X) as a representation of X which is well suited
to capture the main variations in the data, i.e., on the
manifold. When additional criteria (such as sparsity)
are introduced in the learning model, one can no longer
directly view Y = f(X) as an explicit low-dimensional
coordinate system for points on the manifold, but it
retains the property of capturing the main factors of
variation in the data.
4.2. Top-down, Generative Model Perspective
In this section we recover the training criterion for
our denoising autoencoder (eq. 5) from a generative
model perspective. Speciﬁcally we show that training
the denoising autoencoder as described in section 2.3
is equivalent to maximizing a variational bound on a
particular generative model.
Consider
the
generative
model
p(X, e
X, Y )
=
p(Y )p(X|Y )p( e
X|X) where p(X|Y ) = Bs(W′Y +b′) and

Extracting and Composing Robust Features with Denoising Autoencoders
p( e
X|X) = qD( e
X|X).
p(Y ) is a uniform prior over
Y ∈[0, 1]d′. This deﬁnes a generative model with pa-
rameter set θ′ = {W′, b′}.
We will use the previ-
ously deﬁned q0(X, e
X, Y ) = q0(X)qD( e
X|X)δfθ( e
X)(Y )
(equation 4) as an auxiliary model in the context of
a variational approximation of the log-likelihood of
p( e
X). Note that we abuse notation to make it lighter,
and use the same letters X, e
X and Y for diﬀerent
sets of random variables representing the same quan-
tity under diﬀerent distributions: p or q0.
Keep in
mind that whereas we had the dependency structure
X →e
X →Y for q or q0, we have Y →X →e
X for p.
Since p contains a corruption operation at the last
generative stage, we propose to ﬁt p( e
X) to corrupted
training samples. Performing maximum likelihood ﬁt-
ting for samples drawn from q0( e
X) corresponds to min-
imizing the cross-entropy, or maximizing
H = max
θ′ {−IH(q0( e
X)∥p( e
X))}
= max
θ′ {EEq0( e
X)[log p( e
X)]}.
(6)
Let q⋆(X, Y | e
X) be a conditional density, the quan-
tity L(q⋆, e
X) = EEq⋆(X,Y | e
X)
h
log p(X, e
X,Y )
q⋆(X,Y | e
X)
i
is a lower
bound on log p( e
X) since the following can be shown to
be true for any q⋆:
log p( e
X) = L(q⋆, e
X) + IDKL(q⋆(X, Y | e
X)∥p(X, Y | e
X))
Also it is easy to verify that the bound is tight when
q⋆(X, Y | e
X) = p(X, Y | e
X), where the IDKL becomes 0.
We can thus write log p( e
X) = maxq⋆L(q⋆, e
X), and
consequently rewrite equation 6 as
H = max
θ′ {EEq0( e
X)[max
q⋆L(q⋆, e
X)]}
= max
θ′,q⋆{EEq0( e
X)[L(q⋆, e
X)]}
(7)
x
x
˜x
˜x
qD(˜x|x)
gθ′(fθ(˜x))
Figure 2. Manifold learning perspective.
Suppose
training data (×) concentrate near a low-dimensional man-
ifold. Corrupted examples (.) obtained by applying cor-
ruption process qD( e
X|X) will lie farther from the manifold.
The model learns with p(X| e
X) to “project them back” onto
the manifold. Intermediate representation Y can be inter-
preted as a coordinate system for points on the manifold.
where we moved the maximization outside of the ex-
pectation because an unconstrained q⋆(X, Y | e
X) can
in principle perfectly model the conditional distribu-
tion needed to maximize L(q⋆, e
X) for any e
X.
Now
if we replace the maximization over an unconstrained
q⋆by the maximization over the parameters θ of our
q0 (appearing in fθ that maps an x to a y), we get
a lower bound on H: H ≥maxθ′,θ{EEq0( e
X)[L(q0, e
X)]}
Maximizing this lower bound, we ﬁnd
arg max
θ,θ′
{EEq0( e
X)[L(q0, e
X)]}
= arg max
θ,θ′
EEq0(X, e
X,Y )
"
log p(X, e
X, Y )
q0(X, Y | e
X)
#
= arg max
θ,θ′
EEq0(X, e
X,Y)
h
log p(X, e
X, Y)
i
+ EEq0( e
X)
h
IH[q0(X, Y | e
X)]
i
= arg max
θ,θ′
EEq0(X, e
X,Y )
h
log p(X, e
X, Y )
i
.
Note that θ only occurs in Y = fθ( e
X), and θ′ only
occurs in p(X|Y ). The last line is therefore obtained
because q0(X| e
X) ∝qD( e
X|X)q0(X) (none of which de-
pends on (θ, θ′)), and q0(Y | e
X) is deterministic, i.e., its
entropy is constant, irrespective of (θ, θ′). Hence the
entropy of q0(X, Y | e
X) = q0(Y | e
X)q0(X| e
X), does not
vary with (θ, θ′).
Finally, following from above, we
obtain our training criterion (eq. 5):
arg max
θ,θ′
EEq0( e
X)[L(q0, e
X)]
= arg max
θ,θ′
EEq0(X, e
X,Y )[log[p(Y )p(X|Y )p( e
X|X)]]
= arg max
θ,θ′
EEq0(X, e
X,Y )[log p(X|Y )]
= arg max
θ,θ′
EEq0(X, e
X)[log p(X|Y = fθ( e
X))]
= arg min
θ,θ′
EEq0(X, e
X)
h
LIH

X, gθ′(fθ( e
X))
i
where the third line is obtained because (θ, θ′)
have no inﬂuence on EEq0(X, e
X,Y )[log p(Y )] because
we chose p(Y ) uniform,
i.e.
constant,
nor on
EEq0(X, e
X)[log p( e
X|X)], and the last line is obtained
by inspection of the deﬁnition of LIH in eq. 2, when
p(X|Y = fθ( e
X)) is a Bgθ′(fθ( e
X)).
4.3. Other Theoretical Perspectives
Information Theoretic Perspective:
Consider
X ∼q(X), q unknown, Y = fθ( e
X).
It can easily
be shown (Vincent et al., 2008) that minimizing the
expected reconstruction error amounts to maximizing

Extracting and Composing Robust Features with Denoising Autoencoders
a lower bound on mutual information I(X; Y ). Denois-
ing autoencoders can thus be justiﬁed by the objective
that Y captures as much information as possible about
X even as Y is a function of corrupted input.
Stochastic Operator Perspective: Extending the
manifold perspective, the denoising autoencoder can
also be seen as corresponding to a semi-parametric
model from which we can sample (Vincent et al., 2008):
p(X) = 1
n
Pn
i=1
P
˜x p(X| e
X = ˜x)qD(˜x|xi),
where xi is one of the n training examples.
5. Experiments
We performed experiments with the proposed algo-
rithm on the same benchmark of classiﬁcation prob-
lems used in (Larochelle et al., 2007)3.
It contains
diﬀerent variations of the MNIST digit classiﬁcation
problem (input dimensionality d = 28 × 28 = 784),
with added factors of variation such as rotation (rot),
addition of a background composed of random pixels
(bg-rand) or made from patches extracted from a set of
images (bg-img), or combinations of these factors (rot-
bg-img).
These variations render the problems par-
ticularly challenging for current generic learning al-
gorithms.
Each problem is divided into a training,
validation, and test set (10000, 2000, 50000 examples
respectively). A subset of the original MNIST prob-
lem is also included with the same example set sizes
(problem basic). The benchmark also contains addi-
tional binary classiﬁcation problems: discriminating
between convex and non-convex shapes (convex), and
between wide and long rectangles (rect, rect-img).
Neural networks with 3 hidden layers initialized by
stacking denoising autoencoders (SdA-3), and ﬁne
tuned on the classiﬁcation tasks, were evaluated on all
the problems in this benchmark. Model selection was
conducted following a similar procedure as Larochelle
et al. (2007). Several values of hyper parameters (de-
struction fraction ν, layer sizes, number of unsuper-
vised training epochs) were tried, combined with early
stopping in the ﬁne tuning phase. For each task, the
best model was selected based on its classiﬁcation per-
formance on the validation set.
Table 1 reports the resulting classiﬁcation error on the
test set for the new model (SdA-3), together with the
performance reported in Larochelle et al. (2007)4 for
3All the datasets for these problems are available at
http://www.iro.umontreal.ca/∼lisa/icml2007.
4Except that rot and rot-bg-img, as reported on the web-
site from which they are available, have been regenerated
since Larochelle et al. (2007), to ﬁx a problem in the initial
data generation process. We used the updated data and
corresponding benchmark results given on this website.
SVMs with Gaussian and polynomial kernels, 1 and 3
hidden layers deep belief network (DBN-1 and DBN-3)
and a 3 hidden layer deep network initialized by stack-
ing basic autoencoders (SAA-3). Note that SAA-3 is
equivalent to a SdA-3 with ν = 0% destruction. As can
be seen in the table, the corruption+denoising train-
ing works remarkably well as an initialization step, and
in most cases yields signiﬁcantly better classiﬁcation
performance than basic autoencoder stacking with no
noise. On all but one task the SdA-3 algorithm per-
forms on par or better than the best other algorithms,
including deep belief nets. Due to space constraints,
we do not report all selected hyper-parameters in the
table (only showing ν).
But it is worth mentioning
that, for the majority of tasks, the model selection
procedure chose best performing models with an over-
complete ﬁrst hidden layer representation (typically
of size 2000 for the 784-dimensional MNIST-derived
tasks). This is very diﬀerent from the traditional “bot-
tleneck” autoencoders, and made possible by our de-
noising training procedure. All this suggests that the
proposed procedure was indeed able to produce more
useful feature detectors.
Next, we wanted to understand qualitatively the ef-
fect of the corruption+denoising training. To this end
we display the ﬁlters obtained after initial training of
the ﬁrst denoising autoencoder on MNIST digits. Fig-
ure 3 shows a few of these ﬁlters as little image patches,
for diﬀerent noise levels. Each patch corresponds to a
row of the learnt weight matrix W, i.e. the incoming
weights of one of the hidden layer neurons. The beneﬁ-
cial eﬀect of the denoising training can clearly be seen.
Without the denoising procedure, many ﬁlters appear
to have learnt no interesting feature. They look like
the ﬁlters obtained after random initialization.
But
when increasing the level of destructive corruption, an
increasing number of ﬁlters resemble sensible feature
detectors. As we move to higher noise levels, we ob-
serve a phenomenon that we expected: ﬁlters become
less local, they appear sensitive to larger structures
spread out across more input dimensions.
6. Conclusion and Future Work
We have introduced a very simple training principle
for autoencoders, based on the objective of undoing a
corruption process. This is motivated by the goal of
learning representations of the input that are robust to
small irrelevant changes in input. We also motivated
it from a manifold learning perspective and gave an
interpretation from a generative model perspective.
This principle can be used to train and stack autoen-
coders to initialize a deep neural network.
A series

Extracting and Composing Robust Features with Denoising Autoencoders
Table 1. Comparison of stacked denoising autoencoders (SdA-3) with other models.
Test error rate on all considered classiﬁcation problems is reported together with a 95% conﬁdence interval. Best performer
is in bold, as well as those for which conﬁdence intervals overlap. SdA-3 appears to achieve performance superior or
equivalent to the best other model on all problems except bg-rand. For SdA-3, we also indicate the fraction ν of destroyed
input components, as chosen by proper model selection. Note that SAA-3 is equivalent to SdA-3 with ν = 0%.
Dataset
SVMrbf
SVMpoly
DBN-1
SAA-3
DBN-3
SdA-3 (ν)
basic
3.03±0.15
3.69±0.17
3.94±0.17
3.46±0.16
3.11±0.15
2.80±0.14 (10%)
rot
11.11±0.28
15.42±0.32
14.69±0.31
10.30±0.27
10.30±0.27
10.29±0.27 (10%)
bg-rand
14.58±0.31
16.62±0.33
9.80±0.26
11.28±0.28
6.73±0.22
10.38±0.27 (40%)
bg-img
22.61±0.37
24.01±0.37
16.15±0.32
23.00±0.37
16.31±0.32
16.68±0.33 (25%)
rot-bg-img
55.18±0.44
56.41±0.43
52.21±0.44
51.93±0.44
47.39±0.44
44.49±0.44 (25%)
rect
2.15±0.13
2.15±0.13
4.71±0.19
2.41±0.13
2.60±0.14
1.99±0.12 (10%)
rect-img
24.04±0.37
24.05±0.37
23.69±0.37
24.05±0.37
22.50±0.37
21.59±0.36 (25%)
convex
19.13±0.34
19.82±0.35
19.92±0.35
18.41±0.34
18.63±0.34
19.06±0.34 (10%)
(a) No destroyed inputs
(b) 25% destruction
(c) 50% destruction
(d) Neuron A (0%, 10%, 20%, 50% destruction)
(e) Neuron B (0%, 10%, 20%, 50% destruction)
Figure 3. Filters obtained after training the ﬁrst denoising autoencoder.
(a-c) show some of the ﬁlters obtained after training a denoising autoencoder on MNIST samples, with increasing
destruction levels ν. The ﬁlters at the same position in the three images are related only by the fact that the autoencoders
were started from the same random initialization point.
(d) and (e) zoom in on the ﬁlters obtained for two of the neurons, again for increasing destruction levels.
As can be seen, with no noise, many ﬁlters remain similarly uninteresting (undistinctive almost uniform grey patches).
As we increase the noise level, denoising training forces the ﬁlters to diﬀerentiate more, and capture more distinctive
features. Higher noise levels tend to induce less local ﬁlters, as expected. One can distinguish diﬀerent kinds of ﬁlters,
from local blob detectors, to stroke detectors, and some full character detectors at the higher noise levels.

Extracting and Composing Robust Features with Denoising Autoencoders
of image classiﬁcation experiments were performed to
evaluate this new training principle. The empirical re-
sults support the following conclusions: unsupervised
initialization of layers with an explicit denoising crite-
rion helps to capture interesting structure in the input
distribution. This in turn leads to intermediate rep-
resentations much better suited for subsequent learn-
ing tasks such as supervised classiﬁcation. It is possi-
ble that the rather good experimental performance of
Deep Belief Networks (whose layers are initialized as
RBMs) is partly due to RBMs encapsulating a simi-
lar form of robustness to corruption in the represen-
tations they learn, possibly because of their stochas-
tic nature which introduces noise in the representation
during training. Future work inspired by this observa-
tion should investigate other types of corruption pro-
cess, not only of the input but of the representation
itself as well.
Acknowledgments
We thank the anonymous reviewers for their useful
comments that helped improved the paper.
We are
also very grateful for ﬁnancial support of this work by
NSERC, MITACS and CIFAR.
References
Bengio, Y. (2007).
Learning deep architectures for AI
(Technical Report 1312). Universit´e de Montr´eal, dept.
IRO.
Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H.
(2007).
Greedy layer-wise training of deep networks.
Advances in Neural Information Processing Systems 19
(pp. 153–160). MIT Press.
Bengio, Y., & Le Cun, Y. (2007). Scaling learning algo-
rithms towards AI. In L. Bottou, O. Chapelle, D. De-
Coste and J. Weston (Eds.), Large scale kernel machines.
MIT Press.
Bishop, C. M. (1995). Training with noise is equivalent to
tikhonov regularization.
Neural Computation, 7, 108–
116.
Doi, E., Balcan, D. C., & Lewicki, M. S. (2006). A theo-
retical analysis of robust coding over noisy overcomplete
channels. In Y. Weiss, B. Sch¨olkopf and J. Platt (Eds.),
Advances in neural information processing systems 18,
307–314. Cambridge, MA: MIT Press.
Doi, E., & Lewicki, M. S. (2007). A theory of retinal pop-
ulation coding. NIPS (pp. 353–360). MIT Press.
Elad, M., & Aharon, M. (2006). Image denoising via sparse
and redundant representations over learned dictionaries.
IEEE Transactions on Image Processing, 15, 3736–3745.
Gallinari, P., LeCun, Y., Thiria, S., & Fogelman-Soulie, F.
(1987). Memoires associatives distribuees. Proceedings
of COGNITIVA 87. Paris, La Villette.
Hammond, D., & Simoncelli, E. (2007). A machine learning
framework for adaptive combination of signal denoising
methods. 2007 International Conference on Image Pro-
cessing (pp. VI: 29–32).
Hinton, G. (1989). Connectionist learning procedures. Ar-
tiﬁcial Intelligence, 40, 185–234.
Hinton, G., & Salakhutdinov, R. (2006).
Reducing the
dimensionality of data with neural networks. Science,
313, 504–507.
Hinton, G. E., Osindero, S., & Teh, Y. (2006). A fast learn-
ing algorithm for deep belief nets. Neural Computation,
18, 1527–1554.
Hopﬁeld, J. (1982). Neural networks and physical systems
with emergent collective computational abilities. Pro-
ceedings of the National Academy of Sciences, USA, 79.
Larochelle, H., Erhan, D., Courville, A., Bergstra, J., &
Bengio, Y. (2007). An empirical evaluation of deep ar-
chitectures on problems with many factors of variation.
Proceedings of the 24 th International Conference on Ma-
chine Learning (ICML’2007) (pp. 473–480).
LeCun,
Y.
(1987).
Mod`eles
connexionistes
de
l’apprentissage.
Doctoral
dissertation,
Universit´e
de Paris VI.
Lee, H., Ekanadham, C., & Ng, A. (2008). Sparse deep be-
lief net model for visual area V2. In J. Platt, D. Koller,
Y. Singer and S. Roweis (Eds.), Advances in neural in-
formation processing systems 20. Cambridge, MA: MIT
Press.
McClelland,
J.,
Rumelhart,
D.,
&
the
PDP
Re-
search Group (1986).
Parallel distributed processing:
Explorations in the microstructure of cognition, vol. 2.
Cambridge: MIT Press.
Memisevic, R. (2007). Non-linear latent factor models for
revealing structure in high-dimensional data. Doctoral
dissertation, Departement of Computer Science, Univer-
sity of Toronto, Toronto, Ontario, Canada.
Ranzato, M., Boureau, Y.-L., & LeCun, Y. (2008). Sparse
feature learning for deep belief networks. In J. Platt,
D. Koller, Y. Singer and S. Roweis (Eds.), Advances in
neural information processing systems 20. Cambridge,
MA: MIT Press.
Ranzato, M., Poultney, C., Chopra, S., & LeCun, Y.
(2007). Eﬃcient learning of sparse representations with
an energy-based model. Advances in Neural Information
Processing Systems (NIPS 2006). MIT Press.
Roth, S., & Black, M. (2005). Fields of experts: a frame-
work for learning image priors.
IEEE Conference on
Computer Vision and Pattern Recognition (pp. 860–
867).
Utgoﬀ, P., & Stracuzzi, D. (2002). Many-layered learning.
Neural Computation, 14, 2497–2539.
Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A.
(2008). Extracting and composing robust features with
denoising autoencoders (Technical Report 1316). Uni-
versit´e de Montr´eal, dept. IRO.
