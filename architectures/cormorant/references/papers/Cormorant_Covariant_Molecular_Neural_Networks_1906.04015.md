# Paper (Anderson et al. 2019)

> Source: `https://arxiv.org/abs/1906.04015`

---

arXiv:1906.04015v3  [physics.comp-ph]  25 Nov 2019
Cormorant: Covariant Molecular Neural Networks
Brandon Anderson∗‡, Truong-Son Hy∗and Risi Kondor∗†♯
∗Department of Computer Science, †Department of Statistics
The University of Chicago
♯Center for Computational Mathematics, Flatiron Institute
‡ Atomwise
{hytruongson,risi}@uchicago.edu
brandona@jfi.uchicago.edu
Abstract
We propose Cormorant, a rotationally covariant neural network architecture for
learning the behavior and properties of complex many-body physical systems.
We apply these networks to molecular systems with two goals: learning atomic
potential energy surfaces for use in Molecular Dynamics simulations, and learn-
ing ground state properties of molecules calculated by Density Functional Theory.
Some of the key features of our network are that (a) each neuron explicitly corre-
sponds to a subset of atoms; (b) the activation of each neuron is covariant to rota-
tions, ensuring that overall the network is fully rotationally invariant. Furthermore,
the non-linearity in our network is based upon tensor products and the Clebsch-
Gordan decomposition, allowing the network to operate entirely in Fourier space.
Cormorant signiﬁcantly outperforms competing algorithms in learning molecular
Potential Energy Surfaces from conformational geometries in the MD-17 dataset,
and is competitive with other methods at learning geometric, energetic, electronic,
and thermodynamic properties of molecules on the GDB-9 dataset.
1
Introduction
In principle, quantum mechanics provides a perfect description of the forces governing the behavior
of atoms, molecules and crystalline materials such as metals. However, for systems larger than a few
dozen atoms, solving the Schrödinger equation explicitly at every timestep is not a feasible propo-
sition on present day computers. Even Density Functional Theory (DFT) [Hohenberg and Kohn,
1964], a widely used approximation to the equations of quantum mechanics, has trouble scaling to
more than a few hundred atoms.
Consequently, the majority of practical work in molecular dynamics today falls back on fundamen-
tally classical models, where the atoms are essentially treated as solid balls and the forces between
them are given by pre-deﬁned formulae called atomic force ﬁelds or empirical potentials, such as the
CHARMM family of models [Brooks et al., 1983, 2009]. There has been a widespread realization
that this approach has inherent limitations, so in recent years a burgeoning community has formed
around trying to use machine learning to learn more descriptive force ﬁelds directly from DFT
computations [Behler and Parrinello, 2007, Bartók et al., 2010, Rupp et al., 2012, Shapeev, 2015,
Chmiela et al., 2016, Zhang et al., 2018, Schütt et al., 2017, Hirn et al., 2017]. More broadly, there
is considerable interest in using ML methods not just for learning force ﬁelds, but also for predict-
ing many other physical/chemical properties of atomic systems across different branches of materi-
als science, chemistry and pharmacology [Montavon et al., 2013, Gilmer et al., 2017b, Smith et al.,
2017, Yao et al., 2018].
At the same time, there have been signiﬁcant advances in our understanding of the equivariance
and covariance properties of neural networks, starting with [Cohen and Welling, 2016a,b] in the
33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

context of traditional convolutional neural nets (CNNs).
Similar ideas underly generalizations
of CNNs to manifolds [Masci et al., 2015, Monti et al., 2016, Bronstein et al., 2017] and graphs
[Bruna et al., 2014, Henaff et al., 2015]. In the context of CNNs on the sphere, Cohen et al. [2018]
realized the advantage of using “Fourier space” activations, i.e., expressing the activations of neu-
rons in a basis deﬁned by the irreducible representations of the underlying symmetry group (see also
[Esteves et al., 2017]), and these ideas were later generalized to the entire SE(3) group [Weiler et al.,
2018]. Kondor and Trivedi [2018] gave a complete characterization of what operations are allowable
in Fourier space neural networks to preserve covariance, and Cohen et al generalized the framework
even further to arbitrary gauge ﬁelds [Cohen et al., 2019]. There have also been some recent works
where even the nonlinear part of the neural network’s operation is performed in Fourier space: inde-
pendently of each other [Thomas et al., 2018] and [Kondor, 2018] were to ﬁrst to use the Clebsch–
Gordan transform inside rotationally covariant neural networks for learning physical systems, while
[Kondor et al., 2018] showed that in spherical CNNs the Clebsch–Gordan transform is sufﬁcient to
serve as the sole source of nonlinearity.
The Cormorant neural network architecture proposed in the present paper combines some of the
insights gained from the various force ﬁeld and potential learning efforts with the emerging theory
of Fourier space covariant/equivariant neural networks. The important point that we stress in the
following pages is that by setting up the network in such a way that each neuron corresponds to
an actual set of physical atoms, and that each activation is covariant to symmetries (rotation and
translation), we get a network in which the “laws” that individual neurons learn resemble known
physical interactions. Our experiments show that this generality pays off in terms of performance
on standard benchmark datasets.
2
The nature of physical interactions in molecules
Ultimately interactions in molecular systems arise from the quantum structure of electron clouds
around constituent atoms. However, from a chemical point of view, effective atom-atom interactions
break down into a few simple classes based upon symmetry. Here we review a few of these classes in
the context of the multipole expansion, whose structure will inform the design of our neural network.
Scalar interactions.
The simplest type of physical interaction is that between two particles that
are pointlike and have no internal directional degrees of freedom, such as spin or dipole moments.
A classical example is the electrostatic attraction/repulsion between two charges described by the
Coulomb energy
VC = −
1
4πǫ0
qAqB
|rAB| .
(1)
Here qA and qB are the charges of the two particles, rA and rB are their position vectors, rAB =
rA −rB, and ǫ0 is a universal constant. Note that this equation already reﬂects symmetries: the fact
that (1) only depends on the length of rAB and not its direction or the position vectors individually
guarantees that the potential is invariant under both translations and rotations.
Dipole/dipole interactions.
One step up from the scalar case is the interaction between two
dipoles. In general, the electrostatic dipole moment of a set of N charged particles relative to their
center of mass r is just the ﬁrst moment of their position vectors weighted by their charges:
µ =
N
X
i=1
qi(ri −r).
The dipole/dipole contribution to the electrostatic potential energy between two sets of particles A
and B separated by a vector rAB is then given by
Vd/d =
1
4πǫ0
µA · µB
|rAB|3 −3 (µA · rAB)(µB · rAB)
|rAB|5

.
(2)
One reason why dipole/dipole interactions are indispensible for capturing the energetics of
molecules is that most chemical bonds are polarized. However, dipole/dipole interactions also occur
in other contexts, such as the interaction between the magnetic spins of electrons.
2

Quadropole/quadropole interactions.
One more step up the multipole hierarchy is the interac-
tion between quadropole moments. In the electrostatic case, the quadropole moment is the second
moment of the charge density (corrected to remove the trace), described by the matrix
Θ =
N
X
i=1
qi(3rir⊤
i −|ri|2I).
Quadropole/quadropole interactions appear for example when describing the interaction between
benzene rings, but the general formula for the corresponding potential is quite complicated. As a
simpliﬁcation, let us only consider the special case when in some coordinate system aligned with
the structure of A, and at polar angle (θA, φA) relative to the vector rAB connecting A and B, ΘA
can be transformed into a form that is diagonal, with [ΘA]zz = ϑA and [ΘA]xx = [ΘA]yy = −ϑA/2
[Stone, 1997]. We make a similar assumption about the quadropole moment of B. In this case the
interaction energy becomes
Vq/q = 3
4
ϑAϑB
4πǫ0 |rAB|5

1 −5 cosθ
A −5 cos2 θB −15 cos2 θA cos2 θB+
2(4 cosθAθB −sin θA sin θB cos(φA −φB))2
.
(3)
Higher order interactions involve moment tensors of order 3,4,5, and so on. One can appreciate that
the corresponding formulae, especially when considering not just electrostatics but other types of
interactions as well (dispersion, exchange interaction, etc), quickly become very involved.
3
Spherical tensors and representation theory
Fortunately, there is an alternative formalism for expressing molecular interactions, that of spherical
tensors, which makes the general form of physically allowable interactions more transparent. This
formalism also forms the basis of the our Cormorant networks described in the next section.
The key to spherical tensors is understanding how physical quantities transform under rotations.
Speciﬁcally, in our case, under a rotation R:
q 7−→q
µ 7−→Rµ
Θ 7−→RΘR⊤
rAB 7−→R rAB.
Flattening Θ into a vector Θ ∈R9, its transformation rule can equivalently be written as
Θ 7→(R⊗R) Θ, showing its similarity to the other three cases. In general, a k’th order Carte-
sian moment tensor T (k) ∈R3×3×...×3 (or its ﬂattened T (k) ∈R3k equivalent) transforms as
T (k) 7→(R⊗R⊗. . . ⊗R) T (k).
Recall that given a group G, a representation ρ of G is a matrix valued function ρ: G →Cd×d
obeying ρ(xy) = ρ(x)ρ(y) for any two group elements x, y ∈G. It is easy to see that R, and
consequently R ⊗. . . ⊗R are representations of the three dimensional rotation group SO(3). We
also know that because SO(3) is a compact group, it has a countable sequence of unitary so-called
irreducible representations (irreps), and, up to a similarity transformation, any representation can
be reduced to a direct sum of irreps. In the speciﬁc case of SO(3), the irreps are called Wigner
D-matrices and for any positive integer ℓ= 0, 1, 2, . . . there is a single corresponding irrep Dℓ(R),
which is a (2ℓ+ 1) dimensional representation (i.e., as a function, Dℓ: SO(3) →C(2ℓ+1)×(2ℓ+1)).
The ℓ= 0 irrep is the trivial irrep D0(R) = (1).
The above imply that there is a ﬁxed unitary transformation matrix C(k) which reduces the k’th
order rotation operator into a direct sum of irreducible representations:
R⊗R⊗. . . ⊗R
|
{z
}
k
= C(k)hM
ℓ
τℓ
M
i=1
Dℓ(R)
i
C(k)†.
Note that the transformation R⊗R⊗. . . ⊗R contains redundant copies of Dℓ(R), which we de-
note as the multiplicites τℓ. For our present purposes knowing the actual values of the τℓis not that
important, except that τk = 1 and that for any ℓ> k, τℓ= 0. What is important is that T (k), the
vectorized form of the Cartesian moment tensor has a corresponding decomposition
T (k) = C(k)hM
ℓ
τℓ
M
i=1
Qℓ,i
i
.
(4)
3

This is nice, because using the unitarity of Qℓi, it shows that under rotations the individual Qℓ,i
components transform independently as Qℓ,i 7→Dℓ(R)Qℓ,i.
What we have just described is a form of generalized Fourier analysis applied to the transforma-
tion of Cartesian tensors under rotations. For the electrostatic multipole problem it is particularly
relevant, because it turns out that in that case, due to symmetries of T (k), the only nonzero Qℓ,i com-
ponent of (4) is the single one with ℓ= k. Furthermore, for a set of N charged particles (indexing
its components −ℓ, . . . , ℓ) Qℓhas the simple form
[Qℓ]m =
 4π
2ℓ+1
1/2
N
X
i=1
qi (ri)ℓY m
ℓ(θi, φi)
m = −ℓ, . . . , ℓ,
(5)
where (ri, θi, φi) are the coordinates of the i’th particle in spherical polars, and the Y m
ℓ(θ, φ) are
the well known spherical harmonic functions. Qℓis called the ℓ’th spherical moment of the charge
distribution. Note that while T (ℓ) and Qℓconvey exactly the same information, T (ℓ) is a tensor with
3ℓcomponents, while Qℓis just a (2ℓ+1) dimensional vector.
Somewhat confusingly, in physics and chemistry any quantity U that transforms under rotations as
U 7→Dℓ(R)U is often called an (ℓ’th order) spherical tensor, despite the fact that in terms of its
presentation Qℓis just a vector of 2ℓ+ 1 numbers. Also note that since D0(R) = (1), a zeroth
order spherical tensor is just a scalar. A ﬁrst order spherical tensor, on the other hand, can be used
to represent a spatial vector r = (r, θ, φ) by setting [U1]m = r Y m
1 (θ, φ).
3.1
The general form of interactions
The beneﬁt of the spherical tensor formalism is that it makes it very clear how each part of a given
physical equation transforms under rotations. For example, if Qℓand eQℓare two ℓ’th order spherical
tensors, then Q†
ℓeQℓis a scalar, since under a rotation R, by the unitarity of the Wigner D-matrices,
Q†
ℓeQℓ7−→(Dℓ(R) Qℓ)† (Dℓ(R) eQℓ) = Q†
ℓ(Dℓ(R))† Dℓ(R) eQℓ= Q†
ℓeQℓ.
Even the dipole/dipole interaction (2) requires a more sophisticated way of coupling spherical ten-
sors than this, since it involves non-trivial interactions between not just two, but three different quan-
tites: the two dipole moments µA and µB and the the relative position vector rAB. Representing
interactions of this type requires taking tensor products of the constituent variables. For example,
in the dipole/dipole case we need terms of the form QA
ℓ1 ⊗QB
ℓ2. Naturally, these will transform
according to the tensor product of the corresponding irreps:
QA
ℓ1 ⊗QB
ℓ2 7→(Dℓ1(R)⊗Dℓ2(R)) (QA
ℓ1 ⊗QB
ℓ2).
In general, Dℓ1(R)⊗Dℓ2(R) is not an irreducible representation. However it does have a well
studied decomposition into irreducibles, called the Clebsch–Gordan decomposition:
Dℓ1(R)⊗Dℓ2(R) = C†
ℓ1,ℓ2

ℓ1+ℓ2
M
ℓ=|ℓ1−ℓ2|
Dℓ(R)

Cℓ1,ℓ2.
Letting Cℓ1,ℓ2,ℓ∈C(2ℓ+1)×(2ℓ1+1)(2ℓ2+2) be the block of 2ℓ+1 rows in Cℓ1,ℓ2 corresponding to the
ℓcomponent of the direct sum, we see that Cℓ1,ℓ2,ℓ(QA
ℓ1 ⊗QB
ℓ2) is an ℓ’th order spherical tensor. In
particular, given some other spherical tensor quantity Uℓ,
U †
ℓ· Cℓ1,ℓ2,ℓ· (QA
ℓ1 ⊗QB
ℓ2)
is a scalar, and hence it is a candidate for being a term in the potential energy. Note the similarity
of this expression to the bispectrum [Kakarala, 1992, Bendory et al., 2018], which is an already
established tool in the force ﬁeld learning literature [Bartók et al., 2013].
Almost any rotation invariant interaction potential can be expressed in terms of iterated Clebsch–
Gordan products between spherical tensors. In particular, the full electrostatic energy between two
sets of charges A and B separated by a vector r = (r, θ, φ) expressed in multipole form [Jackson,
1999] is
VAB =
1
4πǫ0
∞
X
ℓ=0
∞
X
ℓ′=0
s2ℓ+ 2ℓ′
2ℓ
r
4π
2ℓ+2ℓ′ + 1 r−(ℓ+ℓ′+1) Yℓ+ℓ′(θ, φ) Cℓ1,ℓ2,ℓ+ℓ′ (QA
ℓ⊗QB
ℓ′).
(6)
4

Note the generality of this formula: the ℓ= ℓ′ = 1 case covers the dipole/dipole interaction (2),
the ℓ= ℓ′ = 2 case covers the quadropole/quadropole interaction (3), while the other terms cover
every other possible type of multipole/multipole interaction. Magnetic and other types of interac-
tions, including interactions that involve 3-way or higher order terms, can also be recovered from
appropriate combinations of tensor products and Clebsch–Gordan decompositions.
We emphasize that our discussion of electrostatics is only intended to illustrate the algebraic struc-
ture of interatomic interactions of any type, and is not restricted to electrostatics. In what follows, we
will not explicitly specify what interactions the network will learn. Nevertheless, there are physical
constraints on the interactions arising from symmetries, which we explicitly impose in our design of
Cormorant.
4
CORMORANT: COvaRiant MOleculaR Artiﬁcial Neural neTworks
The goal of using ML in molecular problems is not to encode known physical laws, but to provide
a platform for learning interactions from data that cannot easily be captured in a simple formula.
Nonetheless, the mathematical structure of known physical laws, like those discussed in the previ-
ous sections, give strong hints about how to represent physical interactions in algorithms. In partic-
ular, when using machine learning to learn molecular potentials or similar rotation and translation
invariant physical quantities, it is essential to make sure that the algorithm respects these invariances.
Our Cormorant neural network has invariance to rotations baked into its architecture in a way that
is similar to the physical equations of the previous section: the internal activations are all spherical
tensors, which are then combined at the top of the network in such a way as to guarantee that the
ﬁnal output is a scalar (i.e., is invariant). However, to allow the network to learn interactions that are
more complicated than classical interatomic forces, we allow each neuron to output not just a single
spherical tensor, but a combination of spherical tensors of different orders. We will call an object
consisting of τ0 scalar components, τ1 components transforming as ﬁrst order spherical tensors, τ2
components transforming as second order spherical tensors, and so on, an SO(3)–covariant vector
of type (τ0, τ1, τ2, . . .). The output of each neuron in Cormorant is an SO(3)–vector of a ﬁxed type.
Deﬁnition 1. We say that F is an SO(3)-covariant vector of type τ = (τ0, τ1, τ2, . . . , τL) if it can
be written as a collection of complex matrices F0, F1, . . . , FL, called its isotypic parts, where each
Fℓis a matrix of size (2ℓ+1)×τℓand transforms under rotations as Fℓ7→Dℓ(R) Fℓ.
The second important feature of our architecture is that each neuron corresponds to either a single
atom or a set of atoms forming a physically meaningful subset of the system at hand, for example
all atoms in a ball of a given radius. This condition helps encourage the network to learn physically
meaningful and interpretable interactions. The high level deﬁnition of Cormorant nets is as follows.
Deﬁnition 2. Let S be a molecule or other physical system consisting of N atoms. A “Cormorant”
covariant molecular neural network for S is a feed forward neural network consisting of m neurons
n1, . . . , nm, such that
C1. Every neuron ni corresponds to some subset Si of the atoms. In particular, each input neuron
corresponds to a single atom. Each output neuron corresponds to the entire system S.
C2. The activation of each ni is an SO(3)-vector of a ﬁxed type τi.
C3. The type of each output neuron is τout = (1), i.e., a scalar. 1
Condition (C3) guarantees that whatever function a Cormorant network learns will be invariant to
global rotations. Translation invariance is easier to enforce simply by making sure that the interac-
tions represented by individual neurons only involve relative distances.
4.1
Covariant neurons
The neurons in our network must be such that if each of their inputs is an SO(3)–covariant vector
then so is their output. Classically, neurons perform a simple linear operation such as x 7→Wx + b,
followed by a nonlinearity like a ReLU. In convolutional neural nets the weights are tied together in
1Cormorant can learn data of arbitrary SO(3)-vector outputs. We restrict to scalars here to simplify the
exposition.
5

a speciﬁc way which guarantees that the activation of each layer is covariant to the action of global
translations. Kondor and Trivedi [2018] discuss the generalization of convolution to the action of
compact groups (such as, in our case, rotations) and prove that the only possible linear operation that
is covariant with the group action, is what, in terms of SO(3)–vectors, corresponds to multiplying
each Fℓmatrix from the right by some matrix W of learnable weights.
For the nonlinearity, one option would be to express each spherical tensor as a function on SO(3)
using an inverse SO(3) Fourier transform, apply a pointwise nonlinearity, and then transform the
resulting function back into spherical tensors. This is the approach taken in e.g., [Cohen et al.,
2018]. However, in our case this would be forbiddingly costly, as well as introducing quadra-
ture errors by virtual of having to interpolate on the group, ultimately degrading the network’s co-
variance. Instead, taking yet another hint from the structure of physical interactions, we use the
Clebsch–Gordan transform introduced in 3.1 as a nonlinearity. The general rule for taking the CG
product of two SO(3)–parts Fℓ1 ∈C(2ℓ1+1)×n1 and Gℓ2 ∈C(2ℓ2+1)×n2 gives a collection of parts
[Fℓ1 ⊗cg Gℓ2]|ℓ1−ℓ1|, . . . [Fℓ1 ⊗cg Gℓ2]ℓ1+ℓ1 with columns

[Fℓ1 ⊗cg Gℓ2]ℓ

∗,(i1,i2) = Cℓ1,ℓ2,ℓ([Fℓ1]∗,i1 ⊗[Gℓ2]∗,i2) ,
(7)
i.e., every column of Fℓ1 is separately CG-multiplied with every column of Gℓ2. The ℓ’th part of
the CG-product of two SO(3)–vectors consists of the concatenation of all SO(3)–part matrices with
index ℓcoming from multiplying each part of F with each part of G:
[F ⊗cg G]ℓ=
M
ℓ1
M
ℓ2
[Fℓ1 ⊗cg Gℓ2]ℓ.
Here and in the following ⊕denotes the appropriate concatenation of vectors and matrices. In
Cormorant, however, as a slight departure from (7), to reduce the quadratic blow-up in the number
of columns, we always have n1 = n2 and use the restricted “channel-wise” CG-product,

[Fℓ1 ⊗cg Gℓ2]ℓ

∗,i = Cℓ1,ℓ2,ℓ([Fℓ1]∗,i ⊗[Gℓ2]∗,i) ,
where each column of Fℓ1 is only mixed with the corresponding column of Gℓ2. We note that
similar Clebsch–Gordan nonlinearities were used in [Kondor et al., 2018], and that the Clebsch–
Gordan product is also an essential part of Tensor Field Networks [Thomas et al., 2018].
4.2
One-body and two-body interactions
As stated in Deﬁnition 2, the covariant neurons in a Cormorant net correspond to different subsets
of the atoms making up the physical system to be modeled. For simplicty in our present architecture
there are only two types of neurons: those that correspond to individual atoms and those that corre-
spond to pairs. For a molecule consisting of N atoms, each layer s = 0, 1, . . . , S of the covariant
part of the network has N neurons corresponding to the atoms and N 2 neurons corresponding to the
(i, j) atom pairs. By loose analogy with graph neural networks, we call the corresponding F s
i and
gs
i,j activations vertex and edge activations, respectively.
In accordance with the foregoing, each F s
i activation is an SO(3)–vector consisting of L+1 distinct
parts (F s,0
i
, F s,1
i
, . . . , F s,L
i
), i.e., each F s,ℓ
i
is a (2ℓ+1)×τ s
ℓdimensional complex matrix that trans-
forms under rotations as F s,ℓ
i
7→Dℓ(R) F s,ℓ
i
. The different columns of these matrices are regarded
as the different channels of the network, because they fulﬁll a similar role to channels in conven-
tional convolutional nets. The gs
i,j edge activations also break down into parts (gs,0
i,j , gs,1
i,j , . . . , gs,L
i,j ),
but these are invariant under rotations. Again for simplicity, in the version of Cormorant that we used
in our experiments L is the same in every layer (speciﬁcally L = 3), and the number of channels is
also independent of both s and ℓ, speciﬁcally, τ s
ℓ≡nc = 16.
The actual form of the vertex activations captures “one-body interactions” propagating information
from the previous layer related to the same atom and (indirectly, via the edge activations) “two-body
interactions” capturing interactions between pairs of atoms:
F s−1
i
=
h
F s
i ⊕
 F s−1
i
⊗cg F s−1
i

|
{z
}
one-body part
⊕
 X
j
Gs
i,j ⊗cg F s−1
j

|
{z
}
two-body part
i
· W vertex
s,ℓ
.
(8)
6

Here Gs
i,j are SO(3)–vectors arising from the edge network. Speciﬁcally, Gs,ℓ
i,j = gs,ℓ
i,j Y ℓ(bri,j),
where Y ℓ(bri,j) are the spherical harmonic vectors capturing the relative position of atoms i and j.
The edge activations, in turn, are deﬁned
gs,ℓ
i,j = µs(ri,j)
h gs−1,ℓ
i,j
⊕
 F s−1
i
· F s−1
j

⊕ηs,ℓ(ri,j)

W edge
s,ℓ
i
(9)
where we made the ℓ= 0, 1, . . . , L irrep index explicit. As before, in these formulae, ⊕denotes
concatenation over the channel index c, ηs,ℓ
c (ri,j) are learnable radial functions, and µs
c(ri,j) are
learnable cutoff functions limiting the inﬂuence of atoms that are farther away from atom i. The
learnable parameters of the network are the {W vertex
s,ℓ
} and {W edge
s,ℓ} weight matrices.
Note that the F s−1
i
· F s−1
j
dot product term is the only term in these formulae responsible for
the interaction between different atoms, and that this term always appears in conjunction with the
ηs,ℓ
c (ri,j) radial basis functions and µs
c(ri,j) cutoff functions (as well as the SO(3)–covariant spher-
ical harmonic vector) making sure that interaction scales with the distance between the atoms. More
details of these activation rules are given in the Supplement.
4.3
Overall structure and comparison with other architectures
In addition to the covariant neurons described above, our network also needs neurons to compute
the input featurization and the the ﬁnal output after the covariant layers. Thus, in total, a Cormorant
networks consists of three distinct parts:
1. An input featurization network {F s=0
j
} ←INPUT({Zi, ri,j}) that operates only on atomic
charges/identities and (optionally) a scalar function of relative positions ri,j.
2. An S-layer network {F s+1
i
} ←CGNet({F s
i })of covariant activations F s
i , each of which is a
SO(3)-vector of type τ s
i .
3. A rotation invariant network at the top y ←OUTPUT(LS
s=0{F s
i }) that constructs scalars
from the activations F s
i , and uses them to predict a regression target y.
We leave the details of the input and output featurization to the Supplement.
A key difference between Cormorant and other recent covariant networks (Tensor Field Net-
works [Thomas et al., 2018] and SE(3)-equivariant networks [Weiler et al., 2018]) is the use of
Clebsch-Gordan non-linearities. The Clebsch-Gordan non-linearity results in a complete interac-
tion of every degree of freedom in an activation. This comes at the cost of increased difﬁculty in
training, as discussed in the Supplement. We further note that SE(3)-equivariant networks use a
three-dimensional grid of points to represent data, and ensure both translational and rotational co-
variance (equivariance) of each layer. Cormorant on the other hand uses activations that are covariant
to rotations, and strictly invariant to translations.
5
Experiments
We present experimental results on two datasets of interest to the computational chemistry com-
munity: MD-17 for learning molecular force ﬁelds and potential energy surfaces, and QM-9 for
learning the ground state properties of a set of molecules. The supplement provides a detailed sum-
mary of all hyperparameters, our training algorithm, and the details of the input/output levels used
in both cases. Our code is available at https://github.com/risilab/cormorant.
QM9 [Ramakrishnan et al., 2014] is a dataset of approximately 134k small organic molecules con-
taining the atoms H, C, N, O, F. For each molecule, the ground state conﬁguration is calculated using
DFT, along with a variety of molecular properties. We use the ground state conﬁguration as the input
to our Cormorant, and use a common subset of properties in the literature as regression targets. Ta-
ble 1(a) presents our results averaged over three training runs compared with SchNet [Schütt et al.,
2017], MPNNs [Gilmer et al., 2017b], and wavelet scattering networks [Hirn et al., 2017]. Of the
twelve regression targets considered, we achieve leading or competitive results on six (α, ∆ǫ,
ǫHOMO, ǫLUMO, µ, Cv). The remaining four targets are within 40% of the best result, with the
exception of R2.
MD-17 [Chmiela et al., 2016] is a dataset of eight small organic molecules (see Table 1(b)) con-
taining up to 17 total atoms composed of the atoms H, C, N, O, F. For each molecule, an ab
7

Table 1: Mean absolute error of various prediction targets on QM-9 (left) and conformational ener-
gies (in units of kcal/mol) on MD-17 (right). The best results within a standard deviation of three
Cormorant training runs (in parenthesis) are indicated in bold.
Cormorant
SchNet
NMP
WaveScatt
α (bohr3)
0.085
(0.001)
0.235
0.092
0.160
∆ǫ (eV)
0.061
(0.005)
0.063
0.069
0.118
ǫHOMO (eV)
0.034
(0.002)
0.041
0.043
0.085
ǫLUMO (eV)
0.038
(0.008)
0.034
0.038
0.076
µ (D)
0.038
(0.009)
0.033
0.030
0.340
Cv (cal/mol K)
0.026
(0.000)
0.033
0.040
0.049
G (eV)
0.020
(0.000)
0.014
0.019
0.022
H (eV)
0.021
(0.001)
0.014
0.017
0.022
R2 (bohr2)
0.961
(0.019)
0.073
0.180
0.410
U (eV)
0.021
(0.000)
0.019
0.020
0.022
U0 (eV)
0.022
(0.003)
0.014
0.020
0.022
ZPVE (meV)
2.027
(0.042)
1.700
1.500
2.000
Cormorant
DeepMD
DTNN
SchNet
GDML
sGDML
Aspirin
0.098
0.201
–
0.120
0.270
0.190
Benzene
0.023
0.065
0.040
0.070
0.070
0.100
Ethanol
0.027
0.055
–
0.050
0.150
0.070
Malonaldehyde
0.041
0.092
0.190
0.080
0.160
0.100
Naphthalene
0.029
0.095
–
0.110
0.120
0.120
Salicylic Acid
0.066
0.106
0.410
0.100
0.120
0.120
Toluene
0.034
0.085
0.180
0.090
0.120
0.100
Uracil
0.023
0.085
–
0.100
0.110
0.110
initio molecular dynamics simulation was run using DFT to calculate the ground state energy and
forces. At intermittent timesteps, the energy, forces, and conﬁguration (positions of each atom) were
recorded. For each molecule we use a train/validation/test split of 50k/10k/10k atoms respectively.
The results of these experiments are presented in Table 1(b), where the mean-average error (MAE)
is plotted on the test set for each of molecules. (All units are in kcal/mol, as consistent with the
dataset and the literature.) To the best of our knowledge, the current state-of-the art algorithms on
this dataset are DeepMD [Zhang et al., 2018], DTNN [Schütt et al., 2017], SchNet [Schütt et al.,
2017], GDML [Chmiela et al., 2016], and sGDML [Chmiela et al., 2018]. Since training and test-
ing set sizes were not consistent, we used a training set of 50k molecules to compare with all neural
network based approaches. As can be seen from the table, our Cormorant network outperforms all
competitors.
6
Conclusions
To the best of our knowledge, Cormorant is the ﬁrst neural network architecture in which the opera-
tions implemented by the neurons is directly motivated by the form of known physical interactions.
Rotation and translation invariance are explicitly “baked into” the network by the fact all activations
are represented in spherical tensor form (SO(3)–vectors), and the neurons combine Clebsch–Gordan
products, concatenation of parts and mixing with learnable weights, all of which are covariant op-
erations. In future work we envisage the potentials learned by Cormorant to be directly integrated
in MD simulation frameworks. In this regard, it is very encouraging that on MD-17, which is the
standard benchmark for force ﬁeld learning, Cormorant outperforms all other competing methods.
Learning from derivatives (forces) and generalizing to other compact symmetry groups are natural
extensions of the persent work.
Acknowledgements
This project was supported by DARPA “Physics of AI” grant number HR0011837139, and used
computational resources acquired through NSF MRI 1828629.
We thank E. Thiede for helpful discussion and comments on the manuscript.
8

References
Albert P Bartók, Michael C Payne, Risi Kondor, and Gábor Csányi. Gaussian Approximation Potentials: the
accuracy of quantum mechanics, without the electrons. Phys Rev Lett, 104(13):136403, 2010.
Albert P. Bartók, Risi Kondor, and Gábor Csányi. On representing chemical environments. Phys. Rev. B, 87:
184115, May 2013.
Jörg Behler and Michele Parrinello. Generalized neural-network representation of high-dimensional potential-
energy surfaces. Phys Rev Lett, 98(14):146401, 2007.
Tamir Bendory, Nicolas Boumal, Chao Ma, Zhizhen Zhao, and Amit Singer. Bispectrum inversion with appli-
cation to multireference alignment. Trans. Sig. Proc., 66(4):1037–1050, February 2018. ISSN 1053-587X.
doi: 10.1109/TSP.2017.2775591.
M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst. Geometric deep learning: Going beyond
euclidean data. IEEE Signal Process. Mag., 34(4):18–42, July 2017. ISSN 1053-5888. doi: 10.1109/MSP.
2017.2693418.
B. R. Brooks, C. L. Brooks, A. D. Mackerell, L. Nilsson, R. J. Petrella, B. Roux, Y. Won, G. Archontis, C. Bar-
tels, S. Boresch, and et al. CHARMM: the biomolecular simulation program. Journal of Computational
Chemistry, 30(10):1545–1614, Jul 2009. ISSN 1096-987X.
Bernard R. Brooks, Robert E. Bruccoleri, Barry D. Olafson, David J. States, S. Swaminathan, and Martin
Karplus. CHARMM: A program for macromolecular energy, minimization, and dynamics calculations.
Journal of Computational Chemistry, 4(2):187–217, Jun 1983. ISSN 1096-987X.
J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun. Spectral networks and locally connected networks on graphs.
3, 2014.
Stefan Chmiela, Alexandre Tkatchenko, Huziel E. Sauceda, Igor Poltavsky, Kristof T. Schütt, and Klaus-Robert
Müller. Machine Learning of Accurate Energy-Conserving Molecular Force Fields. (May):1–6, 2016. ISSN
2375-2548.
Stefan Chmiela, Huziel E. Sauceda, Klaus-Robert Muller, and Alexandre Tkatchenko. Towards exact molecular
dynamics simulations with machine-learned force ﬁelds. Nature Communications, 9(1):3887, 2018. doi:
10.1038/s41467-018-06169-2. URL https://doi.org/10.1038/s41467-018-06169-2.
Taco S. Cohen and Max Welling. Group equivariant convolutional networks. CoRR, abs/1602.07576, 2016a.
URL http://arxiv.org/abs/1602.07576.
Taco S. Cohen and Max Welling.
Steerable cnns.
CoRR, abs/1612.08498,
2016b.
URL
http://arxiv.org/abs/1612.08498.
Taco S. Cohen, Mario Geiger, Jonas Köhler, and Max Welling. Spherical cnns. CoRR, abs/1801.10130, 2018.
URL http://arxiv.org/abs/1801.10130.
Taco
S.
Cohen,
Maurice
Weiler,
Berkay
Kicanaoglu,
and
Max
Welling.
Gauge
equivari-
ant convolutional networks and the icosahedral CNN.
CoRR, abs/1902.04615,
2019.
URL
http://arxiv.org/abs/1902.04615.
Carlos
Esteves,
Christine
Allen-Blanchette,
Ameesh
Makadia,
and
Kostas
Daniilidis.
3d
ob-
ject classiﬁcation and retrieval
with spherical
cnns.
CoRR, abs/1711.06721,
2017.
URL
http://arxiv.org/abs/1711.06721.
J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for quantum
chemistry. 70, 2017a.
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message pass-
ing for quantum chemistry. CoRR, abs/1704.01212, 2017b. URL http://arxiv.org/abs/1704.01212.
Mikael Henaff, Joan Bruna, and Yann LeCun. Deep convolutional networks on graph-structured data. CoRR,
abs/1506.05163, 2015. URL http://arxiv.org/abs/1506.05163.
M. Hirn, S. Mallat, and N. Poilvert. Wavelet scattering regression of quantum chemical energies. Multiscale
Modeling & Simulation, 15(2):827–863, Jan 2017. ISSN 1540-3459.
P. Hohenberg and W. Kohn. Inhomogeneous electron gas. Phys. Rev., 136:864–871, 1964.
John David Jackson.
Classical electrodynamics.
Wiley, New York, NY, 3rd ed. edition, 1999.
ISBN
9780471309321. URL http://cdsweb.cern.ch/record/490457.
Ramakrishna Kakarala. Triple correlation on groups. PhD thesis, Department of Mathematics, UC Irvine,
1992.
R. Kondor and S. Trivedi. On the generalization of equivariance and convolution in neural networks to the
action of compact groups. International Conference on Machine Learning (ICML), 2018.
Risi Kondor. N-body networks: a covariant hierarchical neural network architecture for learning atomic poten-
tials. CoRR, abs/1803.01588, 2018. URL http://arxiv.org/abs/1803.01588.
Risi Kondor, Zhen Lin, and Shubhendu Trivedi. Clebsch–gordan nets: a fully fourier space spherical con-
volutional neural network. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and
R. Garnett, editors, Advances in Neural Information Processing Systems 31, pages 10117–10126. Curran
9

Associates, Inc., 2018.
Jonathan Masci,
Davide Boscaini,
Michael M. Bronstein,
and Pierre Vandergheynst.
Geodesic
convolutional neural networks on riemannian manifolds.
CoRR, abs/1501.06297,
2015.
URL
http://arxiv.org/abs/1501.06297.
G. Montavon, M. Rupp, V. Gobre, A. Vazquez-Mayagoitia, K. Hansen, A. Tkatchenko, K-R. Müller, and O. A.
von Lilienfeld. Machine learning of molecular electronic properties in chemical compound space. New J.
Phys., 15, 09 2013.
Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, and Michael M. Bronstein.
Geometric deep learning on graphs and manifolds using mixture model cnns. CoRR, abs/1611.08402, 2016.
URL http://arxiv.org/abs/1611.08402.
Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole von Lilienfeld. Quantum chemistry
structures and properties of 134 kilo molecules. Scientiﬁc Data, 1, 2014.
M. Rupp, A. Tkatchenko, K. R. Müller, and O. A. von Lilienfeld. Fast and accurate modeling of molecular
atomization energies with machine learning. Phys. Rev. Lett., 108, 2012.
Kristof Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko,
and Klaus-Robert Müller. Schnet: A continuous-ﬁlter convolutional neural network for modeling quantum
interactions. 2017.
Kristof T. Schütt, Farhad Arbabzadah, Stefan Chmiela, Klaus R. M uller, and Alexandre Tkatchenko. Quantum-
chemical insights from deep tensor neural networks. Nature Communications, 8:13890, Jan 2017. ISSN
2041-1723.
Alexander V Shapeev. Moment Tensor Potentials: a class of systematically improvable interatomic potentials.
arXiv, December 2015.
J. S. Smith, O. Isayev, and A. E. Roitberg. Ani-1: an extensible neural network potential with dft accuracy at
force ﬁeld computational cost. Chem. Sci., 8:3192–3203, 2017. doi: 10.1039/C6SC05720A.
A.J. Stone. The Theory of Intermolecular Forces. International Series of Monographs on Chemistry. Clarendon
Press, 1997. ISBN 9780198558835.
Nathaniel Thomas, Tess Smidt, Steven M. Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley.
Tensor ﬁeld networks: Rotation- and translation-equivariant neural networks for 3d point clouds. CoRR,
abs/1802.08219, 2018.
Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco Cohen.
3d steerable cnns:
Learning rotationally equivariant features in volumetric data.
CoRR, abs/1807.02547, 2018.
URL
http://arxiv.org/abs/1807.02547.
Kun Yao, John E. Herr, David[space]W. Toth, Ryker Mckintyre, and John Parkhill. The tensormol-0.1 model
chemistry: a neural network augmented with long-range physics. Chem. Sci., 9:2261–2269, 2018. doi:
10.1039/C7SC04934J.
Linfeng
Zhang,
Jiequn
Han,
Han
Wang,
Roberto
Car,
and
Weinan
E.
Deep
poten-
tial
molecular
dynamics:
A
scalable
model
with
the
accuracy
of
quantum
mechanics.
Phys.
Rev.
Lett.,
120:143001,
Apr
2018.
doi:
10.1103/PhysRevLett.120.143001.
URL
https://link.aps.org/doi/10.1103/PhysRevLett.120.143001.
10

7
Architecture
As discussed in the main text, our Cormorant architecture is constructed from three basic building
blocks: (1) an input featurization that takes (Zi, ri) and outputs a scalar, (2) a set of covariant CG
layers that update F s
i to F s+1
i
, (3) a layer that takes the set of covariant activations F s
i , and construct
a permutation and rotation invariant regression target.
7.1
Notation
Throughout this section, we will follow the use the main text, and denote a SO(3)-vector at layer s
by F s = (F s
0 , . . . , F s
L) with maximum weight L. Each SO(3)-vector has corresponding type τ s, and
lives in a representation space F s ∈V s = LLs
ℓ=0 ¯V τ s
ℓ
ℓ, where ¯Vℓ= C(2ℓ+1)×1 is the representation
space for irreducible representation of SO(3) with multiplicity 1. We will also introduce the vector
space for the edge network V s
edge = LLs
ℓ=0 Cτ s
ℓ.
See Table ?? for a more complete table of symbols used in the supplement and main text.
7.2
Overall structure
The Cormorant network is a function CORMORANT ({Zi, ri}) : ZN × RN×3 →R that takes a
set of N charge-positions {Zi, ri} and outputs a single regression target. The
CORMORANT ({Zi, ri}) = OUTPUT (CGNet (INPUT ({Zi, ri})))
(S1)
networks are constructed from three basic units:
1. INPUT ({Zi, ri}) : ZN × RN×3 →( ¯V0)N which takes the N charge-position pairs and
outputs N sets of scalar feature vectors cin. (See section 7.3.)
2. CGNet ({Fi, ri}) : ( ¯V0)N × RN×3 →LS
s=0 (V s)N takes the set of scalar features from
INPUT ({Zi, ri}), along with the set of positions for each atom, and outputs a SO(3)-
vector for each level s = 0, . . . , S using Clebsch-Gordan operations. (See section 7.4.)
3. OUTPUT
LS
s=0(V s)N
→R takes the output of CGNet above, constructs a set of
scalars, and then constructs a permutation-invariant prediction that can be exploited at the
top of the network. (See section 7.5.)
This design is organized in a modular way to separate the input featurization, the covariant SO(3)-
vector layers, and the output regression tasks. Importantly, the INPUT and OUTPUT networks are
different for GDB9 and MD17. However, the covariant SO(3)-vector layers CGNet were identical
in design and hyperparameter choice. We include these designs and choices below.
7.3
Input featurization
7.3.1
MD-17
For MD-17, the input featurization was determined by taking the tensor product ˜Fi = onehoti ⊗⃗Zi,
where onehoti is a one-hot vector determining which of Nspecies atomic species an atom is, and⃗
Zi = (1, ˜Zi, ˜Z2
i ), where ˜Zi = Zi/Zmax, and Zmax is the largest charge in the dataset. We then use
a single learnable mixing matrix to convert this real vector with 3 × Nspecies elements to a complex
representation ℓ= 0 and Nc channels (or τi = (nc).)
We found for MD-17, a complex input featurization network was not signiﬁcantly beneﬁcial, and
that this input parametrization was sufﬁciently expressive.
7.3.2
QM-9
For the dataset QM-9, we used an input featurization based upon message passing neural networks.
We start by creating the vector ˜Fi = onehoti ⊗⃗Zi as deﬁned in the previous section. Using this,
a weighted adjacency matrix is constructed using a mask in the same manner as in the main text:
µij = σ((rcut −rij)/w), with learnable cutoffs/width rcut/w and σ(x) = 1/(1 + exp(−x)) . This
1

mask is used to aggregate neighbors ˜F agg
i
= P
j µij ˜Fj. The result is concatenated with ˜F, and
passed through a MLP with a single hidden layer with 256 neurons and ReLU activations with an
output real vector of length 2 × nc. This is then resized to form a complex SO(3)-vector composed
of a single irrep of type τi = (nc).
7.4
Covariant SO(3)-vector layers
For both datasets, the central covariant SO(3)-vector layers of our Cormorant are identical. In both
cases, we used S = 4 layers with L = 3, followed by a single SO(3)-vector layer with L = 0. The
number of channels of the input tensors at each level is ﬁxed to nc = 16, and similarly the set of
weights W reduce the number of channels of each irreducible representation back to nc = 16.
7.4.1
Overview
The algorithm can be implemented as iterating over the function
CGLayer
 gs
ij, F s
i , ri,

: (V s
edge)N×N × RN×N×3 × (V s)N →(V s+1
edge)N×N × (V s+1)N
where gs
ij ∈(V s
edge)N×N and is an edge network at level s with cs channels for each ℓ∈[0, L], and
F s
i ∈(V s)N is an atom-state vector that lives in the representation space at level s.
The function
 gs+1
ij
, F s+1
i

←CGLayer
 gs
ij, F s
i , ri

is itself constructed in the following way:
• gs+1
ij
←EdgeNetwork
 gs
ij, rij, F s
i

• Gs+1
ij
←Edge2Vertex
 gs+1
ij
, Y ℓ(ˆrij)

• F s+1
i
←VertexNetwork
 F s+1
ij
, F s
i

where:
1. EdgeNetwork
 gs
ij, rij, F s
i

: (V s
edge)N×N × RN×N×3 × (V s)N →(V s+1
edge)N×N is a
pair/edge network that combined the input pair matrix gs
ij at level s, with a position network
Fij,c = Fc (|rij|), and dij ∼Fi ·Fj is a matrix of dot products, all of which will be deﬁned
below. This output is then used to construct a set of representations that will be used as the
input to the VertexNetwork function below.
2. Edge2Vertex
 gs+1
ij
, Yij

: (V s
edge)N×N × (V )N×N →(V s)N×N takes the product
of the scalar pair network gs+1
ij
, with the SO(3)-vector of spherical harmonics Yij =
LL
ℓ=0 Y ℓ(ˆrij), to produce a SO(3)-vector of edge scalar representations that will be con-
sidered in the aggregation step in VertexNetwork.
3. VertexNetwork
 Gs+1
ij
, F s
i

: (V s)N×N × (V s)N →(V s+1)N updates the vertex SO(3)-
vector activations by combining a “Clebsch-Gordan aggregation”, a CG non-linearity, a
skip connection, and a linear mixing layer.
7.4.2
Edge networks
Our edge network is an extension of the “edge networks” in Message Passing Neural Networks
Gilmer et al. [2017a]. The EdgeNetwork function takes three different types of pair features, con-
catenates them, and then mixes them. We express write the edge network (Eq. (9)) in the main text)
with all indices explicitly included:
gs+1
ℓc,ij = ms
c,ij ⊙
X
c′
 M
c1
gs
ℓc1,ij ⊕
M
c2
ds
c2,ij ⊕
M
c3
ηℓc3,ij
!
c′

W edge
s,ℓ

c′c
(S2)
where:
• W edge
s,ℓ
is a weight matrix at layer s for each ℓof the edge network.
• gs
ℓc1,ij is a set of edge activations from the previous layer.
2

• ds
c2,ij
=
LL
ℓ=0 F s
ℓc2i · F s
ℓc2j, is a matrix of dot products, where Fℓci · Fℓcj
=
P
m (−1)m (Fℓci,mFℓcj,−m).2
• ηs
ℓc3,ij = ηs
ℓc3 (|rij|) is a set of learnable basis functions. These functions are of the form
ηs
ℓck,n (r) = r−k  sin (2πκs
ℓnr + φs
ℓn) + i sin
 2π¯κs
ℓnr + ¯φs
ℓn

, where κs
ℓn, ¯κs
ℓn, φs
ℓn, and
¯φs
ℓn are learnable parameters, the list of channels c is found by ﬂattening the matrix indexed
by c3 = (k, n), and i2 = −1.
• µs
ℓc,ij is a mask that is used drop the radial functions smoothly to zero. This mask is
constructed through
µc,ij = σ
 −
 rij −rs
c,soft

/ws
c

,
where σ (x) is the sigmoid activation, rs
c,soft is a soft cutoff that drops off with width ws
c.
7.4.3
From edge scalar representations to SO(3)-vector
The function Gs+1
ij
←Edge2Vertex
 gs+1
ij
, Y ℓ(ˆrij)

will take the scalar output of the edge network
gs+1
ℓc,ij, and construct a set of SO(3)-vector representations using spherical harmonics through:
Gs+1
ℓc,ij = gs
ℓc,ijY ℓ(ˆrij)
(S3)
We note the normalization of the spherical harmonics here is not using the “quantum mechanical”
convention, but rather are normalized such that P
m
Y ℓ
m (ˆr)
2 = 1. This is equivalent to scaling the
QM version by Y ℓ
m (ˆr) →
q
2ℓ+1
4π × Y ℓ
m (ˆr).
7.4.4
Vertex networks
The function VertexNetwork is found by concatenating three operations:
F s+1
ℓ,i
=
 VertexNetwork
 Gs+1
ij
, F s
i

ℓ
(S4)
=
X
c′
 M
c1
F s+1,ag
c1,i
⊕
M
c2
F s+1,nl
c2,i
⊕
M
c3
F s+1,id
c3,i
!
ℓ,c′
 W vertex
s,ℓ

c′c
(S5)
where
1. F s+1,ag
i
= P
j∈N(i) Gs+1
ij
⊗cg F s
j is a CG-aggregation step and Gs
ij is the set of edge
representations calculated by Edge2Vertex.
2. F s+1,nl
i
= F s
i ⊗cg F s
i is a CG non-linearity.
3. F s+1,id
i
= F s
i is just the identity function, or equivalently a skip connection.
4. W vertex
s,ℓ
is a atom feature mixing matrix.
7.5
Output featurization
The output featurization of the network starts with the construction of a set of scalar invariants from
the set of activations F s
i for all atoms i and all levels s = 0 . . . S. We extract three scalar invariants
from each activation F (dropping the i and s indices):
1. Take the ℓ= 0 component: ξ0(F) = F s
ℓ=0.
2. Take the scalar product with itself: ξ1(F) = Re[˜ξ1(F)] + Im[˜ξ1(F)] where ˜ξ1(F) =
Pℓ
m=−ℓ(−1)mF s
ℓ,mF s
ℓ,−m.
3. Calculate the SO(3)-invariant norm: ξ2(F s) = Pℓ
m=−ℓF s
ℓ,m

F s
ℓ,m
∗
.
These are then concatenated together to get a ﬁnal set of scalars:
xi
=
LS
s=0 ξ0(F s
i ) ⊕
LL
ℓ=0(ξ1(F s
i ) ⊕ξ2(F s
i )) and fed into the output network network.
2Note that Fℓci · Fℓcj = P
m (−1)m (Fℓci,mFℓcj,−m) is (up to a constant) just the CG decomposi-
tion Cℓℓ0 (Fℓci ⊗Fℓcj).
The speciﬁc matrix elements of the CG coefﬁcients Cℓℓ0 are ⟨ℓm1ℓm2|00⟩∝
(−1)m1 δm1,−m2.
3

7.5.1
MD-17
The output for the MD-17 network is straightforward. The scalars xi are summed over, and then a
single linear layer is applied: y = A (P
i xi) + b.
7.5.2
QM-9
The output for the QM-9 is constructed using two multi-layer perceptrons (MLPs). First, a MLP
is applied to the scalar representation xi at each site. The result is summed over all sites, forming
a single permutation invariant representation of the molecule. This representation is then used to
predict a single number used as the regression target: y = MLP2 (P
i MLP1(xi)). Here, both
MLP1 and MLP2 have a single hidden layer of size 256, and the intermediate representation has 96
neurons.
7.6
Weight initialization
All CG weights W ℓwere initialized uniformly in the interval [−1, 1], and then scaled by a factor of
W ℓ
τ in
ℓ,τ out
ℓ
∼Unif(−1, 1) ∗g/(τ in
ℓ+ τ out
ℓ
), where τ in
ℓ, τ out
ℓ
and g is the weight gain.
We chose the gain to ensure that the activations at each level were order unity when the network is
initialized. We found that if the gain was too low, the CG products in higher levels would not signiﬁ-
cantly contribute to training, and information would only ﬂow through linear (one-body) operations.
This would result in convergence to poor training error. On the other hand, if the gain is set too high,
the CG non-linearities dominate at initialization and would increase the change of the instabilities
discussed above.
In practice, the gain was hand-tuned by such that the mean of the absolute value of the CG activations
1/(NatomNc(2ℓ+ 1)) P
ℓ,i,c,m |F s
ℓ,i,c,m| at each level was approximately unity for a random mini-
batch. For experimental results presented here, we used a gain of g = 5.
8
Experimental details
We trained our network using the AMSGrad [?] optimizer with a constant learning rate of 5 × 10−4
and a mini-batch size of 25. We trained for 512 and 256 epoch respectively for MD-17 and QM-
9.
For each molecule in MD-17, we uniformly sampled 50k/10k/10k data points in the train-
ing/validation/test splits respectively. In QM-9 the dataset was randomly split to 100k molecules in
the train set, with 10% in the test set, and the remaining in the validation set. We removed the 3054
molecules that failed consistency requirements [Ramakrishnan et al., 2014], and also subtracted the
thermochemical energy [Gilmer et al., 2017a] for the targets Cv, U0, U, G, H, ZPVE.
For both datasets, we used, S = 4 CGLayers with L = 3 and we used Nc = 16 channels at the
output of each CGLayer. This gave networks with 299808 and 154241 parameter respectively for
QM-9 and MD-17. Training time for QM-9 is takes roughly 48 hours on a NVidia 2080 Ti GPU.
Training time for MD-17 varies based upon the molecule being trained, but typically ranges between
26 and 30 hours.
8.1
Training instabilities
Training our Cormorant had several subtleties that we both believe are related to the nature of the
CG non-linearity. We found a poor choice of weight initialization or optimization algorithm will
frequently result in either: (1) an instability resulting in very large training loss (> 106), from
which the network will never recover, or (2) convergence to weights where the activation of CG
non-linearities in higher layers turn off, and the resulting training error is poor.
We believe these difﬁculties are a result of the CG non-linearity, which is quadratic and unbounded.
In fact, our network is just a high-order polynomial function of learnable parameters.3 For the
hyperparameters used in our experiments, the prediction at the top is a sixteenth order polynomial
3This is true for MD-17, although for QM-9, the presence of non-linearities in the fully-connected MLPs
adds a more conventional non-linearity.
4

of our network’s parameters. As a result, in certain regions of parameter space small gradient updates
can result in rapid growth of the output amplitude or a rapid drop in the importance of some channels.
These issues were more signiﬁcant when we used Adam [?] then AMSGrad, and when the network’s
parameters were not initialized in a narrow range. Using the weight initialization scheme discussed
in Sec. 7.6, we were able to consistently converge to low training and validation error, provided we
were limited to at most four CG layers.
5

arXiv:1906.04015v3  [physics.comp-ph]  25 Nov 2019
Symbol
Meaning
Zi
Atomic charge of atom i (e.g., Hydrogen would be Zi = 1, and Carbon would be Zi = 6)
ri
Position of atom i
rij
Relative position of atoms i and j: rij = ri −rj
ˆrij
Unit vector in the direction of rij
s
Index of a CGLayer in Cormorant
S
Top CGLayer
ℓ
Index of an irreducible representation of SO (3)
L
Largest index in a representation
m
Component m ∈[−ℓ, ℓ] of a SO (3)-vector
τ s
Vector of multiplicities of a SO (3)-vector at layer s
nc
Number of output channels of a CGLayer
¯Vℓ
Representation space for irrep with index ℓand multiplicity τℓ= 1: ¯Vℓ= C(2ℓ+1)×1
V s
Representation space for SO (3)-vector of type τ s: V s = LL
ℓ=0 ¯V τ s
ℓ
ℓ
V s
edge
Vector space for an scalar edge network at layer s: V s
edge = LL
ℓ=0 Cτ s
ℓ
F s
i
SO (3)-vector atom-activation at layer s for atom i
Y ℓ(ˆrij)
Spherical harmonics of weight ℓin the direction of ˆrij
Yij
SO (3)-vector of spherical harmonics connecting atoms i and j: Yij = LL
ℓ=0 Y ℓ(ˆrij)
gs
ij
Scalar edge activation layer s connecting atoms i and j
ds
ij
Scalar matrix of dot products ds
ij = F s
i · F s
j
ηs
ij
Radial basis functions between atoms i and j
Gs
ij
SO (3)-vector edge activation layer s connecting atoms i and j
W vertex
s,ℓ
Weight mixing matrix for component ℓof atom SO (3)-vector activations at layer s
W edge
s,ℓ
Weight mixing matrix for component ℓof scalar edge activations at layer s
Table 1: Table of symbols used in the main text and supplement.
1
