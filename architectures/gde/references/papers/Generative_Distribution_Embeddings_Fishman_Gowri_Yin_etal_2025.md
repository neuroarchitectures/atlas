# Generative Distribution Embeddings Fishman Gowri Yin etal 2025

> Source: `Generative_Distribution_Embeddings_Fishman_Gowri_Yin_etal_2025.pdf`

---

                                                            Generative Distribution Embeddings


                                             Nic Fishman1,* , Gokul Gowri2,3,* , Peng Yin2,3 , Jonathan Gootenberg4,5,6, , and Omar Abudayyeh4,5,6,
                                                                           1
                                                                             Department of Statistics, Harvard University
                                                                       2
                                                                          Wyss Institute for Biologically Inspired Engineering
                                                                        3
                                                                          Department of Systems Biology, Harvard University
                                           4
                                             Center for Virology and Vaccine Research, Beth Israel Deaconess Medical Center, Harvard Medical School




arXiv:2505.18150v1 [cs.LG] 23 May 2025
                                         5
                                           Dept. of Medicine, Div. of Engineering in Medicine, Brigham and Women’s Hospital, Harvard Medical School
                                                                     6
                                                                       Gene and Cell Therapy Institute, Mass General Brigham
                                                              *
                                                                Equal contribution: njwfish@gmail.com, ggowri@g.harvard.edu.
                                         
                                           Senior authors jointly supervised this work: jgootenb@bidmc.harvard.edu, omar@abudayyeh.science.




                                                                                           Abstract

                                                    Many real-world problems require reasoning across multiple scales, demanding
                                                    models which operate not on single data points, but on entire distributions. We
                                                    introduce generative distribution embeddings (GDE), a framework that lifts au-
                                                    toencoders to the space of distributions. In GDEs, an encoder acts on sets of
                                                    samples, and the decoder is replaced by a generator which aims to match the input
                                                    distribution. This framework enables learning representations of distributions by
                                                    coupling conditional generative models with encoder networks which satisfy a
                                                    criterion we call distributional invariance. We show that GDEs learn predictive suf-
                                                    ficient statistics embedded in the Wasserstein space, such that latent GDE distances
                                                    approximately recover the W2 distance, and latent interpolation approximately
                                                    recovers optimal transport trajectories for Gaussian and Gaussian mixture distribu-
                                                    tions. We systematically benchmark GDEs against existing approaches on synthetic
                                                    datasets, demonstrating consistently stronger performance. We then apply GDEs
                                                    to six key problems in computational biology: learning representations of cell
                                                    populations from lineage-tracing data (150K cells), predicting perturbation effects
                                                    on single-cell transcriptomes (1M cells), predicting perturbation effects on cellular
                                                    phenotypes (20M single-cell images), modeling tissue-specific DNA methylation
                                                    patterns (253M sequences), designing synthetic yeast promoters (34M sequences),
                                                    and spatiotemporal modeling of viral protein sequences (1M sequences).


                                         1      Introduction

                                         Critical advancements in science and engineering in-
                                         creasingly depend on our ability to reason across mul-
                                         tiple scales: modelling not just individual data points,
                                         but entire populations those datapoints are drawn from.
                                         In applications ranging from single-cell genomics to
                                         DNA sequence design, the relevant unit of analysis is
                                         not an individual sample (e.g., a single cell), but the Figure 1: GDEs leverage distribution-
                                         distribution from which it is drawn (e.g. the cell state invariant encoders (E) and conditional gener-
                                         or the patient they were sampled from). These settings ative models (G) to lift autoencoders to sta-
                                         are fundamentally hierarchical: we observe sets of tistical manifolds where points correspond
                                         samples from latent distributions, which themselves to distributions (M).
                                         are drawn from a meta-distribution. Without directly modeling these distributions population-level


                                         Preprint. Under review.
signals can be lost in the noise. The fundamental challenge is how we can learn representations at the
level of distributions, not just individual data points.
We introduce generative distribution embeddings (GDEs) (Fig. 1), a framework that lifts autoencoders
to the distribution space. In GDE, the encoder maps a finite set of samples – an empirical distribution
– to a latent space, while the decoder is replaced by a conditional generative model that reconstructs
the distribution by sampling from this latent representation. This naturally supports hierarchical data
modeling: treating each observed set as a sample from a latent distribution Pi ∼ Q, where Q is a
meta-distribution over distributions. A central insight is that strong distributional representations
can be learned by coupling conditional generative models with encoders that satisfy a minimal
distributional invariance property, which we motivate through a statistical decision-theoretic lens.
Our framework synthesizes ideas from conditional generation, empirical process theory, and informa-
tion geometry. We show empirically that GDEs behave as approximate predictive sufficient statistics,
capturing distribution-level structure and abstracting away sampling noise. Moreover, the learned
latent spaces exhibit geometric regularity: latent distances correlate with Wasserstein distances (W2 )
between distributions, and linear interpolation in the latent space mimics optimal transport geodesics
restricted distributions in the support of Q, particularly for Gaussian and Gaussian mixture families.
We benchmark GDEs on synthetic datasets with known parametric structure, demonstrating improved
generative fidelity and structure preservation relative to baselines. We then scale our approach to
multiple domains in computational biology, showcasing GDEs’ versatility in modeling distributions
defined across diverse organizing principles such as distinct populations, varying experimental
conditions, spatial arrangements, and temporal dynamics. We demonstrate six applications: learning
representations of cell populations from from lineage tracing data (150K cells), predicting single-cell
transcriptomic responses to perturbations (1M cells), predicting cellular phenotypic responses to
perturbations (20M cell images), modeling tissue-specific DNA methylation (253M sequences),
designing synthetic promoters (34M sequences), and modeling spatiotemporal distributions of viral
protein sequences (1M sequences). Across these domains, GDEs offer a flexible and scalable
framework for distribution-level inference. Code for all experiments is available here.

2    Setting and methods
We observe n sets of samples Si,m = {xij }m j=1 ,
where each set is drawn i.i.d. from an unknown
distribution Pi ∈ P(X ). These distributions are
themselves drawn i.i.d. from a meta-distribution
Q:
              Pi ∼ Q, xij ∼ Pi .
Our goal is to learn an encoder–generator pair
(E, G) such that encoding a finite sample Si,m Figure 2: Right: Histogram of loss for fixed Pi
and decoding the embedding yields a distribu- different set sizes. Left: First two PCs of latent
tion close to Pi . Formally, we seek              representation of empirical distributions of MNIST
                                              (1) data from Fig. 3 for fixed Pi for different set sizes.
                      d
         G(E(Si,m )) −→ Pi as m → ∞.
This limit reflects a key property of our framework: the encoder must distill structural information
about Pi from finite, noisy observations. This motivates two constraints on E: (i) Permutation
invariance: E(S) = E(π(S)) for any permutation π, (ii) Proportional invariance: E(S) = E(S k ) for
any k. Together, these
                    Pmimply distributional invariance: the encoder depends only on the empirical
                  1
measure Pm = m         j=1 δxj , that is, E(Sm ) = ϕ(Pm ) for some measurable map ϕ.


    Proposition 1. (Informal statement of Corollary 1) Any class of encoder-generator archi-
    tectures that consistently recovers the data-generating distribution from i.i.d. samples must
    encode exactly the information in the empirical distribution—no more, no less.

In addition to the encoder capable of embedding Sm , we need a generator capable of learning to
regenerate the distribution. We can repurpose any conditional generative model (e.g., VAE, DDPM,
GAN) to minimize a divergence d between the true distribution and the decoded sample:
                              min EP ∼Q ESm ∼P ⊗m [d(P, G(E(Sm )))] .
                              E,G


                                                   2
This allows our framework to unify conditional generation and representation learning for distributions.
In Appendix D.2 we give a theoretical justification for this objective:

    Proposition 2. (Informal statement of Theorem 2) Fixing P , under mild regularity condi-
    tions: (i) a distributionally invariant encoder will have asymptotically normal distribution
    embeddings; (ii) for a suitable divergence, the plug-in loss, ℓbm := d(P, G(E(Pm ))), is
    asymptotically unbiased and normally distributed around the population loss; (iii) a global
    minimizer will recover the true data distribution as m → ∞: G(E(Pm )) ⇒ P. See Fig. 2.


3   From labels to distributions
In many real-world cases, we observe a dataset D = {(xk , yk )}Nk=1 rather than sets drawn from Q.
To apply our framework, we group data points into sets {xij }m
                                                             j=1 whose empirical distributions Pi,m
approximate draws from Q. The grouping should reflect prior knowledge about the structure of the
label space Y and enables us to shape the GDE latent space for downstream applications (see Sec. 5).
When Y is discrete, we can form sets by grouping datapoints with the same label (e.g., MNIST digits,
cells by clone identity in Sec. 6.2, cell transcriptomes by perturbation in Sec. 6.3, or epigenetic
samples by tissue in Sec. 6.5). If there is some semantic similarity between discrete labels we can
define sets proportional to those similarity metrics, paralleling contrastive learning, where labels
define semantic neighborhoods.
When Y is continuous or structured (e.g. spatial coordinates for the xij as in Sec. 6.4 or temporal
in the viral protein sequences in Sec. 6.7), we can use a similarity kernel to sample points near a
target yi∗ : wik = exp(−d(yk , yi∗ )2 /(2σ 2 )), defining a probabilistic neighborhood in the label space,
enforcing the consideration of the local structure.
When labels are noisy measurements (e.g. expression associated with DNA sequences as in Sec. 6.6),
we can invert the noise model yk = yk∗ + ϵk by sampling a latent target yi∗ and computing likelihood
weights wik = p(yk | yi∗ ), yielding samples that reflect the uncertainty of the data.
All these constructions can be unified as instances of the following framework: let Q(Y) be a prior
                                                   (Y )
over label distributions. For each set i, we draw Pi ∼ Q(Y) compute weights
                                       (Y )                             N
                                    dPi                              1 X
                            wik =             (yk ),   P̂emp (y) =       δyk (y),
                                    dP̂emp                           N
                                                                       k=1
and sample xij from D accordingly. This framework subsumes the above examples and giving us a
general set of tools for shaping the GDE latent space, as we will illustrate in Sec. 6.

4   Related work
Several lines of literature have tried to learn distribution embeddings or summary statistics. Kernel
methods, such as kernel mean embedding (KME) and set kernels, provide nonparametric approaches
to represent probability measures as points in a reproducing kernel Hilbert space, enabling tasks
like distributional regression and classification [1, 2, 3, 4, 5]. GDEs naturally nest these methods as
particular choices of distributionally invariant encoders. GDEs also generalize the approach in [6],
where they develop a particular encoder and VAE-based generator.
Distribution embeddings have also been studied from a geometric perspective. Building on theoretical
foundations from Amari [7], several works model distributions as points on a manifold imbued
with the Fisher-Rao metric [8, 9, 10, 11]. These methods are either not generative or restricted to
categorical distributions. Building on the work of Otto [12], others have considered learning flows
over Wasserstein spaces [13, 14] (see Appendix C.2 for background on Wasserstein spaces), primarily
focused on leveraging distribution encodings for transport problems as opposed to GDEs which aim
to auto-encode distributions. GDEs are complementary to these works, and can be plugged in to
many of these frameworks. One recent method closely related to GDEs, Wasserstein Wormhole
[15], aims to represent distributions as points in a space where Euclidean distances match Sinkhorn
divergences in the sample space. Wasserstein Wormhole is a particular instantiation of a GDE, using
an attention-based encoder and generator that only samples a fixed number of points.


                                                        3
A related body of work aims to learn informative summary statistics [16, 17, 18, 19, 20]. These
methods typically consider a supervised setting with a particular inferential target. For example, in
the context of likelihood-free inference, one aims to learn summary statistics which are maximally
informative about the parameters of a generative model [18, 19].
GDEs are distinct from these approaches along several dimensions: first, we generalize these methods
under a common framework with a central objective of re-sampling the encoded distribution (1);
second, we develop theory to guide the design and analysis of GDEs toward this objective; third, we
show that distribution embedding is deeply related to generative modeling, enabling domain-specific
generative models to be bootstrapped into high-quality GDEs to tackle multiscale problems.
On the architectural side, the encoder in the GDE framework requires a distributionally invariant
model. While distributional invariance is a concept introduced in this work, it requires permutation
invariance, which has been well-studied [21, 22, 23]. Some permutation invariant approaches, such
as deep sets [21], are not distributionally invariant due to proportional sensitivity, while others, such
as mean-pooled attention layers, are also distributionally invariant (as shown in Appendix D.2).
A key contribution of our work is the observation that any conditional generative model can be
repurposed to learn distributional representations. Recent work in the vision domain has found that
conditional diffusion models can induce strong image representations [24]. Our work formalizes and
generalizes this finding. We demonstrate in practice that a number of modern techniques, including
variational autoencoders [25], Sinkhorn-based generative models [26], sliced Wasserstein models
[27], denoising diffusion models [28], and autoregressive sequence models [29, 30], can be leveraged
to learn GDEs. This is by no means exhaustive: any other conditional generative modelling approach
[31, 32], including those which will emerge in the future, can be used in the GDE framework.


5     Statistical and geometric properties of GDEs

Conceptually, a key feature of GDEs is the ability to decouple observational noise from the structural
properties of the underlying distribution. We formalize this through two complementary perspectives:
first, as learning predictive sufficient statistics, and second, as embedding statistical manifolds.

5.1   Learning an approximate predictive sufficient statistic

The core objective of GDEs is to recover the true data-generating distribution P from finite, noisy
samples. This goal corresponds to learning an aggregate representation that separates the signal (the
structure of P ) from sampling noise. Formally, this objective is captured by the notion of asymptotic
predictive sufficiency, which emerges from Bayesian notions of sufficiency [33, Section 5.1.4].
Given a sample Sm ∼ P ⊗m a statistic T (Sm ) is asymptotically predictively sufficient if

                     P(xnew ∈ A | T (Sm )) ≈ P(xnew ∈ A | Sm ) as m → ∞,

for all measurable sets A ⊆ X . In our setting, the encoder E(Sm ) is such a statistic. Our condition
(1) is essentially a restatement of this condition: E(Sm ) asymptotically determines P .
Essentially, a predictive sufficient statistic distills the structural properties of the meta-distribution
while marginalizing over sampling variability in the observed data. Generative distributional em-
beddings satisfy this property in practice: they recover consistent representations of underlying
distributions, even across diverse domains and observational sample spaces.
We demonstrate this using the multinomial distribution. We learn GDEs of 3-dimensional multinomial
distributions using a mean-pooled graph neural network (GNN) encoder and a conditional diffusion
generator. The model’s latent space is able to recover the structure of the multinomial simplex (Fig. 3).
Next, we use two real-world datasets with discrete class labels and conditionally sample observations
according to label identities, which are drawn from the same family of 3-dimensional multinomial
distributions. For both a three-digit subset of MNIST and a set of three synthetic DNA sequence
patterns, GDEs (using 2D and 1D convolutional encoders and diffusion and HyenaDNA generators,
respectively) recover the same structure of the underlying multinomial simplex in the latent space.
Despite coming from three different domains and using three vastly different architectures, the latent
geometry learned between these three experiments is nearly identical.


                                                    4
Figure 3: L2 in GDE latent space compared to W2 distance. Normalized distances from the center,
p = (1/3, 1/3, 1/3). The plots to the left show GDE L2 learned from empirical distributions.
MNIST and DNA distributions are constructed by sampling conditional on class label according to a
multinomial, for MNIST subsetted to images of (0, 1, 2) and a synthetic DNA dataset with 3 patterns
respectively. Rightmost plot shows the Gaussian approximation for the W2 between multinomials.



In fact, the learned geometry is rather particular: the L2 distance in GDE latent spaces in all three cases
closely resemble W2 distances between multinomials (computed under a Gaussian approximation).
This points to a geometric interpretation of GDEs, bringing us to our second theoretical perspective.



5.2   Learning a statistical manifold

We present a geometric formulation of the GDE
problem using optimal transport. Let X denote
the sample space and let P2 (X ) be the set of
Borel probability measures on X with finite sec-
ond moment. The 2-Wasserstein distance cap-
tures the minimal cost of transporting mass from
one distribution to another, and endows P2 (X )
with a geodesic structure: interpolants between
distributions correspond to the paths obtained by
pushing forward mass along straight lines under
the optimal transport map.
Otto observed that these interpolants can be in-
terpreted as geodesics in an infinite-dimensionalFigure 4: Top row: Trajectories between pairs
Riemannian manifold. See Appendix C.2 for        of Gaussians under optimal transport (left) and
details: critically, Wasserstein geodesics admit GDE (right). Bottom row: Similar comparison for
a geometric interpretation that behaves analo-   Gaussian mixture models, we compute the “OT”
gously to Riemannian geometry.                   by finding the optimal pairing between Gaussians
For our meta-distribution Q over P2 (X ) we de- and computing the OT. Inset ternary plots show
fine the statistical manifold M := supp(Q), a mixture weights during interpolation.
submanifold of P2 (X ) with geometry inherited from the Wasserstein metric. Geodesics within M
then correspond to constrained optimal transport paths that remain inside the support of Q.
Given this setup, we can finally shed light on GDEs through the geometric framing. An embedding
ψ : M → Rd is said to be a smooth embedding if it is injective, differentiable, and has injective
differential everywhere. That is, ψ preserves the local topology and differentiable structure of the
manifold. So we can re-interpret the encoder E as an (approximate) smooth embedding of M.
Isometric embeddings preserve the actual distances of the metric on the manifold. Although every
Riemannian manifold can be isometrically embedded into Euclidean space, these embeddings are
quite complex [34]. There is no clear reason to expect that GDEs should preserve the Wasserstein
metric, yet empirically we find that they behave like approximate isometric embeddings. First, we
train GDEs on samples from 2D Gaussian distributions and observe that latent space L2 distances
correlate with true W2 distances at rank correlation ρ = 0.96. Second, we use 3-component Gaussian
mixtures and a W2 distance restricted to the mixture family [35], again finding high correlation
(ρ = 0.76). In both cases, linear latent interpolants recover the optimal transport path (Fig. 4).


                                                    5
      Proposition 3. (Informal statement of Theorem 3) A smooth map ψ : M → Rd is an
      asymptotically predictively sufficient statistic if and only if it is a smooth embedding.

These results support a simple intuition: GDEs summarize noisy data like predictive sufficient
statistics and preserve transport-based geometry like smooth embeddings of the statistical manifold.

5.3    Learning task-specific geometries by prior-weighting

Sufficient statistics and statistical manifolds are classically understood as fixed: they are determined
by the family of distributions under consideration, not learned from data. In contrast, we learn an
embedding from samples drawn from distributions, themselves drawn from a meta-distribution Q.
As a result, the representations we learn—and the geometry they induce—are shaped by Q.
                                                       In earlier examples, we used approximately uni-
                                                       form Q over a region of the statistical manifold
                                                       M, and observed that the learned geometry ap-
                                                       proximates that of Wasserstein space. We now
                                                       demonstrate that non-uniform priors Q warp this
                                                       geometry, and that this effect can be leveraged to
                                                       induce task-relevant structure in the latent space.
Figure 5: Similar to Fig. 3 we show the GDE
distances of multinomials from p = ( 13 , 13 , 13 ). When Q is non-uniform, regions with higher
We shift the prior asymmetrically by changing α1     prior density tend to expand to preserve finer
while fixing α2 = α3 = 1. This shifts the focus of   distinctions, while lower-density regions con-
the model, leading to a different learned geometry.  tract. We verify this in the synthetic multinomial
                                                     setting by training GDE models on empirical dis-
tributions sampled from skewed Dirichlet priors: α = (2−5 , 1, 1) and α = (25 , 1, 1), in contrast to
the uniform α = (1, 1, 1) (Fig. 5). As expected, distances stretch in regions where Q concentrates.
This result shows that the learned embedding geometry is prior-weighted: it reflects not only intrinsic
distributional distances, but also the sampling density induced by Q. Prior-weighting is not a side
effect, it is critical a tool: by choosing Q strategically—using task-informed sampling strategies as in
Sec. 3—we can shape the latent space to reflect meaningful distinctions in downstream objectives.


6     Applications

We first benchmark our approach and then demonstrate the generality of GDEs on tasks across
the biological sciences, spanning several data domains: DNA sequences, protein sequences, gene
expression data, and microscopy data. Throughout, we explore different combinations of encoder-
generator pairs, see App. A for a detailed discussion of architectures and training dynamics.

6.1    Benchmarking GDEs on synthetic distribution datasets

The design space of GDEs is large: any distributionally invariant encoder can be coupled to any
conditional generative model. To guide our implementation choices, we systematically benchmark
architectures using synthetic datasets. Included in the benchmarked models are two existing methods
that GDEs generalize, kernel mean embeddings and Wasserstein Wormhole [5, 15].
We benchmark 30 combinations of en- Table 1: Wasserstein reconstruction error across synthetic
coders and generators on multivari- distributional datasets. Computed as W2 for normal and
ate normal distributions in 5 dimen- GMM, and as Sinkhorn divergence for MNIST and FMNIST.
sions. For evaluation we compute
                                       Model              Normal GMM MNIST FMNIST
the Wasserstein reconstruction error
from ground truth distribution by es- KME + DDPM            0.04     2.17   80.46      111.01
timating means and covariance matri- W2 Wormhole            0.20     2.88   263.29     320.18
ces from generated samples and using   GDE                  0.02     1.82    63.79     102.21
the closed-form for W2 between Gaus-
sians. We find that mean-pooled GNNs with skip-connections coupled with DDPM generators provide


                                                   6
the highest quality generations, outperforming existing techniques. For synthetic distributions we
present results for this architecture; see App. B.1 for full experimental results.
In Table 1 we additionally benchmark this GDE architecture on three more sophisticated datasets: (1)
3-component Gaussian mixtures in 5 dimensions, (2) mixtures of MNIST [36] images according to
categorical distributions of 3 classes, and (3) an analogous dataset using Fashion-MNIST [37]. For
image datasets, where W2 distances are not tractable, we instead compute the Sinkhorn divergences
between pretrained Resnet18 [38] representations of generated and ground truth samples. In all cases,
our chosen GDE architecture outperforms existing techniques, including kernel mean embeddings
(with DDPM for generation) and Wasserstein Wormhole.

6.2   Modeling clonal populations in lineage-traced scRNA-seq data

While many methods have been developed for learning representations of single cells from single cell
RNA sequencing (scRNA-seq) data [39], methods for learning representations of cell populations
remain relatively underexplored. This task is relevant to the analysis of lineage tracing data, where
the unit of interest is a clone, or a population of cells that arise from the same progenitor.
                                                 Using lineage-traced scRNA-seq data from mouse
                                                 hematopoietic stem cells [40], we apply GDEs to learn
                                                 clone-level representations by treating the set of cells
                                                 within a clone as samples from an empirical distribu-
                                                 tion. Following prior frameworks [41], we evaluate
                                                 the ability of representations to predict future clonal
                                                 gene expression based on the mutual information (MI)
Figure 6: 2D embeddings of lineage-traced between a clone’s representation at an early timepoint
scRNA-seq data, hued by pointwise mutual during differentiation and its representation at a late
information between clonal representation at timepoint. We find that GDEs with a CVAE genera-
early timepoint and clonal fate.                 tor outperform Wasserstein Wormhole embeddings by
                                                 over 2 bits (Fig. 6). We next ask if this increase in
predictive power is due to improved representations within certain cell types (e.g., neutrophils or
monocytes). Decomposing MI estimates into their pointwise contributions [42], reveals contributions
across the entire cell state space rather than any particular cell subtype (Fig. 6).

6.3   Predicting transcriptional responses to genetic perturbations

A central goal in genomics is to predict the transcriptional effects of genetic perturbations[43, 44, 45].
We evaluate GDE for genetic perturbation prediction, using the Perturb-seq data of Replogle et al.
[46] that profiled gene expression responses to CRISPRi knockdown of thousands of genes.
We formulate a distributional perturbation prediction task as Table 2: GenePT predicting held-
follows: given the identity of a perturbation, predict the full out perturbations in mean expres-
distribution of transcriptional responses. We compare two ap- sion space and GDE latent space.
proaches. In the first case, we train a linear model to predict
                                                                                R2       MSE
the mean expression profile directly. In the second case, we
predict the GDE embedding (trained on sets of cells subject to    Mean 0.378290 1.854997
the same perturbation, via a Resnet-GNN encoder and CVAE          GDE 0.457941 1.500731
generator as in Sec. 6.2) of the perturbation-induced expression
distribution and then recover the mean via a learned linear projection from the embedding space.
In both cases, we use a ridge regression on top of GenePT embeddings [47] to enable zero-shot
generalization across perturbation conditions, demonstrating that GDE improves both R2 and MSE
in Tab. 2. See Appendix B.3 for full details.

6.4   Learning morphological cellular responses to genetic perturbations

We apply GDEs to pooled image-based CRISPR screening data from Funk et al. [48], which profiles
the phenotypic effects of perturbing 5,072 essential human genes in HeLa cells. The dataset includes
over 20 million single-cell microscopy images with four stains, capturing diverse phenotypic variation.
Each perturbation induces a distribution over cell morphologies based on perturbation groupings.
We treat these as empirical distributions and train a GDE model to reconstruct them. To explore the


                                                    7
role of inductive biases, we instantiate GDEs with two different priors: a spatial prior that models
positional image structure (see App. B.4), and a perturbation prior that captures latent variation across
perturbation conditions. These approaches capture spatial and perturbation sets, respectively.
                                          Qualitatively, the model learns to reproduce phenotypic
                                          features, including nuclear shape, cytoplasmic texture, and
                                          boundary sharpness across perturbations (Fig. 7). Quan-
                                          titatively, similar to Sec. 6.3 we hold out 30% of the most
                                          perturbative perturbations and use ridge regression with
                                          GenePT to enable zero-shot generalization across perturba-
                                          tions by predicting the GDE embedding. We then sample
Figure 7: Real and generated OPS data conditional on the predicted embedding and compute the
for the heldout RACGAP1 Knockout.         nuclear signal intensity. The predictions on these held-out
                                          perturbations achieved an R2 = 0.7055 and an MSE of
0.00068, indicating a strong zero-shot generalization of phenotypic outcomes.


6.5   Determining tissue-specific methylation signatures from bisulfite sequencing reads

Analyzing sequencing data typically extensive preprocessing, including alignment to a reference
genome. GDEs present an alternative, where sequencing reads can be modeled directly – without
alignment or other preprocessing steps. To demonstrate this capability, we show that GDEs can detect
tissue-specific DNA methylation patterns directly from bisulfite sequencing (BS-seq) reads. BS-seq
measures methylation indirectly through substitution errors: methylated cytosines remain unchanged,
while unmethylated cytosines are substituted as thymines. Using publicly available methylation
data from diverse tissues [49], we simulate sample-specific BS-seq read distributions by imposing
corresponding base substitutions to the reference genome (see Appendix B.5 for details).
Critically, we do not provide the GDE model with any explicit information about methylation signals,
the structure of the experimental assay, or a reference genome. The model has access only to sets
of sequencing reads grouped by both patient and tissue type. For the GDE model architecture,
we choose a 1D convolutional network encoder, and the decoder is a HyenaDNA model [30]. To
support large-scale inference over tens of millions of reads per patient, we process 200,000 reads at a
time through the encoder and aggregate the resulting embeddings using a simple mean, justified by
Theorem 2. This design allows the model to scale efficiently while preserving distributional fidelity.
Our approach enables end-to-end learning of methylation signatures from tissue-specific read distribu-
tions. There are two levels of tissue classification, a coarse level with 37 categories and a fine-grained
level classification with 83 tissues. Training a linear classifier on top of the GDE latent space, we
achieve a test accuracy of 60% on the coarse task and 35% on the fine-grained classification.


6.6   Decoding yeast promoter sequence activity with GDEs

We evaluate our GDE perfor-
mance on a large-scale dataset
from a massively parallel re-
porter assay measuring transcrip-
tional activity across 34 million
randomly generated yeast pro-
moter sequences [50]. Each pro-
moter consists of a random 80 nu-
cleotide DNA sequence embed-
ded in a fixed DNA scaffold and
assayed for expression in yeast
cells. Because the sequences are Figure 8: The PCA (left) of the GDE latent space of quantile
randomly sampled, there is no embeddings with underlying 34 million promoter sequences and
shared structure across examples the recovered distribution of TFBS (right) as measured by motif
so unconditional generative mod- counts in both the real and reconstructed data.
els cannot learn anything meaningful. Instead, the signal lies entirely in how distributions over
sequences give rise to distributions over expression levels, due to the presence of transcription factor


                                                    8
binding sites (TFBS): short, position-specific DNA motifs that interact with transcription factors and
control gene expression in a combinatorial and quantitative manner [50].
We construct a distributional learning task where each training example is a set of sequences sampled
from a narrow expression quantile; we hold out the top 5 quantiles. We train a GDE with a 1D
convolutional network over the one-hot encoded sequences as the encoder and HyenaDNA [30] as
the decoder. As shown in Fig. 8, the learned GDE embeddings reflect a smooth gradient across
expression quantiles. Using the set of all known yeast TFBS [51] we can identify the motifs present
in each of the real and generated sequences. Reconstructed motif distributions closely match those
of the input, indicating that the model learns to represent biologically meaningful variation across
promoter sets. Further details are available in App. B.6.

6.7   Modeling spatiotemporal distributions of viral lineages

Powerful modeling approaches have been developed to represent individual protein sequences
[52, 53, 54, 55, 29]. Here, we show that the GDE framework can naturally lift these modeling
approaches to learn representations of distributions of sequences. In particular, we model distributions
of SARS-CoV2 spike protein sequences over time and location. Using a dataset from the Global
Initiative on Sharing All Influenza Data (GISAID) [56], we group sequences by sampling month and
site location and treat each group as an empirical distribution over protein sequences. We embed
these distributions using a GDE which couples the ESM architecture [54] to a mean-pooled GNN as
the encoder and a conditional ProGen2 architecture [29] as the generator.
                                      As shown in Fig. 9, the learned latent space organizes samples
                                      chronologically, suggesting that GDEs capture time-varying
                                      signal about sequence distributions. And indeed, this is ob-
                                      served quantitatively: ridge regression on GDE representations
                                      predicts the month of held out sequence distributions with mean
                                      absolute error (MAE) of 1.83 ± 0.01 months, an improvement
                                      over the baseline of mean-pooled ESM embeddings with MAE
                                      of 2.24 ± 0.01 months (errors reported as mean ± s.e.m. over
                                      10 random train/test splits). See App B.7 for further details.

Figure 9: GDE representations         Similarly, we also observe a spatial signal, albeit much weaker.
of protein sequence distributions.    An SVM trained to classify distributions by country achieves
Each point corresponds to a set of    0.28 ± 0.001 accuracy from GDE representations, compared
SARS-CoV2 spike sequences ob-         to 0.25 ± 0.003 from mean-pooled ESM embeddings. Both
tained from one lab in one month.     approaches slightly outperform the baseline of predicting the
                                      most common dataset label (‘USA’ with accuracy 0.21).


7     Discussion

We introduce generative distribution embeddings, a framework that couples distribution-invariant
encoders with conditional generators to learn structured representations of distributions. Finite sample
sets are mapped by smooth embeddings that asymptotically identify the underlying distribution,
enabling consistent reconstruction in the large-sample limit. We formalized these properties via
connections to predictive sufficiency and statistical manifold embeddings, and proved that a broad
class of encoder architectures is asymptotically normal and unbiased when trained via a plug-in loss.
We demonstrated GDEs across a diverse set of large-scale biological problems. These applications
highlight the generality of GDEs and their ability to operate directly on measurement data while
modeling population-level structure. Crucially, GDEs support flexible distributional constructions
(e.g. spatial neighborhoods, time windows, expression quantiles), showing that a wide range of
problems can be cast as population-level modeling tasks. Code for model training and dataset
preprocessing is available at this Github repository.

Limitations GDEs rely on sensible choices of meta-distributional priors (i.e. construction of
sets, Sec. 3), often requiring careful, domain-specific design. GDEs also pose practical engineering
challenges (propagate gradients to the encoder through the generator, scaling to large set sizes)
discussed in App. A. On the theoretical side, the current formalism assumes exchangeable samples,


                                                   9
and does not admit non-i.i.d. samples within a distribution. Regarding geometry, we provide empirical
but not mechanistic evidence that GDEs learn isometries across domains.

Extensions GDEs can serve as a tool for generalization (akin to meta flow matching [14]), can be
expanded to settings where the i.i.d. assumption within sets of samples does not hold, and extended to
semi-supervised settings. More broadly, GDEs point toward questions at the intersection of empirical
process theory, information geometry, and generative modeling; we hope this connection can be
explored more deeply in future work.

References
 [1] Alex Smola, Arthur Gretton, Le Song, and Bernhard Schölkopf. A Hilbert space em-
     bedding for distributions. In Lecture Notes in Computer Science, Lecture notes in com-
     puter science, pages 13–31. Springer Berlin Heidelberg, Berlin, Heidelberg, 2007. ISBN
     9783540752240,9783540752257. doi: 10.1007/978-3-540-75225-7\_5.
 [2] Krikamol Muandet, K Fukumizu, Francesco Dinuzzo, and B Scholkopf. Learning from dis-
     tributions via support measure machines. Neural Information Processing Systems, 25:10–18,
     29 February 2012.
 [3] Junier B Oliva, B Póczos, and J Schneider. Distribution to Distribution Regression. International
     Conference on Machine Learning, 28(3):1049–1057, 16 June 2013.
 [4] Zoltan Szabo, Arthur Gretton, Barnabas Poczos, and Bharath Sriperumbudur. Two-stage
     sampled learning theory on distributions. In Artificial Intelligence and Statistics, pages 948–957.
     PMLR, 21 February 2015.
 [5] Krikamol Muandet, Kenji Fukumizu, Bharath Sriperumbudur, and Bernhard Schölkopf. Kernel
     mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine
     Learning, 10(1-2):1–141, 2017. ISSN 1935-8237,1935-8245. doi: 10.1561/2200000060.
 [6] Harrison Edwards and Amos Storkey. Towards a Neural Statistician. In International Conference
     on Learning Representations, 6 February 2017.
 [7] S I Amari and H Nagaoka. Methods of information geometry. 191, 2000.
 [8] K Carter, R Raich, W Finn, and A Hero. FINE: Fisher Information Nonparametric Embedding.
     IEEE transactions on pattern analysis and machine intelligence, 31(11):2093–2098, 14 February
     2008. ISSN 0162-8828,1939-3539. doi: 10.1109/TPAMI.2009.67.
 [9] Yonghyeon Lee, Seungyeon Kim, Jinwon Choi, and F Park. A statistical manifold framework
     for point cloud data. International Conference on Machine Learning, pages 12378–12402,
     2022.
[10] Hannes Stärk, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, R Barzilay, and
     T Jaakkola. Dirichlet flow matching with applications to DNA sequence design. International
     Conference on Machine Learning, 235:46495–46513, 8 February 2024. doi: 10.48550/arXiv.
     2402.05841.
[11] Oscar Davis, Samuel Kessler, Mircea Petrache, I Ceylan, Michael M Bronstein, and A Bose.
     Fisher flow matching for generative modeling over discrete data. Neural Information Processing
     Systems, abs/2405.14664:139054–139084, 23 May 2024. doi: 10.48550/arXiv.2405.14664.
[12] Felix Otto. The geometry of dissipative evolution equations: the porous medium equation.
     2001.
[13] Doron Haviv, Aram-Alexandre Pooladian, Dana Pe’er, and Brandon Amos. Wasserstein flow
     matching: Generative modeling over families of distributions, 2024. URL https://arxiv.
     org/abs/2411.00698.
[14] Lazar Atanackovic, Xi Zhang, Brandon Amos, Mathieu Blanchette, Leo J Lee, Yoshua Bengio,
     Alexander Tong, and Kirill Neklyudov. Meta Flow Matching: Integrating Vector Fields on the
     Wasserstein Manifold. In The Thirteenth International Conference on Learning Representations,
     4 October 2024.


                                                  10
[15] D Haviv, Russell Z Kunes, Thomas Dougherty, Cassandra Burdziak, T Nawy, Anna Gilbert,
     and D Pe’er. Wasserstein Wormhole: Scalable optimal transport distance with transformers.
     International Conference on Machine Learning, 235:17697–17718, 15 April 2024. doi: 10.
     48550/arXiv.2404.09411.
[16] John W Fisher, III, A Ihler, and Paul A Viola. Learning informative statistics: A nonparametnic
     approach. Neural Information Processing Systems, pages 900–906, 29 November 1999. doi:
     10.5555/3009657.3009784.
[17] Paul Joyce and Paul Marjoram. Approximately sufficient statistics and bayesian computation.
     Statistical applications in genetics and molecular biology, 7(1):Article26, 30 August 2008.
     ISSN 1544-6115,2194-6302. doi: 10.2202/1544-6115.1389.
[18] Justin Alsing, Benjamin Wandelt, and Stephen Feeney. Massive optimal data compression and
     density estimation for scalable, likelihood-free inference in cosmology. Monthly notices of the
     Royal Astronomical Society, 477(3):2874–2885, 1 July 2018. ISSN 0035-8711,1365-2966. doi:
     10.1093/mnras/sty819.
[19] Yanzhi Chen, Dinghuai Zhang, Michael U Gutmann, Aaron Courville, and Zhanxing Zhu.
     Neural Approximate Sufficient Statistics for Implicit Models. In International Conference on
     Learning Representations, 2 October 2020.
[20] Maxime Peyrard and Kyunghyun Cho. Meta-Statistical Learning: Supervised Learning of
     Statistical Inference, 2025.
[21] M Zaheer, Satwik Kottur, Siamak Ravanbakhsh, B Póczos, R Salakhutdinov, and Alex Smola.
     Deep Sets. Advances in neural information processing systems, 30, 10 March 2017.
[22] E Wagstaff, F Fuchs, Martin Engelcke, Michael A Osborne, and I Posner. Universal approxima-
     tion of functions on sets. Journal of machine learning research: JMLR, 23(151):151:1–151:56,
     5 July 2021. ISSN 1532-4435,1533-7928.
[23] Lily H Zhang, Veronica Tozzo, J Higgins, and R Ranganath. Set norm and equivariant skip
     connections: Putting the deep in Deep Sets. International Conference on Machine Learning,
     162:26559–26574, 23 June 2022. doi: 10.48550/arXiv.2206.11925.
[24] Drew A Hudson, Daniel Zoran, Mateusz Malinowski, Andrew K Lampinen, Andrew Jaegle,
     James L McClelland, Loic Matthey, Felix Hill, and Alexander Lerchner. SODA: Bottleneck
     diffusion models for representation learning. In 2024 IEEE/CVF Conference on Computer
     Vision and Pattern Recognition (CVPR), pages 23115–23127. IEEE, 16 June 2024. doi: 10.
     1109/cvpr52733.2024.02181.
[25] Diederik P Kingma and M Welling. An Introduction to Variational Autoencoders. Found.
     Trends Mach. Learn., 12(4):307–392, 6 June 2019. doi: 10.1561/2200000056.
[26] Aude Genevay, Gabriel Peyre, and Marco Cuturi. Learning Generative Models with Sinkhorn
     Divergences. In International Conference on Artificial Intelligence and Statistics, pages 1608–
     1617. PMLR, 31 March 2018.
[27] Soheil Kolouri, Kimia Nadjahi, Umut Simsekli, R Badeau, and G Rohde. Generalized Sliced
     Wasserstein Distances. Neural Information Processing Systems, 32:261–272, 1 February 2019.
[28] Jonathan Ho, Ajay Jain, and P Abbeel. Denoising Diffusion Probabilistic Models. Neural
     Information Processing Systems, abs/2006.11239:6840–6851, 19 June 2020.
[29] Erik Nijkamp, Jeffrey A Ruffolo, Eli N Weinstein, Nikhil Naik, and Ali Madani. ProGen2: Ex-
     ploring the boundaries of protein language models. Cell systems, 14(11):968–978.e3, 15 Novem-
     ber 2023. ISSN 2405-4712,2405-4720. doi: 10.1016/j.cels.2023.10.002.
[30] Eric D Nguyen, Michael Poli, Marjan Faizi, A Thomas, Callum Birch-Sykes, Michael Wornow,
     Aman Patel, Clayton M Rabideau, Stefano Massaroli, Y Bengio, Stefano Ermon, S Baccus,
     and Christopher Ré. HyenaDNA: Long-range genomic sequence modeling at single nucleotide
     resolution. Neural Information Processing Systems, 36:43177–43201, 27 June 2023. doi:
     10.48550/arXiv.2306.15794.


                                                11
[31] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
     Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the
     ACM, 63(11):139–144, 22 October 2020. ISSN 0001-0782,1557-7317. doi: 10.1145/3422622.

[32] Y Lipman, Ricky T Q Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow
     Matching for generative modeling. International Conference on Learning Representations,
     abs/2210.02747, 6 October 2022.

[33] José M Bernardo and Adrian FM Smith. Bayesian theory, volume 405. John Wiley & Sons,
     2009.

[34] John Nash. The imbedding problem for riemannian manifolds. Annals of mathematics, 63(1):
     20–63, 1956.

[35] Julie Delon and Agnes Desolneux. A wasserstein-type distance in the space of gaussian mixture
     models. SIAM Journal on Imaging Sciences, 13(2):936–970, 2020.

[36] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs
     [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

[37] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for
     benchmarking machine learning algorithms. CoRR, abs/1708.07747, 2017. URL http://
     arxiv.org/abs/1708.07747.

[38] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
     recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
     pages 770–778. IEEE, June 2016. ISBN 9781467388511,9781467388528. doi: 10.1109/cvpr.
     2016.90.

[39] Romain Lopez, Jeffrey Regier, Michael B Cole, Michael I Jordan, and Nir Yosef. Deep genera-
     tive modeling for single-cell transcriptomics. Nature methods, 15(12):1053–1058, December
     2018. ISSN 1548-7091,1548-7105. doi: 10.1038/s41592-018-0229-2.

[40] Caleb Weinreb, Alejo Rodriguez-Fraticelli, Fernando D Camargo, and Allon M Klein. Lineage
     tracing on transcriptional landscapes links state to fate during differentiation. Science, 367
     (6479), 14 February 2020. ISSN 0036-8075,1095-9203. doi: 10.1126/science.aaw3381.

[41] Gokul Gowri, Xiao-Kang Lun, Allon M Klein, and Peng Yin. Approximating mutual informa-
     tion of high-dimensional variables using learned representations. In A Globerson, L Mackey,
     D Belgrave, A Fan, U Paquet, J Tomczak, and C Zhang, editors, Advances in Neural Information
     Processing Systems, volume 37, pages 132843–132875. Curran Associates, Inc., 2024.

[42] Xianghao Kong, Ollie Liu, Han Li, Dani Yogatama, and Greg Ver Steeg. Interpretable Diffu-
     sion via Information Decomposition. In The Twelfth International Conference on Learning
     Representations, 13 October 2023.

[43] Omar O Abudayyeh and Jonathan S Gootenberg. Programmable biology through artificial
     intelligence: from nucleic acids to proteins to cells. Nature Methods, 21(8):1384–1386, 2024.

[44] Charlotte Bunne, Yusuf Roohani, Yanay Rosen, Ankit Gupta, Xikun Zhang, Marcel Roed, Theo
     Alexandrov, Mohammed AlQuraishi, Patricia Brennan, Daniel B Burkhardt, et al. How to
     build the virtual cell with artificial intelligence: Priorities and opportunities. Cell, 187(25):
     7045–7063, 2024.

[45] Julia Joung, Sai Ma, Tristan Tay, Kathryn R Geiger-Schuller, Paul C Kirchgatterer, Vanessa K
     Verdine, Baolin Guo, Mario A Arias-Garcia, William E Allen, Ankita Singh, et al. A transcrip-
     tion factor atlas of directed differentiation. Cell, 186(1):209–229, 2023.

[46] Joseph M Replogle, Reuben A Saunders, Angela N Pogson, Jeffrey A Hussmann, Alexander
     Lenail, Alina Guna, Lauren Mascibroda, Eric J Wagner, Karen Adelman, Gila Lithwick-Yanai,
     et al. Mapping information-rich genotype-phenotype landscapes with genome-scale perturb-seq.
     Cell, 185(14):2559–2575, 2022.


                                                 12
[47] Yiqun Chen and James Zou. Genept: a simple but effective foundation model for genes and
     cells built from chatgpt. bioRxiv, pages 2023–10, 2024.

[48] Luke Funk, Kuan-Chung Su, Jimmy Ly, David Feldman, Avtar Singh, Brittania Moodie, Paul C
     Blainey, and Iain M Cheeseman. The phenotypic landscape of essential human genes. Cell, 185
     (24):4634–4653, 2022.

[49] Netanel Loyfer, Judith Magenheim, Ayelet Peretz, Gordon Cann, Joerg Bredno, Agnes
     Klochendler, Ilana Fox-Fisher, Sapir Shabi-Porat, Merav Hecht, Tsuria Pelet, Joshua Moss,
     Zeina Drawshy, Hamed Amini, Patriss Moradi, Sudharani Nagaraju, Dvora Bauman, David
     Shveiky, Shay Porat, Uri Dior, Gurion Rivkin, Omer Or, Nir Hirshoren, Einat Carmon, Alon
     Pikarsky, Abed Khalaileh, Gideon Zamir, Ronit Grinbaum, Machmud Abu Gazala, Ido Mizrahi,
     Noam Shussman, Amit Korach, Ori Wald, Uzi Izhar, Eldad Erez, Vladimir Yutkin, Yaacov
     Samet, Devorah Rotnemer Golinkin, Kirsty L Spalding, Henrik Druid, Peter Arner, A M James
     Shapiro, Markus Grompe, Alex Aravanis, Oliver Venn, Arash Jamshidi, Ruth Shemer, Yu-
     val Dor, Benjamin Glaser, and Tommy Kaplan. A DNA methylation atlas of normal human
     cell types. Nature, 613(7943):355–364, 4 January 2023. ISSN 0028-0836,1476-4687. doi:
     10.1038/s41586-022-05580-6.

[50] Carl G de Boer, Eeshit Dhaval Vaishnav, Ronen Sadeh, Esteban Luis Abeyta, Nir Friedman, and
     Aviv Regev. Deciphering eukaryotic gene-regulatory logic with 100 million random promoters.
     Nature biotechnology, 38(1):56–65, 2020.

[51] Carl G De Boer and Timothy R Hughes. Yetfasco: a database of evaluated yeast transcription
     factor sequence specificities. Nucleic acids research, 40(D1):D169–D179, 2012.

[52] Adam J Riesselman, John B Ingraham, and Debora S Marks. Deep generative models of genetic
     variation capture the effects of mutations. Nature methods, 15(10):816–822, October 2018.
     ISSN 1548-7091,1548-7105. doi: 10.1038/s41592-018-0138-4.

[53] Tristan Bepler and Bonnie Berger. Learning the protein language: Evolution, structure, and
     function. Cell systems, 12(6):654–669.e3, 16 June 2021. ISSN 2405-4712,2405-4720. doi:
     10.1016/j.cels.2021.05.017.

[54] Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo,
     Myle Ott, C Lawrence Zitnick, Jerry Ma, and Rob Fergus. Biological structure and function
     emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of
     the National Academy of Sciences of the United States of America, 118(15), 13 April 2021.
     ISSN 0027-8424,1091-6490. doi: 10.1073/pnas.2016239118.

[55] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin,
     Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan Dos Santos Costa, Maryam Fazel-Zarandi,
     Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomic-
     level protein structure with a language model. Science, 379(6637):1123–1130, 17 March 2023.
     ISSN 0036-8075,1095-9203. doi: 10.1126/science.ade2574.

[56] Yuelong Shu and John McCauley. GISAID: Global initiative on sharing all influenza data -
     from vision to reality. Euro surveillance : bulletin Europeen sur les maladies transmissibles
     [Euro surveillance : European communicable disease bulletin], 22(13), 30 March 2017. ISSN
     1025-496X,1560-7917. doi: 10.2807/1560-7917.ES.2017.22.13.30494.

[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
     Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information
     processing systems, 30, 2017.

[58] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
     recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition,
     pages 770–778, 2016.

[59] Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding.
     arXiv preprint arXiv:1810.04805, 2018.


                                                13
[60] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin,
     Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level
     protein structure with a language model. Science, 379(6637):1123–1130, 2023.
[61] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony
     Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. Huggingface’s transform-
     ers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019.
[62] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
     Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
[63] Alexander Kraskov, Harald Stögbauer, and Peter Grassberger. Estimating mutual information.
     Physical review. E, Statistical, nonlinear, and soft matter physics, 69(6 Pt 2):066138, June 2004.
     ISSN 1539-3755. doi: 10.1103/PhysRevE.69.066138.
[64] Netanel Loyfer, Jonathan Rosenski, and Tommy Kaplan. wgbstools: A computational suite for
     DNA methylation sequencing data representation, visualization, and analysis. bioRxiv, page
     2024.05.08.593132, 10 May 2024. doi: 10.1101/2024.05.08.593132.
[65] Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the
     monge-kantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000.
[66] Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge
     University Press, 2019.
[67] David Blackwell. An analog of the minimax theorem for vector payoffs. 1956.
[68] Aad W Van Der Vaart and Jon A Wellner. Weak convergence. Springer, 1996.




                                                 14
A     Architectures and Training Dynamics
In this section we outline some general details about the architectures and training dynamics for GDEs.
In the following section we will give more detailed explanations about each specific experiment, in
addition to full details available in the codebase. All of these findings are somewhat provisional, and
there is significant scope for future work to further explore these design choices, but we hope this is a
useful complement to our codebase for researchers trying to train their own GDEs.

A.1     Encoder Architectures

Our framework utilizes permutation-invariant encoders to map input sets Sm = {x1 , ..., xm }, where
each xi ∈ Rd , to a fixed-dimensional latent representation z ∈ Rl . We primarily employ several
types of set encoders, including variants based on self-attention, Graph Neural Network (GNN)-style
pooling, and residual connections. All encoders typically conclude by applying a final pooling
operation (e.g., mean pooling) across the element representations, followed by a linear projection and
a non-linearity (e.g., SELU) to produce the final latent vector z.

A.1.1    Simple Self-Attention Encoder
This encoder provides a baseline transformer-based approach. It first applies a linear layer followed
by a SELU activation to project input elements xi into a hidden dimension H. It then processes these
representations through a series of multi-head Self-Attention blocks [57]. This architecture directly
models pairwise interactions within the set.

A.1.2    Simple GNN Encoder
The simple GNN-style encoder offers an alternative based on iterative pooling and non-linear
transformations, distinct from the standard DeepSets [21] sum-decomposition. It starts with an
MLP projection into the hidden dimension H. Subsequently, it applies a sequence of layers, each
performing a pooling operation across the set followed by an MLP. This structure iteratively refines
element representations based on aggregated set information.
Pooling Operations: Our theoretical framework (see Appendix D.2) justifies the use of pooling
operations that correspond to M/Z-estimators. We focus on mean pooling but additionally implement
median pooling as an illustrative example. Notably, max pooling is generally not suitable in this
context as its non-differentiability breaks the convergence guarantees we are interested in for Eq.
(1), see the remarks in App D.2 for details. Future work might thoroughly explore which pooling
operations lead to the greatest flexibility and stability for distribution embedding.

A.1.3    ResNet-GNN Encoder
To improve gradient flow and enable deeper architectures, we enhance the GNN-style encoder with
residual connections. This encoder first projects each input element xi into H using an MLP. It
then processes the set through a series of blocks where each block k computes an intermediate
                 (k)
representation hi for each element i. The core operation within a block uses mean pooling (or
median pooling). Inspired by ResNet [58, 23], we incorporate skip connections. The input to block k
includes the output from the previous block h(k−1) , a linear projection of the original input x, and
the output of the initial MLP projection. Formally:

                  h(k) = LayerNorm(PooledFC(h(k−1) ) + h(k−1) + Lineark (x))
where h(0) is the output of the initial input projection combined with a projection of x, followed by
Layer Normalization. This structure ensures the original input signal is preserved.

A.1.4    ResNet-Transformer Encoder
This variant follows the same residual structure as the ResNet-MLP encoder but replaces the layers
with standard multi-head Self-Attention blocks [57]. This potentially allows the model to learn more
complex interactions while benefiting from the improved training dynamics of residual connections.
The skip connection mechanism remains identical to the ResNet-MLP version.


                                                  15
A.1.5     Encoder Comparison
Transformer-based encoders (Simple Self-Attention and ResNet-Transformer) often leverage pre-
trained weights effectively and can converge in fewer epochs compared to GNN-style approaches.
However, this typically comes at a higher computational cost per epoch and during inference due to
the quadratic complexity of self-attention with respect to set size m. With sufficient training, we find
that the GNN-based architectures, particularly the ResNet-GNN, achieve strong performance, often
rivaling the transformer variants while being more computationally efficient for large sets.

Alternative Generative Strategies and Sampling The Wasserstein Wormhole [15] uses a self-attention
decoder with fixed positional embeddings that can map the latent z back to samples. One potential
method replaces fixed positional embeddings with samples drawn from a simple distribution (e.g.,
Gaussian) transforming this into a true generator. But this incurs substantial computational costs (e.g.,
quadratic cost in the number of generated samples for attention-based sampling decoders), and it is
not clear this would lead to significant improvements in performance.
It also becomes less obvious how to adapt existing generator architectures using this approach. One
option is to use self-attention to construct sample-specific condtional signals from the latent z and the
noise vector, and then condition the generator on this signal. This is significantly more complex, and
is not clear that this would lead to significant improvements in performance.

A.2     Adapting Pre-trained Models

Our framework is designed to flexibly incorporate pre-trained models, leveraging their learned
representations and generative capabilities. We adapt pre-trained models for both the encoder and the
generator components.

A.2.1     Encoder Adaptation
For tasks involving complex input modalities like natural language or protein sequences, we can
utilize pre-trained transformer-based encoders such as BERT [59] or ESM [60] as powerful feature
extractors. These pre-trained models can serve as the initial feature extraction layer, whose outputs
{h1 , ..., hN } are then fed into the subsequent aggregation layers of our set encoders (e.g., ResNet-
GNN or ResNet-Transformer, see subsection A.1).
The adaptation process typically involves:
        1. Loading Pre-trained Weights: We load the desired pre-trained encoder model using
           standard libraries like Hugging Faces transformers [61].
        2. Feature Extraction: For each element xi in the input set X = {x1 , ..., xN }, we pass it
           through the pre-trained transformer to obtain a contextualized representation hi . Often, the
           output embedding corresponding to a special token (like [CLS] in BERT) or the mean/max-
           pooled output of the final hidden states is used.
        3. Set Aggregation: These element-wise feature vectors {h1 , ..., hN } are then fed into the
           subsequent layers of our chosen set encoder (e.g., ResNet-MLP or ResNet-Transformer
           layers) which perform the permutation-invariant aggregation to produce the final latent
           representation z.
        4. Fine-tuning (Optional): Depending on the task and dataset size, the pre-trained encoder’s
           weights might be kept frozen initially or fine-tuned jointly with the rest of the model during
           end-to-end training.

A.2.2     Generator Adaptation and Conditioning
A core strength of our approach is the ability to use large pre-trained causal language models (LMs),
such as GPT-2 [62], ProGen2 [29], or specialized models like HyenaDNA [30], as the conditional
generator pθ (x|z).
The adaptation involves:
        1. Loading Pre-trained Weights: We load the chosen pre-trained causal LM and its associated
           tokenizer using ‘transformers‘ [61].


                                                   16
        2. Prefix Conditioning: The primary challenge is to effectively condition the generatorś
           output on the latent set representation z produced by the encoder. In practice, we find
           prefix tuning to be an effective and widely applicable method. The latent vector z ∈ RL
           is projected, typically via a small MLP Wp , into one or more vectors p = Wp (z) that
           have the same hidden dimension as the LM. These projected vectors p are then treated
           as continuous "prefix" embeddings prepended to the actual input sequence embeddings
           E(x<T ) before they are processed by the transformer layers. The model learns to interpret
           this prefix as the conditioning signal specifying the target distribution. Mathematically, the
           input embedding sequence to the transformer becomes [p; E(x<T )]. The attention mask is
           adjusted accordingly to allow all sequence tokens x<T to attend to the prefix p.
        3. Fine-tuning: The pre-trained generator weights can be either frozen or fine-tuned. Fine-
           tuning the entire model allows the LM to adapt its generation process based on the condi-
           tioning prefix p. Freezing the LM backbone and only training the conditioning projection
           Wp (and potentially adapter layers) can be more parameter-efficient.

A.3     Training Details and Considerations

A.3.1     Learning Rate Schedule
For simpler models we use a fixed learning rate, but for more complex models we typically employ a
cosine annealing learning rate schedule during training. This involves starting with an initial learning
rate and gradually decreasing it towards zero following a cosine curve over the course of training
epochs. This schedule is often effective in achieving stable convergence and good final performance.
In general we have found that whatever the current state of the art for training the (unconditional)
generator is, that will generally give good results when learning the encoder-generator jointly.

A.3.2     Performance and Convergence
Our experiments generally indicate that this training setup, combined with the described architectures
and adaptation strategies, leads to strong performance across various tasks and datasets presented in
the main paper. As noted in subsection A.1.5, the choice of encoder can impact convergence speed
and computational cost.

A.3.3     Set Size and Batching Trade-offs
We observe that achieving optimal performance sometimes necessitates using large input set sizes (N ).
However, processing large sets can significantly increase the computational and memory requirements
per batch, particularly for the attention mechanisms in transformer-based encoders or generators.
This often forces a reduction in the overall batch size to fit within hardware constraints. Smaller
batch sizes can, in turn, lead to increased variance in the loss gradients, potentially slowing down or
destabilizing training. Careful tuning of the set size N , batch size, and learning rate parameters is
often required to balance performance and training efficiency for a given task and hardware setup.

A.3.4     Gradient Propogation Challenges
A potential challenge arises, particularly with deeper encoder and generator architectures. The
encoder only receives a learning signal indirectly through the generator via the shared latent variable
z. If the generator itself struggles to utilize the latent information effectively, or if the dimensionality
L of z creates an information bottleneck, the gradients flowing back to the encoder can become
weak or noisy. This can make training deep encoders difficult. Addressing this might require more
sophisticated generator architectures capable of integrating the latent information more effectively or
alternative training schemes with auxiliary losses directly on the encoder. We found these issues in
the simple encoder architectures, but they seemed to be alleviated in the ResNet-based architectures.




                                                    17
B     Experiments
B.1   Additional semi-synthetic experimental results




Figure 10: Left: Distance correlation showing high alignment between latent GDE distances and
analytical W2 distances (Spearman ρ = 0.96). Left: Distance correlation showing high alignment
between latent GDE distances and the OT-GMM distance [35], which is a W2 metric restricted to the
subspace of GMMs (Spearman ρ = 0.76).




Figure 11: Expanding on Fig. 5 we show that the Pearson correlation between the W2 (computed via
normal approximation) and the latent GDE distances decreases as α1 deviates from 1, while keeping
fixed α2 = α3 = 1.


Table 3: W2 reconstruction error of 30 possible GDE implementations (including two existing
methods generalized by GDE, Wasserstein Wormhole and kernel mean embeddings) on 5-dimensional
multivariate Gaussians. Covariance matrices sampled from Wishart distribution with scale of 1, and
means sampled uniformly from [0, 5]. Further results included in Table 4.

 Gen. ↓ \\ Enc. →     Mean     Kernel mean     GNN     Med.-GNN       ResNet-GNN       SelfAttn.
 Sinkhorn             0.05         0.14         0.09       0.10            0.05          0.06
 Sliced W2            0.03         0.04         0.07       0.07            0.03          0.04
 CVAE                 0.16         0.16         0.19       0.20            0.15          0.17
 DDPM                 0.03         0.04         0.06       0.05            0.02          0.07
 Wormhole             0.14         0.15         0.72       0.49            0.14          0.20



                                               18
Table 4: W2 reconstruction error (mean ± s.e.m. over 5 trials) for 30 possible GDE implementations
(including two existing methods generalized by GDE, Wasserstein Wormhole and kernel mean
embeddings) on 5-dimensional multivariate Gaussians. Covariance matrices sampled from Wishart
distribution with scale of 0.1, and means sampled uniformly from [0, 5].

         Gen. ↓ \\ Enc. →     Kernel mean          GNN          ResNet-GNN         Self-Attn.
         CVAE                 0.15 ± 0.011      0.12 ± 0.006     0.12 ± 0.009     0.11 ± 0.007
         DDPM                 0.15 ± 0.008      0.13 ± 0.020     0.09 ± 0.003     0.10 ± 0.005
         Direct SW            0.15 ± 0.008      0.13 ± 0.007     0.13 ± 0.009     0.15 ± 0.001
         Direct Sinkhorn      0.29 ± 0.008      0.22 ± 0.010     0.17 ± 0.005     0.19 ± 0.010
         Wormhole             0.23 ± 0.021      0.72 ± 0.090     0.24 ± 0.011     0.34 ± 0.021


B.2     Lineage-traced scRNA-seq experiments

B.2.1    Data preprocessing details
We use lineage tracing data from Weinreb et al. [40]. The single-cell RNA sequencing (scRNA-seq)
count matrices were preprocessed following standard procedures. Specifically, counts for each cell
were normalized by rescaling to 104 counts per cell, followed by log transformation. Finally, the
top 104 highly variable genes (HVGs) were selected. Cell-type annotations and two-dimensional
SPRING embeddings were obtained directly from the annotations provided in Weinreb et al.

B.2.2    Mutual information estimation
We compute mutual information as a sample mean of pointwise mutual information estimates. To
estimate pointwise mutual information in the representation space, we use the nonparametric nearest-
neighbor estimator introduced by Kraskov et al. [63] with k = 3. This estimator has been shown to
be effective in this setting: model latent spaces with tens of dimesions [41].

B.2.3    GDE modelling architecture
We use a Resnet-GNN architecture as the encoder and a CVAE as the generator. We use 64 latent
dimensions, with 2 hidden layers of size 128.

B.3     Perturbation Prediction

B.3.1    Data preprocessing details
We use the pre-processed h5ad file from [46] including 104 genes. We compute the 10% most
perturbative perturbations by examining the differentially expressed genes and then randomly select
20 of those perturbations to hold out. We hold these out across all cell types.

B.3.2    GDE modelling architecture
We use a Resnet-GNN architecture as the encoder and a CVAE as the generator, similar to the
architecture in the lineage-tracing experiment, except we use a larger hidden state (1024) and a
larger latent space (256). We include a perturbation prediction loss during training which trains a
linear model with pairwise interactions between the control cell distribution embedding and the gene
embedding to predict the difference in mean expression through a linear head. This structures the
latent space for our downstream perturbation prediction task.

B.3.3    Perturbation Prediction
We fit a ridge regression to predict (1) the difference in mean expression and (2) the difference between
the perturbed embedding and the control for each perturbation using GenePT gene embeddings [47]
with cross-validation to perform grid search over λ. We then compute the predictions on the held-out
perturbations and use a linear head to predict the mean expression from the latent difference. Finally
we compute the R2 score and the MSE.


                                                   19
B.4     Optical pooled screening dataset

B.4.1    Data preprocessing details

We use phenotyping images with assigned perturbation barcodes from Funk et al. [48]. We analyze
only two of the measured channels: DAPI and GFP. Each image is a 64x64 bounding box surrounding
a single cell (center-padded or center-cropped from the original bounding box as necessary). Image
intensities are normalized to a minimum of −1 and a maximum of 1. Using the set of perturbative
perturbations computed in [48] we randomly select 30% to holdout during training for evaluation.

B.4.2    GDE modelling architecture

For the encoder architecture, we extend our GNN approach to 2D convolutional layers, standard for
image processing. For the generator we use a U-net architecture standard in diffusion for images, but
upscaled in expressivity relative to our MNIST and Fashion-MNIST examples.

B.4.3    Perturbation Prediction

We find that empirically, our diffusion approach struggles to model the padded border of the cells.
So, at inference time we condition on the border to generate our predictions. Using GenePT, we
train a ridge regression with grid search (similar to App. B.3) to predict the perturbation distribution
embeddings. We also construct a nearest neighbor model using the GenePT embeddings to sample
the padding. We then condition on the padding and the predicted latent to sample a set of 1,000 cells
from each heldout perturbation. We then compute the DAPI intensity and compare with the ground
truth, computing the R2 and the MSE.

B.5     Methylation atlas of human tissues

B.5.1    Simulating raw bisulfite-sequencing reads from methylation patterns

While sample-specific methylation patterns are published in [49], the raw sequencing reads are not
public due to patient privacy considerations. Here, we instead use the published methylation patterns
(in the form of .pat files) to simulate bisulfite sequencing reads. For each methylation site entry of
the .pat file, we use wgbstools[64] to find the 100 preceding bases of the HG38 genome reference,
and append to the CpG sequence. We omit all CpG sites with unknown methylation status. We
subsample 107 sequencing reads per sample.

B.5.2    GDE modelling architecture

We use a 1D convolutional neural network as our encoder, with mean pooling at each layer (analogous
to the fully connected GNN with an MLP, but using convolutional layers). For the generator, we
use HyenaDNA [30]. We additionally include a linear classification head on top of the distribution
embedding, co-trained with a cross-entropy loss.

B.6     GPRA

B.6.1    Data processing details

We collect all sequences in the Gal and Gly conditions from [50] and process them into 100 quantiles
by measured expression, totaling 34 million sequences. We one-hot encode these sequences for
ACTGN, and tokenize them using the HyenaDNA tokenizer. We break these sequences into 100
quantiles and hold out the top 5 quantiles during training. During training, we construct sets by
selecting a “center” quantile and then randomly sampling from that quantile and the two adjacent
quantiles.

B.6.2    GDE modelling architecture

We use the same architecture as in the methylation experiment (App. B.5).


                                                  20
B.6.3    Details for Fig. 8
We encode a random subsample of 130K sequences from each quantile in the Gal condition to
construct the set embeddings (the larger dots). We then compute the PCA of these embeddings. We
embed all the DNA sequences as sets of size one and project them to the PCA. For the histograms of
the TFBS motifs we leverage the PWMs from [51]. We wrote a simple unidirectional motif scanning
procedure in Torch to facilitate efficient scanning, and used a threshold of 5 to determine hits. We
then sum over the motifs to derive the motif count per sequence, and then compute the histogram by
plotting the distribution of these counts by quantile.

B.7     Spatiotemporal distribution of viral lineages

B.7.1    Data preprocessing details
We obtain all SARS-CoV2 spike sequences deposited up to April 2025 in GISAID [56]. We group
sequences by submission month and lab of collection. We discard sequences with improperly
formatted date fields. During tokenization, we truncate sequences to 1000 amino acids.

B.7.2    GDE modelling architecture
The encoder couples the ESM-50M [54] architecture coupled to a mean-pooled GNN, while the
generator uses the Progen2-150M architecture [29] with prefix conditioning. We initialize (but do not
freeze) the protein language models with their pretrained weights. We use a 128 dimensional latent
space.




                                                 21
C     Background
C.1     Frequentist, Bayesian, and Predictive Sufficiency

Sufficiency is a classical notion in statistics that formalizes when a statistic retains all information
about a parameter or distribution. In this appendix, we distinguish three forms of sufficiency relevant
to modern generative modeling and provide canonical examples.

C.1.1    Frequentist Sufficiency
Let {Pθ : θ ∈ Θ} be a parametric family of probability distributions on a sample space X . A statistic
T (X1 , . . . , Xn ) is frequentist sufficient for θ if the conditional distribution of the data given T does
not depend on θ:
                       Pθ (X1 , . . . , Xn | T (X1 , . . . , Xn )) = (independent of θ).
Intuitively, the likelihood depends on the data only through T .

C.1.2    Bayesian Sufficiency
Given a prior π(θ) over the parameter space, a statistic T is Bayesian sufficient for θ if the posterior
depends on the data only through T :
                             π(θ | X1 , . . . , Xn ) = π(θ | T (X1 , . . . , Xn )).
Bayesian sufficiency holds if and only if T is a sufficient statistic in the sense that the posterior is
conditionally independent of the data given T .

C.1.3    Predictive Sufficiency
A weaker notion, often relevant in nonparametric and distributional settings, is predictive sufficiency.
A statistic T is predictive sufficient if the distribution of a new sample Xnew given T is the same as
given the full data:
           P(Xnew ∈ B | T (X1 , . . . , Xn )) = P(Xnew ∈ B | X1 , . . . , Xn ),       ∀B ∈ B(X ).
This requires only that T contains enough information to match the predictive distribution of future
data.

C.1.4    Implications and Comparisons
There is a strict hierarchy among these definitions:
              Frequentist sufficiency ⇒ Bayesian sufficiency ⇒ Predictive sufficiency.
The first implication follows from the factorization of the likelihood, and the second follows because
the posterior predictive is a marginal of the posterior. However, the reverse implications do not
hold in general, especially in infinite-dimensional or nonparametric models. In particular, predictive
sufficiency may hold in settings where no finite-dimensional parameter exists.

C.1.5    Examples
Example 1 (Gaussian Mean). Let X1 , . . . , Xn ∼ N (µ, σ 2 ) with known σ 2 . Then the sample mean
X̄n is sufficient for µ in all three senses: frequentist, Bayesian, and predictive. The likelihood,
posterior, and predictive distributions all depend on the data only through X̄n .
Example 2 (Gaussian Mixture Model). Let X1 , . . . , Xn ∼ P where P is a finite mixture of
Gaussians:
                                              K
                                              X
                                       P =       πk N (µk , Σk ).
                                               k=1
The sufficient statistics for this model (under known K) are:

        • the soft assignment (responsibility) weights for each component,


                                                      22
        • the empirical means and covariances of points assigned to each component,
        • the mixture proportions.
These are sufficient in both the frequentist and Bayesian senses. In many applications, they are
approximated via the Expectation-Maximization algorithm or variational inference.
Example 3 (Uniform(0, θ)). Let X1 , . . . , Xn ∼ Unif(0, θ). Then the sample maximum
                                         Tn = max{X1 , . . . , Xn }
is the minimal sufficient statistic for θ in both the frequentist and Bayesian senses. It also suffices for
prediction of future samples, since the predictive distribution under θ is uniform on [0, θ], and Tn
provides all information about θ.

C.1.6    Nonparametric Extensions
In the nonparametric regime where P is not indexed by a finite-dimensional parameter,
                                                                                 Pn       predictive
sufficiency remains well-defined. For instance, the empirical measure Pn = n1 i=1 δXi is always
predictive sufficient under exchangeable models. In this setting, stronger forms of sufficiency may
not exist, but predictive sufficiency still supports meaningful generative modeling.

C.2     Otto’s Geometry and Statistical Submanifolds

This appendix summarizes the formal Riemannian structure of the Wasserstein space P2 (X ) intro-
duced by Otto [12], and defines statistical manifolds as submanifolds equipped with a geometry
induced from this structure. This provides the mathematical foundation for interpreting generative dis-
tributional encoders (GDEs) as learning smooth geometric embeddings of constrained distributional
families.

C.2.1    Wasserstein Space and the Benamou–Brenier Formulation
Let X ⊆ Rd be a domain, and let P2 (X ) denote the space of Borel probability measures on X with
finite second moment. The 2-Wasserstein distance between two measures µ0 , µ1 ∈ P2 (X ) is defined
by the optimal transport problem
                                                 Z
                       W22 (µ0 , µ1 ) :=  inf          ∥x − y∥2 dγ(x, y),
                                          γ∈Π(µ0 ,µ1 )      X ×X
where Π(µ0 , µ1 ) denotes the set of couplings with marginals µ0 and µ1 . An equivalent dynamic
formulation, due to Benamou and Brenier [65], expresses the Wasserstein distance as a variational
problem over time-dependent flows:
                                                   Z 1Z
                      2
                    W2 (µ0 , µ1 ) =      inf              ∥vt (x)∥2 dµt (x) dt,
                                             (µt ,vt )       0     X
                                      ∂t µt +∇·(µt vt )=0

subject to boundary conditions µ0 , µ1 and the continuity equation, which ensures mass conservation.

C.2.2    Otto’s Riemannian Structure
Otto observed that the Benamou–Brenier problem defines a formal Riemannian structure on P2 (X ),
where the tangent space at a measure µ consists of velocity fields v such that the continuity equation
describes admissible perturbations. The inner product between two such velocity fields v1 , v2 ∈ Tµ P2
is defined as                                    Z
                              ⟨v1 , v2 ⟩Tµ P2 :=   v1 (x) · v2 (x) dµ(x).
                                                    X
This makes P2 (X ) a formal infinite-dimensional Riemannian manifold, and Wasserstein geodesics
become curves of minimal kinetic energy under this metric.
For absolutely continuous µ0 , the optimal transport map T : X → X from µ0 to µ1 induces a
geodesic (µt )t∈[0,1] by pushing µ0 along linear interpolations:
                                    µt = ((1 − t)id + tT )# µ0 .
These displacement interpolants travel at constant speed under the W2 metric and solve the geodesic
equation associated with the Otto metric.


                                                     23
C.2.3    Statistical Manifolds as Submanifolds of Wasserstein Space
Let Q be a probability distribution over P2 (X ), and define the statistical manifold as the support of
Q:
                                       M := supp(Q) ⊂ P2 (X ).
We endow M with the induced Riemannian structure from P2 (X ), by restricting the Otto metric
to velocity fields that remain tangent to M. That is, Tµ M ⊂ Tµ P2 is a subspace of velocity fields
preserving membership in M, and the inner product is
                                      Z
                    ⟨v1 , v2 ⟩Tµ M :=   v1 (x) · v2 (x) dµ(x), for v1 , v2 ∈ Tµ M.
                                      X

This leads to a constrained transport problem defining geodesics within M:
                                                         Z 1Z
                  2
                W2,M  (µ0 , µ1 ) :=         inf                 ∥vt (x)∥2 dµt (x) dt.
                                            (µt ,vt )           0   X
                                   µt ∈M, ∂t µt +∇·(µt vt )=0

This is simply the Wasserstein variational problem, but restricted to paths that lie within the submani-
fold M. It defines the geometry relevant to learning distributions drawn from Q.

C.2.4    Examples and Application to GDEs
Typical examples of such submanifolds include:

        • Gaussian families N (µ, Σ), where geodesics can be computed in closed form;
        • Mixture models with a fixed number of components, see [35];
        • general parametric families.

In this work, we treat the statistical manifold M = supp(Q) as the set of data-generating distributions
and interpret GDEs as learning a smooth embedding of this submanifold into Euclidean space. While
GDEs do not explicitly minimize Wasserstein distances, we observe empirically that their learned
latent geometries often approximate the structure of W2,M , suggesting that they act as approximate
isometric embeddings of this constrained transport geometry.




                                                   24
D     Theory
Throughout, let (X , B) be a Polish space. Let P ∈ P(X ) denote a probability  Pmlaw on X . Given
                                                                             1
m ∈ N, let Sm = (X1 , . . . , Xm ) be an i.i.d. sample from P , and let Pm = m  i=1 δXi denote the
empirical measure.
We use P1 , P2 to denote two (possibly distinct) probability laws on X , and S1 , S2 for independent
samples from P1 , P2 respectively.
For signed measures ν, µ on (X , B) define
                                                             R
                              dBL (ν, µ) :=       sup            f d(ν − µ) .
                                              f :X →[−1,1]
                                                Lip(f )≤1

We use ∥ · ∥BL for the corresponding norm ∥ν∥BL := dBL (ν, 0) and recall that dTV (ν, µ) ≤
dBL (ν, µ).
All random variables are defined on a common probability space unless otherwise specified.

D.1   Necessity of Distributional Invariance

Motivation Our goal is to design encoder architectures that flexibly model unknown data distribu-
tions while guaranteeing consistent generation of the underlying law as sample size grows. Since the
true distribution P is not known in advance, the encoder must be constructed to generalize across all
possible P , without leaking spurious information tied to the specific realization or sample size. If the
encoder depends on sample-level artifacts—such as ordering, multiplicity, or the raw sample size—it
may encode features that a generator can exploit, breaking the guarantee that
                                        d
                           G(E(Sm )) −
                                     →P        as m → ∞,           Sm ∼ P ⊗m .
This risk arises even under either permutation or proportional invariance on their own: both permit
dependencies that vanish only in expectation and are insufficient to ensure correct extrapolation with
increasing m. For example, encoders based on unnormalized sum aggregations (e.g., DeepSets) will
vary with m even when the empirical distribution is unchanged, leading to divergence at inference
time.
To formalize this constraint, we draw on Blackwell’s theory of experiments, which provides a general
framework for comparing the informativeness of statistical summaries. We adopt his game-theoretic
perspective—viewing the encoder as a player that chooses an experiment, and the generator as an
adversary that exploits the information it receives—and use this to characterize the minimal structural
conditions an encoder must satisfy to guarantee asymptotic consistency. In particular, we show
that dependence on the empirical distribution is necessary and sufficient: it is the least informative
summary that still retains all information required to identify the law, so the generator cannot learn
any spurious information that will fail to extrapolate at inference time.

Setting We consider the following general setting: Let (X , B) be a Polish space. We are interested
in measurable summaries of infinite i.i.d. sequences S ∼ P ∞ , where P ∈ P(X ) is an unknown
probability law. The goal is to characterize the minimal invariance properties required for encoders to
guarantee consistent recovery of P from finite samples.
Definition 1 (Distributional Invariance). A function E : X m → Z is distributionally invariant if for
any Sm ∈ X m , E(Sm ) depends only on the empirical measure Pm of Sm ; that is, for any permutation
                                             π
π of {1, . . . , m} and any Sm , E(Sm ) = E(Sm ), and ϕ is invariant to proportional duplications of the
sample.
Definition 2 (Asymptotic Distributional Invariance). A sequence of functions Em : X m → Z is
asymptotically distributionally invariant if for every P ∈ P(X ), there exists a sequence of measurable
functions ϕm : Pm (X ) → Z such that
                                                          
                        PSm ∼P ⊗m Em (Sm ) = ϕm (Pm ) → 1 as m → ∞,
where Pm is the empirical measure of Sm .


                                                  25
Lemma 1 (Strong Law of Large Numbers for Empirical Measures). Let P P           ∈ P(X ) and Sm =
                                                                              1   m
(X1 , . . . , Xm ) be i.i.d. samples from P . Then the empirical measure Pm = m   i=1 δXi converges
almost surely to P in the weak topology as m → ∞.
Lemma 2 (Wainwright’s Rademacher–tail bound [66, Thm. 4.10]). Let F be a class of measurable
functions f : X → R that is b-uniformly bounded, i.e. ∥f ∥∞ ≤ b for all f ∈ F. For any integer
n ≥ 1 and any δ > 0,
                                                                                    2
                                                                                       
                Pn − P F ≤ 2 Rn (F) + δ with probability at least 1 − exp − nδ     2b2 ,

where Rn (F) is the (deterministic) Rademacher complexity defined in [66, Eq. 4.13].
To formalize what can go wrong, we introduce an adversarial two-player game, adapted from the
decision-theoretic framework introduced by [67]. This game reveals the informational limits of
summary statistics for distinguishing distributions.

1. Player 1 selects a measurable summary rule T : X N → T before seeing any data.
2. Player 2 observes T and chooses two probability laws P1 , P2 ∈ P(X ).
3. Nature draws two independent infinite i.i.d. sequences S1 ∼ P1∞ , S2 ∼ P2∞ .

The induced decision problem is whether the summary T is consistent with P1 = P2 or not. The
payoff structure is:

                                       T (S1 ) = T (S2 )   T (S1 ) ̸= T (S2 )
                           P1 = P2          (1, 0)              (0, 1)
                           P1 ̸= P2         (0, 1)              (1, 0)
Player 1 aims to minimize both types of errors: introducing spurious distinctions when P1 = P2 , and
failing to distinguish when P1 ̸= P2 . In Blackwell’s terms, the goal is to find a summary that is as
informative as possible for this class of binary decision problems.
Definition 3 (Asymptotically Blackwell–optimal summary). Let (T , dT ) be a separable metric space
and let Tn : X n → T be measurable. Fix any deterministic sequence εn ↓ 0. We say that (Tn ) is
asymptotically Blackwell–optimal if, for every pair of probability laws P1 , P2 ∈ P(X ),
                                         h                          i
                 (i) P1 = P2 =⇒ PP1∞ dT Tn (S1 ), Tn (S2 ) > εn −−−−→ 0,                          (2)
                                                                          n→∞
                                           h                         i
               (ii)   P1 ̸= P2 =⇒ PP1∞ ×P2∞ dT Tn (S1 ), Tn (S2 ) ≤ εn −−−−→ 0.                    (3)
                                                                                n→∞

Main Result
Theorem
  Pn       1 (Empirical distribution characterises optimal asymptotic summaries). Let Pn (S) =
1                                                                          ∞
           i be the empirical distribution of the first n samples of S ∼ P
n   i=1 δ S                                                                  . For measurable
                  n
summaries Tn : X → T :
  (i) (Blackwell optimality) the empirical distribution Tnemp (S) = Pn (S) is asymptotically optimal
      in the game of Definition 3.
  (ii) (Asymptotic information equivalence) If (Tn ) is asymptotically optimal, there exist measurable
       maps fn : P(X ) → T and gn : T → P(X ) such that for every P ∈ P(X )
                                                                                     
           PP ⊗n Tn (S) = fn (Pn (S)) −−−−→ 1,           PP ⊗n Pn (S) = gn (Tn (S)) −−−−→ 1.
                                           n→∞                                         n→∞
      In particular, Tn and Pn are asymptotically Blackwell-equivalent in probability.
We can immediately conclude the implications for encoders here:
Corollary 1 (Necessity of asymptotic distributional sufficiency). Suppose we design an encoder E
and decoder G with the goal that, for any unknown distribution P ∈ P(X ),
                                       d
                           G(E(Sm )) −
                                     →P       as m → ∞,       Sm ∼ P ⊗m .
Then, in order for this convergence to hold for any P , the encoder architecture must satisfy two
conditions:


                                                  26
                                         Pm Em (Sm ) must (eventually) depend only on the
     (i) Asymptotic distributional invariance:
                                       1
         empirical distribution Pm = m
                                          j=1 δxj —that is, for every P , PP ⊗m Em (Sm ) =
         ϕm (Pm ) → 1.
    (ii) Distributional expressivity: the class of encoder functions must be rich enough to approxi-
         mate any measurable function of Pm .

These are constraints on the encoder architecture, not on the learned function after training. If Em
encodes any features not measurable with respect to Pm —such as sample order, indexing artifacts,
or features sensitive to repeated observations—the generator can exploit these to fit P incorrectly,
breaking consistency. Either permutation or proportional invariance alone are not sufficient: only
sufficiency with respect to the empirical distribution rules out such failure modes.

Proof

Proof. Step 1: Soundness. The class FBL is 1-uniformly bounded, and its empirical Rademacher
complexity satisfies Rn (FBL ) = O(n−1/2 ). Equip T with dBL and set εn = n−1/2 .
By the triangle inequality,
                                                                          
                  dBL Pn (S1 ), Pn (S2 ) ≤ dBL Pn (S1 ), P + dBL Pn (S2 ), P .
Applying Lemma 2 [66] to each term with tolerance εn /2 and union-bounding yields
                                                                  nε2 
                       h                            i
                      P dBL Pn (S1 ), Pn (S2 ) > εn ≤ 2 exp − 8n .

Choosing εn = n−1/4 therefore fulfils (2).
                                                         a.s.
If P1 ̸= P2 , the strong
                          law gives dBL Pn (Si ), Pi −−→ 0, hence for large n the event
dBL Pn (S1 ), Pn (S2 ) ≤ εn is impossible, establishing (3).
Step 2: Tn is a function of Pn with high probability. Let ηn ↓ 0 be an arbitrary deterministic
sequence. We will show
                                            
                   PP ⊗n Tn (S) = fn (Pn (S)) ≥ 1 − ηn for all large n,
for some measurable fn : P(X ) → T . That is, Tn differs from a measurable function of the empirical
distribution with probability o(1), which suffices for the asymptotic game.
Now fix η > 0 and define
              n                                                               o
        Iη := n ≥ 1 : ∃ P ∈ P(X ) s.t. PP ⊗n Tn (S) is not σ(Pn )-measurable > η .
If Iη were infinite for some positive η, we would construct a single distribution P† to be any
accumulation point of the sequence of counter-example measures forcing
                                                      2
                           P Tn (S1 ) ̸= Tn (S2 ) ≥ η2 for all n ∈ Iη ,
                                                

contradicting optimality condition (2).
Since asymptotic optimality holds, every η > 0 gives a finite Iη , so we can choose N (η) such that
                                           
         PP ⊗n Tn (S) is σ(Pn )-measurable ≥ 1 − η for all n ≥ N (η) and every P.
Taking η = ηn and letting fn be any measurable selector on the high-probability event (where Tn is
σ(Pn )-measurable) proves the claim that Tn is eventually a function of Pn with probability 1 − ηn .
Step 3: Recoverability of Pn from Tn . If fn collapses two distinct empirical measures m ̸= m′ ,
take P1 = m, P2 = m′ . Then dBL (m, m′ ) > 0 while Tn (S1) = Tn (S2 ) a.s., violating (3). Hence
there exists a measurable gn : T → P(X ) with Pn = gn Tn a.s.
Remark. The argument above shows that any asymptotically optimal summary Tn is both a measur-
able function of Pn and sufficient to recover Pn almost surely. In Blackwell’s terminology, this means
Tn and Pn are asymptotically equivalent experiments: they contain exactly the same information for
distinguishing distributions from i.i.d. samples. In future work it would be interesting to consider
how to design architectures with similar properties under different forms of dependence.


                                                 27
D.2      A Complete Large-m Analysis of the Plug-in Loss

Motivation We analyze the statistical properties of the plug-in loss used to train distributional
encoders and generators. Our goal is to understand the asymptotic behavior of this loss as the sample
size grows, and to establish conditions under which the learned generator recovers the true data
distribution. This analysis provides a principled foundation for the training objectives used in our
framework.

Setting First we establish some notation and definitions.
Definition 4 (Hadamard differentiability). A map T : D → Y between normed spaces is Hadamard
differentiable at x ∈ D if there exists a continuous linear operator DTx such that for every sequence
ht → h in D and t ↓ 0, T (x+thtt )−T (x) −→ DTx [h].
Definition 5 (Fréchet differentiability). Let T : D → Y be a map between normed vector spaces. T
is Fréchet differentiable at x ∈ D if there exists a bounded linear operator A : D → E such that
                                       ∥T (x + h) − T (x) − A(h)∥E
                                  lim                              = 0.
                               ∥h∥D →0            ∥h∥D
The operator A is called the Fréchet derivative of T at x.

We work in the following general setting:
Assumption 1 (Data and Empirical Measure). (X , B) is a Polish space; P ∈ P(X ) is thePtrue data
                                                                                       1 m
law. Observations Sm = (X1 , . . . , Xm ) are i.i.d. P . The empirical measure is Pm = m i=1 δXi .
Assumption 2 (Encoder regularity). For each probability law P ∈ P(X ) the encoder ϕ : P(X ) →
Rd satisfies

        (i) Distributional invariance: Em (Sm ) = ϕ(Pm ) depends on the sample only via its empirical
            measure.
       (ii) Pathwise (Hadamard) differentiability: ϕ is pathwise differentiable at P and its canonical
            gradient1 ψP : X → Rd belongs to L2 (P ).
       (iii) Asymptotic linearity (AL): the estimator obeys
                                                                    m
                              √                              1 X
                                  m ϕ(Pm ) − ϕ(P )         = √       ψP (Xi ) + op (1).
                                                               m i=1

           where EX∼P [ψP (X)] = 0, so ψP .

Under these conditions
              √                             d                                    
                m ϕ(Pm ) − ϕ(P )             ⇒ N 0, Σϕ ,
                                             =                   Σϕ := VarX∼P ψP (X) .

Assumption 3 (Generator). G : Rd → P(X ) is Fréchet differentiable on a neighbourhood of
µ := ϕ(P ) and its derivative factors through L2 (P ), i.e.
                  Dµ G = T ◦ A,         where A : Rd → L20 (P ), T : L20 (P ) → M0 (X )
are bounded linear maps and L20 (P ) denotes zero-mean square-integrable functions.
Assumption 4 (Divergence). The discrepancy d : P(X )2 → R+ satisfies

        (i) (Hadamard differentiability) the map Q 7→ d(P, Q) is Hadamard differentiable at Q0 =
            G(µ) tangentially to M0 (X ), with continuous linear derivative D2 d(P, Q0 ) : M0 (X ) →
            R;
       (ii) (Separating property) d(P, Q) = 0 =⇒ P = Q;
       (iii) (Weak-continuity) if d(Qn , Q) → 0 then Qn ⇒ Q.
   1
    In the semiparametric sense of 68, i.e. the unique influence function representing the functional derivative
along M0 (X ).


                                                      28
We define a general loss function ℓ(P, Q), where P is the true distribution and Q is a model output
(e.g., a divergence such as KL or Wasserstein).
The plug-in loss is
                                       ℓbm := ℓ(P, G(ϕ(Pm )))
and the population loss is
                                      ℓ∗ := ℓ(P, G(ϕ(P )))
where Pm is the empirical measure of the sample, ϕ is the encoder, and G is the generator.
Lemma 3 (Donsker’s Theorem for Empirical Measures). Let (X , B) be a Polish space, and let
P ∈ P(X ). Let {Xi }m
                    i=1 be i.i.d. samples from P , and let Pm be the empirical measure:
                                                        m
                                                   1 X
                                         Pm =           δX .
                                                   m i=1 i
Define the empirical process:              √
                                          m(Pm − P ).
Then, viewed as an element of the Banach space ℓ∞ (F) of bounded real-valued functions on F,
where F is any P -Donsker class of measurable functions, we have:
                                    √               d
                                      m(Pm − P ) −→ GP ,
where GP is a P -Brownian bridge, a mean-zero tight Gaussian process indexed by F with covariance
function                                                              
                       Cov GP (f ), GP (g) = CovX∼P f (X), g(X) .
Lemma 4 (Functional Delta Method, [68, Thm. 3.9.4]). Let (D, ∥ · ∥D ) and (E, ∥ · ∥E ) be normed
vector spaces. Let T : D → E be a map that is Hadamard differentiable at a point z ∈ D tangentially
to a subset D0 ⊆ D, with continuous linear derivative denoted DTz : D0 → E.
Suppose:
(a) There exist random elements Zm taking values in D such that:
                                      √                d
                                        m(Zm − z) −→ Z
    for some tight limit Z taking values in D0 .
(b) Z is tight and Borel measurable.
Then:                           √                     d
                                    m T (Zm ) − T (z) −→ DTz (Z),
where DTz (Z) is a random element of E.
In particular, if Z is Gaussian in D0 and DTz is continuous and linear, then DTz (Z) is Gaussian in
E.

Main Result
Theorem 2 (Large-m behaviour of the plug-in loss). Assume 1, 2, 3, and 4. Let µ := ϕ(P ) and
                     
ℓbm := d P, G(ϕ(Pm )) .
We now combine the regularity assumptions with empirical-process theory to quantify the estimation
error of the plug-in loss

(a) Asymptotic normality of the Encoder.
               √                     d                                           
                 m ϕ(Pm ) − ϕ(P ) =   ⇒ N 0, Σϕ ,                Σϕ := VarX∼P ψP (X) .

(b) Unbiasedness of the loss. E[ℓbm ] = ℓ∗ + O(m−1 ),          ℓ∗ := d P, G(µ) .
                                                                              

(c) Asymptotic normality of the loss.
          √                d
             m (ℓbm − ℓ∗ ) =
                           ⇒ N 0, σ 2 ,        σ 2 = ∇µ ℓ⊤ Σϕ ∇µ ℓ,
                                                                                          
                                                                          ℓ(θ) := d P, G(θ) .


                                                   29
                                                                                           
(d) Sufficiency of the loss. If (ϕ⋆ , G ⋆ ) minimises P 7→ d P, G(ϕ(P )) , then G ⋆ ϕ⋆ (Pm ) ⇒ P in
    probability as m → ∞.

Proof. Step 1: Asymptotic Normality of the encoder (a). Assumption 2(iii) (asymptotic linearity)
gives
                                                   m
                    √                         1 X
                      m ϕ(Pm ) − ϕ(P ) = √            ψP (Xi ) + op (1),
                                               m i=1
and the classical multivariate CLT yields the stated convergence.
                                          √         d
Let ∆m := ϕ(Pm ) − µ so that, by (a),              ⇒ N (0, Σϕ ).
                                              m ∆m =

Step 2 (unbiasedness). Because E[∆m ] = 0 by Assumption 2(iii) and ℓ is twice continuously
differentiable in a neighbourhood of µ, a Taylor expansion gives E[ℓbm − ℓ∗ ] = 12 tr ∇2µ ℓ Var ∆m +
                                                                                     

O(m−1 ) = O(m−1 ).
Step 3: Asymptotic Normality of the loss (c). Apply the functional delta method twice:
                                                                                     √
1. to the generator G : Rd → P(X ), using Fréchet differentiability and the fact that m ∆m is tight,
obtaining
                          G(µ + ∆m ) = G(µ) + Dµ G[∆m ] + op (m−1/2 );

2. to the divergence Q 7→ d(P, Q) at Q0 := G(µ), with linear derivative ∂2 d(P, Q0 )[ · ].
                                                               √
Combining the two expansions produces the linear functional of m ∆m displayed in (c) and hence
the Gaussian limit with variance σ 2 = ∇µ ℓ⊤ Σϕ ∇µ ℓ.
                                                                                           
Step 4: Sufficiency of the loss (d). If (ϕ⋆ , G ⋆ ) is optimal, then d P, G ⋆ (ϕ⋆ (P )) = 0,
so G ⋆ (ϕ⋆ (P )) = P by Assumption 4(ii). Repeating the expansion from (c) with (ϕ⋆ , G ⋆ )
shows d P, G ⋆ (ϕ⋆ (Pm )) = Op (m−1/2 ), and consistency for weak convergence then implies
G ⋆ (ϕ⋆ (Pm )) ⇒ P.

Encoders: examples, counter-examples, and CLTs The only encoder requirement entering
Theorem 2 is Assumption 2. We now show that it is satisfied by a large family of permutation-invariant
architectures built from asymptotically-linear (M/Z) poolers.

Generic K-layer pool–concat encoder Fix K ∈ N. Given a set of samples Sm = {x1 , . . . , xm }
define recursively
     (0)                             (ℓ−1)      (ℓ)           (ℓ−1)
    hi = ψ(xi ), h̄(ℓ) = T (ℓ) h1:m , hi = MLPℓ hi                   , h̄(ℓ) ,
                                                                            
                                                                               ℓ = 1, . . . , K,
                                                                  (K)   
and set the encoder output to be another pooler ϕ(Pm ) = T (K+1) h1:m .
We call a permutation-invariant functional an asymptotically linear (AL) pooler if it is root-m
consistent and admits an influence-function expansion; precise details follow.
Definition 6 (Asymptotically-linear pooler). A symmetric map T : X m → Rd is an AL pooler at law
P if there exists ψP ∈ L2 (P ) such that
                                                             m
                     √                               1 X
                         m T (X1:m ) − ϕ(P )       = √       ψP (Xi ) + op (1).
                                                       m i=1

Examples: mean, median, trimmed mean, Huber M -estimator, M-quantiles, studentised Z-estimators
with finite variance.
Proposition 4 (CLT for K-layer AL pool–concat encoders). Assume

     (i) each T (ℓ) (ℓ = 1, . . . , K + 1) is an AL pooler at P ;
    (ii) each MLPℓ and the base feature map ψ : X → Rp are C 2 with bounded derivatives, and
         weights are frozen as m → ∞.


                                                    30
Then the encoder ϕ is distributionally invariant, pathwise differentiable, and satisfies the CLT of
Assumption 2 with
                             √                       d            
                                m ϕ(Pm ) − ϕ(P ) =    ⇒ N 0, Σϕ ,
for some finite covariance matrix Σϕ .

Sketch. The composition of Lipschitz maps (MLPℓ ) with AL poolers is Hadamard differentiable
by repeated application of the delta method (iterating Lemma 4, [68]). Plugging each AL ex-
pansionPinto the chain yields an overall AL expansion whose leading empirical-process term is
         m
m−1/2 i=1 ψP⋆ (Xi ) for some L2 (P ) function ψP⋆ , giving the CLT.

Instantiation to common architectures
Corollary 2 (DeepSets, Transformers without positional enc.). Encoder architectures of either type
below satisfy Assumption 2 and Proposition 4:

(a) DeepSets / fully-connected GNN with global mean: T (ℓ) and T (K+1) are sample means;
(b) Self-attention block with mean head:          T (ℓ) are sample means; MLPℓ includes the
    softmax-attention update.

Why max-pooling fails The max functional Tmax (x1:m ) = maxi xi is not Hadamard differentiable
at continuous laws: Its influence function is identically 0 whenever the maximum is attained at a
unique point and undefined when it is not. Consequently, the centered statistic m1/2 {Tmax (Pm ) −
Tmax (P )} has a non-Gaussian limit—the Gumbel extreme-value law—so Assumption 2(iii) fails.
Using max-pooling inside a deep encoder, therefore breaks the loss-CLT of Theorem 2. (Softmax
pooling with temperature τ > 0, on the other hand, is smooth and becomes a valid AL pooler.)

The table below summarises the status of common poolers.

            Pooler                           AL / CLT?      Influence fcn. ψP in L2 (P )?
            Sample mean                           ✓                       ✓
            Huber M -estimator (δ fixed)          ✓                       ✓
            Sample median                         ✓                       ✓
            Top-k or max                          ×                       ×
            Softmax (τ > 0 fixed)                 ✓                       ✓

Generators All neural generators considered in the experiments— MLPs, Transformer decoders,
and diffusion-score networks with fixed weights—are compositions of C 2 maps on finite-dimensional
spaces and therefore satisfy Assumption 3.

Smooth Approximation of Non-Regular Statistics The theory developed here establishes that
Hadamard differentiability of the encoder ensure asymptotic normality and consistency and in
subsection 5.1 we develop the idea that our encoders learn sufficient statistics. But what if the
sufficient statistic of interest is not Hadamard differentiable? The sample maximum is a classic
example: it is the minimal sufficient statistic for the endpoint of a uniform distribution (see Example
3), yet it is not asymptotically normal.
Let X1 , . . . , Xn ∼ Uniform(0, θ). The sample maximum
                                    X(n) := max{X1 , . . . , Xn }
satisfies
                                                   d
                                     n(θ − X(n) ) −
                                                  → Exp(1/θ),
so it converges to θ but its asymptotic distribution is exponential, not Gaussian. This occurs be-
cause the maximum is not a smooth functional of the empirical distribution: it fails Hadamard
differentiability, so the functional delta method does not apply.


                                                  31
A natural remedy is to approximate the max by a smooth function. A standard choice is the log-sum-
exp:                                                                !
                                                           n
                                                     1    X
                                                                λXi
                            LSEλ (X1 , . . . , Xn ) = log     e       .
                                                     λ    i=1
For fixed λ, this is Hadamard differentiable and thus amenable to the theory developed above. As
λ → ∞, LSEλ → maxi Xi , so we recover the max in the limit.
Corollary 3 (Smooth approximation suffices for asymptotic normality). Let T (Pm ) be a non-smooth
statistic (e.g., the maximum), and let T (λ) (Pm ) be a family of smooth approximations (e.g., LSEλ )
such that T (λ) (Pm ) → T (Pm ) pointwise. Then for any fixed λ, T (λ) (Pm ) is Hadamard differentiable
and admits asymptotically normal plug-in estimators. Moreover, if λn → ∞ slowly as n → ∞, this
family can approximate T (Pm ) arbitrarily closely while retaining asymptotic normality.
Thus, even when the true sufficient statistic is not regular, a Hadamard differentiable encoder can still
be learned to approximate it. This ensures that the asymptotic guarantees from Theorem 2 continue
to hold. This also highlight why we cannot use e.g. max-pooling in the encoder, since that would
break our CLT.

D.3    Embeddings and Predictive Sufficiency

Setting. Let M ⊂ P(X ) be the statistical manifold introduced in Section D.2.
Here we assume the statistical manifold M is d–dimensional (in the usual differential-geometric
sense), so dim TP M = d for every P ∈ M.
                                                   i.i.d.
For PP ∈ M observe Sm = (X1 , . . . , Xm ) ∼ P and write the empirical measure Pm =
      m
m−1 i=1 δXi .
Throughout we use the plug-in predictor Pm . Given a statistic Tm = ϕ(Pm ) with ϕ : M → Rd ,
define a measurable reconstruction map R : ϕ(M) → M and set
                                          ϕ
                                                    
                                        Pm  := R Tm .
Definition 7 (Predictive sufficiency). The statistic Tm = ϕ(Pm ) is asymptotically predictive sufficient
if there exists a reconstruction R such that, for every P ∈ M,
                                             ϕ              P ⊗m
                                       Pm − Pm TV
                                                  −−−−→ 0.
                                                            m→∞

This notion of sufficiency coincides with the one used in Section D.2: both ask that the predictor
                                ϕ
available to the decoder (here Pm ) converges in total variation to the full plug-in predictor Pm .
Theorem 3 (Embedding ⇐⇒ Predictive sufficiency). Assume ϕ is C 1 and satisfies the encoder
regularity conditions of Assumption 2. Then the following are equivalent.
      (i) Smooth embedding: ϕ is injective and its differential dϕP : TP M → Rd is bijective for
          every P ∈ M.
      (ii) Predictive sufficiency: Tm = ϕ(Pm ) is asymptotically plug-in sufficient in the sense of
           Definition 7.

Proof. Throughout, ∥ · ∥BL denotes the bounded–Lipschitz norm on signed measures, and ∥ · ∥TV ≤
∥ · ∥BL .

Step 1: (i) =⇒ (ii).
Step 1(a): global inverse and Lipschitz constant. Because ϕ is a C 1 diffeomorphism onto its image,
the inverse–function theorem supplies, for every P ∈ M, an open neighbourhood UP ⊂ M on which
R ≡ ϕ−1 is also C 1 . Shrink UP so that the operator norm of dRQ is bounded by some LP < ∞ for
all Q ∈ UP ; then R is LP -Lipschitz on UP under ∥ · ∥BL .
Step 1(b): stochastic linearisation of ϕ(Pm ). Encoder regularity (Assumption 2) gives
                                                     m
                  √                             1 X
                     m ϕ(Pm ) − ϕ(P ) =         √       ψP (Xi ) + oP (1) in Rd ,
                                                  m i=1


                                                   32
so ∥ϕ(Pm ) − ϕ(P )∥ = OP (m−1/2 ).
Step 1(c): reconstructing Pm . For m large enough Pm ∈ UP with probability one, whence
                 Pm − R ϕ(Pm ) BL ≤ LP ϕ(Pm ) − ϕ(P ) = OP (m−1/2 ).
                                  

Dividing by ∥ · ∥TV concludes plug-in sufficiency.

Step 2: (ii) =⇒ (i).
Step 2(a): Continuity of ϕ. Suppose Pn → P in M but ϕ(Pn ) ̸→ ϕ(P ). Choose ε > 0 and a
                                                                             (n)
subsequence (still indexed by n) with ∥ϕ(Pn ) − ϕ(P )∥ ≥ ε. For each n draw Sm ∼ Pn⊗mn with
                               (n)                                                   (n)
mn ↑ ∞ slowly enough that ∥Pmn − Pn ∥TV ≤ ε/4 w.p. ≥ 1 − ε. By sufficiency, ∥R(ϕ(Pmn )) −
  (n)                                                                               (n)
Pmn ∥TV ≤ ε/4 with the same probability. The triangle inequality then forces ∥R(ϕ(Pmn )) −
                                    (n)   d
P ∥TV ≥ ε/2, contradicting R(ϕ(Pmn )) −
                                      → P . Hence ϕ is continuous at every point.
Step 2(b): Injectivity of ϕ. Assume ϕ(P1 ) = ϕ(P2 ) with P1 ̸= P2 . Choose a measurable set B for
which P1 (B) ̸= P2 (B). Under P1⊗m we have Pm (B) → P1 (B) almost surely, while sufficiency
yields R(Tm )(B) → P1 (B). Repeating under P2⊗m forces P2 (B) = P1 (B), contradiction. Hence
ϕ is injective.
Step 2(c): Injectivity of dϕP . Suppose there is v ∈ TP M \ {0} with dϕP [v] = 0. Pick a C 1 path
t 7→ Pt in M with P0 = P and ∂t Pt |0 = v. Taylor expansion of ϕ(Pt ) yields ∥ϕ(Pt )−ϕ(P )∥ = o(t),
whereas ∥Pt − P ∥BL = Θ(t). Setting t = m−1/2 violates sufficiency exactly as in the previous step.
Hence dϕP is injective; because the tangent and target spaces share the same (finite) dimension, it is
bijective.
Step 2(d): Smooth embedding. Injectivity, continuity, and bijective differentials for all P ∈ M imply
that ϕ is a smooth embedding.
Remark (Identifiability is automatic). Because each P ∈ M already defines a unique predictive
distribution, any statistic that is plug-in sufficient must be injective; no separate identifiability
condition is required.




                                                 33
E     Extensions
E.1   Extension to Multiscale Settings

In many applications, data is naturally organized across multiple scales. For example, we may observe
distributions of samples at a fine scale (e.g., single cells), grouped into entities at a coarser scale (e.g.,
patients), which themselves may belong to larger groups (e.g., hospitals). More generally, we may
observe hierarchical data in which each level exhibits internal distributional structure.
Our framework naturally extends to such multiscale settings. At each scale s, we observe a set
of units indexed by i = 1, . . . , n(s) . Each unit i at scale s is associated with: a set of samples
  (s)      (s)                                         (s)                            (s+1)
Si,m = {xij }m j=1 , drawn i.i.d. from a distribution Pi   and a higher-scale sample xi     ∈ X (s+1) ,
representing the corresponding entity at scale s + 1.
                                     (s)
The lower-scale distributions Pi are drawn i.i.d. from a meta-distribution Q(s) over P(X (s) ), while
                           (s+1)                   (s+1)          (s+1)
the higher-scale samples xi      are drawn from Pi       , where Pi     ∼ Q(s+1) .
                         (s)
Each lower-scale set Si,m defines an empirical measure
                                                         m
                                         (s)        1 X
                                        Pi,m =           δ (s) ∈ Pm (X (s) ).
                                                    m j=1 xij

At each scale we learn: an encoder E (s) : Pm (X (s) ) → Rds mapping lower-scale empirical
distributions into latent space, an encoder E (s+1) : X (s+1) → Rds+1 mapping higher-scale samples
into the corresponding latent space, and generators G (s) : Rds → P(X (s) ) and G (s+1) : Rds+1 →
P(X (s+1) ) at each scale.
To link adjacent scales, we introduce deterministic maps
                           f (s) : Rds → Rds+1               and g (s) : Rds+1 → Rds ,
which project embeddings upward and downward between latent spaces.
We jointly train to enforce: Approximate identity at each scale:
                                  (s)              (s)                        (s+1)         (s+1)
                   G (s) (E (s) (Si,m )) ≈ Pi ,          G (s+1) (E (s+1) (xi         )) ≈ Pi       ,
                                                                                                    (s)
and co-embedding consistency: the mapped lower-scale embedding f (s) (E (s) (Si,m )) should align
                                                     (s+1)
with the higher-scale embedding E (s+1) (xi                  ) and vice versa via g (s) .
Formally, we optimize objectives of the form:
                                             (s)                   (s)
                               L = d Pi , G (s) (E (s) (Si,m ))
                                                                         
                                                                                                          (4)
                                                (s+1)                       (s+1) 
                                 +         d Pi        , G (s+1) (E (s+1) (xi    ))                       (5)
                                                            (s)                (s+1) 2
                                 +         ∥f (s) (E (s) (Si,m )) − E (s+1) (xi     )∥                    (6)
                                                             (s+1)            (s)
                                 +         ∥g (s) (E (s+1) (Si,m )) − E (s) (xi )∥2                       (7)

where d is a divergence or distance (e.g., KL divergence, Wasserstein distance) defined by the
generative model. One natural approach would be to let f (s) , g (s) both be the identity, forcing the
model to learn a co-embedding across scales. But this may be too rigid and we might prefer more
flexilbity in practice.
This bi-directional coupling ensures that embeddings at adjacent scales are mutually predictive
and geometrically aligned, while each scale individually satisfies distributional invariance and
approximate identity. The framework naturally generalizes to hierarchies involving more than two
scales by recursively composing the maps f (s) and g (s) across levels.




                                                              34
F    Broader impacts
Generative distribution embeddings provide a general framework for modeling data across scales.
They are broadly applicable to a wide variety of problems, including those with direct societal
consequences, for example in healthcare. In these settings, it will be critical to consider any potential
inequities induced by GDEs, as is the case for any modelling approach. Lastly, we acknowledge
the environmental impact of this paper which used nontrivial amounts of computational resources,
estimated to be about 54kg CO2 .




                                                  35

