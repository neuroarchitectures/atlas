# Paper (Radosavovic et al. 2020)

> Source: `https://arxiv.org/abs/2003.13678`

---

Designing Network Design Spaces
Ilija Radosavovic
Raj Prateek Kosaraju
Ross Girshick
Kaiming He
Piotr Doll´ar
Facebook AI Research (FAIR)
Abstract
In this work, we present a new network design paradigm.
Our goal is to help advance the understanding of net-
work design and discover design principles that generalize
across settings. Instead of focusing on designing individual
network instances, we design network design spaces that
parametrize populations of networks. The overall process
is analogous to classic manual design of networks, but ele-
vated to the design space level. Using our methodology we
explore the structure aspect of network design and arrive at
a low-dimensional design space consisting of simple, regu-
lar networks that we call RegNet. The core insight of the
RegNet parametrization is surprisingly simple: widths and
depths of good networks can be explained by a quantized
linear function. We analyze the RegNet design space and
arrive at interesting ﬁndings that do not match the current
practice of network design. The RegNet design space pro-
vides simple and fast networks that work well across a wide
range of ﬂop regimes. Under comparable training settings
and ﬂops, the RegNet models outperform the popular Efﬁ-
cientNet models while being up to 5× faster on GPUs.
1. Introduction
Deep convolutional neural networks are the engine of vi-
sual recognition. Over the past several years better architec-
tures have resulted in considerable progress in a wide range
of visual recognition tasks. Examples include LeNet [15],
AlexNet [13], VGG [26], and ResNet [8]. This body of
work advanced both the effectiveness of neural networks as
well as our understanding of network design. In particular,
the above sequence of works demonstrated the importance
of convolution, network and data size, depth, and residuals,
respectively. The outcome of these works is not just partic-
ular network instantiations, but also design principles that
can be generalized and applied to numerous settings.
While manual network design has led to large advances,
ﬁnding well-optimized networks manually can be challeng-
ing, especially as the number of design choices increases. A
popular approach to address this limitation is neural archi-
tecture search (NAS). Given a ﬁxed search space of possible
A
B
C
B
C
A
Figure 1. Design space design. We propose to design network de-
sign spaces, where a design space is a parametrized set of possible
model architectures. Design space design is akin to manual net-
work design, but elevated to the population level. In each step of
our process the input is an initial design space and the output is a
reﬁned design space of simpler or better models. Following [21],
we characterize the quality of a design space by sampling models
and inspecting their error distribution. For example, in the ﬁgure
above we start with an initial design space A and apply two reﬁne-
ment steps to yield design spaces B then C. In this case C ⊆B ⊆A
(left), and the error distributions are strictly improving from A to B
to C (right). The hope is that design principles that apply to model
populations are more likely to be robust and generalize.
networks, NAS automatically ﬁnds a good model within the
search space. Recently, NAS has received a lot of attention
and shown excellent results [34, 18, 29].
Despite the effectiveness of NAS, the paradigm has lim-
itations. The outcome of the search is a single network in-
stance tuned to a speciﬁc setting (e.g., hardware platform).
This is sufﬁcient in some cases; however, it does not enable
discovery of network design principles that deepen our un-
derstanding and allow us to generalize to new settings. In
particular, our aim is to ﬁnd simple models that are easy to
understand, build upon, and generalize.
In this work, we present a new network design paradigm
that combines the advantages of manual design and NAS.
Instead of focusing on designing individual network in-
stances, we design design spaces that parametrize popula-
tions of networks.1 Like in manual design, we aim for in-
terpretability and to discover general design principles that
describe networks that are simple, work well, and general-
ize across settings. Like in NAS, we aim to take advantage
of semi-automated procedures to help achieve these goals.
1We use the term design space following [21], rather than search space,
to emphasize that we are not searching for network instances within the
space. Instead, we are designing the space itself.
1
arXiv:2003.13678v1  [cs.CV]  30 Mar 2020

The general strategy we adopt is to progressively design
simpliﬁed versions of an initial, relatively unconstrained,
design space while maintaining or improving its quality
(Figure 1). The overall process is analogous to manual de-
sign, elevated to the population level and guided via distri-
bution estimates of network design spaces [21].
As a testbed for this paradigm, our focus is on ex-
ploring network structure (e.g., width, depth, groups, etc.)
assuming standard model families including VGG [26],
ResNet [8], and ResNeXt [31]. We start with a relatively
unconstrained design space we call AnyNet (e.g., widths
and depths vary freely across stages) and apply our human-
in-the-loop methodology to arrive at a low-dimensional de-
sign space consisting of simple “regular” networks, that we
call RegNet. The core of the RegNet design space is sim-
ple: stage widths and depths are determined by a quantized
linear function. Compared to AnyNet, the RegNet design
space has simpler models, is easier to interpret, and has a
higher concentration of good models.
We design the RegNet design space in a low-compute,
low-epoch regime using a single network block type on Im-
ageNet [3]. We then show that the RegNet design space
generalizes to larger compute regimes, schedule lengths,
and network block types. Furthermore, an important prop-
erty of the design space design is that it is more interpretable
and can lead to insights that we can learn from. We analyze
the RegNet design space and arrive at interesting ﬁndings
that do not match the current practice of network design.
For example, we ﬁnd that the depth of the best models is sta-
ble across compute regimes (∼20 blocks) and that the best
models do not use either a bottleneck or inverted bottleneck.
We compare top REGNET models to existing networks
in various settings. First, REGNET models are surprisingly
effective in the mobile regime. We hope that these sim-
ple models can serve as strong baselines for future work.
Next, REGNET models lead to considerable improvements
over standard RESNE(X)T [8, 31] models in all metrics.
We highlight the improvements for ﬁxed activations, which
is of high practical interest as the number of activations
can strongly inﬂuence the runtime on accelerators such as
GPUs. Next, we compare to the state-of-the-art EFFICIENT-
NET [29] models across compute regimes. Under compa-
rable training settings and ﬂops, REGNET models outper-
form EFFICIENTNET models while being up to 5× faster on
GPUs. We further test generalization on ImageNetV2 [24].
We note that network structure is arguably the simplest
form of a design space design one can consider. Focusing
on designing richer design spaces (e.g., including operators)
may lead to better networks. Nevertheless, the structure will
likely remain a core component of such design spaces.
In order to facilitate future research we will release all
code and pretrained models introduced in this work.2
2https://github.com/facebookresearch/pycls
2. Related Work
Manual network design. The introduction of AlexNet [13]
catapulted network design into a thriving research area.
In the following years, improved network designs were
proposed; examples include VGG [26], Inception [27,
28], ResNet [8], ResNeXt [31], DenseNet [11], and Mo-
bileNet [9, 25]. The design process behind these networks
was largely manual and focussed on discovering new design
choices that improve accuracy e.g., the use of deeper models
or residuals. We likewise share the goal of discovering new
design principles. In fact, our methodology is analogous to
manual design but performed at the design space level.
Automated network design. Recently, the network design
process has shifted from a manual exploration to more au-
tomated network design, popularized by NAS. NAS has
proven to be an effective tool for ﬁnding good models,
e.g., [35, 23, 17, 20, 18, 29]. The majority of work in NAS
focuses on the search algorithm, i.e., efﬁciently ﬁnding the
best network instances within a ﬁxed, manually designed
search space (which we call a design space). Instead, our
focus is on a paradigm for designing novel design spaces.
The two are complementary: better design spaces can im-
prove the efﬁciency of NAS search algorithms and also lead
to existence of better models by enriching the design space.
Network scaling. Both manual and semi-automated net-
work design typically focus on ﬁnding best-performing net-
work instances for a speciﬁc regime (e.g., number of ﬂops
comparable to ResNet-50). Since the result of this proce-
dure is a single network instance, it is not clear how to adapt
the instance to a different regime (e.g., fewer ﬂops). A com-
mon practice is to apply network scaling rules, such as vary-
ing network depth [8], width [32], resolution [9], or all three
jointly [29]. Instead, our goal is to discover general design
principles that hold across regimes and allow for efﬁcient
tuning for the optimal network in any target regime.
Comparing networks. Given the vast number of possible
network design spaces, it is essential to use a reliable com-
parison metric to guide our design process. Recently, the
authors of [21] proposed a methodology for comparing and
analyzing populations of networks sampled from a design
space. This distribution-level view is fully-aligned with our
goal of ﬁnding general design principles. Thus, we adopt
this methodology and demonstrate that it can serve as a use-
ful tool for the design space design process.
Parameterization. Our ﬁnal quantized linear parameteri-
zation shares similarity with previous work, e.g. how stage
widths are set [26, 7, 32, 11, 9]. However, there are two key
differences. First, we provide an empirical study justifying
the design choices we make. Second, we give insights into
structural design choices that were not previously under-
stood (e.g., how to set the number of blocks in each stages).
2

3. Design Space Design
Our goal is to design better networks for visual recog-
nition. Rather than designing or searching for a single best
model under speciﬁc settings, we study the behavior of pop-
ulations of models. We aim to discover general design prin-
ciples that can apply to and improve an entire model pop-
ulation. Such design principles can provide insights into
network design and are more likely to generalize to new set-
tings (unlike a single model tuned for a speciﬁc scenario).
We rely on the concept of network design spaces intro-
duced by Radosavovic et al. [21]. A design space is a large,
possibly inﬁnite, population of model architectures. The
core insight from [21] is that we can sample models from
a design space, giving rise to a model distribution, and turn
to tools from classical statistics to analyze the design space.
We note that this differs from architecture search, where the
goal is to ﬁnd the single best model from the space.
In this work, we propose to design progressively simpli-
ﬁed versions of an initial, unconstrained design space. We
refer to this process as design space design. Design space
design is akin to sequential manual network design, but el-
evated to the population level. Speciﬁcally, in each step of
our design process the input is an initial design space and
the output is a reﬁned design space, where the aim of each
design step is to discover design principles that yield popu-
lations of simpler or better performing models.
We begin by describing the basic tools we use for design
space design in §3.1. Next, in §3.2 we apply our method-
ology to a design space, called AnyNet, that allows uncon-
strained network structures. In §3.3, after a sequence of de-
sign steps, we obtain a simpliﬁed design space consisting of
only regular network structures that we name RegNet. Fi-
nally, as our goal is not to design a design space for a single
setting, but rather to discover general principles of network
design that generalize to new settings, in §3.4 we test the
generalization of the RegNet design space to new settings.
Relative to the AnyNet design space, the RegNet de-
sign space is: (1) simpliﬁed both in terms of its dimension
and type of network conﬁgurations it permits, (2) contains
a higher concentration of top-performing models, and (3) is
more amenable to analysis and interpretation.
3.1. Tools for Design Space Design
We begin with an overview of tools for design space de-
sign. To evaluate and compare design spaces, we use the
tools introduced by Radosavovic et al. [21], who propose to
quantify the quality of a design space by sampling a set of
models from that design space and characterizing the result-
ing model error distribution. The key intuition behind this
approach is that comparing distributions is more robust and
informative than using search (manual or automated) and
comparing the best found models from two design spaces.
40
45
50
55
60
65
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[39.0|49.0] AnyNetX
10
20
30
40
50
depth
40
50
60
70
error
200
400
600
800
1000
w4
40
50
60
70
error
Figure 2. Statistics of the AnyNetX design space computed with
n = 500 sampled models. Left: The error empirical distribution
function (EDF) serves as our foundational tool for visualizing the
quality of the design space. In the legend we report the min error
and mean error (which corresponds to the area under the curve).
Middle: Distribution of network depth d (number of blocks) ver-
sus error. Right: Distribution of block widths in the fourth stage
(w4) versus error. The blue shaded regions are ranges containing
the best models with 95% conﬁdence (obtained using an empirical
bootstrap), and the black vertical line the most likely best value.
To obtain a distribution of models, we sample and train
n models from a design space. For efﬁciency, we primarily
do so in a low-compute, low-epoch training regime. In par-
ticular, in this section we use the 400 million ﬂop3 (400MF)
regime and train each sampled model for 10 epochs on the
ImageNet dataset [3]. We note that while we train many
models, each training run is fast: training 100 models at
400MF for 10 epochs is roughly equivalent in ﬂops to train-
ing a single ResNet-50 [8] model at 4GF for 100 epochs.
As in [21], our primary tool for analyzing design space
quality is the error empirical distribution function (EDF).
The error EDF of n models with errors ei is given by:
F(e) = 1
n
n
X
i=1
1[ei < e].
(1)
F(e) gives the fraction of models with error less than e. We
show the error EDF for n = 500 sampled models from the
AnyNetX design space (described in §3.2) in Figure 2 (left).
Given a population of trained models, we can plot and
analyze various network properties versus network error,
see Figure 2 (middle) and (right) for two examples taken
from the AnyNetX design space. Such visualizations show
1D projections of a complex, high-dimensional space, and
can help obtain insights into the design space. For these
plots, we employ an empirical bootstrap4 [5] to estimate
the likely range in which the best models fall.
To summarize: (1) we generate distributions of models
obtained by sampling and training n models from a design
space, (2) we compute and plot error EDFs to summarize
design space quality, (3) we visualize various properties of a
design space and use an empirical bootstrap to gain insight,
and (4) we use these insights to reﬁne the design space.
3Following common practice, we use ﬂops to mean multiply-adds.
Moreover, we use MF and GF to denote 106 and 109 ﬂops, respectively.
4Given n pairs (xi, ei) of model statistic xi (e.g. depth) and corre-
sponding error ei, we compute the empirical bootstrap by: (1) sampling
with replacement 25% of the pairs, (2) selecting the pair with min error
in the sample, (3) repeating this 104 times, and ﬁnally (4) computing the
95% CI for the min x value. The median gives the most likely best value.
3

3, r, r
w0, r/2, r/2
w4, r/32, r/32
n, 1, 1
stem
body
head
w0, r/2, r/2
w1, r/4, r/4
stage 1
stage 2
wi-1, 2ri, 2ri
wi, ri, ri
block 1
block 2
w4, r/32, r/32
wi, ri, ri
(a) network 
(b) body 
(c) stage i
stage 4
block di
…
…
Figure 3. General network structure for models in our design
spaces. (a) Each network consists of a stem (stride-two 3×3 conv
with w0 = 32 output channels), followed by the network body that
performs the bulk of the computation, and then a head (average
pooling followed by a fully connected layer) that predicts n output
classes. (b) The network body is composed of a sequence of stages
that operate at progressively reduced resolution ri. (c) Each stage
consists of a sequence of identical blocks, except the ﬁrst block
which uses stride-two conv. While the general structure is simple,
the total number of possible network conﬁgurations is vast.
3.2. The AnyNet Design Space
We next introduce our initial AnyNet design space. Our
focus is on exploring the structure of neural networks as-
suming standard, ﬁxed network blocks (e.g., residual bot-
tleneck blocks). In our terminology the structure of the net-
work includes elements such as the number of blocks (i.e.
network depth), block widths (i.e. number of channels), and
other block parameters such as bottleneck ratios or group
widths. The structure of the network determines the distri-
bution of compute, parameters, and memory throughout the
computational graph of the network and is key in determin-
ing its accuracy and efﬁciency.
The basic design of networks in our AnyNet design
space is straightforward. Given an input image, a network
consists of a simple stem, followed by the network body that
performs the bulk of the computation, and a ﬁnal network
head that predicts the output classes, see Figure 3a. We
keep the stem and head ﬁxed and as simple as possible, and
instead focus on the structure of the network body that is
central in determining network compute and accuracy.
The network body consists of 4 stages operating at pro-
gressively reduced resolution, see Figure 3b (we explore
varying the number of stages in §3.4). Each stage consists
of a sequence of identical blocks, see Figure 3c. In total, for
each stage i the degrees of freedom include the number of
blocks di, block width wi, and any other block parameters.
While the general structure is simple, the total number of
possible networks in the AnyNet design space is vast.
Most of our experiments use the standard residual bottle-
necks block with group convolution [31], shown in Figure 4.
We refer to this as the X block, and the AnyNet design space
built on it as AnyNetX (we explore other blocks in §3.4).
While the X block is quite rudimentary, we show it can be
surprisingly effective when network structure is optimized.
1×1, s=2
wi, ri, ri
⨁
wi, ri, ri
wi/bi, ri, ri
wi/bi, ri, ri
1×1, s=1
3×3, gi, s=1
1×1, s=1
(a) X block, s=1
wi, ri, ri
⨁
wi-1, 2ri, 2ri
wi/bi, 2ri, 2ri
wi/bi, ri, ri
1×1, s=1
3×3, gi, s=2
1×1, s=1
(b) X block, s=2
Figure 4. The X block is based on the standard residual bottleneck
block with group convolution [31]. (a) Each X block consists of a
1×1 conv, a 3×3 group conv, and a ﬁnal 1×1 conv, where the 1×1
convs alter the channel width. BatchNorm [12] and ReLU follow
each conv. The block has 3 parameters: the width wi, bottleneck
ratio bi, and group width gi. (b) The stride-two (s = 2) version.
The AnyNetX design space has 16 degrees of freedom as
each network consists of 4 stages and each stage i has 4 pa-
rameters: the number of blocks di, block width wi, bottle-
neck ratio bi, and group width gi. We ﬁx the input resolution
r = 224 unless otherwise noted. To obtain valid models,
we perform log-uniform sampling of di ≤16, wi ≤1024
and divisible by 8, bi ∈{1, 2, 4}, and gi ∈{1, 2, . . . , 32}
(we test these ranges later). We repeat the sampling until
we obtain n = 500 models in our target complexity regime
(360MF to 400MF), and train each model for 10 epochs.5
Basic statistics for AnyNetX were shown in Figure 2.
There are (16·128·3·6)4 ≈1018 possible model conﬁgu-
rations in the AnyNetX design space. Rather than searching
for the single best model out of these ∼1018 conﬁgurations,
we explore whether there are general design principles that
can help us understand and reﬁne this design space. To do
so, we apply our approach of designing design spaces. In
each step of this approach, our aims are:
1. to simplify the structure of the design space,
2. to improve the interpretability of the design space,
3. to improve or maintain the design space quality,
4. to maintain model diversity in the design space.
We now apply this approach to the AnyNetX design space.
AnyNetXA. For clarity, going forward we refer to the initial,
unconstrained AnyNetX design space as AnyNetXA.
AnyNetXB. We ﬁrst test a shared bottleneck ratio bi = b
for all stages i for the AnyNetXA design space, and refer
to the resulting design space as AnyNetXB. As before, we
sample and train 500 models from AnyNetXB in the same
settings. The EDFs of AnyNetXA and AnyNetXB, shown in
Figure 5 (left), are virtually identical both in the average and
best case. This indicates no loss in accuracy when coupling
the bi. In addition to being simpler, the AnyNetXB is more
amenable to analysis, see for example Figure 5 (right).
5Our training setup in §3 exactly follows [21]. We use SGD with mo-
mentum of 0.9, mini-batch size of 128 on 1 GPU, and a half-period cosine
schedule with initial learning rate of 0.05 and weight decay of 5·10−5.
Ten epochs are usually sufﬁcient to give robust population statistics.
4

40
45
50
55
60
65
70
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[39.0|49.0] AnyNetXA
[39.0|49.2] AnyNetXB
40
50
60
70
error
[39.0|49.2] AnyNetXB
[38.9|49.4] AnyNetXC
1
2
4
b
40
50
60
70
error
Figure 5. AnyNetXB (left) and AnyNetXC (middle) introduce a
shared bottleneck ratio bi = b and shared group width gi = g,
respectively. This simpliﬁes the design spaces while resulting in
virtually no change in the error EDFs. Moreover, AnyNetXB and
AnyNetXC are more amendable to analysis. Applying an empir-
ical bootstrap to b and g we see trends emerge, e.g., with 95%
conﬁdence b ≤2 is best in this regime (right). No such trends are
evident in the individual bi and gi in AnyNetXA (not shown).
0
1
2
3
4
5
6
7
8
64
256
1024
width
di = [1, 2, 2, 4]
wi = [48, 80, 352, 560]
g = 8, b = 1, e = 38.9%
0
2
4
6
8
10
di = [2, 3, 1, 6]
wi = [64, 128, 448, 656]
g = 8, b = 2, e = 39.7%
0
4
8
12
16
20
24
28
di = [2, 7, 17, 6]
wi = [48, 48, 208, 608]
g = 8, b = 2, e = 40.4%
0
1
2
3
4
block index
64
256
1024
width
di = [1, 1, 1, 2]
wi = [48, 704, 32, 48]
g = 8, b = 2, e = 72.0%
0
2
4
6
8
10
block index
di = [3, 2, 2, 4]
wi = [48, 576, 128, 32]
g = 8, b = 4, e = 65.7%
0
2
4
6
8
10
12
14
16
18
block index
di = [1, 1, 9, 9]
wi = [64, 960, 64, 64]
g = 8, b = 4, e = 65.0%
Figure 6. Example good and bad AnyNetXC networks, shown in
the top and bottom rows, respectively. For each network, we plot
the width wj of every block j up to the network depth d. These
per-block widths wj are computed from the per-stage block depths
di and block widths wi (listed in the legends for reference).
AnyNetXC. Our second reﬁnement step closely follows the
ﬁrst. Starting with AnyNetXB, we additionally use a shared
group width gi = g for all stages to obtain AnyNetXC. As
before, the EDFs are nearly unchanged, see Figure 5 (mid-
dle). Overall, AnyNetXC has 6 fewer degrees of freedom
than AnyNetXA, and reduces the design space size nearly
four orders of magnitude. Interestingly, we ﬁnd g > 1 is
best (not shown); we analyze this in more detail in §4.
AnyNetXD. Next, we examine typical network structures of
both good and bad networks from AnyNetXC in Figure 6.
A pattern emerges: good network have increasing widths.
We test the design principle of wi+1 ≥wi, and refer to the
design space with this constraint as AnyNetXD. In Figure 7
(left) we see this improves the EDF substantially. We return
to examining other options for controlling width shortly.
AnyNetXE. Upon further inspection of many models (not
shown), we observed another interesting trend. In addition
to stage widths wi increasing with i, the stage depths di
likewise tend to increase for the best models, although not
necessarily in the last stage. Nevertheless, we test a design
space variant AnyNetXE with di+1 ≥di in Figure 7 (right),
and see it also improves results. Finally, we note that the
constraints on wi and di each reduce the design space by 4!,
with a cumulative reduction of O(107) from AnyNetXA.
40
50
60
70
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[38.9|49.4] AnyNetXC
[38.7|43.2] + wi + 1
wi
[47.5|56.1] + wi + 1 = wi
[52.2|63.2] + wi + 1
wi
40
45
50
55
error
[38.7|43.2] AnyNetXD
[38.7|42.7] + di + 1
di
[39.0|46.6] + di + 1 = di
[41.0|48.8] + di + 1
di
Figure 7. AnyNetXD (left) and AnyNetXE (right). We show var-
ious constraints on the per stage widths wi and depths di. In both
cases, having increasing wi and di is beneﬁcial, while using con-
stant or decreasing values is much worse. Note that AnyNetXD
= AnyNetXC + wi+1 ≥wi, and AnyNetXE = AnyNetXD +
di+1 ≥di. We explore stronger constraints on wi and di shortly.
0
3
6
9
12
15
18
21
block index
32
64
128
256
512
1024
width
0
2
4
6
8
10
12
14
block index
32
64
128
256
512
width
di = [2, 4, 5, 5]
wi = [64, 128, 272, 704]
g = 4, b = 2, e = 38.7%
wa = 30, w0 = 80, wm = 1.9
0
1
2
3
4
5
6
7
8
9
block index
di = [1, 1, 3, 5]
wi = [32, 144, 256, 448]
g = 16, b = 1, e = 39.7%
wa = 61, w0 = 48, wm = 2.2
0
1
2
3
4
5
efit
40
50
60
70
80
error
AnyNetXC
0
1
2
3
4
5
efit
AnyNetXD
0
1
2
3
4
5
efit
AnyNetXE
Figure 8. Linear ﬁts. Top networks from the AnyNetX design
space can be well modeled by a quantized linear parameterization,
and conversely, networks for which this parameterization has a
higher ﬁtting error eﬁt tend to perform poorly. See text for details.
3.3. The RegNet Design Space
To gain further insight into the model structure, we show
the best 20 models from AnyNetXE in a single plot, see Fig-
ure 8 (top-left). For each model, we plot the per-block width
wj of every block j up to the network depth d (we use i and
j to index over stages and blocks, respectively). See Fig-
ure 6 for reference of our model visualization.
While there is signiﬁcant variance in the individual mod-
els (gray curves), in the aggregate a pattern emerges. In par-
ticular, in the same plot we show the line wj = 48·(j+1) for
0 ≤j ≤20 (solid black curve, please note that the y-axis is
logarithmic). Remarkably, this trivial linear ﬁt seems to ex-
plain the population trend of the growth of network widths
for top models. Note, however, that this linear ﬁt assigns a
different width wj to each block, whereas individual models
have quantized widths (piecewise constant functions).
To see if a similar pattern applies to individual models,
we need a strategy to quantize a line to a piecewise constant
function. Inspired by our observations from AnyNetXD and
AnyNetXE, we propose the following approach. First, we
introduce a linear parameterization for block widths:
uj = w0 + wa · j
for
0 ≤j < d
(2)
This parameterization has three parameters: depth d, initial
width w0 > 0, and slope wa > 0, and generates a differ-
ent block width uj for each block j < d. To quantize uj,
5

40
45
50
55
60
65
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[39.0|49.0] AnyNetXA
[38.7|42.7] AnyNetXE
[38.2|41.0] RegNetX
38
40
42
44
46
error
[38.2|41.0] RegNetX 
[38.0|40.7] RegNetX wm = 2
[38.2|40.1] RegNetX w0 = wa
2
8
32
128
sample size
40
45
50
55
error
AnyNetXA
RegNetX
Figure 9. RegNetX design space. See text for details.
we introduce an additional parameter wm > 0 that controls
quantization as follows. First, given uj from Eqn. (2), we
compute sj for each block j such that the following holds:
uj = w0 · wsj
m
(3)
Then, to quantize uj, we simply round sj (denoted by ⌊sj⌉)
and compute quantized per-block widths wj via:
wj = w0 · w⌊sj⌉
m
(4)
We can convert the per-block wj to our per-stage format by
simply counting the number of blocks with constant width,
that is, each stage i has block width wi = w0·wi
m and num-
ber of blocks di = P
j 1[⌊sj⌉= i]. When only considering
four stage networks, we ignore the parameter combinations
that give rise to a different number of stages.
We test this parameterization by ﬁtting to models from
AnyNetX. In particular, given a model, we compute the
ﬁt by setting d to the network depth and performing a grid
search over w0, wa and wm to minimize the mean log-ratio
(denoted by eﬁt) of predicted to observed per-block widths.
Results for two top networks from AnyNetXE are shown
in Figure 8 (top-right). The quantized linear ﬁts (dashed
curves) are good ﬁts of these best models (solid curves).
Next, we plot the ﬁtting error eﬁt versus network error
for every network in AnyNetXC through AnyNetXE in Fig-
ure 8 (bottom). First, we note that the best models in each
design space all have good linear ﬁts. Indeed, an empiri-
cal bootstrap gives a narrow band of eﬁt near 0 that likely
contains the best models in each design space. Second, we
note that on average, eﬁt improves going from AnyNetXC
to AnyNetXE, showing that the linear parametrization natu-
rally enforces related constraints to wi and di increasing.
To further test the linear parameterization, we design a
design space that only contains models with such linear
structure. In particular, we specify a network structure via 6
parameters: d, w0, wa, wm (and also b, g). Given these, we
generate block widths and depths via Eqn. (2)-(4). We refer
to the resulting design space as RegNet, as it contains only
simple, regular models. We sample d < 64, w0, wa < 256,
1.5 ≤wm ≤3 and b and g as before (ranges set based on
eﬁt on AnyNetXE).
The error EDF of RegNetX is shown in Figure 9
(left). Models in RegNetX have better average error than
AnyNetX while maintaining the best models. In Figure 9
(middle) we test two further simpliﬁcations. First, using
wm = 2 (doubling width between stages) slightly improves
the EDF, but we note that using wm ≥2 performs better
(shown later). Second, we test setting w0 = wa, further
restriction
dim.
combinations
total
AnyNetXA
none
16
(16·128·3·6)4
∼1.8·1018
AnyNetXB
+ bi+1 = bi
13
(16·128·6)4·3
∼6.8·1016
AnyNetXC
+ gi+1 = gi
10
(16·128)4·3·6
∼3.2·1014
AnyNetXD
+ wi+1 ≥wi
10
(16·128)4·3·6/(4!)
∼1.3·1013
AnyNetXE
+ di+1 ≥di
10
(16·128)4·3·6/(4!)2
∼5.5·1011
RegNet
quantized linear
6
∼644·6·3
∼3.0·108
Table 1. Design space summary. See text for details.
35
40
45
50
55
60
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
flops=800M
[35.8|44.5] AnyNetXA
[35.1|38.5] AnyNetXE
[34.6|36.8] RegNetX
30
35
40
45
50
55
60
epochs=50
[30.0|38.8] AnyNetXA
[30.0|32.5] AnyNetXE
[29.4|31.5] RegNetX
40
45
50
55
60
65
70
stages=5
[40.4|49.9] AnyNetXA
[38.4|42.8] AnyNetXE
[37.9|41.4] RegNetX
45
50
55
60
65
70
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[42.9|53.1] AnyNetRA
[42.0|45.7] AnyNetRE
[41.9|44.3] RegNetR
50
60
70
80
error
[47.0|61.2] AnyNetVA
[46.8|56.4] AnyNetVE
[46.0|49.2] RegNetV
50
55
60
65
70
error
[48.1|58.9] AnyNetVRA
[47.4|53.3] AnyNetVRE
[46.6|49.3] RegNetVR
Figure 10. RegNetX generalization. We compare RegNetX to
AnyNetX at higher ﬂops (top-left), higher epochs (top-middle),
with 5-stage networks (top-right), and with various block types
(bottom). In all cases the ordering of the design spaces is consis-
tent and we see no signs of design space overﬁtting.
simplifying the linear parameterization to uj = wa ·(j +1).
Interestingly, this performs even better. However, to main-
tain the diversity of models, we do not impose either restric-
tion. Finally, in Figure 9 (right) we show that random search
efﬁciency is much higher for RegNetX; searching over just
∼32 random models is likely to yield good models.
Table 1 shows a summary of the design space sizes (for
RegNet we estimate the size by quantizing its continuous
parameters). In designing RegNetX, we reduced the dimen-
sion of the original AnyNetX design space from 16 to 6 di-
mensions, and the size nearly 10 orders of magnitude. We
note, however, that RegNet still contains a good diversity
of models that can be tuned for a variety of settings.
3.4. Design Space Generalization
We designed the RegNet design space in a low-compute,
low-epoch training regime with only a single block type.
However, our goal is not to design a design space for a sin-
gle setting, but rather to discover general principles of net-
work design that can generalize to new settings.
In Figure 10, we compare the RegNetX design space to
AnyNetXA and AnyNetXE at higher ﬂops, higher epochs,
with 5-stage networks, and with various block types (de-
scribed in the appendix). In all cases the ordering of the
design spaces is consistent, with RegNetX > AnyNetXE >
AnyNetXA. In other words, we see no signs of overﬁtting.
These results are promising because they show RegNet can
generalize to new settings. The 5-stage results show the
regular structure of RegNet can generalize to more stages,
where AnyNetXA has even more degrees of freedom.
6

10
20
30
40
50
60
d
all
good
best
1.0
1.5
2.0
2.5
3.0
3.5
4.0
b
all
good
best
1.50
1.75
2.00
2.25
2.50
2.75
3.00
wm
all
good
best
0.2
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
0
5
10
15
20
25
30
g
all
good
best
0.2
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
0
50
100
150
200
250
wa
all
good
best
0.2
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
0
50
100
150
200
250
w0
all
good
best
Figure 11. RegNetX parameter trends. For each parameter and
each ﬂop regime we apply an empirical bootstrap to obtain the
range that contains best models with 95% conﬁdence (shown with
blue shading) and the likely best model (black line), see also Fig-
ure 2. We observe that for best models the depths d are remarkably
stable across ﬂops regimes, and b = 1 and wm ≈2.5 are best.
Block and groups widths (wa, w0, g) tend to increase with ﬂops.
ﬂops
params acts.
1×1 conv
w2r2
w2
wr2
3×3 conv
32w2r2
32w2
wr2
3×3 gr conv
32wgr2
32wg wr2
3×3 dw conv
32wr2
32w
wr2
0.2
0.4
0.6
0.8
1.0
1.2
1.4
1.6
flops (B)
10
20
30
40
50
60
inference time (ms)
r=0.643
2
4
6
8
10
12
14
activations (M)
10
20
30
40
50
60
r=0.897
0.2
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
5
10
15
20
25
30
activations (M)
all
good
best
6.5 f
ResNet
ResNeXt
0.2
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
0
50
100
150
200
params (M)
all
good
best
3.0 + 5.5 f
ResNet
ResNeXt
0.2
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
0
50
100
150
200
250
300
inference time (ms)
all
good
best
24 f + 6.0 f
ResNet
ResNeXt
Figure 12. Complexity metrics.
Top: Activations can have a
stronger correlation to runtime on hardware accelerators than ﬂops
(we measure inference time for 64 images on an NVIDIA V100
GPU). Bottom: Trend analysis of complexity vs. ﬂops and best ﬁt
curves (shown in blue) of the trends for best models (black curves).
4. Analyzing the RegNetX Design Space
We next further analyze the RegNetX design space and
revisit common deep network design choices. Our analysis
yields surprising insights that don’t match popular practice,
which allows us to achieve good results with simple models.
As the RegNetX design space has a high concentration
of good models, for the following results we switch to sam-
pling fewer models (100) but training them for longer (25
epochs) with a learning rate of 0.1 (see appendix). We do
so to observe more ﬁne-grained trends in network behavior.
RegNet trends. We show trends in the RegNetX parame-
ters across ﬂop regimes in Figure 11. Remarkably, the depth
of best models is stable across regimes (top-left), with an
optimal depth of ∼20 blocks (60 layers). This is in contrast
to the common practice of using deeper models for higher
ﬂop regimes. We also observe that the best models use a
bottleneck ratio b of 1.0 (top-middle), which effectively re-
moves the bottleneck (commonly used in practice). Next,
we observe that the width multiplier wm of good models
is ∼2.5 (top-right), similar but not identical to the popular
recipe of doubling widths across stages. The remaining pa-
rameters (g, wa, w0) increase with complexity (bottom).
25.0
27.5
30.0
32.5
35.0
37.5
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[31.1|33.4] 400MF U
[30.7|32.6] 400MF C
[28.3|30.3] 800MF U
[28.0|29.3] 800MF C
[24.1|25.9] 3.2GF U
[24.0|25.2] 3.2GF C
0.2
0.4
0.8
1.6
3.2
6.4
flops (B)
25
30
35
40
error
RegNetX U
RegNetX C
0.2
0.4
0.8
1.6
3.2
6.4
flops (B)
0
50
100
150
200
params (M)
RegNetX U
RegNetX C
Figure 13. We reﬁne RegNetX using various constraints (see
text). The constrained variant (C) is best across all ﬂop regimes
while being more efﬁcient in terms of parameters and activations.
32
34
36
38
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[30.7|32.6] b = 1, g
32
[30.5|33.3] b
1, g
32
[33.2|35.5] b
1, g = 1
30
35
40
45
error
[30.7|32.6] 400MF r = 224
[31.2|34.1] 400MF r
448
[28.0|29.3] 800MF r = 224
[28.2|30.8] 800MF r
448
30
31
32
33
34
error
[30.7|32.6] RegNetX
[29.8|31.3] RegNetY
Figure 14. We evaluate RegNetX with alternate design choices.
Left: Inverted bottleneck ( 1
8 ≤b ≤1) degrades results and depth-
wise conv (g = 1) is even worse. Middle: Varying resolution r
harms results. Right: RegNetY (Y=X+SE) improves the EDF.
Complexity analysis. In addition to ﬂops and parameters,
we analyze network activations, which we deﬁne as the size
of the output tensors of all conv layers (we list complexity
measures of common conv operators in Figure 12, top-left).
While not a common measure of network complexity, acti-
vations can heavily affect runtime on memory-bound hard-
ware accelerators (e.g., GPUs, TPUs), for example, see Fig-
ure 12 (top). In Figure 12 (bottom), we observe that for the
best models in the population, activations increase with the
square-root of ﬂops, parameters increase linearly, and run-
time is best modeled using both a linear and a square-root
term due to its dependence on both ﬂops and activations.
RegNetX constrained. Using these ﬁndings, we reﬁne the
RegNetX design space. First, based on Figure 11 (top), we
set b = 1, d ≤40, and wm ≥2. Second, we limit param-
eters and activations, following Figure 12 (bottom). This
yields fast, low-parameter, low-memory models without af-
fecting accuracy. In Figure 13, we test RegNetX with theses
constraints and observe that the constrained version is supe-
rior across all ﬂop regimes. We use this version in §5, and
further limit depth to 12 ≤d ≤28 (see also Appendix D).
Alternate design choices. Modern mobile networks often
employ the inverted bottleneck (b < 1) proposed in [25]
along with depthwise conv [1] (g = 1). In Figure 14 (left),
we observe that the inverted bottleneck degrades the EDF
slightly and depthwise conv performs even worse relative to
b = 1 and g ≥1 (see appendix for further analysis). Next,
motivated by [29] who found that scaling the input image
resolution can be helpful, we test varying resolution in Fig-
ure 14 (middle). Contrary to [29], we ﬁnd that for RegNetX
a ﬁxed resolution of 224×224 is best, even at higher ﬂops.
SE. Finally,
we evaluate RegNetX with the popular
Squeeze-and-Excitation (SE) op [10] (we abbreviate X+SE
as Y and refer to the resulting design space as RegNetY). In
Figure 14 (right), we see that RegNetY yields good gains.
7

0
2
4
6
8
10
12
16
32
64
128
256
512
1024
width
RegNetX-200MF
di = [1, 1, 4, 7]
wi = [24, 56, 152, 368]
g = 8, b = 1, e = 30.8%
wa = 36, w0 = 24, wm = 2.5
0
3
6
9
12
15
18
21
16
32
64
128
256
512
1024
RegNetX-400MF
di = [1, 2, 7, 12]
wi = [32, 64, 160, 384]
g = 16, b = 1, e = 27.2%
wa = 24, w0 = 24, wm = 2.5
0
2
4
6
8
10
12
14
16
32
64
128
256
512
1024
RegNetX-600MF
di = [1, 3, 5, 7]
wi = [48, 96, 240, 528]
g = 24, b = 1, e = 25.5%
wa = 37, w0 = 48, wm = 2.2
0
2
4
6
8
10
12
14
16
32
64
128
256
512
1024
RegNetX-800MF
di = [1, 3, 7, 5]
wi = [64, 128, 288, 672]
g = 16, b = 1, e = 24.8%
wa = 36, w0 = 56, wm = 2.3
0
2
4
6
8
10
12
14
16
32
64
128
256
512
1024
2048
width
RegNetX-1.6GF
di = [2, 4, 10, 2]
wi = [72, 168, 408, 912]
g = 24, b = 1, e = 22.9%
wa = 34, w0 = 80, wm = 2.2
0
3
6
9
12
15
18
21
24
32
64
128
256
512
1024
2048
RegNetX-3.2GF
di = [2, 6, 15, 2]
wi = [96, 192, 432, 1008]
g = 48, b = 1, e = 21.6%
wa = 26, w0 = 88, wm = 2.2
0
3
6
9
12
15
18
21
32
64
128
256
512
1024
2048
RegNetX-4.0GF
di = [2, 5, 14, 2]
wi = [80, 240, 560, 1360]
g = 40, b = 1, e = 21.3%
wa = 39, w0 = 96, wm = 2.4
0
2
4
6
8
10
12
14
16
32
64
128
256
512
1024
2048
RegNetX-6.4GF
di = [2, 4, 10, 1]
wi = [168, 392, 784, 1624]
g = 56, b = 1, e = 20.7%
wa = 61, w0 = 184, wm = 2.1
0
3
6
9
12
15
18
21
block index
64
128
256
512
1024
2048
4096
width
RegNetX-8.0GF
di = [2, 5, 15, 1]
wi = [80, 240, 720, 1920]
g = 120, b = 1, e = 20.5%
wa = 50, w0 = 80, wm = 2.9
0
2
4
6
8
10
12
14
16
18
block index
64
128
256
512
1024
2048
4096
RegNetX-12GF
di = [2, 5, 11, 1]
wi = [224, 448, 896, 2240]
g = 112, b = 1, e = 20.3%
wa = 73, w0 = 168, wm = 2.4
0
3
6
9
12
15
18
21
block index
64
128
256
512
1024
2048
4096
RegNetX-16GF
di = [2, 6, 13, 1]
wi = [256, 512, 896, 2048]
g = 128, b = 1, e = 20.0%
wa = 56, w0 = 216, wm = 2.1
0
3
6
9
12
15
18
21
block index
64
128
256
512
1024
2048
4096
RegNetX-32GF
di = [2, 7, 13, 1]
wi = [336, 672, 1344, 2520]
g = 168, b = 1, e = 19.5%
wa = 70, w0 = 320, wm = 2.0
ﬂops
params
acts
batch
infer
train
error
(B)
(M)
(M)
size
(ms)
(hr)
(top-1)
REGNETX-200MF
0.2
2.7
2.2
1024
10
2.8
31.1±0.09
REGNETX-400MF
0.4
5.2
3.1
1024
15
3.9
27.3±0.15
REGNETX-600MF
0.6
6.2
4.0
1024
17
4.4
25.9±0.03
REGNETX-800MF
0.8
7.3
5.1
1024
21
5.7
24.8±0.09
REGNETX-1.6GF
1.6
9.2
7.9
1024
33
8.7
23.0±0.13
REGNETX-3.2GF
3.2
15.3
11.4
512
57
14.3
21.7±0.08
REGNETX-4.0GF
4.0
22.1
12.2
512
69
17.1
21.4±0.19
REGNETX-6.4GF
6.5
26.2
16.4
512
92
23.5
20.8±0.07
REGNETX-8.0GF
8.0
39.6
14.1
512
94
22.6
20.7±0.07
REGNETX-12GF
12.1
46.1
21.4
512
137
32.9
20.3±0.04
REGNETX-16GF
15.9
54.3
25.5
512
168
39.7
20.0±0.11
REGNETX-32GF
31.7
107.8
36.3
256
318
76.9
19.5±0.12
Figure 15. Top REGNETX models. We measure inference time
for 64 images on an NVIDIA V100 GPU; train time is for 100
epochs on 8 GPUs with the batch size listed. Network diagram
legends contain all information required to implement the models.
5. Comparison to Existing Networks
We now compare top models from the RegNetX and
RegNetY design spaces at various complexities to the state-
of-the-art on ImageNet [3]. We denote individual models
using small caps, e.g. REGNETX. We also sufﬁx the models
with the ﬂop regime, e.g. 400MF. For each ﬂop regime, we
pick the best model from 25 random settings of the RegNet
parameters (d, g, wm, wa, w0), and re-train the top model 5
times at 100 epochs to obtain robust error estimates.
Resulting top REGNETX and REGNETY models for each
ﬂop regime are shown in Figures 15 and 16, respectively.
In addition to the simple linear structure and the trends we
analyzed in §4, we observe an interesting pattern. Namely,
the higher ﬂop models have a large number of blocks in the
third stage and a small number of blocks in the last stage.
This is similar to the design of standard RESNET models.
Moreover, we observe that the group width g increases with
complexity, but depth d saturates for large models.
Our goal is to perform fair comparisons and provide sim-
ple and easy-to-reproduce baselines. We note that along
with better architectures, much of the recently reported
gains in network performance are based on enhancements to
the training setup and regularization scheme (see Table 7).
As our focus is on evaluating network architectures, we per-
form carefully controlled experiments under the same train-
ing setup. In particular, to provide fair comparisons to clas-
sic work, we do not use any training-time enhancements.
0
2
4
6
8
10
12
16
32
64
128
256
512
1024
width
RegNetY-200MF
di = [1, 1, 4, 7]
wi = [24, 56, 152, 368]
g = 8, b = 1, e = 29.5%
wa = 36, w0 = 24, wm = 2.5
0
2
4
6
8
10
12
14
16
32
64
128
256
512
1024
RegNetY-400MF
di = [1, 3, 6, 6]
wi = [48, 104, 208, 440]
g = 8, b = 1, e = 25.8%
wa = 28, w0 = 48, wm = 2.1
0
2
4
6
8
10
12
14
16
32
64
128
256
512
1024
RegNetY-600MF
di = [1, 3, 7, 4]
wi = [48, 112, 256, 608]
g = 16, b = 1, e = 24.4%
wa = 33, w0 = 48, wm = 2.3
0
2
4
6
8
10
12
16
32
64
128
256
512
1024
RegNetY-800MF
di = [1, 3, 8, 2]
wi = [64, 128, 320, 768]
g = 16, b = 1, e = 23.5%
wa = 39, w0 = 56, wm = 2.4
0
3
6
9
12
15
18
21
24
32
64
128
256
512
1024
2048
width
RegNetY-1.6GF
di = [2, 6, 17, 2]
wi = [48, 120, 336, 888]
g = 24, b = 1, e = 22.0%
wa = 21, w0 = 48, wm = 2.6
0
3
6
9
12
15
18
32
64
128
256
512
1024
2048
RegNetY-3.2GF
di = [2, 5, 13, 1]
wi = [72, 216, 576, 1512]
g = 24, b = 1, e = 20.7%
wa = 43, w0 = 80, wm = 2.7
0
3
6
9
12
15
18
21
32
64
128
256
512
1024
2048
RegNetY-4.0GF
di = [2, 6, 12, 2]
wi = [128, 192, 512, 1088]
g = 64, b = 1, e = 20.5%
wa = 31, w0 = 96, wm = 2.2
0
3
6
9
12
15
18
21
24
32
64
128
256
512
1024
2048
RegNetY-6.4GF
di = [2, 7, 14, 2]
wi = [144, 288, 576, 1296]
g = 72, b = 1, e = 19.9%
wa = 33, w0 = 112, wm = 2.3
0
2
4
6
8
10
12
14
16
block index
64
128
256
512
1024
2048
4096
width
RegNetY-8.0GF
di = [2, 4, 10, 1]
wi = [168, 448, 896, 2016]
g = 56, b = 1, e = 19.9%
wa = 77, w0 = 192, wm = 2.2
0
2
4
6
8
10
12
14
16
18
block index
64
128
256
512
1024
2048
4096
RegNetY-12GF
di = [2, 5, 11, 1]
wi = [224, 448, 896, 2240]
g = 112, b = 1, e = 19.5%
wa = 73, w0 = 168, wm = 2.4
0
2
4
6
8
10
12
14
16
block index
64
128
256
512
1024
2048
4096
RegNetY-16GF
di = [2, 4, 11, 1]
wi = [224, 448, 1232, 3024]
g = 112, b = 1, e = 19.5%
wa = 106, w0 = 200, wm = 2.5
0
2
4
6
8
10 12 14 16 18
block index
64
128
256
512
1024
2048
4096
RegNetY-32GF
di = [2, 5, 12, 1]
wi = [232, 696, 1392, 3712]
g = 232, b = 1, e = 19.0%
wa = 116, w0 = 232, wm = 2.5
ﬂops
params
acts
batch
infer
train
error
(B)
(M)
(M)
size
(ms)
(hr)
(top-1)
REGNETY-200MF
0.2
3.2
2.2
1024
11
3.1
29.6±0.11
REGNETY-400MF
0.4
4.3
3.9
1024
19
5.1
25.9±0.16
REGNETY-600MF
0.6
6.1
4.3
1024
19
5.2
24.5±0.07
REGNETY-800MF
0.8
6.3
5.2
1024
22
6.0
23.7±0.03
REGNETY-1.6GF
1.6
11.2
8.0
1024
39
10.1
22.0±0.08
REGNETY-3.2GF
3.2
19.4
11.3
512
67
16.5
21.0±0.05
REGNETY-4.0GF
4.0
20.6
12.3
512
68
16.8
20.6±0.08
REGNETY-6.4GF
6.4
30.6
16.4
512
104
26.1
20.1±0.04
REGNETY-8.0GF
8.0
39.2
18.0
512
113
28.1
20.1±0.09
REGNETY-12GF
12.1
51.8
21.4
512
150
36.0
19.7±0.06
REGNETY-16GF
15.9
83.6
23.0
512
189
45.6
19.6±0.16
REGNETY-32GF
32.3
145.0
30.3
256
319
76.0
19.0±0.12
Figure 16. Top REGNETY models (Y=X+SE). The benchmark-
ing setup and the ﬁgure format is the same as in Figure 15.
5.1. State-of-the-Art Comparison: Mobile Regime
Much of the recent work on network design has focused
on the mobile regime (∼600MF). In Table 2, we compare
REGNET models at 600MF to existing mobile networks.
We observe that REGNETS are surprisingly effective in this
regime considering the substantial body of work on ﬁnding
better mobile networks via both manual design [9, 25, 19]
and NAS [35, 23, 17, 18].
We emphasize that REGNET models use our basic 100
epoch schedule with no regularization except weight de-
cay, while most mobile networks use longer schedules
with various enhancements, such as deep supervision [16],
Cutout [4], DropPath [14], AutoAugment [2], and so on. As
such, we hope our strong results obtained with a short train-
ing schedule without enhancements can serve as a simple
baseline for future work.
ﬂops (B)
params (M)
top-1 error
MOBILENET [9]
0.57
4.2
29.4
MOBILENET-V2 [25]
0.59
6.9
25.3
SHUFFLENET [33]
0.52
-
26.3
SHUFFLENET-V2 [19]
0.59
-
25.1
NASNET-A [35]
0.56
5.3
26.0
AMOEBANET-C [23]
0.57
6.4
24.3
PNASNET-5 [17]
0.59
5.1
25.8
DARTS [18]
0.57
4.7
26.7
REGNETX-600MF
0.60
6.2
25.9±0.03
REGNETY-600MF
0.60
6.1
24.5±0.07
Table 2. Mobile regime. We compare existing models using orig-
inally reported errors to RegNet models trained in a basic setup.
Our simple RegNet models achieve surprisingly good results
given the effort focused on this regime in the past few years.
8

0.8
1.6
3.2
6.4
12.8
flops (B)
21
22
23
24
25
error
ResNet
ResNeXt
RegNetX
8
16
32
64
params (M)
ResNet
ResNeXt
RegNetX
8
16
32
activations (M)
ResNet
ResNeXt
RegNetX
Figure 17. ResNe(X)t comparisons. REGNETX models versus
RESNE(X)T-(50,101,152) under various complexity metrics. As
all models use the identical components and training settings, all
observed gains are from the design of the RegNetX design space.
ﬂops
params
acts
infer
train
top-1 error
(B)
(M)
(M)
(ms)
(hr)
ours±std [orig]
RESNET-50
4.1
22.6
11.1
53
12.2
23.2±0.09 [23.9]
REGNETX-3.2GF
3.2
15.3
11.4
57
14.3
21.7±0.08
RESNEXT-50
4.2
25.0
14.4
78
18.0
21.9±0.10 [22.2]
RESNET-101
7.8
44.6
16.2
90
20.4
21.4±0.11 [22.0]
REGNETX-6.4GF
6.5
26.2
16.4
92
23.5
20.8±0.07
RESNEXT-101
8.0
44.2
21.2
137
31.8
20.7±0.08 [21.2]
RESNET-152
11.5
60.2
22.6
130
29.2
20.9±0.12 [21.6]
REGNETX-12GF
12.1
46.1
21.4
137
32.9
20.3±0.04
(a) Comparisons grouped by activations.
RESNET-50
4.1
22.6
11.1
53
12.2
23.2±0.09 [23.9]
RESNEXT-50
4.2
25.0
14.4
78
18.0
21.9±0.10 [22.2]
REGNETX-4.0GF
4.0
22.1
12.2
69
17.1
21.4±0.19
RESNET-101
7.8
44.6
16.2
90
20.4
21.4±0.11 [22.0]
RESNEXT-101
8.0
44.2
21.2
137
31.8
20.7±0.08 [21.2]
REGNETX-8.0GF
8.0
39.6
14.1
94
22.6
20.7±0.07
RESNET-152
11.5
60.2
22.6
130
29.2
20.9±0.12 [21.6]
RESNEXT-152
11.7
60.0
29.7
197
45.7
20.4±0.06 [21.1]
REGNETX-12GF
12.1
46.1
21.4
137
32.9
20.3±0.04
(b) Comparisons grouped by ﬂops.
Table 3. RESNE(X)T comparisons. (a) Grouped by activations,
REGNETX show considerable gains (note that for each group GPU
inference and training times are similar). (b) REGNETX models
outperform RESNE(X)T models under ﬁxed ﬂops as well.
5.2. Standard Baselines Comparison: ResNe(X)t
Next, we compare REGNETX to standard RESNET [8] and
RESNEXT [31] models. All of the models in this experiment
come from the exact same design space, the former being
manually designed, the latter being obtained through design
space design. For fair comparisons, we compare REGNET
and RESNE(X)T models under the same training setup (our
standard REGNET training setup). We note that this results
in improved RESNE(X)T baselines and highlights the impor-
tance of carefully controlling the training setup.
Comparisons are shown in Figure 17 and Table 3. Over-
all, we see that REGNETX models, by optimizing the net-
work structure alone, provide considerable improvements
under all complexity metrics. We emphasize that good REG-
NET models are available across a wide range of compute
regimes, including in low-compute regimes where good
RESNE(X)T models are not available.
Table 3a shows comparisons grouped by activations
(which can strongly inﬂuence runtime on accelerators such
as GPUs). This setting is of particular interest to the re-
search community where model training time is a bottle-
neck and will likely have more real-world use cases in the
future, especially as accelerators gain more use at inference
time (e.g., in self-driving cars). REGNETX models are quite
effective given a ﬁxed inference or training time budget.
0.4
0.8
1.6
3.2
6.4
12.8
flops (B)
20
22
24
26
error
EfficientNet
RegNetX
RegNetY
4
8
16
32
64
params (M)
EfficientNet
RegNetX
RegNetY
4
8
16
32
64
128
activations (M)
EfficientNet
RegNetX
RegNetY
Figure 18. EFFICIENTNET comparisons. REGNETs outperform
the state of the art, especially when considering activations.
ﬂops
params
acts
batch
infer
train
top-1 error
(B)
(M)
(M)
size
(ms)
(hr)
ours±std [orig]
EFFICIENTNET-B0
0.4
5.3
6.7
256
34
11.7
24.9±0.03 [23.7]
REGNETY-400MF
0.4
4.3
3.9
1024
19
5.1
25.9±0.16
EFFICIENTNET-B1
0.7
7.8
10.9
256
52
15.6
24.1±0.16 [21.2]
REGNETY-600MF
0.6
6.1
4.3
1024
19
5.2
24.5±0.07
EFFICIENTNET-B2
1.0
9.2
13.8
256
68
18.4
23.4±0.06 [20.2]
REGNETY-800MF
0.8
6.3
5.2
1024
22
6.0
23.7±0.03
EFFICIENTNET-B3
1.8
12.0
23.8
256
114
32.1
22.5±0.05 [18.9]
REGNETY-1.6GF
1.6
11.2
8.0
1024
39
10.1
22.0±0.08
EFFICIENTNET-B4
4.2
19.0
48.5
128
240
65.1
21.2±0.06 [17.4]
REGNETY-4.0GF
4.0
20.6
12.3
512
68
16.8
20.6±0.08
EFFICIENTNET-B5
9.9
30.0
98.9
64
504
135.1
21.5±0.11 [16.7]
REGNETY-8.0GF
8.0
39.2
18.0
512
113
28.1
20.1±0.09
Table 4. EFFICIENTNET comparisons using our standard train-
ing schedule.
Under comparable training settings, REGNETY
outperforms EFFICIENTNET for most ﬂop regimes.
Moreover,
REGNET models are considerably faster, e.g., REGNETX-F8000
is about 5× faster than EFFICIENTNET-B5. Note that originally
reported errors for EFFICIENTNET (shown grayed out), are much
lower but use longer and enhanced training schedules, see Table 7.
5.3. State-of-the-Art Comparison: Full Regime
We focus our comparison on EFFICIENTNET [29], which
is representative of the state of the art and has reported im-
pressive gains using a combination of NAS and an interest-
ing model scaling rule across complexity regimes.
To enable direct comparisons, and to isolate gains due
to improvements solely of the network architecture, we opt
to reproduce the exact EFFICIENTNET models but using our
standard training setup, with a 100 epoch schedule and no
regularization except weight decay (effect of longer sched-
ule and stronger regularization are shown in Table 7). We
optimize only lr and wd, see Figure 22 in appendix. This is
the same setup as REGNET and enables fair comparisons.
Results are shown in Figure 18 and Table 4. At low ﬂops,
EFFICIENTNET outperforms the REGNETY. At intermediate
ﬂops, REGNETY outperforms EFFICIENTNET, and at higher
ﬂops both REGNETX and REGNETY perform better.
We also observe that for EFFICIENTNET, activations scale
linearly with ﬂops (due to the scaling of both resolution and
depth), compared to activations scaling with the square-root
of ﬂops for REGNETs. This leads to slow GPU training and
inference times for EFFICIENTNET. E.g., REGNETX-8000 is
5× faster than EFFICIENTNET-B5, while having lower error.
6. Conclusion
In this work, we present a new network design paradigm.
Our results suggest that designing network design spaces is
a promising avenue for future research.
9

ﬂops
params
acts
infer
train
error
(B)
(M)
(M)
(ms)
(hr)
(top-1)
RESNET-50
4.1
22.6
11.1
53
12.2
35.0±0.20
REGNETX-3.2GF
3.2
15.3
11.4
57
14.3
33.6±0.25
RESNEXT-50
4.2
25.0
14.4
78
18.0
33.5±0.10
RESNET-101
7.8
44.6
16.2
90
20.4
33.2±0.24
REGNETX-6.4GF
6.5
26.2
16.4
92
23.5
32.6±0.15
RESNEXT-101
8.0
44.2
21.2
137
31.8
32.1±0.30
RESNET-152
11.5
60.2
22.6
130
29.2
32.2±0.22
REGNETX-12GF
12.1
46.1
21.4
137
32.9
32.0±0.27
(a) Comparisons grouped by activations.
RESNET-50
4.1
22.6
11.1
53
12.2
35.0±0.20
RESNEXT-50
4.2
25.0
14.4
78
18.0
33.5±0.10
REGNETX-4.0GF
4.0
22.1
12.2
69
17.1
33.2±0.20
RESNET-101
7.8
44.6
16.2
90
20.4
33.2±0.24
RESNEXT-101
8.0
44.2
21.2
137
31.8
32.1±0.30
REGNETX-8.0GF
8.0
39.6
14.1
94
22.6
32.5±0.18
RESNET-152
11.5
60.2
22.6
130
29.2
32.2±0.22
RESNEXT-152
11.7
60.0
29.7
197
45.7
31.5±0.26
REGNETX-12GF
12.1
46.1
21.4
137
32.9
32.0±0.27
(b) Comparisons grouped by ﬂops.
Table 5. RESNE(X)T comparisons on ImageNetV2.
ﬂops
params
acts
batch
infer
train
error
(B)
(M)
(M)
size
(ms)
(hr)
(top-1)
EFFICIENTNET-B0
0.4
5.3
6.7
256
34
11.7
37.1±0.22
REGNETY-400MF
0.4
4.3
3.9
1024
19
5.1
38.3±0.26
EFFICIENTNET-B1
0.7
7.8
10.9
256
52
15.6
36.4±0.10
REGNETY-600MF
0.6
6.1
4.3
1024
19
5.2
36.9±0.17
EFFICIENTNET-B2
1.0
9.2
13.8
256
68
18.4
35.3±0.25
REGNETY-800MF
0.8
6.3
5.2
1024
22
6.0
35.7±0.40
EFFICIENTNET-B3
1.8
12.0
23.8
256
114
32.1
34.4±0.27
REGNETY-1.6GF
1.6
11.2
8.0
1024
39
10.1
33.9±0.19
EFFICIENTNET-B4
4.2
19.0
48.5
128
240
65.1
32.5±0.23
REGNETY-4.0GF
4.0
20.6
12.3
512
68
16.8
32.3±0.28
EFFICIENTNET-B5
9.9
30.0
98.9
64
504
135.1
31.5±0.17
REGNETY-8.0GF
8.0
39.2
18.0
512
113
28.1
31.3±0.08
Table 6. EFFICIENTNET comparisons on ImageNetV2.
Appendix A: Test Set Evaluation
In the main paper we perform all experiments on the Im-
ageNet [3] validation set. Here we evaluate our models on
the ImageNetV2 [24] test set (original test set unavailable).
Evaluation setup. To study generalization of models de-
veloped on ImageNet, the authors of [24] collect a new test
set following the original procedure (ImageNetV2). They
ﬁnd that the overall model ranks are preserved on the new
test set. The absolute errors, however, increase. We repeat
the comparisons from §5 on the ImageNetV2 test set.
RESNE(X)T comparisons.
We compare to RESNE(X)T
models in Table 5. We observe that while model ranks are
generally consistent, the gap between them decreases. Nev-
ertheless, REGNETX models still compare favorably, and
provide good models across ﬂop regimes, including in low-
compute regimes where good RESNE(X)T models are not
available. Best results can be achieved using REGNETY.
EFFICIENTNET comparisons. We compare to EFFICIENT-
NET models in Table 6. As before, we observe that the
model ranks are generally consistent but the gap decreases.
Overall, the results conﬁrm that the REGNET models per-
form comparably to state-of-the-art EFFICIENTNET while be-
ing up to 5× faster on GPUs.
0.2
0.4
0.8
1.6
3.2
6.4
flops (B)
20
22
24
26
28
30
error
12
d
28
d = 20
21.0
21.5
22.0
22.5
23.0
23.5
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[21.0|21.7] 3 stage
[20.7|21.0] 4 stage
20.8
21.0
21.2
21.4
21.6
error
[21.0|21.2] b = 1/2
[20.7|21.0] b = 1
Figure 19. Additional ablations.
Left: Fixed depth networks
(d = 20) are effective across ﬂop regimes. Middle: Three stage
networks perform poorly at high ﬂops. Right: Inverted bottleneck
(b < 1) is also ineffective at high ﬂops. See text for more context.
22
24
26
28
30
32
34
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[29.8|31.3] 400MF ReLU
[28.7|30.4] 400MF Swish
[22.1|22.7] 6.4GF ReLU
[22.6|23.0] 6.4GF Swish
0.2
0.4
0.8
1.6
3.2
6.4
flops (B)
22
24
26
28
30
32
34
36
error
ReLU g
1
Swish g
1
0.2
0.4
0.8
1.6
3.2
6.4
flops (B)
ReLU g = 1
Swish g = 1
Figure 20. Swish vs. ReLU. Left: RegNetY performs better with
Swish than ReLU at 400MF but worse at 6.4GF. Middle: Results
across wider ﬂop regimes show similar trends. Right: If, however,
g is restricted to be 1 (depthwise conv), Swish is much better.
Appendix B: Additional Ablations
In this section we perform additional ablations to further
support or supplement the results of the main text.
Fixed depth. In §5 we observed that the depths of our top
models are fairly stable (∼20 blocks). In Figure 19 (left) we
compare using ﬁxed depth (d = 20) across ﬂop regimes.
To compare to our best results, we trained each model for
100 epochs. Surprisingly, we ﬁnd that ﬁxed-depth networks
can match the performance of variable depth networks for
all ﬂop regimes, in both the average and best case. Indeed,
these ﬁxed depth networks match our best results in §5.
Fewer stages. In §5 we observed that the top REGNET mod-
els at high ﬂops have few blocks in the fourth stage (one or
two). Hence we tested 3 stage networks at 6.4GF, trained for
100 epochs each. In Figure 19 (middle), we show the results
and observe that the three stage networks perform consid-
erably worse. We note, however, that additional changes
(e.g., in the stem or head) may be necessary for three stage
networks to perform well (left for future work).
Inverted Bottleneck. In §4 we observed that using the in-
verted bottleneck (b < 1) degrades performance. Since our
results were in a low-compute regime, in Figure 19 (right)
we re-test at 6.4GF and 100 epochs. Surprisingly, we ﬁnd
that in this regime b < 1 degrades results further.
Swish vs. ReLU Many recent methods employ the
Swish [22] activation function, e.g. [29]. In Figure 20, we
study RegNetY with Swish and ReLU. We ﬁnd that Swish
outperforms ReLU at low ﬂops, but ReLU is better at high
ﬂops.
Interestingly, if g is restricted to be 1 (depthwise
conv), Swish performs much better than ReLU. This sug-
gests that depthwise conv and Swish interact favorably, al-
though the underlying reason is not at all clear.
10

3.0
2.5
2.0
1.5
1.0
0.5
log10(lr)
40
60
80
100
error
6.0
5.5
5.0
4.5
4.0
3.5
3.0
log10(wd)
40
60
80
100
error
8
7
6
5
4
log10(lr wd)
40
60
80
100
error
0.2
0.4
0.8
1.6
3.2
flops (B)
5
4
3
2
1
0
1
log10(lr)
all
good
best
lr = 0.1
0.2
0.4
0.8
1.6
3.2
flops (B)
8
7
6
5
4
3
2
log10(wd)
all
good
best
wd = 5 10
5
0.2
0.4
0.8
1.6
3.2
flops (B)
9
8
7
6
5
4
3
log10(lr wd)
all
good
best
wd lr = 5 10
6
5
10
50
epochs
5
4
3
2
1
0
1
log10(lr)
all
good
best
lr = 0.1
5
10
50
epochs
8
7
6
5
4
3
2
log10(wd)
all
good
best
wd = 5 10
5
5
10
50
epochs
9
8
7
6
5
4
3
log10(lr wd)
all
good
best
wd lr = 5 10
6
Figure 21. Optimization settings. For these results, we generate a
population of RegNetX models while also randomly varying the
initial learning rate (lr) and weight decay (wd) for each model.
These results use a batch size of 128 and are trained on 1 GPU.
Top: Distribution of model error versus lr, wd, and also lr·wd
(at 10 epochs and 400MF). Applying an empirical bootstrap, we
see that clear trends emerge, especially for lr and lr·wd. Middle:
We repeat this experiment but across various ﬂop regimes (trained
for 10 epochs each); the trends are stable. Bottom: Similarly, we
repeat the above while training for various number of epochs (in
the 400MF regime), and observe the same trends. Based on these
results, we use an lr = 0.1 and wd = 5·10−5 starting with §4
across all training schedules and ﬂop regimes.
B0
B1
B2
B3
B4
B5
model
5
4
3
2
1
0
1
log10(lr)
all
good
best
lr = 0.2
B0
B1
B2
B3
B4
B5
model
8
7
6
5
4
3
2
log10(wd)
all
good
best
wd = 1 10
5
B0
B1
B2
B3
B4
B5
model
9
8
7
6
5
4
3
log10(lr wd)
all
good
best
wd lr = 2 10
6
Figure 22. We repeat the sweep over lr and wd for EFFICIENT-
NET training each model for 25 epochs. The lr (with reference to
a batch size of 128) and wd is stable across regimes. We use these
values for all EFFICIENTNET experiments in the main text (adjust-
ing for batch size accordingly). See Figure 21 for comparison.
Appendix C: Optimization Settings
Our basic training settings follow [21] as discussed in §3.
To tune the learning rate lr and weight decay wd for REG-
NET models, we perform a study, described in Figure 21.
Based on this, we set lr = 0.1 and wd = 5·10−5 for all
models in §4 and §5. To enable faster training of our ﬁ-
nal models at 100 epochs, we increase the number of GPUs
to 8, while keeping the number of images per GPU ﬁxed.
When scaling the batch size, we adjust lr using the linear
scaling rule and apply 5 epoch gradual warmup [6].
To enable fair comparisons, we repeat the same op-
timization for EFFICIENTNET in Figure 22.
Interestingly,
learning rate and weight decay are again stable across com-
plexity regimes. Finally, in Table 7 we report the sizable
effect of training enhancement on EFFICIENTNET-B0. The
gap may be even larger for larger models (see Table 4).
ﬂops (B) params (M) acts (M) epochs enhance
error
EFFICIENTNET-B0
0.39
5.3
6.7
100
25.6
EFFICIENTNET-B0
0.39
5.3
6.7
250
25.0
EFFICIENTNET-B0
0.39
5.3
6.7
250
✓
24.4
EFFICIENTNET-B0 [29]
0.39
5.3
6.7
350
✓✓✓
23.7
Table 7. Training enhancements to EFFICIENTNET-B0.
Our
EFFICIENTNET-B0 reproduction with DropPath [14] and a 250
epoch training schedule (third row), achieves results slightly infe-
rior to original results (bottom row), which additionally used RM-
SProp [30], AutoAugment [2], etc. Without these enhancements
to the training setup results are ∼2% lower (top row), highlighting
the importance of carefully controlling the training setup.
wi, ri, ri
⨁
wi, ri, ri
wi/bi, ri, ri
wi/bi, ri, ri
1×1
3×3, gi
1×1
(a) X block
wi, ri, ri
⨁
wi, ri, ri
wi/bi, ri, ri
wi/bi, ri, ri
1×1
3×3
1×1
(b) R block
wi, ri, ri
3×3
(d) VR block
wi, ri, ri
⨁
wi, ri, ri
3×3
(c) V block
wi, ri, ri
40.0
42.5
45.0
47.5
50.0
52.5
error
0.0
0.2
0.4
0.6
0.8
1.0
cumulative prob.
[46.6|49.3] RegNetVR
[46.0|49.2] RegNetV
[41.9|44.3] RegNetR
[38.2|41.0] RegNetX
Figure 23. Block types used in our generalization experiments (see
§3.4 and Figure 10). Left: Block diagrams. Right: Comparison of
RegNet with the four block types. The X block performs best.
Interestingly, for the V block, residual connections give no gain.
Appendix D: Implementation Details
We conclude with additional implementation details.
Group width compatibility. When sampling widths w and
groups widths g for our models, we may end up with incom-
patible values (i.e. w not divisible by g). To address this, we
employ a simple strategy. Namely, we set g = w if g > w
and round w to be divisible by g otherwise. The ﬁnal w can
be at most 1/3 different from the original w (proof omitted).
For models with bottlenecks, we apply this strategy to the
bottleneck width instead (and adjust widths accordingly).
Group width ranges. As discussed in §4, we notice the
general trend that the group widths of good models are
larger in higher compute regimes. To account for this, we
gradually adjust the group width ranges for higher compute
regimes. For example, instead of sampling g ≤32, at 3.2GF
we use 16 ≤g ≤64 and allow any g divisible by 8.
Block types. In §3, we showed that the RegNet design
space generalizes to different block types.
We describe
these additional block types, shown in Figure 23, next:
1. R block: same as the X block except without groups,
2. V block: a basic block with only a single 3×3 conv,
3. VR block: same as V block plus residual connections.
We note that good parameter values may differ across block
types. E.g., in contrast to the X block, for the R block using
b > 1 is better than b = 1. Our approach is robust to this.
Y block details. To obtain the Y block, we add the SE op
after the 3×3 conv of the X block, and we use an SE reduc-
tion ratio of 1/4. We experimented with these choices but
found that they performed comparably (not shown).
11

References
[1] F. Chollet. Xception: Deep learning with depthwise separa-
ble convolutions. In CVPR, 2017. 7
[2] E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le.
AutoAugment: Learning augmentation policies from data.
arXiv:1805.09501, 2018. 8, 11
[3] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-
Fei. Imagenet: A large-scale hierarchical image database. In
CVPR, 2009. 2, 3, 8, 10
[4] T. DeVries and G. W. Taylor.
Improved regular-
ization of convolutional neural networks with cutout.
arXiv:1708.04552, 2017. 8
[5] B. Efron and R. J. Tibshirani. An introduction to the boot-
strap. CRC press, 1994. 3
[6] P.
Goyal,
P.
Doll´ar,
R.
Girshick,
P.
Noordhuis,
L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He.
Accurate, large minibatch sgd: Training imagenet in 1 hour.
arXiv:1706.02677, 2017. 11
[7] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
rectiﬁers: Surpassing human-level performance on imagenet
classiﬁcation. In ICCV, 2015. 2
[8] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
for image recognition. In CVPR, 2016. 1, 2, 3, 9
[9] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang,
T. Weyand, M. Andreetto, and H. Adam. Mobilenets: Efﬁ-
cient convolutional neural networks for mobile vision appli-
cations. arXiv:1704.04861, 2017. 2, 8
[10] J. Hu, L. Shen, and G. Sun.
Squeeze-and-excitation net-
works. In CVPR, 2018. 7
[11] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger.
Densely connected convolutional networks. In CVPR, 2017.
2
[12] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift. In
ICML, 2015. 4
[13] A. Krizhevsky, I. Sutskever, and G. E. Hinton.
Imagenet
classiﬁcation with deep convolutional neural networks. In
NIPS, 2012. 1, 2
[14] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet:
Ultra-deep neural networks without residuals.
In ICLR,
2017. 8, 11
[15] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E.
Howard, W. Hubbard, and L. D. Jackel. Backpropagation
applied to handwritten zip code recognition. Neural compu-
tation, 1989. 1
[16] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-
supervised nets. In AISTATS, 2015. 8
[17] C. Liu, B. Zoph, M. Neumann, J. Shlens, W. Hua, L.-J. Li,
L. Fei-Fei, A. Yuille, J. Huang, and K. Murphy. Progressive
neural architecture search. In ECCV, 2018. 2, 8
[18] H. Liu, K. Simonyan, and Y. Yang. Darts: Differentiable
architecture search. In ICLR, 2019. 1, 2, 8
[19] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun. Shufﬂenet v2:
Practical guidelines for efﬁcient cnn architecture design. In
ECCV, 2018. 8
[20] H. Pham, M. Y. Guan, B. Zoph, Q. V. Le, and J. Dean. Ef-
ﬁcient neural architecture search via parameter sharing. In
ICML, 2018. 2
[21] I. Radosavovic, J. Johnson, S. Xie, W.-Y. Lo, and P. Doll´ar.
On network design spaces for visual recognition. In ICCV,
2019. 1, 2, 3, 4, 11
[22] P. Ramachandran, B. Zoph, and Q. V. Le. Searching for ac-
tivation functions. arXiv:1710.05941, 2017. 10
[23] E. Real, A. Aggarwal, Y. Huang, and Q. V. Le. Regularized
evolution for image classiﬁer architecture search. In AAAI,
2019. 2, 8
[24] B. Recht, R. Roelofs, L. Schmidt, and V. Shankar. Do ima-
genet classiﬁers generalize to imagenet? arXiv:1902.10811,
2019. 2, 10
[25] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C.
Chen.
Mobilenetv2: Inverted residuals and linear bottle-
necks. In CVPR, 2018. 2, 7, 8
[26] K. Simonyan and A. Zisserman. Very deep convolutional
networks for large-scale image recognition. In ICLR, 2015.
1, 2
[27] C. Szegedy,
W. Liu,
Y. Jia,
P. Sermanet,
S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. In CVPR, 2015. 2
[28] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna.
Rethinking the inception architecture for computer vision. In
CVPR, 2016. 2
[29] M. Tan and Q. V. Le. Efﬁcientnet: Rethinking model scaling
for convolutional neural networks. ICML, 2019. 1, 2, 7, 9,
10, 11
[30] T. Tieleman and G. Hinton. Lecture 6.5-rmsprop: Divide
the gradient by a running average of its recent magnitude.
Coursera: Neural networks for machine learning, 2012. 11
[31] S. Xie, R. Girshick, P. Doll´ar, Z. Tu, and K. He. Aggregated
residual transformations for deep neural networks. In CVPR,
2017. 2, 4, 9
[32] S. Zagoruyko and N. Komodakis. Wide residual networks.
In BMVC, 2016. 2
[33] X. Zhang, X. Zhou, M. Lin, and J. Sun.
Shufﬂenet: An
extremely efﬁcient convolutional neural network for mobile
devices. In CVPR, 2018. 8
[34] B. Zoph and Q. V. Le. Neural architecture search with rein-
forcement learning. In ICLR, 2017. 1
[35] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le. Learning
transferable architectures for scalable image recognition. In
CVPR, 2018. 2, 8
12
