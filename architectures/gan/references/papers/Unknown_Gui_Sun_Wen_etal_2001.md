# Unknown Gui Sun Wen etal 2001

> Source: `Unknown_Gui_Sun_Wen_etal_2001.pdf`

---

                                         JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                       1




                                             A Review on Generative Adversarial Networks:
                                                 Algorithms, Theory, and Applications
                                                                        Jie Gui, Zhenan Sun, Yonggang Wen, Dacheng Tao, Jieping Ye

                                              Abstract—Generative adversarial networks (GANs) are a hot research topic recently. GANs have been widely studied since 2014, and
                                              a large number of algorithms have been proposed. However, there is few comprehensive study explaining the connections among
                                              different GANs variants, and how they have evolved. In this paper, we attempt to provide a review on various GANs methods from the
                                              perspectives of algorithms, theory, and applications. Firstly, the motivations, mathematical representations, and structure of most GANs
                                              algorithms are introduced in details. Furthermore, GANs have been combined with other machine learning algorithms for specific
                                              applications, such as semi-supervised learning, transfer learning, and reinforcement learning. This paper compares the commonalities




arXiv:2001.06937v1 [cs.LG] 20 Jan 2020
                                              and differences of these GANs methods. Secondly, theoretical issues related to GANs are investigated. Thirdly, typical applications of
                                              GANs in image processing and computer vision, natural language processing, music, speech and audio, medical field, and data
                                              science are illustrated. Finally, the future open research problems for GANs are pointed out.

                                              Index Terms—Deep Learning, Generative Adversarial Networks, Algorithm, Theory, Applications.

                                                                                                                       F

                                         1    I NTRODUCTION


                                         G     ENERATIVE adversarial networks (GANs) have become
                                               a hot research topic recently. Yann LeCun, a legend in
                                         deep learning, said in a Quora post “GANs are the most
                                                                                                                           relevant work is predictability minimization [2]. The connec-
                                                                                                                           tions between predictability minimization and GANs can be
                                                                                                                           found in [3], [4].
                                         interesting idea in the last 10 years in machine learning.”                           The popularity and importance of GANs have led to sev-
                                         There are a large number of papers related to GANs accord-                        eral previous reviews. The difference from previous work is
                                         ing to Google scholar. For example, there are about 11,800                        summarized in the following.
                                         papers related to GANs in 2018. That is to say, there are                             1)   GANs for specific applications: There are surveys of
                                         about 32 papers everyday and more than one paper every                                     using GANs for specific applications such as image
                                         hour related to GANs in 2018.                                                              synthesis and editing [5], audio enhancement and
                                             GANs consist of two models: a generator and a dis-                                     synthesis [6].
                                         criminator. These two models are typically implemented by                             2)   General survey: The earliest relevant review was
                                         neural networks, but they can be implemented with any                                      probably the paper by Wang et al. [7] which mainly
                                         form of differentiable system that maps data from one space                                introduced the progress of GANs before 2017. Ref-
                                         to the other. The generator tries to capture the distribution                              erences [8], [9] mainly introduced the progress of
                                         of true examples for new data example generation. The                                      GANs prior to 2018. The reference [10] introduced
                                         discriminator is usually a binary classifier, discriminating                               the architecture-variants and loss-variants of GANs
                                         generated examples from the true examples as accurately                                    only related to computer vision. Other related work
                                         as possible. The optimization of GANs is a minimax opti-                                   can be found in [11]–[13].
                                         mization problem. The optimization terminates at a saddle
                                         point that is a minimum with respect to the generator and                         As far as we know, this paper is the first to provide a
                                         a maximum with respect to the discriminator. That is, the                         comprehensive survey on GANs from the algorithm, theory,
                                         optimization goal is to reach Nash equilibrium [1]. Then,                         and application perspectives which introduces the latest
                                         the generator can be thought to have captured the real                            progress. Furthermore, our paper focuses on applications
                                         distribution of true examples.                                                    not only to image processing and computer vision, but also
                                             Some previous work has adopted the concept of making                          sequential data such as natural language processing, and
                                         two neural networks compete with each other. The most                             other related areas such as medical field.
                                                                                                                               The remainder of this paper is organized as follows: The
                                         J. Gui is with the Department of Computational Medicine and Bioinformatics,       related work is discussed in Section 2. Sections 3-5 introduce
                                         University of Michigan, USA (e-mail: guijie@ustc.edu).                            GANs from the algorithm, theory, and applications perspec-
                                         Z. Sun is with the Center for Research on Intelligent Perception and
                                         Computing, Chinese Academy of Sciences, Beijing 100190, China (e-mail:
                                                                                                                           tives, respectively. Tables 1 and 2 show GANs’ algorithms
                                         znsun@nlpr.ia.ac.cn).                                                             and applications which will be discussed in Sections 3 and
                                         Y. Wen is with the School of Computer Science and Engineering, Nanyang            5, respectively. The open research problems are discussed in
                                         Technological University, Singapore 639798 (e-mail: ygwen@ntu.edu.sg).            Section 6 and Section 7 concludes the survey.
                                         D. Tao is with the UBTECH Sydney Artificial Intelligence Center and
                                         the School of Information Technologies, the Faculty of Engineering and
                                         Information Technologies, the University of Sydney, Australia. (e-mail:           2   R ELATED WORK
                                         Dacheng.Tao@Sydney.edu.au).
                                         J. Ye is with DiDi AI Labs, P.R. China and University of Michigan, Ann            GANs belong to generative algorithms. Generative algo-
                                         Arbor.(e-mail:jpye@umich.edu).                                                    rithms and discriminative algorithms are two categories of
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                    2

                                   TABLE 1: A overview of GANs’ algorithms discussed in Section 3

          GANs’ Representative variants                 InfoGAN [14], cGANs [15], CycleGAN [16], f -GAN [17], WGAN [18], WGAN-GP [19],
                                                                                            LS-GAN [20]
                             Objective function         LSGANs [21], [22], hinge loss based GAN [23]–[25], MDGAN [26], unrolled GAN [27],
                                                                                     SN-GANs [23], RGANs [28]
                                   Skills                                        ImprovedGANs [29], AC-GAN [30]
                                                      LAPGAN [31], DCGANs [32], PGGAN [33], StackedGAN [34], SAGAN [35], BigGANs [36],
   GANs training                                                 StyleGAN [37], hybrids of autoencoders and GANs (EBGAN [38],
                                 Structure                                 BEGAN [39], BiGAN [40]/ALI [41], AGE [42]),
                                                                      multi-discriminator learning (D2GAN [43], GMAN [44]),
                                                                      multi-generator learning (MGAN [45], MAD-GAN [46]),
                                                                                 multi-GAN learning (CoGAN [47])
                       Semi-supervised learning          CatGANs [48], feature matching GANs [29], VAT [49], ∆-GAN [50], Triple-GAN [51]
                                                             DANN [52], CycleGAN [53], DiscoGAN [54], DualGAN [55], StarGAN [56],
 Task driven GANs            Transfer learning                               CyCADA [57], ADDA [58], [59], FCAN [60],
                                                                    unsupervised pixel-level domain adaptation (PixelDA) [61]
                           Reinforcement learning                                            GAIL [62]


                                            TABLE 2: Applications of GANs discussed in Section 5

                   Field                                  Subfield                                          Method
                                                       Super-resolution             SRGAN [63], ESRGAN [64], Cycle-in-Cycle GANs [65],
                                                                                                   SRDGAN [66], TGAN [67]
                                                                                      DR-GAN [68], TP-GAN [69], PG2 [70], PSGAN [71],
                                               Image synthesis and manipulation                APDrawingGAN [72], IGAN [73],
  Image processing and computer vision                                               introspective adversarial networks [74], GauGAN [75]
                                                       Texture synthesis                     MGAN [76], SGAN [77], PSGAN [78]
                                                       Object detection                  Segan [79], perceptual GAN [80], MTGAN [81]
                                                            Video                  VGAN [82], DRNET [83], Pose-GAN [84], video2video [85],
                                                                                                         MoCoGan [86]
                                               Natural language processing (NLP)        RankGAN [87], IRGAN [88], [89], TAC-GAN [90]
             Sequential data                                 Music                       RNN-GAN (C-RNN-GAN) [91], ORGAN [92],
                                                                                                       SeqGAN [93], [94]



machine learning algorithms. If a machine learning algo-                   It may fail to represent the complexity of true data distribu-
rithm is based on a fully probabilistic model of the observed              tion and learn the high-dimensional data distributions [100].
data, this algorithm is generative. Generative algorithms
have become more popular and important due to their wide                   2.1.2   Implicit density model
practical applications.                                                    An implicit density model does not directly estimate or
                                                                           fit the data distribution. It produces data instances from
2.1     Generative algorithms                                              the distribution without an explicit hypothesis [101] and
                                                                           utilizes the produced examples to modify the model. Prior
Generative algorithms can be classified into two classes:
                                                                           to GANs, the implicit density model generally needs to be
explicit density model and implicit density model.
                                                                           trained utilizing either ancestral sampling [102] or Markov
                                                                           chain-based sampling, which is inefficient and limits their
2.1.1    Explicit density model                                            practical applications. GANs belong to the directed implicit
An explicit density model assumes the distribution and                     density model category. A detailed summary and relevant
utilizes true data to train the model containing the distri-               papers can be found in [103].
bution or fit the distribution parameters. When finished,
new examples are produced utilizing the learned model or                   2.1.3 The comparison between GANs and other generative
distribution. The explicit density models include maximum                  algorithms
likelihood estimation (MLE), approximate inference [95],                   GANs were proposed to overcome the disadvantages of
[96], and Markov chain method [97]–[99]. These explicit                    other generative algorithms. The basic idea behind adver-
density models have an explicit distribution, but have lim-                sarial learning is that the generator tries to create as real-
itations. For instance, MLE is conducted on true data and                  istic examples as possible to deceive the discriminator. The
the parameters are updated directly based on the true data,                discriminator tries to distinguish fake examples from true
which leads to an overly smooth generative model. The gen-                 examples. Both the generator and discriminator improve
erative model learned by approximate inference can only                    through adversarial learning. This adversarial process gives
approach the lower bound of the objective function rather                  GANs notable advantages over other generative algorithms.
than directly approach the objective function, because of                  More specifically, GANs have advantages over other gener-
the difficulty in solving the objective function. The Markov               ative algorithms as follows:
chain algorithm can be used to train generative models, but
it is computationally expensive. Furthermore, the explicit                    1)   GANs can parallelize the generation, which is im-
density model has the problem of computational tractability.                       possible for other generative algorithms such as
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                  3

             PixelCNN [104] and fully visible belief networks             3.1.1.1 Original minimax game:
             (FVBNs) [105], [106].                                  The objective function of GANs [3] is
      2)     The generator design has few restrictions.
                                                                         min max V (D, G) = Ex∼pdata (x) [log D (x)]
      3)     GANs are subjectively thought to produce better              G    D                                                          (1)
             examples than other methods.                                  +Ez∼pz (z) [log (1 − D (G (z)))] .
                                                                                                                                      T
Refer to [103] for more detailed discussions about this.            log D (x) is the cross-entropy between [1 0]           and
                                                                                       T
                                                                    [D (x) 1 − D (x)] .       Similarly,    log (1 − D (G (z)))
                                                                                                                     T
2.2        Adversarial idea                                         is    the    cross-entropy      between     [0 1]      and
                                                                                               T
                                                                    [D (G (z)) 1 − D (G (z))] . For fixed G, the optimal
The adversarial idea has been successfully applied to many
                                                                    discriminator D is given by [3]:
areas such as machine learning, artificial intelligence, com-
puter vision and natural language processing. The recent                            ∗                pdata (x)
                                                                                   DG (x) =                         .                     (2)
event that AlphaGo [107] defeats world’s top human player                                        pdata (x) + pg (x)
engages public interest in artificial intelligence. The interme-    The minmax game in (1) can be reformulated as:
diate version of AlphaGo utilizes two networks competing
with each other.                                                      C(G) = max V (D, G)
                                                                                D
    Adversarial examples [108]–[117] have the adversarial                               ∗
                                                                      = Ex∼pdata [log DG  (x)]
idea, too. Adversarial examples are those examples which                                       ∗
                                                                         +Ez∼pz [log (1 − DG      (G (z)))]
are very different from the real examples, but are classified                           ∗                          ∗                      (3)
                                                                      = Ex∼pdata h[log DG (x)] + Ex∼pg i[log (1 − DG (x))]
into a real category very confidently, or those that are                                   pdata (x)
                                                                      = Ex∼pdata log 1 (p (x)+p (x))
slightly different than the real examples, but are classified                       h 2 data         g i
                                                                                          pg (x)
into a wrong category. This is a very hot research topic                 +Ex∼pg 1 (p (x)+p (x)) − 2 log 2.
                                                                                      2   data       g
recently [112], [113]. To be against adversarial attacks [118],
[119], references [120], [121] utilize GANs to conduct the          The definition of KullbackLeibler (KL) divergence and
right defense.                                                      Jensen-Shannon (JS) divergence between two probabilistic
    Adversarial machine learning [122] is a minimax prob-           distributions p (x) and q (x) are defined as
                                                                                             Z
lem. The defender, who builds the classifier that we want                                                p (x)
                                                                               KL( pk q) = p (x) log           dx,       (4)
to work correctly, is searching over the parameter space to                                              q (x)
find the parameters that reduce the cost of the classifier as
much as possible. Simultaneously, the attacker is searching                           1        p+q    1       p+q
over the inputs of the model to maximize the cost.                     JS( pk q) =      KL( pk     ) + KL( qk     ).                      (5)
                                                                                      2         2     2        2
    The adversarial idea exists in adversarial networks, ad-        Therefore, (3) is equal to
versarial learning, and adversarial examples. However, they
                                                                                              p     +p                  p       +pg
have different objectives.                                            C(G) = KL( pdata k data2 g ) + KL( pg k data2                   )
                                                                           −2 log 2                                                       (6)
                                                                      = 2JS( pdata k pg ) − 2 log 2.
3     A LGORITHMS
                                                                    Thus, the objective function of GANs is related to both KL
In this section, we first introduce the original GANs. Then,
                                                                    divergence and JS divergence.
the representative variants, training, evaluation of GANs,
                                                                            3.1.1.2 Non-saturating game:
and task-driven GANs are introduced.
                                                                    It is possible that the Equation (1) cannot provide sufficient
                                                                    gradient for G to learn well in practice. Generally speak-
3.1        Generative Adversarial Nets (GANs)                       ing, G is poor in early learning and samples are clearly
The GANs framework is straightforward to implement                  different from the training data. Therefore, D can reject the
when the models are both neural networks. In order to learn         generated samples with high confidence. In this situation,
the generator’s distribution pg over data x, a prior on input       log (1 − D (G (z))) saturates. We can train G to maximize
noise variables is defined as pz (z) [3] and z is the noise vari-   log (D (G (z))) rather than minimize log (1 − D (G (z))).
able. Then, GANs represent a mapping from noise space to            The cost for the generator then becomes
data space as G (z, θg ), where G is a differentiable function                J (G) = Ez∼pz (z) [− log (D (G (z)))]
represented by a neural network with parameters θg . Other                                                                                (7)
                                                                              = Ex∼pg [− log (D (x))] .
than G, the other neural network D (x, θd ) is also defined
with parameters θd and the output of D (x) is a single scalar.      This new objective function results in the same fixed point
D (x) denotes the probability that x was from the data              of the dynamics of D and G but provides much larger
rather than the generator G. The discriminator D is trained         gradients early in learning. The non-saturating game is
to maximize the probability of giving the correct label to          heuristic, not being motivated by theory. However, the
both training data and fake samples generated from the              non-saturating game has other problems such as unstable
                                                                                                                         ∗
generator G. G is trained to minimize log (1 − D (G (z)))           numerical gradient for training G. With optimal DG     , we
simultaneously .                                                    have
                                                                                      ∗                           ∗
                                                                      Ex∼pg [−hlog (DG  (x))] +          h (1 − DG (x))]
                                                                                              i Ex∼pg [log
                                                                                     (1−D ∗ (x))
                                                                                                                     i
3.1.1 Objective function                                                                                           g    p (x)
                                                                      = Ex∼pg log D∗ G(x)            = Ex∼pg log pdata (x)                (8)
                                                                                        G
Different objective functions can be used in GANs.                    = KL( pg k pdata ).
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                                4

                          ∗
Therefore, Ex∼pg [− log (DG (x))] is equal to                        6
                                                                                                                               Original
                       ∗
       Ex∼pg [− log (DG   (x))]                                                                                                Non-saturating
                                             ∗               (9)
       = KL( pg k pdata ) − Ex∼pg [log (1 − DG (x))] .                                                                         Maximum likelihood cost
                                                                     0
From (3) and (6), we have
                   ∗                         ∗
   Ex∼pdata [log DG  (x)] + Ex∼pg [log (1 − DG (x))]
                                                            (10)
   = 2JS( pdata k pg ) − 2 log 2.                                    -5
                                                                                               High gradient              Low gradient
                            ∗
Therefore, Ex∼pg [log (1 − DG (x))] equals
                    ∗                                               -10
 Ex∼pg [log (1 − DG   (x))]                                                   When samples are
                                                ∗           (11)
 = 2JS( pdata k pg ) − 2 log 2 − Ex∼pdata [log DG (x)] .                      possible to be fake, we
                                                                              want to learn it to                          Gradient is
By substituting (11) into (9), (9) reduces to                       -15       improve the generator.                       dominated by the
                           ∗
                                                                              However, gradient in                         region where
           Ex∼pg [− log (DG   (x))]                                           this region is almost                        samples in very
           = KL( pg k pdata ) − 2JS( pdata k pg )+          (12)              vanishing                                    good quality.
                          ∗                                         -20
           Ex∼pdata [log DG  (x)] + 2 log 2.                              0      0.1    0.2    0.3      0.4    0.5       0.6    0.7     0.8    0.9       1
From (12), we can see that the optimization of the alternative                                                 (     )
G loss in the non-saturating game is contradictory because
the first term aims to make the divergence between the             Fig. 1: The three curves of “Original”, “Non-saturating”, and
generated distribution and the real distribution as small as       “Maximum likelihood cost” denotes log (1 − D (G (z))),
possible while the second term aims to make the divergence         − log (D (G (z))), and −D(G(z))/(1 − D(G(z))) in (1), (7),
between these two distributions as large as possible due           and (13), respectively. The cost that the generator has for
to the negative sign. This will bring unstable numerical           generating a sample G(z) is only decided by the discrimi-
gradient for training G. Furthermore, KL divergence is not a       nator’s response to that generated sample. The larger prob-
symmetrical quantity, which is reflected from the following        ability the discriminator gives the real label to the generated
two examples                                                       sample, the less cost the generator gets. This figure is repro-
                                                                   duced from [103], [123].
   •     If pdata (x) → 0 and pg (x) → 1, we have
         KL( pg k pdata ) → +∞.
   •     If pdata (x) → 1 and pg (x) → 0, we have                              from gradient vanishing. The heuristically motivated
         KL( pg k pdata ) → 0.                                                 non-saturating game does not have this problem.
The penalizations for two errors made by G are completely                •     Second, maximum likelihood game also has the
different. The first error is that G produces implausible sam-                 problem that almost all of the gradient is from the
ples and the penalization is rather large. The second error is                 right end of the curve, which means that a rather
that G does not produce real samples and the penalization is                   small number of samples in each minibatch dominate
quite small. The first error is that the generated samples are                 the gradient computation. This demonstrates that
inaccurate while the second error is that generated samples                    variance reduction methods could be an important
are not diverse enough. Based on this, G prefers producing                     research direction for improving the performance of
repeated but safe samples rather than taking risk to produce                   GANs based on maximum likelihood game.
different but unsafe samples, which has the mode collapse                •     Third, the heuristically motivated non-saturating
problem.                                                                       game has lower sample variance, which is the pos-
        3.1.1.3 Maximum likelihood game:                                       sible reason that it is more successful in real applica-
There are many methods to approximate (1) in GANs.                             tions.
Under the assumption that the discriminator is optimal,
                                                                   GAN Lab [124] is proposed as the interactive visualization
minimizing
                                                                   tool designed for non-experts to learn and experiment with
       J (G) = Ez∼pz (z) − exp σ −1 (D (G (z)))                    GANs. Bau et al. [125] present an analytic framework to
                                                 
                                                           (13)    visualize and understand GANs.
       = Ez∼pz (z) [−D (G (z))/(1 − D (G (z)))] ,
where σ is the logistic sigmoid function, equals minimiz-
ing (1) [123]. The demonstration of this equivalence can           3.2        GANs’ representative variants
be found in Section 8.3 of [103]. Furthermore, there are           There are many papers related to GANs [126]–[131] such as
other possible ways of approximating maximum likelihood            CSGAN [132] and LOGAN [133]. In this subsection, we will
within the GANs framework [17]. A comparison of original           introduce GANs’ representative variants.
zero-sum game, non-saturating game, and maximum likeli-
hood game is shown in Fig. 1.                                      3.2.1       InfoGAN
   Three observations can be obtained from Fig. 1.
                                                                   Rather than utilizing a single unstructured noise vector z ,
   •     First, when the sample is possible to be fake, that is    InfoGAN [14] proposes to decompose the input noise vector
         on the left end of the figure, both the maximum like-     into two parts: z , which is seen as incompressible noise; c,
         lihood game and the original minimax game suffer          which is called the latent code and will target the significant
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                       5

structured semantic features of the real data distribution.            y                    G(y)                    x
InfoGAN [14] aims to solve
                                                                                  G
   min max VI (D, G) = V (D, G) − λI(c; G(z, c)),           (14)
    G        D                                                                                           D              D
                                                                                                             fake             real
where V (D, G) is the objective function of original GAN,
G(z, c) is the generated sample, I is the mutual information,
and λ is the tunable regularization parameter. Maximizing
I(c; G(z, c)) means maximizing the mutual information be-
tween c and G(z, c) to make c contain as much important                                     y                       y
and meaningful features of the real samples as possible.
However, I(c; G(z, c)) is difficult to optimize directly in        Fig. 2: Illustration of pix2pix: Training a conditional GANs
practice since it requires access to the posterior P (c|x).        to map grayscale→color. The discriminator D learns to
Fortunately, we can have a lower bound of I(c; G(z, c))            classify between real grayscale, color tuples and fake (syn-
by defining an auxiliary distribution Q(c|x) to approximate        thesized by the generator). The generator G learns to fool
P (c|x). The final objective function of InfoGAN [14] is           the discriminator. Different from the original GANs, both
                                                                   the generator and discriminator observe the input grayscale
     min max VI (D, G) = V (D, G) − λLI (c; Q),             (15)
        G        D                                                 image and there is no noise input for the generator of
                                                                   pix2pix.
where LI (c; Q) is the lower bound of I(c; G(z, c)). InfoGAN
has several variants such as causal InfoGAN [134] and semi-
supervised InfoGAN (ss-InfoGAN) [135].                             correct class label, LC . LS is equivalent to L in (17). LC is
                                                                   defined as
3.2.2       Conditional GANs (cGANs)                                           LC = E [log P ( C = c| Xreal )]
                                                                                                                             (18)
GANs can be extended to a conditional model if both the                         +E [log (P ( C = c| Xf ake ))] .
discriminator and generator are conditioned on some extra          The discriminator and generator of AC-GAN is to maximize
information y . The objective function of conditional GANs         LC + LS and LC − LS , respectively. AC-GAN was the
[15] is:                                                           first variant of GANs that was able to produce recognizable
    min max V (D, G) = Ex∼pdata (x) [log D ( x| y)]                examples of all the ImageNet [151] classes.
        G     D                                             (16)       Discriminators of most cGANs based methods [31], [41],
                 +Ez∼pz (z) [log (1 − D (G ( z| y)))] .            [152]–[154] feed conditional information y into the discrimi-
By comparing (15) and (16), we can see that the generator          nator by simply concatenating (embedded) y to the input or
of InfoGAN is similar to that of cGANs. However, the latent        to the feature vector at some middle layer. cGANs with pro-
code c of InfoGAN is not known, and it is discovered by            jection discriminator [155] adopts an inner product between
training. Furthermore, InfoGAN has an additional network           the condition vector y and the feature vector.
Q to output the conditional variables Q(c|x).                          Isola et al. [156] used cGANs and sparse regularization
    Based on cGANs, we can generate samples conditioning           for image-to-image translation. The corresponding software
on class labels [30], [136], text [34], [137], [138], bounding     is called pix2pix. In GANs, the generator learns a mapping
box and keypoints [139]. In [34], [140], text to photo-realistic   from random noise z to G (z). In contrast, there is no noise
image synthesis is conducted with stacked generative ad-           input in the generator of pix2pix. A novelty of pix2pix is
versarial networks (SGAN) [141]. cGANs have been used              that the generator of pix2pix learns a mapping from an
for convolutional face generation [142], face aging [143],         observed image y to output image G (y), for example, from
image translation [144], synthesizing outdoor images having        a grayscale image to a color image. The objective of cGANs
specific scenery attributes [145], natural image description       in [156] can be expressed as
[146], and 3D-aware scene manipulation [147]. Chrysos et                     LcGAN s (D, G) = Ex,y [log D (x, y)]
al. [148] proposed robust cGANs. Thekumparampil et al.                                                                       (19)
                                                                               +Ey [log (1 − D (y, G (y)))] .
[149] discussed the robustness of conditional GANs to noisy
labels. Conditional CycleGAN [16] uses cGANs with cyclic              Furthermore, l1 distance is used:
consistency. Mode seeking GANs (MSGANs) [150] proposes                            Ll1 (G) = Ex,y [kx − G(y)k1 ] .            (20)
a simple yet effective regularization term to address the
mode collapse issue for cGANs.                                     The final objective of [156] is
    The discriminator of original GANs [3] is trained to                              LcGAN s (D, G) + λLl1 (G) ,            (21)
maximize the log-likelihood that it assigns to the correct
source [30]:                                                       where λ is the free parameter. As a follow-up to pix2pix,
                                                                   pix2pixHD [157] used cGANs and feature matching loss for
              L = E [log P ( S = real| Xreal )]                    high-resolution image synthesis and semantic manipulation.
                                                            (17)
                +E [log (P ( S = f ake| Xf ake ))] ,               With the discriminators, the learning problem is a multi-task
                                                                   learning problem:
which is equal to (1). The objective function of the auxiliary
                                                                                             X
classifier GAN (AC-GAN) [30] has two parts: the loglikeli-                  min max                LGAN (G, Dk ).           (22)
hood of the correct source, LS , and the loglikelihood of the                 G       D1 ,D2 ,D3
                                                                                                   k=1,2,3
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                    6

The training set is given as a set of pairs of corresponding             quantitatively evaluated GANs with divergences proposed
images {(si , xi )}, where xi is a natural photo and si is a             for training. Uehara et al. [162] extend the f -GAN further,
corresponding semantic label map. The ith-layer feature ex-              where the f -divergence is directly minimized in the gen-
                                             (i)
tractor of discriminator Dk is denoted as Dk (from input to              erator step and the ratio of the distributions of real and
the ith layer of Dk ). The feature matching loss LF M (G, Dk )           generated data are predicted in the discriminator step.
is:
     LF M (G, Dk ) =                                                     3.2.5    Integral Probability Metrics (IPMs)
            T    h
                     (i)         (i)
                                                          i
                                                                  (23)   Denoting P the set of all Borel probability measures on a
               1
            P
     E(s,x)   Ni    Dk (s, x) − Dk (s, G (s))                 ,          topological space (M, A). The integral probability metric
            i=1                                       1
                                                                         (IPM) [163] between two probability distributions P ∈ P
where Ni is the number of elements in each layer and                     and Q ∈ P is defined as
T denotes the total number of layers. The final objective                                        Z          Z
function of [157] is                                                             γF (P, Q) = sup     f dP −     f dQ ,         (26)
                X                                                                                f ∈F     M              M
min max              (LGAN (G, Dk ) + λLF M (G, Dk )). (24)
 G   D1 ,D2 ,D3
                  k=1,2,3                                                where F is a class of real-valued bounded measurable
                                                                         functions on M . Nonparametric density estimation and
3.2.3 CycleGAN                                                           convergence rates for GANs under Besov IPM Losses is
Image-to-image translation is a class of graphics and vision             discussed in [164]. IPMs include such as RKHS-induced
problems where the goal is to learn the mapping between                  maximum mean discrepancy (MMD) as well as the Wasser-
an output image and an input image using a training                      stein distance used in Wasserstein GANs (WGAN).
set of aligned image pairs. When paired training data is                        3.2.5.1 Maximum Mean Discrepancy (MMD):
available, reference [156] can be used for these image-to-               The maximum mean discrepancy (MMD) [165] is a measure
image translation tasks. However, reference [156] can not be             of the difference between two distributions P and Q given
used for unpaired data (no input/output pairs), which was                by the supremum over a function space F of differences
well solved by Cycle-consistent GANs (CycleGAN) [53].                    between the expectations with regard to two distributions.
CycleGAN is an important progress for unpaired data. It                  The MMD is defined by:
is proved that cycle-consistency is an upper bound of the                         M M D(F, P, Q) =
conditional entropy [158]. CycleGAN can be derived as a                             sup (EX∼P [f (X)] − EY ∼Q [f (Y )]) .                 (27)
special case within the proposed variational inference (VI)                         f ∈F
framework [159], naturally establishing its relationship with
                                                                         MMD has been used for deep generative models [166]–[168]
approximate Bayesian inference methods.
                                                                         and model criticism [169].
    The basic idea of DiscoGAN [54] and CycleGAN [53]
                                                                                3.2.5.2 Wasserstein GAN (WGAN):
is nearly the same. Both of them were proposed separately
                                                                         WGAN [18] conducted a comprehensive theoretical analysis
nearly at the same time. The only difference between Cy-
                                                                         of how the Earth Mover (EM) distance behaves in com-
cleGAN [53] and DualGAN [55] is that DualGAN uses the
                                                                         parison with popular probability distances and divergences
loss format advocated by Wasserstein GAN (WGAN) rather
                                                                         such as the total variation (TV) distance, the Kullback-
than the sigmoid cross-entropy loss used in CycleGAN.
                                                                         Leibler (KL) divergence, and the Jensen-Shannon (JS) diver-
                                                                         gence utilized in the context of learning distributions. The
3.2.4 f -GAN                                                             definition of the EM distance is
As we know, Kullback-Leibler (KL) divergence measures the
difference between two given probability distributions. A                    W (pdata , pg ) =          inf        E(x,y)∈γ [kx − yk] ,   (28)
                                                                                                 γ∈Π(pdata ,pg )
large class of assorted divergences are the so called Ali-
Silvey distances, also known as the f -divergences [160].                where Π (pdata , pg ) denotes the set of all joint distributions
Given two probability distributions P and Q which have,                  γ (x, y) whose marginals are pdata and pg , respectively.
respectively, an absolutely continuous density function p                However, the infimum in (28) is highly intractable. The
and q with regard to a base measure dx defined on the                    reference [18] uses the following equation to approximate
domain X , the f -divergence is defined,                                 the EM distance
                        Z                
                                    p (x)                                   max Ex∼pdata (x) [fw (x)] − Ez∼pz (z) [fw (G (z))] ,          (29)
          Df (P kQ ) = q (x)f               dx.       (25)                  w∈W
                                    q (x)
                            X                                            where there is a parameterized family of functions
Different choices of f recover popular divergences as special            {fw }w∈W that are all K -Lipschitz for some K and fw can
cases of f -divergence. For example, if f (a) = a log a, f -             be realized by the discriminator D. When D is optimized,
divergence becomes KL divergence. The original GANs                      (29) denotes the approximated EM distance. Then the aim
[3] is a special case of f -GAN [17] which is based on f -               of G is to minimize (29) to make the generated distribution
divergence. The reference [17] shows that any f -divergence              as close to the real distribution as possible. Therefore, the
can be used for training GAN. Furthermore, the reference                 overall objective function of WGAN is
[17] discusses the advantages of different choices of di-
                                                                          min max Ex∼pdata (x) [fw (x)] − Ez∼pz (z) [fw (G (z))]
vergence functions on both the quality of the produced                     G w∈W
                                                                                                                                           (30)
generative models and training complexity. Im et al. [161]                = min max Ex∼pdata (x) [D (x)] − Ez∼pz (z) [D (G (z))] .
                                                                              G     D
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                      7

By comparing (1) and (30), we can see three differences
between the objective function of original GANs and that
of WGAN:                                                                          min LG = Ez∼pz (z) [Lθ (G (z))] ,         (34)
                                                                                   G
   •    First, there is no log in the objective function of
                                                                              +
        WGAN.                                                      where [y] = max(0, y), λ is the free tuning-parameter, and
   •    Second, the D in original GANs is utilized as a            θ is the paramter of the discriminator D.
        binary classifier while D utilized in WGAN is to
        approximate the Wasserstein distance, which is a
        regression task. Therefore, the sigmoid in the last        3.2.7    Summary
        layer of D is not used in the WGAN. The output             There is a website called “The GAN Zoo” (https:
        of the discriminator of the original GANs is between       //github.com/hindupuravinash/the-gan-zoo) which lists
        zero and one while there is no constraint for that of      many GANs’ variants. Please refer to this website for more
        WGAN.                                                      details.
   •    Third, the D in WGAN is required to be K -Lipschitz
        for some K and therefore WGAN uses weight clip-
        ping.
                                                                   3.3     GANs Training
Compared with traditional GANs training, WGAN can
improve the stability of learning and provide meaningful           Despite the theoretical existence of unique solutions, GANs
learning curves useful for hyperparameter searches and             training is hard and often unstable for several reasons [29],
debugging. However, it is a challenging task to approxi-           [32], [179]. One difficulty is from the fact that optimal
mate the K -Lipschitz constraint which is required by the          weights for GANs correspond to saddle points, and not
Wasserstein-1 metric. WGAN-GP [19] is proposed by utiliz-          minimizers, of the loss function.
ing gradient penalty for restricting K -Lipschitz constraint           There are many papers on GANs training. Yadav et al.
and the objective function is                                      [180] stabilized GANs with prediction methods. By using
                                                                   independent learning rates, [181] proposed a two time-
         L = −Ex∼pdata
                    h [D (x)] + Ex̃∼pg [D  i (x̃)]                 scale update rule (TTUR) for both discriminator and gen-
                                         2                  (31)
           +λEx̂∼px̂ (k∇x̂ D (x̂)k2 − 1)                           erator to ensure that the model can converge to a stable
                                                                   local Nash equilibrium. Arjovsky [179] made theoretical
where the first two terms are the objective function of            steps towards fully understanding the training dynamics of
WGAN and x̂ is sampled from the distribution px̂ which             GANs; analyzed why GANs was hard to train; studied and
samples uniformly along straight lines between pairs of            proved rigorously the problems including saturation and
points sampled from the real data distribution pdata and           instability that occurred when training GANs; examined a
the generated distribution pg . There are some other methods       practical and theoretically grounded direction to mitigate
closely related to WGAN-GP such as DRAGAN [170]. Wu et             these problems; and introduced new tools to study them.
al. [171] propose a novel and relaxed version of Wasserstein-      Liang et al. [182] think that GANs training is a continual
1 metric: Wasserstein divergence (W-div), which does not           learning problem [183].
require the K -Lipschitz constraint. Based on W-div, Wu et al.         One method to improve GANs training is to assess
[171] introduce a Wasserstein divergence objective for GANs        the empirical “symptoms” that might occur in training.
(WGAN-div), which can faithfully approximate W-div by              These symptoms include: the generative model collapsing
optimization. CramerGAN [172] argues that the Wasserstein          to produce very similar samples for diverse inputs [29];
distance leads to biased gradients, suggesting the Cramr           the discriminator loss converging quickly to zero [179],
distance between two distributions. Other papers related to        providing no gradient updates to the generator; difficulties
WGAN can be found in [173]–[178].                                  in making the pair of models converge [32].
                                                                       We will introduce GANs training from three perspec-
3.2.6 Loss Sensitive GAN (LS-GAN)
                                                                   tives: objective function, skills, and structure.
Similar to WGAN, LS-GAN [20] also has a Lipschitz con-
straint. It is assumed in LS-GAN that pdata lies in a set of
Lipschitz densities with a compact support. In LS-GAN , the        3.3.1    Objective function
loss function Lθ (x) is parameterized with θ and LS-GAN
                                                                As we can see from Subsection 3.1, utilizing the original
assumes that a generated sample should have larger loss
                                                                objective function in equation (1) will have the gradient van-
than a real one. The loss function can be trained to satisfy
                                                                ishing problem for training G and utilizing the alternative
the following constraint:
                                                                G loss (12) in non-saturating game will get the mode col-
           Lθ (x) ≤ Lθ (G (z)) − ∆ (x, G (z))             (32) lapse problem. These problems are caused by the objective
                                                                function and cannot be solved by changing the structures
where ∆ (x, G (z)) is the margin measuring the difference of GANs. Re-designing the objective function is a natural
between generated sample G(z) and real sample x. The solution to mitigate these problems. Based on the theoretical
objective function of LS-GAN is                                 flaws of GANs, many objective function based variants
 min LD = Ex∼pdata (x) [Lθ (x)]                                 have been proposed to change the objective function of
  D                                                             GANs based on theoretical analyses such as least squares
 +λE x∼pdata (x), [∆ (x, G (z)) + Lθ (x) − Lθ (G (z))]+ ,  (33)
       z∼p (z)
                                                                generative adversarial networks [21], [22].
           z
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                            8

        3.3.1.1 Least squares generative adversarial net-          D give high score to real samples and low score to the
works (LSGANs) :                                                   generated (“fake”) samples. However, the discriminator in
LSGANs [21], [22] are proposed to overcome the vanishing           EBGAN attributes low energy (score) to the real samples
gradient problem in the original GANs. This work shows             and higher energy to the generated ones. EBGAN has more
that the decision boundary for D of original GAN penalizes         stable behavior than original GANs during training.
very small error to update G for those generated samples                   3.3.1.4 Boundary equilibrium generative adversar-
which are far from the decision boundary. LSGANs adopt             ial networks (BEGAN):
the least squares loss rather than the cross-entropy loss in       Similar to EBGAN [38], dual-agent GAN (DA-GAN) [186],
the original GANs. Suppose that the a-b coding is used             [187], and margin adaptation for GANs (MAGANs) [188],
for the LSGANs’ discriminator [21], where a and b are the          BEGAN also uses an auto-encoder as the discriminator.
labels for generated sample and real sample, respectively.         Using proportional control theory, BEGAN proposes a novel
The LSGANs’ discriminator loss VLSGAN (D) and generator            equilibrium method to balance generator and discriminator
loss VLSGAN (G) are defined as:                                    in training, which is fast, stable, and robust to parameter
                                     h
                                                  2
                                                    i              changes.
    min VLSGAN (D) = Ex∼pdata (x) (D (x) − b)                              3.3.1.5 Mode regularized generative adversarial
      D                    h                 i          (35)
                                           2
                +Ez∼pz (z) (D (G (z)) − a) ,                       networks (MDGAN) :
                                                                   Che et al. [26] argue that GANs’ unstable training and model
                             h                 i                   collapse is due to the very special functional shape of the
                                             2
   min VLSGAN (G) = Ez∼pz (z) (D (G (z)) − c) ,             (36)   trained discriminators in high dimensional spaces, which
    G
                                                                   can make training stuck or push probability mass in the
where c is the value that G hopes for D to believe for             wrong direction, towards that of higher concentration than
generated samples. The reference [21] shows that there are         that of the real data distribution. Che et al. [26] introduce
two advantages of LSGANs in comparison with the original           several methods of regularizing the objective, which can
GANs:                                                              stabilize the training of GAN models. The key idea of
   •    The new decision boundary produced by D penal-             MDGAN is utilizing an encoder E (x) : x → z to produce
        izes large error to those generated samples which          the latent variable z for the generator G rather than utilizing
        are far from the decision boundary, which makes            noise. This procedure has two advantages:
        those “low quality” generated samples move toward
        the decision boundary. This is good for generating            •     Encoder guarantees the correspondence between z
        higher quality samples.                                             (E(x)) and x, which makes G capable of covering di-
   •    Penalizing the generated samples far from the de-                   verse modes in the data space. Therefore, it prevents
        cision boundary can supply more gradient when                       the mode collapse problem.
        updating the G, which overcomes the vanishing gra-            •     Because the reconstruction of encoder can add more
        dient problems in the original GANs.                                information to the generator G, it is not easy for the
                                                                            discriminator D to distinguish between real samples
       3.3.1.2 Hinge loss based GAN:                                        and generated ones.
Hinge loss based GAN is proposed and used in [23]–[25]
and its objective function is V (D, G):                            The loss function for the generator and the encoder of
                                                                 MDGAN is
   VD Ĝ, D = Ex∼pdata (x) [min(0, −1 + D (x))]
                    h                  i        (37)                      LG = −Ez∼pz(z) [log (D (G (z)))]
         +Ez∼pz (z) min(0, −1 − D Ĝ(z) ) .                                                                     
                                                                                          λ1 d (x, G ◦ E (x))                     (41)
                                                                            +Ex∼pdata (x)                         ,
                            h         i                                                 +λ2 log D (G ◦ E (x))
         VD G, D̂ = −Ez∼pz (z) D̂ (G(z)) .                  (38)

The softmax cross-entropy loss [184] is also used in GANs.                                    
                                                                                                  λ1 d (x, G ◦ E (x))
                                                                                                                          
        3.3.1.3 Energy-based generative adversarial net-                  LE = Ex∼pdata (x)                                   ,   (42)
                                                                                                  +λ2 log D (G ◦ E (x))
work (EBGAN):
EBGAN’s discriminator is seen as an energy function, giving        where both λ1 and λ2 are free tuning parameters, d is the
high energy to the fake (“generated”) samples and lower            distance metric such as Euclidean distance, and G ◦ E (x) =
energy to the real samples. As for the energy function, please     G (E (x)).
refer to [185] for the corresponding tutorial. Given a positive           3.3.1.6 Unrolled GAN:
margin m , the loss functions for EBGAN can be defined as          Metz et al. [27] introduce a technique to stabilize GANs
follows:                                                           by defining the generator objective with regard to an un-
          LD (x, z) = D(x) + [m − D(G(z))] ,
                                                  +
                                                            (39)   rolled optimization of the discriminator. This allows training
                                                                   to be adjusted between utilizing the current value of the
                                                                   discriminator, which is usually unstable and leads to poor
                    LG (z) = D(G(z)),                       (40)   solutions, and utilizing the optimal discriminator solution
                                                                   in the generator’s objective, which is perfect but infeasible
           +
where [y] = max(0, y) is the rectified linear unit (ReLU)          in real applications. Let f (θG , θD ) denote the objective
function. Note that in the original GANs, the discriminator        function of the original GANs.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                           9

   A local optimal solution of the discriminator parameters        This modification can be understood in the following way
 ∗
θD  can be expressed as the fixed point of an iterative            [28]: D estimates the probability that the given real sample
optimization procedure,                                            is more realistic than a randomly sampled generated sam-
                           0                                       ple. Similarly, Drev (x̃) = sigmoid (C (xg ) − C (xr )) can be
                          θD = θD ,                         (43)
                                                                   defined as the probability that the given generated sample
                                                                   is more realistic than a randomly sampled real sample. The
                                              k
                                                                  discriminator and generator loss functions of the Relativistic
                k+1    k      df        θG , θD
               θD   = θD + ηk             k
                                                  ,         (44)   Standard GAN (RSGAN) is:
                                        dθD
                                                                   LRSGAN
                                                                    D     = −E(xr ,xg ) [log (sigmoid (C (xr ) − C (xg )))] , (51)
                     ∗                   k
                    θD (θG ) =    lim   θD ,                (45)
                                 k→∞

where η k is the learning rate. By unrolling for K steps, a        LRSGAN
                                                                    G     = −E(xr ,xg ) [log (sigmoid (C (xg ) − C (xr )))] . (52)
surrogate objective for the update of the generator is created
                                  K
                                                                  Most GANs can be parametrized:
          fK (θG , θD ) = f θG , θD (θG , θD ) .           (46)
When K = 0, this objective is the same as the standard                 LGAN
                                                                        D   = Exr [f1 (C (xr ))] + Exg [f2 (C (xg ))] ,          (53)
GAN objective. When K → ∞, this objective is the true
                                      ∗
generator objective function f (θG , θD (G)). By adjusting the
number of unrolling steps K , we are capable of interpolat-            LGAN
                                                                        G   = Exr [g1 (C (xr ))] + Exg [g2 (C (xg ))] ,          (54)
ing between standard GAN training dynamics with their
related pathologies, and more expensive gradient descent           where f1 , f2 , g1 , g2 are scalar-to-scalar functions. If we adopt
on the true generator loss. The generator and discriminator        a relativistic discriminator, the loss functions of these GANs
parameter updates of unrolled GAN using this surrogate             become:
loss are
                               dfK (θG , θD )                              LRGAN
                                                                            D    = E(xr ,xg ) [f1 (C (xr ) − C (xg ))]
                θG = θG − η                   ,             (47)                                                       ,         (55)
                                   dθG                                        +E(xr ,xg ) [f2 (C (xg ) − C (xr ))]

                                 df (θG , θD )
                 θD = θD + η                   .            (48)           LRGAN = E(xr ,xg ) [g1 (C (xr ) − C (xg ))]
                                     dθD                                    G                                                    (56)
                                                                              +E(xr ,xg ) [g2 (C (xg ) − C (xr ))] .
Metz et al. [27] show how this method solves mode collapse,
stabilizes training of GANs, and increases diversity and
coverage of the generated distribution by the generator.           3.3.2   Skills
        3.3.1.7 Spectrally normalized GANs (SN-GANs):
                                                                   NIPS 2016 held a workshop on adversarial training, with
SN-GANs [23] propose a novel weight normalization
                                                                   an invited talk by Soumith Chintala named ”How to train
method named spectral normalization to make the training
                                                                   a GAN.” This talk has an assorted tips and tricks. For
of the discriminator stable. This new normalization tech-
                                                                   example, this talk suggests that if you have labels, train-
nique is computationally efficient and easy to be integrated
                                                                   ing the discriminator to also classify the examples: AC-
into existing methods. The spectral normalization [23] uses
                                                                   GAN [30]. Refer to the GitHub repository associated with
a simple method to make the weight matrix W satisfy the
                                                                   Soumith’s talk: https://github.com/soumith/ganhacks for
Lipschitz constraint σ (W ) = 1:
                                                                   more advice.
                 W̄SN (W ) := W/σ (W ) ,                    (49)       Salimans et al. [29] proposed very useful and improved
                                                                   techniques for training GANs (ImprovedGANs), such as
where W is the weight matrix of each layer in D and σ (W )         feature matching, minibatch discrimination, historical aver-
is the spectral norm of W . It is shown that [23] SN-GANs          aging, one-sided label smoothing, and virtual batch normal-
can generate images of equal or better quality in comparison       ization.
with the previous training stabilization methods. In theory,
spectral normalization is capable of being applied to all
GANs variants. Both BigGANs [36] and SAGAN [35] use                3.3.3   Structure
the spectral normalization and have good performances on
                                                                   The original GANs utilized multi-layer perceptron (MLP).
the Imagenet.
                                                                   Specific type of structure may be good for specific applica-
        3.3.1.8 Relativistic GANs (RGANs):
                                                                   tions e.g., recurrent neural network (RNN) for time series
In the original GANs, the discriminator can be defined,
                                                                   data and convolutional neural network (CNN) for images.
according to the non-transformed layer C(x), as D(x) =
                                                                          3.3.3.1 The original GANs:
sigmoid(C(x)). A simple way to make discriminator rela-
                                                                   The original GANs used MLP as the generator G and
tivistic (i.e., making the output of D depend on both real and
                                                                   discriminator D. MLP can be only used for small datasets
generated samples) [28] is to sample from real/generated
                                                                   such as CIFAR-10 [189], MNIST [190], and the Toronto Face
data pairs x̃ = (xr , xg ) and define it as
                                                                   Database (TFD) [191]. However, MLP does not have good
          D (x̃) = sigmoid (C (xr ) − C (xg )) .            (50)   generalization on more complex images [10].
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                    10

        3.3.3.2 Laplacian generative adversarial networks        greatly. BigGANs successfully generates images with quite
(LAPGAN) and SinGAN:                                             high resolution up to 512 by 512 pixels. If you do not have
LAPGAN [31] is proposed for producing higher resolution          enough data, it can be a challenging task to replicate the
images in comparison with the original GANs. LAPGAN              BigGANs results from scratch. Lucic et al. [199] propose to
uses a cascade of CNN within a Laplacian pyramid frame-          train BigGANs quality model with fewer labels. BigBiGAN
work [192] to generate high quality images.                      [200], based on BigGANs, extends it to representation learn-
    SinGAN [193] learns a generative model from a single         ing by adding an encoder and modifying the discriminator.
natural image. SinGAN has a pyramid of fully convolutional       BigBiGAN achieve the state of the art in both unsupervised
GANs, each of which learns the patch distribution at a           representation learning on ImageNet and unconditional im-
different scale of the image. Similar to SinGAN, InGAN           age generation.
[194] also learns a generative model from a single natural            In the original GANs [3], G and D are defined by MLP.
image.                                                           Karras et al. [37] proposed a StyleGAN architecture for
        3.3.3.3 Deep convolutional generative adversarial        GANs, which wins the CVPR 2019 best paper honorable
networks (DCGANs):                                               mention. StyleGAN’s generator is a really high-quality gen-
In original GANs, G and D are defined by MLP. Because            erator for other generation tasks like generating faces. It is
CNN are better at images than MLP, G and D are defined by        particular exciting because it allows to separate different
deep convolutional neural networks (DCNNs) in DCGANs             factors such as hair, age and sex that are involved in con-
[32], which have better performance. Most current GANs           trolling the appearance of the final example and we can then
are at least loosely based on the DCGANs architecture [32].      control them separately from each other. StyleGAN [37] has
Three key features of the DCGANs architecture are listed as      also been used in such as generating high-resolution fashion
follows:                                                         model images wearing custom outfits [201].
                                                                          3.3.3.7 Hybrids of autoencoders and GANs:
   •    First, the overall architecture is mostly based on
                                                                 An autoencoder is a type of neural networks used for
        the all-convolutional net [195]. This architecture has
                                                                 learning efficient data codings in an unsupervised way. The
        neither pooling nor “unpooling” layers. When G
                                                                 autoencoder has an encoder and a decoder. The encoder
        needs to increase the spatial dimensionality of the
                                                                 aims to learn a representation (encoding) for a set of data,
        representation, it uses transposed convolution (de-
                                                                 z = E(x), typically for dimensionality reduction. The de-
        convolution) with a stride greater than 1.
                                                                 coder aims to reconstruct the data x̂ = g (z). That is to say,
   •    Second, utilize batch normalization for most layers of
                                                                 the decoder tries to generate from the reduced encoding a
        both G and D. The last layer of G and first layer of
                                                                 representation as close as possible to its original input x.
        D are not batch normalized, in order that the neural
                                                                      GANs with an autoencoder: Adversarial autoencoder
        network can learn the correct mean and scale of the
                                                                 (AAE) [202] is a probabilistic autoencoder based on GANs.
        data distribution.
                                                                 Adversarial variational Bayes (AVB) [203], [204] is proposed
   •    Third, utilize the Adam optimizer instead of SGD
                                                                 to unify variational autoencoders (VAEs) and GANs. Sun et
        with momentum.
                                                                 al. [205] proposed a UNsupervised Image-to-image Transla-
       3.3.3.4 Progressive GAN:                                  tion (UNIT) framework that are based on GANs and VAEs.
In Progressive GAN (PGGAN) [33], a new training method-          Hu et al. [206] aimed to establish formal connections be-
ology for GAN is proposed. The structure of Progressive          tween GANs and VAEs through a new formulation of them.
GAN is based on progressive neural networks that is first        By combining a VAE with a GAN, Larsen et al. [207] utilize
proposed in [196]. The key idea of Progressive GAN is to         learned feature representations in the GAN discriminator as
grow both the generator and discriminator progressively:         basis for the VAE reconstruction. Therefore, element-wise
starting from a low resolution, adding new layers that           errors is replaced with feature-wise errors to better capture
model increasingly fine details as training progresses.          the data distribution while offering invariance towards such
       3.3.3.5 Self-Attention Generative Adversarial Net-        as translation. Rosca et al. [208] proposed variational ap-
work (SAGAN):                                                    proaches for auto-encoding GANs. By adopting an encoder-
SAGAN [35] is proposed to allow attention-driven, long-          decoder architecture for the generator, disentangled rep-
range dependency modeling for image generation tasks.            resentation GAN (DR-GAN) [68] addresses pose-invariant
Spectral normalization technique has only been applied to        face recognition, which is a hard problem due to the drastic
the discriminator [23]. SAGAN uses spectral normalization        changes in an image for each diverse pose.
for both generator and discriminator and it is found that this        GANs with an encoder: References [40], [42] only add
improves training dynamics. Furthermore, it is confirmed         an encoder to GANs. The original GANs [3] can not learn
that the two time-scale update rule (TTUR) [181] is effective    the inverse mapping - projecting data back into the latent
in SAGAN.                                                        space. To solve this problem, Donahue et al. [40] proposed
    Note that AttnGAN [197] utilizes attention over word         Bidirectional GANs (BiGANs), which can learn this inverse
embeddings within an input sequence rather than self-            mapping through the encoder, and show that the resulting
attention over internal model states.                            learned feature representation is useful. Similarly, Dumoulin
       3.3.3.6 BigGANs and StyleGAN:                             et al. [41] proposed the adversarially learned inference (ALI)
Both BigGANs [36] and StyleGAN [37], [198] made great            model, which also utilizes the encoder to learn the latent
advances in the quality of GANs.                                 feature distribution. The structure of BiGAN and ALI is
    BigGANs [36] is a large scale TPU implementation of          shown in Fig. 3(a). Besides the discriminator and generator,
GANs, which is pretty similar to SAGAN but scaled up             BiGAN also has an encoder, which is used for mapping the
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                      11

                                                                    whilst the other discriminator, conversely, favoring data
    Latent    Data                          Latent      Data
    space     space                         space       space       from the true distribution, and the generator generates data
                                                                    to fool both discriminators. The reference [43] develops
                 ( )                    R              ( )
                                  Real     ( )
                                                                    theoretical analysis to show that, given the optimal discrim-
                         ,
                            OR  D or                                inators, optimizing the generator of D2GAN is minimizing
                      , ( )       Fake?                (   )
                                                             R
                                                                    both KL and reverse KL divergences between true distribu-
                                                                    tion and the generated distribution, thus effectively over-
               (a) BiGAN/ALI                   (b) AGE              coming the mode collapsing problem. Generative multi-
                                                                    adversarial network (GMAN) [44] further extends GANs
 Fig. 3: The structures of (a) BiGAN and ALI and (b) AGE.           to a generator and multiple discriminators. Albuquerque et
                                                                    al. [211] performed multi-objective training of GANs with
                                                                    multiple discriminators.
data back to the latent space. The input of discriminator is a             3.3.3.9 Multi-generator learning:
pair of data composed of data and its corresponding latent Multi-generator GAN (MGAN) [45] is proposed to train
code. For real data x, the pair are x, E(x) where E(x) is the GANs with a mixture of generators to avoid the mode
obtained from the encoder E . For the generated data G(z), collapsing problem. More specially, MGAN has one binary
this pair is G(z), z where z is the noise vector generating discriminator, K generators, and a multi-class classifier.
the data G(z) through the generator G. Similar to (1), the The distinguishing feature of MGAN is that the generated
objective function of BiGAN is                                      samples are produced from multiple generators, and then
  min max V (D, E, G) = Ex∼pdata (x) [log D (x, E(x))]              only one of them will be randomly selected as the final
  G,E   D                                                      (57) output similar to the mechanism of a probabilistic mixture
    +Ez∼pz (z) [log (1 − D (G (z) , z))] .                          model. The classifier shows which generator a generated
The generator of [40], [42] can be seen the decoder since the sample is from.
generator map the vectors from latent space to data space,              The most closely related to MGAN is multi-agent diverse
which performs the same function as the decoder.                    GANs (MAD-GAN) [46]. The difference between MGAN
    Similar to utilizing an encoding process to model the and MAD-GAN can be found in [45]. SentiGAN [212] uses
distribution of latent samples, Gurumurthy et al. [209] a mixture of generators and a multi-class discriminator to
model the latent space as a mixture of Gaussians and learn generate sentimental texts.
the mixture components that maximize the likelihood of                     3.3.3.10 Multi-GAN learning:
generated samples under the data generating distribution.           Coupled GAN (CoGAN) [47] is proposed for learning a joint
    In an encoding-decoding model, the output (also known distribution of two-domain images. CoGAN is composed
as a reconstruction), ought to be similar to the input in the of a pair of GANs - GAN1 and GAN2, each of which
ideal case. Generally, the fidelity of reconstructed samples synthesizes images in one domain. Two GANs respectively
synthesized utilizing a BiGAN/ALI is poor. With an addi- considering structure and style are proposed in [213] based
tional adversarial cost on the distribution of data samples on cGANs. Causal fairness-aware GANs (CFGAN) [214]
and their reconstructions [158], the fidelity of samples may used two generators and two discriminators for generating
be improved. Other related methods include such as vari- fair data. The structures of GANs, D2GAN, MGAN, and
ational discriminator bottleneck (VDB) [210] and MDGAN CoGAN are shown in Fig. 4.
(detailed in Paragraph 3.3.1.5).                                           3.3.3.11 Summary:
    Combination of a generator and an encoder: Different There are many GANs’ variants and milestone ones are
from previous hybrids of autoencoders and GANs, Ad- shown in Fig. 5. Due to space limitation, only limited
versarial Generator-Encoder (AGE) Network [42] is set up number of variants are shown in Fig. 5.
directly between the generator and the encoder, and no                  GANs’ objective function based variants can be gener-
external mappings are trained in the learning process. The alized to structure variants. Compared with other objective
structure of AGE is shown in Fig. 3(b) which R is the recon- function based variants, both SN-GANs and RGANs show
struction loss function. In AGE, there are two reconstruction the stronger generalization ability. These two objective func-
losses: the latent variable z and E(G(z)), the data x and tion based variants can be generalized to the other objective
G(E(x)). AGE is similar to CycleGAN. However, there are function based variants. Spectral normalization is capable
two differences between them:                                       of being generalized to any type of GANs’ variants while
    •   CycleGAN [53] is used for two modalities of the RGANs is able to be generalized to any IPM-based GANs.
        image such as grayscale and color. AGE acts between
        latent space and true data space.                           3.4 Evaluation metrics for GANs
    •   There is a discriminator for each modality of Cycle-
        GAN and there is no discriminator in AGE.                   In this subsection, we show evaluation metrics [215], [216]
                                                                    that are used for GANs.
        3.3.3.8 Multi-discriminator learning:
GANs have a discriminator together with a generator. Dif-
ferent from GANs, dual discriminator GAN (D2GAN) [43] 3.4.1 Inception Score (IS)
has a generator and two binary discriminators. D2GAN is Inception score (IS) is proposed in [29], which uses the
analogous to a minimax game, wherein one discriminator Inception model [217] for every generated image to get
gives high scores for samples from generated distribution the conditional label distribution p (y|x). Images that have
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                      12


                                     ( )            Real
                                                                                  which is the Fréchet distance (also called the Wasserstein-2
                                           OR     D or                            distance) between the two Gaussian distributions. However,
                                                    Fake?                         the IS and FID cannot well handle the overfitting problem.
                                 (a) GANs
                                                                                  To mitigate this problem, the kernel inception distance (KID)
                                                                                  was proposed in [220].
                                  ( )           Real or Fake?
                                                                                  3.4.4 Multi-scale structural similarity (MS-SSIM)
                                                Real or Fake?                     Structural similarity (SSIM) [221] is proposed to measure the
                                                                                  similarity between two images. Different from single scale
                                 (b) D2GAN
                                                                                  SSIM measure, the MS-SSIM [222] is proposed for multi-
                                                                                  scale image quality assessment. It quantitatively evaluates
                ⋯                ~              classifier                        image similarity by attempting to predict human perceptual
           shared                                               Which generator
           parameter                                   ⋯                          similarity judgment. The range of MS-SSIM values is be-
                                                                was utilized?
                                                  shared                          tween 0.0 and 1.0 and lower MS-SSIM values means percep-
               ⋯
                                                  parameter
           shared
       ⋮   parameter   ⋮
                                                     ⋯          Real or Fake?
                                                                                  tually more dissimilar images. References [30], [223] used
               ⋯                            discriminator                         MS-SSIM to measure the diversity of fake data. Reference
                                                                                  [224] suggested that MS-SSIM should only be taken into
                                  (c) MGAN                                        account together with the FID and IS metrics for testing
                    Generators             Discriminators                         sample diversity.
                 ⋯         ⋯                ⋯         ⋯
           shared weight                        shared weight                     3.4.5 Summary
                ⋯          ⋯                ⋯        ⋯                            How to select a good evaluation metric for GANs is still a
                                                                                  hard problem [225]. Xu et al. [219] proposed an empirical
                                 (d) CoGAN
                                                                                  study on evaluation metrics of GANs. Karol Kurach [224]
Fig. 4: The structures of GANs [3], D2GAN [43], MGAN [45],                        conducted a large-scale study on regularization and normal-
and CoGAN [47].                                                                   ization in GANs. There are some other comparative study
                                                                                  of GANs such as [226]. Reference [227] presented several
                                                                                  measures as meta-measures to guide researchers to select
meaningful objects ought to have a conditional label distri-                      quantitative evaluation metrics. An appropriate evaluation
bution p (y|x) with low entropy. Furthermore, the model is                        metric ought to differentiate true samples from fake ones,
expected
R         to produce diverse images. Therefore, the marginal                      verify mode drop, mode collapse, and detect overfitting. It
  p (y|x = G (z))dz ought to have high entropy. In combina-                       is hoped that there will be better methods to evaluate the
tion of these two requirements, the IS is:                                        quality of the GANs model in the future.
                 exp(Ex KL (p (y|x) ||p (y))),                            (58)
                                                                                  3.5   Task driven GANs
where exponentiating results is for easy comparison of the                        The focus of this paper is on GANs. There are closely
values.                                                                           associated fields for specific tasks with an enormous volume
    A higher IS indicates that the generative model can                           of literature.
produce high quality samples and the generated samples
are also diverse. However, the IS also has disadvantages. If                      3.5.1 Semi-Supervised Learning
the generative model falls into mode collapse, the IS might
                                                                                  A research field where GANs are very successful is the ap-
be still good while the real case is pretty bad. To address this
                                                                                  plication of generative models to semi-supervised learning
issue, an independent Wasserstein critic [218] is proposed
                                                                                  [228], [229], as proposed but not shown in the first GANs
to be trained independently for the validation dataset to
                                                                                  paper [3].
measure mode collapse and overfitting.
                                                                                      GANs have been successfully used for semi-supervised
                                                                                  learning at least since CatGANs [48]. Feature matching
3.4.2 Mode score (MS)
                                                                                  GANs [29] got good performance with a small number of
The mode score (MS) [26], [219] is an improved version of                         labels on datasets such as MNIST, SVHN, and CIFAR-10.
the IS. Different from IS, MS can measure the dissimilarity                           Odena [230] extends GANs to the semi-supervised learn-
between the real distribution and generated distribution.                         ing by forcing the discriminator network to output class la-
                                                                                  bels. Generally speaking, when we train GANs, we actually
3.4.3 Fréchet Inception Distance (FID)                                           do not use the discriminator in the end. The discriminator is
FID was also proposed [181] to evaluate GANs. For a                               only used to guide the learning process, but the discrimina-
suitable feature function φ (the default one is the Inception                     tor is not used to generate the data after we have trained the
network’s convolutional feature), FID models φ (pdata ) and                       generator. We only use the generator to generate the data
φ (pg ) as Gaussian random variables with empirical means                         and abandon discriminator at last. For traditional GANs,
µr , µg and empirical covariance Cr , Cg and computes                             the discriminator is a two-class classifier, which outputs cat-
                                                                                  egory one for real data and category two for generated data.
                  data , pg ) = kµr − µg k
            F ID(p                                                               In semi-supervised learning, the discriminator is upgraded
                                                                          (59)
                                              
                                          1/2
               +tr Cr + Cg − 2(Cr Cg )          ,                                 to be a multi-class classifier. At the end of the training, the
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                           13



                             LAPGAN                                                                PGGAN         BigGANs

                                                            CycleGAN                             video2video

          original GANs        DCGANs       ImprovedGANs    PACGAN      WGAN     WGAN-GP                          SinGAN
                                                                         BEGAN
                                                                                              hinge loss based
                                                                                                   GAN           SAGAN

            Year: 2014         2015             2016                     2017                       2018         2019

                            Fig. 5: A road map of GANs. Milestone variants are shown in this figure.


classifier is the thing that we are interested in. For semi-           [238], [239]. By using synthetic data and domain adaptation
supervised learning, if we want to learn an N class classifier,        [237], the number of real-world examples needed to achieve
we make GANs with a discriminator which can predict N +1               a given level of performance is reduced by up to 50 times,
classes the input is from, where an extra class corresponds to         utilizing only randomly generated simulated objects.
the outputs of G. Therefore, suppose that we want to learn                  Recent studies have shown remarkable success in image-
to classify two classes, apples and oranges. We can make a             to-image translation [240]–[245] for two domains. However,
classifier which has three different labels: one - the class of        existing methods such as CycleGAN [53], DiscoGAN [54],
real apples, two - the class of real oranges, and three - the          and DualGAN [55], cannot be used directly for more than
class of generated data. The system learns on three kinds of           two domains, since different approaches should be built
data: real labeled data, unlabeled real data, and fake data.           independently for every pair of domains. StarGAN [56]
    Real labeled data: We can tell the discriminator to                well solve this problem which can conduct image-to-image
maximize the probability of the correct class. For example,            translations for multiple domains using only a single model.
if we have an apple photo and it is labeled as an apple,               Other related works can be found in [246], [247]. CoGAN
we can maximize the probability of the apple class in this             [47] can be also used for multiple domains.
discriminator.                                                              Learning fair representations is a closely related problem
    Unlabeled real data: Suppose we have a photo, we                   to domain transfer. Note that different formulations of ad-
do not know whether it is an apple or an orange but we                 versarial objectives [248]–[251] achieve different notations of
know that it is a real photo. In this situation, we train the          fairness.
discriminator to maximize the sum of the probabilities over                 Domain adaptation [61], [252] can be seen as a subset of
all the real classes.                                                  transfer learning [253]. Recent visual domain adaptation
    Fake data: When we obtain a generated example from                 (VDA) methods include: visual appearance adaptation,
the generator, we train the discriminator to classify it as a          representation adaptation, and output adaptation, which
fake example.                                                          can be thought of as using domain adaptation based on
    Miyato et al. [49] proposed virtual adversarial training           the original input, features, and outputs of the domains,
(VAT): a regularization method for both supervised and                 respectively.
semi-supervised learning. Dai et al. [231] show that given                  Visual appearance adaptation: CycleGAN [53] is a rep-
the discriminator objective, good semi-supervised learning             resentative method in this category. CyCADA [57] is pro-
indeed requires a bad generator from the theory perspective,           posed for visual appearance adaptation based on Cycle-
and propose the definition of a preferred generator. A tri-            GAN.
angle GAN (∆-GAN) [50] is proposed for semi-supervised                      Representation adaptation: The key of adversarial dis-
cross-domain joint distribution matching and ∆-GAN is                  criminative domain adaptation (ADDA) [58], [59] is to learn
closely related to Triple-GAN [51]. Madani et al. [232] used           feature representations that a discriminator cannot differ-
semi-supervised learning with GANs for chest X-ray classi-             entiate which domain they belong to. Sankaranarayanan et
fication.                                                              al. [254] focused on adapting the representations learned by
    Future improvements to GANs can be expected to                     segmentation networks across real and synthetic domains
simultaneously produce further improvements to semi-                   based on GANs. Fully convolutional adaptation networks
supervised learning and unsupervised learning such as self-            (FCAN) [60] is proposed for semantic segmentation which
supervised learning [233].                                             combines visual appearance adaptation and representation
                                                                       adaptation.
3.5.2 Transfer learning                                                     Output adaptation: Tsai [255] made the outputs of the
Ganin et al. [234] introduce a domain-adversarial training             source and target images have a similar structure so that the
approach of neural networks for domain adaptation, where               discriminator cannot differentiate them.
training data and test data are from similar but different                  Other transfer learning based GANs can be found in [52],
distributions. The Professor Forcing algorithm [235] uses ad-          [256]–[262].
versarial domain adaptation for training recurrent network.
Shrivastava et al. [236] used GANs for simulated training              3.5.3 Reinforcement learning
data. A novel extension of pixel-level domain adaptation               Generative models can be integrated into reinforcement
named GraspGAN [237] was proposed for robotic grasping                 learning (RL) [107] in different ways [103], [263]. Reference
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                   14

[264] has already discussed connections between GANs and               Ground truth         MSE        Adversarial loss
actor-critic methods. The connections among GANs, inverse
reinforcement learning (IRL), and energy-based models is
studied in [265]. These connections to RL are possible to be
useful for the development of both GANs and RL. Further-
more, GANs were combined with reinforcement learning
for synthesizing programs for images [266]. The competitive
multi-agent learning framework proposed in [267] is also
related to GANs and works on learning robust grasping
                                                                Fig. 6: Lotter et al. [276] show an excellent description of
policies by an adversary.
                                                                the significance of being capable of modeling multi-modal
    Imitation Learning: The connection between imitation
                                                                data. In this instance, a model is trained to predict the
learning and EBGAN is discussed in [268]. Ho and Ermon
                                                                next frame in a video. The video describes a computer
[269] show that an instantiation of their framework draws
                                                                rendering of a moving 3D model of a man’s head. The
an analogy between GANs and imitation learning, from
                                                                image on the left is the ground truth, an instance of an
which they derive a model-free imitation learning method
                                                                actual frame of a video, which the model would like to
that has significant performance gains over existing model-
                                                                predict. The image in the middle is what happens when the
free algorithms in imitating complex behaviors in large
                                                                model is trained using mean squared error (MSE) between
and high-dimensional environments. Song et al. [62] pro-
                                                                the model’s predicted next frame and the actual next frame.
posed multi-agent generative adversarial imitation learning
                                                                The model is forced to select only one answer for what
(GAIL) and Guo et al. [270] proposed generative adversarial
                                                                the next frame will be. Since there are multiple possible
self-imitation learning. A multi-agent GAIL framework is
                                                                answers, corresponding to slightly diverse positions of the
used in a deconfounded multi-agent environment recon-
                                                                head, the answer that the model selects is an average over
struction (DEMER) approach [271] to learn the environment.
                                                                multiple slightly diverse images. This causes the faces to
DEMER is tested in the real application of Didi Chuxing and
                                                                have a blurring effect. Utilizing an additional GANs loss,
has achieved good performances.
                                                                the image on the right is capable of knowing that there are
                                                                multiple possible outputs, each of which is recognizable and
3.5.4   Multi-modal learning                                    clear as a realistic, satisfying image. Images are from [276].
Generative models, especially GANs, make machine learn-
ing be able to work with multi-modal outputs. In many
tasks, an input may correspond to multiple diverse correct      4.1   Maximum likelihood estimation (MLE)
outputs, each of which is an acceptable answer. Traditional     Not all generative models use maximum likelihood estima-
ways of training machine learning methods, such as mini-        tion (MLE). Some generative models do not utilize MLE,
mizing the mean squared error (MSE) between the model’s         but can be made to do so (GANs belong to this category). It
predicted output and a desired output, are not capable of       can be simply proved that minimizing the Kullback-Leibler
training models that can produce many different correct         Divergence (KLD) between pdata (x) and pg (x) is equal to
outputs. One instance of such a case is predicting the next     maximizing the log likelihood as the number of samples m
frame in a video sequence, as shown in Fig. 6. Multi-modal      increases:
image-to-image translation related works can be found in                θ∗ = arg min KLD ( pdata k pg )
[272]–[275].                                                                    θ    R                 pg (x)
                                                                        = arg min − pdata (x) log pdata    (x) dx
                                                                              θ
3.5.5   Other task driven GANs
                                                                                  R
                                                                        = arg min pdata (x) log pdata (x) dx
GANs have been used for feature learning such as feature                    Rθ                                            (60)
                                                                          − pdata R(x) log pg (x) dx
selection [277], hashing [278]–[285], and metric learning
                                                                        = arg max pdata (x) log pg (x) dx
[286].                                                                        θ              Pm
                                                                                           1
    MisGAN [287] was proposed to learn from incomplete                  = arg max lim m         i=1 log pg (xi ).
                                                                               θ      m→∞
data with GANs. Evolutionary GANs are proposed in [288].
Ponce et al. [289] combined GANs and genetic algorithms             The model probability distribution pθ (x) is replaced
to evolve images for visual neurons. GANs have also been        with pg (x) for notation consistency. Refer to Chapter 5 of
used in other machine learning tasks [290] such as active       [298] for more information on MLE and other statistical
learning [291], [292], online learning [293], ensemble learn-   estimators.
ing [294], zero-shot learning [295], [296], and multi-task
learning [297].                                                 4.2   Mode collapse
                                                                GANs are hard to train, and it has been observed [26], [29]
                                                                that they often suffer from mode collapse [299], [300], in
4   T HEORY
                                                                which the generator learns to generate samples from only
In this section, we first introduce maximum likelihood es-      a few modes of the data distribution but misses many
timation. Then, we introduce mode collapse. Finally, other      other modes, even if samples from the missing modes exist
theoretical issues such as inverse mapping and memoriza-        throughout the training data. In the worst case, the gener-
tion are discussed.                                             ator produces simply a single sample (complete collapse)
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                    15

[179], [301]. In this subsection, we will first introduce two     4.3.3 Inverse mapping
viewpoints of GANs mode collapse. Then, we will intro-            GANs cannot learn the inverse mapping - projecting data
duce methods that propose new objective functions or new          back into the latent space. BiGANs [40] (detailed in Sec-
structures to solve mode collapse.                                tion 3.3.3.7) is proposed as a way of learning this inverse
                                                                  mapping. Dumoulin et al. [41] introduce the adversarially
4.2.1 Two viewpoints: divergence and algorithmic                  learned inference (ALI) model (detailed in Section 3.3.3.7),
We can resolve and understand GANs mode collapse and              which jointly learns an inference network and a generation
instability from two viewpoints: divergence and algorith-         network utilizing an adversarial process. Arora et al. [310]
mic.                                                              show the theoretical limitations of Encoder-Decoder GAN
    Divergence viewpoint: Roth et al. [302] stabilizd train-      architectures such as BiGANs [40] and ALI [41]. Creswell et
ing of GANs and their variants such as f -divergence based        al. [311] invert the generator of GANs.
GANs (f -GAN) through regularization.
    Algorithmic viewpoint: The numerics of common algo-           4.3.4 Mathematical perspective such as optimization
rithms for training GANs are analyzed and a new algorithm         Mohamed et al. [312] frame GANs within the algorithms
that has better convergence properties is proposed in [303].      for learning in implicit generative models that only specify
Mescheder et al. [304] showed that which training methods         a stochastic procedure with which to generate data. Gidel
for GANs do actually converge.                                    et al. [313] looked at optimization approaches designed
                                                                  for GANs and casted GANs optimization problems in the
4.2.2 Methods overcoming mode collapse                            general variational inequality framework. The convergence
Objective function based methods: Deep regret analytic            and robustness of training GANs with regularized optimal
GAN (DRAGAN) [170] suggests that the mode collapse                transport is disscussed in [314].
exists due to the occurence of a fake local Nash equilibrium
in the nonconvex problem. DRAGAN solves this problem              4.3.5 Memorization
by constraining gradients of the discriminator around the         As for “memorization of GANs”, Nagarajan et al. [315]
real data manifold. It adds a gradient penalizing term which      argue that making the generator “learn to memorize” the
biases the discriminator to have a gradient norm of 1 around      training data is a more difficult task than making it “learn
the real data manifold. Other methods such as EBGAN and           to output realistic but unseen data”.
Unrolled GAN (detailed in Section 3.3) also belong to this
category.
    Structure based methods: Representative methods in            5     A PPLICATIONS
this category include such as MAD-GAN [46] and MRGAN              As discussed earlier, GANs are a powerful generative model
[26] (detailed in Section 3.3)                                    which can generate realistic-looking samples with a random
    There are also other methods to reduce mode collapse in       vector z . We neither need to know an explicit true data
GANs. For example, PACGAN [305] eases the pain of mode            distribution nor have any mathematical assumptions. These
collapse by changing input to the discriminator.                  advantages allow GANs to be widely applied to many areas
                                                                  such as image processing and computer vision, sequential
4.3   Other theoretical issues                                    data.
4.3.1 Do GANs actually learn the distribution?
References [41], [301], [306] have both empirically and the-      5.1   Image processing and computer vision
oretically brought the concern to light that distributions        The most successful applications of GANs are in image
learned by GANs suffer from mode collapse. In contrast,           processing and computer vision, such as image super-
Bai et al. [307] show that GANs can in principle learn            resolution, image synthesis and manipulation, and video
distributions in Wasserstein distance (or KL-divergence in        processing.
many situations) with polynomial sample complexity, if the
discriminator class has strong discriminating power against       5.1.1 Super-resolution (SR)
the particular generator class (instead of against all possible   SRGAN [63], GANs for SR, is the first framework able to
generators). Liang et al. [308] studied how well GANs learn       infer photo-realistic natural images for upscaling factors.
densities, including nonparametric and parametric target          To further improve the visual quality of SRGAN, Wang
distributions. Singh et al. [309] further studied nonparamet-     et al. [64] thoroughly study three key components of SR-
ric density estimation with adversarial losses.                   GAN and improve each of them to derive an Enhanced
                                                                  SRGAN (ESRGAN). For example, ESRGAN uses the idea
4.3.2 Divergence/Distance                                         from relativistic GANs [28] to have the discriminator predict
Arora et al. [301] show that training of GAN may not have         relative realness rather than the absolute value. Benefiting
good generalization properties; e.g., training may look suc-      from these improvements, ESRGAN won the first place in
cessful but the generated distribution may be far from real       the PIRM2018-SR Challenge (region 3) [316] and got the
data distribution in standard metrics. The popular distances      best perceptual index. Based on CycleGAN [53], the Cycle-
such as Wasserstein and Jensen-Shannon (JS) may not gen-          in-Cycle GANs [65] is proposed for unsupervised image
eralize. However, generalization does occur by introducing        SR. SRDGAN [66] is proposed to learn the noise prior for
a novel notion of distance between distributions, the neural      SR with DualGAN [55]. Deep tensor generative adversarial
net distance. Are there other useful divergences?                 nets (TGAN) [67] is proposed to generate large high-quality
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                         16

images by exploring tensor structures. There are methods                   5.1.2.3 Interaction between a human being and an
specific for face SR [317]–[319]. Other related methods can        image generation process:
be found in [320]–[323].                                           There are many applications that involve interaction be-
                                                                   tween a human being and an image generation process.
5.1.2   Image synthesis and manipulation                           Realistic image manipulation is difficult because it requires
                                                                   modifying the image in a user-controlled way, while making
        5.1.2.1 Face:                                              it appear realistic. If the user does not have efficient artistic
Pose related: Disentangled representation learning GAN             skill, it is easy to deviate from the manifold of natural
(DR-GAN) [324] is proposed for pose-invariant face recog-          images while editing. Interactive GAN (IGAN) [73] defines
nition. Huang et al. [69] proposed a Two-Pathway GAN               a class of image editing operations, and constrain their
(TP-GAN) for photorealistic frontal view synthesis by si-          output to lie on that learned manifold at all times. Intro-
multaneously perceiving local details and global structures.       spective adversarial networks [74] also offer this capability
Ma et al. [70] proposed the novel Pose Guided Person               to perform interactive photo editing and have demonstrated
Generation Network (PG2 ) that synthesizes person images           their results mostly in face editing. GauGAN [75] can turn
in arbitrary poses, based on a novel pose and an image of          doodles into stunning, photorealistic landscapes.
that person. Cao et al. [325] proposed a high fidelity pose
invariant model for high-resolution face frontalization based
on GANs. Siarohin et al. [326] proposed deformable gans for        5.1.3   Texture synthesis
pose-based human image generation. Pose-robust spatial-            Texture synthesis is a classical problem in image field.
aware GAN (PSGAN) for customizable makeup transfer is              Markovian GANs (MGAN) [76] is a texture synthesis
proposed in [71].                                                  method based on GANs. By capturing the texture data of
    Portrait related: APDrawingGAN [72] is proposed to             Markovian patches, MGAN can generate stylized videos
generate artistic portrait drawings from face photos with          and images very quickly, in order to realize real-time texture
hierarchical GANs. APDrawingGAN has a software based               synthesis. Spatial GAN (SGAN) [77] was the first to apply
on wechat and the results are shown in Fig. 8. GANs have           GANs with fully unsupervised learning in texture synthesis.
also been used in other face related applications such as          Periodic spatial GAN (PSGAN) [78] is a variant of SGAN,
facial attribute changes [327] and portrait editing [328]–         which can learn periodic textures from a single image or
[331].                                                             complicated big dataset.
    Face generation: The quality of generated faces by GANs
is improved year by year, which can be found in Sebastian          5.1.4   Object detection
Nowozin’s GAN lecture materials1 . As we can see from Fig-
                                                                   How can we learn an object detector that is invariant to
ure 7, the generated faces based on original GANs [3] are of
                                                                   deformations and occlusions? One way is using a data-
low visual quality and can only serves as a proof of concept.
                                                                   driven strategy - collect large-scale datasets which have
Radford et al. [32] used better neural network architectures:
                                                                   object examples under different conditions. We hope that the
deep convolutional neural networks for generating faces.
                                                                   final classifier can use these instances to learn invariances.
Roth et al. [302] addressed the instability problems of GAN
                                                                   Is it possible to see all the deformations and occlusions in a
training, allowing for larger architectures such as the ResNet
                                                                   dataset? Some deformations and occlusions are so rare that
to be utilized. Karras et al. [33] utilized multiscale training,
                                                                   they hardly happen in practical applications; yet we want to
allowing megapixel face image generation at high fidelity.
                                                                   learn a method invariant to such situations. Wang et al. [346]
    Face generation [332]–[340] is somewhat easy because
                                                                   used GANs to generate instances with deformations and
there is only one class of object. Every object is a face and
                                                                   occlusions. The aim of the adversary is to generate instances
most face data sets tend to be composed of people looking
                                                                   that are difficult for the object detector to classify. By using a
straight into the camera. Most people have been registered
                                                                   segmentor and GANs, Segan [79] detected objects occluded
in terms of putting nose and eyes and other landmarks in
                                                                   by other objects in an image. To deal with the small object
consistent locations.
                                                                   detection problem, Li et al. [80] proposed perceptual GAN
        5.1.2.2 General object:
                                                                   and Bai et al. [81] proposed an end-to-end multi-task GAN
It is a little harder to have GANs work on assorted data
                                                                   (MTGAN).
sets like ImageNet [151] which has a thousand different
object classes. However, we have seen rapid progress over
the recent few years. The quality of these images has been         5.1.5   Video applications
improved year by year [304].                                       Reference [82] is the first paper to use GANs for video
    While most papers use GANs to synthesize images in             generation. Villegas et al. [347] proposed a deep neural
two dimensions [341], [342], Wu et al. [343] synthesized           network for the prediction of future frames in natural video
three-dimensional (3-D) samples using GANs and volumet-            sequences using GANs. Denton and Birodkar [83] proposed
ric convolutions. Wu et al. [343] synthesized novel objects        a new model named disentangled representation net (DR-
including cars, chairs, sofa, and tables. Im et al. [344] gen-     NET) that learns disentangled image representations from
erated images with recurrent adversarial networks. Yang et         video based on GANs. A novel video-to-video synthesis
al. [345] proposed layered recursive GANs (LR-GAN) for             approach (video2video) under the generative adversarial
image generation.                                                  learning framework was proposed in [85]. MoCoGan [86]
                                                                   is proposed to decompose motion and content for video
  1. https://github.com/nowozin/mlss2018-madrid-gan                generation [348]–[350].
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                17




               [Goodfellow et al., 2014]          [Radford et al., 2015]           [Roth et al., 2017]      [Karras et al., 2018]
                University of Montreal            Facebook AI Research            Microsoft and ETHZ              NVIDIA

                                                            Fig. 7: Face image synthesis.


                                                                           have been widely used in image-to-text (image caption)
                                                                           [391], [392], too.
                                                                              Furthermore, GANs have been widely utilized in other
                                                                           NLP applications such as question answer selection [393],
                                                                           [394], poetry generation [395], talent-job fit [396], and review
                                                                           detection and geneneration [397], [398].
                                                                              Music: GANs have been used for generating music
                                                                           such as continuous RNN-GAN (C-RNN-GAN) [91], Object-
                   (a) photo        (b) portrait drawings
                                                                           Reinforced GAN (ORGAN) [92], and SeqGAN [93], [94].
                                                                              Speech and Audio: GANs have been used for speech
Fig. 8: Given a photo such as (a), APDrawingGAN can                        and audio analysis such as synthesis [399]–[401], enhance-
produce the corresponding artistic portrait drawings (b).                  ment [402], and recognition [403], .

                                                                           5.3   Other applications
   GANs have also been used in other video applications
such as video prediction [84], [351], [352] and video retar-               Medical field: GANs have been widely utilized in medical
geting [353].                                                              field such as generating and designing DNA [404], [405],
                                                                           drug discovery [406], generating multi-label discrete patient
5.1.6 Other image and vision applications                                  records [407], medical image processing [408]–[415], dental
GANs have been utilized in other image processing and                      restorations [416], and doctor recommendation [417].
computer vision tasks [354]–[357] such as object transfig-                     Data science: GANs have been used in data generating
uration [358], [359], semantic segmentation [360], visual                  [214], [418]–[426], neural networks generating [427], data
saliency prediction [361], object tracking [362], [363], image             augmentation [428], [429], spatial representation learning
dehazing [364]–[366], natural image matting [367], image                   [430], network embedding [431], heterogeneous information
inpainting [368], [369], image fusion [370], image completion              networks [432], and mobile user profiling [433].
[371], and image classification [372].                                         GANs have been widely used in many other areas
    Creswell et al. [373] show that the representations                    such as malware detection [434], chess game playing [435],
learned by GANs can also be used for retrieval. GANs have                  steganography [436]–[439], privacy-preserving [440]–[442],
also been used for anticipating where people will look [374],              social robot [443], and network pruning [444], [445].
[375].
                                                                            6    O PEN RESEARCH PROBLEMS
5.2   Sequential data                                                      Because GANs have become popular throughout the deep
GANs also have achievements in sequential data such as                     learning area, its limitations have recently been improved
natural language, music, speech, voice [376], [377], and time              [446], [447]. There are still open research problems for
series [378]–[381].                                                        GANs.
    Natural language processing (NLP): IRGAN [88], [89]                        GANs for discrete data: GANs rely on the generated
is proposed for information retrieval (IR). Li et al. [382]                samples being completely differentiable with respect to the
used adversarial learning for neural dialogue generation.                  generative parameters. Therefore, GANs cannot produce
GANs have also been used in text generation [87], [383]–                   discrete data directly, such as hashing code and one-hot
[385] and speech language processing [94]. Kbgan [386] is                  word. Solving this problem is very important since it could
proposed to generate high-quality negative examples and                    unlock the potential of GANs for NLP and hashing. Good-
used in knowledge graph embeddings. Adversarial REward                     fellow [103] suggested three ways to solve this problem:
Learning (AREL) [387] is proposed for visual storytelling.                 using Gumbel-softmax [448], [449] or the concrete distri-
DSGAN [388] is proposed for distant supervision relation                   bution [450]; utilizing the REINFORCE algorithm [451];
extraction. ScratchGAN [389] is proposed to train a language               training the generator to sample continuous values that can
GAN from scratch – without maximum likelihood pre-                         be transformed to discrete ones (such as sampling word
training.                                                                  embeddings directly).
    Qiao et al. [90] learn text-to-image generation by re-                     There are other methods towards this research direction.
description and text conditioned auxiliary classifier GAN                  Song et al. [278] used a continuous function to approximate
(TAC-GAN) [390] is also proposed for text to image. GANs                   the sign function for hashing code. Gulrajani et al. [19]
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                 18

modelled discrete data with a continuous generator. Hjelm        R EFERENCES
et al. [452] introduced an algorithm for training GANs with
                                                                 [1]    L. J. Ratliff, S. A. Burden, and S. S. Sastry, “Characterization and
discrete data that utilizes the estimated difference measure            computation of local nash equilibria in continuous games,” in
from the discriminator to compute importance weights for                Annual Allerton Conference on Communication, Control, and Com-
generated samples, and thus providing a policy gradient for             puting, pp. 917–924, 2013.
training the generator. Other related work can be found in       [2]    J. Schmidhuber, “Learning factorial codes by predictability mini-
                                                                        mization,” Neural Computation, vol. 4, no. 6, pp. 863–879, 1992.
[453], [454]. More work needs to be done in this interesting     [3]    I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-
area.                                                                   Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adver-
    New Divergences: New families of Integral Probability               sarial nets,” in Neural Information Processing Systems, pp. 2672–
                                                                        2680, 2014.
Metrics (IPMs) for training GANs such as Fisher GAN [455],
                                                                 [4]    J. Schmidhuber, “Unsupervised minimax: Adversarial curiosity,
[456], mean and covariance feature matching GAN (McGan)                 generative adversarial networks, and predictability minimiza-
[457], and Sobolev GAN [458], have been proposed. Are                   tion,” arXiv preprint arXiv:1906.04493, 2019.
there any other interesting classes of divergences? This         [5]    X. Wu, K. Xu, and P. Hall, “A survey of image synthesis and
                                                                        editing with generative adversarial networks,” Tsinghua Science
deserves further study.                                                 and Technology, vol. 22, no. 6, pp. 660–674, 2017.
    Estimation uncertainty: Generally speaking, as we have       [6]    N. Torres-Reyes and S. Latifi, “Audio enhancement and synthesis
more data, uncertainty estimation reduces. GANs do not                  using generative adversarial networks: A survey,” International
give the distribution that generated the training examples              Journal of Computer Applications, vol. 182, no. 35, pp. 27–31, 2019.
                                                                 [7]    K. Wang, C. Gou, Y. Duan, Y. Lin, X. Zheng, and F.-Y. Wang,
and GANs aim to generate new samples that come from                     “Generative adversarial networks: introduction and outlook,”
the same distribution of the training examples. Therefore,              IEEE/CAA Journal of Automatica Sinica, vol. 4, no. 4, pp. 588–598,
GANs have neither a likelihood nor a well-defined posterior.            2017.
There are early attempts towards this research direction         [8]    Y. Hong, U. Hwang, J. Yoo, and S. Yoon, “How generative
                                                                        adversarial networks and their variants work: An overview,”
such as Bayesian GAN [459]. Although we can use GANs                    ACM Computing Surveys, vol. 52, no. 1, pp. 1–43, 2019.
to generate data, how can we measure the uncertainty of          [9]    A. Creswell, T. White, V. Dumoulin, K. Arulkumaran, B. Sen-
the well-trained generator? This is another interesting future          gupta, and A. A. Bharath, “Generative adversarial networks: An
                                                                        overview,” IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 53–
issue.                                                                  65, 2018.
    Theory: As for generalization, Zhang et al. [460] devel-     [10]   Z. Wang, Q. She, and T. E. Ward, “Generative adversarial net-
oped generalization bounds between the true distribution                works: A survey and taxonomy,” arXiv preprint arXiv:1906.01529,
and learned distribution under different evaluation metrics.            2019.
                                                                 [11]   S. Hitawala, “Comparative study on generative adversarial net-
When evaluated with neural distance, the bounds in [460]                works,” arXiv preprint arXiv:1801.04271, 2018.
show that generalization is guaranteed as long as the dis-       [12]   M. Zamorski, A. Zdobylak, M. Zieba, and J. Swiatek, “Generative
criminator set is small enough, regardless of the size of the           adversarial networks: recent developments,” in International Con-
hypothesis set or generator. Arora et al. [306] proposed a              ference on Artificial Intelligence and Soft Computing, pp. 248–258,
                                                                        Springer, 2019.
novel test for estimating support size using the birthday        [13]   Z. Pan, W. Yu, X. Yi, A. Khan, F. Yuan, and Y. Zheng, “Recent
paradox of discrete probability and show that GAN does                  progress on generative adversarial networks (gans): A survey,”
suffer mode collapse even when images are of higher visual              IEEE Access, vol. 7, pp. 36322–36333, 2019.
                                                                 [14]   X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and
quality. More deep theoretical research is well worth study-
                                                                        P. Abbeel, “Infogan: Interpretable representation learning by
ing. How do we test for generalization empirically? Useful              information maximizing generative adversarial nets,” in Neural
theory should enable choice of model class, capacity, and               Information Processing Systems, pp. 2172–2180, 2016.
architectures. This is an interesting issue to be investigated   [15]   M. Mirza and S. Osindero, “Conditional generative adversarial
                                                                        nets,” arXiv preprint arXiv:1411.1784, 2014.
in future work.                                                  [16]   Y. Lu, Y.-W. Tai, and C.-K. Tang, “Conditional cyclegan
    Others: There are other important research problems for             for attribute guided face image generation,” arXiv preprint
GANs such as evaluation (detailed in Subsection 3.4) and                arXiv:1705.09966, 2017.
mode collapse (detailed in Subsection 4.2)                       [17]   S. Nowozin, B. Cseke, and R. Tomioka, “f-gan: Training genera-
                                                                        tive neural samplers using variational divergence minimization,”
                                                                        in Neural Information Processing Systems, pp. 271–279, 2016.
                                                                 [18]   M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein genera-
7   C ONCLUSIONS                                                        tive adversarial networks,” in International Conference on Machine
                                                                        Learning, pp. 214–223, 2017.
This paper presents a comprehensive review of various            [19]   I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C.
aspects of GANs. We elaborate on several perspectives, i.e.,            Courville, “Improved training of wasserstein gans,” in Neural
                                                                        Information Processing Systems, pp. 5767–5777, 2017.
algorithm, theory, applications, and open research problems.     [20]   G.-J. Qi, “Loss-sensitive generative adversarial networks on lips-
We believe this survey will help readers to gain a thorough             chitz densities,” International Journal of Computer Vision, pp. 1–23,
understanding of the GANs research area.                                2019.
                                                                 [21]   X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. Paul Smolley,
                                                                        “Least squares generative adversarial networks,” in IEEE Inter-
                                                                        national Conference on Computer Vision, pp. 2794–2802, 2017.
ACKNOWLEDGMENTS                                                  [22]   X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley,
                                                                        “On the effectiveness of least squares generative adversarial
The authors would like to thank the NetEase course taught               networks,” IEEE Transactions on Pattern Analysis and Machine
by Shuang Yang, Ian Good fellow’s invited talk at AAAI 19,              Intelligence, vol. 41, no. 12, pp. 2947–2960, 2019.
CVPR 2018 tutorial on GANs, Sebastian Nowozin’s MLSS             [23]   T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, “Spectral nor-
                                                                        malization for generative adversarial networks,” in International
2018 GAN lecture materials. The authors also would like to              Conference on Learning Representations, 2018.
thank the helpful discussions with group members of Umich        [24]   J. H. Lim and J. C. Ye, “Geometric gan,” arXiv preprint
Yelab and Foreseer research group.                                      arXiv:1705.02894, 2017.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                             19

[25]   D. Tran, R. Ranganath, and D. M. Blei, “Deep and hierarchical          [50]   Z. Gan, L. Chen, W. Wang, Y. Pu, Y. Zhang, H. Liu, C. Li, and
       implicit models,” arXiv preprint arXiv:1702.08896, 2017.                      L. Carin, “Triangle generative adversarial networks,” in Neural
[26]   T. Che, Y. Li, A. P. Jacob, Y. Bengio, and W. Li, “Mode regularized           Information Processing Systems, pp. 5247–5256, 2017.
       generative adversarial networks,” in International Conference on       [51]   L. Chongxuan, T. Xu, J. Zhu, and B. Zhang, “Triple generative ad-
       Learning Representations, 2017.                                               versarial nets,” in Neural Information Processing Systems, pp. 4088–
[27]   L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein, “Unrolled gen-             4098, 2017.
       erative adversarial networks,” arXiv preprint arXiv:1611.02163,        [52]   H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, and M. Marc-
       2016.                                                                         hand, “Domain-adversarial neural networks,” arXiv preprint
[28]   A. Jolicoeur-Martineau, “The relativistic discriminator: a key                arXiv:1412.4446, 2014.
       element missing from standard gan,” in International Conference        [53]   J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-
       on Learning Representation, 2019.                                             image translation using cycle-consistent adversarial networks,”
[29]   T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford,                in International Conference on Computer Vision, pp. 2223–2232, 2017.
       and X. Chen, “Improved techniques for training gans,” in Neural        [54]   T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim, “Learning to
       Information Processing Systems, pp. 2234–2242, 2016.                          discover cross-domain relations with generative adversarial net-
[30]   A. Odena, C. Olah, and J. Shlens, “Conditional image synthesis                works,” in International Conference on Machine Learning, pp. 1857–
       with auxiliary classifier gans,” in International Conference on Ma-           1865, 2017.
       chine Learning, pp. 2642–2651, 2017.                                   [55]   Z. Yi, H. Zhang, P. Tan, and M. Gong, “Dualgan: Unsupervised
[31]   E. L. Denton, S. Chintala, R. Fergus, et al., “Deep generative image          dual learning for image-to-image translation,” in International
       models using a laplacian pyramid of adversarial networks,” in                 Conference on Computer Vision, pp. 2849–2857, 2017.
       Neural Information Processing Systems, pp. 1486–1494, 2015.            [56]   Y. Choi, M. Choi, M. Kim, J.-W. Ha, S. Kim, and J. Choo, “Star-
[32]   A. Radford, L. Metz, and S. Chintala, “Unsupervised represen-                 gan: Unified generative adversarial networks for multi-domain
       tation learning with deep convolutional generative adversarial                image-to-image translation,” in IEEE Conference on Computer Vi-
       networks,” arXiv preprint arXiv:1511.06434, 2015.                             sion and Pattern Recognition, pp. 8789–8797, 2018.
[33]   T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive growing    [57]   J. Hoffman, E. Tzeng, T. Park, J.-Y. Zhu, P. Isola, K. Saenko, A. A.
       of gans for improved quality, stability, and variation,” in Interna-          Efros, and T. Darrell, “Cycada: Cycle-consistent adversarial do-
       tional Conference on Learning Representations, 2018.                          main adaptation,” in International Conference on Machine Learning,
[34]   H. Zhang, T. Xu, H. Li, S. Zhang, X. Wang, X. Huang, and D. N.                2018.
       Metaxas, “Stackgan: Text to photo-realistic image synthesis with       [58]   E. Tzeng, J. Hoffman, K. Saenko, and T. Darrell, “Adversarial dis-
       stacked generative adversarial networks,” in IEEE International               criminative domain adaptation,” in IEEE Conference on Computer
       Conference on Computer Vision, pp. 5907–5915, 2017.                           Vision and Pattern Recognition, pp. 7167–7176, 2017.
[35]   H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena, “Self-              [59]   S. Wang and L. Zhang, “Catgan: Coupled adversarial transfer for
       attention generative adversarial networks,” in International Con-             domain generation,” arXiv preprint arXiv:1711.08904, 2017.
       ference on Machine Learning, pp. 7354–7363, 2019.                      [60]   Y. Zhang, Z. Qiu, T. Yao, D. Liu, and T. Mei, “Fully convolutional
                                                                                     adaptation networks for semantic segmentation,” in IEEE Con-
[36]   A. Brock, J. Donahue, and K. Simonyan, “Large scale gan training
                                                                                     ference on Computer Vision and Pattern Recognition, pp. 6810–6818,
       for high fidelity natural image synthesis,” in International Confer-
                                                                                     2018.
       ence on Learning representations, 2019.
                                                                              [61]   K. Bousmalis, N. Silberman, D. Dohan, D. Erhan, and D. Krish-
[37]   T. Karras, S. Laine, and T. Aila, “A style-based generator archi-
                                                                                     nan, “Unsupervised pixel-level domain adaptation with genera-
       tecture for generative adversarial networks,” in IEEE Conference
                                                                                     tive adversarial networks,” in IEEE conference on Computer Vision
       on Computer Vision and Pattern Recognition, pp. 4401–4410, 2019.
                                                                                     and Pattern Recognition, pp. 3722–3731, 2017.
[38]   J. Zhao, M. Mathieu, and Y. LeCun, “Energy-based generative
                                                                              [62]   J. Song, H. Ren, D. Sadigh, and S. Ermon, “Multi-agent generative
       adversarial network,” in International Conference on Learning Rep-
                                                                                     adversarial imitation learning,” in Neural Information Processing
       resentations, 2017.
                                                                                     Systems, pp. 7461–7472, 2018.
[39]   D. Berthelot, T. Schumm, and L. Metz, “Began: Boundary                 [63]   C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham,
       equilibrium generative adversarial networks,” arXiv preprint                  A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al., “Photo-
       arXiv:1703.10717, 2017.                                                       realistic single image super-resolution using a generative adver-
[40]   J. Donahue, P. Krähenbühl, and T. Darrell, “Adversarial feature             sarial network,” in IEEE Conference on Computer Vision and Pattern
       learning,” arXiv preprint arXiv:1605.09782, 2016.                             Recognition, pp. 4681–4690, 2017.
[41]   V. Dumoulin, I. Belghazi, B. Poole, O. Mastropietro, A. Lamb,          [64]   X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, Y. Qiao, and
       M. Arjovsky, and A. Courville, “Adversarially learned inference,”             C. Change Loy, “Esrgan: Enhanced super-resolution generative
       arXiv preprint arXiv:1606.00704, 2016.                                        adversarial networks,” in European Conference on Computer Vision,
[42]   D. Ulyanov, A. Vedaldi, and V. Lempitsky, “It takes (only) two:               pp. 63–79, 2018.
       Adversarial generator-encoder networks,” in AAAI Conference on         [65]   Y. Yuan, S. Liu, J. Zhang, Y. Zhang, C. Dong, and L. Lin, “Unsu-
       Artificial Intelligence, pp. 1250–1257, 2018.                                 pervised image super-resolution using cycle-in-cycle generative
[43]   T. Nguyen, T. Le, H. Vu, and D. Phung, “Dual discriminator gen-               adversarial networks,” in IEEE Conference on Computer Vision and
       erative adversarial nets,” in Neural Information Processing Systems,          Pattern Recognition Workshops, pp. 701–710, 2018.
       pp. 2670–2680, 2017.                                                   [66]   J. Guan, C. Pan, S. Li, and D. Yu, “Srdgan: learning the noise prior
[44]   I. Durugkar, I. Gemp, and S. Mahadevan, “Generative multi-                    for super resolution with dual generative adversarial networks,”
       adversarial networks,” in International Conference on Learning                arXiv preprint arXiv:1903.11821, 2019.
       Representations, 2017.                                                 [67]   Z. Ding, X.-Y. Liu, M. Yin, W. Liu, and L. Kong, “Tgan: Deep
[45]   Q. Hoang, T. D. Nguyen, T. Le, and D. Phung, “Multi-generator                 tensor generative adversarial nets for large image generation,”
       generative adversarial nets,” arXiv preprint arXiv:1708.02556,                arXiv preprint arXiv:1901.09953, 2019.
       2017.                                                                  [68]   L. Q. Tran, X. Yin, and X. Liu, “Representation learning by
[46]   A. Ghosh, V. Kulharia, V. P. Namboodiri, P. H. Torr, and P. K.                rotating your faces,” IEEE Transactions on Pattern Analysis and
       Dokania, “Multi-agent diverse generative adversarial networks,”               Machine Intelligence, 2019.
       in IEEE Conference on Computer Vision and Pattern Recognition,         [69]   R. Huang, S. Zhang, T. Li, and R. He, “Beyond face rotation:
       pp. 8513–8521, 2018.                                                          Global and local perception gan for photorealistic and identity
[47]   M.-Y. Liu and O. Tuzel, “Coupled generative adversarial net-                  preserving frontal view synthesis,” in International Conference on
       works,” in Neural Information Processing Systems, pp. 469–477,                Computer Vision, pp. 2439–2448, 2017.
       2016.                                                                  [70]   L. Ma, X. Jia, Q. Sun, B. Schiele, T. Tuytelaars, and L. Van Gool,
[48]   J. T. Springenberg, “Unsupervised and semi-supervised learning                “Pose guided person image generation,” in Neural Information
       with categorical generative adversarial networks,” arXiv preprint             Processing Systems, pp. 406–416, 2017.
       arXiv:1511.06390, 2015.                                                [71]   W. Jiang, S. Liu, C. Gao, J. Cao, R. He, J. Feng, and S. Yan,
[49]   T. Miyato, S.-i. Maeda, S. Ishii, and M. Koyama, “Virtual adver-              “Psgan: Pose-robust spatial-aware gan for customizable makeup
       sarial training: a regularization method for supervised and semi-             transfer,” arXiv preprint arXiv:1909.06956, 2019.
       supervised learning,” IEEE Transactions on Pattern Analysis and        [72]   R. Yi, Y.-J. Liu, Y.-K. Lai, and P. L. Rosin, “Apdrawinggan:
       Machine Intelligence, vol. 41, no. 8, pp. 1979–1993, 2019.                    Generating artistic portrait drawings from face photos with hier-
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                                20

       archical gans,” in IEEE Conference on Computer Vision and Pattern       [96]  D. J. Rezende, S. Mohamed, and D. Wierstra, “Stochastic back-
       Recognition, pp. 10743–10752, 2019.                                           propagation and approximate inference in deep generative mod-
[73]   J.-Y. Zhu, P. Krähenbühl, E. Shechtman, and A. A. Efros, “Gen-              els,” arXiv preprint arXiv:1401.4082, 2014.
       erative visual manipulation on the natural image manifold,” in          [97] G. E. Hinton, T. J. Sejnowski, and D. H. Ackley, Boltzmann ma-
       European Conference on Computer Vision, pp. 597–613, 2016.                    chines: Constraint satisfaction networks that learn. Carnegie-Mellon
[74]   A. Brock, T. Lim, J. M. Ritchie, and N. Weston, “Neural photo                 University, Department of Computer Science Pittsburgh, 1984.
       editing with introspective adversarial networks,” arXiv preprint        [98] D. H. Ackley, G. E. Hinton, and T. J. Sejnowski, “A learning
       arXiv:1609.07093, 2016.                                                       algorithm for boltzmann machines,” Cognitive science, vol. 9,
[75]   T. Park, M.-Y. Liu, T.-C. Wang, and J.-Y. Zhu, “Semantic image                no. 1, pp. 147–169, 1985.
       synthesis with spatially-adaptive normalization,” in IEEE Con-          [99] G. E. Hinton, S. Osindero, and Y.-W. Teh, “A fast learning
       ference on Computer Vision and Pattern Recognition, pp. 2337–2346,            algorithm for deep belief nets,” Neural computation, vol. 18, no. 7,
       2019.                                                                         pp. 1527–1554, 2006.
[76]   C. Li and M. Wand, “Precomputed real-time texture synthesis             [100] A. Nguyen, A. Dosovitskiy, J. Yosinski, T. Brox, and J. Clune,
       with markovian generative adversarial networks,” in European                  “Synthesizing the preferred inputs for neurons in neural net-
       Conference on Computer Vision, pp. 702–716, 2016.                             works via deep generator networks,” in Neural Information Pro-
[77]   N. Jetchev, U. Bergmann, and R. Vollgraf, “Texture synthesis                  cessing Systems, pp. 3387–3395, 2016.
       with spatial generative adversarial networks,” arXiv preprint           [101] Y. Bengio, E. Laufer, G. Alain, and J. Yosinski, “Deep generative
       arXiv:1611.08207, 2016.                                                       stochastic networks trainable by backprop,” in International Con-
[78]   U. Bergmann, N. Jetchev, and R. Vollgraf, “Learning texture                   ference on Machine Learning, pp. 226–234, 2014.
       manifolds with the periodic spatial gan,” in Proceedings of the 34th    [102] Y. Bengio, L. Yao, G. Alain, and P. Vincent, “Generalized denois-
       International Conference on Machine Learning-Volume 70, pp. 469–              ing auto-encoders as generative models,” in Neural Information
       477, JMLR. org, 2017.                                                         Processing Systems, pp. 899–907, 2013.
[79]   K. Ehsani, R. Mottaghi, and A. Farhadi, “Segan: Segmenting and          [103] I. Goodfellow, “Nips 2016 tutorial: Generative adversarial net-
       generating the invisible,” in IEEE Conference on Computer Vision              works,” arXiv preprint arXiv:1701.00160, 2016.
       and Pattern Recognition, pp. 6144–6153, 2018.                           [104] T. Salimans, A. Karpathy, X. Chen, and D. P. Kingma, “Pix-
[80]   J. Li, X. Liang, Y. Wei, T. Xu, J. Feng, and S. Yan, “Perceptual gen-         elcnn++: Improving the pixelcnn with discretized logistic
       erative adversarial networks for small object detection,” in IEEE             mixture likelihood and other modifications,” arXiv preprint
       Conference on Computer Vision and Pattern Recognition, pp. 1222–              arXiv:1701.05517, 2017.
       1230, 2017.                                                             [105] B. J. Frey, G. E. Hinton, and P. Dayan, “Does the wake-sleep
[81]   Y. Bai, Y. Zhang, M. Ding, and B. Ghanem, “Sod-mtgan: Small ob-               algorithm produce good density estimators?,” in Advances in
       ject detection via multi-task generative adversarial network,” in             neural information processing systems, pp. 661–667, 1996.
       Proceedings of the European Conference on Computer Vision (ECCV),       [106] B. J. Frey, J. F. Brendan, and B. J. Frey, Graphical models for machine
       pp. 206–221, 2018.                                                            learning and digital communication. MIT press, 1998.
[82]   C. Vondrick, H. Pirsiavash, and A. Torralba, “Generating videos         [107] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van
       with scene dynamics,” in Neural Information Processing Systems,               Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershel-
       pp. 613–621, 2016.                                                            vam, M. Lanctot, et al., “Mastering the game of go with deep
[83]   E. L. Denton et al., “Unsupervised learning of disentangled repre-            neural networks and tree search,” Nature, vol. 529, no. 7587,
       sentations from video,” in Neural Information Processing Systems,             pp. 484–489, 2016.
       pp. 4414–4423, 2017.                                                    [108] K. Eykholt, I. Evtimov, E. Fernandes, B. Li, A. Rahmati, C. Xiao,
[84]   J. Walker, K. Marino, A. Gupta, and M. Hebert, “The pose                      A. Prakash, T. Kohno, and D. Song, “Robust physical-world
       knows: Video forecasting by generating pose futures,” in IEEE                 attacks on deep learning visual classification,” in IEEE Conference
       International Conference on Computer Vision, pp. 3332–3341, 2017.             on Computer Vision and Pattern Recognition, pp. 1625–1634, 2018.
[85]   T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, G. Liu, A. Tao, J. Kautz, and         [109] A. Kurakin, I. Goodfellow, and S. Bengio, “Adversarial examples
       B. Catanzaro, “Video-to-video synthesis,” in Neural Information               in the physical world,” arXiv preprint arXiv:1607.02533, 2016.
       Processing Systems, pp. 1152–1164, 2018.                                [110] G. Elsayed, S. Shankar, B. Cheung, N. Papernot, A. Kurakin,
[86]   S. Tulyakov, M.-Y. Liu, X. Yang, and J. Kautz, “Mocogan: De-                  I. Goodfellow, and J. Sohl-Dickstein, “Adversarial examples that
       composing motion and content for video generation,” in IEEE                   fool both computer vision and time-limited humans,” in Neural
       Conference on Computer Vision and Pattern Recognition, pp. 1526–              Information Processing Systems, pp. 3910–3920, 2018.
       1535, 2018.                                                             [111] X. Jia, X. Wei, X. Cao, and H. Foroosh, “Comdefend: An effi-
[87]   K. Lin, D. Li, X. He, Z. Zhang, and M.-T. Sun, “Adversarial                   cient image compression model to defend adversarial examples,”
       ranking for language generation,” in Neural Information Processing            in IEEE Conference on Computer Vision and Pattern Recognition,
       Systems, pp. 3155–3165, 2017.                                                 pp. 6084–6092, 2019.
[88]   J. Wang, L. Yu, W. Zhang, Y. Gong, Y. Xu, B. Wang, P. Zhang,            [112] A. Athalye, N. Carlini, and D. Wagner, “Obfuscated gradients
       and D. Zhang, “Irgan: A minimax game for unifying generative                  give a false sense of security: Circumventing defenses to adver-
       and discriminative information retrieval models,” in International            sarial examples,” in International Conference on Machine Learning,
       ACM SIGIR conference on Research and Development in Information               2018.
       Retrieval, pp. 515–524, ACM, 2017.                                      [113] D. Zügner, A. Akbarnejad, and S. Günnemann, “Adversarial at-
[89]   S. Lu, Z. Dou, X. Jun, J.-Y. Nie, and J.-R. Wen, “Psgan: A mini-              tacks on neural networks for graph data,” in SIGKDD Conference
       max game for personalized search with limited and noisy click                 on Knowledge Discovery and Data Mining, pp. 2847–2856, 2018.
       data,” in ACM SIGIR Conference on Research and Development in           [114] Y. Dong, F. Liao, T. Pang, H. Su, J. Zhu, X. Hu, and J. Li, “Boost-
       Information Retrieval, pp. 555–564, 2019.                                     ing adversarial attacks with momentum,” in IEEE Conference on
[90]   T. Qiao, J. Zhang, D. Xu, and D. Tao, “Mirrorgan: Learning                    Computer Vision and Pattern Recognition, pp. 9185–9193, 2018.
       text-to-image generation by redescription,” in IEEE Conference on       [115] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Good-
       Computer Vision and Pattern Recognition, pp. 1505–1514, 2019.                 fellow, and R. Fergus, “Intriguing properties of neural networks,”
[91]   O. Mogren, “C-rnn-gan: Continuous recurrent neural networks                   arXiv preprint arXiv:1312.6199, 2013.
       with adversarial training,” arXiv preprint arXiv:1611.09904, 2016.      [116] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and
[92]   G. L. Guimaraes, B. Sanchez-Lengeling, C. Outeiral, P. L. C.                  harnessing adversarial examples,” arXiv preprint arXiv:1412.6572,
       Farias, and A. Aspuru-Guzik, “Objective-reinforced generative                 2014.
       adversarial networks (organ) for sequence generation models,”           [117] J. Kos, I. Fischer, and D. Song, “Adversarial examples for gener-
       arXiv preprint arXiv:1705.10843, 2017.                                        ative models,” in IEEE Security and Privacy Workshops, pp. 36–42,
[93]   S.-g. Lee, U. Hwang, S. Min, and S. Yoon, “A seqgan for poly-                 2018.
       phonic music generation,” arXiv preprint arXiv:1710.11418, 2017.        [118] P. Samangouei, M. Kabkab, and R. Chellappa, “Defense-gan:
[94]   L. Yu, W. Zhang, J. Wang, and Y. Yu, “Seqgan: Sequence genera-                Protecting classifiers against adversarial attacks using generative
       tive adversarial nets with policy gradient,” in AAAI Conference on            models,” arXiv preprint arXiv:1805.06605, 2018.
       Artificial Intelligence, pp. 2852–2858, 2017.                           [119] N. Akhtar and A. Mian, “Threat of adversarial attacks on deep
[95]   D. P. Kingma and M. Welling, “Auto-encoding variational bayes,”               learning in computer vision: A survey,” IEEE Access, vol. 6,
       arXiv preprint arXiv:1312.6114, 2013.                                         pp. 14410–14430, 2018.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                              21

[120] S. Shen, G. Jin, K. Gao, and Y. Zhang, “Ape-gan: Adversarial per-      [144] H. Tang, D. Xu, N. Sebe, Y. Wang, J. J. Corso, and Y. Yan, “Multi-
      turbation elimination with gan,” arXiv preprint arXiv:1707.05474,            channel attention selection gan with cascaded semantic guidance
      2017.                                                                        for cross-view image translation,” in IEEE Conference on Computer
[121] H. Lee, S. Han, and J. Lee, “Generative adversarial trainer:                 Vision and Pattern Recognition, pp. 2417–2426, 2019.
      Defense to adversarial perturbations with gan,” arXiv preprint         [145] L. Karacan, Z. Akata, A. Erdem, and E. Erdem, “Learning to
      arXiv:1705.03387, 2017.                                                      generate images of outdoor scenes from attributes and semantic
[122] L. Huang, A. D. Joseph, B. Nelson, B. I. Rubinstein, and J. Tygar,           layouts,” arXiv preprint arXiv:1612.00215, 2016.
      “Adversarial machine learning,” in ACM Workshop on Security            [146] B. Dai, S. Fidler, R. Urtasun, and D. Lin, “Towards diverse
      and Artificial Intelligence, pp. 43–58, 2011.                                and natural image descriptions via a conditional gan,” in IEEE
[123] I. J. Goodfellow, “On distinguishability criteria for estimating             International Conference on Computer Vision, pp. 2970–2979, 2017.
      generative models,” arXiv preprint arXiv:1412.6515, 2014.              [147] S. Yao, T. M. Hsu, J.-Y. Zhu, J. Wu, A. Torralba, B. Freeman,
[124] M. Kahng, N. Thorat, D. H. P. Chau, F. B. Viégas, and M. Watten-            and J. Tenenbaum, “3d-aware scene manipulation via inverse
      berg, “Gan lab: Understanding complex deep generative models                 graphics,” in Neural Information Processing Systems, pp. 1887–1898,
      using interactive visual experimentation,” IEEE Transactions on              2018.
      Visualization and Computer Graphics, vol. 25, no. 1, pp. 1–11, 2018.   [148] G. G. Chrysos, J. Kossaifi, and S. Zafeiriou, “Robust
[125] D. Bau, J.-Y. Zhu, H. Strobelt, B. Zhou, J. B. Tenenbaum, W. T.              conditional generative adversarial networks,” arXiv preprint
      Freeman, and A. Torralba, “Gan dissection: Visualizing and                   arXiv:1805.08657, 2018.
      understanding generative adversarial networks,” in International       [149] K. K. Thekumparampil, A. Khetan, Z. Lin, and S. Oh, “Robust-
      Conference on Learning Representations, 2019.                                ness of conditional gans to noisy labels,” in Neural Information
[126] M. Kocaoglu, C. Snyder, A. G. Dimakis, and S. Vishwanath,                    Processing Systems, pp. 10271–10282, 2018.
      “Causalgan: Learning causal implicit generative models with            [150] Q. Mao, H.-Y. Lee, H.-Y. Tseng, S. Ma, and M.-H. Yang, “Mode
      adversarial training,” arXiv preprint arXiv:1709.02023, 2017.                seeking generative adversarial networks for diverse image syn-
[127] S. Feizi, F. Farnia, T. Ginart, and D. Tse, “Understanding gans: the         thesis,” in IEEE Conference on Computer Vision and Pattern Recog-
      lqg setting,” arXiv preprint arXiv:1710.10793, 2017.                         nition, pp. 1429–1437, 2019.
[128] F. Farnia and D. Tse, “A convex duality framework for gans,” in        [151] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei,
      Neural Information Processing Systems, pp. 5248–5258, 2018.                  “Imagenet: A large-scale hierarchical image database,” in IEEE
[129] J. Zhao, J. Li, Y. Cheng, T. Sim, S. Yan, and J. Feng, “Under-               Conference on Computer Vision and Pattern Recognition, pp. 248–
      standing humans in crowded scenes: Deep nested adversarial                   255, 2009.
      learning and a new benchmark for multi-human parsing,” in              [152] G. Perarnau, J. Van De Weijer, B. Raducanu, and J. M. Álvarez,
      ACM Multimedia Conference on Multimedia Conference, pp. 792–800,             “Invertible conditional gans for image editing,” arXiv preprint
      ACM, 2018.                                                                   arXiv:1611.06355, 2016.
[130] A. Jahanian, L. Chai, and P. Isola, “On the”steerability” of genera-   [153] M. Saito, E. Matsumoto, and S. Saito, “Temporal generative ad-
      tive adversarial networks,” arXiv preprint arXiv:1907.07171, 2019.           versarial nets with singular value clipping,” in IEEE International
[131] B. Zhu, J. Jiao, and D. Tse, “Deconstructing generative adversarial          Conference on Computer Vision, pp. 2830–2839, 2017.
      networks,” arXiv preprint arXiv:1901.09465, 2019.                      [154] K. Sricharan, R. Bala, M. Shreve, H. Ding, K. Saketh, and
[132] K. B. Kancharagunta and S. R. Dubey, “Csgan: cyclic-synthesized              J. Sun, “Semi-supervised conditional gans,” arXiv preprint
      generative adversarial networks for image-to-image transforma-               arXiv:1708.05789, 2017.
      tion,” arXiv preprint arXiv:1901.03554, 2019.                          [155] T. Miyato and M. Koyama, “cgans with projection discriminator,”
[133] Y. Wu, J. Donahue, D. Balduzzi, K. Simonyan, and T. Lill-                    arXiv preprint arXiv:1802.05637, 2018.
      icrap, “Logan: Latent optimisation for generative adversarial          [156] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image
      networks,” arXiv preprint arXiv:1912.00953, 2019.                            translation with conditional adversarial networks,” in IEEE Con-
[134] T. Kurutach, A. Tamar, G. Yang, S. J. Russell, and P. Abbeel,                ference on Computer Vision and Pattern Recognition, pp. 1125–1134,
      “Learning plannable representations with causal infogan,” in                 2017.
      Neural Information Processing Systems, pp. 8733–8744, 2018.            [157] T.-C. Wang, M.-Y. Liu, J.-Y. Zhu, A. Tao, J. Kautz, and B. Catan-
[135] A. Spurr, E. Aksan, and O. Hilliges, “Guiding infogan with semi-             zaro, “High-resolution image synthesis and semantic manipula-
      supervision,” in Joint European Conference on Machine Learning and           tion with conditional gans,” in IEEE Conference on Computer Vision
      Knowledge Discovery in Databases, pp. 119–134, Springer, 2017.               and Pattern Recognition, pp. 8798–8807, 2018.
[136] A. Nguyen, J. Clune, Y. Bengio, A. Dosovitskiy, and J. Yosinski,       [158] C. Li, H. Liu, C. Chen, Y. Pu, L. Chen, R. Henao, and L. Carin,
      “Plug & play generative networks: Conditional iterative genera-              “Alice: Towards understanding adversarial learning for joint
      tion of images in latent space,” in IEEE Conference on Computer              distribution matching,” in Neural Information Processing Systems,
      Vision and Pattern Recognition, pp. 4467–4477, 2017.                         pp. 5495–5503, 2017.
[137] S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee,      [159] L. C. Tiao, E. V. Bonilla, and F. Ramos, “Cycle-consistent adver-
      “Generative adversarial text to image synthesis,” in International           sarial learning as approximate bayesian inference,” arXiv preprint
      Conference on Machine Learning, pp. 1–10, 2016.                              arXiv:1806.01771, 2018.
[138] S. Hong, D. Yang, J. Choi, and H. Lee, “Inferring semantic layout      [160] I. Csiszár, P. C. Shields, et al., “Information theory and statistics:
      for hierarchical text-to-image synthesis,” in IEEE Conference on             A tutorial,” Foundations and Trends R in Communications and Infor-
      Computer Vision and Pattern Recognition, pp. 7986–7994, 2018.                mation Theory, vol. 1, no. 4, pp. 417–528, 2004.
[139] S. E. Reed, Z. Akata, S. Mohan, S. Tenka, B. Schiele, and H. Lee,      [161] D. J. Im, H. Ma, G. Taylor, and K. Branson, “Quantitatively
      “Learning what and where to draw,” in Neural Information Pro-                evaluating gans with divergences proposed for training,” in
      cessing Systems, pp. 217–225, 2016.                                          International Conference on Learning Representation, 2018.
[140] H. Zhang, T. Xu, H. Li, S. Zhang, X. Wang, X. Huang, and               [162] M. Uehara, I. Sato, M. Suzuki, K. Nakayama, and Y. Matsuo,
      D. Metaxas, “Stackgan++: Realistic image synthesis with stacked              “Generative adversarial nets from a density ratio estimation
      generative adversarial networks,” IEEE Transactions on Pattern               perspective,” arXiv preprint arXiv:1610.02920, 2016.
      Analysis and Machine Intelligence, vol. 41, no. 8, pp. 1947–1962,      [163] B. K. Sriperumbudur, A. Gretton, K. Fukumizu, B. Schölkopf,
      2019.                                                                        and G. R. Lanckriet, “Hilbert space embeddings and metrics
[141] X. Huang, Y. Li, O. Poursaeed, J. Hopcroft, and S. Belongie,                 on probability measures,” Journal of Machine Learning Research,
      “Stacked generative adversarial networks,” in IEEE Conference on             vol. 11, no. Apr, pp. 1517–1561, 2010.
      Computer Vision and Pattern Recognition, pp. 5077–5086, 2017.          [164] A. Uppal, S. Singh, and B. Poczos, “Nonparametric density
[142] J. Gauthier, “Conditional generative adversarial nets for convo-             estimation & convergence of gans under besov ipm losses,” in
      lutional face generation,” Class Project for Stanford CS231N: Con-           Neural Information Processing Systems, pp. 9086–9097, 2019.
      volutional Neural Networks for Visual Recognition, Winter semester,    [165] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and
      vol. 2014, no. 5, p. 2, 2014.                                                A. Smola, “A kernel two-sample test,” Journal of Machine Learning
[143] G. Antipov, M. Baccouche, and J.-L. Dugelay, “Face aging with                Research, vol. 13, no. Mar, pp. 723–773, 2012.
      conditional generative adversarial networks,” in 2017 IEEE In-         [166] G. K. Dziugaite, D. M. Roy, and Z. Ghahramani, “Training
      ternational Conference on Image Processing (ICIP), pp. 2089–2093,            generative neural networks via maximum mean discrepancy
      IEEE, 2017.                                                                  optimization,” arXiv preprint arXiv:1505.03906, 2015.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                                 22

[167] C.-L. Li, W.-C. Chang, Y. Cheng, Y. Yang, and B. Póczos, “Mmd           [192] P. Burt and E. Adelson, “The laplacian pyramid as a compact
      gan: Towards deeper understanding of moment matching net-                      image code,” IEEE Transactions on Communications, vol. 31, no. 4,
      work,” in Neural Information Processing Systems, pp. 2203–2213,                pp. 532–540, 1983.
      2017.                                                                    [193] T. R. Shaham, T. Dekel, and T. Michaeli, “Singan: Learning a gen-
[168] Y. Li, K. Swersky, and R. Zemel, “Generative moment match-                     erative model from a single natural image,” in IEEE International
      ing networks,” in International Conference on Machine Learning,                Conference on Computer Vision, pp. 4570–4580, 2019.
      pp. 1718–1727, 2015.                                                     [194] A. Shocher, S. Bagon, P. Isola, and M. Irani, “Ingan: Capturing
[169] D. J. Sutherland, H.-Y. Tung, H. Strathmann, S. De, A. Ramdas,                 and retargeting the “dna” of a natural image,” in IEEE Interna-
      A. Smola, and A. Gretton, “Generative models and model criti-                  tional Conference on Computer Vision, pp. 4492–4501, 2019.
      cism via optimized maximum mean discrepancy,” arXiv preprint             [195] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller,
      arXiv:1611.04488, 2016.                                                        “Striving for simplicity: The all convolutional net,” arXiv preprint
[170] N. Kodali, J. Abernethy, J. Hays, and Z. Kira, “On convergence                 arXiv:1412.6806, 2014.
      and stability of gans,” arXiv preprint arXiv:1705.07215, 2017.           [196] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirk-
[171] J. Wu, Z. Huang, J. Thoma, D. Acharya, and L. Van Gool, “Wasser-               patrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progres-
      stein divergence for gans,” in European Conference on Computer                 sive neural networks,” arXiv preprint arXiv:1606.04671, 2016.
      Vision, pp. 653–668, 2018.                                               [197] T. Xu, P. Zhang, Q. Huang, H. Zhang, Z. Gan, X. Huang, and
[172] M. G. Bellemare, I. Danihelka, W. Dabney, S. Mohamed, B. Lak-                  X. He, “Attngan: Fine-grained text to image generation with
      shminarayanan, S. Hoyer, and R. Munos, “The cramer distance                    attentional generative adversarial networks,” in IEEE Conference
      as a solution to biased wasserstein gradients,” arXiv preprint                 on Computer Vision and Pattern Recognition, pp. 1316–1324, 2018.
      arXiv:1705.10743, 2017.                                                  [198] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila,
[173] H. Petzka, A. Fischer, and D. Lukovnicov, “On the regulariza-                  “Analyzing and improving the image quality of stylegan,” arXiv
      tion of wasserstein gans,” in International Conference on Learning             preprint arXiv:1912.04958, 2019.
      Representations, pp. 1–24, 2018.                                         [199] M. Lučić, M. Tschannen, M. Ritter, X. Zhai, O. Bachem, and
[174] F. Juefei-Xu, V. N. Boddeti, and M. Savvides, “Gang of gans: Gen-              S. Gelly, “High-fidelity image generation with fewer labels,” in
      erative adversarial networks with maximum margin ranking,”                     International Conference on Machine Learning, pp. 4183–4192, 2019.
      arXiv preprint arXiv:1704.04865, 2017.                                   [200] J. Donahue and K. Simonyan, “Large scale adversarial represen-
[175] C.-C. Hsu, H.-T. Hwang, Y.-C. Wu, Y. Tsao, and H.-M. Wang,                     tation learning,” arXiv preprint arXiv:1907.02544, 2019.
      “Voice conversion from unaligned corpora using variational               [201] G. Yildirim, N. Jetchev, R. Vollgraf, and U. Bergmann, “Gen-
      autoencoding wasserstein generative adversarial networks,” in                  erating high-resolution fashion model images wearing custom
      Interspeech, pp. 3364–3368, 2017.                                              outfits,” arXiv preprint arXiv:1908.08847, 2019.
[176] J. Adler and S. Lunz, “Banach wasserstein gan,” in Neural Infor-         [202] A. Makhzani, J. Shlens, N. Jaitly, I. Goodfellow, and B. Frey, “Ad-
      mation Processing Systems, pp. 6754–6763, 2018.                                versarial autoencoders,” arXiv preprint arXiv:1511.05644, 2015.
[177] Y.-S. Chen, Y.-C. Wang, M.-H. Kao, and Y.-Y. Chuang, “Deep               [203] L. Mescheder, S. Nowozin, and A. Geiger, “Adversarial varia-
      photo enhancer: Unpaired learning for image enhancement from                   tional bayes: Unifying variational autoencoders and generative
      photographs with gans,” in IEEE Conference on Computer Vision                  adversarial networks,” in International Conference on Machine
      and Pattern Recognition, pp. 6306–6314, 2018.                                  Learning, pp. 2391–2400, 2017.
[178] S. Athey, G. Imbens, J. Metzger, and E. Munro, “Using wasser-            [204] X. Yu, X. Zhang, Y. Cao, and M. Xia, “Vaegan: A collaborative
      stein generative adversial networks for the design of monte carlo              filtering framework based on adversarial variational autoen-
      simulations,” arXiv preprint arXiv:1909.02210, 2019.                           coders,” in International Joint Conference on Artificial Intelligence,
[179] M. Arjovsky and L. Bottou:, “Towards principled methods for                    pp. 4206–4212, 2019.
      training generative adversarial networks,” in International Con-         [205] M.-Y. Liu, T. Breuel, and J. Kautz, “Unsupervised image-to-image
      ference on Learning Representations, 2017.                                     translation networks,” in Neural Information Processing Systems,
[180] A. Yadav, S. Shah, Z. Xu, D. Jacobs, and T. Goldstein, “Stabi-                 pp. 700–708, 2017.
      lizing adversarial nets with prediction methods,” arXiv preprint         [206] Z. Hu, Z. Yang, R. Salakhutdinov, and E. P. Xing, “On unifying
      arXiv:1705.07364, 2017.                                                        deep generative models,” arXiv preprint arXiv:1706.00550, 2017.
[181] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and                  [207] A. B. L. Larsen, S. K. Sønderby, H. Larochelle, and O. Winther,
      S. Hochreiter, “Gans trained by a two time-scale update rule                   “Autoencoding beyond pixels using a learned similarity metric,”
      converge to a local nash equilibrium,” in Neural Information                   in International Conference on Machine Learning, pp. 1558–1566,
      Processing Systems, pp. 6626–6637, 2017.                                       2016.
[182] K. J. Liang, C. Li, G. Wang, and L. Carin, “Generative adversarial       [208] M. Rosca, B. Lakshminarayanan, D. Warde-Farley, and S. Mo-
      network training is a continual learning problem,” arXiv preprint              hamed, “Variational approaches for auto-encoding generative
      arXiv:1811.11083, 2018.                                                        adversarial networks,” arXiv preprint arXiv:1706.04987, 2017.
[183] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual learning              [209] S.     Gurumurthy,         R.     Kiran       Sarvadevabhatla,      and
      with deep generative replay,” in Advances in Neural Information                R. Venkatesh Babu, “Deligan: Generative adversarial networks
      Processing Systems, pp. 2990–2999, 2017.                                       for diverse and limited data,” in IEEE Conference on Computer
[184] M. Lin, “Softmax gan,” arXiv preprint arXiv:1704.06191, 2017.                  Vision and Pattern Recognition, pp. 166–174, 2017.
[185] Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, and F. Huang,               [210] X. B. Peng, A. Kanazawa, S. Toyer, P. Abbeel, and S. Levine,
      “A tutorial on energy-based learning,” Predicting structured data,             “Variational discriminator bottleneck: Improving imitation learn-
      vol. 1, no. 0, 2006.                                                           ing, inverse rl, and gans by constraining information flow,” in
[186] J. Zhao, L. Xiong, P. K. Jayashree, J. Li, F. Zhao, Z. Wang, P. S.             International Conference on Learning Representations, 2019.
      Pranata, P. S. Shen, S. Yan, and J. Feng, “Dual-agent gans for           [211] I. Albuquerque, J. Monteiro, T. Doan, B. Considine, T. Falk,
      photorealistic and identity preserving profile face synthesis,” in             and I. Mitliagkas, “Multi-objective training of generative ad-
      Advances in Neural Information Processing Systems, pp. 66–76, 2017.            versarial networks with multiple discriminators,” arXiv preprint
[187] J. Zhao, L. Xiong, J. Li, J. Xing, S. Yan, and J. Feng, “3d-aided              arXiv:1901.08680, 2019.
      dual-agent gans for unconstrained face recognition,” IEEE Trans-         [212] K. Wang and X. Wan, “Sentigan: Generating sentimental texts via
      actions on Pattern Analysis and Machine Intelligence, vol. 41, no. 10,         mixture adversarial networks.,” in International Joint Conference on
      pp. 2380–2394, 2018.                                                           Artificial Intelligence, pp. 4446–4452, 2018.
[188] R. Wang, A. Cully, H. J. Chang, and Y. Demiris, “Magan: Margin           [213] X. Wang and A. Gupta, “Generative image modeling using style
      adaptation for generative adversarial networks,” arXiv preprint                and structure adversarial networks,” in European Conference on
      arXiv:1704.03817, 2017.                                                        Computer Vision, pp. 318–335, 2016.
[189] A. Krizhevsky, G. Hinton, et al., “Learning multiple layers of           [214] D. Xu, Y. Wu, S. Yuan, L. Zhang, and X. Wu, “Achieving causal
      features from tiny images,” tech. rep., Citeseer, 2009.                        fairness through generative adversarial networks,” in Interna-
[190] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, et al., “Gradient-based            tional Joint Conference on Artificial Intelligence, pp. 1452–1458, 2019.
      learning applied to document recognition,” Proceedings of the            [215] Z. Wang, G. Healy, A. F. Smeaton, and T. E. Ward, “Use of
      IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.                                    neural signals to evaluate the quality of generative adversarial
[191] J. Susskind, A. Anderson, and G. E. Hinton, “The toronto face                  network performance in facial image generation,” arXiv preprint
      dataset,” U. Toronto, Tech. Rep. UTML TR, vol. 1, p. 2010, 2010.               arXiv:1811.04172, 2018.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                           23

[216] Z. Wang, Q. She, A. F. Smeaton, T. E. Ward, and G. Healy,                    canonical adaptation networks,” in IEEE Conference on Computer
      “Neuroscore: A brain-inspired evaluation metric for generative               Vision and Pattern Recognition, pp. 12627–12637, 2019.
      adversarial networks,” arXiv preprint arXiv:1905.04243, 2019.          [239] L. Pinto, J. Davidson, and A. Gupta, “Supervision via com-
[217] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Re-            petition: Robot adversaries for learning tasks,” in International
      thinking the inception architecture for computer vision,” in IEEE            Conference on Robotics and Automation, pp. 1601–1608, 2017.
      Conference on Computer Vision and Pattern Recognition, pp. 2818–       [240] A. Anoosheh, E. Agustsson, R. Timofte, and L. Van Gool, “Com-
      2826, 2016.                                                                  bogan: Unrestrained scalability for image domain translation,”
[218] I. Danihelka, B. Lakshminarayanan, B. Uria, D. Wierstra, and                 in IEEE Conference on Computer Vision and Pattern Recognition
      P. Dayan, “Comparison of maximum likelihood and gan-based                    Workshops, pp. 783–790, 2018.
      training of real nvps,” arXiv preprint arXiv:1705.05263, 2017.         [241] X. Chen, C. Xu, X. Yang, and D. Tao, “Attention-gan for object
[219] Q. Xu, G. Huang, Y. Yuan, C. Guo, Y. Sun, F. Wu, and K. Wein-                transfiguration in wild images,” in European Conference on Com-
      berger, “An empirical study on evaluation metrics of generative              puter Vision, pp. 164–180, 2018.
      adversarial networks,” arXiv preprint arXiv:1806.07755, 2018.          [242] C. Wang, C. Xu, C. Wang, and D. Tao, “Perceptual adversarial
[220] M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton, “De-              networks for image-to-image transformation,” IEEE Transactions
      mystifying mmd gans,” arXiv preprint arXiv:1801.01401, 2018.                 on Image Processing, vol. 27, no. 8, pp. 4066–4079, 2018.
[221] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, et al., “Image   [243] J. Cao, H. Huang, Y. Li, J. Liu, R. He, and Z. Sun, “Biphasic
      quality assessment: from error visibility to structural similarity,”         learning of gans for high-resolution image-to-image translation,”
      IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612,          arXiv preprint arXiv:1904.06624, 2019.
      2004.                                                                  [244] M. Amodio and S. Krishnaswamy, “Travelgan: Image-to-image
[222] Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural           translation by transformation vector learning,” in IEEE Conference
      similarity for image quality assessment,” in Asilomar Conference             on Computer Vision and Pattern Recognition, pp. 8983–8992, 2019.
      on Signals, Systems & Computers, vol. 2, pp. 1398–1402, 2003.          [245] M.-Y. Liu, X. Huang, A. Mallya, T. Karras, T. Aila, J. Lehtinen, and
[223] W. Fedus, M. Rosca, B. Lakshminarayanan, A. M. Dai, S. Mo-                   J. Kautz, “Few-shot unsupervised image-to-image translation,”
      hamed, and I. Goodfellow, “Many paths to equilibrium: Gans do                arXiv preprint arXiv:1905.01723, 2019.
      not need to decrease a divergence at every step,” 2018.                [246] Z. He, W. Zuo, M. Kan, S. Shan, and X. Chen, “Attgan: Facial
[224] K. Kurach, M. Lučić, X. Zhai, M. Michalski, and S. Gelly, “A               attribute editing by only changing what you want,” IEEE Trans-
      large-scale study on regularization and normalization in gans,” in           actions on Image Processing, 2019.
      International Conference on Machine Learning, pp. 3581–3590, 2019.     [247] M. Liu, Y. Ding, M. Xia, X. Liu, E. Ding, W. Zuo, and S. Wen,
[225] L. Theis, A. v. d. Oord, and M. Bethge, “A note on the evaluation            “Stgan: A unified selective transfer network for arbitrary image
      of generative models,” in International Conference on Learning               attribute editing,” in IEEE Conference on Computer Vision and
      Representations, 2015.                                                       Pattern Recognition, pp. 3673–3682, 2019.
[226] M. Lucic, K. Kurach, M. Michalski, S. Gelly, and O. Bousquet,          [248] H. Edwards and A. Storkey, “Censoring representations with an
      “Are gans created equal? a large-scale study,” in Neural Informa-            adversary,” arXiv preprint arXiv:1511.05897, 2015.
      tion Processing Systems, pp. 700–709, 2018.                            [249] A. Beutel, J. Chen, Z. Zhao, and E. H. Chi, “Data decisions
[227] A. Borji, “Pros and cons of gan evaluation measures,” Computer               and theoretical implications when adversarially learning fair
      Vision and Image Understanding, vol. 179, pp. 41–65, 2019.                   representations,” arXiv preprint arXiv:1707.00075, 2017.
[228] E. Denton, S. Gross, and R. Fergus, “Semi-supervised learning          [250] B. H. Zhang, B. Lemoine, and M. Mitchell, “Mitigating unwanted
      with context-conditional generative adversarial networks,” arXiv             biases with adversarial learning,” in AAAI/ACM Conference on AI,
      preprint arXiv:1611.06430, 2016.                                             Ethics, and Society, pp. 335–340, 2018.
[229] M. Ding, J. Tang, and J. Zhang, “Semi-supervised learning on           [251] D. Madras, E. Creager, T. Pitassi, and R. Zemel, “Learning ad-
      graphs with generative adversarial nets,” in CM International                versarially fair and transferable representations,” in International
      Conference on Information and Knowledge Management, pp. 913–922,             Conference on Machine Learning, 2018.
      ACM, 2018.                                                             [252] L. Hu, M. Kan, S. Shan, and X. Chen, “Duplex generative ad-
[230] A. Odena, “Semi-supervised learning with generative adversarial              versarial network for unsupervised domain adaptation,” in IEEE
      networks,” arXiv preprint arXiv:1606.01583, 2016.                            Conference on Computer Vision and Pattern Recognition, pp. 1498–
[231] Z. Dai, Z. Yang, F. Yang, W. W. Cohen, and R. R. Salakhutdinov,              1507, 2018.
      “Good semi-supervised learning that requires a bad gan,” in            [253] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE
      Neural Information Processing Systems, pp. 6510–6520, 2017.                  Transactions on Knowledge and Data Engineering, vol. 22, no. 10,
[232] A. Madani, M. Moradi, A. Karargyris, and T. Syeda-Mahmood,                   pp. 1345–1359, 2009.
      “Semi-supervised learning with generative adversarial networks         [254] S. Sankaranarayanan, Y. Balaji, A. Jain, S. Nam Lim, and R. Chel-
      for chest x-ray classification with ability of data domain adapta-           lappa, “Learning from synthetic data: Addressing domain shift
      tion,” in International Symposium on Biomedical Imaging, pp. 1038–           for semantic segmentation,” in IEEE Conference on Computer Vi-
      1042, 2018.                                                                  sion and Pattern Recognition, pp. 3752–3761, 2018.
[233] T. Chen, X. Zhai, M. Ritter, M. Lucic, and N. Houlsby, “Self-          [255] Y.-H. Tsai, W.-C. Hung, S. Schulter, K. Sohn, M.-H. Yang, and
      supervised generative adversarial networks,” in IEEE Conference              M. Chandraker, “Learning to adapt structured output space for
      on Computer Vision and Pattern Recognition, 2019.                            semantic segmentation,” in IEEE Conference on Computer Vision
[234] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle,                 and Pattern Recognition, pp. 7472–7481, 2018.
      F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-                 [256] J. Shen, Y. Qu, W. Zhang, and Y. Yu, “Wasserstein distance guided
      adversarial training of neural networks,” The Journal of Machine             representation learning for domain adaptation,” arXiv preprint
      Learning Research, vol. 17, no. 1, pp. 2096–2030, 2016.                      arXiv:1707.01217, 2017.
[235] A. M. Lamb, A. G. A. P. Goyal, Y. Zhang, S. Zhang, A. C.               [257] S. Benaim and L. Wolf, “One-sided unsupervised domain map-
      Courville, and Y. Bengio, “Professor forcing: A new algorithm                ping,” in Neural Information Processing Systems, pp. 752–762, 2017.
      for training recurrent networks,” in Neural Information Processing     [258] H. Zhao, S. Zhang, G. Wu, J. M. Moura, J. P. Costeira, and G. J.
      Systems, pp. 4601–4609, 2016.                                                Gordon, “Adversarial multiple source domain adaptation,” in
[236] A. Shrivastava, T. Pfister, O. Tuzel, J. Susskind, W. Wang, and              Neural Information Processing Systems, pp. 8559–8570, 2018.
      R. Webb, “Learning from simulated and unsupervised images              [259] M. Long, Z. Cao, J. Wang, and M. I. Jordan, “Conditional ad-
      through adversarial training,” in IEEE Conference on Computer                versarial domain adaptation,” in Neural Information Processing
      Vision and Pattern Recognition, pp. 2107–2116, 2017.                         Systems, pp. 1640–1650, 2018.
[237] K. Bousmalis, A. Irpan, P. Wohlhart, Y. Bai, M. Kelcey, M. Kalakr-     [260] W. Hong, Z. Wang, M. Yang, and J. Yuan, “Conditional generative
      ishnan, L. Downs, J. Ibarz, P. Pastor, K. Konolige, et al., “Using           adversarial network for structured domain adaptation,” in IEEE
      simulation and domain adaptation to improve efficiency of deep               Conference on Computer Vision and Pattern Recognition, pp. 1335–
      robotic grasping,” in International Conference on Robotics and Au-           1344, 2018.
      tomation, pp. 4243–4250, 2018.                                         [261] K. Saito, K. Watanabe, Y. Ushiku, and T. Harada, “Maximum clas-
[238] S. James, P. Wohlhart, M. Kalakrishnan, D. Kalashnikov, A. Irpan,            sifier discrepancy for unsupervised domain adaptation,” in IEEE
      J. Ibarz, S. Levine, R. Hadsell, and K. Bousmalis, “Sim-to-real via          Conference on Computer Vision and Pattern Recognition, pp. 3723–
      sim-to-sim: Data-efficient robotic grasping via randomized-to-               3732, 2018.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                            24

[262] R. Volpi, P. Morerio, S. Savarese, and V. Murino, “Adversar-          [284] J. Zhang, Z. Wei, I. C. Duta, F. Shen, L. Liu, F. Zhu, X. Xu,
      ial feature augmentation for unsupervised domain adaptation,”               L. Shao, and H. T. Shen, “Generative reconstructive hashing for
      in IEEE Conference on Computer Vision and Pattern Recognition,              incomplete video analysis,” in ACM International Conference on
      pp. 5495–5504, 2018.                                                        Multimedia, pp. 845–854, ACM, 2019.
[263] X. Chen, S. Li, H. Li, S. Jiang, Y. Qi, and L. Song, “Gener-          [285] Y. Wang, L. Zhang, F. Nie, X. Li, Z. Chen, and F. Wang, “We-
      ative adversarial user model for reinforcement learning based               gan: Deep image hashing with weighted generative adversarial
      recommendation system,” in International Conference on Machine              networks,” IEEE Transactions on Multimedia, 2019.
      Learning, pp. 1052–1061, 2019.                                        [286] G. Dai, J. Xie, and Y. Fang, “Metric-based generative adversarial
[264] D. Pfau and O. Vinyals, “Connecting generative adversarial net-             network,” in ACM International Conference on Multimedia, pp. 672–
      works and actor-critic methods,” arXiv preprint arXiv:1610.01945,           680, 2017.
      2016.                                                                 [287] S. C.-X. Li, B. Jiang, and B. Marlin, “Misgan: Learning from
[265] C. Finn, P. Christiano, P. Abbeel, and S. Levine, “A connec-                incomplete data with generative adversarial networks,” in Inter-
      tion between generative adversarial networks, inverse rein-                 national Conference on Learning Representations, 2019.
      forcement learning, and energy-based models,” arXiv preprint          [288] C. Wang, C. Xu, X. Yao, and D. Tao, “Evolutionary generative
      arXiv:1611.03852, 2016.                                                     adversarial networks,” IEEE Transactions on Evolutionary Compu-
[266] Y. Ganin, T. Kulkarni, I. Babuschkin, S. Eslami, and O. Vinyals,            tation, 2019.
      “Synthesizing programs for images using reinforced adversar-
                                                                            [289] C. R. Ponce, W. Xiao, P. F. Schade, T. S. Hartmann, G. Kreiman,
      ial learning,” in International Conference on Machine Learning,
                                                                                  and M. S. Livingstone, “Evolving images for visual neurons
      pp. 1666–1675, 2018.
                                                                                  using a deep generative network reveals coding principles and
[267] T. Bansal, J. Pachocki, S. Sidor, I. Sutskever, and I. Mor-                 neuronal preferences,” Cell, vol. 177, no. 4, pp. 999–1009, 2019.
      datch, “Emergent complexity via multi-agent competition,” arXiv
                                                                            [290] Y. Wang, Y. Xia, T. He, F. Tian, T. Qin, C. Zhai, and T.-Y.
      preprint arXiv:1710.03748, 2017.
                                                                                  Liu, “Multi-agent dual learning,” in International Conference on
[268] J. Yoo, H. Ha, J. Yi, J. Ryu, C. Kim, J.-W. Ha, Y.-H. Kim,
                                                                                  Learning Representations, 2019.
      and S. Yoon, “Energy-based sequence gans for recommenda-
      tion and their connection to imitation learning,” arXiv preprint      [291] J.-J. Zhu and J. Bento, “Generative adversarial active learning,”
      arXiv:1706.09200, 2017.                                                     arXiv preprint arXiv:1702.07956, 2017.
[269] J. Ho and S. Ermon, “Generative adversarial imitation learning,”      [292] M.-K. Xie and S.-J. Huang, “Learning class-conditional gans with
      in Neural Information Processing Systems, pp. 4565–4573, 2016.              active sampling,” in ACM SIGKDD International Conference on
[270] Y. Guo, J. Oh, S. Singh, and H. Lee, “Generative adversarial self-          Knowledge Discovery & Data Mining, pp. 998–1006, ACM, 2019.
      imitation learning,” arXiv preprint arXiv:1812.00950, 2018.           [293] P. Grnarova, K. Y. Levy, A. Lucchi, T. Hofmann, and A. Krause,
[271] W. Shang, Y. Yu, Q. Li, Z. Qin, Y. Meng, and J. Ye, “Environ-               “An online learning approach to generative adversarial net-
      ment reconstruction with hidden confounders for reinforcement               works,” arXiv preprint arXiv:1706.03269, 2017.
      learning based recommendation,” in ACM SIGKDD International           [294] I. O. Tolstikhin, S. Gelly, O. Bousquet, C.-J. Simon-Gabriel, and
      Conference on Knowledge Discovery & Data Mining, pp. 566–576,               B. Schölkopf, “Adagan: Boosting generative models,” in Neural
      2019.                                                                       Information Processing Systems, pp. 5424–5433, 2017.
[272] J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang,     [295] Y. Zhu, M. Elhoseiny, B. Liu, X. Peng, and A. Elgammal, “A gen-
      and E. Shechtman, “Toward multimodal image-to-image transla-                erative adversarial approach for zero-shot learning from noisy
      tion,” in Neural Information Processing Systems, pp. 465–476, 2017.         texts,” in IEEE conference on computer vision and pattern recognition,
[273] A. Almahairi, S. Rajeswar, A. Sordoni, P. Bachman, and                      pp. 1004–1013, 2018.
      A. Courville, “Augmented cyclegan: Learning many-to-many              [296] P. Qin, X. Wang, W. Chen, C. Zhang, W. Xu, and W. Y. Wang,
      mappings from unpaired data,” arXiv preprint arXiv:1802.10151,              “Generative adversarial zero-shot relational learning for knowl-
      2018.                                                                       edge graphs,” in AAAI Conference on Artificial Intelligence, 2020.
[274] X. Huang, M.-Y. Liu, S. Belongie, and J. Kautz, “Multimodal un-       [297] P. Yang, Q. Tan, H. Tong, and J. He, “Task-adversarial co-
      supervised image-to-image translation,” in European Conference              generative nets,” in ACM SIGKDD International Conference on
      on Computer Vision (ECCV), pp. 172–189, 2018.                               Knowledge Discovery & Data Mining, pp. 1596–1604, ACM, 2019.
[275] H.-Y. Lee, H.-Y. Tseng, J.-B. Huang, M. Singh, and M.-H. Yang,        [298] I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT
      “Diverse image-to-image translation via disentangled represen-              press, 2016.
      tations,” in European Conference on Computer Vision, pp. 35–51,       [299] A. Srivastava, L. Valkov, C. Russell, M. U. Gutmann, and C. Sut-
      2018.                                                                       ton, “Veegan: Reducing mode collapse in gans using implicit
[276] W. Lotter, G. Kreiman, and D. Cox, “Unsupervised learning of                variational learning,” in Neural Information Processing Systems,
      visual structure using predictive generative networks,” arXiv               pp. 3308–3318, 2017.
      preprint arXiv:1511.06380, 2015.                                      [300] D. Bau, J.-Y. Zhu, J. Wulff, W. Peebles, H. Strobelt, B. Zhou, and
[277] J. Jordon, J. Yoon, and M. van der Schaar, “Knockoffgan: Gener-             A. Torralba, “Seeing what a gan cannot generate,” in Proceedings
      ating knockoffs for feature selection using generative adversarial          of the IEEE International Conference on Computer Vision, pp. 4502–
      networks,” in International Conference on Learning Representations,         4511, 2019.
      2019.
                                                                            [301] S. Arora, R. Ge, Y. Liang, T. Ma, and Y. Zhang, “Generalization
[278] J. Song, T. He, L. Gao, X. Xu, A. Hanjalic, and H. T. Shen, “Binary
                                                                                  and equilibrium in generative adversarial nets (gans),” in Inter-
      generative adversarial networks for image retrieval,” in AAAI
                                                                                  national Conference on Machine Learning, pp. 224–232, 2017.
      Conference on Artificial Intelligence, pp. 394–401, 2018.
                                                                            [302] K. Roth, A. Lucchi, S. Nowozin, and T. Hofmann, “Stabilizing
[279] J. Zhang, Y. Peng, and M. Yuan, “Unsupervised generative ad-
                                                                                  training of generative adversarial networks through regulariza-
      versarial cross-modal hashing,” in AAAI Conference on Artificial
                                                                                  tion,” in Neural Information Processing Systems, pp. 2018–2028,
      Intelligence, pp. 539–546, 2018.
                                                                                  2017.
[280] J. Zhang, Y. Peng, and M. Yuan, “Sch-gan: Semi-supervised
      cross-modal hashing by generative adversarial network,” IEEE          [303] L. Mescheder, S. Nowozin, and A. Geiger, “The numerics of
      Transactions on Cybernetics, vol. 50, no. 2, pp. 489–502, 2020.             gans,” in Neural Information Processing Systems, pp. 1825–1835,
[281] K. Ghasedi Dizaji, F. Zheng, N. Sadoughi, Y. Yang, C. Deng, and             2017.
      H. Huang, “Unsupervised deep generative adversarial hashing           [304] L. Mescheder, A. Geiger, and S. Nowozin, “Which training meth-
      network,” in IEEE Conference on Computer Vision and Pattern                 ods for gans do actually converge?,” in International Conference on
      Recognition, pp. 3664–3673, 2018.                                           Machine Learning, pp. 3481–3490, 2018.
[282] G. Wang, Q. Hu, J. Cheng, and Z. Hou, “Semi-supervised gen-           [305] Z. Lin, A. Khetan, G. Fanti, and S. Oh, “Pacgan: The power of
      erative adversarial hashing for image retrieval,” in Proceedings of         two samples in generative adversarial networks,” in Advances in
      the European Conference on Computer Vision (ECCV), pp. 469–485,             Neural Information Processing Systems, pp. 1498–1507, 2018.
      2018.                                                                 [306] S. Arora, A. Risteski, and Y. Zhang, “Do gans learn the distri-
[283] Z. Qiu, Y. Pan, T. Yao, and T. Mei, “Deep semantic hashing with             bution? some theory and empirics,” in International Conference on
      generative adversarial networks,” in ACM SIGIR Conference on                Learning Representations, 2018.
      Research and Development in Information Retrieval, pp. 225–234,       [307] Y. Bai, T. Ma, and A. Risteski, “Approximability of discriminators
      ACM, 2017.                                                                  implies diversity in gans,” arXiv preprint arXiv:1806.10586, 2018.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                           25

[308] T. Liang, “On how well generative adversarial networks learn           [330] B. Dolhansky and C. Canton Ferrer, “Eye in-painting with ex-
      densities: Nonparametric and parametric results,” arXiv preprint             emplar generative adversarial networks,” in IEEE Conference on
      arXiv:1811.03179, 2018.                                                      Computer Vision and Pattern Recognition, pp. 7902–7911, 2018.
[309] S. Singh, A. Uppal, B. Li, C.-L. Li, M. Zaheer, and B. Póczos,        [331] A. Pumarola, A. Agudo, A. M. Martinez, A. Sanfeliu, and
      “Nonparametric density estimation with adversarial losses,” in               F. Moreno-Noguer, “Ganimation: Anatomically-aware facial ani-
      Neural Information Processing Systems, pp. 10246–10257, Curran               mation from a single image,” in European Conference on Computer
      Associates Inc., 2018.                                                       Vision, pp. 818–833, 2018.
[310] S. Arora, A. Risteski, and Y. Zhang, “Theoretical limita-              [332] W. Yin, Y. Fu, L. Sigal, and X. Xue, “Semi-latent gan: Learning to
      tions of encoder-decoder gan architectures,” arXiv preprint                  generate and modify facial images from attributes,” arXiv preprint
      arXiv:1711.02651, 2017.                                                      arXiv:1704.02166, 2017.
[311] A. Creswell and A. A. Bharath, “Inverting the generator of             [333] C. Donahue, Z. C. Lipton, A. Balsubramani, and J. McAuley,
      a generative adversarial network,” IEEE Transactions on Neural               “Semantically decomposing the latent spaces of generative ad-
      Networks and Learning Systems, vol. 30, no. 7, pp. 1967–1974, 2019.          versarial networks,” in International Conference on Learning Repre-
[312] S. Mohamed and B. Lakshminarayanan, “Learning in implicit                    sentations, 2018.
      generative models,” arXiv preprint arXiv:1610.03483, 2016.             [334] A. Duarte, F. Roldan, M. Tubau, J. Escur, S. Pascual, A. Salvador,
[313] G. Gidel, H. Berard, G. Vignoud, P. Vincent, and S. Lacoste-Julien,          E. Mohedano, K. McGuinness, J. Torres, and X. Giro-i Nieto,
      “A variational inequality perspective on generative adversarial              “Wav2pix: speech-conditioned face generation using generative
      networks,” in International Conference on Learning Representations,          adversarial networks,” in International Conference on Acoustics,
      2019.                                                                        Speech and Signal Processing, vol. 3, 2019.
[314] M. Sanjabi, J. Ba, M. Razaviyayn, and J. D. Lee, “On the conver-       [335] B. Gecer, S. Ploumpis, I. Kotsia, and S. Zafeiriou, “Ganfit: Gen-
      gence and robustness of training gans with regularized optimal               erative adversarial network fitting for high fidelity 3d face re-
      transport,” in Neural Information Processing Systems, pp. 7091–              construction,” in IEEE Conference on Computer Vision and Pattern
      7101, 2018.                                                                  Recognition, pp. 1155–1164, 2019.
[315] V. Nagarajan, C. Raffel, and I. J. Goodfellow, “Theoretical insights   [336] Z. Shu, M. Sahasrabudhe, R. Alp Guler, D. Samaras, N. Paragios,
      into memorization in gans,” in Neural Information Processing Sys-            and I. Kokkinos, “Deforming autoencoders: Unsupervised dis-
      tems Workshop, 2018.                                                         entangling of shape and appearance,” in European Conference on
[316] Y. Blau, R. Mechrez, R. Timofte, T. Michaeli, and L. Zelnik-Manor,           Computer Vision, pp. 650–665, 2018.
      “The 2018 pirm challenge on perceptual image super-resolution,”        [337] C. Fu, X. Wu, Y. Hu, H. Huang, and R. He, “Dual variational
      in European Conference on Computer Vision Workshops, pp. 334–355,            generation for low-shot heterogeneous face recognition,” in Neu-
      2018.                                                                        ral Information Processing Systems, 2019.
[317] X. Yu and F. Porikli, “Ultra-resolving face images by discrimi-        [338] Y. Lu, Y.-W. Tai, and C.-K. Tang, “Attribute-guided face gen-
      native generative networks,” in European conference on computer              eration using conditional cyclegan,” in European Conference on
      vision, pp. 318–333, Springer, 2016.                                         Computer Vision, pp. 282–297, 2018.
[318] H. Zhu, A. Zheng, H. Huang, and R. He, “High-resolution talking        [339] J. Cao, Y. Hu, B. Yu, R. He, and Z. Sun, “3d aided duet gans for
      face generation via mutual information approximation,” arXiv                 multi-view face image synthesis,” IEEE Transactions on Informa-
      preprint arXiv:1812.06589, 2018.                                             tion Forensics and Security, vol. 14, no. 8, pp. 2028–2042, 2019.
[319] H. Huang, R. He, Z. Sun, and T. Tan, “Wavelet domain generative        [340] Y. Liu, Q. Li, and Z. Sun, “Attribute-aware face aging with
      adversarial network for multi-scale face hallucination,” Interna-            wavelet-based generative adversarial networks,” in IEEE Confer-
      tional Journal of Computer Vision, vol. 127, no. 6-7, pp. 763–784,           ence on Computer Vision and Pattern Recognition, pp. 11877–11886,
      2019.                                                                        2019.
[320] C. K. Sønderby, J. Caballero, L. Theis, W. Shi, and F. Huszár,        [341] J. Bao, D. Chen, F. Wen, H. Li, and G. Hua, “Cvae-gan: fine-
      “Amortised map inference for image super-resolution,” in Inter-              grained image generation through asymmetric training,” in IEEE
      national Conference on Learning Representations, 2017.                       International Conference on Computer Vision, pp. 2745–2754, 2017.
[321] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-     [342] H. Dong, S. Yu, C. Wu, and Y. Guo, “Semantic image synthesis via
      time style transfer and super-resolution,” in European Conference            adversarial learning,” in IEEE International Conference on Computer
      on Computer Vision, pp. 694–711, Springer, 2016.                             Vision, pp. 5706–5714, 2017.
[322] X. Wang, K. Yu, C. Dong, and C. Change Loy, “Recovering                [343] J. Wu, C. Zhang, T. Xue, B. Freeman, and J. Tenenbaum, “Learn-
      realistic texture in image super-resolution by deep spatial feature          ing a probabilistic latent space of object shapes via 3d generative-
      transform,” in IEEE Conference on Computer Vision and Pattern                adversarial modeling,” in Neural Information Processing Systems,
      Recognition, pp. 606–615, 2018.                                              pp. 82–90, 2016.
[323] W. Zhang, Y. Liu, C. Dong, and Y. Qiao, “Ranksrgan: Generative         [344] D. J. Im, C. D. Kim, H. Jiang, and R. Memisevic, “Generat-
      adversarial networks with ranker for image super-resolution,” in             ing images with recurrent adversarial networks,” arXiv preprint
      International Conference on Computer Vision, 2019.                           arXiv:1602.05110, 2016.
[324] L. Tran, X. Yin, and X. Liu, “Disentangled representation learning     [345] J. Yang, A. Kannan, D. Batra, and D. Parikh, “Lr-gan: Layered
      gan for pose-invariant face recognition,” in IEEE Conference on              recursive generative adversarial networks for image generation,”
      Computer Vision and Pattern Recognition, pp. 1415–1424, 2017.                arXiv preprint arXiv:1703.01560, 2017.
[325] J. Cao, Y. Hu, H. Zhang, R. He, and Z. Sun, “Learning a high           [346] X. Wang, A. Shrivastava, and A. Gupta, “A-fast-rcnn: Hard
      fidelity pose invariant model for high-resolution face frontal-              positive generation via adversary for object detection,” in IEEE
      ization,” in Advances in Neural Information Processing Systems,              Conference on Computer Vision and Pattern Recognition, pp. 2606–
      pp. 2867–2877, 2018.                                                         2615, 2017.
[326] A. Siarohin, E. Sangineto, S. Lathuilière, and N. Sebe, “De-          [347] R. Villegas, J. Yang, S. Hong, X. Lin, and H. Lee, “Decomposing
      formable gans for pose-based human image generation,” in IEEE                motion and content for natural video sequence prediction,” arXiv
      Conference on Computer Vision and Pattern Recognition, pp. 3408–             preprint arXiv:1706.08033, 2017.
      3416, 2018.                                                            [348] E. Santana and G. Hotz, “Learning a driving simulator,” arXiv
[327] C. Wang, C. Wang, C. Xu, and D. Tao, “Tag disentangled gen-                  preprint arXiv:1608.01230, 2016.
      erative adversarial networks for object image re-rendering,” in        [349] C. Chan, S. Ginosar, T. Zhou, and A. A. Efros, “Everybody dance
      International Joint Conference on Aritifical Intelligence, 2017.             now,” arXiv preprint arXiv:1808.07371, 2018.
[328] Z. Shu, E. Yumer, S. Hadap, K. Sunkavalli, E. Shechtman, and           [350] A. Clark, J. Donahue, and K. Simonyan, “Efficient video genera-
      D. Samaras, “Neural face editing with intrinsic image disen-                 tion on complex datasets,” arXiv preprint arXiv:1907.06571, 2019.
      tangling,” in IEEE Conference on Computer Vision and Pattern           [351] M. Mathieu, C. Couprie, and Y. LeCun, “Deep multi-scale
      Recognition, pp. 5541–5550, 2017.                                            video prediction beyond mean square error,” arXiv preprint
[329] H. Chang, J. Lu, F. Yu, and A. Finkelstein, “Pairedcyclegan:                 arXiv:1511.05440, 2015.
      Asymmetric style transfer for applying and removing makeup,”           [352] X. Liang, L. Lee, W. Dai, and E. P. Xing, “Dual motion gan for
      in IEEE Conference on Computer Vision and Pattern Recognition,               future-flow embedded video prediction,” in IEEE International
      pp. 40–48, 2018.                                                             Conference on Computer Vision, pp. 1744–1752, 2017.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                             26

[353] A. Bansal, S. Ma, D. Ramanan, and Y. Sheikh, “Recycle-gan: Un-          [376] F. Fang, J. Yamagishi, I. Echizen, and J. Lorenzo-Trueba, “High-
      supervised video retargeting,” in European Conference on Computer             quality nonparallel voice conversion based on cycle-consistent
      Vision, pp. 119–135, 2018.                                                    adversarial network,” in IEEE International Conference on Acous-
[354] X. Liang, H. Zhang, L. Lin, and E. Xing, “Generative semantic                 tics, Speech and Signal Processing, pp. 5279–5283, 2018.
      manipulation with mask-contrasting gan,” in European Conference         [377] T. Kaneko and H. Kameoka, “Parallel-data-free voice conver-
      on Computer Vision, pp. 558–573, 2018.                                        sion using cycle-consistent adversarial networks,” arXiv preprint
[355] T. Kim, B. Kim, M. Cha, and J. Kim, “Unsupervised visual                      arXiv:1711.11293, 2017.
      attribute transfer with reconfigurable generative adversarial net-      [378] C. Esteban, S. L. Hyland, and G. Rätsch, “Real-valued (medical)
      works,” arXiv preprint arXiv:1707.09798, 2017.                                time series generation with recurrent conditional gans,” arXiv
[356] Y. Chen, Y.-K. Lai, and Y.-J. Liu, “Cartoongan: Generative adver-             preprint arXiv:1706.02633, 2017.
      sarial networks for photo cartoonization,” in IEEE Conference on        [379] K. G. Hartmann, R. T. Schirrmeister, and T. Ball, “Eeg-gan:
      Computer Vision and Pattern Recognition, pp. 9465–9474, 2018.                 Generative adversarial networks for electroencephalograhic (eeg)
[357] R. Villegas, J. Yang, D. Ceylan, and H. Lee, “Neural kinematic                brain signals,” arXiv preprint arXiv:1806.01875, 2018.
      networks for unsupervised motion retargetting,” in IEEE Confer-         [380] C. Donahue, J. McAuley, and M. Puckette, “Synthesizing
      ence on Computer Vision and Pattern Recognition, pp. 8639–8648,               audio with generative adversarial networks,” arXiv preprint
      2018.                                                                         arXiv:1802.04208, vol. 1, 2018.
[358] S. Zhou, T. Xiao, Y. Yang, D. Feng, Q. He, and W. He, “Gene-            [381] D. Li, D. Chen, B. Jin, L. Shi, J. Goh, and S.-K. Ng, “Mad-
      gan: Learning object transfiguration and attribute subspace from              gan: Multivariate anomaly detection for time series data with
      unpaired data,” arXiv preprint arXiv:1705.04932, 2017.                        generative adversarial networks,” in International Conference on
[359] H. Wu, S. Zheng, J. Zhang, and K. Huang, “Gp-gan: Towards                     Artificial Neural Networks, pp. 703–716, Springer, 2019.
      realistic high-resolution image blending,” 2019.                        [382] J. Li, W. Monroe, T. Shi, S. Jean, A. Ritter, and D. Jurafsky, “Ad-
[360] N. Souly, C. Spampinato, and M. Shah, “Semi supervised seman-                 versarial learning for neural dialogue generation,” arXiv preprint
      tic segmentation using generative adversarial network,” in IEEE               arXiv:1701.06547, 2017.
      International Conference on Computer Vision, pp. 5688–5696, 2017.       [383] Y. Zhang, Z. Gan, and L. Carin, “Generating text via adversarial
[361] J. Pan, C. C. Ferrer, K. McGuinness, N. E. O’Connor, J. Tor-                  training,” in NIPS workshop on Adversarial Training, vol. 21, 2016.
      res, E. Sayrol, and X. Giro-i Nieto, “Salgan: Visual saliency           [384] W. Fedus, I. Goodfellow, and A. M. Dai, “Maskgan: better text
      prediction with generative adversarial networks,” arXiv preprint              generation via filling in the ,” arXiv preprint arXiv:1801.07736,
      arXiv:1701.01081, 2017.                                                       2018.
[362] Y. Song, C. Ma, X. Wu, L. Gong, L. Bao, W. Zuo, C. Shen, R. W.          [385] S. Yang, J. Liu, W. Wang, and Z. Guo, “Tet-gan: Text effects
      Lau, and M.-H. Yang, “Vital: Visual tracking via adversarial                  transfer via stylization and destylization,” in AAAI Conference on
      learning,” in IEEE Conference on Computer Vision and Pattern                  Artificial Intelligence, pp. 1238–1245, 2019.
      Recognition, pp. 8990–8999, 2018.                                       [386] L. Cai and W. Y. Wang, “Kbgan: Adversarial learning for knowl-
[363] Y. Han, P. Zhang, W. Huang, Y. Zha, G. D. Cooper, and Y. Zhang,               edge graph embeddings,” arXiv preprint arXiv:1711.04071, 2017.
      “Robust visual tracking using unlabeled adversarial instance            [387] X. Wang, W. Chen, Y.-F. Wang, and W. Y. Wang, “No metrics
      generation and regularized label smoothing,” Pattern Recognition,             are perfect: Adversarial reward learning for visual storytelling,”
      pp. 1–15, 2019.                                                               arXiv preprint arXiv:1804.09160, 2018.
[364] D. Engin, A. Genç, and H. Kemal Ekenel, “Cycle-dehaze: En-             [388] P. Qin, W. Xu, and W. Y. Wang, “Dsgan: generative adversarial
      hanced cyclegan for single image dehazing,” in IEEE Conference                training for distant supervision relation extraction,” arXiv preprint
      on Computer Vision and Pattern Recognition Workshops, pp. 825–833,            arXiv:1805.09929, 2018.
      2018.                                                                   [389] C. d. M. d’Autume, M. Rosca, J. Rae, and S. Mohamed, “Training
[365] X. Yang, Z. Xu, and J. Luo, “Towards perceptual image dehazing                language gans from scratch,” arXiv preprint arXiv:1905.09922,
      by physics-based disentanglement and adversarial training,” in                2019.
      AAAI conference on artificial intelligence, pp. 7485–7492, 2018.        [390] A. Dash, J. C. B. Gamboa, S. Ahmed, M. Liwicki, and M. Z.
[366] W. Liu, X. Hou, J. Duan, and G. Qiu, “End-to-end single image                 Afzal, “Tac-gan-text conditioned auxiliary classifier generative
      fog removal using enhanced cycle consistent adversarial net-                  adversarial network,” arXiv preprint arXiv:1703.06412, 2017.
      works,” arXiv preprint arXiv:1902.01374, 2019.                          [391] T.-H. Chen, Y.-H. Liao, C.-Y. Chuang, W.-T. Hsu, J. Fu, and
[367] S. Lutz, K. Amplianitis, and A. Smolic, “Alphagan: Generative                 M. Sun, “Show, adapt and tell: Adversarial training of cross-
      adversarial networks for natural image matting,” arXiv preprint               domain image captioner,” in IEEE International Conference on
      arXiv:1807.10088, 2018.                                                       Computer Vision, pp. 521–530, 2017.
[368] R. A. Yeh, C. Chen, T. Yian Lim, A. G. Schwing, M. Hasegawa-            [392] R. Shetty, M. Rohrbach, L. Anne Hendricks, M. Fritz, and
      Johnson, and M. N. Do, “Semantic image inpainting with deep                   B. Schiele, “Speaking the same language: Matching machine to
      generative models,” in IEEE Conference on Computer Vision and                 human captions by adversarial training,” in IEEE International
      Pattern Recognition, pp. 5485–5493, 2017.                                     Conference on Computer Vision, pp. 4135–4144, 2017.
[369] J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang, “Generative    [393] S. Rao and H. Daumé III, “Answer-based adversarial train-
      image inpainting with contextual attention,” in IEEE Conference               ing for generating clarification questions,” arXiv preprint
      on Computer Vision and Pattern Recognition, pp. 5505–5514, 2018.              arXiv:1904.02281, 2019.
[370] X. Liu, Y. Wang, and Q. Liu, “Psgan: a generative adversarial           [394] X. Yang, M. Khabsa, M. Wang, W. Wang, A. Awadallah, D. Kifer,
      network for remote sensing image pan-sharpening,” in IEEE                     and C. L. Giles, “Adversarial training for community question
      International Conference on Image Processing, pp. 873–877, IEEE,              answer selection based on multi-scale matching,” 2019.
      2018.                                                                   [395] B. Liu, J. Fu, M. P. Kato, and M. Yoshikawa, “Beyond narrative
[371] S. Iizuka, E. Simo-Serra, and H. Ishikawa, “Globally and locally              description: Generating poetry from images by multi-adversarial
      consistent image completion,” ACM Transactions on Graphics,                   training,” in ACM Multimedia Conference on Multimedia Conference,
      vol. 36, no. 4, p. 107, 2017.                                                 pp. 783–791, ACM, 2018.
[372] F. Liu, L. Jiao, and X. Tang, “Task-oriented gan for polsar image       [396] Y. Luo, H. Zhang, Y. Wen, and X. Zhang, “Resumegan: An
      classification and clustering,” IEEE Transactions on Neural Net-              optimized deep representation learning framework for talent-
      works and Learning Systems, vol. 30, no. 9, pp. 2707–2719, 2019.              job fit via adversarial learning,” in Proceedings of the 28th ACM
[373] A. Creswell and A. A. Bharath, “Adversarial training for sketch               International Conference on Information and Knowledge Management,
      retrieval,” in European Conference on Computer Vision, pp. 798–809,           pp. 1101–1110, ACM, 2019.
      2016.                                                                   [397] C. Garbacea, S. Carton, S. Yan, and Q. Mei, “Judge the judges:
[374] M. Zhang, K. Teck Ma, J. Hwee Lim, Q. Zhao, and J. Feng,                      A large-scale evaluation study of neural language models for
      “Deep future gaze: Gaze anticipation on egocentric videos using               online review generation,” in Conference on Empirical Methods
      adversarial networks,” in IEEE Conference on Computer Vision and              in Natural Language Processing & International Joint Conference on
      Pattern Recognition, pp. 4372–4381, 2017.                                     Natural Language Processing, 2019.
[375] M. Zhang, K. T. Ma, J. Lim, Q. Zhao, and J. Feng, “Anticipating         [398] H. Aghakhani, A. Machiry, S. Nilizadeh, C. Kruegel, and G. Vi-
      where people will look using adversarial networks,” IEEE Trans-               gna, “Detecting deceptive reviews using generative adversarial
      actions on Pattern Analysis and Machine Intelligence, vol. 41, no. 8,         networks,” in IEEE Security and Privacy Workshops, pp. 89–95,
      pp. 1783–1796, 2019.                                                          2018.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                                                                                           27

[399] K. Takuhiro, K. Hirokazu, H. Nobukatsu, I. Yusuke, H. Kaoru,                 with variational info generative adversarial networks,” arXiv
      and K. Kunio, “Generative adversarial network-based postfil-                 preprint arXiv:1701.04568, 2017.
      tering for statistical parametric speech synthesis,” in IEEE In-       [420] X. Wang, Z. Man, M. You, and C. Shen, “Adversarial generation
      ternational Conference on Acoustics, Speech and Signal Processing,           of training examples: applications to moving vehicle license plate
      pp. 4910–4914, 2017.                                                         recognition,” arXiv preprint arXiv:1707.03124, 2017.
[400] Y. Saito, S. Takamichi, and H. Saruwatari, “Statistical parametric     [421] B. Chang, Q. Zhang, S. Pan, and L. Meng, “Generating handwrit-
      speech synthesis incorporating generative adversarial networks,”             ten chinese characters using cyclegan,” in IEEE Winter Conference
      IEEE/ACM Transactions on Audio, Speech, and Language Processing,             on Applications of Computer Vision, pp. 199–207, 2018.
      vol. 26, no. 1, pp. 84–96, 2018.                                       [422] L. Sixt, B. Wild, and T. Landgraf, “Rendergan: Generating realistic
[401] C. Donahue, J. McAuley, and M. Puckette, “Adversarial audio                  labeled data,” Frontiers in Robotics and AI, vol. 5, p. 66, 2018.
      synthesis,” arXiv preprint arXiv:1802.04208, 2018.                     [423] D. Xu, S. Yuan, L. Zhang, and X. Wu, “Fairgan: Fairness-aware
[402] S. Pascual, A. Bonafonte, and J. Serra, “Segan: Speech enhance-              generative adversarial networks,” in IEEE International Conference
      ment generative adversarial network,” in Interspeech, pp. 3642–              on Big Data, pp. 570–575, 2018.
      3646, 2017.                                                            [424] M.-C. Lee, B. Gao, and R. Zhang, “Rare query expansion through
[403] C. Donahue, B. Li, and R. Prabhavalkar, “Exploring speech                    generative adversarial networks in search advertising,” in ACM
      enhancement with generative adversarial networks for robust                  SIGKDD International Conference on Knowledge Discovery & Data
      speech recognition,” in IEEE International Conference on Acoustics,          Mining, pp. 500–508, ACM, 2018.
      Speech and Signal Processing, pp. 5024–5028, 2018.                     [425] M. O. Turkoglu, W. Thong, L. Spreeuwers, and B. Kicanaoglu,
[404] N. Killoran, L. J. Lee, A. Delong, D. Duvenaud, and B. J. Frey,              “A layer-based sequential framework for scene generation with
      “Generating and designing dna with deep generative models,”                  gans,” arXiv preprint arXiv:1902.00671, 2019.
      arXiv preprint arXiv:1712.06148, 2017.                                 [426] A. El-Nouby, S. Sharma, H. Schulz, D. Hjelm, L. El Asri, S. E.
[405] A. Gupta and J. Zou, “Feedback gan (fbgan) for dna: A novel                  Kahou, Y. Bengio, and G. W. Taylor, “Tell, draw, and repeat:
      feedback-loop architecture for optimizing protein functions,”                Generating and modifying images based on continual linguistic
      arXiv preprint arXiv:1804.01694, 2018.                                       instruction,” in International Conference on Computer Vision, 2019.
[406] M. Benhenda, “Chemgan challenge for drug discovery: can                [427] N. Ratzlaff and L. Fuxin, “Hypergan: A generative model for
      ai reproduce natural chemical diversity?,” arXiv preprint                    diverse, performant neural networks,” in International Conference
      arXiv:1708.08227, 2017.                                                      on Machine Learning, pp. 5361–5369, 2019.
[407] E. Choi, S. Biswal, B. Malin, J. Duke, W. F. Stewart, and J. Sun,      [428] M. Frid-Adar, I. Diamant, E. Klang, M. Amitai, J. Goldberger, and
      “Generating multi-label discrete patient records using generative            H. Greenspan, “Gan-based synthetic medical image augmenta-
      adversarial networks,” arXiv preprint arXiv:1703.06490, 2017.                tion for increased cnn performance in liver lesion classification,”
[408] W. Dai, J. Doyle, X. Liang, H. Zhang, N. Dong, Y. Li, and E. P.              Neurocomputing, vol. 321, pp. 321–331, 2018.
      Xing, “Scan: Structure correcting adversarial network for chest x-     [429] Q. Wang, H. Yin, H. Wang, Q. V. H. Nguyen, Z. Huang, and
      rays organ segmentation,” arXiv preprint arXiv:1703.08770, vol. 1,           L. Cui, “Enhancing collaborative filtering with generative aug-
      2017.                                                                        mentation,” in Proceedings of the 25th ACM SIGKDD International
[409] T. Schlegl, P. Seeböck, S. M. Waldstein, U. Schmidt-Erfurth, and            Conference on Knowledge Discovery & Data Mining, pp. 548–556,
      G. Langs, “Unsupervised anomaly detection with generative                    2019.
      adversarial networks to guide marker discovery,” in International      [430] Y. Zhang, Y. Fu, P. Wang, X. Li, and Y. Zheng, “Unifying inter-
      Conference on Information Processing in Medical Imaging, pp. 146–            region autocorrelation and intra-region structures for spatial
      157, Springer, 2017.                                                         embedding via collective adversarial learning,” in ACM SIGKDD
[410] J. M. Wolterink, A. M. Dinkla, M. H. Savenije, P. R. Seevinck,               International Conference on Knowledge Discovery & Data Mining,
      C. A. van den Berg, and I. Išgum, “Deep mr to ct synthesis                  pp. 1700–1708, 2019.
      using unpaired data,” in International Workshop on Simulation and      [431] H. Gao, J. Pei, and H. Huang, “Progan: Network embedding
      Synthesis in Medical Imaging, pp. 14–23, 2017.                               via proximity generative adversarial network,” in Proceedings
[411] T. M. Quan, T. Nguyen-Duc, and W.-K. Jeong, “Compressed sens-                of the 25th ACM SIGKDD International Conference on Knowledge
      ing mri reconstruction using a generative adversarial network                Discovery & Data Mining, pp. 1308–1316, 2019.
      with a cyclic loss,” IEEE Transactions on Medical Imaging, vol. 37,    [432] B. Hu, Y. Fang, and C. Shi, “Adversarial learning on hetero-
      no. 6, pp. 1488–1497, 2018.                                                  geneous information networks,” in ACM SIGKDD International
[412] M. Mardani, E. Gong, J. Y. Cheng, S. S. Vasanawala, G. Za-                   Conference on Knowledge Discovery & Data Mining, pp. 120–129,
      harchuk, L. Xing, and J. M. Pauly, “Deep generative adversarial              2019.
      neural networks for compressive sensing mri,” IEEE Transactions        [433] P. Wang, Y. Fu, H. Xiong, and X. Li, “Adversarial substructured
      on Medical Imaging, vol. 38, no. 1, pp. 167–179, 2018.                       representation learning for mobile user profiling,” in Proceedings
[413] Y. Xue, T. Xu, H. Zhang, L. R. Long, and X. Huang, “Segan:                   of the 25th ACM SIGKDD International Conference on Knowledge
      Adversarial network with multi-scale l1 loss for medical image               Discovery & Data Mining, pp. 130–138, 2019.
      segmentation,” Neuroinformatics, vol. 16, no. 3-4, pp. 383–392,        [434] W. Hu and Y. Tan, “Generating adversarial malware examples for
      2018.                                                                        black-box attacks based on gan,” arXiv preprint arXiv:1702.05983,
[414] Q. Yang, P. Yan, Y. Zhang, H. Yu, Y. Shi, X. Mou, M. K. Kalra,               2017.
      Y. Zhang, L. Sun, and G. Wang, “Low-dose ct image denoising            [435] M. Chidambaram and Y. Qi, “Style transfer generative adversar-
      using a generative adversarial network with wasserstein dis-                 ial networks: Learning to play chess differently,” arXiv preprint
      tance and perceptual loss,” IEEE Transactions on Medical Imaging,            arXiv:1702.06762, 2017.
      vol. 37, no. 6, pp. 1348–1357, 2018.                                   [436] C. Chu, A. Zhmoginov, and M. Sandler, “Cyclegan, a master of
[415] G. St-Yves and T. Naselaris, “Generative adversarial networks                steganography,” arXiv preprint arXiv:1712.02950, 2017.
      conditioned on brain activity reconstruct seen images,” in IEEE        [437] D. Volkhonskiy, I. Nazarov, B. Borisenko, and E. Burnaev,
      International Conference on Systems, Man, and Cybernetics, pp. 1054–         “Steganographic generative adversarial networks,” arXiv preprint
      1061, 2018.                                                                  arXiv:1703.05502, 2017.
[416] J.-J. Hwang, S. Azernikov, A. A. Efros, and S. X. Yu, “Learn-          [438] H. Shi, J. Dong, W. Wang, Y. Qian, and X. Zhang, “Ssgan: secure
      ing beyond human expertise with generative models for dental                 steganography based on generative adversarial networks,” in
      restorations,” arXiv preprint arXiv:1804.00064, 2018.                        Pacific Rim Conference on Multimedia, pp. 534–544, Springer, 2017.
[417] B. Tian, Y. Zhang, X. Chen, C. Xing, and C. Li, “Drgan: A gan-         [439] J. Hayes and G. Danezis, “Generating steganographic images
      based framework for doctor recommendation in chinese on-line                 via adversarial training,” in Neural Information Processing Systems,
      qa communities,” in International Conference on Database Systems             pp. 1954–1963, 2017.
      for Advanced Applications, pp. 444–447, 2019.                          [440] M. Abadi and D. G. Andersen, “Learning to protect commu-
[418] Z. Zheng, L. Zheng, and Y. Yang, “Unlabeled samples generated                nications with adversarial neural cryptography,” arXiv preprint
      by gan improve the person re-identification baseline in vitro,” in           arXiv:1610.06918, 2016.
      IEEE International Conference on Computer Vision, pp. 3754–3762,       [441] A. N. Gomez, S. Huang, I. Zhang, B. M. Li, M. Osama, and
      2017.                                                                        L. Kaiser, “Unsupervised cipher cracking using discrete gans,”
[419] M. Gorijala and A. Dukkipati, “Image generation and editing                  arXiv preprint arXiv:1801.04883, 2018.
JOURNAL OF LATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015                    28

[442] B. K. Beaulieu-Jones, Z. S. Wu, C. Williams, R. Lee, S. P. Bhavnani,
      J. B. Byrd, and C. S. Greene, “Privacy-preserving generative
      deep neural networks support clinical data sharing,” BioRxiv,
      p. 159756, 2018.
[443] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi, “Social
      gan: Socially acceptable trajectories with generative adversarial
      networks,” in IEEE Conference on Computer Vision and Pattern
      Recognition, pp. 2255–2264, 2018.
[444] H. Shu, Y. Wang, X. Jia, K. Han, H. Chen, C. Xu, Q. Tian,
      and C. Xu, “Co-evolutionary compression for unpaired image
      translation,” in International Conference on Computer Vision, 2019.
[445] S. Lin, R. Ji, C. Yan, B. Zhang, L. Cao, Q. Ye, F. Huang, and
      D. Doermann, “Towards optimal structured cnn pruning via
      generative adversarial learning,” in IEEE Conference on Computer
      Vision and Pattern Recognition, pp. 2790–2799, 2019.
[446] C. Daskalakis, A. Ilyas, V. Syrgkanis, and H. Zeng, “Training gans
      with optimism,” 2018.
[447] A. Bora, E. Price, and A. G. Dimakis, “Ambientgan: Generative
      models from lossy measurements,” in International Conference on
      Learning Representations, 2018.
[448] M. J. Kusner and J. M. Hernández-Lobato, “Gans for sequences
      of discrete elements with the gumbel-softmax distribution,” arXiv
      preprint arXiv:1611.04051, 2016.
[449] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with
      gumbel-softmax,” arXiv preprint arXiv:1611.01144, 2016.
[450] C. J. Maddison, A. Mnih, and Y. W. Teh, “The concrete distri-
      bution: A continuous relaxation of discrete random variables,”
      arXiv preprint arXiv:1611.00712, 2016.
[451] R. J. Williams, “Simple statistical gradient-following algorithms
      for connectionist reinforcement learning,” Machine learning,
      vol. 8, no. 3-4, pp. 229–256, 1992.
[452] R. D. Hjelm, A. P. Jacob, T. Che, A. Trischler, K. Cho, and
      Y. Bengio, “Boundary-seeking generative adversarial networks,”
      in International Conference on Learning Representations, 2018.
[453] T. Che, Y. Li, R. Zhang, R. D. Hjelm, W. Li, Y. Song, and
      Y. Bengio, “Maximum-likelihood augmented discrete generative
      adversarial networks,” arXiv preprint arXiv:1702.07983, 2017.
[454] Z. Junbo, Y. Kim, K. Zhang, A. Rush, and Y. LeCun, “Adversari-
      ally regularized autoencoders for generating discrete structures,”
      arXiv preprint arXiv:1706.04223, 2017.
[455] Y. Mroueh and T. Sercu, “Fisher gan,” in Neural Information
      Processing Systems, pp. 2513–2523, 2017.
[456] J. Yoo, Y. Hong, Y. Noh, and S. Yoon, “Domain adaptation using
      adversarial learning for autonomous navigation,” arXiv preprint
      arXiv:1712.03742, 2017.
[457] Y. Mroueh, T. Sercu, and V. Goel, “Mcgan: Mean and covariance
      feature matching gan,” arXiv preprint arXiv:1702.08398, 2017.
[458] Y. Mroueh, C.-L. Li, T. Sercu, A. Raj, and Y. Cheng, “Sobolev gan,”
      arXiv preprint arXiv:1711.04894, 2017.
[459] Y. Saatci and A. G. Wilson, “Bayesian gan,” in Neural Information
      Processing Systems, pp. 3622–3631, 2017.
[460] P. Zhang, Q. Liu, D. Zhou, T. Xu, and X. He, “On the
      discrimination-generalization tradeoff in gans,” arXiv preprint
      arXiv:1711.02771, 2017.

