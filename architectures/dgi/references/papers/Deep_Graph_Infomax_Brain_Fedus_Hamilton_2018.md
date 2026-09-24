# Deep Graph Infomax Brain Fedus Hamilton 2018

> Source: `Deep_Graph_Infomax_Brain_Fedus_Hamilton_2018.pdf`

---

Deep Graph Infomax
Petar Veličković1,2 , William Fedus2,3,5 , William L. Hamilton2,4 ,
Pietro Liò1 , Yoshua Bengio2,3 and R Devon Hjelm6,2,3

1 University of Cambridge          2 Mila    3 Université de Montréal
4 McGill University       5 Google Brain          6 Microsoft Research Montréal

Università degli Studi di Siena Seminar Series                                    17 December 2018
Introduction


 I In this ICLR 2019 submission, we study the problem of
   unsupervised graph representation learning.
     I That is, learning “good” representations for nodes in an
       unlabelled graph.
     I Very important problem: many graphs in the wild are unlabelled!



 I We rely on a synergy of mutual information maximisation
   and graph convolutional networks—enabling simultaneously
   leveraging local and global information propagation in a graph.

 I Very promising results on the learnt embeddings—often
   competitive with supervised learning!
Roadmap for today




 I A tale of many (random) walks;




 I The (DIM) fairy godmother;




 I Happily ever (mutually) informative.
Random walk-based objectives



 I Thus far, the field of unsupervised node embeddings has been
   dominated by random walk-based methods.

 I Some methods go even simpler—and use a link prediction
   objective (“random walks of length 1”):
     I   see e.g. GAE (Kipf & Welling, NIPS BDL 2016) or EP (Duran &
         Niepert, NIPS 2017).


 I I will briefly present two such methods, and outline a few
   reasons why we should also consider alternative objectives.
Overview of DeepWalk (Perozzi et al., KDD 2014)


 I Start by random features ~ i for each node i.


 I Sample a random walk Wi , starting from node i.


 I For node x at step j, x = Wi [j], and a node y at step
   k 2 [j w, j + w], y = Wi [k], modify ~ x to maximise
   log P(y| ~ x ) (obtained from a neural network classifier).

 I Inspired by skip-gram models in natural language processing:
   to obtain a good vector representation of a word, its vector
   should allow us to easily predict the words that surround it.
Overview of DeepWalk, cont’d


 I Expressing the full P(y| ~ x ) distribution directly, even for a
   single layer neural network, where
                                                     ⇣         ⌘
                                                exp ~  wyT ~ x
                                   wyT ~ x ) = P
             P(y | ~ x ) = softmax( ~                  ⇣          ⌘
                                                   exp   w
                                                         ~ T~
                                                 z         z    x


   is prohibitive for large graphs, as we need to normalise across
   the entire space of nodes—making most updates vanish.

 I To rectify, DeepWalk expresses it as a hierarchical softmax—a
   tree of binary classifiers, each halving the node space.
DeepWalk in action




Later improved by LINE (Tang et al., WWW 2015) and node2vec
(Grover & Leskovec, KDD 2016), but main idea stays the same.
Some limitations


 I These kinds of techniques are in principle very powerful—due
   to their relation to methods such as PageRank—but have
   known limitations:
     I They tend to over-emphasise proximity information at the
       expense of structural information (as discussed in struc2vec;
       Ribeiro et al., 2017)
     I Performance can be highly dependent on the choice of

       hyperparameters (as discussed in both DeepWalk and
       node2vec papers).
     I Much harder to apply in inductive settings, especially on

       entirely unseen graphs!

 I To alleviate some of these issues, let’s use something fresh out
   the toolbox. . .
Graph convolutional networks




Let’s assume that we have (intermediate) features,
H = {~h1 , . . . , ~hN } (~hi 2 RF ) in the nodes. . .
Graph convolutional networks




In a nutshell, obtain higher-level representations of a node i by
leveraging its neighbourhood, Ni !
             ~h`+1 = g ` (~h` , ~h` , ~h` , . . . ) (a, b, c, · · · 2 Ni )
               i            a b c

where g ` is the `-th graph convolutional layer.
GCNs meet random walks!


 I If we, instead, use a stack of (inductive) graph convolutional
   layers to derive our embeddings, ~zi . . .
 I . . . and then express our random walk objective on these
   embeddings. . .
 I . . . we can fix both the stability and inductivity issues
   simultaneously!
 I First done in GraphSAGE (Hamilton et al., NIPS 2017), with the
   following objective:

           L(~zu ) =   log (~zuT ~zv )   Q · Evn ⇠Pn (v ) log (~zuT ~zvn )

   where Pn is a negative sampling distribution.
Should GCNs meet random walks?


  I Random walk objectives essentially enforce a bias towards
    proximal nodes having similar embeddings.
  I But a graph convolutional network summarises local patches
    (centered around each of the nodes) of the graph by design!
  I As neighbours experience a lot of patch-level mixing, they will
    already have similar embeddings as a result of using a GCN!
  I It may be argued that random walks may fail to provide any
    useful signal at all in this case.
=) In the age of the GCN, we should perhaps seek a move
   towards non-local objectives.
Roadmap for today




 I A tale of many (random) walks;




 I The (DIM) fairy godmother;




 I Happily ever (mutually) informative.
Towards non-local objectives


 I As was the case in many other situations, we should think of
   graphs as generalisations of images, and therefore it is natural
   to seek inspiration from advances in the vision domain.

 I Luckily, we live in very exciting times, with two highly relevant
   techniques being released in the past two months!
     I   Contrastive Predictive Coding (CPC; Oord et al., 2018)
     I   Deep InfoMax (DIM; Hjelm et al., 2018)

 I Both techniques rely on mutual information maximisation
   between different parts of the input, in the latent space!
Contrastive Predictive Coding

                                ct                   Predictions



   gar       gar      gar       gar


                                      zt      zt+1         zt+2       zt+3      zt+4


   genc     genc      genc     genc        genc        genc        genc      genc

  xt 3      xt 2     xt 1       xt         xt+1        xt+2        xt+3      xt+4



Relies on an autoregressive model (LSTM/PixelCNN) to produce
context, ~ct , from latents, ~zt , and therefore unlikely to be trivially
applicable to graphs (requires a node ordering).
And now. . .




. . . a brief slide-deck intermission!
(Many thanks to Devon Hjelm)
           DEEP INFOMAX (DIM)
           Or: how to exploit structure for fun and profit
                             Devon Hjelm (MSR / MILA)


Work with: Alex Fedorov (MRN), Sam Lavoie (MILA), Karan Grewal (U Toronto),
Adam Trischler, and Yoshua Bengio

Paper title: Learning deep representations by information estimation and maximization
Found at: https://arxiv.org/abs/1808.06670
Github code: https://github.com/rdevon/DIM
                       OVERVIEW


• Goal: Learn good representations

• Simple setting: representation of images

• What is a “good” representation?

• Deep INFOMAX

• Future / ongoing work
        LEARNING REPRESENTATIONS OF
                  IMAGES

         Input image              MxM            Feature
                               feature map       vector


                                                               ?
                                                 Y

                                         M




                                 M


Step 1: encode image into Step 2: Summarize the image into a       Step 3: Profit!
   MxM feature map            single (1D) feature vector
        LEARNING REPRESENTATIONS OF
                  IMAGES

         Input image              MxM            Feature
                               feature map       vector         Happy?

                                                                 Hungry?
                                                 Y

                                         M                       Walky?


                                                      Who’s a Good boy?
                                 M


Step 1: encode image into Step 2: Summarize the image into a   Step 3: Profit!
   MxM feature map            single (1D) feature vector
              WHY DO WE NEED “GOOD”
                 REPRESENTATIONS?
•   Low dimensional “simple”
    summary of the data
                                              ̧
•   Deal with limited data
    (semi-supervised, zero-shot
    learning, etc)

•   Multi-model learning          ?   ̦   ?
    (visually grounded NLP)

•   Discover underlying
    structure / discovery
              WHY DO WE NEED “GOOD”
                 REPRESENTATIONS?
•   Low dimensional “simple”              Representation
    summary of the data           Start
                                                            Goal
•   Deal with limited data
    (semi-supervised, zero-shot
    learning, etc)

•   Multi-model learning
    (visually grounded NLP)

•   Discover underlying                                    State
    structure / discovery                                  space

•   Planning in RL
                          DEEP IMPLICIT INFOMAX
1.    Encode your image

2.    Concatenate feature vector with
      feature map at every location

3.    1x1 convolution                                      Concat feature vector
                                                             at every location            M x M Scores
4.    Call this “real”
                                            Feature                Y                        “Real”
5.    Encode another image                   vector

                                                                                                     M
6.    Concatenate 1st feature vector with                              M
      feature map of new image              Y                                                 M
                                                                           1x1convolutional
7.    1x1 convolution                                          M
                                                                             discriminator
                                                                   Y                       “Fake”
8.    Call this “fake”.

9.    Loss function averaged over                                                                    M
      locations

10.   Train the whole thing like a GAN                                                        M
                                                M x M features drawn from another image
      discriminator (aka like a binary
      classifier)

11.   Profit!
                               DEEP IMPLICIT INFOMAX

Simple classification experiment
• Fully supervised = normal classifier

•   Build a small classifier with 200 hidden units
    on representation
      • conv = last conv layer
      • fc = next to last 1d layer
      • Y = final representation
      • SS = fine tuned with labels

Results
• DIM outperforms all models compared by
   large margins on most tasks

•   Performs comparably to BiGAN with
    random crops

•   Global version, DIM(G) tries to maximize
    mutual information with full input
                    THANKS!

                 Acknowledgements:
   Alex Fedorov (MRN), Sam Lavoie (MILA), Karan
Grewal (U Toronto), Adam Trischler, and Yoshua Bengio
      Found at: https://arxiv.org/abs/1808.06670
    Github code: https://github.com/rdevon/DIM
. . . and we’re back!


 I DIM has many properties that are suitable to being leveraged
   in a graph setting.

 I Namely, all we need is a global summary, and some positive
   and negative patch representations—nobody really cares how
   we got them!
 I While DIM focused on image classification, and therefore
   required the summary to encode useful information of the
   input, for node classification we care about maximising mutual
   information between different local parts of the input.
     I   In practice, both of these should happen simultaneously—a
         local-global objective is a scalable proxy for all-pairs local-local.
Capturing structural roles


 I This becomes all the more important in graphs, where
   structural roles may be an excellent predictor (as argued in
   both struc2vec and GraphWave (Donnat et al., KDD 2018).




 I A local-global objective would allow every node to see
   arbitrarily far in the graph, looking for structural similarities to
   reinforce its local representation!
Roadmap for today




 I A tale of many (random) walks;




 I The (DIM) fairy godmother;




 I Happily ever (mutually) informative.
Towards a graph method


 I Finally, we will look at how to generalise the ideas from DIM
   into the graph domain.

 I To-do list:
      I Obtaining patch representations;
      I Obtaining global summaries;
      I Obtaining negative patch representations;
      I Discriminating positive and negative patch-summary pairs.




 I Most of these will come with absolutely no hassle (reusing
   standard ideas from graph neural networks)!
Representations and summaries

 I We may obtain patch representations, ~
                                        hi , by using any graph
   convolutional encoder, E, we’d like!
     I   In our work, we use the GCN (Kipf & Welling, ICLR 2017) for
         transductive tasks and GraphSAGE (Hamilton et al., NIPS
         2017) for inductive tasks.
                                      ⇣ 1        1
                                                      ⌘
                            E(X, A) =  D̂ 2 ÂD̂ 2 X⇥

 I Graph summaries, ~
                    s, are obtained by a readout function, R.
     I   We have attempted several popular readouts (such as set2vec),
         but simple averaging was found to work best.
                                            N
                                       1 X~
                                R(H) =    hi
                                       N
                                           i=1
Scaling to large graphs



 I As graphs grow in size (>100,000 nodes), storing them in GPU
   memory in their entirety quickly becomes infeasible.

 I This means we must resort to subsampling (computing patch
   representations only for a chosen batch of nodes at once), as
   done in GraphSAGE.

 I We found that summarising only the nodes in the batch still
   works well!
Encoding and summarisation on large graphs




       ~h1             ~h2             ~h3




                       ~s
Obtaining negative samples


  I DIM produces negative patch representations by simply using
     another image from the training set as a “fake” input.
  I For multi-graph datasets (such as PPI), we are able to re-use
     this approach (+ dropout for more variability).
  I However, most graph datasets are single-graph!

=) Must specify a (stochastic) corruption function, C, to obtain
                     e A)
   negative graphs (X, e from the positive one.

  I Can then re-use the encoder, E, on the negative graph, to
                                            ~e
     obtain negative patch representations, h  i.
Choice of corruption function


 I We seek a negative graph that will be in some ways
   “comparable” to the original graph, while encoding mostly
   “nonsensical” patches.
                                    e obtained by row-wise
 I Here, we utilise node shuffling (X
   shuffling of the feature matrix, X), while keeping the adjacency
   matrix fixed (Ae = A).

 I This corruption has many desirable properties:
     I All input node features are preserved;
     I The adjacency matrix is isomorphic to the original one;
     I High likelihood of useless neighbourhoods.
     I Trivial to apply, and works really well! :)
Node shuffling example


             ~x3                                ~x4

       ~x2                ~x6             ~x2                ~x3




                   ~x4          ⇠                     ~x6

 ~x1                                ~x5

                    ~x5                                ~x1
Maximising mutual information


 I We treat (~
                                       ~ej , ~s) as negative examples.
             hi , ~s) as positive, and (h

 I Just like in DIM, we use a discriminator, D, which is a simple
   bilinear binary classifier between these two:
                                        ⇣        ⌘
                          D(~hi , ~s) =  ~hT W~s
                                           i


   and optimising its binary cross-entropy implies maximising the
   mutual information based on the Jensen-Shannon divergence:
           0                                                           1
             N
             X       h     ⇣           M
                                    ⌘i X         ✓         ✓       ◆◆
        1                                                    ~e ~
   L=      @   E(X,A) log D ~hi , ~s +   E(X,
                                           e A)
                                              e  log 1     D h j, s
                                                                       A
      N +M
                 i=1                       j=1
Deep Graph Infomax



We have arrived at Deep Graph Infomax (DGI)!

         (X, A)                      (H, A)

                         E
           ~xi                         ~hi         D    +
                                               R
C                                                  ~s

           ~x            E             ~e
            ej                          hj         D


         (e
          X, e
             A)                      (e
                                      H, e
                                         A)
Quantitative evaluation: Datasets

 I Verified the potential of the method in a variety of node
   classification benchmarks: transductive, inductive, large
   graphs, multi-graph.
   Dataset    Task       Nodes        Edges       Features   Classes
     Cora     Trans.       2,708        5,429      1,433         7
   Citeseer   Trans.       3,327        4,732      3,703         6
   Pubmed     Trans.      19,717       44,338       500          3
    Reddit     Ind.      231,443     11,606,919     602         41
                          56,944                               121
     PPI       Ind.                   818,716       50
                       (24 graphs)                           (multilbl.)

 I In all cases, the encoder was trained on the training set, and
   the embeddings are then used for simple linear classification
   (logistic regression) to evaluate their potential.
Quantitative results: Transductive



Data      Method         Cora          Citeseer      Pubmed
X         Raw features   47.9 ± 0.4%   49.3 ± 0.2%   69.1 ± 0.3%
A, Y      LP             68.0%         45.3%         63.0%
A         DeepWalk       67.2%         43.2%         65.3%
X, A      DW + fts.      70.7 ± 0.6%   51.4 ± 0.5%   74.3 ± 0.9%
X, A      Random-Init    69.3 ± 1.4%   61.9 ± 1.6%   69.6 ± 1.9%
X, A      DGI (ours)     82.3 ± 0.6%   71.8 ± 0.7%   76.8 ± 0.6%
X, A, Y   GCN            81.5%         70.3%         79.0%
X, A, Y   Planetoid      75.7%         64.7%         77.2%
Quantitative results: Inductive

  Data      Method           Reddit          PPI
  X         Raw features     0.585           0.422
  A         DeepWalk         0.324           —
  X, A      DW + fts.        0.691           —
  X, A      GraphSAGE-GCN    0.908           0.465
  X, A      GraphSAGE-mean   0.897           0.486
  X, A      GraphSAGE-LSTM   0.907           0.482
  X, A      GraphSAGE-pool   0.892           0.502
  X, A      Random-Init      0.933 ± 0.001   0.626 ± 0.002
  X, A      DGI (ours)       0.940 ± 0.001   0.638 ± 0.002
  X, A, Y   FastGCN          0.937           —
  X, A, Y   Avg. pooling     0.958 ± 0.001   0.969 ± 0.002
Qualitative results: “evolving” t-SNE




 I Silhouette score of 0.234.

 I Favourable to 0.158 for EP (Duran & Niepert, NIPS 2017).
Qualitative results: DGI insights




 I No particular structure found in negative graph (as expected!).

 I Few “hot” nodes surrounded by many colder ones in t-SNE
   space—implies a specialisation of individual dimensions of the
   embeddings for different purposes!
Qualitative results: DGI insights, cont’d


                              Visualizing top positive/negative samples
                                                                            ·10 2
                    500
                                                                               0.04
                    400
                                                                               0.02
                    300
             Node                                                              0.00
                    200
                                                                               0.02
                    100
                                                                               0.04
                      0
                          0       100      200    300        400      500
                                            Dimension


We can see specialisation!
If we remove dimensions, ordered by how biased they are. . .
Qualitative results: DGI insights, concluded

                DGI classification: robustness to removing dimensions                      DGI discriminator: robustness to removing dimensions
                85                                                                      2,500
                                                                                                     + (p #)
                                                                                                       (p #)




                                                                        Examples misclassified by D
                                                                                        2,000
                80                                                                                   + (p ")
                                                                                                       (p ")


Test accuracy
                             DGI (p ")                                                  1,500
                             DGI (p #)
                75
                             GCN
                                                                                        1,000
                             Random-Init

                70                                                                                    500


                                                                                                        0
                65
                     0      100      200     300    400     500                                             0   100      200     300     400   500
                                  Dimensions removed                                                                  Dimensions removed


Can remove over half the dimensions and stay competitive with
the supervised GCN!
Qualitative results: Corruption function study

                       Classification accuracy for A corruption                              Classification accuracy for (X, A) corruption

                                                                                                                                e
                                                                                                                                X 6= X
                80                                                                                                              GCN
                                                                                        82



Test accuracy                                                           Test accuracy
                                                e
                                                X=X
                75
                                                GCN
                                                                                        81
                                                Random-Init


                70                                                                      80



                     10 6   10 5   10 4 10 3 10 2       10 1      100                        10 6   10 5   10 4 10 3 10 2       10 1     100
                                   Corruption rate, ⇢                                                      Corruption Rate, ⇢


Direct corruption of A works too—but is harder (requires us not to
get too dense), and more expensive to compute at every step.
Thank you!




             Questions?
    petar.velickovic@cst.cam.ac.uk
      http://www.cst.cam.ac.uk/⇠pv273/
      https://arxiv.org/abs/1809.10341

