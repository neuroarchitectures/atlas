# Fast Feedforward Networks Belcak Wattenhofer 2023

> Source: `Fast_Feedforward_Networks_Belcak_Wattenhofer_2023.pdf`

---

                                                                                       Fast Feedforward Networks

                                                                                     Peter Belcak and Roger Wattenhofer
                                                                                                       ETH Zürich
                                                                                              {belcak,wattenhofer}@ethz.ch




arXiv:2308.14711v2 [cs.LG] 18 Sep 2023
                                                                    Abstract
                                           We break the linear link between the layer size and its infer-
                                           ence cost by introducing the fast feedforward1 (FFF) archi-
                                           tecture, a log-time alternative to feedforward networks.
                                           We demonstrate that FFFs are up to 220x faster than feedfor-
                                           ward networks, up to 6x faster than mixture-of-experts net-
                                           works, and exhibit better training properties than mixtures of
                                           experts thanks to noiseless conditional execution.
                                           Pushing FFFs to the limit, we show that they can use as little
                                           as 1% of layer neurons for inference in vision transformers
                                           while preserving 94.2% of predictive performance.


                                                                Introduction
                                         The feedforward layer is a parameter-heavy building block
                                         of transformer models (Vaswani et al. 2017). Growing to
                                         tens of thousands of hidden neurons in recent years, the cost
                                         of feedforward layer inference is now in the sights of those
                                         seeking to make large models faster.
                                             It has been recognized that in very large networks, only
                                         a small portion of the feedforward hidden neurons plays a
                                         role in determining the output for any single input, and that
                                         it is possible to design networks that are modular in order to
                                         utilize this fact (Bengio et al. 2015).
                                             The most recent work on the modularization of feedfor-
                                         ward layers aims at architectural designs that implicitly en-
                                         courage sparsity (Shazeer et al. 2017; Lepikhin et al. 2020;
                                         Fedus, Zoph, and Shazeer 2022). They share the common
                                         approach of subdividing the feedforward layer into separate
                                         blocks of neurons – “experts” – and training a gating layer
                                         to determine the mixture of experts to be used in the forward
                                         pass. Inference acceleration is then achieved by using only
                                         the best-scoring k blocks, or a variant thereof. This approach
                                         scales down the inference time by a constant but remains lin-         Figure 1: A fast feedforward network set in comparison to its
                                         ear in the width of the feedforward layer. Moreover, it relies        peers. Bottom. Illustrations of the resulting regionalization
                                         on noisy gating to allow for load balancing among the ex-             of the input space and varying boundary hardness.
                                         perts, complicating training and encouraging duplicity.
                                         Outline. We introduce the Fast Feedforward (FFF) archi-               of the region boundaries and the neural blocks assigned to
                                         tecture – a peer of the feedforward (FF) architecture that            these regions. This is achieved by tree-conditional execution
                                         accesses blocks of its neurons in logarithmic time. FFF di-           of neurons: a small subset of node neurons is set apart to
                                         vides the input space into disjoint regions by means of a dif-        choose what mixtures of leaf neuron blocks are to be com-
                                         ferentiable binary tree and performs simultaneous learning            puted to produce the final output (Figure 1). As the training
                                                                                                               progresses, the region boundaries harden, and the mixtures
                                            1
                                                https://github.com/pbelcak/fastfeedforward                     tend toward selecting only one log-time-accessible leaf.
Formally, let f : DI → DO be the learning target.                    User manual. Suppose that you have an existing architec-
    The naive approach is to train a feedforward layer (FF) F        ture featuring a FF layer of width w and want to replace it
of width w to approximate f on DI , i.e. F ≈DI f .                   with a FFF layer.
    The mixture-of-experts (MoE) approach (Shazeer et al.               Case 1: “I want faster inference”. Choose ℓ << w such
2017) is to choose an expert width e that does not hin-              that ℓ fits your target inference budget, and then experiment
der performance, and then train separate expert blocks               with different depths d ≥ log2 (w/ℓ) to achieve the desired
E1 , . . . , E⌈w/e⌉ of neurons mixed by the partially random-        performance. Note that ℓ still needs to be large enough to be
ized output of a gating network of width g = ⌈w/e⌉. The              able to learn the partial function on its region, and that the
learning target f is then approximated on DI under the mix-          final training width 2d ℓ might end up being larger than w.
ture of k best-scoring experts, i.e. m1 Eb1 +...+mk Ebk ≈DI             Case 2: “I want a partition of the input space”. Choose
f . For the corresponding FF network of width w, a MoE net-          d such that 2d meets your expectation on the number of
work with g = ⌈w/e⌉ experts uses ke neurons for inference            regions. Then experiment with ℓ ≥ w2−d for best perfor-
at the mixture overhead of g.                                        mance. Note that ℓ again needs to be large enough to be able
    Fast feedforward networks (FFFs) are designed to lever-          to learn the partial function on its region, and that for large d
age the fact that different regions of the input space activate      you might have to actively counter the effects of overfrage-
different sets of neurons in wide networks. FFFs of depth d          mentation.
jointly learn a tree partition R1 , . . . , R2d of the input space
determined by their nodes, and 2d small leaf feedforward             Contributions.
networks L1 , . . . , L2d of width ℓ, which are trained so that      1. We introduce the fast feedforward (FFF) architecture, a
each Li approximates the partial target function f |Ri on Ri ,          peer to the feedforward (FF) architecture that uses only a
i.e. Li ≈Ri f . Crucially, an FFF with depth d and leaf width           log-time-accessible fraction of its neurons at any point.
ℓ can use all 2d ℓ hidden neurons to learn f , but requires only     2. We investigate the effect of leaf size and depth on the
one leaf of width ℓ to compute its output for any ι ∈ DI , and          predictive performance of FFFs as models in their own
does so at the lookup overhead of only d neurons.                       right and show that in sufficiently large settings, FFFs
    To draw a comparison, the leaves Li are to FFFs what                give performance comparable to FFs of the same train-
experts Ej are to MoEs, but the FFF tree network is re-                 ing width while carrying out the inference significantly
gionalizing the input space rather than noisily voting on ex-           faster. We further show that FFFs deliver better memo-
pert prowess. For the corresponding feedforward network                 rization and generalization performance than the FFs of
of width w and a choice of leaf size ℓ, one can choose                  the same inference size.
d = log2 ⌈w/ℓ⌉ and access the leaf in O(d) = O(log w)
instead of O(g) = O(w) time. Coincidentally, the FFF ap-             3. We compare the FFF architecture against the mixture-of-
proach also happens to represent a differentiable relaxation            experts (MoE) approach in terms of their predictive per-
of the classical notion of k-d trees (Bentley 1975).                    formance and inference speed as the number of blocks
    The process enabling the fragmentation of a large feedfor-          increases, and support the claimed advantages of the de-
ward layer to a number of smaller leaf layers while preserv-            sign experimentally.
ing its predictive performance is hardening. In training, the        4. We demonstrate that FFFs can be feasibly used in place
nodes of FFF recursively perform a soft choice ⟨1 − p, p⟩               of FFs as parts of larger, more complex architectures such
over the outputs of their two children (a “mixture of child             as transformers.
experts” if we must), and based on the final loss incurred,
                                                                     The text culminates into a comparison with related work.
the optimizer updates both the weights of the children and
the weights of the parent that computed the choice. During
inference, a single hard decision (i.e. “proceed to left/right                                Algorithm
child”) is made depending on the rounding result [p] at each         Denote the network input, output dimension dimI , dimO .
node. Hardening is the process of nodes learning boundaries          Denote ⟨a, b, c⟩-feedforward network a feedforward net-
such that the soft choices they make for individual inputs go        work with a inputs, b neurons, and c outputs.
from indecisive mixtures (e.g. ⟨49, 51⟩) toward more deci-              Notice that we override the terminology designed for
sive ones (e.g. ⟨3, 97⟩). We observe that in settings where          multi-layer networks and talk of only one set of neurons that
representation power is plentiful (i.e. wide leaves and deep         has both input and output weights. For example, we would
FFFs), the process often takes place on its own. In settings         refer to the BERT-base feedforward layer with input dimen-
where more representational power may be warranted (e.g.             sion 768, 3072 hidden neurons, and 768 output neurons as
wide vision transformers with FFF leaf bottlenecks), this            to the feedforward layer with 3072 neurons, each with 768
process either takes place but stalls prematurely, or takes          inputs and 768 outputs. This greatly simplifies our presenta-
place at a very low rate, and we choose to encourage it with         tion.
the addition of a hardening loss.
    Thanks to hardening, the performance of the soft training        Definition. A fast feedforward network of depth d ≥ 0,
setting carries over to inference. As a byproduct, the learned       node size n, and leaf size ℓ is a pair ⟨N , L⟩.
regions can also be used as a partition of the input space for          N := {N0,0 , . . . , Nd−1,2d−1 −1 } is the set of node
interpretability, surgical model editing, catastrophic forget-       ⟨dimI , n, 1⟩-feedforward networks with an addi-
ting mitigation, reduction of replay data budget, etc..              tional sigmoidal activation on the output. These nodes
 Algorithm 1: FFF forward pass.                                  Hardening. For the predictive performance of
                                                                 F ORWARDT to carry over to F ORWARDI , one must
  Input: Input sample ι ∈ DI , the root node N0,0
                                                                 not lose predictive information when rounding the choice
  Output: Output ∈ DO of the FFF for ι
                                                                 scores. This loss of information is minimal when the
                                                                boundary decisions have been properly hardened (cf. Intro-
  Function F O R W A R D T ι, Nm,n :                             duction). As hinted at above, hardening at a node generally
     if Nm,n ∈ L then                                            does not have to involve adjustment to the boundary as
         return Nm,n (ι)                                         a manifold in space – progressive uniform rescaling of
     else                                                        boundary coefficients (i.e. squashing of the final sigmoid
         cm,n ← Nm,n (ι)                                        toward the step function to make the boundary more clearly
         return cm,n F O R W A R D T ι, Nm+1,2n+1               defined) suffices.
          + (1 − cm,n ) F O R W A R D T ι, Nm+1,2n                  Interpreting the node choice scores as Bernoulli probabili-
     end                                                         ties, hardening can be tracked by monitoring the batch mean
                                                                 of entropies of the choices scores at each node. In our exper-
                                   
  Function F O R W A R D I ι, Nm,n :                             imentation, we found that rounding choice pairs ⟨1 − p, p⟩
     if Nm,n ∈ L then                                            with entropies below 0.10 tends to lead to only very modest
         return Nm,n (ι)                                         deviations from the F ORWARDT performance. For situations
     else                                                        where the hardening of node decisions does not occur to a
         cm,n ← Nm,n (ι)                                         sufficient extent on its own, hardening can be encouraged by
                                                                 the addition of hardening loss.
         if cm,n ≥ 12 then                                         Let Lpred be the loss due to the outputs of FFF. Then one
             return F O R W A R D I ι, Nm+1,2n+1                 can take the total loss to be Ltotal := Lpred + hLharden with
         else                                                                              X X
             return F O R W A R D I ι, Nm+1,2n                                  Lharden :=             H (N (ι)) ,
         end                                                                                ι∈B N ∈N
     end
                                                                 where B ⊆ DI is a batch of samples, H(p) the entropy of a
                                                                 Bernoulli random variable, and h the training hyperparame-
                                                                 ter controlling the effect of the hardening loss.
form a balanced differentiable binary tree such that
Nm+1,2n , Nm+1,2n+1 are the children of Nm,n .                   Overfragmentation. If pushed to the extreme, allowing
   L := {Nd,0 , . . . , Nd,2d −1 } is the set of leaf            fast feedforward networks to learn too many hard bound-
⟨dimI , ℓ, dimO ⟩-feedforward networks. All weights are          aries leads to overfragmentation – a phenomenon in which
trainable by default and the forward pass is governed by the     the network divides the input space into exponentially many
fully deterministic Algorithm 1.                                 disjoint regions and learns to approximate parts of the learn-
   The nodes of FFF are arranged in a differentiable dimI -d     ing target in a way that is too specific for each region. Over-
tree that makes a soft choice over the leaves in the form of     fragmentation has two direct consequences: localized over-
a stochastic vector c. In training, FFF performs a mixture of    fitting and the “shrinking batch problem”.
experts over all leaves in L, with the choice weights of the         Localized overfitting denotes a tail process that occurs
mixture c computed by ascending through the tree from the        once the model has sufficiently hardened, in which the re-
root node N0,0 (cf. F ORWARDT ). During inference, the de-       gion boundaries are no longer flexible and certain leaves
cision at each node is taken to be the closer of {0, 1}, and     learn to overfit the training data on their regions of respon-
the forward pass algorithm proceeds from the root, always        sibility. This is because they stop receiving meaningfully
choosing only one branch depending on the local node deci-       large gradient updates from the neighboring regions, but
sion (cf. F ORWARDI ).                                           may be responsible for handling test samples that are not
                                                                 well understood by the training data for their region. Local-
Regions of responsibility and their boundaries. The tree         ized overfitting manifests itself just like classical overfitting
component of the FFF yields a partition of the input space.      – the validation performance ceases to improve or deterio-
Each leaf is responsible for exactly one region of this parti-   rates while learning on the training set continues. It can be
tion, even though during training, its prediction on its own     mitigated by randomized child transpositions – the soft de-
region of responsibility is mixed with the predictions of        cisions ⟨1 − p, p⟩ at each node can be randomly transposed
other leaves. The boundaries between the individual regions      with some low probability into ⟨p, 1 − p⟩. This is to expose
are determined by the node networks.                             their children to the training data of neighboring regions in
   In the case when n = 1 and there is no activation on the      order to aid generalization performance and does to some
node network but the head sigmoid, the boundary is the ac-       extent already happen for soft boundaries, but it becomes
tivation plane of the hidden neuron. The norm of the plane’s     rare as the boundaries harden.
normal vector (=weights of the neuron) affects how quickly           The FFF variant of the shrinking batch problem is also a
the sigmoid goes from 0 to 1 around the boundary (cf. Fig-       result of the leaf region boundaries hardening, and it arises in
ure 1 bottom-left). This determines how clearly the bound-       situations when the batch size becomes too small for the par-
ary is defined.                                                  tition of the input space learned by the FFF tree. If the par-
tition of the input space is too finely grained and the bound-                            Experiments
aries hardened, each leaf ends up receiving meaningful gra-        We conduct a number of experiments to (1) explore the ef-
dient updates from only a small fraction of the training sam-      fects of assigning neurons with learnable regions of influ-
ples, resulting in inaccurate gradient descent progress. Batch     ence, (2) compare the predictive performance and speed of
shrinking leads to poor learning performance (e.g. low train-      FFFs to that of MoEs, and (3) assess the feasibility of FFFs
ing set accuracy, early stalling, chaotic development) but can     as parts of deeper architectures. The task of each experiment
be mitigated – naively with larger batch sizes, gradient accu-     is image classification, and we evaluate the classification ac-
mulation, and smaller learning rates; in full generality with      curacy of the softmax of output logits in the usual way. For
localized optimization.                                            FFFs, we measure the accuracy of making “hard” decisions
We consider the complexity of our algorithms in terms of the       at every node (i.e. we use F ORWARDI ).
parameters d, n, ℓ. Note that we found n = 1 to suffice in all        For each dataset considered, we use the designated train-
our experiments, but we keep n in for the sake of generality.      ing and test sets as provided. We further split the full training
                                                                   set 9 : 1 into training and validation subsets.
Training complexity. The training forward pass
                                                                      To compare the qualities of individual models, we mea-
F ORWARDT ascends through d levels of the tree, passing
                                                                   sure four quantities.
through node neurons to compute the final choice vector c in
                                                                      Memorization accuracy (MA ). Interpreting (fast) feed-
O((2d − 1)n) time. Then, the leaf outputs are computed and
                                                                   forward networks as model memories (Bau et al. 2020), we
mixed by c in O(2d ℓ) time. This means O(2d (ℓ + n) − n)
                                                                   measure their ability to learn the training set by computing
time for the forward pass, and a (d + 1)-step backward pass
                                                                   the accuracy of overfitted networks on the training set. That
back to the decision on the root.
                                                                   is, we train networks until their accuracy in training stops
   From the implementation standpoint, we express the as-
                                                                   improving, and then run a test on the training data. A result-
cent through the tree as a single loop making d identical
                                                                   ing MA of 100% means that the network has successfully
batched computations, and then perform the final leaf for-
                                                                   memorized all the predictions for the training data.
ward pass and expert mixture.
                                                                      Generalization accuracy (GA ). Treating (fast) feedfor-
Inference complexity. F ORWARDI ascends through the                ward networks as predictive models in their own right, we
FFF tree in d steps, always executing exactly one node net-        measure their ability to correctly predict the classes of pre-
work. Then, it performs inference on one leaf, leading to          viously unseen samples in the test set. For this, we train net-
O(dn + ℓ) time.                                                    works until their validation accuracy stops improving, and
   In terms of the implementation, the ascent from the             use the best model in terms of the validation accuracy for
root through the tree is executed as a batched computa-            evaluation.
tion of an indexed set of weights and biases (multiply-and-           Inference time and speedup. Our own implementa-
accumulate), comparison of the logit to 0, and advancing of        tion of the FFF algorithms is provided through pip
the index depending on the result of the comparison.               install fastfeedforward and on GitHub1 . We
   In our experience of using ahead-of-time compilation for        compile our implementation of F ORWARDI for NVIDIA
CUDA, the selective indexing of weights for node decisions         A100 GPUs using PyTorch 2.0.1 model compilation in the
manifested itself in the native code as a simple offset in the     reduce-overhead mode. We then run each model 104
data load for batched matrix multiplication, having only a         times with batch size 2048 on a single NVIDIA A100 GPU.
small constant implementation overhead on the hardware             Where relevant for comparison, we report the mean infer-
level when compared to feedforward layers.                         ence time t• per single forward pass under repeated trials, to-
                                                                   gether with its standard deviation. We further report speedup
Size and width. Fast feedfoward networks consist of neu-           – the fraction tF F /tF F F , where tF F F is the mean inference
rons of two types: node and leaf neurons. For clarity and to       time for the given FFF model and tF F is the mean infer-
make direct comparisons to the corresponding feedforward           ence time for the vanilla feedforward network of the same
networks, we distinguish between variants of network size          training width. Simply put, speedup says how much faster
and width.                                                         the FFF was than the FF with the same number of neurons
   An FFF with d, n, ℓ as in the definition has training size of   available for making predictions in training, measured using
(2d −1)n+2d ℓ – these are all the neurons of the network, and      this choice of software and hardware. The means and devia-
they are all affected by optimization. It further has inference    tions for speedups are in the appendix.
size of dn + ℓ, as these are the neurons engaged to produce
inference output by F ORWARDI .                                                    Explorative evaluation
   However, only the neurons of leaves produce output, with
the node neurons being involved solely in the computation          To examine the nature of fast feedforward networks as an al-
of the mixture of the outputs of individual leaves. Therefore,     ternative to feedforward networks, we measure the effect of
we say that the FFF has training width of 2d ℓ and inference       their parameters on their predictive performance and speed
width ℓ. Note that the FFF with all weights of node networks       in the context.
set to 0 is equivalent to a vanilla feedforward network with
2d ℓ neurons (up to a uniform rescaling of the output weights,     Evaluation with training counterparts
which is learnable). We refer to the difference between the        Subject. We investigate the relationship between the config-
training/inference size and width as overhead.                     uration of leaf size, depth, and training width and the memo-
     Model                                                             USPS
                             16                           32                           64                         128
                  MA       GA     speedup      MA       GA        speedup   MA       GA     speedup      MA      GA     speedup
   vanilla FF    100.0    93.1      1.00x     100.0    93.7        1.00x    100.0    94.1    1.00x      100.0    94.2     1.00x
         ℓ=8      99.3    92.2      1.07x      99.2    91.8        1.16x    99.2     92.3    1.53x       99.5    92.1     2.56x
   FFF   ℓ=4      94.1    87.6      0.98x      97.2    89.5        1.08x    97.6     90.6    1.56x       97.1    90.3     2.40x
         ℓ=2      92.0    85.5      0.90x      93.4    86.4        1.07x    90.6     84.4    1.39x       94.3    88.1     2.22x
         ℓ=1      83.4    77.0      0.85x      77.3    74.2        0.99x    79.2     77.1    1.34x       81.4    77.8     2.12x
     Model                                                             MNIST
   vanilla FF     98.0    95.2      1.00x     100.0    96.6        1.00x    100.0    97.7    1.00x      100.0    98.1     1.00x
         ℓ=8      94.6    93.1      1.13x      96.5    93.9        1.50x    97.7     94.2    2.20x       99.3    94.9     3.39x
   FFF   ℓ=4      91.6    90.8      1.33x      96.2    93.1        1.35x    96.7     93.3    2.33x       97.6    93.6     3.29x
         ℓ=2      92.1    90.3      1.19x      94.0    91.4        1.48x    95.2     92.1    2.33x       96.2    92.4     3.47x
         ℓ=1      91.7    89.9      1.04x      94.4    92.0        1.26x    94.5     91.4    1.91x       94.1    92.0     3.93x
     Model                                                         FashionMNIST
   vanilla FF     91.0    86.4      1.00x      94.8    87.8        1.00x    98.5     89.0    1.00x       99.3    89.6     1.00x
         ℓ=8      86.7    84.2      1.34x      87.8    85.2        1.44x    88.8     85.2    2.02x       90.5    86.1     3.78x
   FFF   ℓ=4      86.4    83.3      1.27x      86.6    84.5        1.32x    89.1     85.1    2.02x       89.0    85.4     3.41x
         ℓ=2      84.5    83.0      1.24x      85.4    82.9        1.34x    87.2     84.1    2.01x       87.3    84.3     3.28x
         ℓ=1      79.7    78.4      1.04x      79.4    77.8        1.29x    79.9     79.5    1.90x       78.7    77.7     2.92x

Table 1: The results of the explorative experimentation on FFFs. Reading top-to-bottom shows the effect of decreasing the leaf
size and correspondingly increasing the depth. Left-to-right: The effect of increasing the training width and model depth while
keeping the leaf size constant. Diagonally bottom-left-to-top-right: The effect of keeping the depth constant while increasing
the leaf size and training width. Emphasis and emphasis mark the best speedups per ℓ, dataset and dataset, respectively.


rization and generalization performance of fast feedforward            Discussion. Our results are listed in Table 1. A visualiza-
networks. For each FFF, we also consider the performance               tion of the hardening process can be found in Figure 5 of
of a FF of the same training width. We make the compar-                the appendix. The general observations are that each of: in-
ison with them having the same training width rather than              creasing width, increasing leaf size, and increasing leaf size
training size since only leaf neurons are directly involved in         while keeping the depth constant; universally help memo-
computing the classification prediction for the inputs given.          rization and generalization performance. We make several
                                                                       specific observations in relation to our contributions.
                                                                          FFFs perform comparably to FFs. For sufficiently large
Method. We train fast feedforward networks for train-                  widths and depths on USPS and MNIST, fast feedforward
ing widths w = 16, 32, 64, 128, leaf sizes ℓ = 1, 2, 4,                networks are only slightly (2-3%) worse than vanilla feed-
datasets USPS (Hull 1994), MNIST (LeCun, Cortes, and                   forward networks. Coincidentally, these are also the config-
Burges 2010), and FashionMNIST (Xiao, Rasul, and Voll-                 urations in which FFFs deliver the best inference speed im-
graf 2017). For each w, ℓ configuration we compute the                 provements over classical feedforward layers as measured
depth as log2 (w/ℓ). The set of widths has been chosen on              on our hardware.
purpose: notice that any fast feedforward network with w, ℓ               Notice the performances of the FFFs with w = 128, ℓ = 8
as above has inference width smaller than 16 – the narrow-             across datasets relative to FFs with w = 16. The perfor-
est of our configurations. For each width, we further train a          mance is remarkably close and even exceeds that of FFs, all
vanilla feedforward network as a baseline.                             that while the inference size of these FFFs (12) remains be-
   We feed the networks flattened images. For ease of com-             low that of the FFs (16).
parison, we use batch size 256 and pure SGD optimization                  On FashionMNIST we observe the same trends but note
with learning rate of 0.2 irrespective of the size or the depth        that an FFF beyond our testing range (w = 512, ℓ = 8)
of the networks, but we note that deeper FFFs have bene-               was eventually necessary to bring FFFs close (MA =97.1,
fited from larger batch sizes and smaller learning rates. We           GA =88.1) to the performance of FFs.
engage the hardening loss with scale parameter h = 3.0. We                Speedup increases with width. The wider and deeper
execute 10 runs for each configuration, and since this is an           networks become while keeping the leaf size constant, the
evaluation of architectural limits, we report the performance          more significant the inference speed improvement delivered
of the best model. Means and deviations are in the appendix.           by the fast feedforward architecture.
Figure 2: A visualization of the comparison of memorization and generalization performance of fast feedforward (d=2,6) and
feedforward (d=0) networks. Horizontally: the inference size in neurons. Vertically: accuracy.


   Constant width leads to a speed-performance trade-            feedforward networks of varying depths and leaf sizes with
off. If the training width is kept fixed, there is clearly a     respect to their inference size. We make the inference size
trade-off between increasing depth (and therefore increasing     the point of comparison with feedforward networks since,
speed) and performance (cf. Table 1 top-to-bottom).              unlike the inference width, it is directly proportional to the
   Increasing the depth of the network while keeping the         computational cost of inference.
training width constant results in a gradual decrease in per-
formance due to the FFF having smaller leaves, hence our         Method. We train fast feedforward networks for ℓ =
note in the “User manual” (cf. Introduction). Later we will      2, 4, 6, 8, 16, 32, d = 2, 6, datasets SVHN (Netzer et al.
see that in deeper architectures the trade-off in constant-      2011), CIFAR10, CIFAR100 (Krizhevsky, Hinton et al.
width networks between inference speed and performance           2009), and for each ℓ, d configuration we compute the in-
decrease due to small leaf size appears to be lessened in        ference size as ℓ + d. For each inference size, we further
multi-layer architectures.                                       train a vanilla feedforward network as a baseline.
   Large-depth small-leaf networks exhibit overfragmen-             We train the networks with all parameters as above except
tation. We observe (USPS, FashionMNIST) that decreasing          h, where we do not engage the hardening loss (h = 0) as
the leaf size while increasing depth (keeping the width con-     we found that the hardening tended to occur on its own. We
stant) leads to quickly worsening memorization and general-      execute 10 runs for each configuration and report the best
ization performance. The difference is particularly stark with   performances.
greater depths (ℓ=1,2, w=64, 128). This is well explained by     Discussion. Our results are sparse because of all the dif-
overfragmentation, since especially for ℓ = 1, w = 128,          ferent possible sums of leaf size and depth and are shown
each leaf receives only 2 samples per batch on average. The      in Figure 2. Aligned with intuition, we observe that bigger
results on USPS for ℓ=1, w=16,32 are a model example of          depth and larger leaf sizes lead to better memorization and
this: we see that FFF with ℓ=1,d=4 delivers MA of 83.4,          generalization performance.
but that its deeper cousin ℓ=1,d=5 – which has more of the          Further, FFFs outperform FFs of the same inference
same-sized leaves at its disposal – yields MA of only 77.3.      size. FFFs of varying depths and sizes consistently outper-
TL;DR. FFFs give predictive performance comparable to            form the FFs with widths equal to the FFF inference sizes,
FFs of the same training width, are faster as the training       both in terms of MA and GA . In terms of MA , the difference
width increases, and if pushed to the limit exhibit speed-       is stark and grows with the depth and leaf size. In terms of
performance trade-offs and overfragmentation.                    GA , FFFs initially gain an edge over FFs, but later the per-
                                                                 formances of all models unite toward plateauing out at the
Evaluation with inference counterparts                           limit of the naive, single-layer feedforward-only approach.
Subject. Similarly to the experimentation above with the         TL;DR. FFFs deliver performance more readily than FFs
main consideration for training width, we now evaluate fast      of the same inference width.
             Width                                                       Model
                                 feedforward              mixture-of-experts (e=16, k=2)         fast feedforward (ℓ=32)
                          MA     ETT      GA     ETT      MA       ETT      GA       ETT       MA      ETT     GA     ETT
           w = 64        87.2     307    49.3     55      57.8     5354     29.4     4880      85.8    302     45.9    22
           w = 128       95.5     200    51.5     46      62.0     6074     33.6      938      90.1    305     45.5    22
           w = 256       99.9     105    52.0     48      62.4     2001     33.9      372      91.2    244     44.4    17
           w = 512       99.9      85    52.4     31      65.4     3834     34.5      315      96.2    175     43.7    10
           w = 1024      99.9      82    53.0     21      65.3     1575     35.2      327      96.0    180     41.3     9

Table 2: The results of the comparison of feedforward, mixture-of-experts, and fast feedforward networks, for various training
widths. The inference width is fixed to 32 for mixture-of-experts and fast feedforward networks. The ETT columns to the right
of metric columns list the “epochs to train”, i.e. the number of training epochs that have elapsed until the score to the left was
observed.


               Comparative evaluation                                    training width, and suffer from localized overfitting with the
The direct contender architecture to fast feedforward, com-              increasing depth.
ing along with its own set of design parameters, is the                     FFFs outperform MoEs of the same training width.
mixture-of-expert layer, which we take in its original form              We observe that FFFs consistently deliver better MA and
(Shazeer et al. 2017).                                                   GA scores than the MoE networks of the same training
                                                                         width. We further see that they do so at ETTs smaller by an
Predictive performance comparison                                        order of magnitude. We attribute this difference mainly to
Subject. We compare FFFs against MoEs and FFs of vary-                   the learnably controlled noise introduced to the expert mix-
ing training widths in terms of their predictive performance.            ture computation to aid load balancing and generalization.
We keep the leaf and expert width constant and focus on the              Without the noise, however, MoE networks would overfit to
ability of the architectures to deliver good memorization and            learn only with a handful of experts. We also experimented
generalization properties as well as on the training compute             with varying values of wimportance and wload , but we found
necessary to reach those properties.                                     those to be broadly detrimental to the load balancing effort.
                                                                         Our final values of batch importance and load were consis-
Method. We experiment on the unaugmented CIFAR10                         tent with those arrived at in Shazeer et al. (2017).
dataset. We train feedforward, mixture-of-experts, and fast
feedforward networks of increasing size so that they always              TL;DR. FFFs deliver representational power more readily
agree in the training width. To keep the inference width the             than the MoEs of equal training widths.
same, we set the leaf width to 32 and expert width to 16
with always engaging k = 2 experts. Note that while single-              Inference speed comparison
expert networks can be used for inference, they are not able
to propagate gradients to the gating network (cf. Shazeer                Subject. Since the operations involved in the computation
et al. (2017)). We take widths w = 26 , 27 , . . . , 210 , which         of the expert/leaf network output are the same, the difference
correspond to 16- to 64-expert MoE networks and FFFs of                  in inference speed between mixture-of-experts and fast feed-
depths 1 to 5. To encourage importance equality and load                 forward networks comes solely from the functioning of the
balancing in MoEs, we set wimportance = wload = 0.1 in                   gating/lookup mechanism. We therefore keep the expert/leaf
line with previous work. To encourage FFF hardening, we                  width constant and measure the time needed to execute in-
use h = 3.0. We train all models width batch size 4096 for               ference forward passes of feedforward, mixture-of-experts,
7000 epochs at most, with early stopping after 350 epochs                and fast feedforward networks across repeated trials for in-
where no improvement in the respective validation accura-                creasing numbers of experts/leaves (i.e. wider and wider net-
cies is seen. All models have been trained with the Adam op-             works).
timizer, learning rate of 0.001, with the learning rate halving             To add realism, we simulate the conditions of a BERT-
on 250-epoch training accuracy plateaus.                                 base (Devlin et al. 2018) feedforward layer, setting the input
                                                                         and output widths of all neurons to 768 each.
Discussion. The results are listed in Table 2. On the outset,
we observe that the MA and GA benefit from larger training               Method. We consider FF models of width 32 × 21 to
width across all models except GA on FFFs, where we see                  32 × 25 , where we highlight the 32 neuron blocks for di-
the unmitigated localized overfitting negatively affecting the           rect comparison with the other models. We further evaluate
performance.                                                             MoE models with expert width e = 32 and 21 to 215 ex-
   FFFs are the fastest to deliver MA and GA . We see that               perts, and FFF models with leaf width e = 32 and depths
the FFFs are the fastest (in terms of ETT) to deliver both               1 to 15. To eliminate the effect of the mixture computation
MA and GA , but, consistently with our previous explorative              on our measurements, we keep e = ℓ and set k = 1, even
evaluation, deliver slightly lower MA than FFs of the same               though this MoE parameter configuration is not trainable (cf.
Figure 3: A visualization of the performance measure-              Figure 4: A close-up on the visualization of the performance
ment results. The horizontal axis denotes the number of            measurement results. The axes and values are as in Figure 3.
blocks/experts/leaves and is scaled logarithmically, the point
values are the mean inference times per single forward pass,
and the error bars show the standard deviation.                    128, and a baseline vision transformer with feedforward lay-
                                                                   ers of width w = 128. The fast feedforward layers have leaf
                                                                   size ℓ = 1, 2, 4, 8, 16, 32 and depths log2 (w/ℓ). We try three
above). When measuring, each model performs inference on           levels of hardening: h = 5, 10, ∞, where ∞ denotes that the
BERT-base inputs with batch size 256 exactly 20000 times.          FFF tree has been effectively frozen from the beginning (i.e.
                                                                   the boundaries are not trainable). We use Adam optimizer
Discussion. The inference speed measurement results are
                                                                   with the initial learning rate of 4e − 4 and learning rate halv-
visualized in Figures 3–4. We observe that both MoEs and
                                                                   ing on 50-epoch validation accuracy plateaus. For each ℓ,d-
FFFs offer significant acceleration to the inference speed
                                                                   configuration, we report the generalization performance of
when compared to the FF baseline. However, Figure 4,
                                                                   the best model and the measured speedup at the feedforward
shows the clear tendency of MoE model inference time to
                                                                   layers (not the whole transformer).
grow exponentially with the exponent of the expert count, in
stark contrast with the linear relationship between the two        Discussion. The results of our experimentation can be
exhibited by the FFF models. This is fully aligned with our        seen in Table 3. A visualization of the hardening process
theoretical analysis of the inference time complexity of the       across the layers of the transformer can be found in Figure 6
two architectures.                                                 of the appendix. In line with our assessment of the algorithm
TL;DR. We have experimentally confirmed the exponen-               complexity, the measured speedup at the feedforward layers
tial difference between the inference time complexity of the       increases with decreasing leaf size. Further:
MoE’s and FFF’s internal mechanism.                                   Single-neuron FFFs suffice. We observe that even fast
                                                                   feedforward layers with inference width 1 are sufficient for
                                                                   the vision transformer to deliver reasonable performance,
   Fast feedforward layers as building blocks                      with relative decrease in performance of only 5.8%.
Subject. We demonstrate that fast feedforward networks                The effects of overfragmentation are suppressed. We
can be used as layers in place of standard feedforward layers      observe that the generalization performance of GA suffers
in the transformer architecture, thus giving it a significant      only relatively mildly due to the increase in depth and de-
performance boost. Previously, we noted that leaf sizes that       crease in leaf size, which is in stark contrast with the results
are too small for a given problem may lead to the occurrence       of Table 1. We attribute this to the depth of the transformer
of overfragmentation. Here we push our experimental setup          and take it as an encouraging sign of the feasibility of FFFs
to the limit in terms of leaf size and investigate the effect of   as replacements for FFs.
overfragmentation in a deep transformer.
                                                                   TL;DR. Fast feedforward layers can deliver inference ac-
Method. We experiment on the CIFAR10 dataset of                    celeration in vision transformers over feedforward layers of
32x32-pixel 3-channel images, with random horizontal, ver-         the same training width at the cost of a small performance
tical flipping, and random linear augmentations (translate,        decrease.
rotate, scale). As models, we use 4-layer vision transform-
ers with patch size 4, hidden dimension 128, input dropout
0.1, and no layer dropout.                                                               Related work
   We consider vision transformers with their feedforward          Our work overlaps with the research efforts in two areas of
layers replaced by fast feedforward layers of training width       inference acceleration.
                   Model                                                Property
                              depth   training width   training size   inference width    inference size    speedup     GA
        FF          w = 128      –              128    128 (100%)          128 (100%)       128 (100%)       1.00x     84.7
                    ℓ = 32       2              128    131 (102%)           32 (25%)          34 (27%)       2.44x     83.6
                    ℓ = 16       3              128    135 (105%)           16 (13%)          19 (15%)       2.80x     83.2

         fast FF
                    ℓ=8          4              128    143 (112%)            8 (6%)           12 (9%)        3.29x     82.8
                    ℓ=4          5              128    159 (124%)            4 (3%)            9 (7%)        3.39x     81.6
                    ℓ=2          6              128    191 (149%)            2 (1%)            8 (6%)        3.47x     80.1
                    ℓ=1          7              128    255 (199%)            1 (1%)            8 (6%)        3.93x     79.8

Table 3: The results of the testing of vision transformers leveraging feedforward and fast feedforward layers. All sizes are given
in neurons. Bracketed percentages describe quantities relative to their counterparts in the vanilla feedforward layers. GA is the
generalization accuracy of the fully trained vision transformer and “speedup” gives the performance improvement over vanilla
feedforward layers in our testing setup.


Conditional execution. Although nowadays largely inac-                 randomization to avoid the formation of a strong preference
tive due to the tendency to move away from custom archi-               for only a handful experts. This holds true even when mul-
tectures, modified designs for MLP and CNN models were                 tiple layers of expert mixers are introduced. In direct com-
proposed to allow for their partial execution where possible.          parison, a fast feedforward network of depth d = log2 n
   A number of methods were proposed (Davis and Arel                   reduces the inference by a factor of n and requires only
2013; Bengio, Léonard, and Courville 2013; Bengio et al.              O(d) = O(log n) time to decide on which leaf to use. Ad-
2015; Almahairi et al. 2016) to learn either policy distribu-          mittedly, to compensate for the effect of having only one
tions or additional controlling neural components to decide            leaf to make the decision, the leaves of the fast feedforward
which blocks of layer neurons to execute during forward                layer might have to be slightly wider than the blocks of the
pass.                                                                  corresponding mixture-of-experts layer.
   In comparison, fast feedforward networks completely
conceal the learning of leaf regions from the user (save from          Regionalization. An additional advantage of FFF over all
the hyperparameter h if used) and come in an inference-                of the related work is that there is a direct correspondence
ready form once trained, requiring no adjustment when in-              between parts of the network used in inference and alge-
cluded as a part of transformers.                                      braically identifiable regions of the input space. This can be
   In a notable generalization of this line of work to deep ar-        leveraged to mitigate catastrophic forgetting when editing
chitectures, Ioannou et al. (2016) proposed an approach peer           models and to significantly reduce replay data budgets by
to deep convolutional neural networks that learns to route             applying the learned partition of the input space to partition
the input through a sequence of chosen intermediate layers.            the training data.
While our method quickly routes the signal to a single-leaf
feedforward neural network, it draws no comparison to deep                                    References
networks.
                                                                       Almahairi, A.; Ballas, N.; Cooijmans, T.; Zheng, Y.;
Modular “mixture-of-experts” models. Very large mod-                   Larochelle, H.; and Courville, A. 2016. Dynamic capacity
els practically demand modularity. The most straightforward            networks. In International Conference on Machine Learn-
way to modularize large transformer models in order to re-             ing, 2549–2558. PMLR.
duce their inference cost is to subdivide their feedforward
layers into n blocks of neurons, and then train a controlling          Bau, D.; Liu, S.; Wang, T.; Zhu, J.-Y.; and Torralba, A.
classifier to choose which block to involve in forward pass.           2020. Rewriting a deep generative model. In Computer
This is usually done by training a wide softmax-activated              Vision–ECCV 2020: 16th European Conference, Glasgow,
linear layer to produce a stochastic vector of mixture scores          UK, August 23–28, 2020, Proceedings, Part I 16, 351–369.
to be applied to the outputs per block in order to produce the         Springer.
final output. Several variants of this method have been pro-           Bengio, E.; Bacon, P.-L.; Pineau, J.; and Precup, D. 2015.
posed and tested across a variety of large models (Shazeer             Conditional computation in neural networks for faster mod-
et al. 2017; Lepikhin et al. 2020; Fedus, Zoph, and Shazeer            els. arXiv preprint arXiv:1511.06297.
2022).
   We thoroughly compare fast feedforward to mixture-of-               Bengio, Y.; Léonard, N.; and Courville, A. 2013. Estimat-
expert networks in earlier sections. To briefly summarise,             ing or propagating gradients through stochastic neurons for
the mixture-of-experts approach reduces the layer inference            conditional computation. arXiv preprint arXiv:1308.3432.
width by a factor of n/k, where k is the number of best-               Bentley, J. L. 1975. Multidimensional binary search trees
scoring blocks to engage in inference, but requires O(n)               used for associative searching. Communications of the ACM,
time to select the k blocks, and often relies on controlled            18(9): 509–517.
Davis, A.; and Arel, I. 2013. Low-rank approximations
for conditional feedforward computation in deep neural net-
works. arXiv preprint arXiv:1312.4461.
Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018.
Bert: Pre-training of deep bidirectional transformers for lan-
guage understanding. arXiv preprint arXiv:1810.04805.
Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch trans-
formers: Scaling to trillion parameter models with simple
and efficient sparsity. The Journal of Machine Learning Re-
search, 23(1): 5232–5270.
Hull, J. J. 1994. A database for handwritten text recogni-
tion research. IEEE Transactions on pattern analysis and
machine intelligence, 16(5): 550–554.
Ioannou, Y.; Robertson, D.; Zikic, D.; Kontschieder, P.;
Shotton, J.; Brown, M.; and Criminisi, A. 2016. Decision
forests, convolutional networks and the models in-between.
arXiv preprint arXiv:1603.01250.
Krizhevsky, A.; Hinton, G.; et al. 2009. Learning multiple
layers of features from tiny images.
LeCun, Y.; Cortes, C.; and Burges, C. 2010. MNIST
handwritten digit database. ATT Labs [Online]. Available:
http://yann.lecun.com/exdb/mnist, 2.
Lepikhin, D.; Lee, H.; Xu, Y.; Chen, D.; Firat, O.; Huang,
Y.; Krikun, M.; Shazeer, N.; and Chen, Z. 2020. Gshard:
Scaling giant models with conditional computation and au-
tomatic sharding. arXiv preprint arXiv:2006.16668.
Netzer, Y.; Wang, T.; Coates, A.; Bissacco, A.; Wu, B.; and
Ng, A. Y. 2011. Reading digits in natural images with unsu-
pervised feature learning.
Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.;
Hinton, G.; and Dean, J. 2017. Outrageously large neu-
ral networks: The sparsely-gated mixture-of-experts layer.
arXiv preprint arXiv:1701.06538.
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. At-
tention is all you need. Advances in neural information pro-
cessing systems, 30.
Xiao, H.; Rasul, K.; and Vollgraf, R. 2017. Fashion-mnist:
a novel image dataset for benchmarking machine learning
algorithms. arXiv preprint arXiv:1708.07747.
             Extended results of Table 1
Table 4 lists the means and standard deviations of our explo-
rative evaluation of fast feedforward networks in comparison
with feedforward networks.
   On top of the observations made in the main text, we
see that the variance of MA and GA when performing re-
peated runs increases with the decreasing leaf size and is
most clearly seen for small training widths.

      Visualization of the hardening process
To provide some insight and intuition into the hardening of
the node decision boundaries, we track the batch mean en-
                                                                     Figure 5: The evolution of batched mean decision entropy in
tropies across the nodes of fast feedforward layers and report
                                                                     a fast feedforward layer with ℓ = 8, d = 2, 3, 4, h = 3.0,
on their evolution as training progresses visually.
                                                                     trained on the MNIST dataset.
   Figure 5 shows the evolution of the batched mean entropy
for three models of varying depths whose training was ter-
minated by early stopping on validation F ORWARDI accu-
racy plateau. We observe that by keeping the dataset (and
therefore the complexity of the input space) and leaf size
constant, the batch mean accuracy of deep layers converges
faster for deeper networks. We attribute this simply to the
fact that the deeper the fast feedforward network in this ex-
periment, the more leaves were available to selectively cater
to sub-spaces of the input space, allowing the networks to
make cleaner separations between the regions learned.
   Figure 6 reports the evolution of batched mean entropies
per layer for a 4-layer visual transformer model with fast
feedforward networks in place of the vanilla feedforward
networks. We notice that early into the training, entropies of       Figure 6: The evolution of batched mean decision en-
the layers closer to the input (i.e. the earlier/lower layers) are   tropies across a vision 4-layer transformer of dimension 128
faster to converge. However, we see that the entropy of the          equipped with fast feedforward layers of ℓ = 32, d = 2, h =
second layer then ceases to meaningfully decrease, and that          0.10, trained on the CIFAR10 dataset.
the entropies of the first layer actually begin to climb. This
can be explained by hardened decision boundaries showing
signs of bottleneck behavior in higher layers of the trans-                                    Code
former, thus necessitating that the lower layers contribute          We provide our implementations as a Python package built
with learned representations of higher quality.                      on top of the PyTorch framework.
   Interpreting the stalling or climbing of the entropy as a           Just use pip install fastfeedforward. Docu-
sign of the need for more representational power, we note            mentation is provided with the code.
across multiple experiments that the ideal shape for a vision
transformer, even on a relatively simple dataset such as CI-
FAR10, appears to be more toward the shape of a pyramid
than a uniform-width line of layers.

  Relationship to multi-gating MoE networks
More recent literature on mixture-of-experts networks (Lep-
ikhin et al. 2020; Fedus, Zoph, and Shazeer 2022) suggests
the use of a hierarchical, two-layer gating structure. Save
from the inherent randomness of the gating network selec-
tion, this can be seen as a small step towards FFFs. However,
FFFs attend to clearly specified regions of space rather than
being noisily voted in as the most appropriate experts for a
given input (which we have shown gives them much better
training and inference properties), and further, even under
such an approach, the inference time reduction of mixture-
of-expert networks remains constant in the width of the tar-
get network.
        Model                                                                                        USPS
                                     16                                          32                                           64                                        128
                     MA            GA             time           MA            GA             time             MA           GA            time           MA            GA            time
  vanilla FF      100.0 ± 0.0   93.1 ± 0.4    0.13 ± 0.02ms   100.0 ± 0.0   93.7 ± 0.4    0.16 ± 0.01ms   100.0 ± 0.0    94.1 ± 0.3   0.24 ± 0.02ms   100.0 ± 0.0   94.2 ± 0.3   0.41 ± 0.03ms
            ℓ=8   99.3 ± 0.5    92.2 ± 0.4    0.12 ± 0.03ms    99.2 ± 0.5   91.8 ± 0.6    0.14 ± 0.03ms     99.2 ± 0.4   92.3 ± 0.7   0.16 ± 0.03ms   99.5 ± 0.8    92.1 ± 0.4   0.16 ± 0.02ms
            ℓ=4   94.1 ± 0.5    87.6 ± 0.7    0.13 ± 0.03ms    97.2 ± 3.4   89.5 ± 2.0    0.15 ± 0.03ms     97.6 ± 2.3   90.6 ± 1.4   0.16 ± 0.02ms   97.1 ± 1.3    90.3 ± 1.2   0.17 ± 0.02ms
  fast FF
            ℓ=2   92.0 ± 9.2    85.5 ± 8.0    0.14 ± 0.03ms   93.4 ± 10.0   86.4 ± 8.7    0.15 ± 0.02ms     90.6 ± 7.7   84.4 ± 6.0   0.18 ± 0.03ms   94.3 ± 8.6    88.1 ± 7.1   0.18 ± 0.03ms
            ℓ=1   83.4 ± 12.1   77.0 ± 11.1   0.15 ± 0.02ms    77.3 ± 5.8   74.2 ± 5.0    0.16 ± 0.01ms     79.2 ± 8.1   77.1 ± 7.1   0.18 ± 0.02ms   81.4 ± 9.2    77.8 ± 8.3   0.19 ± 0.02ms
        Model                                                                                        MNIST
  vanilla FF      98.0 ± 0.9    95.2 ± 0.5    0.34 ± 0.11ms   100.0 ± 0.0   96.6 ± 0.2    0.42 ± 0.06ms   100.0 ± 0.0    97.7 ± 0.2   0.69 ± 0.10ms   100.0 ± 0.0   98.1 ± 0.1   1.13 ± 0.06ms
            ℓ=8   94.6 ± 19.5   93.1 ± 16.6   0.30 ± 0.11ms    96.5 ± 2.3    93.9 ± 1.2   0.28 ± 0.05ms     97.7 ± 4.3   94.2 ± 2.4   0.31 ± 0.06ms   99.3 ± 1.0    94.9 ± 0.6   0.33 ± 0.08ms
            ℓ=4   91.6 ± 29.3   90.8 ± 27.2   0.26 ± 0.07ms   96.2 ± 24.3   93.1 ± 23.9   0.31 ± 0.09ms     96.7 ± 1.0   93.3 ± 0.6   0.30 ± 0.06ms   97.6 ± 0.6    93.6 ± 0.5   0.34 ± 0.08ms
  fast FF
            ℓ=2   92.1 ± 7.3    90.3 ± 5.6    0.28 ± 0.08ms    94.0 ± 1.4    91.4 ± 1.0   0.28 ± 0.05ms     95.2 ± 1.8   92.1 ± 1.2   0.30 ± 0.07ms   96.2 ± 1.4    92.4 ± 0.6   0.32 ± 0.06ms
            ℓ=1   91.7 ± 7.4    89.9 ± 6.4    0.33 ± 0.11ms    94.4 ± 3.5    92.0 ± 3.1   0.33 ± 0.07ms     94.5 ± 1.8   91.4 ± 1.1   0.36 ± 0.09ms   94.1 ± 0.9    92.0 ± 0.7   0.29 ± 0.03ms
        Model                                                                                   FashionMNIST
  vanilla FF      91.0 ± 0.7    86.4 ± 0.4    0.34 ± 0.10ms   94.8 ± 0.9    87.8 ± 0.2    0.42 ± 0.07ms     98.5 ± 0.8   89.0 ± 0.4   0.64 ± 0.07ms   99.3 ± 0.4    89.6 ± 0.2   1.13 ± 0.05ms
            ℓ=8   86.7 ± 12.1   84.2 ± 10.9   0.26 ± 0.07ms   87.8 ± 17.6   85.2 ± 16.1   0.29 ± 0.10ms     88.8 ± 5.6   85.2 ± 3.8   0.32 ± 0.05ms   90.5 ± 1.7    86.1 ± 1.0   0.30 ± 0.06ms
            ℓ=4   84.5 ± 25.0   83.0 ± 24.5   0.27 ± 0.11ms    86.6 ± 8.6    84.5 ± 6.4   0.32 ± 0.08ms     89.1 ± 3.5   85.1 ± 2.3   0.32 ± 0.07ms   89.0 ± 0.7    85.4 ± 0.7   0.33 ± 0.07ms
  fast FF
            ℓ=2   83.6 ± 21.0   82.5 ± 11.0   0.28 ± 0.10ms    85.4 ± 8.4    82.9 ± 6.5   0.32 ± 0.09ms     87.2 ± 7.1   84.1 ± 5.9   0.32 ± 0.06ms   85.3 ± 5.2    81.5 ± 3.7   0.35 ± 0.08ms
            ℓ=1   86.4 ± 9.0    83.3 ± 8.0    0.33 ± 0.09ms    79.4 ± 6.2    77.8 ± 5.5   0.33 ± 0.07ms     79.9 ± 3.5   79.5 ± 3.7   0.34 ± 0.08ms   78.7 ± 4.6    77.7 ± 3.8   0.39 ± 0.08ms

Table 4: The detailed results of the explorative experimentation on FFFs. Reading top-to-bottom shows the effect of decreasing the leaf size and correspondingly
increasing the depth. Left-to-right: The effect of increasing the training width and model depth while keeping the leaf size constant. Diagonally bottom-left-to-top-
right: The effect of keeping the depth constant while increasing the leaf size and training width.

