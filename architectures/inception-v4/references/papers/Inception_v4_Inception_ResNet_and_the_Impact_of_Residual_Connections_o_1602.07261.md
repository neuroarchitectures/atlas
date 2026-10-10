# Paper (Szegedy et al. 2016)

> Source: `https://arxiv.org/abs/1602.07261`

---

Inception-v4, Inception-ResNet and
the Impact of Residual Connections on Learning
Christian Szegedy
Google Inc.
1600 Amphitheatre Pkwy, Mountain View, CA
szegedy@google.com
Sergey Ioffe
sioffe@google.com
Vincent Vanhoucke
vanhoucke@google.com
Alex Alemi
alemi@google.com
Abstract
Very deep convolutional networks have been central to
the largest advances in image recognition performance in
recent years. One example is the Inception architecture that
has been shown to achieve very good performance at rel-
atively low computational cost. Recently, the introduction
of residual connections in conjunction with a more tradi-
tional architecture has yielded state-of-the-art performance
in the 2015 ILSVRC challenge; its performance was similar
to the latest generation Inception-v3 network. This raises
the question of whether there are any beneﬁt in combining
the Inception architecture with residual connections. Here
we give clear empirical evidence that training with residual
connections accelerates the training of Inception networks
signiﬁcantly. There is also some evidence of residual Incep-
tion networks outperforming similarly expensive Inception
networks without residual connections by a thin margin. We
also present several new streamlined architectures for both
residual and non-residual Inception networks. These varia-
tions improve the single-frame recognition performance on
the ILSVRC 2012 classiﬁcation task signiﬁcantly. We fur-
ther demonstrate how proper activation scaling stabilizes
the training of very wide residual Inception networks. With
an ensemble of three residual and one Inception-v4, we
achieve 3.08% top-5 error on the test set of the ImageNet
classiﬁcation (CLS) challenge.
1. Introduction
Since the 2012 ImageNet competition [11] winning en-
try by Krizhevsky et al [8], their network “AlexNet” has
been successfully applied to a larger variety of computer
vision tasks, for example to object-detection [4], segmen-
tation [10], human pose estimation [17], video classiﬁca-
tion [7], object tracking [18], and superresolution [3]. These
examples are but a few of all the applications to which deep
convolutional networks have been very successfully applied
ever since.
In this work we study the combination of the two most
recent ideas: Residual connections introduced by He et al.
in [5] and the latest revised version of the Inception archi-
tecture [15]. In [5], it is argued that residual connections are
of inherent importance for training very deep architectures.
Since Inception networks tend to be very deep, it is natu-
ral to replace the ﬁlter concatenation stage of the Inception
architecture with residual connections. This would allow
Inception to reap all the beneﬁts of the residual approach
while retaining its computational efﬁciency.
Besides a straightforward integration, we have also stud-
ied whether Inception itself can be made more efﬁcient by
making it deeper and wider. For that purpose, we designed
a new version named Inception-v4 which has a more uni-
form simpliﬁed architecture and more inception modules
than Inception-v3. Historically, Inception-v3 had inherited
a lot of the baggage of the earlier incarnations. The techni-
cal constraints chieﬂy came from the need for partitioning
the model for distributed training using DistBelief [2]. Now,
after migrating our training setup to TensorFlow [1] these
constraints have been lifted, which allowed us to simplify
the architecture signiﬁcantly. The details of that simpliﬁed
architecture are described in Section 3.
In this report, we will compare the two pure Inception
variants, Inception-v3 and v4, with similarly expensive hy-
brid Inception-ResNet versions.
Admittedly, those mod-
els were picked in a somewhat ad hoc manner with the
main constraint being that the parameters and computa-
tional complexity of the models should be somewhat similar
to the cost of the non-residual models. In fact we have tested
bigger and wider Inception-ResNet variants and they per-
formed very similarly on the ImageNet classiﬁcation chal-
1
arXiv:1602.07261v2  [cs.CV]  23 Aug 2016

lenge [11] dataset.
The last experiment reported here is an evaluation of an
ensemble of all the best performing models presented here.
As it was apparent that both Inception-v4 and Inception-
ResNet-v2 performed similarly well, exceeding state-of-
the art single frame performance on the ImageNet valida-
tion dataset, we wanted to see how a combination of those
pushes the state of the art on this well studied dataset. Sur-
prisingly, we found that gains on the single-frame perfor-
mance do not translate into similarly large gains on ensem-
bled performance. Nonetheless, it still allows us to report
3.1% top-5 error on the validation set with four models en-
sembled setting a new state of the art, to our best knowl-
edge.
In the last section, we study some of the classiﬁcation
failures and conclude that the ensemble still has not reached
the label noise of the annotations on this dataset and there
is still room for improvement for the predictions.
2. Related Work
Convolutional networks have become popular in large
scale image recognition tasks after Krizhevsky et al. [8].
Some of the next important milestones were Network-in-
network [9] by Lin et al., VGGNet [12] by Simonyan et al.
and GoogLeNet (Inception-v1) [14] by Szegedy et al.
Residual connection were introduced by He et al. in [5]
in which they give convincing theoretical and practical ev-
idence for the advantages of utilizing additive merging of
signals both for image recognition, and especially for object
detection. The authors argue that residual connections are
inherently necessary for training very deep convolutional
models. Our ﬁndings do not seem to support this view, at
least for image recognition. However it might require more
measurement points with deeper architectures to understand
the true extent of beneﬁcial aspects offered by residual con-
nections. In the experimental section we demonstrate that
it is not very difﬁcult to train competitive very deep net-
works without utilizing residual connections. However the
use of residual connections seems to improve the training
speed greatly, which is alone a great argument for their use.
The Inception deep convolutional architecture was intro-
duced in [14] and was called GoogLeNet or Inception-v1 in
our exposition. Later the Inception architecture was reﬁned
in various ways, ﬁrst by the introduction of batch normaliza-
tion [6] (Inception-v2) by Ioffe et al. Later the architecture
was improved by additional factorization ideas in the third
iteration [15] which will be referred to as Inception-v3 in
this report.
Conv
+
Relu activation
Relu activation
Conv
Figure 1. Residual connections as introduced in He et al. [5].
Conv
+
Relu activation
Relu activation
1x1 Conv
Figure 2. Optimized version of ResNet connections by [5] to shield
computation.
3. Architectural Choices
3.1. Pure Inception blocks
Our older Inception models used to be trained in a par-
titioned manner, where each replica was partitioned into a
multiple sub-networks in order to be able to ﬁt the whole
model in memory. However, the Inception architecture is
highly tunable, meaning that there are a lot of possible
changes to the number of ﬁlters in the various layers that
do not affect the quality of the fully trained network. In
order to optimize the training speed, we used to tune the
layer sizes carefully in order to balance the computation be-
tween the various model sub-networks. In contrast, with the
introduction of TensorFlow our most recent models can be
trained without partitioning the replicas. This is enabled in
part by recent optimizations of memory used by backprop-
agation, achieved by carefully considering what tensors are
needed for gradient computation and structuring the compu-

tation to reduce the number of such tensors. Historically, we
have been relatively conservative about changing the archi-
tectural choices and restricted our experiments to varying
isolated network components while keeping the rest of the
network stable. Not simplifying earlier choices resulted in
networks that looked more complicated that they needed to
be. In our newer experiments, for Inception-v4 we decided
to shed this unnecessary baggage and made uniform choices
for the Inception blocks for each grid size. Plase refer to
Figure 9 for the large scale structure of the Inception-v4 net-
work and Figures 3, 4, 5, 6, 7 and 8 for the detailed struc-
ture of its components. All the convolutions not marked
with “V” in the ﬁgures are same-padded meaning that their
output grid matches the size of their input. Convolutions
marked with “V” are valid padded, meaning that input patch
of each unit is fully contained in the previous layer and the
grid size of the output activation map is reduced accord-
ingly.
3.2. Residual Inception Blocks
For the residual versions of the Inception networks, we
use cheaper Inception blocks than the original Inception.
Each Inception block is followed by ﬁlter-expansion layer
(1 × 1 convolution without activation) which is used for
scaling up the dimensionality of the ﬁlter bank before the
addition to match the depth of the input. This is needed to
compensate for the dimensionality reduction induced by the
Inception block.
We tried several versions of the residual version of In-
ception.
Only two of them are detailed here.
The ﬁrst
one “Inception-ResNet-v1” roughly the computational cost
of Inception-v3, while “Inception-ResNet-v2” matches the
raw cost of the newly introduced Inception-v4 network. See
Figure 15 for the large scale structure of both varianets.
(However, the step time of Inception-v4 proved to be signif-
icantly slower in practice, probably due to the larger number
of layers.)
Another small technical difference between our resid-
ual and non-residual Inception variants is that in the case
of Inception-ResNet, we used batch-normalization only on
top of the traditional layers, but not on top of the summa-
tions. It is reasonable to expect that a thorough use of batch-
normalization should be advantageous, but we wanted to
keep each model replica trainable on a single GPU. It turned
out that the memory footprint of layers with large activa-
tion size was consuming disproportionate amount of GPU-
memory. By omitting the batch-normalization on top of
those layers, we were able to increase the overall number
of Inception blocks substantially. We hope that with bet-
ter utilization of computing resources, making this trade-off
will become unecessary.
3x3 Conv
(32 stride 2 V)
Input 
(299x299x3)
3x3 Conv
(32 V)
3x3 Conv
(64)
3x3 MaxPool
(stride 2 V)
3x3 Conv
(96 stride 2 V)
Filter concat
1x1 Conv
(64)
3x3 Conv
(96 V)
1x1 Conv
(64)
7x1 Conv
(64)
1x7 Conv
(64)
Filter concat
3x3 Conv
(96 V)
MaxPool
(stride=2 V)
3x3 Conv
(192 V)
Filter concat
299x299x3
149x149x32
147x147x32
147x147x64
73x73x160
71x71x192
35x35x384
Figure 3. The schema for stem of the pure Inception-v4 and
Inception-ResNet-v2 networks. This is the input part of those net-
works. Cf. Figures 9 and 15

1x1 Conv 
(96)
1x1 Conv
(64)
1x1 Conv
(64)
3x3 Conv
(96)
3x3 Conv
(96)
3x3 Conv
(96)
Filter concat
Filter concat
Avg Pooling
1x1 Conv
(96)
Figure 4. The schema for 35 × 35 grid modules of the pure
Inception-v4 network. This is the Inception-A block of Figure 9.
1x1 Conv 
(384)
1x1 Conv
(192)
1x1 Conv
(192)
1x7 Conv
(224)
1x7 Conv
(192)
7x1 Conv
(224)
Filter concat
Filter concat
Avg Pooling
1x1 Conv
(128)
1x7 Conv
(256)
1x7 Conv
(224)
7x1 Conv
(256)
Figure 5. The schema for 17 × 17 grid modules of the pure
Inception-v4 network. This is the Inception-B block of Figure 9.
1x1 Conv 
(256)
1x1 Conv
(384)
1x1 Conv
(384)
3x1 Conv
(256)
1x3 Conv
(448)
3x1 Conv
(512)
Filter concat
Filter concat
Avg Pooling
1x1 Conv
(256)
1x3 Conv
(256)
1x3 Conv
(256)
3x1 Conv
(256)
Figure 6. The schema for 8×8 grid modules of the pure Inception-
v4 network. This is the Inception-C block of Figure 9.
1x1 Conv
(k)
3x3 Conv
(n stride 2 V)
3x3 Conv
(l)
3x3 Conv
(m stride 2 V)
Filter concat
Filter concat
3x3 MaxPool 
(stride 2 V)
Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
Different variants of this blocks (with various number of ﬁlters)
are used in Figure 9, and 15 in each of the new Inception(-v4, -
ResNet-v1, -ResNet-v2) variants presented in this paper. The k, l,
m, n numbers represent ﬁlter bank sizes which can be looked up
in Table 1.
1x1 Conv
(256)
1x1 Conv
(192)
1x7 Conv
(256)
3x3 Conv
(320 stride 2 V)
Filter concat
Filter concat
3x3 MaxPool 
(stride 2 V)
3x3 Conv
(192 stride 2 V)
7x1 Conv
(320)
Figure 8. The schema for 17 × 17 to 8 × 8 grid-reduction mod-
ule. This is the reduction module used by the pure Inception-v4
network in Figure 9.

Stem
Input (299x299x3)
299x299x3
4 x Inception-A
Output: 35x35x384
Output: 35x35x384
Reduction-A
Output: 17x17x1024
7 x Inception-B
3 x Inception-C
Reduction-B
Avarage Pooling
Dropout (keep 0.8)
Output: 17x17x1024
Output: 8x8x1536
Output: 8x8x1536
Output: 1536
Softmax
Output: 1536
Output: 1000
Figure 9. The overall schema of the Inception-v4 network. For the
detailed modules, please refer to Figures 3, 4, 5, 6, 7 and 8 for the
detailed structure of the various components.
1x1 Conv 
(32)
1x1 Conv
(32)
1x1 Conv
(32)
3x3 Conv
(32)
3x3 Conv
(32)
3x3 Conv
(32)
1x1 Conv
(256 Linear)
+
Relu activation
Relu activation
Figure 10. The schema for 35 × 35 grid (Inception-ResNet-A)
module of Inception-ResNet-v1 network.
1x1 Conv
(128)
1x1 Conv
(128)
1x7 Conv
(128)
7x1 Conv
(128)
1x1 Conv
(896 Linear)
+
Relu activation
Relu activation
Figure 11. The schema for 17 × 17 grid (Inception-ResNet-B)
module of Inception-ResNet-v1 network.
1x1 Conv
(256)
3x3 Conv
(256 stride 2 V)
Filter concat
Previous 
Layer
3x3 MaxPool 
(stride 2 V)
3x3 Conv
(384 stride 2 V)
3x3 Conv
(256)
1x1 Conv
(256)
3x3 Conv
(256 stride 2 V)
1x1 Conv
(256)
Figure 12. “Reduction-B” 17×17 to 8×8 grid-reduction module.
This module used by the smaller Inception-ResNet-v1 network in
Figure 15.

1x1 Conv
(192)
1x1 Conv
(192)
1x3 Conv
(192)
3x1 Conv
(192)
1x1 Conv
(1792 Linear)
+
Relu activation
Relu activation
Figure 13. The schema for 8×8 grid (Inception-ResNet-C) module
of Inception-ResNet-v1 network.
3x3 Conv
(32 stride 2 V)
Input 
(299x299x3)
3x3 Conv
(32 V)
3x3 Conv
(64)
3x3 MaxPool
(stride 2 V)
1x1 Conv
(80)
299x299x3
149x149x32
147x147x32
147x147x64
73x73x64
73x73x80
3x3 Conv
(192 V)
71x71x192
3x3 Conv
(256 stride 2 V)
35x35x256
Figure 14. The stem of the Inception-ResNet-v1 network.

Stem
Input (299x299x3)
299x299x3
5 x Inception-resnet-A
Output: 35x35x256
Output: 35x35x256
Reduction-A
Output: 17x17x896
10 x
Inception-resnet-B
5 x Inception-resnet-C
Reduction-B
Average Pooling
Dropout (keep 0.8)
Output: 17x17x896
Output: 8x8x1792
Output: 8x8x1792
Output: 1792
Softmax
Output: 1792
Output: 1000
Figure 15. Schema for Inception-ResNet-v1 and Inception-
ResNet-v2 networks. This schema applies to both networks but
the underlying components differ. Inception-ResNet-v1 uses the
blocks as described in Figures 14, 10, 7, 11, 12 and 13. Inception-
ResNet-v2 uses the blocks as described in Figures 3, 16, 7,17, 18
and 19. The output sizes in the diagram refer to the activation
vector tensor shapes of Inception-ResNet-v1.

1x1 Conv 
(32)
1x1 Conv
(32)
1x1 Conv
(32)
3x3 Conv
(32)
3x3 Conv
(48)
3x3 Conv
(64)
1x1 Conv
(384 Linear)
+
Relu activation
Relu activation
Figure 16. The schema for 35 × 35 grid (Inception-ResNet-A)
module of the Inception-ResNet-v2 network.
1x1 Conv
(192)
1x1 Conv
(128)
1x7 Conv
(160)
7x1 Conv
(192)
1x1 Conv
(1154 Linear)
+
Relu activation
Relu activation
Figure 17. The schema for 17 × 17 grid (Inception-ResNet-B)
module of the Inception-ResNet-v2 network.
1x1 Conv
(256)
3x3 Conv
(320 stride 2 V)
Filter concat
Previous 
Layer
3x3 MaxPool 
(stride 2 V)
3x3 Conv
(384 stride 2 V)
3x3 Conv
(288)
1x1 Conv
(256)
3x3 Conv
(288 stride 2 V)
1x1 Conv
(256)
Figure 18. The schema for 17 × 17 to 8 × 8 grid-reduction mod-
ule. Reduction-B module used by the wider Inception-ResNet-v1
network in Figure 15.
1x1 Conv
(192)
1x1 Conv
(192)
1x3 Conv
(224)
3x1 Conv
(256)
1x1 Conv
(2048 Linear)
+
Relu activation
Relu activation
Figure 19. The schema for 8×8 grid (Inception-ResNet-C) module
of the Inception-ResNet-v2 network.
Network
k
l
m
n
Inception-v4
192
224
256
384
Inception-ResNet-v1
192
192
256
384
Inception-ResNet-v2
256
256
384
384
Table 1. The number of ﬁlters of the Reduction-A module for the
three Inception variants presented in this paper. The four numbers
in the colums of the paper parametrize the four convolutions of
Figure 7

Activation 
Scaling
+
Relu activation
Relu activation
Inception
Figure 20. The general schema for scaling combined Inception-
resnet moduels. We expect that the same idea is useful in the gen-
eral resnet case, where instead of the Inception block an arbitrary
subnetwork is used. The scaling block just scales the last linear
activations by a suitable constant, typically around 0.1.
3.3. Scaling of the Residuals
Also we found that if the number of ﬁlters exceeded
1000, the residual variants started to exhibit instabilities and
the network has just “died” early in the training, meaning
that the last layer before the average pooling started to pro-
duce only zeros after a few tens of thousands of iterations.
This could not be prevented, neither by lowering the learn-
ing rate, nor by adding an extra batch-normalization to this
layer.
We found that scaling down the residuals before adding
them to the previous layer activation seemed to stabilize the
training. In general we picked some scaling factors between
0.1 and 0.3 to scale the residuals before their being added to
the accumulated layer activations (cf. Figure 20).
A similar instability was observed by He et al. in [5] in
the case of very deep residual networks and they suggested a
two-phase training where the ﬁrst “warm-up” phase is done
with very low learning rate, followed by a second phase
with high learning rata. We found that if the number of
ﬁlters is very high, then even a very low (0.00001) learning
rate is not sufﬁcient to cope with the instabilities and the
training with high learning rate had a chance to destroy its
effects. We found it much more reliable to just scale the
residuals.
Even where the scaling was not strictly necessary, it
never seemed to harm the ﬁnal accuracy, but it helped to
stabilize the training.
4. Training Methodology
We have trained our networks with stochastic gradient
utilizing the TensorFlow [1] distributed machine learning
system using 20 replicas running each on a NVidia Kepler
GPU. Our earlier experiments used momentum [13] with a
decay of 0.9, while our best models were achieved using
20
40
60
80
100
120
140
160
180
200
Epoch
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
Error (top-1) %
inception-v3
inception-resnet-v1
Figure 21. Top-1 error evolution during training of pure Inception-
v3 vs a residual network of similar computational cost. The eval-
uation is measured on a single crop on the non-blacklist images of
the ILSVRC-2012 validation set. The residual model was train-
ing much faster, but reached slightly worse ﬁnal accuracy than the
traditional Inception-v3.
RMSProp [16] with decay of 0.9 and ϵ = 1.0. We used a
learning rate of 0.045, decayed every two epochs using an
exponential rate of 0.94. Model evaluations are performed
using a running average of the parameters computed over
time.
5. Experimental Results
First we observe the top-1 and top-5 validation-error evo-
lution of the four variants during training. After the exper-
iment was conducted, we have found that our continuous
evaluation was conducted on a subset of the validation set
which omitted about 1700 blacklisted entities due to poor
bounding boxes.
It turned out that the omission should
have been only performed for the CLSLOC benchmark, but
yields somewhat incomparable (more optimistic) numbers
when compared to other reports including some earlier re-
ports by our team. The difference is about 0.3% for top-1
error and about 0.15% for the top-5 error. However, since
the differences are consistent, we think the comparison be-
tween the curves is a fair one.
On the other hand, we have rerun our multi-crop and en-
semble results on the complete validation set consisting of
50000 images. Also the ﬁnal ensemble result was also per-
formed on the test set and sent to the ILSVRC test server
for validation to verify that our tuning did not result in an
over-ﬁtting. We would like to stress that this ﬁnal validation
was done only once and we have submitted our results only
twice in the last year: once for the BN-Inception paper and
later during the ILSVR-2015 CLSLOC competition, so we
believe that the test set numbers constitute a true estimate
of the generalization capabilities of our model.
Finally, we present some comparisons, between various
versions of Inception and Inception-ResNet. The models
Inception-v3 and Inception-v4 are deep convolutional net-

20
40
60
80
100
120
140
160
180
200
Epoch
3.0
3.5
4.0
4.5
5.0
5.5
6.0
6.5
7.0
7.5
8.0
8.5
9.0
9.5
Error (top-5) %
inception-v3
inception-resnet-v1
Figure 22. Top-5 error evolution during training of pure Inception-
v3 vs a residual Inception of similar computational cost. The eval-
uation is measured on a single crop on the non-blacklist images of
the ILSVRC-2012 validation set. The residual version has trained
much faster and reached slightly better ﬁnal recall on the valida-
tion set.
20
40
60
80
100
120
140
160
Epoch
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
Error (top-1) %
inception-v4
inception-resnet-v2
Figure 23. Top-1 error evolution during training of pure Inception-
v3 vs a residual Inception of similar computational cost. The eval-
uation is measured on a single crop on the non-blacklist images of
the ILSVRC-2012 validation set. The residual version was train-
ing much faster and reached slightly better ﬁnal accuracy than the
traditional Inception-v4.
Network
Top-1 Error
Top-5 Error
BN-Inception [6]
25.2%
7.8%
Inception-v3 [15]
21.2%
5.6%
Inception-ResNet-v1
21.3%
5.5%
Inception-v4
20.0%
5.0%
Inception-ResNet-v2
19.9%
4.9%
Table 2. Single crop - single model experimental results. Reported
on the non-blacklisted subset of the validation set of ILSVRC
2012.
works not utilizing residual connections while Inception-
ResNet-v1 and Inception-ResNet-v2 are Inception style net-
works that utilize residual connections instead of ﬁlter con-
catenation.
Table 2 shows the single-model, single crop top-1 and
top-5 error of the various architectures on the validation set.
20
40
60
80
100
120
140
160
Epoch
3
4
5
6
7
8
9
Error (top-5) %
inception-v4
inception-resnet-v2
Figure 24. Top-5 error evolution during training of pure Inception-
v4 vs a residual Inception of similar computational cost. The eval-
uation is measured on a single crop on the non-blacklist images
of the ILSVRC-2012 validation set. The residual version trained
faster and reached slightly better ﬁnal recall on the validation set.
20
40
60
80
100
120
140
160
Epoch
2.5
3.0
3.5
4.0
4.5
5.0
5.5
6.0
6.5
7.0
7.5
8.0
8.5
9.0
9.5
Error (top-5) %
inception-v4
inception-resnet-v2
inception-v3
inception-resnet-v1
Figure 25. Top-5 error evolution of all four models (single model,
single crop). Showing the improvement due to larger model size.
Although the residual version converges faster, the ﬁnal accuracy
seems to mainly depend on the model size.
20
40
60
80
100
120
140
160
Epoch
18
19
20
21
22
23
24
25
26
27
28
29
Error (top-1) %
inception-v4
inception-resnet-v2
inception-v3
inception-resnet-v1
Figure 26. Top-1 error evolution of all four models (single model,
single crop). This paints a similar picture as the top-5 evaluation.
Table 3 shows the performance of the various models
with a small number of crops: 10 crops for ResNet as was
reported in [5]), for the Inception variants, we have used the
12 crops evaluation as as described in [14].

Network
Crops
Top-1 Error
Top-5 Error
ResNet-151 [5]
10
21.4%
5.7%
Inception-v3 [15]
12
19.8%
4.6%
Inception-ResNet-v1
12
19.8%
4.6%
Inception-v4
12
18.7%
4.2%
Inception-ResNet-v2
12
18.7%
4.1%
Table 3. 10/12 crops evaluations - single model experimental re-
sults. Reported on the all 50000 images of the validation set of
ILSVRC 2012.
Network
Crops
Top-1 Error
Top-5 Error
ResNet-151 [5]
dense
19.4%
4.5%
Inception-v3 [15]
144
18.9%
4.3%
Inception-ResNet-v1
144
18.8%
4.3%
Inception-v4
144
17.7%
3.8%
Inception-ResNet-v2
144
17.8%
3.7%
Table 4. 144 crops evaluations - single model experimental results.
Reported on the all 50000 images of the validation set of ILSVRC
2012.
Network
Models
Top-1 Error
Top-5 Error
ResNet-151 [5]
6
–
3.6%
Inception-v3 [15]
4
17.3%
3.6%
Inception-v4 +
3× Inception-ResNet-v2
4
16.5%
3.1%
Table 5. Ensemble results with 144 crops/dense evaluation. Re-
ported on the all 50000 images of the validation set of ILSVRC
2012. For Inception-v4(+Residual), the ensemble consists of one
pure Inception-v4 and three Inception-ResNet-v2 models and were
evaluated both on the validation and on the test-set. The test-set
performance was 3.08% top-5 error verifying that we don’t over-
ﬁt on the validation set.
Table 4 shows the single model performance of the var-
ious models using. For residual network the dense evalua-
tion result is reported from [5]. For the inception networks,
the 144 crops strategy was used as described in [14].
Table 5 compares ensemble results. For the pure resid-
ual network the 6 models dense evaluation result is reported
from [5]. For the inception networks 4 models were ensem-
bled using the 144 crops strategy as described in [14].
6. Conclusions
We have presented three new network architectures in
detail:
• Inception-ResNet-v1: a hybrid Inception version that
has a similar computational cost to Inception-v3
from [15].
• Inception-ResNet-v2: a costlier hybrid Inception ver-
sion with signiﬁcantly improved recognition perfor-
mance.
• Inception-v4: a pure Inception variant without residual
connections with roughly the same recognition perfor-
mance as Inception-ResNet-v2.
We studied how the introduction of residual connections
leads to dramatically improved training speed for the Incep-
tion architecture. Also our latest models (with and without
residual connections) outperform all our previous networks,
just by virtue of the increased model size.
References
[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen,
C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghe-
mawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia,
R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man´e,
R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster,
J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker,
V. Vanhoucke, V. Vasudevan, F. Vi´egas, O. Vinyals, P. War-
den, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. Tensor-
Flow: Large-scale machine learning on heterogeneous sys-
tems, 2015. Software available from tensorﬂow.org.
[2] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao,
A. Senior, P. Tucker, K. Yang, Q. V. Le, et al. Large scale dis-
tributed deep networks. In Advances in Neural Information
Processing Systems, pages 1223–1231, 2012.
[3] C. Dong, C. C. Loy, K. He, and X. Tang. Learning a deep
convolutional network for image super-resolution. In Com-
puter Vision–ECCV 2014, pages 184–199. Springer, 2014.
[4] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich fea-
ture hierarchies for accurate object detection and semantic
segmentation. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2014.
[5] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learn-
ing for image recognition. arXiv preprint arXiv:1512.03385,
2015.
[6] S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift. In
Proceedings of The 32nd International Conference on Ma-
chine Learning, pages 448–456, 2015.
[7] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar,
and L. Fei-Fei. Large-scale video classiﬁcation with con-
volutional neural networks.
In Computer Vision and Pat-
tern Recognition (CVPR), 2014 IEEE Conference on, pages
1725–1732. IEEE, 2014.
[8] A. Krizhevsky, I. Sutskever, and G. E. Hinton.
Imagenet
classiﬁcation with deep convolutional neural networks. In
Advances in neural information processing systems, pages
1097–1105, 2012.
[9] M. Lin, Q. Chen, and S. Yan. Network in network. arXiv
preprint arXiv:1312.4400, 2013.
[10] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional
networks for semantic segmentation. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recogni-
tion, pages 3431–3440, 2015.
[11] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,
S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,

et al.
Imagenet large scale visual recognition challenge.
2014.
[12] K. Simonyan and A. Zisserman. Very deep convolutional
networks for large-scale image recognition. arXiv preprint
arXiv:1409.1556, 2014.
[13] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the
importance of initialization and momentum in deep learning.
In Proceedings of the 30th International Conference on Ma-
chine Learning (ICML-13), volume 28, pages 1139–1147.
JMLR Workshop and Conference Proceedings, May 2013.
[14] C. Szegedy,
W. Liu,
Y. Jia,
P. Sermanet,
S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition,
pages 1–9, 2015.
[15] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna.
Rethinking the inception architecture for computer vision.
arXiv preprint arXiv:1512.00567, 2015.
[16] T. Tieleman and G. Hinton. Divide the gradient by a run-
ning average of its recent magnitude. COURSERA: Neural
Networks for Machine Learning, 4, 2012. Accessed: 2015-
11-05.
[17] A. Toshev and C. Szegedy. Deeppose: Human pose estima-
tion via deep neural networks. In Computer Vision and Pat-
tern Recognition (CVPR), 2014 IEEE Conference on, pages
1653–1660. IEEE, 2014.
[18] N. Wang and D.-Y. Yeung. Learning a deep compact image
representation for visual tracking.
In Advances in Neural
Information Processing Systems, pages 809–817, 2013.
