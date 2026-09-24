# Mask R CNN He Gkioxari 2017

> Source: `Mask_R_CNN_He_Gkioxari_2017.pdf`

---

                                                                                              Mask R-CNN

                                                                Kaiming He         Georgia Gkioxari            Piotr Dollár         Ross Girshick
                                                                                      Facebook AI Research (FAIR)


                                                                 Abstract




arXiv:1703.06870v2 [cs.CV] 5 Apr 2017
                                                                                                                                                                 class
                                            We present a conceptually simple, flexible, and general                                                              box
                                        framework for object instance segmentation. Our approach
                                        efficiently detects objects in an image while simultaneously                                    RoIAlign
                                        generating a high-quality segmentation mask for each in-                                                        conv        conv
                                        stance. The method, called Mask R-CNN, extends Faster
                                        R-CNN by adding a branch for predicting an object mask in
                                        parallel with the existing branch for bounding box recogni-
                                        tion. Mask R-CNN is simple to train and adds only a small
                                        overhead to Faster R-CNN, running at 5 fps. Moreover,                 Figure 1. The Mask R-CNN framework for instance segmentation.
                                        Mask R-CNN is easy to generalize to other tasks, e.g., al-
                                        lowing us to estimate human poses in the same framework.              a fixed set of categories without differentiating object in-
                                        We show top results in all three tracks of the COCO suite of          stances.1 Given this, one might expect a complex method
                                        challenges, including instance segmentation, bounding-box             is required to achieve good results. However, we show that
                                        object detection, and person keypoint detection. Without              a surprisingly simple, flexible, and fast system can surpass
                                        tricks, Mask R-CNN outperforms all existing, single-model             prior state-of-the-art instance segmentation results.
                                        entries on every task, including the COCO 2016 challenge                  Our method, called Mask R-CNN, extends Faster R-CNN
                                        winners. We hope our simple and effective approach will               [34] by adding a branch for predicting segmentation masks
                                        serve as a solid baseline and help ease future research in            on each Region of Interest (RoI), in parallel with the ex-
                                        instance-level recognition. Code will be made available.              isting branch for classification and bounding box regres-
                                                                                                              sion (Figure 1). The mask branch is a small FCN applied
                                                                                                              to each RoI, predicting a segmentation mask in a pixel-to-
                                        1. Introduction                                                       pixel manner. Mask R-CNN is simple to implement and
                                                                                                              train given the Faster R-CNN framework, which facilitates
                                           The vision community has rapidly improved object de-               a wide range of flexible architecture designs. Additionally,
                                        tection and semantic segmentation results over a short pe-            the mask branch only adds a small computational overhead,
                                        riod of time. In large part, these advances have been driven          enabling a fast system and rapid experimentation.
                                        by powerful baseline systems, such as the Fast/Faster R-                  In principle Mask R-CNN is an intuitive extension of
                                        CNN [12, 34] and Fully Convolutional Network (FCN) [29]               Faster R-CNN, yet constructing the mask branch properly
                                        frameworks for object detection and semantic segmenta-                is critical for good results. Most importantly, Faster R-
                                        tion, respectively. These methods are conceptually intuitive          CNN was not designed for pixel-to-pixel alignment be-
                                        and offer flexibility and robustness, together with fast train-       tween network inputs and outputs. This is most evident in
                                        ing and inference time. Our goal in this work is to develop a         how RoIPool [18, 12], the de facto core operation for at-
                                        comparably enabling framework for instance segmentation.              tending to instances, performs coarse spatial quantization
                                           Instance segmentation is challenging because it requires           for feature extraction. To fix the misalignment, we pro-
                                        the correct detection of all objects in an image while also           pose a simple, quantization-free layer, called RoIAlign, that
                                        precisely segmenting each instance. It therefore combines             faithfully preserves exact spatial locations. Despite being
                                        elements from the classical computer vision tasks of ob-                 1 Following common terminology, we use object detection to denote
                                        ject detection, where the goal is to classify individual ob-          detection via bounding boxes, not masks, and semantic segmentation to
                                        jects and localize each using a bounding box, and semantic            denote per-pixel classification without differentiating instances. Yet we
                                        segmentation, where the goal is to classify each pixel into           note that instance segmentation is both semantic and a form of detection.


                                                                                                          1
                                                                                                                                                                            umbrella.98                       bus.99

                                                                                                                           umbrella.98
                                                                                                                                                                 person1.00



                                                                                                                                         person1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                          person1.00
                                                                                                                                                                                backpack1.00
                                                                                                                                                                                                                                                                                              person1.00               person.99
                                                                                                                                     handbag.96                                                                                                                                                                                                                                                                                           person.99
                                                                    person1.00                                                                                                                                                              person1.00        person1.00
 person1.00                                                                                                                                                                                                           person1.00
                                                                                                                                                                                                                   person.95           person.98
                 person1.00
                              person1.00     person1.00 person.94                      person1.00                                                                                                  person1.00 person.89

                                                                                                       person1.00                                                                                                                                                                                                                                           sheep.99
                                                                                                                                                                                                                                              backpack.99
                                                                                                                                                                                                                                                                                                                                                                       sheep.99                                           sheep.86
                                                                                                                                                                                                                                                                                                              backpack.93                                                                                          sheep.82            sheep.96
                                                                                                                                                                                                                                                                                                                                                        sheep.96                         sheep.93       sheep.91        sheep.95 sheep.96    sheep1.00
                                                                                                                                                                                                                                                                                                                                        sheep1.00
                                                                                                                                                                                                                                                                                                                                                                                  sheep.99
                                                                                                                                                                                                                                                                                                                                                                                            sheep1.00
                                                                                                                                                                                                                                                                                                                                                              sheep.99
                                                                                                                                                                                                                                                                                                                                                       sheep.96

                                                                                                                                                                                                                                                                                                                                     sheep.99




                                                                                                                                                                                               person.99
                                                        bottle.99
                                 dining table.96


                                                                       bottle.99
                                                                                                            bottle.99




                                                                                                                                                      person.99person1.00
                                                                                                                              person1.00
                                                                                                                                                                                                                                                                                                traffic light.96                                                                                        tv.99



                                                                                                                                                                                                                                                                                                                                                            chair.98                                                                                       chair.99
                                                                                                                                                                                                                                                                                                                                      chair.90
                                                                                                                                                                                                                                                                                                                                          dining table.99                                                   chair.96                     wine glass.97
                                                                                                                                                                                                                                                                                                                                                                              chair.86
                                                                                                                                                                                                                                                                                                                                                                                                         bottle.99wine glass.93                                    chair.99
                                                                                                                                                                                                                                                                                                                                                                                                                          bowl.85                wine glass1.00

                                                                                                                          elephant1.00
                                                                                                                                                                                                                                                                                                                                                                                                                         wine glass.99
                                                                                                                                                                                                                                                                                                                                                                                    wine glass1.00
                                                                                                                                                                                                                                                                 person1.00                                                          chair.96                                                                            chair.99                        fork.95

                 person1.00                                                                                                                                                                           traffic light.95                                                                                                                                                                                                          bowl.81
                                           person1.00
                                                                                                                                                                                                   traffic light.92                                                        traffic light.84
                                                                                                                                                                                                                                                                                                           person1.00    person.85
                                                                                                            person.96                                                                             truck1.00                                                                                                person.99
                                           motorcycle1.00                                           person.96person1.00
                                                                                                       person.83                                                                               person1.00
    motorcycle1.00                                                               person.98
                                                                                         person.99person.91
                                                                                     person.90                                                                                                                           person.87   car.99        car.92
                                                                                                                                                                                                                                                       person.99
                                                                                                         person.92                                                                                                                             car.99        car.93
                                                                                                                                                                                                                          car1.00
                                                                                                                                                                                                                                                       motorcycle.95
                                                                                                                                                                                                                                                                                                                                                                           knife.83



                                                                                                                                                                                                                                                                                                                        person.96




Figure 2. Mask R-CNN results on the COCO test set. These results are based on ResNet-101 [19], achieving a mask AP of 35.7 and
running at 5 fps. Masks are shown in color, and bounding box, category, and confidences are also shown.

a seemingly minor change, RoIAlign has a large impact: it                                                                                                                                                     2. Related Work
improves mask accuracy by relative 10% to 50%, showing
bigger gains under stricter localization metrics. Second, we                                                                                                                                                  R-CNN: The Region-based CNN (R-CNN) approach [13]
found it essential to decouple mask and class prediction: we                                                                                                                                                  to bounding-box object detection is to attend to a manage-
predict a binary mask for each class independently, without                                                                                                                                                   able number of candidate object regions [38, 20] and evalu-
                                                                                                                                                                                                              ate convolutional networks [25, 24] independently on each
competition among classes, and rely on the network’s RoI
classification branch to predict the category. In contrast,                                                                                                                                                   RoI. R-CNN was extended [18, 12] to allow attending to
FCNs usually perform per-pixel multi-class categorization,                                                                                                                                                    RoIs on feature maps using RoIPool, leading to fast speed
which couples segmentation and classification, and based                                                                                                                                                      and better accuracy. Faster R-CNN [34] advanced this
on our experiments works poorly for instance segmentation.                                                                                                                                                    stream by learning the attention mechanism with a Region
                                                                                                                                                                                                              Proposal Network (RPN). Faster R-CNN is flexible and ro-
   Without bells and whistles, Mask R-CNN surpasses all                                                                                                                                                       bust to many follow-up improvements (e.g., [35, 27, 21]),
previous state-of-the-art single-model results on the COCO                                                                                                                                                    and is the current leading framework in several benchmarks.
instance segmentation task [28], including the heavily-
engineered entries from the 2016 competition winner. As                                                                                                                                                       Instance Segmentation: Driven by the effectiveness of R-
a by-product, our method also excels on the COCO object                                                                                                                                                       CNN, many approaches to instance segmentation are based
detection task. In ablation experiments, we evaluate multi-                                                                                                                                                   on segment proposals. Earlier methods [13, 15, 16, 9] re-
ple basic instantiations, which allows us to demonstrate its                                                                                                                                                  sorted to bottom-up segments [38, 2]. DeepMask [32] and
robustness and analyze the effects of core factors.                                                                                                                                                           following works [33, 8] learn to propose segment candi-
                                                                                                                                                                                                              dates, which are then classified by Fast R-CNN. In these
   Our models can run at about 200ms per frame on a GPU,
                                                                                                                                                                                                              methods, segmentation precedes recognition, which is slow
and training on COCO takes one to two days on a single
                                                                                                                                                                                                              and less accurate. Likewise, Dai et al. [10] proposed a com-
8-GPU machine. We believe the fast train and test speeds,
                                                                                                                                                                                                              plex multiple-stage cascade that predicts segment proposals
together with the framework’s flexibility and accuracy, will
                                                                                                                                                                                                              from bounding-box proposals, followed by classification.
benefit and ease future research on instance segmentation.
                                                                                                                                                                                                              Instead, our method is based on parallel prediction of masks
   Finally, we showcase the generality of our framework                                                                                                                                                       and class labels, which is simpler and more flexible.
via the task of human pose estimation on the COCO key-                                                                                                                                                           Most recently, Li et al. [26] combined the segment pro-
point dataset [28]. By viewing each keypoint as a one-hot                                                                                                                                                     posal system in [8] and object detection system in [11] for
binary mask, with minimal modification Mask R-CNN can                                                                                                                                                         “fully convolutional instance segmentation” (FCIS). The
be applied to detect instance-specific poses. Without tricks,                                                                                                                                                 common idea in [8, 11, 26] is to predict a set of position-
Mask R-CNN surpasses the winner of the 2016 COCO key-                                                                                                                                                         sensitive output channels fully convolutionally. These
point competition, and at the same time runs at 5 fps. Mask                                                                                                                                                   channels simultaneously address object classes, boxes, and
R-CNN, therefore, can be seen more broadly as a flexible                                                                                                                                                      masks, making the system fast. But FCIS exhibits system-
framework for instance-level recognition and can be readily                                                                                                                                                   atic errors on overlapping instances and creates spurious
extended to more complex tasks.                                                                                                                                                                               edges (Figure 5), showing that it is challenged by the fun-
   We will release code to facilitate future research.                                                                                                                                                        damental difficulties of segmenting instances.


                                                                                                                                                                                               2
3. Mask R-CNN                                                       Mask Representation: A mask encodes an input object’s
                                                                    spatial layout. Thus, unlike class labels or box offsets
    Mask R-CNN is conceptually simple: Faster R-CNN has             that are inevitably collapsed into short output vectors by
two outputs for each candidate object, a class label and a          fully-connected (fc) layers, extracting the spatial structure
bounding-box offset; to this we add a third branch that out-        of masks can be addressed naturally by the pixel-to-pixel
puts the object mask. Mask R-CNN is thus a natural and in-          correspondence provided by convolutions.
tuitive idea. But the additional mask output is distinct from          Specifically, we predict an m × m mask from each RoI
the class and box outputs, requiring extraction of much finer       using an FCN [29]. This allows each layer in the mask
spatial layout of an object. Next, we introduce the key ele-        branch to maintain the explicit m × m object spatial lay-
ments of Mask R-CNN, including pixel-to-pixel alignment,            out without collapsing it into a vector representation that
which is the main missing piece of Fast/Faster R-CNN.               lacks spatial dimensions. Unlike previous methods that re-
                                                                    sort to fc layers for mask prediction [32, 33, 10], our fully
Faster R-CNN: We begin by briefly reviewing the Faster
                                                                    convolutional representation requires fewer parameters, and
R-CNN detector [34]. Faster R-CNN consists of two stages.
                                                                    is more accurate as demonstrated by experiments.
The first stage, called a Region Proposal Network (RPN),
                                                                       This pixel-to-pixel behavior requires our RoI features,
proposes candidate object bounding boxes. The second
                                                                    which themselves are small feature maps, to be well aligned
stage, which is in essence Fast R-CNN [12], extracts fea-
                                                                    to faithfully preserve the explicit per-pixel spatial corre-
tures using RoIPool from each candidate box and performs
                                                                    spondence. This motivated us to develop the following
classification and bounding-box regression. The features
                                                                    RoIAlign layer that plays a key role in mask prediction.
used by both stages can be shared for faster inference. We
refer readers to [21] for latest, comprehensive comparisons         RoIAlign: RoIPool [12] is a standard operation for extract-
between Faster R-CNN and other frameworks.                          ing a small feature map (e.g., 7×7) from each RoI. RoIPool
                                                                    first quantizes a floating-number RoI to the discrete granu-
Mask R-CNN: Mask R-CNN adopts the same two-stage                    larity of the feature map, this quantized RoI is then subdi-
procedure, with an identical first stage (which is RPN). In         vided into spatial bins which are themselves quantized, and
the second stage, in parallel to predicting the class and box       finally feature values covered by each bin are aggregated
offset, Mask R-CNN also outputs a binary mask for each              (usually by max pooling). Quantization is performed, e.g.,
RoI. This is in contrast to most recent systems, where clas-        on a continuous coordinate x by computing [x/16], where
sification depends on mask predictions (e.g. [32, 10, 26]).         16 is a feature map stride and [·] is rounding; likewise, quan-
Our approach follows the spirit of Fast R-CNN [12] that             tization is performed when dividing into bins (e.g., 7×7).
applies bounding-box classification and regression in par-          These quantizations introduce misalignments between the
allel (which turned out to largely simplify the multi-stage         RoI and the extracted features. While this may not impact
pipeline of original R-CNN [13]).                                   classification, which is robust to small translations, it has a
    Formally, during training, we define a multi-task loss on       large negative effect on predicting pixel-accurate masks.
each sampled RoI as L = Lcls + Lbox + Lmask . The clas-                 To address this, we propose an RoIAlign layer that re-
sification loss Lcls and bounding-box loss Lbox are identi-         moves the harsh quantization of RoIPool, properly aligning
cal as those defined in [12]. The mask branch has a Km2 -           the extracted features with the input. Our proposed change
dimensional output for each RoI, which encodes K binary             is simple: we avoid any quantization of the RoI boundaries
masks of resolution m × m, one for each of the K classes.           or bins (i.e., we use x/16 instead of [x/16]). We use bilinear
To this we apply a per-pixel sigmoid, and define Lmask as           interpolation [22] to compute the exact values of the input
the average binary cross-entropy loss. For an RoI associated        features at four regularly sampled locations in each RoI bin,
with ground-truth class k, Lmask is only defined on the k-th        and aggregate the result (using max or average).2
mask (other mask outputs do not contribute to the loss).                RoIAlign leads to large improvements as we show in
    Our definition of Lmask allows the network to generate          §4.2. We also compare to the RoIWarp operation proposed
masks for every class without competition among classes;            in [10]. Unlike RoIAlign, RoIWarp overlooked the align-
we rely on the dedicated classification branch to predict the       ment issue and was implemented in [10] as quantizing RoI
class label used to select the output mask. This decouples          just like RoIPool. So even though RoIWarp also adopts
mask and class prediction. This is different from common            bilinear resampling motivated by [22], it performs on par
practice when applying FCNs [29] to semantic segmenta-              with RoIPool as shown by experiments (more details in Ta-
tion, which typically uses a per-pixel softmax and a multino-       ble 2c), demonstrating the crucial role of alignment.
mial cross-entropy loss. In that case, masks across classes            2 We sample four regular locations, so that we can evaluate either max
compete; in our case, with a per-pixel sigmoid and a binary         or average pooling. In fact, interpolating only a single value at each bin
loss, they do not. We show by experiments that this formu-          center (without pooling) is nearly as effective. One could also sample more
lation is key for good instance segmentation results.               than four locations per bin, which we found to give diminishing returns.


                                                                3
                                                                                               Faster R-CNN                               Faster R-CNN
Network Architecture: To demonstrate the generality of                                         w/ ResNet [19]                              w/ FPN [27]
                                                                                                          class                                      class
our approach, we instantiate Mask R-CNN with multiple                     7×7        7×7 ave                            7×7
                                                                     RoI ×1024 res5 ×2048    2048                 RoI   ×256    1024    1024
                                                                                                          box                                        box
architectures. For clarity, we differentiate between: (i) the
convolutional backbone architecture used for feature ex-
                                                                                             14×14      14×14           14×14   14×14   28×28      28×28
traction over an entire image, and (ii) the network head                                      ×256       ×80      RoI   ×256 ×4 ×256     ×256       ×80

for bounding-box recognition (classification and regression)                                              mask                                       mask
and mask prediction that is applied separately to each RoI.          Figure 3. Head Architecture: We extend two existing Faster R-
    We denote the backbone architecture using the nomen-             CNN heads [19, 27]. Left/Right panels show the heads for the
clature network-depth-features. We evaluate ResNet [19]              ResNet C4 and FPN backbones, from [19] and [27], respectively,
and ResNeXt [40] networks of depth 50 or 101 layers. The             to which a mask branch is added. Numbers denote spatial resolu-
original implementation of Faster R-CNN with ResNets                 tion and channels. Arrows denote either conv, deconv, or fc layers
[19] extracted features from the final convolutional layer           as can be inferred from context (conv preserves spatial dimension
of the 4-th stage, which we call C4. This backbone with              while deconv increases it). All convs are 3×3, except the output
                                                                     conv which is 1×1, deconvs are 2×2 with stride 2, and we use
ResNet-50, for example, is denoted by ResNet-50-C4. This
                                                                     ReLU [30] in hidden layers. Left: ‘res5’ denotes ResNet’s fifth
is a common choice used in [19, 10, 21, 36].
                                                                     stage, which for simplicity we altered so that the first conv oper-
    We also explore another more effective backbone re-              ates on a 7×7 RoI with stride 1 (instead of 14×14 / stride 2 as in
cently proposed by Lin et al. [27], called a Feature Pyra-           [19]). Right: ‘×4’ denotes a stack of four consecutive convs.
mid Network (FPN). FPN uses a top-down architecture with
lateral connections to build an in-network feature pyramid
from a single-scale input. Faster R-CNN with an FPN back-            [12]. N is 64 for the C4 backbone (as in [12, 34]) and 512
bone extracts RoI features from different levels of the fea-         for FPN (as in [27]). We train on 8 GPUs (so effective mini-
ture pyramid according to their scale, but otherwise the             batch size is 16) for 160k iterations, with a learning rate of
rest of the approach is similar to vanilla ResNet. Using a           0.02 which is decreased by 10 at the 120k iteration. We use
ResNet-FPN backbone for feature extraction with Mask R-              a weight decay of 0.0001 and a momentum of 0.9.
CNN gives excellent gains in both accuracy and speed. For               The RPN anchors span 5 scales and 3 aspect ratios, fol-
further details on FPN, we refer readers to [27].                    lowing [27]. For convenient ablation, RPN is trained sep-
    For the network head we closely follow architectures             arately and does not share features with Mask R-CNN, un-
presented in previous work to which we add a fully con-              less specified. For every entry in this paper, RPN and Mask
volutional mask prediction branch. Specifically, we ex-              R-CNN have the same backbones and so they are shareable.
tend the Faster R-CNN box heads from the ResNet [19]
and FPN [27] papers. Details are shown in Figure 3. The              Inference: At test time, the proposal number is 300 for the
head on the ResNet-C4 backbone includes the 5-th stage of            C4 backbone (as in [34]) and 1000 for FPN (as in [27]). We
ResNet (namely, the 9-layer ‘res5’ [19]), which is compute-          run the box prediction branch on these proposals, followed
intensive. For FPN, the backbone already includes res5 and           by non-maximum suppression [14]. The mask branch is
thus allows for a more efficient head that uses fewer filters.       then applied to the highest scoring 100 detection boxes. Al-
    We note that our mask branches have a straightforward            though this differs from the parallel computation used in
structure. More complex designs have the potential to im-            training, it speeds up inference and improves accuracy (due
prove performance but are not the focus of this work.                to the use of fewer, more accurate RoIs). The mask branch
                                                                     can predict K masks per RoI, but we only use the k-th mask,
3.1. Implementation Details                                          where k is the predicted class by the classification branch.
   We set hyper-parameters following existing Fast/Faster            The m×m floating-number mask output is then resized to
R-CNN work [12, 34, 27]. Although these decisions were               the RoI size, and binarized at a threshold of 0.5.
made for object detection in original papers [12, 34, 27], we            Note that since we only compute masks on the top 100
found our instance segmentation system is robust to them.            detection boxes, Mask R-CNN adds marginal runtime to its
                                                                     Faster R-CNN counterpart (e.g., ∼20% on typical models).
Training: As in Fast R-CNN, an RoI is considered positive
if it has IoU with a ground-truth box of at least 0.5 and            4. Experiments: Instance Segmentation
negative otherwise. The mask loss Lmask is defined only on
positive RoIs. The mask target is the intersection between              We perform a thorough comparison of Mask R-CNN to
an RoI and its associated ground-truth mask.                         the state of the art along with comprehensive ablation exper-
    We adopt image-centric training [12]. Images are resized         iments. We use the COCO dataset [28] for all experiments.
such that their scale (shorter edge) is 800 pixels [27]. Each        We report the standard COCO metrics including AP (aver-
mini-batch has 2 images per GPU and each image has N                 aged over IoU thresholds), AP50 , AP75 , and APS , APM ,
sampled RoIs, with a ratio of 1:3 of positive to negatives           APL (AP at different scales). Unless otherwise noted, AP


                                                                 4
                                                                                                                                                                                                                                                           person1.00         person1.00      person1.00

                                                                                                                                                                                                                                        person1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       person.93
                        person1.00        person.99
                                                         person.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         person.95
   person1.00                                                    person1.00                           umbrella.97
                                                                                                                                                                                                                                                                                                                                                                 person1.00             person1.00 person.98                                                                                                    person.98       person.93
                                                                                                                  umbrella.97                                                                                                                                                                                                                                               person1.00          surfboard1.00
                                     person1.00                                                            person.99                                               umbrella.96 umbrella.99                                                                                                                                                         person1.00
                                                           skateboard.91                                                                                                                                                                                                                                                                                   person.91                                                                                                                                                                               person1.00
bench.76                                                                                                                          umbrella1.00                       person1.00                                                                                                                                                                                                                                                                                                                                                           person1.00
                                                                                                      person.99
                                                                                                      umbrella.89person.98                            umbrella1.00                                                                                                                                                                                          surfboard1.00 surfboard.98 surfboard1.00          person.74
                                                                                                                           person1.00
                                                                                                                           person.89person1.00person1.00                      person1.00umbrella.98
                                                                                                                                                           person1.00 handbag.97                                                                                                                                                            surfboard1.00
                                                                                                                                          person.95
                                                                                                                                           person.80                                     person1.00                                                                  person1.00
                                                                                                                                                                                           backpack.98                                                                                     person1.00                                                                                                                                                                              horse1.00                                                                                person.99
                                                                                                                         backpack.95      backpack.96                                                                                              person1.00                                                                                                                                                                                               horse1.00 horse1.00
                                                                               handbag.81                                                                                                                                                                                                                       baseball bat.99
                                                                                                                                                        handbag.85


                                                                 skateboard.83


                                                                                                                                                                                                                                                                                                        baseball bat.85                                                                                                                                                                                                                            skateboard.82
                                                                                                                                                                                           bicycle.93
                                                                                                                                                                                                                                                      baseball bat.98dog1.00




                                                                                                         person.99
                                                                                                                                                                                                                                                                                                                                                                    kite.72
                                                                                                                                                                                                                                                                                                                                                                    kite.89   kite.81
                                                                                                                                                                                                                                                                                                                                                                                           kite1.00               kite.99
                                                                                                                                                                                                                                                                                                                                                                kite.98
                                                                                                                                         person.99                              person1.00                                                                                                                                                                                                                             kite.89                person1.00
                                                                                                                                                                                                                                                                                                                                                                                                        kite.73                                                                                              car.87
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      car.93
                                                                                                                                                                                                                                                                                                                                                                                                                    kite.88
                                                                                                                                                                                                                                                                                                                                                  kite.82                                                                        kite.98
zebra1.00
zebra.90                 zebra.99                     zebra.96
                                                                                                                                                                                                                                                                                                                                           kite.97                            kite.84                   kite.86 kite.88
                                                                                                                                                                                                                                                                                                                                                                                                                    kite.95
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           car.95
                        zebra.74                                                                                                                                               person.82        person1.00                                                                                                                                                                                                                                                                                                             car.95
            zebra.99                                           zebra.96                zebra.76                                                                                                                                                                                                                                                               kite.99
                                          zebra.99                                                                                                                                                                                                                                                                                                                                                                                                                                                           car.97
                                                           zebra.88                   zebra1.00                                                                                                                                                                                                         person.87                                                                                                     kite.95
                                                                                                                                                                                                                                                                                                                                                                                                                      kite.84                                                                                     car.99
                                                                 zebra1.00                                                                                                                                          person.95person.72
                                                                                                                                                                                                                                            person.99  person.92 person.94
                                                                                                                                                                                                                                                           person.95            person.88    person.97 person.99
                                                                                                         frisbee1.00                                                                                                  person.97 person.77person.97 person.98    person.82
                                                                                                                                                                                                                                                                        person.89       person.97
                                                                                                                                                                                                                                                                                    person.83                                                                                                            kite.93
                                                                                                                                                                                                                          person.99 person.86    person.81
                                                                                                                                                                                                                                                                                                    person.77                                                                                                                                                                                                          car.78
                                                                                                                                                                                                                                                            person.88
                                                                                                                                                                                                                                                      person.98                                                                                                                                                                                                                                                                 traffic light.73
                                                                                                                                                                                                                                                                      person.94 person.88
                                                                                                                                                                                                                                                                                    person.96
                                                                                                                                                                                                                                                                                      person.96person.99person.86
                                                                                                                                                                                                                                                                                                    person.99                                                                                                     person.80                 skateboard.99                                                          car.98 truck.88     car.93
                                                                                                                                                                                                                                                                                                                                                                                  person.87person.71
                                                                                                                                                                                                                                                                                                                                                                              person.78                            person.98                                                                                                                        bus1.00
                                                                                                                                                                                                                                                                                                                                                                        person.77
                                                                                                                                                                                                                                                                                                                                                                    person.98
                                                                                                                                                                                                                                                                                                                                                                      person.98                                  person.89
                                                                                                                                                                                                                                                                                                                                                                                                                person.99
                                                                                                      person.80                                                                                                                                                                         person.91                      chair.96                              person.94    person.72            person1.00
                                                                                                                                                                                                                                                                                                                                                                                                    person.99 person.81
                                                                                                                                                                                                                                                                                                                                                                                                             person.84
                                                                                                                                                                                                                                        chair.98                   chair.78
                                                                                                                                                                                                                                                               dining         cup.93
                                                                                                                                                                                                                                                                         cup.79
                                                                                                                                                                                                                                                                      table.81                                                                 person.95                                                  person.82
                                                                                                                                                                                                                                                                                                                                                                                                          person.72
                                                                                                                                                                                                                                                                                                                                                                                                                  person1.00
                                                                                                                                                                                                                                                                     chair.85       dining table.96              dining table.75            person.94                                       person.99           person1.00                                                                                             car1.00
                                                                                                                                                                                                                                                                       chair.89                                                                       person.99
                                                                                                                                                                                                                                                                                                                                                            person.96         person1.00 person.98 motorcycle.72
                                                                                                                                                                                                                                                                                                                                                                                                           person1.00                                         person.91
                                                                                                                                                                                                                                                                                             cup.75     cup.71                                                                    person1.00
                                                                                                                                                                                                                                                                                                                                                                         person1.00                                                              person1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                       person.99
                                                                                                                                                                                                                                                                   chair.99         chair.99      chair.98    chair.95                                person1.00                                                                                    person.99              person1.00person1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           person.99
                                                                                                                                                                                                                     dining table.78                                                                                                                                                                                                                               person.98               person.80
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          person1.00
                                                                                                                                                                                                                                wine glass.80                           chair.92                                                                                                                                                                                                                                               car.95truck.86      car.98
                                                                                                                                                                                                                    chair.95           cup.83                 wine glass.80
                                                                                                                                                                                                                            cup.71
                                                                                                                                                                                                                        cup.98                                                 chair.85                                                                                                                               handbag.80                                                                                                   bus1.00               car.93
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   skateboard.98                                          car.97
                                                                                                                                                                                                                                                                      diningchair.83
                                                                                                                                                                                                                                                                             table.91
                                                                                                                                                                                                                    chair.87 chair.97                   chair.94
                                                                                                                                                                                                                                                                                        wine glass.91                 wine
                                                                                                                                                                                                                                                                               cup.96         wine glass.93            wineglass.94
                                                                                                                                                                                                                                                                                                                            glass.94                                                                                                                                                                                  car.99                    car.82
                                                                                                                                                                                                                                                                                                                    wine glass.83                                                                                                                                                               person1.00       car.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             car.99
                                                                                                                                                                                                                                                                                        cup.91                                                                                                                                                                                    couch.82
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       person.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 person.90                        person.99                              car.98

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      car.96
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           car.91car.94
                                                                                                                                                                                                                                                                                                                                       potted plant.92                                      backpack.88
                                                                                                                                                                                                                                                                                                                                                                               person.86person1.00  handbag.91
                                                                                                                                                                                                                                                                                                                                                                                                   person.76
                                                                                                                                                                                                                                                                                                                                                                                                    person1.00        person.78
                                                                                                                                                                                                                                                                                                                                                                                                              person.98
                                                                                                                                                                                                                                                                                                                                                                                                           person.78
                                                                                                                                                                                                                                                                                                  person.88                                                             person1.00                                                                                                                                       car.98
                                                                                                                                                                                                                                                   person1.00
                                                                                                                                                                                                                                                                                                                                                                                                           person.98                                                                                                           car.78
                                                                                                                                                                                    traffic light.87                                                                              tv.98      tv.84

                                                                                                                                                                                                                                                                                         person1.00
                                                                                                                                             traffic light.99                                                       person1.00                                    bottle.97
elephant1.00                                                                                                                                                                                                                                                                                                                               bird.93
                          elephant1.00            elephant1.00                      elephant.97                                                                  traffic light.71
                                                                                                                                                                                                                                                                                                                                       bench.97                                                       handbag.73                                                                                               stop sign.88
                                                                                                                                person.99                                                                                                                                         wine glass.99
                                                                                                                person1.00                                                                       person.98
                                                                                                                                                                                                 person.97
                                                                                                                                                                                              person.95                                                                                                                                                                                                                                        person.77
                                                                                                      person.92 person.74               person.99
                                                                                                                                        person.73       person1.00
                                                                                                                                                         person.95       person1.00
                                                                                                                                                                     person.98
                                                                                                                                                                 person1.00         person.99
                                                                                                                                                                                   person.95
                                                                                                                                                                               person.99 person.95                                       dining table.95
                                                                                                                                                  person1.00
                                                                                                                                              person.80                                                                                                              wine glass1.00
                                                                                                             person.87
                                                                                                               person.98
                                                                                                      person1.00                                      person.95
                                                                                                                                                person1.00                     person.99                                                                                                                                                                                                                                                                                              chair.93
elephant.99                                                                                                                                                                                                                                                                                                                                                                                                                                                     person.87                chair.81
                                                                                                                       tie.85                                                                                                                                                                                                                                                                                                                                                        chair.97
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 chair.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                             chair.99
                                                                                                                                                                                                       handbag.88                            wine glass1.00                                                                                                                                                                                                    person.97                 chair.81                                                   suitcase1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                              chair.93
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                chair.94
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    chair.92
                                                                                                                                                        handbag.88                                                                                                                                                                                                                       cell clock.73
                                                                                                                                                                                                                                                                                                                                                                                              phone.77                                                  person.81
                                                                                                                                                                                                                                                                                                                                                                                                                                                 person.90                                                                          suitcase.98
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            chair.81
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       chair.98
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   chair.83  chair.91
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                chair.80
                                                                                                                                                                        handbag.99                                                                                                                                                                                                                                                         person.96               person.71
                                                                                                                                                                                                                                                                                                                                                                                                                                                               person.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      person.94             chair.71
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       person.98




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      chair.73                    suitcase.93                        suitcase.96
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             suitcase.72




                                                                                                                                                                                                                                                                                                                                                                                                                                                       person1.00

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  suitcase1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             suitcase.88
                                                                                                                                                                                                                                                                                                                                                                                                          person.98
                                                                                                                                                                                                                                                                                                                                                                                                                        person1.00                                                                                                                                suitcase.99
                                                                                                                                                     donut.86                                                                                                                                                                                                                         horse.97 person.96
                                                                                                                              donut.95                                                                                                                                                                                                                                    person.96
                                                                                                                                                                                                                                                                                                                                                                               person.97 person.98 horse.99
                                                                                                                                     donut.89                               donut.89donut.89                                                                                                                                                                                                                                                                                 sports ball.99
                           traffic light.99                                                                            donut.90
                                                                               traffic light1.00                                                                                                                                                                                                                                                                                        person1.00
                                                                                                                                                    donut.93donut.99                                                                                                                                                                                                    person1.00
                                                                                                                                                                                                                                                                                                                                                         horse.77
                                                                                                                                               donut.86            donut.96
                                                                                                                                                   donut.95
                                                                                                                                                         donut.98      donut.81                                                                                                                                                                                                                                                                         tennis racket1.00
                    car.95                                                                                                                  donut.89              donut1.00
                                                                                                                   donut1.00                   donut.96donut.98
                          car.81                                                                           donut.99                                            donut.98
                           car.89                                                                                     donut.98                                                                                                                                                                                                          cow.93
                       car.98                                                                         donut.89 donut.97                                                                                                                                                                                                                                                                                person.99
               car.97         car.91
                                                                                                                             donut.95
     car.96                        car.94                                                             donut.94
     car.97                          car.94                                                                       donut.98      donut1.00                                                                                                                                         truck.92
                           person.87      car.95
                          bicycle.86     car.97                                                                                                                                                                                                                  truck.96
                                              car1.00                                                 donut.95       donut.99                                                                                                      truck.97        truck.99
     car.96                                       car.98
                                                     car.97
                                                       car.99                                                                                                                                                                                                                                        truck.99
                                                                                                                                                                                                                    truck.93
                                                                                                                                                                                                                    bus.90
person.99
   car.95                                                   car.97                                                                          donut.90                                                                      bus.99
                                                                     car1.00       parking meter.98     donut.96
                 car.99
                                                                                                                                                                                    donut1.00
car.97                                                                                                                                               donut.88
car.86




Figure 4. More results of Mask R-CNN on COCO test images, using ResNet-101-FPN and running at 5 fps, with 35.7 mask AP (Table 1).

                                                                                                                                                                                                 backbone                                                                                                AP                            AP50                          AP75                                  APS                             APM                          APL
                                                                                               MNC [10]                                                                                          ResNet-101-C4                                                                                           24.6                          44.3                          24.8                                  4.7                             25.9                         43.6
                                                                                               FCIS [26] +OHEM                                                                                   ResNet-101-C5-dilated                                                                                   29.2                          49.5                           -                                    7.1                             31.3                         50.0
                                                                                               FCIS+++ [26] +OHEM                                                                                ResNet-101-C5-dilated                                                                                   33.6                          54.5                           -                                     -                               -                            -
                                                                                               Mask R-CNN                                                                                        ResNet-101-C4                                                                                           33.1                          54.9                          34.8                                  12.1                            35.6                         51.1
                                                                                               Mask R-CNN                                                                                        ResNet-101-FPN                                                                                          35.7                          58.0                          37.8                                  15.5                            38.1                         52.4
                                                                                               Mask R-CNN                                                                                        ResNeXt-101-FPN                                                                                         37.1                          60.0                          39.4                                  16.9                            39.9                         53.5

Table 1. Instance segmentation mask AP on COCO test-dev. MNC [10] and FCIS [26] are the winners of the COCO 2015 and 2016
segmentation challenges, respectively. Without bells and whistles, Mask R-CNN outperforms the more complex FCIS+++, which includes
multi-scale train/test, horizontal flip test, and OHEM [35]. All entries are single-model results.


is evaluating using mask IoU. As in previous work [5, 27],                                                                                                                                                                                                                                                         expect many such improvements to be applicable to ours.
we train using the union of 80k train images and a 35k sub-                                                                                                                                                                                                                                                            Mask R-CNN outputs are visualized in Figures 2 and 4.
set of val images (trainval35k), and report ablations on                                                                                                                                                                                                                                                           Mask R-CNN achieves good results even under challeng-
the remaining 5k subset of val images (minival). We also                                                                                                                                                                                                                                                           ing conditions. In Figure 5 we compare our Mask R-CNN
report results on test-dev [28], which has no disclosed                                                                                                                                                                                                                                                            baseline and FCIS+++ [26]. FCIS+++ exhibits systematic
labels. Upon publication, we will upload our full results on                                                                                                                                                                                                                                                       artifacts on overlapping instances, suggesting that it is chal-
test-std to the public leaderboard, as recommended.                                                                                                                                                                                                                                                                lenged by the fundamental difficulty of instance segmenta-
                                                                                                                                                                                                                                                                                                                   tion. Mask R-CNN shows no such artifacts.
4.1. Main Results
                                                                                                                                                                                                                                                                                                                   4.2. Ablation Experiments
    We compare Mask R-CNN to the state-of-the-art meth-
ods in instance segmentation in Table 1. All instantia-                                                                                                                                                                                                                                                              We run a number of ablations to analyze Mask R-CNN.
tions of our model outperform baseline variants of pre-                                                                                                                                                                                                                                                            Results are shown in Table 2 and discussed in detail next.
vious state-of-the-art models. This includes MNC [10]
                                                                                                                                                                                                                                                                                                                   Architecture: Table 2a shows Mask R-CNN with various
and FCIS [26], the winners of the COCO 2015 and 2016
                                                                                                                                                                                                                                                                                                                   backbones. It benefits from deeper networks (50 vs. 101)
segmentation challenges, respectively. Without bells and
                                                                                                                                                                                                                                                                                                                   and advanced designs including FPN and ResNeXt3 . We
whistles, Mask R-CNN with ResNet-101-FPN backbone
                                                                                                                                                                                                                                                                                                                   note that not all frameworks automatically benefit from
outperforms FCIS+++ [26], which includes multi-scale
                                                                                                                                                                                                                                                                                                                   deeper or advanced networks (see benchmarking in [21]).
train/test, horizontal flip test, and online hard example min-
ing (OHEM) [35]. While outside the scope of this work, we                                                                                                                                                                                                                                                                         3 We use the 64×4d variant of ResNeXt [40].




                                                                                                                                                                                                                                                                                                 5
FCIS


        umbrella.99                  umbrella1.00                                                                                                     person1.00 person1.00
         person1.00
                                                  person1.00                                                                             person1.00                                                                                               person1.00
                                                                                                                                                                           person1.00




Mask R-CNN
                                                                                                                                                                                                                                                                                     person1.00
                                   person1.00
                                                                                                                                                                                                                                                                        person1.00                                         train1.00
                      person1.00                                                  person1.00          person1.00                                                                                                                                                                                               train.99
                                                                     person1.00
                                                                                              person1.00
                                                                                         person.99             person1.00                                                                                       giraffe1.00   giraffe1.00
                                                                 person.99                                                  person1.00
                                                car.99 car.93
                                                                                                                                                                                                                                                                                                            train.80

                                                                                                                                         person.95


                                                                                                   handbag.93                                                                                                                                                  tie.95
                                                                                           skateboard.98
                                                                                                                                                                                                                                                                                                  tie1.00
                                                                                                                                                                  sports ball.98

                                                                                                                                                                                        sports ball1.00

                                                                                                     skateboard.99




  Figure 5. FCIS+++ [26] (top) vs. Mask R-CNN (bottom, ResNet-101-FPN). FCIS exhibits systematic artifacts on overlapping objects.

       net-depth-features                       AP              AP50                    AP75                                                           AP                           AP50                      AP75                                                         align? bilinear? agg.                          AP             AP50    AP75
        ResNet-50-C4                            30.3            51.2                    31.5                                   softmax                 24.8                         44.1                      25.1                          RoIPool [12]                                                    max           26.9           48.8    26.4
        ResNet-101-C4                           32.7            54.2                    34.3                                   sigmoid                 30.3                         51.2                      31.5                                                                                X         max           27.2           49.2    27.1
                                                                                                                                                                                                                                            RoIWarp [10]
       ResNet-50-FPN                            33.6            55.2                    35.3                                                           +5.5                         +7.1                      +6.4                                                                                X         ave           27.1           48.9    27.1
       ResNet-101-FPN                           35.4            57.3                    37.5                                                                                                                                                                                    X                 X         max           30.2           51.0    31.8
                                                                                                                                                                                                                                              RoIAlign
      ResNeXt-101-FPN                           36.7            59.5                    38.9                                                                                                                                                                                    X                 X         ave           30.3           51.2    31.5
  (a) Backbone Architecture: Better back-                                                                             (b) Multinomial vs. Independent Masks                                                                          (c) RoIAlign (ResNet-50-C4): Mask results with various RoI
  bones bring expected gains: deeper networks                                                                         (ResNet-50-C4): Decoupling via per-                                                                            layers. Our RoIAlign layer improves AP by ∼3 points and
  do better, FPN outperforms C4 features, and                                                                         class binary masks (sigmoid) gives large                                                                       AP75 by ∼5 points. Using proper alignment is the only fac-
  ResNeXt improves on ResNet.                                                                                         gains over multinomial masks (softmax).                                                                        tor that contributes to the large gap between RoI layers.

                               AP                     AP50              AP75                           APbb                          APbb
                                                                                                                                       50             APbb
                                                                                                                                                        75                                                                    mask branch                                                                        AP                    AP50     AP75
       RoIPool                 23.6                   46.5              21.6                           28.2                          52.7             26.9                                           MLP               fc: 1024→1024→80·282                                                                      31.5                  53.7     32.8
       RoIAlign                30.9                   51.8              32.1                           34.0                          55.3             36.4                                           MLP           fc: 1024→1024→1024→80·282                                                                     31.5                  54.0     32.6
                               +7.3                   + 5.3             +10.5                          +5.8                          +2.6             +9.5                                           FCN        conv: 256→256→256→256→256→80                                                                     33.6                  55.2     35.3
  (d) RoIAlign (ResNet-50-C5, stride 32): Mask-level and box-level                                                                                                                            (e) Mask Branch (ResNet-50-FPN): Fully convolutional networks (FCN) vs.
  AP using large-stride features. Misalignments are more severe than                                                                                                                          multi-layer perceptrons (MLP, fully-connected) for mask prediction. FCNs im-
  with stride-16 features (Table 2c), resulting in massive accuracy gaps.                                                                                                                     prove results as they take advantage of explicitly encoding spatial layout.

             Table 2. Ablations for Mask R-CNN. We train on trainval35k, test on minival, and report mask AP unless otherwise noted.

  Multinomial vs. Independent Masks: Mask R-CNN de-                                                                                                                                                           AP by about 3 points over RoIPool, with much of the gain
  couples mask and class prediction: as the existing box                                                                                                                                                      coming at high IoU (AP75 ). RoIAlign is insensitive to
  branch predicts the class label, we generate a mask for each                                                                                                                                                max/average pool; we use average in the rest of the paper.
  class without competition among classes (by a per-pixel sig-                                                                                                                                                    Additionally, we compare with RoIWarp proposed in
  moid and a binary loss). In Table 2b, we compare this to                                                                                                                                                    MNC [10] that also adopt bilinear sampling. As discussed
  using a per-pixel softmax and a multinomial loss (as com-                                                                                                                                                   in §3, RoIWarp still quantizes the RoI, losing alignment
  monly used in FCN [29]). This alternative couples the tasks                                                                                                                                                 with the input. As can be seen in Table 2c, RoIWarp per-
  of mask and class prediction, and results in a severe loss                                                                                                                                                  forms on par with RoIPool and much worse than RoIAlign.
  in mask AP (5.5 points). This suggests that once the in-                                                                                                                                                    This highlights that proper alignment is key.
  stance has been classified as a whole (by the box branch),
                                                                                                                                                                                                                  We also evaluate RoIAlign with a ResNet-50-C5 back-
  it is sufficient to predict a binary mask without concern for
                                                                                                                                                                                                              bone, which has an even larger stride of 32 pixels. We use
  the categories, which makes the model easier to train.
                                                                                                                                                                                                              the same head as in Figure 3 (right), as the res5 head is not
  Class-Specific vs. Class-Agnostic Masks: Our default in-                                                                                                                                                    applicable. Table 2d shows that RoIAlign improves mask
  stantiation predicts class-specific masks, i.e., one m×m                                                                                                                                                    AP by a massive 7.3 points, and mask AP75 by 10.5 points
  mask per class. Interestingly, Mask R-CNN with class-                                                                                                                                                       (50% relative improvement). Moreover, we note that with
  agnostic masks (i.e., predicting a single m×m output re-                                                                                                                                                    RoIAlign, using stride-32 C5 features (30.9 AP) is more ac-
  gardless of class) is nearly as effective: it has 29.7 mask AP                                                                                                                                              curate than using stride-16 C4 features (30.3 AP, Table 2c).
  vs. 30.3 for the class-specific counterpart on ResNet-50-C4.                                                                                                                                                RoIAlign largely resolves the long-standing challenge of
  This further highlights the division of labor in our approach                                                                                                                                               using large-stride features for detection and segmentation.
  which largely decouples classification and segmentation.                                                                                                                                                        Finally, RoIAlign shows a gain of 1.5 mask AP and 0.5
  RoIAlign: An evaluation of our proposed RoIAlign layer is                                                                                                                                                   box AP when used with FPN, which has finer multi-level
  shown in Table 2c. For this experiment we use the ResNet-                                                                                                                                                   strides. For keypoint detection that requires finer alignment,
  50-C4 backbone, which has stride 16. RoIAlign improves                                                                                                                                                      RoIAlign shows large gains even with FPN (Table 6).


                                                                                                                                                                                                          6
                                              backbone                    APbb    APbb
                                                                                    50   APbb
                                                                                           75    APbb
                                                                                                   S     APbb
                                                                                                           M     APbb
                                                                                                                   L
                 Faster R-CNN+++ [19]         ResNet-101-C4                34.9   55.7    37.4    15.6    38.7    50.9
                 Faster R-CNN w FPN [27]      ResNet-101-FPN               36.2   59.1    39.0    18.2    39.0    48.2
                 Faster R-CNN by G-RMI [21]   Inception-ResNet-v2 [37]     34.7   55.5    36.7    13.5    38.1    52.0
                 Faster R-CNN w TDM [36]      Inception-ResNet-v2-TDM      36.8   57.7    39.2    16.2    39.8    52.1
                 Faster R-CNN, RoIAlign       ResNet-101-FPN               37.3   59.6    40.3    19.8    40.2    48.8
                 Mask R-CNN                   ResNet-101-FPN               38.2   60.3    41.7    20.1    41.1    50.2
                 Mask R-CNN                   ResNeXt-101-FPN              39.8   62.3    43.4    22.1    43.2    51.2
Table 3. Object detection single-model results (bounding box AP), vs. state-of-the-art on test-dev. Mask R-CNN using ResNet-101-
FPN outperforms the base variants of all previous state-of-the-art models (the mask output is ignored in these experiments). The gains of
Mask R-CNN over [27] come from using RoIAlign (+1.1 APbb ), multitask training (+0.9 APbb ), and ResNeXt-101 (+1.6 APbb ).

Mask Branch: Segmentation is a pixel-to-pixel task and                   ant takes ∼400ms as it has a heavier box head (Figure 3), so
we exploit the spatial layout of masks by using an FCN.                  we do not recommend using the C4 variant in practice.
In Table 2e, we compare multi-layer perceptrons (MLP)                        Although Mask R-CNN is fast, we note that our design
and FCNs, using a ResNet-50-FPN backbone. Using FCNs                     is not optimized for speed, and better speed/accuracy trade-
gives a 2.1 mask AP gain over MLPs. We note that we                      offs could be achieved [21], e.g., by varying image sizes and
choose this backbone so that the conv layers of the FCN                  proposal numbers, which is beyond the scope of this paper.
head are not pre-trained, for a fair comparison with MLP.
                                                                         Training: Mask R-CNN is also fast to train. Training with
4.3. Bounding Box Detection Results                                      ResNet-50-FPN on COCO trainval35k takes 32 hours
                                                                         in our synchronized 8-GPU implementation (0.72s per 16-
   We compare Mask R-CNN to the state-of-the-art COCO                    image mini-batch), and 44 hours with ResNet-101-FPN. In
bounding-box object detection in Table 3. For this result,               fact, fast prototyping can be completed in less than one day
even though the full Mask R-CNN model is trained, only                   when training on the train set. We hope such rapid train-
the classification and box outputs are used at inference (the            ing will remove a major hurdle in this area and encourage
mask output is ignored). Mask R-CNN using ResNet-101-                    more people to perform research on this challenging topic.
FPN outperforms the base variants of all previous state-of-
the-art models, including the single-model variant of G-                 5. Mask R-CNN for Human Pose Estimation
RMI [21], the winner of the COCO 2016 Detection Chal-
lenge. Using ResNeXt-101-FPN, Mask R-CNN further im-                         Our framework can easily be extended to human pose
proves results, with a margin of 3.0 points box AP over                  estimation. We model a keypoint’s location as a one-hot
the best previous single model entry from [36] (which used               mask, and adopt Mask R-CNN to predict K masks, one for
Inception-ResNet-v2-TDM).                                                each of K keypoint types (e.g., left shoulder, right elbow).
   As a further comparison, we trained a version of Mask                 This task helps demonstrate the flexibility of Mask R-CNN.
R-CNN but without the mask branch, denoted by “Faster                        We note that minimal domain knowledge for human pose
R-CNN, RoIAlign” in Table 3. This model performs better                  is exploited by our system, as the experiments are mainly to
than the model presented in [27] due to RoIAlign. On the                 demonstrate the generality of the Mask R-CNN framework.
other hand, it is 0.9 points box AP lower than Mask R-CNN.               We expect that domain knowledge (e.g., modeling struc-
This gap of Mask R-CNN on box detection is therefore due                 tures [6]) will be complementary to our simple approach,
solely to the benefits of multi-task training.                           but it is beyond the scope of this paper.
   Lastly, we note that Mask R-CNN attains a small gap                   Implementation Details: We make minor modifications to
between its mask and box AP: e.g., 2.7 points between 37.1               the segmentation system when adapting it for keypoints.
(mask, Table 1) and 39.8 (box, Table 3). This indicates that             For each of the K keypoints of an instance, the training
our approach largely closes the gap between object detec-                target is a one-hot m × m binary mask where only a single
tion and the more challenging instance segmentation task.                pixel is labeled as foreground. During training, for each vis-
                                                                         ible ground-truth keypoint, we minimize the cross-entropy
4.4. Timing
                                                                         loss over an m2 -way softmax output (which encourages a
Inference: We train a ResNet-101-FPN model that shares                   single point to be detected). We note that as in instance seg-
features between the RPN and Mask R-CNN stages, follow-                  mentation, the K keypoints are still treated independently.
ing the 4-step training of Faster R-CNN [34]. This model                    We adopt the ResNet-FPN variant, and the keypoint head
runs at 195ms per image on an Nvidia Tesla M40 GPU (plus                 architecture is similar to that in Figure 3 (right). The key-
15ms CPU time resizing the outputs to the original resolu-               point head consists of a stack of eight 3×3 512-d conv lay-
tion), and achieves statistically the same mask AP as the                ers, followed by a deconv layer and 2× bilinear upscaling,
unshared one. We also report that the ResNet-101-C4 vari-                producing an output resolution of 56×56. We found that


                                                                   7
Figure 6. Keypoint detection results on COCO test using Mask R-CNN (ResNet-50-FPN), with person segmentation masks predicted
from the same model. This model has a keypoint AP of 63.1 and runs at 5 fps.

                                          kp     kp     kp      kp
                                APkp   AP50    AP75   APM    APL                                                 APbb
                                                                                                                   person     APmask
                                                                                                                                person        APkp
CMU-Pose+++ [6]                 61.8   84.9    67.5   57.1   68.2            Faster R-CNN                         52.5          -              -
G-RMI [31]†                     62.4   84.0    68.5   59.1   68.1            Mask R-CNN, mask-only                53.6         45.8            -
                                                                             Mask R-CNN, keypoint-only            50.7          -             64.2
Mask R-CNN, keypoint-only       62.7   87.0    68.4   57.4   71.1
                                                                             Mask R-CNN, keypoint & mask          52.0         45.1           64.7
Mask R-CNN, keypoint & mask     63.1   87.3    68.7   57.8   71.4
                                                                          Table 5. Multi-task learning of box, mask, and keypoint about the person
Table 4. Keypoint detection AP on COCO test-dev. Ours                     category, evaluated on minival. All entries are trained on the same data
(ResNet-50-FPN) is a single model that runs at 5 fps. CMU-                for fair comparisons. The backbone is ResNet-50-FPN. The entry with
Pose+++ [6] is the 2016 competition winner that uses multi-scale          64.2 AP on minival has 62.7 AP on test-dev. The entry with 64.7
testing, post-processing with CPM [39], and filtering with an ob-         AP on minival has 63.1 AP on test-dev (see Table 4).
ject detector, adding a cumulative ∼5 points (clarified in personal
communication). † : G-RMI was trained on COCO plus MPII [1]                                                 kp      kp        kp         kp
(25k images), using two models (Inception-ResNet-v2 + ResNet-                                    APkp    AP50    AP75    APM       APL
101). As they use more data, this is not a direct comparison with                   RoIPool       59.8    86.2    66.7      55.1   67.4
Mask R-CNN.                                                                         RoIAlign      64.2    86.6    69.7      58.7   73.0
                                                                          Table 6. RoIAlign vs. RoIPool for keypoint detection on minival.
a relatively high resolution output (compared to masks) is
required for keypoint-level localization accuracy.
    Models are trained on all COCO trainval35k im-                        son category) improves the APkp to 63.1 (Table 4) on
ages that contain annotated keypoints. To reduce overfit-                 test-dev. More ablations of multi-task learning on
ting, as this training set is smaller, we train the models us-            minival are in Table 5. Adding the mask branch to the
ing image scales randomly sampled from [640, 800] pixels;                 box-only (i.e., Faster R-CNN) or keypoint-only versions
inference is on a single scale of 800 pixels. We train for 90k            consistently improves these tasks. However, adding the
iterations, starting from a learning rate of 0.02 and reducing            keypoint branch reduces the box/mask AP slightly, suggest-
it by 10 at 60k and 80k iterations. We use bounding-box                   ing that while keypoint detection benefits from multitask
non-maximum suppression with a threshold of 0.5. Other                    training, it does not in turn help the other tasks. Neverthe-
implementations are identical as in §3.1.                                 less, learning all three tasks jointly enables a unified system
                                                                          to efficiently predict all outputs simultaneously (Figure 6).
Experiments on Human Pose Estimation: We evaluate
the person keypoint AP (APkp ) using ResNet-50-FPN. We                        We also investigate the effect of RoIAlign on keypoint
have experimented with ResNet-101 and found it achieves                   detection (Table 6). Though this ResNet-50-FPN backbone
similar results, possibly because deeper models benefit from              has finer strides (e.g., 4 pixels on the finest level), RoIAlign
more training data, but this dataset is relatively small.                 still shows significant improvement over RoIPool and in-
   Table 4 shows that our result (62.7 APkp ) is 0.9 points               creases APkp by 4.4 points. This is because keypoint detec-
higher than the COCO 2016 keypoint detection winner [6]                   tions are more sensitive to localization accuracy. This again
that uses a multi-stage processing pipeline (see caption of               indicates that alignment is essential for pixel-level localiza-
Table 4). Our method is considerably simpler and faster.                  tion, including masks and keypoints.
   More importantly, we have a unified model that can si-                    Given the effectiveness of Mask R-CNN for extracting
multaneously predict boxes, segments, and keypoints while                 object bounding boxes, masks, and keypoints, we expect it
running at 5 fps. Adding a segment branch (for the per-                   be an effective framework for other instance-level tasks.


                                                                      8
                               training data AP [val]   AP     AP50   person                                        rider                                                         car                                                        truck                                                             bus                                                      train                                       mcycle bicycle
          InstanceCut [23]   fine + coarse     15.8     13.0   27.9    10.0                                          8.0                                                         23.7                                                         14.0                                                             19.5                                                     15.2                                          9.3    4.7
          DWT [4]            fine              19.8     15.6   30.0    15.1                                         11.7                                                         32.9                                                         17.1                                                             20.4                                                     15.0                                          7.9    4.9
          SAIS [17]          fine                -      17.4   36.7    14.6                                         12.9                                                         35.7                                                         16.0                                                             23.2                                                     19.0                                         10.3    7.8
          DIN [3]            fine + coarse       -      20.0   38.8    16.5                                         16.7                                                         25.7                                                         20.6                                                             30.0                                                     23.4                                         17.1   10.1
          Mask R-CNN         fine              31.5     26.2   49.9    30.5                                         23.7                                                         46.9                                                         22.8                                                             32.2                                                     18.6                                         19.1   16.0
          Mask R-CNN         fine + COCO       36.4     32.0   58.1    34.8                                         27.0                                                         49.1                                                         30.1                                                             40.9                                                     30.9                                         24.1   18.7
   Table 7. Results on Cityscapes val (‘AP [val]’ column) and test (remaining columns) sets. Our method uses ResNet-50-FPN.

A. Experiments on Cityscapes
                                                                                                                                                                                                                                                                                         person:0.99        person:1.00
                                                                                             person:1.00
                                                                                       rider:0.59
                                                                                       person:0.79                                                                                                                                                                                                                             person:1.00
                                                                           person:1.00                  person:1.00                                                                       person:1.00                                                                    person:1.00
                                                                                 person:0.66
                                                                                 person:0.59                           person:1.00       bus:1.00                                                               person:1.00
                                                                                                                                                                  bus:0.95                                                        person:1.00         person:1.00
                                                                                                                                                                  truck:0.66                                                  person:1.00                  person:1.00
                                                                                                                                                                                                                                                                  person:0.98                                                                                                                                                                        car:1.00
                                                                                                                                              person:0.99
                                                                                                                                                       car:0.98                                                                                                      person:0.82                                                                                          person:0.92
                                                                          person:0.99person:0.67                                         person:1.00                                                    person:0.99
                                                                                                                                                                                                          person:0.98
                                                                                                                                                                                                            person:0.73                                                 car:0.52                                                                                                   car:1.00            car:0.64




    We further report instance segmentation results on the
                                                                                       person:0.82                                                                        person:0.98
                                                                                                                                                                         person:0.94
                                                                                                                                                                       person:0.94                                                                                                                                                                                           car:0.68       car:1.00                                                                                                                                                                            person:1.00
                                                                                                              car:0.81                                 car:0.98      person:0.98
                                                                                                                                                                             car:0.95
                                                                                                                                                                               car:1.00                                                                                                                                                                      car:0.95
                                                                                                                                                                                                                                                                                                                                                          car:0.57                     car:0.68
                                                                                                                                                                                                                                                                                                                                                                                  car:0.52                                      person:0.82
                                                                                                                                                                                                                                                                                                                                                                                                                                   person:0.63car:1.00                                                    rider:0.68                                      person:0.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       person:0.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               person:0.93
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           person:0.97                          person:0.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               person:0.98
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              person:0.73
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           person:0.98
                                                                                                                                                                                                                                                                                                                                                                         bicycle:0.83                                   car:0.99car:1.00                                                                 person:0.72
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              car:1.00    car:1.00                 person:0.84
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    person:0.98     person:0.86
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       person:0.99
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                person:0.72
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 person:0.72
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      car:0.69
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                car:1.00      car:1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        car:0.95
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            car:0.95                                                        person:0.91
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           bicycle:0.56




Cityscapes [7] dataset. This dataset has fine annotations
for 2975 training images, 500 validation images, and 1525
test images. It has 20k coarse training images without in-
stance annotations, which we do not use. All images are at                                                                                          person:1.00



                                                                                                                                                                                                                              car:1.00                                                        car:1.00                                         car:1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                         car:1.00                                                                                                         person:1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                person:1.00
                                                                                                                                                                                                                  car:1.00                                         car:1.00
                                                                                                                                                                                                                                                                                                                 car:1.00                                                                                                                           person:0.82                                              person:1.00 person:1.00




a fixed resolution of 2048×1024 pixels. The instance seg-
                                                                                                                                                                                                                                           car:1.00                           car:1.00
                                                                                                                                                                      car:0.97            person:0.78                                     car:0.50      car:1.00                                                                                                                                                                      car:1.00
                                                                                person:0.73                                                                      car:0.72
                                                                                                                                                               person:0.98                     person:1.00
                                                                                                                                                                                           person:0.58
                                                                                                                                                                                                   car:1.00                  car:0.65                                                                                                                                                                                     bus:0.75
                                                                                              person:0.85car:1.00                         car:1.00
                                                                                                                          car:1.00 car:1.00                 car:0.72
                                                                                                                                                                 car:0.76                                                                                                                                                                                                                       car:1.00 car:1.00
                                                                          car:1.00    person:0.93
                                                                                                car:1.00              car:1.00         car:0.98 car:0.88                                                                                                                                                                                                                     car:1.00                                       car:1.00 car:0.99
                                                                                   car:1.00                                                                                                                                                                                                                                                                                                                         car:0.89        car:0.67




mentation task involves 8 object categories, whose numbers
of instances on the fine training set are:
 person     rider    car      truck   bus    train   mcycle bicycle
 17.9k      1.8k    26.9k      0.5k   0.4k   0.2k     0.7k   3.7k                  person:1.00
                                                                                                                person:1.00

                                                                                                                                                                                                                                                                     person:1.00
                                                                                                                                                                                                                                                            person:1.00
                                                                                                                                                                                                                                                                            person:1.00
                                                                                                                                                                                                                                                                                                   person:1.00
                                                                                                                                                                                                                                                                                                                            person:1.00
                                                                                                                                                                                                                                                                                                                                 person:1.00
                                                                                                                                                                                                                                                                                                                                               person:1.00
                                                                                                                                                                                                                                                                                                                                               person:0.75
                                                                                                                                                                                                                                                                                                                                                  person:0.93                                                           person:1.00    person:1.00


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  person:1.00
                                                                                                                                                                                                                                         person:1.00person:1.00   person:0.92
                                                                                                                                                                                                       person:0.99                            person:1.00person:0.97                                                                                                                                                                                                                                                   person:1.00 person:0.96
                                                                                                                                                                    person:1.00
                                                                                                                                                              person:1.00
                                                                                                                                                    person:1.00                  person:1.00person:1.00
                                                                                                                                                                          person:1.00             person:1.00person:0.98                                                                     person:1.00                                                  car:1.00                                                                                                         rider:0.94
                                                                                                                                     person:1.00                                         person:0.99
                                                                                                                                                                               person:1.00                            person:1.00
                                                                                                                                                             person:0.70                              person:0.59                                                                                          person:0.96                                                                                  car:0.99                                                                                                                     car:1.00
                                                                                                                                                                                                                                                                                                                                                                                                                                                           person:0.88
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 person:0.89
                                                                                                                                                                                                                                                                                                                                                                                                                                                         car:0.89                                                           bicycle:0.97
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      bicycle:0.99




Instance segmentation performance on this task is measured
by the COCO-style mask AP (averaged over IoU thresh-                      person:0.91




olds); AP50 (i.e., mask AP at an IoU of 0.5) is also reported.
Implementation: We apply our Mask R-CNN models with                       Figure 7. Mask R-CNN results on Cityscapes test (32.0 AP).
the ResNet-FPN-50 backbone; we have tested the 101-layer                  The bottom-right image shows a failure prediction.
counterpart and found it performs similarly due to the small
dataset size. We train with image scale (shorter side) ran-               report a result using COCO pre-training. To do this, we ini-
domly sampled from [800, 1024], which reduces overfit-                    tialize the corresponding 7 categories in Cityscapes from a
ting; inference is on a single scale of 1024 pixels. We use a             pre-trained COCO Mask R-CNN model (rider being ran-
mini-batch size of 1 image per GPU (so effectively 8 on 8                 domly initialized). We fine-tune this model for 4k iterations
GPUs) and train the model for 24k iterations, starting from               in which the learning rate is reduced at 3k iterations, which
a learning rate of 0.01 and reducing it to 0.001 at 18k itera-            takes ∼1 hour for training given the COCO model.
tions. Other implementation details are identical as in §3.1.                 The COCO pre-trained Mask R-CNN model achieves
Results: Table 7 compares our results to the state of the                 32.0 AP on test, almost a 6 point absolute improvement
art on the val and test sets. Without using the coarse                    over the fine-only counterpart. This indicates the impor-
training set, our method achieves 26.2 AP on test, which                  tant role the amount of training data plays. It also suggests
is over 30% relative improvement over the best entry, which               that instance segmentation methods on Cityscapes might
uses both fine + coarse labels. Compared to the best                      be influenced by their low-shot learning performance. We
entry using fine labels only (17.4 AP), we achieve a ∼50%                 show that using COCO pre-training is an effective strategy
improvement. It takes ∼4 hours of training on a single 8-                 for mitigating the limited data issue involving this dataset.
GPU machine to obtain this result.                                            Finally, we observed a bias between the val and test
    For the person and car categories, the Cityscapes dataset             AP, as is also observed from the results of [23, 4]. We
exhibits a large number of within-category overlapping in-                found that this bias is mainly caused by the truck, bus,
stances (on average 6 people and 9 cars per image). We                    and train categories, with the fine-only model having
argue that within-category overlap is a core difficulty of in-            val/test AP of 28.8/22.8, 53.5/32.2, and 33.0/18.6, re-
stance segmentation. Our method shows massive improve-                    spectively. This suggests that there is a domain shift on
ment on these two categories over the best existing entries               these categories, which also have little training data. COCO
(relative ∼85% improvement on person from 16.5 to 30.5                    pre-training helps to improve results the most on these cat-
and ∼30% improvement on car from 35.7 to 46.9).                           egories; however, the domain shift persists with 38.0/30.1,
    A main challenge of the Cityscapes dataset is training                57.5/40.9, and 41.2/30.9 val/test AP, respectively. Note
models in a low-data regime, particularly for the categories              that for the person and car categories we do not see any
of truck, bus, and train, which have about 200-500 train-                 such bias (val/test AP are within ±1 point).
ing samples each. To partially remedy this issue, we further                  Example results on Cityscapes are shown in Figure 7.


                                                                      9
References                                                                 [22] M. Jaderberg, K. Simonyan, A. Zisserman, and
                                                                                K. Kavukcuoglu.         Spatial transformer networks.        In
 [1] M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. 2D                 NIPS, 2015. 3
     human pose estimation: New benchmark and state of the art
                                                                           [23] A. Kirillov, E. Levinkov, B. Andres, B. Savchynskyy, and
     analysis. In CVPR, 2014. 8
                                                                                C. Rother. Instancecut: from edges to instances with multi-
 [2] P. Arbeláez, J. Pont-Tuset, J. T. Barron, F. Marques, and                 cut. In CVPR, 2017. 9
     J. Malik. Multiscale combinatorial grouping. In CVPR,                 [24] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet clas-
     2014. 2                                                                    sification with deep convolutional neural networks. In NIPS,
 [3] A. Arnab and P. H. Torr. Pixelwise instance segmentation                   2012. 2
     with a dynamically instantiated network. In CVPR, 2017. 9             [25] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E.
 [4] M. Bai and R. Urtasun. Deep watershed transform for in-                    Howard, W. Hubbard, and L. D. Jackel. Backpropagation
     stance segmentation. In CVPR, 2017. 9                                      applied to handwritten zip code recognition. Neural compu-
 [5] S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Inside-                  tation, 1989. 2
     outside net: Detecting objects in context with skip pooling           [26] Y. Li, H. Qi, J. Dai, X. Ji, and Y. Wei. Fully convolutional
     and recurrent neural networks. In CVPR, 2016. 5                            instance-aware semantic segmentation. In CVPR, 2017. 2,
 [6] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh. Realtime multi-                3, 5, 6
     person 2d pose estimation using part affinity fields. In CVPR,        [27] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and
     2017. 7, 8                                                                 S. Belongie. Feature pyramid networks for object detection.
 [7] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler,                   In CVPR, 2017. 2, 4, 5, 7
     R. Benenson, U. Franke, S. Roth, and B. Schiele. The                  [28] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-
     Cityscapes dataset for semantic urban scene understanding.                 manan, P. Dollár, and C. L. Zitnick. Microsoft COCO: Com-
     In CVPR, 2016. 9                                                           mon objects in context. In ECCV, 2014. 2, 4, 5
 [8] J. Dai, K. He, Y. Li, S. Ren, and J. Sun. Instance-sensitive          [29] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional
     fully convolutional networks. In ECCV, 2016. 2                             networks for semantic segmentation. In CVPR, 2015. 1, 3, 6
 [9] J. Dai, K. He, and J. Sun. Convolutional feature masking for          [30] V. Nair and G. E. Hinton. Rectified linear units improve re-
     joint object and stuff segmentation. In CVPR, 2015. 2                      stricted boltzmann machines. In ICML, 2010. 4
[10] J. Dai, K. He, and J. Sun. Instance-aware semantic segmen-            [31] G. Papandreou, T. Zhu, N. Kanazawa, A. Toshev, J. Tomp-
     tation via multi-task network cascades. In CVPR, 2016. 2, 3,               son, C. Bregler, and K. Murphy. Towards accurate multi-
     4, 5, 6                                                                    person pose estimation in the wild. In CVPR, 2017. 8
[11] J. Dai, Y. Li, K. He, and J. Sun. R-FCN: Object detection via         [32] P. O. Pinheiro, R. Collobert, and P. Dollar. Learning to seg-
     region-based fully convolutional networks. In NIPS, 2016. 2                ment object candidates. In NIPS, 2015. 2, 3
[12] R. Girshick. Fast R-CNN. In ICCV, 2015. 1, 2, 3, 4, 6                 [33] P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Dollár. Learn-
[13] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich fea-               ing to refine object segments. In ECCV, 2016. 2, 3
     ture hierarchies for accurate object detection and semantic           [34] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: To-
     segmentation. In CVPR, 2014. 2, 3                                          wards real-time object detection with region proposal net-
[14] R. Girshick, F. Iandola, T. Darrell, and J. Malik. Deformable              works. In NIPS, 2015. 1, 2, 3, 4, 7
     part models are convolutional neural networks. In CVPR,               [35] A. Shrivastava, A. Gupta, and R. Girshick. Training region-
     2015. 4                                                                    based object detectors with online hard example mining. In
[15] B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Simul-              CVPR, 2016. 2, 5
     taneous detection and segmentation. In ECCV. 2014. 2                  [36] A. Shrivastava, R. Sukthankar, J. Malik, and A. Gupta. Be-
                                                                                yond skip connections: Top-down modulation for object de-
[16] B. Hariharan, P. Arbeláez, R. Girshick, and J. Malik. Hyper-
                                                                                tection. arXiv:1612.06851, 2016. 4, 7
     columns for object segmentation and fine-grained localiza-
     tion. In CVPR, 2015. 2                                                [37] C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4,
                                                                                inception-resnet and the impact of residual connections on
[17] Z. Hayder, X. He, and M. Salzmann. Shape-aware instance
                                                                                learning. In ICLR Workshop, 2016. 7
     segmentation. In CVPR, 2017. 9
                                                                           [38] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W.
[18] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling
                                                                                Smeulders. Selective search for object recognition. IJCV,
     in deep convolutional networks for visual recognition. In
                                                                                2013. 2
     ECCV. 2014. 1, 2
                                                                           [39] S.-E. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh. Con-
[19] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning
                                                                                volutional pose machines. In CVPR, 2016. 8
     for image recognition. In CVPR, 2016. 2, 4, 7
                                                                           [40] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He. Aggregated
[20] J. Hosang, R. Benenson, P. Dollár, and B. Schiele. What                   residual transformations for deep neural networks. In CVPR,
     makes for effective detection proposals? PAMI, 2015. 2                     2017. 4, 5
[21] J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara,
     A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, et al.
     Speed/accuracy trade-offs for modern convolutional object
     detectors. In CVPR, 2017. 2, 3, 4, 5, 7


                                                                      10

