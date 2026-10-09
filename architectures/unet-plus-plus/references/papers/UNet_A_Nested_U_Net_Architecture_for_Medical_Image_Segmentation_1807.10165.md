# Paper (Zhou et al. 2018)

> Source: `https://arxiv.org/abs/1807.10165`

---

UNet++: A Nested U-Net Architecture
for Medical Image Segmentation
Zongwei Zhou, Md Mahfuzur Rahman Siddiquee,
Nima Tajbakhsh, and Jianming Liang
Arizona State University
{zongweiz,mrahmans,ntajbakh,jianming.liang}@asu.edu
Abstract. In this paper, we present UNet++, a new, more powerful ar-
chitecture for medical image segmentation. Our architecture is essentially
a deeply-supervised encoder-decoder network where the encoder and de-
coder sub-networks are connected through a series of nested, dense skip
pathways. The re-designed skip pathways aim at reducing the semantic
gap between the feature maps of the encoder and decoder sub-networks.
We argue that the optimizer would deal with an easier learning task when
the feature maps from the decoder and encoder networks are semantically
similar. We have evaluated UNet++ in comparison with U-Net and wide
U-Net architectures across multiple medical image segmentation tasks:
nodule segmentation in the low-dose CT scans of chest, nuclei segmen-
tation in the microscopy images, liver segmentation in abdominal CT
scans, and polyp segmentation in colonoscopy videos. Our experiments
demonstrate that UNet++ with deep supervision achieves an average
IoU gain of 3.9 and 3.4 points over U-Net and wide U-Net, respectively.
1
Introduction
The state-of-the-art models for image segmentation are variants of the encoder-
decoder architecture like U-Net [9] and fully convolutional network (FCN) [8].
These encoder-decoder networks used for segmentation share a key similarity:
skip connections, which combine deep, semantic, coarse-grained feature maps
from the decoder sub-network with shallow, low-level, ﬁne-grained feature maps
from the encoder sub-network. The skip connections have proved eﬀective in
recovering ﬁne-grained details of the target objects; generating segmentation
masks with ﬁne details even on complex background. Skip connections is also
fundamental to the success of instance-level segmentation models such as Mask-
RCNN, which enables the segmentation of occluded objects. Arguably, image
segmentation in natural images has reached a satisfactory level of performance,
but do these models meet the strict segmentation requirements of medical im-
ages?
Segmenting lesions or abnormalities in medical images demands a higher level
of accuracy than what is desired in natural images. While a precise segmentation
mask may not be critical in natural images, even marginal segmentation errors in
medical images can lead to poor user experience in clinical settings. For instance,
arXiv:1807.10165v1  [cs.CV]  18 Jul 2018

2
Z. Zhou, et al.
the subtle spiculation patterns around a nodule may indicate nodule malignancy;
and therefore, their exclusion from the segmentation masks would lower the
credibility of the model from the clinical perspective. Furthermore, inaccurate
segmentation may also lead to a major change in the subsequent computer-
generated diagnosis. For example, an erroneous measurement of nodule growth
in longitudinal studies can result in the assignment of an incorrect Lung-RADS
category to a screening patient. It is therefore desired to devise more eﬀective
image segmentation architectures that can eﬀectively recover the ﬁne details of
the target objects in medical images.
To address the need for more accurate segmentation in medical images, we
present UNet++, a new segmentation architecture based on nested and dense
skip connections. The underlying hypothesis behind our architecture is that the
model can more eﬀectively capture ﬁne-grained details of the foreground ob-
jects when high-resolution feature maps from the encoder network are gradually
enriched prior to fusion with the corresponding semantically rich feature maps
from the decoder network. We argue that the network would deal with an easier
learning task when the feature maps from the decoder and encoder networks are
semantically similar. This is in contrast to the plain skip connections commonly
used in U-Net, which directly fast-forward high-resolution feature maps from the
encoder to the decoder network, resulting in the fusion of semantically dissim-
ilar feature maps. According to our experiments, the suggested architecture is
eﬀective, yielding signiﬁcant performance gain over U-Net and wide U-Net.
2
Related Work
Long et al. [8] ﬁrst introduced fully convolutional networks (FCN), while U-
Net was introduced by Ronneberger et al. [9]. They both share a key idea: skip
connections. In FCN, up-sampled feature maps are summed with feature maps
skipped from the encoder, while U-Net concatenates them and add convolutions
and non-linearities between each up-sampling step. The skip connections have
shown to help recover the full spatial resolution at the network output, mak-
ing fully convolutional methods suitable for semantic segmentation. Inspired
by DenseNet architecture [5], Li et al. [7] proposed H-denseunet for liver and
liver tumor segmentation. In the same spirit, Drozdzalet al. [2] systematically
investigated the importance of skip connections, and introduced short skip con-
nections within the encoder. Despite the minor diﬀerences between the above
architectures, they all tend to fuse semantically dissimilar feature maps from
the encoder and decoder sub-networks, which, according to our experiments,
can degrade segmentation performance.
The other two recent related works are GridNet [3] and Mask-RCNN [4].
GridNet is an encoder-decoder architecture wherein the feature maps are wired in
a grid fashion, generalizing several classical segmentation architectures. GridNet,
however, lacks up-sampling layers between skip connections; and thus, it does not
represent UNet++. Mask-RCNN is perhaps the most important meta framework
for object detection, classiﬁcation and segmentation. We would like to note that

UNet++: A Nested U-Net Architecture
3
Fig. 1: (a) UNet++ consists of an encoder and decoder that are connected
through a series of nested dense convolutional blocks. The main idea behind
UNet++ is to bridge the semantic gap between the feature maps of the encoder
and decoder prior to fusion. For example, the semantic gap between (X0,0,X1,3)
is bridged using a dense convolution block with three convolution layers. In the
graphical abstract, black indicates the original U-Net, green and blue show dense
convolution blocks on the skip pathways, and red indicates deep supervision.
Red, green, and blue components distinguish UNet++ from U-Net. (b) Detailed
analysis of the ﬁrst skip pathway of UNet++. (c) UNet++ can be pruned at
inference time, if trained with deep supervision.
UNet++ can be readily deployed as the backbone architecture in Mask-RCNN
by simply replacing the plain skip connections with the suggested nested dense
skip pathways. Due to limited space, we were not able to include results of
Mask RCNN with UNet++ as the backbone architecture; however, the interested
readers can refer to the supplementary material for further details.
3
Proposed Network Architecture: UNet++
Fig. 1a shows a high-level overview of the suggested architecture. As seen,
UNet++ starts with an encoder sub-network or backbone followed by a decoder
sub-network. What distinguishes UNet++ from U-Net (the black components in
Fig. 1a) is the re-designed skip pathways (shown in green and blue) that connect
the two sub-networks and the use of deep supervision (shown red).

4
Z. Zhou, et al.
3.1
Re-designed skip pathways
Re-designed skip pathways transform the connectivity of the encoder and de-
coder sub-networks. In U-Net, the feature maps of the encoder are directly re-
ceived in the decoder; however, in UNet++, they undergo a dense convolution
block whose number of convolution layers depends on the pyramid level. For
example, the skip pathway between nodes X0,0 and X1,3 consists of a dense
convolution block with three convolution layers where each convolution layer
is preceded by a concatenation layer that fuses the output from the previous
convolution layer of the same dense block with the corresponding up-sampled
output of the lower dense block. Essentially, the dense convolution block brings
the semantic level of the encoder feature maps closer to that of the feature maps
awaiting in the decoder. The hypothesis is that the optimizer would face an
easier optimization problem when the received encoder feature maps and the
corresponding decoder feature maps are semantically similar.
Formally, we formulate the skip pathway as follows: let xi,j denote the output
of node Xi,j where i indexes the down-sampling layer along the encoder and j
indexes the convolution layer of the dense block along the skip pathway. The
stack of feature maps represented by xi,j is computed as
xi,j =
(
H
 xi−1,j
,
j = 0
H
h
xi,kj−1
k=0 , U(xi+1,j−1)
i
,
j > 0
(1)
where function H(·) is a convolution operation followed by an activation func-
tion, U(·) denotes an up-sampling layer, and [ ] denotes the concatenation layer.
Basically, nodes at level j = 0 receive only one input from the previous layer
of the encoder; nodes at level j = 1 receive two inputs, both from the encoder
sub-network but at two consecutive levels; and nodes at level j > 1 receive j + 1
inputs, of which j inputs are the outputs of the previous j nodes in the same
skip pathway and the last input is the up-sampled output from the lower skip
pathway. The reason that all prior feature maps accumulate and arrive at the
current node is because we make use of a dense convolution block along each
skip pathway. Fig. 1b further clariﬁes Eq. 1 by showing how the feature maps
travel through the top skip pathway of UNet++.
3.2
Deep supervision
We propose to use deep supervision [6] in UNet++, enabling the model to op-
erate in two modes: 1) accurate mode wherein the outputs from all segmenta-
tion branches are averaged; 2) fast mode wherein the ﬁnal segmentation map
is selected from only one of the segmentation branches, the choice of which de-
termines the extent of model pruning and speed gain. Fig. 1c shows how the
choice of segmentation branch in fast mode results in architectures of varying
complexity.
Owing to the nested skip pathways, UNet++ generates full resolution feature
maps at multiple semantic levels, {x0,j, j ∈{1, 2, 3, 4}}, which are amenable to

UNet++: A Nested U-Net Architecture
5
Table 1: The image segmentation datasets used in our experiments.
Dataset
Images
Input Size
Modality
Provider
cell nuclei
670
96×96
microscopy
Data Science Bowl 2018
colon polyp
7,379
224×224
RGB video
ASU-Mayo [10,11]
liver
331
512×512
CT
MICCAI 2018 LiTS Challenge
lung nodule
1,012
64×64×64
CT
LIDC-IDRI [1]
Table 2: Number of convolutional kernels in U-Net and wide U-Net.
encoder / decoder
X0,0/X0,4
X1,0/X1,3
X2,0/X2,2
X3,0/X3,1
X4,0/X4,0
U-Net
32
64
128
256
512
wide U-Net
35
70
140
280
560
deep supervision. We have added a combination of binary cross-entropy and dice
coeﬃcient as the loss function to each of the above four semantic levels, which
is described as:
L(Y, ˆY ) = −1
N
N
X
b=1
 
1
2 · Yb · log ˆYb + 2 · Yb · ˆYb
Yb + ˆYb
!
(2)
where ˆYb and Yb denote the ﬂatten predicted probabilities and the ﬂatten ground
truths of bth image respectively, and N indicates the batch size.
In summary, as depicted in Fig. 1a, UNet++ diﬀers from the original U-Net
in three ways: 1) having convolution layers on skip pathways (shown in green),
which bridges the semantic gap between encoder and decoder feature maps; 2)
having dense skip connections on skip pathways (shown in blue), which improves
gradient ﬂow; and 3) having deep supervision (shown in red), which as will be
shown in Section 4 enables model pruning and improves or in the worst case
achieves comparable performance to using only one loss layer.
4
Experiments
Datasets: As shown in Table 1, we use four medical imaging datasets for model
evaluation, covering lesions/organs from diﬀerent medical imaging modalities.
For further details about datasets and the corresponding data pre-processing,
we refer the readers to the supplementary material.
Baseline models: For comparison, we used the original U-Net and a customized
wide U-Net architecture. We chose U-Net because it is a common performance
baseline for image segmentation. We also designed a wide U-Net with similar
number of parameters as our suggested architecture. This was to ensure that
the performance gain yielded by our architecture is not simply due to increased
number of parameters. Table 2 details the U-Net and wide U-Net architecture.
Implementation details: We monitored the Dice coeﬃcient and Intersection
over Union (IoU), and used early-stop mechanism on the validation set. We also

6
Z. Zhou, et al.
Fig. 2: Qualitative comparison between U-Net, wide U-Net, and UNet++, show-
ing segmentation results for polyp, liver, and cell nuclei datasets (2D-only for a
distinct visualization).
used Adam optimizer with a learning rate of 3e-4. Architecture details for U-
Net and wide U-Net are shown in Table 2. UNet++ is constructed from the
original U-Net architecture. All convolutional layers along a skip pathway (Xi,j)
use k kernels of size 3×3 (or 3×3×3 for 3D lung nodule segmentation) where
k = 32 × 2i. To enable deep supervision, a 1×1 convolutional layer followed by
a sigmoid activation function was appended to each of the target nodes: {x0,j |
j ∈{1, 2, 3, 4}}. As a result, UNet++ generates four segmentation maps given an
input image, which will be further averaged to generate the ﬁnal segmentation
map. More details can be founded at github.com/Nested-UNet.
Results: Table 3 compares U-Net, wide U-Net, and UNet++ in terms of the
number parameters and segmentation accuracy for the tasks of lung nodule
segmentation, colon polyp segmentation, liver segmentation, and cell nuclei seg-
mentation. As seen, wide U-Net consistently outperforms U-Net except for liver
segmentation where the two architectures perform comparably. This improve-
ment is attributed to the larger number of parameters in wide U-Net. UNet++
without deep supervision achieves a signiﬁcant performance gain over both U-
Net and wide U-Net, yielding average improvement of 2.8 and 3.3 points in
IoU. UNet++ with deep supervision exhibits average improvement of 0.6 points
over UNet++ without deep supervision. Speciﬁcally, the use of deep supervi-
sion leads to marked improvement for liver and lung nodule segmentation, but
such improvement vanishes for cell nuclei and colon polyp segmentation. This
is because polyps and liver appear at varying scales in video frames and CT

UNet++: A Nested U-Net Architecture
7
Table 3: Segmentation results (IoU: %) for U-Net, wide U-Net and our suggested
architecture UNet++ with and without deep supervision (DS).
Architecture
Params
Dataset
cell nuclei
colon polyp
liver
lung nodule
U-Net [9]
7.76M
90.77
30.08
76.62
71.47
Wide U-Net
9.13M
90.92
30.14
76.58
73.38
UNet++ w/o DS
9.04M
92.63
33.45
79.70
76.44
UNet++ w/ DS
9.04M
92.52
32.12
82.90
77.21
Fig. 3: Complexity, speed, and accuracy of UNet++ after pruning on (a) cell
nuclei, (b) colon polyp, (c) liver, and (d) lung nodule segmentation tasks respec-
tively. The inference time is the time taken to process 10k test images using one
NVIDIA TITAN X (Pascal) with 12 GB memory.
slices; and thus, a multi-scale approach using all segmentation branches (deep
supervision) is essential for accurate segmentation. Fig. 2 shows a qualitative
comparison between the results of U-Net, wide U-Net, and UNet++.
Model pruning: Fig. 3 shows segmentation performance of UNet++ after ap-
plying diﬀerent levels of pruning. We use UNet++ Li to denote UNet++ pruned
at level i (see Fig. 1c for further details). As seen, UNet++ L3 achieves on av-
erage 32.2% reduction in inference time while degrading IoU by only 0.6 points.
More aggressive pruning further reduces the inference time but at the cost of
signiﬁcant accuracy degradation.
5
Conclusion
To address the need for more accurate medical image segmentation, we pro-
posed UNet++. The suggested architecture takes advantage of re-designed skip
pathways and deep supervision. The re-designed skip pathways aim at reducing
the semantic gap between the feature maps of the encoder and decoder sub-
networks, resulting in a possibly simpler optimization problem for the optimizer

8
Z. Zhou, et al.
to solve. Deep supervision also enables more accurate segmentation particularly
for lesions that appear at multiple scales such as polyps in colonoscopy videos.
We evaluated UNet++ using four medical imaging datasets covering lung nodule
segmentation, colon polyp segmentation, cell nuclei segmentation, and liver seg-
mentation. Our experiments demonstrated that UNet++ with deep supervision
achieved an average IoU gain of 3.9 and 3.4 points over U-Net and wide U-Net,
respectively.
Acknowledgments This research has been supported partially by NIH under
Award Number R01HL128785, by ASU and Mayo Clinic through a Seed Grant
and an Innovation Grant. The content is solely the responsibility of the authors
and does not necessarily represent the oﬃcial views of NIH.
References
1. S. G. Armato, G. McLennan, L. Bidaut, M. F. McNitt-Gray, C. R. Meyer, A. P.
Reeves, B. Zhao, D. R. Aberle, C. I. Henschke, E. A. Hoﬀman, et al. The lung
image database consortium (lidc) and image database resource initiative (idri):
a completed reference database of lung nodules on ct scans.
Medical physics,
38(2):915–931, 2011.
2. M. Drozdzal, E. Vorontsov, G. Chartrand, S. Kadoury, and C. Pal. The importance
of skip connections in biomedical image segmentation. In Deep Learning and Data
Labeling for Medical Applications, pages 179–187. Springer, 2016.
3. D. Fourure, R. Emonet, E. Fromont, D. Muselet, A. Tremeau, and C. Wolf.
Residual conv-deconv grid network for semantic segmentation.
arXiv preprint
arXiv:1707.07958, 2017.
4. K. He, G. Gkioxari, P. Doll´ar, and R. Girshick. Mask r-cnn. In Computer Vision
(ICCV), 2017 IEEE International Conference on, pages 2980–2988. IEEE, 2017.
5. G. Huang, Z. Liu, K. Q. Weinberger, and L. van der Maaten. Densely connected
convolutional networks. In Proceedings of the IEEE conference on computer vision
and pattern recognition, volume 1, page 3, 2017.
6. C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-supervised nets. In
Artiﬁcial Intelligence and Statistics, pages 562–570, 2015.
7. X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, and P. A. Heng. H-denseunet: Hybrid
densely connected unet for liver and liver tumor segmentation from ct volumes.
arXiv preprint arXiv:1709.07330, 2017.
8. J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for seman-
tic segmentation. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 3431–3440, 2015.
9. O. Ronneberger, P. Fischer, and T. Brox.
U-net: Convolutional networks for
biomedical image segmentation.
In International Conference on Medical image
computing and computer-assisted intervention, pages 234–241. Springer, 2015.
10. N. Tajbakhsh, J. Y. Shin, S. R. Gurudu, R. T. Hurst, C. B. Kendall, M. B. Gotway,
and J. Liang.
Convolutional neural networks for medical image analysis: Full
training or ﬁne tuning? IEEE transactions on medical imaging, 35(5):1299–1312,
2016.
11. Z. Zhou, J. Shin, L. Zhang, S. Gurudu, M. Gotway, and J. Liang. Fine-tuning
convolutional neural networks for biomedical image analysis: actively and incre-
mentally. In IEEE conference on computer vision and pattern recognition (CVPR),
pages 7340–7351, 2017.
