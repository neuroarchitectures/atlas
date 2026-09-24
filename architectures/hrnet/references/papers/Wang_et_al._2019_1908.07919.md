# Deep High-Resolution Representation Learning for Visual Recognition

High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams in parallel; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at https://github.com/HRNet.

## I Introduction

Deep convolutional neural networks (DCNNs) have achieved state-of-the-art results in many computer vision tasks, such as image classification, object detection, semantic segmentation, human pose estimation, and so on. The strength is that DCNNs are able to learn richer representations than conventional hand-crafted representations.

Most recently-developed classification networks, including AlexNet [77], VGGNet [126], GoogleNet [133], ResNet [54], etc., follow the design rule of LeNet-55 [81]. The rule is depicted in Figure 1 (a): gradually reduce the spatial size of the feature maps, connect the convolutions from high resolution to low resolution in series, and lead to a low-resolution representation, which is further processed for classification.

High-resolution representations are needed for position-sensitive tasks, e.g., semantic segmentation, human pose estimation, and object detection. The previous state-of-the-art methods adopt the high-resolution recovery process to raise the representation resolution from the low-resolution representation outputted by a classification or classification-like network as depicted in Figure 1 (b), e.g., Hourglass [105], SegNet [3], DeconvNet [107], U-Net [119], SimpleBaseline [152], and encoder-decoder [112]. In addition, dilated convolutions are used to remove some down-sample layers and thus yield medium-resolution representations [19, 181].

We present a novel architecture, namely High-Resolution Net (HRNet), which is able to maintain high-resolution representations through the whole process. We start from a high-resolution convolution stream, gradually add high-to-low resolution convolution streams one by one, and connect the multi-resolution streams in parallel. The resulting network consists of several (44 in this paper) stages as depicted in Figure 2, and the nnth stage contains nn streams corresponding to nn resolutions. We conduct repeated multi-resolution fusions by exchanging the information across the parallel streams over and over.

The high-resolution representations learned from HRNet are not only semantically strong but also spatially precise. This comes from two aspects. (i) Our approach connects high-to-low resolution convolution streams in parallel rather than in series. Thus, our approach is able to maintain the high resolution instead of recovering high resolution from low resolution, and accordingly the learned representation is potentially spatially more precise. (ii) Most existing fusion schemes aggregate high-resolution low-level and high-level representations obtained by upsampling low-resolution representations. Instead, we repeat multi-resolution fusions to boost the high-resolution representations with the help of the low-resolution representations, and vice versa. As a result, all the high-to-low resolution representations are semantically strong.

We present two versions of HRNet. The first one, named as HRNetV11, only outputs the high-resolution representation computed from the high-resolution convolution stream. We apply it to human pose estimation by following the heatmap estimation framework. We empirically demonstrate the superior pose estimation performance on the COCO keypoint detection dataset [94].

The other one, named as HRNetV22, combines the representations from all the high-to-low resolution parallel streams. We apply it to semantic segmentation through estimating segmentation maps from the combined high-resolution representation. The proposed approach achieves state-of-the-art results on PASCAL-Context, Cityscapes, and LIP with similar model sizes and lower computation complexity. We observe similar performance for HRNetV11 and HRNetV22 over COCO pose estimation, and the superiority of HRNetV22 to HRNet11 in semantic segmentation.

In addition, we construct a multi-level representation, named as HRNetV22p, from the high-resolution representation output from HRNetV22, and apply it to state-of-the-art detection frameworks, including Faster R-CNN, Cascade R-CNN [12], FCOS [136], and CenterNet [36], and state-of-the-art joint detection and instance segmentation frameworks, including Mask R-CNN [53], Cascade Mask R-CNN, and Hybrid Task Cascade [16]. The results show that our method gets detection performance improvement and in particular dramatic improvement for small objects.

## II Related Work

We review closely-related representation learning techniques developed mainly for human pose estimation [57], semantic segmentation and object detection, from three aspects: low-resolution representation learning, high-resolution representation recovering, and high-resolution representation maintaining. Besides, we mention about some works related to multi-scale fusion.

Learning low-resolution representations. The fully-convolutional network approaches [99, 124] compute low-resolution representations by removing the fully-connected layers in a classification network, and estimate their coarse segmentation maps. The estimated segmentation maps are improved by combining the fine segmentation score maps estimated from intermediate low-level medium-resolution representations [99], or iterating the processes [76]. Similar techniques have also been applied to edge detection, e.g., holistic edge detection [157].

The fully convolutional network is extended, by replacing a few (typically two) strided convolutions and the associated convolutions with dilated convolutions, to the dilation version, leading to medium-resolution representations [181, 19, 168, 18, 86]. The representations are further augmented to multi-scale contextual representations [181, 19, 21] through feature pyramids for segmenting objects at multiple scales.

Recovering high-resolution representations. An upsample process can be used to gradually recover the high-resolution representations from the low-resolution representations. The upsample subnetwork could be a symmetric version of the downsample process (e.g., VGGNet), with skipping connection over some mirrored layers to transform the pooling indices, e.g., SegNet [3] and DeconvNet [107], or copying the feature maps, e.g., U-Net [119] and Hourglass [105, 27, 165, 68, 163, 9, 31, 8, 134], encoder-decoder [112], and so on. An extension of U-Net, full-resolution residual network [114], introduces an extra full-resolution stream that carries information at the full image resolution, to replace the skip connections, and each unit in the downsample and upsample subnetworks receives information from and sends information to the full-resolution stream.

The asymmetric upsample process is also widely studied. RefineNet [90] improves the combination of upsampled representations and the representations of the same resolution copied from the downsample process. Other works include: light upsample process [7, 24, 152, 92], possibly with dilated convolutions used in the backbone [63, 113, 89]; light downsample and heavy upsample processes [141], recombinator networks [55]; improving skip connections with more or complicated convolutional units [111, 180, 64], as well as sending information from low-resolution skip connections to high-resolution skip connections [189] or exchanging information between them [49]; studying the details of the upsample process [147]; combining multi-scale pyramid representations [22, 154]; stacking multiple DeconvNets/U-Nets/Hourglass [44, 149] with dense connections [135].

Maintaining high-resolution representations. Our work is closely related to several works that can also generate high-resolution representations, e.g., convolutional neural fabrics [123], interlinked CNNs [188], GridNet [42], and multi-scale DenseNet [58].

The two early works, convolutional neural fabrics [123] and interlinked CNNs [188], lack careful design on when to start low-resolution parallel streams, and how and where to exchange information across parallel streams, and do not use batch normalization and residual connections, thus not showing satisfactory performance. GridNet [42] is like a combination of multiple U-Nets and includes two symmetric information exchange stages: the first stage passes information only from high resolution to low resolution, and the second stage passes information only from low resolution to high resolution. This limits its segmentation quality. Multi-scale DenseNet [58] is not able to learn strong high-resolution representations as there is no information received from low-resolution representations.

Multi-scale fusion. Multi-scale fusion11 1 In this paper, Multi-scale fusion and multi-resolution fusion are interchangeable, but in other contexts, they may not be interchangeable. is widely studied [11, 19, 24, 42, 157, 181, 66, 161, 122, 123, 58, 188]. The straightforward way is to feed multi-resolution images separately into multiple networks and aggregate the output response maps [137]. Hourglass [105], U-Net [119], and SegNet [3] combine low-level features in the high-to-low downsample process into the same-resolution high-level features in the low-to-high upsample process progressively through skip connections. PSPNet [181] and DeepLabV2/3 [19] fuse the pyramid features obtained by pyramid pooling module and atrous spatial pyramid pooling. Our multi-scale (resolution) fusion module resembles the two pooling modules. The differences include: (1) Our fusion outputs four-resolution representations other than only one, and (2) our fusion modules are repeated several times which is inspired by deep fusion [143, 155, 129, 178, 184].

Our approach. Our network connects high-to-low convolution streams in parallel. It maintains high-resolution representations through the whole process, and generates reliable high-resolution representations with strong position sensitivity through repeatedly fusing the representations from multi-resolution streams.

This paper represents a very substantial extension of our previous conference paper [130] with an additional material added from our unpublished technical report [131] as well as more object detection results under recently-developed start-of-the-art object detection and instance segmentation frameworks. The main technical novelties compared with [130] lie in threefold. (1) We extend the network (named as HRNetV11) proposed in [130], to two versions: HRNetV22 and HRNetV22p, which explore all the four-resolution representations. (2) We build the connection between multi-resolution fusion and regular convolution, which provides an evidence for the necessity of exploring all the four-resolution representations in HRNetV22 and HRNetV22p. (3) We show the superiority of HRNetV22 and HRNetV22p over HRNetV11 and present the applications of HRNetV22 and HRNetV22p in a broad range of vision problems, including semantic segmentation and object detection.

## III High-Resolution Networks

We input the image into a stem, which consists of two stride-22 3×33\times 3 convolutions decreasing the resolution to 14\frac{1}{4}, and subsequently the main body that outputs the representation with the same resolution (14\frac{1}{4}). The main body, illustrated in Figure 2 and detailed below, consists of several components: parallel multi-resolution convolutions, repeated multi-resolution fusions, and representation head that is shown in Figure 4.

### III-A Parallel Multi-Resolution Convolutions

We start from a high-resolution convolution stream as the first stage, gradually add high-to-low resolution streams one by one, forming new stages, and connect the multi-resolution streams in parallel. As a result, the resolutions for the parallel streams of a later stage consists of the resolutions from the previous stage, and an extra lower one.

An example network structure illustrated in Figure 2, containing 44 parallel streams, is logically as follows,

|  | 𝒩11→𝒩21→𝒩31→𝒩41↘𝒩22→𝒩32→𝒩42↘𝒩33→𝒩43↘𝒩44,\begin{array}[]{llll}\mathcal{N}_{11}&\rightarrow~~\mathcal{N}_{21}&\rightarrow~~\mathcal{N}_{31}&\rightarrow~~\mathcal{N}_{41}\\ &\searrow~~\mathcal{N}_{22}&\rightarrow~~\mathcal{N}_{32}&\rightarrow~~\mathcal{N}_{42}\\ &&\searrow~~\mathcal{N}_{33}&\rightarrow~~\mathcal{N}_{43}\\ &&&\searrow~~\mathcal{N}_{44},\end{array} |  | (1) |
|---|---|---|---|

where 𝒩s​r\mathcal{N}_{sr} is a sub-stream in the ssth stage and rr is the resolution index. The resolution index of the first stream is r=1r=1. The resolution of index rr is 12r−1\frac{1}{2^{r-1}} of the resolution of the first stream.

### III-B Repeated Multi-Resolution Fusions

The goal of the fusion module is to exchange the information across multi-resolution representations. It is repeated several times (e.g., every 44 residual units).

Let us look at an example of fusing 33-resolution representations, which is illustrated in Figure 3. Fusing 22 representations and 44 representations can be easily derived. The input consists of three representations: {𝐑ri,r=1,2,3}\{\mathbf{R}_{r}^{i},r=1,2,3\}, with rr is the resolution index, and the associated output representations are {𝐑ro,r=1,2,3}\{\mathbf{R}_{r}^{o},r=1,2,3\}. Each output representation is the sum of the transformed representations of the three inputs: 𝐑ro=f1​r​(𝐑1i)+f2​r​(𝐑2i)+f3​r​(𝐑3i)\mathbf{R}_{r}^{o}=f_{1r}(\mathbf{R}_{1}^{i})+f_{2r}(\mathbf{R}_{2}^{i})+f_{3r}(\mathbf{R}_{3}^{i}). The fusion across stages (from stage 33 to stage 44) has an extra output: 𝐑4o=f14​(𝐑1i)+f24​(𝐑2i)+f34​(𝐑3i)\mathbf{R}_{4}^{o}=f_{14}(\mathbf{R}_{1}^{i})+f_{24}(\mathbf{R}_{2}^{i})+f_{34}(\mathbf{R}_{3}^{i}).

The choice of the transform function fx​r​(⋅)f_{xr}(\cdot) is dependent on the input resolution index xx and the output resolution index rr. If x=rx=r, fx​r​(𝐑)=𝐑f_{xr}(\mathbf{R})=\mathbf{R}. If x<rx<r, fx​r​(𝐑)f_{xr}(\mathbf{R}) downsamples the input representation 𝐑\mathbf{R} through (r−s)(r-s) stride-22 3×33\times 3 convolutions. For instance, one stride-22 3×33\times 3 convolution for 2×2\times downsampling, and two consecutive stride-22 3×33\times 3 convolutions for 4×4\times downsampling. If x>rx>r, fx​r​(𝐑)f_{xr}(\mathbf{R}) upsamples the input representation 𝐑\mathbf{R} through the bilinear upsampling followed by a 1×11\times 1 convolution for aligning the number of channels. The functions are depicted in Figure 3.

### III-C Representation Head

We have three kinds of representation heads that are illustrated in Figure 4, and call them as HRNetV11, HRNetV22, and HRNetV11p, respectively.

HRNetV11. The output is the representation only from the high-resolution stream. Other three representations are ignored. This is illustrated in Figure 4 (a).

HRNetV22. We rescale the low-resolution representations through bilinear upsampling without changing the number of channels to the high resolution, and concatenate the four representations, followed by a 1×11\times 1 convolution to mix the four representations. This is illustrated in Figure 4 (b).

HRNetV22p. We construct multi-level representations by downsampling the high-resolution representation output from HRNetV22 to multiple levels. This is depicted in Figure 4 (c).

In this paper, we will show the results of applying HRNetV11 to human pose estimation, HRNetV22 to semantic segmentation, and HRNetV22p to object detection.

### III-D Instantiation

The main body contains four stages with four parallel convolution streams. The resolutions are 1/41/4, 1/81/8, 1/161/16, and 1/321/32. The first stage contains 44 residual units where each unit is formed by a bottleneck with the width 6464, and is followed by one 3×33\times 3 convolution changing the width of feature maps to CC. The 22nd, 33rd, 44th stages contain 11, 44, 33 modularized blocks, respectively. Each branch in multi-resolution parallel convolution of the modularized block contains 44 residual units. Each unit contains two 3×33\times 3 convolutions for each resolution, where each convolution is followed by batch normalization and the nonlinear activation ReLU. The widths (numbers of channels) of the convolutions of the four resolutions are CC, 2​C2C, 4​C4C, and 8​C8C, respectively. An example is depicted in Figure 2.

### III-E Analysis

We analyze the modularized block that is divided into two components: multi-resolution parallel convolutions (Figure 5 (a)), and multi-resolution fusion (Figure 5 (b)). The multi-resolution parallel convolution resembles the group convolution. It divides the input channels into several subsets of channels and performs a regular convolution over each subset over different spatial resolutions separately, while in the group convolution, the resolutions are the same. This connection implies that the multi-resolution parallel convolution enjoys some benefit of the group convolution.

The multi-resolution fusion unit resembles the multi-branch full-connection form of the regular convolution, illustrated in Figure 5 (c). A regular convolution can be divided as multiple small convolutions as explained in [178]. The input channels are divided into several subsets, and the output channels are also divided into several subsets. The input and output subsets are connected in a fully-connected fashion, and each connection is a regular convolution. Each subset of output channels is a summation of the outputs of the convolutions over each subset of input channels. The differences lie in that our multi-resolution fusion needs to handle the resolution change. The connection between multi-resolution fusion and regular convolution provides an evidence for exploring all the four-resolution representations done in HRNetV22 and HRNetV22p.

| Method | Backbone | Pretrain | Input size | #Params | GFLOPs | AP\operatorname{AP} | AP50\operatorname{AP}^{50} | AP75\operatorname{AP}^{75} | APM\operatorname{AP}^{M} | APL\operatorname{AP}^{L} | AR\operatorname{AR} |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 88-stage Hourglass [105] | 88-stage Hourglass | N | 256×192256\times 192 | 25.125.1M | 14.314.3 | 66.966.9 | −- | −- | −- | −- | −- |
| CPN [24] | ResNet-50 | Y | 256×192256\times 192 | 27.027.0M | 6.206.20 | 68.668.6 | −- | −- | −- | −- | −- |
| CPN + OHKM [24] | ResNet-50 | Y | 256×192256\times 192 | 27.027.0M | 6.206.20 | 69.469.4 | −- | −- | −- | −- | −- |
| SimpleBaseline [152] | ResNet-50 | Y | 256×192256\times 192 | 34.034.0M | 8.908.90 | 70.4{70.4} | 88.6{88.6} | 78.3{78.3} | 67.1{67.1} | 77.2{77.2} | 76.3{76.3} |
| SimpleBaseline [152] | ResNet-101 | Y | 256×192256\times 192 | 53.053.0M | 12.412.4 | 71.4{71.4} | 89.3{89.3} | 79.3{79.3} | 68.1{68.1} | 78.1{78.1} | 77.1{77.1} |
| SimpleBaseline [152] | ResNet-152 | Y | 256×192256\times 192 | 68.668.6M | 15.715.7 | 72.0{72.0} | 89.3{89.3} | 79.8{79.8} | 68.7{68.7} | 78.9{78.9} | 77.8{77.8} |
| HRNetV11 | HRNetV11-W3232 | N | 256×192256\times 192 | 28.528.5M | 7.107.10 | 73.473.4 | 89.589.5 | 80.780.7 | 70.270.2 | 80.180.1 | 78.978.9 |
| HRNetV11 | HRNetV11-W3232 | Y | 256×192256\times 192 | 28.528.5M | 7.107.10 | 74.474.4 | 90.590.5 | 81.981.9 | 70.870.8 | 81.081.0 | 79.879.8 |
| HRNetV11 | HRNetV11-W4848 | Y | 256×192256\times 192 | 63.663.6M | 14.614.6 | 75.175.1 | 90.690.6 | 82.282.2 | 71.571.5 | 81.881.8 | 80.480.4 |
| SimpleBaseline [152] | ResNet-152 | Y | 384×288384\times 288 | 68.668.6M | 35.635.6 | 74.3{74.3} | 89.6{89.6} | 81.1{81.1} | 70.5{70.5} | 79.7{79.7} | 79.7{79.7} |
| HRNetV11 | HRNetV11-W3232 | Y | 384×288384\times 288 | 28.528.5M | 16.016.0 | 75.875.8 | 90.690.6 | 82.7{82.7} | 71.971.9 | 82.882.8 | 81.081.0 |
| HRNetV11 | HRNetV11-W4848 | Y | 384×288384\times 288 | 63.663.6M | 32.932.9 | 76.3\mathbf{76.3} | 90.8\mathbf{90.8} | 82.9\mathbf{82.9} | 72.3\mathbf{72.3} | 83.4\mathbf{83.4} | 81.2\mathbf{81.2} |

| Method | Backbone | Input size | #Params | GFLOPs | AP\operatorname{AP} | AP50\operatorname{AP}^{50} | AP75\operatorname{AP}^{75} | APM\operatorname{AP}^{M} | APL\operatorname{AP}^{L} | AR\operatorname{AR} |
|---|---|---|---|---|---|---|---|---|---|---|
| Bottom-up: keypoint detection and grouping |  |  |  |  |  |  |  |  |  |  |
| OpenPose [15] | −\hfil- | −- | −- | −- | 61.861.8 | 84.984.9 | 67.567.5 | 57.157.1 | 68.268.2 | 66.566.5 |
| Associative Embedding [104] | −\hfil- | −- | −- | −- | 65.565.5 | 86.886.8 | 72.372.3 | 60.660.6 | 72.672.6 | 70.270.2 |
| PersonLab [108] | −\hfil- | −- | −- | −- | 68.768.7 | 89.089.0 | 75.475.4 | 64.164.1 | 75.575.5 | 75.475.4 |
| MultiPoseNet [72] | −\hfil- | −- | −- | −- | 69.669.6 | 86.386.3 | 76.676.6 | 65.065.0 | 76.376.3 | 73.573.5 |
| Top-down: human detection and single-person keypoint detection |  |  |  |  |  |  |  |  |  |  |
| Mask-RCNN [53] | ResNet-50-FPN | −- | −- | −- | 63.163.1 | 87.387.3 | 68.768.7 | 57.857.8 | 71.471.4 | −- |
| G-RMI [109] | ResNet-101 | 353×257353\times 257 | 42.642.6M | 57.057.0 | 64.964.9 | 85.585.5 | 71.371.3 | 62.362.3 | 70.070.0 | 69.769.7 |
| Integral Pose Regression [132] | ResNet-101 | 256×256256\times 256 | 45.045.0M | 11.011.0 | 67.867.8 | 88.288.2 | 74.874.8 | 63.963.9 | 74.074.0 | −- |
| G-RMI + extra data [109] | ResNet-101 | 353×257353\times 257 | 42.642.6M | 57.057.0 | 68.568.5 | 87.187.1 | 75.575.5 | 65.865.8 | 73.373.3 | 73.373.3 |
| CPN [24] | ResNet-Inception | 384×288384\times 288 | −- | −- | 72.172.1 | 91.491.4 | 80.080.0 | 68.768.7 | 77.277.2 | 78.578.5 |
| RMPE [38] | PyraNet [165] | 320×256320\times 256 | 28.128.1M | 26.726.7 | 72.372.3 | 89.289.2 | 79.179.1 | 68.068.0 | 78.678.6 | −- |
| CFN [60] | −\hfil- | −- | −- | −- | 72.672.6 | 86.186.1 | 69.769.7 | 78.378.3 | 64.164.1 | −- |
| CPN (ensemble) [24] | ResNet-Inception | 384×288384\times 288 | −- | −- | 73.073.0 | 91.791.7 | 80.980.9 | 69.569.5 | 78.178.1 | 79.079.0 |
| SimpleBaseline [152] | ResNet-152 | 384×288384\times 288 | 68.668.6M | 35.635.6 | 73.7{73.7} | 91.9{91.9} | 81.1{81.1} | 70.3{70.3} | 80.0{80.0} | 79.0{79.0} |
| HRNetV11 | HRNetV11-W3232 | 384×288384\times 288 | 28.528.5M | 16.016.0 | 74.974.9 | 92.592.5 | 82.882.8 | 71.371.3 | 80.980.9 | 80.180.1 |
| HRNetV11 | HRNetV11-W4848 | 384×288384\times 288 | 63.663.6M | 32.932.9 | 75.5{75.5} | 92.5{92.5} | 83.3{83.3} | 71.9{71.9} | 81.5{81.5} | 80.5{80.5} |
| HRNetV11 + extra data | HRNetV11-W4848 | 384×288384\times 288 | 63.663.6M | 32.932.9 | 77.0\mathbf{77.0} | 92.7\mathbf{92.7} | 84.5\mathbf{84.5} | 73.4\mathbf{73.4} | 83.1\mathbf{83.1} | 82.0\mathbf{82.0} |

## IV Human Pose Estimation

Human pose estimation, a.k.a. keypoint detection, aims to detect the locations of KK keypoints or parts (e.g., elbow, wrist, etc) from an image 𝐈\mathbf{I} of size W×H×3W\times H\times 3. We follow the state-of-the-art framework and transform this problem to estimating KK heatmaps of size W4×H4\frac{W}{4}\times\frac{H}{4}, {𝐇1,𝐇2,…,𝐇K}\{\mathbf{H}_{1},\mathbf{H}_{2},\dots,\mathbf{H}_{K}\}, where each heatmap 𝐇k\mathbf{H}_{k} indicates the location confidence of the kkth keypoint.

We regress the heatmaps over the high-resolution representations output by HRNetV11. We empirically observe that the performance is almost the same for HRNetV11 and HRNetV22, and thus we choose HRNetV11 as its computation complexity is a little lower. The loss function, defined as the mean squared error, is applied for comparing the predicted heatmaps and the groundtruth heatmaps. The groundtruth heatmaps are generated by applying 22D Gaussian with standard deviation of 22 pixel centered on the groundtruth location of each keypoint. Some example results are given in Figure 6.

Dataset. The COCO dataset [94] contains over 200,000200,000 images and 250,000250,000 person instances labeled with 1717 keypoints. We train our model on the COCO train2017 set, including 57​K57K images and 150​K150K person instances. We evaluate our approach on the val2017 and test-dev2017 sets, containing 50005000 images and 20​K20K images, respectively.

Evaluation metric. The standard evaluation metric is based on Object Keypoint Similarity (OKS): OKS=∑iexp(−di2/2s2ki2)δ(vi>0)∑iδ⁡(vi>0).\operatorname{OKS}=\frac{\sum_{i}\exp(-d_{i}^{2}/2s^{2}k_{i}^{2})\delta(v_{i}>0)}{\sum_{i}\delta(v_{i}>0)}. Here did_{i} is the Euclidean distance between the detected keypoint and the corresponding ground truth, viv_{i} is the visibility flag of the ground truth, ss is the object scale, and kik_{i} is a per-keypoint constant that controls falloff. We report standard average precision and recall scores22 2 http://cocodataset.org/#keypoints-eval: AP50\operatorname{AP}^{50} (AP\operatorname{AP} at OKS=0.50\operatorname{OKS}=0.50), AP75\operatorname{AP}^{75}, AP\operatorname{AP} (the mean of AP\operatorname{AP} scores at 1010 OKS\operatorname{OKS} positions, 0.50,0.55,…,0.90,0.950.50,0.55,\dots,0.90,0.95); APM\operatorname{AP}^{M} for medium objects, APL\operatorname{AP}^{L} for large objects, and AR\operatorname{AR} (the mean of AR\operatorname{AR} scores at 1010 OKS\operatorname{OKS} positions, 0.50,0.55,…,0.90,0.950.50,0.55,\dots,0.90,0.95).

Training. We extend the human detection box in height or width to a fixed aspect ratio: height:width=4:3\operatorname{height}:\operatorname{width}=4:3, and then crop the box from the image, which is resized to a fixed size, 256×192256\times 192 or 384×288384\times 288. The data augmentation scheme includes random rotation ([−45​°,45​°][$$,$$]), random scale ([0.65,1.35][0.65,1.35]), and flipping. Following [146], half body data augmentation is also involved.

We use the Adam optimizer [71]. The learning schedule follows the setting [152]. The base learning rate is set as 1​e−31\mathrm{e}{-3}, and is dropped to 1​e−41\mathrm{e}{-4} and 1​e−51\mathrm{e}{-5} at the 170170th and 200200th epochs, respectively. The training process is terminated within 210210 epochs. The models are trained on 44 V100100 GPUs and it takes around 6060 (8080) hours for HRNet-W3232 (HRNet-W4848).

Testing. The two-stage top-down paradigm similar as [109, 24, 152] is used: detect the person instance using a person detector, and then predict detection keypoints.

We use the same person detectors provided by SimpleBaseline33 3 https://github.com/Microsoft/human-pose-estimation.pytorch for both the val and test-dev sets. Following [152, 105, 24], we compute the heatmap by averaging the heatmaps of the original and flipped images. Each keypoint location is predicted by adjusting the highest heatvalue location with a quarter offset in the direction from the highest response to the second highest response.

Results on the val set. We report the results of our method and other state-of–the-art methods in Table I. The network - HRNetV11-W3232, trained from scratch with the input size 256×192256\times 192, achieves an AP score 73.473.4, outperforming other methods with the same input size. (i) Compared to Hourglass [105], our network improves AP by 6.56.5 points, and the GFLOP of our network is much lower and less than half, while the numbers of parameters are similar and ours is slightly larger. (ii) Compared to CPN [24] w/o and w/ OHKM, our network, with slightly larger model size and slightly higher complexity, achieves 4.84.8 and 4.04.0 points gain, respectively. (iii) Compared to the previous best-performed method SimpleBaseline [152], our HRNetV11-W3232 obtains significant improvements: 3.03.0 points gain for the backbone ResNet-5050 with a similar model size and GFLOPs, and 1.41.4 points gain for the backbone ResNet-152152 whose model size (#Params) and GFLOPs are twice as many as ours.

Our network can benefit from (i) training from the model pretrained on the ImageNet: The gain is 1.01.0 points for HRNetV11-W3232; (ii) increasing the capacity by increasing the width: HRNetV11-W4848 gets 0.70.7 and 0.50.5 points gain for the input sizes 256×192256\times 192 and 384×288384\times 288, respectively.

Considering the input size 384×288384\times 288, our HRNetV11-W3232 and HRNetV11-W4848, get the 75.875.8 and 76.376.3 AP, which have 1.41.4 and 1.21.2 improvements compared to the input size 256×192256\times 192. In comparison to SimpleBaseline [152] that uses ResNet-152152 as the backbone, our HRNetV11-W3232 and HRNetV11-W4848 attain 1.51.5 and 2.02.0 points gain in terms of AP at 45%45\% and 92.4%92.4\% computational cost, respectively.

Results on the test-dev set. Table II reports the pose estimation performances of our approach and the existing state-of-the-art approaches. Our approach is significantly better than bottom-up approaches. On the other hand, our small network, HRNetV11-W3232, achieves an AP of 74.974.9. It outperforms all the other top-down approaches, and is more efficient in terms of model size (#Params) and computation complexity (GFLOPs). Our big model, HRNetV11-W4848, achieves the highest AP score 75.575.5. Compared to SimpleBaseline [152] with the same input size, our small and big networks receive 1.21.2 and 1.81.8 improvements, respectively. With the additional data from AI Challenger [148] for training, our single big network can obtain an AP of 77.077.0.

|  | backbone | #param. | GFLOPs | mIoU |
|---|---|---|---|---|
| UNet++ [189] | ResNet-101101 | 59.559.5M | 748.5748.5 | 75.575.5 |
| Dilated-ResNet [54] | D-ResNet-101101 | 52.152.1M | 1661.61661.6 | 75.775.7 |
| DeepLabv3 [20] | D-ResNet-101101 | 58.058.0M | 1778.71778.7 | 78.578.5 |
| DeepLabv3+ [22] | D-Xception-7171 | 43.543.5M | 1444.61444.6 | 79.679.6 |
| PSPNet [181] | D-ResNet-101101 | 65.965.9M | 2017.62017.6 | 79.779.7 |
| HRNetV22 | HRNetV22-W4040 | 45.245.2M | 493.2493.2 | 80.280.2 |
| HRNetV22 | HRNetV22-W4848 | 65.965.9M | 696.2696.2 | 81.181.1 |
| HRNetV22 + OCR [170] | HRNetV22-W4848 | 70.370.3M | 1206.31206.3 | 81.6\mathbf{81.6} |

|  | backbone | mIoU | iIoU cla. | IoU cat. | iIoU cat. |
|---|---|---|---|---|---|
| Model learned on the train set |  |  |  |  |  |
| PSPNet [181] | D-ResNet-101101 | 78.478.4 | 56.756.7 | 90.690.6 | 78.678.6 |
| PSANet [182] | D-ResNet-101101 | 78.678.6 | - | - | - |
| PAN [82] | D-ResNet-101101 | 78.678.6 | - | - | - |
| AAF [69] | D-ResNet-101101 | 79.179.1 | - | - | - |
| HRNetV22 | HRNetV22-W4848 | 80.4\mathbf{80.4} | 59.2\mathbf{59.2} | 91.5\mathbf{91.5} | 80.8\mathbf{80.8} |
| Model learned on the train+val set |  |  |  |  |  |
| GridNet [42] | - | 69.569.5 | 44.144.1 | 87.987.9 | 71.171.1 |
| LRR-4x [46] | - | 69.769.7 | 48.048.0 | 88.288.2 | 74.774.7 |
| DeepLab [19] | D-ResNet-101101 | 70.470.4 | 42.642.6 | 86.486.4 | 67.767.7 |
| LC [84] | - | 71.171.1 | - | - | - |
| Piecewise [91] | VGG-1616 | 71.671.6 | 51.751.7 | 87.387.3 | 74.174.1 |
| FRRN [114] | - | 71.871.8 | 45.545.5 | 88.988.9 | 75.175.1 |
| RefineNet [90] | ResNet-101101 | 73.673.6 | 47.247.2 | 87.987.9 | 70.670.6 |
| PEARL [65] | D-ResNet-101101 | 75.475.4 | 51.651.6 | 89.289.2 | 75.175.1 |
| DSSPN [88] | D-ResNet-101101 | 76.676.6 | 56.256.2 | 89.689.6 | 77.877.8 |
| LKM [111] | ResNet-152152 | 76.976.9 | - | - | - |
| DUC-HDC [144] | - | 77.677.6 | 53.653.6 | 90.190.1 | 75.275.2 |
| SAC [176] | D-ResNet-101101 | 78.178.1 | - | - | - |
| DepthSeg [73] | D-ResNet-101101 | 78.278.2 | - | - | - |
| ResNet38 [151] | WResNet-38 | 78.478.4 | 59.159.1 | 90.990.9 | 78.178.1 |
| BiSeNet [166] | ResNet-101101 | 78.978.9 | - | - | - |
| DFN [167] | ResNet-101101 | 79.379.3 | - | - | - |
| PSANet [182] | D-ResNet-101101 | 80.180.1 | - | - | - |
| PADNet [159] | D-ResNet-101101 | 80.380.3 | 58.858.8 | 90.890.8 | 78.578.5 |
| CFNet [173] | D-ResNet-101101 | 79.679.6 | - | - | - |
| Auto-DeepLab [95] | - | 80.480.4 | - | - | - |
| DenseASPP [181] | WDenseNet-161161 | 80.680.6 | 59.159.1 | 90.990.9 | 78.178.1 |
| SVCNet [33] | ResNet-101101 | 81.081.0 | - | - | - |
| ANN [195] | D-ResNet-101101 | 81.381.3 | - | - | - |
| CCNet [61] | D-ResNet-101101 | 81.481.4 | - | - | - |
| DANet [43] | D-ResNet-101101 | 81.581.5 | - | - | - |
| HRNetV22 | HRNetV22-W4848 | 81.681.6 | 61.8\mathbf{61.8} | 92.1\mathbf{92.1} | 82.2\mathbf{82.2} |
| HRNetV22 + OCR [170] | HRNetV22-W4848 | 82.5\mathbf{82.5} | 61.761.7 | 92.1\mathbf{92.1} | 81.681.6 |

|  | backbone | mIoU (5959) | mIoU (6060) |
|---|---|---|---|
| FCN-88s [125] | VGG-1616 | - | 35.135.1 |
| BoxSup [29] | - | - | 40.540.5 |
| HO_CRF [2] | - | - | 41.341.3 |
| Piecewise [91] | VGG-1616 | - | 43.343.3 |
| DeepLab-v22 [19] | D-ResNet-101101 | - | 45.745.7 |
| RefineNet [90] | ResNet-152152 | - | 47.347.3 |
| UNet++ [189] | ResNet-101101 | 47.747.7 | - |
| PSPNet [181] | D-ResNet-101101 | 47.847.8 | - |
| Ding et al. [32] | ResNet-101101 | 51.651.6 | - |
| EncNet [172] | D-ResNet-101101 | 52.652.6 | - |
| DANet [43] | D-ResNet-101101 | 52.652.6 | - |
| ANN [195] | D-ResNet-101101 | 52.852.8 | - |
| SVCNet [33] | ResNet-101101 | 53.253.2 | - |
| CFNet [173] | D-ResNet-101101 | 54.0{54.0} | - |
| APCN [51] | D-ResNet-101101 | 55.655.6 | - |
| HRNetV22 | HRNetV22-W4848 | 54.0{54.0} | 48.348.3 |
| HRNetV22 + OCR [170] | HRNetV22-W4848 | 56.2\mathbf{56.2} | 50.1\mathbf{50.1} |

|  | backbone | extra. | pixel acc. | avg. acc. | mIoU |
|---|---|---|---|---|---|
| Attention+SSL [47] | VGG1616 | Pose | 84.3684.36 | 54.9454.94 | 44.7344.73 |
| DeepLabV33+ [22] | D-ResNet-101101 | - | 84.0984.09 | 55.6255.62 | 44.8044.80 |
| MMAN [100] | D-ResNet-101101 | - | - | - | 46.8146.81 |
| SS-NAN [183] | ResNet-101101 | Pose | 87.5987.59 | 56.0356.03 | 47.9247.92 |
| MuLA [106] | Hourglass | Pose | 88.50\mathbf{88.50} | 60.5060.50 | 49.3049.30 |
| JPPNet [87] | D-ResNet-101101 | Pose | 86.3986.39 | 62.3262.32 | 51.3751.37 |
| CE2P [98] | D-ResNet-101101 | Edge | 87.3787.37 | 63.2063.20 | 53.1053.10 |
| HRNetV22 | HRNetV22-W4848 | N | 88.2188.21 | 67.4367.43 | 55.9055.90 |
| HRNetV22 + OCR [170] | HRNetV22-W4848 | N | 88.2488.24 | 67.84\mathbf{67.84} | 56.48\mathbf{56.48} |

## V Semantic Segmentation

Semantic segmentation is a problem of assigning a class label to each pixel. Some example results by our approach are given in Figure 7. We feed the input image to the HRNetV22 (Figure 4 (b)) and then pass the resulting 15​C15C-dimensional representation at each position to a linear classifier with the softmax loss to predict the segmentation maps. The segmentation maps are upsampled (44 times) to the input size by bilinear upsampling for both training and testing. We report the results over two scene parsing datasets, PASCAL-Context [103] and Cityscapes [28], and a human parsing dataset, LIP [47]. The mean of class-wise intersection over union (mIoU) is adopted as the evaluation metric.

|  | Faster R-CNN [53] | Cascade R-CNN [13] | FCOS [136] | CenterNet [36] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | R-5050 | H-1818 | R-101101 | H-3232 | X-101101 | H-4848 | R-5050 | H-1818 | R-101101 | H-3232 | X-101101 | H-4848 | R-5050 | H-1818 | R-101101 | H-3232 | HG-5252 | H-4848 | HG-104104 | H-6464 |
| #param. (M) | 39.839.8 | 26.226.2 | 57.857.8 | 45.045.0 | 94.994.9 | 79.479.4 | 69.469.4 | 55.155.1 | 88.488.4 | 74.974.9 | 127.3127.3 | 111.0111.0 | 32.032.0 | 17.517.5 | 51.051.0 | 37.337.3 | 104.8104.8 | 73.673.6 | 210.1210.1 | 127.7127.7 |
| GFLOPs | 172.3172.3 | 159.1159.1 | 239.4239.4 | 245.3245.3 | 381.8381.8 | 399.1399.1 | 226.2226.2 | 207.8207.8 | 298.7298.7 | 300.8300.8 | 448.3448.3 | 466.5466.5 | 190.0190.0 | 180.3180.3 | 261.2261.2 | 273.3273.3 | 227.0227.0 | 217.1217.1 | 388.4388.4 | 318.5318.5 |
|  | Cascade Mask R-CNN [13] | Hybrid Task Cascade [16] | Mask R-CNN [53] |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  | R-5050 | H-1818 | R-101101 | H-3232 | X-101101 | H-4848 | R-5050 | H-1818 | R-101101 | H-3232 | X-101101 | H-4848 | R-5050 | H-1818 | R-101101 | H-3232 |  |  |  |  |
| #param. (M) | 77.377.3 | 63.163.1 | 96.396.3 | 82.982.9 | 135.2135.2 | 118.9118.9 | 80.380.3 | 66.166.1 | 99.399.3 | 85.985.9 | 138.2138.2 | 121.9121.9 | 44.444.4 | 30.130.1 | 63.463.4 | 49.949.9 |  |  |  |  |
| GFLOPs | 431.7431.7 | 413.1413.1 | 504.1504.1 | 506.2506.2 | 653.7653.7 | 671.9671.9 | 476.9476.9 | 458.3458.3 | 549.2549.2 | 551.4551.4 | 698.9698.9 | 717.0717.0 | 266.5266.5 | 247.9247.9 | 338.8338.8 | 341.0341.0 |  |  |  |  |

| backbone | LS | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|
| Faster R-CNN [92] |  |  |  |  |  |  |  |
| ResNet-5050-FPN | 1×1\times | 36.736.7 | 58.358.3 | 39.939.9 | 20.920.9 | 39.839.8 | 47.947.9 |
| HRNetV22p-W1818 | 1×1\times | 36.236.2 | 57.357.3 | 39.339.3 | 20.720.7 | 39.039.0 | 46.846.8 |
| ResNet-5050-FPN | 2×2\times | 37.637.6 | 58.758.7 | 41.341.3 | 21.421.4 | 40.8{40.8} | 49.7{49.7} |
| HRNetV22p-W1818 | 2×2\times | 38.0{38.0} | 58.9{58.9} | 41.5{41.5} | 22.6{22.6} | 40.8{40.8} | 49.649.6 |
| ResNet-101101-FPN | 1×1\times | 39.239.2 | 61.161.1 | 43.043.0 | 22.322.3 | 42.942.9 | 50.950.9 |
| HRNetV22p-W3232 | 1×1\times | 39.639.6 | 61.061.0 | 43.343.3 | 23.723.7 | 42.542.5 | 50.550.5 |
| ResNet-101101-FPN | 2×2\times | 39.839.8 | 61.461.4 | 43.443.4 | 22.922.9 | 43.643.6 | 52.452.4 |
| HRNetV22p-W3232 | 2×2\times | 40.9{40.9} | 61.8{61.8} | 44.8{44.8} | 24.4{24.4} | 43.7{43.7} | 53.3{53.3} |
| X-101101-6464×\times44d-FPN | 1×1\times | 41.341.3 | 63.4{63.4} | 45.245.2 | 24.524.5 | 45.8{45.8} | 53.353.3 |
| HRNetV22p-W4848 | 1×1\times | 41.341.3 | 62.862.8 | 45.145.1 | 25.1{25.1} | 44.544.5 | 52.952.9 |
| X-101101-6464×\times44d-FPN | 2×2\times | 40.840.8 | 62.162.1 | 44.644.6 | 23.223.2 | 44.544.5 | 53.753.7 |
| HRNetV22p-W4848 | 2×2\times | 41.8{41.8} | 62.862.8 | 45.9{45.9} | 25.025.0 | 44.744.7 | 54.6{54.6} |
| Cascade R-CNN [13] |  |  |  |  |  |  |  |
| ResNet-5050-FPN | 20​e20e | 41.141.1 | 59.159.1 | 44.844.8 | 22.522.5 | 44.4{44.4} | 54.9{54.9} |
| HRNetV22p-W1818 | 20​e20e | 41.3{41.3} | 59.2{59.2} | 44.9{44.9} | 23.7{23.7} | 44.244.2 | 54.154.1 |
| ResNet-101101-FPN | 20​e20e | 42.542.5 | 60.760.7 | 46.346.3 | 23.723.7 | 46.146.1 | 56.956.9 |
| HRNetV22p-W3232 | 20​e20e | 43.7{43.7} | 61.7{61.7} | 47.7{47.7} | 25.6{25.6} | 46.5{46.5} | 57.4{57.4} |
| X-101101-6464×\times44d-FPN | 20​e20e | 44.7{44.7} | 63.1{63.1} | 49.0{49.0} | 25.825.8 | 48.3{48.3} | 58.8{58.8} |
| HRNetV22p-W4848 | 20​e20e | 44.644.6 | 62.762.7 | 48.748.7 | 26.3{26.3} | 48.148.1 | 58.558.5 |

| backbone | LS | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|
| FCOS [136] |  |  |  |  |  |  |  |
| ResNet-5050-FPN | 2×2\times | 37.137.1 | 55.9{55.9} | 39.839.8 | 21.321.3 | 41.0{41.0} | 47.847.8 |
| HRNetV22p-W1818 | 2×2\times | 37.7{37.7} | 55.355.3 | 40.2{40.2} | 22.0{22.0} | 40.840.8 | 48.8{48.8} |
| ResNet-101101-FPN | 2×2\times | 41.441.4 | 60.3{60.3} | 44.844.8 | 25.025.0 | 45.6{45.6} | 53.153.1 |
| HRNetV22p-W3232 | 2×2\times | 41.9{41.9} | 60.3{60.3} | 45.0{45.0} | 25.1{25.1} | 45.6{45.6} | 53.2{53.2} |
| CenterNet [36] |  |  |  |  |  |  |  |
| Hourglass-5252 | - | 41.341.3 | 59.259.2 | 43.943.9 | 23.623.6 | 43.843.8 | 55.855.8 |
| HRNetV22p-W4848 | - | 43.4{43.4} | 61.8{61.8} | 45.6{45.6} | 23.8{23.8} | 47.1{47.1} | 59.3{59.3} |
| Hourglass-104104 | - | 44.8{44.8} | 62.462.4 | 48.2{48.2} | 25.9{25.9} | 48.9{48.9} | 58.858.8 |
| HRNetV22p-W6464 | - | 44.044.0 | 62.5{62.5} | 47.347.3 | 23.923.9 | 48.248.2 | 60.2{60.2} |

| backbone | LS | mask | bbox |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| AP | APS | APM | APL | AP | APS | APM | APL |  |  |
| Mask R-CNN [53] |  |  |  |  |  |  |  |  |  |
| ResNet-5050-FPN | 1×1\times | 34.234.2 | 15.715.7 | 36.836.8 | 50.250.2 | 37.837.8 | 22.122.1 | 40.940.9 | 49.349.3 |
| HRNetV22p-W1818 | 1×1\times | 33.833.8 | 15.615.6 | 35.635.6 | 49.849.8 | 37.137.1 | 21.921.9 | 39.539.5 | 47.947.9 |
| ResNet-5050-FPN | 2×2\times | 35.035.0 | 16.016.0 | 37.5{37.5} | 52.0{52.0} | 38.638.6 | 21.721.7 | 41.641.6 | 50.950.9 |
| HRNetV22p-W1818 | 2×2\times | 35.3{35.3} | 16.9{16.9} | 37.5{37.5} | 51.851.8 | 39.2{39.2} | 23.7{23.7} | 41.7{41.7} | 51.0{51.0} |
| ResNet-101101-FPN | 1×1\times | 36.136.1 | 16.216.2 | 39.039.0 | 53.053.0 | 40.040.0 | 22.622.6 | 43.443.4 | 52.352.3 |
| HRNetV22p-W3232 | 1×1\times | 36.736.7 | 17.317.3 | 39.039.0 | 53.053.0 | 40.940.9 | 24.524.5 | 43.943.9 | 52.252.2 |
| ResNet-101101-FPN | 2×2\times | 36.736.7 | 17.017.0 | 39.539.5 | 54.854.8 | 41.041.0 | 23.423.4 | 44.444.4 | 53.953.9 |
| HRNetV22p-W3232 | 2×2\times | 37.6{37.6} | 17.8{17.8} | 40.0{40.0} | 55.0{55.0} | 42.3{42.3} | 25.0{25.0} | 45.4{45.4} | 54.9{54.9} |
| Cascade Mask R-CNN [13] |  |  |  |  |  |  |  |  |  |
| ResNet-5050-FPN | 20​e20e | 36.6{36.6} | 19.0{19.0} | 37.437.4 | 50.750.7 | 42.3{42.3} | 23.723.7 | 45.7{45.7} | 56.4{56.4} |
| HRNetV22p-W1818 | 20​e20e | 36.436.4 | 17.017.0 | 38.6{38.6} | 52.9{52.9} | 41.941.9 | 23.8{23.8} | 44.944.9 | 55.055.0 |
| ResNet-101101-FPN | 20​e20e | 37.637.6 | 19.7{19.7} | 40.840.8 | 52.452.4 | 43.343.3 | 24.424.4 | 46.946.9 | 58.058.0 |
| HRNetV22p-W3232 | 20​e20e | 38.5{38.5} | 18.918.9 | 41.1{41.1} | 56.1{56.1} | 44.5{44.5} | 26.1{26.1} | 47.9{47.9} | 58.5{58.5} |
| X-101101-6464×\times44d-FPN | 20​e20e | 39.439.4 | 20.8{20.8} | 42.7{42.7} | 54.154.1 | 45.745.7 | 26.226.2 | 49.6{49.6} | 60.060.0 |
| HRNetV22p-W4848 | 20​e20e | 39.5{39.5} | 19.719.7 | 41.841.8 | 56.9{56.9} | 46.0{46.0} | 27.5{27.5} | 48.948.9 | 60.1{60.1} |
| Hybrid Task Cascade [16] |  |  |  |  |  |  |  |  |  |
| ResNet-5050-FPN | 20​e20e | 38.1{38.1} | 20.3{20.3} | 41.1{41.1} | 52.852.8 | 43.2{43.2} | 24.924.9 | 46.4{46.4} | 57.8{57.8} |
| HRNetV22p-W1818 | 20​e20e | 37.937.9 | 18.818.8 | 39.939.9 | 55.2{55.2} | 43.143.1 | 26.6{26.6} | 46.046.0 | 56.956.9 |
| ResNet-101101-FPN | 20​e20e | 39.439.4 | 21.4{21.4} | 42.4{42.4} | 54.454.4 | 44.944.9 | 26.426.4 | 48.348.3 | 59.9{59.9} |
| HRNetV22p-W3232 | 20​e20e | 39.6{39.6} | 19.119.1 | 42.042.0 | 57.9{57.9} | 45.3{45.3} | 27.0{27.0} | 48.4{48.4} | 59.559.5 |
| X-101101-6464×\times44d-FPN | 20​e20e | 40.8{40.8} | 22.7{22.7} | 44.2{44.2} | 56.356.3 | 46.9{46.9} | 28.0{28.0} | 50.7{50.7} | 62.1{62.1} |
| HRNetV22p-W4848 | 20​e20e | 40.740.7 | 19.719.7 | 43.443.4 | 59.359.3 | 46.846.8 | 28.0{28.0} | 50.250.2 | 61.761.7 |
| X-101101-6464×\times44d-FPN | 28​e28e | 40.740.7 | 20.020.0 | 44.1{44.1} | 59.9{59.9} | 46.846.8 | 27.527.5 | 51.0{51.0} | 61.761.7 |
| HRNetV22p-W4848 | 28​e28e | 41.0{41.0} | 20.8{20.8} | 43.943.9 | 59.9{59.9} | 47.0{47.0} | 28.8{28.8} | 50.350.3 | 62.2{62.2} |

Cityscapes. The Cityscapes dataset [28] contains 5,0005,000 high quality pixel-level finely annotated scene images. The finely-annotated images are divided into 2,975/500/1,5252,975/500/1,525 images for training, validation and testing. There are 3030 classes, and 1919 classes among them are used for evaluation. In addition to the mean of class-wise intersection over union (mIoU), we report other three scores on the test set: IoU category (cat.), iIoU class (cla.) and iIoU category (cat.).

We follow the same training protocol [181, 182]. The data are augmented by random cropping (from 1024×20481024\times 2048 to 512×1024512\times 1024), random scaling in the range of [0.5,2][0.5,2], and random horizontal flipping. We use the SGD optimizer with the base learning rate of 0.010.01, the momentum of 0.90.9 and the weight decay of 0.00050.0005. The poly learning rate policy with the power of 0.90.9 is used for dropping the learning rate. All the models are trained for 120​K120K iterations with the batch size of 1212 on 44 GPUs and syncBN.

Table III provides the comparison with several representative methods on the Cityscapes val set in terms of parameter and computation complexity and mIoU class. (i) HRNetV22-W4040 (4040 indicates the width of the high-resolution convolution), with similar model size to DeepLabv33+ and much lower computation complexity, gets better performance: 4.74.7 points gain over UNet++, 1.71.7 points gain over DeepLabv3 and about 0.50.5 points gain over PSPNet, DeepLabv3+. (ii) HRNetV22-W4848, with similar model size to PSPNet and much lower computation complexity, achieves much significant improvement: 5.65.6 points gain over UNet++, 2.62.6 points gain over DeepLabv3 and about 1.41.4 points gain over PSPNet, DeepLabv3+. In the following comparisons, we adopt HRNetV22-W4848 that is pretrained on ImageNet and has similar model size as most Dilated-ResNet-101101 based methods.

Table IV provides the comparison of our method with state-of-the-art methods on the Cityscapes test set. All the results are with six scales and flipping. Two cases w/o using coarse data are evaluated: One is about the model learned on the train set, and the other is about the model learned on the train+val set. In both cases, HRNetV22-W4848 achieves the superior performance.

PASCAL-Context. The PASCAL-Context dataset [103] includes 4,9984,998 scene images for training and 5,1055,105 images for testing with 5959 semantic labels and 11 background label.

The data augmentation and learning rate policy are the same as Cityscapes. Following the widely-used training strategy [172, 32], we resize the images to 480×480480\times 480 and set the initial learning rate to 0.0040.004 and weight decay to 0.00010.0001. The batch size is 1616 and the number of iterations is 60​K60K.

We follow the standard testing procedure [172, 32]. The image is resized to 480×480480\times 480 and then fed into our network. The resulting 480×480480\times 480 label maps are then resized to the original image size. We evaluate the performance of our approach and other approaches using six scales and flipping.

Table V provides the comparison of our method with state-of-the-art methods. There are two kinds of evaluation schemes: mIoU over 5959 classes and 6060 classes (5959 classes + background). In both cases, HRNetV22-W4848 achieves state-of-the-art results except that the result from [51] is higher than ours without using the OCR scheme [170].

LIP. The LIP dataset [47] contains 50,46250,462 elaborately annotated human images, which are divided into 30,46230,462 training images, and 10,00010,000 validation images. The methods are evaluated on 2020 categories (1919 human part labels and 11 background label). Following the standard training and testing settings [98], the images are resized to 473×473473\times 473 and the performance is evaluated on the average of the segmentation maps of the original and flipped images.

The data augmentation and learning rate policy are the same as Cityscapes. The training strategy follows the recent setting [98]. We set the initial learning rate to 0.0070.007 and the momentum to 0.90.9 and the weight decay to 0.00050.0005. The batch size is 4040 and the number of iterations is 110110K.

Table VI provides the comparison of our method with state-of-the-art methods. The overall performance of HRNetV22-W4848 performs the best with fewer parameters and lighter computation cost. We also would like to mention that our networks do not use extra information such as pose or edge.

|  | backbone | size | LS | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|---|---|
| MLKP [142] | VGG1616 | - | - | 28.628.6 | 52.452.4 | 31.631.6 | 10.810.8 | 33.433.4 | 45.145.1 |
| STDN [187] | DenseNet-169169 | 513513 | - | 31.831.8 | 51.051.0 | 33.633.6 | 14.414.4 | 36.136.1 | 43.443.4 |
| DES [179] | VGG1616 | 512512 | - | 32.832.8 | 53.253.2 | 34.634.6 | 13.913.9 | 36.036.0 | 47.647.6 |
| CoupleNet [194] | ResNet-101101 | - | - | 33.133.1 | 53.553.5 | 35.435.4 | 11.611.6 | 36.336.3 | 50.150.1 |
| DeNet [139] | ResNet-101101 | 512512 | - | 33.833.8 | 53.453.4 | 36.136.1 | 12.312.3 | 36.136.1 | 50.850.8 |
| RFBNet [96] | VGG1616 | 512512 | - | 34.434.4 | 55.755.7 | 36.436.4 | 17.617.6 | 37.037.0 | 47.647.6 |
| DFPR [74] | ResNet-101101 | 512512 | 1×1\times | 34.634.6 | 54.354.3 | 37.337.3 | - | - | - |
| PFPNet [70] | VGG1616 | 512512 | - | 35.235.2 | 57.657.6 | 37.937.9 | 18.718.7 | 38.638.6 | 45.945.9 |
| RefineDet[177] | ResNet-101101 | 512512 | - | 36.436.4 | 57.557.5 | 39.539.5 | 16.616.6 | 39.939.9 | 51.451.4 |
| Relation Net [56] | ResNet-101101 | 600600 | - | 39.039.0 | 58.658.6 | 42.942.9 | - | - | - |
| C-FRCNN [25] | ResNet-101101 | 800800 | 1×1\times | 39.039.0 | 59.759.7 | 42.842.8 | 19.419.4 | 42.442.4 | 53.053.0 |
| RetinaNet [93] | ResNet-101101-FPN | 800800 | 1.5×1.5\times | 39.139.1 | 59.159.1 | 42.342.3 | 21.821.8 | 42.742.7 | 50.250.2 |
| Deep Regionlets [160] | ResNet-101101 | 800800 | 1.5×1.5\times | 39.339.3 | 59.859.8 | - | 21.721.7 | 43.743.7 | 50.950.9 |
| FitnessNMS [140] | ResNet-101101 | 768768 | - | 39.539.5 | 58.058.0 | 42.642.6 | 18.918.9 | 43.543.5 | 54.154.1 |
| DetNet [86] | DetNet5959-FPN | 800800 | 2×2\times | 40.340.3 | 62.162.1 | 43.843.8 | 23.623.6 | 42.642.6 | 50.050.0 |
| CornerNet [79] | Hourglass-104104 | 511511 | - | 40.540.5 | 56.556.5 | 43.143.1 | 19.419.4 | 42.742.7 | 53.953.9 |
| M2Det [185] | VGG1616 | 800800 | ∼10×\sim 10\times | 41.041.0 | 59.759.7 | 45.045.0 | 22.122.1 | 46.5{46.5} | 53.853.8 |
| Faster R-CNN [92] | ResNet-101101-FPN | 800800 | 1×1\times | 39.339.3 | 61.361.3 | 42.742.7 | 22.122.1 | 42.142.1 | 49.749.7 |
| Faster R-CNN | HRNetV22p-W3232 | 800800 | 1×1\times | 39.539.5 | 61.261.2 | 43.043.0 | 23.323.3 | 41.741.7 | 49.149.1 |
| Faster R-CNN [92] | ResNet-101101-FPN | 800800 | 2×2\times | 40.340.3 | 61.861.8 | 43.943.9 | 22.622.6 | 43.143.1 | 51.051.0 |
| Faster R-CNN | HRNetV22p-W3232 | 800800 | 2×2\times | 41.141.1 | 62.362.3 | 44.944.9 | 24.024.0 | 43.143.1 | 51.451.4 |
| Faster R-CNN [92] | ResNet-152152-FPN | 800800 | 2×2\times | 40.640.6 | 62.162.1 | 44.344.3 | 22.622.6 | 43.443.4 | 52.052.0 |
| Faster R-CNN | HRNetV22p-W4040 | 800800 | 2×2\times | 42.142.1 | 63.263.2 | 46.146.1 | 24.624.6 | 44.544.5 | 52.652.6 |
| Faster R-CNN [17] | X-101101-6464×4\times 4d-FPN | 800800 | 2×2\times | 41.141.1 | 62.862.8 | 44.844.8 | 23.523.5 | 44.144.1 | 52.352.3 |
| Faster R-CNN | HRNetV22p-W4848 | 800800 | 2×2\times | 42.442.4 | 63.6{63.6} | 46.446.4 | 24.924.9 | 44.644.6 | 53.053.0 |
| Cascade R-CNN [12]∗ | ResNet-101101-FPN | 800800 | ∼1.6×\sim 1.6\times | 42.842.8 | 62.162.1 | 46.346.3 | 23.723.7 | 45.545.5 | 55.255.2 |
| Cascade R-CNN | ResNet-101101-FPN | 800800 | ∼1.6×\sim 1.6\times | 43.143.1 | 61.761.7 | 46.746.7 | 24.124.1 | 45.945.9 | 55.055.0 |
| Cascade R-CNN | HRNetV22p-W3232 | 800800 | ∼1.6×\sim 1.6\times | 43.743.7 | 62.062.0 | 47.447.4 | 25.525.5 | 46.046.0 | 55.355.3 |
| Cascade R-CNN | X-101101-64×464\times 4d-FPN | 800800 | ∼1.6×\sim 1.6\times | 44.944.9 | 63.763.7 | 48.948.9 | 25.925.9 | 47.747.7 | 57.157.1 |
| Cascade R-CNN | HRNetV22p-W4848 | 800800 | ∼1.6×\sim 1.6\times | 44.844.8 | 63.163.1 | 48.648.6 | 26.026.0 | 47.347.3 | 56.356.3 |
| FCOS [136] | ResNet-5050-FPN | 800800 | 2×2\times | 37.337.3 | 56.456.4 | 39.739.7 | 20.420.4 | 39.639.6 | 47.547.5 |
| FCOS | HRNetV22p-W1818 | 800800 | 2×2\times | 37.837.8 | 56.156.1 | 40.440.4 | 21.621.6 | 39.839.8 | 47.447.4 |
| FCOS [136] | ResNet-101101-FPN | 800800 | 2×2\times | 39.239.2 | 58.858.8 | 41.641.6 | 21.821.8 | 41.741.7 | 50.050.0 |
| FCOS | HRNetV22p-W3232 | 800800 | 2×2\times | 40.540.5 | 59.359.3 | 43.343.3 | 23.423.4 | 42.642.6 | 51.051.0 |
| CenterNet[36] | Hourglass-5252 | 511511 | −- | 41.641.6 | 59.459.4 | 44.244.2 | 22.522.5 | 43.143.1 | 54.154.1 |
| CenterNet | HRNetV22-W4848 | 511511 | −- | 43.543.5 | 62.162.1 | 46.546.5 | 22.222.2 | 46.5{46.5} | 57.8{57.8} |
| Cascade Mask R-CNN [13] | ResNet-101101-FPN | 800800 | ∼1.6×\sim 1.6\times | 44.044.0 | 62.362.3 | 47.947.9 | 24.324.3 | 46.946.9 | 56.756.7 |
| Cascade Mask R-CNN | HRNetV22p-W3232 | 800800 | ∼1.6×\sim 1.6\times | 44.744.7 | 62.562.5 | 48.648.6 | 25.825.8 | 47.147.1 | 56.356.3 |
| Cascade Mask R-CNN [13] | X-101101-64×464\times 4d-FPN | 800800 | ∼1.6×\sim 1.6\times | 45.945.9 | 64.564.5 | 50.050.0 | 26.626.6 | 49.049.0 | 58.658.6 |
| Cascade Mask R-CNN | HRNetV22p-W4848 | 800800 | ∼1.6×\sim 1.6\times | 46.146.1 | 64.064.0 | 50.350.3 | 27.127.1 | 48.648.6 | 58.358.3 |
| Hybrid Task Cascade [16] | ResNet-101101-FPN | 800800 | ∼1.6×\sim 1.6\times | 45.145.1 | 64.364.3 | 49.049.0 | 25.225.2 | 48.048.0 | 58.258.2 |
| Hybrid Task Cascade | HRNetV22p-W3232 | 800800 | ∼1.6×\sim 1.6\times | 45.645.6 | 64.164.1 | 49.449.4 | 26.726.7 | 47.747.7 | 58.058.0 |
| Hybrid Task Cascade [16] | X-101101-64×464\times 4d-FPN | 800800 | ∼1.6×\sim 1.6\times | 47.247.2 | 66.566.5 | 51.451.4 | 27.727.7 | 50.150.1 | 60.360.3 |
| Hybrid Task Cascade | HRNetV22p-W4848 | 800800 | ∼1.6×\sim 1.6\times | 47.047.0 | 65.865.8 | 51.051.0 | 27.927.9 | 49.449.4 | 59.759.7 |
| Hybrid Task Cascade [16] | X-101101-64×464\times 4d-FPN | 800800 | ∼2.3×\sim 2.3\times | 47.247.2 | 66.666.6 | 51.351.3 | 27.527.5 | 50.150.1 | 60.660.6 |
| Hybrid Task Cascade | HRNetV22p-W4848 | 800800 | ∼2.3×\sim 2.3\times | 47.347.3 | 65.965.9 | 51.251.2 | 28.028.0 | 49.749.7 | 59.859.8 |

## VI COCO Object Detection

We perform the evaluation on the MS COCO 20172017 detection dataset, which contains about 118118k images for training, 55k for validation (val) and ∼20\sim 20k testing without provided annotations (test-dev). The standard COCO-style evaluation is adopted. Some example results by our approach are given in Figure 8.

We apply our multi-level representations (HRNetV22p)44 4 Same as FPN [93], we also use 55 levels., shown in Figure 4 (c), for object detection. The data is augmented by standard horizontal flipping. The input images are resized such that the shorter edge is 800 pixels [92]. Inference is performed on a single image scale.

We compare our HRNet with the standard models: ResNet [54] and ResNeXt [156]. We evaluate the detection performance on COCO val. under two anchor-based frameworks: Faster R-CNN [118] and Cascade R-CNN [12], and two recently-developed anchor-free frameworks: FCOS [136] and CenterNet [36]. We train the Faster R-CNN and Cascade R-CNN models for both our HRNetV22p and the ResNet on the public MMDetection platform [17] with the provided training setup, except that we use the learning rate schedule suggested in [52] for 2×2\times, and FCOS [136] and CenterNet [36] from the implementations provided by the authors. Table VII summarizes #parameters and GFLOPs. Table VIII and Table IX report detection scores.

We also evaluate the performance of joint detection and instance segmentation, under three frameworks: Mask R-CNN [53], Cascade Mask R-CNN [13], and Hybrid Task Cascade [16]. The results are obtained on the public MMDetection platform [17] and are in Table X.

There are several observations. On the one hand, as shown in Tables VIII and IX, the overall object detection performance of HRNetV22 is better than ResNet under similar model size and computation complexity. In some cases, for 1×1\times, HRNetV2p-W1818 performs worse than ResNet-5050-FPN, which might come from insufficient optimization iterations. On the other hand, as shown in Table X, the overall object detection and instance segmentation performance is better than ResNet and ResNeXt. In particular, under the Hybrid Task Cascade framework, the HRNet performs slightly worse than ResNeXt-101101-6464×\times44d-FPN for 20​e20e, but better for 28​e28e. This implies that our HRNet benefits more from longer training.

Table XI reports the comparison of our network to state-of-the-art single-model object detectors on COCO test-dev without using multi-scale training and multi-scale testing that are done in [97, 115, 85, 128, 127, 110]. In the Faster R-CNN framework, our networks perform better than ResNets with similar parameter and computation complexity: HRNetV22p-W3232 vs. ResNet-101101-FPN, HRNetV22p-W4040 vs. ResNet-152152-FPN, HRNetV22p-W4848 vs. X-101101-64×464\times 4d-FPN. In the Cascade R-CNN and CenterNet framework, our HRNetV22 also performs better. In the Cascade Mask R-CNN and Hybrid Task Cascade frameworks, the HRNet gets the overall better performance.

## VII Ablation Study

We perform the ablation study for the components in HRNet over two tasks: human pose estimation on COCO validation and semantic segmentation on Cityscapes validation. We mainly use HRNetV1-W3232 for human pose estimation, and HRNetV2-W4848 for semantic segmentation. All results of pose estimation are obtained over the input size 256×192256\times 192. We also present the results for comparing HRNetV11 and HRNetV22.

Representations of different resolutions. We study how the representation resolution affects the pose estimation performance by checking the quality of the heatmap estimated from the feature maps of each resolution from high to low.

We train two HRNetV11 networks initialized by the model pretrained for the ImageNet classification. Our network outputs four response maps from high-to-low resolutions. The quality of heatmap prediction over the lowest-resolution response map is too low and the AP score is below 1010 points. The AP scores over the other three maps are reported in Figure 9. The comparison implies that the resolution does impact the keypoint prediction quality.

| Method | Final | Across | Within | Pose (AP)(\operatorname{AP}) | Segmentation (mIoU)(\operatorname{mIoU}) |
|---|---|---|---|---|---|
| (a) | ✓ |  |  | 70.870.8 | 74.874.8 |
| (b) | ✓ | ✓ |  | 71.971.9 | 75.475.4 |
| (c) | ✓ | ✓ | ✓ | 73.473.4 | 76.476.4 |

Repeated multi-resolution fusion. We empirically analyze the effect of the repeated multi-resolution fusion. We study three variants of our network. (a) W/o intermediate fusion units (11 fusion): There is no fusion between multi-resolution streams except the final fusion unit. (b) W/ across-stage fusion units (33 fusions): There is no fusion between parallel streams within each stage. (c) W/ both across-stage and within-stage fusion units (totally 88 fusions): This is our proposed method. All the networks are trained from scratch. The results on COCO human pose estimation and Cityscapes semantic segmentation (validation) given in Table XII show that the multi-resolution fusion unit is helpful and more fusions lead to better performance.

We also study other possible choices for the fusion design: (i) use bilinear downsample to replace strided convolutions, and (ii) use the multiplication operation to replace the sum operation. In the former case, the COCO pose estimation AP score and the Cityscapes segmentation mIoU score are reduced to 72.672.6 and 74.274.2. The reason is that downsampling reduces the volume size (width ×\times height ×\times #channels) of the representation maps, and strided convolutions learn better volume size reduction than bilinear downsampling. In the later case, the results are much worse: 54.754.7 and 66.066.0, respectively. The possible reason might be that multiplication increases the training difficulty as pointed in [145].

Resolution maintenance. We study the performance of a variant of the HRNet: all the four high-to-low resolution streams are added at the beginning and the depths of the four streams are the same; the fusion schemes are the same to ours. Both the HRNets and the variants (with similar #Params and GFLOPs) are trained from scratch.

The human pose estimation performance (AP) on COCO val for the variant is 72.572.5, which is lower than 73.473.4 for HRNetV11-W3232. The segmentation performance (mIoU) on Cityscapes val for the variant is 75.775.7, which is lower than 76.476.4 for HRNetV22-W4848. We believe that the reason is that the low-level features extracted from the early stages over the low-resolution streams are less helpful. In addition, another simple variant, only the high-resolution stream of similar #parameters and GFLOPs without low-resolution parallel streams shows much lower performance on COCO and Cityscapes.

V11 vs. V22. We compare HRNetV22 and HRNetV22p, to HRNetV11 on pose estimation, semantic segmentation and COCO object detection. For human pose estimation, the performance is similar. For example, HRNetV22-W3232 (w/o ImageNet pretraining) achieves the AP score 73.673.6, which is slightly higher than 73.473.4 HRNetV11-W3232.

The segmentation and object detection results, given in Figure 10 (a) and Figure 10 (b), imply that HRNetV22 outperforms HRNetV11 significantly, except that the gain is minor in the large model case (1×1\times) in segmentation for Cityscapes. We also test a variant (denoted by HRNetV11h), which is built by appending a 1×11\times 1 convolution to align the dimension of the output high-resolution representation with the dimension of HRNetV22. The results in Figure 10 (a) and Figure 10 (b) show that the variant achieves slight improvement to HRNetV11, implying that aggregating the representations from low-resolution parallel convolutions in our HRNetV22 is essential for improving the capability.

|  | Pose estimation | Segmentation | Detection |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
| SB-ResNet-152152 | HRNetV11-W4848 | PSPNet | DeepLabV33 | HRNetV22-W4848 | ResNet-101101 | ResNeXt-101101 | HRNetV22p-W3232 | HRNetV22p-W4848 |  |
| training memory | 14.814.8G | 7.37.3G | 14.414.4G | 13.313.3G | 13.913.9G | 5.45.4G | 9.59.5G | 8.58.5G | 11.311.3G |
| inference memory/image | 0.290.29G | 0.270.27G | 1.601.60G | 1.151.15G | 1.791.79G | 0.620.62G | 0.770.77G | 0.510.51G | 0.790.79G |
| training second/iteration | 1.0851.085 | 1.2311.231 | 0.8370.837 | 0.8500.850 | 0.6920.692 | 0.5500.550 | 1.1831.183 | 0.6900.690 | 0.9650.965 |
| inference second/image | 0.030​(0.012)0.030~(0.012) | 0.058​(0.017)0.058~(0.017) | 0.3970.397 | 0.4110.411 | 0.1500.150 | 0.0870.087 | 0.1440.144 | 0.1010.101 | 0.1160.116 |
| score | 72.072.0 | 75.175.1 | 79.779.7 | 78.578.5 | 81.181.1 | 39.839.8 | 40.840.8 | 40.940.9 | 41.841.8 |

## VIII Conclusions

In this paper, we present a high-resolution network for visual recognition problems. There are three fundamental differences from existing low-resolution classification networks and high-resolution representation learning networks: (i) Connect high and low resolution convolutions in parallel other than in series; (ii) Maintain high resolution through the whole process instead of recovering high resolution from low resolution; and (iii) Fuse multi-resolution representations repeatedly, rendering rich high-resolution representations with strong position sensitivity.

The superior results on a wide range of visual recognition problems suggest that our proposed HRNet is a stronger backbone for computer vision problems. Our research also encourages more research efforts for designing network architectures directly for specific vision problems other than extending, remediating or repairing representations learned from low-resolution networks (e.g., ResNet or VGGNet).

Discussions. There is a possible misunderstanding: the memory cost of the HRNet is larger as the resolution is higher. In fact, the memory cost of the HRNet for all the three applications, human pose estimation, semantic segmentation and object detection, is comparable to state-of-the-arts except that the training memory cost in object detection is a little larger.

In addition, we summarize the runtime cost comparison on the PyTorch 1.01.0 platform. The training and inference time cost of the HRNet is comparable to previous state-of-the-arts except that (1) the inference time of the HRNet for segmentation is much smaller and (2) the training time of the HRNet for pose estimation is a little larger, but the cost on the MXNet 1.5.11.5.1 platform, which supports static graph inference, is similar as SimpleBaseline. We would like to highlight that for semantic segmentation the inference cost is significantly smaller than PSPNet and DeepLabv33. Table XIII summarizes memory and time cost comparisons 55 5 The detailed comparisons are given in the supplementary file..

Future and followup works. We will study the combination of the HRNet with other techniques for semantic segmentation and instance segmentation. Currently, we have results (mIoU), which are depicted in Tables III IV V VI, by combining the HRNet with the object-contextual representation (OCR) scheme [170] 66 6 We empirically observed that the HRNet combined with ASPP [20] or PPM [181] did not get a performance improvement on Cityscape, but got a slight improvement on PASCAL-Context and LIP., a variant of object context [171, 59]. We will conduct the study by further increasing the resolution of the representation, e.g., to 12\frac{1}{2} or even a full resolution.

The applications of the HRNet are not limited to the above that we have done, and are suitable to other position-sensitive vision applications, such as facial landmark detection77 7 We provide the facial landmark detection results in the supplementary file., super-resolution, optical flow estimation, depth estimation, and so on. There are already followup works, e.g., image stylization [83], inpainting [50], image enhancement [62], image dehazing [1], temporal pose estimation [6], and drone object detection [190].

It is reported in [26] that a slightly-modified HRNet combined with ASPP achieved the best performance for Mapillary panoptic segmentation in the single model case. In the COCO + Mapillary Joint Recognition Challenge Workshop at ICCV 2019, the COCO DensePose challenge winner and almost all the COCO keypoint detection challenge participants adopted the HRNet. The OpenImage instance segmentation challenge winner (ICCV 2019) also used the HRNet.

## Appendix A Network Instantiation

Our current design (except the standard stem and the head,) contains four stages, as shown in Table XIV. Each stage consists of modularized blocks, repeated 11, 11, 44, and 33 times, respectively for the four stages. The modularized block consists of 11 (22, 33 and 44) branches for the 11st (22nd, 33rd and 44th) stages. Each branch corresponds to different resolution, and is compose of four residual units and one multi-resolution fusion unit (See Figure 3 in the main paper).

| Resolution | Stage 11 | Stage 22 | Stage 33 | Stage 44 |
|---|---|---|---|---|
| 4×4\times | [1×1,643×3,641×1,256]\left[\begin{array}[]{c}{1\times 1,64}\\[-0.81949pt] {3\times 3,64}\\[-0.81949pt] {1\times 1,256}\end{array}\right] ×\times4×14\times 1 | [3×3,C3×3,C]×4×1\left[\begin{array}[]{c}{3\times 3,C}\\[-0.81949pt] {3\times 3,C}\end{array}\right]\times 4\times 1 | [3×3,C3×3,C]×4×4\left[\begin{array}[]{c}{3\times 3,C}\\[-0.81949pt] {3\times 3,C}\end{array}\right]\times 4\times 4 | [3×3,C3×3,C]×4×3\left[\begin{array}[]{c}{3\times 3,C}\\[-0.81949pt] {3\times 3,C}\end{array}\right]\times 4\times 3 |
| 8×8\times |  | [3×3,2​C3×3,2​C]×4×1\left[\begin{array}[]{c}{3\times 3,2C}\\[-0.81949pt] {3\times 3,2C}\end{array}\right]\times 4\times 1 | [3×3,2​C3×3,2​C]×4×4\left[\begin{array}[]{c}{3\times 3,2C}\\[-0.81949pt] {3\times 3,2C}\end{array}\right]\times 4\times 4 | [3×3,2​C3×3,2​C]×4×3\left[\begin{array}[]{c}{3\times 3,2C}\\[-0.81949pt] {3\times 3,2C}\end{array}\right]\times 4\times 3 |
| 16×16\times |  |  | [3×3,4​C3×3,4​C]×4×4\left[\begin{array}[]{c}{3\times 3,4C}\\[-0.81949pt] {3\times 3,4C}\end{array}\right]\times 4\times 4 | [3×3,4​C3×3,4​C]×4×3\left[\begin{array}[]{c}{3\times 3,4C}\\[-0.81949pt] {3\times 3,4C}\end{array}\right]\times 4\times 3 |
| 32×32\times |  |  |  | [3×3,8​C3×3,8​C]×4×3\left[\begin{array}[]{c}{3\times 3,8C}\\[-0.81949pt] {3\times 3,8C}\end{array}\right]\times 4\times 3 |

## Appendix B Network Pretraining

We pretrain our network, which is augmented by a classification head shown in Figure 11, on ImageNet [120]. The classification head is described as below. First, the four-resolution feature maps are fed into a bottleneck and the output channels are increased from CC, 2​C2C, 4​C4C, and 8​C8C to 128128, 256256, 512512, and 10241024, respectively. Then, we downsample the high-resolution representation by a 22-strided 3×33\times 3 convolution outputting 256256 channels and add it to the representation of the second-high-resolution. This process is repeated two times to get 10241024 feature channels over the small resolution. Last, we transform the 10241024 channels to 20482048 channels through a 1×11\times 1 convolution, followed by a global average pooling operation. The output 20482048-dimensional representation is fed into the classifier.

|  | #Params. | GFLOPs | top-1 err. | top-5 err. |
|---|---|---|---|---|
| Residual branch formed by two 3×33\times 3 convolutions |  |  |  |  |
| ResNet-3838 | 28.328.3M | 3.803.80 | 24.6%24.6\% | 7.4%7.4\% |
| HRNet-W1818-C | 21.321.3M | 3.993.99 | 23.1%\mathbf{23.1\%} | 6.5%\mathbf{6.5\%} |
| ResNet-7272 | 48.448.4M | 7.467.46 | 23.3%23.3\% | 6.7%6.7\% |
| HRNet-W3030-C | 37.737.7M | 7.557.55 | 21.9%\mathbf{21.9\%} | 5.9%\mathbf{5.9\%} |
| ResNet-106106 | 64.964.9M | 11.111.1 | 22.7%22.7\% | 6.4%6.4\% |
| HRNet-W4040-C | 57.657.6M | 11.811.8 | 21.1%\mathbf{21.1\%} | 5.6%\mathbf{5.6\%} |
| Residual branch formed by a bottleneck |  |  |  |  |
| ResNet-5050 | 25.625.6M | 3.823.82 | 23.3%23.3\% | 6.6%6.6\% |
| HRNet-W4444-C | 21.921.9M | 3.903.90 | 23.0%\mathbf{23.0\%} | 6.5%\mathbf{6.5\%} |
| ResNet-101101 | 44.644.6M | 7.307.30 | 21.6%21.6\% | 5.8%\mathbf{5.8\%} |
| HRNet-W7676-C | 40.840.8M | 7.307.30 | 21.5%\mathbf{21.5\%} | 5.8%\mathbf{5.8\%} |
| ResNet-152152 | 60.260.2M | 10.710.7 | 21.2%21.2\% | 5.7%5.7\% |
| HRNet-W9696-C | 57.557.5M | 10.210.2 | 21.0%\mathbf{21.0\%} | 5.6%\mathbf{5.6\%} |

We adopt the same data augmentation scheme for training images as in [54], and train our models for 100100 epochs with a batch size of 256256. The initial learning rate is set to 0.10.1 and is reduced by 1010 times at epoch 3030, 6060 and 9090. We use SGD with a weight decay of 0.00010.0001 and a Nesterov momentum of 0.90.9. We adopt standard single-crop testing, so that 224×224224\times 224 pixels are cropped from each image. The top-11 and top-55 error are reported on the validation set.

Table XV shows our ImageNet classification results. As a comparison, we also report the results of ResNets. We consider two types of residual units: One is formed by a bottleneck, and the other is formed by two 3×33\times 3 convolutions. We follow the PyTorch implementation of ResNets and replace the 7×77\times 7 convolution in the input stem with two 22-strided 3×33\times 3 convolutions decreasing the resolution to 1/41/4 as in our networks. When the residual units are formed by two 3×33\times 3 convolutions, an extra bottleneck is used to increase the dimension of output feature maps from 512512 to 20482048. One can see that under similar #parameters and GFLOPs, our results are comparable to and slightly better than ResNets.

In addition, we look at the results of two alternative schemes: (i) the feature maps on each resolution go through a global pooling separately and then are concatenated together to output a 15​C15C-dimensional representation vector, named HRNet-Wxx-Ci; (ii) the feature maps on each resolution are fed into several 22-strided residual units (bottleneck, each dimension is increased to the double) to increase the dimension to 512512, and concatenate and average-pool them together to reach a 20482048-dimensional representation vector, named HRNet-Wxx-Cii, which is used in [130]. Table XVI shows such an ablation study. One can see that the proposed manner is superior to the two alternatives.

|  | #Params. | GFLOPs | top-1 err. | top-5 err. |
|---|---|---|---|---|
| HRNet-W2727-Ci | 21.421.4M | 5.555.55 | 26.0%26.0\% | 7.7%7.7\% |
| HRNet-W2525-Cii | 21.721.7M | 5.045.04 | 24.1%24.1\% | 7.1%7.1\% |
| HRNet-W1818-C | 21.321.3M | 3.993.99 | 23.1%\mathbf{23.1}\% | 6.5%\mathbf{6.5\%} |
| HRNet-W3636-Ci | 37.537.5M | 9.009.00 | 24.3%24.3\% | 7.3%7.3\% |
| HRNet-W3434-Cii | 36.736.7M | 8.298.29 | 22.8%22.8\% | 6.3%6.3\% |
| HRNet-W3030-C | 37.737.7M | 7.557.55 | 21.9%\mathbf{21.9\%} | 5.9%\mathbf{5.9\%} |
| HRNet-W4545-Ci | 58.258.2M | 13.413.4 | 23.6%23.6\% | 7.0%7.0\% |
| HRNet-W4343-Cii | 56.356.3 M | 12.512.5 | 22.2%22.2\% | 6.1%6.1\% |
| HRNet-W4040-C | 57.657.6M | 11.811.8 | 21.1%\mathbf{21.1\%} | 5.6%\mathbf{5.6\%} |

## Appendix C Training/Inference Cost

Tables XVII, XVIII and XIX provide GPU memory comparisons between HRNets and other standard networks for both training and inference in the PyTorch platform. Compared to state-of-the-arts for human pose estimation, the training and inference memory costs of the HRNet are similar or lower for similar parameter complexity (Table XVII). Compared to state-of-the-arts for semantic segmentation, the training and inference memory costs are similar (Table XVIII) for similar parameter complexity. Compared to state-of-the-arts for object detection for similar parameter complexity, the training and inference memory costs are similar or slightly higher (Table XIX).

In addition, we provide the runtime cost comparison. (1) For semantic segmentation, the time cost of the HRNet for training is slightly smaller and for inference significantly smaller than PSPNet and DeepLabv3 (Table XVIII). (2) For object detection, the time cost of the HRNet for training is larger than ResNet based networks and smaller than ResNext based networks, and for inference the HRNet is smaller for similar GFLOPs (Table XIX). (3) For human pose estimation, the time cost of the HRNet for training is similar and for inference larger; and the time cost of the HRNet for training and inference in the MXNet platform is similar as SimpleBaseline (Table XVII).

| backbone | GFLOPs | #params | train sec./iter. | train mem. | infer sec./batch | infer mem./batch | AP\operatorname{AP} |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b​sbs = 1 | b​sbs = 4 | b​sbs = 8 | b​sbs = 16 | b​sbs = 1 | b​sbs = 4 | b​sbs = 8 | b​sbs = 16 |  |  |  |  |  |  |
| SB-Res-5050 | 8.908.90 | 34.034.0M | 0.946/0.2110.946/0.211 | 11.911.9G | 0.012/0.0050.012/0.005 | 0.013/0.0100.013/0.010 | 0.015/0.0170.015/0.017 | 0.024/0.0270.024/0.027 | 0.160.16G | 0.170.17G | 0.210.21G | 0.300.30G | 70.4{70.4} |
| SB-Res-101101 | 12.412.4 | 53.053.0M | 1.008/0.3201.008/0.320 | 13.213.2G | 0.020/0.0090.020/0.009 | 0.021/0.0140.021/0.014 | 0.024/0.0230.024/0.023 | 0.035/0.0380.035/0.038 | 0.230.23G | 0.240.24G | 0.280.28G | 0.370.37G | 71.4{71.4} |
| SB-Res-152152 | 15.715.7 | 68.668.6M | 1.085/0.4151.085/0.415 | 14.814.8G | 0.030/0.0120.030/0.012 | 0.033/0.0190.033/0.019 | 0.035/0.0310.035/0.031 | 0.048/0.0510.048/0.051 | 0.290.29G | 0.310.31G | 0.350.35G | 0.430.43G | 72.0{72.0} |
| HRNetV11-W3232 | 7.107.10 | 28.528.5M | 1.153/0.3891.153/0.389 | 5.75.7G | 0.057/0.0150.057/0.015 | 0.059/0.0170.059/0.017 | 0.061/0.0200.061/0.020 | 0.062/0.0310.062/0.031 | 0.130.13G | 0.150.15G | 0.190.19G | 0.280.28G | 74.474.4 |
| HRNetV11-W4848 | 14.614.6 | 63.663.6M | 1.231/0.5071.231/0.507 | 7.37.3G | 0.058/0.0170.058/0.017 | 0.060/0.0210.060/0.021 | 0.062/0.0330.062/0.033 | 0.066/0.0510.066/0.051 | 0.270.27G | 0.300.30G | 0.320.32G | 0.440.44G | 75.175.1 |

| backbone | GFLOPs | #params | train sec./iter. | train mem. | infer sec./batch | infer mem./batch | mIoU |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b​sbs = 1 | b​sbs = 4 | b​sbs = 8 | b​sbs = 1 | b​sbs = 4 | b​sbs = 8 |  |  |  |  |  |  |
| Dilated-ResNet | 1661.61661.6 | 52.152.1M | 0.66110.6611 | 12.412.4G | 0.33510.3351 | 1.28821.2882 | 2.70392.7039 | 1.131.13G | 3.923.92G | 7.647.64G | 75.775.7 |
| PSPNet | 2017.62017.6 | 65.965.9M | 0.83680.8368 | 14.414.4G | 0.39720.3972 | 1.52961.5296 | 3.20033.2003 | 1.601.60G | 5.045.04G | 9.819.81G | 79.779.7 |
| DeepLabv33 | 1778.71778.7 | 58.058.0M | 0.85020.8502 | 13.313.3G | 0.41130.4113 | 1.53071.5307 | 3.20003.2000 | 1.151.15G | 3.953.95G | 7.677.67G | 78.578.5 |
| HRNetV22-W4848 | 696.2696.2 | 65.965.9M | 0.69200.6920 | 13.913.9G | 0.15020.1502 | 0.054210.05421 | 1.10321.1032 | 1.791.79G | 6.376.37G | 12.512.5G | 81.181.1 |

| backbone | GFLOPs | #params | sec./iter. | train mem. | infer sec./batch | infer mem./batch | AP\operatorname{AP} | AP50\operatorname{AP}_{50} | AP75\operatorname{AP}_{75} | APS\operatorname{AP}_{S} | APM\operatorname{AP}_{M} | APL\operatorname{AP}_{L} |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b​sbs = 1 | b​sbs = 4 | b​sbs = 8 | b​sbs = 1 | b​sbs = 4 | b​sbs = 8 |  |  |  |  |  |  |  |  |  |  |  |
| ResNet-5050 | 172.34172.34 | 39.7739.77M | 0.41670.4167 | 3.43.4G | 0.07390.0739 | 0.24950.2495 | 0.49210.4921 | 0.550.55G | 1.681.68G | 2.932.93G | 37.637.6 | 58.758.7 | 41.341.3 | 21.421.4 | 40.840.8 | 49.749.7 |
| ResNet-101101 | 239.37239.37 | 57.8357.83M | 0.55020.5502 | 5.45.4G | 0.08700.0870 | 0.30180.3018 | 0.60890.6089 | 0.620.62G | 1.761.76G | 3.253.25G | 39.839.8 | 61.461.4 | 43.443.4 | 22.922.9 | 43.643.6 | 52.452.4 |
| X-101101-64×464\times 4d | 381.83381.83 | 94.8594.85M | 1.18281.1828 | 9.59.5G | 0.14380.1438 | 0.48320.4832 | 0.97640.9764 | 0.770.77G | 1.921.92G | 3.413.41G | 40.840.8 | 62.162.1 | 44.644.6 | 23.223.2 | 44.544.5 | 53.753.7 |
| HRNetV22p-W1818 | 159.06159.06 | 26.1826.18M | 0.61660.6166 | 6.26.2G | 0.09680.0968 | 0.23200.2320 | 0.44710.4471 | 0.330.33G | 1.011.01G | 1.911.91G | 38.038.0 | 58.958.9 | 41.541.5 | 22.622.6 | 40.840.8 | 49.649.6 |
| HRNetV22p-W3232 | 245.33245.33 | 45.0445.04M | 0.69010.6901 | 8.58.5G | 0.10140.1014 | 0.27380.2738 | 0.55460.5546 | 0.510.51G | 1.511.51G | 2.832.83G | 40.940.9 | 61.861.8 | 44.844.8 | 24.424.4 | 43.743.7 | 53.353.3 |
| HRNetV22p-W4848 | 399.12399.12 | 79.4279.42M | 0.96480.9648 | 11.311.3G | 0.11620.1162 | 0.37620.3762 | 0.72960.7296 | 0.790.79G | 2.162.16G | 3.993.99G | 41.841.8 | 62.862.8 | 45.945.9 | 25.025.0 | 44.744.7 | 54.654.6 |

## Appendix D Facial Landmark Detection

Facial landmark detection a.k.a. face alignment is a problem of detecting the keypoints from a face image. We perform the evaluation over four standard datasets: WFLW [149], AFLW [75], COFW [10], and 300300W [121]. We mainly use the normalized mean error (NME) for evaluation. We use the inter-ocular distance as normalization for WFLW, COFW, and 300300W, and the face bounding box as normalization for AFLW. We also report area-under-the-curve scores (AUC) and failure rates.

We follow the standard scheme [149] for training. All the faces are cropped by the provided boxes according to the center location and resized to 256×256256\times 256. We augment the data by ±30\pm 30 degrees in-plane rotation, 0.75−1.250.75-1.25 scaling, and randomly flipping. The base learning rate is 0.00010.0001 and is dropped to 0.000010.00001 and 0.0000010.000001 at the 3030th and 5050th epochs. The models are trained for 6060 epochs with the batch size of 1616 on one GPU. Different from semantic segmentation, the heatmaps are not upsampled from 1/41/4 to the input size, and the loss function is optimized over the 1/41/4 maps.

At testing, each keypoint location is predicted by transforming the highest heatvalue location from 1/41/4 to the original image space and adjusting it with a quarter offset in the direction from the highest response to the second highest response [23].

We adopt HRNetV22-W1818 for face landmark detection whose parameter and computation cost are similar to or smaller than models with widely-used backbones: ResNet-5050 and Hourglass [105]. HRNetV22-W1818: #parameters =9.3=9.3M, GFLOPs =4.3=4.3G; ResNet-5050: #parameters =25.0=25.0M, GFLOPs =3.8=3.8G; Hourglass: #parameters =25.1=25.1M, GFLOPs =19.1=19.1G. The numbers are obtained on the input size 256×256256\times 256. It should be noted that the facial landmark detection methods adopting ResNet-5050 and Hourglass as backbones introduce extra parameter and computation overhead.

WFLW. The WFLW dataset [149] is a recently-built dataset based on the WIDER Face [164]. There are 7,5007,500 training and 2,5002,500 testing images with 9898 manual annotated landmarks. We report the results on the test set and several subsets: large pose (326326 images), expression (314314 images), illumination (698698 images), make-up (206206 images), occlusion (736736 images) and blur (773773 images).

Table XX provides the comparison of our method with state-of-the-art methods. Our approach is significantly better than other methods on the test set and all the subsets, including LAB that exploits extra boundary information [149] and PDB that uses stronger data augmentation [40].

AFLW. The AFLW [75] dataset is a widely used benchmark dataset, where each image has 1919 facial landmarks. Following [191, 149], we train our models on 20,00020,000 training images, and report the results on the AFLW-Full set (4,3864,386 testing images) and the AFLW-Frontal set (13141314 testing images selected from 43864386 testing images).

Table XXI provides the comparison of our method with state-of-the-art methods. Our approach achieves the best performance among methods without extra information and stronger data augmentation and even outperforms DCFE with extra 33D information. Our approach performs slightly worse than LAB that uses extra boundary information [149] and PDB [40] that uses stronger data augmentation.

COFW. The COFW dataset [10] consists of 1,3451,345 training and 507507 testing faces with occlusions, where each image has 2929 facial landmarks.

Table XXII provides the comparison of our method with state-of-the-art methods. HRNetV22 outperforms other methods by a large margin. In particular, it achieves the better performance than LAB with extra boundary information and PDB with stronger data augmentation.

300300W. The dataset [121] is a combination of HELEN [80], LFPW [5], AFW [193], XM2VTS and IBUG datasets, where each face has 6868 landmarks. Following [117], we use the 3,1483,148 training images, which contains the training subsets of HELEN and LFPW and the full set of AFW. We evaluate the performance using two protocols, full set and test set. The full set contains 689689 images and is further divided into a common subset (554554 images) from HELEN and LFPW, and a challenging subset (135135 images) from IBUG. The official test set, used for competition, contains 600600 images (300300 indoor and 300300 outdoor images).

Table XXIII provides the results on the full set, and its two subsets: common and challenging. Table XXIV provides the results on the test set. In comparison to Chen et al. [23] that uses Hourglass with large parameter and computation complexity as the backbone, our scores are better except the AUC0.08 scores. Our HRNetV22 gets the overall best performance among methods without extra information and stronger data augmentation, and is even better than LAB with extra boundary information and DCFE [141] that explores extra 33D information.

|  | backbone | test | pose | expr. | illu. | mu | occu. | blur |
|---|---|---|---|---|---|---|---|---|
| ESR [14] | - | 11.1311.13 | 25.8825.88 | 11.4711.47 | 10.4910.49 | 11.0511.05 | 13.7513.75 | 12.2012.20 |
| SDM [158] | - | 10.2910.29 | 24.1024.10 | 11.4511.45 | 9.329.32 | 9.389.38 | 13.0313.03 | 11.2811.28 |
| CFSS [191] | - | 9.079.07 | 21.3621.36 | 10.0910.09 | 8.308.30 | 8.748.74 | 11.7611.76 | 9.969.96 |
| DVLN [150] | VGG-16 | 6.086.08 | 11.5411.54 | 6.786.78 | 5.735.73 | 5.985.98 | 7.337.33 | 6.886.88 |
| Our approach | HRNetV22-W1818 | 4.60\mathbf{4.60} | 7.94\mathbf{7.94} | 4.85\mathbf{4.85} | 4.55\mathbf{4.55} | 4.29\mathbf{4.29} | 5.44\mathbf{5.44} | 5.42\mathbf{5.42} |
| Model trained with extra info. |  |  |  |  |  |  |  |  |
| LAB (w/ B) [149] | Hourglass | 5.275.27 | 10.2410.24 | 5.515.51 | 5.235.23 | 5.155.15 | 6.796.79 | 6.326.32 |
| PDB (w/ DA) [40] | ResNet-5050 | 5.115.11 | 8.758.75 | 5.365.36 | 4.934.93 | 5.415.41 | 6.376.37 | 5.815.81 |

|  | backbone | full | frontal |
|---|---|---|---|
| RCN [55] | - | 5.605.60 | 5.365.36 |
| CDM [169] | - | 5.435.43 | 3.773.77 |
| ERT [67] | - | 4.354.35 | 2.752.75 |
| LBF [116] | - | 4.254.25 | 2.742.74 |
| SDM [158] | - | 4.054.05 | 2.942.94 |
| CFSS [191] | - | 3.923.92 | 2.682.68 |
| RCPR [10] | - | 3.733.73 | 2.872.87 |
| CCL [192] | - | 2.722.72 | 2.172.17 |
| DAC-CSR [41] |  | 2.272.27 | 1.811.81 |
| TSR [101] | VGG-S | 2.172.17 | - |
| CPM + SBR [35] | CPM | 2.142.14 | - |
| SAN [34] | ResNet-152152 | 1.911.91 | 1.851.85 |
| DSRN [102] | - | 1.861.86 | - |
| LAB (w/o B) [149] | Hourglass | 1.851.85 | 1.621.62 |
| Our approach | HRNetV2-W1818 | 1.57\mathbf{1.57} | 1.46\mathbf{1.46} |
| Model trained with extra info. |  |  |  |
| DCFE (w/ 33D) [141] | - | 2.172.17 | - |
| PDB (w/ DA) [40] | ResNet-5050 | 1.471.47 | - |
| LAB (w/ B) [149] | Hourglass | 1.251.25 | 1.141.14 |

|  | backbone | NME | FR0.1 |
|---|---|---|---|
| Human | - | 5.605.60 | - |
| ESR [14] | - | 11.2011.20 | 36.0036.00 |
| RCPR [10] | - | 8.508.50 | 20.0020.00 |
| HPM [45] | - | 7.507.50 | 13.0013.00 |
| CCR [39] | - | 7.037.03 | 10.9010.90 |
| DRDA [174] | - | 6.466.46 | 6.006.00 |
| RAR [153] | - | 6.036.03 | 4.144.14 |
| DAC-CSR [41] | - | 6.036.03 | 4.734.73 |
| LAB (w/o B) [149] | Hourglass | 5.585.58 | 2.762.76 |
| Our approach | HRNetV22-W1818 | 3.45\mathbf{3.45} | 0.19\mathbf{0.19} |
| Model trained with extra info. |  |  |  |
| PDB (w/ DA) [40] | ResNet-5050 | 5.075.07 | 3.163.16 |
| LAB (w/ B) [149] | Hourglass | 3.923.92 | 0.390.39 |

|  | backbone | common | challenging | full |
|---|---|---|---|---|
| RCN [55] | - | 4.674.67 | 8.448.44 | 5.415.41 |
| DSRN [102] | - | 4.124.12 | 9.689.68 | 5.215.21 |
| PCD-CNN [78] | - | 3.673.67 | 7.627.62 | 4.444.44 |
| CPM + SBR [35] | CPM | 3.283.28 | 7.587.58 | 4.104.10 |
| SAN [34] | ResNet-152 | 3.343.34 | 6.606.60 | 3.983.98 |
| DAN [76] | - | 3.193.19 | 5.245.24 | 3.593.59 |
| Our approach | HRNetV22-W1818 | 2.87\mathbf{2.87} | 5.15\mathbf{5.15} | 3.32\mathbf{3.32} |
| Model trained with extra info. |  |  |  |  |
| LAB (w/ B) [149] | Hourglass | 2.982.98 | 5.195.19 | 3.493.49 |
| DCFE (w/ 33D) [141] | - | 2.762.76 | 5.225.22 | 3.243.24 |

|  | backbone | NME | AUC0.08 | AUC0.1 | FR0.08 | FR0.1 |
|---|---|---|---|---|---|---|
| Balt. et al. [4] | - | - | 19.5519.55 | - | 38.8338.83 | - |
| ESR [14] | - | 8.478.47 | 26.0926.09 | - | 30.5030.50 | - |
| ERT [67] | - | 8.418.41 | 27.0127.01 | - | 28.8328.83 | - |
| LBF [116] | - | 8.578.57 | 25.2725.27 | - | 33.6733.67 | - |
| Face++ [186] | - | - | 32.8132.81 | - | 13.0013.00 | - |
| SDM [158] | - | 5.835.83 | 36.2736.27 | - | 13.0013.00 | - |
| CFAN [175] | - | 5.785.78 | 34.7834.78 | - | 14.0014.00 | - |
| Yan et al. [162] | - | - | 34.9734.97 | - | 12.6712.67 | - |
| CFSS [191] | - | 5.745.74 | 36.5836.58 | - | 12.3312.33 | - |
| MDM [138] | - | 4.784.78 | 45.3245.32 | - | 6.806.80 | - |
| DAN [76] | - | 4.304.30 | 47.0047.00 | - | 2.672.67 | - |
| Chen et al. [23] | Hourglass | 3.963.96 | 53.64\mathbf{53.64} | - | 2.502.50 | - |
| Deng et al. [30] | - | - | - | 47.5247.52 | - | 5.505.50 |
| Fan et al. [37] | - | - | - | 48.0248.02 | - | 14.8314.83 |
| DReg + MDM [48] | ResNet101 | - | - | 52.1952.19 | - | 3.673.67 |
| JMFA [31] | Hourglass | - | - | 54.8554.85 | - | 1.001.00 |
| Our approach | HRNetV22-W1818 | 3.85\mathbf{3.85} | 52.09 | 61.55\mathbf{61.55} | 1.00\mathbf{1.00} | 0.33\mathbf{0.33} |
| Model trained with extra info. |  |  |  |  |  |  |
| LAB (w/ B) [149] | Hourglass | - | - | 58.8558.85 | - | 0.830.83 |
| DCFE (w/ 33D) [141] | - | 3.883.88 | 52.4252.42 | - | 1.831.83 | - |

## Appendix E More object detection and instance results on COCO val2017

| Backbone | LS | APb\operatorname{AP}^{b} | AP50b\operatorname{AP}^{b}_{50} | AP75b\operatorname{AP}^{b}_{75} | APSb\operatorname{AP}^{b}_{S} | APMb\operatorname{AP}^{b}_{M} | APLb\operatorname{AP}^{b}_{L} | APm\operatorname{AP}^{m} | AP50m\operatorname{AP}^{m}_{50} | AP75m\operatorname{AP}^{m}_{75} | APSm\operatorname{AP}^{m}_{S} | APMm\operatorname{AP}^{m}_{M} | APLm\operatorname{AP}^{m}_{L} |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FCOS |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R-50 (c) | 1x | 36.7 | 55.8 | 39.2 | 21.0 | 40.7 | 48.4 | - | - | - | - | - | - |
| R-101 (c) | 1x | 39.1 | 58.5 | 41.8 | 22.0 | 43.5 | 51.1 | - | - | - | - | - | - |
| R-50 (c) | 2x | 36.9 | 55.8 | 39.1 | 20.4 | 40.1 | 49.2 | - | - | - | - | - | - |
| R-101 (c) | 2x | 39.1 | 58.6 | 41.7 | 22.1 | 42.4 | 52.5 | - | - | - | - | - | - |
| HRNetV2-W18 | 1x | 35.2 | 52.9 | 37.3 | 20.4 | 37.8 | 46.1 | - | - | - | - | - | - |
| HRNetV2-W32 | 1x | 38.2 | 56.2 | 40.9 | 22.2 | 41.8 | 50.0 | - | - | - | - | - | - |
| HRNetV2-W18 | 2x | 37.7 | 55.9 | 40.1 | 22.0 | 40.8 | 48.5 | - | - | - | - | - | - |
| HRNetV2-W32 | 2x | 40.3 | 58.7 | 43.3 | 23.6 | 43.4 | 52.9 | - | - | - | - | - | - |
| FCOS (mstrain) |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R-50 (c) | 2x | 38.7 | 58.0 | 41.4 | 23.4 | 42.8 | 49.0 | - | - | - | - | - | - |
| R-101 (c) | 2x | 40.8 | 60.1 | 43.8 | 24.5 | 44.5 | 52.8 | - | - | - | - | - | - |
| HRNetV2-W18 | 2x | 38.1 | 56.3 | 40.6 | 22.9 | 41.1 | 48.6 | - | - | - | - | - | - |
| HRNetV2-W32 | 2x | 41.4 | 60.3 | 44.2 | 25.2 | 44.8 | 52.3 | - | - | - | - | - | - |
| HRNetV2-W48 | 2x | 42.9 | 61.9 | 45.9 | 26.4 | 46.7 | 54.6 | - | - | - | - | - | - |
| X-101-64x4d | 2x | 42.8 | 62.6 | 45.7 | 26.5 | 46.9 | 54.5 | - | - | - | - | - | - |
| Mask R-CNN |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R-50 (c) | 1x | 37.4 | 58.9 | 40.4 | 21.7 | 41.0 | 49.1 | 34.3 | 55.8 | 36.4 | 18.0 | 37.6 | 47.3 |
| R-101 (c) | 1x | 39.9 | 61.5 | 43.6 | 23.9 | 44.0 | 51.8 | 36.1 | 57.9 | 38.7 | 19.8 | 39.8 | 49.5 |
| R-50 | 1x | 37.3 | 59.0 | 40.2 | 21.9 | 40.9 | 48.1 | 34.2 | 55.9 | 36.2 | 18.2 | 37.5 | 46.3 |
| R-101 | 1x | 39.4 | 60.9 | 43.3 | 23.0 | 43.7 | 51.4 | 35.9 | 57.7 | 38.4 | 19.2 | 39.7 | 49.7 |
| HRNetV2-W18 | 1x | 37.3 | 58.2 | 40.7 | 22.1 | 40.2 | 47.6 | 34.2 | 55.0 | 36.2 | 18.4 | 36.7 | 46.0 |
| HRNetV2-W32 | 1x | 40.7 | 61.9 | 44.6 | 25.1 | 44.4 | 51.8 | 36.8 | 58.7 | 39.5 | 20.9 | 40.0 | 49.3 |
| X-101-32x4d | 1x | 41.1 | 62.8 | 45.0 | 24.0 | 45.4 | 52.6 | 37.1 | 59.4 | 39.8 | 19.7 | 41.1 | 50.1 |
| X-101-64x4d | 1x | 42.1 | 63.8 | 46.3 | 24.4 | 46.6 | 55.3 | 38.0 | 60.6 | 40.9 | 20.2 | 42.1 | 52.4 |
| R-50 | 2x | 38.5 | 59.9 | 41.8 | 22.6 | 42.0 | 50.5 | 35.1 | 56.8 | 37.0 | 18.9 | 38.0 | 48.3 |
| R-101 | 2x | 40.3 | 61.5 | 44.1 | 22.2 | 44.8 | 52.9 | 36.5 | 58.1 | 39.1 | 18.4 | 40.2 | 50.4 |
| HRNetV2-W18 | 2x | 39.2 | 60.1 | 42.9 | 24.2 | 42.1 | 50.8 | 35.7 | 57.3 | 38.1 | 17.6 | 37.8 | 52.3 |
| HRNetV2-W32 | 2x | 42.3 | 62.7 | 46.1 | 26.1 | 45.5 | 54.7 | 37.6 | 59.7 | 40.3 | 21.4 | 40.5 | 51.2 |
| X-101-32x4d | 2x | 41.4 | 62.5 | 45.4 | 24.0 | 45.4 | 54.5 | 37.1 | 59.4 | 39.5 | 19.9 | 40.6 | 51.3 |
| X-101-64x4d | 2x | 42.0 | 63.1 | 46.1 | 23.9 | 45.8 | 55.6 | 37.7 | 59.9 | 40.4 | 19.6 | 41.3 | 52.5 |
| Cascade Mask R-CNN |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R-50 | 1x | 41.2 | 59.1 | 45.1 | 23.3 | 44.5 | 54.5 | 35.7 | 56.3 | 38.6 | 18.5 | 38.6 | 49.2 |
| R-101 | 1x | 42.6 | 60.7 | 46.7 | 23.8 | 46.4 | 56.9 | 37.0 | 58.0 | 39.9 | 19.1 | 40.5 | 51.4 |
| X-101-32x4d | 1x | 44.4 | 62.6 | 48.6 | 25.4 | 48.1 | 58.7 | 38.2 | 59.6 | 41.2 | 20.3 | 41.9 | 52.4 |
| X-101-64x4d | 1x | 45.4 | 63.7 | 49.7 | 25.8 | 49.2 | 60.6 | 39.1 | 61.0 | 42.1 | 20.5 | 42.6 | 54.1 |
| R-50 | 20e | 42.3 | 60.5 | 46.0 | 23.7 | 45.7 | 56.4 | 36.6 | 57.6 | 39.5 | 19.0 | 39.4 | 50.7 |
| R-101 | 20e | 43.3 | 61.3 | 47.0 | 24.4 | 46.9 | 58.0 | 37.6 | 58.5 | 40.6 | 19.7 | 40.8 | 52.4 |
| HRNetV2-W18 | 20e | 41.9 | 59.6 | 45.7 | 23.8 | 44.9 | 55.0 | 36.4 | 56.8 | 39.3 | 17.0 | 38.6 | 52.9 |
| HRNetV2-W32 | 20e | 44.5 | 62.3 | 48.6 | 26.1 | 47.9 | 58.5 | 38.5 | 59.6 | 41.9 | 18.9 | 41.1 | 56.1 |
| HRNetV2-W48 | 20e | 46.0 | 63.7 | 50.3 | 27.5 | 48.9 | 60.1 | 39.5 | 61.1 | 42.8 | 19.7 | 41.8 | 56.9 |
| X-101-32x4d | 20e | 44.7 | 63.0 | 48.9 | 25.9 | 48.7 | 58.9 | 38.6 | 60.2 | 41.7 | 20.9 | 42.1 | 52.7 |
| X-101-64x4d | 20e | 45.7 | 64.1 | 50.0 | 26.2 | 49.6 | 60.0 | 39.4 | 61.3 | 42.9 | 20.8 | 42.7 | 54.1 |
| Hybrid Task Cascade |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R-50 | 1x | 42.1 | 60.8 | 45.9 | 23.9 | 45.5 | 56.2 | 37.3 | 58.2 | 40.2 | 19.5 | 40.6 | 51.7 |
| R-50 | 20e | 43.2 | 62.1 | 46.8 | 24.9 | 46.4 | 57.8 | 38.1 | 59.4 | 41.0 | 20.3 | 41.1 | 52.8 |
| R-101 | 20e | 44.9 | 63.8 | 48.7 | 26.4 | 48.3 | 59.9 | 39.4 | 60.9 | 42.4 | 21.4 | 42.4 | 54.4 |
| HRNetV2-W18 | 20e | 43.1 | 61.5 | 46.8 | 26.6 | 46.0 | 56.9 | 37.9 | 59.0 | 40.6 | 18.8 | 39.9 | 55.2 |
| HRNetV2-W32 | 20e | 45.3 | 63.6 | 49.1 | 27.0 | 48.4 | 59.5 | 39.6 | 61.2 | 43.0 | 19.1 | 42.0 | 57.9 |
| HRNetV2-W48 | 20e | 46.8 | 65.3 | 51.1 | 28.0 | 50.2 | 61.7 | 40.7 | 62.6 | 44.2 | 19.7 | 43.4 | 59.3 |
| HRNetV2-W48 | 28e | 47.0 | 65.5 | 51.0 | 28.8 | 50.3 | 62.2 | 41.0 | 63.0 | 44.7 | 20.8 | 43.9 | 59.9 |
| X-101-32x4d | 20e | 46.1 | 65.1 | 50.2 | 27.5 | 49.8 | 61.2 | 40.3 | 62.2 | 43.5 | 22.3 | 43.7 | 55.5 |
| X-101-64x4d | 20e | 46.9 | 66.0 | 51.2 | 28.0 | 50.7 | 62.1 | 40.8 | 63.3 | 44.1 | 22.7 | 44.2 | 56.3 |
| X-101-64x4d | 28e | 46.8 | 65.6 | 50.9 | 27.5 | 51.0 | 61.7 | 40.7 | 63.1 | 43.9 | 20.0 | 44.1 | 59.9 |
