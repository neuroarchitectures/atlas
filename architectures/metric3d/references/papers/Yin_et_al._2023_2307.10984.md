# Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image

Reconstructing accurate 3D scenes from images is a long-standing vision task. Due to the ill-posedness of the single-image reconstruction problem, most well-established methods are built upon multi-view geometry. State-of-the-art (SOTA) monocular metric depth estimation methods can only handle a single camera model and are unable to perform mixed-data training due to the metric ambiguity. Meanwhile, SOTA monocular methods trained on large mixed datasets achieve zero-shot generalization by learning affine-invariant depths, which cannot recover real-world metrics. In this work, we show that the key to a zero-shot single-view metric depth model lies in the combination of large-scale data training and resolving the metric ambiguity from various camera models. We propose a canonical camera space transformation module, which explicitly addresses the ambiguity problems and can be effortlessly plugged into existing monocular models. Equipped with our module, monocular models can be stably trained over 88 million of images with thousands of camera models, resulting in zero-shot generalization to in-the-wild images with unseen camera settings.

Experiments demonstrate SOTA performance of our method on 77 zero-shot benchmarks. Notably, our method won the championship in the 2nd Monocular Depth Estimation Challenge. Our method enables the accurate recovery of metric 3D structures on randomly collected internet images, paving the way for plausible single-image metrology. The potential benefits extend to downstream tasks, which can be significantly improved by simply plugging in our model. For example, our model relieves the scale drift issues of monocular-SLAM (Fig. 1), leading to high-quality metric scale dense mapping. The code is available at https://github.com/YvanYin/Metric3D.

## 1 Introduction

3D reconstruction from images is the core of many computer vision applications, such as autonomous driving and robotics. Main-stream methods leverage multi-view geometry [21] to confidently recover 3D structures. However, these methods cannot be applied to a single image, making 3D reconstruction hard without a prior. State-of-the-art transferable methods, such as MiDaS [40], LeReS [69], and HDN [73], learn such a prior from a large dataset, but they can only output affine-invariant depths, i.e., which are accurate only up to an unknown offset and scale. Though monocular metric depth estimation methods [71, 4] work on a single dataset with a single camera model, they cannot generalize to unseen cameras or scenes. This work aims to address the above problems by learning a zero-shot, single view, metric depth model.

According to the predicted depth, existing methods are categorized into learning metric depth [71, 64, 4, 63], learning relative depth [57, 58, 8, 7], and learning affine-invariant depth [69, 68, 40, 39, 73]. Although the metric depth methods [71, 64, 66, 4, 63] have achieved impressive accuracy on various benchmarks, they must train and test on the dataset with the same camera intrinsics. Therefore, the training datasets of metric depth methods are often small, as it is hard to collect a large dataset covering diverse scenes using one identical camera. The consequence is that all these models are not transferable – they generalize poorly to images in the wild, not to mention the camera parameters of test images can vary too. A compromise is to learn the relative depth [8, 57], which only represents one point being further or closer to another one. The application of relative depth is very limited. Learning affine-invariant depth finds a trade-off between the above two categories of methods. With large-scale data, they decouple the metric information during training and achieve impressive robustness and generalization ability. The recent state-of-the-art LeReS [69] can recover 3D scenes in the wild, but only up to an unknown scale and shift.

This work focuses on learning a zero-shot transferable model to recover metric 3D from a single image. First, we analyze the metric ambiguity issues in monocular depth estimation and study different camera parameters in depth, including the pixel size, focal length, and sensor size. We observe that the focal length is the critical factor for accurate metric recovery. By design, LeReS [69] does not take the focal length information into account during training. As shown in Sec. 3.1, only from the image appearance, various focal lengths may cause metric ambiguity, thus they decouple the depth scale in training. To solve the problem of varying focal lengths, CamConv [15] encodes the camera model in the network, which enforces the network to implicitly understand camera models from the image appearance and then bridges the imaging size to the real-world size. However, training data contains limited images and types of cameras, which challenges data diversity and network capacity. In contrast, we propose a canonical camera transformation method in training. It is inspired by the human body reconstruction methods. To improve reconstructed shape quality on countless poses, they map all samples to a canonical pose space [37] to reduce pose variance. Similarly, we transform all training data to a canonical camera space where the processed images are coarsely regarded as captured by the same camera. To achieve such transformation, we propose two different methods. The first one tries to adjust the image appearance to simulate the canonical camera, while the other one transforms the ground-truth labels for supervision. Camera models are not encoded in the network, making our method easily applicable to existing architectures. During inference, a de-canonical transformation is employed to recover metric information. To further boost the depth accuracy, we propose a random proposal normalization loss. It is inspired by the scale-shift invariant loss [69, 40, 73], which decouples the depth scale to emphasize the single image’s distribution. However, they perform on the whole image, which inevitably squeezes the fine-grained depth difference. We propose to randomly crop several patches from images and enforce the scale-shift invariant loss [69, 40] on them. Our loss emphasizes the local geometry and distribution of the single image.

With the proposed method, we can easily scale up model training to 8 million images from 11 datasets of diverse scene types (indoor and outdoor) and camera models (tens of thousands of different cameras), leading to zero-shot transferability and a significantly improved accuracy. Our model can accurately reconstruct metric 3D from randomly collected Internet images, enabling plausible single-image metrology. Different from affine-invariant depth models, our model can also directly improve various downstream tasks. As an example (Fig. 1), with the predicted metric depths from our model, we can significantly reduce the scale drift of monocular SLAM [52, 51] systems, achieving much better mapping quality with real-world metric recovery. Our model also enables large-scale 3D reconstruction [23]. The model achieves the championship in the 2nd Monocular Depth Estimation Challenge [50]. To summarize, our main contributions are:

- • We propose a canonical and de-canonical camera transformation method to solve the metric depth ambiguity problems from various cameras setting. It enables the learning of strong zero-shot monocular metric depth models from large-scale datasets.
We propose a canonical and de-canonical camera transformation method to solve the metric depth ambiguity problems from various cameras setting. It enables the learning of strong zero-shot monocular metric depth models from large-scale datasets.

- • We propose a random proposal normalization loss to effectively boost the depth accuracy;
We propose a random proposal normalization loss to effectively boost the depth accuracy;

- • Our model achieves state-of-the-art performance on 77 zero-shot benchmarks. It can perform high-quality 3D metric structure recovery in the wild and benefit several downstream tasks, such as mono-SLAM [52, 35], 3D scene reconstruction [23], and metrology [75].
Our model achieves state-of-the-art performance on 77 zero-shot benchmarks. It can perform high-quality 3D metric structure recovery in the wild and benefit several downstream tasks, such as mono-SLAM [52, 35], 3D scene reconstruction [23], and metrology [75].

## 2 Related Work

3D reconstruction from a single image. Reconstructing various objects from a single image has been well studied [1, 54, 56]. They can produce high-quality 3D models of cars, planes, tables, and human body [41, 42]. The main challenge is how to best recover objects’ details, how to represent them with limited memory, and how to generalize to more diverse objects. However, all these methods rely on learning priors specific to a certain object class or instance, typically from 3D supervision, and can therefore not work for full scene reconstruction. Apart from these reconstructing objects works, several works focus on scene reconstruction [61] from a single image. Saxena et al. [43] construct the scene based on the assumption that the whole scene can be segmented into several small planes. With planes’ orientation and location, the 3D structure can be represented. Recently, LeReS [69] propose to use a strong monocular depth estimation model to do scene reconstruction. However, they can only recover the shape up to a scale. Zhang et al. [74] recently propose a zero-shot geometry-preserving depth estimation model that is capable of making depth predictions up to an unknown scale, without requiring scale-invariant depth annotations for training. In contrast to these works, our method can recover the metric 3D structure.

Supervised monocular depth estimation. After several benchmarks [47, 17] are established, neural network based methods [71, 66, 4] have dominated since then. Several approaches regress the continuous depth from the aggregation of information in an image [14]. As depth distribution corresponding to different RGBs can vary to a large extent, some methods [66, 4] discretize the depth and formulate this problem to a classification [64], which often achieves better performance. The generalization issue of deep models for 3D metric recovery is related to two problems. The first one is to generalize to diverse scenes, while the other one is how to predict accurate metric information under various camera settings. The first problem has been well addressed by recent methods. Some works [58, 57, 64] propose to construct a large-scale relative depth dataset, such as DIW [7] and OASIS [8], and then they target learning the relative relations. However, the relative depth loses geometric structure information. To improve the recovered geometry quality, learning affine-invariant depth methods, such as MiDaS [40], LeReS [69], and HDN [73] are proposed. By mixing large-scale data, state-of-the-art performance and the generalization over scenes are improved continuously. Note that by design, these methods are unable to recover the metric information. How to achieve both strong generalization and accurate metric information over diverse scenes is the key problem that we attempt to tackle.

Large-scale data training. Recently, various natural language problems and computer vision problems [65, 38, 29] have achieved impressive progress with large-scale data training. CLIP [38] is a promising classification model, which is trained on billions of paired image and language descriptions data. It achieved state-of-the-art performance over several classification benchmarks by zero-shot testing. For depth prediction, large-scale data training has been widely applied. Ranft et al. [40] mixed over 2 million data in training, LeReS [68] collected over 300300 thousands data, Eftekhar et al. [13] also merged millions of data to build a strong depth prediction model.

## 3 Method

Preliminaries. We consider the pin-hole camera model with intrinsic parameters are: [[f^/δ,0,u0\nicefrac{{\hat{f}}}{{\delta}},0,u_{0}], [0,f^/δ,v00,\nicefrac{{\hat{f}}}{{\delta}},v_{0}], [0,0,10,0,1]], where f^\hat{f} is the focal length (in micrometers), δ\delta is the pixel size (in micrometers), and (u0,v0)(u_{0},v_{0}) is the principle center. f=f^/δf=\nicefrac{{\hat{f}}}{{\delta}} is the pixel-represented focal length used in vision algorithms.

### 3.1 Metric Ambiguity Analysis

Fig. 3 presents an example of photos taken by different cameras and at different distances. Only from the image’s appearance, one may think the last two photos are taken at a similar location by the same camera. In fact, due to different focal lengths, these are captured at different locations. Thus, camera intrinsic parameters are critically important for the metric estimation from a single image, as otherwise, the problem is ill posed. To avoid such metric ambiguity, recent methods, such as MiDaS [40] and LeReS [69], decouple the metric from the supervision and compromise learning the affine-invariant depth.

Fig. 4 (A) shows a simple pin-hole perspective projection. Object AA locating at dad_{a} is projected to A′A^{\prime}. Based on the principle of similarity, we have the equation:

|  | da=S^​[f^S′^]=S^⋅α\vskip-10.00002ptd_{a}=\hat{S}\Bigl[\frac{\hat{f}}{\hat{S^{\prime}}}\Bigr]=\hat{S}\cdot\alpha |  | (1) |
|---|---|---|---|

where S^\hat{S} and S′^\hat{S^{\prime}} are the real and imaging size respectively. ⋅^\hat{\cdot} denotes variables are in the physical metric (e.g., millimeter). To recover dad_{a} from a single image, focal length, imaging size of the object, and real-world object size must be available. Estimating the focal length from a single image is a challenging and ill-posed problem. Although several methods [69, 22] have explored, the accuracy is still far from being satisfactory. Here, we simplify the problem by assuming the focal length of a training/test image is available. In contrast, understanding the imaging size is much easier for a neural network. To obtain the real-world object size, a neural network needs to understand the semantic scene layout and the object, at which a neural network excels. We define α=f^/S′^\alpha=\nicefrac{{\hat{f}}}{{\hat{S^{\prime}}}}, so dad_{a} is proportional to α\alpha.

We make the following observations regarding sensor size, pixel size, and focal length.

O1: Sensor size and pixel size do not affect the metric depth estimation. Based on the perspective projection (Fig. 4 (A)), the sensor size only affects the field of view (FOV) and is irrelevant to α\alpha, thus does not affect the metric depth estimation. For the pixel size, we assume two cameras with different pixel sizes (δ1=2​δ2\delta_{1}=2\delta_{2}) but the same focal length f^\hat{f} to capture the same object locating at dad_{a}. Fig. 4 (B) shows their captured photos. According to the preliminaries, the pixel-represented focal length f1=12​f2f_{1}=\frac{1}{2}f_{2}. As the second camera has a smaller pixel size, although in the same projected imaging size S′^\hat{S^{\prime}}, the pixel-represented image resolution is S1′=12​S2′S^{\prime}_{1}=\frac{1}{2}S^{\prime}_{2}. According to Eq. (1), f^δ1⋅S1′=f^δ2⋅S2′\frac{\hat{f}}{\delta_{1}\cdot S^{\prime}_{1}}=\frac{\hat{f}}{\delta_{2}\cdot S^{\prime}_{2}}, i.e. α1=α2\alpha_{1}=\alpha_{2}, so d1=d2d_{1}=d_{2}. Therefore, different camera sensors would not affect the metric depth estimation.

O2: The focal length is vital for metric depth estimation. Fig. 3 shows the metric ambiguity issue caused by the unknown focal length. Fig. 5 illustrates this. If two cameras (f^1=2​f^2\hat{f}_{1}=2\hat{f}_{2}) are at distances d1=2​d2d_{1}=2d_{2}, the imaging sizes on cameras are the same. Thus, only from the appearance, the network will be confused when supervised with different labels. Based on this observation, we propose a canonical camera transformation method to solve the supervision and image appearance conflicts.

### 3.2 Canonical Camera Transformation

The core idea is to set up a canonical camera space ((fxc,fyc)(f_{x}^{c},f_{y}^{c}), fxc=fyc=fcf_{x}^{c}=f_{y}^{c}=f^{c} in experiments) and transform all training data to this space. Consequently, all data can roughly be regarded as captured by the canonical camera. We propose two transformation methods, i.e. either transforming the input image (𝐈∈ℝH×W×3\mathbf{I}\in\mathbb{R}^{H\times W\times 3}) or the ground-truth (GT) label (𝐃∈ℝH×W\mathbf{D}\in\mathbb{R}^{H\times W}). The original intrinsics are {f,u0,v0}\{f,u_{0},v_{0}\}.

Method1: transforming depth labels (CSTM_label). Fig. 3’s ambiguity is for depths. Thus our first method directly transforms the ground-truth depth labels to solve this problem. Specifically, we scale the ground-truth depth (𝐃∗\mathbf{D}^{*}) with the ratio ωd=fcf\omega_{d}=\frac{f^{c}}{f} in training, i.e., 𝐃c∗=ωd​𝐃∗\mathbf{D}^{*}_{c}=\omega_{d}\mathbf{D}^{*}. The original camera model is transformed to {fc,u0,v0}\{f^{c},u_{0},v_{0}\}. In inference, the predicted depth (𝐃c\mathbf{D}_{c}) is in the canonical space and needs to perform a de-canonical transformation to recover the metric information, i.e., 𝐃=1ωd​𝐃c\mathbf{D}=\frac{1}{\omega_{d}}\mathbf{D}_{c}. Note the input 𝐈\mathbf{I} does not perform any transformation, i.e., 𝐈c=𝐈\mathbf{I}_{c}=\mathbf{I}.

Method2: transforming input images (CSTM_image). From another view, the ambiguity is caused by the similar image appearance. Thus this method is to transform the input image to simulate the canonical camera imaging effect. Specifically, the image 𝐈\mathbf{I} is resized with the ratio ωr=fcf\omega_{r}=\frac{f^{c}}{f}, i.e., 𝐈c=𝒯⁡(𝐈,ωr)\mathbf{I}_{c}=\mathcal{T}(\mathbf{I},\omega_{r}), where 𝒯⁡(⋅)\mathcal{T}(\cdot) denotes image resize. The optical center is resized, thus the canonical camera model is {fc,ωr​u0,ωr​v0}\{f^{c},\omega_{r}u_{0},\omega_{r}v_{0}\}. The ground-truth labels are resized without any scaling, i.e., 𝐃c∗=𝒯⁡(𝐃∗,ωr)\mathbf{D}^{*}_{c}=\mathcal{T}(\mathbf{D}^{*},\omega_{r}). In inference, the de-canonical transformation is to resize the prediction to the original size without scaling, i.e., 𝐃=𝒯⁡(𝐃c,1ωr)\mathbf{D}=\mathcal{T}(\mathbf{D}_{c},\frac{1}{\omega_{r}}).

Fig. 2 shows the pipeline. After performing either transformation, we randomly crop a patch for training. The cropping only adjusts the FOV and the optical center, thus not causing any metric ambiguity issues. In the labels transformation method ωr=1\omega_{r}=1 and ωd=fcf\omega_{d}=\frac{f^{c}}{f}, while ωd=1\omega_{d}=1 and ωr=fcf\omega_{r}=\frac{f^{c}}{f} in the images transformation method. The training objective is as follows:

|  | minθ⁡|𝒩d​(𝐈c,θ)−𝐃c∗|\min_{\theta}\left|\mathcal{N}_{d}(\mathbf{I}_{c},\theta)-\mathbf{D}^{*}_{c}\right|\vskip-5.0pt |  | (2) |
|---|---|---|---|

where θ\theta is the network’s (𝒩d​(⋅)\mathcal{N}_{d}(\cdot)) parameters, 𝐃c∗\mathbf{D}^{*}_{c} and 𝐈c\mathbf{I}_{c} are transformed ground-truth depth labels and images.

Mix-data training is an effective way to boost generalization. We collect 1111 datasets for training, see the supplementary materials for details. In the mixed data, over 10K different cameras are included. All collected training data have included paired camera intrinsic parameters, which are used in our canonical transformation module.

Supervision. To further boost the performance, we propose a random proposal normalization loss (RPNL). The scale-shift invariant loss [40, 69] is widely applied for the affine-invariant depth estimation, which decouples the depth scale to emphasize the single image distribution. However, such normalization based on the whole image inevitably squeezes the fine-grained depth difference, particularly in close regions. Inspired by this, we propose to randomly crop several patches (pi⁡(i=0,…,M)∈ℝhi×wip_{i(i=0,...,M)}\in\mathbb{R}^{h_{i}\times w_{i}}) from the ground truth 𝐃c∗\mathbf{D}^{*}_{c} and the predicted depth 𝐃c\mathbf{D}_{c}. Then we employ the median absolute deviation normalization [48] for paired patches. By normalizing the local statistics, we can enhance local contrast. The loss function is as follows:

|  | LRPNL=1M​N∑piM∑jN|dpi,j∗−μ⁡(dpi,j∗)1N​∑jN|dpi,j∗−μ⁡(dpi,j∗)|−\displaystyle L_{{\rm RPNL}}=\frac{1}{MN}\sum_{p_{i}}^{M}\sum_{j}^{N}\lvert\frac{d^{*}_{p_{i},j}-\mu(d^{*}_{p_{i},j})}{\frac{1}{N}\sum_{j}^{N}\left|d^{*}_{p_{i},j}-\mu(d^{*}_{p_{i},j})\right|}- |  |  |
|---|---|---|---|
|  | dpi,j−μ⁡(dpi,j)1N​∑jN|dpi,j−μ⁡(dpi,j)||\displaystyle\frac{d_{p_{i},j}-\mu(d_{p_{i},j})}{\frac{1}{N}\sum_{j}^{N}\left|d_{p_{i},j}-\mu(d_{p_{i},j})\right|}\rvert |  | (3) |

where d∗∈𝐃c∗d^{*}\in\mathbf{D}^{*}_{c} and d∈𝐃cd\in\mathbf{D}_{c} are the ground truth and predicted depth respectively. μ⁡(⋅)\mu(\cdot) and is the median of depth. MM is the number of proposal crops, which is set to 32. During training, proposals are randomly cropped from the image by 0.1250.125 to 0.50.5 of the original size. Furthermore, several other losses are employed, including the scale-invariant logarithmic loss [14] Ls​i​l​o​gL_{silog}, pair-wise normal regression loss [69]LPWNL_{{\rm PWN}}, virtual normal loss [64] LVNLL_{{\rm VNL}}. Note Ls​i​l​o​gL_{silog} is a variant of L1 loss. The overall losses are as follows.

|  | L=LPWN+LVNL+Ls​i​l​o​g+LRPNL.\displaystyle L=L_{{\rm PWN}}+L_{{\rm VNL}}+L_{silog}+L_{{\rm RPNL}}. |  |
|---|---|---|

## 4 Experiments

Dataset details. We collect 1111 public RGB-D datasets, and over 88 million data for training. It spreads over diverse indoor and outdoor scenes. Note that all datasets have provided camera intrinsic parameters. Apart from the test split of training datasets, we collect 77 unseen datasets for robustness and generalization evaluation. Details of employed data are reported in the supplementary materials.

Implementation details. We employ an UNet architecture with the ConvNext-large [33] backbone. ImageNet-22K pre-trained weights are used for initialization. We use AdamW with a batch size of 192192, an initial learning rate 0.00010.0001 for all layers, and the polynomial decaying method with the power of 0.90.9. We train our final model on 4848 A100 GPUs for 500500K iterations. Following the DiverseDepth [64], we balance all datasets in a mini-batch to ensure each dataset accounts for an almost equal ratio. During training, images are processed by the canonical camera transformation module, flipped horizontally with a 50%50\% chance, and then randomly cropped into 512×960512\times 960 pixels. For the ablation experiments, training settings are different as we sample 50005000 images from each dataset for training. We trained on 88 GPUs for 150150K iterations.

Evaluation details. a) To show the robustness of our metric depth estimation method, we test on 8 zero-shot benchmarks, including NYUv2 [47], KITTI [17], NuScenes [6], 7-scenes [46], iBIMS-1 [26], DIODE [53], ETH3D [45]. Following previous works [71], absolute relative error (AbsRel), the accuracy under threshold (δi<1.25i,i=1,2,3\delta_{i}<1.25^{i},i=1,2,3), root mean squared error (RMS), root mean squared error in log space (RMS_log), and log10 error (log10) metrics are employed. b) Furthermore, we also follow current affine-invariant depth benchmarks [69, 73] (Tab. 4) to evaluate the generalization ability on 55 zero-shot datasets, i.e., NYUv2, DIODE, ETH3D, ScanNet [11], and KITTI. We mainly compare with large-scale data trained models. Note that in this benchmark we follow existing methods to apply the scale shift alignment before evaluation. c) To evaluate our metric 3D reconstruction quality, we randomly sample 9 unseen scenes from NYUv2 and use colmap [44] to obtain the camera poses for multi-frame reconstruction. Chamfer l1l_{1} distance and the F-score [25] are used to evaluate the reconstruction accuracy. d) In dense-SLAM experiments, following Li et al. [31], we test on the KITTI odometry benchmark [17] and evaluate the average translational RMS drift (%,tr​e​l\%,t_{rel}) and rotational RMS drift (°/100​m,rr​e​l\degree/100m,r_{rel}) errors [17]. Note that all these depth and reconstruction evaluations use the same trained model.

### 4.1 Zero-shot Generalization Test

| NYUv2 Benchmark |  |  |  |  |  |  |
|---|---|---|---|---|---|---|
| Method | 𝜹𝟏\boldsymbol{\delta_{1}}↑\uparrow | 𝜹𝟐\boldsymbol{\delta_{2}}↑\uparrow | 𝜹𝟑\boldsymbol{\delta_{3}}↑\uparrow | AbsRel↓\downarrow | log10↓\downarrow | RMS↓\downarrow |
| Li et al. [30] | 0.7880.788 | 0.9580.958 | 0.9910.991 | 0.1430.143 | 0.0630.063 | 0.6350.635 |
| Laina et al. [28] | 0.8110.811 | 0.9530.953 | 0.9880.988 | 0.1270.127 | 0.0550.055 | 0.5730.573 |
| VNL [66] | 0.8750.875 | 0.9760.976 | 0.9940.994 | 0.1080.108 | 0.0480.048 | 0.4160.416 |
| TrDepth [63] | 0.9000.900 | 0.9830.983 | 0.9960.996 | 0.1060.106 | 0.0450.045 | 0.3650.365 |
| Adabins [4] | 0.9030.903 | 0.984{0.984} | 0.997¯\underline{0.997} | 0.1030.103 | 0.0440.044 | 0.3640.364 |
| NeWCRFs [71] | 0.922{0.922} | 0.992\boldsymbol{0.992} | 0.998\boldsymbol{0.998} | 0.0950.095 | 0.0410.041 | 0.334¯\underline{0.334} |
| Ours CSTM_image | 0.925\boldsymbol{0.925} | 0.9830.983 | 0.9940.994 | 0.092¯\underline{0.092} | 0.040¯\underline{0.040} | 0.341{0.341} |
| Ours CSTM_label | 0.944¯\underline{0.944} | 0.986¯\underline{0.986} | 0.9950.995 | 0.083\boldsymbol{0.083} | 0.035\boldsymbol{0.035} | 0.310\boldsymbol{0.310} |
| KITTI Benchmark |  |  |  |  |  |  |
| Method | 𝜹𝟏\boldsymbol{\delta_{1}}↑\uparrow | 𝜹𝟐\boldsymbol{\delta_{2}}↑\uparrow | 𝜹𝟑\boldsymbol{\delta_{3}}↑\uparrow | AbsRel ↓\downarrow | RMS ↓\downarrow | RMS_log ↓\downarrow |
| Guo et al. [20] | 0.9020.902 | 0.9690.969 | 0.9860.986 | 0.0900.090 | 3.2583.258 | 0.1680.168 |
| VNL [66] | 0.938{0.938} | 0.990{0.990} | 0.998{0.998} | 0.072{0.072} | 3.2583.258 | 0.117{0.117} |
| TrDepth [63] | 0.9560.956 | 0.9940.994 | 0.9990.999 | 0.0640.064 | 2.7552.755 | 0.0980.098 |
| Adabins [4] | 0.9640.964 | 0.9950.995 | 0.9990.999 | 0.058¯\underline{0.058} | 2.3602.360 | 0.0880.088 |
| NeWCRFs [71] | 0.974\boldsymbol{0.974} | 0.997\boldsymbol{0.997} | 0.999¯\underline{0.999} | 0.052\boldsymbol{0.052} | 2.129\boldsymbol{2.129} | 0.079\boldsymbol{0.079} |
| Ours CSTM_image | 0.967¯\underline{0.967} | 0.995¯\underline{0.995} | 0.999\boldsymbol{0.999} | 0.0600.060 | 2.843{2.843} | 0.087¯\underline{0.087} |
| Ours CSTM_label | 0.964{0.964} | 0.993{0.993} | 0.998{0.998} | 0.0580.058 | 2.770{2.770} | 0.092{0.092} |

| Method | Basement_0001a | Bedroom_0015 | Dining_room_0004 | Kitchen_0008 | Classroom_0004 | Playroom_0002 | Office_0024 | Office_0004 | Dining_room_0033 |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow | C-l1l_{1}↓\downarrow | F-score ↑\uparrow |  |
| RCVD [27] | 0.364 | 0.276 | 0.074 | 0.582 | 0.462 | 0.251 | 0.053 | 0.620 | 0.187 | 0.327 | 0.791 | 0.187 | 0.324 | 0.241 | 0.646 | 0.217 | 0.445 | 0.253 |
| SC-DepthV2 [5] | 0.254 | 0.275 | 0.064 | 0.547 | 0.749 | 0.229 | 0.049 | 0.624 | 0.167 | 0.267 | 0.426 | 0.263 | 0.482 | 0.138 | 0.516 | 0.244 | 0.356 | 0.247 |
| DPSNet [23] | 0.243 | 0.299 | 0.195 | 0.276 | 0.995 | 0.186 | 0.269 | 0.203 | 0.296 | 0.195 | 0.141 | 0.485 | 0.199 | 0.362 | 0.210 | 0.462 | 0.222 | 0.493 |
| DPT [69] | 0.698 | 0.251 | 0.289 | 0.226 | 0.396 | 0.364 | 0.126 | 0.388 | 0.780 | 0.193 | 0.605 | 0.269 | 0.454 | 0.245 | 0.364 | 0.279 | 0.751 | 0.185 |
| LeReS [69] | 0.081 | 0.555 | 0.064 | 0.616 | 0.278 | 0.427 | 0.147 | 0.289 | 0.143 | 0.480 | 0.145 | 0.503 | 0.408 | 0.176 | 0.096 | 0.497 | 0.241 | 0.325 |
| Ours | 0.042 | 0.736 | 0.059 | 0.610 | 0.159 | 0.485 | 0.050 | 0.645 | 0.145 | 0.445 | 0.036 | 0.814 | 0.069 | 0.638 | 0.045 | 0.700 | 0.060 | 0.663 |

Evaluation on metric depth benchmarks. To evaluate the accuracy of predicted metric depth, firstly, we compare with state-of-the-art (SOTA) metric depth prediction methods on NYUv2 [47], KITTI [18]. We use the same model to do all evaluations. Results are reported in Tab. 1. Without any fine-tuning or metric adjustment, we can achieve comparable performance with SOTA methods, which are trained on benchmarks for hundreds of epochs.

Furthermore, We collect 66 unseen datasets to do more metric accuracy evaluation. These datasets contain a wide range of indoor and outdoor scenes, including rooms, buildings, and driving scenes. The camera models are also various, e.g. 7scenes has a short focal length (around 500), while ETH3D is 2000. We mainly compare with the SOTA metric depth estimation methods and take their NYUv2 and KITTI models for indoor and outdoor scenes evaluation respectively. From Tab. 3, we observe that although 7Scenes is similar to NYUv2 and NuScenes is similar to KITTI, existing methods face a noticeable performance decrease. In contrast, our model is more robust.

| Method | DIODE(Indoor) | iBIMS-1 | 7Scenes | DIODE(Outdoor) | ETH3D | NuScenes |
|---|---|---|---|---|---|---|
| Indoor scenes (AbsRel↓\downarrow/RMS↓\downarrow) | Outdoor scenes (AbsRel↓\downarrow/RMS↓\downarrow) |  |  |  |  |  |
| Adabins [4] | 0.443 / 1.963 | 0.212 / 0.901 | 0.218 / 0.428 | 0.865 / 10.35 | 1.271 / 6.178 | 0.445 / 10.658 |
| NewCRFs [71] | 0.404 / 1.867 | 0.206 / 0.861 | 0.240 / 0.451 | 0.854 / 9.228 | 0.890 / 5.011 | 0.400 / 12.139 |
| Ours_CSTM_label | 0.252 / 1.440 | 0.160 / 0.521 | 0.183 / 0.363 | 0.414 / 6.934 | 0.416 / 3.017 | 0.154 / 7.097 |
| Ours_CSTM_image | 0.268 / 1.429 | 0.144 / 0.646 | 0.189 / 0.388 | 0.535 / 6.507 | 0.342 / 2.965 | 0.147 / 5.889 |

| Method | Backbone | #Params | NYUv2 | KITTI | DIODE | ScanNet | ETH3D | Rank |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AbsRel↓\downarrow | δ1\delta_{1}↑\uparrow | AbsRel↓\downarrow | δ1\delta_{1}↑\uparrow | AbsRel↓\downarrow | δ1\delta_{1}↑\uparrow | AbsRel↓\downarrow | δ1\delta_{1}↑\uparrow | AbsRel↓\downarrow | δ1\delta_{1}↑\uparrow |  |  |  |  |
| DiverseDepth [64] | ResNeXt50 [60] | 25M | 0.1170.117 | 0.8750.875 | 0.1900.190 | 0.7040.704 | 0.3760.376 | 0.6310.631 | 0.1080.108 | 0.8820.882 | 0.2280.228 | 0.6940.694 | 7.77.7 |
| MiDaS [40] | ResNeXt101 | 88M | 0.1110.111 | 0.8850.885 | 0.2360.236 | 0.6300.630 | 0.3320.332 | 0.7150.715 | 0.1110.111 | 0.8860.886 | 0.1840.184 | 0.7520.752 | 7.27.2 |
| Leres [69] | ResNeXt101 |  | 0.0900.090 | 0.916{0.916} | 0.149{0.149} | 0.784{0.784} | 0.271{0.271} | 0.766{0.766} | 0.095{0.095} | 0.912{0.912} | 0.171{0.171} | 0.777{0.777} | 5.45.4 |
| Omnidata [13] | ViT-base |  | 0.074 | 0.945 | 0.149 | 0.835 | 0.339 | 0.742 | 0.077 | 0.935 | 0.166 | 0.778 | 4.94.9 |
| HDN [73] | ViT-Large [12] | 306M | 0.0690.069 | 0.9480.948 | 0.1150.115 | 0.8670.867 | 0.2460.246 | 0.7800.780 | 0.0800.080 | 0.9390.939 | 0.1210.121 | 0.8330.833 | 3.73.7 |
| DPT-large [39] | ViT-Large |  | 0.098 | 0.903 | 0.10 | 0.901 | 0.182 | 0.758 | 0.078 | 0.938 | 0.078 | 0.946 | 3.83.8 |
| Ours CSTM_image | ConvNeXt-large [33] | 198M | 0.058¯\underline{0.058} | 0.963¯\underline{0.963} | 0.053 | 0.965¯\underline{0.965} | 0.211¯\underline{0.211} | 0.825 | 0.074 | 0.942 | 0.064 | 0.965 | 1.31.3 |
| Ours CSTM_label | ConvNeXt-large |  | 0.050 | 0.966 | 0.058¯\underline{0.058} | 0.970 | 0.2240.224 | 0.805¯\underline{0.805} | 0.074 | 0.941¯\underline{0.941} | 0.066¯\underline{0.066} | 0.964¯\underline{0.964} | 1.81.8 |

Generalization over diverse scenes. Affine-invariant depth benchmarks decouple the scale’s effect, which aims to evaluate the model’s generalization ability to diverse scenes. Recent impact works, such as MiDaS, LeReS, and DPT, achieved promising performance on them. Following them, we test on 5 datasets and manually align the scale and shift to the ground-truth depth before evaluation. Results are reported in Tab. 4. Although our method enforces the network to recover more challenging metric information, our method outperforms them by a large margin on most datasets.

### 4.2 Applications Based on Our Method

In these experiments, we apply the CSTM_image model to various tasks.

3D scene reconstruction . To demonstrate our work can recover the 3D metric shape in the wild, we first do the quantitative comparison on 9 NYUv2 scenes, which are unseen during training. We predict the per-frame metric depth and then fuse them together with provided camera poses. Results are reported in Tab. 2. We compare with the video consistent depth prediction method (RCVD [27]), the unsupervised video depth estimation method (SC-DepthV2 [5]), the 3D scene shape recovery method (LeReS [69]), affine-invariant depth estimation method (DPT [39]), and the multi-view stereo reconstruction method (DPSNet [23]). Apart from DPSNet and our method, other methods have to align the scale with the ground truth depth for each frame. Although our method does not aim for the video or multi-view reconstruction problem, our method can achieve promising consistency between frames and reconstruct much more accurate 3D scenes than others on these zero-shot scenes. From the qualitative comparison in Fig. 6. our reconstructions have much less noise and outliers.

Dense-SLAM mapping. Monocular SLAM is an important robotics application. It only relies on a monocular video input to create the trajectory and dense 3D mapping. Owing to limited photometric and geometric constraints, existing methods face serious scale drift problems in large scenes and cannot recover the metric information. Our robust metric depth estimation method is a strong depth prior to the SLAM system. To demonstrate this benefit, we naively input our metric depth to the SOTA SLAM system, Droid-SLAM [52], and evaluate the trajectory on KITTI. We do not do any tuning on the original system. Trajectory comparisons are reported in Tab. 5. As Droid-SLAM can access accurate per-frame metric depth, like an RGB-D SLAM, the translation drift (tr​e​lt_{rel}) decreases significantly. Furthermore, with our depths, Droid-SLAM can perform denser and more accurate 3D mapping. An example is shown in Fig. 1 and more cases are shown in the supplementary materials.

We also test on the ETH3D SLAM benchmarks. Results are reported in Tab. 6. Droid with our depths has much better SLAM performance. As the ETH3D scenes are all small-scale indoor scenes, the performance improvement is less than that on KITTI.

| Method | Seq 00 | Seq 02 | Seq 05 | Seq 06 | Seq 08 | Seq 09 | Seq 10 |
|---|---|---|---|---|---|---|---|
| Translational RMS drift (tr​e​l,↓t_{rel},\downarrow) / Rotational RMS drift (rr​e​l,↓r_{rel},\downarrow) |  |  |  |  |  |  |  |
| GeoNet [70] | 27.6/5.72 | 42.24/6.14 | 20.12/7.67 | 9.28/4.34 | 18.59/7.85 | 23.94/9.81 | 20.73/9.1 |
| VISO2-M [49] | 12.66/2.73 | 9.47/1.19 | 15.1/3.65 | 6.8/1.93 | 14.82/2.52 | 3.69/1.25 | 21.01/3.26 |
| ORB-V2 [36] | 11.43/0.58 | 10.34/0.26 | 9.04/0.26 | 14.56/0.26 | 11.46/0.28 | 9.3/0.26 | 2.57/0.32 |
| Droid [52] | 33.9/0.29 | 34.88/0.27 | 23.4/0.27 | 17.2/0.26 | 39.6/0.31 | 21.7/0.23 | 7/0.25 |
| Droid+Ours | 1.44/0.37 | 2.64/0.29 | 1.44/0.25 | 0.6/0.2 | 2.2/0.3 | 1.63/0.22 | 2.73/0.23 |

|  | Einstein_global | Manquin4 | Motion1 | Plantscene3 | sfm_house_loop | sfm_lab_room2 |
|---|---|---|---|---|---|---|
|  | Average trajectory error (↓\downarrow) |  |  |  |  |  |
| Droid | 4.7 | 0.88 | 0.83 | 0.78 | 5.64 | 0.55 |
| Droid + Ours | 1.5 | 0.69 | 0.62 | 0.34 | 4.03 | 0.53 |
| Droid + GT | 0.7 | 0.006 | 0.024 | 0.006 | 0.96 | 0.013 |

Metrology in the wild. To show the robustness and accuracy of our recovered metric 3D, we download Flickr photos captured by various cameras and collect coarse camera intrinsic parameters from their metadata. We use our CSTM_image model to reconstruct their metric shape and measure structures’ sizes (marked in red in Fig. 7), while the ground-truth sizes are in blue. It shows that our measured sizes are very close to the ground-truth sizes.

### 4.3 Ablation Study

Ablation on canonical transformation. We study the effect of our proposed canonical transformation for the input images (‘CSTM_input’) and the canonical transformation for the ground-truth labels (‘CSTM_output’). Results are reported in Tab. 7. We train the model on sampled mixed data (55K images) and test it on 6 datasets. A naive baseline (‘Ours w/o CSTM’) is to remove CSTM modules and enforce the same supervision as ours. Without CSTM, the model is unable to converge when training on mixed metric datasets and cannot achieve metric prediction ability on zero-shot datasets. This is why recent mixed-data training methods compromise learning the affine-invariant depth to avoid metric issues. In contrast, our two CSTM methods both can enable the model to achieve the metric prediction ability, and they can achieve comparable performance. Tab. 1 also shows comparable performance. Therefore, both adjusting the supervision and the input image appearance during training can solve the metric ambiguity issues. Furthermore, we compare with CamConvs [15], which encodes the camera model in the decoder with a 4-channel feature. ‘CamConvs’ employ the same training schedule, model, and training data as ours. This method enforces the network to implicitly understand various camera models from the image appearance and then bridges the imaging size to the real-world size. We believe that this method challenges the data diversity and network capacity, thus their performance is worse than ours.

| Method | DDAD | Lyft | DS | NS | KITTI | NYU |
|---|---|---|---|---|---|---|
| Test set of train. data (AbsRel↓\downarrow) | Zero-shot test set (AbsRel↓\downarrow) |  |  |  |  |  |
| w/o CSTM | 0.5300.530 | 0.5820.582 | 0.3940.394 | 1.001.00 | 0.5680.568 | 0.5840.584 |
| CamConvs [15] | 0.2950.295 | 0.3150.315 | 0.2130.213 | 0.4230.423 | 0.1780.178 | 0.3330.333 |
| Ours CSTM_image | 0.1900.190 | 0.2350.235 | 0.1820.182 | 0.1970.197 | 0.0970.097 | 0.2100.210 |
| Ours CSTM_label | 0.1830.183 | 0.2210.221 | 0.2010.201 | 0.2130.213 | 0.0810.081 | 0.2120.212 |

Ablation on canonical space. We study the effect of the canonical camera here, i.e., the canonical focal length. We train the model on the small sampled dataset and test it on the validation set of training data and testing data. The average AbsRel error is calculated. We experiment on 3 different focal lengths, i.e., 500, 1000, 1500. Experiments show that f​o​c​a​l=1000focal=1000 has slightly better performance than others, see Fig. 8 for details. Thus we set the canonical focal length to 1000 in our experiments.

Effectiveness of the random proposal normalization loss. To show the effectiveness of our proposed random proposal normalization loss (RPNL), we experiment on the sampled small dataset. Results are shown in Tab. 8. We test on the DDAD, Lyft, DrivingStereo (DS), NuScenes (NS), KITTI, and NYUv2. The ‘baseline’ employs all losses except our RPNL. We compare it with ‘baseline + RPNL’ and ‘baseline + SSIL [40]’. We can observe that our proposed random proposal normalization loss can further improve the performance. In contrast, the scale-shift invariant loss [40], which does the normalization on the whole image, can only slightly improve the performance.

| Method | DDAD | Lyft | DS | NS | KITTI | NYUv2 |
|---|---|---|---|---|---|---|
| Test set of train. data (AbsRel↓\downarrow) | Zero-shot test set (AbsRel↓\downarrow) |  |  |  |  |  |
| baseline | 0.2040.204 | 0.2510.251 | 0.1840.184 | 0.2070.207 | 0.1040.104 | 0.2300.230 |
| baseline + SSIL [40] | 0.1970.197 | 0.2630.263 | 0.2590.259 | 0.2060.206 | 0.1050.105 | 0.2160.216 |
| baseline + RPNL | 0.190 | 0.235 | 0.182 | 0.197 | 0.097 | 0.210 |

## 5 Conclusion

In this paper, we tackle the problem of reconstructing the 3D metric scene from a single monocular image. To solve the depth ambiguity in image appearance caused by various focal lengths, we propose a canonical camera space transformation method. With our method, we can easily merge millions of data captured by 10k cameras to train one metric depth model. To improve the robustness, we collected over 88M data for training. Several zero-shot evaluations show the effectiveness and robustness of our work. We further show the ability to do metrology on randomly collected internet images and dense mapping on large-scale scenes.

## Acknowledgements

This work was in part supported by National Key R&D Program of China (No. 2022ZD0118700).

## 6 Appendix

### 6.1 Datasets and Training and Testing

We collect over 88M data from 11 public datasets for training. Datasets are listed in Tab. 9. The autonomous driving datasets, including DDAD [19], Lyft [24], DrivingStereo [62], Argoverse2 [55], DSEC [16], and Pandaset [59], have provided LiDar and camera intrinsic and extrinsic parameters. We project the LiDar to image planes to obtain ground-truth depths. In contrast, Cityscapes [10], DIML [9], and UASOL [2] only provide calibrated stereo images. We use draftstereo [32] to achieve pseudo ground-truth depths. Mapillary PSD [34] dataset provides paired RGB-D, but the depth maps are achieved from a structure-from-motion method. The camera intrinsic parameters are estimated from the SfM. We believe that such achieved metric information is noisy. Thus we do not enforce learning-metric-depth loss on this data, i.e., Ls​i​l​o​gL_{silog}, to reduce the effect of noises. For the Taskonomy [72] dataset, we follow LeReS [68] to obtain the instance planes, which are employed in the pair-wise normal regression loss. During training, we employ the training strategy from [67] to balance all datasets in each training batch.

The testing data is listed in Tab. 9. All of them are captured by high-quality sensors. In testing, we employ their provided camera intrinsic parameters to perform our proposed canonical space transformation.

| Datasets | Scenes | Label | Size | # Cam. |
|---|---|---|---|---|
| Training Data |  |  |  |  |
| DDAD [19] | Outdoor | LiDar | ∼\sim80K | 36+ |
| Lyft [24] | Outdoor | LiDar | ∼\sim50K | 6+ |
| Driving Stereo (DS) [62] | Outdoor | Stereo† | ∼\sim181K | 1 |
| DIML [9] | Outdoor | Stereo† | ∼\sim122K | 10 |
| Arogoverse2 [55] | Outdoor | LiDar | ∼\sim3515K | 6+ |
| Cityscapes [10] | Outdoor | Stereo† | ∼\sim170K | 1 |
| DSEC [16] | Outdoor | LiDar | ∼\sim26K | 1 |
| Mapillary PSD [34] | Outdoor | SfM‡ | 750K | 1000+ |
| Pandaset [59] | Outdoor | LiDar | ∼\sim48K | 6 |
| UASOL [2] | Outdoor | Stereo† | ∼\sim137K | 1 |
| Taskonomy [72] | Indoor | LiDar | ∼\sim4M | ∼\sim1M |
| Testing Data |  |  |  |  |
| NYU [47] | Indoor | Kinect | 654 | 1 |
| KITTI [17] | Outdoor | LiDar | 652 | 4 |
| ScanNet [11] | Indoor | Kinect | 700 | 1 |
| NuScenes (NS) [6] | Outdoor | LiDar | 10K | 6 |
| ETH3D [45] | Outdoor | LiDar | 431 | 1 |
| DIODE [53] | In/Out | LiDar | 771 | 1 |
| 7Scenes [46] | Indoor | Kinect | 17k | 1 |
| iBims-1 [26] | Indoor | LiDar | 100 | 1 |

- † ‘Stereo’: we use RaftStereo [32] to retrieve the pseudo ground truth.
‘Stereo’: we use RaftStereo [32] to retrieve the pseudo ground truth.

- ‡ ‘SfM’: pseudo ground truth is retrieved by structure from motion.
‘SfM’: pseudo ground truth is retrieved by structure from motion.

### 6.2 Details for Some Experiments

Evaluation of zero-shot 3D scene reconstruction. In this experiment, we use all methods’ released models to predict each frame’s depth and use the ground-truth poses and camera intrinsic parameters to reconstruct point clouds. When evaluating the reconstructed point cloud, we employ the iterative closest point (ICP) [3] algorithm to match the predicted point clouds with ground truth by a pose transformation matrix. Finally, we evaluate the Chamfer ℓ1\ell_{1} distance and F-score on the point cloud.

Reconstruction of in-the-wild scenes. We collect several photos from Flickr. From their associated camera metadata, we can obtain the focal length f^\hat{f} and the pixel size δ\delta. According to f^/δ\nicefrac{{\hat{f}}}{{\delta}}, we can obtain the pixel-represented focal length for 3D reconstruction and achieve the metric information. We use meshlab software to measure some structures’ size on point clouds. More visual results are shown in Fig. 11.

Generalization of metric depth estimation. To evaluate our method’s robustness of metric recovery, we test on 8 zero-shot datasets, i.e. NYU, KITTI, DIODE (indoor and outdoor parts), ETH3D, iBims-1, NuScenes, and 7Scenes. Details are reported in Tab. 9. We use the officially provided focal length to predict the metric depths. All benchmarks use the same depth model for evaluation. We don’t perform any scale alignment.

Evaluation on affine-invariant depth benchmarks. We follow existing affine-invariant depth estimation methods to evaluate 5 zero-shot datasets. Before evaluation, we employ the least square fitting to align the scale and shift with ground truth [69]. Previous methods’ performance is cited from their papers.

Dense-SLAM Mapping. This experiment is conducted on the KITTI odometry benchmark. We use our model to predict metric depths, and then naively input them to the Droid-SLAM system as an initial depth. We do not perform any finetuning but directly run their released codes on KITTI. With Droid-SLAM predicted poses, we unproject depths to the 3D point clouds and fuse them together to achieve dense metric mapping. More qualitative results are shown in Fig. 10.

### 6.3 More Visual Results

Reconstructing 360°NuScenes scenes. Current autonomous driving cars are equipped with several pin-hole cameras to capture 360°views. Capturing the surround-view depth is important for autonomous driving. We sampled some scenes from the testing data of NuScenes. With our depth model, we can obtain the metric depths for 6-ring cameras. With the provided camera intrinsic and extrinsic parameters, we unproject the depths to the 3D point cloud and merge all views together. See Fig. 12 for details. Note that 6-ring cameras have different camera intrinsic parameters. We can observe that all views’ point clouds can be fused together consistently.

Qualitative comparison of depth estimation. In Figs. 9, 13, 14, and 15, We show the qualitative comparison of depth maps with Adabins [4], NewCRFs [71], and Omnidata [13]. Our results have much less artifacts.
