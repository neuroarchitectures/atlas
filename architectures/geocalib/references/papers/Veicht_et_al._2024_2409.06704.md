# GeoCalib: Learning Single-image Calibration with Geometric Optimization

From a single image, visual cues can help deduce intrinsic and extrinsic camera parameters like the focal length and the gravity direction. This single-image calibration can benefit various downstream applications like image editing and 3D mapping. Current approaches to this problem are based on either classical geometry with lines and vanishing points or on deep neural networks trained end-to-end. The learned approaches are more robust but struggle to generalize to new environments and are less accurate than their classical counterparts. We hypothesize that they lack the constraints that 3D geometry provides. In this work, we introduce GeoCalib, a deep neural network that leverages universal rules of 3D geometry through an optimization process. GeoCalib is trained end-to-end to estimate camera parameters and learns to find useful visual cues from the data. Experiments on various benchmarks show that GeoCalib is more robust and more accurate than existing classical and learned approaches. Its internal optimization estimates uncertainties, which help flag failure cases and benefit downstream applications like visual localization. The code and trained models are publicly available at https://github.com/cvg/GeoCalib.

## 1 Introduction

Camera calibration consists of estimating the intrinsic and extrinsic parameters of a camera. This information is required for most image-based 3D applications, including metrology, 3D reconstruction, and novel view synthesis. This problem has been extensively studied, and many tools based on 3D geometry are available [51, 59, 73]. Since the process of image formation is well-understood, such tools can very accurately calibrate a camera from images taken in controlled lab conditions. The calibration can also be estimated in uncontrolled conditions, which generally requires additional sensors or multiple images observing the same scene, using structure-from-motion [74, 5, 60, 57] or SLAM [33, 96, 40].

In some applications, multiple images of the same scene are not available, such as in image editing, or multi-view constraints are not sufficient to accurately estimate the camera parameters, for example due to limited visual overlap across view. This occurs frequently when dealing with in-the-wild, crowd-sourced imagery, where each image is captured by a different camera [76, 75, 89, 3]. Visual cues visible in a single image can however help estimate some camera parameters, like the gravity direction, focal length, or distortion coefficients, without the need for multi-view cues.

Example of such geometric cues are straight lines, curves, and vanishing points. Estimating camera parameters from them has been extensively studied [50, 62, 10, 88, 6]. Because we have a good model of projective geometry, these approaches are extremely accurate. They are however limited to man-made environments in which straight lines are visible, and catastrophically fail when this condition is not met. Such low robustness significantly impairs their wide adoption.

Recent research has tackled the task of single-image calibration with deep neural networks (DNNs) trained in a supervised manner [52, 14, 45, 38, 77]. These approaches can leverage many more geometric and semantic cues and thus exhibit an impressively high robustness. To generalize well to different environment, they however require large amounts of training data that is costly to acquire. They are also far less accurate than their classical counterparts based on 3D geometry (Fig. 1). Intuitively, each deep network needs to relearn projective geometry from scratch when trained. Given finite model capacity, this can only be approximated within the domain of the training data, without any guarantee outside.

In this work, we introduce GeoCalib, a DNN that leverages our knowledge of projective geometry through an optimization process. As this optimization is differentiable, GeoCalib learns end-to-end to estimate the vertical direction and the camera intrinsics given a single image. Our approach can thus learn the right visual cues without explicit supervision but does not need to learn the process of estimating camera parameters, which is better achieved with knowledge of 3D geometry. This improves the generalization to different environments with negligible overhead, which is important for practical applications. Experiments on various benchmarks show that GeoCalib is both more robust and more accurate than existing classical and learned approaches.

Compared to black-box deep networks, GeoCalib has multiple practical benefits. When a subset of the parameters is known, e.g. the intrinsic parameters, GeoCalib can more accurately estimate the remaining parameters by leveraging this prior information. This makes it possible to handle different camera models, such as pinhole and fisheye, without any retraining. GeoCalib is also more interpretable: we can easily visualize the cues that it relies on, and the optimization uncertainties help flag failure cases and can benefit downstream applications. To support this, we show that GeoCalib can readily improve the accuracy of visual positioning.

## 2 Related work

for single-image calibration rely on parallel lines [31] that intersect at a vanishing point (VP) in the image. VP estimation in a single image can be categorized in two directions: i) unconstrained VP detection [12, 42, 94] which relies on coplanar lines and ii) approaches that assume a Manhattan world (3 orthogonal directions) [81, 10, 21] that is often found in man-made environments but not in natural ones. The vertical VP generally corresponds to the gravity direction. VP detection has been extensively studied with approaches based on RANSAC [50, 10, 88, 6], optimization [11, 82], exhaustive search [56, 66, 9], and even deep learning [81, 7, 48, 95]. Some approaches also recover the focal length [62, 50] or the distortion parameters, using curved lines, circular arcs, or covariant regions [50, 65]. Our approach also solves an optimization based on low-level cues, including but not limited to lines, making it much more robust in natural and outdoor scenes.

has recently gained popularity. Existing research is divided into two categories: direct and indirect methods. Direct methods directly regress the camera parameters (focal length, distortion, horizon line) or classify them into bins [52, 36, 94]. This simple approach is typically less accurate than traditional methods because of missing geometric constraints. Indirect methods aim to alleviate this issue by predicting observable, lower-level cues. This includes classifying horizontal/vertical lines [45, 77] and regressing surface normals [90] or vector fields [38]. These cues are often used as auxiliary supervision [77, 45] or as inputs to a decoder network that regresses camera parameters [38]. Compared to classical algorithms, DNNs are usually more robust because they do not assume any scene configuration. They however fail to match their accuracy when sufficient geometric constraints are available. In this work, we combine deep priors with robust optimization, which enables the network to focus on well-constrained priors, boosting the estimation accuracy.

has been recently popular in geometric computer vision, including for pose estimation [17, 71, 19, 30], camera tracking [20, 54, 79, 93, 80], and image matching [69, 18] or alignment [85, 63]. In these, a DNN typically predicts observations on the image level, based on which an optimization problem is solved. It can be made differentiable by unrolling a fixed number of optimization steps or using implicit derivatives [67]. The DNN can be trained by only supervising the optimized quantity. The robustness to incorrect priors can be improved with carefully selected loss functions or by formulating the optimization through differentiable RANSAC [16, 13, 87].

For the task of single-image calibration, UprightNet [90] is closest to our work. It aligns normals in camera and world frames to estimate the gravity direction by solving a simpler weighted least squares problem with closed form solution. It cannot recover the camera intrinsics and its training requires ground-truth normals. In contrast, we use cues that are solely derived from the camera parameters and sufficient to constraint all of them. We show that the differentiable optimization improves both the generalization and the accuracy of GeoCalib.

## 3 Single-image calibration with GeoCalib

A 3D point 𝐏=[XYZ]⊤{\bm{\mathrm{P}}}=\begin{bmatrix}X&Y&Z\end{bmatrix}^{\top}, expressed in the coordinate frame of the camera, maps to an image point 𝐩∈ℝ2{\bm{\mathrm{p}}}\,{\in}\,\mathbb{R}^{2} with the projection function Π⁡(⋅)\Pi\left(\cdot\right). Assuming square pixels and no skew, we write for a pinhole camera Π⁡(𝐏)=f​[uv]⊤+𝐜\Pi\left({\bm{\mathrm{P}}}\right)=f\begin{bmatrix}u&v\end{bmatrix}^{\top}+\bm{\mathrm{c}}, where [uv]=1Z​[XY]\begin{bmatrix}u&v\end{bmatrix}=\frac{1}{Z}\begin{bmatrix}X&Y\end{bmatrix} are the normalized image coordinates, ff is the focal length in pixels, and 𝐜\bm{\mathrm{c}} is the principal point, which can often be assumed to be at the center of the image unless it has been cropped.

To handle lens distortion, various models are commonly used [27, 64, 72, 83, 84, 29, 39]. Many of them, including those based on polynomials, transform [uv]\begin{bmatrix}u&v\end{bmatrix} into [udvd]:=𝒟⁡([uv],𝐤)\begin{bmatrix}u_{d}&v_{d}\end{bmatrix}:=\mathcal{D}\left(\begin{bmatrix}u&v\end{bmatrix},\bm{\mathrm{k}}\right), where 𝐤∈ℝD\bm{\mathrm{k}}\in\mathbb{R}^{D} are DD distortion parameters. The undistortion function 𝒟−1\mathcal{D}^{-1} can be computed using an iterative process [22] or inverse parameters [26]. We generally refer to ff, 𝐜\bm{\mathrm{c}}, and 𝐤\bm{\mathrm{k}} as intrinsic parameters.

The gravity direction in the camera coordinate frame is the unit vector 𝐠=[𝐠x𝐠y𝐠z]\bm{\mathrm{g}}=\begin{bmatrix}\bm{\mathrm{g}}_{x}&\bm{\mathrm{g}}_{y}&\bm{\mathrm{g}}_{z}\end{bmatrix}. It can be decomposed into two Euler angles, the roll, and the pitch. The third degree of freedom of the rotation, the yaw, is generally not observable from a single image unless one makes additional restrictive assumptions, e.g. Manhattan-world. Our goal is to estimate 𝐠\bm{\mathrm{g}}, ff, and 𝐤\bm{\mathrm{k}}, collectively referred to as 𝜽\bm{\mathrm{\theta}}, given a single image 𝐈∈ℝH×W\bm{\mathrm{I}}\in\mathbb{R}^{H{\times}W} as input.

GeoCalib first infers visual cues – a Perspective Field, with associated confidences, from the input image with a DNN. An iterative optimization then fits the camera parameters 𝜽\bm{\mathrm{\theta}} to be consistent with these cues (Fig. 2, Sec. 3.1). The DNN is then trained end-to-end by supervising the result of the optimization. We detail in Sec. 3.2 the multiple practical benefits of this simple approach.

### 3.1 Perspective alignment

We choose the Perspective Field [38] as visual cues driving the optimization. It is an over-parameterized, pixel-wise representation of the camera parameters and consists, for each pixel 𝐩{\bm{\mathrm{p}}}, of a unit up-vector 𝐮𝐩\bm{\mathrm{u}}_{\bm{\mathrm{p}}} and a latitude φ𝐩\varphi_{\bm{\mathrm{p}}}. The up-vector is the projection of the up direction at 𝐏{\bm{\mathrm{P}}}, while the latitude φ𝐩\varphi_{\bm{\mathrm{p}}} is the angle between the ray 𝐧=[uv1]⊤\bm{\mathrm{n}}=\begin{bmatrix}u&v&1\end{bmatrix}^{\top} at pixel 𝐩{\bm{\mathrm{p}}} and the horizontal plane:

|  | 𝐮𝐩=limt→0Π⁡(𝐏−t​𝐠)−Π⁡(𝐏)‖Π⁡(𝐏−t​𝐠)−Π⁡(𝐏)‖2φ𝐩=arcsin⁡(𝐧⊤​𝐠‖𝐧‖2)\bm{\mathrm{u}}_{{\bm{\mathrm{p}}}}=\lim_{t\rightarrow 0}\frac{\Pi({\bm{\mathrm{P}}}-t\bm{\mathrm{g}})-\Pi({\bm{\mathrm{P}}})}{\left\lVert\Pi({\bm{\mathrm{P}}}-t\bm{\mathrm{g}})-\Pi({\bm{\mathrm{P}}})\right\rVert_{2}}\quad\quad\quad\varphi_{{\bm{\mathrm{p}}}}=\arcsin\left(\frac{\bm{\mathrm{n}}^{\top}\bm{\mathrm{g}}}{\left\lVert\bm{\mathrm{n}}\right\rVert_{2}}\right) |  | (1) |
|---|---|---|---|

This representation is visually observable: the up-vector corresponds to the direction of vertical lines, and the latitude is 0​°0\degree on the horizon. It is not only geometric, but also semantic: the up-vector can be deduced from objects known to have a canonical orientation, like trees and humans. Unlike purely geometric cues like lines, it can be inferred in a wider range of environments. It is applicable to arbitrary camera models, though Jin et al. [38] mainly focused on perspective projection. Here we also apply it to distorted images.

We predict a pixel-wise Perspective Field (𝐮^𝐩,φ^𝐩)(\hat{\bm{\mathrm{u}}}_{\bm{\mathrm{p}}},\hat{\varphi}_{\bm{\mathrm{p}}}) from the input image using a DNN. We employ a SegNeXt [32] encoder-decoder architecture, modified such that the outputs have the same spatial resolutions as the input. This provides more accurate constraints for the subsequent optimization.

up-vector σ𝐮\sigma_{\bm{\mathrm{u}}}

latitude σφ\sigma_{\varphi}

Predicting the Perspective Field is easy in some areas, such as near the horizon or vertical lines, but difficult in others, such as in texture-less areas. It needs to be accurate only for a subset of pixels to sufficiently constrain the camera parameters. The neural network thus also predicts a pixel-wise confidence (σ𝐮𝐩,σφ𝐩)∈[0,1](\sigma_{\bm{\mathrm{u}}_{\bm{\mathrm{p}}}},\sigma_{\varphi_{\bm{\mathrm{p}}}})\in[0,1] for each component of the Perspective Field. The confidences are not directly supervised, but rather learned implicitly to help the optimization converge to the right solution. Figure 3 shows the predicted confidence after training. The up-vector is not limited to vertical lines; it is also confidently predicted on upright objects and curves in distorted images.

We then find the camera parameters 𝜽\bm{\mathrm{\theta}} that generate a Perspective Field (𝐮⁡(𝜽),φ⁡(𝜽))\left(\bm{\mathrm{u}}(\bm{\mathrm{\theta}}),\varphi(\bm{\mathrm{\theta}})\right) similar to the one that is observed (𝐮^,φ^)(\hat{\bm{\mathrm{u}}},\hat{\varphi}). This amounts to minimizing the objective function

|  | E⁡(𝜽)=∑𝐩∈H×Wσ𝐮𝐩​∥𝐮𝐩​(𝜽)−𝐮^𝐩⏟𝐫𝐮𝐩∥22+σφ𝐩​∥sin⁡φ𝐩​(𝜽)−sin⁡φ^𝐩⏟𝐫φ𝐩∥22.E(\bm{\mathrm{\theta}})=\sum_{{\bm{\mathrm{p}}}\in{H{\times}W}}\sigma_{\bm{\mathrm{u}}_{\bm{\mathrm{p}}}}\lVert\underbrace{\bm{\mathrm{u}}_{\bm{\mathrm{p}}}\left(\bm{\mathrm{\theta}}\right)-\hat{\bm{\mathrm{u}}}_{\bm{\mathrm{p}}}}_{\bm{\mathrm{r}}_{\bm{\mathrm{u}}_{\bm{\mathrm{p}}}}}\rVert_{2}^{2}+\sigma_{\varphi_{\bm{\mathrm{p}}}}\lVert\underbrace{\sin\varphi_{\bm{\mathrm{p}}}\left(\bm{\mathrm{\theta}}\right)-\sin\hat{\varphi}_{\bm{\mathrm{p}}}}_{\bm{\mathrm{r}}_{\varphi_{\bm{\mathrm{p}}}}}\rVert_{2}^{2}\kern 5.0pt. |  | (2) |
|---|---|---|---|

This requires an explicit expression of the up-vector for the distortion camera model by solving for the limit of Eq. 1. We notice that the up-vector is collinear with the directional derivative of the projection function along the up direction, i.e. 𝐮𝐩∝−∇Π(𝐏)⋅𝐠\bm{\mathrm{u}}_{\bm{\mathrm{p}}}\propto-\nabla\Pi({\bm{\mathrm{P}}})\cdot\bm{\mathrm{g}}. For a radial distortion model such that 𝒟⁡([uv],𝐤)=d⁡(u,v,𝐤)​[uv]\mathcal{D}\left(\begin{bmatrix}u&v\end{bmatrix},\bm{\mathrm{k}}\right)=d(u,v,\bm{\mathrm{k}})\begin{bmatrix}u&v\end{bmatrix}, the up-vector can be expressed as

|  | 𝐮𝐩​(f,𝐠,𝐤)∝(1+1d⁡(u,v,𝐤)​[uv]​∂d⁡(u,v,𝐤)∂(u,v))​[u​𝐠z−𝐠xv​𝐠z−𝐠y].\bm{\mathrm{u}}_{\bm{\mathrm{p}}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)\propto\left(1+\frac{1}{d(u,v,\bm{\mathrm{k}})}\begin{bmatrix}u\\ v\end{bmatrix}\frac{\partial d(u,v,\bm{\mathrm{k}})}{\partial(u,v)}\right)\begin{bmatrix}u\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{x}\\ v\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{y}\end{bmatrix}\kern 5.0pt. |  | (3) |
|---|---|---|---|

This can be derived similarly for other distortion models, as shown in Appendix 0.D.

Since Eq. 2 describes a non-linear least-squares problem, we find its minimum with the Levenberg–Marquardt (LM) algorithm [46, 58]. We parametrize the gravity 𝐠\bm{\mathrm{g}} on the S2S^{2} manifold [34, 35] since it has unit norm and the focal length by log⁡f\log f since it is strictly positive. The distortion parameters generally live in the Euclidean space, unless constrained by the camera model. At each iteration, we compute the update 𝜹∈ℝ3+D\bm{\mathrm{\delta}}\in\mathbb{R}^{3{+}D} by solving the linear system

|  | 𝜹=−(𝐇+λ​diag⁡(𝐇))−1​𝐉⊤​𝐖𝐫with𝐉=∂𝐫∂𝜽and𝐇=𝐉⊤​𝐖𝐉,\bm{\mathrm{\delta}}=-\left(\bm{\mathrm{H}}+\lambda\operatorname{diag}\left(\bm{\mathrm{H}}\right)\right)^{-1}\bm{\mathrm{J}}^{\top}\bm{\mathrm{W}}\bm{\mathrm{r}}\quad\text{with}\quad\bm{\mathrm{J}}=\frac{\partial\bm{\mathrm{r}}}{\partial\bm{\mathrm{\theta}}}\quad\text{and}\quad\bm{\mathrm{H}}=\bm{\mathrm{J}}^{\top}\bm{\mathrm{W}}\bm{\mathrm{J}}\kern 5.0pt, |  |
|---|---|---|

where 𝐫\bm{\mathrm{r}} stacks the residuals 𝐫𝐮𝐩\bm{\mathrm{r}}_{\bm{\mathrm{u}}_{\bm{\mathrm{p}}}} and 𝐫φ𝐩\bm{\mathrm{r}}_{\varphi_{\bm{\mathrm{p}}}} and 𝐖\bm{\mathrm{W}} is a weight matrix whose diagonal stacks the corresponding confidences σ𝐮\sigma_{\bm{\mathrm{u}}} and σφ\sigma_{\varphi}. The damping factor λ\lambda interpolates between the Gauss-Newton (λ=0\lambda{=}0) and gradient descent (λ→∞\lambda\!\!\to\!\!\infty) strategies. It is usually adjusted at each iteration using heuristics [46, 58, 55]. We derive 𝐉\bm{\mathrm{J}} analytically and provide it in Appendix 0.E. The optimization stops when the update 𝜹\bm{\mathrm{\delta}} is sufficiently small.

The LM algorithm requires an initial guess. Since the objective function is not necessarily convex, the choice of initialization should matter. We consider several strategies. The trivial strategy initializes 𝐠=[010]\bm{\mathrm{g}}{=}\begin{bmatrix}0&1&0\end{bmatrix}, 𝐤=𝟎\bm{\mathrm{k}}{=}\bm{\mathrm{0}}, and f=0.7​max⁡(W,H)f{=}0.7\max\left(W,H\right), assuming a common sensor size. Other, more complex strategies are described in Appendix 0.B. In practice, we have found that neither of them improves the performance over the trivial initialization.

The LM optimization is differentiable, either by unrolling its steps and applying backpropagation to each of them [71, 80] or by leveraging implicit derivatives [67]. GeoCalib is thus end-to-end differentiable, from the estimated camera parameters to the input image. We train it by supervising the estimated parameters 𝜽^\hat{\bm{\mathrm{\theta}}} but also the intermediate perspective field, which accelerates the convergence of the training. This translates into the loss function

|  | ℒ=∥𝜽^−𝜽¯∥γ+β​∑𝐩∈H×Wσ𝐮𝐩​∥𝐮^𝐩−𝐮​(𝜽¯)𝐩∥γ+σφ𝐩​∥φ^𝐩−φ​(𝜽¯)𝐩∥γ,\mathcal{L}=\lVert\hat{\bm{\mathrm{\theta}}}-\bar{\bm{\mathrm{\theta}}}\rVert_{\gamma}+\beta\sum_{{\bm{\mathrm{p}}}\in{H{\times}W}}\sigma_{\bm{\mathrm{u}}_{{\bm{\mathrm{p}}}}}\lVert\hat{\bm{\mathrm{u}}}_{{\bm{\mathrm{p}}}}-\bm{\mathrm{u}}(\bar{\bm{\mathrm{\theta}}})_{{\bm{\mathrm{p}}}}\rVert_{\gamma}+\sigma_{\varphi_{{\bm{\mathrm{p}}}}}\lVert\hat{\varphi}_{{\bm{\mathrm{p}}}}-\varphi(\bar{\bm{\mathrm{\theta}}})_{{\bm{\mathrm{p}}}}\rVert_{\gamma}\kern 5.0pt, |  | (4) |
|---|---|---|---|

where 𝜽¯\bar{\bm{\mathrm{\theta}}} are the ground-truth parameters, γ\gamma is the L1 loss, and β\beta balances the two terms. The confidences are supervised only by the LM optimization and their gradient is detached in Eq. 4. They cancel the second term in areas that the network deems not useful for the optimization. No capacity is thus wasted to unnecessarily improve the prediction of the Perspective Field for all pixels.

### 3.2 Practical benefits

Because GeoCalib is based on well-understood optimization processes, it can be flexibly adapted to the constraints of real-world applications. None of the existing learned approaches provide such a degree of flexibility.

Unlike most existing learned approaches, our formulation is not tied to a specific distortion model. One can train GeoCalib with a given model but optimize a different one at inference time without any retraining, as long as the Perspective Field is similarly observable. This is a significant gain given the large number of different distortion models used in practice. Our experiments show that a version of GeoCalib trained only on pinhole images can more accurately estimate a radial distortion than all existing approaches that operate on a single image. This makes it possible to handle heterogeneous datasets captured by many different cameras.

A subset of the camera parameters might sometimes be already available. This includes camera intrinsics from a factory calibration, EXIF metadata, or structure-from-motion or a gravity direction estimated by an inertial measurement unit or assumed constant when the motion is restricted to be planar. Our optimization can leverage this information as a hard constraint, by fixing the associated parameters, or as a soft prior, by biasing the parameters with a simple regularization term, in which case the uncertainty of the priors can be leveraged. Our experiments show that such information directly improves the estimation accuracy of the remaining parameters.

We can also couple the optimization across multiple images. For example, images captured by the same cameras share identical intrinsics, which should be optimized jointly, while each image has its own gravity direction. Unlike classical geometry tools, this does not require any covisibility across the views. Differently, images taken by a multi-camera rig have distinct intrinsics but have a common gravity direction, given known relative poses.

The uncertainty of parameters estimated by the LM algorithm can be easily obtained via their covariance matrix, 𝚺𝜽=𝐇−1\bm{\mathrm{\Sigma}}_{\bm{\mathrm{\theta}}}=\bm{\mathrm{H}}^{-1} computed at convergence. It can be further decomposed into an uncertainty per parameter 𝐠\bm{\mathrm{g}}, ff, and 𝐤\bm{\mathrm{k}}. This uncertainty can flag examples for which the estimate is likely wrong, e.g. due to poor geometric constraints in the optimization or excessively low-confident visual cues, for example due to occlusion, motion blur, or fully texture-less view (Fig. 4). This makes it easier to integrate the estimates of GeoCalib into a larger probabilistic framework, such as a factor graph [23, 24].

0.5​°0.5\degree/1.2​°/1.2\degree

0.9​°0.9\degree/1.4​°/1.4\degree

2.1​°2.1\degree/2.2​°/2.2\degree

3.4​°3.4\degree/3.2​°/3.2\degree

3.4​°3.4\degree/4.7​°/4.7\degree

5.3​°5.3\degree/5.4​°/5.4\degree

7.3​°7.3\degree/9.5​°/9.5\degree

8.6​°8.6\degree/10.0​°/10.0\degree

0.2​°0.2\degree/0.7​°/0.7\degree

0.7​°0.7\degree/1.0​°/1.0\degree

1.3​°1.3\degree/1.5​°/1.5\degree

2.4​°2.4\degree/2.6​°/2.6\degree

4.3​°4.3\degree/4.8​°/4.8\degree

4.5​°4.5\degree/6.0​°/6.0\degree

6.2​°6.2\degree/6.0​°/6.0\degree

7.6​°7.6\degree/9.6​°/9.6\degree

Our work is inspired by Jin et al. [38], who train a DNN to extract camera parameters from a pre-trained Perspective Field. This black-box approach is sensitive to the training distribution and lacks the flexibility described above. They also propose to refine its estimates with a first-order optimization based on Adam [41]. We found this to converge much slower than our LM optimization. Training it end-to-end and learning appropriate confidences is thus not practical. Unlike our approach, this optimization is sensitive to the initialization and yields marginal accuracy gains. We demonstrate the value of our improvements in an ablation study.

a) ground-truth

b) final prediction

c) observed latitude

d) observed up-vect.

## 4 Implementation

As prior works relied on proprietary datasets [36, 52, 38], reproducing them has been difficult. We have therefore assembled a new dataset, OpenPano, of panoramas that are publicly available from various sources [1, 2, 15]. OpenPano consists of 2.9k panoramas split into 2.6k/147/148 for training/validation/testing. Sampling 16 square images per panorama yields about 42k training images. Compared to the dataset used in [38], ours is much smaller but exhibits a better balance between indoor and outdoor scenes. It will be made publicly available to facilitate future research. See Sec. 0.F.1 for more details.

We trained variants of GeoCalib on the pinhole and distorted images from OpenPano, each for 49 epochs, with a learning rate of 10−410^{-4} and a batch size of 24. We linearly warm up the learning rate in the first 1000 steps and drop it after 25 and 42 epochs by a factor of 10. The LM optimization is stopped after 10 iterations with an initial damping λ=0.1\lambda{=}0.1. The entire training takes one day on two RTX 4090 GPUs. Calibrating a single image takes about 100ms.

|  | Approach | Roll [degrees] | Pitch [degrees] | FoV [degrees] |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow |  |  |  |  |  |  |  |
| Stanford2D3D [8] | DeepCalib* [52] | 01.59 | 33.8 | 63.9 | 79.2 | 02.58 | 21.6 | 46.9 | 65.7 | 06.67 | 08.1 | 20.6 | 37.6 |
| Perceptual [36] | 02.08 | 26.8 | 53.8 | 70.7 | 03.17 | 21.5 | 41.8 | 57.8 | 13.84 | 02.8 | 07.7 | 16.1 |  |
| CTRL-C [45] | 03.04 | 23.2 | 43.0 | 56.9 | 03.43 | 18.3 | 38.6 | 53.8 | 08.50 | 07.7 | 18.2 | 31.5 |  |
| MSCC [77] | 03.43 | 13.5 | 36.8 | 57.3 | 02.64 | 22.6 | 45.0 | 60.5 | 05.81 | 09.6 | 23.8 | 41.6 |  |
| ParamNet [38] | 02.52 | 20.6 | 48.5 | 68.1 | 02.78 | 20.9 | 44.2 | 61.5 | 07.79 | 07.4 | 18.0 | 33.2 |  |
| ParamNet* [38] | 01.14 | 44.6 | 73.9 | 84.8 | 01.94 | 29.2 | 56.7 | 73.1 | 09.01 | 05.8 | 14.3 | 27.8 |  |
| SVA [50] | - | 21.7 | 24.6 | 25.8 | - | 15.4 | 19.9 | 22.4 | - | 06.2 | 11.5 | 15.2 |  |
| UVP [62] | 00.52 | 65.3 | 74.6 | 79.1 | 00.95 | 51.2 | 63.0 | 69.2 | 03.65 | 22.2 | 39.5 | 51.3 |  |
| GeoCalib* | 00.40 | 83.1 | 91.8 | 94.8 | 00.93 | 52.3 | 74.8 | 84.6 | 03.21 | 17.4 | 40.0 | 59.4 |  |
| TartanAir [86] | DeepCalib* [52] | 01.95 | 24.7 | 55.4 | 71.5 | 03.27 | 16.3 | 38.8 | 58.5 | 08.07 | 01.5 | 08.8 | 27.2 |
| Perceptual [36] | 02.24 | 23.2 | 48.6 | 66.7 | 02.86 | 23.5 | 44.6 | 61.5 | 15.06 | 05.1 | 08.9 | 17.1 |  |
| CTRL-C [45] | 01.68 | 32.8 | 59.1 | 74.1 | 02.39 | 24.6 | 48.6 | 65.2 | 05.64 | 10.7 | 25.4 | 43.5 |  |
| MSCC [77] | 03.50 | 15.0 | 37.2 | 57.7 | 03.48 | 18.8 | 38.6 | 54.3 | 11.18 | 04.4 | 11.8 | 23.0 |  |
| ParamNet [38] | 02.33 | 23.3 | 51.4 | 71.0 | 02.87 | 19.9 | 43.8 | 62.9 | 06.04 | 08.5 | 22.5 | 40.8 |  |
| ParamNet* [38] | 01.63 | 34.5 | 59.2 | 73.9 | 03.05 | 19.4 | 42.0 | 60.3 | 08.21 | 06.0 | 16.8 | 31.6 |  |
| SVA [50] | 09.48 | 32.4 | 39.6 | 44.1 | 18.46 | 21.2 | 28.8 | 34.5 | 43.01 | 08.8 | 16.1 | 21.6 |  |
| UVP [62] | 00.89 | 52.1 | 64.8 | 71.9 | 02.48 | 36.2 | 48.8 | 58.6 | 09.15 | 15.8 | 25.8 | 35.7 |  |
| GeoCalib* | 00.43 | 71.3 | 83.8 | 89.8 | 01.49 | 38.2 | 62.9 | 76.6 | 04.90 | 14.1 | 30.4 | 47.6 |  |
| MegaDepth [47] | DeepCalib* [52] | 01.41 | 34.6 | 65.4 | 79.4 | 05.19 | 11.9 | 27.8 | 44.8 | 11.14 | 05.6 | 12.1 | 22.9 |
| Perceptual [36] | 01.07 | 47.9 | 72.4 | 83.2 | 03.49 | 19.8 | 39.1 | 54.2 | 13.40 | 02.9 | 08.2 | 16.8 |  |
| CTRL-C [45] | 00.88 | 54.5 | 75.0 | 84.2 | 04.80 | 16.6 | 33.2 | 46.5 | 18.65 | 02.0 | 05.8 | 12.8 |  |
| MSCC [77] | 00.90 | 53.1 | 72.8 | 82.1 | 05.73 | 19.0 | 33.2 | 44.3 | 10.80 | 06.0 | 14.6 | 26.2 |  |
| ParamNet [38] | 01.46 | 37.0 | 66.4 | 80.8 | 03.53 | 15.8 | 37.3 | 57.1 | 10.98 | 05.3 | 12.8 | 24.0 |  |
| ParamNet* [38] | 01.17 | 43.4 | 70.7 | 82.2 | 03.99 | 15.4 | 34.5 | 53.3 | 11.01 | 03.2 | 10.1 | 21.3 |  |
| SVA [50] | - | 31.9 | 35.0 | 36.2 | - | 13.6 | 20.6 | 24.9 | - | 09.4 | 16.1 | 21.1 |  |
| UVP [62] | 00.51 | 69.2 | 81.6 | 86.9 | 04.59 | 21.6 | 36.2 | 47.4 | 10.92 | 08.2 | 18.7 | 29.8 |  |
| GeoCalib* | 00.36 | 82.6 | 90.6 | 94.0 | 01.94 | 32.4 | 53.3 | 67.5 | 04.46 | 13.6 | 31.7 | 48.2 |  |
| LaMAR [70] | DeepCalib* [52] | 01.15 | 44.1 | 73.9 | 84.8 | 04.68 | 10.8 | 28.3 | 49.8 | 10.93 | 00.7 | 13.0 | 24.0 |
| Perceptual [36] | 01.29 | 40.0 | 68.9 | 81.6 | 02.83 | 21.2 | 44.7 | 62.6 | 17.78 | 03.0 | 05.3 | 10.7 |  |
| CTRL-C [45] | 01.20 | 43.5 | 70.9 | 82.5 | 01.94 | 27.6 | 54.7 | 70.2 | 05.64 | 09.8 | 24.6 | 43.2 |  |
| MSCC [77] | 01.44 | 39.6 | 60.7 | 72.8 | 03.02 | 20.9 | 41.8 | 55.7 | 14.78 | 03.2 | 08.3 | 16.8 |  |
| ParamNet [38] | 01.30 | 38.7 | 69.4 | 82.8 | 02.77 | 19.0 | 44.7 | 65.7 | 15.15 | 01.8 | 06.2 | 13.2 |  |
| ParamNet* [38] | 00.93 | 51.7 | 77.0 | 86.0 | 02.15 | 27.0 | 52.7 | 70.2 | 14.71 | 02.8 | 06.8 | 14.3 |  |
| SVA [50] | - | 08.6 | 09.2 | 09.7 | - | 03.4 | 05.7 | 07.0 | - | 01.2 | 02.7 | 04.1 |  |
| UVP [62] | 00.38 | 72.7 | 81.8 | 85.7 | 01.34 | 42.3 | 59.9 | 69.4 | 05.57 | 15.6 | 30.6 | 43.5 |  |
| GeoCalib* | 00.28 | 86.4 | 92.5 | 95.0 | 00.87 | 55.0 | 76.9 | 86.2 | 03.03 | 19.1 | 41.5 | 60.0 |  |

## 5 Experiments

We evaluate GeoCalib on single-image calibration and radial distortion estimation in a wide range of scenarios. Experiments are performed on a diverse range of real-world images (indoor, outdoor, and natural environments), and we analyze the impacts of our design decisions. We show some qualitative examples in Fig. 5. We also evaluate our gravity estimation as a prior for visual localization.

### 5.1 Gravity and Field-of-View estimation

We first compare GeoCalib to existing deep and classical approaches with pinhole images from four real-world datasets.

Following [38], we evaluate uncalibrated pinhole scenarios, i.e. no lens distortion. For each image, we evaluate the gravity estimation in terms of angular roll and pitch errors (in degrees), and the focal length in terms of the vertical field-of-view error (FoV, in degrees). For each metric, we report the median error and the Area Under the recall Curve (AUC) up to 1/5/10°.

We conduct this experiment on four popular datasets not seen during training. i) Stanford2D3D [8] consists of images samples from 360° panoramas captured inside university buildings. ii) TartanAir [86] provides images from photo-realistic simulated environments and captures dynamic objects and various appearance changes. Its images exhibit large variations in gravity directions but share the same FoV. For both datasets, we use the same test set as [38], with about 2k images each. iii) The MegaDepth dataset [47] was built from crowd-sourced images covering popular phototourism landmarks using SfM. We align the respective 3D models to gravity using COLMAP [74] and sample a total of 2k images with varying intrinsics from the scenes in the IMC 2021 test set [37]. iv) LaMAR [70] is an AR dataset for visual localization, captured over multiple years in university buildings and a city center. We use 2k images from the phone sequences.

We benchmark our method against the deep methods DeepCalib [52], CTRL-C [45], Perceptual [36], MSCC [77] and ParamNet [38]. We follow the suggested evaluation setup of each method and refer to Appendix 0.F for more details. For fairness, we also retrain DeepCalib [52] and ParamNet [38] on our dataset. We also evaluate two recent classical approaches based on VPs: SVA [50] and UVP [62], which both assume a Manhattan world (3 orthogonal vanishing points) and rely on minimal solvers for VP estimation.

Table 1 shows that GeoCalib largely improves on top of all deep single-image calibration networks, and outperforms classical methods in all metrics, except for the finest threshold on FoV on TartanAir [86] and Stanford2D3D [8]. UVP [62] assumes a Manhattan world, and this stronger assumption about scene configuration enables slightly more accurate predictions on easy samples, but completely fails in other scenarios. In contrast, GeoCalib is the first deep method that consistently matches or surpasses the accuracy of classical methods without any assumption on the scene, thus combining the accuracy of classical methods with the robustness of deep neural networks. We conclude that our method is more accurate in gravity estimation than FoV, likely due to the weaker constraint from latitude, which is generally harder to predict.

| Approach | FoV [degrees] | 𝐤1\bm{\mathrm{k}}_{1} | Pixel Distortion Error |  |  |  |
|---|---|---|---|---|---|---|
| error ↓\downarrow | error ↓\downarrow | recall ⊳\triangleright 0.5/1/3/5px ↑\uparrow |  |  |  |  |
| Perceptual [36] | 13.50 | (spherical) | 00.97 | 03.42 | 19.67 | 30.01 |
| SVA [50] | - | (divisional) | 22.28 | 32.01 | 52.50 | 63.67 |
| DeepCalib* [52] | 10.80 | 00.09 | 23.89 | 34.32 | 55.82 | 67.06 |
| ParamNet* [38] | 11.17 | 00.10 | 22.56 | 32.37 | 53.07 | 64.34 |
| GeoCalib - pinhole* | 04.55 | 00.07 | 25.92 | 37.10 | 60.40 | 72.01 |
| GeoCalib* | 04.38 | 00.06 | 28.36 | 40.28 | 64.14 | 75.57 |

### 5.2 Lens distortion

We now evaluate how GeoCalib can handle radial distortion and estimate it.

We evaluate camera distortion estimation on the MegaDepth dataset [47] with crowd-sourced, distorted images. These images were taken from various sensors and thus exhibit a diverse distribution of distortions. The ground-truth distortion models were obtained from SfM [74]. Similar to these GT camera models, we use a polynomial distortion model [27] with one parameter and report the median error. We also report the pixel distortion error, defined as the Euclidean distance between the pixel distorted by the ground truth camera model and the distortion from the predicted model but with GT focal length. In contrast to previous works [50], we use the GT focal length to disentangle the distortion estimation from FoV, which can bias the evaluation if the FoV estimate is incorrect. We average the pixel distortion errors over each image and compute the recall of the average distortion error within an image over the entire dataset. For completeness, we also report the median FoV error for each method.

We compare our method against Perceptual [36], which uses a spherical distortion model [84], SVA [50] (divisional model [29]), as well as DeepCalib [52] and PerspectiveNet+ParamNet [38] trained on our dataset with a single parameter polynomial distortion model [27]. We evaluate both variants of GeoCalib trained with pinhole and distorted images.

Table 2 reports the results. GeoCalib-pinhole already improves over all baselines, suggesting that the model can zero-shot generalize to radial distortion through optimization. Training our model on distorted images further boosts the accuracy. Figure 6 shows that GeoCalib can accurately handle strong lens distortion on a diverse range of images.

### 5.3 Insights

| Dataset | known? | Gravity [degrees] | FoV [degrees] |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|
| gravity | FoV | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow |  |  |  |  |  |
| Stanford2D3D | ✗ | ✗ | 01.12 | 45.6 | 71.5 | 82.7 | 03.21 | 17.4 | 40.0 | 59.4 |
| ✗ | ✓ | 00.86 | 57.0 | 79.0 | 87.0 | - | - | - | - |  |
| ✓ | ✗ | - | - | - | - | 02.30 | 24.7 | 50.2 | 67.5 |  |
| TartanAir | ✗ | ✗ | 01.69 | 31.6 | 58.6 | 73.4 | 04.90 | 14.1 | 30.4 | 47.6 |
| ✗ | ✓ | 01.18 | 44.7 | 66.8 | 78.6 | - | - | - | - |  |
| ✓ | ✗ | - | - | - | - | 03.21 | 20.0 | 40.4 | 56.9 |  |
| MegaDepth | ✗ | ✗ | 02.10 | 28.1 | 50.8 | 65.8 | 04.46 | 13.6 | 31.7 | 48.2 |
| ✗ | ✓ | 01.41 | 37.6 | 62.3 | 74.2 | - | - | - | - |  |
| ✓ | ✗ | - | - | - | - | 02.42 | 27.0 | 48.8 | 64.3 |  |
| LaMAR | ✗ | ✗ | 00.99 | 50.1 | 74.5 | 84.5 | 03.03 | 19.1 | 41.5 | 60.0 |
| ✗ | ✓ | 00.82 | 58.2 | 80.2 | 88.2 | - | - | - | - |  |
| ✓ | ✗ | - | - | - | - | 02.28 | 23.5 | 50.0 | 67.4 |  |

| Approach | DUC 1 | DUC2 |  |  |  |  |
|---|---|---|---|---|---|---|
| Recall at (0.25m,10°) / (0.5m,10°) / (1.0m,10°) ↑\uparrow |  |  |  |  |  |  |
| Baseline (SP+SG) | 43.4 | 66.7 | 78.3 | 51.9 | 74.8 | 78.6 |
| + gravity in refinement | 44.4 | 68.2 | 78.8 | 54.2 | 74.8 | 78.6 |
| + gravity in RANSAC | 44.9 | 69.2 | 79.3 | 55.0 | 76.3 | 78.6 |
| gravity from UVP [62] | 43.9 | 67.7 | 78.8 | 52.7 | 76.3 | 78.6 |

We perform an extensive ablation study to verify the design decisions of our method. We start from the closest baseline PerspectiveNet + ParamNet [38], trained on 360 cities, and train 5 different models on our OpenPano dataset. We evaluate the trained models on the 2k test images sampled from OpenPano as well as on the previous benchmarks, averaging their results.

We plot the results in Fig. 7. (1) While OpenPano is about 5 times smaller than 360 Cities, it is more balanced across different domains, resulting in improvements on the test set and on the benchmarks. (2) The improved architecture as described in Sec. 3 slightly increases the accuracy both in- and out-of-domain. (3) When switching from ParamNet [38] to our second-order optimization, the FoV estimation drops in domain but gets more accurate across the benchmarks. This shows that ParamNet is unable to generalize to out-of-domain images. (4) Supervising the result of the optimization and (5) learning uncertainties significantly boost the accuracy across the board as this i) allows GeoCalib to increase the weight for accurate predictions, improving both accuracy and robustness, and ii) enables it to focus more on well-constrained areas of the image, while saving capacity in less informative regions. (6) Finally, replacing the SegFormer [92] backbone with the more recent SegNeXt [32] also brings improvements. Despite relying on classical geometry, GeoCalib will thus benefit from further advances in deep learning architectures. See Appendix 0.A for an extended analysis.

In many real-world scenarios, we either have access to GT gravity (from IMU) or known intrinsics. We conduct an additional experiment as in Sec. 5.1 to showcase how GeoCalib can leverage such priors. Here, we either fix the gravity direction or the intrinsics (FoV) in the optimization (Eq. 2) and predict the other. We report the angular error of gravity and FoV.

Table 3 shows that partial calibration consistently improves the accuracy of our joint optimization. Fixing the FoV improves the gravity estimation by up to 13.1%13.1\% AUC@1° on TartanAir [86]. On the other hand, fixing the gravity direction improves the intrinsics estimation across all thresholds equally.

We evaluate GeoCalib’s ability to jointly estimate camera intrinsics for multiple images captured by the same camera. We randomly sample sets of NN images from the TartanAir [86] dataset and report the AUC of the gravity and FoV at 1° and 5°.

Figure 8 shows that our joint optimization estimates progressively more accurate FoVs as more images are considered simultaneously. This also improves the estimate of the gravity, even though it is different for each image. In contrast, simply averaging the independently-estimated FoVs over all images is less effective and cannot benefit the gravity estimation. Thus, GeoCalib can effectively leverage geometric priors across images, even if they do not have any visual overlap.

### 5.4 Indoor Visual Localization

We evaluate the accuracy of gravity-aided visual localization on the InLoc [78] indoor dataset following hloc [68, 25, 69]. We report the pose recall under three different thresholds on the two locations DUC1 and DUC2. The absolute pose is estimated with RANSAC and LM-refinement from 2D-3D correspondences.

Because the gravity estimates are noisy, we avoid upright solvers [43, 44] and instead add soft constraints. In pose refinement, we add a regularization on the rotation towards the estimated gravity [4]. In RANSAC, we add a constant reward to the MSAC score if the estimated gravity is within 2​σ2\sigma of our estimated gravity uncertainty. As a baseline, we use the gravity estimated by UVP [62].

We report the results in Tab. 4. Adding the gravity constraint in both RANSAC and pose refinement improves the localization accuracy. The baseline UVP [62] is less effective because its gravity estimates are less accurate.

## 6 Conclusion

This paper introduces GeoCalib, a new approach to single-image calibration that combines the best of learning and geometry. Thanks to its differentiable optimization, it learns strong priors that make it both more accurate and more robust than existing approaches, with a strong generalization to different environments. GeoCalib offers much flexibility in terms of camera models, priors, and uncertainties and is thus easier to integrate in downstream applications.

Thanks to Xu Song for helping to evaluate MSCC. Thanks to Jean-François Lalonde for allowing us to release models trained on the Laval Indoor dataset. Thanks to Shaohui Liu and Linfei Pan for their valuable feedback. Philipp Lindenberger was supported by an ETH Zurich RobotX Fellowship.

## Appendix 0.A Extended ablation study

|  | Approach | Gravity [degrees] | FoV [degrees] |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow |  |  |  |  |  |
| OpenPano | PerspectiveNet+ParamNet [38] | 04.99 | 04.5 | 23.0 | 43.7 | 06.63 | 09.1 | 21.0 | 37.7 |
| + our dataset (OpenPano) | 03.08 | 09.4 | 38.1 | 60.2 | 04.40 | 13.1 | 31.0 | 51.5 |  |
| + better architecture | 03.01 | 09.2 | 38.9 | 60.9 | 04.23 | 13.9 | 32.0 | 52.5 |  |
| + ParamNet →\rightarrow SGD optim | 03.03 | 10.3 | 39.1 | 61.0 | 04.75 | 12.4 | 29.3 | 49.2 |  |
| + ParamNet →\rightarrow LM optim | 03.03 | 09.9 | 39.1 | 61.0 | 04.58 | 12.6 | 30.2 | 50.7 |  |
| + end-to-end training | 02.65 | 14.0 | 43.7 | 64.0 | 04.54 | 12.5 | 30.2 | 49.7 |  |
| + learned uncertainties | 02.24 | 18.8 | 49.1 | 67.3 | 03.85 | 16.3 | 35.5 | 54.4 |  |
| + SegFormer →\rightarrow SegNext = ours | 01.88 | 23.1 | 54.6 | 71.1 | 03.02 | 19.4 | 41.9 | 60.7 |  |
| MegaDepth | PerspectiveNet+ParamNet [38] | 04.06 | 05.4 | 28.3 | 51.5 | 10.98 | 05.3 | 12.8 | 24.0 |
| + our dataset (OpenPano) | 04.51 | 06.8 | 28.0 | 48.6 | 11.01 | 03.2 | 10.1 | 21.3 |  |
| + better architecture | 04.35 | 08.8 | 29.9 | 50.4 | 11.23 | 03.9 | 10.0 | 20.8 |  |
| + ParamNet →\rightarrow SGD optim | 04.22 | 07.4 | 29.5 | 50.9 | 08.88 | 05.8 | 15.1 | 29.0 |  |
| + ParamNet →\rightarrow LM optim | 04.12 | 08.2 | 30.1 | 51.2 | 08.70 | 05.4 | 15.6 | 29.6 |  |
| + end-to-end training | 03.31 | 10.8 | 35.4 | 57.6 | 07.22 | 08.1 | 18.4 | 34.8 |  |
| + learned uncertainties | 02.36 | 20.7 | 48.1 | 65.1 | 04.80 | 11.7 | 28.9 | 46.3 |  |
| + SegFormer →\rightarrow SegNext = ours | 02.10 | 28.1 | 50.8 | 65.8 | 04.46 | 13.6 | 31.7 | 48.2 |  |
| LaMAR | PerspectiveNet+ParamNet [38] | 03.46 | 06.9 | 34.4 | 59.4 | 15.15 | 01.8 | 06.2 | 13.2 |
| + our dataset (OpenPano) | 02.57 | 13.6 | 44.7 | 65.3 | 14.71 | 02.8 | 06.8 | 14.3 |  |
| + better architecture | 02.53 | 14.8 | 46.0 | 66.2 | 13.93 | 02.3 | 06.4 | 14.4 |  |
| + ParamNet →\rightarrow SGD optim | 02.54 | 14.2 | 45.4 | 66.0 | 11.71 | 04.9 | 12.5 | 23.0 |  |
| + ParamNet →\rightarrow LM optim | 02.54 | 14.4 | 45.4 | 66.1 | 11.24 | 04.6 | 12.5 | 23.6 |  |
| + end-to-end training | 02.14 | 19.5 | 51.0 | 71.2 | 06.34 | 07.9 | 20.2 | 38.5 |  |
| + learned uncertainties | 01.31 | 37.3 | 67.2 | 80.3 | 03.66 | 15.9 | 36.3 | 54.8 |  |
| + SegFormer →\rightarrow SegNeXt = ours | 00.99 | 50.1 | 74.5 | 84.5 | 03.03 | 19.1 | 41.5 | 60.0 |  |

We perform an extensive ablation study to verify the design decisions of our method. We start from the closest baseline PerspectiveNet + ParamNet [38], trained on 360 cities, and train 5 different models on our OpenPano dataset. We evaluate the trained models on the MegaDepth [47] and LaMAR [70] test splits and on the 2k test images sampled from OpenPano . We report the median and AUC of angular gravity and FoV errors.

We report the results in Tab. 5. 1) Re-training PerspectiveNet + ParamNet [38] on OpenPano, which is 5 times smaller than 360 cities [38], yields large improvements on the test split of OpenPano and LaMAR, but no improvements or slightly weaker generalization to MegaDepth. We conclude that despite the smaller size of our dataset, it enables strong generalization with higher quality and more evenly sampled images across domains at a fraction of the size. 2) The improved architecture, as described in Sec. 3 increases the accuracy both in- and out-of-domain. The improved low-level features enable a more accurate prediction of up-vectors, which results in more precise gravity estimations. 3) Replacing ParamNet with the optimization proposed in [38] drops the accuracy in-domain, but improves FoV estimation on MegaDepth and LaMAR. This suggests that ParamNet is not able to generalize well on FoV estimation. 4) When switching from ParamNet [38] to our second-order optimization, we observe that it does not improve over the optimization proposed in [38] out-of-the-box. 5) However, propagating the gradients through the LM-optimization enables the network to learn globally consistent up-vectors and latitude, which results in more robustness and generalization, concluded from large improvements for gravity and FoV estimation both in LaMAR and MegaDepth. 6) Letting the network predict uncertainties on the perspective fields and propagating them through the LM optimization greatly improves the accuracy. These uncertainties i) allow the network to increase the weight for accurate predictions, improving both accuracy and robustness, and ii) enable the network to focus more on well-constrained areas of the image, while saving capacity in less informative regions. This results in a much stronger generalization of our model, supported by the 2x improvements at fine thresholds on MegaDepth and LaMAR. 7) Finally, replacing the SegFormer [92] architecture with the more efficient SegNeXt [32] increases our model’s accuracy, showing that GeoCalib will benefit from advances in deep learning architectures despite relying on classical geometry.

|  | Approach | Gravity [degrees] | FoV [degrees] |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow | error ↓\downarrow | AUC ⊳\triangleright 1/5/10° ↑\uparrow |  |  |  |  |  |
| OpenPano | Trivial | 34.36 | 00.1 | 00.3 | 01.1 | 22.60 | 01.8 | 05.5 | 11.0 |
| Heuristic | 04.74 | 06.0 | 26.5 | 46.2 | 06.94 | 09.1 | 21.3 | 37.1 |  |
| Solver | 06.01 | 11.4 | 30.3 | 41.5 | 11.79 | 10.4 | 21.8 | 32.0 |  |
| GeoCalib |  |  |  |  |  |  |  |  |  |
| +Trivial | 01.88 | 23.1 | 54.6 | 71.1 | 03.02 | 19.4 | 41.9 | 60.7 |  |
| +Heuristic | 01.90 | 23.1 | 54.6 | 71.0 | 03.09 | 19.3 | 41.8 | 60.5 |  |
| +Solver | 02.03 | 21.4 | 51.8 | 67.7 | 03.31 | 18.3 | 39.7 | 57.8 |  |
| MegaDepth | Trivial | 21.90 | 00.7 | 02.8 | 07.4 | 21.07 | 00.5 | 02.8 | 08.3 |
| Heuristic | 06.93 | 04.6 | 16.7 | 35.7 | 08.45 | 05.2 | 14.8 | 29.4 |  |
| Solver | 08.69 | 07.0 | 18.8 | 31.9 | 16.67 | 04.0 | 10.5 | 19.0 |  |
| GeoCalib |  |  |  |  |  |  |  |  |  |
| +Trivial | 02.10 | 28.1 | 50.8 | 65.8 | 04.46 | 13.6 | 31.7 | 48.2 |  |
| +Heuristic | 02.09 | 28.2 | 51.0 | 65.9 | 04.50 | 13.6 | 31.6 | 48.1 |  |
| +Solver | 02.14 | 27.8 | 50.7 | 65.6 | 04.54 | 13.4 | 31.4 | 47.8 |  |

## Appendix 0.B Initialization strategies

We describe and evaluate additional initialization strategies mentioned in Sec. 3.1.

The trivial strategy initializes 𝐠=[010]\bm{\mathrm{g}}{=}\begin{bmatrix}0&1&0\end{bmatrix}, 𝐤=𝟎\bm{\mathrm{k}}{=}\bm{\mathrm{0}}, and f=0.7​max⁡(W,H)f{=}0.7\max\left(W,H\right), assuming a common sensor size.

A heuristic strategy was proposed in [38]. We observe that, at the principal point 𝐜\bm{\mathrm{c}}, 𝐮𝐜∝−[𝐠x𝐠y]{\bm{\mathrm{u}}_{\bm{\mathrm{c}}}\propto{-}\begin{bmatrix}\bm{\mathrm{g}}_{x}&\bm{\mathrm{g}}_{y}\end{bmatrix}}. Additionally, from Eq. 1, 𝐠z=sin⁡φ𝐜\bm{\mathrm{g}}_{z}{=}\sin\varphi_{\bm{\mathrm{c}}}. Therefore, 𝐠=[α​𝐮𝐜⊤sin⁡φ𝐜]\bm{\mathrm{g}}{=}\begin{bmatrix}\alpha\bm{\mathrm{u}}_{\bm{\mathrm{c}}}^{\top}&\sin\varphi_{\bm{\mathrm{c}}}\end{bmatrix}, where α\alpha is adjusted to ensure ‖𝐠‖=1\left\lVert\bm{\mathrm{g}}\right\rVert{=}1. Notably, the vertical field of view is linked to the difference in latitude, expressed as vFoV=φ(𝐜x,0)−φ(𝐜x,H)\text{vFoV}{=}\varphi_{(\bm{\mathrm{c}}_{x},0)}-\varphi_{(\bm{\mathrm{c}}_{x},H)}, which can be converted into an initial estimate for f.

Generalizing the heuristic strategy, we derive a simple minimal solver to infer 𝐠\bm{\mathrm{g}} and ff from any combination of two up-vectors and one latitude value. We embed this solver in a RANSAC [28] loop to maximize the confidence-weighted inlier count of up-vectors and latitudes. This can provide a more accurate estimate.

We evaluate each initialization strategy and demonstrate its impact on the final estimation when used to initialize our optimization. We perform the evaluation on the OpenPano and MegaDepth [47] datasets and report the median and AUC of the angular gravity and vFoV errors. The results in Tab. 6 show that the increased accuracy of the heuristic strategy and the solver does not translate to any improvements in the final estimate of our model. We hypothesize that, once trained, the observed Perspective Field is sufficiently good to ensure convergence from any initial point.

a) Precision and recall curves

b) Calibration plots

## Appendix 0.C Uncertainty estimation

We evaluate the optimization uncertainties estimated by GeoCalib, as described in Sec. 3.2, on the OpenPano dataset.

We show precision and recall curves for gravity and vertical Field-of-View (vFoV) in Fig. 9-a). To assess the quality of our uncertainties, we discretize the errors by increasing thresholds and determine how accurately the predicted uncertainties classify samples into their respective categories. This means that for each threshold, we classify samples based on whether their predicted uncertainty surpasses the threshold or not and calculate the corresponding precision and recall values. The precision curve indicates that samples with low uncertainty generally exhibit low error rates. However, the low recall values for small errors suggest that the model is under-confident in its predictions.

We also show corresponding calibration plots in Fig. 9-b). To compute them, we bin the predictions based on the uncertainty and calculate the mean error per bin. We observe that the calibration tightly follows the optimal calibration of y=xy{=}x for samples with low uncertainty. For samples with high uncertainty, our model tends to be under-confident in its prediction. The error is thus generally bounded by the predicted uncertainty.

## Appendix 0.D Distortion models

Following Eq. 1, the up-vector is collinear with the directional derivative of the projection function Π⁡(⋅)\Pi(\cdot) along the up direction −𝐠-\bm{\mathrm{g}}, i.e. 𝐮𝐩∝−∇Π(𝐏)⋅𝐠\bm{\mathrm{u}}_{\bm{\mathrm{p}}}\propto-\nabla\Pi({\bm{\mathrm{P}}})\cdot\bm{\mathrm{g}}. We consider a radial distortion model such that 𝒟⁡([uv],𝐤)=d⁡(u,v,𝐤)​[uv]\mathcal{D}\left(\begin{bmatrix}u&v\end{bmatrix},\bm{\mathrm{k}}\right)=d(u,v,\bm{\mathrm{k}})\begin{bmatrix}u&v\end{bmatrix}. Then

|  | ∇Π​(𝐏)\displaystyle\nabla\Pi({\bm{\mathrm{P}}}) | ∝∂𝒟⁡([uv],𝐤)∂𝐏\displaystyle\propto\frac{\partial\mathcal{D}\left(\begin{bmatrix}u&v\end{bmatrix},\bm{\mathrm{k}}\right)}{\partial{\bm{\mathrm{P}}}} |  | (5) |
|---|---|---|---|---|
|  |  | ∝d⁡(u,v,𝐤)​∂[uv]⊤∂𝐏+[uv]​∂d⁡(u,v,𝐤)∂[uv]​∂[uv]⊤∂𝐏\displaystyle\propto d(u,v,\bm{\mathrm{k}})\frac{\partial\begin{bmatrix}u&v\end{bmatrix}^{\top}}{\partial{\bm{\mathrm{P}}}}+\begin{bmatrix}u\\ v\end{bmatrix}\frac{\partial d(u,v,\bm{\mathrm{k}})}{\partial\begin{bmatrix}u&v\end{bmatrix}}\frac{\partial\begin{bmatrix}u&v\end{bmatrix}^{\top}}{\partial{\bm{\mathrm{P}}}} |  | (6) |
|  |  | ∝(d⁡(u,v,𝐤)+[uv]​∂d⁡(u,v,𝐤)∂(u,v))​∂[uv]⊤∂𝐏\displaystyle\propto\left(d(u,v,\bm{\mathrm{k}})+\begin{bmatrix}u\\ v\end{bmatrix}\frac{\partial d(u,v,\bm{\mathrm{k}})}{\partial(u,v)}\right)\frac{\partial\begin{bmatrix}u&v\end{bmatrix}^{\top}}{\partial{\bm{\mathrm{P}}}} |  | (7) |
|  |  | ∝(1+1d⁡(u,v,𝐤)​[uv]​∂d⁡(u,v,𝐤)∂(u,v))​[10−u01−v].\displaystyle\propto\left(1+\frac{1}{d(u,v,\bm{\mathrm{k}})}\begin{bmatrix}u\\ v\end{bmatrix}\frac{\partial d(u,v,\bm{\mathrm{k}})}{\partial(u,v)}\right)\begin{bmatrix}1&0&-u\\ 0&1&-v\end{bmatrix}\kern 5.0pt. |  | (8) |

The up-vector is thus such that

|  | 𝐮𝐩​(f,𝐠,𝐤)∝(1+1d⁡(u,v,𝐤)​[uv]​∂d⁡(u,v,𝐤)∂(u,v))​[u​𝐠z−𝐠xv​𝐠z−𝐠y].\bm{\mathrm{u}}_{\bm{\mathrm{p}}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)\propto\left(1+\frac{1}{d(u,v,\bm{\mathrm{k}})}\begin{bmatrix}u\\ v\end{bmatrix}\frac{\partial d(u,v,\bm{\mathrm{k}})}{\partial(u,v)}\right)\begin{bmatrix}u\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{x}\\ v\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{y}\end{bmatrix}\kern 5.0pt. |  | (9) |
|---|---|---|---|

For the pinhole camera model, this simplifies to

|  | 𝐮𝐩​(f,𝐠,𝐤)∝[u​𝐠z−𝐠xv​𝐠z−𝐠y].\bm{\mathrm{u}}_{\bm{\mathrm{p}}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)\propto\begin{bmatrix}u\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{x}\\ v\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{y}\end{bmatrix}\kern 5.0pt. |  | (10) |
|---|---|---|---|

For a polynomial radial distortion with d⁡(u,v,𝐤)=1+𝐤1​r2+𝐤2​r4d(u,v,\bm{\mathrm{k}})=1+\bm{\mathrm{k}}_{1}r^{2}+\bm{\mathrm{k}}_{2}r^{4}, where 𝐤=(𝐤1,𝐤2)\bm{\mathrm{k}}=(\bm{\mathrm{k}}_{1},\bm{\mathrm{k}}_{2}) are the distortion parameters and r2=u2+v2r^{2}=u^{2}+v^{2}, we have

|  | ∂d⁡(u,v,𝐤)∂[uv]=2​(𝐤1+2​𝐤2​r2)​[uv].\frac{\partial d(u,v,\bm{\mathrm{k}})}{\partial\begin{bmatrix}u&v\end{bmatrix}}=2(\bm{\mathrm{k}}_{1}+2\bm{\mathrm{k}}_{2}r^{2})\begin{bmatrix}u&v\end{bmatrix}\kern 5.0pt. |  | (11) |
|---|---|---|---|

## Appendix 0.E Analytical Jacobians

We now report the analytical expression of the jacobians used in the optimization for the pinhole camera model. For each pixel, we write the up-vector and latitude

|  | 𝐮=norm⁡([u​𝐠z−𝐠xv​𝐠z−𝐠y])=norm⁡(𝐮¯)sin⁡(φ)=norm⁡(𝐧)⊤​𝐠\bm{\mathrm{u}}=\operatorname{norm}\left(\begin{bmatrix}u\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{x}\\ v\bm{\mathrm{g}}_{z}-\bm{\mathrm{g}}_{y}\end{bmatrix}\right)=\operatorname{norm}\left(\bar{\bm{\mathrm{u}}}\right)\quad\quad\sin\left(\varphi\right)=\operatorname{norm}\left(\bm{\mathrm{n}}\right)^{\top}\bm{\mathrm{g}}\kern 5.0pt |  | (12) |
|---|---|---|---|

where 𝐧=[uv1]⊤\bm{\mathrm{n}}=\begin{bmatrix}u&v&1\end{bmatrix}^{\top} is the ray and norm⁡(𝐱)=𝐱/‖𝐱‖2\operatorname{norm}\left(\bm{\mathrm{x}}\right)=\nicefrac{{\bm{\mathrm{x}}}}{{\left\lVert\bm{\mathrm{x}}\right\rVert_{2}}} normalizes the input vector to have unit length. We write 𝐉=[𝐉𝐮𝐉φ]\bm{\mathrm{J}}=\begin{bmatrix}\bm{\mathrm{J}}_{\bm{\mathrm{u}}}&\bm{\mathrm{J}}_{\varphi}\end{bmatrix} and use the general chain rule:

|  | 𝐉𝐮\displaystyle\bm{\mathrm{J}}_{\bm{\mathrm{u}}} | =∂𝐫𝐮∂𝜹=∂𝐫𝐮∂𝐮⁡(𝜽)​∂𝐮⁡(𝜽)∂𝜽​∂𝜽∂𝜹,\displaystyle=\frac{\partial\bm{\mathrm{r}}_{\bm{\mathrm{u}}}}{\partial\bm{\mathrm{\delta}}}=\frac{\partial\bm{\mathrm{r}}_{\bm{\mathrm{u}}}}{\partial\bm{\mathrm{u}}\left(\bm{\mathrm{\theta}}\right)}\frac{\partial\bm{\mathrm{u}}\left(\bm{\mathrm{\theta}}\right)}{\partial\bm{\mathrm{\theta}}}\frac{\partial\bm{\mathrm{\theta}}}{\partial\bm{\mathrm{\delta}}}, |  | (13) |
|---|---|---|---|---|
|  | 𝐉φ\displaystyle\bm{\mathrm{J}}_{\varphi} | =∂𝐫φ∂𝜹=∂𝐫φ∂sin⁡(φ⁡(𝜽))​∂sin⁡(φ⁡(𝜽))∂𝜽​∂𝜽∂𝜹.\displaystyle=\frac{\partial\bm{\mathrm{r}}_{\varphi}}{\partial\bm{\mathrm{\delta}}}=\frac{\partial\bm{\mathrm{r}}_{\varphi}}{\partial\sin{\left(\varphi\left(\bm{\mathrm{\theta}}\right)\right)}}\frac{\partial\sin{\left(\varphi\left(\bm{\mathrm{\theta}}\right)\right)}}{\partial\bm{\mathrm{\theta}}}\frac{\partial\bm{\mathrm{\theta}}}{\partial\bm{\mathrm{\delta}}}. |  | (14) |

where we calculate the jacobians at 𝜹=0\bm{\mathrm{\delta}}{=}0. By simple derivation we can calculate the partial derivate for 𝐮\bm{\mathrm{u}} as

|  | ∂𝐮⁡(f,𝐠,𝐤)∂𝐠\displaystyle\frac{\partial\bm{\mathrm{u}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)}{\partial\bm{\mathrm{g}}} | =(1‖𝐮¯‖2​𝕀−𝐮¯​𝐮¯⊤‖𝐮¯‖23)​[−10u0−1v],\displaystyle=\left(\frac{1}{\left\lVert\bar{\bm{\mathrm{u}}}\right\rVert}_{2}\mathbb{I}-\frac{\bar{\bm{\mathrm{u}}}\bar{\bm{\mathrm{u}}}^{\top}}{\left\lVert\bar{\bm{\mathrm{u}}}\right\rVert_{2}^{3}}\right)\begin{bmatrix}-1&0&u\\ 0&-1&v\end{bmatrix}, |  | (15) |
|---|---|---|---|---|
|  | ∂𝐮⁡(f,𝐠,𝐤)∂f\displaystyle\frac{\partial\bm{\mathrm{u}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)}{\partial f} | =(1‖𝐮¯‖2​𝕀−𝐮¯​𝐮¯⊤‖𝐮¯‖23)​[𝐠z𝐠z]​[−(x−𝐜x)/f2−(y−𝐜y)/f2].\displaystyle=\left(\frac{1}{\left\lVert\bar{\bm{\mathrm{u}}}\right\rVert}_{2}\mathbb{I}-\frac{\bar{\bm{\mathrm{u}}}\bar{\bm{\mathrm{u}}}^{\top}}{\left\lVert\bar{\bm{\mathrm{u}}}\right\rVert_{2}^{3}}\right)\begin{bmatrix}\bm{\mathrm{g}}_{z}\\ \bm{\mathrm{g}}_{z}\end{bmatrix}\begin{bmatrix}\nicefrac{{-(x-\bm{\mathrm{c}}_{x})}}{{f^{2}}}\\ \nicefrac{{-(y-\bm{\mathrm{c}}_{y})}}{{f^{2}}}\end{bmatrix}. |  | (16) |

Similarly, we can calculate the partial derivatives for sin⁡(φ)\sin\left(\varphi\right) as

|  | ∂sin⁡(φ𝐩​(f,𝐠,𝐤))∂𝐠\displaystyle\frac{\partial\sin\left(\varphi_{\bm{\mathrm{p}}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)\right)}{\partial\bm{\mathrm{g}}} | =norm⁡(𝐧)⊤​𝕀,\displaystyle=\operatorname{norm}\left(\bm{\mathrm{n}}\right)^{\top}\mathbb{I}, |  | (17) |
|---|---|---|---|---|
|  | ∂sin⁡(φ𝐩​(f,𝐠,𝐤))∂f\displaystyle\frac{\partial\sin\left(\varphi_{\bm{\mathrm{p}}}\left(f,\bm{\mathrm{g}},\bm{\mathrm{k}}\right)\right)}{\partial f} | =𝐠⊤​(1‖𝐧‖2​𝕀−𝐧𝐧⊤‖𝐧‖23)​[−(x−𝐜x)/f2−(y−𝐜y)/f20].\displaystyle=\bm{\mathrm{g}}^{\top}\left(\frac{1}{\left\lVert\bm{\mathrm{n}}\right\rVert}_{2}\mathbb{I}-\frac{\bm{\mathrm{n}}\bm{\mathrm{n}}^{\top}}{\left\lVert\bm{\mathrm{n}}\right\rVert_{2}^{3}}\right)\begin{bmatrix}\nicefrac{{-(x-\bm{\mathrm{c}}_{x})}}{{f^{2}}}\\ \nicefrac{{-(y-\bm{\mathrm{c}}_{y})}}{{f^{2}}}\\ 0\end{bmatrix}. |  | (18) |

The Jacobians for radially distorted cameras can be derived in a similar way.

## Appendix 0.F Implementation details

In this section we provide more information and details on our dataset, how we train our models and evaluate our baselines.

### 0.F.1 Dataset

For our dataset we collect 360° panoramas in equirectangular format which covers 180° vertically and 360° horizontally. The dataset comprises 2912 panoramas sourced from hdrmaps [1], polyhaven [2], and parts of the Laval Photometric Indoor HDR dataset [15], where we manually select panoramas that are aligned with the gravity. The resulting dataset contains a good balance of about 800 outdoor and 2.1k indoor panoramas which are split into 2616 training, 147 validation and 148 testing panoramas. We create two iterations of the dataset: one featuring a simple pinhole camera and another assuming a radially distorted camera model. For both datasets we sample 16 crops per panorama with roll, pitch and vFoV sampled uniformly within [±45​°][\pm 45\degree], [±45​°][\pm 45\degree], and [20,105][20,105] respectively. To capture the relationship between the distortion parameter 𝐤\bm{\mathrm{k}} and the vFoV, we sample 𝐤^=𝐤/vFoV\hat{\bm{\mathrm{k}}}=\nicefrac{{\bm{\mathrm{k}}}}{{\mathrm{vFoV}}} from a truncated normal with μ=0\mu{=}0 and σ=0.07\sigma{=}0.07 bounded to [-0.3, 0.3] for the distorted dataset. Since some panoramas are padded with black pixels to be equirectangular, we remove images with ≥1%\geq 1\% of black pixels. After this, we are left with 37015 images for training, 2100 for validation and 2128 for testing. Figure 10 shows the diversity of our dataset with difficult and easy samples, indoor and outdoor images as well as variation in gravity and camera intrinsics.

### 0.F.2 Training

GeoCalib is trained for 75k steps with a batch size of 24 and a base learning rate of 10−410^{-4} using the AdamW [53] optimizer. We make use of a linear warm-up schedule in the first 1k steps and drop the learning rate after 40k and 65k steps by a factor of 10. The input resolution is 320×320320\times 320 and we apply extensive color, blur and resize augmentations during training. During inference, non-square images are first resized to 320320 along the shorter edge and afterwards cropped to fit the required multiple of 32 by our model. The resulting camera is re-scaled to fit the original image dimensions.

### 0.F.3 Indoor Visual Localization

We use the P3P RANSAC solver from PoseLib [44] as a baseline, and add the gravity constraint in their framework. We add a constant, multiplicative reward to the MSAC scoring if the angle between the estimated gravity from our network and the gravity obtained from the samples’ pose estimate (the second column of the rotation matrix) is smaller than twice the uncertainty predicted by our network (in degrees).

In contrast to upright solvers [43], this weak constraint is less susceptible to noise and can directly utilize the uncertainty estimates of our model.

In the non-linear absolute pose refinement, we add an L2-regularization term to the optimization, which penalizes the deviation of the rotation matrix from our estimated gravity. We use a simple L2-loss and weigh the regularization by the number of inlier correspondences. We use the Ceres-solver [4] to build the non-linear pose refinement, similar to COLMAP [74].

### 0.F.4 Evaluation

We provide further details on our evaluation setup.

To account for inaccuracies in the ground truth, especially for datasets derived from structure-from-motion like MegaDepth and LaMAR, we compute the AUC for curves clipped to a minimum of 1°\degree. When an approach fails to return a valid result [50, 62], we set its error to +∞+\infty.

We re-implement the model and its training following the paper. We train the model for 20k steps using the Adam optimizer [41] with a learning rate of 10−410^{-4} and a batch size of 32 on our distorted dataset. The input images are resized to 320×320320\times 320 and we use the same augmentations as for GeoCalib. We follow the parameterization proposed in [52] and train the model using 256 bins per parameter. The model is trained to minimize the negative log likelihood, and we apply early stopping on the validation set.

We run inference using the online demo provided by the authors, which is based on a ConvNeXt CNN backbone [49].

We make use of the official code released by the authors to run inference on our benchmarks using the model weights trained on the SUN360 [91] dataset. The input resolution is 512×512512\times 512.

We ask the authors to run inference on our benchmarks without sharing the ground-truth parameters.

We re-implement ParamNet and validate our implementation by comparing its inference to the original code-base on various benchmarks. The model is re-trained on our pinhole dataset following the schedule proposed in the official repo: we first pre-train PerspectiveNet, then add ParamNet to train the full model. Each is trained for 45k steps with a base learning rate of 0.010.01 using the SGD optimizer and a batch size of 64. After a 1k step warm-up, the learning rate drops by a factor of 10 after 30k and 40k steps. The input resolution is 320×320320\times 320, and we use the same augmentation and inference resizing patterns as in GeoCalib. To support radially distorted camera models (Tab. 2), we add another prediction head to ParamNet, using the parametrization of [52] for the distortion value k. We then re-train PerspectiveNet and ParamNet on our distorted dataset.

We use the official implementation released by the authors to run inference. The method fails when too few lines or coplanar repeats are visible in the image, which happens for about 58% of the samples in MegaDepth [47] and about 89% of images on Lamar [70]. Images are resized to a maximum of 3​k×3​k3k\times 3k.

To run inference using UVP, we make the upright assumption and follow the suggested configuration using DeepLSD [61] to extract lines. We resize the short side of the image to 320320 as recommended by the authors, which we also found to work best. The approach generally does not return any value when too few lines are observable in the image.

## Appendix 0.G Qualitative Results

We show qualitative results in Figs. 11, 12, 14, 13 and 15.

a) ground-truth

b) final prediction

c) observed latitude

d) observed up-vect.

a) ground-truth

b) final prediction

c) observed latitude

d) observed up-vect.

a) ground-truth

b) final prediction

c) observed latitude

d) observed up-vect.

a) ground-truth

b) final prediction

c) observed latitude

d) observed up-vect.

a) ground-truth

b) final prediction

c) observed latitude

d) observed up-vect.
