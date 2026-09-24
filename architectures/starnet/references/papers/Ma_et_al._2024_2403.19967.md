# Rewrite the Stars

Recent studies have drawn attention to the untapped potential of the "star operation" (element-wise multiplication) in network design. While intuitive explanations abound, the foundational rationale behind its application remains largely unexplored. Our study attempts to reveal the star operation’s ability to map inputs into high-dimensional, non-linear feature spaces—akin to kernel tricks—without widening the network. We further introduce StarNet, a simple yet powerful prototype, demonstrating impressive performance and low latency under compact network structure and efficient budget. Like stars in the sky, the star operation appears unremarkable but holds a vast universe of potential. Our work encourages further exploration across tasks, with codes available at https://github.com/ma-xu/Rewrite-the-Stars.

## 1 Introduction

The learning paradigms have imperceptibly and gradually evolved in the past decade. Since AlexNet [33], a myriad of deep networks [49, 23, 32, 4, 37] have emerged, each building on the other. Despite their characteristic insights and contributions, this line of models is mostly based on the blocks that blend linear projection (i.e., convolution and linear layers) with non-linear activations. Since [56], self-attention has dominated natural language processing, and later computer vision [14]. The most distinctive feature of self-attention is mapping features to different spaces and then constructing an attention matrix through dot-product multiplication. However, this implementation is not efficient, and results in the attention complexity scaling quadratically with the increase in the number of tokens.

Recently, a new learning paradigm has been gaining increased attention: fusing different subspace features through element-wise multiplication. We refer to this paradigm as ‘star operation’ for simplicity (owing to the element-wise multiplication symbol resembling a star). Star operation exhibits promising performance and efficiency across various research fields, including Natural Language Processing (i.e., Monarch Mixer [16], Mamba [17], Hyena Hierarchy [44], GLU [48], etc.), Computer Vision (i.e., FocalNet [60], HorNet [45], VAN [18], etc.), and more. For illustration, we construct a ‘demo block’ for image classification, as depicted on the left in Fig. 1. By stacking multiple demo blocks following a stem layer, we construct a straightforward model named DemoNet. Keeping all other factors constant, we observe that element-wise multiplication (star operation) consistently surpasses summation in performance, as illustrated on the right in Fig. 1. Although the star operation is remarkably simple, it raises the question: why does it deliver such gratifying results? In response to this, several assumptive explanations have been proposed. For instance, FocalNet [60] posits that star operations can act as modulation or gating mechanisms, dynamically altering input features. HorNet [45] suggests that the advantage lies in harnessing high-ordered features. Meanwhile, both VAN [18] and Monarch Mixer [16] attribute the effectiveness to convolutional attention. While these preliminary explanations provide some insights, they are largely based on intuition and assumptions, lacking comprehensive analysis and strong evidence. Consequently, the foundational rationale behind remain unexamined, posing a challenge to better understanding and effectively leveraging the star operations.

In this work, we explain the strong representative ability of star operation by explicitly demonstrating that: the star operation possesses the capability to map inputs into an exceedingly high-dimensional, non-linear feature space. Instead of relying on intuitive or assumptive high-level explanations, we delve deep into the details of star operations. By rewriting and reformulating star operations, we uncover that this seemingly simple operation can generate a new feature space comprising approximately (d2)2\left(\frac{d}{\sqrt{2}}\right)^{2} linearly independent dimensions, as detailed in Sec. 3.1. The way star operation achieves such non-linear high dimensions is distinct from traditional neural networks that increase the network width (aka channel number). Rather, the star operation is analogous to kernel functions that conduct pairwise multiplication of features across distinct channels, particularly polynomial kernel functions [47, 25]. When incorporated into neural networks and with the stacking of multiple layers, each layer contributes to an exponential increase in the implicit dimensional complexity. With just a few layers, the star operation enables the attainment of nearly infinite dimensions within a compact feature space, as elaborated in Sec .3.2. Operating within a compact feature space while benefiting from implicit high dimensions, that is where the star operation captivates with its unique charm.

Drawing from the aforementioned insights, we infer that the star operation may inherently be more suited to efficient, compact networks as opposed to the conventionally used large models. To validate this, we introduce a proof-of-concept efficient network, StarNet, characterized by its conciseness and efficiency. The detailed architecture of StarNet can be found in Fig. 3. StarNet is notably straightforward, devoid of sophisticated designs and fine-tuned hyper-parameters. In terms of design philosophy, StarNet clearly diverges from existing networks, as illustrated in Table 1. Capitalizing on the efficacy of the star operation, our StarNet can even surpass various meticulously designed efficient models, like MobileNetv3 [27], EdgeViT [42], FasterNet [6], etc. For example, our StarNet-S4 outperforms EdgeViT-XS by 0.9% top-1 accuracy on ImageNet-1K validation set while running 3×3\times faster on iPhone13 and CPU, and 2×2\times faster on GPU. These results not only empirically validate our insights regarding the star operation but also underscore its practical value in real-world applications.

We succinctly summarize and emphasize the key contributions of this work as follows:

- • Foremost, we demonstrated the effectiveness of star operations, as illustrated in Fig. 1. We unveiled that the star operation possesses the capability to project features into an exceedingly high-dimensional implicit feature space, akin to polynomial kernel functions, as detailed in Sec. 3.
Foremost, we demonstrated the effectiveness of star operations, as illustrated in Fig. 1. We unveiled that the star operation possesses the capability to project features into an exceedingly high-dimensional implicit feature space, akin to polynomial kernel functions, as detailed in Sec. 3.

- • We validated our analysis through empirical results (refer Fig. 1, Table 2, and Table 3, etc.), theoretical exploration (in Sec. 3), and visual representation (see Fig. 2).
We validated our analysis through empirical results (refer Fig. 1, Table 2, and Table 3, etc.), theoretical exploration (in Sec. 3), and visual representation (see Fig. 2).

- • Drawing inspiration from our analysis, we identify the utility of the star operation in the realm of efficient networks and present a proof-of-concept model, StarNet. Remarkably, StarNet achieves promising performance without the need for intricate designs or meticulously selected hyper-parameters, surpassing numerous efficient designs.
Drawing inspiration from our analysis, we identify the utility of the star operation in the realm of efficient networks and present a proof-of-concept model, StarNet. Remarkably, StarNet achieves promising performance without the need for intricate designs or meticulously selected hyper-parameters, surpassing numerous efficient designs.

- • It is noteworthy that there exists a multitude of unexplored possibilities based on the star operation, such as learning without activations and refining operations within implicit dimensions. We envision that our analysis can serve as a guiding framework, steering researchers away from haphazard network design attempts.
It is noteworthy that there exists a multitude of unexplored possibilities based on the star operation, such as learning without activations and refining operations within implicit dimensions. We envision that our analysis can serve as a guiding framework, steering researchers away from haphazard network design attempts.

| Main Insight | Networks |
|---|---|
| DW-Conv | MobileNetv2 [28, 46] |
| Feature Shuffle | ShuffleNet [65], ShuffleNetv2 [40] |
| Feature Re-use | GhostNet [19], FasterNet [6] |
| NAS | EfficientNet [50, 51], MnasNet [52] |
| Re-parameterization | MobileOne [55], FasterViT [21] |
| Hybrid Architecture | Mobile-Former [9], EdgeViT [42] |
| Implicit high dimension | StarNet(ours) |

## 2 Related Work

Recent efforts have demonstrated that utilizing element-wise multiplication can be a more effective choice than summation in network design for feature aggregation, as exemplified by FocalNet [60], VAN [18], Conv2Former [26], HorNet [45], and more [36, 35, 58, 61]. To elucidate its superiority, intuitive explanations have been developed, including modulation mechanism, high-order features, and the integration of convolutional attention, etc. Although many tentative explanations have been proposed and empirical improvements have been achieved, the foundational rationale behind has remained unexamined. In this work, we explicitly emphasize that the element-wise multiplication is crucial, regardless of trivial architectural modifications. It has the capacity to implicitly transform input features into exceptionally high and nonlinear dimensions in a novel manner, but operate in low-dimensional space.

The inclusion of high-dimensional and nonlinear features is crucial in both traditional machine learning algorithms [3, 20] and deep learning networks [34, 33, 63, 49]. This necessity stems from the intricate nature of real-world data and the inherent capacity of models to represent this complexity. Nevertheless, it is important to recognize that these two lines of approaches achieve this goal from different perspectives. In the era of deep learning, we typically start by linearly projecting low-dimensional features into a high-dimensional space and then introduce non-linearity using activation functions (e.g., ReLU, GELU, etc.). In contrast, we can simultaneously attain high-dimensionality and non-linearity using kernel tricks [47, 10] in traditional machine learning algorithms. For instance, a polynomial kernel function k⁡(x1,x2)=(γ​x1⋅x2+c)dk\left(x_{1},x_{2}\right)=\left(\gamma x_{1}\cdot x_{2}+c\right)^{d} can project the input feature x1,x2∈ℝnx_{1},x_{2}\in\mathbb{R}^{n} into a (n+1)d\left(n+1\right)^{d} high-dimensional non-linear feature space; a Gaussian kernel function k⁡(x1,x2)=exp⁡(−‖x1‖2)​exp​(−‖x2‖2)​∑i=0+∞(2​x1⊺​x2)ii!k\left(x_{1},x_{2}\right)=\mathrm{exp}\left(-\left\|x_{1}\right\|^{2}\right)\mathrm{exp}\left(-\left\|x_{2}\right\|^{2}\right)\sum_{i=0}^{+\infty}\frac{\left(2x_{1}^{\intercal}x_{2}\right)^{i}}{i!} can result in an infinite-dimensional feature space through Taylor expansion. As a comparison, we can observe that classical machine learning kernel methods and neural networks differ in their implementation and comprehension of high-dimensional and non-linear features. In this work, we demonstrate that the star operation can obtain a high-dimensional and non-linear feature space within a low-dimensional input, akin to the principles of kernel tricks. A simple visualization experiment shown in Fig 2 further illustrates the connections between star operation and polynomial kernel functions.

Efficient Networks strive to strike the ideal balance between computational complexity and performance. In recent years, numerous innovative concepts have been introduced to enhance the efficiency of networks. These include depth-wise convolution [28, 46], feature re-use [19, 6], and re-parameterization [55], among others. A comprehensive summary can be found in Table 1. In stark contrast to all previous methods, we demonstrate that the star operation can serve as a novel methodology for efficient networks. It has the unique capability to implicitly consider extremely high-dimensional features while performing computations in a low-dimensional space. The salient merit is discriminative in distinguishing star operation from other technologies in the realm of efficient networks, and makes it particularly suitable for efficient network design. With star operations, we demonstrate that a straightforward network can easily outperform heavily handcrafted designs.

## 3 Rewrite the Stars

We start by rewriting the star operation to explicitly showcase its ability to achieve exceedingly high dimensions. We then demonstrate that after multiple layers, star can significantly increase the implicit dimensions to nearly infinite dimensionality. Discussions are presented subsequently.

### 3.1 Star Operation in One layer

In a single layer of neural networks, the star operation is typically written as (W1T​X+B1)∗(W2T​X+B2)\left(\mathrm{W}_{1}^{\mathrm{T}}\mathrm{X}+\mathrm{B}_{1}\right)*\left(\mathrm{W}_{2}^{\mathrm{T}}\mathrm{X}+\mathrm{B}_{2}\right), signifying the fusion of two linearly transformed features through element-wise multiplication. For convenience, we consolidate the weight matrix and bias into a single entity, denoted by W=[WB]\mathrm{W}=\begin{bmatrix}\mathrm{W}\\ \mathrm{B}\end{bmatrix}, and similarly, X=[X1]\mathrm{X}=\begin{bmatrix}\mathrm{X}\\ 1\end{bmatrix}, resulting star operation (W1T​X)∗(W2T​X)\left(\mathrm{W}_{1}^{\mathrm{T}}\mathrm{X}\right)*\left(\mathrm{W}_{2}^{\mathrm{T}}\mathrm{X}\right). To simplify our analysis, we focus on the scenario involving a one-output channel transformation and a single-element input. Specifically, we define w1,w2,x∈ℝ(d+1)×1w_{1},w_{2},x\in\mathbb{R}^{\left(d+1\right)\times 1}, where dd is the input channel number. It can be readily extended to accommodate multiple output channels W1,W2∈ℝ(d+1)×(d′+1)\mathrm{W}_{1},\mathrm{W}_{2}\in\mathbb{R}^{\left(d+1\right)\times\left(d^{\prime}+1\right)} and to handle multiple feature elements, with X∈ℝ(d+1)×n\mathrm{X}\in\mathbb{R}^{\left(d+1\right)\times n}.

Generally, we can rewrite the star operation by:

|  |  | w1T​x∗w2T​x\displaystyle\>\>\>\>\>\>w_{1}^{\mathrm{T}}x*w_{2}^{\mathrm{T}}x |  | (1) |
|---|---|---|---|---|
|  |  | =(∑i=1d+1w1i​xi)∗(∑j=1d+1w2j​xj)\displaystyle=\left(\sum_{i=1}^{d+1}w_{1}^{i}x^{i}\right)*\left(\sum_{j=1}^{d+1}w_{2}^{j}x^{j}\right) |  | (2) |
|  |  | =∑i=1d+1∑j=1d+1w1i​w2j​xi​xj\displaystyle=\sum_{i=1}^{d+1}\sum_{j=1}^{d+1}w_{1}^{i}w_{2}^{j}x^{i}x^{j} |  | (3) |
|  |  | =α(1,1)​x1​x1+⋯+α(4,5)​x4​x5+⋯+α(d+1,d+1)​xd+1​xd+1⏟(d+2)​(d+1)/2​items\displaystyle=\begin{matrix}\underbrace{\alpha_{\left(1,1\right)}x^{1}x^{1}+\cdots+\alpha_{\left(4,5\right)}x^{4}x^{5}+\cdots+\alpha_{\left(d+1,d+1\right)}x^{d+1}x^{d+1}}\\ {\scriptstyle{\color[rgb]{0.5,0.5,0.5}\left(d+2\right)\left(d+1\right)/2\>\>\>\mathrm{items}}}\end{matrix} |  | (4) |

where we use i,ji,j to index the channel and α\alpha is a coefficient for each item:

|  | α(i,j)={w1i​w2jif​i==j,w1i​w2j+w1j​w2iif​i!=j.\alpha_{\left(i,j\right)}=\left\{\begin{matrix}w_{1}^{i}w_{2}^{j}&\mathrm{if}\>i==j,\\ w_{1}^{i}w_{2}^{j}+w_{1}^{j}w_{2}^{i}&\mathrm{if}\>i!=j.\\ \end{matrix}\right. |  | (5) |
|---|---|---|---|

Upon rewriting the star operation delineated in Eq. 1, we can expand it into a composition of (d+2)​(d+1)2\frac{\left(d+2\right)\left(d+1\right)}{2} distinct items, as presented in Eq. 4. Of note is that each item (besides α(d+1,:)xd+1x\alpha_{\left(d+1,:\right)}x^{d+1}x) exhibits a non-linear association with xx, indicating that they are individual and implicit dimensions. Therefore, we perform computations within a dd-dimensional space using a computationally efficient star operation, yet we achieve a representation in a (d+2)​(d+1)2≈(d2)2\frac{\left(d+2\right)\left(d+1\right)}{2}\approx\left(\frac{d}{\sqrt{2}}\right)^{2} (considering d≫2d\gg 2) implicit dimensional feature space, significantly amplifying the feature dimensions without incurring any additional computational overhead within a single layer. Of note is that this prominent property shares a similar philosophy as kernel functions, and we refer readers to [25, 47] for a wider and deeper outlook.

### 3.2 Generalized to multiple layers

Next, we demonstrate that by stacking multiple layers, we can exponentially increase the implicit dimensions to nearly infinite in a recursive manner.

Considering an initial network layer with a width of dd, the application of one star operation yields the expression ∑i=1d+1∑j=1d+1w1i​w2j​xi​xj\sum_{i=1}^{d+1}\sum_{j=1}^{d+1}w_{1}^{i}w_{2}^{j}x^{i}x^{j}, as detailed in Eq. 3. This results in a representation within an implicit feature space of ℝ(d2)21\mathbb{R}^{\left(\frac{d}{\sqrt{2}}\right)^{2^{1}}}. Let OlO_{l} denote the output of ll-th star operation, we get:

|  | O1\displaystyle O_{1} | =∑i=1d+1∑j=1d+1w(1,1)iw(1,2)jxixj∈ℝ(d2)21\displaystyle=\sum_{i=1}^{d+1}\sum_{j=1}^{d+1}w_{(1,1)}^{i}w_{(1,2)}^{j}x^{i}x^{j}\;\;\;\;\;\;\;\;\in\mathbb{R}^{\left(\frac{d}{\sqrt{2}}\right)^{2^{1}}} |  | (6) |
|---|---|---|---|---|
|  | O2\displaystyle O_{2} | =W2,1TO1∗W2,2TO1∈ℝ(d2)22\displaystyle=\mathrm{W}_{2,1}^{\mathrm{T}}\mathrm{O_{1}}*\mathrm{W}_{2,2}^{\mathrm{T}}\mathrm{O_{1}}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\in\mathbb{R}^{\left(\frac{d}{\sqrt{2}}\right)^{2^{2}}} |  | (7) |
|  | O3\displaystyle O_{3} | =W3,1TO2∗W3,2TO2∈ℝ(d2)23\displaystyle=\mathrm{W}_{3,1}^{\mathrm{T}}\mathrm{O_{2}}*\mathrm{W}_{3,2}^{\mathrm{T}}\mathrm{O_{2}}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\in\mathbb{R}^{\left(\frac{d}{\sqrt{2}}\right)^{2^{3}}} |  | (8) |
|  | ⋯\displaystyle\cdots |  | (9) |  |
|  | Ol\displaystyle O_{l} | =Wl,1TOl−1∗Wl,2TOl−1∈ℝ(d2)2l\displaystyle=\mathrm{W}_{l,1}^{\mathrm{T}}\mathrm{O_{l-1}}*\mathrm{W}_{l,2}^{\mathrm{T}}\mathrm{O_{l-1}}\;\;\;\;\;\;\;\;\;\;\;\;\;\in\mathbb{R}^{\left(\frac{d}{\sqrt{2}}\right)^{2^{l}}} |  | (10) |

That is, with ll layers, we can implicitly obtain a feature space belonging to ℝ(d2)2l\mathbb{R}^{\left(\frac{d}{\sqrt{2}}\right)^{2^{l}}}. For instance, given a 10-layer isotropic network with a width of 128, the implicit feature dimension number achieved through the star operation approximates 90102490^{1024}, which can be reasonably approximated as infinite dimensions. Therefore, by stacking multiple layers, even just a few, star operations can substantially amplify the implicit dimensions in an exponential manner.

### 3.3 Special Cases

Not all star operations adhere to the formulation presented in Eq. 1, where each branch undergoes a transformation. For instance, VAN[18] and SENet [30] incorporate one identity branch, while GENet-θ−\theta^{-} [29] operates without any learnable transformation. Subsequently, we will delve into the intricacies of these unique cases.

Case I: Non-Linear Nature of 𝐖𝟏\mathbf{W_{1}} and/or 𝐖𝟐\mathbf{W_{2}}. In practical scenarios, a significant number of studies (e.g., Conv2Former, FocalNet, etc.) implement the transformation functions W1\mathrm{W}_{1} and/or W2\mathrm{W}_{2} as non-linear by incorporating activation functions. Nonetheless, a critical aspect is their maintenance of channel communications, as depicted in Eq. 2. Importantly, the number of implicit dimensions remains unchanged (approximately d22\frac{d^{2}}{2}), thereby not affecting our analysis in Sec. 3.1. Hence, we can simply use the linear transformation as a demonstration.

Case II: 𝐖𝟏𝐓​𝐗∗𝐗\mathbf{W_{1}^{T}X*X}. When removing the transformation W2\mathrm{W}_{2}, the implicit dimension number decreases from approximately d22\frac{d^{2}}{2} to 2​d2d.

Case III: 𝐗∗𝐗\mathbf{X*X}. In this case, star operation converts the feature from a feature space {x1,x2,⋯,xd}∈ℝd\left\{x^{1},x^{2},\cdots,x^{d}\right\}\in\mathbb{R}^{d} to a new space characterized by {x1​x1,x2​x2,⋯,xd​xd}∈ℝd\left\{x^{1}x^{1},x^{2}x^{2},\cdots,x^{d}x^{d}\right\}\in\mathbb{R}^{d}.

There are several notable aspects to consider. First, star operations and their special cases are commonly (though not necessarily) integrated with spatial interactions, typically implemented via pooling or convolution, as exemplified in VAN [18]. Many of these approaches emphasize the benefits of an expanded receptive field, yet often overlook the advantages conferred by implicit high-dimensional spaces. Second, it is feasible to combine these special cases, as demonstrated in Conv2Former [26], which merges aspects of Case I and Case II, and in GENet-θ−\theta^{-} [29], which blends elements of Case I and Case III. Lastly, although Cases II and III may not significantly increase the implicit dimensions in a single layer, the use of linear layers (primarily for channel communication) and skip connections can cumulatively achieve high implicit dimensions across multiple layers.

| Width | 96 | 128 | 160 | 192 | 224 | 256 | 288 |
|---|---|---|---|---|---|---|---|
| sum acc. | 51.1 | 58.5 | 62.7 | 66.2 | 69.6 | 71.4 | 72.5 |
| star acc. | 57.6 | 64.0 | 68.2 | 71.7 | 73.9 | 75.3 | 76.1 |
| acc. gap | 6.5↑\uparrow | 5.5↑\uparrow | 5.5↑\uparrow | 5.5↑\uparrow | 4.3↑\uparrow | 3.9↑\uparrow | 3.6↑\uparrow |

| Depth | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|
| sum acc. | 63.8 | 66.3 | 68.2 | 68.8 | 69.9 | 69.6 | 70.6 |
| star acc. | 70.3 | 71.8 | 72.9 | 72.9 | 73.9 | 75.4 | 75.4 |
| acc. gap | 6.5↑\uparrow | 5.5↑\uparrow | 4.7↑\uparrow | 4.1↑\uparrow | 4.0↑\uparrow | 5.8↑\uparrow | 4.8↑\uparrow |

### 3.4 Empirical Study

To substantiate and validate our analysis, we conduct extensive studies on the star operation from various perspectives.

Initially, we empirically validate the superiority of the star operation compared to simple summation. As illustrated in Fig. 1, we build an isotropic network, referred to as DemoNet, for this demonstration. DemoNet is designed to be straightforward, consisting of a convolutional layer that reduces the input resolution by a factor of 16, followed by a sequence of homogeneous demo blocks for feature extraction (refer to Fig. 1, left side). Within each demo block, we apply either the star operation or the summation operation to amalgamate features from two distinct branches. By varying the network’s width and depth, we explore the distinctive attributes of each operation. The implementation details of DemoNet are provided in the supplementary Algorithm 1.

From Table 2 and Table 3, we can see that star operation consistently outperforms sum operation, regardless of the network depth and width. This phenomenon verifies the effectiveness and superiority of star operation. Moreover, we observed that with the increase in network width, the performance gains brought by the star operation gradually diminish. However, we did not observe a similar phenomenon in the case of varying depths. This disparity in behavior suggests two key insights: 1) The gradual decrease in the gains brought by the star operation as shown in Table 2 is not a consequence of the model’s enlarged size; 2) Based on this, it implies that the star operation does intrinsically expand the network’s dimensionality, which in turn lessens the incremental benefits of widening the network.

Subsequently, we visually analyze and discern the differences between the star and summation operations. For this purpose, we visualize the decision boundaries of these two operations on the toy 2D moon dataset [43], which consists of two sets of moon-shaped 2D points. In terms of the model configuration, we eliminate normalization and convolutional layers from the demo block. Given the relatively straightforward nature of this dataset, we configure the model with a width of 100 and a depth of 4.

Fig. 2 (top row) displays the decision boundaries delineated by the sum and star operations. It is evident that the star operation delineates a significantly more precise and effective decision boundary compared to the sum operation. Notably, the observed differences in decision boundaries do not stem from non-linearity, as both operations incorporate activation functions in their respective building blocks. The primary distinction arises from the star operation’s capability to attain exceedingly high dimensionality, a characteristic we have previously analyzed in detail.

As aforementioned, the star operation functions analogously to kernel functions, particularly the polynomial kernel function. To corroborate this, we also illustrate the decision boundaries of SVM with both Gaussian and polynomial kernels (implemented using the scikit-learn package [43]) in Fig. 2 (bottom row). In line with our expectations, the decision boundary produced by the star operation closely mirrors that of the polynomial kernel, while markedly diverging from the Gaussian kernel. This compelling evidence further substantiates the correctness of our analysis.

| operation | w/ act. | w/o act. | Accuracy drop |
|---|---|---|---|
| sum | 66.2 | 32.4 | 33.8↓\downarrow |
| star | 71.7 | 70.5 | 1.2↓\downarrow |

Activation functions are fundamental and indispensable components in neural networks. Commonly employed activations like ReLU and GELU, however, are subject to certain drawbacks such as ‘mean shift’ [4, 24, 22] and information loss [46], among others. The prospect of excluding activation functions from networks is an intriguing and potentially advantageous concept. Nevertheless, without activation functions, traditional neural networks would collapse into a single-layer network due to the lack of non-linearity.

In this study, while our primary focus is on the implicit high-dimensional feature achieved via star operations, the aspect of non-linearity also holds profound importance. To investigate this, we experiment by removing all activations from DemoNet, thus creating an activation-free network. The results in Table 4 are highly encouraging. As expected, the performance of the summation operation deteriorates markedly upon the removal of all activations, from 66.2% to 32.4%. In stark contrast, the star operation experiences only a minimal impact from the elimination of activations, evidenced by a mere 1.2% decrease in accuracy. This experiment not only corroborates our theoretical analysis but also paves the way for expansive avenues in future research.

### 3.5 Open Discussions & Broader Impacts

Although based on the simple operation, our analysis lays the groundwork for exploring fundamental challenges in deep learning. Below, we outline several promising and intriguing research questions that merit further investigation, where the star operation could play a pivotal role.

I. Are activation functions truly indispensable? In our research, we have concentrated on the aspect of implicit high dimensions introduced by star operations. Notably, star operations also incorporate non-linearity, a characteristic that distinguishes kernel functions from other linear machine learning methods. A preliminary experiment in our study demonstrated the potential feasibility of eliminating activation layers in neural networks.

II. How do star operations relate to self-attention and matrix multiplication? Self-attention utilizes matrix multiplication to produce matrices in ℝn×n\mathbb{R}^{n\times n}. It can be demonstrated that matrix multiplication in self-attention shares similar attributes (non-linearity and high dimensionality) with element-wise multiplication. Notably, matrix multiplication facilitates global interactions, in contrast to the element-wise multiplication. However, matrix multiplication alters the input shape, necessitating additional operations (e.g., pooling, another round of matrix multiplication, etc.) to reconcile tensor shape, a complication avoided by element-wise multiplication. PolyNL [2] provides preliminary efforts in this direction. Our analysis could offer new insights into the effectiveness of self-attention and contribute to the revisiting on ‘dynamic’ features [7, 8, 12] in neural networks.

III. How to optimize the coefficient distribution in implicit high-dimensional spaces? Traditional neural networks can learn a distinct set of weight coefficients for each channel, but the coefficients for each implicit dimension in star operations, akin to kernel functions, are fixed. For instance, in the polynomial kernel function k⁡(x1,x2)=(γ​x1⋅x2+c)dk\left(x_{1},x_{2}\right)=\left(\gamma x_{1}\cdot x_{2}+c\right)^{d}, the coefficient distribution can be adjusted via hyper-parameters. In star operations, while the weights W1\mathrm{W}_{1} and W2\mathrm{W}_{2} are learnable, they offer only limited scope for fine-tuning the distribution, as opposed to allowing for customized coefficients for each channel as in traditional neural networks. This constraint might explain why extremely high dimensions result in only moderate performance improvements. Notably, skip connections seem to aid in smoothing the coefficient distribution [57], and dense connections (as in DenseNet [32]) may offer additional benefits. Furthermore, employing exponential functions could provide a direct mapping to implicit infinite dimensions, similar to Gaussian kernel functions.

| Variant | embed | depth | Params | FLOPs |
|---|---|---|---|---|
| StarNet-s1 | 24 | [2, 2, 8, 3] | 2.9M | 425M |
| StarNet-s2 | 32 | [1, 2, 6, 2] | 3.7M | 547M |
| StarNet-s3 | 32 | [2, 2, 8, 4] | 5.8M | 757M |
| StarNet-s4 | 32 | [3, 3, 12, 5] | 7.5M | 1075M |

## 4 Proof-of-Concept: StarNet

Given the unique advantage of the star operation — its ability to compute in a low-dimensional space while yielding high-dimensional features — we identify its utility in the domain of efficient network architectures. Consequently, we introduce StarNet as a proof-of-concept model. StarNet is characterized by an exceedingly minimalist design and a significant reduction in human intervention. Despite its simplicity, StarNet showcases exceptional performance, underscoring the efficacy of the star operation.

### 4.1 StarNet Architecture

StarNet is structured as a 4-stage hierarchical architecture, utilizing convolutional layer for down-sampling and a modified demo block for feature extraction. To meet the requirement of efficiency, we replace Layer Normalization with Batch Normalization and place it after the depth-wise convolution (can be fused during inference). Drawing inspiration from MobileNeXt[66], we incorporate a depth-wise convolution at the end of each block. The channel expansion factor is consistently set at 4, with network width doubling at each stage. The GELU activation in the demo block is substituted with ReLU6, following the MobileNetv2 [46] design. The StarNet framework is illustrated in Fig. 3. We only vary the block numbers and input embedding channel number to build different sizes of StarNet, as detailed in Table 5.

While many advanced design techniques (like re-parameterization, integrating with attention, SE-block, etc.) can empirically enhance performance, but will also obscure our contributions. By deliberately eschewing these sophisticated design elements and minimizing human design intervention, we underscore the pivotal role of star operations in the conceptualization and functionality of StarNet.

| Model | Top-1 | Params | FLOPs | Latency (ms) |  |  |
|---|---|---|---|---|---|---|
|  | (%) | (M) | (M) | Mobile | GPU | CPU |
| MobileOne-S0 [55] | 71.4 | 2.1 | 275 | 0.7 | 1.1 | 2.2 |
| ShuffleV2-1.0 [40] | 69.4 | 2.3 | 146 | 4.1 | 2.2 | 3.8 |
| MobileV3-S0.75 [27] | 65.4 | 2.4 | 44 | 5.5 | 1.8 | 2.5 |
| GhostNet0.5 [19] | 66.2 | 2.6 | 42 | 10.0 | 2.9 | 4.8 |
| MobileV3-S [27] | 67.4 | 2.9 | 66 | 6.5 | 1.8 | 2.6 |
| StarNet-S1 | 73.5 | 2.9 | 425 | 0.7 | 2.3 | 4.3 |
| MobileV2-1.0 [46] | 72.0 | 3.4 | 300 | 0.9 | 2.2 | 3.2 |
| ShuffleV2 1.5 [40] | 72.6 | 3.5 | 299 | 5.9 | 2.2 | 4.9 |
| Mobileformer-52 [9] | 68.7 | 3.6 | 52 | 6.6 | 8.3 | 26.0 |
| FasterNet-T0 [6] | 71.9 | 3.9 | 338 | 0.7 | 2.5 | 5.7 |
| StarNet-S2 | 74.8 | 3.7 | 547 | 0.7 | 2.0 | 4.5 |
| MobileV3-L0.75 [27] | 73.3 | 4.0 | 155 | 10.9 | 2.2 | 4.4 |
| EdgeViT-XXS [42] | 74.4 | 4.1 | 559 | 1.8 | 8.9 | 12.6 |
| MobileOne-S1 [55] | 75.9 | 4.8 | 825 | 0.9 | 1.5 | 6.0 |
| GhostNet1.0 [19] | 73.9 | 5.2 | 141 | 7.9 | 3.6 | 7.0 |
| EfficientNet-B0 [50] | 77.1 | 5.3 | 390 | 1.6 | 3.4 | 8.8 |
| MobileV3-L [27] | 75.2 | 5.4 | 219 | 11.4 | 2.5 | 5.2 |
| StarNet-S3 | 77.3 | 5.8 | 757 | 0.9 | 2.7 | 6.7 |
| EdgeViT-XS [42] | 77.5 | 6.8 | 1166 | 3.5 | 12.1 | 18.3 |
| MobileV2-1.4 [46] | 74.7 | 6.9 | 585 | 1.1 | 2.8 | 5.4 |
| GhostNet1.3 [19] | 75.7 | 7.3 | 226 | 9.7 | 3.9 | 11.0 |
| ShuffleV2-2.0 [40] | 74.9 | 7.4 | 591 | 19.9 | 2.6 | 9.7 |
| Fasternet-T1 [6] | 76.2 | 7.6 | 855 | 0.9 | 3.3 | 9.7 |
| MobileOne-S2 [55] | 77.4 | 7.8 | 1299 | 1.0 | 2.0 | 8.9 |
| StarNet-S4 | 78.4 | 7.5 | 1075 | 1.0 | 3.7 | 9.4 |

### 4.2 Experimental Results

We adhere to a standard training recipe from DeiT [53] to ensure a fair comparison when training our StarNet models. All models are trained from scratch over 300 epochs, utilizing the AdamW optimizer [39] with an initial learning rate of 3e-3 and a batch size of 2048. Comprehensive training details are provided in the supplementary materials. For benchmark purposes, our PyTorch models are converted to the ONNX format [13] to facilitate latency evaluations on both CPU (Intel Xeon CPU E5-2680 v4 @ 2.40GHz) and GPU (P100). Additionally, we deploy the models on iPhone13 using CoreML-Tools [1] to assess latency on mobile devices. Detailed settings for these benchmarks are also available in the supplementary materials.

The experimental results are presented in Table 6. With minimal handcrafted design, our StarNet is able to deliver promising performance in comparison with many other state-of-the-art efficient models. Notably, StarNet achieves a top-1 accuracy of 73.5% in just 0.7 seconds on iPhone 13 device, surpassing MobileOne-S0 by 2.1% (73.5% vs. 71.4%) at the same latency. When scaling the model to a 1G FLOPs budget, StarNet continues to exhibit remarkable performance, outperforming MobileOne-S2 by 1.0%, and surpassing EdgeViT-XS by 0.9% while being three times faster (1.0 ms vs. 3.5 ms). This impressive efficiency, given the model’s straightforward design, can be mainly attributed to the fundamental role of the star operation. Fig. 4 further illustrates the latency-accuracy trade-off among various models. If we can further push the performance of StarNet to a higher level? We believe that through careful hyper-parameter optimization, leveraging insights from Table 1, and applying training enhancements such as more epochs or distillation, substantial improvements could be made to StarNet’s performance. However, achieving a high-performance model is not our primary goal, as such enhancements could potentially obscure the core contributions of the star operation. We left the engineering work behind.

### 4.3 More Ablation studies

The star operation is identified as the sole contributor to the high performance of our model. To empirically validate this assertion, we systematically replaced the star operation with summation in our implementation. Specifically, this entailed substituting the ‘*’ operator with a ‘+’ in the model’s architecture.

The results are delineated in Table 7. Removing all star operations resulted in a notable performance decline, with a 3.1% accuracy drop observed. Interestingly, the impact of the star operation on performance appears minimal in the first and second stages of the model. This observation is logical. With a very narrow width, the ReLU6 activation results in some features turning to zero. In the context of the star operation, this leads to numerous dimensions in its implicit high-dimensional space also becoming zero, thereby restraining its full potential. However, its contribution becomes more pronounced in the last two stages (more channels), leading to 1.6% and 1.6% improvements, respectively. Last three rows in Table 7 also validate our analysis.

Theoretically, multiplication operations (such as the star operation in our study) are understood to have a higher computational complexity compared to simpler summation operations, as indicated in several related works [5, 15]. However, practical latency outcomes may not always align with theoretical predictions. We conducted benchmarks to compare the latency of replacing all star operations with summation, with the results detailed in Table 8. From the table, we observed that the latency impact is contingent on the hardware. In practice, the star operation did not result in any additional latency on GPU and iPhone devices relative to the summation operation. However, the summation operation was slightly more efficient than the star operation on CPU (e.g., 8.4ms vs. 9.4 ms for StarNet-S4). Given the considerable performance gap, this minor latency overhead on CPU can be deemed negligible.

| Stage 1 | Stage 2 | Stage 3 | Stage 4 | Top-1 |
|---|---|---|---|---|
| sum | sum | sum | sum | 75.3 |
| star | sum | sum | sum | 75.1 |
| star | star | sum | sum | 75.2 |
| star | star | star | sum | 76.8 |
| star | star | star | star | 78.4 |
| sum | sum | star | sum | 76.4 |
| sum | sum | sum | star | 76.9 |
| sum | sum | star | star | 78.4 |

| StarNet | Mobile (ms) sum / star | GPU (ms) sum / star | CPU (ms) sum / star |
|---|---|---|---|
| S1 | 0.7 / 0.7 (——) | 2.3 / 2.3 (——) | 3.8 / 4.3 (++0.5) |
| S2 | 0.8 / 0.7 (−- 0.1) | 2.0 / 2.0 (——) | 4.2 / 4.5 (++0.3) |
| S3 | 0.9 / 0.9 (——) | 2.7 / 2.7 (——) | 5.9 / 6.7 (++0.8) |
| S4 | 1.1 / 1.0 (−- 0.1) | 3.7 / 3.7 (——) | 8.4 / 9.4 (++1.0) |

We present a comprehensive analysis regarding the placement of activation functions (ReLU6) within our network block. For clarity, x1x_{1} and x2x_{2} are used to denote the outputs of the two branches, with StarNet-S4 serving as the demonstrative model.

Here, we investigated four approaches to implementing activation functions within StarNet: 1) employing no activations, 2) activating both branches, 3) activating post star operation, and 4) activating a single branch, which is our default practice. The results, as depicted in Table 9, demonstrate that activating only one branch yields the highest accuracy, reaching 78.4%. Inspiringly, the complete removal of activations from StarNet (except one in the stem layer) leads to a mere 2.8% reduction in accuracy, bringing it down to 75.6%, a performance that is still competitive with some strong baselines in Table 6. These findings, consistent with Table 4, underscore the potential of activation-free networks.

| x1x_{1}*x2x_{2} | act(x1x_{1})*act(x2x_{2}) | act(x1x_{1}*x2x_{2}) | act(x1x_{1})*x2x_{2} |
|---|---|---|---|
| (no act.) | (act. both) | (post act.) | (act. one) |
| 75.6 | 78.0 | 77.0 | 78.4 |

In StarNet, the star operation is typically implemented as act⁡(W1T​X)∗(W2T​X)\mathrm{act}\left(\mathrm{W}_{1}^{\mathrm{T}}\mathrm{X}\right)*\left(\mathrm{W}_{2}^{\mathrm{T}}\mathrm{X}\right), as detailed in Sec. 3.1. This standard approach allows StarNet-S4 to reach an accuracy of 84.4%. However, alternative implementations are possible. We experimented with a variation: (W2T​act​(W1T​X))∗X\left(\mathrm{W}_{2}^{\mathrm{T}}\mathrm{act}\left(\mathrm{W}_{1}^{\mathrm{T}}\mathrm{X}\right)\right)*\mathrm{X}, where W1∈ℝd×d′\mathrm{W}_{1}\in\mathbb{R}^{d\times d^{\prime}} is designed to expand the width, and W2∈ℝd′×d\mathrm{W}_{2}\in\mathbb{R}^{d^{\prime}\times d} restores it back to dd. This adjustment results in the transformation of only one branch, while the other remains unaltered. We vary d′d^{\prime} to ensure the same computational complexity as StarNet-S4. By doing so, the performance dropped from 78.4% to 74.4%. While a better and careful design might mitigate this performance gap (see supplementary), the marked difference in accuracy emphasizes the efficacy of our initial implementation in harnessing the star operation’s capabilities, and underscores the critical importance of transforming both branches in star operation.

## 5 Conclusion

In this study, we have delved into the intricate details of the star operation, going beyond the intuitive and plausible explanations as in previous research. We recontextualized the star operations, uncovering that their strong representational capacity is derived from implicitly high-dimensional spaces. In many ways, the star operation mirrors the behavior of polynomial kernel functions. Our analysis was rigorously validated through empirical, theoretical, and visual methods. Our results were mathematically and theoretically solid, aligning coherently with the analysis we have presented. Building on this foundation, we positioned the star operation within the realm of efficient network designs and introduced StarNet, a simple prototype network. StarNet’s impressive performance, achieved without the reliance on sophisticated designs or meticulously chosen hyper-parameters, stands as a testament to the efficacy of star operations. Furthermore, our exploration of the star operation opens up numerous potential research avenues, as we discussed above.

Supplementary Material

This supplementary document elaborates on the implementation details of DemoNet, as showcased in Figure 1, Table 3, and Table 2. It also covers the simple network used in visualizing the decision boundary, illustrated in Figure 2, and the implementation of our StarNet presented in Table 6 (details can be found in Section A). Moreover, we provide a more granular view of decision boundary visualization in Section B. Section C delves into our exploratory studies on ultra-compact models.The analysis of activations is presented in Sec. D. Additionally, a detailed examination of block design is discussed in Section E.

## A Implementation Details

### A.1 Model Architecture

In our isotropic DemoNet, featured in Fig. 1, Table 3, and Table 2, detailed implementation is provided in Algorithm 1. We adjust the depth or width values to facilitate the experiments showcased in the aforementioned figures and tables.

For the illustration of 2D points, as shown in Fig. 2, we have further simplified the DemoNet structure. In this adaptation, all convolutional layers within the original DemoNet have been removed. Additionally, we replaced the GELU activation function with ReLU to streamline the architecture. The details of this simplified DemoNet are outlined in Algorithm 2.

For ease of reproduction, we include a separate file in the supplementary materials dedicated to our StarNet. Detailed information regarding its architecture is also available in Section 4.1.

### A.2 Training Recipes

We next provide detailed training recipe for each experiment.

For all DemoNet variants, we utilize a consistent and standard training recipe. While it’s acknowledged that specialized and finely-tuned training recipes could better suit different model sizes and potentially yield enhanced performance, as in the cases of DemoNet(width=96, depth=12) (approximately 1.26M parameters) and DemoNet(width=288, depth=12) (approximately 9.68M parameters), achieving superior performance with DemoNet is not the primary goal of this work. Our objective is to provide a fair comparison among the various DemoNet variants; hence, the same training recipe is applied across all. Details of this training recipe for DemoNet are presented in Table 10.

| config | value |
|---|---|
| image size | 224 |
| optimizer | AdamW [39] |
| base learning rate | 4e-3 |
| weight decay | 0.05 |
| optimizer momentum | β1,β2=0.9,0.999\beta_{1},\beta_{2}{=}0.9,0.999 |
| batch size | 2048 |
| learning rate schedule | cosine decay [38] |
| warmup epochs | 5 |
| training epochs | 300 |
| AutoAugment | rand-m9-mstd0.5-inc1 [11] |
| label smoothing [41] | 0.1 |
| mixup [64] | 0.8 |
| cutmix [62] | 1.0 |
| color jitter | 0.4 |
| drop path [31] | 0. |
| AMP | False |
| ema | 0.9998 |

Given the inherent simplicity of 2D points, we have eliminated all data augmentation processes and reduced the number of training epochs. The specific training recipe employed for this streamlined process is detailed in Table 11.

| config | value |
|---|---|
| optimizer | SGD |
| base learning rate | 0.1 |
| minimal learning rate | 0.005 |
| learning rate schedule | cosine decay [38] |
| weight decay | 2e-4 |
| optimizer momentum | 0.9 |
| batch size | 32 |
| training epochs | 30 |

Owing to its small model size and straightforward architectural design, StarNet necessitates fewer augmentations for training regularization. Importantly, we opted not to use the commonly employed Exponential Moving Average (EMA) and learnable layer-scale technique [54]. While these methods could potentially enhance performance, they may obscure the unique contributions of our work. The detailed training recipe for various StarNet models is provided in Table 12.

| config | value |
|---|---|
| image size | 224 |
| optimizer | AdamW [39] |
| base learning rate | 3e-3 |
| weight decay | 0.025 |
| optimizer momentum | β1,β2=0.9,0.999\beta_{1},\beta_{2}{=}0.9,0.999 |
| batch size | 2048 |
| learning rate schedule | cosine decay [38] |
| warmup epochs | 5 |
| training epochs | 300 |
| AutoAugment | rand-m1-mstd0.5-inc1 [11] |
| label smoothing [41] | 0.1 |
| mixup [64] | 0.8 |
| cutmix [62] | 0.2 |
| color jitter | 0. |
| drop path [31] | 0.(S1/S2/S3), 0.1(S4) |
| AMP | False |
| EMA | None |
| layer-scale | None |

### A.3 Latency Benchmark Settings

All models listed in Table 6 have been converted from Pytorch code to the ONNX format [13], to enable latency evaluations on different hardware: CPU (Intel Xeon CPU E5-2680 v4 @ 2.40GHz) and GPU (P100). We have conducted benchmarks with a batch size of one, mirroring real-world application scenarios. The benchmark involves a warm-up of 50 iterations, followed by the computation of the average latency over 500 iterations. It’s important to note that all models were benchmarked on the same device to ensure fairness in comparisons. For CPU benchmarks, we utilize 4 threads to optimize performance. In the case of GPU evaluations, we’ve adapted the pointwise convolutional layer in StarNet to a linear layer, incorporating a permutation operation. This modification was prompted by observations of marginally faster inference speeds. It’s important to note that, mathematically, a pointwise convolutional layer is equivalent to a linear layer. Therefore, this change does not lead to any variances in performance or alterations in the architecture of our StarNet.

Thanks to MobileOne [55], we have utilized their open-sourced iOS benchmark application for all CoreML models. Our latency benchmark settings follow those used in MobileOne, with the only difference being the inclusion of additional models in our tests. We have observed that the initial run of the iOS benchmark consistently yields slightly faster results. To account for this, each model is run three times, and we report the latency of the final run. Although there are minor latency variations when testing on iPhone, these discrepancies (typically less than 0.05 ms) do not affect our analysis.

## B Decision Boundary Visualization

We provide more analyses for the decision boundary visualization, as shown in Fig. 2.

Firstly, as a supplement to Fig. 2, we present more comprehensive results. The decision boundary can exhibit significant variation due to randomness in different tests. To illustrate this, we display the decision boundaries from an additional four runs (with no fixed seed) in Fig. 5. As evidenced, the star operation not only surpasses the summation operation in representational capability, but it also demonstrates greater robustness, exhibiting minimal variance.

Next, our investigation extends to a more in-depth analysis of the decision boundary. Considering the network has 4 blocks, we experimented with a mix of summation and star operations, applying them in various combinations as demonstrated in Figure 6. The visual results from this exploration suggest that employing star operations in the early blocks of the network yields the most significant benefits.

Impact of hyper-parameters in SVM. It should be noted that parameter adjustments will lead to different visualization results, which may challenge our claim based on Fig. 2. However, the change is not significant due to the intrinsic differences between ploy and rbf kernels, as shown in Fig. 7. Our star-based network decision boundary is consistently similar to the decision boundary of poly-SVM. Hence, not only theoretical proof, our visual experiments also hold firmly.

## C Exploring Extremely Small Models

In this section, we explore the performance of StarNet under extremely small parameters (around 0.5M, 1.0M, and 1.5M). For these extremely small variants, we further tune the MLP expansion ratio besides block number and base embedding width. The detailed configurations of these very small variants are presented in Table 13.

In this section, we delve into the performance of StarNet when configured with extremely small number of parameters, specifically around 0.5M, 1.0M, and 1.5M. For these ultra-compact variants, we go beyond just adjusting the block number and base embedding width; we also finely tune the MLP expansion ratio. The specific configurations for these very small StarNet variants are detailed in Table 13.

| Variant | width | depth | expand | Params | FLOPs |
|---|---|---|---|---|---|
| StarNet-050 | 16 | [1, 1, 3, 1] | 3 | 0.54M | 92.8M |
| StarNet-100 | 20 | [1, 2, 4, 1] | 4 | 1.04M | 187.1M |
| StarNet-150 | 24 | [1, 2, 4, 2] | 3 | 1.56M | 229.0M |

| Model | Top-1 | Params | FLOPs | Latency (ms) |  |  |
|---|---|---|---|---|---|---|
| (%) | (M) | (M) | Mobile | GPU | CPU |  |
| MobileNetv2-050* | 66.0 | 1.97 | 104.5 | 0.7 | 1.4 | 1.5 |
| StarNet-050 | 55.7 | 0.54 | 92.8 | 0.5 | 1.0 | 1.3 |
| StarNet-100 | 64.3 | 1.04 | 187.1 | 0.6 | 1.3 | 2.3 |
| StarNet-150 | 68.0 | 1.56 | 229.0 | 0.6 | 1.5 | 2.6 |

The results presented in Table 14 demonstrate that our ultra-compact variants of StarNet are also promising in performance. When compared to MobileNetv2-050, which is trained for longer training period and introduces approximately 25% more parameters (1.97M vs. 1.56M), our StarNet-150 variant still outperforms in both top-1 accuracy and speed on mobile devices.

## D Activation Analysis

### D.1 Analysis on removing all activations

In Sec. 3.5, we explored the possibility of eliminating all activation functions, presenting preliminary findings in Table 4 and Table 9. This section delves deeper into the detailed analyses of removing all activations. Utilizing DemoNet as our model for illustration, we provide further results in Table 15 and Table 16.

| Width | 96 | 128 | 160 | 192 | 224 | 256 | 288 |
|---|---|---|---|---|---|---|---|
| w/ act. | 57.6 | 64.0 | 68.2 | 71.7 | 73.9 | 75.3 | 76.1 |
| w/o act. | 57.6 | 64.0 | 67.5 | 70.5 | 73.0 | 75.3 | 75.7 |
| acc. gap | - | - | 0.7↓\downarrow | 1.2↓\downarrow | 0.9↓\downarrow | - | 0.4↓\downarrow |

| Depth | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|
| w/ act. | 70.3 | 71.8 | 72.9 | 72.9 | 73.9 | 75.4 | 75.4 |
| w/o act. | 69.4 | 70.5 | 72.6 | 73.5 | 74.2 | 74.7 | 75.5 |
| acc. gap | 0.9↓\downarrow | 1.3↓\downarrow | 0.3↑\uparrow | 0.6↑\uparrow | 0.3↑\uparrow | 0.7↓\downarrow | 0.1↑\uparrow |

In most cases, we observe slight performance drop when removing all activations from DemoNet (using star operation). However, an intriguing observation emerged: in some instances, the removal of all activations resulted in similar or even better performance. This was notably evident when the depth\mathrm{depth} values were set to 14, 16, 18, and 22, as detailed in Table 16. This phenomenon implies that the star operation may inherently provide sufficient non-linearity, akin to what is typically achieved through activation layers. We believe that a more thorough investigation in this direction could yield valuable insights.

### D.2 Exploring activation types

In our StarNet design, we adopt the ReLU6 activation function, following MobileNetv2. Additionally, we have experimented with various other activation functions, with the results of these explorations presented in Table 17. Empirically, we found that StarNet-S4 delivers the best performance when equipped with the ReLU6 activation function.

| Act. | ReLU | GELU | LeakyReLU | HardSwish | ReLU6 |
|---|---|---|---|---|---|
|  | max⁡(0,x)\mathrm{max}(0,x) | x2​(1+erf​(x2))\frac{x}{2}\left(1+\mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right) | max⁡(0.01​x,x)\mathrm{max}(0.01x,x) | x​ReLU6⁡(x+3)6x\frac{\mathrm{ReLU6}\left(x+3\right)}{6} | min⁡(max⁡(0,x),6)\mathrm{min}(\mathrm{max}(0,x),6) |
| Acc. | 77.6 | 77.8 | 77.7 | 77.7 | 78.4 |

## E Exploring Block Designs

We provide detailed ablation studies on the design of the StarNet block. To this end, we present five implementations of star operation in StarNet, as illustrated in Fig. 8. Of note is that Block I can be considered as a standard implementation of star operation, Block II and Block III can be considered as instantiations of special case II, which is discussed in Sec.3.3, Block IV and Block V can be considered as different implementations of special case III. All the block variants have same parameters and FLOPs via varying expansion ratio, ensuring same computational complexity for fair comparison. We test all these block variants based on StarNet-S4 architecture, and report the performance in Table 18. Empirically, we see strong performance for Block I, Block IV, and Block V, while Block II and Block III perform worse. The detailed results suggest that the strong performance stems from the star operation rather than specific block design. We consider the default star operation (Block I) as the essential building block in our StarNet.

We have conducted detailed ablation studies on the design of the StarNet block. To this end, we explored five different implementations of the star operation in StarNet, as depicted in Fig.8. Notably, Block I represents a standard implementation of the star operation. Blocks II and III are instantiations of the special case II, discussed in Sec.3.3, while Blocks IV and V offer alternative takes on special case III. All these block variants maintain the same number of parameters and FLOPs by adjusting the expansion ratio, ensuring the same computational complexity for a fair comparison. These block variants were tested within the StarNet-S4 architecture framework, with their performance detailed in Table18. Empirically, Blocks I, IV, and V demonstrated strong performance, with minimal performance gap. These findings suggest that the effectiveness is more attributable to the star operation itself rather than to the specific design of the blocks. Therefore, we use the default star operation (Block I) as the foundational block in our StarNet.

| Star Op. | Design | Top-1 |
|---|---|---|
| Standard | Block I | 78.4 |
| Case II | Block II | 74.4 |
| Block III | 74.4 |  |
| Case III | Block IV | 78.5 |
| Block V | 78.6 |  |

## F More Latency Analysis on StarNet

CPU latency Visualization. To better understand the latency of the proof-of-concept model StarNet, we further plot the CPU-latency trade-off in Fig. 9.

Mobile Device latency Robustness. In Table 6, we presented the mobile device latency, which is tested on a iPhone13 mobile phone. We further conducted latency testing on 4 different mobile devices, including iPhone12, iPhone12 Pro Max, iPhone 13, and iPhone14, to test the latency stability of different models. Results in Tab. 19 demonstrate that, despite varying latency results for some models, StarNet always shows stable inference across different phones, attributed to its simple design.

| Model | Top-1 | iPhone devices Latency (ms) |  |  |  |  |
|---|---|---|---|---|---|---|
| (%) | 12 | 12PM | 13 | 14 | avg±\pmstd |  |
| MobileOne-S0 | 71.4 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7±\pm0.02 |
| ShuffleV2-1.0 | 69.4 | 4.1 | 4.1 | 0.8 | 0.8 | 2.4±\pm1.66 |
| MobileV3-S0.75 | 65.4 | 5.5 | 5.3 | 6.5 | 6.6 | 6.0±\pm0.59 |
| GhostNet0.5 | 66.2 | 10.0 | 7.3 | 9.7 | 8.8 | 8.9±\pm1.05 |
| MobileV3-S | 67.4 | 6.5 | 5.8 | 7.5 | 7.7 | 6.9±\pm0.75 |
| StarNet-S1 | 73.5 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7±\pm0.00 |
| MobileV2-1.0 | 72.0 | 0.9 | 0.9 | 1.0 | 0.9 | 0.9±\pm0.04 |
| ShuffleV21.5 | 72.6 | 5.9 | 5.9 | 1.2 | 1.2 | 3.5±\pm2.34 |
| M-Former-52 | 68.7 | 6.6 | 6.3 | 8.3 | 8.2 | 7.4±\pm0.93 |
| FasterNet-T0 | 71.9 | 0.7 | 0.7 | 0.8 | 0.7 | 0.7±\pm0.02 |
| StarNet-S2 | 74.8 | 0.7 | 0.8 | 0.8 | 0.8 | 0.8±\pm0.01 |
| MobileV3-L0.75 | 73.3 | 10.9 | 11.4 | 12.4 | 14.2 | 12.2±\pm1.26 |
| EdgeViT-XXS | 74.4 | 1.8 | 1.8 | 1.2 | 1.2 | 1.5±\pm0.31 |
| MobileOne-S1 | 75.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9±\pm0.03 |
| GhostNet1.0 | 73.9 | 7.9 | 7.4 | 10.1 | 10.2 | 8.9±\pm1.28 |
| EfficientNet-B0 | 77.1 | 1.6 | 1.5 | 1.7 | 1.7 | 1.6±\pm0.10 |
| MobileV3-L | 75.2 | 11.4 | 12.9 | 15.1 | 14.9 | 13.6±\pm1.52 |
| StarNet-S3 | 77.3 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9±\pm0.02 |
| EdgeViT-XS | 77.5 | 3.5 | 3.5 | 1.6 | 1.6 | 2.5±\pm0.96 |
| MobileV2-1.4 | 74.7 | 1.1 | 1.2 | 1.3 | 1.3 | 1.2±\pm0.08 |
| GhostNet1.3 | 75.7 | 9.7 | 9.0 | 12.4 | 12.5 | 10.9±\pm1.57 |
| ShuffleV2-2.0 | 74.9 | 19.9 | 16.9 | 1.8 | 1.5 | 10.0±\pm8.44 |
| Fasternet-T1 | 76.2 | 0.9 | 0.9 | 1.0 | 1.0 | 1.0±\pm0.03 |
| MobileOne-S2 | 77.4 | 1.0 | 1.1 | 1.2 | 1.2 | 1.1±\pm0.07 |
| StarNet-S4 | 78.4 | 1.0 | 1.0 | 1.1 | 1.1 | 1.1±\pm0.03 |
