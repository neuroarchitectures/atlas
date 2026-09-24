# CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows

We present CSWin Transformer, an efficient and effective Transformer-based backbone for general-purpose vision tasks. A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token. To address this issue, we develop the Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. We provide a mathematical analysis of the effect of the stripe width and vary the stripe width for different layers of the Transformer network which achieves strong modeling capability while limiting the computation cost. We also introduce Locally-enhanced Positional Encoding (LePE), which handles the local positional information better than existing encoding schemes. LePE naturally supports arbitrary input resolutions, and is thus especially effective and friendly for downstream tasks. Incorporated with these designs and a hierarchical structure, CSWin Transformer demonstrates competitive performance on common vision tasks. Specifically, it achieves 85.4% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task, and 52.2 mIOU on the ADE20K semantic segmentation task, surpassing previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4, and +2.0 respectively under the similar FLOPs setting. By further pretraining on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K and high segmentation performance on ADE20K with 55.7 mIoU.

## 1 Introduction

Transformer-based architectures [17, 53, 38, 60] have recently achieved competitive performances compared to their CNN counterparts in various vision tasks. By leveraging the multi-head self-attention mechanism, these vision Transformers demonstrate a high capability in modeling the long-range dependencies, which is especially helpful for handling high-resolution inputs in downstream tasks, e.g., object detection and segmentation. Despite the success, the Transformer architecture with full-attention mechanism [17] is computationally inefficient.

To improve the efficiency, one typical way is to limit the attention region of each token from full-attention to local/windowed attention [38, 55]. To bridge the connection between windows, researchers further proposed halo and shift operations to exchange information through nearby windows. However, the receptive field is enlarged quite slowly and it requires stacking a great number of blocks to achieve global self-attention. A sufficiently large receptive field is crucial to the performance especially for the downstream tasks(e.g., object detection and segmentation). Therefore it is important to achieve large receptive filed efficiently while keeping the computation cost low.

In this paper, we present the Cross-Shaped Window (CSWin) self-attention, which is illustrated in Figure 1 and compared with existing self-attention mechanisms. With CSWin self-attention, we perform the self-attention calculation in the horizontal and vertical stripes in parallel, with each stripe obtained by splitting the input feature into stripes of equal width. This stripe width is an important parameter of the cross-shaped window because it allows us to achieve strong modelling capability while limiting the computation cost. Specifically, we adjust the stripe width according to the depth of the network: small widths for shallow layers and larger widths for deep layers. A larger stripe width encourages a stronger connection between long-range elements and achieves better network capacity with a small increase in computation cost. We will provide a mathematical analysis of how the stripe width affects the modeling capability and computation cost.

It is worthwhile to note that with CSWin self-attention mechanism, the self-attention in horizontal and vertical stripes are calculated in parallel. We split the multi-heads into parallel groups and apply different self-attention operations onto different groups. This parallel strategy introduces no extra computation cost while enlarging the area for computing self-attention within each Transformer block. This strategy is fundamentally different from existing self-attention mechanisms [56, 38, 69, 25] that apply the same attention operation across multi-heads((Figure 1 b,c,d,e), and perform different attention operations sequentially(Figure 1 c,e). We will show through ablation analysis that this difference makes CSWin self-attention much more effective for general vision tasks.

Based on the CSWin self-attention mechanism, we follow the hierarchical design and propose a new vision Transformer architecture named “CSWin Transformer” for general-purpose vision tasks. This architecture provides significantly stronger modeling power while limiting computation cost. To further enhance this vision Transformer, we introduce an effective positional encoding, Locally-enhanced Positional Encoding (LePE), which is especially effective and friendly for input varying downstream tasks such as object detection and segmentation. Compared with previous positional encoding methods [56, 46, 12], our LePE imposes the positional information within each Transformer block and directly operates on the attention results instead of the attention calculation. The LePE makes CSWin Transformer more effective and friendly for the downstream tasks.

As a general vision Transformer backbone, the CSWin Transformer demonstrates strong performance on image classification, object detection and semantic segmentation tasks. Under the similar FLOPs and model size, CSWin Transformer variants significantly outperforms previous state-of-the-art (SOTA) vision Transformers. For example, our base variant CSWin-B achieves 85.4% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task, 51.7 mIOU on the ADE20K semantic segmentation task, surpassing previous state-of-the-art Swin Transformer counterpart by +1.2, +2.0, 1.4 and +2.0 respectively. Under a smaller FLOPs setting, our tiny variant CSWin-T even shows larger performance gains, i.e.,, +1.4 point on ImageNet classification, +3.0 box AP, +2.0 mask AP on COCO detection and +4.6 on ADE20K segmentation. Furthermore, when pretraining CSWin Transformer on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K and high segmentation performance on ADE20K with 55.7 mIoU.

## 2 Related Work

Vision Transformers. Convolutional neural networks (CNN) have dominated the computer vision field for many years and achieved tremendous successes [35, 47, 50, 22, 28, 7, 27, 51, 26, 45, 49]. Recently, the pioneering work ViT [17] demonstrates that pure Transformer-based architectures can also achieve very competitive results, indicating the potential of handling the vision tasks and natural language processing (NLP) tasks under a unified framework. Built upon the success of ViT, many efforts have been devoted to designing better Transformer based architectures for various vision tasks, including low-level image processing [5, 57], image classification [53, 66, 64, 13, 20, 58, 11, 60, 11, 65, 31, 54, 18, 23], object detection [3, 73] and semantic segmentation [59, 70, 48]. Rather than concentrating on one special task, some recent works [58, 69, 38] try to design a general vision Transformer backbone for general-purpose vision tasks. They all follow the hierarchical Transformer architecture but adopt different self-attention mechanisms. The main benefit of the hierarchical design is to utilize the multi-scale features and reduce the computation complexity by progressively decreasing the number of tokens. In this paper,we propose a new hierarchical vision Transformer backbone by introducing cross-shaped window self-attention and locally-enhanced positional encoding.

Efficient Self-attentions. In the NLP field, many efficient attention mechanisms [9, 42, 10, 32, 34, 52, 44, 1] have been designed to improve the Transformer efficiency for handling long sequences. Since the image resolution is often very high in vision tasks, designing efficient self-attention mechanisms is also very crucial. However, many existing vision Transformers [17, 53, 66, 60] still adopt the original full self-attention, whose computation complexity is quadratic to the image size. To reduce the complexity, the recent vision Transformers [38, 55] adopt the local self-attention mechanism [43] and its shifted/haloed version to add the interaction across different local windows. Besides, axial self-attention[25] and criss-cross attention[30] propose calculating attention within stripe windows along horizontal or/and vertical axis. While the performance of axial attention is limited by its sequential mechanism and restricted window size, criss-cross attention is inefficient in practice due to its overlapped window design and ineffective due to its restricted window size. They are the most related works with our CSWin, which could be viewed as a much general and efficient format of these previous works.

Positional Encoding. Since self-attention is permutation-invariant and ignores the token positional information, positional encoding is widely used in Transformers to add such positional information back. Typical positional encoding mechanisms include absolute positional encoding (APE) [56], relative positional encoding (RPE) [46, 38] and conditional positional encoding (CPE) [12]. APE and RPE are often defined as the sinusoidal functions of a series of frequencies or the learnable parameters, which are designed for a specific input size and are not friendly to varying input resolutions. CPE takes the feature as input and can generate the positional encoding for arbitrary input resolutions. Then the generated positional encoding will be added onto the input feature. Our LePE shares a similar spirit as CPE, but proposes to add the positional encoding as a parallel module to the self-attention operation and operates on projected values in each Transformer block. This design decouples positional encoding from the self-attention calculation, and can enforce stronger local inductive bias.

## 3 Method

### 3.1 Overall Architecture

The overall architecture of CSWin Transformer is illustrated in Figure 2. For an input image with size of H×W×3H\times W\times 3, we follow [60] and leverage the overlapped convolutional token embedding (7×77\times 7 convolution layer with stride 4) ) to obtain H4×W4\frac{H}{4}\times\frac{W}{4} patch tokens, and the dimension of each token is CC. To produce a hierarchical representation, the whole network consists of four stages. A convolution layer (3×33\times 3, stride 2) is used between two adjacent stages to reduce the number of tokens and double the channel dimension. Therefore, the constructed feature maps have H2i+1×W2i+1\frac{H}{2^{i+1}}\times\frac{W}{2^{i+1}} tokens for the it​hi^{th} stage, which is similar to traditional CNN backbones like VGG/ResNet. Each stage consists of NiN_{i} sequential CSWin Transformer Blocks and maintains the number of tokens. CSWin Transformer Block has the overall similar topology as the vanilla multi-head self-attention Transformer block with two differences: 1) It replaces the self-attention mechanism with our proposed Cross-Shaped Window Self-Attention; 2) In order to introduce the local inductive bias, LePE is added as a parallel module to the self-attention branch.

### 3.2 Cross-Shaped Window Self-Attention

Despite the strong long-range context modeling capability, the computation complexity of the original full self-attention mechanism is quadratic to feature map size. Therefore, it will suffer from huge computation cost for vision tasks that take high resolution feature maps as input, such as object detection and segmentation. To alleviate this issue, existing works [38, 55] suggest to perform self-attention in a local attention window and apply halo or shifted window to enlarge the receptive filed. However, the token within each Transformer block still has limited attention area and requires stacking more blocks to achieve global receptive filed. To enlarge the attention area and achieve global self-attention more efficiently, we present the cross-shaped window self-attention mechanism, which is achieved by performing self-attention in horizontal and vertical stripes in parallel that form a cross-shaped window.

Horizontal and Vertical Stripes. According to the multi-head self-attention mechanism, the input feature X∈𝑹(H×W)×CX\in\bm{R}^{(H\times W)\times C} will be first linearly projected to KK heads, and then each head will perform local self-attention within either the horizontal or vertical stripes.

For horizontal stripes self-attention, XX is evenly partitioned into non-overlapping horizontal stripes [X1,..,XM][X^{1},..,X^{M}] of equal width s​wsw, and each of them contains s​w×Wsw\times W tokens. Here, s​wsw is the stripe width and can be adjusted to balance the learning capacity and computation complexity. Formally, suppose the projected queries, keys and values of the kt​hk^{th} head all have dimension dkd_{k}, then the output of the horizontal stripes self-attention for kt​hk^{th} head is defined as:

|  |  | X=[X1,X2,…,XM],\displaystyle X=[X^{1},X^{2},\dots,X^{M}], |  | (1) |
|---|---|---|---|---|
|  |  | Yki=Attention​(Xi​WkQ,Xi​WkK,Xi​WkV),\displaystyle Y^{i}_{k}=\text{Attention}(X^{i}W^{Q}_{k},X^{i}W^{K}_{k},X^{i}W^{V}_{k}), |  |  |
|  |  | H​-​Attentionk​(X)=[Yk1,Yk2,…,YkM]\displaystyle\mathrm{H\text{-}Attention}_{k}(X)=[Y^{1}_{k},Y^{2}_{k},\dots,Y^{M}_{k}] |  |  |

Where where​Xi∈𝑹(s​w×W)×C\text{where}~X^{i}\in\bm{R}^{(sw\times W)\times C} and M=H/s​wM=H/sw, i=1,…,Mi=1,\dots,M. WkQ∈𝑹C×dkW^{Q}_{k}\in\bm{R}^{C\times d_{k}}, WkK∈𝑹C×dkW^{K}_{k}\in\bm{R}^{C\times d_{k}}, WkV∈𝑹C×dkW^{V}_{k}\in\bm{R}^{C\times d_{k}} represent the projection matrices of queries, keys and values for the kt​hk^{th} head respectively, and dkd_{k} is set as C/KC/K. The vertical stripes self-attention can be similarly derived, and its output for kt​hk^{th} head is denoted as V​-​Attentionk​(X)\mathrm{V\text{-}Attention}_{k}(X).

Assuming natural images do not have directional bias, we equally split the KK heads into two parallel groups (each has K/2K/2 heads, KK is often an even value). The first group of heads perform horizontal stripes self-attention while the second group of heads perform vertical stripes self-attention. Finally the output of these two parallel groups will be concatenated back together.

|  |  | CSWin​-​Attention​(X)=Concat⁡(head1,…,headK)​WO\displaystyle\mathrm{CSWin\text{-}Attention}(X)=\mathrm{Concat}(\mathrm{head_{1}},...,\mathrm{head_{K}})W^{O} |  | (2) |
|---|---|---|---|---|
|  |  | headk={H​-​Attentionk​(X)k=1,…,K/2V​-​Attentionk​(X)k=K/2+1,…,K\displaystyle\mathrm{head_{k}}=\begin{cases}\mathrm{H\text{-}Attention}_{k}(X)&k=1,\dots,K/2\\ \mathrm{V\text{-}Attention}_{k}(X)&k=K/2+1,\dots,K\end{cases} |  |  |

Where WO∈𝑹C×CW^{O}\in\bm{R}^{C\times C} is the commonly used projection matrix that projects the self-attention results into the target output dimension (set as CC by default). As described above, one key insight in our self-attention mechanism design is splitting the multi-heads into different groups and applying different self-attention operations accordingly. In other words, the attention area of each token within one Transformer block is enlarged via multi-head grouping. By contrast, existing self-attention mechanisms apply the same self-attention operations across different multi-heads. In the experiment parts, we will show that this design will bring better performance.

Computation Complexity Analysis. The computation complexity of CSWin self-attention is:

|  | Ω⁡(CSWin)=H​W​C∗(4​C+s​w∗H+s​w∗W)\displaystyle\Omega(\text{CSWin})=HWC*(4C+sw*H+sw*W) |  | (3) |
|---|---|---|---|

For high-resolution inputs, considering H,WH,W will be larger than CC in the early stages and smaller than CC in the later stages, we choose small s​wsw for early stages and larger s​wsw for later stages. In other words, adjusting s​wsw provides the flexibility to enlarge the attention area of each token in later stages in an efficient way. Besides, to make the intermediate feature map size divisible by s​wsw for 224×224224\times 224 input, we empirically set s​wsw to 1,2,7,71,2,7,7 for four stages by default.

Locally-Enhanced Positional Encoding. Since the self-attention operation is permutation-invariant, it will ignore the important positional information within the 2D image. To add such information back, different positional encoding mechanisms have been utilized in existing vision Transformers. In Figure 3, we show some typical positional encoding mechanisms and compare them with our proposed locally-enhanced positional encoding. In details, APE [56] and CPE [12] add the positional information into the input token before feeding into the Transformer blocks, while RPE [46] and our LePE incorporate the positional information within each Transformer block. But different from RPE that adds the positional information within the attention calculation (i.e., Softmax​(Q​KT)\text{Softmax}(QK^{T})), we consider a more straightforward manner and impose the positional information upon the linearly projected values. Meanwhile, we notice that RPE introduces bias in a per head manner, while our LePE is a per-channel bias, which may show more potential to serve as positional embeddings.

Mathematically, we denote the input sequence as x=(x1,…,xn)x=(x_{1},\dots,x_{n}) of nn elements, and the output of the attention z=(z1,…,zn)z=(z_{1},\dots,z_{n}) of the same length, where xi,zi∈RCx_{i},z_{i}\in R^{C}. Self-attention computation could be formulated as:

|  | zi=∑j=1nαi​j​vj,αi​j=e​x​p​(qiT​kj/d)\vskip-2.84526ptz_{i}=\sum_{j=1}^{n}{\alpha_{ij}v_{j}},\alpha_{ij}=exp(q_{i}^{T}k_{j}/\sqrt{d})\vskip-2.84526pt |  | (4) |
|---|---|---|---|

where qi,ki,viq_{i},k_{i},v_{i} are the q​u​e​u​e,k​e​yqueue,key and v​a​l​u​evalue get by a linear transformation of the input xix_{i} and dd is the feature dimension. Then our Locally-Enhanced position encoding performs as a learnable per-element bias and Eq.4 could be formulated as:

|  | zik=∑j=1n(αi​jk+βi​jk)​vjk\vskip-2.84526ptz_{i}^{k}=\sum_{j=1}^{n}{(\alpha_{ij}^{k}+\beta_{ij}^{k})v_{j}^{k}}\vskip-2.84526pt |  | (5) |
|---|---|---|---|

where zikz_{i}^{k} represents the kt​hk^{th} element of vector ziz_{i}. To make the LePE suitable to varying input size, we set a distance threshold to the LePE and set it to 00 if the Chebyshev distance of token ii and jj is greater than a threshold τ\tau (τ=3\tau=3 in the default setting).

### 3.3 CSWin Transformer Block

Equipped with the above self-attention mechanism and positional embedding mechanism, CSWin Transformer block is formally defined as:

|  |  | X^l=CSWin-Attention​(LN​(Xl−1))+Xl−1,\displaystyle{{\hat{{X}}}^{l}}=\text{CSWin-Attention}\left({\text{LN}\left({{{{X}}^{l-1}}}\right)}\right)+{{X}}^{l-1}, |  |  |
|---|---|---|---|---|
|  |  | Xl=MLP​(LN​(X^l))+X^l,\displaystyle{{{X}}^{l}}=\text{MLP}\left({\text{LN}\left({{{\hat{{X}}}^{l}}}\right)}\right)+{{\hat{{X}}}^{l}}, |  | (6) |

where Xl{{X}}^{l} denotes the output of ll-th Transformer block or the precedent convolutional layer of each stage.

| Models | #Dim | #Blocks | s​wsw | #heads | #Param. | FLOPs |
|---|---|---|---|---|---|---|
| CSWin-T | 64 | 1,2,21,1 | 1,2,7,7 | 2,4,8,16 | 23M | 4.3G |
| CSWin-S | 64 | 2,4,32,2 | 1,2,7,7 | 2,4,8,16 | 35M | 6.9G |
| CSWin-B | 96 | 2,4,32,2 | 1,2,7,7 | 4,8,16,32 | 78M | 15.0G |
| CSWin-L | 144 | 2,4,32,2 | 1,2,7,7 | 6,12,24,48 | 173M | 31.5G |

### 3.4 Architecture Variants

For a fair comparison with other vision Transformers under similar settings, we build four different variants of CSWin Transformer as shown in Table 1: CSWin-T (Tiny), CSWin-S (Small), CSWin-B (Base), CSWin-L (Large). They are designed by changing the base channel dimension CC and the block number of each stage. In all these variants, the expansion ratio of each MLP is set as 44. The head number of the four stages is set as 2,4,8,162,4,8,16 in the first three variants and 6,12,24,486,12,24,48 in the last variant respectively.

## 4 Experiments

To show the effectiveness of CSWin Transformer as a general vision backbone, we conduct experiments on ImageNet-1K [16] classification, COCO [37] object detection, and ADE20K [72] semantic segmentation. We also perform comprehensive ablation studies to analyze each component of CSWin Transformer. As most of the methods we compared did not report downstream inference speed, we use an extra section to report it for simplicity.

| Method | Image Size | #Param. | FLOPs | Throughput | Top-1 |
|---|---|---|---|---|---|
| Eff-B4 [51] | 3802380^{2} | 19M | 4.2G | 349/s | 82.9 |
| Eff-B5 [51] | 4562456^{2} | 30M | 9.9G | 169/s | 83.6 |
| Eff-B6 [51] | 5282528^{2} | 43M | 19.0G | 96/s | 84.0 |
| DeiT-S [53] | 2242224^{2} | 22M | 4.6G | 940/s | 79.8 |
| DeiT-B [53] | 2242224^{2} | 87M | 17.5G | 292/s | 81.8 |
| DeiT-B [53] | 3842384^{2} | 86M | 55.4G | 85/s | 83.1 |
| PVT-S [58] | 2242224^{2} | 25M | 3.8G | 820/s | 79.8 |
| PVT-M [58] | 2242224^{2} | 44M | 6.7G | 526/s | 81.2 |
| PVT-L [58] | 2242224^{2} | 61M | 9.8G | 367/s | 81.7 |
| T2Tt-14 [66] | 2242224^{2} | 22M | 6.1G | – | 81.7 |
| T2Tt-19 [66] | 2242224^{2} | 39M | 9.8G | – | 82.2 |
| T2Tt-24 [66] | 2242224^{2} | 64M | 15.0G | – | 82.6 |
| CvT-13 [60] | 2242224^{2} | 20M | 4.5G | – | 81.6 |
| CvT-21 [60] | 2242224^{2} | 32M | 7.1G | – | 82.5 |
| CvT-21 [60] | 3842384^{2} | 32M | 24.9G | – | 83.3 |
| Swin-T [38] | 2242224^{2} | 29M | 4.5G | 755/s | 81.3 |
| Swin-S [38] | 2242224^{2} | 50M | 8.7G | 437/s | 83.0 |
| Swin-B [38] | 2242224^{2} | 88M | 15.4G | 278/s | 83.3 |
| Swin-B [38] | 3842384^{2} | 88M | 47.0G | 85/s | 84.2 |
| CSWin-T | 2242224^{2} | 23M | 4.3G | 701/s | 82.7 |
| CSWin-S | 2242224^{2} | 35M | 6.9G | 437/s | 83.6 |
| CSWin-B | 2242224^{2} | 78M | 15.0G | 250/s | 84.2 |
| CSWin-B | 3842384^{2} | 78M | 47.0G |  | 85.4 |

| Method | Param | Size | FLOPs | Top-1 | Method | Param | Size | FLOPs | Top-1 |
|---|---|---|---|---|---|---|---|---|---|
| R-101x3 | 388M | 3842 | 204.6G | 84.4 | R-152x4 | 937M | 4802 | 840.5G | 85.4 |
| ViT-B/16 | 86M | 3842 | 55.4G | 84.0 | ViT-L/16 | 307M | 3842 | 190.7G | 85.2 |
| Swin-B | 88M | 2242 | 15.4G | 85.2 | Swin-L | 197M | 2242 | 34.5G | 86.3 |
| 3842 | 47.1G | 86.4 | 3842 | 103.9G | 87.3 |  |  |  |  |
|  |  | 2242 | 15.0G | 85.9 |  |  | 2242 | 31.5G | 86.5 |
| CSWin-B | 78M | 3842 | 47.0G | 87.0 | CSWin-L | 173M | 3842 | 96.8G | 87.5 |

| Backbone | #Params | FLOPs | Mask R-CNN 1x schedule | Mask R-CNN 3x + MS schedule |  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (M) | (G) | A​PbAP^{b} | A​P50bAP^{b}_{50} | A​P75bAP^{b}_{75} | A​PmAP^{m} | A​P50mAP^{m}_{50} | A​P75mAP^{m}_{75} | A​PbAP^{b} | A​P50bAP^{b}_{50} | A​P75bAP^{b}_{75} | A​PmAP^{m} | A​P50mAP^{m}_{50} | A​P75mAP^{m}_{75} |  |
| Res50 [22] | 44 | 260 | 38.0 | 58.6 | 41.4 | 34.4 | 55.1 | 36.7 | 41.0 | 61.7 | 44.9 | 37.1 | 58.4 | 40.1 |
| PVT-S [58] | 44 | 245 | 40.4 | 62.9 | 43.8 | 37.8 | 60.1 | 40.3 | 43.0 | 65.3 | 46.9 | 39.9 | 62.5 | 42.8 |
| ViL-S [69] | 45 | 218 | 44.9 | 67.1 | 49.3 | 41.0 | 64.2 | 44.1 | 47.1 | 68.7 | 51.5 | 42.7 | 65.9 | 46.2 |
| TwinsP-S [11] | 44 | 245 | 42.9 | 65.8 | 47.1 | 40.0 | 62.7 | 42.9 | 46.8 | 69.3 | 51.8 | 42.6 | 66.3 | 46.0 |
| Twins-S [11] | 44 | 228 | 43.4 | 66.0 | 47.3 | 40.3 | 63.2 | 43.4 | 46.8 | 69.2 | 51.2 | 42.6 | 66.3 | 45.8 |
| Swin-T [38] | 48 | 264 | 42.2 | 64.6 | 46.2 | 39.1 | 61.6 | 42.0 | 46.0 | 68.2 | 50.2 | 41.6 | 65.1 | 44.8 |
| CSWin-T | 42 | 279 | 46.7 | 68.6 | 51.3 | 42.2 | 65.6 | 45.4 | 49.0 | 70.7 | 53.7 | 43.6 | 67.9 | 46.6 |
| Res101 [22] | 63 | 336 | 40.4 | 61.1 | 44.2 | 36.4 | 57.7 | 38.8 | 42.8 | 63.2 | 47.1 | 38.5 | 60.1 | 41.3 |
| X101-32 [63] | 63 | 340 | 41.9 | 62.5 | 45.9 | 37.5 | 59.4 | 40.2 | 44.0 | 64.4 | 48.0 | 39.2 | 61.4 | 41.9 |
| PVT-M [58] | 64 | 302 | 42.0 | 64.4 | 45.6 | 39.0 | 61.6 | 42.1 | 44.2 | 66.0 | 48.2 | 40.5 | 63.1 | 43.5 |
| ViL-M [69] | 60 | 261 | 43.4 | —- | —- | 39.7 | —- | —- | 44.6 | 66.3 | 48.5 | 40.7 | 63.8 | 43.7 |
| TwinsP-B [11] | 64 | 302 | 44.6 | 66.7 | 48.9 | 40.9 | 63.8 | 44.2 | 47.9 | 70.1 | 52.5 | 43.2 | 67.2 | 46.3 |
| Twins-B [11] | 76 | 340 | 45.2 | 67.6 | 49.3 | 41.5 | 64.5 | 44.8 | 48.0 | 69.5 | 52.7 | 43.0 | 66.8 | 46.6 |
| Swin-S [38] | 69 | 354 | 44.8 | 66.6 | 48.9 | 40.9 | 63.4 | 44.2 | 48.5 | 70.2 | 53.5 | 43.3 | 67.3 | 46.6 |
| CSWin-S | 54 | 342 | 47.9 | 70.1 | 52.6 | 43.2 | 67.1 | 46.2 | 50.0 | 71.3 | 54.7 | 44.5 | 68.4 | 47.7 |
| X101-64 [63] | 101 | 493 | 42.8 | 63.8 | 47.3 | 38.4 | 60.6 | 41.3 | 44.4 | 64.9 | 48.8 | 39.7 | 61.9 | 42.6 |
| PVT-L[58] | 81 | 364 | 42.9 | 65.0 | 46.6 | 39.5 | 61.9 | 42.5 | 44.5 | 66.0 | 48.3 | 40.7 | 63.4 | 43.7 |
| ViL-B [69] | 76 | 365 | 45.1 | —- | —- | 41.0 | —- | —- | 45.7 | 67.2 | 49.9 | 41.3 | 64.4 | 44.5 |
| TwinsP-L [11] | 81 | 364 | 45.4 | —- | —- | 41.5 | —- | —- | —- | —- | —- | —- | —- | —- |
| Twins-L [11] | 111 | 474 | 45.9 | —- | —- | 41.6 | —- | —- | —- | —- | —- | —- | —- | —- |
| Swin-B [38] | 107 | 496 | 46.9 | —- | —- | 42.3 | —- | —- | 48.5 | 69.8 | 53.2 | 43.4 | 66.8 | 46.9 |
| CSWin-B | 97 | 526 | 48.7 | 70.4 | 53.9 | 43.9 | 67.8 | 47.3 | 50.8 | 72.1 | 55.8 | 44.9 | 69.1 | 48.3 |

### 4.1 ImageNet-1K Classification

For fair comparison, we follow the training strategy in DeiT [53] as other baseline Transformer architectures [60, 38]. Specifically, all our models are trained for 300 epochs with the input size of 224×224224\times 224. We use the AdamW optimizer with weight decay of 0.05 for CSWin-T/S and 0.1 for CSWin-B. The default batch size and initial learning rate are set to 1024 and 0.001, and the cosine learning rate scheduler with 20 epochs linear warm-up is used. We apply increasing stochastic depth [29] augmentation for CSWin-T, CSWin-S, and CSWin-B with the maximum rate as 0.1, 0.3, 0.5 respectively. When reporting the results of 384×384384\times 384 input, we fine-tune the models for 30 epochs with the weight decay of 1​e​-​81e\text{-}8, learning rate of 1​e​-​51e\text{-}5, batch size of 512512.

In Table 11, we compare our CSWin Transformer with state-of-the-art CNN and Transformer architectures. With the limitation of pages, we only compare with a few classical methods here and make a comprehensive comparison in the supplemental materials.

It shows that our CSWin Transformers outperform previous state-of-the-art vision Transformers by large margins. For example, CSWin-T achieves 82.7% Top-1 accuracy with only 4.3G FLOPs, surpassing CvT-13, Swin-T and DeiT-S by 1.1%, 1.4% and 2.9% respectively. And for the small and base model setting, our CSWin-S and CSWin-B also achieve the best performance. When finetuned on the 384×384384\times 384 input, a similar trend is observed, which well demonstrates the powerful learning capacity of our CSWin Transformers.

Compared with state-of-the-art CNNs, we find our CSWin Transformer is the only Transformer based architecture that achieves comparable or even better results than EfficientNet [51] under the small and base settings, while using less computation complexity . It is also worth noting that neural architecture search is used in EfficientNet but not in our CSWin Transformer design.

We further pre-train CSWin Transformer on ImageNet-21K dataset, which contains 14.2M images and 21K classes. Models are trained for 90 epochs with the input size of 224×224224\times 224. We use the AdamW optimizer with weight decay of 0.1 for CSWin-B and 0.2 for CSWin-L, and the default batch size and initial learning rate are set to 2048 and 0.001. When fine-tuning on ImageNet-1K, we train the models for 30 epochs with the weight decay of 1​e​-​81e\text{-}8, learning rate of 1​e​-​51e\text{-}5, batch size of 512512. The increasing stochastic depth [29] augmentation for both CSWin-B and CSWin-L is set to 0.1.

Table.3 reports the results of pre-training on ImageNet-21K. Compared to the results of CSWin-B pre-trained on ImageNet-1K, the large-scale data of ImageNet-21K brings a 1.6%∼\thicksim1.7% gain. CSWin-B and CSWin-L achieve 87.0% and 87.5% top-1 accuracy, surpassing previous methods.

| Backbone | #Params | FLOPs | Cascade Mask R-CNN 3x +MS |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| (M) | (G) | A​PbAP^{b} | A​P50bAP^{b}_{50} | A​P75bAP^{b}_{75} | A​PmAP^{m} | A​P50mAP^{m}_{50} | A​P75mAP^{m}_{75} |  |
| Res50 [22] | 82 | 739 | 46.3 | 64.3 | 50.5 | 40.1 | 61.7 | 43.4 |
| Swin-T [38] | 86 | 745 | 50.5 | 69.3 | 54.9 | 43.7 | 66.6 | 47.1 |
| CSWin-T | 80 | 757 | 52.5 | 71.5 | 57.1 | 45.3 | 68.8 | 48.9 |
| X101-32 [63] | 101 | 819 | 48.1 | 66.5 | 52.4 | 41.6 | 63.9 | 45.2 |
| Swin-S [38] | 107 | 838 | 51.8 | 70.4 | 56.3 | 44.7 | 67.9 | 48.5 |
| CSWin-S | 92 | 820 | 53.7 | 72.2 | 58.4 | 46.4 | 69.6 | 50.6 |
| X101-64 [63] | 140 | 972 | 48.3 | 66.4 | 52.3 | 41.7 | 64.0 | 45.1 |
| Swin-B [38] | 145 | 982 | 51.9 | 70.9 | 56.5 | 45.0 | 68.4 | 48.7 |
| CSWin-B | 135 | 1004 | 53.9 | 72.6 | 58.5 | 46.4 | 70.0 | 50.4 |

### 4.2 COCO Object Detection

Next, we evaluate CSWin Transformer on the COCO objection detection task with the Mask R-CNN [21] and Cascade Mask R-CNN [2] framework respectively. Specifically, we pretrain the backbones on the ImageNet-1K dataset and follow the finetuning strategy used in Swin Transformer [38] on the COCO training set.

We compare CSWin Transformer with various backbones: previous CNN backbones ResNet [22], ResNeXt(X) [62], and Transformer backbones PVT [58], Twins [11], and Swin [38]. Table 4 reports the results of the Mask R-CNN framework with “1×1\times” (12 training epoch) and “3×+MS3\times+\text{MS}” (36 training epoch with multi-scale training) schedule. It shows that our CSWin Transformer variants clearly outperforms all the CNN and Transformer counterparts. In details, our CSWin-T outperforms Swin-T by +4.5 box AP, +3.1 mask AP with the 1×1\times schedule and +3.0 box AP, +2.0 mask AP with the 3×3\times schedule respectively. We also achieve similar performance gain on small and base configuration.

Table 5 reports the results with the Cascade Mask R-CNN framework. Though Cascade Mask R-CNN is overall stronger than Mask R-CNN, we observe CSWin Transformers still surpass the counterparts by promising margins under different model configurations.

| Backbone | Semantic FPN 80k | Upernet 160k |  |  |  |  |
|---|---|---|---|---|---|---|
| #Param. | FLOPs | mIoU | #Param. | FLOPs | SS/MS mIoU |  |
| Res50 [22] | 28.5 | 183 | 36.7 | —- | —- | —-/—- |
| PVT-S [58] | 28.2 | 161 | 39.8 | —- | —- | —-/—- |
| TwinsP-S [11] | 28.4 | 162 | 44.3 | 54.6 | 919 | 46.2/47.5 |
| Twins-S [11] | 28.3 | 144 | 43.2 | 54.4 | 901 | 46.2/47.1 |
| Swin-T [38] | 31.9 | 182 | 41.5 | 59.9 | 945 | 44.5/45.8 |
| CSWin-T | 26.1 | 202 | 48.2 | 59.9 | 959 | 49.3/50.7 |
| Res101 [22] | 47.5 | 260 | 38.8 | 86.0 | 1029 | —-/44.9 |
| PVT-M [58] | 48.0 | 219 | 41.6 | —- | —- | —-/—- |
| TwinsP-B [11] | 48.1 | 220 | 44.9 | 74.3 | 977 | 47.1/48.4 |
| Twins-B [11] | 60.4 | 261 | 45.3 | 88.5 | 1020 | 47.7/48.9 |
| Swin-S [38] | 53.2 | 274 | 45.2 | 81.3 | 1038 | 47.6/49.5 |
| CSWin-S | 38.5 | 271 | 49.2 | 64.6 | 1027 | 50.4/51.5 |
| X101-64 [63] | 86.4 | — | 40.2 | —- | —- | —-/—- |
| PVT-L [58] | 65.1 | 283 | 42.1 | —- | —- | —-/—- |
| TwinsP-L [11] | 65.3 | 283 | 46.4 | 91.5 | 1041 | 48.6/49.8 |
| Twins-L [11] | 103.7 | 404 | 46.7 | 133.0 | 1164 | 48.8/50.2 |
| Swin-B [38] | 91.2 | 422 | 46.0 | 121.0 | 1188 | 48.1/49.7 |
| CSWin-B | 81.2 | 464 | 49.9 | 109.2 | 1222 | 51.1/52.2 |
| Swin-B†\dagger [38] | —- | —- | —- | 121.0 | 1841 | 50.0/51.7 |
| Swin-L†\dagger [38] | —- | —- | —- | 234.0 | 3230 | 52.1/53.5 |
| CSWin-B†\dagger | —- | —- | —- | 109.2 | 1941 | 51.8/52.6 |
| CSWin-L†\dagger | —- | —- | —- | 207.7 | 2745 | 54.0/55.7 |

### 4.3 ADE20K Semantic Segmentation

We further investigate the capability of CSWin Transformer for Semantic Segmentation on the ADE20K [72] dataset. Here we employ the semantic FPN [33] and Upernet [61] as the basic framework. For fair comparison, we follow previous works [58, 38] and train Semantic FPN 80k iterations with batch size as 16, and Upernet 160k iterations with batch size as 16, more details are provided in the supplementary material. In Table 6, we report the results of different methods in terms of mIoU and Multi-scale tested mIoU (MS mIoU). It can be seen that, our CSWin Transformers significantly outperform previous state-of-the-arts under different configurations. In details, CSWin-T, CSWin-S, CSWin-B achieve +6.7, +4.0, +3.9 higher mIOU than the Swin counterparts with the Semantic FPN framework, and +4.8, +2.8, +3.0 higher mIOU with the Upernet framework. Compared to the CNN counterparts, the performance gain is very promising and demonstrates the potential of vision Transformers again. When using the ImageNet-21K pre-trained model, our CSWin-L further achieves 55.7 mIoU and surpasses the previous best model by +2.2 mIoU, while using less computation complexity.

| Model | Cascade Mask R-CNN on COCO | UperNet on ADE20K |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|
| #Param. | FLOPs | FPS | APb/m | #Param. | FLOPs | FPS | mIoU |  |
| Swin-T | 86M | 745G | 15.3 | 50.5/43.7 | 60M | 945G | 18.5 | 44.5 |
| CSWin-T | 80M | 757G | 14.2 | 52.5/45.3 | 60M | 959G | 17.3 | 49.3 |
| Swin-S | 107M | 838G | 12.0 | 51.8/44.7 | 81M | 1038G | 15.2 | 47.6 |
| CSWin-S | 92M | 820G | 11.7 | 53.7/46.4 | 65M | 1027G | 15.6 | 50.4 |
| Swin-B | 145M | 982G | 11.2 | 51.9/45.0 | 121M | 1188G | 9.92 | 48.1 |
| CSWin-B | 135M | 1004G | 9.6 | 53.9/46.4 | 109M | 1222G | 9.08 | 51.1 |

| Model | Attention | ImageNet | COCO | ADE20K |  |  |  |  |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reigon | #Param. | FLOPs | FPS | Top1(%) | #Param. | FLOPs | FPS | APb | APm | #Param. | FLOPs | FPS | mIoU(%) |  |
| Axial | H | 23M | 4.2G | 735 | 81.8 | 42M | 258G | 27.9 | 43.4 | 39.4 | 26M | 186G | 50.3 | 42.6 |
| CSWin (fix sw=1) | H | 23M | 4.1G | 721 | 81.9 | 42M | 258G | 26.8 | 45.2 | 40.8 | 26M | 179G | 49.1 | 47.5 |
| Criss-Cross | H*2-1 | 23M | 4.2G | 187 | 82.2 | 42M | 263G | 5.5 | 45.2 | 40.9 | 26M | 186G | 17.6 | 47.4 |
| CSWin (fix sw=2) | H*2 | 23M | 4.2G | 718 | 82.2 | 42M | 263G | 25.1 | 45.6 | 41.4 | 26M | 186G | 47.2 | 47.6 |
| CSWin (sw=1,2,7,7; Seq) | sw×\timesH | 23M | 4.3G | 711 | 82.4 | 42M | 279G | 22.3 | 45.1 | 41.1 | 26M | 202G | 45.2 | 46.2 |
| CSWin (sw=1,2,7,7) | sw×\timesH | 23M | 4.3G | 701 | 82.7 | 42M | 279G | 21.1 | 46.7 | 42.2 | 26M | 202G | 44.8 | 48.2 |

### 4.4 Inference Speed.

Here we report the inference speed of our CSWin and Swin works. For downstream tasks, we report the FPS of Cascade Mask R-CNN for object detection on COCO and UperNet for semantic segmentation on ADE20K. In most cases, the speed of our model is only slightly slower than Swin (less than 10%), but our model outperforms Swin by large margins. For example, on COCO, CSWin-S are +1.9% box AP and +1.7% mask AP higher than Swin-S with similar inference speed(11.7 FPS vs. 12 FPS). Note that our CSWin-T performs better than Swin-B on box AP(+0.6%), mask AP(+0.3%) with much faster inference speed(14.2 FPS vs. 11.2 FPS), indicating our CSWin achieves better accuracy/FPS trade-offs.

### 4.5 Ablation Study

To better understand CSWin Transformers, we compare each key component with the previous works under a completely fair setting that we use the same architecture and hyper-parameter for the following experiments, and only vary one component for each ablation. For time consideration, we use Mask R-CNN with 1x schedule as the default setting for detection and instance segmentation evaluation, and Semantic FPN with 80k iterations and single-scale test for segmentation evaluation.

Parallel Multi-Head Grouping. We first study the effectiveness of our novel “Parallel Multi-Head Grouping” strategy. Here we compare Axial-Attention [25] and Criss-Cross-Attention [30] under the CSWin-T backbone. “Attention region” is used as the computation cost metric for detailed comparison. To simplify, we assume the attention is calculated on a square input that H=WH=W.

In Table.8, we find that the “parallel multi-head grouping” is efficient and effective, especially for downstream tasks. When we replace the Parallel manner with Sequential, the performance of CSWin degrades on all tasks. When comparing with previous methods under the similar attention region constrain, our s​w=1sw=1 CSWin performs slightly better than Axial on ImageNet, while outperforming it by a large margin on downstream tasks. Our s​w=2sw=2 CSWin performs slightly better than Criss-Cross Attention, while the speed of CSWin is 2×∼5×2\times\sim 5\times faster than it on different tasks, this further proves that our “parallel” design is much more efficient.

Dynamic Stripe Width . In Fig.4 we study the trade off between stripe width and accuracy. We find that with the increase of stripe width, the compution cost(FLOPS) increase, and the Top-1 classification accuracy improves greatly at the beginning and slows down when the width is large enough. Our default setting [1,2,7,7] achieves a good trade-off between accuracy and FLOPs.

Attention Mechanism Comparison. Following the above analysis on each component of CSWin self-attention, we further compare with existing self-attention mechanisms. As some of the methods need even layers in each stage, for a fair comparison, we use the Swin-T [38] as backbone and only change the self-attention mechanism. In detail, we use 2,2,6,22,2,6,2 blocks for the four stages with the 96 base channel, non-overlapped token embedding [17], and RPE [38]. The results are reported in Table 9. Obviously, our CSWin self-attention mechanism performs better than existing self-attention mechanisms across all the tasks.

Positional Encoding Comparison. The proposed LePE is specially designed to enhance the local positional information on downstream tasks for various input resolutions. Here we use CSWin-T as the backbone and only very the position encoding. In Table 10, we compare our LePE with other recent positional encoding mechanisms(APE [17], CPE [12], and RPE [46]) for image classification, object detection and image segmentation. Besides, we also test the variants without positional encoding (No PE) and CPE*, which is obtained by applying CPE before every Transformer block. According to the comparison results, we see that: 1) Positional encoding can bring performance gain by introducing the local inductive bias; 2) Though RPE achieves similar performance on the classification task with fixed input resolution, our LePE performs better (+1.2 box AP and +0.9 mask AP on COCO, +0.9 mIoU on ADE20K) on downstream tasks where the input resolution varies; 3) Compared to APE and CPE, our LePE also achieves better performance.

|  | ImageNet | COCO | ADE20K |  |
|---|---|---|---|---|
|  | Top1(%) | APb | APm | mIoU(%) |
| Sliding window [43] | 81.4 | — | — | —- |
| Shifted window [38] | 81.3 | 42.2 | 39.1 | 41.5 |
| Spatially Sep [11] | 81.5 | 42.7 | 39.5 | 42.9 |
| Sequential Axial [25] | 81.5 | 40.4 | 37.6 | 39.8 |
| Criss-Cross [30] | 81.7 | 42.9 | 39.7 | 43.0 |
| Cross-shaped window | 82.2 | 43.4 | 40.2 | 43.4 |

|  | ImageNet | COCO | ADE20K |  |
|---|---|---|---|---|
|  | Top1(%) | APb | APm | mIoU(%) |
| No PE | 82.5 | 44.8 | 41.1 | 47.0 |
| APE [17] | 82.6 | 45.1 | 41.1 | 45.7 |
| CPE [12] | 82.2 | 45.8 | 41.6 | 46.1 |
| CPE* [12] | 82.4 | 45.4 | 41.3 | 46.6 |
| RPE [46] | 82.7 | 45.5 | 41.3 | 46.6 |
| LePE | 82.7 | 46.7 | 42.2 | 48.2 |

## 5 Conclusion

In this paper, we have presented a new Vision Transformer architecture named CSWin Transformer. The core design of CSWin Transformer is the CSWin Self-Attention, which performs self-attention in the horizontal and vertical stripes by splitting the multi-heads into parallel groups. This multi-head grouping design can enlarge the attention area of each token within one Transformer block efficiently. On the other hand, the mathematical analysis also allows us to increase the stripe width along the network depth to further enlarge the attention area with subtle extra computation cost. We further introduce locally-enhanced positional encoding into CSWin Transformer for downstream tasks. We achieved the state-of-the-art performance on various vision tasks under constrained computation complexity. We are looking forward to applying it for more vision tasks.

## Experiment Details

In this section, we provide more detailed experimental settings about ImageNet and downstream tasks.

ImageNet-1K Classification. For a fair comparison, we follow the training strategy in DeiT [53]. Specifically, all our models are trained for 300 epochs with the input size of 224×224224\times 224. We use the AdamW optimizer with weight decay of 0.05 for CSWin-T/S and 0.1 for CSWin-B. The default batch size and initial learning rate are set to 2048 and 2​e−32e-3 respectively, and the cosine learning rate scheduler with 20 epochs linear warm-up is used. We adopt most of the augmentation in [53], including RandAugment [15] (rand-m9-mstd0.5-inc1) , Mixup [68] (p​r​o​b=0.8)(prob=0.8), CutMix [67] (p​r​o​b=1.0)(prob=1.0), Random Erasing [71] (p​r​o​b=0.25)(prob=0.25) and Exponential Moving Average [40] (e​m​aCLOSE(ema-OPENd​e​c​a​y=0.99984)decay=0.99984), increasing stochastic depth [29] (p​r​o​b=0.2,0.4,0.5prob=0.2,0.4,0.5 for CSWin-T, CSWin-S, and CSWin-B respectively).

When fine-tuning with 384×384384\times 384 input, we follow the setting in [38] that fine-tune the models for 30 epochs with the weight decay of 1​e​-​81e\text{-}8, learning rate of 5​e​-​65e\text{-}6, batch size of 256256. We notice that a large ratio of stochastic depth is beneficial for fine-tuning and keeping it the same as the training stage.

COCO Object Detection and Instance Segmentation. We use two classical object detection frameworks: Mask R-CNN [21] and Cascade Mask R-CNN [2] based on the implementation from mmdetection [6]. For Mask R-CNN, we train it with ImageNet-1K pretrained model with two settings: 1×1\times schedule and 3×3\times+MS schedule. For 1×1\times schedule, we train the model with single-scale input (image is resized to the shorter side of 800 pixels, while the longer side does not exceed 1333 pixels) for 12 epochs. We use AdamW [39] optimizer with a learning rate of 0.0001, weight decay of 0.05 and batch size of 16. The learning rate declines at the 8 and 11 epoch with decay rate 0.1. The stochastic depth is also same as the ImageNet-1K setting that 0.1, 0.3, 0.5 for CSWin-T, CSWin-S, and CSWin-B respectively. For 3×3\times+MS schedule, we train the model with multi-scale input (image is resized to the shorter side between 480 and 800 while the longer side is no longer than 1333) for 36 epochs. The other settings are same as the 1×1\times except we decay the learning rate at epoch 27 and 33. When it comes to Cascade Mask R-CNN, we use the same 3×3\times+MS schedule as Mask R-CNN.

ADE20K Semantic segmentation. Here we consider two semantic segmentation frameworks: UperNet [61] and Semantic FPN [33] based on the implementation from mmsegmentaion [14]. For UperNet, we follow the setting in [38] and use AdamW [39] optimizer with initial learning rate 6​e−56e^{-5}, weight decay of 0.01 and batch size of 16 (8 GPUs with 2 images per GPU) for 160K iterations. The learning rate warmups with 1500 iterations at the beginning and decays with a linear decay strategy. We use the default augmentation setting in mmsegmentation including random horizontal flipping, random re-scaling (ratio range [0.5, 2.0]) and random photo-metric distortion. All the models are trained with input size 512×512512\times 512. The stochastic depth is set to 0.2, 0.4, 0.6 for CSWin-T, CSWin-S, and CSWin-B respectively. When it comes to testing, we report both single-scale test result and multi-scale test ([0.5, 0.75, 1.0, 1.25, 1.5, 1.75]×\times of that in training).

For Semantic FPN, we follow the setting in [58]. We use AdamW [39] optimizer with initial learning rate 1​e−41e^{-4}, weight decay of 1​e−41e^{-4} and batch size of 16 (4 GPUs with 4 images per GPU) for 80K iterations.

## More Experimetns

With the limitation of pages, we only compare with a few classical methods in our paper, here we make a comprehensive comparison with more current methods on ImageNet-1K. We find that our CSWin performs best in concurrent works.

| ImageNet-1K 2242 trained models |  |  |  |
|---|---|---|---|
| Method | #Param. | FLOPs | Top-1 |
| Reg-4G [41] | 21M | 4.0G | 80.0 |
| Eff-B4* [51] | 19M | 4.2G | 82.9 |
| DeiT-S [53] | 22M | 4.6G | 79.8 |
| PVT-S [58] | 25M | 3.8G | 79.8 |
| T2T-14 [66] | 22M | 5.2G | 81.5 |
| ViL-S [69] | 25M | 4.9G | 82.0 |
| TNT-S [20] | 24M | 5.2G | 81.3 |
| CViT-15 [4] | 27M | 5.6G | 81.0 |
| Visf-S [8] | 40M | 4.9G | 82.3 |
| LViT-S [36] | 22M | 4.6G | 80.8 |
| CoaTL-S [64] | 20M | 4.0G | 81.9 |
| CPVT-S [12] | 23M | 4.6G | 81.5 |
| Swin-T [38] | 29M | 4.5G | 81.3 |
| CvT-13 [60] | 20M | 4.5G | 81.6 |
| CSWin-T | 23M | 4.3G | 82.7 |
| ImageNet-1K 3842 finetuned models |  |  |  |
| CvT-13 [60] | 20M | 16.3G | 83.0 |
| T2T-14 [66] | 22M | 17.1G | 83.3 |
| CViTc-15 [4] | 28M | 21.4G | 83.5 |
| CSWin-T | 23M | 14.0G | 84.3 |

| ImageNet-1K 2242 trained models |  |  |  |
|---|---|---|---|
| Method | #Param. | FLOPs | Top-1 |
| Reg-8G [41] | 39M | 8.0G | 81.7 |
| Eff-B5* [51] | 30M | 9.9G | 83.6 |
| PVT-M [58] | 44M | 6.7G | 81.2 |
| PVT-L [58] | 61M | 9.8G | 81.7 |
| T2T-19 [66] | 39M | 8.9G | 81.9 |
| T2Tt-19 [66] | 39M | 9.8G | 82.2 |
| ViL-M [69] | 40M | 8.7G | 83.3 |
| MViT-B [19] | 37M | 7.8G | 83.0 |
| CViT-18 [4] | 43M | 9.0G | 82.5 |
| CViTc-18 [4] | 44M | 9.5G | 82.8 |
| Twins-B [11] | 56M | 8.3G | 83.2 |
| Swin-S [38] | 50M | 8.7G | 83.0 |
| CvT-21 [60] | 32M | 7.1G | 82.5 |
| CSWin-S | 35M | 6.9G | 83.6 |
| ImageNet-1K 3842 finetuned models |  |  |  |
| CvT-21 [60] | 32M | 24.9G | 83.3 |
| CViTc-18 [4] | 45M | 32.4G | 83.9 |
| CSWin-S | 35M | 22.0G | 85.0 |

| ImageNet-1K 2242 trained models |  |  |  |
|---|---|---|---|
| Method | #Param. | FLOPs | Top-1 |
| Reg-16G [41] | 84M | 16.0G | 82.9 |
| Eff-B6* [51] | 43M | 19.0G | 84.0 |
| DeiT-B [53] | 87M | 17.5G | 81.8 |
| PiT-B [24] | 74M | 12.5G | 82.0 |
| T2T-24 [66] | 64M | 14.1G | 82.3 |
| T2Tt-24 [66] | 64M | 15.0G | 82.6 |
| CPVT-B [12] | 88M | 17.6G | 82.3 |
| TNT-B [20] | 66M | 14.1G | 82.8 |
| ViL-B [69] | 56M | 13.4G | 83.2 |
| Twins-L [11] | 99M | 14.8G | 83.7 |
| Swin-B [38] | 88M | 15.4G | 83.3 |
| CSWin-B | 78M | 15.0G | 84.2 |
| ImageNet-1K 3842 finetuned models |  |  |  |
| ViT-B/16 [17] | 86M | 49.3G | 77.9 |
| DeiT-B [53] | 86M | 55.4G | 83.1 |
| Swin-B [38] | 88M | 47.0G | 84.2 |
| CSWin-B | 78M | 47.0G | 85.4 |
