# Revisiting Feature Prediction for Learning Visual Representations from Video

This paper explores feature prediction as a stand-alone objective for unsupervised learning from video and introduces V-JEPA, a collection of vision models trained solely using a feature prediction objective, without the use of pretrained image encoders, text, negative examples, reconstruction, or other sources of supervision. The models are trained on 2 million videos collected from public datasets and are evaluated on downstream image and video tasks. Our results show that learning by predicting video features leads to versatile visual representations that perform well on both motion and appearance-based tasks, without adaption of the model’s parameters; e.g., using a frozen backbone. Our largest model, a ViT-H/16 trained only on videos, obtains 81.9%81.9\% on Kinetics-400, 72.2%72.2\% on Something-Something-v2, and 77.9%77.9\% on ImageNet1K.

## 1 Introduction

Humans possess the remarkable ability to map low-level signals originating from the retina into a semantic spatio-temporal understanding of the world; synthesizing notions such as objects and global motion (Spelke et al., 1995). A long-standing goal of the machine learning community is to identify the principles or objectives that may guide such unsupervised learning in humans (Field, 1994; Berkes and Wiskott, 2005; Hinton, 1989). One related hypothesis is based on the predictive feature principle (Rao and Ballard, 1999), which posits that representations of temporally adjacent sensory stimuli should be predictive of each other.

In this work, we revisit feature prediction as a stand-alone objective for unsupervised learning of visual representations from video. Numerous advances in the field — such as the standard use of transformer architectures in vision (Dosovitskiy et al., 2020), the maturing of masked autoencoding frameworks (Xie et al., 2021; Bao et al., 2021; He et al., 2021), query-based feature pooling (Chen et al., 2022), joint-embedding predictive architectures (JEPA) (LeCun, 2022; Assran et al., 2023; Baevski et al., 2022b), and larger datasets — form a unique arsenal of tools, which we integrate in a modern and conceptually simple method, the video joint-embedding predictive architecture or V-JEPA, which is based solely on feature prediction, without using pretrained image encoders, text, negative examples, human annotations, or pixel-level reconstruction.

We seek to answer the simple question:

How effective is feature prediction as a stand-alone objective for unsupervised learning from video with modern tools?

To that end, we pretrain a family of V-JEPA models on a dataset of 2 million videos collected from publicly available datasets by combining a masked modeling prediction task with a joint-embedding predictive architecture (see Figure 2). We measure performance on several downstream image and video tasks, using both frozen evaluation and end-to-end fine-tuning. Our findings suggest that feature prediction can indeed serve as an effective stand-alone objective for unsupervised learning from video, while using significantly shorter training schedules than pixel prediction methods. Specifically:

- • Feature prediction leads to versatile visual representations that perform well across downstream image and video tasks without adaption of the model’s weights; i.e., using a frozen backbone. V-JEPA achieves the best performance among methods we consider (+6% accuracy) on the SomethingSomething-v2 task, which requires fine-grained temporal understanding. V-JEPA is also competitive on tasks like Kinetics400, where appearance-based features are sufficient and hence state-of-the-art image models such as DINOv2 excel (Figure 1 and Table 6).
Feature prediction leads to versatile visual representations that perform well across downstream image and video tasks without adaption of the model’s weights; i.e., using a frozen backbone. V-JEPA achieves the best performance among methods we consider (+6% accuracy) on the SomethingSomething-v2 task, which requires fine-grained temporal understanding. V-JEPA is also competitive on tasks like Kinetics400, where appearance-based features are sufficient and hence state-of-the-art image models such as DINOv2 excel (Figure 1 and Table 6).

- • Models trained with feature prediction are superior to pixel prediction approaches under a frozen evaluation protocol (attentive probing) and are competitive with pixel prediction under full fine-tuning, while using significantly shorter training schedules (Tables 6 and 6).
Models trained with feature prediction are superior to pixel prediction approaches under a frozen evaluation protocol (attentive probing) and are competitive with pixel prediction under full fine-tuning, while using significantly shorter training schedules (Tables 6 and 6).

- • Models trained with feature prediction are more label-efficient than pixel prediction approaches. Decreasing the available number of labeled examples results in an increase in the performance gap between V-JEPA and pixel-reconstruction models (Table 7).
Models trained with feature prediction are more label-efficient than pixel prediction approaches. Decreasing the available number of labeled examples results in an increase in the performance gap between V-JEPA and pixel-reconstruction models (Table 7).

## 2 Related Works

One way to encourage temporally adjacent representations to be predictive of each other is to ensure that they vary slowly over time. Early works targeting predictive features encouraged representations of individual video frames to be locally temporally invariant, while preventing representation collapse by using spectral methods, as in SFA (Wiskott and Sejnowski, 2002), SSA (Kayser et al., 2001), and Simulated Fixations (Zou et al., 2012). More recently, Goroshin et al. (2015); Wang et al. (2010) train a siamese convolutional network to map the representations of two subsequent frames to the same point, while encouraging distant frames to have diverse representations via a pair-wise margin loss and a triplet loss, respectively. Other works (Oord et al., 2018; Surís et al., 2021; Feichtenhofer et al., 2021) implement temporal invariance using noise-contrastive estimation (Gutmann and Hyvärinen, 2012). Our exploration in this paper goes beyond temporal invariance and explores feature prediction using masked modeling.

Going beyond local invariance, a family of works trains a predictor network to map the representation of a frame or clip at one time-step to a distinct representation at another time-step. Srivastava et al. (2015); Vondrick et al. (2016); Wang et al. (2023b) train such a video feature predictor network on top of a frozen pretrained image or video encoder. Unfreezing the target feature extractor, several methods train the video encoder and the predictor network simultaneously, while preventing collapse by using a supervised action forecasting loss (Girdhar and Grauman, 2021), or by using the representations of distant clips as negative samples in a contrastive loss (Han et al., 2019; Han et al., 2020; Tan et al., 2023), often focusing on small convolutional encoders (Han et al., 2019; Han et al., 2020). The idea of learning a representation by predicting missing information in feature space is also core to the joint-embedding predictive architecture (JEPA) (LeCun, 2022), which combines a siamese encoder with a predictor network. JEPAs have been successfully instantiated in several modalities, such as with audio data (Baevski et al., 2022b) and image data (Zhou et al., 2021; Oquab et al., 2023; Assran et al., 2023). In this work, we extend this paradigm to video data by leveraging recent advances in self-supervised learning.

The use of vision transformers (Dosovitskiy et al., 2020; Li et al., 2022) has become standard practice in self-supervised learning with joint-embedding architectures (Chen et al., 2021; Caron et al., 2021; Oquab et al., 2023; Zhou et al., 2021; Assran et al., 2022), and unlocked masked image modeling in pixel space by parameterizing the pixel decoder as a transformer with learnable mask tokens (Dosovitskiy et al., 2020; Xie et al., 2021; He et al., 2021; Bao et al., 2021), demonstrating a step-change in the representation quality of autoencoding methods (Vincent et al., 2010). This line of generative methods was subsequently extended to video data using spatio-temporal masking (Tong et al., 2022; Feichtenhofer et al., 2022; Wang et al., 2023a; Kalluri et al., 2023; Gupta et al., 2023). It was also recently shown that the representations of masked image autoencoders could be significantly improved by using learnable pooling mechanisms based on cross-attention (Chen et al., 2022). Finally, through careful selection of design choices, the non-contrastive collapse prevention strategy in BYOL (Grill et al., 2020) was recently made to work with image feature prediction methods (Baevski et al., 2022b; Assran et al., 2023), which demonstrated the ability to learn representations that can be leveraged for various downstream tasks without relying on invariance to hand-crafted image transformations.

Approaches that predict in pixel space must dedicate significant model capacity and compute to capture all the low-level detail in the visual input. By contrast, approaches that predict in latent space have the flexibility to eliminate irrelevant or unpredictable pixel-level details from the target representation (Vondrick et al., 2016). Predicting in representation space has been shown to lead to versatile representations that perform well across many downstream tasks through linear probing or low-shot adaptation (Assran et al., 2023; Oquab et al., 2023; Assran et al., 2022), while demonstrating an efficiency gain during pretraining compared to pixel level reconstruction (Assran et al., 2023; Baevski et al., 2022b; Baevski et al., 2022a). The works of Baevski et al. (2022a); Baevski et al. (2022b) additionally show that predicting in representation space results in competitive end-to-end fine-tuning performance in the image, audio and text domains. In this work, we extend these findings to the video modality.

## 3 Methodology: Video-JEPA

Our goal is to explore the effectiveness of feature prediction as a stand-alone objective for learning visual representations from video. To that end, we use a joint-embedding predictive architecture (JEPA) (LeCun, 2022); see Figure 2. The main idea behind a JEPA is to learn by predicting the representation of an input yy from the representation of another input xx. The basic architecture is made up of an encoder, Eθ​(⋅)E_{\theta}(\cdot), which computes the representation of the inputs, and a predictor, Pϕ​(⋅)P_{\phi}(\cdot), which predicts the representation of yy from the representation of xx, conditioned on a variable zz indicating the transformation (or corruption) between xx and yy. Conditioning on zz enables the generation of distinct predictions for various transformations of xx.

### 3.1 Training Objective

We train our visual encoder Eθ​(⋅)E_{\theta}(\cdot) to satisfy the constraint that representations computed from one part of the video, yy, should be predictable from representations computed from another part of the video, xx. The predictor network Pϕ​(⋅)P_{\phi}(\cdot), which maps the representation of xx to the representation of yy, is trained simultaneously with the encoder, and is provided specification of the spatio-temporal positions of yy through the conditioning variable z←Δyz\leftarrow\Delta_{y}.

Naively implementing the objective using the regression

|  | minimizeθ,ϕ∥Pϕ​(Eθ​(x),Δy)−Eθ​(y)∥1,\text{minimize}_{\theta,\phi}\quad\lVert P_{\phi}(E_{\theta}(x),\Delta_{y})-E_{\theta}(y)\rVert_{1}, |  |
|---|---|---|

would admit a trivial solution, where the encoder outputs a constant representation, regardless of its input. In practice, we use the following modified objective to prevent representation collapse,

|  | minimizeθ,ϕ∥Pϕ(Eθ(x),Δy)−sg(E¯θ(y))∥1,\text{minimize}_{\theta,\phi}\quad\rVert P_{\phi}(E_{\theta}(x),\Delta_{y})-\text{sg}(\overline{E}_{\theta}(y))\lVert_{1}, |  | (1) |
|---|---|---|---|

where sg​(⋅)\text{sg}(\cdot) denotes a stop-gradient operation, which does not backpropagate through its argument, and E¯θ​(⋅)\overline{E}_{\theta}(\cdot) is an exponential moving average of the network Eθ​(⋅)E_{\theta}(\cdot). The use of an exponential-moving average feature extractor along with a stop-gradient and a predictor has been used as a collapse prevention strategy for image pretraining (Grill et al., 2020), and studied empirically (Xie et al., 2021) and theoretically (Tian et al., 2021). In fact, the objective in equation (1) is similar to the loss of Assran et al. (2023) used for image pretraining, but we modify it to use an ℓ1\ell_{1} regression, which we found to be more stable.

A theoretical motivation for the effectiveness of this collapse prevention strategy was proposed in Grill et al. (2020) for the BYOL method. We provide a simple adaptation of their analysis for our ℓ1\ell_{1} loss. For ease of exposition, we will disregard the effect of the conditioning variable zz and consider one dimensional representations. Denote the representation E¯θ​(y)\overline{E}_{\theta}(y) by a random variable YY. The optimal predictor under equation (1) is thus given by the following functional expression,

|  | P⋆​(Eθ​(x))\displaystyle P^{\star}(E_{\theta}(x)) | =argminP​∥P⁡(Eθ​(x))−Y∥1\displaystyle=\text{argmin}_{P}\lVert P(E_{\theta}(x))-Y\rVert_{1} |  |
|---|---|---|---|
|  |  | =median​(Y|Eθ​(x)).\displaystyle=\text{median}(Y|E_{\theta}(x)). |  |

Substituting this expression for the optimal predictor into the loss function and evaluating the expected gradient of the encoder gives

|  | ∇θ𝔼​∥P⋆​(Eθ​(x))−Y∥1=∇θMAD​(Y|Eθ​(x)),\nabla_{\theta}\mathbb{E}\lVert P^{\star}(E_{\theta}(x))-Y\rVert_{1}=\nabla_{\theta}\text{MAD}(Y|E_{\theta}(x)), |  |
|---|---|---|

where MAD(⋅|Eθ(x))\text{MAD}(\cdot\ |E_{\theta}(x)) is the median absolute deviation of a random variable conditioned on Eθ​(x)E_{\theta}(x). Thus, in the case where the predictor is optimal, the encoder must learn to capture as much information about the video as possible to minimize the deviation of the target. The hypothesis is that incorporating an exponential moving average to compute the representation of yy ensures that the predictor evolves faster than the encoder and remains close to optimal, thereby preventing collapse.

### 3.2 Prediction Task: Predicting yy from xx

The feature prediction task is based on a masked modeling formulation (He et al., 2021; Tong et al., 2022); i.e., regions xx and yy from the video are sampled using masking. To sample yy from a video, we sample several (possibly overlapping) spatially continuous blocks with various aspect ratios and repeat the spatial blocks across the entire temporal dimension of the video; xx is taken to be the complement. Masking a large continuous block that covers the full temporal dimension limits information leakage due to the spatial and temporal redundancy of videos, and results in a harder prediction task (Tong et al., 2022).

We leverage two types of masks: short-range masks, where we take the union of 88 randomly sampled target blocks covering 15% of each frame, and long-range masks, where we take the union of 22 randomly sampled target blocks covering 70% of each frame. In both cases, the aspect ratio for all sampled blocks is randomly chosen in the range (0.75,1.5)(0.75,1.5). Given that both short-range and long-range masks are produced by sampling many blocks and taking their union, the result is an average masking ratio of ∼90%\sim 90\%. We refer to our masking strategy as multi-block, and compare it to other possible masking strategies in Section 4.

### 3.3 Network Parameterization

We use a Vision Transformer (ViT) (Dosovitskiy et al., 2020; Arnab et al., 2021) as our video backbone. To process a video with a transformer network, we split the video clip into a 3D grid of LL spatio-temporal patches, where a patch consists of a 16×1616\times 16 pixel block spanning 22 consecutive frames; we refer to these spatio-temporal patches as tokens. This sequence of tokens is then directly processed by the stack of transformer blocks. Inputs xx and yy correspond to masked regions of a video, we apply the video masks by simply dropping a subset of the tokens. We apply masking at the input of the xx-encoder, and at the output of the yy-encoder to construct contextualized targets (Baevski et al., 2022b). The encoder is parameterized using standard ViT networks, while the predictor is a narrow transformer implemented using 1212 blocks with an embedding dimension of 384384. Taking inspiration from masked autoencoders (He et al., 2021), our predictor takes as input the sequence of embeddings produced by the xx-encoder as well as a sequence of learnable mask tokens with positional embeddings indicating the spatio-temporal positions of the yy tokens. The output of the predictor is an embedding vector for each mask token; see Figure 3 and refer to Appendix B for more details.

|  |  | Frozen Evaluation | Fine-Tuning |  |  |
|---|---|---|---|---|---|
|  |  | K400 | SSv2 | IN1K | K400-ft |
| Target | Arch. | (16×\times1×\times1) | (16×\times1×\times1) |  | (16×\times5×\times3) |
| Pixels | ViT-L/16 | 68.6 | 66.0 | 73.3 | 85.4 |
| Features | ViT-L/16 | 73.7 | 66.2 | 74.8 | 85.6 |

|  |  |  | Frozen Evaluation |  |  |  |
|---|---|---|---|---|---|---|
|  |  |  | K400 | SSv2 | IN1K | Avg. |
| Arch. | Data | #Samples | (16×\times1×\times1) | (16×\times1×\times1) |  |  |
| ViT-L/16 | K710 | 700K | 75.8 | 63.2 | 73.7 | 70.9 |
| K710+SSv2 | 900K | 72.9 | 67.4 | 72.8 | 71.0 |  |
| K710+HT | 1900K | 74.5 | 64.2 | 74.8 | 71.1 |  |
| VideoMix2M | 2000K | 73.7 | 66.2 | 74.8 | 71.5 |  |
| ViT-H/16 | K710+SSv2 | 900K | 75.7 | 66.8 | 73.7 | 72.0 |
| VideoMix2M | 2000K | 74.0 | 68.5 | 75.9 | 72.8 |  |

### 3.4 Pretraining Data and Evaluation Setup

We combine several public datasets to construct an unsupervised video pretraining dataset, which we refer to as VideoMix2M. Specifically, we combine the videos from HowTo100M (HT) (Miech et al., 2019), Kinetics-400/600/700 (K710) (Kay et al., 2017), and Something-Something-v2 (SSv2) (Goyal et al., 2017), and remove any overlap with the validation sets of Kinetics-400/600/700 and Something-Something-v2, resulting in approximately 2 million videos. We train a ViT-L/16, a ViT-H/16, and a ViT-H/16384 transformer model on VideoMix2M. We use a batch size of 3072 for the ViT-L/16 and ViT-H/16 models, and a batch size of 2400 for the ViT-H/16384 model. Each model takes as input a video clip of 16 frames sampled with a frame-skip of 4, corresponding to roughly 3 second clips on average. The ViT-L/16 and ViT-H/16 process the video at a spatial resolution of 224, while the ViT-H/16384 uses an input resolution of 384; cf. Appendix C.

Pretrained models are evaluated on downstream video and image tasks. On video tasks, we use a subset of the VideoGLUE benchmark (Yuan et al., 2023) to test for various capabilities; specifically, we investigate action recognition on Kinetics-400 (K400) (Kay et al., 2017), motion classification on Something-Something-v2 (SSv2) (Goyal et al., 2017), and action localization on AVA (Gu et al., 2018). Action classification on Kinetics evaluates the appearance-based understanding of the model, as many action classes in the dataset can be inferred from the presence of specific objects in the video (Sevilla-Lara et al., 2021). Motion classification on Something-Something-v2 evaluates the temporal understanding of the model, as action classes in the dataset are decoupled from the appearance/presence of specific objects in the video (Goyal et al., 2017). Finally, action localization on AVA evaluates the ability of the model to understand and localize motions in the video. We follow standard practice and report accuracy on K400 and SSv2 by sampling several spatial and temporal views. For static image tasks, we explore object recognition on ImageNet (Russakovsky et al., 2015), scene classification on Places205 (Zhou et al., 2014), and fine-grained recognition on iNaturalist 2021 (Van Horn et al., 2018).

## 4 What Matters for Learning Representations from Video?

In this section we isolate the contributions of several design choices, including: a) the use of a feature prediction versus pixel prediction objective, b) the construction of the pretraining data distribution, c) the feature pooling strategy for leveraging the model’s representations in downstream tasks, and d) the masking strategy, towards identifying: what to predict from what?

### 4.1 Predicting Representations versus Pixels

We first ablate the effect of computing the prediction loss in representation space. We train a pair of ViT-L/16 models using either a V-JEPA feature prediction loss, or a mean-squared error loss with the normalized pixel values, as in masked autoencoders (He et al., 2021), and perform a sweep over the learning rate and weight decay schedules for both approaches. All models are pretrained on VideoMix2M for 90K iterations with a batch size of 3072 using multi-block masking. We examine performance on Kinetics-400 (K400), Something-Something-v2 (SSv2), and ImageNet-1K (IN1K), using a frozen backbone with an attentive probe, and report top-1 accuracy using a single center view. We also examine end-to-end fine-tuning performance of the models on Kinetics-400.

Results of this comparison are reported in Table 2 and indicate that predicting in feature space provides a consistent performance improvement over pixel space prediction in both frozen evaluation of the video backbone, as well as end-to-end fine-tuning.

### 4.2 Pretraining Data Distribution

Next we study the impact of the pretraining data distribution in Table 2. Leveraging large scale datasets has been critical for enabling the surge of advancements in other modalities, such as text and images (Kaplan et al., 2020; Cherti et al., 2023). We investigate whether a similar trend holds for video data. To control for the possible confounding variable of compute budget, we pretrain all models in Table 2 for 90K iterations using a batch-size of 3072. We report downstream results on K400, SSv2, and IN1K using a frozen backbone with an attentive probe, and report top-1 accuracy using a single center view.

Table 2 shows that average performance across tasks monotonically increases as we increase the size of the pretraining dataset, but the best task-specific performance is obtained by independently selecting the pretraining data for each specific downstream task. For instance, the L/16 obtains its best SSv2 performance when pretrained on K710+SSv2, its best K400 performance when pretrained only on K710, and its best IN1K performance when pretrained only on K710+HT. The best average performance across all tasks is achieved by pretraining VideoMix2M, which combines all the data sources. Similarly, the H/16 pretrained on K710+SSv2 achieves a greater K400 score than the H/16 pretrained on VideoMix2M, however, the top performing H/16 on average is pretrained on VideoMix2M.

|  |  | Frozen Evaluation |  |  |  |
|---|---|---|---|---|---|
|  |  | K400 | SSv2 |  |  |
|  |  | (16×\times1×\times1) | (16×\times1×\times1) |  |  |
| Method | Arch. | Avg. | Att. | Avg. | Att. |
| V-JEPA | ViT-L/16 | 56.7 | 73.7 | 50.1 | 66.2 |

### 4.3 Evaluation: Attentive Probing

Next we explore the feature pooling strategy for applying the model’s representations in downstream tasks. Since the prediction objective in equation (1) is unnormalized, there is no a priori reason for the encoder to yield a linearly separable subspace (Chen et al., 2020). Thus, rather than using a linear operation (averaging) to pool the features output of the frozen backbone, we explore a learnable non-linear pooling strategy. Specifically, when evaluating the frozen pretrained backbone on downstream tasks, we learn a cross-attention layer with a learnable query token. The output of the cross-attention layer is then added back to the query token (residual connection), and then fed into two-layer MLP with a single GeLU activation, followed by a LayerNorm, and finally a linear classifier.

In Table 3 we see that using adaptive pooling with a learnable cross-attention layer leads to a significant improvement of +17+17 points on K400 and +16.1+16.1 points on SSv2. Using an attentive-probe is also beneficial for other baseline models as reported in Appendix E.

### 4.4 Prediction Task: Predicting yy from xx

We conduct an ablation on the masking strategy used in V-JEPA pretraining. We examine the following masking strategies: random-tube[r] in which xx is obtained by removing a random fraction rr of tubes (spatial patches extended across the entire temporal duration) from the video, causal multi-block[p] in which xx is restricted to the first pp frames of the 16-frame video, which are then masked with a random set of spatio-temporal blocks, and multi-block in which xx obtained by masking a random set of spatio-temporal blocks from the entire video. Spatio-temporal blocks are sampled using the parameters described in Section 3.2; an ablation on the size and quantity of masked spatio-temporal blocks is provided in Appendix E.4.

|  | Frozen Evaluation |  |  |
|---|---|---|---|
|  | K400 | SSv2 | IN1K |
| Masking | (16×\times1×\times1) | (16×\times1×\times1) |  |
| random-tube[0.9] | 51.5 | 46.4 | 55.6 |
| causal multi-block[6] | 61.3 | 49.8 | 66.9 |
| causal multi-block[12] | 71.9 | 63.6 | 72.2 |
| multi-block | 72.9 | 67.4 | 72.8 |

Table 4 indicates that the best results are obtained by sampling xx using a multi-block strategy, wherein the network is forced to make predictions after removing large continuous blocks in the video. When xx is only sampled from the first few frames of the video, as in the causal multi-block strategy, we observe a decrease in downstream performances. Finally, the random-tube strategy, wherein 90% of the tubes in the video are randomly masked, leads to features of low-semantic quality when combined with our feature prediction objective.

|  |  |  |  | Frozen Evaluation w/ Att. Pooling | Fine-Tuning |  |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  | #Samples |  | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 | K400-ft | SSv2-ft |
| Method | Arch. | Seen | Iter. | (16×\times8×\times3) | (16×\times2×\times3) |  |  |  |  | (16×\times5×\times3) | (16×\times2×\times3) |
| Methods pretrained using pixel prediction |  |  |  |  |  |  |  |  |  |  |  |
| OmniMAE | ViT-L/16 | 2400M | 1170K | 65.6 | 60.6 | 14.4 | 75.1 | 59.8 | 66.1 | 84.0 | 74.2 |
| VideoMAE | ViT-L/16 | 410M | 400K | 77.8 | 65.5 | 21.6 | 71.1 | 59.3 | 64.6 | 85.4 | 74.3 |
| Hiera | Hiera-L | 770M | 1500K | 75.5 | 64.2 | 15.8 | 68.9 | 58.5 | 56.9 | 87.3 | 75.1 |
| V-JEPA | ViT-L/16 | 270M | 90K | 80.8 | 69.5 | 25.6 | 74.8 | 60.3 | 67.8 | 85.6 | 75.1 |

|  |  |  |  | Video Tasks | Image Tasks |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 |
| Method | Arch. | Params. | Data | (16×\times8×\times3) | (16×\times2×\times3) |  |  |  |  |
| Methods pretrained on Images |  |  |  |  |  |  |  |  |  |
| I-JEPA | ViT-H/16512 | 630M | IN22K | 79.7 | 50.0 | 19.8 | 84.4 | 66.5 | 85.7 |
| OpenCLIP | ViT-G/14 | 1800M | LAION | 81.8 | 34.8 | 23.2 | 85.3 | 70.2 | 83.6 |
| DINOv2 | ViT-g/14 | 1100M | LVD-142M | 83.4 | 50.6 | 24.3 | 86.2 | 68.4 | 88.8 |
| Methods pretrained on Videos |  |  |  |  |  |  |  |  |  |
| MVD | ViT-L/16 | 200M | IN1K+K400 | 79.4 | 66.5 | 19.7 | 73.3 | 59.4 | 65.7 |
| OmniMAE | ViT-H/16 | 630M | IN1K+SSv2 | 71.4 | 65.4 | 16.0 | 76.3 | 60.6 | 72.4 |
| VideoMAE | ViT-H/16 | 630M | K400 | 79.8 | 66.2 | 20.7 | 72.3 | 59.1 | 65.5 |
| VideoMAEv2 | ViT-g/14 | 1100M | Un.Hybrid | 71.2 | 61.2 | 12.9 | 71.4 | 60.6 | 68.3 |
| Hiera | Hiera-H | 670M | K400 | 77.0 | 64.7 | 17.5 | 71.4 | 59.5 | 61.7 |
| V-JEPA | ViT-L/16 | 200M | VideoMix2M | 80.8 | 69.5 | 25.6 | 74.8 | 60.3 | 67.8 |
| ViT-H/16 | 630M | 82.0 | 71.4 | 25.8 | 75.9 | 61.7 | 67.9 |  |  |
| ViT-H/16384 | 630M | 81.9 | 72.2 | 25.0 | 77.4 | 62.8 | 72.6 |  |  |

## 5 Comparison with Prior Work

In Section 5.1, we investigate the impact of feature prediction by comparing V-JEPA with video approaches that rely on pixel prediction, while using a similar architecture for all baselines. Subsequently, in Section 5.2, we remove the architectural constraint and report the best performance across architectures for self-supervised video and image pretraining approaches. Finally, we explore the label-efficiency of V-JEPA relative to other self-supervised video pretraining approaches in Section 5.3. We further detail the evaluation setup in Appendix D.

### 5.1 Comparison with Pixel Prediction

To investigate the effectiveness of feature prediction pretraining, we first compare V-JEPA to video masked modeling models relying on a pixel prediction loss. We control for the possible confounding factor of model architecture by evaluating all models using either a ViT-L/16 encoder, or a Hiera-L encoder, which has a similar number of parameters. For the pixel prediction baselines we consider VideoMAE (Tong et al., 2022; Wang et al., 2023a), which trains vision transformer autoencoders exclusively on video, Hiera (Ryali et al., 2023), which trains a hierarchical transformer autoencoder on video, and OmniMAE (Girdhar et al., 2023), which trains a vision transformer autoencoder on static images and video simultaneously.

Table 6 examines both frozen evaluation with an attentive probe on downstream video and image tasks, as well as end-to-end fine-tuning. In frozen evaluation, V-JEPA outperforms the baselines on all downstream tasks, except ImageNet, where we achieve 74.8%74.8\% compared to 75.1%75.1\% of an OmniMAE model trained directly on ImageNet; hence, V-JEPA achieves comparable ImageNet performance despite only pretraining on video.

Under the fine-tuning protocol, V-JEPA also achieves the best performance of any model trained with a ViT-L/16, and matches the performance of the Hiera-L on SSv2, which benefits from a hierachical prior (Ryali et al., 2023). The V-JEPA models achieve this result while processing significantly fewer samples during pretraining (Figure 4), demonstrating the efficiency of feature prediction as a learning principle.

### 5.2 Comparison with State-of-the-Art

Next, in Table 6, we inspect how the V-JEPA models pretrained on video stack up next to the largest state-of-the-art self-supervised image and video models when freezing the backbone encoder and training an attentive probe on top. Our image pretrained baselines include OpenCLIP (Cherti et al., 2023), DINOv2 (Oquab et al., 2023), and I-JEPA (Assran et al., 2023). The OpenCLIP model is trained with a contrastive image-text alignment objective, DINOv2 and I-JEPA are trained with self-supervision. These models are known to excel in their frozen-evaluation performance (Oquab et al., 2023); i.e., their ability to produce visual features that can be applied to many downstream tasks simultaneously, without end-to-end fine-tuning, and thus provide highly competitive baselines. Our video pretrained baselines include VideoMAE (Tong et al., 2022), OmniMAE (Girdhar et al., 2023), Hiera (Ryali et al., 2023), VideoMAEv2 (Wang et al., 2023a), and MVD (Wang et al., 2023b). The OpenCLIP, DINOv2 and VideoMAEv2 models are parameterized as Giant/Gigantic vision transformer architectures containing over 1B parameters trained on large-scale image or video datasets.

|  |  | Frozen Evaluation |  |  |  |  |  |
|---|---|---|---|---|---|---|---|
|  |  | K400 | SSv2 |  |  |  |  |
|  |  | (16×\times8×\times3) | (16×\times2×\times3) |  |  |  |  |
|  |  | 5% | 10% | 50% | 5% | 10% | 50% |
| Method | Arch. | (∼\sim29 samples per class) | (∼\sim58 samples per class) | (∼\sim287 samples per class) | (∼\sim48 samples per class) | (∼\sim96 samples per class) | (∼\sim440 samples per class) |
| MVD | ViT-L/16 | 62.6 ±\pm 0.2 | 68.3 ±\pm 0.2 | 77.2 ±\pm 0.3 | 42.9 ±\pm 0.8 | 49.5 ±\pm 0.6 | 61.0 ±\pm 0.2 |
| VideoMAE | ViT-H/16 | 62.3 ±\pm 0.3 | 68.5 ±\pm 0.2 | 78.2 ±\pm 0.1 | 41.4 ±\pm 0.8 | 48.1 ±\pm 0.2 | 60.5 ±\pm 0.4 |
| VideoMAEv2 | ViT-g/14 | 37.0 ±\pm 0.3 | 48.8 ±\pm 0.4 | 67.8 ±\pm 0.1 | 28.0 ±\pm 1.0 | 37.3 ±\pm 0.3 | 54.0 ±\pm 0.3 |
| V-JEPA | ViT-H/16 | 67.0 ±\pm 0.2 | 72.1 ±\pm 0.1 | 80.2 ±\pm 0.2 | 51.9 ±\pm 0.3 | 57.5 ±\pm 0.4 | 67.3 ±\pm 0.2 |
| ViT-H/16384 | 68.2 ±\pm 0.2 | 72.8 ±\pm 0.2 | 80.6 ±\pm 0.2 | 54.0 ±\pm 0.2 | 59.3 ±\pm 0.5 | 67.9 ±\pm 0.2 |  |

Compared to large-scale video baselines, the V-JEPA models outperform all previous models on every downstream video and image task with notable margin (see Table 6). Our H/16 model outperforms the largest publicly available VideoMAE, VideoMAEv2, OmniMAE, MVD, and Hiera models by at least +5+5 points in motion understanding (Something-Something-v2), +2+2 points in action recognition (Kinetics-400), +5+5 points on action detection (AVA), +1+1 point on object recognition (ImageNet-1K), +2+2 points in scene recognition (Places205), and +0.2+0.2 points on fine-grained recognition (iNaturalist). Moreover, when comparing pretraining wallclock time in Figure 5, we see that V-JEPA achieves this performance with a roughly 2×2\times speedup compared to the large pixel prediction models.

On tasks that require a fine-grained understanding of motion (Something-Something-v2), the V-JEPA models provide a major improvement (over +21+21 points) compared to large-scale image baselines, such as DINOv2, OpenCLIP, and I-JEPA. Self-supervised pretraining from videos allows to model dynamic concepts that are not easily learned from static image datasets. Similarly, we observe that the V-JEPA models outperform image-based pretraining on action localization.

On Kinetics-400, we find image models to perform well; e.g., while DINOv2 (Oquab et al., 2023) previously reported 78.4%78.4\% on K400 with a linear probe, we improve the frozen evaluation of the g/14 model to 83.4%83.4\% by using an attentive probe. In this case, our H/16 model achieves 82.0%82.0\% top-1 accuracy. It is worth noting that the label for many Kinetics videos can be inferred using appearance-based cues, without requiring an understanding of motion (Sevilla-Lara et al., 2021).

The V-JEPA models narrow the gap with image models on image classification tasks. In particular, V-JEPA achieves a score of 77.4%77.4\% on ImageNet using a one-layer attentive probe, which can be further improved to 77.9%\bf{77.9\%} using a two-layer attentive probe. More generally, we hypothesize that the datasets used to train V-JEPA and other video models are too constrained and lack the visual diversity of the internet-scale pretraining data used by the images models; as such, there is value in focusing future work on building diverse publicly available video datasets.

### 5.3 Label-efficiency

We examine the label-efficiency of V-JEPA compared to other self-supervised video models by measuring the ability of the pretrained backbones to adapt to downstream tasks with few labels. Specifically, we investigate the performance of the frozen models on Kinetics-400 and Something-Something-v2 as we vary the percentage of labeled examples from each dataset available for training the attentive probe. We train the probes in several low-shot settings: using either 5% of the train set, 10%, or 50%, and take 3 random splits in each setting to obtain more robust metrics, resulting in 9 different evaluation experiments for each model. Table 7 reports the mean performances and standard deviation using the K400 and SSv2 validation sets.

We find V-JEPA to be more label-efficient than other self-supervised video models: decreasing the available number of labeled examples for training the attentive probe results in an increase in the performance gap between V-JEPA and the other models. In particular, the performance of the largest V-JEPA model on K400 drops by 12% to 68.2% top-1 when we reduce the number of labeled examples by a factor of 10×10\times (from roughly 287 examples per class to 29 examples per class). By contrast, VideoMAEv2 drops by 30% to 37.0% top-1, VideoMAE drops by 15.9% to 62.3% top-1, and MVD drops by 14.6% to 62.6% top-1.

Similar observations hold on SSv2. The performance of the largest V-JEPA model on SSv2 drops by 13.9% to 54.0% top-1 when we reduce the number of labeled examples by a factor of 10×10\times (from roughly 440 examples per class to 48 examples per class). By contrast, VideoMAEv2 drops by 26% to 28.0% top-1, VideoMAE drops by 19.1% to 41.4% top-1, and MVD drops by 18.1% to 42.9% top-1.

## 6 Evaluating the Predictor

Next, we seek to qualitatively inspect the V-JEPA models. Recall that the predictor network in V-JEPA predicts the representations of a masked spatio-temporal region yy from a visible region xx, given the positional information of the masked regions (see Section 3). To qualitatively investigate the grounding of the feature-space predictions, we freeze the pretrained encoder and predictor networks and train a conditional diffusion decoder to map the V-JEPA predictions to interpretable pixels. Notably, the decoder is only fed the representations predicted for the missing regions of the video, and does not have access to the unmasked regions of the video (see Figure 6(a)).

Given a masked video, we use the V-JEPA pretrained models to predict the representations of the missing regions, and then use the decoder to project the representations to pixel space. Figure 6(b) shows decoder outputs for various random seeds. Qualities that are common across samples represent information that is contained in the predictor representation.

Figure 6(b) shows that the V-JEPA feature predictions are indeed grounded, and exhibit spatio-temporal consistency with the unmasked regions of the video. Specifically, the samples in Figure 6(b) show that the V-JEPA predictor correctly captures positional uncertainty and produces a variety of visual objects at various locations with consistent motion. Some of the samples also demonstrate an understanding of object-permanence, as the visual objects remain consistent after partial occlusion.

## 7 Conclusion

In this work, we explored the effectiveness of feature prediction as a stand-alone objective for unsupervised learning from video and introduced V-JEPA, a collection of vision models trained solely using a self-supervised feature prediction objective. The V-JEPA models demonstrate the ability to solve various downstream image and video tasks without adaption of the model parameters, and outperform previous video representation learning approaches in frozen evaluation on action recognition, spatio-temporal action detection, and image classification tasks. Additionally, we show that pretraining V-JEPA on videos is particularly effective for solving downstream tasks requiring fine-grained motion understanding, while large-scale image models trained on internet scale datasets fall short on such tasks. Finally, we empirically observed that V-JEPA models are label-efficient learners, and exhibit good performance on downstream tasks, even when only few labeled examples are available.

## Appendix A Extended Related Works

We first review approaches for learning visual perception from static images before discussing strategies for learning from video.

### Weakly-Supervised Learning from Static Images

One family of approaches for learning visual perception from static images trains a visual encoder to predict the representations of text captions often found accompanying images from the Web, as in CLIP (Radford et al., 2021) or CoCa (Yu et al., 2022). The largest open source CLIP model to date, numbering 2B parameters and trained on over 2B web-scraped images (Cherti et al., 2023), demonstrates impressive performance on a wide range of downstream image and video tasks. Notably, this is achieved using only the light-weight adaptation of task-specific heads, also referred to as frozen-evaluation, and does not require expensive end-to-end fine-tuning of the pretrained model.

### Self-Supervised Learning from Static Images

Other approaches for learning from static images leverage unsupervised objectives. Initial works on self-supervised approaches are based on sparse coding or hand-crafted pretext tasks, such as colorization (Larsson et al., 2016; Larsson et al., 2017), rotation prediction (Gidaris et al., 2020), and jigsaws (Noroozi and Favaro, 2016). More recent approaches leverage invariance-based objectives by training a visual encoder to be invariant to hand-crafted image transformations (Wu et al., 2018; Chen et al., 2020).

Another family of methods learn representations using denoising autoencoders (Vincent et al., 2008); image inpainting is one popular instantiation of this idea (Pathak et al., 2016). More recently, masked autoencoders (He et al., 2021) train an encoder-decoder transformer to predict missing pixels of a masked image. Follow-up work addresses the indeterminism of pixel reconstruction by exploring instantiations of masked image modeling in latent space (Baevski et al., 2022b; Assran et al., 2023; Baevski et al., 2022a). These approaches can be seen as applications of the predictive feature principle in the image modality.

There are also various methods that combine both masked image modeling and invariance criteria to learn visual representations from static images, such as iBOT (Zhou et al., 2021) and DINOv2 (Zhou et al., 2021; Oquab et al., 2023), the latter is currently the most competitive instantiation of self-supervised learning with static images, scaled to a model with over 1.1B parameters trained on a curated dataset of 142M images.

### Weakly-Supervised Learning from Videos

One family of approaches for learning visual perception from videos relies on weakly-supervised guidance from closed captioning, often computed from an ASR transcription of audio data accompanying internet videos. For instance, VideoBERT (Sun et al., 2019; Xu et al., 2021) trains a video encoder to predict masked spans in the textual closed captions. Similarly, VideoCLIP (Xu et al., 2021) trains a video encoder to predict the representation of video captions computed by a text encoder. Follow-up work such as MERLOT (Zellers et al., 2022), VATT (Akbari et al., 2021), and InternVideo (Wang et al., 2022) extended VideoCLIP by incorporating additional unsupervised objectives.

### Self-Supervised Learning from Videos

Similar to unsupervised learning from images, a family of unsupervised video representation learning approaches enforces a spatio-temporal representation of a video clip to be invariant to hand-crafted spatio-temporal data augmentations (Parthasarathy et al., 2022). However, one obvious insight is that the temporal ordering of visual information in video can provide implicit supervision. Indeed, this insight is the key insight leveraged by many works on unsupervised video learning. Towards leveraging temporal information as supervision, some approaches train a visual encoder by predicting the temporal ordering of frames (Xu et al., 2019; Lee et al., 2017). Other approaches seek to predict low-level motion vectors computed from optical flow (Pintea et al., 2014), or to predict mixing pixels in video frames, using either a frame-interpolation objective (Kalluri et al., 2023) or a denoising autoencoder (Tong et al., 2022; Feichtenhofer et al., 2022; Wang et al., 2023a).

## Appendix B Extended Description of V-JEPA

In this section, we provide an in-depth description of our approach V-JEPA that is illustrated in Figure 3.

Unless stated otherwise, during during pretraining, we always randomly sample a clip of 16 frames from each input video with a temporal stride of 4 between sampled frames. An input video clip therefore covers 64 frames in total, or roughly 2 seconds of a given video running at 30 frames per second. We then resize the video’s spatial dimensions to 224×224224\times 224, resulting in an overall shape of 16×224×224×316\times 224\times 224\times 3 for the entire clip. Since ViT networks process a 1D sequence of tokens, we must convert an input video clip into a 1D token sequence. To do so, we apply a 3D convolution comprising dd filters of size 2×16×162\times 16\times 16 with a temporal stride of 22 and a spatial stride of 1616, resulting in a tensor of shape 8×14×14×d8\times 14\times 14\times d. Next we add absolute 3D sin-cos positional embeddings to the spatio-temporal feature map and flatten it, resulting in a 1D token sequence of shape 1568×d1568\times d. This process is demonstrated in Figure 7.

We sample both a video clip, and a video mask in each iteration. We denote a video clip represented as a 1D token sequence of length L=1568L=1568 by xL=(x1,…,xL)x_{L}=(x_{1},\ldots,x_{L}). Similarly, given a mask of M<LM<L patches, leaving N=L−MN=L-M patches unmasked, we denote the indices of masked patches by (i1,…,iM)(i_{1},\ldots,i_{M}) and its complement (the indices of unmasked patches) by (j1,…,jN)(j_{1},\ldots,j_{N}).

Computing the xx-representations. To compute the V-JEPA loss, we first produce the xx-representations by masking the video clip and feeding it into the xx-encoder; we denote the masked video by xN=(xj1,…,xjN)x_{N}=(x_{j_{1}},\ldots,x_{j_{N}}). Applying the xx-encoder Eθ​(⋅)E_{\theta}(\cdot) to the masked clip gives a sequence of patch representations, denoted as zN=Eθ​(xN)=(zj1,…,zjN).z_{N}=E_{\theta}(x_{N})=(z_{j_{1}},\ldots,z_{j_{N}}).

Predicting the target. Next, the V-JEPA predictor network Pϕ​(⋅,⋅)P_{\phi}(\cdot,\cdot) takes as input the tokens produced by the xx-encoder and predicts the missing regions in the video clip, which are specified by a set of learnable mask tokens. Specifically, the mask tokens are parameterized as the sum of a shared learnable vector and an absolute 3D sin-cos positional embedding, denoted by mM=(mi1,…,miM)m_{M}=(m_{i_{1}},\ldots,m_{i_{M}}). The output of the predictor is thus given by, s^M=Pϕ​(zN,mM)=(s^i1,…,s^iM),\hat{s}_{M}=P_{\phi}(z_{N},m_{M})=(\hat{s}_{i_{1}},\ldots,\hat{s}_{i_{M}}), corresponding to a dd-dimensional output for each of the MM masked patches.

Computing the yy-representations. Finally to compute the prediction targets, the entire unmasked video clip is processed by the yy-encoder to obtain a set of target representations, denoted by sL=E¯θ​(xL)=(s1,…,sL).s_{L}=\overline{E}_{\theta}(x_{L})=(s_{1},\ldots,s_{L}). The V-JEPA loss is now computed as

|  | Loss=1M​∑k∈(i1,…,iM)∥s^k−sk∥1,\text{Loss}=\frac{1}{M}\sum_{k\in(i_{1},\ldots,i_{M})}\lVert\hat{s}_{k}-s_{k}\rVert_{1}, |  | (2) |
|---|---|---|---|

which is simply the average L1L_{1} distance between the output of the predictor and the yy-encoder. We then compute a gradient update with respect to the parameters of the xx-encoder, θ\theta, and the predictor, ϕ\phi, and subsequently update the parameters of the yy-encoder as an exponential moving average of the context encoder weights (Polyak average).

To increase the efficiency of V-JEPA, we use a multi-masking strategy (Caron et al., 2020; Baevski et al., 2022a), which enables us to amortize the cost of the target computation. As mentioned in Section 3, for a given video clip, we sample 2 different masks, short-range and long-range. While we need to forward propagate the xx-encoder and predictor separately for each mask, we only need to compute the yy-representation once.

## Appendix C Pretraining details

| Hyper-parameter | ViT-L/16224 | ViT-H/16224 | ViT-H/16384 |
|---|---|---|---|
| data |  |  |  |
| datasets | VideoMix2M | VideoMix2M | VideoMix2M |
| resolution | 224 | 224 | 384 |
| num_frames | 16 | 16 | 16 |
| temporal_stride | 4 | 4 | 4 |
| horizontal_flip | true | true | true |
| random_resize_scale | (0.3, 1.0) | (0.3, 1.0) | (0.3, 1.0) |
| random_resize_aspect_ratio | (0.75, 1.35) | (0.75, 1.35) | (0.75, 1.35) |
| masking |  |  |  |
| block_aspect_ratio | (0.75, 1.5) | (0.75, 1.5) | (0.75, 1.5) |
| shortrange_mask_num_blocks | 8 | 8 | 8 |
| shortrange_mask_spatial_scale | 0.15 | 0.15 | 0.15 |
| longrange_mask_num_blocks | 2 | 2 | 2 |
| longrange_mask_spatial_scale | 0.7 | 0.7 | 0.7 |
| optimization |  |  |  |
| batch_size | 3072 | 3072 | 2400 |
| total_number_of_iterations | 90000 | 90000 | 90000 |
| warmup_iterations | 12000 | 12000 | 12000 |
| lr | 6.25e-4 | 6.25×10−4\times 10^{-4} | 6.25×10−4\times 10^{-4} |
| start_lr | 2×10−4\times 10^{-4} | 2×10−4\times 10^{-4} | 2×10−4\times 10^{-4} |
| final_lr | 1×10−6\times 10^{-6} | 1×10−6\times 10^{-6} | 1×10−6\times 10^{-6} |
| start_momentum | 0.998 | 0.998 | 0.998 |
| final_momentum | 1.0 | 1.0 | 1.0 |
| start_weight_decay | 0.04 | 0.04 | 0.04 |
| final_weight_decay | 0.4 | 0.4 | 0.4 |
| scheduler_scale_factor | 1.25 | 1.25 | 1.25 |
| architecture |  |  |  |
| patch_size | 16 | 16 | 16 |
| tubelet_size | 2 | 2 | 2 |
| pred_depth | 12 | 12 | 12 |
| pred_embed_dim | 384 | 384 | 384 |
| hardware |  |  |  |
| dtype | bfloat16 | bfloat16 | bfloat16 |
| accelerator | A100 80G | A100 80G | A100 80G |

In section, we report V-JEPA pretraining details. Table 8 summarizes the main hyperparameters used during pretraining.

We use Vision Transformer (Dosovitskiy et al., 2020) (ViT) architectures for the xx-encoder and yy-encoder. We train three V-JEPA encoders: a ViT-L/16224, a ViT-H/16224 and a ViT-H/16384. All three encoders take as input a short video clip of 16 frames with a temporal stride of 4 between consecutive frames. The subscripts, 224224 and 384384, indicate the spatial resolution of the video clip. V-JEPA flattens the video clip into a sequence of non-overlapping spatio-temporal patches of size 16×16×216\times 16\times 2 (see Figure 7). For all three models, the predictor is designed as a narrow ViT architecture, consisting of 12 transformer blocks with an embedding dimension of 384. For simplicity, we keep the number of self-attention heads in the predictor equal to that of the backbone used for the context-encoder/target-encoder. V-JEPA is pretrained without using a [cls] token.

We use AdamW (Loshchilov and Hutter, 2017) to optimize the xx-encoder and predictor weights. The ViT-L/16224 and ViT-H/16224 models use a batch size of 30723072 while the ViT-H/16384 uses a batch size of 24002400. Models are trained for a total of 90,000 iterations. The learning rate is linearly increased from 2×10−42\times 10^{-4} to 6.25×10−46.25\times 10^{-4} during the first 12,00012,000 iterations of pretraining, and decayed to 10−610^{-6} following a cosine schedule. Weight-decay is also linearly increased from 0.040.04 to 0.40.4 throughout pretraining. The yy-encoder weights are initialized identically to the xx-encoder, and subsequently updated as an exponential moving average (EMA) (Tarvainen and Valpola, 2017) of the xx-encoder weights using a momentum value which starts at 0.9980.998 and is linearly increased to 1.01.0 during training (Caron et al., 2021; Assran et al., 2022). We scale all hyper-parameter schedules 25% beyond the actual training schedule. Specifically, the learning rate schedule, weight-decay schedule, and EMA schedule are computed assuming a training length of 112,500 iterations, even though we only train our model for 90,000 iterations. We found the last 25%25\% of the default scheduler period to update hyper-parameters too aggressively, and simply truncating the schedulers improved performance.

As described in Section 3, we propose a 3D Multi-Block masking strategy. We use two type of masks: short-range masks, where we take the union of 88 randomly sampled target blocks with a spatial scale of 0.150.15, and long-range masks, where we take the union of 22 randomly sampled target blocks with a spatial scale of 0.70.7. In both cases, the aspect ratio for all sampled blocks is randomly chosen in the range (0.75,1.5)(0.75,1.5).

## Appendix D Evaluation details

### D.1 Frozen classification

| Hyper-parameter | K400 | SSv2 | IN1K | Place205 | iNat21 |
|---|---|---|---|---|---|
| data |  |  |  |  |  |
| num_clips | 8 | 1 | N.A. | N.A. | N.A. |
| num_frames | 16 | 16 | N.A. | N.A. | N.A. |
| temporal_stride | 4 | 4 | N.A. | N.A. | N.A. |
| horizontal_flip | true | true | true | true | true |
| random_resize_scale | (0.08, 1.0) | (0.08, 1.0) | (0.08, 1.0) | (0.08, 1.0) | (0.08, 1.0) |
| random_resize_aspect_ratio | (0.75, 1.33) | (0.75, 1.33) | (0.75, 1.33) | (0.75, 1.33) | (0.75, 1.33) |
| auto_augment | false | false | true | true | true |
| optimization |  |  |  |  |  |
| batch_size | 256 | 256 | 1024 | 1024 | 1024 |
| epochs | 20 | 20 | 20 | 20 | 20 |
| lr | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| final_lr | 0 | 0 | 0 | 0 | 0 |
| weight_decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

Given an input video, xLx_{L}, the V-JEPA target encoder E¯θ​(⋅)\overline{E}_{{\theta}}(\cdot) outputs a sequence of LL tokens, Eθ​(xL)=(s1,…,sL)E_{\theta}(x_{L})=(s_{1},\ldots,s_{L}), where si∈ℝds_{i}\in\mathbb{R}^{d}. To pool this sequence of tokens into a single feature vector, we apply a lightweight non-linear cross-attention block which replace the self-attention operation of a transformer block with cross attention. Specifically, the cross-attention performs the following computation:

|  | ∑i=1Lexp⁡(q⊤​𝐖𝐤​si)∑jexp⁡(q⊤​𝐖𝐤​sj)​𝐖𝐯​si,\sum^{L}_{i=1}{\frac{\exp(q^{\top}{\bf W_{k}}s_{i})}{\sum_{j}\exp(q^{\top}{\bf W_{k}}s_{j})}{\bf W_{v}}s_{i}}, |  |
|---|---|---|

where 𝐖𝐤,𝐖𝐯∈𝐑𝐝×𝐝\bf{W_{k}},{\bf W_{v}}\in R^{d\times d} are the key and value matrices, and q∈Rdq\in R^{d} is a learnable query token. The output of the cross-attention is then added back to the query token (residual connection), and then fed into two-layer MLP with a single GeLU activation, followed by a LayerNorm, and finally a linear classifier. The parameters of the cross-attention block are jointly learned with that of the linear classifier for the downstream task, while the encoder parameters are kept frozen. Note that, in practice, we actually use an attentive probe with 12 heads, each of dimension 1212. In Appendix E we show that baselines benefit from the attentive probing protocol.

For all the tasks, we use AdamW optimizer with a cosine scheduler (no warmup) that decays the learning rate from 0.0010.001 to 00. We use a fixed weight-decay of 0.010.01 and apply simple data augmentations (random resized crops and horizontal flips) during training of the attentive probe, except on image tasks, where we apply AutoAugment (Dogus Cubuk et al., 2019). Table 9 reports the hyperparameters for each downstream evaluation.

Unless stated otherwise, our attentive probe takes 8 clips of 16 frames as input on Kinetics, and 2 clips of 16 frames on Something-Somethingv2 to increase the temporal coverage of the video. Specifically, we first divide a video in 8 (or 2) equal-length temporal segments, and sample 1 clip at random per segment. The video encoder E¯θ\overline{E}_{{\theta}} processes each clip separately and produces a clip-level feature map. The feature maps for each clip are then concatenated together and fed to the attentive probe. At test time, we average the prediction of 3 spatial views following standard practice in video classification.

To evaluate the video models on image tasks, we simply duplicate input images to generate still video clips of 16 frames. We perform this duplication operation simply for convenience in evaluation of the video models, however we find this step to be unnecessary in general. Given a video tokenizer implemented as a 3D-conv with a temporal stride of 22, it is sufficient to simply duplicate the image into a 2 frame video clip. This would result in the same number of input tokens as that produced by a static image model with a 2D-conv tokenizer.

To evaluate image models such as DINOv2 and OpenCLIP on video tasks, we simply process each frame independently with the image encoder to produce a frame-level feature map. The feature maps for each frame are then concatenated and fed to the attentive probe, just as we do with the clip-level feature maps when evaluating video models.

### D.2 Frozen detection

We evaluate our model on the AVA (Gu et al., 2018) spatio-temporal localization of human actions dataset, containing 211k training and 57k validation video segments. We follow the experimental protocol of (Feichtenhofer et al., 2021), and use precomputed masks from a pretrained Faster-RCNN adapted to videos, which uses a ResNeXt-101-FPN backbone and is pretrained on ImageNet and COCO. We train a linear classifier on top of the frozen V-JEPA features to classify the extracted regions of interest and report mean Average Precision (mAP) on the 60 most common classes. Hyper-parameters are provided in Table 10. Our frozen features are obtained by concatenating the last layer of the transformer encoder with three intermediate layers. We use a batch size of 64 and pretrain for 30 epochs with AdamW using a learning rate of 0.0001 with 2 epochs of warmup and a weight decay of 0.05.

| Hyper-parameter | ViT-L/16 | ViT-H/16 |
|---|---|---|
| out_layers | [18, 20, 22, 24] | [26, 28, 30, 32] |
| batch_size | 64 | 64 |
| epochs | 30 | 30 |
| opt | AdamW | AdamW |
| opt_eps | 0.00000001 | 0.00000001 |
| momentum | 0.9 | 0.9 |
| weight_decay | 0.05 | 0.05 |
| lr | 0.0001 | 0.0001 |
| warmup_lr | 0.000001 | 0.000001 |
| min_lr | 0.000001 | 0.000001 |
| warmup_epochs | 2 | 2 |
| warmup_steps | 1 | 1 |

### D.3 Finetuning

Following Tong et al. (2022), we finetune a linear layer on top of our model, using a layer decay schema and mixup as the data augmentation pipeline. We provide all hyper-parameters for both K400 and SSv2 in Table 11.

| Hyper-parameter | K400 | SSv2 |  |  |
|---|---|---|---|---|
| data |  |  |  |  |
| num_segments | 1 |  |  |  |
| num_frames | 16 |  |  |  |
| sampling_rate | 4 |  |  |  |
| resolution | 224 |  |  |  |
| model |  |  |  |  |
| model_name | ViT-L/16 | ViT-H/16 | ViT-L/16 | ViT-H/16 |
| drop_path | 0.1 | 0.2 | 0.2 | 0.2 |
| head_drop_rate | 0. | 0. | 0.5 | 0.5 |
| optimization |  |  |  |  |
| batch_size | 256 | 1024 | 256 | 256 |
| epochs | 35 | 25 | 15 | 15 |
| opt | adamw |  |  |  |
| opt_eps | 0.00000001 |  |  |  |
| momentum | 0.9 |  |  |  |
| weight_decay | 0.05 |  |  |  |
| lr | 0.002 | 0.0005 | 0.0005 | 0.0005 |
| layer_decay | 0.75 | 0.75 | 0.75 | 0.75 |
| warmup_lr | 1e-6 | 1e-8 | 1e-6 | 1e-6 |
| min_lr | 1e-6 | 1e-5 | 1.5e-4 | 1.5e-3 |
| warmup_epochs | 5 |  |  |  |
| augmentations |  |  |  |  |
| color_jitter | 0.4 |  |  |  |
| horizontal_flip | True | True | False | False |
| num_sample | 2 |  |  |  |
| aa | rand-m7-n4-mstd0.5-inc1 |  |  |  |
| smoothing | 0.1 |  |  |  |
| train_interpolation | bicubic |  |  |  |
| test_num_segment | 5 | 5 | 2 | 2 |
| test_num_crop | 3 | 3 | 3 | 3 |
| erase |  |  |  |  |
| prob | 0.25 |  |  |  |
| mode | pixel |  |  |  |
| count | 1 |  |  |  |
| split | False |  |  |  |
| mixup |  |  |  |  |
| mixup | 0.8 |  |  |  |
| cutmix | 1.0 |  |  |  |
| mixup_prob | 1.0 |  |  |  |
| mixup_switch_prob | 0.5 |  |  |  |
| mixup_mode | batch |  |  |  |

## Appendix E Extra Results

### E.1 Frozen Evaluation.

|  |  | K400 | SSv2 |  |  |
|---|---|---|---|---|---|
| Method | Arch. | Lin. | Att. | Lin. | Att. |
| VideoMAE | ViT-L/16 | 52.5 | 77.8 | 41.3 | 61.2 |
| V-JEPA | ViT-L/16 | 56.7 | 80.8 | 50.1 | 69.5 |

|  |  | K400 | SSv2 | IN1K | Place205 | iNat21 |  |  |  |  |  |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Method | Arch. | Lin. | Att. | Lin. | Att. | Lin. | Att. | Lin. | Att. | Lin. | Att. |
| DINOv2 | ViT-g/14 | 78.4 | 83.4 | 38.3 | 50.0 | 86.5 | 86.2 | 67.5 | 68.4 | 85.7 | 88.8 |
| OpenCLIP | ViT-G/14 | 78.3 | 81.8 | 35.8 | 34.8 | 86.2 | 85.3 | 69.8 | 70.2 | 76.0 | 83.6 |

Table 12 shows that V-JEPA and VideoMAE benefit from using a non-linear attentive probe and multiple clips on the K400 and SSv2 downstream tasks. Additionally, Table 13 shows that attentive probing leads to better performance on average for DINOv2 and OpenCLIP models. Since attentive probing and multiclips eval improves the performance of all models, we use it as our default protocol in frozen evaluation.

| Method | Arch. | 1 Clip | 8 Clips |
|---|---|---|---|
| VideoMAE | ViT-L/16 | 69.4 | 77.8 |
| V-JEPA | ViT-L/16 | 73.7 | 80.9 |

We examine the impact of changing the temporal coverage of a model during downstream evaluation on K400 action classification. In Table 14, we evaluate VideoMAE and V-JEPA models using an attentive probe with access to either the feature map of 1 clip randomly sampled from the video, or the concatenated feature map of 8 clips randomly sampled from the video. To sample 8 clips from a video, we first divide the video into 8 equal length temporal segments, and sample 1 clip at random from each segment. A single clip corresponds to ≈\approx 2 seconds of a video on average, while 8 clips correspond to ≈\approx 16 seconds. The video encoders processes each clip separately to produce a clip-level feature map, which are then concatenated at the input to the attentive probe.

Increasing the temporal coverage from 1 clip per video to 8 clips improves the performance of both V-JEPA and VideoMAE on K400 action classification. We therefore use the multiclip attentive probing setup as our default evaluation pipeline.

| Method | Arch. | Pretraining Data | #Samples Seen | K400 | SSv2 |
|---|---|---|---|---|---|
| (16×\times5×\times3) | (16×\times2×\times3) |  |  |  |  |
| VideoMAEv1 | ViT-L/16 | K400|\lvertSSv2 | 380M|\lvert410M | 85.4 | 74.3 |
| ViT-H/16 | K400|\lvertSSv2 | 380M|\lvert410M | 86.6 | 74.8 |  |
| VideoMAEv2 | ViT-H/16 | Un.Hybrid | 1600M | 86.9 | 76.8 |
| MVD | ViT-L/16 | K400+IN1K | 2400M | 86.4 | 76.7 |
| ViT-H/16 | K400+IN1K | 2400M | 87.2 | 77.3 |  |
| V-JEPA | ViT-L/16 | VideoMix2M | 270M | 85.6 | 75.1 |
| ViT-H/16 | VideoMix2M | 270M | 86.6 | 77.0 |  |

### E.2 Finetuning

In Table 15, we evaluate V-JEPA using finetuning (separately) on K400 and SSv2. We compare V-JEPA with VideoMAEv2 (Wang et al., 2023a), VideoMAE (Tong et al., 2022) and MVD (Wang et al., 2023b) using a ViT-L/16 or a ViT-H/16 architecture. V-JEPA obtains competitive performance using a finetuning protocol. With a ViTiH/16 architecture, V-JEPA outperforms by 1.2%1.2\% VideoMAE and +0.3%+0.3\% VideoMAEv2 on the SSv2 dataset, while obtaining comparable performance on K400. V-JEPA also obtains performance similar to MVD on the SSv2 dataset. The MVD model achieves the best performance across models on the K400 dataset, and is trained using the image dataset ImageNet1K, in contrast to the other methods in the table, which only use video data. Additionally MVD requires the processing of significantly more samples during pretraining due to the cost of training the teacher encoder networks in a pre-pre-training step.

### E.3 Sample Efficiency of pretraining

We compare the sample efficiency of pretraining various state-of-the-art image and video models. Specifically, we look at the number of samples (image or video clips) processed by the network during pretraining, which is larger than the size of the pretraining dataset for multi-epoch training. Notably, our results with V-JEPA are obtained while processing an order of magnitude fewer samples than previous methods, and notably two orders of magnitude fewer samples than OpenCLIP. We believe that further investment towards improving the video pretraining data distribution could lead to substantial gains in downstream image and video tasks.

| Method | Arch. | Data | #Samples Seen |
|---|---|---|---|
| OpenCLIP | ViT-G/14 | LAION-2B | 39000M |
| DINOv2 | ViT-g/14 | LVD 142M | 1900M |
| VideoMAEv2 | ViT-g/14 | UnlabeledHybrid | 1600M |
| V-JEPA | ViT-H/16384 | VideoMix2M | 210M |

### E.4 Masking Strategy

An important component of the V-JEPA pretraining strategy is the 3D clip masking strategy. In this section, we detail 26 ablation experiments exploring different masks. For all the experiments, we pretrain a ViT-B/16 pretrained on K400. Figure 8 presents a summary of those results.

Figure 8(c) shows the effect of changing the spatial and temporal masking ratio. Figure 8(b) ablates the number of sampled blocks used to construct the masks given a fixed effective masking ratio of 90%90\%. Finally, in Figure 8(a) we examine our multi-masking strategy and find that sampling two masks for each clip (long-range and short-range) to be more effective than sampling just a single mask for each clip.

In Figure 8(c), we explore different average spatial and temporal masking ratio, i.e. the spatial/temporal ratio of the area that is covered by a mask on average for a clip. Recall that each mask is constructed by sampling several (possibly overlapping) blocks and taking their union. We change the average spatial or temporal masking ratio by changing a block spatial or temporal size, as well as the overall number of blocks. We found that low spatial or temporal coverage results in a trivial prediction task, which degrades downstream performance. Based on those results, we sample masks that remove roughly 90%90\% of the frame and extend along the entire temporal dimension of the clip by default.

In Figure 8(b) , we explore different block size given an effective spatial masking ratio of 90% and temporal ratio of 100%. We keep the masking ratio approximately constant by changing the block size and the number of block at the same time. We find that sampling several blocks to perform better than sampling a single large block. Figure 9 visually illustrates the effect of sampling several smaller blocks to construct a mask.

In Figure 8(a), we explore the effect of sampling various number of masks per samples. We find that sampling two masks for each clip, with different spatial block sizes for each, to be more effective than sampling just a single mask. We hypothesize that this masking strategy induces complementary tasks. In our experiment, we use this as our default masks sampling.
