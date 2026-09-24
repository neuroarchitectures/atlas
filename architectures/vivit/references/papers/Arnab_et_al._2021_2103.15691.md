# ViViT: A Video Vision Transformer

We present pure-transformer based models for video classification, drawing upon the recent success of such models in image classification. Our model extracts spatio-temporal tokens from the input video, which are then encoded by a series of transformer layers. In order to handle the long sequences of tokens encountered in video, we propose several, efficient variants of our model which factorise the spatial- and temporal-dimensions of the input. Although transformer-based models are known to only be effective when large training datasets are available, we show how we can effectively regularise the model during training and leverage pretrained image models to be able to train on comparatively small datasets. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple video classification benchmarks including Kinetics 400 and 600, Epic Kitchens, Something-Something v2 and Moments in Time, outperforming prior methods based on deep 3D convolutional networks. To facilitate further research, we release code at https://github.com/google-research/scenic.

## 1 Introduction

Approaches based on deep convolutional neural networks have advanced the state-of-the-art across many standard datasets for vision problems since AlexNet [38]. At the same time, the most prominent architecture of choice in sequence-to-sequence modelling (e.g. in natural language processing) is the transformer [68], which does not use convolutions, but is based on multi-headed self-attention. This operation is particularly effective at modelling long-range dependencies and allows the model to attend over all elements in the input sequence. This is in stark contrast to convolutions where the corresponding “receptive field” is limited, and grows linearly with the depth of the network.

The success of attention-based models in NLP has recently inspired approaches in computer vision to integrate transformers into CNNs [75, 7], as well as some attempts to replace convolutions completely [49, 3, 53]. However, it is only very recently with the Vision Transformer (ViT) [18], that a pure-transformer based architecture has outperformed its convolutional counterparts in image classification. Dosovitskiy et al. [18] closely followed the original transformer architecture of [68], and noticed that its main benefits were observed at large scale – as transformers lack some of the inductive biases of convolutions (such as translational equivariance), they seem to require more data [18] or stronger regularisation [64].

Inspired by ViT, and the fact that attention-based architectures are an intuitive choice for modelling long-range contextual relationships in video, we develop several transformer-based models for video classification. Currently, the most performant models are based on deep 3D convolutional architectures [8, 20, 21] which were a natural extension of image classification CNNs [27, 60]. Recently, these models were augmented by incorporating self-attention into their later layers to better capture long-range dependencies [75, 23, 79, 1].

As shown in Fig. 1, we propose pure-transformer models for video classification. The main operation performed in this architecture is self-attention, and it is computed on a sequence of spatio-temporal tokens that we extract from the input video. To effectively process the large number of spatio-temporal tokens that may be encountered in video, we present several methods of factorising our model along spatial and temporal dimensions to increase efficiency and scalability. Furthermore, to train our model effectively on smaller datasets, we show how to reguliarise our model during training and leverage pretrained image models.

We also note that convolutional models have been developed by the community for several years, and there are thus many “best practices” associated with such models. As pure-transformer models present different characteristics, we need to determine the best design choices for such architectures. We conduct a thorough ablation analysis of tokenisation strategies, model architecture and regularisation methods. Informed by this analysis, we achieve state-of-the-art results on multiple standard video classification benchmarks, including Kinetics 400 and 600 [35], Epic Kitchens 100 [13], Something-Something v2 [26] and Moments in Time [45].

## 2 Related Work

Architectures for video understanding have mirrored advances in image recognition. Early video research used hand-crafted features to encode appearance and motion information [41, 69]. The success of AlexNet on ImageNet [38, 16] initially led to the repurposing of 2D image convolutional networks (CNNs) for video as “two-stream” networks [34, 56, 47]. These models processed RGB frames and optical flow images independently before fusing them at the end. Availability of larger video classification datasets such as Kinetics [35] subsequently facilitated the training of spatio-temporal 3D CNNs [8, 22, 65] which have significantly more parameters and thus require larger training datasets. As 3D convolutional networks require significantly more computation than their image counterparts, many architectures factorise convolutions across spatial and temporal dimensions and/or use grouped convolutions [59, 66, 67, 81, 20]. We also leverage factorisation of the spatial and temporal dimensions of videos to increase efficiency, but in the context of transformer-based models.

Concurrently, in natural language processing (NLP), Vaswani et al. [68] achieved state-of-the-art results by replacing convolutions and recurrent networks with the transformer network that consisted only of self-attention, layer normalisation and multilayer perceptron (MLP) operations. Current state-of-the-art architectures in NLP [17, 52] remain transformer-based, and have been scaled to web-scale datasets [5]. Many variants of the transformer have also been proposed to reduce the computational cost of self-attention when processing longer sequences [10, 11, 37, 62, 63, 73] and to improve parameter efficiency [40, 14]. Although self-attention has been employed extensively in computer vision, it has, in contrast, been typically incorporated as a layer at the end or in the later stages of the network [75, 7, 32, 77, 83] or to augment residual blocks [30, 6, 9, 57] within a ResNet architecture [27].

Although previous works attempted to replace convolutions in vision architectures [49, 53, 55], it is only very recently that Dosovitisky et al. [18] showed with their ViT architecture that pure-transformer networks, similar to those employed in NLP, can achieve state-of-the-art results for image classification too. The authors showed that such models are only effective at large scale, as transformers lack some of inductive biases of convolutional networks (such as translational equivariance), and thus require datasets larger than the common ImageNet ILSRVC dataset [16] to train. ViT has inspired a large amount of follow-up work in the community, and we note that there are a number of concurrent approaches on extending it to other tasks in computer vision [71, 74, 84, 85] and improving its data-efficiency [64, 48]. In particular, [4, 46] have also proposed transformer-based models for video.

In this paper, we develop pure-transformer architectures for video classification. We propose several variants of our model, including those that are more efficient by factorising the spatial and temporal dimensions of the input video. We also show how additional regularisation and pretrained models can be used to combat the fact that video datasets are not as large as their image counterparts that ViT was originally trained on. Furthermore, we outperform the state-of-the-art across five popular datasets.

## 3 Video Vision Transformers

We start by summarising the recently proposed Vision Transformer [18] in Sec. 3.1, and then discuss two approaches for extracting tokens from video in Sec. 3.2. Finally, we develop several transformer-based architectures for video classification in Sec. 3.3 and 3.4.

### 3.1 Overview of Vision Transformers (ViT)

Vision Transformer (ViT) [18] adapts the transformer architecture of [68] to process 2D images with minimal changes. In particular, ViT extracts NN non-overlapping image patches, xi∈ℝh×wx_{i}\in\mathbb{R}^{h\times w}, performs a linear projection and then rasterises them into 1D tokens zi∈ℝdz_{i}\in\mathbb{R}^{d}. The sequence of tokens input to the following transformer encoder is

|  | 𝐳=[zc​l​s,𝐄​x1,𝐄​x2,…,𝐄​xN]+𝐩,\mathbf{z}=[z_{cls},\mathbf{E}x_{1},\mathbf{E}x_{2},\ldots,\mathbf{E}x_{N}]+\mathbf{p}, |  | (1) |
|---|---|---|---|

where the projection by 𝐄\mathbf{E} is equivalent to a 2D convolution. As shown in Fig. 1, an optional learned classification token zc​l​sz_{cls} is prepended to this sequence, and its representation at the final layer of the encoder serves as the final representation used by the classification layer [17]. In addition, a learned positional embedding, 𝐩∈ℝN×d\mathbf{p}\in\mathbb{R}^{N\times d}, is added to the tokens to retain positional information, as the subsequent self-attention operations in the transformer are permutation invariant. The tokens are then passed through an encoder consisting of a sequence of LL transformer layers. Each layer ℓ\ell comprises of Multi-Headed Self-Attention [68], layer normalisation (LN) [2], and MLP blocks as follows:

|  | 𝐲ℓ\displaystyle\mathbf{y}^{\ell} | =MSA​(LN​(𝐳ℓ))+𝐳ℓ\displaystyle=\text{MSA}(\text{LN}(\mathbf{z}^{\ell}))+\mathbf{z}^{\ell} |  | (2) |
|---|---|---|---|---|
|  | 𝐳ℓ+1\displaystyle\mathbf{z}^{\ell+1} | =MLP​(LN​(𝐲ℓ))+𝐲ℓ.\displaystyle=\text{MLP}(\text{LN}(\mathbf{y}^{\ell}))+\mathbf{y}^{\ell}. |  | (3) |

The MLP consists of two linear projections separated by a GELU non-linearity [28] and the token-dimensionality, dd, remains fixed throughout all layers. Finally, a linear classifier is used to classify the encoded input based on zc​l​sL∈ℝdz_{cls}^{L}\in\mathbb{R}^{d}, if it was prepended to the input, or a global average pooling of all the tokens, 𝐳L\mathbf{z}^{L}, otherwise.

As the transformer [68], which forms the basis of ViT [18], is a flexible architecture that can operate on any sequence of input tokens 𝐳∈ℝN×d\mathbf{z}\in\mathbb{R}^{N\times d}, we describe strategies for tokenising videos next.

### 3.2 Embedding video clips

We consider two simple methods for mapping a video 𝐕∈ℝT×H×W×C\mathbf{V}\in\mathbb{R}^{T\times H\times W\times C} to a sequence of tokens 𝐳~∈ℝnt×nh×nw×d\mathbf{\tilde{z}}\in\mathbb{R}^{n_{t}\times n_{h}\times n_{w}\times d}. We then add the positional embedding and reshape into ℝN×d\mathbb{R}^{N\times d} to obtain 𝐳\mathbf{z}, the input to the transformer.

As illustrated in Fig. 2, a straightforward method of tokenising the input video is to uniformly sample ntn_{t} frames from the input video clip, embed each 2D frame independently using the same method as ViT [18], and concatenate all these tokens together. Concretely, if nh⋅nwn_{h}\cdot n_{w} non-overlapping image patches are extracted from each frame, as in [18], then a total of nt⋅nh⋅nwn_{t}\cdot n_{h}\cdot n_{w} tokens will be forwarded through the transformer encoder. Intuitively, this process may be seen as simply constructing a large 2D image to be tokenised following ViT. We note that this is the input embedding method employed by the concurrent work of [4].

An alternate method, as shown in Fig. 3, is to extract non-overlapping, spatio-temporal “tubes” from the input volume, and to linearly project this to ℝd\mathbb{R}^{d}. This method is an extension of ViT’s embedding to 3D, and corresponds to a 3D convolution. For a tubelet of dimension t×h×wt\times h\times w, nt=⌊Tt⌋n_{t}=\lfloor\frac{T}{t}\rfloor, nh=⌊Hh⌋n_{h}=\lfloor\frac{H}{h}\rfloor and nw=⌊Ww⌋n_{w}=\lfloor\frac{W}{w}\rfloor, tokens are extracted from the temporal, height, and width dimensions respectively. Smaller tubelet dimensions thus result in more tokens which increases the computation. Intuitively, this method fuses spatio-temporal information during tokenisation, in contrast to “Uniform frame sampling” where temporal information from different frames is fused by the transformer.

### 3.3 Transformer Models for Video

As illustrated in Fig. 1, we propose multiple transformer-based architectures. We begin with a straightforward extension of ViT [18] that models pairwise interactions between all spatio-temporal tokens, and then develop more efficient variants which factorise the spatial and temporal dimensions of the input video at various levels of the transformer architecture.

This model simply forwards all spatio-temporal tokens extracted from the video, 𝐳0\mathbf{z}^{0}, through the transformer encoder. We note that this has also been explored concurrently by [4] in their “Joint Space-Time” model. In contrast to CNN architectures, where the receptive field grows linearly with the number of layers, each transformer layer models all pairwise interactions between all spatio-temporal tokens, and it thus models long-range interactions across the video from the first layer. However, as it models all pairwise interactions, Multi-Headed Self Attention (MSA) [68] has quadratic complexity with respect to the number of tokens. This complexity is pertinent for video, as the number of tokens increases linearly with the number of input frames, and motivates the development of more efficient architectures next.

As shown in Fig. 4, this model consists of two separate transformer encoders. The first, spatial encoder, only models interactions between tokens extracted from the same temporal index. A representation for each temporal index, hi∈ℝdh_{i}\in\mathbb{R}^{d}, is obtained after LsL_{s} layers: This is the encoded classification token, zc​l​sLsz_{cls}^{L_{s}} if it was prepended to the input (Eq. 1), or a global average pooling from the tokens output by the spatial encoder, 𝐳Ls\mathbf{z}^{L_{s}}, otherwise. The frame-level representations, hih_{i}, are concatenated into 𝐇∈ℝnt×d\mathbf{H}\in\mathbb{R}^{n_{t}\times d}, and then forwarded through a temporal encoder consisting of LtL_{t} transformer layers to model interactions between tokens from different temporal indices. The output token of this encoder is then finally classified.

This architecture corresponds to a “late fusion” [34, 56, 72, 46] of temporal information, and the initial spatial encoder is identical to the one used for image classification. It is thus analogous to CNN architectures such as [24, 34, 72, 86] which first extract per-frame features, and then aggregate them into a final representation before classifying them. Although this model has more transformer layers than Model 1 (and thus more parameters), it requires fewer floating point operations (FLOPs), as the two separate transformer blocks have a complexity of 𝒪⁡((nh⋅nw)2+nt2)\mathcal{O}({(n_{h}\cdot n_{w})^{2}+n_{t}^{2})} compared to 𝒪⁡((nt⋅nh⋅nw)2)\mathcal{O}((n_{t}\cdot n_{h}\cdot n_{w})^{2}) of Model 1.

This model, in contrast, contains the same number of transformer layers as Model 1. However, instead of computing multi-headed self-attention across all pairs of tokens, 𝐳ℓ\mathbf{z}^{\ell}, at layer ll, we factorise the operation to first only compute self-attention spatially (among all tokens extracted from the same temporal index), and then temporally (among all tokens extracted from the same spatial index) as shown in Fig. 5. Each self-attention block in the transformer thus models spatio-temporal interactions, but does so more efficiently than Model 1 by factorising the operation over two smaller sets of elements, thus achieving the same computational complexity as Model 2. We note that factorising attention over input dimensions has also been explored in [29, 78], and concurrently in the context of video by [4] in their “Divided Space-Time” model.

This operation can be performed efficiently by reshaping the tokens 𝐳\mathbf{z} from ℝ1×nt⋅nh⋅nw⋅d\mathbb{R}^{1\times n_{t}\cdot n_{h}\cdot n_{w}\cdot d} to ℝnt×nh⋅nw⋅d\mathbb{R}^{n_{t}\times n_{h}\cdot n_{w}\cdot d} (denoted by 𝐳s\mathbf{z}_{s}) to compute spatial self-attention. Similarly, the input to temporal self-attention, 𝐳t\mathbf{z}_{t} is reshaped to ℝnh⋅nw×nt⋅d\mathbb{R}^{n_{h}\cdot n_{w}\times n_{t}\cdot d}. Here we assume the leading dimension is the “batch dimension”. Our factorised self-attention is defined as

|  | 𝐲sℓ\displaystyle\mathbf{y}_{s}^{\ell} | =MSA⁡(LN⁡(𝐳sℓ))+𝐳sℓ\displaystyle=\msa(\lnorm(\mathbf{z}^{\ell}_{s}))+\mathbf{z}^{\ell}_{s} |  | (4) |
|---|---|---|---|---|
|  | 𝐲tℓ\displaystyle\mathbf{y}_{t}^{\ell} | =MSA⁡(LN⁡(𝐲sℓ))+𝐲sℓ\displaystyle=\msa(\lnorm(\mathbf{y}^{\ell}_{s}))+\mathbf{y}^{\ell}_{s} |  | (5) |
|  | 𝐳ℓ+1\displaystyle\mathbf{z}^{\ell+1} | =MLP⁡(LN⁡(𝐲tℓ))+𝐲tℓ.\displaystyle=\mlp(\lnorm(\mathbf{y}_{t}^{\ell}))+\mathbf{y}_{t}^{\ell}. |  | (6) |

We observed that the order of spatial-then-temporal self-attention or temporal-then-spatial self-attention does not make a difference, provided that the model parameters are initialised as described in Sec. 3.4. Note that the number of parameters, however, increases compared to Model 1, as there is an additional self-attention layer (cf. Eq. 7). We do not use a classification token in this model, to avoid ambiguities when reshaping the input tokens between spatial and temporal dimensions.

Finally, we develop a model which has the same computational complexity as Models 2 and 3, while retaining the same number of parameters as the unfactorised Model 1. The factorisation of spatial- and temporal dimensions is similar in spirit to Model 3, but we factorise the multi-head dot-product attention operation instead (Fig. 6). Concretely, we compute attention weights for each token separately over the spatial- and temporal-dimensions using different heads. First, we note that the attention operation for each head is defined as

|  | Attention⁡(𝐐,𝐊,𝐕)=Softmax⁡(𝐐𝐊⊤dk)​𝐕.\displaystyle\attention(\mathbf{Q},\mathbf{K},\mathbf{V})=\softmax\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_{k}}}\right)\mathbf{V}. |  | (7) |
|---|---|---|---|

In self-attention, the queries 𝐐=𝐗𝐖q\mathbf{Q}=\mathbf{X}\mathbf{W}_{q}, keys 𝐊=𝐗𝐖k\mathbf{K}=\mathbf{X}\mathbf{W}_{k}, and values 𝐕=𝐗𝐖v\mathbf{V}=\mathbf{X}\mathbf{W}_{v} are linear projections of the input 𝐗\mathbf{X} with 𝐗,𝐐,𝐊,𝐕∈ℝN×d\mathbf{X},\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}. Note that in the unfactorised case (Model 1), the spatial and temporal dimensions are merged as N=nt⋅nh⋅nwN=n_{t}\cdot n_{h}\cdot n_{w}.

The main idea here is to modify the keys and values for each query to only attend over tokens from the same spatial- and temporal index by constructing 𝐊s,𝐕s∈ℝnh⋅nw×d\mathbf{K}_{s},\mathbf{V}_{s}\in\mathbb{R}^{n_{h}\cdot n_{w}\times d} and 𝐊t,𝐕t∈ℝnt×d\mathbf{K}_{t},\mathbf{V}_{t}\in\mathbb{R}^{n_{t}\times d}, namely the keys and values corresponding to these dimensions. Then, for half of the attention heads, we attend over tokens from the spatial dimension by computing 𝐘s=Attention⁡(𝐐,𝐊s,𝐕s)\mathbf{Y}_{s}=\attention(\mathbf{Q},\mathbf{K}_{s},\mathbf{V}_{s}), and for the rest we attend over the temporal dimension by computing 𝐘t=Attention⁡(𝐐,𝐊t,𝐕t)\mathbf{Y}_{t}=\attention(\mathbf{Q},\mathbf{K}_{t},\mathbf{V}_{t}). Given that we are only changing the attention neighbourhood for each query, the attention operation has the same dimension as in the unfactorised case, namely 𝐘s,𝐘t∈ℝN×d\mathbf{Y}_{s},\mathbf{Y}_{t}\in\mathbb{R}^{N\times d}. We then combine the outputs of multiple heads by concatenating them and using a linear projection [68], 𝐘=Concat⁡(𝐘s,𝐘t)​𝐖O\mathbf{Y}=\concat(\mathbf{Y}_{s},\mathbf{Y}_{t})\mathbf{W}_{O}.

### 3.4 Initialisation by leveraging pretrained models

ViT [18] has been shown to only be effective when trained on large-scale datasets, as transformers lack some of the inductive biases of convolutional networks [18]. However, even the largest video datasets such as Kinetics [35], have several orders of magnitude less labelled examples when compared to their image counterparts [16, 39, 58]. As a result, training large models from scratch to high accuracy is extremely challenging. To sidestep this issue, and enable more efficient training we initialise our video models from pretrained image models. However, this raises several practical questions, specifically on how to initialise parameters not present or incompatible with image models. We now discuss several effective strategies to initialise these large-scale video classification models.

A positional embedding 𝐩\mathbf{p} is added to each input token (Eq. 1). However, our video models have ntn_{t} times more tokens than the pretrained image model. As a result, we initialise the positional embeddings by “repeating” them temporally from ℝnw⋅nh×d\mathbb{R}^{n_{w}\cdot n_{h}\times d} to ℝnt⋅nh⋅nw×d\mathbb{R}^{n_{t}\cdot n_{h}\cdot n_{w}\times d}. Therefore, at initialisation, all tokens with the same spatial index have the same embedding which is then fine-tuned.

When using the “tubelet embedding” tokenisation method (Sec. 3.2), the embedding filter 𝐄\mathbf{E} is a 3D tensor, compared to the 2D tensor in the pretrained model, 𝐄image\mathbf{E}_{\text{image}}. A common approach for initialising 3D convolutional filters from 2D filters for video classification is to “inflate” them by replicating the filters along the temporal dimension and averaging them [8, 22] as

|  | 𝐄=1t​[𝐄image,…,𝐄image,…,𝐄image].\mathbf{E}=\frac{1}{t}[\mathbf{E}_{\text{image}},\ldots,\mathbf{E}_{\text{image}},\ldots,\mathbf{E}_{\text{image}}]. |  | (8) |
|---|---|---|---|

We consider an additional strategy, which we denote as “central frame initialisation”, where 𝐄\mathbf{E} is initialised with zeroes along all temporal positions, except at the centre ⌊t2⌋\lfloor\frac{t}{2}\rfloor,

|  | 𝐄=[𝟎,…,𝐄image,…,𝟎].\mathbf{E}=[\mathbf{0},\ldots,\mathbf{E}_{\text{image}},\ldots,\mathbf{0}]. |  | (9) |
|---|---|---|---|

Therefore, the 3D convolutional filter effectively behaves like “Uniform frame sampling” (Sec. 3.2) at initialisation, while also enabling the model to learn to aggregate temporal information from multiple frames as training progresses.

The transformer block in Model 3 (Fig. 5) differs from the pretrained ViT model [18], in that it contains two multi-headed self attention (MSA) modules. In this case, we initialise the spatial MSA module from the pretrained module, and initialise all weights of the temporal MSA with zeroes, such that Eq. 5 behaves as a residual connection [27] at initialisation.

## 4 Empirical evaluation

We first present our experimental setup and implementation details in Sec. 4.1, before ablating various components of our model in Sec. 4.2. We then present state-of-the-art results on five datasets in Sec. 4.3.

### 4.1 Experimental Setup

Our backbone architecture follows that of ViT [18] and BERT [17]. We consider ViT-Base (ViT-B, LL=1212, NHN_{H}=1212, dd=768768), ViT-Large (ViT-L, LL=2424, NHN_{H}=1616, dd=10241024), and ViT-Huge (ViT-H, LL=3232, NHN_{H}=1616, dd=12801280), where LL is the number of transformer layers, each with a self-attention block of NHN_{H} heads and hidden dimension dd. We also apply the same naming scheme to our models (e.g., ViViT-B/16x2 denotes a ViT-Base backbone with a tubelet size of h×w×t=16×16×2h\times w\times t=16\times 16\times 2). In all experiments, the tubelet height and width are equal. Note that smaller tubelet sizes correspond to more tokens at the input, and thus more computation.

We train our models using synchronous SGD and momentum, a cosine learning rate schedule and TPU-v3 accelerators. We initialise our models from a ViT image model trained either on ImageNet-21K [16] (unless otherwise specified) or the larger JFT [58] dataset. We implement our method using the Scenic library [15] and have released our code and models.

We evaluate the performance of our proposed models on a diverse set of video classification datasets:

Kinetics [35] consists of 10-second videos sampled at 25fps from YouTube. We evaluate on both Kinetics 400 and 600, containing 400 and 600 classes respectively. As these are dynamic datasets (videos may be removed from YouTube), we note our dataset sizes are approximately 267 000 and 446 000 respectively.

Epic Kitchens-100 consists of egocentric videos capturing daily kitchen activities spanning 100 hours and 90 000 clips [13]. We report results following the standard “action recognition” protocol. Here, each video is labelled with a “verb” and a “noun” and we therefore predict both categories using a single network with two “heads”. The top-scoring verb and action pair predicted by the network form an “action”, and action accuracy is the primary metric.

Moments in Time [45] consists of 800 000, 3-second YouTube clips that capture the gist of a dynamic scene involving animals, objects, people, or natural phenomena.

Something-Something v2 (SSv2) [26] contains 220 000 videos, with durations ranging from 2 to 6 seconds. In contrast to the other datasets, the objects and backgrounds in the videos are consistent across different action classes, and this dataset thus places more emphasis on a model’s ability to recognise fine-grained motion cues.

|  | Top-1 accuracy |
|---|---|
| Uniform frame sampling | 78.5 |
| Tubelet embedding |  |
| Random initialisation [25] | 73.2 |
| Filter inflation [8] | 77.6 |
| Central frame | 79.2 |

The input to our network is a video clip of 32 frames using a stride of 2, unless otherwise mentioned, similar to [21, 20]. Following common practice, at inference time, we process multiple views of a longer video and average per-view logits to obtain the final result. Unless otherwise specified, we use a total of 4 views per video (as this is sufficient to “see” the entire video clip across the various datasets), and ablate these and other design choices next.

### 4.2 Ablation study

We first consider the effect of different input encoding methods (Sec. 3.2) using our unfactorised model (Model 1) and ViViT-B on Kinetics 400. As we pass 32-frame inputs to the network, sampling 8 frames and extracting tubelets of length t=4t=4 correspond to the same number of tokens in both cases. Table 1 shows that tubelet embedding initialised using the “central frame” method (Eq. 9) performs well, outperforming the commonly-used “filter inflation” initialisation method [8, 22] by 1.6%, and “uniform frame sampling” by 0.7%. We therefore use this encoding method for all subsequent experiments.

|  | K400 | EK | FLOPs (×109\times 10^{9}) | Params (×106\times 10^{6}) | Runtime (ms) |
|---|---|---|---|---|---|
| Model 1: Spatio-temporal | 80.0 | 43.1 | 455.2 | 88.9 | 58.9 |
| Model 2: Fact. encoder | 78.8 | 43.7 | 284.4 | 115.1 | 17.4 |
| Model 3: Fact. self-attention | 77.4 | 39.1 | 372.3 | 117.3 | 31.7 |
| Model 4: Fact. dot product | 76.3 | 39.5 | 277.1 | 88.9 | 22.9 |
| Model 2: Ave. pool baseline | 75.8 | 38.8 | 283.9 | 86.7 | 17.3 |

| LtL_{t} | 0 | 1 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| Top-1 | 75.8 | 78.6 | 78.8 | 78.8 | 78.9 |

We compare our proposed model variants (Sec. 3.3) across the Kinetics 400 and Epic Kitchens datasets, both in terms of accuracy and efficiency, in Tab. 2. In all cases, we use the “Base” backbone and tubelet size of 16×216\times 2. Model 2 (“Factorised Encoder”) has an additional hyperparameter, the number of temporal transformers, LtL_{t}. We set Lt=4L_{t}=4 for all experiments and show in Tab. 3 that the model is not sensitive to this choice.

The unfactorised model (Model 1) performs the best on Kinetics 400. However, it can also overfit on smaller datasets such as Epic Kitchens, where we find our “Factorised Encoder” (Model 2) to perform the best. We also consider an additional baseline (last row), based on Model 2, where we do not use any temporal transformer, and simply average pool the frame-level representations from the spatial encoder before classifying. This average pooling baseline performs the worst, and has a larger accuracy drop on Epic Kitchens, suggesting that this dataset requires more detailed modelling of temporal relations.

As described in Sec. 3.3, all factorised variants of our model use significantly fewer FLOPs than the unfactorised Model 1, as the attention is computed separately over spatial- and temporal-dimensions. Model 4 adds no additional parameters to the unfactorised Model 1, and uses the least compute. The temporal transformer encoder in Model 2 operates on only ntn_{t} tokens, which is why there is a barely a change in compute and runtime over the average pooling baseline, even though it improves the accuracy substantially (3% on Kinetics and 4.9% on Epic Kitchens). Finally, Model 3 requires more compute and parameters than the other factorised models, as its additional self-attention block means that it performs another query-, key-, value- and output-projection in each transformer layer [68].

|  | Top-1 accuracy |
|---|---|
| Random crop, flip, colour jitter | 38.4 |
| + Kinetics 400 initialisation | 39.6 |
| + Stochastic depth [31] | 40.2 |
| + Random augment [12] | 41.1 |
| + Label smoothing [61] | 43.1 |
| + Mixup [82] | 43.7 |

| (a) Accuracy | (b) Compute |
|---|---|

| (a) Accuracy | (b) Compute |
|---|---|

Pure-transformer architectures such as ViT [18] are known to require large training datasets, and we observed overfitting on smaller datasets like Epic Kitchens and SSv2, even when using an ImageNet pretrained model. In order to effectively train our models on such datasets, we employed several regularisation strategies that we ablate using our “Factorised encoder” model in Tab. 4. We note that these regularisers were originally proposed for training CNNs, and that [64] have recently explored them for training ViT for image classification.

Each row of Tab. 4 includes all the methods from the rows above it, and we observe progressive improvements from adding each regulariser. Overall, we obtain a substantial overall improvement of 5.3% on Epic Kitchens. We also achieve a similar improvement of 5% on SSv2 by using all the regularisation in Tab. 4. Note that the Kinetics-pretrained models that we initialise from are from Tab. 2, and that all Epic Kitchens models in Tab. 2 were trained with all the regularisers in Tab. 4. For larger datasets like Kinetics and Moments in Time, we do not use these additional regularisers (we use only the first row of Tab. 4), as we obtain state-of-the-art results without them. The appendix contains hyperparameter values and additional details for all regularisers.

Figure 7 compares the ViViT-B and ViViT-L backbones for the unfactorised spatio-temporal model. We observe consistent improvements in accuracy as the backbone capacity increases. As expected, the compute also grows as a function of the backbone size.

| Crop size | 224 | 288 | 320 |
|---|---|---|---|
| Accuracy | 80.3 | 80.7 | 81.0 |
| GFLOPs | 1446 | 2919 | 3992 |
| Runtime | 58.9 | 147.6 | 238.8 |

We first analyse the performance as a function of the number of tokens along the temporal dimension in Fig. 8. We observe that using smaller input tubelet sizes (and therefore more tokens) leads to consistent accuracy improvements across all of our model architectures. At the same time, computation in terms of FLOPs increases accordingly, and the unfactorised model (Model 1) is impacted the most.

We then vary the number of tokens fed into the model by increasing the spatial crop-size from the default of 224 to 320 in Tab. 5. As expected, there is a consistent increase in both accuracy and computation. We note that when comparing to prior work we consistently obtain state-of-the-art results (Sec. 4.3) using a spatial resolution of 224, but we also highlight that further improvements can be obtained at higher spatial resolutions.

| Method | Top 1 | Top 5 | Views | TFLOPs |
|---|---|---|---|---|
| blVNet [19] | 73.5 | 91.2 | – | – |
| STM [33] | 73.7 | 91.6 | – | – |
| TEA [42] | 76.1 | 92.5 | 10×310\times 3 | 2.10 |
| TSM-ResNeXt-101 [43] | 76.3 | – | – | – |
| I3D NL [75] | 77.7 | 93.3 | 10×310\times 3 | 10.77 |
| CorrNet-101 [70] | 79.2 | – | 10×310\times 3 | 6.72 |
| ip-CSN-152 [66] | 79.2 | 93.8 | 10×310\times 3 | 3.27 |
| LGD-3D R101 [51] | 79.4 | 94.4 | – | – |
| SlowFast R101-NL [21] | 79.8 | 93.9 | 10×310\times 3 | 7.02 |
| X3D-XXL [20] | 80.4 | 94.6 | 10×310\times 3 | 5.82 |
| TimeSformer-L [4] | 80.7 | 94.7 | 1×31\times 3 | 7.14 |
| ViViT-L/16x2 FE | 80.6 | 92.7 | 1×11\times 1 | 3.98 |
| ViViT-L/16x2 FE | 81.7 | 93.8 | 1×31\times 3 | 11.94 |
| Methods with large-scale pretraining |  |  |  |  |
| ip-CSN-152 [66] (IG [44]) | 82.5 | 95.3 | 10×310\times 3 | 3.27 |
| ViViT-L/16x2 FE (JFT) | 83.5 | 94.3 | 1×31\times 3 | 11.94 |
| ViViT-H/14x2 (JFT) | 84.9 | 95.8 | 4×34\times 3 | 47.77 |

| Method | Top 1 | Top 5 |
|---|---|---|
| AttentionNAS [76] | 79.8 | 94.4 |
| LGD-3D R101 [51] | 81.5 | 95.6 |
| SlowFast R101-NL [21] | 81.8 | 95.1 |
| X3D-XL [20] | 81.9 | 95.5 |
| TimeSformer-L [4] | 82.2 | 95.6 |
| ViViT-L/16x2 FE | 82.9 | 94.6 |
| ViViT-L/16x2 FE (JFT) | 84.3 | 94.9 |
| ViViT-H/14x2 (JFT) | 85.8 | 96.5 |

|  | Top 1 | Top 5 |
|---|---|---|
| TSN [72] | 25.3 | 50.1 |
| TRN [86] | 28.3 | 53.4 |
| I3D [8] | 29.5 | 56.1 |
| blVNet [19] | 31.4 | 59.3 |
| AssembleNet-101 [54] | 34.3 | 62.7 |
| ViViT-L/16x2 FE | 38.5 | 64.1 |

| Method | Action | Verb | Noun |
|---|---|---|---|
| TSN [72] | 33.2 | 60.2 | 46.0 |
| TRN [86] | 35.3 | 65.9 | 45.4 |
| TBN [36] | 36.7 | 66.0 | 47.2 |
| TSM [43] | 38.3 | 67.9 | 49.0 |
| SlowFast [21] | 38.5 | 65.6 | 50.0 |
| ViViT-L/16x2 FE | 44.0 | 66.4 | 56.8 |

| Method | Top 1 | Top 5 |
|---|---|---|
| TRN [86] | 48.8 | 77.6 |
| SlowFast [20, 80] | 61.7 | – |
| TimeSformer-HR [4] | 62.5 | – |
| TSM [43] | 63.4 | 88.5 |
| STM [33] | 64.2 | 89.8 |
| TEA [42] | 65.1 | – |
| blVNet [19] | 65.2 | 90.3 |
| ViVIT-L/16x2 FE | 65.9 | 89.9 |

In our experiments so far, we have kept the number of input frames fixed at 32. We now increase the number of frames input to the model, thereby increasing the number of tokens proportionally.

Figure 9 shows that as we increase the number of frames input to the network, the accuracy from processing a single view increases, since the network incorporates longer temporal context. However, common practice on datasets such as Kinetics [21, 75, 42] is to average results over multiple, shorter “views” of the same video clip. Figure 9 also shows that the accuracy saturates once the number of views is sufficient to cover the whole video. As a Kinetics video consists of 250 frames, and we sample frames with a stride of 2, our model which processes 128 frames requires just a single view to “see” the whole video and achieve its maximum accuarcy.

Note that we used ViViT-L/16x2 Factorised Encoder (Model 2) here. As this model is more efficient it can process more tokens, compared to the unfactorised Model 1 which runs out of memory after 48 frames using tubelet length t=2t=2 and a “Large” backbone. Models processing more frames (and thus more tokens) consistently achieve higher single- and multi-view accuracy, in line with our observations in previous experiments (Tab. 5, Fig. 8). Moroever, observe that by processing more frames (and thus more tokens) with Model 2, we are able to achieve higher accuracy than Model 1 (with fewer total FLOPs as well).

Finally, we observed that for Model 2, the number of FLOPs effectively increases linearly with the number of input frames as the overall computation is dominated by the initial Spatial Transformer. As a result, the total number of FLOPs for the number of temporal views required to achieve maximum accuracy is constant across the models. In other words, ViViT-L/16x2 FE with 32 frames requires 995.3 GFLOPs per view, and 4 views to saturate multi-view accuracy. The 128-frame model requires 3980.4 GFLOPs but only a single view. As shown by Fig. 9, the latter model achieves the highest accuracy.

### 4.3 Comparison to state-of-the-art

Based on our ablation studies in the previous section, we compare to the current state-of-the-art using two of our model variants. We primarily use our Factorised Encoder model (Model 2), as it can process more tokens than Model 1 to achieve higher accuracy.

Tables 6(a) and 6(c) show that our spatio-temporal attention models outperform the state-of-the-art on Kinetics 400 and 600 respectively. Following standard practice, we take 3 spatial crops (left, centre and right) [21, 20, 66, 75] for each temporal view, and notably, we require significantly fewer views than previous CNN-based methods.

We surpass the previous CNN-based state-of-the-art using ViViT-L/16x2 Factorised Encoder (FE) pretrained on ImageNet, and also outperform [4] who concurrently proposed a pure-transformer architecture. Moreover, by initialising our backbones from models pretrained on the larger JFT dataset [58], we obtain further improvements. Although these models are not directly comparable to previous work, we do also outperform [66] who pretrained on the large-scale, Instagram dataset [44]. Our best model uses a ViViT-H backbone pretrained on JFT and significantly advances the best reported results on Kinetics 400 and 600 to 84.9% and 85.8%, respectively.

We surpass the state-of-the-art by a significant margin as shown in Tab. 6(c). We note that the videos in this dataset are diverse and contain significant label noise, making this task challenging and leading to lower accuracies than on other datasets.

Table 6(e) shows that our Factorised Encoder model outperforms previous methods by a significant margin. In addition, our model obtains substantial improvements for Top-1 accuracy of “noun” classes, and the only method which achieves higher “verb” accuracy used optical flow as an additional input modality [43, 50]. Furthermore, all variants of our model presented in Tab. 2 outperformed the existing state-of-the-art on action accuracy. We note that we use the same model to predict verbs and nouns using two separate “heads”, and for simplicity, we do not use separate loss weights for each head.

Finally, Tab. 6(e) shows that we achieve state-of-the-art Top-1 accuracy with our Factorised encoder model (Model 2), albeit with a smaller margin compared to previous methods. Notably, our Factorised encoder model significantly outperforms the concurrent TimeSformer [4] method by 2.9%, which also proposes a pure-transformer model, but does not consider our Factorised encoder variant or our additional regularisation.

SSv2 differs from other datasets in that the backgrounds and objects are quite similar across different classes, meaning that recognising fine-grained motion patterns is necessary to distinguish classes from each other. Our results suggest that capturing these fine-grained motions is an area of improvement and future work for our model. We also note an inverse correlation between the relative performance of previous methods on SSv2 (Tab. 6(e)) and Kinetics (Tab. 6(a)) suggesting that these two datasets evaluate complementary characteristics of a model.

## 5 Conclusion and Future Work

We have presented four pure-transformer models for video classification, with different accuracy and efficiency profiles, achieving state-of-the-art results across five popular datasets. Furthermore, we have shown how to effectively regularise such high-capacity models for training on smaller datasets and thoroughly ablated our main design choices. Future work is to remove our dependence on image-pretrained models. Finally, going beyond video classification towards more complex tasks is a clear next step.

## Appendix

## Appendix A Additional experimental details

In this appendix, we provide additional experimental details. Section A.1 provides additional details about the regularisers we used and Sec. A.2 details the training hyperparamters used for our experiments.

|  | K400 | K600 | MiT | EK | SSv2 |
|---|---|---|---|---|---|
| Optimisation |  |  |  |  |  |
| Optimiser | Synchronous SGD |  |  |  |  |
| Momentum | 0.9 |  |  |  |  |
| Batch size | 64 |  |  |  |  |
| Learning rate schedule | cosine with linear warmup |  |  |  |  |
| Linear warmup epochs | 2.5 |  |  |  |  |
| Base learning rate | 0.1 | 0.1 | 0.25 | 0.5 | 0.5 |
| Epochs | 30 | 30 | 10 | 50 | 35 |
| Data augmentation |  |  |  |  |  |
| Random crop probability | 1.0 |  |  |  |  |
| Random flip probability | 0.5 |  |  |  |  |
| Scale jitter probability | 1.0 |  |  |  |  |
| Maximum scale | 1.33 |  |  |  |  |
| Minimum scale | 0.9 |  |  |  |  |
| Colour jitter probability | 0.8 | 0.8 | 0.8 | – | – |
| Rand augment number of layers [12] | – | – | – | 2 | 2 |
| Rand augment magnitude [12] | – | – | – | 15 | 20 |
| Other regularisation |  |  |  |  |  |
| Stochastic droplayer rate, pdropp_{\text{drop}} [31] | – | – | – | 0.2 | 0.3 |
| Label smoothing λ\lambda [61] | – | – | – | 0.2 | 0.3 |
| Mixup α\alpha [82] | – | – | – | 0.1 | 0.3 |

### A.1 Further details about regularisers

In this section, we provide additional details and list the hyperparameters of the additional regularisers that we employed in Tab. 4. Hyperparameter values for all our experiments are listed in Tab. 7.

Stochastic depth regularisation was originally proposed for training very deep residual networks [31]. Intuitively, the outputs of a layer, ℓ\ell, are “dropped out” with probability, pdrop​(ℓ)p_{\text{drop}}(\ell) during training, by setting the output of the layer to be equal to its input.

Following [31], we linearly increase the probability of dropping a layer according to its depth within the network,

|  | pdrop​(ℓ)=ℓL​pdrop,p_{\text{drop}}(\ell)=\frac{\ell}{L}p_{\text{drop}}, |  | (10) |
|---|---|---|---|

where ℓ\ell is the index of the layer in the network, and LL is the total number of layers.

Random augment [12] randomly applies data augmentation transformations sequentially to an input example. We follow the public implementation11 1 https://github.com/tensorflow/models/blob/master/official/vision/beta/ops/augment.py, but modify the data augmentation operations to be temporally consistent throughout the video (in other words, the same transformation is applied on each frame of the video).

The authors define two hyperparameters for Random augment, “number of layers” , the number of augmentation transformations to apply sequentially to a video and “magnitude”, the strength of the transformation that is shared across all augmentation operations. Our values for these parameters are shown in Tab. 7.

Label smoothing was proposed by [61] originally to regularise training Inception-v3. Concretely, the label distribution used during training, y~\tilde{y}, is a mixture of the one-hot ground-truth label, yy, and a uniform distribution, uu, to encourage the network to produce less confident predictions during training:

|  | y~=(1−λ)​y+λ​u.\tilde{y}=(1-\lambda)y+\lambda u. |  | (11) |
|---|---|---|---|

There is therefore one scalar hyperparamter, λ∈[0,1]\lambda\in[0,1].

Mixup [82] constructs virtual training examples which are a convex combination of pairs of training examples and their labels. Concretely, given (xi,yi)(x_{i},y_{i}) and (xj,yj)(x_{j},y_{j}) where xix_{i} denotes an input vector and yiy_{i} a one-hot input label, mixup constructs the virtual training example,

|  | x~\displaystyle\tilde{x} | =λ​xi+(1−λ)​xj\displaystyle=\lambda x_{i}+(1-\lambda)x_{j} |  |  |
|---|---|---|---|---|
|  | y~\displaystyle\tilde{y} | =λ​yi+(1−λ)​yj.\displaystyle=\lambda y_{i}+(1-\lambda)y_{j}. |  | (12) |

λ∈[0,1]\lambda\in[0,1], and is sampled from a Beta distribution, Beta​(α,α)\text{Beta}(\alpha,\alpha). Our choice of the hyperparameter α\alpha is detailed in Tab. 7.

### A.2 Training hyperparameters

Table 7 details the hyperparamters for all of our experiments. We use synchronous SGD with momentum, a cosine learning rate schedule with linear warmup, and a batch size of 64 for all experiments. As aforementioned, we only employed additional regularisation when training on the smaller Epic Kitchens and Something-Something v2 datasets.
