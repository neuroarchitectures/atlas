# Paper (Ye et al. 2023)

> Source: `https://arxiv.org/abs/2308.06721`

---

IP-Adapter: Text Compatible Image Prompt Adapter for
Text-to-Image Diffusion Models
Hu Ye, Jun Zhang∗, Sibo Liu, Xiao Han, Wei Yang
Tencent AI Lab
{huye, junejzhang, siboliu, haroldhan, willyang}@tencent.com
ABSTRACT
Recent years have witnessed the strong power of large text-to-image diffusion models for the
impressive generative capability to create high-fidelity images. However, it is very tricky to generate
desired images using only text prompt as it often involves complex prompt engineering. An alternative
to text prompt is image prompt, as the saying goes: "an image is worth a thousand words". Although
existing methods of direct fine-tuning from pretrained models are effective, they require large
computing resources and are not compatible with other base models, text prompt, and structural
controls. In this paper, we present IP-Adapter, an effective and lightweight adapter to achieve image
prompt capability for the pretrained text-to-image diffusion models. The key design of our IP-Adapter
is decoupled cross-attention mechanism that separates cross-attention layers for text features and
image features. Despite the simplicity of our method, an IP-Adapter with only 22M parameters
can achieve comparable or even better performance to a fully fine-tuned image prompt model. As
we freeze the pretrained diffusion model, the proposed IP-Adapter can be generalized not only to
other custom models fine-tuned from the same base model, but also to controllable generation using
existing controllable tools. With the benefit of the decoupled cross-attention strategy, the image
prompt can also work well with the text prompt to achieve multimodal image generation. The project
page is available at https://ip-adapter.github.io.
1
Introduction
Image generation has made remarkable strides with the success of recent large text-to-image diffusion models like
GLIDE [1], DALL-E 2 [2], Imagen [3], Stable Diffusion (SD) [4], eDiff-I [5] and RAPHAEL [6]. Users can write text
prompt to generate images with the powerful text-to-image diffusion models. But writing good text prompt to generate
desired content is not easy, as complex prompt engineering [7] is often required. Moreover, text is not informative to
express complex scenes or concepts, which can be a hindrance to content creation. Considering the above limitations of
the text prompt, we may ask if there are other prompt types to generate images. A natural choice is to use the image
prompt, since an image can express more content and details compared to text, just as often said: "an image is worth a
thousand words". DALL-E 2[2] makes the first attempt to support image prompt, the diffusion model is conditioned
on image embedding rather than text embedding, and a prior model is required to achieve the text-to-image ability.
However, most existing text-to-image diffusion models are conditioned on text to generate images, for example, the
popular SD model is conditioned on the text features extracted from a frozen CLIP [8] text encoder. Could image prompt
be also supported on these text-to-image diffusion models? Our work attempts to enable the generative capability with
image prompt for these text-to-image diffusion models in a simple manner.
Prior works, such as SD Image Variations2 and Stable unCLIP3, have demonstrated the effectiveness of fine-tuning
the text-conditioned diffusion models directly on image embedding to achieve image prompt capabilities. However,
the disadvantages of this approach are obvious. First, it eliminates the original ability to generate images using text,
and large computing resources are often required for such fine-tuning. Second, the fine-tuned models are typically not
reusable, as the image prompt ability cannot be directly transferred to the other custom models derived from the same
text-to-image base models. Moreover, the new models are often incompatible with existing structural control tools such
as ControlNet [9], which poses significant challenges for downstream applications. Due to the drawbacks of fine-tuning,
∗Corresponding author
2https://huggingface.co/lambdalabs/sd-image-variations-diffusers
3https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip
arXiv:2308.06721v1  [cs.CV]  13 Aug 2023

wearing sunglasses
+
+
+
+
+
Image Prompt
Structural Controls
Realistic Model 
Inpainting
Text Prompt
Variation
Anime Model 
Realistic Model Anime Model 
Figure 1: Various image synthesis with our proposed IP-Adapter applied on the pretrained text-to-image diffusion
models with different styles. The examples on the right show the results of image variations, multimodal generation,
and inpainting with image prompt, while the left examples show the results of controllable generation with image
prompt and additional structural conditions.
some studies [10] opt to replace the text encoder with an image encoder while avoiding fine-tuning the diffusion model.
Although this method is effective and simple, it still has several drawbacks. At first, only the image prompt is supported,
preventing users from simultaneously using text and image prompt to create images. Furthermore, merely fine-tuning
the image encoder is often not sufficient to guarantee image quality, and could lead to generalization issues.
In this study, we are curious about whether it is possible to achieve image prompt capability without modifying the
original text-to-image models. Fortunately, previous works are encouraging. Recent advances in controllable image
generation, such as ControlNet [9] and T2I-adapter [11], have demonstrated that an additional network can be effectively
plugged in the existing text-to-image diffusion models to guide the image generation. Most of the studies focus on image
generation with additional structure control such as user-drawn sketch, depth map, semantic segmentation map, etc.
Besides, image generation with style or content provided by reference image has also been achieved by simple adapters,
such as the style adapter of T2I-adapter [11] and global controller of Uni-ControlNet [12]. To achieve this, image
features extracted from CLIP image encoder are mapped to new features by a trainable network and then concatenated
with text features. By replacing the original text features, the merged features are fed into the UNet of the diffusion
model to guide image generation. These adapters can be seen as a way to have the ability to use image prompt, but the
generated image is only partially faithful to the prompted image. The results are often worse than the fine-tuned image
prompt models, let alone the model trained from scratch.
We argue that the main problem of the aforementioned methods lies in the cross-attention modules of text-to-image
diffusion models. The key and value projection weights of the cross-attention layer in the pretrained diffusion model are
trained to adapt the text features. Consequently, merging image features and text features into the cross-attention layer
only accomplishes the alignment of image features to text features, but this potentially misses some image-specific
information and eventually leads to only coarse-grained controllable generation (e.g., image style) with the reference
image.
To this end, we propose a more effective image prompt adapter named IP-Adapter to avoid the shortcomings of
the previous methods. Specifically, IP-Adapter adopts a decoupled cross-attention mechanism for text features and
image features. For every cross-attention layer in the UNet of diffusion model, we add an additional cross-attention
layer only for image features. In the training stage, only the parameters of the new cross-attention layers are trained,
while the original UNet model remains frozen. Our proposed adapter is lightweight but very efficient: the generative
performance of an IP-Adapter with only 22M parameters is comparable to a fully fine-tuned image prompt model from
the text-to-image diffusion model. More importantly, our IP-Adapter exhibits excellent generalization capabilities and
is compatible with text prompt. With our proposed IP-Adapter, various image generation tasks can be easily achieved,
as illustrated in Figure 1.
To sum up, our contributions are as follows:
• We present IP-Adapter, a lightweight image prompt adaptation method with the decoupled cross-attention
strategy for existing text-to-image diffusion models. Quantitative and qualitative experimental results show
that a small IP-Adapter with about 22M parameters is comparable or even better than the fully fine-tuned
models for image prompt based generation.
2

• Our IP-Adapter is reusable and flexible. IP-Adapter trained on the base diffusion model can be generalized to
other custom models fine-tuned from the same base diffusion model. Moreover, IP-Adapter is compatible
with other controllable adapters such as ControlNet, allowing for an easy combination of image prompt with
structure controls.
• Due to the decoupled cross-attention strategy, image prompt is compatible with text prompt to achieve
multimodal image generation.
2
Related Work
We focus on designing an image prompt adapter for the existing text-to-image diffusion models. In this section, we
review recent works on text-to-image diffusion models, as well as relevant studies on adapters for large models.
2.1
Text-to-Image Diffusion Models
Large text-to-image models are mainly divided into two categories: autoregressive models and diffusion models.
Early works, such as DALLE [13], CogView [14, 15] and Make-A-Scene [16], are autoregressive models. For the
autoregressive model, an image tokenizer like VQ-VAE [17] is used to convert an image to tokens, then an autoregressive
transformer [18] conditioned on text tokens is trained to predict image tokens. However, autoregressive models often
require large parameters and computing resources to generate high-quality images, as seen in Parti [19].
Recently, diffusion models (DMs) [20, 21, 22, 23] has emerged as the new state-of-the-art model for text-to-image
generation. As a pioneer, GLIDE uses a cascaded diffusion architecture with a 3.5B text-conditional diffusion model at
64×64 resolution and a 1.5B text-conditional upsampling diffusion model at 256×256 resolution. DALL-E 2 employs
a diffusion model conditioned image embedding, and a prior model was trained to generate image embedding by giving
a text prompt. DALL-E 2 not only supports text prompt for image generation but also image prompt. To enhance the
text understanding, Imagen adopts T5 [24], a large transformer language model pretrained on text-only data, as the text
encoder of diffusion model. Re-Imagen [25] uses retrieved information to improve the fidelity of generated images for
rare or unseen entities. SD is built on the latent diffusion model [4], which operates on the latent space instead of pixel
space, enabling SD to generate high-resolution images with only a diffusion model. To improve text alignment, eDiff-I
was designed with an ensemble of text-to-image diffusion models, utilizing multiple conditions, including T5 text,
CLIP text, and CLIP image embeddings. Versatile Diffusion [26] presents a unified multi-flow diffusion framework to
support text-to-image, image-to-text, and variations within a single model. To achieve controllable image synthesis,
Composer [27] presents a joint fine-tuning strategy with various conditions on a pretrained diffusion model conditioned
on image embedding. RAPHAEL introduces a mixture-of-experts (MoEs) strategy [28, 29] into the text-conditional
image diffusion model to enhance image quality and aesthetic appeal.
An attractive feature of DALL-E 2 is that it can also use image prompt to generate image variations. Hence, there are
also some works to explore to support image prompt for the text-to-image diffusion models conditioned only on text.
SD Image Variations model is fine-tuned from a modified SD model where the text features are replaced with the image
embedding from CLIP image encoder. Stable unCLIP is also a fine-tuned model on SD, in which the image embedding
is added to the time embedding. Although the fine-tuning model can successfully use image prompt to generate images,
it often requires a relatively large training cost, and it fails to be compatible with existing tools, e.g., ControlNet [9].
2.2
Adapters for Large Models
As fine-tuning large pre-trained models is inefficient, an alternative approach is using adapters, which add a few trainable
parameters but freeze the original model. Adapters have been used in the field of NLP for a long time [30]. Recently,
adapters have been utilized to achieve vision-language understanding for large language models [31, 32, 33, 34, 35].
With the popularity of recent text-to-image models, adapters have also been used to provide additional control for the
generation of text-to-image models. ControlNet [9] first proves that an adapter could be trained with a pretrained text-to-
image diffusion model to learn task-specific input conditions, e.g., canny edge. Almost concurrently, T2I-adapter [11]
employs a simple and lightweight adapter to achieve fine-grained control in the color and structure of the generated
images. To reduce the fine-tuning cost, Uni-ControlNet [12] presents a multi-scale condition injection strategy to learn
an adapter for various local controls.
Apart from the adapters for structural control, there are also works for the controllable generation conditioned on the
content and style of the provided image. ControlNet Shuffle 1 trained to recompose images, can be used to guide the
1https://github.com/lllyasviel/ControlNet-v1-1-nightly
3

A girl with sunglasses
Text 
Encoder
Image
Encoder
Linear
LN
Image Features
Text Features
Decoupled Cross-Attention
𝒙𝒕
𝒙𝒕"𝟏
Cross 
Attention
Cross 
Attention
Denoising U-Net
Frozen 
modules
Trainable 
modules
Figure 2: The overall architecture of our proposed IP-Adapter with decoupled cross-attention strategy. Only the newly
added modules (in red color) are trained while the pretrained text-to-image model is frozen.
generation by a user-provided image. Moreover, ControlNet Reference-only1 was presented to achieve image variants
on SD model through simple feature injection without training. In the updated version of T2I-adapter, a style adapter
is designed to control the style of generated images using a reference image by appending image features extracted
from the CLIP image encoder to text features. The global control adapter of Uni-ControlNet also projects the image
embedding from CLIP image encoder into condition embeddings by a small network and concatenates them with
the original text embeddings, and it is used to guide the generation with the style and content of reference image.
SeeCoder [10] presents a semantic context encoder to replace the original text encoder to generate image variants.
Although the aforementioned adapters are lightweight, their performance is hardly comparable to that of the fine-tuned
image prompt models, let alone one trained from scratch. In this study, we introduce a decoupled cross-attention
mechanism to achieve a more effective image prompt adapter. The proposed adapter remains simple and small but
outperforms previous adapter methods, and is even comparable to fine-tuned models.
3
Method
In this section, we first introduce some preliminaries about text-to-image diffusion models. Then, we depict in detail
the motivation and the design of the proposed IP-Adapter.
3.1
Prelimiaries
Diffusion models are a class of generative models that comprise two processes: a diffusion process (also known as
the forward process), which gradually adds Gaussian noise to the data using a fixed Markov chain of T steps, and a
denoising process that generates samples from Gaussian noise with a learnable model. Diffusion models can also be
conditioned on other inputs, such as text in the case of text-to-image diffusion models. Typically, the training objective
of a diffusion model, denoted as ϵθ, which predicts noise, is defined as a simplified variant of the variational bound:
Lsimple = Ex0,ϵ∼N(0,I),c,t∥ϵ −ϵθ
 xt, c, t

∥2,
(1)
where x0 represents the real data with an additional condition c, t ∈[0, T] denotes the time step of diffusion process,
xt = αtx0 + σtϵ is the noisy data at t step, and αt, σt are predefined functions of t that determine the diffusion process.
Once the model ϵθ is trained, images can be generated from random noise in an iterative manner. Generally, fast
samplers such as DDIM [21], PNDM [36] and DPM-Solver [37, 38], are adopted in the inference stage to accelerate
the generation process.
For the conditional diffusion models, classifier guidance [23] is a straightforward technique used to balance image
fidelity and sample diversity by utilizing gradients from a separately trained classifier. To eliminate the need for training
1https://github.com/Mikubill/sd-webui-controlnet
4

a classifier independently, classifier-free guidance [39] is often employed as an alternative method. In this approach,
the conditional and unconditional diffusion models are jointly trained by randomly dropping c during training. In the
sampling stage, the predicted noise is calculated based on the prediction of both the conditional model ϵθ(xt, c, t) and
unconditional model ϵθ(xt, t):
ˆϵθ(xt, c, t) = wϵθ(xt, c, t) + (1 −w)ϵθ(xt, t),
(2)
here, w, often named guidance scale or guidance weight, is a scalar value that adjusts the alignment with condition c.
For text-to-image diffusion models, classifier-free guidance plays a crucial role in enhancing the image-text alignment
of generated samples.
In our study, we utilize the open-source SD model as our example base model to implement the IP-Adapter. SD is a
latent diffusion model conditioned on text features extracted from a frozen CLIP text encoder. The architecture of the
diffusion model is based on a UNet [40] with attention layers. Compared to pixel-based diffusion models like Imagen,
SD is more efficient since it is constructed on the latent space from a pretrained auto-encoder model.
3.2
Image Prompt Adapter
In this paper, the image prompt adapter is designed to enable a pretrained text-to-image diffusion model to generate
images with image prompt. As mentioned in previous sections, current adapters struggle to match the performance of
fine-tuned image prompt models or the models trained from scratch. The major reason is that the image features cannot
be effectively embedded in the pretrained model. Most methods simply feed concatenated features into the frozen
cross-attention layers, preventing the diffusion model from capturing fine-grained features from the image prompt. To
address this issue, we present a decoupled cross-attention strategy, in which the image features are embedded by newly
added cross-attention layers. The overall architecture of our proposed IP-Adapter is demonstrated in Figure 2. The
proposed IP-Adapter consists of two parts: an image encoder to extract image features from image prompt, and adapted
modules with decoupled cross-attention to embed image features into the pretrained text-to-image diffusion model.
3.2.1
Image Encoder
Following most of the methods, we use a pretrained CLIP image encoder model to extract image features from the
image prompt. The CLIP model is a multimodal model trained by contrastive learning on a large dataset containing
image-text pairs. We utilize the global image embedding from the CLIP image encoder, which is well-aligned with
image captions and can represent the rich content and style of the image. In the training stage, the CLIP image encoder
is frozen.
To effectively decompose the global image embedding, we use a small trainable projection network to project the image
embedding into a sequence of features with length N (we use N = 4 in this study), the dimension of the image features
is the same as the dimension of the text features in the pretrained diffusion model. The projection network we used in
this study consists of a linear layer and a Layer Normalization [41].
3.2.2
Decoupled Cross-Attention
The image features are integrated into the pretrained UNet model by the adapted modules with decoupled cross-attention.
In the original SD model, the text features from the CLIP text encoder are plugged into the UNet model by feeding into
the cross-attention layers. Given the query features Z and the text features ct, the output of cross-attention Z′ can be
defined by the following equation:
Z′ = Attention(Q, K, V) = Softmax(QK⊤
√
d
)V,
(3)
where Q = ZWq, K = ctWk, V = ctWv are the query, key, and values matrices of the attention operation
respectively, and Wq, Wk, Wv are the weight matrices of the trainable linear projection layers.
A straightforward method to insert image features is to concatenate image features and text features and then feed them
into the cross-attention layers. However, we found this approach to be insufficiently effective. Instead, we propose a
decoupled cross-attention mechanism where the cross-attention layers for text features and image features are separate.
To be specific, we add a new cross-attention layer for each cross-attention layer in the original UNet model to insert
image features. Given the image features ci, the output of new cross-attention Z′′ is computed as follows:
Z′′ = Attention(Q, K′, V′) = Softmax(Q(K′)⊤
√
d
)V′,
(4)
5

where, Q = ZWq, K′ = ciW′
k and V′ = ciW′
v are the query, key, and values matrices from the image features. W′
k
and W′
v are the corresponding weight matrices. It should be noted that we use the same query for image cross-attention
as for text cross-attention. Consequently, we only need add two paramemters W′
k, W′
v for each cross-attention layer. In
order to speed up the convergence, W′
k and W′
v are initialized from Wk and Wv. Then, we simply add the output of
image cross-attention to the output of text cross-attention. Hence, the final formulation of the decoupled cross-attention
is defined as follows:
Znew = Softmax(QK⊤
√
d
)V + Softmax(Q(K′)⊤
√
d
)V′
where Q = ZWq, K = ctWk, V = ctWv, K′ = ciW′
k, V′ = ciW′
v
(5)
Sine we freeze the original UNet model, only the W′
k and W′
v are trainable in the above decoupled cross-attention.
3.2.3
Training and Inference
During training, we only optimize the IP-Adapter while keeping the parameters of the pretrained diffusion model fixed.
The IP-Adapter is also trained on the dataset with image-text pairs1, using the same training objective as original SD:
Lsimple = Ex0,ϵ,ct,ci,t∥ϵ −ϵθ
 xt, ct, ci, t

∥2.
(6)
We also randomly drop image conditions in the training stage to enable classifier-free guidance in the inference stage:
ˆϵθ(xt, ct, ci, t) = wϵθ(xt, ct, ci, t) + (1 −w)ϵθ(xt, t)
(7)
Here, we simply zero out the CLIP image embedding if the image condition is dropped.
As the text cross-attention and image cross-attention are detached, we can also adjust the weight of the image condition
in the inference stage:
Znew = Attention(Q, K, V) + λ · Attention(Q, K′, V′)
(8)
where λ is weight factor, and the model becomes the original text-to-image diffusion model if λ = 0.
4
Experiments
4.1
Experimental Setup
4.1.1
Training Data
To train the IP-Adapter, we build a multimodal dataset including about 10 million text-image pairs from two open
source datasets - LAION-2B [42] and COYO-700M [43].
4.1.2
Implementation Details
Our experiments are based on SD v1.52, and we use OpenCLIP ViT-H/14 [44] as the image encoder. There are 16
cross-attention layers in SD model, and we add a new image cross-attention layer for each of these layers. The total
trainable parameters of our IP-Adapter including a projection network and adapted modules, amount to about 22M,
making the IP-Adapter quite lightweight. We implement our IP-Adapter with HuggingFace diffusers library [45] and
employ DeepSpeed ZeRO-2 [13] for fast training. IP-Adapter is trained on a single machine with 8 V100 GPUs for
1M steps with a batch size of 8 per GPU. We use the AdamW optimizer [46] with a fixed learning rate of 0.0001 and
weight decay of 0.01. During training, we resize the shortest side of the image to 512 and then center crop the image
with 512 × 512 resolution. To enable classifier-free guidance, we use a probability of 0.05 to drop text and image
individually, and a probability of 0.05 to drop text and image simultaneously. In the inference stage, we adopt DDIM
sampler with 50 steps, and set the guidance scale to 7.5. When only using image prompt, we set the text prompt to
empty and λ = 1.0.
1Note that it is also possible to train the model without text prompt since using image prompt only is informative to guide the
final generation.
2https://huggingface.co/runwayml/stable-diffusion-v1-5
6

Figure 3: The visual comparison of our proposed IP-Adapter with other methods conditioned on different kinds and
styles of images.
7

Table 1: Quantitative comparison of the proposed IP-Adapter with other methods on COCO validation set. The best
results are in bold.
Method
Reusable to
custom models
Compatible with
controllable tools
Multimodal
prompts
Trainable
parameters
CLIP-T ↑
CLIP-I ↑
Training from scratch
Open unCLIP
%
%
%
893M
0.608
0.858
Kandinsky-2-1
%
%
%
1229M
0.599
0.855
Versatile Diffusion
%
%
!
860M
0.587
0.830
Fine-tunining from text-to-image model
SD Image Variations
%
%
%
860M
0.548
0.760
SD unCLIP
%
%
%
870M
0.584
0.810
Adapters
Uni-ControlNet (Global Control)
!
!
!
47M
0.506
0.736
T2I-Adapter (Style)
!
!
!
39M
0.485
0.648
ControlNet Shuffle
!
!
!
361M
0.421
0.616
IP-Adapter
!
!
!
22M
0.588
0.828
Figure 4: The generated images of different diffusion models with our proposed IP-Adapter. The IP-Adapter is only
trained once.
4.2
Comparison with Existing Methods
To demonstrate the effectiveness of our method, we compare our IP-Adapter with other existing methods on generation
with image prompt. We select three types of methods: training from scratch, fine-tuning from text-to-image model, and
8

Figure 5: Visualization of generated samples with image prompt and additional structural conditions. Note that we
don’t need fine-tune the IP-Adapter.
Figure 6: Comparison of our IP-Adapter with other methods on different structural conditions.
adapters. For the method trained from scratch, we select 3 open source models: open unCLIP1 which is a reproduction
of DALL-E 2, Kandinsky-2-1 2 which is a mixture of DALL-E 2 and latent diffusion, and Versatile Diffusion [26].
For the fine-tuned models, we choose SD Image Variations and SD unCLIP. For the adapters, we compare our IP-
Adapter with the style-adapter of T2I-Adapter, the global controller of Uni-ControlNet, ControlNet Shuffle, ControlNet
Reference-only and SeeCoder.
1https://github.com/kakaobrain/karlo
2https://github.com/ai-forever/Kandinsky-2
9

Figure 7: Examples of image-to-image and inpainting with image prompt by our IP-Adapter.
4.2.1
Quantitative Comparison
We use the validation set of COCO2017 [47] containing 5,000 images with captions for quantitative evaluation. For a
fair comparison, we generate 4 images conditioned on the image prompt for each sample in the dataset, resulting in
total 20,000 generated images for each method. We use two metrics to evaluate the alignment with the image condition:
• CLIP-I: the similarity in CLIP image embedding of generated images with the image prompt.
• CLIP-T: the CLIPScore [48] of the generated images with captions of the image prompts.
We calculate the average value of the two metrics on all generated images with CLIP ViT-L/141 model. As the open
source SeeCoder is used with additional structural controls and ControlNet Reference-only is released under the web
framework, we only conduct qualitative evaluations. The comparison results are shown in Table 1. As we observe, our
method is much better than other adapters, and is also comparable or even better than the fine-tuned model with only
22M parameters.
4.2.2
Qualitative Comparison
We also select various kinds and styles of images to qualitatively evaluate our method. For privacy reasons, the images
with real face are synthetic. For SeeCoder, we also use the scribble control with ControlNet to generate images. For
ControlNet Reference-only, we also input the captions generated with BLIP caption model [49]. For each image prompt,
we random generate 4 samples and select the best one for each method to ensure fairness. As we can see in Figure 3,
the proposed IP-Adapter is mostly better than other adapters both in image quality and alignment with the reference
image. Moreover, our method is slightly better than the fine-tuned models, and also comparable to the models trained
from scratch in most cases.
In conclusion, the proposed IP-Adapter is lightweight and effective method to achieve the generative capability with
image prompt for the pretrained text-to-image diffusion models.
4.3
More Results
Although the proposed IP-Adapter is designed to achieve the generation with image prompt, its robust generalization
capabilities allow for a broader range of applications. As shown in Table 1, our IP-Adapter is not only reusable to
custom models, but also compatible with existing controllable tools and text prompt. In this part, we show more results
that our adapter can generate.
4.3.1
Generalizable to Custom Models
As we freeze the original diffusion model in the training stage, the IP-Adapter can also be generalizable to the custom
models fine-tuned from SD v1.5 like other adapters (e.g., ControlNet). In other words, once IP-Adapter is trained,
it can be directly reusable on custom models fine-tuned from the same base model. To validate this, we select three
1https://huggingface.co/openai/clip-vit-large-patch14
10

Figure 8: Generated examples of our IP-Adapter with multimodal prompts.
community models from HuggingFace model library1: Realistic Vision V4.0, Anything v4, and ReV Animated. These
models are all fine-tuned from SD v1.5. As shown in Figure 4, our IP-Adapter works well on these community
models. Furthermore, the generated images can mix the style of the community models, for example, we can generate
anime-style images when using the anime-style model Anything v4. Interestingly, our adapter can be directly applied to
SD v1.4, as SD v1.5 is trained with more steps based on SD v1.4.
4.3.2
Structure Control
For text-to-image diffusion models, a popular application is that we can create images with additional structure control.
As our adapter does not change the original network structure, we found that the IP-Adapter is fully compatible with
existing controllable tools. As a result, we can also generate controllable images with image prompt and additional
conditions. Here, we combine our IP-Adapter with two existing controllable tools, ControlNet and T2I-Adapter. Figure
5 shows various samples that are generated with image prompt and different structure controls: the samples of the first
two rows are generated with ControlNet models, while the samples in the last row are generated with T2I-Adapters.
Our adapter effectively works with these tools to produce more controllable images without fine-tuning.
We also compare our adapter with other adapters on the structural control generation, the results are shown in Figure
6. For T2I-Adapter and Uni-ControlNet, we use the default composable multi-conditions. For SeeCoder and our
IP-Adapter, we use ControlNet to achieve structural control. For ControlNet Shuffle and ControlNet Reference-only,
we use multi-ControlNet. As we can see, our method not only outperforms other methods in terms of image quality, but
also produces images that better align with the reference image.
4.3.3
Image-to-Image and Inpainting
Apart from text-to-image generation, text-to-image diffusion models also can achieve text-guided image-to-image
and inpainting with SDEdit [50]. As demonstrated in Figure 7, we can also obtain image-guided image-to-image and
inpainting by simply replacing text prompt with image prompt.
4.3.4
Multimodal Prompts
For the fully fine-tuned image prompt models, the original text-to-image ability is almost lost. However, with the
proposed IP-Adapter, we can generate images with multimodal prompts including image prompt and text prompt. We
found that this capability performs particularly well on community models. In the inference stage with multimodal
prompts, we adjust λ to make a balance between image prompt and text prompt. Figure 8 displays various results with
multimodal prompts using Realistic Vision V4.0 model. As we can see, we can use additional text prompt to generate
more diverse images. For instance, we can edit attributes and change the scene of the subject conditioned on the image
prompt using simple text descriptions.
1https://huggingface.co/models
11

Figure 9: Comparison with multimodal prompts between our IP-Adapter with other methods.
We also compare our IP-Adapter with other methods including Versatile Diffusion, BLIP Diffusion [31], Uni-ControlNet,
T2I-Adapter, ControlNet Shuffle, and ControlNet Reference-only. The comparison results are shown in Figure 9.
Compared with other existing methods, our method can generate superior results in both image quality and alignment
with multimodal prompts.
4.4
Ablation Study
4.4.1
Importance of Decoupled Cross-Attention
In order to verify the effectiveness of the decoupled cross-attention strategy, we also compare a simple adapter without
decoupled cross-attention: image features are concatenated with text features, and then embedded into the pretrained
12

Figure 10: Comparison results of our IP-Adapter with simple adapter. The decoupled cross-attention strategy is not
used in the simple adapter.
Figure 11: The visual difference of generated samples between the IP-Adapter with global features and the IP-Adapter
with fine-grained features.
cross-attention layers. For a fair comparison, we trained both adapters for 200,000 steps with the same configuration.
Figure 10 provides comparative examples with the IP-Adapter with decoupled cross-attention and the simple adapter.
As we can observe, the IP-Adapter not only can generate higher quality images than the simple adapter, but also can
generate more consistent images with image prompts.
4.4.2
Comparison of Fine-grained Features and Global Features
Since our IP-Adapter utilizes the global image embedding from the CLIP image encoder, it may lose some information
from the reference image. Therefore, we design an IP-Adapter conditioned on fine-grained features. First, we extract
the grid features of the penultimate layer from the CLIP image encoder. Then, a small query network is used to learn
features. Specifically, 16 learnable tokens are defined to extract information from the grid features using a lightweight
transformer model. The token features from the query network serve as input to the cross-attention layers.
The results of the two adapters are shown in Figure 11. Although the IP-Adapter with finer-grained features can generate
more consistent images with image prompt, it can also learn the spatial structure information, which may reduce the
diversity of generated images. However, additional conditions, such as text prompt and structure map, can be combined
with image prompt to generate more diverse images. For instance, we can synthesize novel images with the guidance of
additional human poses.
13

5
Conclusions and Future Work
In this work, we propose IP-Adapter to achieve image prompt capability for the pretrained text-to-image diffusion
models. The core design of our IP-Adapter is based on a decoupled cross-attention strategy, which incorporates separate
cross-attention layers for image features. Both quantitative and qualitative experimental results demonstrate that our
IP-Adapter with only 22M parameters performs comparably or even better than some fully fine-tuned image prompt
models and existing adapters. Furthermore, our IP-Adapter, after being trained only once, can be directly integrated
with custom models derived from the same base model and existing structural controllable tools, thereby expanding
its applicability. More importantly, image prompt can be combined with text prompt to achieve multimodal image
generation.
Despite the effectiveness of our IP-Adapter, it can only generate images that resemble the reference images in content
and style. In other words, it cannot synthesize images that are highly consistent with the subject of a given image
like some existing methods, e.g., Textual Inversion [51] and DreamBooth [52]. In the future, we aim to develop more
powerful image prompt adapters to enhance consistency.
References
[1] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever,
and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models.
arXiv preprint arXiv:2112.10741, 2021.
[2] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image
generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.
[3] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models
with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
[4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image
synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 10684–10695, 2022.
[5] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila,
Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers.
arXiv preprint arXiv:2211.01324, 2022.
[6] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-
image generation via large mixture of diffusion paths. arXiv preprint arXiv:2305.18295, 2023.
[7] Sam Witteveen and Martin Andrews. Investigating prompt engineering in diffusion models. arXiv preprint
arXiv:2211.15462, 2022.
[8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language
supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.
[9] Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. arXiv
preprint arXiv:2302.05543, 2023.
[10] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt-free diffusion:
Taking" text" out of text-to-image diffusion models. arXiv preprint arXiv:2305.16223, 2023.
[11] Chong Mou, Xintao Wang, Liangbin Xie, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter:
Learning adapters to dig out more controllable ability for text-to-image diffusion models.
arXiv preprint
arXiv:2302.08453, 2023.
[12] Shihao Zhao, Dongdong Chen, Yen-Chun Chen, Jianmin Bao, Shaozhe Hao, Lu Yuan, and Kwan-Yee K Wong.
Uni-controlnet: All-in-one control to text-to-image diffusion models. arXiv preprint arXiv:2305.16322, 2023.
[13] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya
Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–
8831. PMLR, 2021.
[14] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao,
Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in Neural
Information Processing Systems, 34:19822–19835, 2021.
14

[15] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via
hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890–16902, 2022.
[16] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene:
Scene-based text-to-image generation with human priors. In European Conference on Computer Vision, pages
89–106. Springer, 2022.
[17] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information
processing systems, 30, 2017.
[18] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and
Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
[19] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku,
Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation.
arXiv preprint arXiv:2206.10789, 2022.
[20] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using
nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR,
2015.
[21] Jiaming Song, Chenlin Meng, and Stefano Ermon.
Denoising diffusion implicit models.
arXiv preprint
arXiv:2010.02502, 2020.
[22] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456,
2020.
[23] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural
information processing systems, 34:8780–8794, 2021.
[24] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei
Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of
Machine Learning Research, 21(1):5485–5551, 2020.
[25] Wenhu Chen, Hexiang Hu, Chitwan Saharia, and William W Cohen. Re-imagen: Retrieval-augmented text-to-
image generator. arXiv preprint arXiv:2209.14491, 2022.
[26] Xingqian Xu, Zhangyang Wang, Eric Zhang, Kai Wang, and Humphrey Shi. Versatile diffusion: Text, images and
variations all in one diffusion model. arXiv preprint arXiv:2211.08332, 2022.
[27] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable
image synthesis with composable conditions. arXiv preprint arXiv:2302.09778, 2023.
[28] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Out-
rageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538,
2017.
[29] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with
simple and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232–5270, 2022.
[30] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo,
Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on
Machine Learning, pages 2790–2799. PMLR, 2019.
[31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
[32] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language
understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
[33] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao.
Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199,
2023.
[34] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xi-
angyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010,
2023.
[35] Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong.
What matters in training a gpt4-style language model with multimodal inputs? arXiv preprint arXiv:2307.02469,
2023.
15

[36] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds.
arXiv preprint arXiv:2202.09778, 2022.
[37] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for
diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems,
35:5775–5787, 2022.
[38] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for
guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022.
[39] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
[40] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image seg-
mentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International
Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.
[41] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450,
2016.
[42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for
training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294,
2022.
[43] Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m:
Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.
[44] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave,
Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt.
Openclip. https://github.com/mlfoundations/open_clip, 2021.
[45] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj,
and Thomas Wolf.
Diffusers: State-of-the-art diffusion models.
https://github.com/huggingface/
diffusers, 2022.
[46] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101,
2017.
[47] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and
C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th
European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755.
Springer, 2014.
[48] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free
evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.
[49] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for
unified vision-language understanding and generation. In International Conference on Machine Learning, pages
12888–12900. PMLR, 2022.
[50] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit:
Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.
[51] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or.
An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint
arXiv:2208.01618, 2022.
[52] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth:
Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
16
