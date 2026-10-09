# Paper (Brooks et al. 2022)

> Source: `https://arxiv.org/abs/2211.09800`

---

InstructPix2Pix: Learning to Follow Image Editing Instructions
Tim Brooks*
Aleksander Holynski*
Alexei A. Efros
University of California, Berkeley
“Replace the fruits with cake”
“Make his jacket out of leather”
“Add fireworks to the sky”
“Turn it into a still from a western”
“Swap sunflowers with roses”
“What would it look like if it were snowing?”
Figure 1. Given an image and an instruction for how to edit that image, our model performs the appropriate edit. Our model does not
require full descriptions for the input or output image, and edits images in the forward pass without per-example inversion or ﬁne-tuning.
Abstract
We propose a method for editing images from human in-
structions: given an input image and a written instruction
that tells the model what to do, our model follows these in-
structions to edit the image. To obtain training data for
this problem, we combine the knowledge of two large pre-
trained models—a language model (GPT-3) and a text-to-
image model (Stable Diffusion)—to generate a large dataset
of image editing examples. Our conditional diffusion model,
InstructPix2Pix, is trained on our generated data, and gen-
eralizes to real images and user-written instructions at in-
ference time. Since it performs edits in the forward pass and
does not require per-example ﬁne-tuning or inversion, our
model edits images quickly, in a matter of seconds. We show
compelling editing results for a diverse collection of input
images and written instructions.
*Denotes equal contribution
More results on our project page: timothybrooks.com/instruct-pix2pix
1. Introduction
We present a method for teaching a generative model to
follow human-written instructions for image editing. Since
training data for this task is difﬁcult to acquire at scale,
we propose an approach for generating a paired dataset
that combines multiple large models pretrained on different
modalities: a large language model (GPT-3 [7]) and a text-
to-image model (Stable Diffusion [52]). These two models
capture complementary knowledge about language and im-
ages that can be combined to create paired training data for
a task spanning both modalities.
Using our generated paired data, we train a conditional
diffusion model that, given an input image and a text in-
struction for how to edit it, generates the edited image. Our
model directly performs the image edit in the forward pass,
and does not require any additional example images, full de-
scriptions of the input/output images, or per-example ﬁne-
tuning. Despite being trained entirely on synthetic exam-
ples (i.e., both generated written instructions and generated
1
arXiv:2211.09800v2  [cs.CV]  18 Jan 2023

imagery), our model achieves zero-shot generalization to
both arbitrary real images and natural human-written in-
structions. Our model enables intuitive image editing that
can follow human instructions to perform a diverse collec-
tion of edits: replacing objects, changing the style of an im-
age, changing the setting, the artistic medium, among oth-
ers. Selected examples can be found in Figure 1.
2. Prior work
Composing large pretrained models
Recent work has
shown that large pretrained models can be combined to
solve multimodal tasks that no one model can perform
alone, such as image captioning and visual question an-
swering (tasks that require the knowledge of both a large
language model and a text-image model). Techniques for
combining pretrained models include joint ﬁnetuning on a
new task [4, 34, 41, 68], communication through prompt-
ing [63,70], composing probability distributions of energy-
based models [11, 38], guiding one model with feedback
from another [62], and iterative optimization [35].
Our
method is similar to prior work in that it leverages the com-
plementary abilities of two pretrained models—GPT-3 [7])
and Stable Diffusion [52]—but differs in that we use these
models to generate paired multi-modal training data.
Diffusion-based generative models
Recent advances in
diffusion models [60] have enabled state-of-the-art image
synthesis [10, 18, 19, 54, 56, 61] as well as generative mod-
els of other modalities such as video [21, 59], audio [31],
text [36] and network parameters [46].
Recent text-to-
image diffusion models [42,49,52,55] have shown to gen-
erate realistic images from arbitrary text captions.
Generative models for image editing
Image editing
models traditionally targeted a single editing task such as
style transfer [15, 16] or translation between image do-
mains [22, 24, 37, 43, 72]. Numerous editing approaches
invert [1–3, 12] or encode [8, 51, 64] images into a latent
space (e.g., StyleGAN [26, 27]) where they can be edited
by manipulating latent vectors. Recent models have lever-
aged CLIP [48] embeddings to guide image editing using
text [5, 9, 14, 29, 32, 42, 45, 71]. We compare with one of
these methods, Text2Live [6], an editing method that opti-
mizes for an additive image layer that maximizes a CLIP
similarity objective.
Recent works have used pretrained text-to-image diffu-
sion models for image editing [5,17,28,39,49]. While some
text-to-image models natively have the ability to edit im-
ages (e.g., DALLE-2 can create variations of images, in-
paint regions, and manipulate the CLIP embedding [49]),
using these models for targeted editing is non-trivial, be-
cause in most cases they offer no guarantees that similar
text prompts will yield similar images.
Recent work by
Hertz et al. [17] tackles this issue with Prompt-to-Prompt,
a method for assimilating the generated images for simi-
lar text prompts, such that isolated edits can be made to a
generated image. We use this method in generating training
data. To edit non-generated (i.e., real) imagery, SDEdit [39]
uses a pretrained model to noise and denoise an input im-
age with a new target prompt. We compare with SDEdit
as a baseline. Other recent works perform local inpainting
given a caption and user-drawn mask [5, 49], generate new
images of a speciﬁc object or concept learned from a small
collection of images [13, 53], or perform editing by invert-
ing (and ﬁne-tuning) a single image, and subsequently re-
generating with a new text description [28]. In contrast to
these approaches, our model takes only a single image and
an instruction for how to edit that image (i.e., not a full de-
scription of any image), and performs the edit directly in
the forward pass without need for a user-drawn mask, addi-
tional images, or per-example inversion or ﬁnetuning.
Learning to follow instructions
Our method differs from
existing text-based image editing works [6,13,17,28,39,53]
in that it enables editing from instructions that tell the model
what action to perform, as opposed to text labels, captions
or descriptions of input/output images. A key beneﬁt of fol-
lowing editing instructions is that the user can just tell the
model exactly what to do in natural written text. There is
no need for the user to provide extra information, such as
example images or descriptions of visual content that re-
mains constant between the input and output images. In-
structions are expressive, precise, and intuitive to write, al-
lowing the user to easily isolate speciﬁc objects or visual at-
tributes to change. Our goal to follow written image editing
instructions is inspired by recent work teaching large lan-
guage models to better follow human instructions for lan-
guage tasks [40,44,69].
Training data generation with generative models
Deep
models typically require large amounts of training data.
Internet data collections are often suitable, but may not
exist in the form necessary for supervision, e.g., paired
data of particular modalities. As generative models con-
tinue to improve, there is growing interest in their use as a
source of cheap and plentiful training data for downstream
tasks [33, 47, 50, 58, 65, 66].
In this paper, we use two
different off-the-shelf generative models (language, text-to-
image) to produce training data for our editing model.
3. Method
We treat instruction-based image editing as a supervised
learning problem: (1) ﬁrst, we generate a paired training
dataset of text editing instructions and images before/after
the edit (Sec. 3.1, Fig. 2a-c), then (2) we train an image
editing diffusion model on this generated dataset (Sec. 3.2,
Fig 2d). Despite being trained with generated images and
2

Stable Diffusion
+ Prompt2Prompt
Input Caption: “photograph of a girl riding a horse”
Instruction: “have her ride a dragon”
Edited Caption: “photograph of a girl riding a dragon”
GPT-3
Input Caption: “photograph of a girl riding a horse”
Edited Caption: “photograph of a girl riding a dragon”
(b) Generate paired images:
(a) Generate text edits:
InstructPix2Pix
“turn her into a snake lady”
Training Data Generation
Instruction-following Diffusion Model
“have her ride a dragon”
“Color the cars pink”
“Make it lit by fireworks”
“convert to brick”
…
(c) Generated training examples:
(d) Inference on real images:
Figure 2. Our method consists of two parts: generating an image editing dataset, and training a diffusion model on that dataset. (a) We
ﬁrst use a ﬁnetuned GPT-3 to generate instructions and edited captions. (b) We then use StableDiffusion [52] in combination with Prompt-
to-Prompt [17] to generate pairs of images from pairs of captions. We use this procedure to create a dataset (c) of over 450,000 training
examples. (d) Finally, our InstructPix2Pix diffusion model is trained on our generated data to edit images from instructions. At inference
time, our model generalizes to edit real images from human-written instructions.
editing instructions, our model is able to generalize to edit-
ing real images using arbitrary human-written instructions.
See Fig. 2 for an overview of our method.
3.1. Generating a Multi-modal Training Dataset
We combine the abilities of two large-scale pretrained
models that operate on different modalities—a large lan-
guage model [7] and a text-to-image model [52]—to gen-
erate a multi-modal training dataset containing text editing
instructions and the corresponding images before and af-
ter the edit. In the following two sections, we describe in
detail the two steps of this process. In Section 3.1.1, we
describe the process of ﬁne-tuning GPT-3 [7] to generate a
collection of text edits: given a prompt describing an im-
age, produce a text instruction describing a change to be
made and a prompt describing the image after that change
(Figure 2a). Then, in Section 3.1.2, we describe the process
of converting the two text prompts (i.e., before and after the
edit) into a pair of corresponding images using a text-to-
image model [52] (Figure 2b).
3.1.1
Generating Instructions and Paired Captions
We ﬁrst operate entirely in the text domain, where we lever-
age a large language model to take in image captions and
produce editing instructions and the resulting text captions
after the edit. For example, as shown in Figure 2a, provided
the input caption “photograph of a girl riding a horse”, our
language model can generate both a plausible edit instruc-
tion “have her ride a dragon” and an appropriately modi-
ﬁed output caption “photograph of a girl riding a dragon”.
Operating in the text domain enables us to generate a large
and diverse collection of edits, while maintaining corre-
spondence between the image changes and text instructions.
Our model is trained by ﬁnetuning GPT-3 on a relatively
small human-written dataset of editing triplets: (1) input
captions, (2) edit instructions, (3) output captions. To pro-
duce the ﬁne-tuning dataset, we sampled 700 input captions
from the LAION-Aesthetics V2 6.5+ [57] dataset and man-
ually wrote instructions and output captions. See Table 1a
for examples of our written instructions and output captions.
Using this data, we ﬁne-tuned the GPT-3 Davinci model for
a single epoch using the default training parameters.
Beneﬁting from GPT-3’s immense knowledge and abil-
ity to generalize, our ﬁnetuned model is able to generate
creative yet sensible instructions and captions. See Table 1b
for example GPT-3 generated data. Our dataset is created by
generating a large number of edits and output captions using
this trained model, where the input captions are real image
captions from LAION-Aesthetics (excluding samples with
duplicate captions or duplicate image URLs). We chose
the LAION dataset due to its large size, diversity of con-
tent (including references to proper nouns and popular cul-
ture), and variety of mediums (photographs, paintings, dig-
ital artwork). A potential drawback of LAION is that it is
quite noisy and contains a number of nonsensical or unde-
scriptive captions—however, we found that dataset noise is
mitigated through a combination of dataset ﬁltering (Sec-
tion 3.1.2) and classiﬁer-free guidance (Section 3.2.1). Our
ﬁnal corpus of generated instructions and captions consists
of 454, 445 examples.
3.1.2
Generating Paired Images from Paired Captions
Next, we use a pretrained text-to-image model to transform
a pair of captions (referring to the image before and after the
edit) into a pair of images. One challenge in turning a pair
of captions into a pair of corresponding images is that text-
to-image models provide no guarantees about image consis-
tency, even under very minor changes of the conditioning
3

Input LAION caption
Edit instruction
Edited caption
Human-written
(700 edits)
Yeﬁm Volkov, Misty Morning
make it afternoon
Yeﬁm Volkov, Misty Afternoon
girl with horse at sunset
change the background to a city
girl with horse at sunset in front of city
painting-of-forest-and-pond
Without the water.
painting-of-forest
...
...
...
GPT-3 generated
(>450,000 edits)
Alex Hill, Original oil painting on can-
vas, Moonlight Bay
in the style of a coloring book
Alex Hill, Original coloring book illustra-
tion, Moonlight Bay
The great elf city of Rivendell, sitting
atop a waterfall as cascades of water
spill around it
Add a giant red dragon
The great elf city of Rivendell, sitting atop a
waterfall as cascades of water spill around
it with a giant red dragon ﬂying overhead
Kate Hudson arriving at the Golden
Globes 2015
make her look like a zombie
Zombie Kate Hudson arriving at the Golden
Globes 2015
...
...
...
Table 1. We label a small text dataset, ﬁnetune GPT-3, and use that ﬁnetuned model to generate a large dataset of text triplets. As the input
caption for both the labeled and generated examples, we use real image captions from LAION. Highlighted text is generated by GPT-3.
(a) Without Prompt-to-Prompt.
(b) With Prompt-to-Prompt.
Figure 3. Pair of images generated using StableDiffusion [52] with
and without Prompt-to-Prompt [17]. For both, the corresponding
captions are “photograph of a girl riding a horse” and “photo-
graph of a girl riding a dragon”.
prompt. For example, two very similar prompts: “a picture
of a cat” and “a picture of a black cat” may produce wildly
different images of cats. This is unsuitable for our purposes,
where we intend to use this paired data as supervision for
training a model to edit images (and not produce a different
random image). We therefore use Prompt-to-Prompt [17],
a recent method aimed at encouraging multiple generations
from a text-to-image diffusion model to be similar. This
is done through borrowed cross attention weights in some
number of denoising steps. Figure 3 shows a comparison of
sampled images with and without Prompt-to-Prompt.
While this greatly helps assimilate generated images,
different edits may require different amounts of change
in image-space.
For instance, changes of larger magni-
tude, such as those which change large-scale image struc-
ture (e.g., moving objects around, replacing with objects of
different shapes), may require less similarity in the gener-
ated image pair. Fortunately, Prompt-to-Prompt has as a
parameter that can control the similarity between the two
images: the fraction of denoising steps p with shared atten-
tion weights. Unfortunately, identifying an optimal value
of p from only the captions and edit text is difﬁcult. We
therefore generate 100 sample pairs of images per caption-
pair, each with a random p ∼U(0.1, 0.9), and ﬁlter these
samples by using a CLIP-based metric: the directional sim-
ilarity in CLIP space as introduced by Gal et al. [14]. This
metric measures the consistency of the change between the
two images (in CLIP space) with the change between the
two image captions. Performing this ﬁltering not only helps
maximize the diversity and quality of our image pairs, but
also makes our data generation more robust to failures of
Prompt-to-Prompt and Stable Diffusion.
3.2. InstructPix2Pix
We use our generated training data to train a conditional
diffusion model that edits images from written instructions.
We base our model on Stable Diffusion, a large-scale text-
to-image latent diffusion model.
Diffusion models [60] learn to generate data samples
through a sequence of denoising autoencoders that estimate
the score [23] of a data distribution (a direction pointing to-
ward higher density data). Latent diffusion [52] improves
the efﬁciency and quality of diffusion models by operat-
ing in the latent space of a pretrained variational autoen-
coder [30] with encoder E and decoder D. For an image
x, the diffusion process adds noise to the encoded latent
z = E(x) producing a noisy latent zt where the noise level
increases over timesteps t ∈T. We learn a network ϵθ that
predicts the noise added to the noisy latent zt given image
conditioning cI and text instruction conditioning cT . We
minimize the following latent diffusion objective:
L = EE(x),E(cI),cT ,ϵ∼N(0,1),t
h
∥ϵ −ϵθ(zt, t, E(cI), cT ))∥2
2
i
(1)
Wang et al. [67] show that ﬁne-tuning a large image dif-
fusion models outperforms training a model from scratch
for image translation tasks, especially when paired training
data is limited. We therefore initialize the weights of our
model with a pretrained Stable Diffusion checkpoint, lever-
4

sT = 3
sT = 7.5
sT = 15
sI =
1.0
sI =
1.2
sI =
1.6
Edit instruction: “Turn him into a cyborg!”
Figure 4. Classiﬁer-free guidance weights over two conditional in-
puts. sI controls similarity with the input image, while sT controls
consistency with the edit instruction.
aging its vast text-to-image generation capabilities. To sup-
port image conditioning, we add additional input channels
to the ﬁrst convolutional layer, concatenating zt and E(cI).
All available weights of the diffusion model are initialized
from the pretrained checkpoints, and weights that operate
on the newly added input channels are initialized to zero.
We reuse the same text conditioning mechanism that was
originally intended for captions to instead take as input the
text edit instruction cT . Additional training details are pro-
vided in the supplemental material.
3.2.1
Classiﬁer-free Guidance for Two Conditionings
Classiﬁer-free diffusion guidance [20] is a method for trad-
ing off the quality and diversity of samples generated by a
diffusion model. It is commonly used in class-conditional
and text-conditional image generation to improve the vi-
sual quality of generated images and to make sampled im-
ages better correspond with their conditioning. Classiﬁer-
free guidance effectively shifts probability mass toward data
where an implicit classiﬁer pθ(c|zt) assigns high likelihood
to the conditioning c. The implementation of classiﬁer-free
guidance involves jointly training the diffusion model for
conditional and unconditional denoising, and combining the
two score estimates at inference time. Training for uncondi-
tional denoising is done by simply setting the conditioning
to a ﬁxed null value c = ∅at some frequency during train-
ing. At inference time, with a guidance scale s ≥1, the
modiﬁed score estimate ˜eθ(zt, c) is extrapolated in the di-
rection toward the conditional eθ(zt, c) and away from the
unconditional eθ(zt, ∅).
˜eθ(zt, c) = eθ(zt, ∅) + s · (eθ(zt, c) −eθ(zt, ∅))
(2)
For our task, the score network eθ(zt, cI, cT ) has two
conditionings: the input image cI and text instruction cT .
We ﬁnd if beneﬁcial to leverage classiﬁer-free guidance
with respect to both conditionings. Liu et al. [38] demon-
strate that a conditional diffusion model can compose score
estimates from multiple different conditioning values. We
apply the same concept to our model with two separate con-
ditioning inputs.
During training, we randomly set only
cI = ∅I for 5% of examples, only cT = ∅T for 5% of ex-
amples, and both cI =∅I and cT =∅T for 5% of examples.
Our model is therefore capable of conditional or uncondi-
tional denoising with respect to both or either conditional
inputs. We introduce two guidance scales, sI and sT , which
can be adjusted to trade off how strongly the generated sam-
ples correspond with the input image and how strongly they
correspond with the edit instruction. Our modiﬁed score
estimate is as follows:
˜eθ(zt, cI, cT ) = eθ(zt, ∅, ∅)
+ sI · (eθ(zt, cI, ∅) −eθ(zt, ∅, ∅))
+ sT · (eθ(zt, cI, cT ) −eθ(zt, cI, ∅))
(3)
In Figure 4, we show the effects of these two parameters
on generated samples. See Appendix B for details of our
classiﬁer-free guidance formulation.
4. Results
We show instruction-based image editing results on a
diverse set of real photographs and artwork, for a vari-
ety of types of edits and instruction wordings. See Fig-
ures 1, 5, 6, 7, 11, 12, 15, 16, 17, 18, and 19 for selected
results. Our model successfully performs many challeng-
ing edits, including replacing objects, changing seasons
and weather, replacing backgrounds, modifying material at-
tributes, converting artistic medium, and a variety of others.
We compare our method qualitatively with a couple re-
cent works, SDEdit [39] and Text2Live [6]. Our model fol-
lows instructions for how to edit the image, but prior works
(including these baseline methods) expect descriptions of
the image (or edit layer). Therefore, we provide them with
the “after-edit” text caption instead of the edit instruction.
We also compare our method quantitatively with SDEdit,
using two metrics measuring image consistency and edit
quality, further described in Section 4.1. Finally, we show
ablations on how the size and quality of generated training
data affect our model’s performance in Section 4.2.
5

Input
“Make it a Modigliani painting”
“Make it a Miro painting”
“Make it an Egyptian sculpture”
“Make it a marble roman sculpture”
Figure 5. Mona Lisa transformed into various artistic mediums.
Input
“Put them in outer space”
“Turn the humans into robots”
Figure 6. The Creation of Adam with new context and subjects (generated at 768 resolution).
“Make it Paris”
“Make it Hong Kong”
“Make it Manhattan”
“Make it Prague”
“Make it evening”
“Put them on roller skates”
“Turn this into 1900s”
“Make it underwater”
“Make it Minecraft”
“Turn this into the space age”
“Make them into Alexander Calder sculptures”
“Make it a Claymation”
Figure 7. The iconic Beatles Abbey Road album cover transformed in a variety of ways.
6

0.00
0.05
0.10
0.15
CLIP Text-Image Direction Similarity
0.6
0.8
1.0
CLIP Image Similarity
Ours
SDEdit (instruction)
SDEdit (caption)
Figure 8. We plot the trade-off between consistency with the input
image (Y-axis) and consistency with the edit (X-axis). For both
metrics, higher is better. For both methods, we ﬁx text guidance
to 7.5, and vary our sI ∈[1.0, 2.2] and SDEdit’s strength (the
amount of denoising) between [0.3, 0.9].
Input
SDEdit-OC [39]
T2L [6]
SDEdit-E [39]
Ours
“Dali Painting of Nimbus Cloud...”
“make it look like a Dali Painting”
“Crowned alias Grace. (Photo by [...]/Netﬂix)”
“add a crown”
“The Road Leads to the Ocean by Ben Heine”
“have the road lead to the ocean”
“Industrial design bedroom furniture...”
“add a bedroom”
Figure 9. Comparison with other editing methods. The input is
transformed either by edit string (last two columns) or the ground-
truth output image caption (middle two columns). We compare our
method against two recent works, SDEdit [39] and Text2Live [6].
We show SDEdit in two conﬁgurations: conditioned on the output
caption (OP) and conditioned on the edit string (E).
4.1. Baseline comparisons
We provide qualitative comparisons with SDEdit [39]
and Text2Live [6], as well as quantitative comparisons with
SDEdit. SDEdit [39] is a technique for editing images with
a pretrained diffusion model, where a partially noised im-
age is passed as input and denoised to produce a new edited
image. We compare with the public Stable Diffusion im-
plementation of SDEdit. Text2Live [6] is a technique for
editing images by generating a color+opacity augmentation
layer, conditioned on a text prompt. We compare with the
public implementation released by the authors.
We compare with both methods qualitatively in Figure 9.
We notice that while SDEdit works reasonably well for
cases where content remains approximately constant and
style is changed, it struggles to preserve identity and iso-
0.00
0.05
0.10
0.15
CLIP Text-Image Direction Similarity
0.8
0.9
1.0
CLIP Image Similarity
Ours
Ours (no CLIP filter)
Ours (10% of data)
Ours (1% of data)
Figure 10. We compare ablated variants of our model (smaller
training dataset, no CLIP ﬁltering) by ﬁxing sT and sweeping val-
ues of sI ∈[1.0, 2.2]. Our proposed conﬁguration performs best.
late individual objects, especially when larger changes are
desired. Additionally, it requires a full output description of
the desired image, rather than an editing instruction. On the
other hand, while Text2Live is able to produce convincing
results for edits involving additive layers, its formulation
limits the categories of edits that it can handle.
Quantitative comparisons with SDEdit are shown in Fig-
ure 8. We plot the tradeoff between two metrics, cosine
similarity of CLIP image embeddings (how much the edited
image agrees with the input image) and the directional CLIP
similarity introduced by [14] (how much the change in text
captions agrees with the change in the images). These are
competing metrics—increasing the degree to which the out-
put images correspond to a desired edit will reduce their
similarity (consistency) with the input image. Still, we ﬁnd
that when comparing our method with SDEdit, our results
have notably higher image consistency (CLIP image simi-
larity) for the same directional similarity values.
4.2. Ablations
In Figure 10, we provide quantitative ablations for both
our choice of dataset size and our dataset ﬁltering approach
described in Section 3.1. We ﬁnd that decreasing the size
of the dataset typically results in decreased ability to per-
form larger (i.e., more signiﬁcant) image edits, instead only
performing subtle or stylistic image adjustments (and thus,
maintaining a high image similarity score, but a low direc-
tional score). Removing the CLIP ﬁltering from our dataset
generation has a different effect: the overall image consis-
tency with the input image is reduced.
We also provide an analysis of the effect of our two
classiﬁer-free guidance scales in Figure 4. Increasing sT
results in a stronger edit applied to the image (i.e., the out-
put agrees more with the instruction), and increasing sI can
help preserve the spatial structure of the input image (i.e.,
the output agrees more with the input image). We ﬁnd that
values of sT in the range 5−10 and values of sI in the range
1−1.5 typically produce the best results. In practice, and for
the results shown in the paper, we ﬁnd it beneﬁcial to adjust
guidance weights for each example to get the best balance
between consistency and edit strength.
7

“Add an eerie 
thunderstorm”
“Turn into an oil 
pastel drawing”
“Insert a train”
“Give it a dark
creepy vibe”
Figure 11. Applying our model recurrently with different instructions results in compounded edits.
Input
Edit instruction: “in a race car video game”
Figure 12. By varying the latent noise, our model can produce many possible image edits for the same input image and instruction.
“Zoom into the image”
“Move it to Mars”
“Color the tie blue”
“Have the people swap places”
Figure 13. Failure cases. Left to right: our model is not capable of performing viewpoint changes, can make undesired excessive changes
to the image, can sometimes fail to isolate the speciﬁed object, and has difﬁculty reorganizing or swapping objects with each other.
5. Discussion
We demonstrate an approach that combines two large
pretrained models, a large language model and a text-to-
image model, to generate a dataset for training a diffusion
model to follow written image editing instructions. While
our method is able to produce a wide variety of compelling
edits to images, including style, medium, and other contex-
tual changes, there still remain a number of limitations.
Our model is limited by the visual quality of the gener-
ated dataset, and therefore by the diffusion model used to
generate the imagery (in this case, Stable Diffusion [52]).
Furthermore, our method’s ability to generalize to new edits
and make correct associations between visual changes and
text instructions is limited by the human-written instruc-
tions used to ﬁne-tune GPT-3 [7], by the ability of GPT-3
to create instructions and modify captions, and by the abil-
ity of Prompt-to-Prompt [17] to modify generated images.
In particular, our model struggles with counting numbers of
objects and with spatial reasoning (e.g., “move it to the left
of the image”, “swap their positions”, or “put two cups on
the table and one on the chair”), just as in Stable Diffusion
and Prompt-to-Prompt. Examples of failures can be found
in Figure 13. Furthermore, there are well-documented bi-
ases in the data and the pretrained models that our method
is based upon, and therefore the edited images from our
method may inherit these biases or introduce other biases
(Figure 14).
Aside from mitigating the above limitations, our work
also opens up questions, such as: how to follow instructions
for spatial reasoning, how to combine instructions with
other conditioning modalities like user interaction, and how
to evaluate instruction-based image editing. Incorporating
human feedback to improve the model is another important
area of future work, and strategies like human-in-the-loop
reinforcement learning could be applied to improve align-
ment between our model and human intentions.
Acknowledgments
We thank Ilija Radosavovic, William
Peebles, Allan Jabri, Dave Epstein, Kﬁr Aberman, Amanda
Buster, and David Salesin. Tim Brooks is funded by an NSF
Graduate Research Fellowship.
Additional funding pro-
vided by a research grant from SAP and a gift from Google.
8

Input
“Make them look like ﬂight attendants”
“Make them look like doctors”
Figure 14. Our method reﬂects biases from the data and models it is based upon, such as correlations between profession and gender.
Input
“Make it a grocery store”
Figure 15. Leighton’s Lady in a Garden moved to a new setting.
Input
“Convert to a realistic photo”
“Turn into a 3D model”
Figure 16. Van Gogh’s Self-Portrait with a Straw Hat in different mediums.
9

Input
“Add boats on the water”
“Replace the mountains with a city skyline”
Figure 17. A landscape photograph shown with different contextual edits. Note that isolated changes also bring along accompanying
contextual effects: the addition of boats also adds wind ripples in the water, and the added city skyline is reﬂected on the lake.
Input
“It is now midnight”
“Add a beautiful sunset”
Figure 18. A photograph of a cityscape edited to show different times of day.
Input
“Apply face paint”
“What would she look like as a
bearded man?”
“Put on a pair of sunglasses”
“She should look 100 years old”
“What if she were in an anime?”
“Make her terrifying”
“Make her more sad”
“Make her James Bond”
“Turn her into Dwayne The Rock
Johnson”
Figure 19. Vermeer’s Girl with a Pearl Earring with a variety of edits.
10

References
[1] Rameen Abdal, Yipeng Qin, and Peter Wonka.
Im-
age2stylegan: How to embed images into the stylegan latent
space? In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision, pages 4432–4441, 2019. 2
[2] Rameen Abdal, Yipeng Qin, and Peter Wonka.
Im-
age2stylegan++: How to edit the embedded images?
In
Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 8296–8305, 2020. 2
[3] Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, and
Amit Bermano.
Hyperstyle: Stylegan inversion with hy-
pernetworks for real image editing.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18511–18521, 2022. 2
[4] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine
Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch,
Katie Millican, Malcolm Reynolds, et al. Flamingo: a vi-
sual language model for few-shot learning. arXiv preprint
arXiv:2204.14198, 2022. 2
[5] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended
diffusion for text-driven editing of natural images. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 18208–18218, 2022. 2
[6] Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kas-
ten, and Tali Dekel. Text2live: Text-driven layered image
and video editing.
In European Conference on Computer
Vision, pages 707–723. Springer, 2022. 2, 5, 7
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Sub-
biah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakan-
tan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Lan-
guage models are few-shot learners. Advances in neural in-
formation processing systems, 33:1877–1901, 2020. 1, 2, 3,
8
[8] Lucy Chai, Jonas Wulff, and Phillip Isola. Using latent space
regression to analyze and leverage compositionality in gans.
In International Conference on Learning Representations,
2021. 2
[9] Katherine Crowson,
Stella
Biderman,
Daniel Kornis,
Dashiell Stander, Eric Hallahan, Louis Castricato, and Ed-
ward Raff. Vqgan-clip: Open domain image generation and
editing with natural language guidance. In European Con-
ference on Computer Vision, pages 88–105. Springer, 2022.
2
[10] Prafulla Dhariwal and Alexander Nichol. Diffusion models
beat gans on image synthesis. Advances in Neural Informa-
tion Processing Systems, 34:8780–8794, 2021. 2
[11] Yilun Du, Shuang Li, and Igor Mordatch. Compositional vi-
sual generation with energy based models. Advances in Neu-
ral Information Processing Systems, 33:6637–6647, 2020. 2
[12] Dave Epstein, Taesung Park, Richard Zhang, Eli Shechtman,
and Alexei A Efros. Blobgan: Spatially disentangled scene
representations. In European Conference on Computer Vi-
sion, pages 616–635. Springer, 2022. 2
[13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patash-
nik, Amit H Bermano, Gal Chechik, and Daniel Cohen-
Or.
An image is worth one word: Personalizing text-to-
image generation using textual inversion.
arXiv preprint
arXiv:2208.01618, 2022. 2
[14] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano,
Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clip-
guided domain adaptation of image generators. ACM Trans-
actions on Graphics (TOG), 41(4):1–13, 2022. 2, 4, 7
[15] Leon A Gatys, Alexander S Ecker, and Matthias Bethge.
A neural algorithm of artistic style.
arXiv preprint
arXiv:1508.06576, 2015. 2
[16] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Im-
age style transfer using convolutional neural networks. In
Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 2414–2423, 2016. 2
[17] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kﬁr Aberman,
Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt im-
age editing with cross attention control.
arXiv preprint
arXiv:2208.01626, 2022. 2, 3, 4, 8, 14
[18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffu-
sion probabilistic models. Advances in Neural Information
Processing Systems, 33:6840–6851, 2020. 2
[19] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet,
Mohammad Norouzi, and Tim Salimans. Cascaded diffusion
models for high ﬁdelity image generation. J. Mach. Learn.
Res., 23:47–1, 2022. 2
[20] Jonathan Ho and Tim Salimans.
Classiﬁer-free diffusion
guidance. arXiv preprint arXiv:2207.12598, 2022. 5
[21] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William
Chan, Mohammad Norouzi, and David J Fleet. Video dif-
fusion models. arXiv preprint arXiv:2204.03458, 2022. 2
[22] Xun Huang, Ming-Yu Liu, Serge Belongie, and Jan Kautz.
Multimodal unsupervised image-to-image translation.
In
ECCV, 2018. 2
[23] Aapo Hyv¨arinen and Peter Dayan.
Estimation of non-
normalized statistical models by score matching.
Journal
of Machine Learning Research, 6(4), 2005. 4, 14
[24] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A
Efros. Image-to-image translation with conditional adver-
sarial networks. CVPR, 2017. 2
[25] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine.
Elucidating the design space of diffusion-based generative
models. arXiv preprint arXiv:2206.00364, 2022. 14
[26] Tero Karras, Samuli Laine, and Timo Aila. A style-based
generator architecture for generative adversarial networks.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 4401–4410, 2019. 2
[27] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten,
Jaakko Lehtinen, and Timo Aila.
Analyzing and improv-
ing the image quality of stylegan.
In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 8110–8119, 2020. 2
[28] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen
Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic:
Text-based real image editing with diffusion models. arXiv
preprint arXiv:2210.09276, 2022. 2
[29] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. Dif-
fusionclip: Text-guided diffusion models for robust image
manipulation. In Proceedings of the IEEE/CVF Conference
11

on Computer Vision and Pattern Recognition, pages 2426–
2435, 2022. 2
[30] Diederik P Kingma and Max Welling. Auto-encoding varia-
tional bayes. arXiv preprint arXiv:1312.6114, 2013. 4
[31] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and
Bryan Catanzaro. Diffwave: A versatile diffusion model for
audio synthesis. arXiv preprint arXiv:2009.09761, 2020. 2
[32] Gihyun Kwon and Jong Chul Ye. Clipstyler: Image style
transfer with a single text condition.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 18062–18071, 2022. 2
[33] Daiqing Li, Huan Ling, Seung Wook Kim, Karsten Kreis,
Sanja Fidler, and Antonio Torralba. Bigdatasetgan: Synthe-
sizing imagenet with pixel-wise annotations. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 21330–21340, 2022. 2
[34] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh,
and Kai-Wei Chang.
Visualbert:
A simple and perfor-
mant baseline for vision and language.
arXiv preprint
arXiv:1908.03557, 2019. 2
[35] Shuang Li, Yilun Du, Joshua B Tenenbaum, Antonio Tor-
ralba, and Igor Mordatch.
Composing ensembles of pre-
trained models via iterative consensus.
arXiv preprint
arXiv:2210.11522, 2022. 2
[36] Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy
Liang, and Tatsunori B Hashimoto.
Diffusion-lm im-
proves controllable text generation.
arXiv preprint
arXiv:2205.14217, 2022. 2
[37] Ming-Yu Liu, Xun Huang, Arun Mallya, Tero Karras, Timo
Aila, Jaakko Lehtinen, and Jan Kautz. Few-shot unsuper-
vised image-to-image translation.
In IEEE International
Conference on Computer Vision (ICCV), 2019. 2
[38] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and
Joshua B Tenenbaum.
Compositional visual genera-
tion with composable diffusion models.
arXiv preprint
arXiv:2206.01714, 2022. 2, 5
[39] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jia-
jun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided
image synthesis and editing with stochastic differential equa-
tions. In International Conference on Learning Representa-
tions, 2021. 2, 5, 7
[40] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Han-
naneh Hajishirzi.
Cross-task generalization via natu-
ral language crowdsourcing instructions.
arXiv preprint
arXiv:2104.08773, 2021. 2
[41] Ron Mokady, Amir Hertz, and Amit H Bermano.
Clip-
cap:
Clip preﬁx for image captioning.
arXiv preprint
arXiv:2111.09734, 2021. 2
[42] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav
Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and
Mark Chen. Glide: Towards photorealistic image generation
and editing with text-guided diffusion models. arXiv preprint
arXiv:2112.10741, 2021. 2
[43] Utkarsh Ojha, Yijun Li, Cynthia Lu, Alexei A. Efros,
Yong Jae Lee, Eli Shechtman, and Richard Zhang.
Few-
shot image generation via cross-domain correspondence. In
CVPR, 2021. 2
[44] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L
Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agar-
wal, Katarina Slama, Alex Ray, et al.
Training language
models to follow instructions with human feedback. arXiv
preprint arXiv:2203.02155, 2022. 2
[45] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or,
and Dani Lischinski.
Styleclip: Text-driven manipulation
of stylegan imagery. In Proceedings of the IEEE/CVF In-
ternational Conference on Computer Vision (ICCV), pages
2085–2094, October 2021. 2
[46] William Peebles, Ilija Radosavovic, Tim Brooks, Alexei A
Efros, and Jitendra Malik. Learning to learn with genera-
tive models of neural network checkpoints. arXiv preprint
arXiv:2209.12892, 2022. 2
[47] William Peebles, Jun-Yan Zhu, Richard Zhang, Antonio Tor-
ralba, Alexei A Efros, and Eli Shechtman. Gan-supervised
dense visual alignment. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 13470–13481, 2022. 2
[48] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In International Conference on Machine Learning,
pages 8748–8763. PMLR, 2021. 2
[49] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu,
and Mark Chen. Hierarchical text-conditional image gen-
eration with clip latents. arXiv preprint arXiv:2204.06125,
2022. 2
[50] Suman Ravuri and Oriol Vinyals.
Classiﬁcation accuracy
score for conditional generative models. Advances in neural
information processing systems, 32, 2019. 2
[51] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan,
Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding
in style: a stylegan encoder for image-to-image translation.
In Proceedings of the IEEE/CVF conference on computer vi-
sion and pattern recognition, pages 2287–2296, 2021. 2
[52] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer.
High-resolution image
synthesis with latent diffusion models.
In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10684–10695, 2022. 1, 2, 3, 4, 8, 14
[53] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch,
Michael Rubinstein, and Kﬁr Aberman. Dreambooth: Fine
tuning text-to-image diffusion models for subject-driven
generation. arXiv preprint arXiv:2208.12242, 2022. 2
[54] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee,
Jonathan Ho, Tim Salimans, David Fleet, and Mohammad
Norouzi.
Palette: Image-to-image diffusion models.
In
ACM SIGGRAPH 2022 Conference Proceedings, pages 1–
10, 2022. 2
[55] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed
Ghasemipour,
Burcu Karagol Ayan,
S Sara Mahdavi,
Rapha Gontijo Lopes, et al.
Photorealistic text-to-image
diffusion models with deep language understanding. arXiv
preprint arXiv:2205.11487, 2022. 2
12

[56] Chitwan Saharia, Jonathan Ho, William Chan, Tim Sali-
mans, David J Fleet, and Mohammad Norouzi. Image super-
resolution via iterative reﬁnement.
IEEE Transactions on
Pattern Analysis and Machine Intelligence, 2022. 2
[57] Christoph Schuhmann, Romain Beaumont, Richard Vencu,
Cade Gordon,
Ross Wightman,
Mehdi Cherti,
Theo
Coombes, Aarush Katta, Clayton Mullis, Mitchell Worts-
man, et al.
Laion-5b:
An open large-scale dataset for
training next generation image-text models. arXiv preprint
arXiv:2210.08402, 2022. 3
[58] Ashish Shrivastava, Tomas Pﬁster, Oncel Tuzel, Joshua
Susskind, Wenda Wang, and Russell Webb.
Learning
from simulated and unsupervised images through adversarial
training. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 2107–2116, 2017. 2
[59] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An,
Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,
Oran Gafni, et al. Make-a-video: Text-to-video generation
without text-video data. arXiv preprint arXiv:2209.14792,
2022. 2
[60] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan,
and Surya Ganguli.
Deep unsupervised learning using
nonequilibrium thermodynamics. In Francis Bach and David
Blei, editors, Proceedings of the 32nd International Con-
ference on Machine Learning, volume 37 of Proceedings
of Machine Learning Research, pages 2256–2265, Lille,
France, 07–09 Jul 2015. PMLR. 2, 4
[61] Yang Song and Stefano Ermon. Generative modeling by esti-
mating gradients of the data distribution. Advances in Neural
Information Processing Systems, 32, 2019. 2
[62] Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf.
Zero-shot image-to-text generation for visual-semantic arith-
metic. arXiv preprint arXiv:2111.14447, 2021. 2
[63] Anthony Meng Huat Tiong, Junnan Li, Boyang Li, Silvio
Savarese, and Steven CH Hoi. Plug-and-play vqa: Zero-shot
vqa by conjoining large pretrained models with zero training.
arXiv preprint arXiv:2210.08773, 2022. 2
[64] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and
Daniel Cohen-Or.
Designing an encoder for stylegan im-
age manipulation. ACM Transactions on Graphics (TOG),
40(4):1–14, 2021. 2
[65] Nontawat Tritrong, Pitchaporn Rewatbowornwong, and Su-
pasorn Suwajanakorn.
Repurposing gans for one-shot se-
mantic part segmentation. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 4475–4485, 2021. 2
[66] Yuri Viazovetskyi, Vladimir Ivashkin, and Evgeny Kashin.
Stylegan2 distillation for feed-forward image manipulation.
In European conference on computer vision, pages 170–186.
Springer, 2020. 2
[67] Tengfei Wang, Ting Zhang, Bo Zhang, Hao Ouyang, Dong
Chen, Qifeng Chen, and Fang Wen.
Pretraining is all
you need for image-to-image translation.
arXiv preprint
arXiv:2205.12952, 2022. 4
[68] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia
Tsvetkov, and Yuan Cao. Simvlm: Simple visual language
model pretraining with weak supervision.
arXiv preprint
arXiv:2108.10904, 2021. 2
[69] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu,
Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and
Quoc V Le. Finetuned language models are zero-shot learn-
ers. arXiv preprint arXiv:2109.01652, 2021. 2
[70] Andy Zeng, Adrian Wong, Stefan Welker, Krzysztof Choro-
manski, Federico Tombari, Aveek Purohit, Michael Ryoo,
Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, et al. So-
cratic models: Composing zero-shot multimodal reasoning
with language. arXiv preprint arXiv:2204.00598, 2022. 2
[71] Wanfeng Zheng, Qiang Li, Xiaoyan Guo, Pengfei Wan,
and Zhongyuan Wang.
Bridging clip and stylegan
through latent alignment for image editing. arXiv preprint
arXiv:2210.04506, 2022. 2
[72] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A
Efros.
Unpaired image-to-image translation using cycle-
consistent adversarial networks. In Computer Vision (ICCV),
2017 IEEE International Conference on, 2017. 2
13

A. Implementation Details
A.1. Instruction and Caption Generation
We ﬁnetune GPT3 to generate edit instructions and
edited captions. The text prompt used during ﬁne-tuning
is the input caption concatenated with "\n##\n" as a
separator token.
The text completion is a concatenation
of the instruction and edited caption with "\n%%\n" as
a separator token in between the two and "\nEND" ap-
pended to the end as the stop token.
During inference,
we sample text completions given new input captions using
temperature=0.7 and frequency penalty=0.1.
We exclude generations where the input and output captions
are the same.
A.2. Paired Image Generation
We generate paired before/after training images from
paired before/after captions using Stable Diffusion [52] in
combination with Prompt-to-Prompt [17]. We use expo-
nential moving average (EMA) weights of the Stable Dif-
fusion v1.5 checkpoint and the improved ft-MSE autoen-
coder weights.
We generate images with 100 denoising
steps using an Euler ancestral sampler with denoising vari-
ance schedule proposed by Kerras et al. [25]. We ensure
the same latent noise is used for both images in each gener-
ated pair (for initial noise as well as noise introduced during
stochastic sampling).
Prompt-to-Prompt replaces cross-attention weights in
the second generated image differently based on the spe-
ciﬁc edit type: word swap, adding a phrase, increasing or
decreasing weight of a word.
We instead replaced self-
attention weights of the second image for the ﬁrst p frac-
tion of steps, and use the same attention weight replacement
strategy for all edits.
We generation 100 pairs of images for each pair of cap-
tions.
We ﬁlter training data for an image-image CLIP
threshold of 0.75 to ensure images are not too different, an
image-caption CLIP threshold of 0.2 to ensure images cor-
respond with their captions, and a directional CLIP similar-
ity of 0.2 to ensure the change in before/after captions cor-
respond with the change in before/after images. For each
each pair of captions, we sort any image pairs that pass all
ﬁlters by the directional CLIP similarity and keep up to 4
examples.
A.3. Training InstructPix2Pix
We train our image editing model for 10,000 steps on
8× 40GB NVIDIA A100 GPUs over 25.5 hours. We train
at 256 × 256 resolution with a total batch size of 1024. We
apply random horizontal ﬂip augmentation and crop aug-
mentation where images are ﬁrst resized randomly between
256 and 288 pixels and then cropped to 256. We use a learn-
ing rate of 10−4 (without any learning rate warm up). We
initialize our model from EMA weights of the Stable Diffu-
sion v1.5 checkpoint, and adopt other training settings from
the public Stable Diffusion code base.
While our model is trained at 256 × 256 resolution, we
ﬁnd it generalized well to 512 × 512 resolution at infer-
ence time, and generate results in this paper at 512 resolu-
tion with 100 denoising steps using an Euler ancestral sam-
pler with denoising variance schedule proposed by Kerras
et al. [25]. Editing an image with our model takes roughly
9 seconds on an A100 GPU.
B. Classiﬁer-free Guidance Details
As discussed in Section 3.2.1, we apply classiﬁer-free
guidance with respect to two conditionings: the input im-
age cI and the text instruction cT . We introduce separate
guidance scales sI and sT that enable separately trading off
the strength of each conditioning. Below is the modiﬁed
score estimate for our model with classiﬁer-free guidance
(copied from Equation 3):
˜eθ(zt, cI, cT ) = eθ(zt, ∅, ∅)
+ sI · (eθ(zt, cI, ∅) −eθ(zt, ∅, ∅))
+ sT · (eθ(zt, cI, cT ) −eθ(zt, cI, ∅))
Our generative model learns P(z|cI, cT ), the probability
distribution of image latents z = E(x) conditioned on an
input image cI and a text instruction cT . We arrive at our
particular classiﬁer-free guidance formulation by express-
ing the conditional probability as follows:
P(z|cT , cI) = P(z, cT , cI)
P(cT , cI)
= P(cT |cI, z)P(cI|z)P(z)
P(cT , cI)
Diffusion models estimate the score [23] of the data dis-
tribution, i.e., the derivative of the log probability. Taking
the logarithm gives us the following expression:
log(P(z|cT , cI)) = log(P(cT |cI, z)) + log(P(cI|z))
+ log(P(z)) −log(P(cT , cI))
Taking the derivative and rearranging we attain:
∇z log(P(z|cT , cI)) = ∇z log(P(z))
+ ∇z log(P(cI|z))
+ ∇z log(P(cT |cI, z))
This corresponds with the terms in our classiﬁer-free
guidance formulation in Equation 3. Our guidance scale
sI effectively shifts probability mass toward data where an
implicit classiﬁer pθ(cI|zt) assigns high likelihood to the
14

image conditioning cI, and our guidance scale sT effec-
tively shifts probability mass toward data where an implicit
classiﬁer pθ(cT |cI, zt) assigns high likelihood to the text
instruction conditioning cT . Our model is capable of learn-
ing these implicit classiﬁers by taking the differences be-
tween estimates with and without the respective conditional
input. Note there are multiple possible formulations such as
switching the positions of cT and cI variables. We found
that our particular decomposition works better for our use
case in practice.
15
