# Paper (Sanh et al. 2019)

> Source: `https://arxiv.org/abs/1910.01108`

---

DistilBERT, a distilled version of BERT: smaller,
faster, cheaper and lighter
Victor SANH, Lysandre DEBUT, Julien CHAUMOND, Thomas WOLF
Hugging Face
{victor,lysandre,julien,thomas}@huggingface.co
Abstract
As Transfer Learning from large-scale pre-trained models becomes more prevalent
in Natural Language Processing (NLP), operating these large models in on-the-
edge and/or under constrained computational training or inference budgets remains
challenging. In this work, we propose a method to pre-train a smaller general-
purpose language representation model, called DistilBERT, which can then be ﬁne-
tuned with good performances on a wide range of tasks like its larger counterparts.
While most prior work investigated the use of distillation for building task-speciﬁc
models, we leverage knowledge distillation during the pre-training phase and show
that it is possible to reduce the size of a BERT model by 40%, while retaining 97%
of its language understanding capabilities and being 60% faster. To leverage the
inductive biases learned by larger models during pre-training, we introduce a triple
loss combining language modeling, distillation and cosine-distance losses. Our
smaller, faster and lighter model is cheaper to pre-train and we demonstrate its
capabilities for on-device computations in a proof-of-concept experiment and a
comparative on-device study.
1
Introduction
Figure 1: Parameter counts of several recently released
pretrained language models.
The last two years have seen the rise
of Transfer Learning approaches in
Natural Language Processing (NLP)
with large-scale pre-trained language
models becoming a basic tool in
many NLP tasks [Devlin et al., 2018,
Radford et al., 2019, Liu et al., 2019].
While these models lead to signiﬁ-
cant improvement, they often have
several hundred million parameters
and current research1 on pre-trained
models indicates that training even
larger models still leads to better per-
formances on downstream tasks.
The trend toward bigger models
raises several concerns. First is the
environmental cost of exponentially scaling these models’ computational requirements as mentioned
in Schwartz et al. [2019], Strubell et al. [2019]. Second, while operating these models on-device
in real-time has the potential to enable novel and interesting language processing applications, the
growing computational and memory requirements of these models may hamper wide adoption.
1See for instance the recently released MegatronLM (https://nv-adlr.github.io/MegatronLM)
EMC^2: 5th Edition Co-located with NeurIPS’19
arXiv:1910.01108v4  [cs.CL]  1 Mar 2020

In this paper, we show that it is possible to reach similar performances on many downstream-tasks
using much smaller language models pre-trained with knowledge distillation, resulting in models
that are lighter and faster at inference time, while also requiring a smaller computational training
budget. Our general-purpose pre-trained models can be ﬁne-tuned with good performances on several
downstream tasks, keeping the ﬂexibility of larger models. We also show that our compressed models
are small enough to run on the edge, e.g. on mobile devices.
Using a triple loss, we show that a 40% smaller Transformer (Vaswani et al. [2017]) pre-trained
through distillation via the supervision of a bigger Transformer language model can achieve similar
performance on a variety of downstream tasks, while being 60% faster at inference time. Further
ablation studies indicate that all the components of the triple loss are important for best performances.
We have made the trained weights available along with the training code in the Transformers2
library from HuggingFace [Wolf et al., 2019].
2
Knowledge distillation
Knowledge distillation [Bucila et al., 2006, Hinton et al., 2015] is a compression technique in which
a compact model - the student - is trained to reproduce the behaviour of a larger model - the teacher -
or an ensemble of models.
In supervised learning, a classiﬁcation model is generally trained to predict an instance class by
maximizing the estimated probability of gold labels. A standard training objective thus involves
minimizing the cross-entropy between the model’s predicted distribution and the one-hot empirical
distribution of training labels. A model performing well on the training set will predict an output
distribution with high probability on the correct class and with near-zero probabilities on other
classes. But some of these "near-zero" probabilities are larger than others and reﬂect, in part, the
generalization capabilities of the model and how well it will perform on the test set3.
Training loss The student is trained with a distillation loss over the soft target probabilities of
the teacher: Lce = P
i ti ∗log(si) where ti (resp. si) is a probability estimated by the teacher
(resp. the student). This objective results in a rich training signal by leveraging the full teacher
distribution. Following Hinton et al. [2015] we used a softmax-temperature: pi =
exp(zi/T )
P
j exp(zj/T )
where T controls the smoothness of the output distribution and zi is the model score for the class i.
The same temperature T is applied to the student and the teacher at training time, while at inference,
T is set to 1 to recover a standard softmax.
The ﬁnal training objective is a linear combination of the distillation loss Lce with the supervised
training loss, in our case the masked language modeling loss Lmlm [Devlin et al., 2018]. We found it
beneﬁcial to add a cosine embedding loss (Lcos) which will tend to align the directions of the student
and teacher hidden states vectors.
3
DistilBERT: a distilled version of BERT
Student architecture In the present work, the student - DistilBERT - has the same general architec-
ture as BERT. The token-type embeddings and the pooler are removed while the number of layers
is reduced by a factor of 2. Most of the operations used in the Transformer architecture (linear
layer and layer normalisation) are highly optimized in modern linear algebra frameworks and our
investigations showed that variations on the last dimension of the tensor (hidden size dimension) have
a smaller impact on computation efﬁciency (for a ﬁxed parameters budget) than variations on other
factors like the number of layers. Thus we focus on reducing the number of layers.
Student initialization In addition to the previously described optimization and architectural choices,
an important element in our training procedure is to ﬁnd the right initialization for the sub-network to
converge. Taking advantage of the common dimensionality between teacher and student networks,
we initialize the student from the teacher by taking one layer out of two.
2https://github.com/huggingface/transformers
3E.g.
BERT-base’s predictions for a masked token in "I think this is the beginning of a
beautiful [MASK]" comprise two high probability tokens (day and life) and a long tail of valid predictions
(future, story, world...).
2

Table 1: DistilBERT retains 97% of BERT performance. Comparison on the dev sets of the
GLUE benchmark. ELMo results as reported by the authors. BERT and DistilBERT results are the
medians of 5 runs with different seeds.
Model
Score
CoLA
MNLI
MRPC
QNLI
QQP
RTE
SST-2
STS-B
WNLI
ELMo
68.7
44.1
68.6
76.6
71.1
86.2
53.4
91.5
70.4
56.3
BERT-base
79.5
56.3
86.7
88.6
91.8
89.6
69.3
92.7
89.0
53.5
DistilBERT
77.0
51.3
82.2
87.5
89.2
88.5
59.9
91.3
86.9
56.3
Table 2: DistilBERT yields to comparable
performance on downstream tasks. Com-
parison on downstream tasks: IMDb (test ac-
curacy) and SQuAD 1.1 (EM/F1 on dev set).
D: with a second step of distillation during
ﬁne-tuning.
Model
IMDb
SQuAD
(acc.)
(EM/F1)
BERT-base
93.46
81.2/88.5
DistilBERT
92.82
77.7/85.8
DistilBERT (D)
-
79.1/86.9
Table 3: DistilBERT is signiﬁcantly smaller
while being constantly faster.
Inference
time of a full pass of GLUE task STS-B (sen-
timent analysis) on CPU with a batch size of
1.
Model
# param.
Inf. time
(Millions)
(seconds)
ELMo
180
895
BERT-base
110
668
DistilBERT
66
410
Distillation We applied best practices for training BERT model recently proposed in Liu et al. [2019].
As such, DistilBERT is distilled on very large batches leveraging gradient accumulation (up to 4K
examples per batch) using dynamic masking and without the next sentence prediction objective.
Data and compute power We train DistilBERT on the same corpus as the original BERT model:
a concatenation of English Wikipedia and Toronto Book Corpus [Zhu et al., 2015]. DistilBERT
was trained on 8 16GB V100 GPUs for approximately 90 hours. For the sake of comparison, the
RoBERTa model [Liu et al., 2019] required 1 day of training on 1024 32GB V100.
4
Experiments
General Language Understanding We assess the language understanding and generalization ca-
pabilities of DistilBERT on the General Language Understanding Evaluation (GLUE) benchmark
[Wang et al., 2018], a collection of 9 datasets for evaluating natural language understanding systems.
We report scores on the development sets for each task by ﬁne-tuning DistilBERT without the use
of ensembling or multi-tasking scheme for ﬁne-tuning (which are mostly orthogonal to the present
work). We compare the results to the baseline provided by the authors of GLUE: an ELMo (Peters
et al. [2018]) encoder followed by two BiLSTMs.4
The results on each of the 9 tasks are showed on Table 1 along with the macro-score (average of
individual scores). Among the 9 tasks, DistilBERT is always on par or improving over the ELMo
baseline (up to 19 points of accuracy on STS-B). DistilBERT also compares surprisingly well to
BERT, retaining 97% of the performance with 40% fewer parameters.
4.1
Downstream task benchmark
Downstream tasks We further study the performances of DistilBERT on several downstream tasks
under efﬁcient inference constraints: a classiﬁcation task (IMDb sentiment classiﬁcation - Maas et al.
[2011]) and a question answering task (SQuAD v1.1 - Rajpurkar et al. [2016]).
As shown in Table 2, DistilBERT is only 0.6% point behind BERT in test accuracy on the IMDb
benchmark while being 40% smaller. On SQuAD, DistilBERT is within 3.9 points of the full BERT.
We also studied whether we could add another step of distillation during the adaptation phase by
ﬁne-tuning DistilBERT on SQuAD using a BERT model previously ﬁne-tuned on SQuAD as a
4We use jiant [Wang et al., 2019] to compute the baseline.
3

Table 4: Ablation study. Variations are relative to the model trained with triple loss and teacher
weights initialization.
Ablation
Variation on GLUE macro-score
∅- Lcos - Lmlm
-2.96
Lce - ∅- Lmlm
-1.46
Lce - Lcos - ∅
-0.31
Triple loss + random weights initialization
-3.69
teacher for an additional term in the loss (knowledge distillation). In this setting, there are thus two
successive steps of distillation, one during the pre-training phase and one during the adaptation phase.
In this case, we were able to reach interesting performances given the size of the model: 79.8 F1 and
70.4 EM, i.e. within 3 points of the full model.
Size and inference speed
To further investigate the speed-up/size trade-off of DistilBERT, we compare (in Table 3) the number
of parameters of each model along with the inference time needed to do a full pass on the STS-
B development set on CPU (Intel Xeon E5-2690 v3 Haswell @2.9GHz) using a batch size of 1.
DistilBERT has 40% fewer parameters than BERT and is 60% faster than BERT.
On device computation We studied whether DistilBERT could be used for on-the-edge applications
by building a mobile application for question answering. We compare the average inference time on
a recent smartphone (iPhone 7 Plus) against our previously trained question answering model based
on BERT-base. Excluding the tokenization step, DistilBERT is 71% faster than BERT, and the whole
model weighs 207 MB (which could be further reduced with quantization). Our code is available5.
4.2
Ablation study
In this section, we investigate the inﬂuence of various components of the triple loss and the student
initialization on the performances of the distilled model. We report the macro-score on GLUE. Table 4
presents the deltas with the full triple loss: removing the Masked Language Modeling loss has little
impact while the two distillation losses account for a large portion of the performance.
5
Related work
Task-speciﬁc distillation Most of the prior works focus on building task-speciﬁc distillation se-
tups. Tang et al. [2019] transfer ﬁne-tune classiﬁcation model BERT to an LSTM-based classiﬁer.
Chatterjee [2019] distill BERT model ﬁne-tuned on SQuAD in a smaller Transformer model previ-
ously initialized from BERT. In the present work, we found it beneﬁcial to use a general-purpose
pre-training distillation rather than a task-speciﬁc distillation. Turc et al. [2019] use the original
pretraining objective to train smaller student, then ﬁne-tuned via distillation. As shown in the abla-
tion study, we found it beneﬁcial to leverage the teacher’s knowledge to pre-train with additional
distillation signal.
Multi-distillation Yang et al. [2019] combine the knowledge of an ensemble of teachers using
multi-task learning to regularize the distillation. The authors apply Multi-Task Knowledge Distillation
to learn a compact question answering model from a set of large question answering models. An
application of multi-distillation is multi-linguality: Tsai et al. [2019] adopts a similar approach to us
by pre-training a multilingual model from scratch solely through distillation. However, as shown in
the ablation study, leveraging the teacher’s knowledge with initialization and additional losses leads
to substantial gains.
Other compression techniques have been studied to compress large models. Recent developments
in weights pruning reveal that it is possible to remove some heads in the self-attention at test time
without signiﬁcantly degrading the performance Michel et al. [2019]. Some layers can be reduced
to one head. A separate line of study leverages quantization to derive smaller models (Gupta et al.
[2015]). Pruning and quantization are orthogonal to the present work.
5https://github.com/huggingface/swift-coreml-transformers
4

6
Conclusion and future work
We introduced DistilBERT, a general-purpose pre-trained version of BERT, 40% smaller, 60% faster,
that retains 97% of the language understanding capabilities. We showed that a general-purpose
language model can be successfully trained with distillation and analyzed the various components
with an ablation study. We further demonstrated that DistilBERT is a compelling option for edge
applications.
References
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional
transformers for language understanding. In NAACL-HLT, 2018.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are
unsupervised multitask learners. 2019.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar S. Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke S. Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. ArXiv,
abs/1907.11692, 2019.
Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green ai. ArXiv, abs/1907.10597, 2019.
Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in
nlp. In ACL, 2019.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser,
and Illia Polosukhin. Attention is all you need. In NIPS, 2017.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac,
Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Transformers: State-of-the-art natural language
processing, 2019.
Cristian Bucila, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In KDD, 2006.
Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. ArXiv,
abs/1503.02531, 2015.
Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja
Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading
books. 2015 IEEE International Conference on Computer Vision (ICCV), pages 19–27, 2015.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A
multi-task benchmark and analysis platform for natural language understanding. In ICLR, 2018.
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke
Zettlemoyer. Deep contextualized word representations. In NAACL, 2018.
Alex Wang, Ian F. Tenney, Yada Pruksachatkun, Katherin Yu, Jan Hula, Patrick Xia, Raghu Pappagari, Shuning
Jin, R. Thomas McCoy, Roma Patel, Yinghui Huang, Jason Phang, Edouard Grave, Najoung Kim, Phu Mon
Htut, Thibault F’evry, Berlin Chen, Nikita Nangia, Haokun Liu, Anhad Mohananey, Shikha Bordia, Nicolas
Patry, Ellie Pavlick, and Samuel R. Bowman. jiant 1.1: A software toolkit for research on general-purpose
text understanding models. http://jiant.info/, 2019.
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning
word vectors for sentiment analysis. In ACL, 2011.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100, 000+ questions for machine
comprehension of text. In EMNLP, 2016.
Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin. Distilling task-speciﬁc
knowledge from bert into simple neural networks. ArXiv, abs/1903.12136, 2019.
Debajyoti Chatterjee. Making neural machine reading comprehension faster. ArXiv, abs/1904.00796, 2019.
Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: The impact
of student initialization on knowledge distillation. ArXiv, abs/1908.08962, 2019.
Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, and Daxin Jiang. Model compression with multi-task knowledge
distillation for web-scale question answering system. ArXiv, abs/1904.09636, 2019.
Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, and Amelia Archer. Small and practical
bert models for sequence labeling. In EMNLP-IJCNLP, 2019.
Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In NeurIPS, 2019.
Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited
numerical precision. In ICML, 2015.
5
