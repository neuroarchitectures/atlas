# Paper (Diao et al. 2020)

> Source: `https://doi.org/https://doi.org/10.18653/v1/2020.findings-emnlp.425`

---

**ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations**

Shizhe Diao, Jiaxin Bai, Yan Song, Tong Zhang, Yonggang Wang

**Abstract**

The pre-training of text encoders normally processes text as a sequence of tokens corresponding to small text units, such as word pieces in English and characters in Chinese.It omits information carried by larger text granularity, and thus the encoders cannot easily adapt to certain combinations of characters.This leads to a loss of important semantic information, which is especially problematic for Chinese because the language does not have explicit word boundaries.In this paper, we propose ZEN, a BERT-based Chinese (Z) text encoder Enhanced by N-gram representations, where different combinations of characters are considered during training, thus potential word or phrase boundaries are explicitly pre-trained and fine-tuned with the character encoder (BERT).Therefore ZEN incorporates the comprehensive information of both the character sequence and words or phrases it contains.Experimental results illustrated the effectiveness of ZEN on a series of Chinese NLP tasks, where state-of-the-art results is achieved on most tasks with requiring less resource than other published encoders.It is also shown that reasonable performance is obtained when ZEN is trained on a small corpus, which is important for applying pre-training techniques to scenarios with limited data. 1

---

- **arXiv ID**: `N/A`
- **Year**: 2020
- **DOI**: https://doi.org/10.18653/v1/2020.findings-emnlp.425
- **PDF**: N/A
