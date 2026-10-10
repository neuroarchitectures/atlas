# Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding

> Source: `https://doi.org/https://doi.org/10.18653/v1/2023.emnlp-demo.49`

---

**Authors:** Hang Zhang, Xin Shane Li, Lidong Bing

**Published:** 2023

**Concepts:** Computer science, Encoder, Audio visual, Embedding, Multimedia

## Abstract

We present Video-LLaMA 1 a multi-modal framework that empowers Large Language Models (LLMs) with the capability of understanding both visual and auditory content in the video.Video-LLaMA bootstraps cross-modal training from the frozen pre-trained visual & audio encoders and the frozen LLMs.Unlike previous works that complement LLMs to process the visual or audio signals only (Zhu et al., 2023;Liu et al., 2023; Huang et al., 2023a), Video-LLaMA enables video comprehension by tackling two challenges: (1) capturing the temporal changes in visual scenes, (2) integrating audio-visual signals.To counter the first challenge, we propose a Video Q-former to assemble a pre-trained image encoder into our video encoder and introduce a video-to-text generation task to learn video-language correspondence.For the second challenge, we leverage ImageBind (Girdhar et al., 2023), a universal embedding model aligning multiple modalities, as the pre-trained audio encoder and introduce an Audio Q-former on top of ImageBind to learn reasonable auditory query embeddings for the LLM module.To align the output of both visual & audio encoders with LLM's embedding space, we first train Video-LLaMA on massive video/image-caption pairs and then tune our model with visual-instruction datasets of moderate amount but higher quality.We found Video-LLaMA shows the ability to perceive and comprehend video content and generate meaningful responses grounded in the visual and auditory information presented in the videos.

**DOI:** https://doi.org/https://doi.org/10.18653/v1/2023.emnlp-demo.49
