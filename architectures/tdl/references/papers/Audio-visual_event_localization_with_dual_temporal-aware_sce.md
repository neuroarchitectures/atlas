# Audio-visual event localization with dual temporal-aware scene understanding and image-text knowledge bridging

> Source: `https://doi.org/https://doi.org/10.1007/s40747-024-01654-2`

---

**Authors:** Pufen Zhang, Jiaxiang Wang, Meng Wan, Song Zhang, Jie Jing, Lianhong Ding, Peng Shi

**Published:** 2024

**Concepts:** Bridging (networking), Audio visual, Dual (grammatical number), Computer science, Computational intelligence

## Abstract

Audio-visual event localization (AVEL) task aims to judge and classify an audible and visible event. Existing methods devote to this goal by transferring pre-trained knowledge as well as understanding temporal dependencies and cross-modal correlations of the audio-visual scene. However, most works comprehend the audio-visual scene from an entangled temporal-aware perspective, ignoring the learning of temporal dependency and cross-modal correlation in both forward and backward temporal-aware views. Recently, transferring the pre-trained knowledge from Contrastive Language-Image Pre-training model (CLIP) has shown remarkable results across various tasks. Nevertheless, since audio-visual knowledge of the AVEL task and image-text alignment knowledge of the CLIP exist heterogeneous gap, how to transfer the image-text alignment knowledge of CLIP into AVEL field has barely been investigated. To address these challenges, a novel Dual Temporal-aware scene understanding and image-text Knowledge Bridging (DTKB) model is proposed in this paper. DTKB consists of forward and backward temporal-aware scene understanding streams, in which temporal dependencies and cross-modal correlations are explicitly captured from dual temporal-aware perspectives. Consequently, DTKB can achieve fine-grained scene understanding for event localization. Additionally, a knowledge bridging (KB) module is proposed to simultaneously transfer the image-text representation and alignment knowledge of CLIP to AVEL task. This module regulates the ratio between audio-visual fusion features and CLIP’s visual features, thereby bridging the image-text alignment knowledge of CLIP and the audio-visual new knowledge for event category prediction. Besides, the KB module is compatible with previous models. Extensive experimental results demonstrate that DTKB significantly outperforms the state-of-the-arts models.

**DOI:** https://doi.org/https://doi.org/10.1007/s40747-024-01654-2
