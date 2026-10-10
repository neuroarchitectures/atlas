# Paper (Zhou et al. 2026)

> Source: `https://arxiv.org/abs/2609.32869`

---

**Whisper-Flash: Acoustically Conditioned Parallel Drafting for Faster Whisper Decoding**

Huapeng Zhou, Huayu Wang, Junkai Wu, Kangqi Wang, Xinyu Wang

**Abstract**

Whisper is a widely used encoder-decoder model for speech recognition. Its encoder reads an utterance in one parallel pass, but its decoder writes the transcript one token at a time, which dominates inference time. Speculative decoding shortens such loops without changing their output: a small drafter guesses several upcoming tokens, and the original model verifies them all in one forward pass. We present Whisper-Flash, a two-layer drafter built on a property of speech recognition: the words still to be written have already been spoken. It reads Whisper's encoded audio and accepted decoder states and proposes eight tokens in a single forward pass. On the complete LibriSpeech test sets, Whisper-Flash processes $3.16\times/2.85\times$ as much audio per second as greedy decoding with identical outputs, and it remains faster at batch sizes up to 96 and under temperature sampling. Ablations show that direct access to the audio matters most.

---

- **arXiv ID**: `2609.32869`
- **Published**: 2026-09-26
- **Categories**: cs.SD, eess.AS
- **PDF**: https://arxiv.org/pdf/2609.32869
