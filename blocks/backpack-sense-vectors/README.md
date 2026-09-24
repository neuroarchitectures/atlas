# Backpack Sense Vectors

## Design Philosophy

A single embedding per word entangles all its meanings. Backpack-LM assigns each vocabulary word k non-contextual *sense vectors*, and the contextualizer (a transformer) predicts non-negative weights mixing them: `o_i = Σ_j Σ_ℓ α_ℓij · C(x_j)_ℓ` — an interpretable, intervenable decomposition where you can read off and edit which senses a prediction used.

## Functionality

- C: V → ℝ^{k×d} sense table; Transformer contextualizer → α weights over (word, sense) pairs; log-linear output head.
- 170M parameters (GPT-2-small scale); non-negativity keeps senses attributable.

## Used By

| Model | Role |
|-------|------|
| Backpack Language Model | Sense-mixing LM head and embedding structure |

## Features

- **Interpretability by construction** — sense contributions are explicit weights.
- **Intervention-friendly** — knock out a sense and observe the effect.

## Evolution

- **Predecessor**: multi-sense embeddings (MSSG), contextualized embeddings (ELMo/BERT — implicit senses).
- **Related**: mixture-of-experts — routing over experts vs routing over senses; interpretable-by-design LM line.
