# RLHF PPO Pipeline

## Design Philosophy

Pretraining optimizes likelihood, not helpfulness. The RLHF recipe post-trains in three stages: SFT on demonstrations → train a reward model on human preference pairs → PPO fine-tuning of the policy against the reward, with a KL penalty to the SFT model keeping the policy from reward-hacking its way into gibberish.

## Functionality

- Stage 1: SFT on demonstrations. Stage 2: reward model r(x, y) trained on pairwise preferences (Bradley-Terry). Stage 3: PPO with reward + KL(π‖π_SFT) penalty; value network alongside the policy.
- InstructGPT is the reference implementation; Gemini/GPT-4 document the same pipeline as their alignment stage.

## Used By

| Model | Role |
|-------|------|
| InstructGPT | The canonical 3-stage pipeline |
| GPT-4 | RLHF post-training node |
| Gemini | RLHF with KL penalty alignment recipe |

## Features

- **Preference supervision** — humans compare, they don't write objectives.
- **KL leash** — the stabilizer that makes RL on LMs trainable at all.

## Evolution

- **Predecessor**: behavioral cloning; back-and-forth RL fine-tuning (summarization).
- **Successor**: DPO and its variants — closed-form preference optimization without an RL loop.
