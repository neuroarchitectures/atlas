# Group Relative Policy Optimization (GRPO)

## Design Philosophy

PPO's value network doubles memory and adds a noisy baseline. GRPO drops the critic entirely: sample G outputs per prompt, use the *group mean reward as the baseline*, and optimize the clipped policy ratio against the advantage `r_i − mean(r)` — critic-free RL fine-tuning, purpose-built for verifiable rewards.

## Functionality

- Per prompt: sample G outputs, compute rewards, advantage A_i = (r_i − mean)/std.
- PPO-style clipped surrogate + KL to reference (k3 estimator); no value network.

## Used By

| Model | Role |
|-------|------|
| DeepSeekMath | 7B policy, G samples per prompt — the RL stage of the math pipeline |
| DeepSeek-R1 lineage | Scaled GRPO for reasoning RL |

## Features

- **Critic-free** — roughly half the RL memory of PPO.
- **Group baseline** — robust without a learned value function when rewards are cheap to sample.

## Evolution

- **Predecessor**: PPO (rlhf-ppo-pipeline); REINFORCE with baselines.
- **Successor**: DAPO, Dr.GRPO — bias fixes in the group normalization.
