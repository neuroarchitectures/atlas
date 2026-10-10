# V-Trace (IMPALA)

## Design Philosophy

In distributed RL, actor policies lag behind the learner, so collected trajectories are off-policy by the time they are used. V-trace corrects for this lag with a truncated importance-sampling correction, keeping updates close to on-policy while allowing fully asynchronous actors.

## Functionality

For each trajectory step `t`, define importance ratios between the behaviour policy `mu` and the target policy `pi` (and between the actor and learner policies). The value target is a recursion with truncated, clipped ratios:

```
v_s = V(s) + delta_s + gamma * c_s * ( v_{s+1} - V(s+1) )
where delta_s = rho_s * ( r_s + gamma * V(s+1) - V(s) )
c_s = min( cbar, pi(a|s)/mu(a|s) )
rho_s = min( rhobar, pi(a|s)/mu(a|s) )
```

The truncation levels `cbar`, `rhobar` bound the correction's variance.

## Used By

IMPALA

## Features

- **Off-policy correction**: uses lagged actor data without bias.
- **Truncated IS**: clips ratios to control variance.
- **Asynchronous-friendly**: actors and learner run independently.
- **Policy-gradient + value targets**: gives both actor and critic updates.

## Evolution

Predecessor: A3C (on-policy, synchronous-ish), importance sampling in off-policy PG. Sibling: GAE (on-policy advantage estimation), GRPO. Successor: distributed actor-learner RL (IMPALA, R2D2).
