# LeHami: Symplectic Latent Dynamics for World Models

**Authors:** Gabriel Yalovetzky, Randall Balestriero, Yann LeCun
**arXiv:** 2610.15267 (2026)

## Abstract

We identify a critical failure mode of unconstrained JEPA world model predictors: predictor contraction. In autoregressive rollout, each prediction step introduces a slight contraction of the latent distribution, which compounds multiplicatively over time, causing gradual collapse — a failure that is invisible during single-step training but devastating during multi-step planning. We introduce LeHami, which replaces the unconstrained predictor with a Hamiltonian flow H(q, p) acting on a structured (q, p) phase space, integrated by an exact symplectic integrator. Proposition 1 proves the symplectic step is an exact symplectic diffeomorphism at any step size and parameters, therefore exactly preserving phase space volume. This excludes three collapse modes: uniform contraction, convergence to asymptotically stable fixed points, and merging of distinct states. The resulting model supports continuous-time integration (any step size from a single trained model) and time reversibility (backward rollout via momentum sign flip). We validate LeHami on pendulum environments and discuss Port-Hamiltonian extensions for dissipative systems.
