# LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

**Authors:** Florian Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero
**arXiv:** 2603.19312 (2026)

## Abstract

We introduce LeWorldModel (LeWM), the first Joint-Embedding Predictive Architecture (JEPA) world model that trains end-to-end from raw pixels in a stable manner. Prior end-to-end JEPA world models required 6 or more loss terms with careful hyperparameter balancing. LeWM uses only two losses: next-embedding prediction (MSE) and SIGReg regularization (isotropic Gaussian distribution matching via Cramér-Wold sketching). With no pretrained encoder, no EMA teacher, no stop-gradient, and no auxiliary supervision, LeWM trains in hours on a single GPU with 15M parameters, achieves competitive planning performance across 2D and 3D control tasks at 48× the speed of baseline world models, and spontaneously encodes physical structure in its latent space — verified via physical quantity probing and surprise evaluation.
