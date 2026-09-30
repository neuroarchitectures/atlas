# Paper (Liu et al. 2022)

> Source: `https://arxiv.org/abs/2209.11885`

---

**Physics-Informed Graph Neural Network for Spatial-temporal Production Forecasting**

Wendi Liu, Michael J. Pyrcz

**Abstract**

Production forecast based on historical data provides essential value for developing hydrocarbon resources. Classic history matching workflow is often computationally intense and geometry-dependent. Analytical data-driven models like decline curve analysis (DCA) and capacitance resistance models (CRM) provide a grid-free solution with a relatively simple model capable of integrating some degree of physics constraints. However, the analytical solution may ignore subsurface geometries and is appropriate only for specific flow regimes and otherwise may violate physics conditions resulting in degraded model prediction accuracy. Machine learning-based predictive model for time series provides non-parametric, assumption-free solutions for production forecasting, but are prone to model overfit due to training data sparsity; therefore may be accurate over short prediction time intervals. We propose a grid-free, physics-informed graph neural network (PI-GNN) for production forecasting. A customized graph convolution layer aggregates neighborhood information from historical data and has the flexibility to integrate domain expertise into the data-driven model. The proposed method relaxes the dependence on close-form solutions like CRM and honors the given physics-based constraints. Our proposed method is robust, with improved performance and model interpretability relative to the conventional CRM and GNN baseline without physics constraints.

---

- **arXiv ID**: `2209.11885`
- **Published**: 2022-09-23
- **Categories**: cs.LG, physics.app-ph
- **PDF**: https://arxiv.org/pdf/2209.11885
