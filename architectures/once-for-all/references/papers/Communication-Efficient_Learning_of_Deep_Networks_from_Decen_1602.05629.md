# Communication-Efficient Learning of Deep Networks from Decentralized\n Data

> Source: `https://arxiv.org/abs/1602.05629`

---

**Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas

**Published:** 2016

**Concepts:** Computer science, Stochastic gradient descent, Federated learning, Constraint (computer-aided design), Principal (computer security)

## Abstract

Modern mobile devices have access to a wealth of data suitable for learning\nmodels, which in turn can greatly improve the user experience on the device.\nFor example, language models can improve speech recognition and text entry, and\nimage models can automatically select good photos. However, this rich data is\noften privacy sensitive, large in quantity, or both, which may preclude logging\nto the data center and training there using conventional approaches. We\nadvocate an alternative that leaves the training data distributed on the mobile\ndevices, and learns a shared model by aggregating locally-computed updates. We\nterm this decentralized approach Federated Learning.\n We present a practical method for the federated learning of deep networks\nbased on iterative model averaging, and conduct an extensive empirical\nevaluation, considering five different model architectures and four datasets.\nThese experiments demonstrate the approach is robust to the unbalanced and\nnon-IID data distributions that are a defining characteristic of this setting.\nCommunication costs are the principal constraint, and we show a reduction in\nrequired communication rounds by 10-100x as compared to synchronized stochastic\ngradient descent.\n

**PDF:** https://arxiv.org/pdf/1602.05629