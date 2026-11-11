# Out-of-distribution detection with the Maximum Probability Theorem

Opened 2022-03-02.

Training code for [Out of Distribution Detection, Generalization, and Robustness Triangle with Maximum Probability Theorem](https://arxiv.org/abs/2203.12145).

The theorem bounds the probability of a model by the distribution that model places on its variables. This tree uses that bound as regularization for CNNs and their energy-based variants.

`run.py` sweeps the regularization strength (alpha), learning rate, batch size, depth, and loss on CIFAR-10, CIFAR-100, and MNIST. The losses are conditional cross-entropy, joint cross-entropy, and a joint probabilistic loss. The networks are the CIFAR models in `src/`.
