# Out-of-distribution detection with the Maximum Probability Theorem

Opened 2022-03-02.

Training code for the generalization paper listed below.

The theorem bounds the probability of a model by the distribution that model places on its variables. This tree uses that bound as regularization for CNNs and their energy-based variants.

`run.py` sweeps the regularization strength (alpha), learning rate, batch size, depth, and loss on CIFAR-10, CIFAR-100, and MNIST. The losses are conditional cross-entropy, joint cross-entropy, and a joint probabilistic loss. The networks are the CIFAR models in `src/`.

## Papers

- Amir Emad Marvasti, Ehsan Emad Marvasti, Ulas Bagci, and Hassan Foroosh. [Maximum Probability Theorem: A Framework for Probabilistic Machine Learning](https://www.computer.org/csdl/journal/ai/2021/03/09446965/1ua0q5D18Yg). *IEEE Transactions on Artificial Intelligence*, 2021.
- Amir Emad Marvasti, Ehsan Emad Marvasti, G. Atia, Ulas Bagci, and Hassan Foroosh. [Rediscovering Deep Neural Networks Through Finite-State Distributions](https://arxiv.org/abs/1809.10073). *IEEE Access*.
- Amir Emad Marvasti, Ehsan Emad Marvasti, and Ulas Bagci. [Out of Distribution Detection, Generalization, and Robustness Triangle with Maximum Probability Theorem](https://arxiv.org/abs/2203.12145). *ICECCME*, 2022.
