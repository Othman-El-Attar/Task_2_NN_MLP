"""MLP feedforward using lists of matrices (as on the whiteboard)."""

import numpy as np

from preprocessing import load_data, split_all_classes


def feedforward(x, weights, biases):
    """x: one sample (1x5). weights: [5x4, 4x2, 2x3]. Returns list of outputs."""
    outputs = []
    for i in range(len(weights)):                     # hidden_layers + 1
        if i == 0:
            z = x @ weights[0]                        # (1x5)(5x4) = 1x4
        else:
            z = outputs[i - 1] @ weights[i]           # (1x4)(4x2) = 1x2 ...
        z = z + biases[i]
        outputs.append(1 / (1 + np.exp(-z)))          # sigmoid
    return outputs


if __name__ == "__main__":
    df = load_data()
    X_train, y_train, X_test, y_test = split_all_classes(df, seed=1)

    rng = np.random.default_rng(1)
    weights = [rng.uniform(-0.5, 0.5, (5, 4)),
               rng.uniform(-0.5, 0.5, (4, 2)),
               rng.uniform(-0.5, 0.5, (2, 3))]
    biases = [np.zeros((1, 4)), np.zeros((1, 2)), np.zeros((1, 3))]

    outputs = feedforward(X_train[0:1], weights, biases)
    print("weight shapes:", [w.shape for w in weights])
    print("output shapes:", [o.shape for o in outputs])
    print("final output :", outputs[-1])