import numpy as np
from sklearn.datasets import make_moons

# ---- Data ----
X, y = make_moons(n_samples=400, noise=0.20, random_state=1)
y = y.reshape(-1, 1).astype(float)

# Standardize
X = (X - X.mean(axis=0)) / X.std(axis=0)


# ---- Helpers ----
def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return (z > 0).astype(float)


def sig(z):
    return 1 / (1 + np.exp(-z))


# ---- Parameter initialization ----
def init(sizes, seed=0):
    rng = np.random.default_rng(seed)
    P = []

    for a, b in zip(sizes[:-1], sizes[1:]):
        # He initialization
        W = rng.normal(
            0,
            np.sqrt(2 / a),
            size=(a, b)
        )
        b_bias = np.zeros((1, b))

        P.append([W, b_bias])

    return P


# ---- Forward + Backward ----
def grads(P, Xb, yb):
    zs = []
    acts = [Xb]
    a = Xb

    # Forward pass
    for i, (W, b) in enumerate(P):
        z = a @ W + b
        zs.append(z)

        if i == len(P) - 1:
            a = sig(z)
        else:
            a = relu(z)

        acts.append(a)

    # Backward pass
    m = Xb.shape[0]
    g = [None] * len(P)

    # BCE loss + sigmoid output
    dz = (acts[-1] - yb) / m

    for i in reversed(range(len(P))):
        dW = acts[i].T @ dz
        db = dz.sum(axis=0, keepdims=True)

        g[i] = [dW, db]

        if i > 0:
            dz = (dz @ P[i][0].T) * drelu(zs[i - 1])

    return g


# ---- Accuracy ----
def accuracy(P, X, y):
    a = X

    for i, (W, b) in enumerate(P):
        z = a @ W + b

        if i == len(P) - 1:
            a = sig(z)
        else:
            a = relu(z)

    return np.mean((a > 0.5) == (y > 0.5))


# ---- Hyperparameters ----
sizes = [2, 16, 16, 1]
lr = 0.5
EPOCHS = 200


# ============================================================
# (A) Batch Gradient Descent
#     All data -> 1 update per epoch
# ============================================================

P = init(sizes, seed=0)

for epoch in range(EPOCHS):
    g = grads(P, X, y)

    for k in range(len(P)):
        P[k][0] -= lr * g[k][0]   # Update weights
        P[k][1] -= lr * g[k][1]   # Update biases

print("Batch GD accuracy:", accuracy(P, X, y))


# ============================================================
# (B) Stochastic Gradient Descent / Mini-batch SGD
#     Batch size = 16
# ============================================================

P = init(sizes, seed=0)
B = 16
idx = np.arange(len(X))

for epoch in range(EPOCHS):

    # Shuffle data every epoch
    np.random.shuffle(idx)

    # Mini-batches
    for s in range(0, len(X), B):
        b = idx[s:s + B]

        g = grads(P, X[b], y[b])

        for k in range(len(P)):
            P[k][0] -= lr * g[k][0]   # Update weights
            P[k][1] -= lr * g[k][1]   # Update biases

print("SGD accuracy:", accuracy(P, X, y))
