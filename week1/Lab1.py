import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1])  # OR

Xa = np.hstack([np.ones((4, 1)), X])

eta = 1
w = np.zeros(Xa.shape[1])

def angle(w, x):
    cos_a = w @ x / (np.linalg.norm(w) * np.linalg.norm(x))
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

for epoch in range(1, 21):
    mistakes = 0

    for x, t in zip(Xa, y):
        z = w @ x
        y_hat = 1 if z >= 0 else 0
        e = t - y_hat

        if e != 0:
            a_old = angle(w, x)
            w = w + eta * e * x

            print(
                f"epoch {epoch}: x={x[1:]}, y={t}, "
                f"angle {a_old:.1f} -> {angle(w, x):.1f}, w={w}"
            )

            mistakes += 1

    if mistakes == 0:
        print(f"Converged after {epoch} epochs, w={w}")
        break
else:
    print("No convergence in 20 epochs: data not linearly separable?")

# Output :
#  cos_a = w @ x / (np.linalg.norm(w) * np.linalg.norm(x))
# epoch 1: x=[0. 0.], y=0, angle nan -> 180.0, w=[-1.  0.  0.]
# epoch 1: x=[0. 1.], y=1, angle 135.0 -> 45.0, w=[0. 0. 1.]
# epoch 2: x=[0. 0.], y=0, angle 90.0 -> 135.0, w=[-1.  0.  1.]
# epoch 2: x=[1. 0.], y=1, angle 120.0 -> 60.0, w=[0. 1. 1.]
# epoch 3: x=[0. 0.], y=0, angle 90.0 -> 125.3, w=[-1.  1.  1.]
# Converged after 4 epochs, w=[-1.  1.  1.]