
import random
import numpy as np

n_features = 10
W = np.random.uniform(-1, 1, n_features)
X = np.random.randn(n_features)
y = 1  # Single training point for simplicity

_lambda = .1


def J(w, X, y):
    margin = 1 - y*np.dot(w, X)
    loss = np.maximum(0, margin)**2
    reg = (_lambda/2) * np.sum(w**2)
    return ((1/2) * loss) + reg

def grad_J(w, X, y):
    margin = 1 - y*np.dot(w, X)
    if margin > 0:
        return -y*X*np.maximum(0, margin)
    else:
        return np.zeros_like(w)


alpha_candidates = np.linspace(0, 1, 20)

print(W)
for j, _ in enumerate(W):
    # for each weight, just update this weight
    for t in range(1000):
        w_t = W.copy()
        g_t_all = -grad_J(w_t, X, y)
        g_t = g_t_all[j]  # just grab the gradient for this weight w

        losses = [
            J(w_t + alpha * np.eye(1, len(W), j).flatten() * g_t, X, y)
            for alpha in alpha_candidates
        ]
        alpha_t = alpha_candidates[np.argmin(losses)]

        # we are only updating just this weight
        W[j] += (alpha_t*g_t)


print('------------')
print(W)
print(np.dot(W, X))


