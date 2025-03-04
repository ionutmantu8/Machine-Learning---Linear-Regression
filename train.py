import numpy as np
from model import compute_cost, compute_gradient

def gradient_descent(X, y, w, b, alpha, num_iters):
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(compute_cost(X, y, w, b))
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]:.2f}")
    return w, b

def train_model(X, y, alpha=0.01, iterations=1000):
    w = np.zeros(X.shape[1])
    b = 0.
    return gradient_descent(X, y, w, b, alpha, iterations)
