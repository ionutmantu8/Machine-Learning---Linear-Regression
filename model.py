import numpy as np

def compute_cost(X, y, w, b):
    """ Compute cost function for linear regression. """
    m = len(X)
    cost = np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    """ Computes the gradient for linear regression using vectorization. """
    m = len(X)
    error = np.dot(X, w) + b - y
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db

def predict(X, w, b):
    """ Predicts output for given input X using trained parameters. """
    return np.dot(X, w) + b
