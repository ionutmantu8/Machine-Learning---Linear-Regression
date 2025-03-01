import numpy as np
from utils import zscore_normalize_features
from model import predict
from train import train_model

# Training data
X_train = np.array([[2104, 5, 1, 45],
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Inversează vârsta și numărul de etaje
X_train[:, 3] = -X_train[:, 3]
X_train[:, 2] = -X_train[:, 2]

# Normalize features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
y_train_norm, y_mu, y_sigma = zscore_normalize_features(y_train.reshape(-1, 1))
y_train_norm = y_train_norm.flatten()

# Train model
w_final, b_final = train_model(X_norm, y_train_norm)

# Predict on new data
X_pred = np.array([[1574, 5, 1, 45]])
X_pred[:, 3] = -X_pred[:, 3]
X_pred[:, 2] = -X_pred[:, 2]
X_pred_norm = (X_pred - X_mu) / X_sigma

y_pred_norm = predict(X_pred_norm, w_final, b_final)
y_pred = y_pred_norm * y_sigma + y_mu

print(f"Prediction for {X_pred}: {y_pred[0] * 1000:.2f}$")
