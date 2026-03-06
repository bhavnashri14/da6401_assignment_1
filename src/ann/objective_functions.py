"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class MSE:

  def forward(self, y_true, y_pred):
    loss = np.mean((y_true - y_pred) ** 2)
    return loss

  def backward(self, y_true, y_pred):
    N = y_true.shape[0]
    dL =2*(y_pred - y_true)/N
    return dL


class CrossEntropy:

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, y_true, logits):
        prob = self.softmax(logits)
        eps = 1e-12
        loss = -np.mean(np.sum(y_true * np.log(prob + eps), axis=1))
        return loss

    def backward(self, y_true, logits):
        N = y_true.shape[0]
        prob = self.softmax(logits)
        return (prob - y_true)/N



