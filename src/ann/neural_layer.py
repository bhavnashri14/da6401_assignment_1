"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

class Layer:
  def __init__(self, in_features,out_features, weight_init='random'):
    self.weight_init = weight_init

  # Initialize weights
    if weight_init == 'random':
      self.W = np.random.randn(in_features, out_features) * 0.01
      self.b = np.zeros((1, out_features))
    elif weight_init == 'xavier':
      self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 /(in_features+out_features) )
      self.b = np.zeros((1, out_features))
    else:
      raise ValueError("weight_init must be 'random' or 'xavier'")

    self.grad_W = None
    self.grad_b = None

  def forward(self, X):
    self.X=X
    return X @ self.W + self.b

  def backward(self, delta):
    
    self.grad_W = (self.X.T @ delta)
    self.grad_b = (np.sum(delta, axis=0, keepdims=True))
    grad_v = delta @ self.W.T
    return grad_v
