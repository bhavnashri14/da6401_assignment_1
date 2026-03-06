"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

class ReLU:
    def __init__(self):
        self.A = None
    def forward(self, Z):
        self.A = Z
        return np.maximum(0, self.A)
    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0
        return dZ

class Sigmoid:
    def __init__(self):
        self.A = None
    def forward(self, Z):
        self.A = 1/(1+np.exp(-Z))
        return self.A
    def backward(self, dA):
        dZ = dA * self.A * (1 - self.A)
        return dZ

class Tanh:
    def __init__(self):
        self.A = None
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A
    def backward(self, dA):
        dZ=dA * (1 - self.A**2)
        return dZ

class Softmax:
    def __init__(self):
        self.A = None
    def forward(self, Z):
        Z_s = Z - np.max(Z, axis=1, keepdims=True)
        eZ = np.exp(Z_s)
        self.A = eZ / np.sum(eZ, axis=1, keepdims=True)
        return self.A

    def backward(self, dA):
        return dA   #Gradient combined in loss function script