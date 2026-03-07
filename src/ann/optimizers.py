"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np

class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay

    def update(self, layer):
        # Apply L2 Regularization (Weight Decay)
        if self.wd > 0:
            layer.grad_W += self.wd * layer.W
        
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.gamma = momentum
        self.wd = weight_decay
        self.v_w = {}
        self.v_b = {}

    def update(self, layer):
        if layer not in self.v_w:
            self.v_w[layer] = np.zeros_like(layer.W)
            self.v_b[layer] = np.zeros_like(layer.b)
        if self.wd > 0:
            layer.grad_W += self.wd * layer.W

        # Velocity update    
        self.v_w[layer] = self.gamma * self.v_w[layer] + self.lr * layer.grad_W
        self.v_b[layer] = self.gamma * self.v_b[layer] + self.lr * layer.grad_b
        
        layer.W -= self.v_w[layer]
        layer.b -= self.v_b[layer]


class NAG:
    
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.gamma = momentum
        self.wd = weight_decay
        self.v_w = {}
        self.v_b = {}

    def update(self, layer):
        if layer not in self.v_w:
            self.v_w[layer] = np.zeros_like(layer.W)
            self.v_b[layer] = np.zeros_like(layer.b)

        v_prev_w = self.v_w[layer].copy()
        v_prev_b = self.v_b[layer].copy()

        grad_W = layer.grad_W.copy()
        grad_b = layer.grad_b.copy()

        if self.wd > 0:
            grad_W += self.wd * layer.W

        # Velocity update
        self.v_w[layer] = self.gamma * self.v_w[layer] + self.lr * grad_W
        self.v_b[layer] = self.gamma * self.v_b[layer] + self.lr * grad_b

        # Lookahead update
        layer.W -= self.gamma * v_prev_w + (1 + self.gamma) * self.v_w[layer]
        layer.b -= self.gamma * v_prev_b + (1 + self.gamma) * self.v_b[layer]

class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.wd = weight_decay
        self.m_w = {}
        self.m_b = {}

    def update(self, layer):
        if layer not in self.m_w:
            self.m_w[layer] = np.zeros_like(layer.W)
            self.m_b[layer] = np.zeros_like(layer.b)

        grad_W = layer.grad_W.copy()
        grad_b = layer.grad_b.copy()

        if self.wd > 0:
            grad_W += self.wd * layer.W

        # Update moving average of squared gradients
        self.m_w[layer] = self.beta * self.m_w[layer] + (1 - self.beta) * (grad_W ** 2)
        self.m_b[layer] = self.beta * self.m_b[layer] + (1 - self.beta) * (grad_b ** 2)

        # Parameter update
        layer.W -= (self.lr / (np.sqrt(self.m_w[layer]) + self.eps)) * grad_W
        layer.b -= (self.lr / (np.sqrt(self.m_b[layer]) + self.eps)) * grad_b
