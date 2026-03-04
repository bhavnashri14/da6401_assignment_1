"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist


#one hot encoder for labels

def one_hot_encoder(y, num_classes=10):

  y_encoded = np.zeros((len(y),num_classes))
  y_encoded[np.arange(len(y)), y] = 1

  return y_encoded

def load_data(dataset='mnist'):

  if dataset == "mnist":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
  elif dataset == "fashion_mnist":
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  else:
    raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

  X_train = X_train.astype(np.float32) / 255.0
  X_test = X_test.astype(np.float32) / 255.0

  X_train = X_train.reshape(X_train.shape[0], -1)  # flatten 28X28 to 784
  X_test = X_test.reshape(X_test.shape[0], -1) 

  y_train = one_hot_encoder(y_train)
  y_test = one_hot_encoder(y_test)

  return X_train, y_train, X_test, y_test










