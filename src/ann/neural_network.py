"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from ann.neural_layer import Layer
from ann.activations import ReLU, Sigmoid, Tanh, Softmax
from ann.optimizers import SGD, Momentum, NAG, RMSProp
from ann.objective_functions import MSE,CrossEntropy

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args, input_size=None, output_size=None):
        """
        Initialize the neural network.

        Args:
          cli_args: Command-line arguments for configuring the network
        """
        self.args = cli_args

        self.input_size = input_size or 784
        self.output_size = output_size or 10

        hidden_sizes = getattr(cli_args,"hidden_size",[128,128])
        num_layers = getattr(cli_args,"num_layers",len(hidden_sizes))
        activation = getattr(cli_args,"activation","relu")
        weight_init = getattr(cli_args,"weight_init","xavier")
        loss = getattr(cli_args,"loss","cross_entropy")
        optimizer = getattr(cli_args,"optimizer","sgd")
        learning_rate = getattr(cli_args,"learning_rate",0.001)
        weight_decay = getattr(cli_args,"weight_decay",0.0)

        self.layers = []
        self.activations = []

        in_dim = self.input_size

        for i in range(num_layers):
          out_dim = hidden_sizes[i]
          self.layers.append(Layer(in_dim,out_dim,weight_init))

          if activation == "relu":
            self.activations.append(ReLU())
          elif activation == "sigmoid":
            self.activations.append(Sigmoid())
          elif activation == "tanh":
            self.activations.append(Tanh())
          
          in_dim = out_dim

        self.layers.append(
            Layer(in_dim, self.output_size, weight_init)
        )

        self.output_activation = Softmax()


        if loss == "mse":
          self.loss = MSE()
        elif loss == "cross_entropy":
          self.loss = CrossEntropy()
        
        if optimizer == "sgd":
          self.optimizer = SGD(lr=learning_rate,weight_decay=weight_decay)
        elif optimizer == "momentum":
          self.optimizer = Momentum(lr=learning_rate,weight_decay=weight_decay)
        elif optimizer == "nag":
          self.optimizer = NAG(lr=learning_rate,weight_decay=weight_decay)
        elif optimizer == "rmsprop":
          self.optimizer = RMSProp(lr=learning_rate,weight_decay=weight_decay)
       
    def set_weights(self, weights):

        for i, layer in enumerate(self.layers):

            layer.W = weights[f"W{i}"]
            layer.b = weights[f"b{i}"]
    
    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):

            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b

        return weights
           

    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        A = X
        for layer_i, act_i in zip(self.layers[:-1],self.activations):
          Z = layer_i.forward(A)
          A = act_i.forward(Z)
        logits = self.layers[-1].forward(A)
        return logits


    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        grad_W = []
        grad_b = []
        delta = self.loss.backward(y_true, y_pred)
        delta = self.layers[-1].backward(delta)
    
        # output layer grads
        grad_W.append(self.layers[-1].grad_W)
        grad_b.append(self.layers[-1].grad_b)
        
        for i in reversed(range(len(self.activations))):
          delta = self.activations[i].backward(delta)
          delta = self.layers[i].backward(delta)
          
          grad_W.insert(0, self.layers[i].grad_W)
          grad_b.insert(0, self.layers[i].grad_b)

        return grad_W, grad_b
              
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for L in self.layers:
          self.optimizer.update(L)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        N = X_train.shape[0]

        for epoch in range(epochs):
          idx = np.random.permutation(N)
          X_train = X_train[idx]
          y_train = y_train[idx]
          total_loss = 0

          for i in range(0,N, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred = self.forward(X_batch)
            loss = self.loss.forward(y_batch, y_pred)
            total_loss+= loss
            self.backward(y_batch,y_pred)
            self.update_weights()

          avg_loss = total_loss/(N//batch_size)
          wandb.log({"epoch": epoch+1, "loss": avg_loss})
          print(f"Epoch {epoch+1} Loss {avg_loss:.4f}")
      

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        pred_labels = np.argmax(y_pred,axis=1)
        true_labels = np.argmax(y,axis=1)

        accuracy = np.mean(pred_labels == true_labels)

        return accuracy
