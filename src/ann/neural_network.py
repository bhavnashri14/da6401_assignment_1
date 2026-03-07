"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
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
        self.hidden_activations = []
        A = X

        for layer_i, act_i in zip(self.layers[:-1],self.activations):
          Z = layer_i.forward(A)
          A = act_i.forward(Z)
          self.hidden_activations.append(A)

        logits = self.layers[-1].forward(A)
        return logits


    
    def backward(self, y_true, logits):
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
        delta = self.loss.backward(y_true, logits)
        delta = self.layers[-1].backward(delta)
        grad_W.insert(0, self.layers[-1].grad_W)
        grad_b.insert(0, self.layers[-1].grad_b)

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
        iter_step = 0
        for epoch in range(epochs):
          idx = np.random.permutation(N)
          X_train_shuffled = X_train[idx]
          y_train_shuffled = y_train[idx]
          total_loss = 0
          all_correct_probs = []
          grad_norms = []
          dead_counts = [0] * len(self.activations)
          
          for i in range(0,N, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            logits = self.forward(X_batch)

            y_pred = self.output_activation.forward(logits)
            y_true = np.argmax(y_batch, axis=1)
            correct_probs = y_pred[np.arange(len(y_true)), y_true]
            all_correct_probs.extend(correct_probs)  

            for layer_idx, layer_act in enumerate(self.hidden_activations):
              dead_neurons = np.sum(np.all(layer_act == 0, axis=0))
              dead_counts[layer_idx] += dead_neurons

            loss = self.loss.forward(y_batch, logits)
            total_loss+= loss
            self.backward(y_batch,logits)
            
            grad_norm = np.linalg.norm(self.layers[0].grad_W)
            grad_norms.append(grad_norm)

            
            # if iter_step < 50:
            #   g = self.layers[0].grad_W #gradients of 5 neurons in first hidden layer
            #   wandb.log({
            #       "grad_n1": g[:,0].mean(),
            #       "grad_n2": g[:,1].mean(),
            #       "grad_n3": g[:,2].mean(),
            #       "grad_n4": g[:,3].mean(),
            #       "grad_n5": g[:,4].mean(),
            #       "iteration": iter_step
            #   })
            #   iter_step+=1
            self.update_weights()

          
          avg_loss = total_loss/(N//batch_size)
          # mean_prob = np.mean(all_correct_probs)
          mean_grad_norm = np.mean(grad_norms)
          wandb.log({"epoch": epoch+1, "loss": avg_loss})
          # wandb.log({"mean_correct_class_prob": mean_prob, "grad_norm_layer1": mean_grad_norm})
          # for layer_i, d in enumerate(dead_counts):
          #      wandb.log({f"dead_neurons_layer_{layer_i+1}": d})
          print(f"Epoch {epoch+1} Loss {avg_loss:.4f}")
          # print("Dead neurons:", dead_counts)
          # print(f"First Layer Gradient Norm: {mean_grad_norm:.4f}")
          # print(f"Mean correct class probability: {mean_prob:.4f}")
        train_acc = float(self.evaluate(X_train, y_train))
        wandb.log({"train_accuracy": train_acc})
        print(f"Training Accuracy: {train_acc:.4f}")


    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        y_pred = self.forward(X)
        pred_labels = np.argmax(y_pred,axis=1)
        true_labels = np.argmax(y,axis=1)
        accuracy = np.mean(pred_labels == true_labels)

        return accuracy
