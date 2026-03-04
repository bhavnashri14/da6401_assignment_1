"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
import numpy as np
import wandb
import argparse
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument('-d', '--dataset', type=str,default='mnist',choices=['mnist','fashion_mnist'],help='Dataset choice')
    parser.add_argument('-e', '--epochs', type=int,default=10,help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int,default=64,help='Batch size')
    parser.add_argument('-l', '--loss', type=str,default='cross_entropy',choices=['mse','cross_entropy'],help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str,default='sgd',choices=['sgd','momentum','nag','rmsprop','adam','nadam'],help='Optimiser')
    parser.add_argument('-lr', '--learning_rate', type=float,default=0.001,help='Learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float,default=0.0,help='L2 regularisation weight decay')
    parser.add_argument('-nhl', '--num_layers', type=int,default=2,help='Batch size')
    parser.add_argument('-sz', '--hidden_size', type=int,nargs='+',default=[128,128],help='Hidden Layer sizes')
    parser.add_argument('-a', '--activation', type=str,default='sigmoid',choices=['sigmoid','tanh','relu'],help='Activation')
    parser.add_argument('-wi', '--weight_init', type=str,default='xavier',choices=['random','xavier'],help='Weight initialisation')
    parser.add_argument('-wp', '--wandb_project',type=str,default='da6401_assignment1',help='Weights and Biases Project ID')
    return parser.parse_args()

def save_model(model, path):

    weights = {}

    for i,layer in enumerate(model.layers):
        weights[f"W{i}"] = layer.W
        weights[f"b{i}"] = layer.b

    np.save(path, weights)

def main():
  """
  Main training function.
  """

  args = parse_arguments()

  wandb.init(
    project="da6401_assignment1",
    config=vars(args)
)

  np.random.seed(42)
  
  X_train, y_train, X_test, y_test = load_data(args.dataset)

  model = NeuralNetwork(args)

  model.train(
        X_train,
        y_train,
        args.epochs,
        args.batch_size
    )

  acc = model.evaluate(X_test, y_test)
  wandb.log({"training_accuracy": acc})
  save_model(model,"model.npy")
  print("Training Accuracy:", acc)
  print("Training complete!")


if __name__ == '__main__':
    main()
