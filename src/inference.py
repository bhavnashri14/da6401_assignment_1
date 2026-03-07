"""
Inference Script
Evaluate trained models on test sets
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import CrossEntropy, MSE

def parse_arguments(args_list=None):
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument('-m','--model_path', type=str,default='/da6401_assignment_1/src/inference.py',help='relative path to model')
    parser.add_argument('-d','--dataset',type=str,default='mnist',choices=['mnist','fashion_mnist'],help='dataset')
    parser.add_argument('-b','--batch_size',type=int,default=64,help='batch size')
    parser.add_argument('-nhl', '--num_layers', type=int,default=2,help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int,nargs='+',default=[128,128],help='Hidden Layer sizes')
    parser.add_argument('-a', '--activation', type=str,default='sigmoid',choices=['sigmoid','tanh','relu'],help='Activation')

    return parser.parse_args(args_list)


def load_model(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data
    
def plot_cm(true,pred,class_names=None):
    cm = confusion_matrix(true,pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig,ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix to confusion_matrix.png")

def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)

    if isinstance(model.loss,CrossEntropy):
      loss = model.loss.forward(y_test,logits)
    else:
      loss = MSE().forward(y_test,logits)
    
    pred = np.argmax(logits,axis=1)
    true = np.argmax(y_test,axis=1)

    plot_cm(true,pred)

    accuracy = np.mean(pred == true)
    
    precision_list = []
    recall_list = []
    f1_list = []

    classes = np.unique(true)

    for c in classes:

      TP = np.sum((pred == c)&(true==c))
      FP = np.sum((pred == c)& (true!=c))
      FN = np.sum((pred != c) & (true == c))
      TN = len(true) - TP - FP - FN

      precision = TP/(TP+FP)
      recall = TP/(TP+FN)
      f1 = 2*precision*recall/(precision+recall)

      precision_list.append(precision)
      recall_list.append(recall)
      f1_list.append(f1)
    
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)


    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



def main(args_list=None):
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments(args_list)
    np.random.seed(42)
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    weights = load_model(args.model_path)
    model = NeuralNetwork(args)
    model.set_weights(weights)
    results = evaluate_model(model, X_test, y_test)

    print("Results:")

    for k,v in results.items():
        if k != "logits":
            print(k, ":", v)
    wandb.init(project="da6401_assignment1_inference")

    wandb.log({
        "test_loss": results["loss"],
        "test_accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1"]
    })
    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()
