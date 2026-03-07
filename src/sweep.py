
import wandb
import numpy as np
import json
from src.ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from sklearn.metrics import f1_score


sweep_config = {
    "method": "random",
    "name": "hyperparameter_sweep",
    "metric": {"name": "validation_f1", "goal": "maximize"},
    "parameters": {
        # Hidden sizes and number of layers are paired
        "hidden_size": {"values": [
            [64, 64],    
            [128, 128], 
            [128, 64],   
            [128, 128, 128],  
            [256, 128, 64],   
            [64, 128, 64, 64] 
        ]},
        "activation": {"values": ["relu", "sigmoid", "tanh"]},
        "optimizer": {"values": ["sgd", "momentum", "rmsprop", "nag"]},
        "learning_rate": {"values": [0.001, 0.01]},
        "batch_size": {"values": [32, 64]},
        "weight_init": {"values": ["xavier", "random"]},
        "weight_decay": {"values": [0.0, 0.0001]},
        "loss": {"values": ["cross_entropy", "mse"]}
    }
}

np.random.seed(42)

best_f1 = -1
best_weights = None
best_config = None


def sweep_train():
    global best_f1, best_weights, best_config

    wandb.init()
    config = wandb.config

    class Args:
        dataset = "mnist"
        epochs = 5
        batch_size = config.batch_size
        hidden_size = config.hidden_size
        activation = config.activation
        optimizer = config.optimizer
        learning_rate = config.learning_rate
        weight_init = config.weight_init
        loss = config.loss
        weight_decay = config.weight_decay
        wandb_project = "da6401_assignment1"

    args = Args()
    args.num_layers = len(args.hidden_size)
    
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    model.train(X_train, y_train, args.epochs, args.batch_size)

    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)
    true = np.argmax(y_test, axis=1)
    f1 = f1_score(true, preds, average="macro")
    test_acc = model.evaluate(X_test, y_test)
    print("Test F1-score:", f1, "test_accuracy:", test_acc)
    wandb.log({"test_f1": f1,"test_accuracy": test_acc})

    # Save best model and config 
    if f1 > best_f1:
        best_f1 = f1
        best_weights = model.get_weights()
        best_config = {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_init": args.weight_init,
            "weight_decay": args.weight_decay
            "loss": args.loss
        }
        
        np.save("best_model.npy", best_weights)
        with open("best_config.json", "w") as f:
            json.dump(best_config, f, indent=4)
        print("Saved new best model and config!")

# -------------------------
# 3. Initialize Sweep
# -------------------------
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="da6401_assignment1")
    wandb.agent(sweep_id, function=sweep_train, count=100)