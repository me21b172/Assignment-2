# Import necessary libraries
import wandb  # For experiment tracking
import gc  # Garbage collection
import torch
from model import ConvNN, train_model, log_predictions  # Custom model and functions
import os
import argparse

def sweepConfig(best_config):
    """Define the configuration for hyperparameter sweep"""
    if not best_config:
        # Default configuration for hyperparameter search
        config = {
            "method": "bayes",  # Bayesian optimization
            "metric": {"name": "val_accuracy", "goal": "maximize"},  # Optimization target
            "parameters":{
                "numberOfFilters": {"values": [16, 32, 64]},  # Possible filter sizes
                "filterOrganisation": {"values": [0.5, 1.0, 1.5]},  # Filter growth rates
                "activation": {"values": ["GeLU"]},  # Activation function
                "hidden_size": {"values": [128, 256, 512]},  # FC layer sizes
                "batch_size": {"values": [32, 64]},  # Batch sizes to try
                "dataAugmentation": {"values": [True, False]},  # Augmentation options
                "learning_rate": {"values": [0.001, 0.01]},  # Learning rate range
                "dropout":{"values": [0.3, 0.4, 0.5]},  # Dropout probabilities
                "num_epochs": {"values": [6]}  # Fixed epoch count for search
            }
        }
    else:
        # Configuration for training best found model
        config = {
            "method": "bayes",
            "metric": {"name": "val_accuracy", "goal": "maximize"},
            "parameters":{
                "numberOfFilters": {"values": [64]},  # Best found values
                "filterOrganisation": {"values": [0.5]},
                "activation": {"values": ["GeLU"]},
                "hidden_size": {"values": [512]},
                "batch_size": {"values": [64]},
                "dataAugmentation": {"values": [True]},
                "learning_rate": {"values": [0.001]},
                "dropout":{"values": [0.5]},
                "num_epochs": {"values": [25]}  # Longer training for final model
            }
        }
    return config

def train():
    """Main training function for wandb sweep"""
    try:
        # Initialize wandb run
        wandb.init()
        config = wandb.config
        
        # Create descriptive run name
        run_name = (f"filters-{config.numberOfFilters}-act-{config.activation}-hidden-{config.hidden_size}-lr-{config.learning_rate}")
        wandb.run.name = run_name
        
        # Set device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with current config parameters
        model = ConvNN(
            numberOfFilters=config.numberOfFilters,
            filterOrganisation=config.filterOrganisation,
            activation=config.activation,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            num_classes=10
        ).to(device)
        
        # Train model with current configuration
        train_model(
            model,
            num_epochs=config.num_epochs,
            device=device,
            batch_size=config.batch_size,
            dataAugmentation=config.dataAugmentation,
            learning_rate=config.learning_rate,
            iswandb=True  # Enable wandb logging
        )
        
        # Log sample predictions
        log_predictions(model=model)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        del model
        
    except Exception as e:
        print(f"[wandb/train] Error occurred: {e}")
        wandb.finish()  # Ensure wandb run is properly closed
    
def wandsweep(sweepId=None, best_config=False):
    """Manage the hyperparameter sweep process"""
    if sweepId is None:
        # Create new sweep if no ID provided
        sweepId = wandb.sweep(sweep=sweepConfig(best_config=best_config), project="CNN-Hyperparameter-Tuning")
        
        if best_config:
            # Run single training with best config
            print("Training best model")
            wandb.agent(sweepId, function=train, count=1)
        else:
            # Run hyperparameter search
            print("Tuning to get best model")
            wandb.agent(sweepId, function=train, count=15)  # 15 trials
    else:
        # Continue existing sweep
        wandb.agent(sweepId, function=train, count=10)
    
    # Clean up resources
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()

def main(args):
    """Main execution function"""
    if args.tune:
        # Run hyperparameter tuning or best model training
        wandsweep(best_config=True)
    else:
        # Standard training without tuning
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with fixed parameters
        model = ConvNN(
            numberOfFilters=64,
            filterOrganisation=1,
            activation="GeLU",
            hidden_size=256,
            dropout=0.5,
            num_classes=10
        ).to(device)
        
        # Train model
        train_model(
            model,
            num_epochs=args.epochs,
            device=device,
            batch_size=64,
            dataAugmentation=True,
            learning_rate=0.01,
            iswandb=False
        )
        return model

if __name__ == "__main__":
    # Disable wandb code saving for cleaner runs
    os.environ["WANDB_DISABLE_CODE"] = "true"
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        prog='Part_A',
        description='Implements the cnn model using pytorch to carry out classification task for iNaturalist-Dataset',
        epilog="That's how you run the above code as simple as that"
    )
    parser.add_argument("-t", "--tune", type=bool, help="Perform parameter tuning or not", default=False)
    parser.add_argument("-e", "--epochs", type=int, help="Number of Epochs", default=5)
    
    # Parse arguments and execute
    args = parser.parse_args()
    if args.tune:
        main(args)
    else:
        model = main(args)