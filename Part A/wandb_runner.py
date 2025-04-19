# parameters : numberOfFilters,filterOrganisation, activation,hidden_size, num_classes, batch_size, dataAugmentation,learning_rate, num_epochs
import wandb
import gc
import torch
from model import ConvNN, train_model,log_predictions
import os
import argparse

def sweepConfig(best_config):
    if not best_config:
        config = {
            "method": "bayes",
            "metric": {"name": "val_accuracy", "goal": "maximize"},
            "parameters":{
                "numberOfFilters": {"values": [16, 32, 64]},
                "filterOrganisation": {"values": [0.5, 1.0 , 1.5]},
                "activation": {"values": ["GeLU"]},
                "hidden_size": {"values": [128 , 256, 512]},
                "batch_size": {"values": [32, 64]},
                "dataAugmentation": {"values": [True, False]},
                "learning_rate": {"values": [0.001, 0.01]},
                "dropout":{"values": [0.3, 0.4, 0.5]},
                "num_epochs": {"values": [6]}
            }
        }
    else:
        config = {
            "method": "bayes",
            "metric": {"name": "val_accuracy", "goal": "maximize"},
            "parameters":{
                "numberOfFilters": {"values": [64]},
                "filterOrganisation": {"values": [0.5]},
                "activation": {"values": ["GeLU"]},
                "hidden_size": {"values": [512]},
                "batch_size": {"values": [64]},
                "dataAugmentation": {"values": [True]},
                "learning_rate": {"values": [0.001]},
                "dropout":{"values": [0.5]},
                "num_epochs": {"values": [25]}
            }
        }
    return config

def train():
    try:
        wandb.init()
        config = wandb.config
        run_name = (f"filters-{config.numberOfFilters}-act-{config.activation}-hidden-{config.hidden_size}-lr-{config.learning_rate}")
        wandb.run.name = run_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Using device:", device)
        model = ConvNN(numberOfFilters=config.numberOfFilters, filterOrganisation=config.filterOrganisation, activation=config.activation,hidden_size=config.hidden_size, dropout=config.dropout,num_classes=10).to(device)
        train_model(model, num_epochs=config.num_epochs,device = device,batch_size = config.batch_size, dataAugmentation = config.dataAugmentation,learning_rate = config.learning_rate, iswandb=True)
        log_predictions(model=model)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        del model
    except Exception as e:
        print(f"[wandb/train] Error occurred: {e}")
        wandb.finish()
    
def wandsweep(sweepId = None, best_config = False):
    if sweepId is None:
        sweepId = wandb.sweep(sweep=sweepConfig(best_config = best_config), project="CNN-Hyperparameter-Tuning")
        if best_config:
            print("Training best model")
            wandb.agent(sweepId, function=train, count=1)
        else:
            print("Tuning to get best model")
            wandb.agent(sweepId, function=train, count=15)
    else:
        wandb.agent(sweepId, function=train, count=10)
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()

def main(args):
    if args.tune:
        # wandsweep(sweepId="me21b172-indian-institute-of-technology-madras/CNN-Hyperparameter-Tuning/u89wmcmf")
        wandsweep(best_config=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConvNN(numberOfFilters=64, filterOrganisation=1, activation="GeLU",hidden_size=256, dropout=0.5,num_classes=10).to(device)
        train_model(model, num_epochs=args.epochs,device = device,batch_size = 64, dataAugmentation = True,learning_rate = 0.01, iswandb=False)
        return model

if __name__ == "__main__":
    os.environ["WANDB_DISABLE_CODE"] = "true"
    # wandsweep(sweepId="me21b172-indian-institute-of-technology-madras/CNN-Hyperparameter-Tuning/u89wmcmf")
    # wandsweep()
    parser = argparse.ArgumentParser(
                    prog='Part_A',
                    description='Implements the cnn model using pytorch to carry out classification task for iNaturalist-Dataset',
                    epilog="That's how you run the above code as simple as that")
    parser.add_argument("-t","--tune",type=bool,help="Perform parameter tuning or not",default=False)
    parser.add_argument("-e","--epochs",type=int,help="Number of Epochs",default=5)
    args = parser.parse_args()
    if args.tune:
        main(args)
    else:
        model = main(args)

    