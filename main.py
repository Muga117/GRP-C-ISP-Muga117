import argparse

import optuna
from train import train_agent
from test import  test_agent
from tune import objective

def main():
    parser = argparse.ArgumentParser(description="Run Sonic RL Project")
    parser.add_argument("--mode", choices=["train", "test", "tune"], required=True,
                        help="Choose whether to train, test, or hypertune the agent"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_agent()
    elif args.mode == "test":
        test_agent()
    elif args.mode == "tune":
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=5),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        study.optimize(objective, n_trials=10)  
        print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()
