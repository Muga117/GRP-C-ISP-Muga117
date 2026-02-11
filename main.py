import argparse

from train import train_agent
from test import  test_agent

def main():
    parser = argparse.ArgumentParser(description="Run Sonic RL Project")
    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Choose whether to train or test the agent"
    )
    parser.add_argument("--game", type=str, default="SonicAndKnuckles3-Genesis")
    parser.add_argument("--level", type=str, default="AngelIslandZone.Act1")
    args = parser.parse_args()

    if args.mode == "train":
        train_agent()
    elif args.mode == "test":
        test_agent(game=args.game, state=args.level)

if __name__ == "__main__":
    main()
