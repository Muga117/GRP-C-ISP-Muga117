[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/inoLPW_E)

<sub>Model Playing 1st level of Sonic 3 and Knuckles</sub>

![Alt Text](model/model.gif)

# Project Overview
This project aims to train a PPO agent to play Sonic levels from Sonic The Hedghehog, Sonic The Hedghehog 2 and Sonic 3 and Knuckles.

Below are the levels that the agent was trained and tested on:
```
GAME_TO_ZONES = {
    "SonicTheHedgehog-Genesis": [
        "GreenHillZone.Act1",
        "GreenHillZone.Act2",
        "GreenHillZone.Act3",
    ],
    "SonicTheHedgehog2-Genesis": [
        "EmeraldHillZone.Act1",
        "EmeraldHillZone.Act2",
    ],
    "SonicAndKnuckles3-Genesis": [
        "AngelIslandZone.Act1",
    ],
}
```
## Methodology
### Preprocessing
```
env = SonicDiscretizer(env)              # Simplifies Sonic's complex controls into a manageable discrete action set
env = RewardScaler(env)                  # Scales rewards to keep training stable
env = StochasticFrameSkip(env, n=4, stickprob=0.25)  
                                         # Repeats actions for several frames with slight randomness to reduce computation and improve robustness
if max_episode_steps is not None:
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
                                         # Caps episode length to avoid overly long or infinite runs
env = WarpFrame(env, width=96, height=96, grayscale=True)
                                         # Downscales and converts frames to grayscale to reduce input size and training complexity
env = AllowBacktracking(env)             # Allows the agent to move backward without penalty to encourage exploration

```
### Actions
```
combos=[
    ["UP"],
    ["LEFT"],
    ["RIGHT"],
    ["LEFT", "DOWN"],
    ["RIGHT", "DOWN"],
    ["DOWN"],
    ["DOWN", "B"],
    ["B"],
]
```
### Training
The agent was trained using a reinforcement learning methodology built on the Proximal Policy Optimization (PPO) algorithm implemented through Stable Baselines3 and a CNN Policy. Training was carried out using curriculum learning, where the agent was first exposed to simpler Sonic levels before progressing to more complex ones, allowing it to gradually build robust skills and improve stability in learning. All experimentation and training were performed on personal hardware.

### Rewards
```
reward = (progress - data.prev_progress) * 9000 # Encourages continuous advancement through the level.
reward = reward + (1 - clip(scenario.frame / frame_limit, 0, 1)) * 1000 # Encourages efficient, fast level completion.
reward = reward - 5000 # Strongly discourages risky actions leading to death.
```

## Project Structure
```
PlatformerRLAgent/
├── .github/
│ └── .keep # Maintains GitHub folder structure
├── .gitignore # Git ignore rules
├── callbacks.py # Custom training callbacks (logging, checkpoints, eval triggers)
├── env.py # Stable Retro environment creation and configuration
├── main.py # Entry point for running the agent or launching workflows
├── paths.py # Centralized path management for models, logs, and ROM directories
├── README.md # Project documentation
├── test.py # Evaluation and testing script for trained models
├── train.py # PPO training script using Stable Baselines3
├── wrappers.py # Observation/action wrappers for preprocessing and shaping
├── models/ # Saved PPO models and checkpoints
├── logs/ # TensorBoard logs, training metrics, episode stats
└── ROMS/ # Game ROMs (user-supplied, not included in repo)
```

## Libraries
Stable Retro -> Reinforcement learning framework designed for training agents on classic video games using emulator-based environments. Built on top of Retro Gym, it provides a clean API for loading ROMs, interacting with retro game states, and collecting image-based observations.

Stable Baselines -> Includes PPO reinforcement learning algorithm.


## Setup
Use this guide to install Stable Retro and Stable Baselines3

https://www.youtube.com/watch?v=vPnJiUR21Og

Clone the Repository
```
git clone https://github.com/Muga117/PlatformerRLAgent.git
cd PlatformerRLAgent/
```
Place your Sonic ROMS obtained online in ROMS/ and then import them into Stable Retro.
```
cd ROMS/
python3 -m retro.import
```

Train a New Model
```
python main.py --mode train
```
Access Logs
```
tensorboard --logdir logs/
```
Test the model
```
python main.py --mode test --game SonicTheHedgehog-Genesis --level GreenHillZone.Act1
python main.py --mode test --game SonicTheHedgehog2-Genesis --level EmeraldHillZone.Act1
python main.py --mode test --game SonicAndKnuckles3-Genesis --level AngelIslandZone.Act1
```



