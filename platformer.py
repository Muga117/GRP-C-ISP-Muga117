import gymnasium as gym
import retro
import numpy as np
from stable_baselines3 import PPO

# Create the environment
env = retro.make(game='SonicAndKnuckles3-Genesis')

# Logging directory
log_dir = "./logs/sonic3/
new_logger = configure(log_dir, ["stdout", "tensorboard"])


