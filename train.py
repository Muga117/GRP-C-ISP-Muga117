import os
import numpy as np
import retro
from stable_baselines3 import PPO
from env import make_env, make_env_fn, wrap_env
import env
from paths import MODEL_PATH, VECNORM_PATH, LOGS_PATH
from callbacks import SingleCheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecMonitor
)

NUM_WORKERS = 16

def train_agent(total_timesteps=5_000_000):
    train_env = SubprocVecEnv([make_env_fn for _ in range(NUM_WORKERS)])
    train_env = VecMonitor(train_env, filename="logs/monitor")
    train_env = VecFrameStack(train_env, n_stack=4)
    
    # Load VecNormalize stats if available
    if os.path.exists(VECNORM_PATH):
        print(f"Loading VecNormalize stats from {VECNORM_PATH}")
        train_env = VecNormalize.load(VECNORM_PATH, train_env)
        train_env.training = True  
    else:
        print("No VecNormalize stats found, creating new one.")
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99)

    # PPO agent 
    if os.path.exists(MODEL_PATH):
        print("Loading existing model from", MODEL_PATH)
        model = PPO.load(MODEL_PATH, env=train_env, device="cuda")

    else:
        print("No existing model found, creating new one.")
        model = PPO(
            "CnnPolicy",
            train_env,
            learning_rate=lambda _: 7.5e-5,
            n_steps=4096,
            batch_size=1024,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=lambda _: 0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=LOGS_PATH
        )

    # Train
    print("Starting training...")
    checkpoint_callback = SingleCheckpointCallback(MODEL_PATH, save_freq=500_000)
    model.learn(total_timesteps=total_timesteps, log_interval=10, callback=checkpoint_callback)

    print("Saving Model...")
    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)

    train_env.close()
    print("Training Complete.")
    


