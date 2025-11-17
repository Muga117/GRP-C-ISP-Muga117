import os
import numpy as np
from stable_baselines3 import PPO
from env import GAME_TO_ZONES, ZONES, make_train_env, wrap_env, make_env
from paths import MODEL_PATH, VECNORM_PATH, LOGS_PATH
from callbacks import SingleCheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor
)

def train_agent(total_timesteps=10_000_000):
    env_fns = []
    for game, zones in GAME_TO_ZONES.items():
        for zone in zones:
            def make_env_fn(game=game, zone=zone):
                def _init():
                    return wrap_env(make_env(game=game, state=zone, render_mode=None))
                return _init
            env_fns.append(make_env_fn())
    
    train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env, filename="logs/monitor")
    train_env = VecFrameStack(train_env, n_stack=4)
    
    # PPO agent 
    if os.path.exists(MODEL_PATH):
        print("Loading existing model from", MODEL_PATH)
        model = PPO.load(MODEL_PATH, env=train_env, device="cuda")

    else:
        print("No existing model found, creating new one.")
        model = PPO(
            "CnnPolicy",
            train_env,
            learning_rate=lambda _: 2e-4,
            n_steps=2048,
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
   
    train_env.close()
    print("Training Complete.")


    


