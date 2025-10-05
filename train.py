import os

from stable_baselines3 import PPO
from env import make_env
from paths import MODEL_PATH, VECNORM_PATH, LOGS_PATH
from callbacks import SingleCheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize
)

NUM_WORKERS = 16

def train_agent(total_timesteps=10_000_000):
    def make_env_fn():
        return make_env(record_video=False)

    train_env = SubprocVecEnv([make_env_fn for _ in range(NUM_WORKERS)])
    train_env = VecFrameStack(train_env, n_stack=4)
    # Load VecNormalize stats if available
    if os.path.exists(VECNORM_PATH):
        print(f"Loading VecNormalize stats from {VECNORM_PATH}")
        train_env = VecNormalize.load(VECNORM_PATH, train_env)
        train_env.training = True  # keep updating stats
    else:
        print("No VecNormalize stats found, creating new one.")
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99)

    # PPO agent 
    if os.path.exists(MODEL_PATH):
        print("Loading existing model from", MODEL_PATH)
        model = PPO.load(MODEL_PATH, env=train_env)
    else:
        print("No existing model found, creating new one.")
        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            learning_rate=7.5e-5,
            ent_coef=0.01,
            n_steps=4096,
            batch_size=1024,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            tensorboard_log=LOGS_PATH
        )

    # Train
    print("Starting training...")
    checkpoint_callback = SingleCheckpointCallback(MODEL_PATH, save_freq=500_000)
    model.learn(total_timesteps=10_000_000, log_interval=10, callback=checkpoint_callback)
    print("Saving Model...")
    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)

    train_env.close()
    print("Training Complete.")