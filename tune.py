from copy import deepcopy
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from callbacks import SingleCheckpointCallback
from env import make_env
import numpy as np
import os

from paths import MODEL_PATH, VECNORM_PATH

NUM_WORKERS = 8

# 🧩 Load the base model
base_model = PPO.load(MODEL_PATH)

def objective(trial):
    def make_env_fn():
        return make_env(record_video=False)

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

    # 🔧 Hyperparameters to tune
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-3, 0.05, log=True)
    clip_range = trial.suggest_float("clip_range", 0.05, 0.3)

    # Apply tuned params
    model = PPO(
        "CnnPolicy",
        train_env,
        learning_rate=lambda _: lr,
        ent_coef=ent_coef,
        clip_range=lambda _: clip_range,
        n_steps=base_model.n_steps,
        batch_size=base_model.batch_size,
        n_epochs=base_model.n_epochs,
        gamma=base_model.gamma,
        gae_lambda=base_model.gae_lambda,
        verbose=0,
        device="cuda"
    )

    model.policy.load_state_dict(base_model.policy.state_dict())

    # 🔁 Fine-tune
    print("Starting Fine-Tuning...")
    checkpoint_callback = SingleCheckpointCallback(MODEL_PATH, save_freq=500_000)
    model.learn(total_timesteps=500_000, callback=checkpoint_callback)

    # 🧪 Evaluate performance 
    mean_reward, _ = evaluate_policy(model, train_env, n_eval_episodes=5)
    train_env.close()
    print("Fine-Tuning Complete.")
    return mean_reward

