"""
Sonic 3 & Knuckles Training Script
- PPO with 8 parallel workers
- Sonic-specific discrete actions
- Warp frames to 96x96 RGB
- Frame skipping and stacking
- AllowBacktracking reward
- VecNormalize for reward scaling
- Headless training (no windows)
"""

import random
import numpy as np
import retro
import gymnasium as gym
import cv2
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecVideoRecorder,
    DummyVecEnv,
    VecNormalize
)

# ==================== Wrappers ====================
class WarpFrame(ObservationWrapper):
    """Resize and optionally grayscale frames"""
    def __init__(self, env, width=96, height=96, grayscale=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        channels = 1 if grayscale else 3
        self.observation_space = Box(low=0, high=255, shape=(channels, height, width), dtype=np.uint8)

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, axis=0)
        else:
            obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
        return obs

class AllowBacktracking(gym.Wrapper):
    """Reward forward progress; ignore backward movement"""
    def __init__(self, env):
        super().__init__(env)
        self._cur_x = 0
        self._max_x = 0

        # Unwrap to the base retro environment
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        self._base_env = base_env

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        # Use base retro environment to access 'x'
        x = self._base_env.data.lookup_value("x")
        rew = max(0, x - self._max_x)
        self._max_x = max(self._max_x, x)
        return obs, rew, terminated, truncated, info



class StochasticFrameSkip(gym.Wrapper):
    """Random frame skip for smoother training"""
    def __init__(self, env, n_min=2, n_max=5):
        super().__init__(env)
        self.n_min = n_min
        self.n_max = n_max

    def step(self, action):
        n = random.randint(self.n_min, self.n_max)
        total_reward = 0.0
        done = False
        truncated = False
        for _ in range(n):
            obs, reward, terminated, truncated_step, info = self.env.step(action)
            total_reward += reward
            if terminated:
                done = True
                terminated = True
                break
            truncated |= truncated_step
        return obs, total_reward, terminated, truncated, info


# ==================== Environment ====================

ZONES = [
    "AngelIslandZone.Act1",
    "HydrocityZone.Act1",
    "MarbleGardenZone.Act1",
    "CarnivalNightZone.Act1",
    "IcecapZone.Act1",
    "LaunchBaseZone.Act1",
    "MushroomHillZone.Act1",
    "FlyingBatteryZone.Act1"
]

def make_env(record_video=False, video_folder='videos/', render_mode=None):
    state = np.random.choice(ZONES)
    env = retro.make(
        game='SonicAndKnuckles3-Genesis',
        state=state,
        use_restricted_actions=retro.Actions.DISCRETE,
        players=1,
        record=False,
        render_mode=render_mode  # headless
    )
    
    env = StochasticFrameSkip(env, n_min=2, n_max=5)
    env = WarpFrame(env, width=96, height=96, grayscale=False)
    env = AllowBacktracking(env)
    return env

# ==================== Main ====================

if __name__ == "__main__":
    NUM_WORKERS = 8

    # Training environment
    train_env = SubprocVecEnv([lambda: make_env(record_video=False) for _ in range(NUM_WORKERS)])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99)

    # Test environment
    test_env_single = make_env(render_mode="rgb_array")
    test_env = DummyVecEnv([lambda: test_env_single])  # wrap in DummyVecEnv for stable-baselines3
    test_env = VecFrameStack(test_env, n_stack=4)      # now test_env exists

    # PPO agent
    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./ppo_sonic_tensorboard/"
    )

    # Train
    print("Starting training...")
    model.learn(total_timesteps=10_000_000, log_interval=10)
    print("Saving Model...")
    model.save("ppo_sonic3")

    print("Loading Video Recorder...")
    test_env = VecVideoRecorder(
        test_env,
        video_folder="./videos/",
        record_video_trigger=lambda x: True,  # record all episodes
        video_length=10000                    # max frames per video
    )

    # Test
    print("Testing trained agent...")
    model = PPO.load("ppo_sonic3", env=test_env)
    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action)  
        print(f"Reward: {reward}")
        

    test_env.close()
