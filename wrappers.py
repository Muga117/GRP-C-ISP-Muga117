import random
import cv2
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from gymnasium import ObservationWrapper


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
    
    
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SonicDiscretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                ["LEFT"],
                ["RIGHT"],
                ["LEFT", "DOWN"],
                ["RIGHT", "DOWN"],
                ["DOWN"],
                ["DOWN", "B"],
                ["B"],
            ],
        )

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

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