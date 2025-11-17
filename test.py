from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecVideoRecorder,
    DummyVecEnv,
    VecNormalize
)
from paths import MODEL_PATH, VIDEOS_PATH
from env import make_test_env
import os

def test_agent():
    print("Creating Test Environment...")
    test_env = DummyVecEnv([lambda: make_test_env()])
    test_env = VecFrameStack(test_env, n_stack=4)      
    #test_env = VecVideoRecorder(
        #test_env,
        #video_folder=VIDEOS_PATH,
        #record_video_trigger=lambda x: x == 0, 
        #video_length=20000                   
    #)

    print("Testing trained agent...")
    model = PPO.load(MODEL_PATH)
    obs = test_env.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action) 
        
    test_env.close()
    print("Testing Complete.")