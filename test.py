from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecVideoRecorder,
    DummyVecEnv,
    VecNormalize
)
from paths import MODEL_PATH, VIDEOS_PATH
from env import make_env

def test_agent():
    print("Creating Test Environment...")
    test_env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
    test_env = VecFrameStack(test_env, n_stack=4)      
    test_env = VecVideoRecorder(
        test_env,
        video_folder=VIDEOS_PATH,
        record_video_trigger=lambda x: True,  
        video_length=10000                    
    )

    print("Testing trained agent...")
    model = PPO.load(MODEL_PATH, env=test_env)
    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = test_env.step(action) 
        
    test_env.close()
    print("Testing Complete.")