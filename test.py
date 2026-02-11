from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    DummyVecEnv,
)
from paths import MODEL_PATH
from env import make_test_env

def test_agent(game="SonicAndKnuckles3-Genesis",state=None):
    print("Creating Test Environment...")
    test_env = DummyVecEnv([lambda: make_test_env(
        game=game,
        state=state
    )])
    test_env = VecFrameStack(test_env, n_stack=4)      

    print("Testing trained agent...")
    model = PPO.load(MODEL_PATH)
    obs = test_env.reset()
    done = [False]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action) 
        
    test_env.close()
    print("Testing Complete.")