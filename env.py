
from paths import VIDEOS_PATH
from wrappers import AllowBacktracking, RewardScaler, SonicDiscretizer, StochasticFrameSkip, WarpFrame, RandomLevelWrapper
from gymnasium.wrappers import TimeLimit
import retro
import numpy as np

ZONES = [
    "AngelIslandZone.Act1",
    "AngelIslandZone.Act2",
    "HydrocityZone.Act1",
    #"MarbleGardenZone.Act1",
    #"CarnivalNightZone.Act1",
    #"IcecapZone.Act1",
    #"LaunchBaseZone.Act1",
    #"MushroomHillZone.Act1",
    #"FlyingBatteryZone.Act1",
    #"SandopolisZone.Act1",
    #"LavaReefZone.Act1",
    #"DeathEggZone.Act1"
]

def make_env(record_video=False, video_folder=VIDEOS_PATH, max_episode_steps=4500, render_mode=None):
    #state = np.random.choice(ZONES)
    env = retro.make(
        game='SonicAndKnuckles3-Genesis',
        #state=state,
        state="AngelIslandZone.Act2",
        scenario='contest',
        use_restricted_actions=retro.Actions.ALL,
        players=1,
        record=False,
        render_mode=render_mode
        #render_mode='human'  
    )
    print("Creating Environment")
    return env
    #env = RandomLevelWrapper(ZONES)

def wrap_env(env, max_episode_steps=4500, scale_rew=True):
    env = SonicDiscretizer(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = WarpFrame(env, width=96, height=96, grayscale=True)
    env = AllowBacktracking(env)
    #if scale_rew:
        #env = RewardScaler(env)
    return env

def make_env_fn(render_mode=None, record_video=False):
    env = make_env(record_video=record_video, render_mode=render_mode)
    return wrap_env(env)
 

