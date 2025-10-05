
from wrappers import AllowBacktracking, RewardScaler, SonicDiscretizer, StochasticFrameSkip, WarpFrame
import retro

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

def make_env(record_video=False, video_folder='videos/', render_mode=None, scale_rew=True):
    #state = np.random.choice(ZONES)
    env = retro.make(
        game='SonicAndKnuckles3-Genesis',
        state="AngelIslandZone.Act1",
        scenario='contest',
        use_restricted_actions=retro.Actions.ALL,
        players=1,
        record=False,
        render_mode=render_mode  # headless
    )
    env = SonicDiscretizer(env)
    print("SonicDiscretizer action_space", env.action_space)
    env = StochasticFrameSkip(env, n_min=2, n_max=5)
    env = WarpFrame(env, width=96, height=96, grayscale=True)
    env = AllowBacktracking(env)
    if scale_rew:
        env = RewardScaler(env)
    return env