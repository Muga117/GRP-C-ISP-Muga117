
from wrappers import AllowBacktracking, RewardScaler, SonicDiscretizer, StochasticFrameSkip, WarpFrame
from gymnasium.wrappers import TimeLimit
import retro

GAME_TO_ZONES = {
    "SonicTheHedgehog-Genesis": [
        "GreenHillZone.Act1",
        "GreenHillZone.Act2",
        "GreenHillZone.Act3",
    ],
    "SonicTheHedgehog2-Genesis": [
        "EmeraldHillZone.Act1",
        "EmeraldHillZone.Act2",
    ],
    "SonicAndKnuckles3-Genesis": [
        "AngelIslandZone.Act1",
    ],
}

def make_env(game, state, render_mode="human"):
    env = retro.make(
        game=game,
        state=state,
        scenario='contest',
        use_restricted_actions=retro.Actions.ALL,
        players=1,
        record=False,
        render_mode=render_mode
    )
    return env

def wrap_env(env, max_episode_steps=4500):
    env = SonicDiscretizer(env)
    env = RewardScaler(env)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = WarpFrame(env, width=96, height=96, grayscale=True)
    env = AllowBacktracking(env)
    return env

def make_train_env(zone, render_mode):
    env = make_env(state=zone, render_mode=render_mode)
    return wrap_env(env)

def make_test_env(game="SonicAndKnuckles3-Genesis",state=None):
    game = game
    state = state
    env = make_env(game=game, state=state, render_mode="human")
    return wrap_env(env)
