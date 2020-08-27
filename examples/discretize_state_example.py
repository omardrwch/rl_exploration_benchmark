from rlxp.env import SquareWorld 
from rlxp.wrappers import DiscretizeStateWrapper
from rlxp.rendering import render_env2d

_env = SquareWorld()
_env.enable_rendering()
env = DiscretizeStateWrapper(_env, 50)
for tt in range(10):
    env.step(env.action_space.sample())

# only the unwrapped environment implements rendering
render_env2d(env.unwrapped)
