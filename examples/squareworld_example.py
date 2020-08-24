from rlxp.env import SquareWorld 
from rlxp.rendering import render_env2d

env = SquareWorld()
env.enable_rendering()
for tt in range(10):
    env.step(env.action_space.sample())

render_env2d(env)