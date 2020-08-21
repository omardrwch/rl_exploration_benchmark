from rlxp.env  import Chain
from rlxp.rendering import render_env2d

env = Chain(10, 0.1)
env.enable_rendering()
for tt in range(10):
    env.step(env.action_space.sample())

render_env2d(env)