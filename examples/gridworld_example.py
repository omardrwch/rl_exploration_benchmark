from rlxp.env  import GridWorld
from rlxp.rendering import render_env2d

env = GridWorld(7, 10, walls=((2,2), (3,3)))
env.enable_rendering()
for tt in range(50):
    env.step(env.action_space.sample())
render_env2d(env)