from rlxp.env import MountainCar 
from rlxp.rendering import render_env2d

env = MountainCar()
env.enable_rendering()
for tt in range(150):
    env.step(env.action_space.sample())

render_env2d(env)
