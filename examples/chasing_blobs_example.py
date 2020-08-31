from rlxp.envs import ChasingBlobs 
from rlxp.rendering import render_env2d

period = 5

env = ChasingBlobs(period)
env.enable_rendering()
env.reset()

for tt in range(50):
    if tt % 2 == 0:
        env.reset()
        print(env._current_configuration)
    env.step(env.action_space.sample())

render_env2d(env)
