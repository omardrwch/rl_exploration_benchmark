"""
A subclass of gym.Env with some minor modifications:

- includes a random number generator by default
- reset() takes a state as optional input, to put the env in a given state (useful 
   for RL algorithms that need a generative model)
"""
import gym 
import numpy as np

class Env(gym.Env):
    def __init__(self, seed_val=-1):
        if seed_val == -1:
            seed_val = np.random.randint(1, 32768) 
        self.random = np.random.RandomState(seed_val)

    def reset(self, state=None):
        """
        Reset the environment to a default state or to a given state.
        """
        raise NotImplementedError

    def seed(self, seed_val):
        """
        Reset random number generator
        """
        self.random = np.random.RandomState(seed_val)