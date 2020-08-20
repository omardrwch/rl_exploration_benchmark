import numpy as np
from rlxp.interface import FiniteMDP 

class RandomMDP(FiniteMDP):
    def __init__(self, S, A, seed=123):

        # define rewards and transitions
        R = np.random.uniform(0, 1, (S, A))
        P = np.random.uniform(0, 1, (S, A, S))
        for ss in range(S):
            for aa in range(A):
                P[ss, aa, :] /= P[ss, aa, :].sum()

        # initialize base class
        super().__init__(R, P, seed)
    
    # override reward function to add noise to mean reward
    def reward_fn(self, state, action, next_state):
        return self.R[state,action] + 0.1 * np.random.randn()



# test it!
env = RandomMDP(3, 2)
env.print()