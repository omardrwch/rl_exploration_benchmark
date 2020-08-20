import numpy as np
import gym
from gym import spaces


class FiniteMDP(gym.Env):
    """
    Base class for a finite MDP.

    Args:
        R    (numpy array): Array of shape (S, A) containing the mean rewards;  S = number of states;  A = number of actions.
        P    (numpy.array): Array of shape (S, A, S) containing the transition probabilities,
                            P[s, a, s'] = Prob(S_{t+1}=s'| S_t = s, A_t = a). Na is the total number of actions.
        seed_val(int): Random number generator seed

    Attributes:
        random   (np.random.RandomState) : random number generator
    """
    def __init__(self, R, P, seed_val=42):
        super().__init__()
        S, A = R.shape

        self.R = R
        self.P = P

        self.observation_space = spaces.Discrete(S)
        self.action_space      = spaces.Discrete(A)

        self.state = None
        self.random = np.random.RandomState(seed_val)

        self._states  = np.arange(S)
        self._actions = np.arange(A)

        self.reset()
        self._check()

    def reset(self, state=0):
        """
        Reset the environment to a default state or to a given state.

        Args:
            state(int)

        Returns:
            state (object)
        """
        self.state = state
        return self.state

    def _check(self):
        """
        Check consistency of the MDP
        """
        # Check that P[s,a, :] is a probability distribution
        for s in self._states:
            for a in self._actions:
                assert abs(self.P[s, a, :].sum() - 1.0) < 1e-15
        
        # Check that dimensions match
        S1, A1 = self.R.shape 
        S2, A2, S3 = self.P.shape 
        assert S1 == S2 == S3 
        assert A1 == A2 

    def seed(self, seed=42):
        """
        Reset random number generator
        """
        self.random = np.random.RandomState(seed)

    def sample_transition(self, s, a):
        """
        Sample a transition s' from P(s'|s,a).

        Args:
            s (int): index of state
            a (int): index of action

        Returns:
            ss (int): index of next state
        """
        prob = self.P[s, a, :]
        s_ = self.random.choice(self._states, p=prob)
        return s_

    def step(self, action):
        """
        Execute a step. Similar to gym function [1].
        [1] https://gym.openai.com/docs/#environments

        Args:
            action (int): index of the action to take

        Returns:
            observation (object)
            reward      (float)
            done        (bool)reward_fn
            info        (dict)
        """
        assert action in self._actions, "Invalid action!"
        next_state = self.sample_transition(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        done = self.is_terminal(self.state)
        info = {}

        self.state = next_state
        observation = next_state
        return observation, reward, done, info

    def is_terminal(self, state):
        """
        Returns true if a state is terminal.
        """
        return False

    def reward_fn(self, state, action, next_state):
        """
        Reward function. Returns mean reward at (state, action) by default.

        Args:
            state      (int): current state
            action     (int): current action
            next_state (int): next state

        Returns:
            reward (float)
        """
        return self.R[state, action]

    def print(self):
        """
        Print the structure of the MDP.
        """
        indent = '    '
        for s in self._states:
            print(("State %d" + indent)%s)
            for a in self._actions:
                print(indent + "Action ", a)
                for ss in self._states:
                    if self.P[s, a, ss] > 0.0:
                        print(2*indent + 'transition to %d with prob %0.2f'%(ss, self.P[s, a, ss]))
            print("~~~~~~~~~~~~~~~~~~~~")


if __name__=='__main__':
    S = 3 
    A = 2 

    R = np.random.uniform(0, 1, (S, A))
    P = np.random.uniform(0, 1, (S, A, S))
    for ss in range(S):
        for aa in range(A):
            P[ss, aa, :] /= P[ss, aa, :].sum()

    env = FiniteMDP(R, P)