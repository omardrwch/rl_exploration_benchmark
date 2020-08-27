import gym 
from copy import deepcopy
from rlxp.utils.binsearch import *


class DiscretizeStateWrapper(gym.Wrapper):
    """
    Discretize an environment with continuous states and discrete actions.

    Note: the environment reset function must accept an initial state as input,
    that is: env.reset(s) puts the environment in the state s.
    """
    def __init__(self, _env, n_bins):
        self.env = deepcopy(_env)
        self.n_bins = n_bins

        # initialize bins 
        assert n_bins > 0, "DiscretizeStateWrapper requires n_bins > 0"
        n_states = 1
        tol = 1e-8
        self.dim = len(self.env.observation_space.low)
        n_states = n_bins ** self.dim
        self._bins = []
        self._open_bins = []
        for dd in range(self.dim):
            range_dd = self.env.observation_space.high[dd] - self.env.observation_space.low[dd]
            epsilon = range_dd / n_bins
            bins_dd = []
            for bb in range(n_bins+1):
                val = self.env.observation_space.low[dd]+epsilon*bb
                bins_dd.append(val)
            self._open_bins.append(tuple(bins_dd[1:])) 
            bins_dd[-1] += tol # "close" the last interval
            self._bins.append(tuple(bins_dd)) 
        # initialize base class
        super().__init__(self.env)

        # set observation space
        self.observation_space = gym.spaces.Discrete(n_states) 

        # List of discretized states
        self.discretized_states = np.zeros((self.dim, n_states))
        for ii in range(n_states):
            self.discretized_states[:, ii] = self.get_continuous_state(ii, False)

    def reset(self, discrete_state = None):
        if discrete_state is None:
            return self.get_discrete_state(self.env.reset())
        else:
            assert self.observation_space.contains(discrete_state)
            continuous_state = self.get_continuous_state( discrete_state, randomize=True )
            return self.get_discrete_state(self.env.reset(continuous_state))
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state = binary_search_nd(next_state, self._bins)
        return next_state, reward, done, info

    def get_discrete_state(self, continuous_state):
        return binary_search_nd(continuous_state, self._bins)

    def get_continuous_state(self, discrete_state, randomize=False):
        assert discrete_state >= 0 and  discrete_state < self.observation_space.n, "invalid discrete_state"
        # get multi-index
        index = unravel_index_uniform_bin(discrete_state, self.dim, self.n_bins)

        # get state 
        continuous_state = np.zeros(self.dim)
        for dd in range(self.dim):
            continuous_state[dd] = self._bins[dd][index[dd]]
            if randomize:
                range_dd = self.env.observation_space.high[dd] - self.env.observation_space.low[dd]
                epsilon = range_dd / self.n_bins
                continuous_state[dd] += epsilon*np.random.uniform()
        return continuous_state