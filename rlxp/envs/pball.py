"""
Parametric family of environments whose state space is a unit sphere according to the p-norm in R^d.

Note: 
    The projection function is only a projection for p \in {2, infinity}. 

----------------------------------------------------------------------
State space:
    x \in R^d: norm_p (x) <= 1

    implemented as gym.spaces.Box representing [0, 1]^d
----------------------------------------------------------------------
Action space:
    {u_1, ..., u_m} such that u_i \in R^d'  for i = 1, ..., m

    implemented as gym.spaces.Discrete(m)
----------------------------------------------------------------------
Reward function (independent of the actions):
    r(x) = \sum_{i=1}^n  b_i  \max( 0,  1 - norm_p( x - x_i )/c_i )

    requirements:
        c_i >= 0
        b_i \in [0, 1]
----------------------------------------------------------------------
Transitions:
    x_{t+1} = A x_t + B u_t + N

    where 
          A: square matrix of size d
          B: matrix of size (d, d')
          N: d-dimensional Gaussian noise with zero mean and covariance matrix sigma*I 
----------------------------------------------------------------------
Initial state:
    d-dimensional Gaussian with mean mu_init and covariance matrix sigma_init*I
----------------------------------------------------------------------

Parameters:
    * p (parameter of the p-norm)
    * List of actions {u_1, ..., u_m}, each action u_i is a d'-dimensional array
    * List of reward amplitudes: {b_1, ..., b_n}
    * List of reward smoothness: {c_1, ..., c_n}
    * List of reward centers:    {x_1, ..., x_n}
    * Array A of size (d, d)
    * Array B of size (d, d')
    * Transition noise sigma
    * Initial state noise sigma_init

Default parameters are provided for a 2D environment, PBall2D.
"""

import numpy as np 
import gym
import rlxp.interface as interface 
from rlxp.rendering import RenderInterface2D, Scene, GeometricPrimitive


def projection_to_pball(x, p):
    """
    Solve the problem:
        min_z  ||x-z||_2^2 
        s.t.   ||z||_p  <= 1
    for p = 2 or p = np.inf 

    If p is not 2 or np.inf, it returns x/norm_p(x) if norm_p(x) > 1

    WARNING: projection_to_pball is not actually a projection for p!=2 or p=!np.inf
    """
    if np.linalg.norm(x, ord=p) <= 1.0:
        return x

    if p == 2:
        z = x/np.linalg.norm(x, ord=p)
        return z

    if p == np.inf:
        z = np.minimum(1.0, np.maximum(x, -1.0))
        return z 
    
    # below it is not a projection
    return x/np.linalg.norm(x, ord=p)
        

class PBall(interface.Env):
    def __init__(self,
                 p, 
                 action_list,
                 reward_amplitudes,
                 reward_smoothness,
                 reward_centers,
                 A, 
                 B,
                 sigma,
                 sigma_init,
                 mu_init):
        assert p >= 1, "PBall requires p>=1"
        if p not in [2, np.inf]:
            print("WARNING: for p!=2 or p!=np.inf, PBall does not make true projections onto the lp ball.")
        self.p                 = p 
        self.d, self.dp        = B.shape   # d and d'
        self.m                 = len(action_list)
        self.action_list       = action_list
        self.reward_amplitudes = reward_amplitudes 
        self.reward_smoothness = reward_smoothness 
        self.reward_centers    = reward_centers
        self.A                 = A  
        self.B                 = B 
        self.sigma             = sigma 
        self.sigma_init        = sigma_init 
        self.mu_init           = mu_init

        # State and action spaces
        low  = -1.0*np.ones(self.d, dtype=np.float32)
        high =      np.ones(self.d, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high)
        self.action_space      = gym.spaces.Discrete(self.m)

        # reward range
        assert len(self.reward_amplitudes) == len(self.reward_smoothness) == len(self.reward_centers)
        assert self.reward_amplitudes.max() <= 1.0 and \
               self.reward_amplitudes.min() >= 0.0, "reward amplitudes b_i must be in [0, 1]"
        assert self.reward_smoothness.min() >  0.0, "reward smoothness c_i must be > 0"
        self.reward_range = (0, 1.0)

        # Initalize state
        self.reset()

    def reset(self, state=None):
        if state is not None:
            self.state = state 
        else:
            self.state = self.mu_init + self.sigma_init*np.random.randn(self.d)
        # projection to unit ball
        self.state = projection_to_pball(self.state, self.p)
        return self.state 

    def step(self, action):
        assert self.action_space.contains(action)

        # next state
        action_vec = self.action_list[action]
        next_s = self.A.dot(self.state) + self.B.dot(action_vec) + self.sigma*np.random.randn(self.d)
        next_s = projection_to_pball(next_s, self.p)

        # done and reward
        done = False
        reward = self.compute_reward_at(self.state)

        # update state
        self.state = next_s 

        return self.state, reward, done, {}
    
    def compute_reward_at(self, x):
        reward = 0
        for ii, b_ii in enumerate(self.reward_amplitudes):
            c_ii = self.reward_smoothness[ii]
            x_ii = self.reward_centers[ii]
            dist = np.linalg.norm(x-x_ii, ord=self.p)
            reward += b_ii * max(0.0, 1.0 - dist/c_ii)
        return reward

    def get_reward_lipschitz_constant(self):
        ratios = self.reward_amplitudes/self.reward_smoothness 
        Lr     = ratios.max()
        return Lr 
    
    def get_transitions_lipschitz_constant(self):
        """
        note: considers a fixed action, returns Lipschitz constant w.r.t. to states.

        If p!=1, p!=2 or p!=np.inf, returns an upper bound on the induced norm
        """
        if self.p == 1:
            order = np.inf 
        else:
            order = self.p / (self.p - 1.0)
        
        if order in [1, 2]:
            return np.linalg.norm(self.A, ord = order)

        # If p!=1, p!=2 or p!=np.inf, return upper bound on the induced norm.
        return np.power(self.d, 1.0/self.p) * np.linalg.norm(self.A, ord = np.inf)

        

class PBall2D(PBall, RenderInterface2D):
    def __init__(self,
                 p = 2, 
                 action_list = [  0.05*np.array([1, 0]),
                                 -0.05*np.array([1, 0]),
                                  0.05*np.array([0, 1]),
                                 -0.05*np.array([0, 1])],
                 reward_amplitudes = np.array([1.0]),
                 reward_smoothness = np.array([0.25]),
                 reward_centers    = [np.array([0.75, 0.0])],
                 A = np.eye(2), 
                 B = np.eye(2),
                 sigma = 0.01,
                 sigma_init = 0.001,
                 mu_init = np.array([0.0, 0.0])
                 ):
        # Initialize PBall
        PBall.__init__(self, p,  action_list, reward_amplitudes,reward_smoothness,
                            reward_centers, A, B, sigma, sigma_init, mu_init)
        
        # Render interface
        RenderInterface2D.__init__(self) 

        # rendering info
        self.set_clipping_area((-1, 1, -1, 1))
        self.set_refresh_interval(100)  # in milliseconds

    def step(self, action):
        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state.copy())
        return PBall.step(self, action)

    def get_background(self):
        bg = Scene()

        n_points_x = 100
        n_points_y = 100

        eps_x = 2.0 / n_points_x
        eps_y = 2.0 / n_points_y


        for ii in range(n_points_x):
            for jj in range(n_points_y):
                xx = -1 + ii*eps_x 
                yy = -1 + jj*eps_y 
                center = np.array([xx+eps_x/2, yy+eps_y/2])

                cc = np.linalg.norm( center, ord=self.p) <= 0.999
                reward = self.compute_reward_at(center)

                sqr = GeometricPrimitive("GL_QUADS")
                sqr.set_color((0.0, reward*cc, cc))
                sqr.add_vertex((xx, yy))
                sqr.add_vertex((xx+eps_x, yy))
                sqr.add_vertex((xx+eps_x, yy+eps_y))
                sqr.add_vertex((xx, yy+eps_y))
                bg.add_shape(sqr)
        return bg

    def get_scene(self, state):
        scene = Scene() 

        agent = GeometricPrimitive("GL_QUADS")
        agent.set_color((0.75, 0.0, 0.5))
        size = 0.05
        x = state[0]
        y = state[1]
        agent.add_vertex((x-size/4.0, y-size))
        agent.add_vertex((x+size/4.0, y-size))
        agent.add_vertex((x+size/4.0, y+size))
        agent.add_vertex((x-size/4.0, y+size))

        agent.add_vertex((x-size, y-size/4.0))
        agent.add_vertex((x+size, y-size/4.0))
        agent.add_vertex((x+size, y+size/4.0))
        agent.add_vertex((x-size, y+size/4.0))
        
        scene.add_shape(agent)
        return scene 

if __name__=='__main__':
    env = PBall2D(p=5)
    print(env.get_transitions_lipschitz_constant())
    print(env.get_reward_lipschitz_constant())

    env.enable_rendering()

    for ii in range(100):
        env.step(1)
        env.step(3)

    from rlxp.rendering import render_env2d
    render_env2d(env)