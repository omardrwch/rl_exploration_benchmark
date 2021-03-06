import numpy as np 
import gym
import rlxp.interface as interface 
from rlxp.rendering import RenderInterface2D, Scene, GeometricPrimitive

class SquareWorld(interface.Env, RenderInterface2D):
    """
    SquareWorld environment with states in [0, 1]^2 and 4 actions 
    
    The agent starts at (start_x, start_y) and, in each state, it can take for actions (0 to 3) representing a
    displacement of (-d, 0), (d, 0), (0, -d) and (0, d), respectively.
         
    The immediate reward received in each state s = (s_x, s_y) is, for any action a,
       r(s, a) = exp( - ((s_x-goal_x)^2 + (s_y-goal_y)^2)/(2*reward_smoothness^2)  )
    """
    def __init__(self):
        # Coordinates of start position
        self._start_x = 0.1 
        self._start_y = 0.1  

        # Coordinates of goal position (where the maximal reward is obtained)
        self._goal_x = 0.75 
        self._goal_y = 0.75 

        # Action displacement
        self._displacement = 0.1 

        # Reward smoothness
        self._reward_smoothness = 0.1 

        # Standard dev of reward noise (gaussian)
        self._reward_noise_stdev = 0.01 

        # Standard dev of transition noise (gaussian)
        self._transition_noise_stdev = 0.01 

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(low =np.array([0.0, 0.0], dtype=np.float32), 
                                                high=np.array([1.0, 1.0], dtype=np.float32))
        self.action_space      = gym.spaces.Discrete(4)

        # Initial state
        self.state = np.array([self._start_x, self._start_y])

        # init base classes
        interface.Env.__init__(self)
        RenderInterface2D.__init__(self) 
        self.reward_range = (0, 1)

        # rendering info
        self.set_clipping_area((0, 1, 0, 1))
        self.set_refresh_interval(500)  # in milliseconds


    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state.copy())

        # done and reward
        done = False 
        reward = np.exp( -0.5*( np.power(self.state[0]-self._goal_x, 2)  
                                +   np.power(self.state[1]-self._goal_y, 2) ) /
                                np.power(self._reward_smoothness, 2) )
        reward += self._reward_noise_stdev*np.random.randn() 

        # next state
        noise = self._transition_noise_stdev*np.random.randn(2)        
        self.state += noise 
        if   (action == 0):
            self.state[0] -= self._displacement
        elif (action == 1):
            self.state[0] += self._displacement
        if   (action == 2):
            self.state[1] -= self._displacement
        elif (action == 3):
            self.state[1] += self._displacement

        # clip
        self.state[0] = min(max(self.state[0], 0.0), 1.0)
        self.state[1] = min(max(self.state[1], 0.0), 1.0)

        
        return self.state.copy(), reward, done, {}
    
    def reset(self, state=None):
        if state is not None:
            self.state = state 
        else:
            self.state = np.array([self._start_x, self._start_y])
        return self.state.copy()
        

    def get_background(self):
        bg = Scene()

        flag = GeometricPrimitive("GL_TRIANGLES")
        flag.set_color((0.0, 0.5 ,0.0))
        flag.add_vertex((self._goal_x, self._goal_y))
        flag.add_vertex((self._goal_x+0.025, self._goal_y+0.075))
        flag.add_vertex((self._goal_x-0.025, self._goal_y+0.075))

        bg.add_shape(flag)

        return bg

    def get_scene(self, state):
        scene = Scene() 

        agent = GeometricPrimitive("GL_QUADS")
        agent.set_color((0.75, 0.0, 0.5))
        size = 0.025
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