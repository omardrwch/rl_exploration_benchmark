"""
Mountain Car environment from OpenAI gym [1], with some changes:

* default reward is 0       (instead of -1)
* reward in goal state is 1 (instead of 0)
* reset function accepts state as optional argument (to use the env as a generative model, for instance)
* render function follows our rendering interface instead.

[1] https://raw.githubusercontent.com/openai/gym/master/gym/envs/classic_control/mountain_car.py
"""

import math
import numpy as np
import rlxp.interface as interface

from gym import spaces
from gym.utils import seeding
from rlxp.rendering import RenderInterface2D, Scene, GeometricPrimitive

class MountainCar(interface.Env, RenderInterface2D):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.

    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 1 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of 0 is awarded if the position of the agent is less than 0.5.

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.

     Episode Termination:
         The car position is more than 0.5
     """
    
    def __init__(self, goal_velocity = 0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        
        self.force=0.001
        self.gravity=0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # init base classes
        interface.Env.__init__(self)
        RenderInterface2D.__init__(self) 
        self.reward_range = (0, 1)

        # rendering info
        self.set_clipping_area((-1.2, 0.6, -0.2, 1.1))
        self.set_refresh_interval(10)  # in milliseconds
        
        # initial reset
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(np.array(self.state))

        position, velocity = self.state
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0.0
        if done:
            reward = 1.0

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def reset(self, _state = None):
        if _state is None:
            self.state = np.array([self.random.uniform(low=-0.6, high=-0.4), 0])
        else:
            if (type(_state) != np.ndarray):
                _state = np.array(_state)
            assert self.observation_space.contains(_state), "Invalid state as argument of reset()."
            self.state = _state
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def get_background(self):
        bg = Scene()
        mountain = GeometricPrimitive("GL_TRIANGLE_FAN")
        flag     = GeometricPrimitive("GL_TRIANGLES")
        mountain.set_color((0.6, 0.3, 0.0))
        flag.set_color((0.0, 0.5, 0.0))

        # Mountain
        mountain.add_vertex((-0.3, -1.0))
        mountain.add_vertex(( 0.6, -1.0))

        n_points  = 50
        obs_range = self.observation_space.high[0]-self.observation_space.low[0]
        eps       = obs_range/(n_points-1)
        for ii in reversed(range(n_points)):
            x = self.observation_space.low[0] + ii*eps 
            y = self._height(x)
            mountain.add_vertex((x, y))
        mountain.add_vertex((-1.2, -1.0))

        # Flag
        goal_x = self.goal_position 
        goal_y = self._height(goal_x)
        flag.add_vertex((goal_x,       goal_y))
        flag.add_vertex((goal_x+0.025, goal_y+0.075))
        flag.add_vertex((goal_x-0.025, goal_y+0.075))

        bg.add_shape(mountain)
        bg.add_shape(flag)

        return bg

    def get_scene(self, state):
        scene = Scene() 

        agent = GeometricPrimitive("GL_QUADS")
        agent.set_color((0.0, 0.0, 0.0))
        size = 0.025
        x = state[0]
        y = self._height(x)
        agent.add_vertex((x-size, y-size))
        agent.add_vertex((x+size, y-size))
        agent.add_vertex((x+size, y+size))
        agent.add_vertex((x-size, y+size))

       
        scene.add_shape(agent)
        return scene 
