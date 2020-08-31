import numpy as np
from rlxp.interface import FiniteMDP 
from rlxp.rendering import RenderInterface2D, Scene, GeometricPrimitive

class Chain(FiniteMDP, RenderInterface2D):
    """
    Simple chain environment. 
    Reward 0.05 in initial state, reward 1.0 in final state (deterministic).

    :param L:         length of the chain
    :param fail_prob: fail probability 
    """
    def __init__(self, L, fail_prob) -> None:
        assert L >= 2
        self.L = L

        # reward range 
        self.reward_range = (0, 1)

        # transition probabilities
        P = np.zeros((L, 2, L))
        for ss in range(L):
            for aa in range(2):
                if ss == 0:
                    P[ss, 0, ss]   = 1.0-fail_prob  # action 0 = don't move
                    P[ss, 1, ss+1] = 1.0-fail_prob  # action 1 = right
                    P[ss, 0, ss+1] = fail_prob  
                    P[ss, 1, ss]   = fail_prob          
                elif ss == L-1:
                    P[ss, 0, ss-1] = 1.0-fail_prob  # action 0 = left
                    P[ss, 1, ss]   = 1.0-fail_prob  # action 1 = don't move
                    P[ss, 0, ss]   = fail_prob  
                    P[ss, 1, ss-1] = fail_prob 
                else:
                    P[ss, 0, ss-1] = 1.0-fail_prob  # action 0 = left
                    P[ss, 1, ss+1] = 1.0-fail_prob  # action 1 = right
                    P[ss, 0, ss+1] = fail_prob  
                    P[ss, 1, ss-1] = fail_prob 
        
        # mean reward
        S = L
        A = 2 
        R = np.zeros((S, A))
        R[L-1, :] = 1.0
        R[0,   :] = 0.05

        # init base classes
        FiniteMDP.__init__(self, R, P)
        RenderInterface2D.__init__(self)

        # rendering info
        self.set_clipping_area((0, L, 0, 1))
        self.set_refresh_interval(500)  # in milliseconds


    def step(self, action):
        assert action in self._actions, "Invalid action!"

        # save state for rendering
        if self.is_render_enabled():
            self.append_state_for_rendering(self.state)
        
        # take step
        next_state = self.sample_transition(self.state, action)
        reward = self.reward_fn(self.state, action, next_state)
        done = self.is_terminal(self.state)
        info = {}

        self.state = next_state
        observation = next_state
        return observation, reward, done, info

    def get_background(self):
        """
        Returne a scene (list of shapes) representing the background
        """
        bg = Scene()
        colors = [ (0.25, 0.25, 0.25),  (0.75, 0.75, 0.75) ]
        for ii in range(self.L):
            shape = GeometricPrimitive("GL_QUADS")
            shape.add_vertex((ii,   0))
            shape.add_vertex((ii+1, 0))
            shape.add_vertex((ii+1, 1))
            shape.add_vertex((ii,   1))
            shape.set_color(colors[ii%2])
            bg.add_shape(shape)
        return bg 

    def get_scene(self, state):
        """
        Return scene (list of shapes) representing a given state
        """
        scene = Scene()

        agent = GeometricPrimitive("GL_QUADS")
        agent.set_color((0.0, 0.5, 0.0))

        agent.add_vertex((state+0.25,   0.25))
        agent.add_vertex((state+0.75,   0.25))
        agent.add_vertex((state+0.75,   0.75))
        agent.add_vertex((state+0.25,   0.75))

        scene.add_shape(agent)
        return scene  
    

