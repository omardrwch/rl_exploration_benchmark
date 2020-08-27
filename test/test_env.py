import pytest
import numpy as np 
from rlxp.interface import FiniteMDP 
from rlxp.env  import Chain, SquareWorld, GridWorld, MountainCar

@pytest.mark.parametrize("L", [
    2, 10
])
def test_chain(L):
    fail_prob = 0.05
    env = Chain(L, fail_prob)
    assert env.R[L-1, 0] == env.R[L-1, 1]  == 1.0
    assert env.reward_range == (0, 1)


def test_squareworld():
    env = SquareWorld()
    env.step(0)
    env.step(1)
    env.step(2)
    env.step(3)
    assert env.reward_range == (0, 1)

def test_gridworld():
    env = GridWorld()
    env.step(0)
    env.step(1)
    env.step(2)
    env.step(3)   
    assert env.reward_range == (0, 1)

def test_mountaincar():
    env = MountainCar()
    env.step(0)
    env.step(1)
    env.step(2)
    assert env.reward_range == (0, 1)


