import pytest
import numpy as np 
from rlxp.interface import FiniteMDP 
from rlxp.env  import Chain

@pytest.mark.parametrize("L", [
    2, 10
])
def test_chain(L):
    fail_prob = 0.05
    env = Chain(L, fail_prob)
    assert env.R[L-1, 0] == env.R[L-1, 1]  == 1.0