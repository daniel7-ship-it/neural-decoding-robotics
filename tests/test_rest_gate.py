import unittest
import numpy as np
from neuralduck.rest_gate import decisions
class RestGateTests(unittest.TestCase):
 def test_ambiguous_movement_abstains_and_strong_movement_passes(self):
  p=np.array([[.1,.8,.1],[.02,.95,.03],[.02,.03,.95],[.99,.005,.005]])
  np.testing.assert_array_equal(decisions(p,True),[0,1,2,0])
  np.testing.assert_array_equal(decisions(p,False),[1,1,2,0])
