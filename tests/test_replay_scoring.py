import unittest
import numpy as np
from neuralduck.rest_gate_review import measure
class ReplayScoring(unittest.TestCase):
 def test_carryover_and_new_rest_activation_are_separate(self):
  row=dict(y=np.array([1,0,0,0,0]),block=np.array([1,2,2,2,2]),time=np.arange(5)*.1,onset=np.zeros(5))
  m=measure(row,np.array([1,1,0,2,2]))
  self.assertEqual(m['transition_carryover_events'],1)
  self.assertEqual(m['new_rest_activation_events'],1)
  self.assertEqual(m['false_rest_events'],2)
