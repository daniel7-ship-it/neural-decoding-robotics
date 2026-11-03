import unittest
import numpy as np
from neuralduck.rest_gate_review import measure
class ReviewMetricsTests(unittest.TestCase):
 def test_late_single_hit_is_not_timely_or_sustained(self):
  row=dict(y=np.array([0,0,1,1,1,1,1,1,1,1]),block=np.array([0,0,1,1,1,1,1,1,1,1]),time=np.arange(1,11)*.1,onset=np.array([0,0,.2,.2,.2,.2,.2,.2,.2,.2]))
  m=measure(row,np.array([0,0,0,0,0,0,0,0,0,1]))
  self.assertEqual(m['movement_detected'],1)
  self.assertEqual(m['deadline_hits'],0)
  self.assertEqual(m['occupancy_80_hits'],0)
 def test_stop_only_counts_movement_to_rest_and_sustained_zero(self):
  row=dict(y=np.array([1,1,0,0,0,0,0,0]),block=np.array([1,1,2,2,2,3,3,3]),time=np.arange(1,9)*.1,onset=np.array([0,0,.2,.2,.2,.5,.5,.5]))
  m=measure(row,np.array([1,1,1,0,1,0,0,0]))
  self.assertEqual(m['stop_transitions'],1)
  self.assertEqual(m['timely_sustained_stops'],0)
if __name__=='__main__':unittest.main()

