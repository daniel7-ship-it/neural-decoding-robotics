import unittest
import numpy as np
from neuralduck.applied_scoring import score_frames
class Boundaries(unittest.TestCase):
 def row(self):return dict(block=np.array([0,1,2]),onset=np.array([0.,.1,.2]),y=np.array([0,1,0]))
 def frames(self,t,c):return [dict(recording_time=a,applied=[0,0,b],finished=False) for a,b in zip(t,c)]
 def test_initial_activity_and_empty_schema(self):
  r=score_frames(self.frames([0,.1],[1,0]),self.row(),.3,observation_end=.2)
  self.assertEqual(r['initial_active_rest'],1);self.assertEqual(r['false_rest_events'],0)
  self.assertEqual(set(r),set(score_frames([],self.row(),.3)))
 def test_stop_then_reactivation(self):
  r=score_frames(self.frames([0,.1,.2,.25],[0,1,0,1]),self.row(),.3,observation_end=.3)
  self.assertEqual(r['timely_sustained_stops'],0);self.assertEqual(r['new_rest_activation_events'],1)
 def test_wrong_and_partial(self):
  r=score_frames(self.frames([0,.1,.15,.2],[0,-1,1,0]),self.row(),.3,observation_end=.3)
  self.assertAlmostEqual(r['wrong_direction_seconds'],.05);self.assertEqual(r['wrong_direction_events'],1)
  self.assertEqual(score_frames(self.frames([.15,.2],[1,0]),self.row(),.3,observation_end=.3)['movement_episodes'],0)
 def test_invalid(self):
  with self.assertRaises(ValueError):score_frames(self.frames([.1,.1],[0,0]),self.row(),.3)
