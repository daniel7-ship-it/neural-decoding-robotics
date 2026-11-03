import unittest
import numpy as np
from neuralduck.applied_scoring import score_frames
class AppliedScoringTest(unittest.TestCase):
 def test_between_decisions_and_partial(self):
  row=dict(block=np.array([0,1,2]),onset=np.array([0.,.1,.2]),y=np.array([0,1,0]))
  frames=[dict(recording_time=i*.02,applied=[0,0,1 if i in (2,6,7,8,9) else 0],finished=False) for i in range(15)]
  r=score_frames(frames,row,.3,observation_end=.3)
  self.assertAlmostEqual(r['rest_seconds'],.2);self.assertAlmostEqual(r['rest_occupancy'],.1)
  self.assertAlmostEqual(r['movement'][0]['delay_seconds'],.02);self.assertEqual(r['false_rest_events'],1)
  self.assertEqual(score_frames(frames[6:],row,.3)['movement_episodes'],0)
if __name__=='__main__':unittest.main()
