import unittest
import numpy as np
from neuralduck.replay_engine import ReplayEngine
from neuralduck.neural_replay import load_checkpoint
from neuralduck.supervised import ROOT
from neuralduck.augmentation import recording
class ReplayEngineTests(unittest.TestCase):
 def test_saved_trace_order_commands_and_completion(self):
  row=recording(45,'clean');decoder,width=load_checkpoint(ROOT/'models/noise-diagnostic/noise-24-101.pt')
  saved=np.load(ROOT/'artifacts/noise-diagnostic/45-clean-noise-24-101-False.npz')
  engine=ReplayEngine(row['x'],row['time'],decoder,width)
  frames=[]
  while not engine.finished:frames.append(engine.step())
  self.assertEqual([d['index'] for d in engine.history],list(range(len(row['x']))))
  np.testing.assert_array_equal([d['prediction'] for d in engine.history],saved['probability'].argmax(1))
  np.testing.assert_array_equal([d['command'] for d in engine.history],saved['command'])
  for d in engine.history:self.assertAlmostEqual(d['applies_at']-d['issued_at'],.02)
  self.assertEqual(frames[0]['applied'],(0.,0.,0.))
  self.assertEqual(frames[-1]['applied'],(0.,0.,0.))
  self.assertTrue(engine.step()['finished'])
  engine.reset(decoder,width)
  self.assertEqual(engine.next_index,0);self.assertEqual(engine.tick,0);self.assertEqual(engine.history,[]);self.assertIsNone(engine.command)
 def test_end_seek_does_not_repeat_final_window(self):
  class Fake:
   def predict_proba(self,x):return np.tile([0.,1.,0.],(len(x),1))
  engine=ReplayEngine(np.zeros((3,48)),[.2,.3,.4],Fake(),48,start_seconds=.4)
  for _ in range(20):engine.step()
  self.assertEqual([d['index'] for d in engine.history],[2])
  self.assertTrue(engine.finished)
if __name__=='__main__':unittest.main()

