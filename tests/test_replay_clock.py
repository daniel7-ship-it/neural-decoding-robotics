import unittest
import numpy as np
from neuralduck.replay_engine import ReplayEngine
class ClockContract(unittest.TestCase):
 class Fake:
  def __init__(self,labels):self.labels=iter(labels);self.calls=0
  def predict_proba(self,x):
   self.calls+=1;return np.eye(3)[[next(self.labels)]]
 def test_late_window_order_applied_timing_and_expiry_gap(self):
  f=self.Fake([1,1,0]);e=ReplayEngine(np.zeros((3,48)),[.035,.071,.46],f,48)
  frames=[]
  for _ in range(30):frames.append(e.step())
  self.assertEqual([d['index'] for d in e.history],[0,1,2]);self.assertEqual(f.calls,3)
  self.assertEqual(e.history[0]['issued_at'],.04)
  self.assertEqual(e.history[1]['issued_at'],.08)
  self.assertEqual(frames[4]['applied'],(0.,0.,0.))
  self.assertGreater(frames[5]['applied'][2],0)
  self.assertEqual(frames[18]['applied'],(0.,0.,0.))
  self.assertTrue(e.finished)
 def test_seek_future_empty_invalid_and_idempotent(self):
  f=self.Fake([1]);e=ReplayEngine(np.zeros((1,48)),[.4],f,48,start_seconds=.2)
  for _ in range(10):self.assertIsNone(e.step()['decision'])
  self.assertEqual(f.calls,0)
  e.reset(f,48,start_seconds=1.)
  self.assertTrue(e.step()['finished']);self.assertTrue(e.step()['finished']);self.assertEqual(f.calls,0)
  empty=ReplayEngine(np.zeros((0,48)),[],f,48)
  self.assertTrue(empty.step()['finished'])
  for times in ([.1,.1],[.2,.1],[float('nan')]):
   with self.assertRaises(ValueError):ReplayEngine(np.zeros((len(times),48)),times,f,48)
 def test_reset_switch_clears_gate_cursor_and_trace(self):
  left=self.Fake([1,1]);right=self.Fake([2,2]);e=ReplayEngine(np.zeros((2,48)),[0,.1],left,48)
  for _ in range(6):e.step()
  e.reset(right,48)
  self.assertEqual(e.history,[]);self.assertEqual(e.frames,[]);self.assertEqual(e.tick,0)
  for _ in range(6):e.step()
  self.assertEqual([d['prediction'] for d in e.history],[2,2]);self.assertEqual(e.history[-1]['command'],2)
