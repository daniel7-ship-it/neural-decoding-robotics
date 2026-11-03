import unittest
import numpy as np
from neuralduck.augmentation import Variation
from neuralduck.neural import VirtualElectrodes
class AugmentationContracts(unittest.TestCase):
 def test_recording_only_deterministic_and_pre_onset(self):
  electrode=VirtualElectrodes(seed=42+29*7919,neuron_seed=42)
  rates=np.ones(96)
  base=np.ones(24,dtype=np.float32)
  for name in ('noise_train','shift_train','gain_train','noise_hard','shift_hard','gain_hard'):
   a=Variation(42,29,name); b=Variation(42,29,name)
   np.testing.assert_array_equal(a.apply(rates,base,electrode,1.9),base)
   np.testing.assert_array_equal(a.apply(rates,base,electrode,2.),b.apply(rates,base,electrode,2.))
   np.testing.assert_array_equal(rates,np.ones(96))
if __name__=='__main__':unittest.main()
