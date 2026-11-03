import unittest,numpy as np
from neuralduck.feature_benchmark import bandpower
class BandpowerTest(unittest.TestCase):
 def test_frequency_and_units(self):
  t=np.arange(320)/160;x=np.sin(2*np.pi*10*t)[None,None,:]*10e-6;f=bandpower(x);self.assertEqual(f.shape,(1,3));self.assertEqual(int(f.argmax(1)[0]),0);np.testing.assert_allclose(bandpower(x*2)[0,0]-f[0,0],np.log(4),atol=1e-5)
if __name__=='__main__':unittest.main()
