import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from neuralduck.eeg_research import fingerprint,new_experiment,fit_normalization,probabilities,save_checkpoint,load_checkpoint
from neuralduck.human_eeg_benchmark import model
class ResearchContracts(unittest.TestCase):
 def test_fingerprint_and_unique_outputs(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'raw';p.write_bytes(b'a');a=fingerprint([p],{'channels':['a','b']});self.assertNotEqual(a,fingerprint([p],{'channels':['b','a']}));p.write_bytes(b'b');self.assertNotEqual(a,fingerprint([p],{'channels':['a','b']}));self.assertNotEqual(new_experiment(root),new_experiment(root))
 def test_reload_full_probabilities_and_training_only_scale(self):
  rng=np.random.default_rng(10);train=rng.normal(size=(4,64,320)).astype('float32');test=rng.normal(20,2,size=(2,64,320)).astype('float32');mean,scale=fit_normalization(train);original=mean.copy();net=model();expected=probabilities(net,test,mean,scale);np.testing.assert_array_equal(mean,original)
  with tempfile.TemporaryDirectory() as root:
   path=Path(root)/'weights.pt';save_checkpoint(path,net,mean,scale,{'class_mapping':['left','right']});restored,m,s,metadata=load_checkpoint(path,model);np.testing.assert_allclose(expected,probabilities(restored,test,m,s),atol=1e-7,rtol=0)
  with self.assertRaises(ValueError):probabilities(net,np.zeros((1,63,320)),mean,scale)
if __name__=='__main__':unittest.main()
