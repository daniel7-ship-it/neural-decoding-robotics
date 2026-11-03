import copy,inspect,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from neuralduck.feature_inference import Decoder,supplement
from neuralduck.feature_benchmark import fit_training_csp,fit_training_scaling
from neuralduck.completion_report import paired,enrich,FEATURE

class SavedInferenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  supplement(FEATURE);cls.manifest=json.loads((FEATURE/'manifest.json').read_text())['manifest']['recordings'];cls.cfg=next(m['configuration'] for m in cls.manifest if m['subject']==2 and m['run']==4);cls.x=np.random.default_rng(10).normal(0,1e-5,(3,64,320)).astype('float32')
 def test_all_representation_loaders_and_no_fitting_or_labels(self):
  for rep,method in [('time','neural'),('bandpower','neural'),('csp','neural'),('bandpower','lda'),('csp','lda')]:
   with self.subTest(rep=rep,method=method):
    d=Decoder(FEATURE,2,rep,method,101)
    with patch('mne.decoding.CSP.fit',side_effect=AssertionError('inference fitted CSP')),patch('sklearn.preprocessing.StandardScaler.fit',side_effect=AssertionError('inference fitted scaler')):
     p=d.predict(self.x,self.cfg['channels'],160)
    self.assertEqual(p.shape,(3,2));np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    self.assertNotIn('labels',inspect.signature(d.predict).parameters)
 def test_reject_config_shape_rate_order_and_nonfinite(self):
  for key,value in [('units','microvolts'),('sampling_hz',128),('annotations',{'T1':1,'T2':0}),('window',dict(duration_seconds=1,offset_seconds=1))]:
   c=copy.deepcopy(self.cfg);c[key]=value
   with self.assertRaises(ValueError):Decoder(FEATURE,2,'time','neural',101,c)
  d=Decoder(FEATURE,2,'time','neural',101)
  for x,ch,sf in [(self.x,self.cfg['channels'][::-1],160),(self.x,self.cfg['channels'],128),(self.x[:,:,:100],self.cfg['channels'],160),(self.x*float('nan'),self.cfg['channels'],160)]:
   with self.assertRaises(ValueError):d.predict(x,ch,sf)
 def test_artifact_integrity_and_missing_transform(self):
  with tempfile.TemporaryDirectory() as td:
   import shutil
   root=Path(td)
   for name in ['manifest.json','inference-contract.json','2-csp-101.pt','2-csp-transform.joblib']:shutil.copy2(FEATURE/name,root/name)
   Decoder(root,2,'csp','neural',101)
   (root/'2-csp-transform.joblib').write_bytes(b'changed')
   with self.assertRaises(ValueError):Decoder(root,2,'csp','neural',101)
   (root/'2-csp-transform.joblib').unlink()
   with self.assertRaises(ValueError):Decoder(root,2,'csp','neural',101)
 def test_csp_and_normalization_only_receive_training_arrays(self):
  train=self.x;labels=np.array([0,1,0]);sentinel=object()
  with patch('neuralduck.feature_benchmark.CSP') as constructor:
   constructor.return_value.fit.return_value=sentinel
   self.assertIs(fit_training_csp(train,labels),sentinel)
   args=constructor.return_value.fit.call_args.args;self.assertIs(args[0],train);self.assertIs(args[1],labels)
  m,s=fit_training_scaling(train,'time');np.testing.assert_equal(m,train.mean((0,2),keepdims=True));np.testing.assert_equal(s,train.std((0,2),keepdims=True))
  # Different held-out recordings cannot affect a fitting API that only receives train.
  heldout=train*1e6
  m2,s2=fit_training_scaling(train,'time');np.testing.assert_equal(m,m2);np.testing.assert_equal(s,s2)
  self.assertFalse(np.allclose(m,heldout.mean((0,2),keepdims=True)))
 def test_fresh_process_csp_prediction(self):
  with tempfile.TemporaryDirectory() as td:
   x=Path(td)/'x.npy';out=Path(td)/'p.npy';np.save(x,self.x)
   code="from neuralduck.feature_inference import Decoder;import numpy as np,sys;d=Decoder(sys.argv[1],2,'csp','neural',101);np.save(sys.argv[3],d.predict(np.load(sys.argv[2]),d.configuration['channels'],160))"
   subprocess.run([sys.executable,'-c',code,str(FEATURE),str(x),str(out)],check=True,capture_output=True)
   d=Decoder(FEATURE,2,'csp','neural',101);np.testing.assert_allclose(np.load(out),d.predict(self.x,self.cfg['channels'],160),atol=1e-6,rtol=0)
 def test_pairing_averages_seeds_within_people(self):
  rows=[dict(subject=1,group='a',score=v) for v in [0,0,1]]+[dict(subject=2,group='a',score=1),dict(subject=1,group='b',score=.5),dict(subject=2,group='b',score=.5)]
  p=paired(rows,lambda r:r['group'],('a','b'));self.assertAlmostEqual(p['mean'],1/6);self.assertEqual(p['participants'],[1,2])
if __name__=='__main__':unittest.main()
