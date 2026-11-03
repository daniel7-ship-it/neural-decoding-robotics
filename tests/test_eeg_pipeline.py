import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import mne,numpy as np
from neuralduck.eeg_pipeline import recording,manifest_document
from neuralduck.eeg_config import CONFIG
from neuralduck.eeg_research import save_checkpoint,load_checkpoint
from neuralduck.human_eeg_benchmark import model
class PipelineTests(unittest.TestCase):
 def test_execution_cache_and_manifest(self):
  sf=160;t=np.arange(1600)/sf;data=np.array([np.sin(2*np.pi*10*t),np.sin(2*np.pi*22*t)])*50e-6
  raw=mne.io.RawArray(data,mne.create_info(['C3','C4'],sf,'eeg'),verbose=False);raw.set_annotations(mne.Annotations([1,5],[3,3],['T1','T2']))
  with tempfile.TemporaryDirectory() as root:
   path=Path(root)/'raw.edf';path.write_bytes(b'fixture');cache=Path(root)/'cache'
   with patch('neuralduck.eeg_pipeline.mne.io.read_raw_edf',side_effect=lambda *a,**k:raw.copy()):
    x,y,g,m,hit=recording(path,2,4,'train',cache);self.assertFalse(hit)
    self.assertTrue(recording(path,2,4,'train',cache)[4]);self.assertEqual(x.shape[-1],320)
    records=[m]
    for kind in ('duration','offset','filter','quality','channels'):
     c=copy.deepcopy(CONFIG)
     if kind=='duration':c['window']['duration_seconds']=1
     if kind=='offset':c['window']['offset_seconds']=.5
     if kind=='filter':c['filter']['low_hz']=15
     if kind=='quality':c['quality']['peak_volts']=1e-9
     if kind=='channels':c['channels']=['C4','C3']
     a,b,good,entry,cached=recording(path,3,8,'validation',cache,c);self.assertFalse(cached);self.assertNotEqual(m['cache_fingerprint'],entry['cache_fingerprint'])
     if kind=='duration':self.assertEqual(a.shape[-1],160)
     if kind=='offset':self.assertNotEqual(m['trials'][0]['start_sample'],entry['trials'][0]['start_sample'])
     if kind=='filter':self.assertFalse(np.allclose(x,a))
     if kind=='quality':self.assertTrue(g.all());self.assertFalse(good.any())
     records.append(entry)
    document,key=manifest_document(records);self.assertEqual({e['split'] for e in document['recordings']},{'train','validation'})
    weights=Path(root)/'model.pt';save_checkpoint(weights,model(),np.zeros((1,64,1)),np.ones((1,64,1)),dict(manifest=document,manifest_sha256=key));_,_,_,metadata=load_checkpoint(weights,model);self.assertEqual(metadata['manifest_sha256'],manifest_document(metadata['manifest']['recordings'])[1])
if __name__=='__main__':unittest.main()
