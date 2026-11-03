import inspect
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from neuralduck.supervised import (NeuralDecoder, causal_features, network, protocol,
                                   train, gated_predictions, metrics, passes, summarize)

class SupervisedContracts(unittest.TestCase):
    def test_acceptance_requires_movement_and_checks_delay(self):
        row=dict(y=np.array([0,0,1,1,1,2,2,2]),time=np.arange(1,9)*.1,
                 block=np.array([0,0,1,1,1,2,2,2]),onset=np.array([0,0,.2,.2,.2,.5,.5,.5]))
        self.assertFalse(passes(summarize([metrics(row,np.zeros(8,dtype=int))]),protocol()['criteria']))
        good=summarize([metrics(row,row['y'])])
        self.assertTrue(passes(good,protocol()['criteria']))
        good['median_onset_delay_seconds']=.6
        self.assertFalse(passes(good,protocol()['criteria']))

    def test_splits_and_training_only_preprocessing(self):
        p=protocol()
        self.assertTrue(min(p['splits']['test'])>10)
        rng=np.random.default_rng(5)
        tr=dict(x=rng.normal(size=(60,48)).astype('float32'),y=np.tile([0,1,2],20))
        va=dict(x=np.full((12,48),100,dtype='float32'),y=np.tile([0,1,2],4))
        short={**p,'max_epochs':1}
        model,_,_=train(tr,va,101,short)
        np.testing.assert_allclose(model.mean,tr['x'].mean(0),atol=1e-6)
        self.assertLess(abs(model.mean).max(),1)

    def test_causal_windows(self):
        recordings=np.random.default_rng(2).normal(size=(100,24)).astype('float32')
        a,ends=causal_features(recordings)
        changed=recordings.copy()
        changed[50:]+=10000
        b,_=causal_features(changed)
        np.testing.assert_array_equal(a[ends<50],b[ends<50])
        np.testing.assert_allclose(a[0,:24],recordings[:20].mean(0))

    def test_input_boundary_and_saved_round_trip(self):
        self.assertEqual(list(inspect.signature(NeuralDecoder.predict_proba).parameters),['self','electrode_features'])
        model=network()
        self.assertEqual(sum(w.numel() for w in model.parameters()),1667)
        decoder=NeuralDecoder(model,np.zeros(48),np.ones(48))
        with self.assertRaises(ValueError): decoder.predict_proba(np.zeros((1,96)))
        with self.assertRaises(ValueError): decoder.predict_proba(np.full((1,48),np.nan))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save(dict(state_dict=model.state_dict(),mean=torch.zeros(48),scale=torch.ones(48)),path)
            restored=NeuralDecoder.load(path)
            x=np.random.default_rng(3).normal(size=(4,48)).astype('float32')
            np.testing.assert_array_equal(decoder.predict_proba(x),restored.predict_proba(x))

    def test_gate_and_event_counts_include_transitions(self):
        pred=np.array([0,1,1,0,2,2,2,0])
        np.testing.assert_array_equal(gated_predictions(pred,np.arange(8)*.1),[0,0,1,0,0,2,2,0])
        row=dict(y=np.zeros(8,dtype=int),time=np.arange(8)*.1,block=np.zeros(8,dtype=int),onset=np.zeros(8))
        result=metrics(row,pred)
        self.assertEqual(result['false_rest_events'],2)
        self.assertEqual(result['long_rest_occupancy'],3/8)

if __name__=='__main__': unittest.main()
