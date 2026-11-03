"""Locked augmented recording comparison, CPU only."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .supervised import ROOT, MODELS, NeuralDecoder, generate_session, join, train, network, ridge, gated_predictions, save_json
from .rest_gate import decisions
from .rest_gate_review import measure
from .closed_loop import ContinuousCircuit
from .neural import VirtualElectrodes

SPEC=ROOT/'docs/augmentation-protocol.json'
OUT=ROOT/'artifacts/augmentation'
WEIGHTS=ROOT/'models/augmentation'

class Variation:
 def __init__(self,seed,session,name):
  self.name=name
  self.rng=np.random.default_rng(seed+session*100003+9001)
  self.shifted=VirtualElectrodes(seed=seed+session*7919,neuron_seed=seed,placement_shift=(.25 if name=='shift_train' else .65)) if name.startswith('shift') else None
 def apply(self,rates,observed,electrode,timestamp):
  x=observed.copy()
  if timestamp<2:return x
  if self.name.startswith('shift'):
   x+=rates@(self.shifted.mixing-electrode.mixing).T*electrode.gain[:,0]
  elif self.name.startswith('noise'):
   x+=self.rng.normal(0,.15 if self.name=='noise_train' else .5,len(x))
  elif self.name.startswith('gain'):
   x[12:]*=.75 if self.name=='gain_train' else .25
  return x.astype(np.float32)

def recording(session,condition):
 return generate_session(session,42,None if condition=='clean' else Variation(42,session,condition),save=False)

def row_metrics(row,prob,gate):
 pred=prob.argmax(1)
 if gate=='strict':pred=decisions(prob,True)
 command=gated_predictions(pred,row['time'])
 m=measure(row,command)
 cm=np.zeros((3,3),dtype=int)
 for y,p in zip(row['y'],prob.argmax(1)):cm[y,p]+=1
 m['confusion_matrix']=cm.tolist()
 m['balanced_accuracy']=float(np.mean(np.diag(cm)/np.maximum(cm.sum(1),1)))
 m['class_recall']=(np.diag(cm)/np.maximum(cm.sum(1),1)).tolist()
 return m,command

def run():
 torch.set_num_threads(1)
 p=json.loads(SPEC.read_text(encoding='utf-8-sig'))
 OUT.mkdir(parents=True,exist_ok=True);WEIGHTS.mkdir(parents=True,exist_ok=True)
 trainrows=[];valrows=[]
 for session in p['sessions']['train']:
  for condition in p['training_conditions']:trainrows.append(recording(session,condition))
  print('train session',session,flush=True)
 for session in p['sessions']['validation']:
  for condition in p['training_conditions']:valrows.append(recording(session,condition))
 tr,va=join(trainrows),join(valrows)
 training={};models={}
 params=dict(max_epochs=p['max_epochs'],patience=p['early_stop_patience'],learning_rate=p['learning_rate'],batch_size=p['batch_size'])
 for seed in p['initializations']:
  model,history,delta=train(tr,va,seed,params)
  name=f'augmented-mlp-{seed}'
  path=WEIGHTS/f'{name}.pt'
  torch.save(dict(state_dict=model.model.state_dict(),mean=torch.tensor(model.mean),scale=torch.tensor(model.scale),classes=['rest','left','right'],architecture=p['architecture'],protocol=p),path)
  models[name]=NeuralDecoder.load(path)
  training[name]=dict(history=history,weight_change_l2=delta)
  print(name,len(history),flush=True)
 lda=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')).fit(tr['x'],tr['y'])
 direct=ridge(1.).fit(tr['x'],tr['drive'])
 joblib.dump(lda,WEIGHTS/'augmented-lda.joblib');joblib.dump(direct,WEIGHTS/'augmented-ridge.joblib')
 models['augmented-lda']=lda;models['augmented-ridge']=direct
 for seed in p['initializations']:models[f'frozen-mlp-{seed}']=NeuralDecoder.load(MODELS/f'mlp-{seed}.pt')
 save_json(OUT/'training.json',training)
 save_json(OUT/'model-lock.json',{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in WEIGHTS.iterdir()})
 print('models locked before fresh test',flush=True)
 results=[]
 for session in p['sessions']['test']:
  for condition in p['evaluation_conditions']:
   row=recording(session,condition)
   for name,model in models.items():
    if name=='augmented-ridge':
     drive=model.predict(row['x']);pred=np.where(drive>=.18,1,np.where(drive<=-.18,2,0));prob=np.eye(3)[pred]
    else:prob=model.predict_proba(row['x'])
    for gate in ('original','strict'):
     m,command=row_metrics(row,prob,gate)
     results.append(dict(session=session,condition=condition,model=name,gate=gate,**m))
     np.savez_compressed(OUT/f'{session}-{condition}-{name}-{gate}.npz',probability=prob,command=command,label=row['y'],time=row['time'])
   for gate in ('original','strict'):
    m,command=row_metrics(row,np.eye(3)[np.zeros(len(row['y']),dtype=int)],gate)
    results.append(dict(session=session,condition=condition,model='always-rest',gate=gate,**m))
   print('test',session,condition,flush=True)
 save_json(OUT/'evaluation.json',dict(protocol=p,rows=results))
 print('evaluation rows',len(results),flush=True)
if __name__=='__main__':run()
