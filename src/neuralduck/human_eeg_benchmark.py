"""Bounded subject-held-out human EEG benchmark; protocol fixed before execution."""
import json,time,copy,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from .eeg import load_physionet,windows_for_run
from .eeg_research import fingerprint,new_experiment,fit_normalization,probabilities,save_checkpoint,load_checkpoint
from .eeg_config import execution_config,CONFIG
from .eeg_pipeline import recording,manifest_document
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/human-eeg';OUT.mkdir(exist_ok=True)
def model():return nn.Sequential(nn.Conv1d(64,16,25,padding=12),nn.ReLU(),nn.AvgPool1d(8),nn.Conv1d(16,16,9,padding=4),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(16,2))
def metrics(y,p,subjects,good):
 result=dict(balanced_accuracy=float(balanced_accuracy_score(y,p)),confusion=confusion_matrix(y,p).tolist(),classification=classification_report(y,p,output_dict=True,zero_division=0),subjects={})
 scores=[]
 for s in np.unique(subjects):
  m=subjects==s;g=m&good;score=float(balanced_accuracy_score(y[m],p[m]));scores.append(score)
  result['subjects'][str(s)]=dict(balanced_accuracy=score,coverage=float(g.sum()/m.sum()),retained_balanced_accuracy=float(balanced_accuracy_score(y[g],p[g])) if g.any() else None,class_coverage={str(c):float(g[y==c].sum()/m[y==c].sum()) for c in (0,1)})
 rng=np.random.default_rng(40);boot=[np.mean(rng.choice(scores,len(scores),replace=True)) for _ in range(2000)]
 result['subject_mean_bootstrap_ci95']=np.percentile(boot,[2.5,97.5]).tolist();return result

def main():
 global OUT
 OUT=new_experiment(ROOT/'artifacts/human-eeg-runs')
 torch.set_num_threads(2);protocol=json.loads((ROOT/os.environ.get('EEG_PROTOCOL','docs/human-eeg-protocol.json')).read_text(encoding='utf-8-sig'));splits={};manifest_records=[]
 for name in ('train','validation','test'):
  xs=[];ys=[];ss=[];gs=[]
  for subject in protocol[name+'_subjects']:
   from mne.datasets import eegbci
   files=eegbci.load_data(subject,[4,8,12],path=str(ROOT/'data/physionet'),update_path=False,verbose=False)
   xx=[];yy=[];gg=[]
   for run,file in zip([4,8,12],files):
    rx,ry,rg,entry,_=recording(file,subject,run,name,ROOT/'data/eeg-cache-v3')
    xx.append(rx);yy.append(ry);gg.extend(rg);manifest_records.append(entry)
   x=np.concatenate(xx);y=np.concatenate(yy);g=np.asarray(gg)
   xs.append(x);ys.append(y);ss.extend([subject]*len(y));gs.extend(g);print(name,subject,len(y),flush=True)
  splits[name]=(np.concatenate(xs),np.concatenate(ys),np.array(ss),np.array(gs))
 manifest,manifest_hash=manifest_document(manifest_records)
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 X,y,_,_=splits['train'];V,vy,_,_=splits['validation'];T,ty,subjects,good=splits['test']
 mean,scale=fit_normalization(X)
 xt=torch.tensor((X-mean)/scale);vt=torch.tensor((V-mean)/scale);tt=torch.tensor((T-mean)/scale);yt=torch.tensor(y)
 results={};predictions={};started=time.perf_counter()
 for seed in protocol['seeds']:
  torch.manual_seed(seed);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');curves=[]
  for epoch in range(protocol['epochs']):
   net.train();losses=[]
   for batch in torch.randperm(len(y)).split(32):
    opt.zero_grad();loss=nn.functional.cross_entropy(net(xt[batch]),yt[batch]);loss.backward();opt.step();losses.append(loss.item())
   net.eval()
   with torch.no_grad():val=nn.functional.cross_entropy(net(vt),torch.tensor(vy)).item()
   curves.append([float(np.mean(losses)),val])
   if val<best:best=val;weights=copy.deepcopy(net.state_dict())
  net.load_state_dict(weights);net.eval()
  with torch.no_grad():p=net(tt).argmax(1).numpy()
  checkpoint=OUT/f'cnn-{seed}.pt';save_checkpoint(checkpoint,net,mean,scale,dict(protocol=protocol,manifest_sha256=manifest_hash,manifest=manifest,seed=seed,selected_epoch=int(np.argmin([v[1] for v in curves]))+1,architecture='temporal-cnn-27970',classes=['left','right']))
  reload,rm,rs,_=load_checkpoint(checkpoint,model)
  np.testing.assert_allclose(probabilities(net,T,mean,scale),probabilities(reload,T,rm,rs),atol=1e-7,rtol=0)
  results[str(seed)]=dict(**metrics(ty,p,subjects,good),parameters=sum(v.numel() for v in net.parameters()),curves=curves);predictions[str(seed)]=p.tolist();print('CNN',seed,results[str(seed)]['balanced_accuracy'],flush=True)
 csp=CSP(n_components=6,reg='ledoit_wolf',log=True,norm_trace=False);clf=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'));clf.fit(csp.fit_transform(X,y),y);p=clf.predict(csp.transform(T));results['csp_lda']=metrics(ty,p,subjects,good)
 import joblib
 joblib.dump(dict(csp=csp,classifier=clf),OUT/'csp-lda.joblib')
 predictions.update(csp_lda=p.tolist(),labels=ty.tolist(),subjects=subjects.tolist(),quality=good.tolist());(OUT/'predictions.json').write_text(json.dumps(predictions));results['runtime_seconds']=time.perf_counter()-started;(OUT/'results.json').write_text(json.dumps(results,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
