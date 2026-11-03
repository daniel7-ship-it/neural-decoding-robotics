"""Development-only EEG diagnostics; preserves original experiment."""
import json,uuid,hashlib,shutil,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from .eeg import load_physionet,windows_for_run
from .human_eeg_benchmark import ROOT,model

def main():
 torch.set_num_threads(2);out=ROOT/'artifacts/eeg-diagnostics'/uuid.uuid4().hex;out.mkdir(parents=True)
 old=ROOT/'artifacts/human-eeg';archive=out/'original';archive.mkdir()
 hashes={}
 for p in old.iterdir():
  if p.suffix in ('.json','.pt','.joblib'):
   shutil.copy2(p,archive/p.name);hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 (out/'archive-hashes.json').write_text(json.dumps(hashes,indent=2))
 results={};quality=[];manifest=[]
 for subject in (2,3,4,5,6):
  split={}
  for run,(raw,events,mapping) in load_physionet(subject,ROOT/'data/physionet').items():
   x,y=windows_for_run(raw,events,mapping,run);split[run]=(x,y-1)
   before=raw.get_data();before=before-before.mean(axis=0,keepdims=True);sf=raw.info['sfreq']
   for trial,(sample,_,code) in enumerate(events):
    if int(code) not in mapping:continue
    a=int(sample+sf);b=a+int(2*sf);epoch=before[:,a:b]
    if epoch.shape[1]!=int(2*sf):continue
    peak=np.max(np.abs(epoch),axis=1)*1e6;ptp=np.ptp(epoch,axis=1)*1e6
    quality.append(dict(subject=subject,run=run,trial=trial,label=int(mapping[int(code)]),peak_uv=peak.tolist(),ptp_uv=ptp.tolist(),extreme_channels=[raw.ch_names[i] for i in np.flatnonzero(peak>200)],flat_channels=[raw.ch_names[i] for i in np.flatnonzero(ptp<.1)]))
   manifest.append(dict(subject=subject,run=run,channels=raw.ch_names,sampling_hz=sf,units='volts',reference='average',trials=len(y)))
  X,y=split[4];V,vy=split[8];T,ty=split[12];mean=X.mean(axis=(0,2),keepdims=True);scale=X.std(axis=(0,2),keepdims=True).clip(1e-8)
  xt=torch.tensor((X-mean)/scale);vt=torch.tensor((V-mean)/scale);tt=torch.tensor((T-mean)/scale)
  torch.manual_seed(101);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');curves=[]
  for epoch in range(40):
   opt.zero_grad();loss=nn.functional.cross_entropy(net(xt),torch.tensor(y));loss.backward();opt.step()
   with torch.no_grad():val=nn.functional.cross_entropy(net(vt),torch.tensor(vy)).item()
   curves.append([loss.item(),val])
   if val<best:best=val;weights=copy.deepcopy(net.state_dict())
  net.load_state_dict(weights)
  with torch.no_grad():p=net(tt).argmax(1).numpy();tr=net(xt).argmax(1).numpy()
  csp=CSP(n_components=6,reg='ledoit_wolf',log=True,norm_trace=False);clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto');clf.fit(csp.fit_transform(X,y),y);cp=clf.predict(csp.transform(T))
  results[str(subject)]=dict(cnn=float(balanced_accuracy_score(ty,p)),cnn_train=float(balanced_accuracy_score(y,tr)),csp=float(balanced_accuracy_score(ty,cp)),curves=curves,predictions=p.tolist(),labels=ty.tolist())
  torch.save(dict(state_dict=weights,mean=torch.tensor(mean),scale=torch.tensor(scale)),out/f'subject-{subject}.pt');print(subject,results[str(subject)]['cnn'],results[str(subject)]['csp'],flush=True)
 (out/'results.json').write_text(json.dumps(results,indent=2));(out/'quality.json').write_text(json.dumps(quality));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
