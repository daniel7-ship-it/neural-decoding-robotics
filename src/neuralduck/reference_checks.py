"""Development-only reference, optimization and channel-policy diagnostics."""
import json,uuid,copy
import numpy as np
import torch,mne
from torch import nn
from mne.datasets import eegbci
from scipy.signal import butter,sosfilt
from sklearn.metrics import balanced_accuracy_score
from .human_eeg_benchmark import ROOT,model
from .eeg_research import fit_normalization,probabilities,save_checkpoint,load_checkpoint
POLICY={'subjects':[2,3,4,5,6],'runs':[4,8,12],'channel_rule':'Flag if abs amplitude >200uV in >50% of training-run imagery epochs, before rereferencing; exclude from reference and zero that channel after filtering in all runs','budget':'seed101, 150 full-batch Adam updates, select minimum run8 loss; 200 updates tiny overfit check','shuffle':'permute training labels only; validation uses same fixed checkpoint rule','scope':'development only; no fresh confirmation'}
def main():
 torch.set_num_threads(2);out=ROOT/'artifacts/reference-checks'/uuid.uuid4().hex;out.mkdir(parents=True);(out/'protocol.json').write_text(json.dumps(POLICY,indent=2));results={}
 for subject in POLICY['subjects']:
  data={};diagnostics=[]
  files=eegbci.load_data(subject,POLICY['runs'],path=str(ROOT/'data/physionet'),update_path=False,verbose=False)
  for run,file in zip(POLICY['runs'],files):
   raw=mne.io.read_raw_edf(file,preload=True,verbose=False);eegbci.standardize(raw);raw.pick('eeg');x=raw.get_data();sf=raw.info['sfreq'];events,ids=mne.events_from_annotations(raw,verbose=False);mapping={ids['T1']:0,ids['T2']:1};trials=[]
   for sample,_,code in events:
    if code in mapping and sample+3*sf<=x.shape[1]:trials.append((int(sample+sf),int(sample+3*sf),mapping[code]))
   data[run]=(x,sf,trials,raw.ch_names)
   ref=x-x.mean(axis=0,keepdims=True)
   for a,b,y in trials:
    diagnostics.append(dict(run=run,label=y,before_peak_uv=(np.max(np.abs(x[:,a:b]),axis=1)*1e6).tolist(),after_peak_uv=(np.max(np.abs(ref[:,a:b]),axis=1)*1e6).tolist()))
  x,sf,trials,names=data[4];bad=np.mean([np.max(np.abs(x[:,a:b]),axis=1)>200e-6 for a,b,y in trials],axis=0)>.5
  if bad.all():raise ValueError('Policy rejects all channels')
  report=dict(bad_channels=[names[i] for i in np.flatnonzero(bad)],reference_units='microvolts',reference=diagnostics)
  for condition in ('original','channel_policy'):
   split={}
   for run,(x,sf,trials,names) in data.items():
    x=x.copy();ref=x[~bad].mean(axis=0,keepdims=True) if condition=='channel_policy' else x.mean(axis=0,keepdims=True);x=x-ref
    x=sosfilt(butter(4,[8,30],btype='bandpass',fs=sf,output='sos'),x,axis=-1)
    if condition=='channel_policy':x[bad]=0
    split[run]=(np.array([x[:,a:b] for a,b,y in trials],dtype=np.float32),np.array([y for a,b,y in trials]))
   X,y=split[4];V,vy=split[8];T,ty=split[12];mean,scale=fit_normalization(X);xt=torch.tensor((X-mean)/scale);vt=torch.tensor((V-mean)/scale);tt=torch.tensor((T-mean)/scale)
   conditions={}
   for shuffled in (False,True):
    torch.manual_seed(101);net=model();opt=torch.optim.Adam(net.parameters(),lr=.001);target=np.random.default_rng(101).permutation(y) if shuffled else y;best=float('inf');curve=[]
    for epoch in range(150):
     opt.zero_grad();loss=nn.functional.cross_entropy(net(xt),torch.tensor(target));loss.backward();opt.step()
     with torch.no_grad():val=nn.functional.cross_entropy(net(vt),torch.tensor(vy)).item()
     curve.append([loss.item(),val])
     if val<best:best=val;weights=copy.deepcopy(net.state_dict())
    net.load_state_dict(weights)
    checkpoint=out/f'{subject}-{condition}-{shuffled}.pt'
    save_checkpoint(checkpoint,net,mean,scale,dict(subject=subject,condition=condition,shuffled=shuffled,channels=names,policy=POLICY))
    restored,rm,rs,_=load_checkpoint(checkpoint,model)
    np.testing.assert_allclose(probabilities(net,T,mean,scale),probabilities(restored,T,rm,rs),atol=1e-7,rtol=0)
    with torch.no_grad():p=net(tt).argmax(1).numpy();train=net(xt).argmax(1).numpy()
    conditions[str(shuffled)]=dict(run12_balanced_accuracy=float(balanced_accuracy_score(ty,p)),train_balanced_accuracy=float(balanced_accuracy_score(target,train)),predictions=p.tolist(),labels=ty.tolist(),curve=curve)
   torch.manual_seed(101);tiny=model();opt=torch.optim.Adam(tiny.parameters(),lr=.003);idx=np.r_[np.flatnonzero(y==0)[:4],np.flatnonzero(y==1)[:4]]
   for _ in range(200):
    opt.zero_grad();loss=nn.functional.cross_entropy(tiny(xt[idx]),torch.tensor(y[idx]));loss.backward();opt.step()
   with torch.no_grad():accuracy=float((tiny(xt[idx]).argmax(1).numpy()==y[idx]).mean())
   report[condition]=dict(tiny_overfit_accuracy=accuracy,conditions=conditions)
  results[str(subject)]=report;(out/'results.json').write_text(json.dumps(results));print(subject,report['bad_channels'],{c:report[c]['conditions']['False']['run12_balanced_accuracy'] for c in ('original','channel_policy')},flush=True)
 print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
