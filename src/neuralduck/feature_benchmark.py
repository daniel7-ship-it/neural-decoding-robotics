"""Exploratory matched EEG representation benchmark."""
import json,copy,time
import numpy as np
import torch,joblib
from torch import nn
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score,recall_score,confusion_matrix
from mne.decoding import CSP
from mne.datasets import eegbci
from .human_eeg_benchmark import ROOT,model
from .eeg_pipeline import recording,manifest_document
from .eeg_research import new_experiment

def bandpower(x):
 f,p=welch(x,fs=160,nperseg=160,axis=-1,scaling='density');return np.concatenate([np.log(np.trapezoid(p[..., (f>=a)&(f<b)],f[(f>=a)&(f<b)],axis=-1).clip(1e-20)) for a,b in [(8,13),(13,20),(20,30)]],axis=1).astype('float32')
def fit_training_csp(training_electrodes, training_labels):
 """The sole CSP fit boundary; validation/evaluation data are not arguments."""
 return CSP(n_components=6,reg='ledoit_wolf',log=True,norm_trace=False).fit(training_electrodes,training_labels)
def fit_training_scaling(training_features, representation):
 axis=(0,2) if representation=='time' else 0
 return training_features.mean(axis=axis,keepdims=True),training_features.std(axis=axis,keepdims=True).clip(1e-8)

def main():
 torch.set_num_threads(2);out=new_experiment(ROOT/'artifacts/feature-benchmark');protocol=dict(subjects=[2,3,4,5,6],runs={'train':4,'validation':8,'evaluation':12},seeds=[101,202,303],epochs=60,lr=.001,features='Welch160Hz 1s segments, log integrated V^2 bands8-13/13-20/20-30; CSP6 log variance',teacher='statistical model must achieve validation BA>=.80 and each recall>=.75 in ALL five development subjects',selection='minimum validation loss',quality='flags only; all trials evaluated',maximum_configurations='three representations; one MLP32 per feature; LDA shrinkage auto; fixed CNN')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];manifest=[];teacher=[]
 for subject in protocol['subjects']:
  split={};files=eegbci.load_data(subject,[4,8,12],path=str(ROOT/'data/physionet'),update_path=False,verbose=False)
  for run,file in zip([4,8,12],files):
   x,y,g,m,_=recording(file,subject,run,'development',ROOT/'data/eeg-cache-v3');split[run]=(x,y,g);manifest.append(m)
  X,y,_=split[4];V,vy,_=split[8];T,ty,good=split[12];csp=fit_training_csp(X,y)
  for representation in ('time','bandpower','csp'):
   transform=lambda x: x if representation=='time' else bandpower(x) if representation=='bandpower' else csp.transform(x).astype('float32')
   A,B,C=map(transform,(X,V,T));mean,scale=fit_training_scaling(A,representation);a,b,c=[torch.tensor((z-mean)/scale,dtype=torch.float32) for z in (A,B,C)]
   for seed in protocol['seeds']:
    torch.manual_seed(seed);net=model() if representation=='time' else nn.Sequential(nn.Linear(A.shape[1],32),nn.ReLU(),nn.Linear(32,2));opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');curves=[];start=time.perf_counter()
    for epoch in range(protocol['epochs']):
     opt.zero_grad();loss=nn.functional.cross_entropy(net(a),torch.tensor(y));loss.backward();opt.step()
     with torch.no_grad():val=nn.functional.cross_entropy(net(b),torch.tensor(vy)).item()
     curves.append([loss.item(),val])
     if val<best:best=val;weights=copy.deepcopy(net.state_dict())
    net.load_state_dict(weights);net.eval()
    with torch.no_grad():p=net(c).softmax(1).numpy()
    path=out/f'{subject}-{representation}-{seed}.pt';torch.save(dict(weights=weights,mean=torch.tensor(mean),scale=torch.tensor(scale),representation=representation),path);pipeline=out/f'{subject}-{representation}-transform.joblib';joblib.dump(csp if representation=='csp' else None,pipeline)
    saved=torch.load(path,weights_only=True);net.load_state_dict(saved['weights']);fresh=transform(T)
    with torch.no_grad():np.testing.assert_allclose(p,net(torch.tensor((fresh-saved['mean'].numpy())/saved['scale'].numpy(),dtype=torch.float32)).softmax(1).numpy(),atol=1e-7,rtol=0)
    rows.append(dict(subject=subject,representation=representation,method='neural',seed=seed,accuracy=float(balanced_accuracy_score(ty,p.argmax(1))),recall=recall_score(ty,p.argmax(1),average=None).tolist(),confusion=confusion_matrix(ty,p.argmax(1)).tolist(),probabilities=p.tolist(),labels=ty.tolist(),coverage=float(good.mean()),parameters=sum(z.numel() for z in net.parameters()),curves=curves,seconds=time.perf_counter()-start))
   if representation!='time':
    scaler=StandardScaler().fit(A);clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(scaler.transform(A),y);vp=clf.predict(scaler.transform(B));p=clf.predict_proba(scaler.transform(C));teacher.append(dict(subject=subject,representation=representation,qualified=bool(balanced_accuracy_score(vy,vp)>=.8 and np.min(recall_score(vy,vp,average=None))>=.75)));joblib.dump(dict(scaler=scaler,classifier=clf,csp=csp if representation=='csp' else None),out/f'{subject}-{representation}-lda.joblib');rows.append(dict(subject=subject,representation=representation,method='lda',seed=0,accuracy=float(balanced_accuracy_score(ty,p.argmax(1))),recall=recall_score(ty,p.argmax(1),average=None).tolist(),probabilities=p.tolist(),labels=ty.tolist(),coverage=float(good.mean())))
   (out/'results.json').write_text(json.dumps(rows));print(subject,representation,flush=True)
 document,key=manifest_document(manifest);(out/'manifest.json').write_text(json.dumps(dict(manifest=document,hash=key)));(out/'teacher.json').write_text(json.dumps(teacher));print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
