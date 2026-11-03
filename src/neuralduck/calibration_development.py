"""Matched personal calibration comparison; development-only by default."""
import copy,json,time,os
import numpy as np
import torch
from torch import nn
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from .human_eeg_benchmark import ROOT,model
from .eeg_pipeline import recording,manifest_document
from .eeg_research import new_experiment,fit_normalization,probabilities,save_checkpoint,load_checkpoint

def run():
 torch.set_num_threads(2);out=new_experiment(ROOT/'artifacts/calibration-development');protocol=dict(scope='development; no fresh confirmation',subjects=[7,8],budgets=[4,8,12],seeds=[101,202,303],updates=50,learning_rate=.001,calibration_run=4,evaluation_run=12,policy='all trials scored; quality flags reported only',selection='first n/2 examples of each label, nested',pool='historical pooled models trained subjects2-6; checkpoint selected subjects7-8, hence development only')
 if os.environ.get('CALIBRATION_FINAL')=='1':
  protocol.update(scope='locked calibrated-person evaluation; run4 calibration/run12 final; run8 unused',subjects=list(range(15,23)),primary='participant-mean paired fine-tuning gain at budget12; average seeds within person',uncertainty='participant bootstrap 95%; eight participants',acceptance='positive lower CI for calibration gain and no claim of continuous control',exclusions='fail explicitly if fewer than budget trials or incompatible channels; no replacements',development_choice='50 updates Adam .001 all layers; one predefined configuration, no hyperparameter search')
  (ROOT/'docs/calibration-final-protocol.json').write_text(json.dumps(protocol,indent=2))
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];manifest=[]
 pooled=ROOT/'artifacts/human-eeg-runs/9770368c070d4f7585ccb9c558a0a890'
 for subject in protocol['subjects']:
  files=eegbci.load_data(subject,[4,12],path=str(ROOT/'data/physionet'),update_path=False,verbose=False);split={}
  for run,file in zip([4,12],files):
   x,y,g,m,_=recording(file,subject,run,'development',ROOT/'data/eeg-cache-v3');split[run]=(x,y,g,m);manifest.append(m)
  X,y,_,cm=split[4];T,ty,good,tm=split[12]
  for seed in protocol['seeds']:
   d=torch.load(pooled/f'cnn-{seed}.pt',weights_only=True);base=model();base.load_state_dict(d['state_dict']);pm=d['mean'].numpy();ps=d['scale'].numpy()
   for budget in protocol['budgets']:
    idx=np.sort(np.r_[np.flatnonzero(y==0)[:budget//2],np.flatnonzero(y==1)[:budget//2]])
    if len(idx)!=budget:raise ValueError('Insufficient calibration trials')
    for method in ('frozen','fine_tuned','subject_specific','csp_lda','shuffled'):
     started=time.perf_counter();losses=[]
     if method=='csp_lda':
      csp=CSP(n_components=2,reg='ledoit_wolf',log=True,norm_trace=False);clf=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto');clf.fit(csp.fit_transform(X[idx],y[idx]),y[idx]);p=clf.predict_proba(csp.transform(T));import joblib;joblib.dump(dict(csp=csp,classifier=clf),out/f'{subject}-{seed}-{budget}-{method}.joblib')
     else:
      torch.manual_seed(seed);net=model();mean,scale=(fit_normalization(X[idx]) if method=='subject_specific' else (pm,ps))
      if method!='subject_specific':net.load_state_dict(base.state_dict())
      if method!='frozen':
       opt=torch.optim.Adam(net.parameters(),lr=protocol['learning_rate']);target=y[idx].copy()
       if method=='shuffled':target=np.random.default_rng(seed).permutation(target)
       xt=torch.tensor((X[idx]-mean)/scale)
       for _ in range(protocol['updates']):
        opt.zero_grad();loss=nn.functional.cross_entropy(net(xt),torch.tensor(target));loss.backward();opt.step();losses.append(loss.item())
      p=probabilities(net,T,mean,scale);path=out/f'{subject}-{seed}-{budget}-{method}.pt';save_checkpoint(path,net,mean,scale,dict(protocol=protocol,calibration_manifest=cm,evaluation_manifest=tm,indices=idx.tolist(),method=method,seed=seed));reloaded,rm,rs,_=load_checkpoint(path,model);np.testing.assert_allclose(p,probabilities(reloaded,T,rm,rs),atol=1e-7,rtol=0)
     rows.append(dict(subject=subject,seed=seed,budget=budget,method=method,balanced_accuracy=float(balanced_accuracy_score(ty,p.argmax(1))),confusion=confusion_matrix(ty,p.argmax(1)).tolist(),probabilities=p.tolist(),labels=ty.tolist(),quality=good.tolist(),losses=losses,seconds=time.perf_counter()-started,collection_seconds=max(cm['trials'][i]['end_sample'] for i in idx)/cm['configuration']['sampling_hz']))
     (out/'results.json').write_text(json.dumps(rows));print(subject,seed,budget,method,rows[-1]['balanced_accuracy'],flush=True)
 document,key=manifest_document(manifest);(out/'manifest.json').write_text(json.dumps(dict(manifest=document,sha256=key)));print('OUTPUT',out,flush=True)
if __name__=='__main__':run()
