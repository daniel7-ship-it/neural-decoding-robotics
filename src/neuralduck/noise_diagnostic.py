import copy,json
import numpy as np
import torch
from torch import nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .supervised import ROOT,MODELS,NeuralDecoder,join,save_json
from .augmentation import recording
from .rest_gate import decisions
from .supervised import gated_predictions
from .rest_gate_review import measure

OUT=ROOT/'artifacts/noise-diagnostic';WEIGHTS=ROOT/'models/noise-diagnostic'
def main():
 torch.set_num_threads(1);p=json.loads((ROOT/'docs/noise-diagnostic-protocol.json').read_text(encoding='utf-8-sig'));OUT.mkdir(exist_ok=True);WEIGHTS.mkdir(exist_ok=True)
 splits={}
 for name in ('train','validation'):
  rows=[]
  for session in p['sessions'][name]:
   for condition in p['conditions']:rows.append(recording(session,condition))
  splits[name]=join(rows);print('generated',name,flush=True)
 tr,va=splits['train'],splits['validation'];models={};history={}
 # Compare feature groups to the original training normalization, on validation only.
 reference=NeuralDecoder.load(MODELS/'mlp-101.pt');z=(va['x']-reference.mean)/reference.scale
 save_json(OUT/'validation-normalization.json',{group:dict(median_absolute_z=float(np.median(abs(z[:,sl]))),p99_absolute_z=float(np.percentile(abs(z[:,sl]),99)),fraction_abs_z_over_5=float(np.mean(abs(z[:,sl])>5))) for group,sl in [('means',slice(0,24)),('standard_deviations',slice(24,48))]})
 for width in (24,48):
  scale=StandardScaler().fit(tr['x'][:,:width]);x=torch.tensor(scale.transform(tr['x'][:,:width]),dtype=torch.float32);vx=torch.tensor(scale.transform(va['x'][:,:width]),dtype=torch.float32);y=torch.tensor(tr['y']);vy=torch.tensor(va['y'])
  for seed in p['seeds']:
   torch.manual_seed(seed);model=nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,3));optimizer=torch.optim.Adam(model.parameters(),lr=.001);loss=nn.CrossEntropyLoss();best=float('inf');stale=0;h=[]
   for epoch in range(40):
    for batch in torch.randperm(len(x)).split(128):
     optimizer.zero_grad();l=loss(model(x[batch]),y[batch]);l.backward();optimizer.step()
    with torch.no_grad():tl=float(loss(model(x),y));vl=float(loss(model(vx),vy));acc=float((model(vx).argmax(1)==vy).float().mean())
    h.append(dict(epoch=epoch+1,training_loss=tl,validation_loss=vl,validation_accuracy=acc))
    if vl<best:best=vl;state=copy.deepcopy(model.state_dict());stale=0
    else:stale+=1
    if stale>=8:break
   model.load_state_dict(state);name=f'noise-{width}-{seed}';models[name]=(model,scale,width);history[name]=h
   torch.save(dict(state_dict=state,mean=torch.tensor(scale.mean_),scale=torch.tensor(scale.scale_),architecture=[width,32,3],parameters=sum(v.numel() for v in model.parameters()),protocol=p),WEIGHTS/f'{name}.pt');print(name,len(h),flush=True)
 lda=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')).fit(tr['x'],tr['y']);joblib.dump(lda,WEIGHTS/'lda.joblib');save_json(OUT/'training.json',history)
 print('locked before final',flush=True);results=[]
 for session in p['sessions']['test']:
  for condition in p['conditions']:
   row=recording(session,condition);predictions={}
   for name,(model,scale,width) in models.items():
    with torch.no_grad():predictions[name]=model(torch.tensor(scale.transform(row['x'][:,:width]),dtype=torch.float32)).softmax(1).numpy()
   for seed in p['seeds']:predictions[f'frozen-48-{seed}']=NeuralDecoder.load(MODELS/f'mlp-{seed}.pt').predict_proba(row['x'])
   predictions['broad-noise-lda']=lda.predict_proba(row['x'])
   for name,prob in predictions.items():
    for strict in (False,True):
     command=gated_predictions(decisions(prob,strict),row['time']);m=measure(row,command);pred=prob.argmax(1);cm=np.zeros((3,3),int)
     for actual,guess in zip(row['y'],pred):cm[actual,guess]+=1
     results.append(dict(session=session,condition=condition,model=name,strict=strict,confusion_matrix=cm.tolist(),balanced_accuracy=float(np.mean(np.diag(cm)/cm.sum(1))),**m))
     np.savez_compressed(OUT/f'{session}-{condition}-{name}-{strict}.npz',probability=prob,command=command,label=row['y'],time=row['time'])
   print('test',session,condition,flush=True)
 save_json(OUT/'evaluation.json',dict(protocol=p,rows=results))
if __name__=='__main__':main()
