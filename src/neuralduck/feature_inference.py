"""Load legacy EEG decoders with verified supplemental artifact contracts.

predict() accepts preprocessed volts windows only. predict_recording() applies the
shared causal source preprocessing first; annotation labels are returned solely
for scoring and never passed to predict(). Only load trusted local joblib files.
"""
import argparse,copy,hashlib,json,platform,tempfile,time
from pathlib import Path
import joblib,numpy as np,torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score,classification_report,confusion_matrix
from .feature_benchmark import bandpower
from .human_eeg_benchmark import model
from .eeg_pipeline import recording,digest

CONTRACT_KEYS=('channels','sampling_hz','window','filter','reference','units','annotations','quality')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def artifact_names(row):
 stem=f"{row['subject']}-{row['representation']}"
 return [stem+'-lda.joblib'] if row['method']=='lda' else [f"{stem}-{row['seed']}.pt",stem+'-transform.joblib']
def supplement(root):
 root=Path(root);manifest=json.loads((root/'manifest.json').read_text());rows=json.loads((root/'results.json').read_text())
 if digest(manifest['manifest'])!=manifest['hash']:raise ValueError('Dataset manifest hash mismatch')
 files=sorted({f for r in rows for f in artifact_names(r)})
 document=dict(schema=1,status='retrospective supplement; original legacy weights unchanged',dataset_manifest_sha256=sha(root/'manifest.json'),protocol_sha256=sha(root/'protocol.json'),results_sha256=sha(root/'results.json'),artifacts={f:sha(root/f) for f in files},classes={'0':'imagined left fist (T1)','1':'imagined right fist (T2)'},evidence='Saved protocol, dataset manifest, results, checkpoint tensor shapes and fitted transforms',unrecoverable=['Original runtime environment and exact source snapshot at training','Per-statistical-model historical training time','Contemporaneous checkpoint-bound hashes; these hashes establish current integrity only'])
 path=root/'inference-contract.json'
 if path.exists():
  if json.loads(path.read_text())!=document:raise ValueError('Supplement contract changed; investigate artifacts')
 else:path.write_text(json.dumps(document,indent=2))
 return document

class Decoder:
 def __init__(self,root,subject,representation,method,seed,configuration=None):
  root=Path(root);self.root=root
  contract=json.loads((root/'inference-contract.json').read_text())
  if sha(root/'manifest.json')!=contract['dataset_manifest_sha256']:raise ValueError('Dataset manifest changed')
  manifest=json.loads((root/'manifest.json').read_text());bound=next(m['configuration'] for m in manifest['manifest']['recordings'] if m['subject']==subject and m['run']==4)
  if representation not in ('time','bandpower','csp') or method not in ('neural','lda') or (representation=='time' and method=='lda'):raise ValueError('Unsupported decoder')
  if configuration is not None:
   for key in CONTRACT_KEYS:
    if configuration.get(key)!=bound.get(key):raise ValueError('Configuration differs from training manifest: '+key)
  if bound['sampling_hz']!=160 or len(bound['channels'])!=64:raise ValueError('Unsupported trained feature contract')
  self.configuration=copy.deepcopy(bound);self.rep=representation;self.method=method;self.classes=contract['classes']
  for name in artifact_names(dict(subject=subject,representation=representation,method=method,seed=seed)):
   if name not in contract['artifacts'] or not (root/name).is_file() or sha(root/name)!=contract['artifacts'][name]:raise ValueError('Missing or changed model artifact: '+name)
  stem=f'{subject}-{representation}'
  if method=='lda':
   self.pipeline=joblib.load(root/f'{stem}-lda.joblib');self.csp=self.pipeline['csp']
   if list(self.pipeline['classifier'].classes_)!=[0,1]:raise ValueError('Class ordering mismatch')
  else:
   saved=torch.load(root/f'{stem}-{seed}.pt',weights_only=True)
   if saved['representation']!=representation:raise ValueError('Representation mismatch')
   self.mean=saved['mean'].numpy();self.scale=saved['scale'].numpy();self.csp=joblib.load(root/f'{stem}-transform.joblib')
   width=192 if representation=='bandpower' else 6
   self.net=model() if representation=='time' else nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,2));self.net.load_state_dict(saved['weights']);self.net.eval()
   expected=(1,64,1) if representation=='time' else (1,width)
   if self.mean.shape!=expected or self.scale.shape!=expected or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale<=0):raise ValueError('Invalid normalization')
  if representation=='csp' and self.csp is None:raise ValueError('Missing CSP transform')
 def features(self,x,channels,sampling_rate):
  c=self.configuration
  if list(channels)!=c['channels'] or sampling_rate!=c['sampling_hz']:raise ValueError('Incompatible channels or sampling rate')
  expected=int(sampling_rate*c['window']['duration_seconds'])
  if not isinstance(x,np.ndarray) or x.ndim!=3 or not len(x) or x.shape[1:]!=(len(channels),expected) or not np.isfinite(x).all():raise ValueError('Invalid electrode windows')
  return x if self.rep=='time' else bandpower(x) if self.rep=='bandpower' else self.csp.transform(x).astype('float32')
 def predict_features(self,f):
  if self.method=='lda':return self.pipeline['classifier'].predict_proba(self.pipeline['scaler'].transform(f))
  with torch.inference_mode():return self.net(torch.tensor((f-self.mean)/self.scale,dtype=torch.float32)).softmax(1).numpy()
 def predict(self,x,channels,sampling_rate):return self.predict_features(self.features(x,channels,sampling_rate))
 def predict_recording(self,path,subject,run,cache):
  if run not in (4,8,12):raise ValueError('Unsupported imagery run')
  x,y,g,m,hit=recording(path,subject,run,'verification',cache,self.configuration)
  if m['configuration']['sampling_hz']!=self.configuration['sampling_hz']:raise ValueError('Source sample rate differs')
  p=self.predict(x,m['configuration']['channels'],m['configuration']['sampling_hz'])
  return p,y,g,m,hit

def timed(fn,repeats=10):
 for _ in range(3):fn()
 values=[]
 for _ in range(repeats):
  t=time.perf_counter();fn();values.append(time.perf_counter()-t)
 return dict(median_seconds=float(np.median(values)),p95_seconds=float(np.quantile(values,.95)))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('experiment');args=parser.parse_args();root=Path(args.experiment);torch.set_num_threads(2);supplement(root)
 rows=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())['manifest']['recordings'];output=[];sources={}
 for subject in sorted({r['subject'] for r in rows}):
  entry=next(m for m in manifest if m['subject']==subject and m['run']==12)
  if sha(entry['source'])!=entry['sha256']:raise ValueError('Source hash mismatch')
  with tempfile.TemporaryDirectory() as temp:
   t=time.perf_counter();data=recording(entry['source'],subject,12,'verification',Path(temp),entry['configuration']);cold=time.perf_counter()-t
   assert not data[4]
   warm=timed(lambda:recording(entry['source'],subject,12,'verification',Path(temp),entry['configuration']),repeats=3)
  sources[subject]=dict(cold_source_preprocessing_seconds=cold,warm_cache_load=warm,cold_repeats=1)
  x,y,g,actual,_=data;c=actual['configuration']
  for r in [r for r in rows if r['subject']==subject]:
   d=Decoder(root,subject,r['representation'],r['method'],r['seed'],c);p=d.predict(x,c['channels'],c['sampling_hz']);delta=float(np.max(np.abs(p-np.array(r['probabilities']))))
   if delta>1e-6 or not np.array_equal(p.argmax(1),np.argmax(r['probabilities'],axis=1)) or not np.array_equal(y,r['labels']):raise AssertionError('Saved prediction mismatch')
   f=d.features(x,c['channels'],160);inference=timed(lambda:d.predict(x,c['channels'],160));core=timed(lambda:d.predict_features(f))
   # A new preprocessing cache for every full-pipeline repetition; OS file cache
   # is uncontrolled. This measures actual source-to-probability execution.
   full=[]
   for repeat in range(3):
    with tempfile.TemporaryDirectory() as temp:
     started=time.perf_counter();fp,_,_,_,hit=d.predict_recording(entry['source'],subject,12,Path(temp));full.append(time.perf_counter()-started)
     assert not hit
     np.testing.assert_allclose(p,fp,atol=1e-6,rtol=0)
   full_timing=dict(median_seconds=float(np.median(full)),p95_seconds=float(np.quantile(full,.95)),repeats=3,processed_cache='empty on every repeat',filesystem_cache='uncontrolled')
   # Audit normalization against train-run recordings, with no fitting or updates.
   train=next(m for m in manifest if m['subject']==subject and m['run']==4)
   tx,_,_,_,_=recording(train['source'],subject,4,'verification',root/'verification-cache',train['configuration']);tf=d.features(tx,c['channels'],160)
   if d.method=='neural':
    axis=(0,2) if d.rep=='time' else 0;np.testing.assert_allclose(d.mean,tf.mean(axis=axis,keepdims=True),atol=1e-6,rtol=1e-5);np.testing.assert_allclose(d.scale,tf.std(axis=axis,keepdims=True).clip(1e-8),atol=1e-6,rtol=1e-5)
   else:np.testing.assert_allclose(d.pipeline['scaler'].mean_,tf.mean(0),rtol=1e-5,atol=1e-6)
   output.append(dict(subject=subject,representation=r['representation'],method=r['method'],seed=r['seed'],max_probability_error=delta,balanced_accuracy=float(balanced_accuracy_score(y,p.argmax(1))),classification=classification_report(y,p.argmax(1),output_dict=True,zero_division=0),confusion=confusion_matrix(y,p.argmax(1)).tolist(),predicted_counts=np.bincount(p.argmax(1),minlength=2).tolist(),trials=len(y),coverage=float(g.mean()),features_and_model=inference,model_with_normalization=core,source_through_prediction=full_timing,normalization_training_audit='passed'))
  print('Verified subject',subject,flush=True)
 result=dict(tolerance=1e-6,threads=2,warmup=3,repeats=10,hardware=dict(platform=platform.platform(),processor=platform.processor(),torch=torch.__version__,device='CPU'),timing_note='Batch wall times; source cold cache once per participant, warm cache 3 repeats; model excludes feature extraction but includes normalization/tensor conversion. Full source-through-prediction measured three times with empty processed cache; OS file cache uncontrolled. Offline batch duration, not live latency.',sources=sources,results=output)
 (root/'verification-complete.json').write_text(json.dumps(result,indent=2));print('Verified',len(output),'max error',max(r['max_probability_error'] for r in output))
if __name__=='__main__':main()
