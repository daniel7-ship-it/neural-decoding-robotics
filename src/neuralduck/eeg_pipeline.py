"""One executable EEG preprocessing and provenance path."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
import mne
from scipy.signal import butter,sosfilt
from mne.datasets import eegbci
from .eeg_config import CONFIG
from .eeg_research import fingerprint

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def recording(path,subject,run,split,cache_dir,configuration=None):
 c=copy.deepcopy(CONFIG if configuration is None else configuration)
 if c['reference']!='average_all_eeg' or c['units']!='volts' or c['filter']['kind']!='butterworth_sos' or not c['filter']['causal'] or c['quality']['policy']!='reject_whole_trial_if_any_channel_fails':raise ValueError('Unsupported preprocessing configuration')
 if set(c['annotations'])!={'T1','T2'} or set(c['annotations'].values())!={0,1}:raise ValueError('Expected binary T1/T2 annotation mapping')
 if c['runs']!=[4,8,12]:raise ValueError('Only imagery runs 4/8/12 supported')
 raw=mne.io.read_raw_edf(path,preload=True,verbose=False);eegbci.standardize(raw);raw.pick('eeg')
 channels=c.get('channels',raw.ch_names)
 if len(set(channels))!=len(channels) or set(channels)!=set(raw.ch_names):raise ValueError('Expected complete unique channel ordering')
 raw.reorder_channels(channels);sf=float(raw.info['sfreq']);c.update(channels=channels,sampling_hz=sf,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 f=c['filter'];w=c['window'];q=c['quality']
 if not 0<f['low_hz']<f['high_hz']<sf/2 or w['duration_seconds']<=0 or w['offset_seconds']<0 or q['peak_volts']<=0 or q['minimum_ptp_volts']<0:raise ValueError('Invalid preprocessing parameters')
 identity=fingerprint([path],c);cache=Path(cache_dir)/f'{identity}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
 if cache.exists():
  with np.load(cache,allow_pickle=False) as d:x,y,good,trials=d['x'],d['y'],d['good'],json.loads(str(d['trials']))
  hit=True
 else:
  data=raw.get_data();data-=data.mean(axis=0,keepdims=True);filtered=sosfilt(butter(f['order'],[f['low_hz'],f['high_hz']],btype='bandpass',fs=sf,output='sos'),data,axis=-1)
  events,ids=mne.events_from_annotations(raw,verbose=False);mapping={ids[k]:v for k,v in c['annotations'].items() if k in ids};xs=[];ys=[];flags=[];trials=[]
  for i,(sample,_,code) in enumerate(events):
   if int(code) not in mapping:continue
   a=int(sample-raw.first_samp+w['offset_seconds']*sf);b=a+int(w['duration_seconds']*sf)
   if a<0 or b>data.shape[1]:continue
   epoch=data[:,a:b];xs.append(filtered[:,a:b]);ys.append(mapping[int(code)]);flags.append(bool(np.isfinite(epoch).all() and np.max(np.abs(epoch))<=q['peak_volts'] and np.all(np.ptp(epoch,axis=1)>=q['minimum_ptp_volts'])))
   trials.append(dict(trial_id=i,label=int(mapping[int(code)]),start_sample=a,end_sample=b))
  x=np.asarray(xs,dtype=np.float32);y=np.asarray(ys,dtype=np.int64);good=np.asarray(flags);np.savez_compressed(cache,x=x,y=y,good=good,trials=json.dumps(trials));hit=False
 manifest=dict(source=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),subject=subject,run=run,split=split,configuration=c,cache_fingerprint=identity,trials=trials)
 return x,y,good,manifest,hit

def manifest_document(records):
 document=dict(schema=1,recordings=records);return document,digest(document)
