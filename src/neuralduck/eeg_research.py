"""Shared provenance, normalization and checkpoint contracts for EEG research."""
import hashlib,json,uuid
from pathlib import Path
import numpy as np
import torch

def fingerprint(files,configuration):
 h=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode())
 for p in files:
  p=Path(p);h.update(str(p.resolve()).encode());h.update(p.read_bytes())
 return h.hexdigest()

def new_experiment(root):
 path=Path(root)/uuid.uuid4().hex;path.mkdir(parents=True,exist_ok=False);return path

def fit_normalization(training_electrodes):
 x=np.asarray(training_electrodes,dtype=np.float32)
 if x.ndim!=3 or not np.isfinite(x).all():raise ValueError('Expected finite trial/channel/sample electrodes')
 return x.mean(axis=(0,2),keepdims=True),x.std(axis=(0,2),keepdims=True).clip(1e-8)

def probabilities(network,electrodes,mean,scale):
 x=np.asarray(electrodes,dtype=np.float32)
 if x.ndim!=3 or x.shape[1]!=mean.shape[1] or not np.isfinite(x).all():raise ValueError('Invalid electrode input')
 network.eval()
 with torch.no_grad():return network(torch.tensor((x-mean)/scale)).softmax(1).numpy()

def save_checkpoint(path,network,mean,scale,metadata):
 torch.save(dict(state_dict=network.state_dict(),mean=torch.tensor(mean),scale=torch.tensor(scale),metadata=metadata),path)

def load_checkpoint(path,factory):
 d=torch.load(path,map_location='cpu',weights_only=True);network=factory();network.load_state_dict(d['state_dict']);network.eval();return network,d['mean'].numpy(),d['scale'].numpy(),d['metadata']
