import json,numpy as np
from pathlib import Path
from neuralduck.supervised import DATA
from neuralduck.rest_gate_review import measure
path=max((Path('artifacts/neural-replay-runs')).glob('*.json'),key=lambda p:p.stat().st_mtime)
t=json.loads(path.read_text());assert t['reason']=='complete'
labels=np.load(DATA/'supervision-19.npz');indices=np.array([d['index'] for d in t['decisions']]);frames=t['frames'];issued=[d['issued_command'] for d in t['decisions']]
phys=np.array([f['recording_time'] for f in frames]);applied=np.array([f['applied'][2] for f in frames]);commands=[]
for d in t['decisions']:
 i=np.flatnonzero(np.isclose(phys,d['issued_at']+t['start_seconds'],atol=1e-8))[0]
 commands.append(1 if applied[i]>0 else 2 if applied[i]<0 else 0)
row={k:labels[k][indices] for k in labels.files};scored=measure(row,np.array(commands))
for key in ('deadline_hits','movement_episodes','movement_detected','false_rest_events','timely_sustained_stops'):
 assert t['metrics'][key]==scored[key],key
assert frames[-1]['finished'] and frames[-1]['applied']==[0.,0.,0.]
print('Complete UI trace: applied-command metrics independently reproduced; terminal zero frame verified:',path.name)
