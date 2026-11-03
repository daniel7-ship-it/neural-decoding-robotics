"""Rescore historical traces without overwriting their original metrics."""
import json
import numpy as np
from .supervised import ROOT,DATA
from .applied_scoring import score_frames

def main():
 out=ROOT/'artifacts/applied-rescore-v2';out.mkdir(exist_ok=True);summary=[]
 for source in (ROOT/'artifacts/neural-replay-runs').glob('*.json'):
  t=json.loads(source.read_text());path=DATA/f"supervision-{t['session']}.npz"
  if not path.exists():continue
  with np.load(path) as d:row={k:d[k] for k in d.files}
  end=float(row['time'][-1]+.1)
  metrics=score_frames(t['frames'],row,end)
  payload=dict(source=str(source),old=t.get('metrics'),corrected=metrics,limitation='Nonterminal last frame excluded because historical trace lacks an explicit interval endpoint.')
  (out/source.name).write_text(json.dumps(payload,indent=2));summary.append(dict(trace=source.name,old_rest=t.get('metrics',{}).get('rest_seconds') if t.get('metrics') else None,new_rest=metrics['rest_seconds']))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print('Rescored',len(summary),'traces')
if __name__=='__main__':main()
