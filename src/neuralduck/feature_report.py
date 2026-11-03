"""Regenerate feature and calibration reporting from saved outputs."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def main():
 p=argparse.ArgumentParser();p.add_argument('experiment');a=p.parse_args();out=Path(a.experiment);rows=json.loads((out/'results.json').read_text());groups=sorted({r['representation']+' '+r['method'] for r in rows});summary={}
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for i,group in enumerate(groups):
  selected=[r for r in rows if r['representation']+' '+r['method']==group];subjects=sorted({r['subject'] for r in selected});scores=[np.mean([r['accuracy'] for r in selected if r['subject']==s]) for s in subjects];summary[group]=dict(mean=float(np.mean(scores)),subjects=dict(zip(map(str,subjects),map(float,scores))));axes[0].scatter([i]*len(scores),scores);axes[0].plot(i,np.mean(scores),'kx')
 axes[0].set(xticks=range(len(groups)),xticklabels=groups,ylim=(0,1),ylabel='Balanced accuracy');axes[0].tick_params(axis='x',rotation=35)
 example=next(r for r in rows if r.get('curves'));curve=np.array(example['curves']);axes[1].plot(curve[:,0],label='Training');axes[1].plot(curve[:,1],label='Validation');axes[1].set(xlabel='Update',ylabel='Cross entropy');axes[1].legend();fig.tight_layout();fig.savefig(out/'features.png',dpi=150)
 teachers=json.loads((out/'teacher.json').read_text());qualified=[rep for rep in ('bandpower','csp') if all(t['qualified'] for t in teachers if t['representation']==rep)];summary['qualified_teachers']=qualified;(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'report.html').write_text('<html><body><h1>Exploratory EEG feature comparison</h1><img src="features.png"><pre>'+json.dumps(summary,indent=2)+'</pre></body></html>')
 calibration=Path('artifacts/calibration-development/6677c8dddba2493f8149ee31e60b57ec');old=json.loads((calibration/'results.json').read_text());details=[]
 for r in old:
  pred=np.argmax(r['probabilities'],axis=1);details.append(dict(subject=r['subject'],seed=r['seed'],budget=r['budget'],method=r['method'],coverage=float(np.mean(r['quality'])),classification=classification_report(r['labels'],pred,output_dict=True,zero_division=0)))
 (calibration/'classification-details.json').write_text(json.dumps(details,indent=2));fig,axes=plt.subplots(1,2,figsize=(11,4));subjects=sorted({r['subject'] for r in old});axes[0].bar(list(map(str,subjects)),[np.mean(next(r for r in old if r['subject']==s)['quality']) for s in subjects]);axes[0].set(ylabel='Quality-pass fraction',xlabel='Participant',ylim=(0,1));r=next(r for r in old if r['method']=='fine_tuned' and r['budget']==12);axes[1].plot(r['labels'],'o-',label='Known label');axes[1].plot(np.argmax(r['probabilities'],axis=1),'x--',label='Prediction');axes[1].legend();axes[1].set(xlabel='Held-out trial',ylabel='Left / right class');fig.tight_layout();fig.savefig(calibration/'coverage-sequence.png',dpi=150);print(json.dumps(summary))
if __name__=='__main__':main()
