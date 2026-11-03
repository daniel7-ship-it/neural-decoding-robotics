"""Regenerate calibration figures and summaries from saved predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('experiment');a=p.parse_args();out=Path(a.experiment);rows=json.loads((out/'results.json').read_text());subjects=sorted({r['subject'] for r in rows});methods=['frozen','fine_tuned','subject_specific','csp_lda'];summary={}
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for method in methods:
  vals=[]
  for budget in (4,8,12):
   vals.append(np.mean([r['balanced_accuracy'] for r in rows if r['method']==method and r['budget']==budget]))
  axes[0].plot([4,8,12],vals,'o-',label=method)
 gains=[]
 for subject in subjects:
  score={m:np.mean([r['balanced_accuracy'] for r in rows if r['subject']==subject and r['budget']==12 and r['method']==m]) for m in methods};summary[str(subject)]=score;gains.append(score['fine_tuned']-score['frozen']);axes[1].plot([0,1],[score['frozen'],score['fine_tuned']],'o-',label=str(subject))
 axes[0].set(xlabel='Labeled calibration trials',ylabel='Balanced accuracy',ylim=(0,1));axes[0].legend();axes[1].set(xticks=[0,1],xticklabels=['Frozen','Calibrated'],ylabel='Balanced accuracy',ylim=(0,1));axes[1].legend(title='Participant');fig.tight_layout();fig.savefig(out/'calibration.png',dpi=150)
 rng=np.random.default_rng(70);ci=np.percentile([np.mean(rng.choice(gains,len(gains),replace=True)) for _ in range(5000)],[2.5,97.5]);result=dict(participants=summary,mean_gain=float(np.mean(gains)),gain_ci95=ci.tolist(),helped=int(np.sum(np.array(gains)>0)),harmed=int(np.sum(np.array(gains)<0)));(out/'summary.json').write_text(json.dumps(result,indent=2));(out/'report.html').write_text('<html><body><h1>Personal EEG calibration</h1><p>Binary prerecorded imagery; no rest/live control claim.</p><img src="calibration.png"><pre>'+json.dumps(result,indent=2)+'</pre></body></html>');print(json.dumps(result))
if __name__=='__main__':main()
