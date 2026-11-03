"""Predeclared frozen-network rest-gate comparison; no fitting."""
import json
import numpy as np
import torch
from .supervised import ROOT, MODELS, NeuralDecoder, generate_session, gated_predictions, save_json

def decisions(probabilities, strict=False):
    pred=probabilities.argmax(1)
    if strict:
        pred=np.where((probabilities.max(1)>=.9)&(probabilities[:,0]<=.05),pred,0)
    return pred

def score(row,command):
    rest=row['y']==0
    previous=np.r_[0,command[:-1]]
    events=rest&(command!=0)&((previous!=command)|~np.r_[False,rest[:-1]])
    long=row['block']==0
    delay=[]; missed=0; offsets=[]
    for b in np.unique(row['block']):
        indices=np.flatnonzero(row['block']==b); label=row['y'][indices[0]]
        if label:
            hits=indices[command[indices]==label]
            if len(hits): delay.append(float(row['time'][hits[0]]-row['onset'][hits[0]]))
            else: missed+=1
        elif b:
            # Time to first zero command on entry to each scheduled rest block.
            hits=indices[command[indices]==0]
            offsets.append(float(row['time'][hits[0]]-row['onset'][hits[0]]) if len(hits) else None)
    return dict(rest_seconds=float(rest.sum()*.1),rest_occupancy=float((command[rest]!=0).mean()),false_rest_events=int(events.sum()),long_rest_false_events=int(events[long].sum()),long_rest_seconds=float(long.sum()*.1),movement_detected=len(delay),movement_missed=missed,onset_delays=delay,rest_offset_delays=offsets)

def run():
    torch.set_num_threads(1)
    p=json.loads((ROOT/'docs/rest-gate-protocol.json').read_text(encoding='utf-8-sig'))
    out=ROOT/'artifacts/rest-gate'; out.mkdir(parents=True,exist_ok=True)
    results=[]
    for session in p['sessions']:
        row=generate_session(session)
        for seed in p['models']:
            probabilities=NeuralDecoder.load(MODELS/f'mlp-{seed}.pt').predict_proba(row['x'])
            for strict in (False,True):
                command=gated_predictions(decisions(probabilities,strict),row['time'])
                results.append(dict(session=session,seed=seed,strict=strict,**score(row,command)))
                np.savez_compressed(out/f'{session}-{seed}-{strict}.npz',command=command,time=row['time'],label=row['y'],probabilities=probabilities)
        print('Completed session',session,flush=True)
    summaries=[]
    for seed in p['models']:
        for strict in (False,True):
            rows=[r for r in results if r['seed']==seed and r['strict']==strict]
            delays=sum([r['onset_delays'] for r in rows],[])
            offsets=sum([r['rest_offset_delays'] for r in rows],[])
            detected=sum(r['movement_detected'] for r in rows); missed=sum(r['movement_missed'] for r in rows)
            s=dict(seed=seed,strict=strict,rest_occupancy=sum(r['rest_occupancy']*r['rest_seconds'] for r in rows)/sum(r['rest_seconds'] for r in rows),false_rest_events=sum(r['false_rest_events'] for r in rows),movement_detection=detected/(detected+missed),movement_missed=missed,median_onset_seconds=float(np.median(delays)) if delays else None,median_rest_offset_seconds=float(np.median([v for v in offsets if v is not None])),rest_blocks_without_stop=sum(v is None for v in offsets),long_rest_false_events_per_minute=sum(r['long_rest_false_events'] for r in rows)/(sum(r['long_rest_seconds'] for r in rows)/60))
            s['passes']=s['movement_detection']>=.9 and s['median_onset_seconds'] is not None and s['median_onset_seconds']<=.5 and s['rest_occupancy']<=.01 and s['long_rest_false_events_per_minute']<=.5
            summaries.append(s)
    save_json(out/'evaluation.json',dict(protocol=p,summaries=summaries,sessions=results))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': run()
