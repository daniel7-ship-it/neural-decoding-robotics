"""Corrected command metrics and frozen disturbance evaluation."""
import json
import numpy as np
import torch
from .supervised import ROOT, MODELS, DATA, NeuralDecoder, causal_features, generate_session, gated_predictions, save_json
from .rest_gate import decisions
from .robustness import RecordingDisturbance

OUT=ROOT/'artifacts/rest-gate-corrected'

def measure(row, command, deadline=.5):
    rest=row['y']==0
    previous=np.r_[0,command[:-1]]
    events=rest&(command!=0)&((command!=previous)|~np.r_[False,rest[:-1]])
    movement=[]; stops=[]
    for block in np.unique(row['block']):
        indices=np.flatnonzero(row['block']==block)
        label=int(row['y'][indices[0]])
        if label:
            correct=indices[command[indices]==label]
            delay=float(row['time'][correct[0]]-row['onset'][correct[0]]) if len(correct) else None
            correct_fraction=float((command[indices]==label).mean())
            movement.append(dict(block=int(block),delay_seconds=delay,within_deadline=delay is not None and delay<=deadline,correct_command_occupancy=correct_fraction,at_least_80_percent_correct_occupancy=correct_fraction>=.8,interruptions=int(np.sum((command[indices][1:]!=label)&(command[indices][:-1]==label))),wrong_direction_samples=int(np.sum((command[indices]!=0)&(command[indices]!=label)))))
        elif block and row['y'][indices[0]-1]!=0:
            zero=indices[command[indices]==0]
            delay=float(row['time'][zero[0]]-row['onset'][zero[0]]) if len(zero) else None
            sustained=bool(len(zero) and np.all(command[zero[0]:indices[-1]+1]==0))
            stops.append(dict(block=int(block),delay_seconds=delay,within_deadline=delay is not None and delay<=deadline,sustained_zero=sustained,timely_sustained_stop=delay is not None and delay<=deadline and sustained))
    delays=[m['delay_seconds'] for m in movement if m['delay_seconds'] is not None]
    offsets=[s['delay_seconds'] for s in stops if s['delay_seconds'] is not None]
    long=row['block']==0
    return dict(transition_carryover_events=int(np.sum(rest & ~np.r_[True,rest[:-1]] & (command!=0))),new_rest_activation_events=int(np.sum(rest & np.r_[True,rest[:-1]] & (command!=0) & (command!=previous))),movement=movement,stops=stops,movement_episodes=len(movement),movement_detected=sum(m['delay_seconds'] is not None for m in movement),deadline_hits=sum(m['within_deadline'] for m in movement),occupancy_80_hits=sum(m['at_least_80_percent_correct_occupancy'] for m in movement),movement_interruptions=sum(m['interruptions'] for m in movement),wrong_direction_samples=sum(m['wrong_direction_samples'] for m in movement),median_onset_seconds=float(np.median(delays)) if delays else None,p95_onset_seconds=float(np.percentile(delays,95)) if delays else None,stop_transitions=len(stops),stop_deadline_hits=sum(s['within_deadline'] for s in stops),timely_sustained_stops=sum(s['timely_sustained_stop'] for s in stops),median_stop_seconds=float(np.median(offsets)) if offsets else None,false_rest_events=int(events.sum()),rest_occupancy=float((command[rest]!=0).mean()) if rest.any() else 0.0,rest_seconds=float(rest.sum()*.1),long_rest_false_events=int(events[long].sum()),long_rest_seconds=float(long.sum()*.1))

def run():
    torch.set_num_threads(1)
    p=json.loads((ROOT/'docs/rest-gate-disturbance-protocol.json').read_text(encoding='utf-8-sig'))
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for session in (23,24,25,26):
        supervision=np.load(DATA/f'supervision-{session}.npz')
        row={k:supervision[k] for k in supervision.files}
        for seed in p['models']:
            for strict in (False,True):
                saved=np.load(ROOT/f'artifacts/rest-gate/{session}-{seed}-{strict}.npz')
                rows.append(dict(stage='rescore',session=session,seed=seed,strict=strict,condition='control',**measure(row,saved['command'])))
    save_json(OUT/'rescored-saved.json',rows)
    for session in p['sessions']:
        for condition in p['conditions']:
            disturbance=RecordingDisturbance(42,session,condition)
            row=generate_session(session,42,disturbance=disturbance,save=False)
            for seed in p['models']:
                probability=NeuralDecoder.load(MODELS/f'mlp-{seed}.pt').predict_proba(row['x'])
                for strict in (False,True):
                    command=gated_predictions(decisions(probability,strict),row['time'])
                    rows.append(dict(stage='fresh',session=session,seed=seed,strict=strict,condition=condition,**measure(row,command)))
                    np.savez_compressed(OUT/f'{session}-{condition}-{seed}-{strict}.npz',command=command,probability=probability,time=row['time'],label=row['y'])
            print(session,condition,flush=True)
    save_json(OUT/'evaluation.json',dict(protocol=p,rows=rows))
    for condition in p['conditions']:
        for strict in (False,True):
            chosen=[r for r in rows if r['stage']=='fresh' and r['condition']==condition and r['strict']==strict]
            print(condition,strict,'events',sum(r['false_rest_events'] for r in chosen),'deadline',sum(r['deadline_hits'] for r in chosen),'/',sum(r['movement_episodes'] for r in chosen),'sustained',sum(r['occupancy_80_hits'] for r in chosen),'stop',sum(r['timely_sustained_stops'] for r in chosen),'/',sum(r['stop_transitions'] for r in chosen),flush=True)

if __name__=='__main__':run()

