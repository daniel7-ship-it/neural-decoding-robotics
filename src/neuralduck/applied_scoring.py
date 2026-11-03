"""Duration-weighted scoring of every applied physics interval."""
import numpy as np

def score_frames(frames, supervision, recording_end, deadline=.5, observation_end=None):
    valid=[f for f in frames if not f['finished']]
    result=dict(movement_episodes=0,movement_detected=0,deadline_hits=0,false_rest_events=0,transition_carryover_events=0,new_rest_activation_events=0,rest_seconds=0.,rest_occupancy=0.,timely_sustained_stops=0,movement=[],initial_active_rest=0,missed_episodes=0,median_onset_seconds=None,p95_onset_seconds=None,false_rest_events_per_minute=None,movement_interruptions=0,wrong_direction_seconds=0.,wrong_direction_events=0)
    if not valid:return result
    starts=np.asarray([f['recording_time'] for f in valid])
    terminal=next((f['recording_time'] for f in frames if f['finished']),None)
    endpoint=observation_end if observation_end is not None else terminal
    if endpoint is None: endpoint=starts[-1] # Last interval is unobserved without an endpoint.
    if not np.isfinite(starts).all() or not np.isfinite(endpoint) or endpoint<starts[-1]:raise ValueError('Invalid observation times')
    ends=np.minimum(np.r_[starts[1:],endpoint],recording_end)
    if np.any(np.diff(starts)<=0):raise ValueError('Frame times must increase')
    commands=np.array([1 if f['applied'][2]>0 else 2 if f['applied'][2]<0 else 0 for f in valid])
    blocks=np.unique(supervision['block']);onsets=np.array([supervision['onset'][supervision['block']==b][0] for b in blocks]);labels=np.array([supervision['y'][supervision['block']==b][0] for b in blocks]);boundaries=np.r_[onsets,recording_end]
    pieces=[]
    for a,z,c in zip(starts,ends,commands):
        cuts=np.r_[a,boundaries[(boundaries>a)&(boundaries<z)],z]
        for left,right in zip(cuts[:-1],cuts[1:]):
            b=int(np.searchsorted(onsets,left,side='right')-1)
            if b>=0 and right>left:pieces.append((left,right,c,b))
    result['initial_active_rest']=0
    rest_active=0.;previous=None
    for a,z,c,b in pieces:
        if labels[b]==0:
            result['rest_seconds']+=z-a;rest_active+=(z-a)*(c!=0)
            if c and (previous is None or previous[2]!=c or previous[3]!=b):
                if previous is None:
                    result['initial_active_rest']+=1
                    previous=(a,z,c,b)
                    continue
                result['false_rest_events']+=1
                key='transition_carryover_events' if previous is not None and labels[previous[3]]!=0 else 'new_rest_activation_events'
                result[key]+=1
        if labels[b] and c and c!=labels[b]:
            result['wrong_direction_seconds']+=z-a
            result['wrong_direction_events']+=int(previous is None or previous[3]!=b or previous[2]!=c)
        previous=(a,z,c,b)
    result['rest_occupancy']=rest_active/result['rest_seconds'] if result['rest_seconds'] else 0.
    for b,label in enumerate(labels):
        rows=[p for p in pieces if p[3]==b]
        if not rows or starts[0]>onsets[b]+1e-9 or ends[-1]<boundaries[b+1]-1e-9:continue
        if label:
            hits=[p[0] for p in rows if p[2]==label];delay=hits[0]-onsets[b] if hits else None
            occupancy=sum(z-a for a,z,c,_ in rows if c==label)/(boundaries[b+1]-onsets[b])
            result['movement_episodes']+=1;result['movement_detected']+=bool(hits);result['deadline_hits']+=delay is not None and delay<=deadline+1e-9
            result['movement'].append(dict(block=int(blocks[b]),delay_seconds=delay,correct_command_occupancy=occupancy,interruptions=int(sum(rows[i-1][2]==label and rows[i][2]!=label for i in range(1,len(rows))))))
        elif b and labels[b-1]:
            zero=next((i for i,p in enumerate(rows) if p[2]==0),None)
            result['timely_sustained_stops']+=zero is not None and rows[zero][0]-onsets[b]<=deadline+1e-9 and all(p[2]==0 for p in rows[zero:])
    delays=[m['delay_seconds'] for m in result['movement'] if m['delay_seconds'] is not None]
    result.update(missed_episodes=result['movement_episodes']-result['movement_detected'],median_onset_seconds=float(np.median(delays)) if delays else None,p95_onset_seconds=float(np.percentile(delays,95)) if delays else None,false_rest_events_per_minute=60*result['false_rest_events']/result['rest_seconds'] if result['rest_seconds'] else None,movement_interruptions=sum(m['interruptions'] for m in result['movement']))
    return {k:(v.item() if isinstance(v,np.generic) else v) for k,v in result.items()}
