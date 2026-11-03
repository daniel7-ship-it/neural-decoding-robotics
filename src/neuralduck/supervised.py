"""Small supervised electrode MLP; offline continuous replay, no robot/RL changes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .closed_loop import ContinuousCircuit, CommandGate, ridge

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/neural-network-protocol.json'
OUT = ROOT / 'artifacts/neural-network'
MODELS = ROOT / 'models/neural-network'
DATA = ROOT / 'data/neural-network'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def protocol():
    p = json.loads(PROTOCOL.read_text(encoding='utf-8-sig'))
    splits = [set(v) for v in p['splits'].values()]
    if any(a & b for i, a in enumerate(splits) for b in splits[i+1:]):
        raise ValueError('Session splits overlap')
    return p


def causal_features(recordings):
    """Window ends at sample i; never consumes a future sample."""
    ends = np.arange(19, len(recordings), 10)
    x = np.array([np.r_[recordings[i-19:i+1].mean(0),
                            recordings[i-19:i+1].std(0)] for i in ends], dtype=np.float32)
    return x, ends


def generate_session(session, anatomy=42, disturbance=None, save=True):
    rng = np.random.default_rng(900000 + session)
    labels = np.repeat(np.arange(3), 20)
    rng.shuffle(labels)
    schedule = [(0, 12000, 0.)]
    for label in labels:
        strength = float(rng.uniform(.3, 1.)) if label else 0.
        schedule.append((int(label), int(rng.integers(150,301)), strength))
    stream = ContinuousCircuit(anatomy, session, disturbance=disturbance)
    observations, truth_labels, drives, blocks, starts = [], [], [], [], []
    tick = 0
    for block, (label, count, strength) in enumerate(schedule):
        start = tick * .01
        drive = strength * (1 if label == 1 else -1 if label == 2 else 0)
        for _ in range(count):
            observations.append(stream.sample(drive, tick*.01))
            truth_labels.append(label)
            drives.append(drive)
            blocks.append(block)
            starts.append(start)
            tick += 1
    recordings = np.asarray(observations, dtype=np.float32)
    x, ends = causal_features(recordings)
    # Ground truth stays in the experimental harness, not the inference object.
    row = dict(x=x, y=np.array(truth_labels)[ends], drive=np.array(drives)[ends],
               block=np.array(blocks)[ends], onset=np.array(starts)[ends],
               time=(ends+1)*.01, session=np.full(len(ends), session))
    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / f'electrodes-{session}.npz', recordings=recordings)
        np.savez_compressed(DATA / f'supervision-{session}.npz', **{k:v for k,v in row.items() if k != 'x'})
    return row


def join(rows):
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def network():
    return nn.Sequential(nn.Linear(48,32), nn.ReLU(), nn.Linear(32,3))


class NeuralDecoder:
    """Only electrode features enter inference. Labels and simulator stay outside."""
    def __init__(self, model, mean, scale):
        self.model = model.eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    def predict_proba(self, electrode_features):
        x = np.asarray(electrode_features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 48 or not np.isfinite(x).all():
            raise ValueError('Expected finite (windows,48) electrode features')
        with torch.no_grad():
            return self.model(torch.from_numpy((x-self.mean)/self.scale)).softmax(1).numpy()

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = network()
        model.load_state_dict(checkpoint['state_dict'])
        return cls(model, checkpoint['mean'].numpy(), checkpoint['scale'].numpy())


def train(train_data, validation, seed, p, shuffle=False):
    torch.manual_seed(seed)
    model = network()
    initial = torch.cat([w.detach().flatten().clone() for w in model.parameters()])
    scaler = StandardScaler().fit(train_data['x'])
    x = torch.tensor(scaler.transform(train_data['x']), dtype=torch.float32)
    vx = torch.tensor(scaler.transform(validation['x']), dtype=torch.float32)
    labels = train_data['y'].copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        for session in np.unique(train_data['session']):
            mask = train_data['session'] == session
            blocks = np.unique(train_data['block'][mask])
            block_labels = [labels[np.flatnonzero(mask & (train_data['block']==b))[0]] for b in blocks]
            shuffled = rng.permutation(block_labels)
            for b, label in zip(blocks, shuffled):
                labels[mask & (train_data['block']==b)] = label
    y = torch.tensor(labels, dtype=torch.long)
    vy = torch.tensor(validation['y'], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()
    history, best, best_state, stale = [], float('inf'), None, 0
    for epoch in range(p['max_epochs']):
        model.train()
        order = torch.randperm(len(x))
        for batch in order.split(p['batch_size']):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            tl = float(loss_fn(model(x),y))
            vl = float(loss_fn(model(vx),vy))
            accuracy = float((model(vx).argmax(1)==vy).float().mean())
        history.append(dict(epoch=epoch+1, training_loss=tl, validation_loss=vl, validation_accuracy=accuracy))
        if vl < best:
            best, best_state, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= p['patience']:
            break
    model.load_state_dict(best_state)
    delta = float(torch.linalg.vector_norm(torch.cat([w.detach().flatten() for w in model.parameters()])-initial))
    decoder = NeuralDecoder(model, scaler.mean_, scaler.scale_)
    return decoder, history, delta


def gated_predictions(pred, times):
    gate = CommandGate()
    result = []
    for label, timestamp in zip(pred,times):
        drive = 1. if label == 1 else -1. if label == 2 else 0.
        command = gate.update(drive, float(timestamp))
        result.append(1 if command.yaw_rate > 0 else 2 if command.yaw_rate < 0 else 0)
    return np.asarray(result)


def metrics(row, pred):
    y = row['y']
    command = gated_predictions(pred, row['time'])
    cm = confusion_matrix(y,pred,labels=[0,1,2])
    recalls = np.diag(cm)/np.maximum(1,cm.sum(1))
    rest = y==0
    long_rest = row['block']==0
    # Count carry-in at rest entry and nonzero direction changes as false events.
    previous = np.r_[0,command[:-1]]
    prior_rest = np.r_[False, rest[:-1]]
    events = rest & (command!=0) & ((command!=previous) | ~prior_rest)
    delays, missed = [], 0
    for block in np.unique(row['block']):
        mask = row['block']==block
        label = y[mask][0]
        if label == 0:
            continue
        hits = np.flatnonzero(mask & (command==label))
        if len(hits):
            delays.append(float(row['time'][hits[0]]-row['onset'][hits[0]]))
        else:
            missed += 1
    return dict(confusion_matrix=cm.tolist(), balanced_accuracy=float(recalls.mean()),
                rest_recall=float(recalls[0]), movement_recall=float(recalls[1:].mean()),
                missed_movement_episodes=missed, detected_movement_episodes=len(delays),
                movement_episode_detection=len(delays)/max(1,len(delays)+missed),
                median_onset_delay_seconds=float(np.median(delays)) if delays else None,
                onset_delays_seconds=delays,
                rest_seconds=float(rest.sum()*.1), false_rest_events=int(events.sum()),
                rest_occupancy=float((command[rest]!=0).mean()),
                long_rest_seconds=float(long_rest.sum()*.1),
                long_rest_false_events=int(events[long_rest].sum()),
                long_rest_false_events_per_minute=float(events[long_rest].sum()/(long_rest.sum()*.1/60)),
                long_rest_occupancy=float((command[long_rest]!=0).mean()))


def summarize(rows):
    cm = np.sum([r['confusion_matrix'] for r in rows],axis=0)
    recalls = np.diag(cm)/cm.sum(1)
    seconds = sum(r['long_rest_seconds'] for r in rows)
    events = sum(r['long_rest_false_events'] for r in rows)
    detected = sum(r['detected_movement_episodes'] for r in rows)
    missed = sum(r['missed_movement_episodes'] for r in rows)
    delays = sum([r['onset_delays_seconds'] for r in rows],[])
    return dict(confusion_matrix=cm.tolist(), balanced_accuracy=float(recalls.mean()), rest_recall=float(recalls[0]),
                movement_recall=float(recalls[1:].mean()), movement_episode_detection=detected/max(1,detected+missed),
                missed_movement_episodes=missed, median_onset_delay_seconds=float(np.median(delays)) if delays else None,
                long_rest_seconds=seconds,long_rest_false_events=events,long_rest_false_events_per_minute=events/(seconds/60),
                long_rest_occupancy=sum(r['long_rest_occupancy']*r['long_rest_seconds'] for r in rows)/seconds,
                all_rest_false_events=sum(r['false_rest_events'] for r in rows),
                all_rest_occupancy=sum(r['rest_occupancy']*r['rest_seconds'] for r in rows)/sum(r['rest_seconds'] for r in rows))


def passes(summary, criteria):
    for key, limit in criteria.items():
        metric = key.rsplit('_', 1)[0].replace('median_detected_onset', 'median_onset')
        value = summary[metric]
        if value is None or (value < limit if key.endswith('_min') else value > limit):
            return False
    return True


def run():
    p = protocol()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'protocol.json',p)
    splits = {}
    # Final test generation is deferred until training and model saving finish.
    for name in ('train','validation'):
        splits[name] = [generate_session(s,p['anatomy_seed']) for s in p['splits'][name]]
        print('Generated',name,flush=True)
    tr, va = join(splits['train']), join(splits['validation'])
    lda = make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')).fit(tr['x'],tr['y'])
    direct = ridge(1.).fit(tr['x'],tr['drive'])
    joblib.dump(lda,MODELS/'lda.joblib')
    joblib.dump(direct,MODELS/'ridge.joblib')
    models, training = {}, {}
    for seed in p['initialization_seeds']:
        name = f'mlp-{seed}'
        decoder, history, delta = train(tr,va,seed,p)
        checkpoint = dict(state_dict=decoder.model.state_dict(), mean=torch.tensor(decoder.mean),
                          scale=torch.tensor(decoder.scale), classes=['rest','left','right'],
                          architecture=[48,32,3], seed=seed, protocol=p)
        torch.save(checkpoint,MODELS/f'{name}.pt')
        reloaded = NeuralDecoder.load(MODELS/f'{name}.pt')
        np.testing.assert_array_equal(decoder.predict_proba(va['x']),reloaded.predict_proba(va['x']))
        models[name] = reloaded
        training[name] = dict(history=history,weight_change_l2=delta,parameters=sum(w.numel() for w in decoder.model.parameters()))
        print(name,'epochs',len(history),'weight change',delta,flush=True)
    shuffled, history, delta = train(tr,va,101,p,shuffle=True)
    models['shuffled-label-mlp'] = shuffled
    training['shuffled-label-mlp'] = dict(history=history,weight_change_l2=delta)
    torch.manual_seed(101)
    scaler = StandardScaler().fit(tr['x'])
    models['untrained-mlp'] = NeuralDecoder(network(),scaler.mean_,scaler.scale_)
    for name in ('shuffled-label-mlp', 'untrained-mlp'):
        decoder = models[name]
        torch.save(dict(state_dict=decoder.model.state_dict(), mean=torch.tensor(decoder.mean),
                        scale=torch.tensor(decoder.scale), classes=['rest','left','right'],
                        architecture=[48,32,3], protocol=p), MODELS/f'{name}.pt')
    save_json(OUT/'training.json',training)
    save_json(OUT/'model-lock.json',dict(protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        checkpoint_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in MODELS.glob('*.pt')}))
    print('Models locked; generating final test sessions',flush=True)
    tests = [generate_session(s,p['anatomy_seed']) for s in p['splits']['test']]
    result = dict(protocol=p,torch_version=torch.__version__,models={})
    for name in [*models,'lda','ridge']:
        session_results = []
        for row in tests:
            start = time.perf_counter()
            probabilities = None
            if name in models:
                probabilities = models[name].predict_proba(row['x'])
                pred = probabilities.argmax(1)
            elif name == 'lda':
                probabilities = lda.predict_proba(row['x'])
                pred = probabilities.argmax(1)
            else:
                drive = direct.predict(row['x'])
                pred = np.where(drive>=.18,1,np.where(drive<=-.18,2,0))
            elapsed = time.perf_counter()-start
            measured = metrics(row,pred)
            measured.update(session=int(row['session'][0]), batch_seconds=elapsed,
                            amortized_inference_seconds=elapsed/len(pred))
            session_results.append(measured)
            payload = dict(prediction=pred,command=gated_predictions(pred,row['time']),label=row['y'],time=row['time'])
            if probabilities is not None:
                payload['uncalibrated_probabilities']=probabilities
            np.savez_compressed(OUT/f"predictions-{name}-{row['session'][0]}.npz",**payload)
        summary = summarize(session_results)
        result['models'][name] = dict(summary=summary,passes_criteria=passes(summary,p['criteria']),sessions=session_results)
        print(name,json.dumps(summary),flush=True)
    save_json(OUT/'evaluation.json',result)
    save_json(OUT/'data-manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.glob('*.npz')})
    plot(training,result)


def plot(training,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,row in training.items():
        if name.startswith('mlp-'):
            axes[0].plot([r['training_loss'] for r in row['history']],label=name+' train')
            axes[0].plot([r['validation_loss'] for r in row['history']],linestyle='--',label=name+' val')
    axes[0].set(xlabel='Epoch (zero indexed)',ylabel='Cross entropy',title='Actual supervised optimization')
    axes[0].legend(fontsize=7)
    names=list(result['models'])
    axes[1].barh(names,[result['models'][n]['summary']['balanced_accuracy'] for n in names])
    axes[1].set(xlim=(0,1),xlabel='Balanced accuracy',title='Held-out sessions 19-22')
    fig.savefig(OUT/'results.png',dpi=140)
    plt.close(fig)


def verify_saved():
    """Fresh-process replay of saved test electrodes; no fitting or tuning."""
    torch.set_num_threads(1)
    results = json.loads((OUT/'evaluation.json').read_text())
    checks = []
    for name in results['models']:
        model = (NeuralDecoder.load(MODELS/f'{name}.pt') if (MODELS/f'{name}.pt').exists()
                 else joblib.load(MODELS/f'{name}.joblib'))
        for session in results['protocol']['splits']['test']:
            x, _ = causal_features(np.load(DATA/f'electrodes-{session}.npz')['recordings'])
            saved = np.load(OUT/f'predictions-{name}-{session}.npz')
            if name == 'ridge':
                drive = model.predict(x)
                pred = np.where(drive>=.18,1,np.where(drive<=-.18,2,0))
            else:
                probabilities = model.predict_proba(x)
                np.testing.assert_array_equal(probabilities, saved['uncalibrated_probabilities'])
                pred = probabilities.argmax(1)
            np.testing.assert_array_equal(pred, saved['prediction'])
            checks.append(dict(model=name,session=session,identical=True))
    save_json(OUT/'reload-verification.json', checks)
    print(f'Fresh-process saved-model replay: {len(checks)} model/session checks identical.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-saved', action='store_true')
    args=parser.parse_args()
    verify_saved() if args.verify_saved else run()

