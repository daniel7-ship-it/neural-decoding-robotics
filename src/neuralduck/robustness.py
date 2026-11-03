"""Pre-registered synthetic recording drift and bounded reward adaptation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .closed_loop import Episode, OUT, fit_decoders, pool_rates
from .neural import VirtualElectrodes


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / 'docs' / 'robustness-protocol.json'
CONDITIONS = ('control', 'shift_sudden', 'shift_gradual', 'gain_sudden',
              'gain_gradual', 'noise_sudden', 'noise_gradual', 'loss_sudden', 'loss_gradual')
PATHS = ('direct', 'neurons', 'populations')
CANDIDATES = ((1., 0.), (.7, 0.), (1.4, 0.), (1., -.15), (1., .15), (-1., 0.))


def locked_protocol():
    spec = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    assert tuple(spec['disturbance']['conditions']) == CONDITIONS
    assert tuple(tuple(x) for x in spec['adaptation']['candidate_output_corrections_gain_bias']) == CANDIDATES
    return spec


def targets(seed, session):
    return tuple(np.random.default_rng(seed + session * 5011).choice(
        [-1., -.7, .7, 1.], size=3, replace=False).tolist())


class RecordingDisturbance:
    """One recording-only change; circuit anatomy/noise RNG remain paired."""

    def __init__(self, seed, session, name):
        if name not in CONDITIONS:
            raise ValueError(f'Unknown disturbance {name}')
        self.name = name
        self.kind = name.split('_')[0]
        self.gradual = name.endswith('gradual')
        self.noise_rng = np.random.default_rng(seed + session * 100003 + 9001)
        self.shifted = (VirtualElectrodes(seed=seed + session * 7919,
                         neuron_seed=seed, placement_shift=.65) if self.kind == 'shift' else None)

    def magnitude(self, timestamp):
        if self.kind == 'control' or timestamp < 2:
            return 0.
        return float(np.clip((timestamp-2)/18, 0, 1)) if self.gradual else 1.

    def apply(self, rates, observed, electrode, timestamp):
        magnitude = self.magnitude(timestamp)
        x = observed.copy()
        if self.kind == 'shift':
            delta = rates @ (self.shifted.mixing-electrode.mixing).T * electrode.gain[:, 0]
            x += magnitude * delta
        elif self.kind == 'gain':
            x[12:] *= 1 - .75*magnitude
        elif self.kind == 'noise':
            # Draw even before onset to keep the added-noise stream paired.
            x += magnitude * self.noise_rng.normal(0, .5, len(x))
        elif self.kind == 'loss':
            x[12:] *= 1-magnitude
        return x.astype(np.float32)


@dataclass(frozen=True)
class CorrectedDecoder:
    base: object
    gain: float
    bias: float

    @property
    def path(self):
        return self.base.path

    @property
    def mean_rates(self):
        return self.base.mean_rates

    def infer(self, electrode_features):
        drive, rates = self.base.infer(electrode_features)
        return float(np.clip(self.gain*drive+self.bias, -1, 1)), rates


def run_episode(model, seed, session, condition):
    disturbance = RecordingDisturbance(seed, session, condition)
    episode = Episode(model, seed, session, disturbance=disturbance,
                      targets=targets(seed, session), rest_seconds=60.)
    metrics = episode.run()
    return episode, metrics


def reward(metrics):
    return (100*sum(t['success'] for t in metrics['tasks'])
            - sum(abs(t['final_error_rad']) for t in metrics['tasks'])
            - 20*metrics['falls'] - .1*metrics['false_movement_events'])


def frozen_block(seeds, session, conditions=CONDITIONS):
    rows = []
    for seed in seeds:
        models, _ = fit_decoders(seed)
        for condition in conditions:
            for path in PATHS:
                episode, metrics = run_episode(models[path], seed, session, condition)
                rows.append({'seed': seed, 'session': session, 'condition': condition,
                             'path': path, 'mode': 'frozen', **metrics})
            print(f'Frozen session {session} seed {seed}: {condition}', flush=True)
            save('robustness-development.partial.json', {'runs': rows})
    return rows


def select_challenge(development):
    sudden = [c for c in CONDITIONS if c.endswith('sudden')]
    # Fixed tie-break order from the registered condition list.
    return min(sudden, key=lambda c: sum(r['target_success_rate'] for r in development
                                          if r['condition'] == c))


def paired_reconstruction(seed, session, condition, models):
    """Exactly the same recordings and truths for every reconstruction model."""
    from .closed_loop import ContinuousCircuit
    stream = ContinuousCircuit(seed, session, disturbance=RecordingDisturbance(seed, session, condition))
    rng = np.random.default_rng(seed + session*31013)
    levels = np.tile([-1., -.6, -.3, 0., .3, .6, 1.], 8)
    rng.shuffle(levels)
    features, truths = [], []
    tick = 0
    for level in levels:
        for _ in range(60):
            stream.sample(float(level), tick*.01)
            if tick % 10 == 9 and stream.features() is not None:
                features.append(stream.features())
                truths.append(np.mean(stream.truth, axis=0))
            tick += 1
    x, truth = np.asarray(features), np.asarray(truths)
    neuron = np.asarray([models['neurons'].infer(row)[1] for row in x])
    population = np.asarray([models['populations'].infer(row)[1] for row in x])
    constant = np.broadcast_to(models['neurons'].mean_rates, truth.shape)
    mse = lambda estimate: float(np.mean((estimate-truth)**2))
    return {'seed': seed, 'session': session, 'condition': condition, 'windows': len(x),
            'same_recordings': True, 'neuron_mse': mse(neuron),
            'population_mse': mse(population), 'constant_mse': mse(constant)}


def adaptation_block(seeds, challenge):
    rows = []
    selections = {}
    for seed in seeds:
        models, _ = fit_decoders(seed)
        selections[str(seed)] = {}
        for path in PATHS:
            candidate_rows = []
            for index, (gain, bias) in enumerate(CANDIDATES):
                model = CorrectedDecoder(models[path], gain, bias)
                _, metrics = run_episode(model, seed, 7, challenge)
                row = {'seed': seed, 'session': 7, 'condition': challenge,
                       'path': path, 'mode': 'online_candidate', 'candidate': index,
                       'gain': gain, 'bias': bias, 'reward': reward(metrics), **metrics}
                rows.append(row)
                candidate_rows.append(row)
                _, control_metrics = run_episode(models[path], seed, 7, challenge)
                rows.append({'seed': seed, 'session': 7, 'condition': challenge,
                             'path': path, 'mode': 'equal_budget_frozen', 'candidate': index,
                             'reward': reward(control_metrics), **control_metrics})
                save('robustness-adaptation.partial.json', {'challenge': challenge,
                     'runs': rows, 'selected_so_far': selections})
            best = max(candidate_rows, key=lambda r: r['reward'])
            selections[str(seed)][path] = {'candidate': best['candidate'],
                                           'gain': best['gain'], 'bias': best['bias'],
                                           'adaptation_reward': best['reward']}
            print(f'Adaptation seed {seed} path {path}: candidate {best["candidate"]}', flush=True)
    return rows, selections


def final_block(seeds, sessions, selections):
    rows, recon = [], []
    for seed in seeds:
        models, _ = fit_decoders(seed)
        for session in sessions:
            for condition in CONDITIONS:
                recon.append(paired_reconstruction(seed, session, condition, models))
                for path in PATHS:
                    spec = selections[str(seed)][path]
                    frozen = models[path]
                    adapted = CorrectedDecoder(frozen, spec['gain'], spec['bias'])
                    for mode, model in (('frozen', frozen), ('adapted_locked', adapted)):
                        _, metrics = run_episode(model, seed, session, condition)
                        rows.append({'seed': seed, 'session': session, 'condition': condition,
                                     'path': path, 'mode': mode, **metrics})
                print(f'Final session {session} seed {seed}: {condition}', flush=True)
                save('robustness-final.partial.json', {'runs': rows, 'paired_reconstruction': recon})
    return rows, recon


def summarize(rows):
    result = {}
    for condition in CONDITIONS:
        result[condition] = {}
        for path in PATHS:
            result[condition][path] = {}
            for mode in ('frozen', 'adapted_locked'):
                matched = [r for r in rows if (r['condition'], r['path'], r['mode']) == (condition, path, mode)]
                if not matched:
                    continue
                tasks = [t for r in matched for t in r['tasks']]
                settled = [t['settling_seconds'] for t in tasks if t['success']]
                result[condition][path][mode] = {
                    'successes': sum(t['success'] for t in tasks), 'targets': len(tasks),
                    'success_rate': float(np.mean([t['success'] for t in tasks])),
                    'settling_successes_mean': float(np.mean(settled)) if settled else None,
                    'overshoot_mean': float(np.mean([t['overshoot_rad'] for t in tasks])),
                    'fall_episodes': sum(r['falls'] for r in matched),
                    'rest_false_events': sum(r['false_movement_events'] for r in matched),
                    'rest_false_events_after_washout': sum(r['false_movement_events_after_1s_washout'] for r in matched),
                    'rest_command_occupancy_mean': float(np.mean([r['rest_command_fraction'] for r in matched])),
                    'rest_seconds': sum(r['rest_seconds'] for r in matched),
                    'latency_seconds_mean': float(np.mean([r['observation_to_command_seconds_mean'] for r in matched]))}
    return result


def save(name, result):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('development', 'adaptation', 'final', 'all'), default='all')
    args = parser.parse_args()
    protocol = locked_protocol()
    seeds = tuple(protocol['seeds'])
    sessions = tuple(protocol['sessions']['final_test'])
    development_path = OUT/'robustness-development.json'
    adaptation_path = OUT/'robustness-adaptation.json'
    if args.phase in ('development', 'all'):
        dev = frozen_block(seeds, protocol['sessions']['development'])
        challenge = select_challenge(dev)
        save('robustness-development.json', {'protocol': protocol, 'challenge_selected': challenge, 'runs': dev})
    else:
        prior = json.loads(development_path.read_text())
        challenge = prior['challenge_selected']
    if args.phase in ('adaptation', 'all'):
        rows, selections = adaptation_block(seeds, challenge)
        save('robustness-adaptation.json', {'challenge': challenge, 'runs': rows, 'selected': selections})
    else:
        selections = json.loads(adaptation_path.read_text())['selected'] if args.phase == 'final' else None
    if args.phase in ('final', 'all'):
        rows, recon = final_block(seeds, sessions, selections)
        result = {'protocol': protocol, 'selected_challenge_from_development': challenge,
                  'adaptation_selected_before_final': selections, 'runs': rows,
                  'paired_reconstruction': recon, 'summary': summarize(rows)}
        save('robustness-final.json', result)
        plot(result)
        print(json.dumps(result['summary'], indent=2), flush=True)


def plot(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
    conditions = list(CONDITIONS)
    x = np.arange(len(conditions))
    for path in PATHS:
        vals = [result['summary'][c][path]['frozen']['success_rate'] for c in conditions]
        axes[0].plot(x, vals, marker='o', label=path)
    axes[0].set(xticks=x, xticklabels=conditions, ylim=(-.05, 1.05),
                title='Frozen control under recording changes', ylabel='Target success rate')
    selected = result['selected_challenge_from_development']
    paths = list(PATHS)
    xp = np.arange(len(paths))
    for i, mode in enumerate(('frozen', 'adapted_locked')):
        axes[1].bar(xp+(i-.5)*.35, [result['summary'][selected][p][mode]['success_rate'] for p in paths],
                    width=.35, label=mode)
    axes[1].set(xticks=xp, xticklabels=paths, ylim=(0, 1.05),
                title=f'Fresh sessions: reward adaptation on {selected}', ylabel='Target success rate')
    for i, mode in enumerate(('frozen', 'adapted_locked')):
        axes[2].bar(x+(i-.5)*.35,
                    [sum(result['summary'][c][p][mode]['rest_false_events_after_washout'] for p in PATHS)
                     for c in conditions], width=.35, label=mode)
    axes[2].set(xticks=x, xticklabels=conditions,
                title='False movement events during rest (after washout)', ylabel='Events across all paths')
    axes[0].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='x', rotation=45)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.suptitle('Synthetic recording robustness · two subject seeds · sessions 9–10 held out')
    fig.savefig(OUT/'robustness-final.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
