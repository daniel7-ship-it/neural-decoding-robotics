"""Interactive synthetic recording drift and reward-selected decoder lab."""
import argparse
import json
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import mujoco
import numpy as np

from .closed_loop import DT, Episode, PROTOCOL, fit_decoders
from .demo import OUT
from .robustness import (CONDITIONS, PATHS, CorrectedDecoder,
                         RecordingDisturbance, targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--path', choices=PATHS, default='direct')
    parser.add_argument('--condition', choices=CONDITIONS, default='shift_sudden')
    parser.add_argument('--fast', action='store_true', help='No real-time pacing (for rendering checks)')
    args = parser.parse_args()
    models, _ = fit_decoders(42)
    selections = {}
    adaptation_result = OUT/'robustness-adaptation.json'
    if adaptation_result.exists():
        selections = json.loads(adaptation_result.read_text())['selected'].get('42', {})

    def make_episode(path, condition, mode):
        decoder = models[path]
        if mode == 'adapted' and path in selections:
            chosen = selections[path]
            decoder = CorrectedDecoder(decoder, chosen['gain'], chosen['bias'])
        return Episode(decoder, 42, 6,
                       disturbance=RecordingDisturbance(42, 6, condition),
                       targets=targets(42, 6), rest_seconds=60)

    choice = {'path': args.path, 'condition': args.condition, 'mode': 'frozen'}
    episode = make_episode(**choice)
    renderer = mujoco.Renderer(episode.sim, height=400, width=600)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(bottom=.25, top=.88, hspace=.4, wspace=.25)
    fig.canvas.manager.set_window_title('Neuralduck — recording drift and adaptation')
    duck_ax, heading_ax, electrodes_ax, activity_ax = axes.ravel()
    duck_ax.axis('off')
    duck_image = duck_ax.imshow(np.zeros((400, 600, 3), dtype=np.uint8))
    target_line, = heading_ax.plot([], [], '--', label='Target')
    heading_line, = heading_ax.plot([], [], label='Actual heading')
    heading_ax.set(xlim=(0, episode.seconds), ylim=(-1.7, 1.7), xlabel='Simulation seconds', ylabel='Heading (rad)')
    heading_ax.legend(loc='upper right')
    electrode_image = electrodes_ax.imshow(np.zeros((24, 20)), origin='lower', aspect='auto', extent=(-.2, 0, 0, 24))
    electrodes_ax.set(xlabel='Recent recording history (seconds)', ylabel='Electrode')
    true_line, = activity_ax.plot(np.zeros(96), label='True simulated mean')
    inferred_line, = activity_ax.plot(np.zeros(96), '--', label='Reconstructed mean')
    activity_ax.set(xlim=(0, 95), ylim=(-.2, 1.4), xlabel='Neuron', ylabel='Rate (arbitrary units)')
    activity_ax.legend(loc='upper right', fontsize=8)
    fig.text(.07, .92, 'Synthetic recording changes · frozen versus reward-selected correction', fontsize=11)
    status = fig.text(.07, .17, '', fontsize=8, family='monospace', va='top')
    radio = RadioButtons(fig.add_axes((.78, .05, .17, .13)), PATHS, active=PATHS.index(args.path))
    condition_button = Button(fig.add_axes((.27, .065, .21, .055)), f'Condition: {choice["condition"]}')
    mode_button = Button(fig.add_axes((.51, .065, .23, .055)), 'Mode: frozen')
    pending = {'restart': False}
    radio.on_clicked(lambda path: (choice.update(path=path), pending.update(restart=True)))

    def next_condition(_):
        choice['condition'] = CONDITIONS[(CONDITIONS.index(choice['condition'])+1) % len(CONDITIONS)]
        condition_button.label.set_text(f'Condition: {choice["condition"]}')
        pending['restart'] = True

    def toggle_mode(_):
        if choice['mode'] == 'frozen' and not selections:
            return
        choice['mode'] = 'adapted' if choice['mode'] == 'frozen' else 'frozen'
        mode_button.label.set_text('Mode: adapted (locked)' if choice['mode'] == 'adapted' else 'Mode: frozen')
        pending['restart'] = True

    condition_button.on_clicked(next_condition)
    mode_button.on_clicked(toggle_mode)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .7, 135, -20
    OUT.mkdir(exist_ok=True)
    plt.show(block=False)
    screenshot_saved = False
    launched = time.perf_counter()
    log = (OUT/'robustness-dashboard-events.jsonl').open('a', encoding='utf-8')
    try:
        while plt.fignum_exists(fig.number) and time.perf_counter()-launched < args.seconds:
            start = time.perf_counter()
            if pending['restart']:
                log.write(json.dumps({'settings': choice, 'event': 'restart_or_change',
                                      'metrics': episode.metrics()})+'\n')
                log.flush()
                renderer.close()
                episode = make_episode(**choice)
                renderer = mujoco.Renderer(episode.sim, height=400, width=600)
                pending['restart'] = False
                screenshot_saved = False
            if episode.tick*DT >= episode.seconds:
                log.write(json.dumps({'settings': choice, 'metrics': episode.metrics()})+'\n')
                log.flush()
                pending['restart'] = True
                continue
            row = episode.step()
            if episode.tick % 10 == 0:
                camera.lookat[:] = episode.data.xpos[episode.robot.trunk]
                renderer.update_scene(episode.data, camera=camera)
                duck_image.set_data(renderer.render())
                rows = episode.trace
                times = [r['time'] for r in rows]
                target_line.set_data(times, [r['target'] if r['segment'] >= 0 else np.nan for r in rows])
                heading_line.set_data(times, [r['heading'] for r in rows])
                samples = np.asarray(episode.stream.observed).T
                electrode_image.set_data(samples)
                electrode_image.set_clim(float(samples.min()), float(samples.max())+.001)
                true_line.set_ydata(np.mean(episode.stream.truth, axis=0))
                inferred_line.set_ydata(episode.estimate if episode.estimate is not None else np.full(96, np.nan))
                activity_ax.set_title('Synthetic truth vs inferred mean activity' if choice['path'] != 'direct'
                                      else 'Synthetic truth (direct path has no activity estimate)', fontsize=10)
                metrics = episode.metrics()
                task = f"target {row['target']:+.2f} rad" if row['segment'] >= 0 else 'continuous rest / warmup'
                completed = [t for i, t in enumerate(metrics['tasks'])
                             if row['time'] >= 2+(i+1)*PROTOCOL.target_seconds]
                chosen = selections.get(choice['path']) if choice['mode'] == 'adapted' else None
                progress = (f"Search 6/6; choice {chosen['candidate']+1}/6; R={chosen['adaptation_reward']:.1f}"
                            if chosen else 'Frozen output correction')
                strength = episode.stream.disturbance.magnitude(row['time'])
                status.set_text(f"{choice['path']} {choice['mode']} | {task} | error {row['error']:+.2f} | drift {strength:.2f}\n"
                    f"Drive {row['decoded_drive']:+.2f} | yaw {row['yaw_command']:+.2f} | targets {sum(t['success'] for t in completed)}/{len(completed)} | rest events {metrics['false_movement_events']}\n"
                    + progress)
                fig.suptitle(f"Recording robustness — {choice['condition']} — {choice['mode']} — {row['time']:.1f}s")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if row['time'] >= 5 and not screenshot_saved:
                    fig.savefig(OUT/'robustness-dashboard.png', dpi=120)
                    screenshot_saved = True
            if args.fast and episode.tick*DT >= min(args.seconds, episode.seconds):
                break
            if not args.fast:
                time.sleep(max(0, DT-(time.perf_counter()-start)))
    finally:
        log.write(json.dumps({'settings': choice, 'partial': episode.tick*DT < episode.seconds,
                              'metrics': episode.metrics()})+'\n')
        log.close()
        renderer.close()
        plt.close(fig)


if __name__ == '__main__':
    main()
