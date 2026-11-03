"""Live continuous neural-feedback task, using the existing Microduck policy."""
import argparse
import json
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import mujoco
import numpy as np

from .closed_loop import DT, Episode, PATHS, PROTOCOL, fit_decoders
from .demo import OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--path', choices=PATHS, default='direct')
    parser.add_argument('--fast', action='store_true', help='No real-time pacing (for rendering checks)')
    args = parser.parse_args()
    models, _ = fit_decoders(42)
    episode = Episode(models[args.path], 42, 4)
    renderer = mujoco.Renderer(episode.sim, height=400, width=600)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(bottom=.25, top=.88, hspace=.4, wspace=.25)
    fig.canvas.manager.set_window_title('Neuralduck — continuous feedback experiment')
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
    fig.text(.07, .92, 'Synthetic circuit → electrodes → decoder → Microduck → heading feedback', fontsize=11)
    status = fig.text(.08, .16, '', fontsize=9, family='monospace', va='top')
    radio = RadioButtons(fig.add_axes((.76, .045, .19, .14)), PATHS, active=PATHS.index(args.path))
    button = Button(fig.add_axes((.53, .06, .17, .055)), 'Restart matched task')
    pending = {'path': None}
    radio.on_clicked(lambda path: pending.update(path=path))
    button.on_clicked(lambda _: pending.update(path=radio.value_selected))
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .7, 135, -20
    OUT.mkdir(exist_ok=True)
    plt.show(block=False)
    screenshot_saved = False
    launched = time.perf_counter()
    log = (OUT/'closed-loop-dashboard-events.jsonl').open('a', encoding='utf-8')
    try:
        while plt.fignum_exists(fig.number) and time.perf_counter()-launched < args.seconds:
            start = time.perf_counter()
            if pending['path'] is not None:
                log.write(json.dumps({'path': episode.decoder.path, 'event': 'restart_or_path_change',
                                      'next_path': pending['path'], 'metrics': episode.metrics()})+'\n')
                log.flush()
                renderer.close()
                episode = Episode(models[pending['path']], 42, 4)
                renderer = mujoco.Renderer(episode.sim, height=400, width=600)
                pending['path'] = None
                screenshot_saved = False
            if episode.tick*DT >= episode.seconds:
                log.write(json.dumps({'path': episode.decoder.path, 'metrics': episode.metrics()})+'\n')
                log.flush()
                pending['path'] = episode.decoder.path
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
                activity_ax.set_title('Mean activity: simulation truth vs electrode estimate', fontsize=10)
                metrics = episode.metrics()
                task = f"target {row['target']:+.2f} rad" if row['segment'] >= 0 else 'continuous rest / warmup'
                completed = [t for i, t in enumerate(metrics['tasks'])
                             if row['time'] >= 2+(i+1)*PROTOCOL.target_seconds]
                status.set_text(f"Path: {episode.decoder.path} | {task} | error {row['error']:+.2f} rad\n"
                    f"Decoded drive {row['decoded_drive']:+.2f} | applied yaw {row['yaw_command']:+.2f} rad/s\n"
                    f"Successful completed targets: {sum(t['success'] for t in completed)}/{len(completed)} | Falls: {metrics['falls']}\n"
                    f"Rest movement events: {metrics['false_movement_events']} | Loop: {episode.seconds:.0f}s\n"
                    'Observation: 200ms; updates: 100ms; expiry: 250ms')
                fig.suptitle(f"Continuous neural feedback — {episode.decoder.path} — {row['time']:.1f}s")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if row['time'] >= 5 and not screenshot_saved:
                    fig.savefig(OUT/'closed-loop-dashboard.png', dpi=120)
                    screenshot_saved = True
            if args.fast and episode.tick*DT >= min(args.seconds, episode.seconds):
                break
            if not args.fast:
                time.sleep(max(0, DT-(time.perf_counter()-start)))
    finally:
        log.write(json.dumps({'path': episode.decoder.path, 'partial': episode.tick*DT < episode.seconds,
                              'metrics': episode.metrics()})+'\n')
        log.close()
        renderer.close()
        plt.close(fig)


if __name__ == '__main__':
    main()
