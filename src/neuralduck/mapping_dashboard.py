"""Interactive electrode experiments and activity reconstruction with Microduck."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import mujoco
import numpy as np

from .demo import OUT, RESEARCH, ROOT
from .mapping import Settings, combine, fit_models, record_session, score_models
from .microduck import MicroduckPolicy, create_simulation
from .neural import LABELS, VelocityCommand, effective_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600)
    args = parser.parse_args()
    seed = 42
    calibrations = {}
    state = {"pending": True, "settings": Settings(), "start": 0., "last_event": None, "capture": True}
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title("Neuralduck — neural mapping experiments")
    grid = fig.add_gridspec(2, 2, left=.07, right=.96, top=.88, bottom=.27, hspace=.38, wspace=.25)
    duck_ax, activity_ax, electrode_ax, metrics_ax = [fig.add_subplot(grid[r, c]) for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))]
    duck_ax.axis("off")
    duck_image = duck_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    true_line, = activity_ax.plot(np.zeros(96), label="True mean activity", color="teal")
    inferred_line, = activity_ax.plot(np.zeros(96), label="Inferred from electrodes", color="darkorange", linestyle="--")
    activity_ax.set(xlim=(0, 95), ylim=(-.25, 1.5), xlabel="Neuron (fixed identity across sessions)", ylabel="Mean rate (arbitrary units)")
    activity_ax.legend(loc="upper right", fontsize=8)
    electrode_image = electrode_ax.imshow(np.zeros((24, 100)), aspect="auto", origin="lower", extent=(0, 1, 0, 24), cmap="viridis")
    electrode_ax.set(xlabel="One-second recording window", ylabel="Selected electrode")
    metrics_ax.axis("off")
    metrics_text = metrics_ax.text(0, 1, "Preparing baseline calibration...", va="top", fontsize=10, family="monospace")
    fig.text(.07, .92, "Synthetic rate circuit • reconstructing activity, not a human brain or its wiring", fontsize=12)
    header = fig.suptitle("Neural mapping experiments — preparing", fontsize=14)
    count_slider = Slider(fig.add_axes((.15, .19, .35, .025)), "Electrodes", 1, 24, valinit=24, valstep=1)
    noise_slider = Slider(fig.add_axes((.15, .14, .35, .025)), "Noise", 0, 2, valinit=.065)
    shift_slider = Slider(fig.add_axes((.15, .09, .35, .025)), "Position shift", -1.5, 1.5, valinit=0)
    radio = RadioButtons(fig.add_axes((.57, .08, .16, .13)), ("No silencing", "Silence left", "Silence right"))
    button = Button(fig.add_axes((.78, .13, .17, .055)), "Apply experiment")
    fig.text(.57, .035, "Count recalibrates; noise/shift/silencing test frozen models.\nDrag sliders, then Apply. Close window to stop.", fontsize=9)

    def apply(_):
        silence = {"No silencing": None, "Silence left": "left", "Silence right": "right"}[radio.value_selected]
        state["settings"] = Settings(int(count_slider.val), float(noise_slider.val), float(shift_slider.val), silence)
        state["pending"] = True
    button.on_clicked(apply)
    plt.show(block=False)
    sim, data, bam = create_simulation(RESEARCH / "src/mjlab_microduck/robot/microduck/scene.xml")
    controller = MicroduckPolicy(sim, data, bam, ROOT / "models/velstand.onnx")
    renderer = mujoco.Renderer(sim, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .65, 135, -18
    OUT.mkdir(exist_ok=True)
    log = (OUT / "mapping-dashboard-events.jsonl").open("a", encoding="utf-8")
    try:
        for step in range(max(1, int(args.seconds / .02))):
            if not plt.fignum_exists(fig.number):
                break
            started = time.perf_counter()
            now = step * .02
            if state["pending"]:
                header.set_text("Calibrating selected electrodes; duck holding zero command")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                settings = state["settings"]
                if settings.electrodes not in calibrations:
                    base = Settings(electrodes=settings.electrodes)
                    train = combine([record_session(seed, s, base) for s in (0, 1, 2)])
                    val = record_session(seed, 3, base)
                    calibrations[settings.electrodes] = fit_models(train, val)
                models = calibrations[settings.electrodes]
                dataset = record_session(seed, 4, settings)
                metrics = score_models(models, dataset)
                # Neither labels nor ground-truth activity enter infer().
                predicted, reconstructed, probabilities, accepted = models.infer(dataset.features)
                state.update(pending=False, start=now, last_event=None, capture=True)
                metrics_text.set_text(
                    "Held-out session 4 (36 windows)\n"
                    f"Balanced accuracy: {metrics['balanced_accuracy']:.1%}\n"
                    f"Mean activity MSE: {metrics['reconstruction_mse']:.5f}\n"
                    f"Constant baseline MSE: {metrics['constant_mean_baseline_mse']:.5f}\n"
                    f"Reconstruction skill: {metrics['reconstruction_skill_vs_constant']:.1%}\n"
                    f"False movement / 60 rest windows: {metrics['false_movement_windows_per_60_rest_windows']:.1f}\n"
                    f"Accepted windows: {metrics['accepted_fraction']:.1%}\n"
                    f"Observation delay: 1s + compute/display\n"
                    f"Electrodes: {settings.electrodes}; noise: {settings.noise:.2f}\n"
                    f"Shift: {settings.shift:+.2f}; silence: {settings.silence or 'none'}")
                log.write(json.dumps({"type": "experiment", "simulation_time": now, "settings": asdict(settings), "metrics": metrics}) + "\n")
                log.flush()
            phase_time = now - state["start"]
            slot = int(phase_time // 2) % len(dataset.labels)
            ready = phase_time % 2 >= 1
            label_id = int(predicted[slot]) if ready else 0
            confidence = float(probabilities[slot].max()) if ready else 0.
            command = VelocityCommand(now, 0, 0, 1.2 if label_id == 1 else -1.2 if label_id == 2 else 0., confidence, now + .3, LABELS[label_id])
            controller.step(effective_command(command, now))
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Nonfinite robot physics state")
            if step % 10 == 0:
                camera.lookat[:] = data.xpos[controller.trunk]
                renderer.update_scene(data, camera=camera)
                duck_image.set_data(renderer.render())
                true_line.set_ydata(dataset.rates[slot])
                inferred_line.set_ydata(reconstructed[slot] if ready else np.full(96, np.nan))
                visible_samples = max(1, min(100, int((phase_time % 2) * 100)))
                frame = dataset.recordings[slot].T.copy()
                frame[:, visible_samples:] = np.nan
                electrode_image.set_data(frame)
                electrode_image.set_extent((0, 1, 0, settings.electrodes))
                electrode_image.set_clim(float(np.min(dataset.recordings)), float(np.max(dataset.recordings)))
                electrode_ax.set_ylim(0, settings.electrodes)
                header.set_text(f"Decoded: {LABELS[label_id] if ready else 'acquiring / stop'} | confidence {confidence:.2f} | actual yaw {controller.body_velocity()[2]:+.2f} rad/s")
                activity_ax.set_title("True vs reconstructed window-mean activity" if ready else "Acquiring: reconstruction withheld until window complete")
                if ready and state["last_event"] != slot:
                    log.write(json.dumps({"type": "decision", "simulation_time": now, "trial": slot,
                                          "decoded": LABELS[label_id], "truth_evaluation_only": LABELS[int(dataset.labels[slot])],
                                          "confidence": confidence, "yaw_command": command.yaw_rate}) + "\n")
                    log.flush()
                    state["last_event"] = slot
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if ready and state["capture"]:
                    fig.savefig(OUT / "mapping-dashboard.png", dpi=120)
                    state["capture"] = False
            time.sleep(max(0, .02 - (time.perf_counter() - started)))
    finally:
        log.close()
        renderer.close()
        plt.close(fig)


if __name__ == "__main__":
    main()
