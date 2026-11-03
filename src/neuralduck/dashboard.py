"""Local research dashboard: official physics plus synthetic neural recordings."""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .demo import OUT, RESEARCH, ROOT, evaluate_synthetic
from .microduck import MicroduckPolicy, create_simulation
from .neural import LABELS, MotorCircuit, SafeCommandDecoder, effective_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--eeg-replay", action="store_true",
                        help="replay saved held-out EEG predictions, not synthetic activity")
    args = parser.parse_args()
    metrics, decoder_model, trials = evaluate_synthetic()
    OUT.mkdir(exist_ok=True)
    (OUT / "synthetic-evaluation.json").write_text(json.dumps(metrics, indent=2))
    circuit = MotorCircuit(seed=20261026 + 3 * 1009)
    sim, data, bam = create_simulation(RESEARCH / "src/mjlab_microduck/robot/microduck/scene.xml")
    policy = MicroduckPolicy(sim, data, bam, ROOT / "models/velstand.onnx")
    decoder = SafeCommandDecoder(decoder_model)
    renderer = mujoco.Renderer(sim, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 0.65
    camera.azimuth = 135
    camera.elevation = -18
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.canvas.manager.set_window_title("Neuralduck — synthetic neural interface / MuJoCo")
    ax_duck, ax_rates, ax_electrodes, ax_map = axes.ravel()
    image = ax_duck.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    ax_duck.axis("off")
    bar = ax_rates.bar(np.arange(96), trials[0].activity, color="teal")
    ax_rates.set(ylim=(0, 1.3), xlabel="Simulated neuron", ylabel="Rate (arbitrary units)",
                 title="Toy rate circuit: activity, not human brain data")
    lines = [ax_electrodes.plot(np.arange(100) * .01, np.zeros(100), lw=.8)[0] for _ in range(6)]
    ax_electrodes.set(xlim=(0, 1), ylim=(-.5, 6), xlabel="Recording window (seconds)",
                      ylabel="Virtual electrode + offset", title="Six of 24 virtual electrodes")
    ax_map.imshow(circuit.recurrent, cmap="coolwarm", vmin=-.04, vmax=.04)
    ax_map.set(title="Known synthetic connectivity", xlabel="Source neuron", ylabel="Target neuron")
    status = fig.suptitle("Initializing pretrained Microduck controller")
    fig.tight_layout(rect=(0, 0, 1, .93))
    plt.show(block=False)
    eeg = None
    if args.eeg_replay:
        path = OUT / "physionet-evaluation.json"
        if not path.exists():
            raise RuntimeError("Run the EEG evaluation first (see README).")
        eeg = json.loads(path.read_text())
        ax_rates.set_title("EEG replay: class probabilities (no synthetic neural activity)")
        ax_rates.clear()
        ax_rates.set(ylim=(0, 1), xticks=[0, 1], xticklabels=["Left", "Right"], ylabel="Probability")
        eeg_bars = ax_rates.bar([0, 1], [.5, .5])
        ax_electrodes.clear()
        ax_electrodes.axis("off")
        ax_electrodes.text(.05, .5, "Recorded PhysioNet subject 1, run 12\n2-second cue-aligned windows\nOpen-loop replay; not live brain control\nNo validated rest detector", transform=ax_electrodes.transAxes)
        ax_map.clear()
        ax_map.axis("off")
        ax_map.text(.05, .5, "CSP + LDA baseline\n15 held-out trials\nSee saved evaluation for limitations", transform=ax_map.transAxes)
    dt, current = .02, None
    trace = []
    try:
        for step in range(max(1, int(args.seconds / dt))):
            if not plt.fignum_exists(fig.number):
                break
            started = time.perf_counter()
            now = step * dt
            trial_seconds = 4 if eeg is not None else 2
            slot = int(now // trial_seconds)
            # A new one-second recording arrives at each trial boundary. During
            # acquisition there is no command; completed windows renew at 5 Hz.
            trial = trials[(slot * 37) % len(trials)]
            if eeg is None:
                if now % 2 >= 1:
                    current = decoder.decode(trial.features, now)
                else:
                    current = None
            else:
                from .neural import VelocityCommand
                probabilities = eeg["test_probabilities"][slot % len(eeg["test_predictions"])]
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
                label = ["left", "right"][index] if confidence >= .62 and now % 4 >= 2 else "rest"
                current = VelocityCommand(now, 0, 0, 1.2 if label == "left" else -1.2 if label == "right" else 0,
                                          confidence, now + .3, label)
            command = effective_command(current, now)
            policy.step(command)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Nonfinite simulator state")
            trace.append((now, *command, *policy.body_velocity()))
            if step % 10 == 0:
                camera.lookat[:] = data.xpos[policy.trunk]
                renderer.update_scene(data, camera=camera)
                image.set_data(renderer.render())
                if eeg is None:
                    for rect, value in zip(bar, trial.activity):
                        rect.set_height(value)
                    for i, line in enumerate(lines):
                        line.set_ydata(trial.recordings[:, i * 4] * .7 + i * .8)
                else:
                    for rect, value in zip(eeg_bars, probabilities):
                        rect.set_height(value)
                label = current.label if current else "acquiring / stop"
                confidence = current.confidence if current else 0
                mode = "Recorded EEG replay" if eeg is not None else "Synthetic neural recordings"
                status.set_text(f"{mode} | decoded: {label} | confidence {confidence:.2f} | yaw command {command[2]:+.2f} rad/s")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                if step == (150 if eeg is not None else 250):
                    fig.savefig(OUT / ("eeg-dashboard.png" if eeg is not None else "neural-dashboard.png"), dpi=120)
            time.sleep(max(0, dt - (time.perf_counter() - started)))
    finally:
        renderer.close()
        np.save(OUT / ("eeg-dashboard-trace.npy" if eeg is not None else "dashboard-trace.npy"), np.asarray(trace))
        plt.close(fig)


if __name__ == "__main__":
    main()

