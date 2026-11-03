"""Reproducible simulation, synthetic decoder evaluation, and local launcher."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .microduck import MicroduckPolicy, create_simulation
from .neural import (LABELS, SafeCommandDecoder, effective_command,
                     evaluate_decoder, fit_decoder, make_session_trials)


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RESEARCH = WORKSPACE / "microduck-rl-research"
OUT = ROOT / "artifacts"


def evaluate_synthetic() -> tuple[dict, object, list]:
    train = sum((make_session_trials(s, seed=20261026, repetitions=36)
                 for s in (0, 1, 2)), [])
    test = make_session_trials(3, seed=20261026, repetitions=48)
    result = evaluate_decoder(train, test)
    model = result.pop("model")
    y = np.array([t.label for t in test])
    x = np.stack([t.features for t in test])
    t0 = time.perf_counter()
    pred = model.predict(x)
    batch_ms = (time.perf_counter() - t0) * 1000.0
    rest = y == 0
    false_events = int(np.sum(pred[rest] != 0))
    result.update({
        "sessions": {"training": [0, 1, 2], "held_out_test": 3},
        "trials": {"train": len(train), "test": len(test)},
        "labels": list(LABELS),
        "per_class_recall": {
            LABELS[i]: float(np.mean(pred[y == i] == i)) for i in range(3)
        },
        "rest_false_commands_per_minute": float(false_events / max(rest.sum(), 1) * 60),
        "rest_trials": int(rest.sum()),
        "coverage": float(np.mean(np.max(model.predict_proba(x), axis=1) >= 0.62)),
        "decoder_batch_latency_ms_mean_per_trial": float(batch_ms / len(x)),
    })
    # Decoder consumes only features. Shuffle baseline trains on shuffled labels.
    rng = np.random.default_rng(83)
    from sklearn.base import clone
    shuffled_scores = []
    for k in range(20):
        yshuf = np.random.default_rng(83 + k).permutation(np.array([t.label for t in train]))
        shuffled = clone(model).fit(np.stack([t.features for t in train]), yshuf)
        shuffled_scores.append(balanced_accuracy_score(y, shuffled.predict(x)))
    result["shuffled_label_baseline_balanced_accuracy_mean"] = float(np.mean(shuffled_scores))
    result["shuffled_label_baseline_balanced_accuracy_sd"] = float(np.std(shuffled_scores))
    result["interpretation"] = "Simple synthetic rate circuit, not EEG or a model of a whole human brain; easy task by construction."
    result["observation_window_seconds"] = 1.0
    result["rest_metric_definition"] = "Non-rest predictions per 60 independent one-second rest windows; not continuous-stream false-trigger events."
    import joblib
    joblib.dump(model, ROOT / "models/synthetic-decoder.joblib")
    # Intervention: silence one population only in a new, held-out session.
    silenced = make_session_trials(3, seed=20261026, repetitions=48, silenced="left")
    ys = np.array([t.label for t in silenced])
    ps = model.predict(np.stack([t.features for t in silenced]))
    result["left_pool_silencing_balanced_accuracy"] = float(balanced_accuracy_score(ys, ps))
    result["left_pool_silencing_confusion_matrix"] = confusion_matrix(ys, ps, labels=[0, 1, 2]).tolist()
    return result, model, test


def replay_commands(test_trials, model, duration: float = 9.0):
    left = [t for t in test_trials if t.label == 1]
    right = [t for t in test_trials if t.label == 2]
    # Interleave held-out neural trials; their truth is retained only for this
    # experiment schedule/analysis. The decoder itself receives feature values.
    chosen = [left[0], right[0], left[1], right[1]] if len(left) > 1 and len(right) > 1 else []
    if not chosen:
        return []
    sequence = []
    for i, trial in enumerate(chosen):
        sequence.append((float(i * 2.0), 1.5, trial.features, trial.label))
    return sequence


def run_robot(model, test_trials, seconds: float, gui: bool, save_frame: bool):
    OUT.mkdir(exist_ok=True)
    scene = RESEARCH / "src/mjlab_microduck/robot/microduck/scene.xml"
    policy_path = ROOT / "models/velstand.onnx"
    mj, data, bam = create_simulation(scene)
    controller = MicroduckPolicy(mj, data, bam, policy_path)
    controller.reset()
    timeline = replay_commands(test_trials, model)
    decoder = SafeCommandDecoder(model)
    dt = 0.02
    total = int(seconds / dt)
    cmd_trace, vel_trace, cmd_events = [], [], []
    sim_time = 0.0
    state = {"frame_saved": False}

    def step_once(viewer=None):
        nonlocal sim_time
        active = None
        for start, span, features, truth in timeline:
            if start <= sim_time < start + span:
                # Every 50 Hz recording update renews the short-lived command.
                active = decoder.decode(features, now=sim_time)
                if not cmd_events or cmd_events[-1]["start"] != start:
                    cmd_events.append({"start": start, "decoded": active.label,
                                       "truth_for_eval_only": LABELS[truth],
                                       "confidence": active.confidence})
                break
        command = effective_command(active, sim_time)
        controller.step(command)
        velocity = controller.body_velocity()
        cmd_trace.append(command)
        vel_trace.append(velocity)
        sim_time += dt
        if viewer is not None:
            viewer.sync()
        return command, velocity

    if gui:
        with mujoco.viewer.launch_passive(mj, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.sync()
            for _ in range(total):
                if not viewer.is_running():
                    break
                tic = time.perf_counter()
                command, velocity = step_once(viewer)
                if save_frame and not state["frame_saved"]:
                    try:
                        renderer = mujoco.Renderer(mj, height=480, width=640)
                        renderer.update_scene(data, camera=-1)
                        import matplotlib.pyplot as plt
                        plt.imsave(OUT / "integration-frame.png", renderer.render())
                        renderer.close()
                        state["frame_saved"] = True
                    except Exception as exc:
                        (OUT / "render-error.txt").write_text(str(exc), encoding="utf-8")
                time.sleep(max(0.0, dt - (time.perf_counter() - tic)))
    else:
        for _ in range(total):
            step_once()
            if save_frame and not state["frame_saved"]:
                try:
                    renderer = mujoco.Renderer(mj, height=480, width=640)
                    renderer.update_scene(data, camera=-1)
                    import matplotlib.pyplot as plt
                    plt.imsave(OUT / "integration-frame.png", renderer.render())
                    renderer.close()
                    state["frame_saved"] = True
                except Exception as exc:
                    (OUT / "render-error.txt").write_text(str(exc), encoding="utf-8")

    commands = np.asarray(cmd_trace)
    velocities = np.asarray(vel_trace)
    actual_time = np.arange(len(commands)) * dt
    active_mask = np.abs(commands[:, 2]) > 0
    tracking = float(np.mean(np.sign(commands[active_mask, 2]) ==
                             np.sign(velocities[active_mask, 2]))) if active_mask.any() else 0.0
    integration = {
        "physics": "BAM M6 XL330 at 7.4V, 0.1 load sag; 5ms physics x4 = 50Hz policy",
        "policy_input_dim": controller.obs_dim,
        "sim_seconds": float(len(commands) * dt),
        "command_events": cmd_events,
        "tracking_yaw_sign_accuracy": tracking,
        "mean_abs_yaw_rate_active": float(np.mean(np.abs(velocities[active_mask, 2]))) if active_mask.any() else 0.0,
        "trunk_height_m": float(data.qpos[int(mj.jnt_qposadr[controller.free]) + 2]),
        "frame_saved": state["frame_saved"],
    }
    np.savez_compressed(OUT / "integration-trace.npz", time=actual_time,
                        command=commands, measured_body_velocity=velocities)
    (OUT / "integration.json").write_text(json.dumps(integration, indent=2), encoding="utf-8")
    return integration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--gui", action="store_true", help="open native MuJoCo viewer")
    parser.add_argument("--no-frame", action="store_true")
    parser.add_argument("--eeg", action="store_true", help="also download/evaluate PhysioNet subject 1")
    args = parser.parse_args()
    metrics, model, test = evaluate_synthetic()
    OUT.mkdir(exist_ok=True)
    (OUT / "synthetic-evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    integration = run_robot(model, test, args.seconds, args.gui, not args.no_frame)
    result = {"synthetic_evaluation": metrics, "integration": integration}
    if args.eeg:
        from .eeg import evaluate_physionet
        eeg = evaluate_physionet(ROOT / "data/physionet")
        (OUT / "physionet-evaluation.json").write_text(json.dumps(eeg, indent=2), encoding="utf-8")
        result["physionet"] = eeg
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
