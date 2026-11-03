"""Continuous synthetic sensory circuit -> electrodes -> decoder -> Microduck.

Synthetic supervised calibration; no human EEG or brain reconstruction claim.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .demo import OUT, RESEARCH, ROOT
from .microduck import MicroduckPolicy, create_simulation
from .neural import MotorCircuit, VirtualElectrodes, VelocityCommand, effective_command


DT = .02
WINDOW = 20  # 200 ms, sampled at 100 Hz
DECISION_STEPS = 5  # 100 ms between decisions
PATHS = ("direct", "neurons", "populations", "constant")


@dataclass(frozen=True)
class Protocol:
    # These definitions are fixed before validation and final evaluation.
    target_seconds: float = 12.
    rest_seconds: float = 12.
    tolerance_rad: float = .20
    settle_seconds: float = 1.
    fall_height_m: float = .065
    fall_up_z: float = .45
    ttl_seconds: float = .25
    sensory_deadband_rad: float = .10
    sensory_scale_rad: float = .65
    targets_rad: tuple = (1., -.9, .65)


PROTOCOL = Protocol()


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class ContinuousCircuit:
    """Persistent 96-rate circuit. Only this component receives sensory drive."""

    def __init__(self, seed, session, noise=.065, shift=0., disturbance=None):
        self.circuit = MotorCircuit(seed=seed)
        self.rates = np.zeros(96)
        self.rng = np.random.default_rng(seed + session * 100003 + 17)
        self.electrodes = VirtualElectrodes(seed=seed + session * 7919,
                                           neuron_seed=seed, noise_sd=noise,
                                           placement_shift=shift)
        self.disturbance = disturbance
        self.observed = deque(maxlen=WINDOW)
        self.truth = deque(maxlen=WINDOW)
        self.drives = deque(maxlen=WINDOW)
        self.times = deque(maxlen=WINDOW)
        self.wall_times = deque(maxlen=WINDOW)

    def sample(self, sensory_drive, timestamp):
        drive = np.zeros(96)
        drive[:48] = .75 * max(0., sensory_drive)
        drive[48:] = .75 * max(0., -sensory_drive)
        target = np.maximum(0, self.circuit.base + drive +
                            self.circuit.recurrent @ self.rates + self.rng.normal(0, .045, 96))
        self.rates += .01 / .08 * (target - self.rates)
        observation = self.electrodes.record(self.rates[None])[0]
        if self.disturbance is not None:
            observation = self.disturbance.apply(self.rates, observation, self.electrodes, timestamp)
        self.observed.append(observation)
        self.truth.append(self.rates.copy())
        self.drives.append(sensory_drive)
        self.times.append(timestamp)
        self.wall_times.append(time.perf_counter())
        return observation

    def features(self):
        if len(self.observed) < WINDOW:
            return None
        x = np.asarray(self.observed)
        return np.r_[x.mean(axis=0), x.std(axis=0)]


def calibration_session(seed, session):
    stream = ContinuousCircuit(seed, session)
    rng = np.random.default_rng(seed + session * 27011)
    levels = np.tile([-1., -.6, -.3, 0., 0., .3, .6, 1.], 8)
    rng.shuffle(levels)
    features, rates, drives = [], [], []
    tick = 0
    for level in levels:
        for _ in range(60):
            stream.sample(level, tick * .01)
            if tick % 10 == 9 and stream.features() is not None:
                features.append(stream.features())
                rates.append(np.mean(stream.truth, axis=0))
                drives.append(np.mean(stream.drives))
            tick += 1
    return np.asarray(features), np.asarray(rates), np.asarray(drives)


def pool_rates(rates):
    x = np.asarray(rates)
    return np.stack((x[..., :48].mean(axis=-1), x[..., 48:].mean(axis=-1)), axis=-1)


def ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


@dataclass
class Decoder:
    path: str
    reconstructor: object
    readout: object
    mean_rates: np.ndarray
    alpha: float
    validation_drive_mse: float

    def infer(self, electrode_features):
        # Public inference boundary: observations only, no sensory error or truth.
        x = np.asarray(electrode_features, dtype=float).reshape(1, -1)
        if x.shape != (1, 48) or not np.isfinite(x).all():
            raise ValueError("Expected 48 finite electrode features")
        if self.path == "constant":
            rates = self.mean_rates.copy()
            command = float(self.readout.predict(pool_rates(rates)[None])[0])
        elif self.path == "direct":
            rates = None
            command = float(self.readout.predict(x)[0])
        elif self.path == "neurons":
            rates = self.reconstructor.predict(x)[0]
            command = float(self.readout.predict(rates[None])[0])
        else:
            pools = self.reconstructor.predict(x)[0]
            rates = np.repeat(pools, 48)
            command = float(self.readout.predict(pools[None])[0])
        return float(np.clip(command, -1, 1)), rates


def fit_decoders(seed):
    train = [calibration_session(seed, s) for s in (0, 1, 2)]
    x, rates, y = (np.concatenate([row[i] for row in train]) for i in range(3))
    vx, vr, vy = calibration_session(seed, 3)
    models = {}
    calibration_metrics = {}
    for path in PATHS:
        best = None
        for alpha in (.1, 1., 10., 100.):
            recon = None
            if path == "direct":
                readout = ridge(alpha).fit(x, y)
            elif path == "neurons":
                recon = ridge(alpha).fit(x, rates)
                # Readout learns from training truth; inference uses estimates.
                readout = ridge(alpha).fit(rates, y)
            else:
                if path == "populations":
                    recon = ridge(alpha).fit(x, pool_rates(rates))
                readout = ridge(alpha).fit(pool_rates(rates), y)
            model = Decoder(path, recon, readout, rates.mean(axis=0), alpha, 0.)
            predictions = [model.infer(f)[0] for f in vx]
            error = float(np.mean((predictions - vy) ** 2))
            model.validation_drive_mse = error
            if best is None or error < best.validation_drive_mse:
                best = model
        models[path] = best
        estimates = [best.infer(f)[1] for f in vx] if path != "direct" else None
        calibration_metrics[path] = {"alpha": best.alpha, "validation_drive_mse": best.validation_drive_mse,
            "validation_rate_mse": None if estimates is None else float(np.mean((estimates-vr)**2))}
    return models, calibration_metrics


class CommandGate:
    """Two consecutive directional decisions, immediate rest, expiring output."""

    def __init__(self, threshold=.18):
        self.threshold = threshold
        self.candidate = 0
        self.count = 0
        self.direction = 0

    def update(self, decoded_drive, now):
        wanted = int(np.sign(decoded_drive)) if abs(decoded_drive) >= self.threshold else 0
        if wanted == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = wanted, 1
        if wanted == 0:
            self.direction = 0
        elif self.count >= 2:
            self.direction = wanted
        else:
            self.direction = 0
        yaw = self.direction * 1.2 * abs(decoded_drive)
        return VelocityCommand(now, 0., 0., yaw, 1., now + PROTOCOL.ttl_seconds, "decoded")


def sensory_input(error, active):
    if not active or abs(error) <= PROTOCOL.sensory_deadband_rad:
        return 0.
    return float(np.clip(error / PROTOCOL.sensory_scale_rad, -1, 1))


def heading(data, trunk):
    rotation = data.xmat[trunk].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


class Episode:
    """One continuous physics/circuit rollout, including target transitions/rest."""

    def __init__(self, model, seed, session, noise=.065, shift=0.,
                 disturbance=None, targets=None, rest_seconds=None):
        self.decoder = model
        self.stream = ContinuousCircuit(seed, session, noise, shift, disturbance)
        self.targets = tuple(PROTOCOL.targets_rad if targets is None else targets)
        self.rest_seconds = PROTOCOL.rest_seconds if rest_seconds is None else rest_seconds
        self.sim, self.data, self.bam = create_simulation(RESEARCH / "src/mjlab_microduck/robot/microduck/scene.xml")
        self.robot = MicroduckPolicy(self.sim, self.data, self.bam, ROOT / "models/velstand.onnx")
        self.gate = CommandGate()
        self.command = None
        self.estimate = None
        self.drive_estimate = 0.
        self.tick = 0
        self.trace = []
        self.decisions = []
        self.wall_start = time.perf_counter()
        self.first_window_wall = None

    def step(self):
        now = self.tick * DT
        warmup = 2.
        segment = int((now - warmup) // PROTOCOL.target_seconds) if now >= warmup else -1
        active = 0 <= segment < len(self.targets)
        target = float(self.targets[segment]) if active else 0.
        actual = heading(self.data, self.robot.trunk)
        error = float(wrap(target - actual))
        sensory = sensory_input(error, active)
        # Two causal 100-Hz samples per 50-Hz robot control step.
        for offset in (0., .01):
            self.stream.sample(sensory, now + offset)
        ready_time = now + DT
        # Existing command drives this interval. The newly completed recording
        # can affect only the following physics interval, never its own past.
        command = effective_command(self.command, now)
        if self.decisions and "wall_observation_to_application_seconds" not in self.decisions[-1]:
            self.decisions[-1]["wall_observation_to_application_seconds"] = time.perf_counter()-self.decisions[-1]["oldest_sample_wall"]
        self.robot.step(command)
        if (self.tick + 1) % DECISION_STEPS == 0 and self.stream.features() is not None:
            start = time.perf_counter()
            self.drive_estimate, self.estimate = self.decoder.infer(self.stream.features())
            self.command = self.gate.update(self.drive_estimate, ready_time)
            compute = time.perf_counter() - start
            wall = time.perf_counter()
            if self.first_window_wall is None:
                self.first_window_wall = wall
            self.decisions.append({"time": ready_time, "oldest_sample": self.stream.times[0],
                "oldest_sample_wall": self.stream.wall_times[0],
                "latest_sample": self.stream.times[-1], "compute_seconds": compute,
                "observation_to_command_sim_seconds": ready_time-self.stream.times[0],
                "rate_mse": None if self.estimate is None else float(np.mean((self.estimate-np.mean(self.stream.truth, axis=0))**2)),
                "constant_rate_mse": float(np.mean((self.decoder.mean_rates-np.mean(self.stream.truth, axis=0))**2))})
        actual = heading(self.data, self.robot.trunk)
        height = float(self.data.xpos[self.robot.trunk, 2])
        up_z = float(self.data.xmat[self.robot.trunk].reshape(3, 3)[2, 2])
        fallen = height < PROTOCOL.fall_height_m or up_z < PROTOCOL.fall_up_z or not np.isfinite(self.data.qpos).all()
        row = {"time": ready_time, "segment": segment if active else -1,
               "rest": now >= warmup + len(self.targets)*PROTOCOL.target_seconds,
               "target": target, "heading": actual, "error": float(wrap(target-actual)),
               "yaw_command": command[2], "height": height, "fallen": bool(fallen),
               "decoded_drive": self.drive_estimate}
        self.trace.append(row)
        self.tick += 1
        return row

    @property
    def seconds(self):
        return 2 + len(self.targets)*PROTOCOL.target_seconds + self.rest_seconds

    def run(self):
        for _ in range(round(self.seconds / DT)):
            self.step()
        return self.metrics()

    def metrics(self):
        tasks = []
        hold = round(PROTOCOL.settle_seconds/DT)
        for segment, target in enumerate(self.targets):
            rows = [r for r in self.trace if r["segment"] == segment]
            if not rows:
                continue
            errors = np.asarray([r["error"] for r in rows])
            inside = (np.abs(errors) <= PROTOCOL.tolerance_rad) & ~np.asarray([r["fallen"] for r in rows])
            # Settling must persist to the end of the target segment, >=1 second.
            bad = np.flatnonzero(~inside)
            index = int(bad[-1]+1) if len(bad) else 0
            settled = len(rows)-index >= hold and not any(r["fallen"] for r in rows)
            initial_sign = np.sign(errors[0])
            overshoot = float(max(0., np.max(-initial_sign*errors)))
            tasks.append({"target_rad": target, "success": bool(settled),
                "settling_seconds": (index+1)*DT if settled else None,
                "overshoot_rad": overshoot, "final_error_rad": float(errors[-1])})
        rest = [r for r in self.trace if r["rest"]]
        moves = np.asarray([abs(r["yaw_command"]) > 1e-6 for r in rest], dtype=bool)
        # Ignore first second of rest (transition washout), report separately.
        washed = moves[round(1/DT):]
        count_events = lambda v: int(np.count_nonzero(v & ~np.r_[False, v[:-1]])) if len(v) else 0
        errors = [d["rate_mse"] for d in self.decisions if d["rate_mse"] is not None]
        durations = [d["observation_to_command_sim_seconds"]+d["compute_seconds"] for d in self.decisions]
        wall_durations = [d["wall_observation_to_application_seconds"] for d in self.decisions
                          if "wall_observation_to_application_seconds" in d]
        return {"tasks": tasks, "target_success_rate": float(np.mean([t["success"] for t in tasks])) if tasks else None,
            "falls": int(any(r["fallen"] for r in self.trace)),
            "false_movement_events": count_events(moves), "rest_seconds": len(rest)*DT,
            "false_movement_events_after_1s_washout": count_events(washed),
            "false_movement_events_per_minute_after_washout": count_events(washed)/max(len(washed)*DT, DT)*60,
            "rest_command_fraction": float(moves.mean()) if len(moves) else None,
            "reconstruction_mse": float(np.mean(errors)) if errors else None,
            "constant_reconstruction_mse": float(np.mean([d["constant_rate_mse"] for d in self.decisions])) if self.decisions else None,
            "observation_to_command_seconds_mean": float(np.mean(durations)) if durations else None,
            "observation_to_command_seconds_max": float(np.max(durations)) if durations else None,
            "wall_observation_to_application_seconds_mean": float(np.mean(wall_durations)) if wall_durations else None,
            "wall_observation_to_application_seconds_max": float(np.max(wall_durations)) if wall_durations else None,
            "latency_definition": "200ms simulated sample acquisition + measured decoder/gate compute; GUI scheduling excluded",
            "wall_latency_definition": "Actual elapsed wall time from oldest recorded sample to command application, including scheduling; accelerated batch wall time is not biological real time",
            "wall_seconds": time.perf_counter()-self.wall_start}


def evaluate(seeds=(42, 1337, 2026), sessions=(4, 5)):
    result = {"protocol": asdict(PROTOCOL), "design": {"train_sessions": [0, 1, 2],
        "validation_session": 3, "test_sessions": list(sessions), "seeds": list(seeds),
        "matching": "Same initial robot state, target sequence, neural/electrode RNG seeds per path; trajectories diverge under feedback.",
        "gate_threshold": .18, "hysteresis_decisions": 2, "window_seconds": .2,
        "decision_interval_seconds": .1, "adaptation": "Deferred pending closed-loop baseline review.",
        "limits": "Toy rate circuit, supervised synthetic targets, short continuous rest; no human EEG claims. No test-based retuning."},
        "calibration": {}, "runs": []}
    OUT.mkdir(exist_ok=True)
    for seed in seeds:
        models, calibration = fit_decoders(seed)
        result["calibration"][str(seed)] = calibration
        for session in sessions:
            for path in PATHS:
                episode = Episode(models[path], seed, session)
                metrics = episode.run()
                result["runs"].append({"seed": seed, "session": session, "path": path, **metrics})
                trace_path = OUT / f"closed-loop-trace-{seed}-{session}-{path}.json"
                trace_path.write_text(json.dumps({"trace": episode.trace, "decisions": episode.decisions}), encoding="utf-8")
                print(f"seed={seed} session={session} {path}: success={metrics['target_success_rate']:.0%} falls={metrics['falls']}", flush=True)
                (OUT / "closed-loop-evaluation.partial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary"] = {}
    for path in PATHS:
        rows = [r for r in result["runs"] if r["path"] == path]
        settled = [t["settling_seconds"] for r in rows for t in r["tasks"] if t["success"]]
        result["summary"][path] = {
            "target_success_rate": float(np.mean([r["target_success_rate"] for r in rows])),
            "successful_targets": sum(t["success"] for r in rows for t in r["tasks"]),
            "total_targets": sum(len(r["tasks"]) for r in rows),
            "episodes_with_fall": sum(r["falls"] for r in rows),
            "mean_settling_seconds_successful_only": float(np.mean(settled)) if settled else None,
            "mean_overshoot_rad": float(np.mean([t["overshoot_rad"] for r in rows for t in r["tasks"]])),
            "mean_reconstruction_mse": None if path == "direct" else float(np.mean([r["reconstruction_mse"] for r in rows])),
            "mean_constant_reconstruction_mse": float(np.mean([r["constant_reconstruction_mse"] for r in rows])),
            "false_events_after_washout": sum(r["false_movement_events_after_1s_washout"] for r in rows),
            "total_rest_seconds": sum(r["rest_seconds"] for r in rows),
            "mean_observation_to_command_seconds": float(np.mean([r["observation_to_command_seconds_mean"] for r in rows]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    parser.add_argument("--sessions", type=int, nargs="+", default=[4, 5])
    args = parser.parse_args()
    if set(args.sessions) & {0, 1, 2, 3}:
        parser.error("Final evaluation sessions must exclude calibration and validation")
    result = evaluate(args.seeds, args.sessions)
    path = OUT / "closed-loop-evaluation.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_results(result)
    print(json.dumps(result["summary"], indent=2))


def plot_results(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    x = np.arange(len(PATHS))
    summary = result['summary']
    axes[0, 0].bar(x, [summary[p]['target_success_rate'] for p in PATHS], color='teal')
    axes[0, 0].set(ylabel='Successful targets / all targets', ylim=(0, 1.1), title='Closed-loop task success')
    for i, path in enumerate(PATHS):
        rows = [r for r in result['runs'] if r['path'] == path]
        settling = [t['settling_seconds'] for r in rows for t in r['tasks'] if t['success']]
        if settling:
            axes[0, 1].scatter(np.full(len(settling), i), settling, alpha=.5)
        mse = [r['reconstruction_mse'] for r in rows if r['reconstruction_mse'] is not None]
        if mse:
            axes[1, 0].scatter(np.full(len(mse), i), mse, color='teal')
        overshoot = [t['overshoot_rad'] for r in rows for t in r['tasks']]
        axes[1, 1].scatter(np.full(len(overshoot), i), overshoot, alpha=.5)
    axes[0, 1].set(ylabel='Seconds (successful targets only)', title='Settling and remaining within ±0.20 rad')
    axes[1, 0].set(ylabel='Mean squared rate error', title='Activity reconstruction (direct: no estimate)', yscale='log')
    axes[1, 1].set(ylabel='Radians', title='Overshoot (constant path stays still)')
    for ax in axes.ravel():
        ax.set_xticks(x, PATHS)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f"Continuous synthetic feedback · {len(result['design']['seeds'])} circuit seeds · {len(result['design']['test_sessions'])} held-out sessions\nSame tasks and initial conditions; no human EEG")
    fig.savefig(OUT/'closed-loop-evaluation.png', dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
