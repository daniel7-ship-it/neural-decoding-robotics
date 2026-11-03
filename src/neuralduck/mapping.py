"""Paired synthetic experiments with known neuron identity across sessions.

Reconstructs window-mean rates, not wiring, spikes, or human brain structure.
Only electrode features enter the learned models at inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .neural import MotorCircuit, VirtualElectrodes


@dataclass(frozen=True)
class Settings:
    electrodes: int = 24
    noise: float = .065
    shift: float = 0.
    silence: str | None = None

    def __post_init__(self):
        if not 1 <= self.electrodes <= 24:
            raise ValueError("electrodes must be between 1 and 24")
        if not np.isfinite([self.noise, self.shift]).all() or self.noise < 0:
            raise ValueError("noise and shift must be finite, noise nonnegative")
        if self.silence not in (None, "left", "right"):
            raise ValueError("silence must be None, left, or right")


@dataclass
class RecordingSet:
    features: np.ndarray
    rates: np.ndarray
    labels: np.ndarray
    recordings: np.ndarray
    sessions: np.ndarray


def channels(count: int):
    # Nested subsets: losing electrodes cannot move the remaining electrodes.
    order = [0, 23, 11, 17, 5, 20, 8, 14, 2, 22, 10, 16,
             4, 19, 7, 13, 1, 21, 9, 15, 3, 18, 6, 12]
    return np.sort(order[:count])


def record_session(seed: int, session: int, settings: Settings, repetitions: int = 12):
    circuit = MotorCircuit(seed=seed)  # Same neuron identities/wiring for all sessions.
    selected = channels(settings.electrodes)
    features, rates_out, labels, recordings = [], [], [], []
    order_rng = np.random.default_rng(seed + session * 100003)
    schedule = order_rng.permutation(np.repeat(np.arange(3), repetitions))
    for index, label in enumerate(schedule):
        trial_seed = seed + session * 100003 + index * 101 + 19
        rng = np.random.default_rng(trial_seed)
        rates = circuit.simulate(int(label), trial_seed, silenced=settings.silence,
                                 drive_scale=float(rng.uniform(.45, 1.4)))
        # Resetting noise and placement seeds pairs intervention/control trials.
        electrodes = VirtualElectrodes(seed=seed + session * 7919,
                                       neuron_seed=seed, noise_sd=settings.noise,
                                       placement_shift=settings.shift)
        electrodes.rng = np.random.default_rng(trial_seed + 43)
        recording = electrodes.record(rates)[:, selected]
        features.append(np.r_[recording.mean(axis=0), recording.std(axis=0)])
        rates_out.append(rates.mean(axis=0))
        labels.append(label)
        recordings.append(recording)
    return RecordingSet(np.asarray(features), np.asarray(rates_out),
                        np.asarray(labels), np.asarray(recordings),
                        np.full(len(labels), session))


def combine(sets):
    return RecordingSet(*(np.concatenate([getattr(s, name) for s in sets])
                          for name in ("features", "rates", "labels", "recordings", "sessions")))


@dataclass
class MappingModels:
    decoder: object
    reconstructor: object
    mean_rates: np.ndarray
    confidence_threshold: float
    ridge_alpha: float
    validation_score: float

    def infer(self, electrode_features: np.ndarray):
        x = np.asarray(electrode_features)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.decoder.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("Expected finite electrode-feature rows matching calibration")
        probs = self.decoder.predict_proba(x)
        predicted = self.decoder.classes_[probs.argmax(axis=1)]
        accepted = probs.max(axis=1) >= self.confidence_threshold
        commands = np.where(accepted, predicted, 0)
        estimated_rates = self.reconstructor.predict(x)
        return commands, estimated_rates, probs, accepted


def fit_models(train: RecordingSet, validation: RecordingSet):
    if set(train.sessions) & set(validation.sessions):
        raise ValueError("Training and validation sessions must be disjoint")
    decoder = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    decoder.fit(train.features, train.labels)
    p = decoder.predict_proba(validation.features)
    best_score, threshold = -1., .5
    for candidate in (.5, .65, .8, .95):
        pred = decoder.classes_[p.argmax(axis=1)]
        gated = np.where(p.max(axis=1) >= candidate, pred, 0)
        score = balanced_accuracy_score(validation.labels, gated)
        if score > best_score:
            best_score, threshold = float(score), candidate
    best_error, best_model, best_alpha = float("inf"), None, None
    for alpha in (.1, 1., 10., 100.):
        recon = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        recon.fit(train.features, train.rates)
        error = np.mean((recon.predict(validation.features) - validation.rates) ** 2)
        if error < best_error:
            best_error, best_model, best_alpha = float(error), recon, alpha
    return MappingModels(decoder, best_model, train.rates.mean(axis=0), threshold,
                         best_alpha, best_score)


def score_models(models: MappingModels, test: RecordingSet):
    t0 = time.perf_counter()
    pred, estimates, probabilities, accepted = models.infer(test.features)
    elapsed = (time.perf_counter() - t0) / len(pred)
    mse = float(np.mean((estimates - test.rates) ** 2))
    baseline = float(np.mean((models.mean_rates - test.rates) ** 2))
    rest = test.labels == 0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test.labels, pred)),
        "confusion_matrix_rest_left_right": confusion_matrix(test.labels, pred, labels=[0, 1, 2]).tolist(),
        "recall_rest_left_right": [float(np.mean(pred[test.labels == i] == i)) for i in range(3)],
        "accepted_fraction": float(accepted.mean()),
        "false_movement_windows_per_60_rest_windows": float(np.mean(pred[rest] != 0) * 60),
        "reconstruction_mse": mse,
        "constant_mean_baseline_mse": baseline,
        "reconstruction_skill_vs_constant": 1 - mse / max(baseline, 1e-12),
        "compute_ms_per_window": elapsed * 1000,
        "observation_seconds_per_decision": 1.,
        "confidence_threshold_validation_selected": models.confidence_threshold,
        "ridge_alpha_validation_selected": models.ridge_alpha,
        "test_windows": int(len(pred)),
    }


CONDITIONS = {
    "control": Settings(),
    "12_electrodes": Settings(electrodes=12),
    "6_electrodes": Settings(electrodes=6),
    "2_electrodes": Settings(electrodes=2),
    "noise_0.5": Settings(noise=.5),
    "noise_1.5": Settings(noise=1.5),
    "shift_0.4": Settings(shift=.4),
    "shift_1.0": Settings(shift=1.),
    "silence_left": Settings(silence="left"),
    "silence_right": Settings(silence="right"),
}


def experiment(seeds=(42, 1337, 2026), repetitions=16):
    rows = []
    for seed in seeds:
        cache = {}
        for name, settings in CONDITIONS.items():
            count = settings.electrodes
            if count not in cache:
                calibration = Settings(electrodes=count)
                train = combine([record_session(seed, s, calibration, repetitions) for s in (0, 1, 2)])
                val = record_session(seed, 3, calibration, repetitions)
                models = fit_models(train, val)
                # Shuffled labels and reconstruction targets share one permutation.
                permutation = np.random.default_rng(seed + 77).permutation(len(train.labels))
                shuffled = RecordingSet(train.features, train.rates[permutation], train.labels[permutation],
                                        train.recordings, train.sessions)
                shuffled_models = fit_models(shuffled, val)
                cache[count] = (models, shuffled_models)
            models, shuffled_models = cache[count]
            test = combine([record_session(seed, s, settings, repetitions) for s in (4, 5)])
            result = score_models(models, test)
            shuffled_result = score_models(shuffled_models, test)
            result.update(seed=int(seed), condition=name, settings=asdict(settings),
                          shuffled_label_balanced_accuracy=shuffled_result["balanced_accuracy"],
                          shuffled_target_reconstruction_mse=shuffled_result["reconstruction_mse"])
            rows.append(result)
        print(f"Completed mapping seed {seed}", flush=True)
    summary = {}
    for name in CONDITIONS:
        selected = [r for r in rows if r["condition"] == name]
        summary[name] = {metric: {"mean": float(np.mean([r[metric] for r in selected])),
                                  "sd_across_seeds": float(np.std([r[metric] for r in selected]))}
                         for metric in ("balanced_accuracy", "reconstruction_mse", "reconstruction_skill_vs_constant",
                                        "false_movement_windows_per_60_rest_windows", "shuffled_label_balanced_accuracy")}
    return {"design": {"seeds": list(seeds), "train_sessions": [0, 1, 2], "validation_session": 3,
                        "test_sessions": [4, 5], "repetitions_per_class_per_session": repetitions,
                        "neuron_identity": "fixed circuit/anatomy within each seed, new electrode jitter and noise per session",
                        "electrode_count": "recalibrate per nested channel subset; noise/shift/silence use frozen control models",
                        "target": "96 window-mean synthetic rates; not temporal spikes or structural connectivity",
                        "limitations": "Toy low-dimensional circuit with strong lateralized cues. Paired windows are not independent subjects; seed SD is not a confidence interval. No live human data."},
            "summary": summary, "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mapping-evaluation.json"))
    parser.add_argument("--repetitions", type=int, default=16)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("repetitions must be >= 2")
    result = experiment(repetitions=args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_summary(result, args.output.with_suffix(".png"))
    print(f"Saved {args.output}")


def plot_summary(result, path):
    import matplotlib.pyplot as plt
    names = list(result["summary"])
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    summaries = result["summary"]
    axes[0].barh(y, [summaries[n]["balanced_accuracy"]["mean"] for n in names],
                 xerr=[summaries[n]["balanced_accuracy"]["sd_across_seeds"] for n in names], color="teal", alpha=.8)
    axes[0].scatter([summaries[n]["shuffled_label_balanced_accuracy"]["mean"] for n in names], y,
                    marker="x", color="darkorange", label="Shuffled-label decoder")
    axes[0].set(yticks=y, yticklabels=names, xlim=(0, 1.05), xlabel="Held-out balanced accuracy", title="Decoding intentions (mean ± seed SD)")
    axes[0].axvline(1/3, color="gray", linestyle=":", label="Three-class chance")
    axes[0].legend(fontsize=8)
    mse = [np.mean([r["reconstruction_mse"] for r in result["runs"] if r["condition"] == n]) for n in names]
    baseline = [np.mean([r["constant_mean_baseline_mse"] for r in result["runs"] if r["condition"] == n]) for n in names]
    axes[1].barh(y - .16, mse, height=.3, label="Inferred activity", color="teal")
    axes[1].barh(y + .16, baseline, height=.3, label="Constant training-mean baseline", color="lightgray")
    axes[1].set(yticks=y, yticklabels=names, xlabel="Mean squared rate error (lower is better)", title="Reconstructing 96 window-mean rates")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle("Toy synthetic circuit · 3 seeds · new recording sessions · not human EEG", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
