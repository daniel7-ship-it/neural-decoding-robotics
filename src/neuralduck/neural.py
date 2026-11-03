"""Interpretable synthetic circuit, electrode projection and safe decoder."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ("rest", "left", "right")


@dataclass(frozen=True)
class Trial:
    session: int
    label: int
    features: np.ndarray
    activity: np.ndarray
    recordings: np.ndarray


class MotorCircuit:
    """A toy rate circuit with two competing pools and shared inhibition.

    Hidden cue enters the circuit only at stimulus onset. A virtual electrode
    array observes a noisy linear mixture of population rates. Decoder APIs
    accept only electrode samples; labels remain in the experiment harness.
    """

    def __init__(self, n_neurons: int = 96, seed: int = 0):
        if n_neurons < 12 or n_neurons % 2:
            raise ValueError("n_neurons must be an even number >= 12")
        self.n_neurons = n_neurons
        self.rng = np.random.default_rng(seed)
        self.left = np.arange(0, n_neurons // 2)
        self.right = np.arange(n_neurons // 2, n_neurons)
        self.base = self.rng.normal(0.0, 0.04, n_neurons)
        self.recurrent = np.eye(n_neurons, dtype=np.float32) * 0.20
        self.recurrent[:n_neurons//2, n_neurons//2:] = -0.18 / (n_neurons//2)
        self.recurrent[n_neurons//2:, :n_neurons//2] = -0.18 / (n_neurons//2)
        self.recurrent += self.rng.normal(0, 0.006, self.recurrent.shape)

    def simulate(self, label: int, seed: int, seconds: float = 1.0,
                 silenced: str | None = None, drive_scale: float = 1.0) -> np.ndarray:
        """Return rate activity (time, neurons); label is a circuit stimulus."""
        if label not in (0, 1, 2):
            raise ValueError("label must be rest=0, left=1 or right=2")
        rng = np.random.default_rng(seed)
        dt, tau = 0.01, 0.08
        n_steps = int(seconds / dt)
        drive = np.zeros(self.n_neurons, dtype=np.float32)
        if label == 1:
            drive[self.left] = 0.75
        elif label == 2:
            drive[self.right] = 0.75
        drive *= drive_scale
        active = np.ones(self.n_neurons, dtype=np.float32)
        if silenced == "left":
            active[self.left] = 0
        elif silenced == "right":
            active[self.right] = 0
        elif silenced is not None:
            raise ValueError("silenced must be None, 'left', or 'right'")
        rates = np.zeros(self.n_neurons, dtype=np.float32)
        trace = np.empty((n_steps, self.n_neurons), dtype=np.float32)
        for t in range(n_steps):
            noise = rng.normal(0, 0.045, self.n_neurons)
            target = np.maximum(0.0, self.base + drive + self.recurrent @ rates + noise)
            rates += (dt / tau) * (target - rates)
            rates *= active
            trace[t] = rates
        return trace


class VirtualElectrodes:
    """Session-specific electrode placement, volume conduction, and noise."""

    def __init__(self, n_channels: int = 24, seed: int = 0,
                 noise_sd: float = 0.065, placement_shift: float = 0.0,
                 neuron_seed: int | None = None):
        if n_channels < 1 or not np.isfinite([noise_sd, placement_shift]).all() or noise_sd < 0:
            raise ValueError("Need positive channel count, finite shift and nonnegative noise")
        self.n_channels = n_channels
        rng = np.random.default_rng(seed)
        anatomy_rng = np.random.default_rng(seed if neuron_seed is None else neuron_seed)
        # Lateralized population topography is stable across sessions; electrode
        # placement varies mildly, preserving brain coordinates while changing
        # the exact sensor mixture. Fully random mixing can invert L/R labels.
        neuron_x = np.r_[anatomy_rng.normal(-0.58, 0.20, 48),
                         anatomy_rng.normal(0.58, 0.20, 48)]
        electrode_x = np.linspace(-1.0, 1.0, n_channels)
        electrode_x += rng.normal(0, 0.10, n_channels)
        electrode_x += placement_shift
        sigma = 0.48
        self.mixing = np.exp(-0.5 * ((electrode_x[:, None] - neuron_x[None, :]) / sigma) ** 2)
        self.mixing *= rng.uniform(0.8, 1.2, self.mixing.shape)
        self.mixing = self.mixing.astype(np.float32)
        self.mixing /= np.linalg.norm(self.mixing, axis=1, keepdims=True) + 1e-8
        self.gain = rng.uniform(0.75, 1.25, (n_channels, 1)).astype(np.float32)
        self.noise_sd = noise_sd
        self.rng = rng

    def record(self, rates: np.ndarray) -> np.ndarray:
        if rates.shape[-1] != self.mixing.shape[-1]:
            raise ValueError("electrode/circuit neuron dimensions differ")
        return (rates @ self.mixing.T) * self.gain.T + self.rng.normal(
            0, self.noise_sd, (rates.shape[0], self.n_channels)
        ).astype(np.float32)


def make_session_trials(session: int, seed: int, repetitions: int = 24,
                        silenced: str | None = None) -> list[Trial]:
    """Build trials; randomized session mappings force independent-session tests."""
    trials: list[Trial] = []
    circuit = MotorCircuit(seed=seed + session * 1009)
    electrodes = VirtualElectrodes(seed=seed + session * 7919)
    for label in range(3):
        for rep in range(repetitions):
            rates = circuit.simulate(label, seed + session * 65537 + label * 997 + rep,
                                     silenced=silenced)
            observed = electrodes.record(rates)
            # Mean and standard deviation of synthetic rate recordings, not EEG bandpower.
            feature = np.concatenate((observed.mean(axis=0), observed.std(axis=0))).astype(np.float32)
            trials.append(Trial(session, label, feature, rates.mean(axis=0), observed))
    return trials


def fit_decoder(trials: list[Trial]):
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(
        np.stack([t.features for t in trials]), np.array([t.label for t in trials])
    )


def evaluate_decoder(train: list[Trial], test: list[Trial]) -> dict:
    model = fit_decoder(train)
    x = np.stack([t.features for t in test])
    y = np.array([t.label for t in test])
    pred = model.predict(x)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "model": model,
    }


@dataclass(frozen=True)
class VelocityCommand:
    timestamp: float
    vx: float
    vy: float
    yaw_rate: float
    confidence: float
    expires_at: float
    label: str


class SafeCommandDecoder:
    """Maps electrode feature rows to timestamped turn commands with expiry."""

    def __init__(self, model, threshold: float = 0.62, ttl_seconds: float = 0.30):
        self.model = model
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

    def decode(self, electrode_features: np.ndarray, now: float) -> VelocityCommand:
        probs = self.model.predict_proba(np.asarray(electrode_features).reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = LABELS[idx] if confidence >= self.threshold else "rest"
        yaw = 1.2 if label == "left" else -1.2 if label == "right" else 0.0
        return VelocityCommand(now, 0.0, 0.0, yaw, confidence,
                               now + self.ttl_seconds, label)


def effective_command(command: VelocityCommand | None, now: float) -> tuple[float, float, float]:
    if command is None or not np.isfinite(now):
        return (0.0, 0.0, 0.0)
    if not np.isfinite([command.timestamp, command.expires_at, command.vx,
                        command.vy, command.yaw_rate, command.confidence]).all() or \
            now < command.timestamp or now >= command.expires_at:
        return (0.0, 0.0, 0.0)
    return (command.vx, command.vy, command.yaw_rate)


