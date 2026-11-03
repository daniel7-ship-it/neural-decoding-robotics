"""Optional PhysioNet EEG Motor Movement/Imagery left-vs-right replay."""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_physionet(subject: int = 1, data_dir: Path | None = None):
    """Download/use subject 1 runs 4, 8, 12; return run-separated raw data.

    PhysioNet's event codes 2/3 in these runs refer to left/right fist motor
    imagery. Run 4 trains, run 8 validates, and run 12 is held out for testing.
    The run split happens before any epoch/window extraction.
    """
    files = eegbci.load_data(subject, runs=[4, 8, 12], path=str(data_dir) if data_dir else None,
                             update_path=False, verbose=False)
    outputs = {}
    for run, filename in zip((4, 8, 12), files):
        raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)
        eegbci.standardize(raw)
        raw.pick("eeg")
        raw.set_eeg_reference("average", projection=False, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        labels = {code: name.lower() for name, code in event_id.items()}
        # EEGBCI annotation IDs are normalized by events_from_annotations; map
        # descriptions by annotation text so the semantics stay explicit.
        mapping = {}
        for name, code in event_id.items():
            low = name.lower()
            if "left" in low:
                mapping[code] = 1
            elif "right" in low:
                mapping[code] = 2
        # Standard run annotations use T1/T2; for runs 4/8/12 T1=left,T2=right.
        if not mapping:
            for description, code in event_id.items():
                if description == "T1":
                    mapping[code] = 1
                elif description == "T2":
                    mapping[code] = 2
        if set(mapping.values()) != {1, 2}:
            raise ValueError(f"Unrecognized imagery annotations: {event_id}")
        outputs[run] = (raw, events, mapping)
    return outputs


def causal_bandpass(raw, low: float = 8.0, high: float = 30.0):
    """Causal one-pass SOS filter; no zero-phase future-sample leakage."""
    sfreq = float(raw.info["sfreq"])
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    x = raw.get_data().astype(np.float64)
    # Continuous recording is filtered before window extraction. Initial filter
    # transients are dropped by the one-second post-cue crop.
    return sosfilt(sos, x, axis=-1).astype(np.float32), sfreq


def windows_for_run(raw, events, mapping, run: int, window_s: float = 2.0,
                    offset_s: float = 1.0):
    x, sfreq = causal_bandpass(raw)
    n = int(window_s * sfreq)
    X, y = [], []
    for sample, _, code in events:
        if int(code) not in mapping:
            continue
        start = int(sample + offset_s * sfreq)
        end = start + n
        if end <= x.shape[1]:
            X.append(x[:, start:end])
            y.append(mapping[int(code)])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def evaluate_physionet(data_dir: Path | None = None) -> dict:
    runs = load_physionet(data_dir=data_dir)
    split = {r: windows_for_run(*runs[r], r) for r in runs}
    Xtr, ytr = split[4]
    Xva, yva = split[8]
    Xte, yte = split[12]
    if not len(Xtr) or not len(Xva) or not len(Xte):
        raise RuntimeError("PhysioNet run annotations did not yield left/right trials")
    csp = CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)
    ztr = csp.fit_transform(Xtr, ytr)
    zva = csp.transform(Xva)
    zte = csp.transform(Xte)
    # Fit scaling/LDA on train only. Validation is reported, never used to fit.
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    clf.fit(ztr, ytr)
    import joblib
    output_dir = Path(__file__).resolve().parents[2] / "models"
    output_dir.mkdir(exist_ok=True)
    joblib.dump({"csp": csp, "classifier": clf, "channels": runs[4][0].ch_names,
                 "sampling_hz": runs[4][0].info["sfreq"], "filter": [8, 30],
                 "window_seconds": 2, "cue_offset_seconds": 1,
                 "subject": 1, "training_run": 4}, output_dir / "eeg-baseline.joblib")
    pv, pt = clf.predict(zva), clf.predict(zte)
    yshuffle = np.random.default_rng(42).permutation(ytr)
    baseline_csp = CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)
    zb = baseline_csp.fit_transform(Xtr, yshuffle)
    baseline = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    baseline.fit(zb, yshuffle)
    pbase = baseline.predict(baseline_csp.transform(Xte))
    return {
        "dataset": "PhysioNet EEG Motor Movement/Imagery, subject 1, runs 4/8/12",
        "channels": int(len(runs[4][0].ch_names)),
        "sampling_hz": float(runs[4][0].info["sfreq"]),
        "n_trials": {str(r): int(len(split[r][1])) for r in split},
        "split": {"train": "run 4", "validation": "run 8", "test": "run 12"},
        "validation_balanced_accuracy": float(balanced_accuracy_score(yva, pv)),
        "test_balanced_accuracy": float(balanced_accuracy_score(yte, pt)),
        "test_confusion_matrix_left_right": confusion_matrix(yte, pt, labels=[1, 2]).tolist(),
        "shuffled_train_label_test_balanced_accuracy": float(balanced_accuracy_score(yte, pbase)),
        "test_labels": yte.tolist(),
        "test_predictions": pt.tolist(),
        "test_probabilities": clf.predict_proba(zte).tolist(),
        "interpretation": "Exploratory single-subject, 15-test-trial result; not evidence of reliable live BCI. Shuffled baseline is one draw only.",
    }
