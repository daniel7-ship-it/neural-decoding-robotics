# Real EEG milestone audit — in progress

The prior dashboard scored applied commands only at neural decision times. Added applied_scoring.py, wired dashboard scoring to every physics frame, and added a hand-calculated test for between-decision activation and partial movement exclusion. Historical results are preserved. Full regression, expanded interval edge cases, saved-trace rescoring, and UI verification remain required before acceptance.

Existing human EEG implementation: src/neuralduck/eeg.py, PhysioNet subject 1 runs 4/8/12, causal 8–30 Hz filtering, 2-second left/right imagery epochs beginning 1 second after cue, train-only CSP (6 components) and shrinkage LDA. Existing artifact models: none identified in this audit. Existing model models/eeg-baseline.joblib must be preserved. Subject 1 test run 12 was previously inspected and cannot serve as untouched confirmation.

Environment: PyTorch 2.14.1+cpu, CUDA unavailable to this installation. No environment changes performed.

The new real-EEG benchmark, dataset comparison, locked protocol, artifact comparison, neural training, and held-out evaluation have NOT been completed. No new human EEG claims are justified.
