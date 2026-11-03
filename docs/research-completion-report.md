# Research completion report — milestone still open

Actual pipeline now uses configuration/raw-file fingerprints for cache filenames, unique run folders and shared train-only normalization. Revised reference diagnostics executed to completion with trained checkpoint probability reload assertions. Output: artifacts/reference-checks/b03741904023446380535fd0004df5d6. Microvolt output generated directly.

Locked fresh-subject replication: subjects12–14, train2–6, validation7–8. Protocol docs/human-eeg-replication-protocol.json. Balanced accuracy percentages: {'101': 56.82, '202': 53.85, '303': 55.63, 'csp_lda': 48.88}. Output artifacts\human-eeg-runs\9770368c070d4f7585ccb9c558a0a890. This is independent replication, not the requested primary-change follow-up. Replay criteria not met. Binary left/right model does not validate rest.

35 tests passed (artifacts/completion-regression.log). Dashboard near-end launch/completion smoke exited successfully (artifacts/completion-dashboard-smoke.log); interactive controls were not visually reverified.

Outstanding: repeated shuffle controls, primary-change follow-up, source/channel/trial metadata completeness, diagnostic plots, interactive UI verification, expanded independent scoring parity, dataset/license documentation. These remain pending, with no external blocker established.

Reproduce with PYTHONPATH=src; python -m neuralduck.reference_checks; set EEG_PROTOCOL=docs/human-eeg-replication-protocol.json and python -m neuralduck.human_eeg_benchmark. Every execution creates a unique output folder.


2026-10-29 verification continuation: benchmark cache identity now uses actual channels, sample rate, units, reference, filter, window, annotation, quality thresholds, and EEG preprocessing source hash. New benchmark checkpoints save selected epoch/config/class mapping and compare reloaded full probabilities on electrode recordings. Cache dependency test passed. Native dashboard inspected at completion; Original, Augmented, Means-only 24, Broad-noise 48, Clean/Noisy, and Restart controls were clicked and visible selection/trace changes confirmed. Full interactive reset trajectory and independent dashboard/scorer parity still need checks.
