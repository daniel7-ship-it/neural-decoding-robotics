# EEG feature comparison — exploratory development results

Five previously studied participants2–6; train run4, validation8, evaluation12. Three neural seeds, 60 updates, minimum validation loss. Same filtered recordings for every method. No new participants.

Mean balanced accuracy: time CNN56.61%; band-power MLP51.25%; CSP MLP50.65%; band-power LDA57.32%; CSP/LDA48.93%. Established features did not improve the tested neural model on average. Band-power LDA is the leading simple candidate, but its small advantage over the CNN is exploratory and does not establish superiority.

No statistical decoder met the predefined teacher rule (validation BA>=80% and both recalls>=75% in every development person). Distillation was therefore omitted. Teacher assistance remains untested, rather than failed.

Sources: https://mne.tools/stable/generated/mne.decoding.CSP.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html . Welch PSD integrated by band produces V²; log power is used as a feature. Train-only CSP and normalization, fixed 8–30Hz preprocessing, 64 channels, 2-second epochs.

Artifacts: artifacts/feature-benchmark/4514bab7aa7e4dba837581882bfe6be4. Includes protocol, manifest, weights, transforms, curves, predictions, class recall, confusion matrices for neural models and quality coverage. Neural checkpoints reloaded and probabilities compared; CSP reload verification currently uses the in-memory transform and needs stronger independent transformer loading verification. Test_bandpower verifies known frequency, shape and amplitude scaling; broader train-only fitting tests remain outstanding.

Reproduce training with PYTHONPATH=src and python -m neuralduck.feature_benchmark. Regenerate figures without training: python -m neuralduck.feature_report artifacts/feature-benchmark/4514bab7aa7e4dba837581882bfe6be4. Figures were opened and inspected. No live EEG/rest claim.

Calibration reporting supplement: classification-details.json and coverage-sequence.png added to artifacts/calibration-development/6677c8dddba2493f8149ee31e60b57ec without retraining. Original files preserved. Complete paired uncertainty against CSP/LDA remains outstanding.

Recommended next comparison: freeze band-power LDA and time CNN configurations, improve provenance/reload verification and quantify paired uncertainty before reserving any new confirmation cohort. Do not expand architecture or add RL.

Limitations: only five development people and few evaluation trials; reporting lacks full paired uncertainty, standalone inference timings and all requested comparison figures. This report records executed evidence, not complete acceptance of every requested deliverable.
