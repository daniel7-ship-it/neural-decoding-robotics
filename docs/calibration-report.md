# Personal calibration research report

Real EDF default-loader parity passed for development subject2 runs4/8/12, with cache reuse. Evidence artifacts/calibration-preprocessing-verification.json.

Final protocol docs/calibration-final-protocol.json was saved before subjects15–22 were loaded. Three seeds, budgets4/8/12; pooled weights pretrained2–6 and selected7–8. All-layer Adam updates50 at .001, no target-person validation, run4 calibration/run12 evaluation. Nested first-per-class subsets; quality flags reported without abstention. This is within-person calibration across runs, not days.

All comparisons executed, weights and preprocessing saved, full probability reload assertions passed. Eight participants. At12 calibration trials, mean fine-tuning gain = -0.89 percentage points; participant-bootstrap95% interval [-5.17,+4.35] points. Three participants helped, five harmed. No evidence of consistent calibration benefit.

frozen: mean 56.29% balanced accuracy.
fine_tuned: mean 55.39% balanced accuracy.
subject_specific: mean 53.91% balanced accuracy.
csp_lda: mean 48.55% balanced accuracy.

Artifacts: artifacts\calibration-development\6677c8dddba2493f8149ee31e60b57ec. Report report.html; plot calibration.png; protocol, complete results, checkpoints and manifests alongside. Regenerate using PYTHONPATH=src and python -m neuralduck.calibration_report artifacts\calibration-development\6677c8dddba2493f8149ee31e60b57ec.

Limitations/deviations: one fixed update configuration, no extensive development search; 15-ish evaluation trials/person; CSP component count2 on tiny calibration sets; no rest validation. Shuffled negative-control runs were also inadvertently executed for final participants by the shared runner; they did not affect selection or updates. Do not interpret them as a formal permutation test. Report does not yet include all requested coverage/sequence figures or per-class recall summaries. Final manifest hash exists beside outputs; individual checkpoints embed calibration/evaluation manifests rather than that aggregate hash.

Recommendation: examine development-only calibration loss and feature distributions before selecting another update policy; do not advance to live control or RL.
