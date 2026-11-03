# EEG completion evidence — 2026-11-02

All non-browser acceptance work completed. Follow-up browser rendering and calibration navigation/table layout were inspected; remaining disclosure/download interaction checks are pending after a monitor-crop capture error. See report-ui-check.md. This is not full interaction acceptance. No models retrained, no new participants, no weights/predictions overwritten.

## What changed

- src/neuralduck/feature_inference.py: five artifact-bound loaders, explicit electrode/source interfaces, source hashes, input checks, legacy supplement, independent verification and separated timings.
- src/neuralduck/feature_benchmark.py: extracted train-only CSP/scaling boundaries used by the benchmark and tests; training was not executed.
- src/neuralduck/completion_report.py: saved-output analysis and readable HTML/figures, participant bootstrap, class and coverage summaries, every neural curve and separate calibration analysis.
- tests/test_feature_inference.py and tests/test_completion_report.py: meaningful inference/integrity/fitting/aggregation/report tests.
- README.md, PROJECT_STATUS.md, CHECKPOINT.md and current checklist reconciled. Older versions archived under docs/history/before-completion-2026-11-02/. Existing launcher preserved.

## Verification and tests

- Fresh standalone process: all55 model configurations reproduced historical probabilities exactly; max error0.0, tolerance1e-6. Predictions and annotation labels matched. Saved normalization audited against train run4. Evidence: artifacts/completion-verification.log and artifacts/feature-benchmark/4514bab7aa7e4dba837581882bfe6be4/verification-complete.json.
- Six new inference tests passed: artifacts/completion-targeted-tests.log.
- Full regression:45 tests passed in25.660 seconds; artifacts/completion-regression-tests.log. Includes real fixed-locomotion forward/left/right checks.
- Final report-only regression after last table additions: artifacts/completion-report-test.log. Regeneration verifies all report resource/anchor targets and checks model hashes before/after.

## Requirement-to-evidence mapping

| Requirement | Evidence / status |
|---|---|
| Independently load every decoder | Decoder API; actual55-row verification; persisted CSP and LDA/scaler loading |
| Artifact-bound channels/rate/windows/classes | inference-contract.json + training manifest; corruption/missing/config/input tests |
| Legacy provenance | Retrospective supplement hashes current artifacts; original weights preserved; unrecoverable environment/time explicitly listed |
| Source versus window boundary | Decoder.predict_recording / Decoder.predict; README usage |
| Causal shared source processing | Existing eeg_pipeline.recording; source-through-prediction path; pipeline regression |
| Labels unavailable to inference | predict signature accepts only electrodes/channels/rate; no fitting calls permitted in inference tests |
| Train-only fitting | fit_training_csp/scaling receive only training arrays; spy tests; saved normalization compared against training recordings |
| Feature units/frequency/dimensions | Existing test_bandpower + representation shape tests; all remain in passing regression |
| Every seed/participant reproducible | verification-complete.json covers45 neural +10 statistical configurations |
| Class precision/recall/confusion/frequencies | consolidated-report/classification-details.json, tables/plots; labels0/1 are T1/T2 left/right imagery |
| Participant/seed/training curves/coverage | feature-summary.png, all-curves.png, confusion.png, participant tables, complete JSON |
| Parameter counts/training times | HTML table; neural counts exact; LDA decision coefficient counts stated; unrecorded LDA historical training time explicitly unavailable |
| Correct inference timing | verification-complete.json separates model+normalization, features+model, cold source, warm cache, and actual full source-through-prediction |
| Paired feature uncertainty | paired.json five comparisons; seeds averaged within five people,10,000 bootstrap draws seed712 |
| Calibration budgets and comparisons | nine paired comparisons; every participant/budget/method table, class metrics, coverage/sequence plots |
| Shuffled-run/limited-selection caveats | HTML calibration section; original report preserved |
| Regeneration without training/download | completion_report module and launcher; model-hash-preservation test; no downloader calls |
| Open and inspect browser report/controls | Rendering/calibration navigation and table layout inspected; all saved figures inspected. Remaining disclosure/download interactions blocked by subsequent monitor-crop capture error; details in report-ui-check.md |
| Current documentation | README, PROJECT_STATUS, checkpoint and this checklist/report |

## Measured paired results

Differences below are percentage points; exploratory participant-bootstrap95% intervals. Five feature participants and eight calibration participants are small samples. Seeds/trials are not treated as independent people. These analyses are retrospective and not multiple-comparison adjusted.

- feature bandpower neural minus bandpower lda: -6.07 pp [-13.81, +1.67].
- feature csp neural minus csp lda: +1.73 pp [-11.55, +14.64].
- feature bandpower neural minus time neural: -5.36 pp [-12.56, +0.71].
- feature csp neural minus time neural: -5.95 pp [-9.46, -2.32].
- feature bandpower lda minus time neural: +0.71 pp [-6.67, +8.10].
- calibration fine_tuned minus frozen budget4: -1.56 pp [-10.31, +7.66].
- calibration fine_tuned minus subject_specific budget4: +3.98 pp [+0.19, +7.70].
- calibration fine_tuned minus csp_lda budget4: +4.84 pp [-1.86, +12.17].
- calibration fine_tuned minus frozen budget8: -6.14 pp [-15.40, +3.39].
- calibration fine_tuned minus subject_specific budget8: -2.19 pp [-5.88, +2.08].
- calibration fine_tuned minus csp_lda budget8: +1.49 pp [-5.58, +10.04].
- calibration fine_tuned minus frozen budget12: -0.89 pp [-5.28, +4.35].
- calibration fine_tuned minus subject_specific budget12: +1.49 pp [-2.64, +5.39].
- calibration fine_tuned minus csp_lda budget12: +6.85 pp [+0.74, +13.54].

## Interpretation

Established features did not improve the tested neural-network averages. Band-power MLP was6.07 points below band-power LDA, with uncertainty including zero; CSP MLP was1.73 points above CSP/LDA, also uncertain. Band-power LDA was0.71 points above time CNN, interval[-6.67,+8.10]; superiority is not established. CSP MLP underperformed time CNN in this exploratory paired analysis.

At12 calibration trials, the tested50-update procedure averaged-0.89 points versus frozen, interval[-5.28,+4.35]. It improved versus the weak tiny-calibration CSP/LDA baseline by6.85 points, but that does not establish benefit over the stronger frozen CNN. No teacher qualified; distillation remains untested. Band-power LDA is a simple candidate for later locked comparison against the fixed time CNN, not a confirmed winner.

## Timing methodology and limits

CPU, Torch threads2,3 warm-ups and10 measured inference batches. Model-only boundary includes normalization and tensor conversion but excludes feature extraction. Cold source processing uses an empty processed cache; warm cached loads are separate. Actual source-through-prediction uses3 repetitions with a fresh processed cache each time; OS file cache is uncontrolled. Batch durations are offline wall times, not live acquisition/command latency. Full hardware and numerical timings are in verification-complete.json. Historical neural seconds include training/validation/save/reload; statistical training times were not recorded and were not fabricated.

## Browser blocker

Computer Use returned: "Computer Use has been stopped for this turn because it could not determine the current browser URL on Windows with enough confidence to enforce policy. Stop your work and send a final message noting why Computer Use ended." The root explicitly resumed only independent non-browser work afterward. Browser action was not retried or bypassed. No screenshot, visual inspection or control acceptance is claimed.

## Launch and reproduce

Open Open-EEG-Research-Report.cmd. It invokes the local .venv, sets PYTHONPATH=src, regenerates the report from saved outputs, and opens artifacts/consolidated-report/report.html. Report-only generation: python -m neuralduck.completion_report. Verification: python -m neuralduck.feature_inference artifacts/feature-benchmark/4514bab7aa7e4dba837581882bfe6be4. Tests: python -m unittest discover -s tests -v. Run in project directory with PYTHONPATH=src and the local .venv Python.

No further scientific experiment was started. Remaining milestone action is disclosure/download interaction acceptance when desktop capture works. Older replay/scoring/license-inventory backlog is separate and is not claimed completed by this EEG phase.

## Follow-up visual QA

The user-authorized follow-up inspected the latest report in Edge and all saved figures directly. A real class-recall legend/color mismatch was corrected; confusion-cell contrast and calibration-legend overlap were improved. Launcher now isolates environment with setlocal and supports --no-open; the actual launcher regeneration test passed in7.882 seconds with model hashes unchanged. Rendering/calibration navigation verified, remaining disclosure/download UI checks not completed because capture failed twice with a monitor-crop error. See report-ui-check.md for exact scope and errors. No scientific outputs changed.
