# Brain-controlled Microduck research

Working Microduck simulation and synthetic neural-signal replay alongside exploratory recorded human EEG classification. Read [current status](PROJECT_STATUS.md) and [completion evidence](docs/eeg-completion-report.md).

## Open the tools

- **Open-Neural-Network-Duck.cmd**: saved neural models control the duck from prerecorded simulated signals.
- **Open-Mapping-Lab.cmd**: virtual electrodes and synthetic activity reconstruction.
- **Open-Closed-Loop-Lab.cmd**: synthetic heading feedback.
- **Open-Robustness-Lab.cmd**: recording disturbances and adaptation comparison.
- **Open-EEG-Research-Report.cmd**: regenerate/open the human EEG report from saved artifacts, with no training or downloads.

Feature balanced accuracy: band-power LDA57.32%, time CNN56.61%, feature MLPs50â€“51%. No confirmed winner or qualified teacher. Tested50-update personal calibration did not consistently help. Human rest detection, live brain control and whole-brain mapping remain unestablished.

All55 saved feature configurations reproduce their probabilities exactly. The report includes participant-level uncertainty, class metrics, coverage, all curves, seed variation, calibration budgets and separated timing. Latest report rendering and calibration navigation/table layout were inspected in Edge; all five saved figures were inspected directly. Disclosure/download interactions remain unverified after a later monitor-crop capture error. Static links/resources and actual launcher regeneration pass; see docs/report-ui-check.md.

## Reproduce without training

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m neuralduck.feature_inference artifacts/feature-benchmark/4514bab7aa7e4dba837581882bfe6be4
.\.venv\Scripts\python.exe -m neuralduck.completion_report
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Verification uses existing source EDFs in the saved manifest, with no downloader. Report regeneration reads saved outputs only: artifacts/consolidated-report/report.html.

## Saved inference

```python
from pathlib import Path
from neuralduck.feature_inference import Decoder
folder = Path('artifacts/feature-benchmark/4514bab7aa7e4dba837581882bfe6be4')
decoder = Decoder(folder, 2, 'csp', 'neural', 101)
p = decoder.predict(x, decoder.configuration['channels'], 160)
# x: preprocessed volts, shape (trials, 64 channels, 320 samples).
# Existing imagery EDF through the shared causal preprocessing path:
p, labels, quality, manifest, cache_hit = decoder.predict_recording(
    existing_edf_path, 2, 12, Path('data/inference-cache'))
```

Supported: time/bandpower/csp neural, bandpower/csp lda. Inference receives only electrodes; labels returned by source processing are for scoring. Artifact hashes and saved training settings bind the loader. Only trusted local artifacts should be loaded. Supplemental legacy provenance cannot recover unrecorded historical environment or statistical training times.

## Research history and next action

See [synthetic training](docs/neural-network-report.md), [noise/replay](docs/neural-replay-noise-report.md), [replay reliability](docs/replay-reliability-report.md), [feature history](docs/feature-benchmark-report.md), [calibration history](docs/calibration-report.md) and [later completion evidence](docs/eeg-completion-report.md).

All subjects1â€“22 are previously inspected. Synthetic97â€“98% accuracy does not transfer to human EEG. No new training, subjects, hardware or locomotion changes in this task. RL stays deferred after synthetic adaptation increased unwanted rest events22â†’323. Next finish the remaining disclosure/download interaction checks when capture works; later confirmation may compare band-power LDA/time CNN under a locked protocol. Earlier current-document snapshots are preserved under docs/history/before-completion-2026-11-02/.

## Provenance and licenses


Official sources: https://github.com/pollen-robotics/microduck and https://github.com/pollen-robotics/microduck_rl . Upstream files were preserved. Our Windows adapter follows the upstream unified observation layout and pose/action conventions. BAM is pinned to commit 57d13ead53206a6bf0db3d66f86506ae8c2ce01a; voltage sag conversion follows that revision's current-based resistance API.

Policy: https://huggingface.co/pollen-robotics/microduck-policies/tree/v8 . Exact revision, hash, and Apache-2.0 card metadata are saved in models/provenance.json; manifest.json records training lineage. Upstream software is Apache-2.0; Microduck simulation model assets have separate CC-BY-NC-SA-4.0 terms in the training repo's LICENSE-MODELS.md. Dataset source/terms: https://physionet.org/content/eegmmidb/1.0.0/ . Keep upstream notices; no redistribution was performed.

Models are saved in models/; results, traces and screenshots are in artifacts/. Only load our locally generated joblib files, since joblib is not a safe format for untrusted models.

