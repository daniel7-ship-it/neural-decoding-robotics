# Project status — 2026-11-02

Synthetic neural-signal Microduck replay works. Human EEG classification remains exploratory; no reliable live human control, rest detection or brain mapping is established.

The EEG feature milestone now has independent artifact-bound loading, exact saved-probability verification for55 configurations, retrospective participant-level paired analyses, separated computation timing and a consolidated report. Open Open-EEG-Research-Report.cmd. Report regeneration uses saved outputs without training or downloads.

Feature balanced accuracy: band-power LDA57.32%, time CNN56.61%, band-power MLP51.25%, CSP MLP50.65%, CSP/LDA48.93%. Five previously inspected participants2–6. Features did not improve neural averages; no teacher qualified and distillation remains untested. No confirmed winner.

Calibration: previously inspected participants15–22, three seeds, budgets4/8/12. Tested50-update fine-tuning at12 trials averaged -0.89 percentage points versus frozen; uncertainty includes zero (3 improved/5 worsened). Extra shuffled final runs and limited development selection are documented. All subjects1–22 have been inspected.

Visual QA update: the latest report rendered in Edge; calibration navigation/table layout and all five saved figure files were inspected. A later capture failed twice with a window-crop/monitor error, so disclosure/download interactions remain unverified. The earlier URL safety stop was not bypassed. Static resource/anchor checks and launcher regeneration pass. See docs/report-ui-check.md.

See docs/eeg-completion-report.md and docs/research-completion-checklist.md. No new model training, participants, RL or locomotion changes. RL remains deferred after prior synthetic adaptation increased unwanted rest events22→323. Next complete the remaining disclosure/download UI checks when capture works; a later locked comparison may retain band-power LDA and time CNN.

Earlier status/checkpoint/README snapshots are preserved in docs/history/before-completion-2026-11-02/. Historical reports remain unchanged.
