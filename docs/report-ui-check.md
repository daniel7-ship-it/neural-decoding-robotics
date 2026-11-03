# Report QA follow-up — 2026-11-02

## Observed through approved Computer Use

Selected returned Edge window67090, title Microduck EEG research report. Current URL was the local artifacts/consolidated-report/report.html file. Refreshed to the latest generated report; verified the new Source-through-prediction batch-ms column and current values, report headings, readable top text and table wrapping in approximately1034-pixel-wide window.

Clicked Calibration navigation link through its observed accessibility element. It navigated to the calibration heading and displayed the paired and class-recall tables. Text and participant differences wrapped inside the content area without page-wide horizontal overflow. This is actual browser rendering/navigation evidence, not an inferred static check.

The earlier URL-confidence safety stop was not bypassed. The later user-authorized turn observed the returned browser state normally. A subsequent capture failed: "encode latest capture frame failed: window crop is outside captured monitor". Refreshed window selection, activated and retried capture once; the same error recurred. Stopped desktop interaction afterward. Remaining disclosure/download click checks are NOT claimed verified.

## Saved-figure inspection and fixes

Inspected feature-summary.png, paired-calibration.png, all-curves.png, calibration-sequences.png and confusion.png using the local image viewer. Found and fixed:

- Class-recall bars previously cycled colors across decoder groups while the legend implied consistent Left/Right colors. They now use explicit blue/orange throughout; regenerated figure re-inspected.
- Confusion-matrix dark cells now use white text for readable contrast.
- Calibration budget-curve legend moved below the axes to avoid obscuring the CSP/LDA curve.

All curves label participant/representation/seed and distinguish solid training from dashed validation. Calibration sequences identify the illustrative seed/budget and quality coverage. No predictions, metrics, weights, or scientific comparisons changed.

## Launcher and static verification

Open-EEG-Research-Report.cmd uses setlocal, checks for the existing local environment, quotes the interpreter, propagates regeneration failures and supports --no-open. The report test executes this actual launcher with --no-open, checks saved model hashes unchanged, all local image/download targets and internal anchor IDs, and parses all JSON evidence files. It passed in7.882 seconds; log artifacts/completion-report-test.log.

No browser-open claim is made for the --no-open test. The report was already opened and inspected separately through Computer Use. Browser disclosure/download interactions remain the specific pending limitation.
