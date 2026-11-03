# Replay consistency update — 2026-10-28

The replay selector now includes Original, Augmented, Means-only24 and Broad-noise48. The latter load noise-24-101.pt and noise-48-101.pt from models/noise-diagnostic/. The screen shows the checkpoint filename, feature count and fixed initialization seed101. Selection is illustrative and not a final-test-based choice of the best seed.

Dashboard completed-block metrics call the same rest_gate_review.measure function as offline evaluation. The current incomplete block is excluded from completed-episode counts. Transition carryover and new activation/direction-change events during continuing rest are separate shared metrics. There is no washout for these counters: continuing rest begins after the entry decision. Historical result files are preserved.

Model and condition switches reset MuJoCo data, robot pose, velocities, time, previous policy action, command vector, motor target/timestamp, decoder gate and scoring history. Reset invariants were checked after a turning rollout. Both new selectors were clicked and the displayed checkpoint/input count verified. Updated replay remains open on Broad-noise48 / Clean.

Parity verification (`artifacts/verify-replay.py`, PYTHONPATH=src) regenerated the identical session45 clean/noise_hard recordings. Both new checkpoints reproduced saved predicted classes and gated command arrays exactly in streaming single-window inference. Probability values match within1e-6; floating point ordering yields tiny differences, so bitwise probability equality is not claimed. Results: artifacts/replay-parity.json. This verifies decoder/command equivalence, not identical robot trajectories between online and offline metrics (the offline decoder experiment has no robot).

Open Open-Neural-Network-Duck.cmd. Existing dashboards remain available. The updated replay still uses session19 and starts at110s; the parity fixture uses full session45, including the gate's initial state. Shared carryover counter test and robot-reset checks passed; existing full22-test suite also passed during the update. No training occurred.
