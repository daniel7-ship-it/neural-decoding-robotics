# Frozen-network rest gate experiment — 2026-10-27

Fixed protocol before evaluation: docs/rest-gate-protocol.json. No retraining, seed selection or threshold tuning. Fresh sessions23–26; all three existing networks; identical recordings per comparison. Movement requires probability>=0.9 and rest probability<=0.05, followed by the original two-decision gate. Probabilities remain uncalibrated.

| Initialization | Original/rest-gated false rest events | Original/rest-gated median onset |
|---|---:|---:|
|101|35 /4|170 /460ms|
|202|36 /8|200 /480ms|
|303|35 /7|150 /475ms|

All variants detected160/160 movement episodes per model. Long-rest false events remained zero. Rest command occupancy fell from0.61–0.76% to0.061–0.121%. Median time to first zero command on entering a rest block fell from100ms to70ms. Events include carryover into rest, so these are command errors rather than necessarily new activations. Rest blocks had no failure to stop. All variants met this phase's predeclared joint command criteria; these differ from the earlier raw-classification criteria.

The stricter gate reduces unwanted commands substantially but adds about260–325ms median movement-onset delay. It does not establish robustness to noise changes, new subjects or human EEG. The three initializations share recordings and must not be treated as independent subjects. No robot integration or RL was added. Original model weights and interfaces remain unchanged.

Reproduce: set PYTHONPATH=src and run `.venv\Scripts\python.exe -m neuralduck.rest_gate`. Results and traces: artifacts/rest-gate/evaluation.json and per-session NPZs. Targeted gate behavior test passed. Next: lock a noise/shift challenge panel on fresh sessions beyond26; evaluate this speed/rest tradeoff before integrating the gate with the robot.
