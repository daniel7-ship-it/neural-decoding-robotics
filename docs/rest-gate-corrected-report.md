# Corrected rest-gate and disturbance evaluation — 2026-10-27

The frozen MLP checkpoints and original/strict gate thresholds were held fixed. Saved sessions23–26 were rescored from recorded commands without retraining. The locked disturbance protocol (`docs/rest-gate-disturbance-protocol.json`) then used new sessions27–28, fixed anatomy42, and matched signals for original versus strict gates. Sudden position shift is +0.65 electrode coordinates; sudden added noise has SD0.5 after2s.

A movement episode now reports any correct hit, a correct hit within500ms, and correct-command occupancy of at least80%. The 95th percentile of detected onset delays is separate from missed episodes. A stop transition is counted only when a movement block enters rest; it must reach zero within500ms and then remain zero for the rest of that block. False rest events retain the prior command-start/change definition.

| Stage / condition | Gate | False rest events | Timely movements | Sustained movements | Sustained stops | p95 detected onset |
|---|---|---:|---:|---:|---:|---:|
| rescore / control | original | 106 | 471/480 | 457/480 | 159/159 | 460 ms |
| rescore / control | strict | 19 | 326/480 | 292/480 | 159/159 | 600 ms |
| fresh / control | original | 67 | 233/240 | 230/240 | 83/84 | 470 ms |
| fresh / control | strict | 10 | 171/240 | 170/240 | 84/84 | 620 ms |
| fresh / shift_sudden | original | 75 | 190/240 | 169/240 | 45/84 | 540 ms |
| fresh / shift_sudden | strict | 8 | 80/240 | 71/240 | 84/84 | 1135 ms |
| fresh / noise_sudden | original | 123 | 123/240 | 37/240 | 0/84 | 140 ms |
| fresh / noise_sudden | strict | 197 | 83/240 | 0/240 | 2/84 | 100 ms |

The strict gate helps on nominal recordings but exceeds the500ms deadline for many movement episodes. Under added noise it increases rest events and almost eliminates sustained movement. This fails the joint reliability goal. Do not connect this gate to the robot or use it as an RL starting policy. Results are deterministic toy-simulation outcomes over two fresh sessions and one anatomy, not evidence of human EEG generalization. The three weight seeds share recordings and are not independent subjects.

Reproduce with `PYTHONPATH=src` and `.venv\Scripts\python.exe -m neuralduck.rest_gate_review`. Full per-session data and command traces are in `artifacts/rest-gate-corrected/`. The protocol was saved before the fresh challenge. The original session23–26 result files remain intact. Next experiment: develop noise-resistant abstention on separate development recordings, measure timely/sustained movement and rest jointly, and lock a new evaluation split before testing. No reinforcement learning or robot integration was performed.
