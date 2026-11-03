# Recording variation experiment — 2026-10-27

The protocol in `docs/augmentation-protocol.json` was saved before training. One fixed simulated anatomy; train sessions29–34, validation35–36, untouched final37–38. Every session has120 seconds of continuous rest. Each train/validation session was recorded under clean, added noise SD0.15, electrode shift+0.25, and right-channel gain0.75. Final sessions also tested harder SD0.5, shift+0.65 and gain0.25. All paired models saw the same held-out electrode features per condition. Label schedules and anatomy were matched across conditions; electrode recordings alone entered inference.

The same 48→32→3 network (1,667 parameters) was trained from three fixed initializations, selected only by validation loss, and saved alongside preprocessing. Frozen original networks and LDA/ridge trained on the same augmented recordings were compared. All original/strict gate thresholds remained fixed. The strict gate requires movement probability>=.9 and rest probability<=.05 before the existing two-decision gate. Ridge retains its signed-drive threshold; its one-hot class encoding means that the strict gate cannot alter it. This is a limitation of the gate comparison.

Scoring now defines a successful stop as zero command reached within500ms and maintained through the rest block. “80% occupancy” is a fraction of correct command decisions, not continuous uninterrupted success; interruptions and wrong-direction samples are retained separately. A missed episode is separate from onset delay; median and p95 are based on detected episodes. Rest-event counts include carryover at every rest transition. The criterion of <=0.5 false events/minute applies to all rest including transitions and is severe under this definition.

| Condition | Model/gate | Timely movement | >=80% correct occupancy | Timely sustained stops | Rest occupancy | Rest events |
|---|---|---:|---:|---:|---:|---:|
| clean | frozen-mlp, original | 234/240 | 234/240 | 75/75 | 0.72% | 57 |
| clean | augmented-mlp, original | 240/240 | 233/240 | 74/75 | 0.78% | 61 |
| clean | augmented-mlp, strict | 137/240 | 162/240 | 75/75 | 0.15% | 15 |
| clean | augmented-lda, original | 76/80 | 76/80 | 25/25 | 1.15% | 23 |
| clean | augmented-ridge, original | 80/80 | 78/80 | 25/25 | 1.69% | 25 |
| noise_train | frozen-mlp, original | 181/240 | 87/240 | 0/75 | 91.53% | 513 |
| noise_train | augmented-mlp, original | 239/240 | 230/240 | 75/75 | 0.94% | 66 |
| noise_train | augmented-mlp, strict | 138/240 | 152/240 | 75/75 | 0.15% | 15 |
| noise_train | augmented-lda, original | 76/80 | 76/80 | 25/25 | 1.15% | 23 |
| noise_train | augmented-ridge, original | 80/80 | 78/80 | 25/25 | 1.78% | 25 |
| shift_train | frozen-mlp, original | 216/240 | 216/240 | 66/75 | 0.95% | 59 |
| shift_train | augmented-mlp, original | 240/240 | 233/240 | 75/75 | 0.68% | 64 |
| shift_train | augmented-mlp, strict | 136/240 | 156/240 | 75/75 | 0.17% | 17 |
| shift_train | augmented-lda, original | 77/80 | 77/80 | 25/25 | 1.30% | 24 |
| shift_train | augmented-ridge, original | 80/80 | 78/80 | 25/25 | 1.87% | 25 |
| gain_train | frozen-mlp, original | 238/240 | 231/240 | 75/75 | 0.87% | 60 |
| gain_train | augmented-mlp, original | 238/240 | 233/240 | 74/75 | 0.63% | 57 |
| gain_train | augmented-mlp, strict | 137/240 | 167/240 | 75/75 | 0.16% | 16 |
| gain_train | augmented-lda, original | 74/80 | 73/80 | 25/25 | 1.09% | 22 |
| gain_train | augmented-ridge, original | 80/80 | 77/80 | 25/25 | 1.66% | 25 |
| noise_hard | frozen-mlp, original | 140/240 | 38/240 | 0/75 | 98.19% | 114 |
| noise_hard | augmented-mlp, original | 149/240 | 69/240 | 2/75 | 52.03% | 1249 |
| noise_hard | augmented-mlp, strict | 34/240 | 1/240 | 51/75 | 5.63% | 314 |
| noise_hard | augmented-lda, original | 75/80 | 74/80 | 25/25 | 1.18% | 23 |
| noise_hard | augmented-ridge, original | 80/80 | 78/80 | 25/25 | 1.78% | 26 |
| shift_hard | frozen-mlp, original | 181/240 | 172/240 | 55/75 | 0.81% | 60 |
| shift_hard | augmented-mlp, original | 237/240 | 231/240 | 75/75 | 0.96% | 60 |
| shift_hard | augmented-mlp, strict | 116/240 | 121/240 | 75/75 | 0.17% | 17 |
| shift_hard | augmented-lda, original | 74/80 | 74/80 | 25/25 | 1.30% | 23 |
| shift_hard | augmented-ridge, original | 76/80 | 75/80 | 25/25 | 1.84% | 25 |
| gain_hard | frozen-mlp, original | 162/240 | 118/240 | 63/75 | 0.84% | 58 |
| gain_hard | augmented-mlp, original | 193/240 | 175/240 | 75/75 | 0.51% | 45 |
| gain_hard | augmented-mlp, strict | 121/240 | 109/240 | 75/75 | 0.15% | 15 |
| gain_hard | augmented-lda, original | 58/80 | 57/80 | 25/25 | 0.94% | 21 |
| gain_hard | augmented-ridge, original | 77/80 | 75/80 | 25/25 | 1.36% | 24 |

Counts for MLP rows combine three weight seeds over two sessions (240 movement episodes,75 movement-to-rest transitions); LDA/ridge rows have one model over two sessions (80/25). Shared recordings mean seeds are not independent subjects. Full per-seed confusion matrices, class recalls, missed episodes, wrong-direction samples, interruptions, onset-delay tails, rest seconds, and all command traces are in `artifacts/augmentation/evaluation.json` and NPZ files. The always-rest control has zero rest events but0 timely movements and fails the movement requirement.

**Answer:** Recording variation greatly improved the MLP at conditions included in training, notably SD0.15 noise. It did not make the MLP reliable at SD0.5 noise: original-gate augmented MLPs reached149/240 timely movements,69/240 occupancy passes,2/75 timely sustained stops, and52% rest command occupancy. Strict gating reduced rest occupancy to5.6% but reached34/240 timely movements and1/240 occupancy passes. Shift extrapolation improved, while severe gain reduction remained mixed. Clean balanced accuracy changed little (frozen97.1–98.3%, augmented96.8–97.6%); original-gate clean timely movement rose234/240 to240/240 but rest events57 to61 and occupancy.7% to.8%. The strict gate remained too slow (137/240 clean timely movements).

Augmented statistical baselines performed strongly at hard noise. LDA reached75/80 timely movements,74/80 occupancy passes,25/25 timely sustained stops, 1.2% rest occupancy; ridge80/80,78/80,25/25,1.8% respectively. Thus the MLP does **not** add demonstrated robustness value over these baselines. Neither baseline meets the <=1% rest occupancy target at hard noise, and transition-inclusive false-event rates also exceed the locked <=0.5/minute criterion. No model meets the full joint acceptance criteria. These are deterministic results from one toy anatomy and two final sessions; no human EEG or across-subject claim follows.

Reproduce with PowerShell from project root: set `PYTHONPATH=src`, then run `.venv\Scripts\python.exe -m neuralduck.augmentation`. This reruns the locked CPU experiment and writes new-model checkpoints to `models/augmentation/`, losses and metrics to `artifacts/augmentation/`. The prior rescore is `artifacts/augmentation/rescored-prior.json`. Original checkpoints remain under `models/neural-network/`.

Recommended next step: investigate why the MLP extrapolates poorly under high noise, compare calibration and input normalization on development sessions, and define a realistic joint threshold that distinguishes rest-transition carryover from sustained false commands. Use fresh sessions beyond38 for any revised method. RL and robot integration remain deferred.
