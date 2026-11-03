# Neural-network duck replay and noise diagnostic — 2026-10-28

Open `Open-Neural-Network-Duck.cmd`. The replay feeds saved 200ms electrode windows through saved PyTorch weights and preprocessing, then the fixed two-decision gate. Its decoded commands drive the existing locomotion policy. Labels only update evaluation text/counters. The robot does not feed back into prerecorded signals. Replay begins at110s of historical session19, preserving about10s of rest before movement. Original versus augmented and clean versus added SD0.5 noise controls were clicked and verified in the native window. The rendered screenshot was inspected. Dashboard counts are illustrative: misses refer to completed intention segments, and unwanted rest starts are counted over the replay rather than a confirmatory trial. Clean augmented replay is left open.

The diagnostic protocol was saved before training: sessions39–42 train,43–44 validation,45–46 final; anatomy42. Added noise levels0,0.15,0.5 were all included in training and testing. Therefore SD0.5 is now in-distribution, not a new extrapolation claim. Both new model types use the broader noise set. Means-only24→32→3 has899 parameters;48→32→3 has1667. Each has three fixed initializations, CPU Adam training,40epoch cap and8epoch validation patience. No test tuning or threshold changes. LDA uses the same recordings.

Validation diagnostic under original preprocessing: 99th-percentile absolute normalized feature value3.27 for means and6.14 for standard deviations;1.97% of standard-deviation entries exceed5 versus zero means entries. This supports a distribution-shift hypothesis but is not a causal proof. Means-only and broad48 are both trained on broader noise, so their comparison does not isolate removal of standard-deviation features under original training.

| Condition | Model | Balanced accuracy | Timely movement | >=80% correct occupancy | Timely sustained stops | Rest events |
|---|---|---:|---:|---:|---:|---:|
| clean | frozen-48 | 97.29% | 231/240 | 231 | 72/72 | 51 |
| clean | noise-24 | 96.37% | 240/240 | 236 | 72/72 | 72 |
| clean | noise-48 | 96.85% | 240/240 | 234 | 71/72 | 60 |
| clean | broad-noise-lda | 93.06% | 75/80 | 75 | 24/24 | 20 |
| noise_train | frozen-48 | 30.84% | 168/240 | 79 | 0/72 | 422 |
| noise_train | noise-24 | 96.37% | 240/240 | 236 | 72/72 | 72 |
| noise_train | noise-48 | 96.86% | 240/240 | 234 | 72/72 | 63 |
| noise_train | broad-noise-lda | 93.05% | 76/80 | 75 | 24/24 | 20 |
| noise_hard | frozen-48 | 12.74% | 132/240 | 36 | 0/72 | 108 |
| noise_hard | noise-24 | 96.33% | 240/240 | 236 | 72/72 | 72 |
| noise_hard | noise-48 | 96.74% | 240/240 | 237 | 71/72 | 70 |
| noise_hard | broad-noise-lda | 93.01% | 77/80 | 74 | 24/24 | 21 |

MLP counts combine three initializations sharing two recordings; they are not independent subjects. LDA counts cover two sessions with one model. Full strict/original results, median and p95 delays, interruption and wrong-direction counts, rest duration/occupancy, and per-episode metrics are in evaluation.json. Rest counts include carryover at transitions; long-rest false events are separately retained. The stricter gate is unchanged and is not selected from test results.

Both newly trained models recover noise performance but have small clean balanced-accuracy regressions and more transition-inclusive rest events. No reliable human-EEG or new-anatomy claim follows. The model in the dashboard is the historical original/augmented network; the new diagnostic networks are saved separately for measured comparison. Next: isolate feature effects with a factorial means-versus-full comparison under identical narrow/broad training, and measure transition stopping separately from established-rest activation on new held-out sessions. RL remains deferred.

Reproduce: set PYTHONPATH=src and run `.venv\Scripts\python.exe -m neuralduck.noise_diagnostic`. Launch replay with `Open-Neural-Network-Duck.cmd`. New weights/preprocessing/architecture/protocol: models/noise-diagnostic/*.pt; baseline: lda.joblib; losses/curves/predictions/metrics: artifacts/noise-diagnostic/. Dashboard screenshot: artifacts/neural-replay-dashboard.png.
