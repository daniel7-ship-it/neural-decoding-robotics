# First supervised neural-network decoder

## Audit and scope

The original electrode classifier (`neural.py`, trained by `demo.py`) is scaled shrinkage LDA, saved as `models/synthetic-decoder.joblib`. The mapping lab (`mapping.py`) trains LDA and ridge reconstruction. Continuous control (`closed_loop.py`) fits standardized ridge models for direct drive, 96-unit reconstruction, and two-population reconstruction in memory. The human EEG pipeline (`eeg.py`) fits causal filtering/CSP/LDA and saves `models/eeg-baseline.joblib`. These are trained statistical models. The official `models/velstand.onnx` is a pretrained locomotion neural network, not our electrode decoder. Previous robustness adaptation selected among six fixed gain/bias corrections; it was not neural-network training or full reinforcement learning.

The new `src/neuralduck/supervised.py` trains a PyTorch MLP alongside all of these. No existing source module, historical model, or result is replaced. This milestone is offline continuous neural decoding, not a new robot dashboard, closed-loop robot integration, EEG experiment, or RL controller.

## Locked design

`docs/neural-network-protocol.json` was written before training. One fixed simulated anatomy (seed42); train sessions11-16, validation17-18, final19-22. Historical sessions0-10 are not used. Each session has 120 seconds of initial continuous rest and 60 randomized blocks (20 rest, 20 left, 20 right), each 1.5-3 seconds. Movement strengths vary from0.3 to1.0. Session duration is approximately255 seconds. Mild session placement changes and native noise are inherited from the existing simulator. Deliberate disturbance robustness is not tested.

At100Hz, trailing20-sample windows produce 24 means plus24 standard deviations. Decisions occur every100ms after the first200ms window. Features never use future samples. Intention labels refer to the current intended command, including transition windows; transition errors are retained. Labels and hidden circuit state never enter `NeuralDecoder.predict_proba`. Complete sessions are assigned before window generation and normalization is fitted only on training sessions. Recordings and supervision are saved separately.

Architecture:48 inputs ->32 ReLU units ->3 logits, 1,667 trainable parameters. Softmax produces uncalibrated class probabilities. Cross entropy, Adam(0.001), batches128, at most100epochs, patience10, best validation-loss weights. Three fixed initializations101/202/303 are all reported. CPU, one PyTorch thread. No test-based model choice or threshold tuning.

LDA and direct ridge are retrained on the identical training electrodes; historical baseline artifacts are preserved. Ridge uses alpha1 and the prior +/-0.18 drive threshold. All models use the existing two-consecutive-decision movement gate, immediate rest, and250ms expiry. Regular100ms replay has no missing decisions; expiry under acquisition dropouts is not tested here. The MLP/LDA use argmax without tuned confidence thresholds. Baseline search budgets are not optimized, so this is a first comparison, not a superiority trial.

An untrained network and block-shuffled-label network are controls. The shuffle preserves within-block temporal correlation but may change class duration proportions; it is a single diagnostic, not a permutation significance test.

## Metrics and acceptance

Raw three-class confusion matrices, balanced accuracy, rest recall and average left/right recall are saved. Gated movement episodes count as detected only after a correct-direction command; misses are reported separately. Delays run from the scheduled onset to decision availability, including recording and gate delay, but exclude robot actuation and GUI scheduling. Compute time is recorded as batch time and amortized throughput, not a guaranteed online latency.

False events count nonzero command starts/direction changes during rest, including movement carried into a rest boundary. Occupancy counts every rest decision with a movement command. Both all-rest metrics (including transitions) and the initial long-rest metrics are retained. The first unavailable100ms decision is excluded from exposure, yielding119.9 evaluated seconds/session and479.6 seconds across four test sessions.

Predeclared criteria: balanced accuracy>=90%, rest recall>=99%, movement-episode detection>=90%, long-rest false events<=0.5/minute, long-rest occupancy<=1%, median detected onset delay<=500ms. An always-rest model fails. These are preliminary synthetic feasibility criteria, not human safety claims. Overlapping windows are correlated; per-session metrics are retained and no window-level statistical significance is claimed.

## Reproduce and load

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-neural-network.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m neuralduck.supervised
.\.venv\Scripts\python.exe -m neuralduck.supervised --verify-saved
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`Train-Neural-Network.cmd` runs the fixed experiment. Rerunning overwrites only this experiment's own files; copy its folders before starting a different protocol. Use fresh test sessions for future tuning/research.

```python
from neuralduck.supervised import NeuralDecoder
model = NeuralDecoder.load('models/neural-network/mlp-101.pt')
probabilities = model.predict_proba(electrode_features)  # finite array [N,48]
```

Main trained weights: `models/neural-network/mlp-{101,202,303}.pt`; checkpoints include weights, normalization, class order, architecture and protocol. LDA/ridge and control networks are saved in the same new directory. Training curves, per-epoch losses/validation accuracy, parameter counts, weight changes, protocol/model/data hashes, test predictions, per-session metrics and reload checks are in `artifacts/neural-network/`. Raw electrodes and separate supervision are in `data/neural-network/`. The optional CPU PyTorch requirements leave the original dependency files intact.

## Implementation notes

An initial run finished training but stopped at a metric-name mismatch in the acceptance report. The reporting lookup was corrected and covered by a regression test. The same predefined experiment was rerun without altering data generation, model settings, thresholds or acceptance limits. This was a reporting repair, not test-guided tuning.

## Measured results

| Model | Balanced accuracy | Rest recall | Missed movement episodes /160 | Median onset | Long-rest false events | All-rest events | Criteria |
|---|---:|---:|---:|---:|---:|---:|---|
| mlp-101 | 97.60% | 99.09% | 0 | 165ms | 0 | 41 | pass |
| mlp-202 | 97.00% | 98.86% | 0 | 220ms | 0 | 38 | fail |
| mlp-303 | 98.44% | 99.32% | 0 | 150ms | 0 | 37 | pass |
| shuffled-label-mlp | 22.70% | 2.36% | 79 | 230ms | 4 | 69 | fail |
| untrained-mlp | 19.12% | 0.21% | 72 | 60ms | 79 | 201 | fail |
| lda | 90.01% | 98.94% | 12 | 310ms | 0 | 42 | fail |
| ridge | 95.13% | 98.16% | 2 | 245ms | 0 | 52 | fail |

Each model sees the same four final sessions and479.6 seconds of long-rest decision exposure. Zero observed events does not establish a zero underlying false-trigger rate. Initializations share these recordings and are not independent subjects. All-rest events include transition carryover and are not directly comparable to the previous robustness result, which excluded a one-second washout.

Two of three MLP initializations pass the aggregate criteria. Seed202 fails the99% raw rest-recall criterion (98.86%). All three detect160/160 movement episodes. No best seed is selected from final results. LDA and ridge fail the joint criteria. The network learns useful synthetic discrimination; superiority across brains, disturbances or human EEG is unestablished.

Training evidence:
- mlp-101: 99 epochs; best validation epoch89; validation loss0.03677, accuracy98.76%; training loss0.10201 -> 0.02543; learned weight-change L2=13.6530;1667 parameters.
- mlp-202: 77 epochs; best validation epoch67; validation loss0.04412, accuracy98.12%; training loss0.09017 -> 0.03384; learned weight-change L2=11.0622;1667 parameters.
- mlp-303: 98 epochs; best validation epoch88; validation loss0.03401, accuracy98.93%; training loss0.09415 -> 0.02403; learned weight-change L2=13.6825;1667 parameters.

Saved checkpoints reproduced all28 model/session prediction arrays exactly in a separate Python process. The results figure was visually checked. Final full-suite and targeted-test logs are retained.

Recommended next experiment: characterize onset/offset errors and rest-transition carryover using development/validation recordings, then lock a revised abstention/gating rule and evaluate on new sessions beyond22. Add a predefined noise/shift panel before robot integration or RL. Preserve the current weights as a frozen supervised reference. The long-rest-only acceptance limit should be supplemented by a transition/offset criterion in that next protocol.
