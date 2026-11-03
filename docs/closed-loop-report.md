# Continuous neural feedback milestone — 2026-10-26

## Outcome

The project now runs a continuous closed loop: measured heading error enters a persistent synthetic sensory/motor circuit; noisy electrodes observe that circuit; a learned decoder issues expiring movement commands; the official pretrained Microduck controller moves the robot; the changed heading supplies the next sensory input. Earlier mapping and EEG tools remain available.

This is engineered sensory feedback through a toy neural circuit, not inferred human intention. The circuit has 96 rate units and strong two-population structure. No synapses, whole brain, spikes, or human EEG sources are reconstructed.

## Launch

- `Open-Closed-Loop-Lab.cmd`: interactive native dashboard, up to one hour. Select direct, neurons, populations, or constant to restart a matched task. The restart button repeats the current task. Close the window to stop.
- `Run-Closed-Loop-Experiments.cmd`: reproduce the full CPU evaluation and result figure.
- Python: set `PYTHONPATH=src`, then use `.venv\Scripts\python.exe -m neuralduck.closed_loop_dashboard` or `-m neuralduck.closed_loop`.

## Design fixed before final evaluation

Calibration uses circuit seeds 42, 1337, and 2026 independently, with sessions 0–2 for fitting and session 3 for selecting ridge regularization from 0.1, 1, 10, and 100. Calibration streams contain randomized sustained signed sensory drives, including rest and transitions. Overlapping windows stay within their session; no random window split is used. The session-3 closed-loop validation was inspected before evaluating sessions 4 and 5. There was no final-test-based tuning.

Each rollout has 2 seconds warmup, three 12-second absolute heading targets (+1.0, -0.9, +0.65 radians), then 12 seconds continuous sensory rest. Success requires heading error at most 0.20 radians throughout the remainder of a target interval for at least its final second, with no fall anywhere in that target interval. Settling time is the start of this final uninterrupted interval. Overshoot is the maximum error past the target relative to initial approach direction. A fall is trunk height below 0.065 m, trunk up-vector z below 0.45, or a nonfinite physics state. Fall counts describe episodes with any fall.

The circuit advances at 100 Hz with state retained across targets and rest. Heading feedback is held constant over each 20 ms robot interval. Electrode windows span 200 ms and use channel means/standard deviations; decisions update every 100 ms. Two consecutive directional decisions are required to start or reverse movement. Below a fixed 0.18 decoded-drive threshold, movement stops immediately. Commands expire after 250 ms. The command is signed decoded drive scaled by 1.2 rad/s. Newly completed recordings can affect only the next physics interval.

All four paths share initial robot state, target sequence, circuit anatomy, session-specific electrode placement, and random noise seeds. Subsequent neural signals and robot trajectories differ because feedback depends on each path's actions. They are paired experimental conditions, not independent human subjects.

## Compared paths

1. **Direct:** 48 electrode features → ridge estimate of signed sensory drive → gate → command.
2. **Neurons:** electrode features → 96 estimated window-mean rates → learned ridge readout → gate → command.
3. **Populations:** electrode features → two estimated population means → learned ridge readout → gate → command. Each mean is repeated across its 48 neurons when scoring activity reconstruction.
4. **Constant:** fixed training-mean activity → learned readout → gate → command.

Synthetic true rates and drives are supervised calibration targets. During inference, the decoder API accepts only electrode features. True rates are used separately for evaluation/display. A direct decoder has no activity reconstruction. Readouts for reconstructed activity are calibrated on training true activity and evaluated on inferred activity. This calibration is possible in this simulator; equivalent human neuronal ground truth is not available from ordinary EEG.

## Final results

24 rollouts: 3 seeds × 2 held-out sessions × 4 paths. Each path received 18 target trials.

| Path | Targets reached | Mean settling, successful only | Mean overshoot | Neural rate MSE |
|---|---:|---:|---:|---:|
| Direct | 18/18 | 1.451 s | 0.079 rad | No estimate |
| 96 neurons | 18/18 | 1.518 s | 0.086 rad | 0.000266 |
| Two population means | 18/18 | 1.453 s | 0.071 rad | 0.001031 |
| Constant activity | 0/18 | No successes | 0.000 rad | 0.206897 |

No episode fell. Each path had zero false movement events in 72 seconds of total rest, accumulated from six 12-second rest segments. After the separately defined one-second transition washout, each path had zero events in 66 seconds. These short, easy synthetic recordings do not establish a low human false-trigger rate. An event is the start of a contiguous nonzero-command interval during rest; command occupancy and raw event counts are also retained. The constant baseline's zero overshoot reflects failing to approach the target, not good tracking.

Mean acquisition-plus-compute latency was approximately 200.5 ms direct, 200.8 ms neurons, and 200.7 ms populations. This is 200 ms of simulated acquisition plus measured inference/gate computation; it excludes display scheduling. Separate per-rollout wall-clock observation-to-command-application measurements include scheduling. Accelerated batch wall-clock durations must not be interpreted as real-time biological acquisition latency. The dashboard logs its actual paced wall-clock measurements, which can grow under GUI load: three native UI-check segments averaged approximately 293 ms, including rendering/scheduling. Neither latency measure includes unknown human neural encoding delay; target settling time measures the full simulated task response.

The detailed reconstruction better recovered individual rate variation than a repeated population average, but showed no control benefit on this task. Small settling-time differences are descriptive; this experiment does not establish statistical superiority. All paths are linear models and the circuit is deliberately low dimensional.

## Verification and artifacts

Nine tests passed, covering previous movement/mapping contracts plus state persistence, full-window availability, causal timestamps, direction hysteresis, immediate rest, command expiry, finite-input inference, and real-policy closed-loop motion. A follow-up targeted check covers metrics before any observations/targets are available. The dashboard rendered successfully; its saved screenshot was inspected for layout. A runtime warning from early empty metrics was fixed and the render check repeated with runtime warnings treated as errors. The initial window exceeded the local display height; its default size was reduced. Native UI checks confirmed all controls are visible, switching to the constant path restarts the task, the restart button works, and switching back to neuron reconstruction resumes feedback-controlled turning. These transitions are recorded in the dashboard event log. The lab was left open on the neuron path.

- `artifacts/closed-loop-evaluation.json`: full protocol, calibration choices, raw episode metrics, summary.
- `artifacts/closed-loop-evaluation.png`: comparison figure.
- `artifacts/closed-loop-trace-*.json`: step traces and decision timestamps/errors.
- `artifacts/closed-loop-validation.json`: exploratory session-3 validation.
- `artifacts/closed-loop-tests.txt`: test output.
- `artifacts/closed-loop-dashboard.png`: rendered interface.
- `artifacts/closed-loop-dashboard-events.jsonl`: dashboard partial/completed episode metrics and path/restart events.

## Adaptation decision and next experiment

Adaptation is deferred. All signal-based paths reached the success ceiling under baseline recording conditions. Adding an adaptive rule now would require introducing and selecting a disturbance and reward protocol after viewing final results, compromising a clean comparison within this milestone.

Next, predefine held-out electrode placement/gain/noise drift, longer rest recordings, and randomized target sequences. Compare frozen decoders first. Then reserve distinct adaptation sessions and fresh final-test sessions to compare a small reward-driven decoder update against the frozen and population baselines. Keep the official locomotion policy fixed. This establishes a measurable failure for adaptation to address before claiming learning benefits. Human EEG validation remains a separate research track.
