# Recording robustness and reward adaptation — 2026-10-26

The [locked protocol](robustness-protocol.json) was saved before development and final testing. The experiment changes only synthetic electrode recordings while the 96-unit circuit anatomy and the official Microduck locomotion policy stay fixed. Session 6 determined the adaptation challenge, session 7 provided a fixed six-episode reward budget, and sessions 9–10 were reserved for fresh evaluation. Sessions 4–5 from the earlier milestone are not reused.

The task samples three targets from four specified angles for each seed/session and holds each for 12 seconds. It then records 60 uninterrupted seconds of rest. Direct electrode decoding, 96-rate reconstruction, and two-population reconstruction run on matched targets and random seeds. Feedback makes their later trajectories different. A separate recording replay feeds identical electrode windows to both reconstruction models and scores against the same simulated neuron rates.

Each disturbance is isolated: +0.65 electrode-coordinate shift, 75% right-channel gain reduction, additional noise SD 0.5, or loss of channels 12–23. Changes start suddenly at two seconds or ramp over 18 seconds. The 60-second rest window counts starts of nonzero movement commands and command occupancy; transition washout is also reported. A false event is a decoded command event, not a human BCI false activation estimate.

The adaptation method chooses one of six bounded gain/bias corrections to a decoder's output. It sees only the decoder's own output and episode-level task reward (success, final errors, falls, rest events). It cannot read hidden neural rates, stimulus labels, or electrode parameters; measured heading error feeds the simulated circuit, not a direct command. Every candidate gets one matched session-7 episode, and a frozen control runs the same number of episodes. The selected correction is locked before sessions 9–10; online candidate performance and final locked performance are reported separately.

Development session 6: the undisturbed control reached 14/18 targets across seeds and paths. Sudden gain drop and sudden channel loss each reached 6/18; the fixed tie rule chose gain drop for adaptation. Sudden electrode shift reached 7/18. Added noise reached 14/18 but produced seven false movement events after the transition washout across six 60-second rest periods, versus zero for undisturbed control. These development figures selected the challenge and are not final-test estimates.

Adaptation session 7: all six seed/path combinations selected candidate 4 of 6, a fixed −0.15 signed-drive bias. Across all 36 exploratory candidate episodes, mean target success was 22% versus 28% for 36 equal-budget frozen replays. Rejected candidates contributed 433 post-washout false rest-command events; the selected candidate contributed zero in its six session-7 episodes. The search's overall online performance was worse than frozen control, so any benefit must be judged on fresh sessions rather than from selecting the highest development reward.

## Fresh held-out results

Sessions 9–10 produced 216 complete rollouts: two circuit seeds × two fresh sessions × nine conditions × three decoder paths × frozen/adapted modes. Each condition/mode has 36 target trials and 720 total simulated seconds of continuous rest. No rollout fell. Mean acquisition-plus-compute latency remained about 201 ms across modes; paced GUI scheduling can add wall time.

| Recording condition | Frozen targets | Locked adaptation targets | Frozen false rest events* | Adapted false rest events* |
|---|---:|---:|---:|---:|
| Undisturbed | 29/36 | 25/36 | 0 | 3 |
| Position shift, sudden | 24/36 | 28/36 | 0 | 0 |
| Position shift, gradual | 30/36 | 24/36 | 0 | 0 |
| Gain drop, sudden (selected challenge) | 14/36 | 19/36 | 0 | 0 |
| Gain drop, gradual | 29/36 | 29/36 | 0 | 0 |
| Added noise, sudden | 29/36 | 26/36 | 11 | 160 |
| Added noise, gradual | 28/36 | 25/36 | 11 | 160 |
| Channel loss, sudden | 13/36 | 16/36 | 0 | 0 |
| Channel loss, gradual | 26/36 | 29/36 | 0 | 0 |

*Event starts after a one-second transition washout in each 60-second rest stream. Raw events and command occupancy are retained in the JSON; each row covers twelve rest streams. The noise conditions share much of their signal after the ramp, so matching counts do not represent independent replication.

The selected challenge improved by five of 36 target trials without adding rest events under that specific disturbance. Successful targets under sudden gain loss took longer to settle on average with adaptation (1.69 versus 1.22 seconds, successful trials only). Across all nine conditions, target success was 221/324 adapted versus 222/324 frozen, while false rest events rose to 323 from 22. The large rest-event increase occurred under added noise: 160 adapted versus 11 frozen for each noise-onset condition. Even the undisturbed control lost four target successes and gained three rest events after applying the correction.

This bounded output search therefore **does not yield a generally safer or more robust controller**. It recovered some target performance for the selected disturbance, with major failures when the recording changed differently. The 36 target trials per condition/mode share only four seed/session pairs and three targets per rollout; the counts are descriptive and do not establish statistical generalization.

## Matched activity reconstruction

The separate offline replay used 335 identical held-out electrode windows per seed/session/condition, 12,060 windows in all, with the same true 96-unit rate targets for both reconstruction methods. Across the 36 paired recording sets, mean rate MSE was 0.02823 for the 96-unit model, 0.02752 for the two-population model repeated across units, and 0.09308 for constant training-mean activity. The detailed model won only 18 of 36 paired sets. Under undisturbed recording it was slightly better (0.00192 versus 0.00246), but sudden electrode shift favored the population estimate (0.03672 versus 0.02908). Detailed reconstruction did not produce a dependable advantage under this protocol. This is supervised reconstruction of synthetic window means, not inference of wiring or human neural activity.

## Validation, artifacts, and next experiment

Thirteen tests passed, covering the existing robot/mapping contract plus fixed neuron anatomy across disturbances, deterministic target/recording generation, timing of sudden and gradual changes, session separation, inference inputs, bounded search choices, and development-only challenge selection. The new native dashboard was rendered and checked through its controls: mode switching loaded the locked candidate, condition cycling changed the recording disturbance, and the plots and status stayed within the local display. A status-line spacing issue found in that check was corrected. Prior mapping, EEG replay, and continuous-feedback launchers remain available.

Raw results are in `artifacts/robustness-development.json`, `robustness-adaptation.json`, and `robustness-final.json`; the comparison figure is `robustness-final.png`, and dashboard events/screenshot are saved separately. Launch the interface with `Open-Robustness-Lab.cmd`; rerun all phases with `Run-Robustness-Experiments.cmd`.

The next experiment should put a validated rest/abstain detector and an explicit false-event limit ahead of any reward update. Then test adaptation that learns from multiple disturbance types on separate sessions, with a fresh final split and a criterion that requires both tracking improvement and acceptable rest behavior. The toy circuit remains low dimensional, and none of these results transfer to human EEG without separate evidence.
