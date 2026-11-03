Human EEG benchmark: run with PYTHONPATH=src and .venv/Scripts/python.exe -m neuralduck.human_eeg_benchmark. Protocol docs/human-eeg-protocol.json. Source data PhysioNet EEGMMIDB https://physionet.org/content/eegmmidb/1.0.0/ . Existing causal preprocessing reused. Heuristic quality checks are abstention flags, not expert artifact annotations. No continuous rest or intention latency claims. Dataset comparison, independent acceptance review, expanded tests and UI verification remain outstanding. Existing raw recordings and models preserved.

Completed benchmark results: {'101': 0.5014, '202': 0.4514, '303': 0.4598, 'csp_lda': 0.4737}
Saved three CNN checkpoints, CSP/LDA model, predictions, curves and subject-level coverage. Replay deferred: acceptance criteria not met.
