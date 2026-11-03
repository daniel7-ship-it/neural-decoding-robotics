# Preprocessing and provenance fix — 2026-10-30

Replaced the benchmark's separate hardcoded extraction and quality loop with eeg_pipeline.recording. The structured configuration now executes referencing, channel ordering, causal Butterworth filtering, sample offset/duration, annotation mapping and quality thresholds. Both quality checks and filtered decoder windows use one pair of sample boundaries. Unsupported processing modes and invalid parameters fail explicitly.

Each cache stores trial metadata. Its identity includes source bytes, actual configuration/channel order/sample rate and pipeline implementation hash. New cache namespace eeg-cache-v3 leaves historical caches intact. Every run saves manifest.json containing all source hashes, subject/run/split assignments, channel configuration and trial labels/boundaries. Checkpoints embed the aggregate manifest and its hash rather than the last subject's fingerprint.

Modified: src/neuralduck/human_eeg_benchmark.py; added src/neuralduck/eeg_pipeline.py and tests/test_eeg_pipeline.py. Updated research-completion-checklist.md to replace contradictory status additions.

Targeted test exercises the actual recording function with an MNE RawArray supplied at the EDF-reader boundary. It verifies duration/sample count, offset/boundaries, filter effects, quality flags, channel ordering, cache misses/hits, multiple subject/split manifest retention, and checkpoint manifest round trip. No downloads or training.

Limitations: the fixture does not test EDF parsing itself; existing MNE EDF loading is retained. Historical weights retain legacy provenance. The trained model has fixed 64-channel input; changing the number of channels is rejected by the pipeline's complete-order rule. Broader scoring/UI/analysis backlog remains outside this focused fix.

Reproduce: set PYTHONPATH=src, then .venv\Scripts\python.exe -m unittest discover -s tests -p test_eeg_pipeline.py -v. Full suite evidence: artifacts/reproducibility-fix-tests.log.

Approximate focused working time: several minutes, not a full 20-minute session. No idle time was added.
