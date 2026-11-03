# Provenance and licenses

- Microduck: https://github.com/pollen-robotics/microduck — local reference commit `96b89d8613f06722c1d64ce75ab0d7124fed9d8f`.
- Microduck RL: https://github.com/pollen-robotics/microduck_rl — local reference commit `b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`.
- Pollen upstream code is Apache-2.0; the included notice is [here](licenses/POLLEN-APACHE-2.0.txt). Simulation assets have separate [CC-BY-NC-SA-4.0 terms](licenses/POLLEN-MODEL-ASSETS.md) and are obtained from upstream rather than copied here.
- `models/velstand.onnx` is Pollen Robotics’ pretrained policy. Source, resolved revision, SHA-256 and Apache-2.0 model-card metadata are preserved in [models/provenance.json](models/provenance.json). [models/manifest.json](models/manifest.json) describes the broader upstream policy family; only velstand is bundled.
- BAM actuator model: https://github.com/Rhoban/bam at `57d13ead53206a6bf0db3d66f86506ae8c2ce01a`.
- Human EEG: [PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/), credited to its creators and PhysioNet. Raw recordings are not bundled. Saved analyses and trained decoders derive from that public dataset; consult its attribution and reuse terms.

The synthetic neural, decoding, evaluation, and integration work is presented as an independent AI-assisted research project. Upstream assets and locomotion training are not claimed as original contributions. No blanket license is newly granted for original project material in this publication; third-party licenses remain applicable to their respective material.
