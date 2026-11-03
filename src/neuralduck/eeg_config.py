"""Concrete preprocessing contract used for EEG cache identity and extraction."""
from pathlib import Path
import hashlib
CONFIG=dict(runs=[4,8,12],units='volts',reference='average_all_eeg',filter=dict(kind='butterworth_sos',order=4,low_hz=8.,high_hz=30.,causal=True),window=dict(offset_seconds=1.,duration_seconds=2.),annotations={'T1':0,'T2':1},quality=dict(peak_volts=200e-6,minimum_ptp_volts=.1e-6,policy='reject_whole_trial_if_any_channel_fails'))
def execution_config(channels,sampling_hz):
 from . import eeg
 source=hashlib.sha256(Path(eeg.__file__).read_bytes()).hexdigest()
 return dict(**CONFIG,channels=list(channels),sampling_hz=float(sampling_hz),implementation_sha256=source)
