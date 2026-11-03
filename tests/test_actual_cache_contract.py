import unittest,tempfile
from pathlib import Path
from neuralduck.eeg_config import execution_config,CONFIG
from neuralduck.eeg_research import fingerprint
class ActualCacheContract(unittest.TestCase):
 def test_dependencies_change_key(self):
  with tempfile.TemporaryDirectory() as root:
   source=Path(root)/'recording.edf';source.write_bytes(b'raw');base=execution_config(['C3','C4'],160);key=fingerprint([source],base)
   changes=[dict(base,channels=['C4','C3']),dict(base,filter=dict(base['filter'],low_hz=9.)),dict(base,window=dict(base['window'],duration_seconds=1.)),dict(base,quality=dict(base['quality'],peak_volts=100e-6)),dict(base,implementation_sha256='changed')]
   for changed in changes:self.assertNotEqual(key,fingerprint([source],changed))
if __name__=='__main__':unittest.main()
