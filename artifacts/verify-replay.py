import json
import numpy as np
import torch
from neuralduck.neural_replay import load_checkpoint
from neuralduck.supervised import ROOT,gated_predictions
from neuralduck.augmentation import recording
checks=[]
for condition in ('clean','noise_hard'):
 row=recording(45,condition)
 for width in (24,48):
  model,n=load_checkpoint(ROOT/f'models/noise-diagnostic/noise-{width}-101.pt')
  # Exact batch reproduction, plus streaming command reproduction from single-window inference.
  prob=model.predict_proba(row['x'][:,:n])
  saved=np.load(ROOT/f'artifacts/noise-diagnostic/45-{condition}-noise-{width}-101-False.npz')
  np.testing.assert_allclose(prob,saved['probability'],atol=1e-6)
  streaming=np.concatenate([model.predict_proba(x[None,:n]) for x in row['x']])
  np.testing.assert_allclose(streaming,saved['probability'],atol=1e-6)
  np.testing.assert_array_equal(streaming.argmax(1),saved['probability'].argmax(1))
  np.testing.assert_array_equal(gated_predictions(streaming.argmax(1),row['time']),saved['command'])
  checks.append(dict(condition=condition,width=width,predictions_and_commands_identical=True))
(ROOT/'artifacts/replay-parity.json').write_text(json.dumps(checks,indent=2))
print(checks)

