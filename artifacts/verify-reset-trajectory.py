import numpy as np
from neuralduck.demo import ROOT,RESEARCH
from neuralduck.microduck import create_simulation,MicroduckPolicy
from neuralduck.neural_replay import reset_robot
scene=RESEARCH/'src/mjlab_microduck/robot/microduck/scene.xml'
a,ad,ab=create_simulation(scene);ap=MicroduckPolicy(a,ad,ab,ROOT/'models/velstand.onnx')
b,bd,bb=create_simulation(scene);bp=MicroduckPolicy(b,bd,bb,ROOT/'models/velstand.onnx')
for _ in range(40):ap.step((0.,0.,1.2))
reset_robot(a,ad,ap)
for tick in range(100):
 command=(0.,0.,1.2 if tick<50 else 0.)
 ap.step(command);bp.step(command)
 np.testing.assert_allclose(ad.qpos,bd.qpos,atol=1e-8,rtol=1e-8)
 np.testing.assert_allclose(ad.qvel,bd.qvel,atol=1e-8,rtol=1e-8)
print('Reset-vs-fresh robot trajectories matched for 100 control steps after prior turning rollout.')
