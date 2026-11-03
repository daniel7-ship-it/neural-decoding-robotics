"""Saved neural decoder controls Microduck from prerecorded simulated electrodes."""
import argparse,time,json,hashlib,uuid
from datetime import datetime,timezone
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons,Button
import mujoco
from .supervised import ROOT,DATA,MODELS,NeuralDecoder,causal_features
from .microduck import MicroduckPolicy,create_simulation
from .demo import RESEARCH
from .closed_loop import CommandGate
from .neural import effective_command
from .rest_gate_review import measure
from torch import nn
from .replay_engine import ReplayEngine
from .applied_scoring import score_frames


class ReplayDecoder(NeuralDecoder):
 def predict_proba(self,electrode_features):
  x=np.asarray(electrode_features,dtype=np.float32)
  if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():raise ValueError('Invalid electrode feature shape or values')
  with torch.no_grad():return self.model(torch.from_numpy((x-self.mean)/self.scale)).softmax(1).numpy()

def load_checkpoint(path):
 checkpoint=torch.load(path,map_location='cpu',weights_only=True)
 width=checkpoint.get('architecture',[48])[0]
 model=nn.Sequential(nn.Linear(width,32),nn.ReLU(),nn.Linear(32,3))
 model.load_state_dict(checkpoint['state_dict'])
 decoder=ReplayDecoder(model,checkpoint['mean'].numpy(),checkpoint['scale'].numpy())
 return decoder,width

def reset_robot(sim,data,robot):
 mujoco.mj_resetData(sim,data)
 robot.last_action[:]=0
 robot.command[:]=0
 robot.reset()

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=3600);p.add_argument('--fast',action='store_true');p.add_argument('--start',type=float,default=110.);p.add_argument('--session',type=int,default=19);args=p.parse_args()
 torch.set_num_threads(1)
 recordings=np.load(DATA/f'electrodes-{args.session}.npz')['recordings']
 labels=np.load(DATA/f'supervision-{args.session}.npz')['y']
 features,ends=causal_features(recordings)
 paths={'Original':MODELS/'mlp-101.pt','Augmented':ROOT/'models/augmentation/augmented-mlp-101.pt','Means-only 24':ROOT/'models/noise-diagnostic/noise-24-101.pt','Broad-noise 48':ROOT/'models/noise-diagnostic/noise-48-101.pt'}
 models={name:load_checkpoint(path) for name,path in paths.items()}
 supervision=np.load(DATA/f'supervision-{args.session}.npz'); scoring={k:supervision[k] for k in supervision.files}
 sim,data,bam=create_simulation(RESEARCH/'src/mjlab_microduck/robot/microduck/scene.xml');robot=MicroduckPolicy(sim,data,bam,ROOT/'models/velstand.onnx')
 renderer=mujoco.Renderer(sim,height=360,width=540);camera=mujoco.MjvCamera();camera.distance=.7;camera.azimuth=135;camera.elevation=-20
 fig,axes=plt.subplots(2,2,figsize=(12,7));fig.subplots_adjust(bottom=.27,top=.88,hspace=.4)
 fig.canvas.manager.set_window_title('Neuralduck — trained neural-network replay')
 axes[0,0].axis('off');image=axes[0,0].imshow(np.zeros((360,540,3),dtype=np.uint8))
 bars=axes[0,1].bar(['rest','left','right'],[1,0,0]);axes[0,1].set(ylim=(0,1),title='Uncalibrated neural-network probabilities')
 electrode=axes[1,0].imshow(np.zeros((24,20)),aspect='auto');axes[1,0].set(title='Actual replay electrode window',xlabel='100Hz samples',ylabel='Channel')
 text=axes[1,1];text.axis('off');status=text.text(0,1,'',va='top',fontsize=10)
 selector=RadioButtons(fig.add_axes([.12,.03,.22,.13]),list(models))
 condition=RadioButtons(fig.add_axes([.48,.03,.24,.13]),['Clean','Added noise SD0.5'])
 state={'model':'Original','condition':'Clean','tick':0,'gate':CommandGate(),'command':None,'events':0,'misses':0,'prior':0,'block_hit':False,'label':0}
 history=[]
 rng=np.random.default_rng(501);noisy=recordings+rng.normal(0,.5,recordings.shape);noisy_features,_=causal_features(noisy)
 engine=ReplayEngine(features,(ends+1)*.01,*models['Original'],start_seconds=args.start)
 run_id=uuid.uuid4().hex;summary={'text':None}
 def finish(reason):
  nonlocal run_id
  if summary['text'] is not None:return
  metrics=score_frames(engine.frames,scoring,engine.end_ns/1e9)
  trace_dir=ROOT/'artifacts/neural-replay-runs';trace_dir.mkdir(exist_ok=True)
  digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
  checkpoint=paths[state['model']];recording_path=DATA/f'electrodes-{args.session}.npz'
  decisions=[dict(index=int(d['index']),probability=d['probability'].tolist(),prediction=int(d['prediction']),issued_command=int(d['command']),available_at=float(d['window_available_at']),issued_at=float(d['issued_at']),eligible_at=float(d['applies_at'])) for d in engine.history]
  frames=[dict(time=float(f['time']),recording_time=float(f['recording_time']),applied=[float(v) for v in f['applied']],finished=bool(f['finished'])) for f in engine.frames]
  payload=dict(run_id=run_id,reason=reason,partial=reason!='complete',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),recording=str(recording_path),recording_sha256=digest(recording_path),session=args.session,condition=state['condition'],added_noise_seed=501 if state['condition']!='Clean' else None,start_seconds=args.start,seek_policy='first available window at/after start, fresh gate',physics_dt=.02,command_ttl=.25,decisions=decisions,frames=frames,metrics=metrics,robot_qpos=data.qpos.tolist(),robot_qvel=data.qvel.tolist())
  (trace_dir/f'{run_id}.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
  summary['text']=f"{reason.upper()} | decisions {len(decisions)}\nTimely movement {metrics['deadline_hits'] if metrics else 0} | missed {metrics['movement_episodes']-metrics['movement_detected'] if metrics else 0}\nRest events {metrics['false_rest_events'] if metrics else 0}\nTrace: {run_id[:12]}...json\nClick Restart or select a model/condition"
 def reset(_):
  nonlocal run_id
  if summary['text'] is None:finish('switch')
  reset_robot(sim,data,robot)
  history.clear()
  engine.features=features if condition.value_selected=='Clean' else noisy_features
  engine.reset(*models[selector.value_selected],start_seconds=args.start)
  state.update(model=selector.value_selected,condition=condition.value_selected,tick=0,gate=CommandGate(),command=None,events=0,misses=0,prior=0,block_hit=False,label=0)
  run_id=uuid.uuid4().hex;summary['text']=None
 selector.on_clicked(reset);condition.on_clicked(reset)
 button=Button(fig.add_axes([.75,.04,.16,.06]),'Restart')
 button.on_clicked(reset)
 fig.suptitle('PRERECORDED SIMULATED SIGNALS → trained neural decoder → Microduck')
 plt.show(block=False);start=time.perf_counter();steps=0;prob=np.array([1.,0,0]);known=0
 try:
  while plt.fignum_exists(fig.number) and time.perf_counter()-start<args.seconds:
   before=time.perf_counter()
   if engine.finished:
    fig.canvas.flush_events();time.sleep(.05);continue
   frame=engine.step();now=frame['time'];replay_time=args.start+now
   if frame['finished']:
    robot.step((0.,0.,0.));finish('complete');status.set_text(summary['text']);fig.canvas.draw_idle();fig.canvas.flush_events();continue
   robot.step(frame['applied'])
   decision=frame['decision']
   index=engine.history[-1]['index'] if engine.history else engine.first
   if decision is not None:
    prob=decision['probability'];known=int(labels[index]);state['command']=engine.command
    applied_yaw=frame['applied'][2]
    applied_class=1 if applied_yaw>0 else 2 if applied_yaw<0 else 0
    history.append((index,applied_class))
   if state['tick']%10==0:
    camera.lookat[:]=data.xpos[robot.trunk];renderer.update_scene(data,camera=camera);image.set_data(renderer.render())
    for bar,value in zip(bars,prob):bar.set_height(value)
    end=ends[max(index,0)];window=(recordings if state['condition']=='Clean' else noisy)[end-19:end+1].T
    electrode.set_data(window);electrode.set_clim(float(window.min()),float(window.max())+.001)
    report=score_frames(engine.frames,scoring,engine.end_ns/1e9)
    carry=report['transition_carryover_events'];fresh=report['new_rest_activation_events']
    status.set_text(f"Replay: {replay_time:.1f}s | elapsed {now:.1f}s\nCheckpoint: {paths[state['model']].name}\nInputs: {models[state['model']][1]} | fixed seed 101\nKnown intention: {['rest','left','right'][known]}\nPrediction: {['rest','left','right'][int(prob.argmax())]}\nIssued yaw: {state['command'].yaw_rate if state['command'] else 0:+.1f} rad/s\nApplied yaw: {frame['applied'][2]:+.1f} rad/s\nTransition carryover events: {carry}\nNew rest activations/direction changes: {fresh}\nCompleted blocks: timely movement {report['deadline_hits'] if report else 0}\nMissed episodes: {report['movement_episodes']-report['movement_detected'] if report else 0}\nTimely sustained stops: {report['timely_sustained_stops'] if report else 0}\nPrerecorded signals; no robot feedback.")
    fig.canvas.draw_idle();fig.canvas.flush_events()
   state['tick']+=1;steps+=1
   if args.fast and steps>=250:break
   if not args.fast:time.sleep(max(0,.02-(time.perf_counter()-before)))
  fig.savefig(ROOT/'artifacts/neural-replay-dashboard.png',dpi=120)
 finally:
  if summary['text'] is None:finish('closed')
  renderer.close();plt.close(fig)
if __name__=='__main__':main()
