"""Timestamp-driven neural replay shared by dashboard and headless tests."""
import numpy as np
from .closed_loop import CommandGate
from .neural import effective_command

DT_NS=20_000_000
WINDOW_NS=100_000_000

class ReplayEngine:
 def __init__(self,features,times,decoder,width,start_seconds=0.):
  x=np.asarray(features);t=np.asarray(times,dtype=float)
  if x.ndim!=2 or t.ndim!=1 or len(x)!=len(t) or x.shape[1]<width or width<1 or not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
   raise ValueError('Replay needs finite 2D features and matching strictly increasing window availability times')
  self.features=x;self.times=t;self.time_ns=np.rint(t*1e9).astype(np.int64)
  if len(t) and np.any(np.diff(self.time_ns)<=0):raise ValueError('Window times collapse at nanosecond resolution')
  self.reset(decoder,width,start_seconds)
 def reset(self,decoder,width,start_seconds=None):
  if width<1 or width>self.features.shape[1]:raise ValueError('Decoder input width exceeds electrode features')
  if start_seconds is not None:
   if not np.isfinite(start_seconds):raise ValueError('Seek time must be finite')
   self.start_seconds=float(start_seconds)
  self.decoder=decoder;self.width=width;self.start_ns=int(round(self.start_seconds*1e9))
  self.first=int(np.searchsorted(self.time_ns,self.start_ns,side='left'))
  self.next_index=self.first;self.tick=0;self.command=None;self.gate=CommandGate();self.history=[];self.frames=[];self.finished=False
  self.end_ns=int(self.time_ns[-1]+WINDOW_NS) if self.first<len(self.time_ns) else self.start_ns
 def step(self):
  if self.finished:return dict(time=self.tick*.02,recording_time=(self.start_ns+self.tick*DT_NS)/1e9,applied=(0.,0.,0.),decisions=[],decision=None,finished=True)
  now_ns=self.tick*DT_NS;absolute_ns=self.start_ns+now_ns;now=now_ns/1e9
  if absolute_ns>=self.end_ns:
   self.finished=True;self.command=None
   frame=dict(time=now,recording_time=absolute_ns/1e9,applied=(0.,0.,0.),decisions=[],decision=None,finished=True)
   self.frames.append(frame);return frame
  applied=effective_command(self.command,now)
  decisions=[]
  while self.next_index<len(self.features) and self.time_ns[self.next_index]<=absolute_ns:
   index=self.next_index
   probability=self.decoder.predict_proba(self.features[index:index+1,:self.width])[0]
   label=int(probability.argmax());drive=1. if label==1 else -1. if label==2 else 0.
   self.command=self.gate.update(drive,now+.02)
   decision=dict(index=index,probability=probability,prediction=label,command=1 if self.command.yaw_rate>0 else 2 if self.command.yaw_rate<0 else 0,window_available_at=self.times[index],issued_at=now,applies_at=now+.02)
   decisions.append(decision);self.history.append(decision);self.next_index+=1
  frame=dict(time=now,recording_time=absolute_ns/1e9,applied=applied,decisions=decisions,decision=decisions[-1] if decisions else None,finished=False)
  self.frames.append(frame);self.tick+=1;return frame
