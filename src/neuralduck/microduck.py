"""Windows-safe Microduck adapter matching upstream policy and BAM contracts."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

STIFF_SOLREF = (-5.0e4, -2.0e2)
STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


def create_simulation(scene_xml: Path, vin: float = 7.4, vin_drop_gain: float = 0.1):
    """Compile upstream scene using the same BAM M6/XL330 motor path as training."""
    from bam.model import load_model
    from bam.mujoco import MujocoController

    bm = load_model(motor_name="xl330", model="m6")
    bm.actuator.kp = 200.0
    bm.actuator.vin = vin
    bm.actuator.max_current = None
    force_limit = bm.actuator.vin * bm.kt.value / bm.R.value
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    names = []
    for actuator in spec.actuators:
        target = actuator.target
        target_name = target.name if hasattr(target, "name") else str(target)
        if target_name.startswith("passive_"):
            continue
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-force_limit, force_limit)
        actuator.ctrllimited = False
        actuator.gear = [1, 0, 0, 0, 0, 0]
        names.append(actuator.name)
        for joint in spec.joints:
            if joint.name == target_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = STIFF_SOLREF
                joint.solimp_friction = STIFF_SOLIMP
                break
    model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    # Current pinned BAM names the voltage-drop parameter as resistance [V/A];
    # upstream's infer script expresses it as gain [V/Nm]. Convert through kt
    # so both formulations produce the same sag: R = gain * kt.
    controller = MujocoController(bm, names, model, data,
                                  vin_drop_resistance=vin_drop_gain * bm.kt.value,
                                  vin_min=6.0)
    controller.last_ts = data.time - model.opt.timestep
    if model.nu != 14:
        raise RuntimeError(f"Expected 14 actuators, found {model.nu}")
    return model, data, controller


class MicroduckPolicy:
    def __init__(self, model, data, bam_controller, onnx_path: Path):
        self.model = model
        self.data = data
        self.bam = bam_controller
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.output = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.obs_dim = int(shape[-1])
        self.nj = model.nu
        self.qpos = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
        self.qvel = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
        self.default = DEFAULT_POSE.copy()
        self.last_action = np.zeros(14, dtype=np.float32)
        self.command = np.zeros(13, dtype=np.float32)
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        if self.obs_dim != 61:
            raise RuntimeError(f"This velstand adapter expects 61D actor observations, policy has {shape}")
        self.reset()

    def reset(self):
        adr = int(self.model.jnt_qposadr[self.free])
        self.data.qpos[adr:adr+7] = [0, 0, 0.125, 1, 0, 0, 0]
        for i, idx in enumerate(self.qpos):
            self.data.qpos[idx] = self.default[i]
        self.bam.q_target[:] = self.default
        self.bam.last_ts = self.data.time - self.model.opt.timestep
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def rotate_inverse(q, vec):
        xyz = q[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - q[0] * t + np.cross(xyz, t)

    def observation(self, command):
        self.command[:] = 0.0
        self.command[:3] = command
        sadr = int(self.model.sensor_adr[self.gyro])
        ang = self.data.sensordata[sadr:sadr+3].astype(np.float32)
        gravity = self.rotate_inverse(self.data.xquat[self.trunk].copy(),
                                      np.array([0, 0, -1], dtype=np.float32)).astype(np.float32)
        qpos = self.data.qpos[self.qpos].astype(np.float32)
        qvel = self.data.qvel[self.qvel].astype(np.float32)
        return np.concatenate((ang, gravity, qpos - self.default, qvel,
                               self.last_action, self.command)).astype(np.float32)

    def step(self, command):
        obs = self.observation(command)
        action = self.session.run([self.output], {self.input: obs.reshape(1, -1)})[0].squeeze(0).astype(np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid policy action shape/values: {action.shape}")
        self.last_action[:] = action
        self.bam.q_target[:] = self.default + action
        for _ in range(4):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)
        return action

    def body_velocity(self):
        jid = self.free
        adr = int(self.model.jnt_dofadr[jid])
        qpos_adr = int(self.model.jnt_qposadr[jid])
        quat = self.data.qpos[qpos_adr+3:qpos_adr+7].astype(np.float32)
        vworld = self.data.qvel[adr:adr+3].astype(np.float32)
        vb = self.rotate_inverse(quat, vworld)
        return float(vb[0]), float(vb[1]), float(self.data.qvel[adr+5])

