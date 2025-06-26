# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aloha2 base task which uses a MuJoCo Menagerie robot model."""

import collections
from collections.abc import Mapping
import copy
import dataclasses
import enum
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer import initializers
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import variation_broadcaster
from dm_env import specs
import immutabledict
import numpy as np
from numpy import typing as npt


HOME_CTRL: npt.NDArray[float] = np.array(
    [0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.002]
)
HOME_CTRL.setflags(write=False)
HOME_QPOS: npt.NDArray[float] = np.array(
    [0.0, -0.959, 1.182, 0.0, -0.274, 0.0, 0.0082, 0.0082]
)
HOME_QPOS.setflags(write=False)


# The linear displacement that corresponds to fully open and closed gripper
# in sim. Note that the sim model does not model the dynamixel values, but
# rather the linear displacement of the fingers in meters.

# SIM_GRIPPER_CTRL_CLOSE controls the range of ctrl values that can be set,
# and is lower than the achievable qpos for the gripper, so that the
# proportional actuator can apply a force when the gripper is in the closed
# position.
# SIM_GRIPPER_QPOS_CLOSE is the value of qpos when the gripper is closed in
# sim.
SIM_GRIPPER_QPOS_OPEN: float = 0.037
SIM_GRIPPER_QPOS_CLOSE: float = 0.0078

# Range used for setting ctrl
SIM_GRIPPER_CTRL_OPEN: float = 0.037
SIM_GRIPPER_CTRL_CLOSE: float = 0.002

# These are follower dynamixel values for OPEN and CLOSED gripper.
FOLLOWER_GRIPPER_OPEN: float = 1.5155
FOLLOWER_GRIPPER_CLOSE: float = -0.06135

LEADER_GRIPPER_OPEN: float = 0.78
LEADER_GRIPPER_CLOSE: float = -0.04

WRIST_CAMERA_POSITION: tuple[float, float, float] = (
    -0.011,
    -0.0814748,
    -0.0095955,
)


@dataclasses.dataclass(frozen=True)
class GripperLimit:
  """Gripper open and close limit.

  Attributes:
    open: Joint position of gripper being open.
    close: Joint position of gripper being closed.
  """

  open: float
  close: float

GRIPPER_LIMITS = immutabledict.immutabledict({
    'sim_qpos': GripperLimit(
        open=SIM_GRIPPER_QPOS_OPEN,
        close=SIM_GRIPPER_QPOS_CLOSE,
    ),
    'sim_ctrl': GripperLimit(
        open=SIM_GRIPPER_CTRL_OPEN,
        close=SIM_GRIPPER_CTRL_CLOSE,
    ),
    'follower': GripperLimit(
        open=FOLLOWER_GRIPPER_OPEN,
        close=FOLLOWER_GRIPPER_CLOSE,
    ),
    'leader': GripperLimit(
        open=LEADER_GRIPPER_OPEN,
        close=LEADER_GRIPPER_CLOSE,
    ),
})


_DEFAULT_PHYSICS_DELAY_SECS: float = 0.3
_DEFAULT_JOINT_OBSERVATION_DELAY_SECS: float = 0.1
_DEFAULT_TABLE_HEIGHT_OFFSET: float = 0.011

_ALL_JOINTS: tuple[str, ...] = (
    r'left\waist',
    r'left\shoulder',
    r'left\elbow',
    r'left\forearm_roll',
    r'left\wrist_angle',
    r'left\wrist_rotate',
    r'left\left_finger',
    r'left\right_finger',
    r'right\waist',
    r'right\shoulder',
    r'right\elbow',
    r'right\forearm_roll',
    r'right\wrist_angle',
    r'right\wrist_rotate',
    r'right\left_finger',
    r'right\right_finger',
)


class GeomGroup(enum.IntFlag):
  NONE = 0
  ARM = enum.auto()
  GRIPPER = enum.auto()
  TABLE = enum.auto()
  OBJECT = enum.auto()
  LEFT = enum.auto()
  RIGHT = enum.auto()


class AlohaTask(composer.Task):
  """The base aloha task."""

  def __init__(
      self,
      control_timestep: float,
      cameras: tuple[str, ...] = ('overhead_cam',),
      camera_resolution: tuple[int, int] = (480, 848),
      joints_observation_delay_secs: (
          variation.Variation | float
      ) = _DEFAULT_JOINT_OBSERVATION_DELAY_SECS,
      image_observation_enabled: bool = True,
      image_observation_delay_secs: (
          variation.Variation | float
      ) = _DEFAULT_PHYSICS_DELAY_SECS,
      update_interval: int = 1,
      table_height_offset: float = _DEFAULT_TABLE_HEIGHT_OFFSET,
      waist_joint_limit: float = np.pi / 2,
      terminate_episode=True,
  ):
    """Initializes a new aloha task.

    Args:
      control_timestep: Float specifying the control timestep in seconds.
      cameras: The default cameras to use.
      camera_resolution: The camera resolution to use for rendering.
      joints_observation_delay_secs: The delay of the joints observation. This
        can be a number or a composer.Variation. If set, also adds
        `undelayed_joints_pos` and `undelayed_joints_vel` observables for
        debugging.
      image_observation_enabled: Whether to enable physics state
        observation, as defined by `physics.get_state()`.
      image_observation_delay_secs: The delay of the
        `delayed_physics_state` observable. Note that the `physics_state`
        observable is not delayed. This can be a number or a composer.Variation.
        When set this also delays the camera observations.
      update_interval: An integer, number of simulation steps between
        successive updates to the value of this observable.
      table_height_offset: The offset to the height of the table in meters.
      waist_joint_limit: The joint limit for the waist joint, in radians. Only
        affects the action spec.
      terminate_episode: Whether to terminate the episode when the task 
        succeeds.
    """

    self._waist_joint_limit = waist_joint_limit
    self._terminate_episode = terminate_episode

    self._scene = Arena(
        camera_resolution=camera_resolution,
        table_height_offset=table_height_offset,
    )
    self._scene.mjcf_model.option.flag.multiccd = 'enable'
    self._scene.mjcf_model.option.noslip_iterations = 0

    self.control_timestep = control_timestep

    self._joints = [
        self._scene.mjcf_model.find('joint', name) for name in _ALL_JOINTS
    ]

    # Add custom camera observable.
    obs_dict = collections.OrderedDict()

    shared_delay = variation_broadcaster.VariationBroadcaster(
        image_observation_delay_secs / self.physics_timestep
    )
    cameras_entities = [
        self.root_entity.mjcf_model.find('camera', name) for name in cameras
    ]
    for camera_entity in cameras_entities:
      # Use calibrated camera extrinsics for wrist cameras.
      if (
          camera_entity.name == 'wrist_cam_left'
          or camera_entity.name == 'wrist_cam_right'
      ):
        camera_entity.pos = WRIST_CAMERA_POSITION
      obs_dict[camera_entity.name] = observable.MJCFCamera(
          camera_entity,
          height=camera_resolution[0],
          width=camera_resolution[1],
          update_interval=update_interval,
          buffer_size=1,
          delay=shared_delay.get_proxy(),
          aggregator=None,
          corruptor=None,
      )
      obs_dict[camera_entity.name].enabled = True

    aloha_observables = AlohaObservables(
        self.root_entity,
    )
    aloha_observables.enable_all()
    obs_dict.update(aloha_observables.as_dict())
    self._task_observables = obs_dict

    if joints_observation_delay_secs:
      self._task_observables['undelayed_joints_pos'] = copy.copy(
          self._task_observables['joints_pos']
      )
      self._task_observables['undelayed_joints_vel'] = copy.copy(
          self._task_observables['joints_vel']
      )
      self._task_observables['joints_pos'].configure(
          delay=joints_observation_delay_secs / self.physics_timestep
      )
      self._task_observables['joints_vel'].configure(
          delay=joints_observation_delay_secs / self.physics_timestep
      )
      self._task_observables['delayed_joints_pos'] = copy.copy(
          self._task_observables['joints_pos']
      )
      self._task_observables['delayed_joints_vel'] = copy.copy(
          self._task_observables['joints_vel']
      )

    self._task_observables['physics_state'].enabled = (
        image_observation_enabled
    )
    if image_observation_delay_secs:
      self._task_observables['delayed_physics_state'] = copy.copy(
          self._task_observables['physics_state']
      )
      self._task_observables['delayed_physics_state'].configure(
          delay=shared_delay.get_proxy(),
      )

    self._all_props = []
    self._all_prop_placers = []
    if self._all_prop_placers:
      self._all_prop_placers.append(
          initializers.PropPlacer(
              props=self._all_props,
              position=deterministic.Identity(),
              ignore_collisions=True,  # Collisions already resolved.
              settle_physics=True,
          )
      )

  @property
  def root_entity(self) -> composer.Entity:
    return self._scene

  @property
  def task_observables(self) -> Mapping[str, observable.Observable]:
    return dict(
        **self._task_observables
    )

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    # 0-5: left arm joints.
    # 6: left arm gripper.
    # 7-12: right arm joints.
    # 13: right arm gripper.
    minimum = physics.model.actuator_ctrlrange[:, 0].astype(np.float32)
    maximum = physics.model.actuator_ctrlrange[:, 1].astype(np.float32)
    minimum[0] = minimum[7] = -self._waist_joint_limit
    maximum[0] = maximum[7] = self._waist_joint_limit

    # Gripper actions are never delta actions.
    minimum[6] = minimum[13] = GRIPPER_LIMITS['follower'].close
    maximum[6] = maximum[13] = GRIPPER_LIMITS['follower'].open

    return specs.BoundedArray(
        shape=(14,),
        dtype=np.float32,
        minimum=minimum,
        maximum=maximum,
    )

  @classmethod
  def convert_gripper(
      cls,
      gripper_value: npt.NDArray[float],
      from_name: str,
      to_name: str,
  ) -> float:
    from_limits = GRIPPER_LIMITS[from_name]
    to_limits = GRIPPER_LIMITS[to_name]
    return (gripper_value - from_limits.close) / (
        from_limits.open - from_limits.close
    ) * (to_limits.open - to_limits.close) + to_limits.close

  def before_step(
      self,
      physics: mjcf.Physics,
      action: npt.ArrayLike,
      random_state: np.random.RandomState,
  ) -> None:
    action_left = action[:7]
    action_right = action[7:]

    left_joints_new = action_left[:6]
    right_joints_new = action_right[:6]

    np.copyto(physics.data.ctrl[:6], left_joints_new)
    np.copyto(physics.data.ctrl[7:13], right_joints_new)

    # Handle the gripper action. The gripper action is the same whether it's
    # delta actions or not.
    left_gripper = action_left[6]
    right_gripper = action_right[6]

    left_gripper_cmd = np.array([left_gripper])
    right_gripper_cmd = np.array([right_gripper])

    left_gripper_ctrl = AlohaTask.convert_gripper(
        left_gripper_cmd, 'follower', 'sim_ctrl'
    )
    right_gripper_ctrl = AlohaTask.convert_gripper(
        right_gripper_cmd, 'follower', 'sim_ctrl'
    )

    np.copyto(physics.data.ctrl[6:7], left_gripper_ctrl)
    np.copyto(physics.data.ctrl[13:14], right_gripper_ctrl)

  def get_reward(self, physics: mjcf.Physics) -> float:
    return 0.0

  def get_discount(self, physics: mjcf.Physics) -> float:
    if self.should_terminate_episode(physics):
      return 0.0
    return 1.0

  def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
    if self._terminate_episode:
      reward = self.get_reward(physics)
      if reward >= 1.0:
        return True
    return False

  def initialize_episode(
      self, physics: mjcf.Physics, random_state: np.random.RandomState
  ) -> None:
    arm_joints_bound = physics.bind(self._joints)

    arm_joints_bound.qpos[:8] = HOME_QPOS
    arm_joints_bound.qpos[8:] = HOME_QPOS

    np.copyto(physics.data.ctrl, np.concatenate([HOME_CTRL, HOME_CTRL]))

    for prop_placer in self._all_prop_placers:
      prop_placer(physics, random_state)


class AlohaObservables(composer.Observables):
  """Bimanual ViperX arm obserables."""

  def as_dict(
      self, fully_qualified: bool = False
  ) -> collections.OrderedDict[str, observable.Observable]:
    return super().as_dict(fully_qualified=fully_qualified)

  @define.observable
  def joints_pos(self) -> observable.Observable:
    def _get_joints_pos(physics):
      left_gripper_pos = physics.data.qpos[6]
      right_gripper_pos = physics.data.qpos[14]

      left_gripper_qpos = AlohaTask.convert_gripper(
          left_gripper_pos, 'sim_qpos', 'follower'
      )
      right_gripper_qpos = AlohaTask.convert_gripper(
          right_gripper_pos, 'sim_qpos', 'follower'
      )

      return np.concatenate([
          physics.data.qpos[:6],
          [left_gripper_qpos],
          physics.data.qpos[8:14],
          [right_gripper_qpos],
      ])

    return observable.Generic(_get_joints_pos)

  @define.observable
  def commanded_joints_pos(self) -> observable.Observable:
    """Returns commanded joint positions, for delta and absolute actions."""
    def _get_joints_cmd(physics):
      # Convert sim ctrl values to the environment-level actions
      left_gripper_ctrl = physics.data.ctrl[6]
      right_gripper_ctrl = physics.data.ctrl[13]

      left_gripper_cmd = AlohaTask.convert_gripper(
          left_gripper_ctrl,
          'sim_ctrl',
          'follower',
      )

      right_gripper_cmd = AlohaTask.convert_gripper(
          right_gripper_ctrl,
          'sim_ctrl',
          'follower',
      )
      return np.concatenate([
          physics.data.ctrl[:6],
          [left_gripper_cmd],
          physics.data.ctrl[7:13],
          [right_gripper_cmd],
      ])
    return observable.Generic(_get_joints_cmd)

  @define.observable
  def joints_vel(self) -> observable.Observable:
    return observable.MJCFFeature(
        'qvel',
        [self._entity.mjcf_model.find('joint', name) for name in _ALL_JOINTS],
    )

  @define.observable
  def physics_state(self) -> observable.Observable:
    return observable.Generic(lambda physics: physics.get_state())


class Arena(composer.Arena):
  """Standard Arena for Aloha.

  Forked from dm_control/manipulation/shared/arenas.py
  """

  def __init__(
      self,
      *args,
      camera_resolution,
      table_height_offset=0.0,
      **kwargs,
  ):
    self._camera_resolution = camera_resolution
    self._table_height_offset = table_height_offset
    self.textures = []
    super().__init__(*args, **kwargs)

  def _build(self, name: str | None = None) -> None:
    """Initializes this arena.

    Args:
      name: (optional) A string, the name of this arena. If `None`, use the
        model name defined in the MJCF file.
    """
    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._mjcf_root = mjcf.from_path(
        os.path.join(
            assets_dir,
            'aloha/scene_pbr.xml',
        ),
        escape_separators=True,
    )
    self._mjcf_root.visual.__getattr__('global').offheight = (
        self._camera_resolution[0]
    )
    self._mjcf_root.visual.__getattr__('global').offwidth = (
        self._camera_resolution[1]
    )

    if self._table_height_offset:
      # Shift the heights of the table, worms eye cam, and frame extrusions
      table = self._mjcf_root.find('body', 'table')
      table.pos[2] += self._table_height_offset

      worms_eye_cam = self._mjcf_root.find('camera', 'worms_eye_cam')
      worms_eye_cam.pos[2] += (
          self._table_height_offset
      )

      extrusion_geom_found = False
      for geom in self._mjcf_root.find_all('geom'):
        mesh = geom.__getattr__('mesh')
        if mesh and mesh.name == 'cell_extrusions':
          geom.pos[2] += self._table_height_offset
          extrusion_geom_found = True
          break
      if not extrusion_geom_found:
        raise ValueError('Frame extrusions not found in scene')

