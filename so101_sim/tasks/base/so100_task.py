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

"""SO100 ARM base task which uses a custom SO100 robot model."""

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

# Import calibration system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'scripts'))
from so101_calibration import SO101Calibration


# SO100 ARM home position (matching real robot_descriptions configuration)
SO100_HOME_CTRL: npt.NDArray[float] = np.array(
    [0.0, -1.57079, 1.57079, 1.57079, -1.57079, 0.0]  # rotation, pitch, elbow, wrist_pitch, wrist_roll, jaw
)
SO100_HOME_CTRL.setflags(write=False)

SO100_HOME_QPOS: npt.NDArray[float] = np.array(
    [0.0, -1.57079, 1.57079, 1.57079, -1.57079, 0.0]  # 6 joints matching real robot
)
SO100_HOME_QPOS.setflags(write=False)

# SO100 gripper limits (in meters for jaw opening)
SO100_GRIPPER_QPOS_OPEN: float = 0.08
SO100_GRIPPER_QPOS_CLOSE: float = 0.0

SO100_GRIPPER_CTRL_OPEN: float = 0.08
SO100_GRIPPER_CTRL_CLOSE: float = 0.0

@dataclasses.dataclass(frozen=True)
class GripperLimit:
  """Gripper open and close limit."""
  open: float
  close: float

SO100_GRIPPER_LIMITS = immutabledict.immutabledict({
    'sim_qpos': GripperLimit(
        open=SO100_GRIPPER_QPOS_OPEN,
        close=SO100_GRIPPER_QPOS_CLOSE,
    ),
    'sim_ctrl': GripperLimit(
        open=SO100_GRIPPER_CTRL_OPEN,
        close=SO100_GRIPPER_CTRL_CLOSE,
    ),
})

_DEFAULT_PHYSICS_DELAY_SECS: float = 0.3
_DEFAULT_JOINT_OBSERVATION_DELAY_SECS: float = 0.1
_DEFAULT_TABLE_HEIGHT_OFFSET: float = 0.0

# SO100 ARM joint names
_SO100_JOINTS: tuple[str, ...] = (
    'so100/rotation',
    'so100/pitch', 
    'so100/elbow',
    'so100/wrist_pitch',
    'so100/wrist_roll',
    'so100/jaw_left',
    'so100/jaw_right',
)

class GeomGroup(enum.IntFlag):
  NONE = 0
  ARM = enum.auto()
  GRIPPER = enum.auto()
  TABLE = enum.auto()
  OBJECT = enum.auto()

class SO100Task(composer.Task):
  """The base SO100 ARM task."""

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
      rotation_joint_limit: float = np.pi,
      terminate_episode=True,
  ):
    """Initializes a new SO100 task.

    Args:
      control_timestep: Control timestep in seconds.
      cameras: Default cameras to use.
      camera_resolution: Camera resolution for rendering.
      joints_observation_delay_secs: Delay of joints observation.
      image_observation_enabled: Whether to enable image observations.
      image_observation_delay_secs: Delay of image observations.
      update_interval: Simulation steps between observation updates.
      table_height_offset: Offset to table height in meters.
      rotation_joint_limit: Joint limit for rotation joint in radians.
      terminate_episode: Whether to terminate episode when task succeeds.
    """

    self._rotation_joint_limit = rotation_joint_limit
    self._terminate_episode = terminate_episode

    # Initialize calibration system
    self._calibration = SO101Calibration()
    
    # Apply calibration to home positions
    self._calibrated_home_ctrl = self._calibration.apply_calibration_to_position(SO100_HOME_CTRL)
    self._calibrated_home_qpos = self._calibration.apply_calibration_to_position(SO100_HOME_QPOS)
    

    self._scene = SO100Arena(
        camera_resolution=camera_resolution,
        table_height_offset=table_height_offset,
    )
    self._scene.mjcf_model.option.flag.multiccd = 'enable'
    self._scene.mjcf_model.option.noslip_iterations = 0

    self.control_timestep = control_timestep

    self._joints = []
    for name in _SO100_JOINTS:
      joint = self._scene.mjcf_model.find('joint', name)
      if joint is not None:
        self._joints.append(joint)

    # Add custom camera observable
    obs_dict = collections.OrderedDict()

    shared_delay = variation_broadcaster.VariationBroadcaster(
        image_observation_delay_secs / self.physics_timestep
    )
    cameras_entities = [
        self.root_entity.mjcf_model.find('camera', name) for name in cameras
    ]
    for camera_entity in cameras_entities:
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

    so100_observables = SO100Observables(self.root_entity)
    so100_observables.enable_all()
    obs_dict.update(so100_observables.as_dict())
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

    self._task_observables['physics_state'].enabled = image_observation_enabled
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
              ignore_collisions=True,
              settle_physics=True,
          )
      )

  @property
  def root_entity(self) -> composer.Entity:
    return self._scene

  @property
  def task_observables(self) -> Mapping[str, observable.Observable]:
    return dict(**self._task_observables)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    # 0-4: SO100 arm joints (rotation, pitch, elbow, wrist_pitch, wrist_roll)
    # 5: SO100 gripper
    minimum = physics.model.actuator_ctrlrange[:, 0].astype(np.float32)
    maximum = physics.model.actuator_ctrlrange[:, 1].astype(np.float32)
    
    # Apply custom rotation joint limits
    minimum[0] = -self._rotation_joint_limit
    maximum[0] = self._rotation_joint_limit

    # Gripper limits
    minimum[5] = SO100_GRIPPER_LIMITS['sim_ctrl'].close
    maximum[5] = SO100_GRIPPER_LIMITS['sim_ctrl'].open

    return specs.BoundedArray(
        shape=(6,),
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
    from_limits = SO100_GRIPPER_LIMITS[from_name]
    to_limits = SO100_GRIPPER_LIMITS[to_name]
    return (gripper_value - from_limits.close) / (
        from_limits.open - from_limits.close
    ) * (to_limits.open - to_limits.close) + to_limits.close

  def before_step(
      self,
      physics: mjcf.Physics,
      action: npt.ArrayLike,
      random_state: np.random.RandomState,
  ) -> None:
    # Apply calibration to action
    action_array = np.array(action)
    calibrated_action = self._calibration.apply_calibration_to_action(action_array)
    
    arm_joints_new = calibrated_action[:5]
    gripper_action = calibrated_action[5]

    # Apply calibrated arm joint actions
    np.copyto(physics.data.ctrl[:5], arm_joints_new)

    # Handle gripper action
    gripper_cmd = np.array([gripper_action])
    gripper_ctrl = SO100Task.convert_gripper(
        gripper_cmd, 'sim_ctrl', 'sim_ctrl'  # For now, no conversion needed
    )
    np.copyto(physics.data.ctrl[5:6], gripper_ctrl)

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
    
    if len(self._joints) > 0:
      arm_joints_bound = physics.bind(self._joints)
      if len(arm_joints_bound.qpos) == len(self._calibrated_home_qpos):
        arm_joints_bound.qpos[:] = self._calibrated_home_qpos
      elif len(arm_joints_bound.qpos) >= len(self._calibrated_home_ctrl):
        arm_joints_bound.qpos[:len(self._calibrated_home_ctrl)] = self._calibrated_home_ctrl
    
    # Set ctrl directly using available actuators
    if physics.data.ctrl.size >= len(self._calibrated_home_ctrl):
      np.copyto(physics.data.ctrl[:len(self._calibrated_home_ctrl)], self._calibrated_home_ctrl)

    for prop_placer in self._all_prop_placers:
      prop_placer(physics, random_state)


class SO100Observables(composer.Observables):
  """SO100 ARM observables."""

  def as_dict(
      self, fully_qualified: bool = False
  ) -> collections.OrderedDict[str, observable.Observable]:
    return super().as_dict(fully_qualified=fully_qualified)

  @define.observable
  def joints_pos(self) -> observable.Observable:
    def _get_joints_pos(physics):
      gripper_pos = physics.data.qpos[5]  # left jaw position
      gripper_qpos = SO100Task.convert_gripper(
          gripper_pos, 'sim_qpos', 'sim_ctrl'
      )
      return np.concatenate([
          physics.data.qpos[:5],  # arm joints
          [gripper_qpos],         # converted gripper position
      ])
    return observable.Generic(_get_joints_pos)

  @define.observable
  def commanded_joints_pos(self) -> observable.Observable:
    def _get_joints_cmd(physics):
      gripper_ctrl = physics.data.ctrl[5]
      gripper_cmd = SO100Task.convert_gripper(
          gripper_ctrl, 'sim_ctrl', 'sim_ctrl'
      )
      return np.concatenate([
          physics.data.ctrl[:5],  # arm joint commands
          [gripper_cmd],          # gripper command
      ])
    return observable.Generic(_get_joints_cmd)

  @define.observable
  def joints_vel(self) -> observable.Observable:
    joints = []
    for name in _SO100_JOINTS:
      joint = self._entity.mjcf_model.find('joint', name)
      if joint is not None:
        joints.append(joint)
    return observable.MJCFFeature('qvel', joints)

  @define.observable
  def physics_state(self) -> observable.Observable:
    return observable.Generic(lambda physics: physics.get_state())


class SO100Arena(composer.Arena):
  """Standard Arena for SO100 ARM."""

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
    """Initializes this arena."""
    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._mjcf_root = mjcf.from_path(
        os.path.join(
            assets_dir,
            'so100/scene_pbr.xml',
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
      # Shift the height of the table
      table = self._mjcf_root.find('body', 'table')
      table.pos[2] += self._table_height_offset