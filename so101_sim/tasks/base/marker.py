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

"""A task with a whiteboard marker."""

import os

from so101_sim.tasks.base import aloha2_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np


_TABLE_HEIGHT = 0.0
_PEN_RESET_HEIGHT = 0.055
_TABLE_ID = 'table'
_LEFT_GRIPPER_ID = r'left\gripper_link'
_RIGHT_GRIPPER_ID = r'right\gripper_link'

pen_uniform_position = distributions.Uniform(
    low=[-0.2, -0.2, _TABLE_HEIGHT + _PEN_RESET_HEIGHT],
    high=[0.225, 0.225, _TABLE_HEIGHT + _PEN_RESET_HEIGHT],
    single_sample=True,
)
pen_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 0.4 - np.pi * 0.5 + np.pi,
        np.pi * 0.4 - np.pi * 0.5 + np.pi,
        single_sample=True,
    ),
)


class Marker(aloha2_task.AlohaTask):
  """A task with a whiteboard marker."""

  def __init__(
      self,
      terminate_episode: bool = True,
      **kwargs,
  ):
    """Initializes a new `Marker` task.

    Args:
      terminate_episode: Whether to terminate episodes upon success.
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )

    self._terminate_episode = terminate_episode

    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    marker_path = os.path.join(assets_dir, 'mujoco', 'Marker', 'marker.xml')
    cap_path = os.path.join(assets_dir, 'mujoco', 'Marker', 'cap.xml')

    marker = composer.ModelWrapperEntity(mjcf.from_path(marker_path))
    cap_mjcf = mjcf.from_path(cap_path)
    cap_geoms = cap_mjcf.find_all('geom')
    for geom in cap_geoms:
      if geom.type == 'sphere':
        geom.size = [0.001]
    cap = composer.ModelWrapperEntity(cap_mjcf)
    self._pen_props = [marker, cap]

    additional_joints = 0
    for prop in self._pen_props:
      self._scene.add_free_entity(prop)

      # remove freejoint.
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()

      additional_joints += 7 + len(
          prop.mjcf_model.find_all('joint', exclude_attachments=True)
      )

    # extra for pen.
    extra_qpos = np.zeros((additional_joints,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))
    self._table = self.root_entity.mjcf_model.find('body', _TABLE_ID).find_all(
        'geom'
    )
    self._left_gripper = self.root_entity.mjcf_model.find(
        'body', _LEFT_GRIPPER_ID
    ).find_all('geom')
    self._right_gripper = self.root_entity.mjcf_model.find(
        'body', _RIGHT_GRIPPER_ID
    ).find_all('geom')

    self._pen_base_placer = initializers.PropPlacer(
        props=[self._pen_props[0]],
        position=pen_uniform_position,
        quaternion=pen_z_rotation,
        ignore_collisions=False,
        max_attempts_per_prop=40,
        settle_physics=True,
    )

  def initialize_episode(
      self, physics: mjcf.Physics, random_state: np.random.RandomState
  ) -> None:
    super().initialize_episode(physics, random_state)
    # place the pen base.
    self._pen_base_placer(physics, random_state)
    # place pen lids
    pen_base_pos, pen_base_rot = self._pen_props[0].get_pose(physics)

    for prop in self._pen_props[1:]:
      pen_lid_placer = initializers.PropPlacer(
          props=[prop],
          position=pen_base_pos,
          quaternion=pen_base_rot,
          ignore_collisions=True,  # will collide with the base.
          max_attempts_per_prop=40,
          settle_physics=True,
      )
      pen_lid_placer(physics, random_state)
