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

"""Wrap the wire around a headphone on a desktop setup."""

import os

from so101_sim.tasks.base import desktop
from so101_sim.utils import oobb_utils
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
from dm_control.utils import transformations
import numpy as np


_TABLE_HEIGHT = 0.0
_LIFT_HEIGHT = 0.08
_HEADPHONE_RESET_HEIGHT = -0.1
_LAPTOP_INITIAL_JOINT_POS = -1.83

BOX_ROTATION = np.array([0.92387977, 0.38268343, 0.0, 0.0])
TOP_OOBB = [
    oobb_utils.Oobb(
        position=np.array([0.0, 0.0, 0.12]),
        rotation=BOX_ROTATION,
        half_extents=np.array([0.015, 0.08, 0.02]),
    ),
]
BEHIND_OOBB = [
    oobb_utils.Oobb(
        position=np.array([0.03, 0.0, 0.06]),
        rotation=BOX_ROTATION,
        half_extents=np.array([0.015, 0.1, 0.02]),
    ),
]
FRONT_OOBB = [
    oobb_utils.Oobb(
        position=np.array([-0.03, 0.0, 0.06]),
        rotation=BOX_ROTATION,
        half_extents=np.array([0.015, 0.1, 0.02]),
    ),
]

headphone_uniform_position = distributions.Uniform(
    low=[-0.2, -0.25, _TABLE_HEIGHT + _HEADPHONE_RESET_HEIGHT],
    high=[0.2, 0.0, _TABLE_HEIGHT + _HEADPHONE_RESET_HEIGHT],
    single_sample=True,
)
headphone_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 30 / 180,
        np.pi * 30 / 180,
        single_sample=True,
    ),
)


class DesktopWrapHeadphone(desktop.Desktop):
  """Wrapping the wire around a headphone on a desktop setup."""

  def __init__(
      self,
      **kwargs,
  ):
    """Initializes a new `DesktopWrapHeadphone` task.

    Args:
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(**kwargs)

    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    self._headphone_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Razer_Kraken_Pro_headset_Full_size_Black_with_cable/model.xml',
            )
        )
    )
    self._scene.add_free_entity(self._headphone_prop)

    freejoint = traversal_utils.get_freejoint(
        self._headphone_prop.mjcf_model.find_all('body')[0]
    )
    if freejoint:
      freejoint.remove()

    additional_joints = 19 * 4  # 19 capsule links for cable
    extra_qpos = np.zeros((7 + additional_joints,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    self._headphone_placers = [
        initializers.PropPlacer(
            props=[self._headphone_prop],
            position=headphone_uniform_position,
            quaternion=headphone_z_rotation,
            ignore_collisions=False,
            max_attempts_per_prop=40,
            settle_physics=False,
        )
    ]

    self._task_state = 'start'

  def get_reward(self, physics):
    headphone_body = self._headphone_prop.mjcf_model.find('body', 'headphone')
    headphone_body_idx = physics.bind(headphone_body).element_id

    # headphone should be above the table.
    headphone_pos = physics.data.xipos[headphone_body_idx]
    if headphone_pos[2] < _LIFT_HEIGHT:
      self._task_state = 'start'
      return 0.0

    # check if any cable part intersects with the top, front, and behind oobbs.
    self._task_state = 'lifted_headphone'
    cable_parts = [
        body
        for body in self._headphone_prop.mjcf_model.find_all('body')
        if 'B_' in body.name
    ]

    bbox_rot_mat = transformations.quat_to_mat(BOX_ROTATION)[:3, :3]
    headphone_rot_mat = (
        physics.data.ximat[headphone_body_idx].reshape(3, 3) @ bbox_rot_mat
    )
    headphone_rot_quat = transformations.mat_to_quat(headphone_rot_mat)

    cable_oobbs = []
    for cable_part in cable_parts:
      cable_geom_ids = list(
          physics.bind(cable_part.find_all('geom')).element_id
      )
      cable_oobbs.extend(
          oobb_utils.get_oobb(
              physics.model,
              physics.data,
              physics.bind(cable_part).element_id,
              cable_geom_ids,
          )
      )

    top_oobbs = []
    for top_oobb in TOP_OOBB:
      top_oobbs.append(
          oobb_utils.transform_oobb(
              top_oobb,
              headphone_pos,
              headphone_rot_quat,
          )
      )

    behind_oobbs = []
    for behind_oobb in BEHIND_OOBB:
      behind_oobbs.append(
          oobb_utils.transform_oobb(
              behind_oobb,
              headphone_pos,
              headphone_rot_quat,
          )
      )

    front_oobbs = []
    for front_oobb in FRONT_OOBB:
      front_oobbs.append(
          oobb_utils.transform_oobb(
              front_oobb,
              headphone_pos,
              headphone_rot_quat,
          )
      )

    cable_top = any(
        any(
            oobb_utils.overlap_oobb_oobb(cable_oobb, top_oobb)
            for top_oobb in top_oobbs
        )
        for cable_oobb in cable_oobbs
    )
    cable_behind = any(
        any(
            oobb_utils.overlap_oobb_oobb(cable_oobb, behind_oobb)
            for behind_oobb in behind_oobbs
        )
        for cable_oobb in cable_oobbs
    )
    cable_front = any(
        any(
            oobb_utils.overlap_oobb_oobb(cable_oobb, front_oobb)
            for front_oobb in front_oobbs
        )
        for cable_oobb in cable_oobbs
    )

    if cable_top and cable_behind and cable_front:
      return 1.0

    return 0.0

  def get_instruction(self):
    if self._task_state == 'lifted_headphone':
      return 'wind the cable around the headphones'
    else:
      return 'pick up the headphones'

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._headphone_placers:
      prop_placer(physics, random_state)

    # set initial joint positions.
    hinge_joint = self._laptop_prop.mjcf_model.find('joint', 'screen_hinge')
    hinge_joint_bound = physics.bind(hinge_joint)
    hinge_joint_bound.qpos[0] = _LAPTOP_INITIAL_JOINT_POS

