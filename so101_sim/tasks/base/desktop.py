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

"""Tasks desktop items on the tabletop."""

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
_LAPTOP_RESET_HEIGHT = 0.055
_QVEL_TOL = 1e-3

laptop_uniform_position = distributions.Uniform(
    low=[0.0, 0.25, _TABLE_HEIGHT + _LAPTOP_RESET_HEIGHT],
    high=[0.0, 0.20, _TABLE_HEIGHT + _LAPTOP_RESET_HEIGHT],
    single_sample=True,
)
laptop_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        - np.pi * 5 / 180,
        np.pi * 5 / 180,
        single_sample=True,
    ),
)


class Desktop(aloha2_task.AlohaTask):
  """A task desktop items (laptop and headphone) on the tabletop."""

  def __init__(
      self,
      **kwargs,
  ):
    """Initializes a new `DesktopTask` task.

    Args:
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )
    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._laptop_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'mujoco',
                'Laptop/model.xml',
            )
        )
    )
    self._scene.add_free_entity(self._laptop_prop)

    freejoint = traversal_utils.get_freejoint(
        self._laptop_prop.mjcf_model.find_all('body')[0]
    )
    if freejoint:
      freejoint.remove()

    # extra joints for laptop.
    additional_joints = len(
        self._laptop_prop.mjcf_model.find_all('joint', exclude_attachments=True)
    )
    extra_qpos = np.zeros((7 + additional_joints,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    self._laptop_placers = [
        initializers.PropPlacer(
            props=[self._laptop_prop],
            position=laptop_uniform_position,
            quaternion=laptop_z_rotation,
            ignore_collisions=False,
            max_attempts_per_prop=40,
            settle_physics=False,
        )
    ]

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._laptop_placers:
      prop_placer(physics, random_state)
