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

"""Drawer task."""

import os

from so101_sim.tasks.base import aloha2_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np


TABLE_HEIGHT = 0.0
RESET_HEIGHT = 0.1

drawer_position = distributions.Uniform(
    low=[-0.15, -0.15, TABLE_HEIGHT + RESET_HEIGHT],
    high=[0.15, 0.15, TABLE_HEIGHT + RESET_HEIGHT],
    single_sample=True,
)
drawer_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 0.25,
        np.pi * 0.25,
        single_sample=True,
    ),
)


class Drawer(aloha2_task.AlohaTask):
  """Drawer task.
  """

  def __init__(
      self,
      **kwargs,
  ):
    """Initializes a new `Drawer` task.

    Args:
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )

    # Add drawer
    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._object_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'edr',
                'drawer/model.xml',
            )
        )
    )
    self._scene.add_free_entity(self._object_prop)

    for prop in [self._object_prop]:
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()

    drawer_pos = drawer_position
    drawer_rot = drawer_rotation

    self._object_container_placers = [
        initializers.PropPlacer(
            props=[self._object_prop],
            position=drawer_pos,
            quaternion=drawer_rot,
            ignore_collisions=True,
            settle_physics=False,
        ),
    ]

    # extra for drawer.
    extra_dofs = 7 + 3
    extra_qpos = np.zeros((extra_dofs,))
    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._object_container_placers:
      prop_placer(physics, random_state)

