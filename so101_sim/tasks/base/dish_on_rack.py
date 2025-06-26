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

"""Disk on rack task."""

import copy
import os

from so101_sim.tasks.base import aloha2_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np


TABLE_HEIGHT = 0.0
_RACK_RESET_HEIGHT = 0.03
_DISH_RESET_HEIGHT = 0.04

_QVEL_TOL = 1e-3


rack_uniform_position = distributions.Uniform(
    low=[-0.1, 0.18, TABLE_HEIGHT + _RACK_RESET_HEIGHT],
    high=[0.1, 0.25, TABLE_HEIGHT + _RACK_RESET_HEIGHT],
    single_sample=True,
)
rack_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 0.02 - np.pi * 0.5,
        np.pi * 0.02 - np.pi * 0.5,
        single_sample=True,
    ),
)

dish_uniform_position = distributions.Uniform(
    low=[0, -0.2, TABLE_HEIGHT + _DISH_RESET_HEIGHT],
    high=[0.25, 0, TABLE_HEIGHT + _DISH_RESET_HEIGHT],
    single_sample=True,
)


class DishOnRack(aloha2_task.AlohaTask):
  """Put a dish on a rack."""

  def __init__(
      self,
      dish_path,
      **kwargs,
  ):
    """Initializes a new `DishOnRack` task.

    Args:
      dish_path: Path to asset of the dish.
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )

    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._rack_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Rubbermaid_Large_Drainer/model.xml',
            )
        )
    )
    self._scene.add_free_entity(self._rack_prop)

    self._dish_prop = composer.ModelWrapperEntity(mjcf.from_path(dish_path))
    for mesh in self._dish_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (0.7, 0.7, 0.7)
    self._scene.add_free_entity(self._dish_prop)

    for prop in [self._dish_prop, self._rack_prop]:
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()

    dish_uniform_position_copy = copy.deepcopy(dish_uniform_position)
    dish_uniform_position_copy.low[2] += self._scene._table_height_offset
    dish_uniform_position_copy.high[2] += self._scene._table_height_offset

    rack_uniform_position_copy = copy.deepcopy(rack_uniform_position)
    rack_uniform_position_copy.low[2] += self._scene._table_height_offset
    rack_uniform_position_copy.high[2] += self._scene._table_height_offset

    self._rack_dish_placers = [
        initializers.PropPlacer(
            props=[self._rack_prop],
            position=rack_uniform_position_copy,
            quaternion=rack_z_rotation,
            ignore_collisions=True,
            settle_physics=False,
        ),
        initializers.PropPlacer(
            props=[self._dish_prop],
            position=dish_uniform_position_copy,
            ignore_collisions=False,
            max_attempts_per_prop=40,
            settle_physics=False,
        ),
        initializers.PropPlacer(
            props=[self._rack_prop, self._dish_prop],
            position=deterministic.Identity(),
            quaternion=deterministic.Identity(),
            ignore_collisions=True,  # Collisions already resolved.
            settle_physics=True,
        ),
    ]

    # extra for dish and rack.
    extra_qpos = np.zeros((14,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._rack_dish_placers:
      prop_placer(physics, random_state)
