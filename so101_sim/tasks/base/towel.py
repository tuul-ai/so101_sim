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

"""Task with a towel on the tabletop."""

import os

from so101_sim.tasks.base import aloha2_task
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
import numpy as np


_TABLE_HEIGHT = 0.0
_TOWEL_RESET_HEIGHT = 0.04

towel_uniform_position = distributions.Uniform(
    low=[-0.1, -0.1, _TABLE_HEIGHT + _TOWEL_RESET_HEIGHT],
    high=[0.1, 0.1, _TABLE_HEIGHT + _TOWEL_RESET_HEIGHT],
    single_sample=True,
)
towel_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 0.05,
        np.pi * 0.05,
        single_sample=True,
    ),
)


class Towel(aloha2_task.AlohaTask):
  """A task with a towel on the tabletop."""

  def __init__(
      self,
      **kwargs,
  ):
    """Initializes a new `Towel` task.

    Args:
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )
    self._towel_height = 16
    self._towel_width = 16

    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._towel_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'mujoco',
                'Towel/model.xml',
            )
        )
    )
    self._scene.attach(self._towel_prop)

    # extra qpos for towel.
    extra_qpos = np.zeros((self._towel_height * self._towel_width * 3,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    self._towel_placers = [
        initializers.PropPlacer(
            props=[self._towel_prop],
            position=towel_uniform_position,
            quaternion=towel_z_rotation,
            ignore_collisions=True,
            max_attempts_per_prop=40,
            settle_physics=False,
        )
    ]

  def get_reward(self, physics):
    return 0.0

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._towel_placers:
      prop_placer(physics, random_state)
