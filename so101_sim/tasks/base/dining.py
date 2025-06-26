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

"""Tasks in a dining scene."""

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


_TABLE_HEIGHT = 0.03
_RESET_HEIGHT = 0.03

uniform_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(-np.pi, np.pi, single_sample=True),
)


class Dining(aloha2_task.AlohaTask):
  """A task set in a dining scene."""

  def __init__(
      self,
      **kwargs,
  ):
    """Initializes a Dining scene.

    Inherit from this class to create a new task using the objects in this
    scene.

    Args:
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )

    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._mug_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'ycb',
                '025_mug/google_64k/model.xml',
            )
        )
    )

    self._plate_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'ycb',
                '029_plate/google_64k/model.xml',
            )
        )
    )
    plate_tex = self._plate_prop.mjcf_model.asset.find('texture', 'texture')
    plate_tex.file = '1k_textures/plate_blue_color.png'

    for mesh in self._plate_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (0.8, 0.8, 0.8)

    self._pen_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'edr',
                'pen/model.xml',
            )
        )
    )
    self._banana_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'ycb',
                '011_banana/google_64k/model.xml',
            )
        )
    )
    self._bowl_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'ycb',
                '024_bowl/google_64k/model.xml',
            )
        )
    )
    bowl_tex = self._bowl_prop.mjcf_model.asset.find('texture', 'texture')
    bowl_tex.file = 'bowl_blue_color.png'
    for mesh in self._bowl_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (1.5, 1.5, 1.5)

    self._container_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028/model.xml',
            )
        )
    )

    for mesh in self._container_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (0.6, 0.6, 0.6)

    self._scene.add_free_entity(self._mug_prop)
    self._scene.add_free_entity(self._pen_prop)
    self._scene.add_free_entity(self._banana_prop)
    self._scene.add_free_entity(self._plate_prop)
    self._scene.add_free_entity(self._bowl_prop)
    self._scene.add_free_entity(self._container_prop)

    self._props = [
        self._plate_prop,
        self._mug_prop,
        self._pen_prop,
        self._banana_prop,
        self._bowl_prop,
        self._container_prop,
    ]

    for prop in self._props:
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()
      # Place all props at the origin.
      body = prop.mjcf_model.find_all('body')[0]
      if body:
        body.pos = [0, 0, 0]

    # extra for mug and mugs.
    extra_qpos = np.zeros((7 * 6,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

  def _sample_props(self, random_state):
    top_left_region = [
        [-0.3, 0.1, _TABLE_HEIGHT + _RESET_HEIGHT],
        [-0.22, 0.2, _TABLE_HEIGHT + _RESET_HEIGHT],
    ]
    top_middle_region = [
        [-0.03, 0.1, _TABLE_HEIGHT + _RESET_HEIGHT],
        [0.03, 0.2, _TABLE_HEIGHT + _RESET_HEIGHT],
    ]
    top_right_region = [
        [0.22, 0.1, _TABLE_HEIGHT + _RESET_HEIGHT],
        [0.3, 0.2, _TABLE_HEIGHT + _RESET_HEIGHT],
    ]
    bottom_left_region = [
        [-0.3, -0.25, _TABLE_HEIGHT + _RESET_HEIGHT],
        [-0.2, -0.1, _TABLE_HEIGHT + _RESET_HEIGHT],
    ]
    bottom_middle_region = [
        [-0.05, -0.25, _TABLE_HEIGHT + _RESET_HEIGHT],
        [0.05, -0.1, _TABLE_HEIGHT + _RESET_HEIGHT],
    ]
    bottom_right_region = [
        [0.2, -0.25, _TABLE_HEIGHT + _RESET_HEIGHT],
        [0.3, -0.1, _TABLE_HEIGHT + _RESET_HEIGHT],
    ]

    top_left_sample = random_state.uniform(
        low=top_left_region[0], high=top_left_region[1]
    )
    top_middle_sample = random_state.uniform(
        low=top_middle_region[0], high=top_middle_region[1]
    )
    top_right_sample = random_state.uniform(
        low=top_right_region[0], high=top_right_region[1]
    )
    bottom_left_sample = random_state.uniform(
        low=bottom_left_region[0], high=bottom_left_region[1]
    )
    bottom_middle_sample = random_state.uniform(
        low=bottom_middle_region[0], high=bottom_middle_region[1]
    )
    bottom_right_sample = random_state.uniform(
        low=bottom_right_region[0], high=bottom_right_region[1]
    )

    regions = [
        top_left_sample,
        top_middle_sample,
        top_right_sample,
        bottom_left_sample,
        bottom_middle_sample,
        bottom_right_sample,
    ]

    top_ordering = [0, 1, 2]
    bottom_ordering = [3, 4, 5]
    random_state.shuffle(top_ordering)
    random_state.shuffle(bottom_ordering)

    return {
        'plate': regions[top_ordering[0]],
        'bowl': regions[top_ordering[1]],
        'container': regions[top_ordering[2]],
        'mug': regions[bottom_ordering[0]],
        'pen': regions[bottom_ordering[1]],
        'banana': regions[bottom_ordering[2]],
    }

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    positions = self._sample_props(random_state)
    prop_list = [
        self._plate_prop,
        self._bowl_prop,
        self._container_prop,
        self._mug_prop,
        self._pen_prop,
        self._banana_prop,
    ]
    prop_placers = [
        initializers.PropPlacer(
            props=prop_list,
            position=deterministic.Sequence([
                positions['plate'],
                positions['bowl'],
                positions['container'],
                positions['mug'],
                positions['pen'],
                positions['banana'],
            ]),
            quaternion=uniform_z_rotation,
            ignore_collisions=True,
            settle_physics=False,
            max_settle_physics_attempts=20,
        ),
        initializers.PropPlacer(
            props=self._props,
            position=deterministic.Identity(),
            quaternion=deterministic.Identity(),
            ignore_collisions=True,  # Collisions already resolved.
            settle_physics=True,
        ),
    ]

    for prop_placer in prop_placers:
      prop_placer(physics, random_state)
