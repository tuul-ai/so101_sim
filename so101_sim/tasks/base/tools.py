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

"""Tasks with tools in a tabletop scene."""

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


_TABLE_HEIGHT = 0.0
_RESET_HEIGHT = 0.04

uniform_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(-np.pi, np.pi, single_sample=True),
)


class Tools(aloha2_task.AlohaTask):
  """A task with tools in a tabletop scene."""

  def __init__(
      self,
      **kwargs,
  ):
    """Initializes a Tools scene.

    Inherit from this class to create a new task using the objects in this
    scene.

    Args:
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )

    assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')
    self._caddy_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Sterilite_Caddy_Blue_Sky_17_58_x_12_58_x_9_14/model.xml',
            )
        )
    )

    for mesh in self._caddy_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (0.5, 0.5, 0.5)

    self._screwdriver_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Craftsman_Grip_Screwdriver_Phillips_Cushion/model.xml',
            )
        )
    )

    self._magnifier_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Magnifying_Glassassrt/model.xml',
            )
        )
    )
    self._can_opener_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Granimals_20_Wooden_ABC_Blocks_Wagon/model.xml',
            )
        )
    )
    for mesh in self._can_opener_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (0.8, 0.8, 0.8)

    self._thumb_drive_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Kingston_DT4000MR_G2_Management_Ready_USB_64GB/model.xml',
            )
        )
    )
    for mesh in self._thumb_drive_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (0.9, 0.9, 0.9)

    self._scissors_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                'gso',
                'Diamond_Visions_Scissors_Red/model.xml',
            )
        )
    )

    self._scene.add_free_entity(self._caddy_prop)
    self._scene.add_free_entity(self._screwdriver_prop)
    self._scene.add_free_entity(self._magnifier_prop)
    self._scene.add_free_entity(self._can_opener_prop)
    self._scene.add_free_entity(self._thumb_drive_prop)
    self._scene.add_free_entity(self._scissors_prop)

    for prop in [
        self._caddy_prop,
        self._screwdriver_prop,
        self._magnifier_prop,
        self._can_opener_prop,
        self._thumb_drive_prop,
        self._scissors_prop,
    ]:
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()

    # extra for mug and mugs.
    extra_qpos = np.zeros((7 * 6,))

    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

  def _sample_props(self, physics, random_state):
    reset_z = _TABLE_HEIGHT + _RESET_HEIGHT
    top_left_region = [
        [-0.3, 0.1, reset_z],
        [-0.22, 0.2, reset_z],
    ]
    top_middle_region = [
        [-0.03, 0.1, reset_z],
        [0.03, 0.2, reset_z],
    ]
    top_right_region = [
        [0.22, 0.1, reset_z],
        [0.3, 0.2, reset_z],
    ]
    bottom_left_region = [
        [-0.3, -0.25, reset_z],
        [-0.2, -0.1, reset_z],
    ]
    bottom_middle_region = [
        [-0.05, -0.25, reset_z],
        [0.05, -0.1, reset_z],
    ]
    bottom_right_region = [
        [0.2, -0.25, reset_z],
        [0.3, -0.1, reset_z],
    ]
    side_left_region = [
        [-0.3, 0.25, reset_z],
        [-0.2, 0.35, reset_z],
    ]
    side_middle_region = [
        [-0.05, 0.25, reset_z],
        [0.05, 0.35, reset_z],
    ]
    side_right_region = [
        [0.2, 0.25, reset_z],
        [0.3, 0.35, reset_z],
    ]

    top_left_region_sample = random_state.uniform(
        low=top_left_region[0], high=top_left_region[1]
    )
    top_middle_region_sample = random_state.uniform(
        low=top_middle_region[0], high=top_middle_region[1]
    )
    top_right_region_sample = random_state.uniform(
        low=top_right_region[0], high=top_right_region[1]
    )
    bottom_left_region_sample = random_state.uniform(
        low=bottom_left_region[0], high=bottom_left_region[1]
    )
    bottom_middle_region_sample = random_state.uniform(
        low=bottom_middle_region[0], high=bottom_middle_region[1]
    )
    bottom_right_region_sample = random_state.uniform(
        low=bottom_right_region[0], high=bottom_right_region[1]
    )
    side_left_region_sample = random_state.uniform(
        low=side_left_region[0], high=side_left_region[1]
    )
    side_middle_region_sample = random_state.uniform(
        low=side_middle_region[0], high=side_middle_region[1]
    )
    side_right_region_sample = random_state.uniform(
        low=side_right_region[0], high=side_right_region[1]
    )

    regions = [
        top_left_region_sample,
        top_middle_region_sample,
        top_right_region_sample,
        bottom_left_region_sample,
        bottom_middle_region_sample,
        bottom_right_region_sample,
        side_left_region_sample,
        side_middle_region_sample,
        side_right_region_sample,
    ]

    top_ordering = [0, 1, 2]
    bottom_ordering = [3, 4, 5]
    side_ordering = [6, 7, 8]
    random_state.shuffle(top_ordering)
    random_state.shuffle(bottom_ordering)
    random_state.shuffle(side_ordering)

    if random_state.rand() < 0.5:
      top_ordering, bottom_ordering = bottom_ordering, top_ordering

    return {
        'caddy': regions[top_ordering[0]],
        'screwdriver': regions[top_ordering[1]],
        'magnifier': regions[top_ordering[2]],
        'can_opener': regions[bottom_ordering[0]],
        'thumb_drive': regions[bottom_ordering[1]],
        'scissors': regions[bottom_ordering[2]],
    }

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    positions = self._sample_props(physics, random_state)
    props = [
        self._caddy_prop,
        self._screwdriver_prop,
        self._magnifier_prop,
        self._can_opener_prop,
        self._thumb_drive_prop,
        self._scissors_prop,
    ]
    self._prop_placers = [
        initializers.PropPlacer(
            props=props,
            position=deterministic.Sequence([
                positions['caddy'],
                positions['screwdriver'],
                positions['magnifier'],
                positions['can_opener'],
                positions['thumb_drive'],
                positions['scissors'],
            ]),
            quaternion=deterministic.Sequence([
                rotations.IDENTITY_QUATERNION,
                uniform_z_rotation,
                uniform_z_rotation,
                uniform_z_rotation,
                uniform_z_rotation,
                uniform_z_rotation,
            ]),
            ignore_collisions=False,
            settle_physics=False,
            max_settle_physics_time=2.0,
            max_settle_physics_attempts=20,
        ),
        initializers.PropPlacer(
            props=props,
            position=deterministic.Identity(),
            quaternion=deterministic.Identity(),
            ignore_collisions=True,  # Collisions already resolved.
            settle_physics=True,
        ),
    ]

    for prop_placer in self._prop_placers:
      prop_placer(physics, random_state)
