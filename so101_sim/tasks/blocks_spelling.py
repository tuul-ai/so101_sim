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

"""Spelling task with blocks."""

import collections
import os

from so101_sim.tasks.base import aloha2_task
from so101_sim.utils import oobb_utils
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
import numpy as np


_RESET_HEIGHT = 0.1
_ROTATION_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

uniform_position = distributions.Uniform(
    low=[-0.3, -0.3, _RESET_HEIGHT],
    high=[0.3, -0.03, _RESET_HEIGHT],
    single_sample=True,
)
z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi,
        np.pi,
        single_sample=True,
    ),
)
target_pos = [0, 0.15, 0.01]
target_size = [0.3, 0.15, 0.025]
target_oobb = oobb_utils.Oobb(
    position=np.array(target_pos),
    rotation=_ROTATION_IDENTITY,
    half_extents=np.array(target_size),
)


def block_xml_pattern(letter, assets_dir):
  return f"""
    <mujoco model="letter_{letter}">
    <compiler assetdir="{assets_dir}"/>
    <asset>
        <texture type="2d" name="texture" file="letter_{letter}.png"/>
        <material name="texture" texture="texture"/>
      </asset>

      <worldbody>
        <body name="letter_a" pos="0 0 0">
          <geom mass="0.044" condim="6" solimp="2 1 0.001" solref="0.01 1" friction="1 0.005 0.0001" pos="0 0 0" size="0.02 0.02 0.01" type="box" name="letter_a"  material="texture"/>
        </body>
      </worldbody>
    </mujoco>
    """


class BlocksSpelling(aloha2_task.AlohaTask):
  """A task for evaluating spelling with blocks.

  There are multiple blocks with letters on it. The agent must place the target
  letters on top of the table.
  """

  def __init__(
      self,
      letters: str = 'ROBOTAGI',
      target_letters: str = 'AI',
      **kwargs,
  ):
    super().__init__(
        **kwargs,
    )

    self._instruction = ''
    self._letter_props = collections.defaultdict(list)
    self._letters = letters
    self._target_letters = target_letters
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets/letters')
    for l in letters:
      prop = composer.ModelWrapperEntity(
          mjcf.from_xml_string(block_xml_pattern(l, assets_dir))
      )
      self._letter_props[l].append(prop)
      self._scene.add_free_entity(prop)

    props_flatten = []
    for props in self._letter_props.values():
      props_flatten.extend(props)
    extra_qpos = np.zeros((7 * len(props_flatten),))

    self._scene.mjcf_model.worldbody.add(
        'site', type='box', size=target_size, pos=target_pos
    )
    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))

    self._placers = [
        initializers.PropPlacer(
            props=props_flatten,
            position=uniform_position,
            quaternion=z_rotation,
            ignore_collisions=False,
            max_attempts_per_prop=40,
            settle_physics=True,
        )
    ]

  def get_reward(self, physics):
    def get_geom_ids(prop):
      return list(physics.bind(prop.mjcf_model.find_all('geom')).element_id)

    def get_body_idx(prop):
      return physics.bind(prop.mjcf_model.find_all('body')[0]).element_id

    def letter_in_place(l):
      props = self._letter_props[l]
      for prop in props:
        target_tool_geom_ids = get_geom_ids(prop)
        target_tool_oobbs = oobb_utils.get_oobb(
            physics.model,
            physics.data,
            get_body_idx(prop),
            target_tool_geom_ids,
        )
        # if one instance of one letter does not overlap, go to the next
        # instance, otherwise go to the next letter.
        if not any(
            oobb_utils.overlap_oobb_oobb(oobb, target_oobb)
            for oobb in target_tool_oobbs
        ):
          continue
        else:
          return True
      return False

    for l in self._target_letters:
      if not letter_in_place(l):
        return 0.0
    non_target_letters = [
        l for l in self._letters if l not in self._target_letters
    ]
    for l in non_target_letters:
      if letter_in_place(l):
        return 0.0
    return 1.0

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._placers:
      prop_placer(physics, random_state)

  def get_instruction(self) -> str:
    target_letters = ', '.join(self._target_letters)
    return 'put the letters ' + target_letters + ' on the top of the table'
