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

"""Put a specific tool in a specific caddy compartment task."""

from so101_sim.tasks.base import tools
from so101_sim.utils import oobb_utils
from so101_sim.utils import success_detector_utils
import numpy as np


_ROTATION_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
_LEFT_COMPARTMENT_OOBB = [
    oobb_utils.Oobb(
        position=np.array([-0.032, 0.0, -0.01]),
        rotation=_ROTATION_IDENTITY,
        half_extents=np.array([0.015, 0.047, 0.025]),
    ),
]
_RIGHT_COMPARTMENT_OOBB = [
    oobb_utils.Oobb(
        position=np.array([0.032, 0.0, -0.01]),
        rotation=_ROTATION_IDENTITY,
        half_extents=np.array([0.015, 0.047, 0.025]),
    ),
]


class ToolsInCaddy(tools.Tools):
  """Put X tool in Y compartment of the caddy task."""

  def __init__(
      self,
      target_tool: str = 'screwdriver',
      target_compartment: str = 'left',
      **kwargs
    ):
    self._target_tool = target_tool
    self._target_compartment = target_compartment
    super().__init__(**kwargs)

  def get_reward(self, physics):
    def get_geom_ids(prop):
      return list(physics.bind(prop.mjcf_model.find_all('geom')).element_id)

    def get_body_idx(prop):
      return physics.bind(prop.mjcf_model.find_all('body')[0]).element_id

    tool_props = {
        'screwdriver': self._screwdriver_prop,
        'magnifier': self._magnifier_prop,
        'can_opener': self._can_opener_prop,
        'thumb_drive': self._thumb_drive_prop,
        'scissors': self._scissors_prop,
    }

    if self._target_tool == 'screwdriver':
      target_prop = tool_props[self._target_tool]
    elif self._target_tool == 'magnifier':
      target_prop = tool_props[self._target_tool]
    elif self._target_tool == 'can_opener':
      target_prop = tool_props[self._target_tool]
    elif self._target_tool == 'thumb_drive':
      target_prop = tool_props[self._target_tool]
    elif self._target_tool == 'scissors':
      target_prop = tool_props[self._target_tool]
    else:
      raise ValueError(
          f'Unknown target tool: {self._target_tool}. Options are: screwdriver,'
          ' magnifier, can_opener, thumb_drive, scissors.'
      )
    target_tool_geom_ids = get_geom_ids(target_prop)

    props_moving = success_detector_utils.any_props_moving(
        [self._caddy_prop] + list(tool_props.values()), physics
    )

    if props_moving:
      return 0.0

    target_tool_oobbs = oobb_utils.get_oobb(
        physics.model,
        physics.data,
        get_body_idx(target_prop),
        target_tool_geom_ids,
    )
    # transform oobbs into the space of each container oobb
    container_body_idx = get_body_idx(self._caddy_prop)
    container_pos = physics.data.xpos[container_body_idx]
    container_rot = physics.data.xquat[container_body_idx]

    if self._target_compartment == 'left':
      for container_oobb in _LEFT_COMPARTMENT_OOBB:
        container_oobb_ws = oobb_utils.transform_oobb(
            container_oobb, container_pos, container_rot
        )
        if not any(
            oobb_utils.overlap_oobb_oobb(oobb, container_oobb_ws)
            for oobb in target_tool_oobbs
        ):
          return 0.0
    elif self._target_compartment == 'right':
      for container_oobb in _RIGHT_COMPARTMENT_OOBB:
        container_oobb_ws = oobb_utils.transform_oobb(
            container_oobb, container_pos, container_rot
        )
        if not any(
            oobb_utils.overlap_oobb_oobb(oobb, container_oobb_ws)
            for oobb in target_tool_oobbs
        ):
          return 0.0
    else:
      raise ValueError(
          f'Unknown target compartment: {self._target_compartment}. Options'
          ' are: left, right.'
      )

    # Check that no other tool is in the any compartment
    for _, other_tool_prop in tool_props.items():
      if other_tool_prop == target_prop:
        continue
      other_tool_geom_ids = get_geom_ids(other_tool_prop)
      other_tool_oobbs = oobb_utils.get_oobb(
          physics.model,
          physics.data,
          get_body_idx(other_tool_prop),
          other_tool_geom_ids,
      )
      for container_oobb in _LEFT_COMPARTMENT_OOBB + _RIGHT_COMPARTMENT_OOBB:
        container_oobb_ws = oobb_utils.transform_oobb(
            container_oobb, container_pos, container_rot
        )
        if any(
            oobb_utils.overlap_oobb_oobb(oobb, container_oobb_ws)
            for oobb in other_tool_oobbs
        ):
          return 0.0

    return 1.0

  def get_instruction(self):
    return (
        f'place the {self._target_tool.replace("_", " ")} in the'
        + f' {self._target_compartment} compartment of the caddy'
    )
