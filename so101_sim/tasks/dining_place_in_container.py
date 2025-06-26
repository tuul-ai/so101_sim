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

"""Dining scene with 'place X in Y comtainer' tasks."""

from so101_sim.tasks.base import dining
from so101_sim.utils import oobb_utils
from so101_sim.utils import success_detector_utils
import numpy as np


_ROTATION_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class DiningPlaceInContainer(dining.Dining):
  """Dining scene with 'place X in Y container' task."""

  def __init__(self,
               task_id: str = 'banana',
               **kwargs):
    self._task_id = task_id
    super().__init__(**kwargs)

    self._object_sets = {
        'banana': {
            'object': self._banana_prop,
            'receptacle': self._bowl_prop,
            'reward': 'bbox',
            'instruction': 'put the banana in the bowl',
            'overlap_boxes': [
                oobb_utils.Oobb(
                    position=np.array([-0.017, -0.045, 0.035]) * 1.5,
                    rotation=_ROTATION_IDENTITY,
                    half_extents=np.array([0.02, 0.02, 0.01]) * 1.5,
                ),
            ],
        },
        'pen': {
            'object': self._pen_prop,
            'receptacle': self._container_prop,
            'reward': 'bbox',
            'instruction': 'put the pen in the white cup',
            'overlap_boxes': [
                # Main box, inside the container.
                oobb_utils.Oobb(
                    position=np.array([0.0, 0.0, 0.02666]) * 0.6,
                    rotation=_ROTATION_IDENTITY,
                    half_extents=np.array([0.04666, 0.04666, 0.025]) * 0.6,
                ),
                # Larger box above the container, designed to catch the
                # tail of the pen as it fans out from the top of the container.
                oobb_utils.Oobb(
                    position=np.array([0.0, 0.0, 0.25]) * 0.6,
                    rotation=_ROTATION_IDENTITY,
                    half_extents=np.array([0.1, 0.1, 0.01666]) * 0.6,
                ),
            ],
        },
        'mug': {
            'object': self._mug_prop,
            'receptacle': self._plate_prop,
            'reward': 'contact',
            'instruction': 'put the red mug on the plate',
        },
    }

  def get_reward(self, physics):
    object_set = self._object_sets[self._task_id]
    if self._task_id not in self._object_sets:
      raise ValueError(f'Unknown task ID: {self._task_id}')

    object_prop = object_set['object']
    receptacle_prop = object_set['receptacle']
    reward_type = object_set['reward']

    props_moving = success_detector_utils.any_props_moving(
        [object_prop, receptacle_prop], physics
    )
    if props_moving:
      return 0.0

    if reward_type == 'bbox':
      object_geom_ids = list(
          physics.bind(object_prop.mjcf_model.find_all('geom')).element_id
      )

      def get_body_idx(mjcf_source):
        return physics.bind(
            mjcf_source.mjcf_model.find_all('body')[0]
        ).element_id

      object_oobbs = oobb_utils.get_oobb(
          physics.model,
          physics.data,
          get_body_idx(object_prop),
          object_geom_ids,
      )

      container_body_idx = get_body_idx(receptacle_prop)
      container_pos = physics.data.xpos[container_body_idx]
      container_rot = physics.data.xquat[container_body_idx]

      for container_oobb in object_set['overlap_boxes']:
        container_oobb_ws = oobb_utils.transform_oobb(
            container_oobb, container_pos, container_rot
        )
        if not any(
            oobb_utils.overlap_oobb_oobb(oobb, container_oobb_ws)
            for oobb in object_oobbs
        ):
          return 0.0

      return 1.0

    elif reward_type == 'contact':
      obj_geom_ids = list(
          physics.bind(receptacle_prop.mjcf_model.find_all('geom')).element_id
      )
      recep_geom_ids = list(
          physics.bind(object_prop.mjcf_model.find_all('geom')).element_id
      )

      all_contact_pairs = []
      for contact in physics.data.contact:

        pair = (contact.geom2, contact.geom1)
        all_contact_pairs.append(pair)
        all_contact_pairs.append((contact.geom1, contact.geom2))

      def _touching(geom1_ids, geom2_ids):
        for contact_pair in all_contact_pairs:
          if contact_pair[0] in geom1_ids and contact_pair[1] in geom2_ids:
            return True
          if contact_pair[0] in geom2_ids and contact_pair[1] in geom1_ids:
            return True
        return False

      is_touching = _touching(obj_geom_ids, recep_geom_ids)

      if is_touching:
        return 1.0

      return 0.0

    else:
      raise ValueError(f'Unknown reward type: {reward_type}')

  def get_instruction(self):
    return self._object_sets[self._task_id]['instruction']
