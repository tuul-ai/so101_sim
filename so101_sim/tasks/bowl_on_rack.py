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

"""Putting a bowl on to rack."""

import os

from so101_sim.tasks.base import dish_on_rack
from so101_sim.utils import success_detector_utils
import numpy as np


class BowlOnRack(dish_on_rack.DishOnRack):
  """Putting a bowl on a rack."""

  def __init__(
      self,
      requires_centering=False,
      **kwargs,
  ):
    """Initializes a new `BowlOnRack` task.

    Args:
      requires_centering: Whether the rack must be centered.
      **kwargs: Additional args to pass to the base class.
    """
    self._requires_centering = requires_centering

    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    bowl_path = os.path.join(
        assets_dir,
        'ycb',
        '024_bowl/google_64k/model.xml',
    )
    super().__init__(
        bowl_path,
        **kwargs,
    )

    # Changes the bowl color to pink
    dish_tex = self._dish_prop.mjcf_model.asset.find(
        'texture', 'texture'
    )
    dish_tex.file = 'bowl_pink_color.png'

  def get_reward(self, physics):
    bowl_geom_ids = list(
        physics.bind(self._dish_prop.mjcf_model.find_all('geom')).element_id
    )
    rack_geom_ids = list(
        physics.bind(self._rack_prop.mjcf_model.find_all('geom')).element_id
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

    is_touching = _touching(bowl_geom_ids, rack_geom_ids)

    props_moving = success_detector_utils.any_props_moving(
        [self._dish_prop, self._rack_prop], physics)
    # Bowl is on rack
    bowl_on_rack = False
    bowl_z = self._dish_prop.get_pose(physics)[0][2]
    rack_z = self._rack_prop.get_pose(physics)[0][2]
    if bowl_z > rack_z + 0.02:
      bowl_on_rack = True

    success = is_touching and not props_moving and bowl_on_rack

    # Rack is in center
    if self._requires_centering:
      rack_pose = self._rack_prop.get_pose(physics)[0][:2]
      in_center = False
      # If we're within 10cm from the center
      if (np.abs(rack_pose) <= np.array([0.1, 0.1])).all():
        in_center = True

      success = success and in_center

    if success:
      return 1.0

    return 0.0

  def get_instruction(self):
    return 'put the bowl on the rack'

