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

"""Remove lid from the marker task."""

from so101_sim.tasks.base import marker
from dm_control import composer
from dm_control import mjcf
import numpy as np


_PEN_AWAY_THRESHOLD = 0.05
_ZERO_LINEAR_VELOCITY_TOLERANCE = 0.02
_ZERO_ANGULAR_VELOCITY_TOLERANCE = 0.04


class MarkerRemoveLid(marker.Marker):
  """A task for removing the lid from a marker pen."""

  def __init__(
      self,
      terminate_episode: bool = True,
      **kwargs,
  ):
    """Initializes a new `MarkerRemoveLid` task.

    Args:
      terminate_episode: Whether to terminate episodes upon success.
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(terminate_episode=terminate_episode, **kwargs)

  def _get_geometry_ids(
      self, geom: mjcf.Element, physics: mjcf.Physics
  ) -> list[np.int64]:
    # element id could be an array or a single element.
    element_id = physics.bind(geom).element_id
    return [element_id] if isinstance(element_id, int) else list(element_id)

  def _get_all_elements_in_contact_with_prop(
      self, physics: mjcf.Physics, prop: composer.ModelWrapperEntity
  ) -> list[np.int64]:
    # element id could be a list or a single element.
    prop_ = set(
        self._get_geometry_ids(prop.mjcf_model.find_all('geom'), physics)
    )
    # contact.geom contains a list of [id1, id2], where id1 and id2 are the
    # geom ids that are in contact. These are not duplicates, i.e. it could be
    # either [id1, id2] or [id2, id1]
    prop_contact_pairs = [
        pair
        for pair in physics.data.contact.geom
        if pair[0] in prop_ or pair[1] in prop_
    ]
    # return the list of elements that are in contact with the prop. These
    # could be the first or the second element in the pair, depending on where
    # the prop id is located in the pair.
    return [
        pair[0] if pair[1] in prop_ else pair[1] for pair in prop_contact_pairs
    ]

  def _is_object_on_table(
      self, physics: mjcf.Physics, object_contacts: list[np.int64]
  ) -> bool:
    table_ids = self._get_geometry_ids(self._table, physics)
    return any(table_id in object_contacts for table_id in table_ids)

  def _is_object_being_grasped(
      self, physics: mjcf.Physics, object_contacts: list[np.int64]
  ) -> bool:
    gripper_ids = self._get_geometry_ids(
        self._left_gripper, physics
    ) + self._get_geometry_ids(self._right_gripper, physics)
    return any(gripper_id in object_contacts for gripper_id in gripper_ids)

  def _is_object_moving(
      self, physics: mjcf.Physics, prop: composer.ModelWrapperEntity
  ) -> bool:
    twist = prop.get_velocity(physics)
    return (
        np.linalg.norm(twist[:3]) > _ZERO_LINEAR_VELOCITY_TOLERANCE
        or np.linalg.norm(twist[3:]) > _ZERO_ANGULAR_VELOCITY_TOLERANCE
    )

  def _is_lid_removed(self, physics: mjcf.Physics) -> bool:
    # if pen has multiple lids, any can be removed.
    base_pos, _ = self._pen_props[0].get_pose(physics)
    for prop in self._pen_props[1:]:
      lid_pos, _ = prop.get_pose(physics)
      if np.linalg.norm(lid_pos - base_pos) > _PEN_AWAY_THRESHOLD:
        return True
    return False

  def get_reward(self, physics: mjcf.Physics) -> float:
    success = self._is_lid_removed(physics)
    if not success:
      return 0.0
    for prop in self._pen_props:
      success = success and not self._is_object_moving(physics, prop)
      object_contacts = self._get_all_elements_in_contact_with_prop(
          physics, prop
      )
      success = success and self._is_object_on_table(physics, object_contacts)
      success = success and not self._is_object_being_grasped(
          physics, object_contacts
      )
    return 1.0 if success else 0.0

  def get_instruction(self) -> str:
    return 'remove the cap from the marker'
