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

from so101_sim.tasks.base import drawer
from so101_sim.utils import success_detector_utils

OPEN_THRESHOLD = 0.05


class DrawerOpen(drawer.Drawer):
  """Open the bottom drawer of the jewelry box task.
  """

  def get_reward(self, physics):
    bottom_joint = self._object_prop.mjcf_model.find(
        'joint', 'joint_bottom_drawer_low'
    )
    bottom_joint_bound = physics.bind(bottom_joint)
    bottom_drawer_open = bottom_joint_bound.qpos[0] > OPEN_THRESHOLD

    props_moving = success_detector_utils.any_props_moving(
        [self._object_prop], physics)

    if not props_moving and bottom_drawer_open:
      return 1.0
    return 0.0

  def get_instruction(self):
    return 'open the bottom drawer of the jewelry box'

