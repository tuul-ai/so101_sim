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

"""Closing a laptop on a tabletop."""

from so101_sim.tasks.base import laptop
from so101_sim.utils import success_detector_utils


_QVEL_TOL = 1e-2
_LAPTOP_CLOSED_JOINT_POS = -0.01
_LAPTOP_INITIAL_JOINT_POS = -1.83


class LaptopClose(laptop.Laptop):
  """Closing a laptop."""

  def get_reward(self, physics):
    hinge_joint = self._laptop_prop.mjcf_model.find('joint', 'screen_hinge')
    hinge_joint_bound = physics.bind(hinge_joint)
    laptop_closed = hinge_joint_bound.qpos[0] > _LAPTOP_CLOSED_JOINT_POS

    props_moving = success_detector_utils.any_props_moving(
        [self._laptop_prop], physics
    )

    if not props_moving and laptop_closed:
      return 1.0
    return 0.0

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)

    # set initial joint positions.
    hinge_joint = self._laptop_prop.mjcf_model.find('joint', 'screen_hinge')
    hinge_joint_bound = physics.bind(hinge_joint)
    hinge_joint_bound.qpos[0] = _LAPTOP_INITIAL_JOINT_POS

  def get_instruction(self):
    return 'close the laptop'
