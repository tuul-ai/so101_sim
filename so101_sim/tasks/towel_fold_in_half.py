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

"""Folding a towel in half."""

from so101_sim.tasks.base import towel
import numpy as np


_EDGE_DISTANCE_THRESHOLD = 0.05


class TowelFoldInHalf(towel.Towel):
  """A task that folds a towel in half."""

  def get_reward(self, physics):
    bottom_left_edge = self._towel_prop.mjcf_model.find(
        'body',
        f'towel_{0}'
    )
    bottom_right_edge = self._towel_prop.mjcf_model.find(
        'body',
        f'towel_{self._towel_height * self._towel_width - self._towel_height}'
    )
    top_left_edge = self._towel_prop.mjcf_model.find(
        'body',
        f'towel_{self._towel_width - 1}'
    )
    top_right_edge = self._towel_prop.mjcf_model.find(
        'body',
        f'towel_{self._towel_width * self._towel_height - 1}'
    )

    bl_pos = physics.data.xipos[physics.bind(bottom_left_edge).element_id]
    br_pos = physics.data.xipos[physics.bind(bottom_right_edge).element_id]
    tl_pos = physics.data.xipos[physics.bind(top_left_edge).element_id]
    tr_pos = physics.data.xipos[physics.bind(top_right_edge).element_id]

    # check if top edges are close to bottom edges
    if (
        np.linalg.norm(tl_pos - bl_pos) < _EDGE_DISTANCE_THRESHOLD and
        np.linalg.norm(tr_pos - br_pos) < _EDGE_DISTANCE_THRESHOLD
    ):
      return 1.0

    return 0.0

  def get_instruction(self):
    return 'fold the pink cloth from top to bottom vertically'
