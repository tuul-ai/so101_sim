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

"""Tests laptop tasks."""

import os
import unittest

from so101_sim.tasks import laptop_close
from dm_control import composer
import numpy as np
from PIL import Image


class LaptopTests(unittest.TestCase):

  def test_laptop_close(self):
    task = laptop_close.LaptopClose(
        control_timestep=0.02,
        cameras=([]),
    )
    env = composer.Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed=123),
    )
    for i in range(5):
      env.reset()
      env.step(np.zeros(env.action_spec().shape))
      ref_img = env.physics.render(
          camera_id="overhead_cam", height=480, width=848
      )
      tmp_dir = os.path.join(
          "/tmp",
          f"laptop_close_{i}.png",
      )
      Image.fromarray(ref_img).save(tmp_dir, format="png")

if __name__ == "__main__":
  unittest.main()
