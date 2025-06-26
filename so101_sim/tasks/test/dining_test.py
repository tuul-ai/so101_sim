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

"""Tests hand over."""

import os
import unittest

from so101_sim.tasks import dining_place_in_container
from dm_control import composer
import numpy as np
from PIL import Image


class DiningPlaceInContainerTests(unittest.TestCase):

  def test_dining_place_in_container_banana(self):
    task = dining_place_in_container.DiningPlaceInContainer(
        task_id="banana",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "banana"
    env = composer.Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed=123),
    )
    env.reset()
    env.step(np.zeros(env.action_spec().shape))
    ref_img = env.physics.render(
        camera_id="overhead_cam", height=480, width=848
    )
    tmp_dir = os.path.join(
        "/tmp",
        f"dining_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_dining_place_in_container_pen(self):
    task = dining_place_in_container.DiningPlaceInContainer(
        task_id="pen",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "pen"
    env = composer.Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed=123),
    )
    env.reset()
    env.step(np.zeros(env.action_spec().shape))
    ref_img = env.physics.render(
        camera_id="overhead_cam", height=480, width=848
    )
    tmp_dir = os.path.join(
        "/tmp",
        f"dining_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_dining_place_in_container_mug(self):
    task = dining_place_in_container.DiningPlaceInContainer(
        task_id="mug",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "mug"
    env = composer.Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed=123),
    )
    env.reset()
    env.step(np.zeros(env.action_spec().shape))
    ref_img = env.physics.render(
        camera_id="overhead_cam", height=480, width=848
    )
    tmp_dir = os.path.join(
        "/tmp",
        f"dining_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")


if __name__ == "__main__":
  unittest.main()
