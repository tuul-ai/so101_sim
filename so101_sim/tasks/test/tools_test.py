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

from so101_sim.tasks import tools_in_caddy
from dm_control import composer
import numpy as np
from PIL import Image


class DiningPlaceInContainerTests(unittest.TestCase):

  def test_tools_in_caddy_screwdriver_left(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="screwdriver",
        target_compartment="left",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "screwdriver_left"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_screwdriver_right(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="screwdriver",
        target_compartment="right",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "screwdriver_right"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_magnifier_left(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="magnifier",
        target_compartment="left",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "magnifier_left"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_magnifier_right(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="magnifier",
        target_compartment="right",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "magnifier_right"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_can_opener_left(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="can_opener",
        target_compartment="left",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "can_opener_left"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_can_opener_right(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="can_opener",
        target_compartment="right",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "can_opener_right"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_thumb_drive_left(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="thumb_drive",
        target_compartment="left",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "thumb_drive_left"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_thumb_drive_right(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="thumb_drive",
        target_compartment="right",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "thumb_drive_right"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_scissors_left(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="scissors",
        target_compartment="left",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "scissors_left"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")

  def test_tools_in_caddy_scissors_right(self):
    task = tools_in_caddy.ToolsInCaddy(
        target_tool="scissors",
        target_compartment="right",
        control_timestep=0.02,
        cameras=([]),
    )
    task_name = "scissors_right"
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
        f"tools_in_caddy_{task_name}.png",
    )
    Image.fromarray(ref_img).save(tmp_dir, format="png")


if __name__ == "__main__":
  unittest.main()
