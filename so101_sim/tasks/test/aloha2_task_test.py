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

import copy
import unittest

from so101_sim.tasks.base import aloha2_task
from dm_control import composer
import numpy as np


class AlohaTaskTest(unittest.TestCase):

  def test_menagerie(self):
    env = self._create_environment(cameras=("overhead_cam",))
    env.reset()
    env.step(np.zeros(env.action_spec().shape))
    env.close()

  def _create_environment(
      self,
      cameras=(),
      table_height_offset=0.0,
      joints_observation_delay_secs=0.0,
      image_observation_delay_secs=0.0,
      **kwargs,
  ):
    task = aloha2_task.AlohaTask(
        control_timestep=0.02,
        cameras=cameras,
        table_height_offset=table_height_offset,
        joints_observation_delay_secs=joints_observation_delay_secs,
        image_observation_delay_secs=image_observation_delay_secs,
        **kwargs,
    )
    return composer.Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed=123),
    )

  def test_action_spec(self):
    env = self._create_environment()
    action_spec = env.action_spec()
    env.close()
    self.assertEqual(action_spec.shape, (14,))
    self.assertEqual(action_spec.dtype, np.float32)

    # Assert that the initial position of the robot is within the action spec
    # limits.
    home_ctrl = aloha2_task.HOME_CTRL
    self.assertTrue(np.all(action_spec.minimum[:6] < home_ctrl[:6]))
    self.assertTrue(np.all(action_spec.minimum[7:13] < home_ctrl[:6]))
    self.assertTrue(np.all(action_spec.maximum[:6] > home_ctrl[:6]))
    self.assertTrue(np.all(action_spec.maximum[7:13] > home_ctrl[:6]))

    home_qpos = aloha2_task.HOME_QPOS
    self.assertTrue(np.all(action_spec.minimum[:6] < home_qpos[:6]))
    self.assertTrue(np.all(action_spec.minimum[7:13] < home_qpos[:6]))
    self.assertTrue(np.all(action_spec.maximum[:6] > home_qpos[:6]))
    self.assertTrue(np.all(action_spec.maximum[7:13] > home_qpos[:6]))

  def test_close_gripper(self):
    env = self._create_environment()
    action_spec = env.action_spec()
    gripper_close = action_spec.minimum[6]
    self.assertEqual(gripper_close, aloha2_task.FOLLOWER_GRIPPER_CLOSE)
    action = np.concatenate([aloha2_task.HOME_CTRL, aloha2_task.HOME_CTRL])
    action[6] = gripper_close
    action[0] = action[7] = 0.1
    env.reset()
    for _ in range(100):
      timestep = env.step(action)

    self.assertEqual(
        env.physics.data.ctrl[6].astype(np.float32),
        aloha2_task.SIM_GRIPPER_CTRL_CLOSE,
    )
    # Note: Looks like the gripper doesn't quite close?
    np.testing.assert_allclose(
        env.physics.data.qpos[6],
        aloha2_task.SIM_GRIPPER_QPOS_CLOSE,
        atol=0.001,
    )
    np.testing.assert_allclose(
        timestep.observation["joints_pos"][6],
        aloha2_task.FOLLOWER_GRIPPER_CLOSE,
        atol=0.01,
    )
    env.close()

  def test_open_gripper(self):
    env = self._create_environment()
    action_spec = env.action_spec()
    gripper_open = action_spec.maximum[6]
    action = np.zeros(env.action_spec().shape)
    action[6] = gripper_open
    env.reset()
    for _ in range(50):
      timestep = env.step(action)
    self.assertGreaterEqual(env.physics.data.qpos[6], 0.035)
    self.assertGreaterEqual(timestep.observation["joints_pos"][6], 1.55)
    self.assertLessEqual(timestep.observation["joints_pos"][6], 1.62)
    env.close()

  def test_table_height_offset_with_pbr_scene(self):
    env = self._create_environment(
        table_height_offset=0.011,
    )
    env.reset()

    np.testing.assert_allclose(
        env.task._scene._mjcf_root.find("body", "table").pos[2],
        -0.721,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        env.task._scene._mjcf_root.find("camera", "worms_eye_cam").pos[2],
        0.061,
        atol=1e-2,
    )
    env.close()

  def test_environment_with_delay(self, delay=0.02):
    task = aloha2_task.AlohaTask(
        control_timestep=0.02,
        cameras=("overhead_cam",),
        image_observation_enabled=True,
        image_observation_delay_secs=delay,  # one timestep (50Hz)
    )
    # add an undelayed camera for comparison
    task._task_observables["undelayed_overhead_camera"] = copy.copy(
        task._task_observables["overhead_cam"]
    )
    task._task_observables["undelayed_overhead_camera"].configure(delay=0)
    task._task_observables["undelayed_overhead_camera"].enabled = True
    env = composer.Environment(
        task,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed=123),
    )
    ts = env.reset()
    initial_physics_state = ts.observation["physics_state"]
    initial_camera_image = ts.observation["undelayed_overhead_camera"]
    assert "physics_state" in ts.observation
    action = np.zeros(env.action_spec().shape)
    ts = env.step(action)

    # after one step
    first_physics_state = ts.observation["physics_state"]
    delayed_camera_image = ts.observation["overhead_cam"]
    np.testing.assert_array_equal(
        ts.observation["delayed_physics_state"], initial_physics_state
    )
    np.testing.assert_array_equal(delayed_camera_image, initial_camera_image)
    ts1 = env.step(action)
    np.testing.assert_array_equal(
        ts1.observation["delayed_physics_state"], first_physics_state
    )
    env.close()

  def test_check_camera_extrinsics(self):
    env = self._create_environment(
        cameras=(
            "overhead_cam",
            "wrist_cam_left",
            "wrist_cam_right",
            "worms_eye_cam",
        ),
    )
    env.reset()
    self.assertEqual(
        env.task._task_observables["wrist_cam_left"]._mjcf_element.pos.tolist(),
        [-0.011, -0.0814748, -0.0095955],
    )
    self.assertEqual(
        env.task._task_observables[
            "wrist_cam_right"
        ]._mjcf_element.pos.tolist(),
        [-0.011, -0.0814748, -0.0095955],
    )
    env.close()

if __name__ == "__main__":
  unittest.main()
