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

r"""Example to run one episode of Gemini Robotics On-Device Sim Eval and save the video.

"""

import time

from absl import app
from so101_sim import task_suite
import mediapy
import numpy as np

try:
    from safari_sdk.model import gemini_robotics_policy
    from safari_sdk.model import genai_robotics
    SAFARI_SDK_AVAILABLE = True
except ImportError:
    # safari_sdk is not available (e.g., on macOS)
    raise ImportError(
        "safari_sdk is required for run_eval.py but is not available. "
        "On macOS, safari_sdk cannot be installed due to Linux-specific dependencies. "
        "This script requires access to Gemini Robotics models."
    )


_DT = 0.02
_IMAGE_SIZE = (480, 848)
_ALOHA_CAMERAS = {
    'overhead_cam': _IMAGE_SIZE,
    'worms_eye_cam': _IMAGE_SIZE,
    'wrist_cam_left': _IMAGE_SIZE,
    'wrist_cam_right': _IMAGE_SIZE,
}
_ALOHA_JOINTS = {'joints_pos': 14}
_INIT_ACTION = np.asarray([
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,
    1.5,
    0.0,
    -0.96,
    1.16,
    0.0,
    -0.3,
    0.0,
    1.5,
])
_SERVE_ID = 'gemini_robotics_on_device'
_NUM_EPISODES_PER_TASK = 5
_PRINT_TIMES = True

SERVER_ADDRESS = 'localhost:60061'
SERVICE_NAME = 'gemini_robotics'
METHOD_NAME = 'sample_actions_json_flat'
FULL_METHOD_NAME = f'/{SERVICE_NAME}/{METHOD_NAME}'


def run_episode(task_name, ep_idx, env):
  """Runs an episode of the given task."""
  print('Task: ', task_name, ' Episode: ', ep_idx)
  print('Homing...')
  timestep = env.reset()
  frames = []

  # Instantiate the policy.
  try:
    print('Creating policy...')
    policy = gemini_robotics_policy.GeminiRoboticsPolicy(
        serve_id=_SERVE_ID,
        task_instruction=env.task.get_instruction(),
        inference_mode=gemini_robotics_policy.InferenceMode.SYNCHRONOUS,
        cameras=_ALOHA_CAMERAS,
        joints=_ALOHA_JOINTS,
        min_replan_interval=25,
        robotics_api_connection=genai_robotics.RoboticsApiConnectionType.LOCAL,
    )
    policy.setup()  # Initialize the policy
    print('GeminiRoboticsPolicy initialized successfully.')
  except ValueError as e:
    print(f'Error initializing policy: {e}')
    raise
  except Exception as e:  # pylint: disable=broad-except
    print(f'An unexpected error occurred during initialization: {e}')
    raise

  print('Running policy...')
  query_time = 0
  step_time = 0
  i = 0
  while not timestep.last():
    i += 1
    frame_start_time = time.time()
    action = policy.step(timestep.observation)

    query_end_time = time.time()
    query_time += query_end_time - frame_start_time

    timestep = env.step(action=action)
    step_time += time.time() - query_end_time
    if i % 100 == 0:
      if _PRINT_TIMES:
        print(
            f'Step: {i}, Query time: {query_time / 100.}, Step time:'
            f' {step_time / 100.}',
            end='\r',
        )
      query_time = 0
      step_time = 0

    frames.append(timestep.observation['overhead_cam'])
  if timestep.reward >= 1.0:
    print('\nEpisode success.')
  else:
    print('\nEpisode failure.')
  success = timestep.reward >= 1.0
  success_str = 'succ' if success else 'fail'
  video_path = f'/tmp/{task_name}_ep{ep_idx}_{success_str}.mp4'
  print('Saving video to ', video_path)
  mediapy.write_video(video_path, frames)
  return success


def main(_):
  success_rates = {}
  for task_name in task_suite.TASK_FACTORIES.keys():
    success_count = 0
    for ep_idx in range(_NUM_EPISODES_PER_TASK):
      env = task_suite.create_task_env(task_name, time_limit=80.0)
      success = run_episode(task_name, ep_idx, env)
      if success:
        success_count += 1
    success_rates[task_name] = success_count / _NUM_EPISODES_PER_TASK
    print(f'----- Task: {task_name}, Success rate: {success_rates[task_name]}')

  print('All Task Success Rates:')
  for task_name, success_rate in success_rates.items():
    print(f'{task_name}:\t{success_rate}')


if __name__ == '__main__':
  app.run(main)
