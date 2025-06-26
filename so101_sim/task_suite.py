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

"""Creates so101_sim task environments using dm_control.composer."""

import inspect
from so101_sim.tasks import blocks_spelling
from so101_sim.tasks import bowl_on_rack
from so101_sim.tasks import desktop_wrap_headphone
from so101_sim.tasks import dining_place_in_container
from so101_sim.tasks import drawer_open
from so101_sim.tasks import hand_over
from so101_sim.tasks import laptop_close
from so101_sim.tasks import marker_remove_lid
from so101_sim.tasks import tools_in_caddy
from so101_sim.tasks import towel_fold_in_half
from so101_sim.tasks import so100_hand_over
from dm_control import composer
import immutabledict
import numpy as np


DEFAULT_CAMERAS = (
    'overhead_cam',
    'worms_eye_cam',
    'wrist_cam_left',
    'wrist_cam_right',
)

DEFAULT_CONTROL_TIMESTEP = 0.02

TASK_FACTORIES = immutabledict.immutabledict({
    'BlocksSpelling': (blocks_spelling.BlocksSpelling, {}),
    'BowlOnRack': (bowl_on_rack.BowlOnRack, {}),
    'DesktopWrapHeadphone': (desktop_wrap_headphone.DesktopWrapHeadphone, {}),
    'DiningPlaceBananaInBowl': (
        dining_place_in_container.DiningPlaceInContainer,
        {'task_id': 'banana'},
    ),
    'DiningPlacePenInContainer': (
        dining_place_in_container.DiningPlaceInContainer,
        {'task_id': 'pen'},
    ),
    'DiningPlaceMugOnPlate': (
        dining_place_in_container.DiningPlaceInContainer,
        {'task_id': 'mug'},
    ),
    'DrawerOpen': (drawer_open.DrawerOpen, {}),
    'HandOverPen': (hand_over.HandOver, {'object_name': 'pen'}),
    'HandOverBanana': (hand_over.HandOver, {'object_name': 'banana'}),
    'LaptopClose': (laptop_close.LaptopClose, {}),
    'MarkerRemoveLid': (marker_remove_lid.MarkerRemoveLid, {}),
    'ToolsPlaceScrewdriverInLeftCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'screwdriver', 'target_compartment': 'left'},
    ),
    'ToolsPlaceScrewdriverInRightCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'screwdriver', 'target_compartment': 'right'},
    ),
    'ToolsPlaceMagnifierInLeftCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'magnifier', 'target_compartment': 'left'},
    ),
    'ToolsPlaceMagnifierInRightCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'magnifier', 'target_compartment': 'right'},
    ),
    'ToolsPlaceCanOpenerInLeftCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'can_opener', 'target_compartment': 'left'},
    ),
    'ToolsPlaceCanOpenerInRightCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'can_opener', 'target_compartment': 'right'},
    ),
    'ToolsPlaceScissorsInLeftCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'scissors', 'target_compartment': 'left'},
    ),
    'ToolsPlaceScissorsInRightCompartment': (
        tools_in_caddy.ToolsInCaddy,
        {'target_tool': 'scissors', 'target_compartment': 'right'},
    ),
    'TowelFoldInHalf': (towel_fold_in_half.TowelFoldInHalf, {}),
    # SO100 ARM tasks
    'SO100HandOverPen': (so100_hand_over.SO100HandOver, {'object_name': 'pen'}),
    'SO100HandOverBanana': (so100_hand_over.SO100HandOver, {'object_name': 'banana'}),
})


def create_task_env(
    task_name: str,
    time_limit: float,
    random_state: np.random.RandomState | int | None = None,
    control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    **kwargs,
) -> composer.Environment:
  """Creates an Aloha sim task environment.

  Args:
      task_name: The registered name of the task to create/
      time_limit: Time limit per episode in seconds.
      random_state: Random seed for the environment.
      control_timestep: Control timestep for the task.
      cameras: Tuple of camera names to use.
      **kwargs: Extra kwargs passed to the environment.

  Returns:
      A configured dm_control.composer.Environment.
  Raises:
      ValueError: If the task_name is not recognized.
  """
  if task_name not in TASK_FACTORIES:
    raise ValueError(
        f'Unknown task_name: {task_name}. Available tasks:'
        f' {list(TASK_FACTORIES.keys())}'
    )

  task_class, task_kwargs = TASK_FACTORIES[task_name]

  # remove any kwargs that are not in the task constructor
  signature = inspect.signature(task_class.__init__)
  task_class_kwargs = set(signature.parameters.keys())
  task_class_kwargs.remove('self')
  kwargs = {k: v for k, v in kwargs.items() if k in task_class_kwargs}
  constructor_kwargs = {
      'control_timestep': control_timestep,
      'cameras': cameras,
      **task_kwargs,
  }
  kwargs.update(constructor_kwargs)

  task_instance = task_class(**kwargs)

  return composer.Environment(
      task_instance,
      random_state=random_state,
      time_limit=time_limit,
      strip_singleton_obs_buffer_dim=True,
      raise_exception_on_physics_error=False,
      delayed_observation_padding=composer.ObservationPadding.INITIAL_VALUE,
  )
