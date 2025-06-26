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

"""SO100 Hand over task."""

import dataclasses
import os

from so101_sim.tasks.base import so100_task
from so101_sim.utils import oobb_utils
from so101_sim.utils import success_detector_utils
from dm_control import composer
from dm_control import mjcf
from dm_control.composer import initializers
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.mjcf import traversal_utils
import numpy as np

ROTATION_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])

TABLE_HEIGHT = 0.4  # SO100 table height
RESET_HEIGHT = 0.05

object_position = distributions.Uniform(
    low=[0.2, -0.1, TABLE_HEIGHT + RESET_HEIGHT],
    high=[0.3, 0.1, TABLE_HEIGHT + RESET_HEIGHT],
    single_sample=True,
)
object_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0.0, 0.0, 1.0),
    angle=distributions.Uniform(
        -np.pi * 0.1,
        np.pi * 0.1,
        single_sample=True,
    ),
)

container_position = distributions.Uniform(
    low=[-0.3, -0.1, TABLE_HEIGHT + RESET_HEIGHT],
    high=[-0.2, 0.1, TABLE_HEIGHT + RESET_HEIGHT],
    single_sample=True,
)


@dataclasses.dataclass(frozen=True)
class SO100HandOverTaskConfig:
  """Configuration for a SO100 handover task.

  Attributes:
    object_model: file path pointing to xml model of the object.
    container_model: file path pointing to xml model of the container.
    container_mesh_scale: how much to scale the container size.
    success_threshold: threshold for distance between the object center of mass
      and the container center of mass for the task to be considered success.
    overlap_boxes: a list of overlap boxes to be used for overlap reward.
    instruction: language goal for the task.
  """

  object_model: str
  container_model: str
  container_mesh_scale: float
  success_threshold: float
  overlap_boxes: list[oobb_utils.Oobb]
  instruction: str


SO100_HANDOVER_CONFIGS = {
    # Pick up a banana and put it in a bowl.
    'banana': SO100HandOverTaskConfig(
        object_model='ycb/011_banana/google_64k/model.xml',
        container_model='ycb/024_bowl/google_64k/model.xml',
        container_mesh_scale=1.5,
        success_threshold=0.1,
        overlap_boxes=[
            oobb_utils.Oobb(
                position=np.array([-0.017, -0.045, 0.035]) * 1.5,
                rotation=ROTATION_IDENTITY,
                half_extents=np.array([0.02, 0.02, 0.01]) * 1.5,
            ),
        ],
        instruction='pick up the banana and put it in the bowl using the SO100 arm',
    ),
    # Pick up a pen and put it in container.
    'pen': SO100HandOverTaskConfig(
        object_model='edr/pen/model.xml',
        container_model='gso/BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028/model.xml',
        container_mesh_scale=0.6,
        success_threshold=0.05,
        overlap_boxes=[
            # Main box, inside the container.
            oobb_utils.Oobb(
                position=np.array([0.0, 0.0, 0.02666]) * 0.6,
                rotation=ROTATION_IDENTITY,
                half_extents=np.array([0.04666, 0.04666, 0.025]) * 0.6,
            ),
            # Larger box above the container.
            oobb_utils.Oobb(
                position=np.array([0.0, 0.0, 0.25]) * 0.6,
                rotation=ROTATION_IDENTITY,
                half_extents=np.array([0.1, 0.1, 0.01666]) * 0.6,
            ),
        ],
        instruction='pick up the pen and put it in the container using the SO100 arm',
    ),
}


class SO100HandOver(so100_task.SO100Task):
  """SO100 Hand over task.

  The goal is to pick up the object with the SO100 arm gripper and place it
  in a container.
  """

  def __init__(
      self,
      object_name,
      reward_based_on_overlap=True,
      **kwargs,
  ):
    """Initializes a new `SO100HandOver` task.

    Args:
      object_name: str specifying the name of the object to be picked up.
      reward_based_on_overlap: Whether to reward based on overlap between the
        object and the container.
      **kwargs: Additional args to pass to the base class.
    """
    super().__init__(
        **kwargs,
    )

    if object_name not in SO100_HANDOVER_CONFIGS.keys():
      raise ValueError(
          f'Invalid object name: {object_name}, must be one of'
          f' {SO100_HANDOVER_CONFIGS.keys()}'
      )
    task_config = SO100_HANDOVER_CONFIGS[object_name]
    self._container_mesh_scale = task_config.container_mesh_scale
    self._dist_threshold = task_config.success_threshold
    self._overlap_boxes = task_config.overlap_boxes
    self._instruction = task_config.instruction
    object_path = task_config.object_model
    container_path = task_config.container_model

    # Adds object
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    self._object_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                object_path
            )
        )
    )
    self._scene.add_free_entity(self._object_prop)

    # Adds container
    self._container_prop = composer.ModelWrapperEntity(
        mjcf.from_path(
            os.path.join(
                assets_dir,
                container_path,
            )
        )
    )
    if object_name == 'banana':
      # Change bowl color to blue
      container_tex = self._container_prop.mjcf_model.asset.find(
          'texture', 'texture'
      )
      container_tex.file = 'bowl_blue_color.png'

    for mesh in self._container_prop.mjcf_model.find_all('mesh'):
      mesh.scale = (
          self._container_mesh_scale,
          self._container_mesh_scale,
          self._container_mesh_scale,
      )
    container_body = self._container_prop.mjcf_model.find_all('body')[0]
    for geom in container_body.find_all('site'):
      if geom.type == 'box':
        geom.size = geom.size * self._container_mesh_scale
        geom.pos = geom.pos * self._container_mesh_scale

    self._scene.add_free_entity(self._container_prop)

    for prop in [self._container_prop] + [self._object_prop]:
      freejoint = traversal_utils.get_freejoint(
          prop.mjcf_model.find_all('body')[0]
      )
      if freejoint:
        freejoint.remove()

    self._object_container_placers = [
        initializers.PropPlacer(
            props=[self._object_prop],
            position=object_position,
            quaternion=object_rotation,
            ignore_collisions=True,
            settle_physics=False,
        ),
        initializers.PropPlacer(
            props=[self._container_prop],
            position=container_position,
            ignore_collisions=False,
            settle_physics=False,
        ),
        initializers.PropPlacer(
            props=[self._object_prop, self._container_prop],
            position=deterministic.Identity(),
            quaternion=deterministic.Identity(),
            ignore_collisions=True,  # Collisions already resolved.
            settle_physics=True,
        ),
    ]

    # extra for object.
    extra_qpos = np.zeros((14,))
    scene_key = self.root_entity.mjcf_model.find('key', 'neutral_pose')
    if scene_key:
      scene_key.qpos = np.copy(np.concatenate([scene_key.qpos, extra_qpos]))
    self._reward_based_on_overlap = reward_based_on_overlap

  def get_reward(self, physics):
    object_geom_ids = list(
        physics.bind(self._object_prop.mjcf_model.find_all('geom')).element_id
    )

    props_moving = success_detector_utils.any_props_moving(
        [self._object_prop, self._container_prop], physics)

    if self._reward_based_on_overlap:
      if props_moving:
        return 0.0

      def get_body_idx(mjcf_source):
        return physics.bind(
            mjcf_source.mjcf_model.find_all('body')[0]
        ).element_id
      object_oobbs = oobb_utils.get_oobb(
          physics.model,
          physics.data,
          get_body_idx(self._object_prop),
          object_geom_ids,
      )
      # transform oobbs into the space of each container oobb
      container_body_idx = get_body_idx(self._container_prop)
      container_pos = physics.data.xpos[container_body_idx]
      container_rot = physics.data.xquat[container_body_idx]

      for container_oobb in self._overlap_boxes:
        container_oobb_ws = oobb_utils.transform_oobb(
            container_oobb, container_pos, container_rot
        )
        if not any(
            oobb_utils.overlap_oobb_oobb(oobb, container_oobb_ws)
            for oobb in object_oobbs
        ):
          return 0.0

      return 1.0

    # Simple distance-based reward as fallback
    container_geom_ids = list(
        physics.bind(
            self._container_prop.mjcf_model.find_all('geom')
        ).element_id
    )
    so100_gripper_geom_ids = list(
        physics.bind(
            self.root_entity.mjcf_model.find(
                'body', 'so100/hand_link'
            ).find_all('geom')
        ).element_id
    )

    all_contact_pairs = []
    for contact in physics.data.contact:
      pair = (contact.geom2, contact.geom1)
      all_contact_pairs.append(pair)
      all_contact_pairs.append((contact.geom1, contact.geom2))
    def _touching(geom1_ids, geom2_ids):
      for contact_pair in all_contact_pairs:
        if contact_pair[0] in geom1_ids and contact_pair[1] in geom2_ids:
          return True
        if contact_pair[0] in geom2_ids and contact_pair[1] in geom1_ids:
          return True
      return False

    # Check if object is in container
    object_in_container = False
    object_pos = self._object_prop.get_pose(physics)[0][:2]
    container_pos = self._container_prop.get_pose(physics)[0][:2]
    if np.linalg.norm(container_pos - object_pos) < self._dist_threshold:
      object_in_container = True

    if (
        not props_moving
        and _touching(object_geom_ids, container_geom_ids)
        and object_in_container
    ):
      return 1.0
    
    return 0.0

  def initialize_episode(self, physics, random_state):
    super().initialize_episode(physics, random_state)
    for prop_placer in self._object_container_placers:
      prop_placer(physics, random_state)

  def get_instruction(self):
    return self._instruction