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

"""A package that provides utilities for extracting OOBB from Mujoco."""

import dataclasses
from typing import Iterable

import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True)
class Aabb:
  """A class representing an oriented bounding box (OOBB)."""

  min3: np.ndarray
  max3: np.ndarray


@dataclasses.dataclass(frozen=True)
class Oobb:
  """A class representing an oriented bounding box (OOBB).

  Attributes:
    position: The position of the OOBB.
    rotation: The rotation of the OOBB.
    half_extents: The half extents of the OOBB.
  """
  position: np.ndarray
  rotation: np.ndarray
  half_extents: np.ndarray


def get_aabb_from_vertices(pts: np.ndarray) -> Aabb:
  """Get the AABB from a set of points."""
  return Aabb(min3=np.min(pts, axis=0), max3=np.max(pts, axis=0))


def get_vertices_aabb(aabb: Aabb) -> np.ndarray:
  """Get the corners of an AABB."""
  def lerp3(a: np.ndarray, b: np.ndarray, i: np.ndarray) -> np.ndarray:
    """3d linear interpolation between a and b."""
    return a * (np.array([1.0] * 3) - i) + b * i

  pts = []
  for i in range(8):
    iz = i // 4
    ixy = i % 4
    pts.append(lerp3(aabb.min3, aabb.max3, np.array([ixy % 2, ixy // 2, iz])))
  return np.array(pts)


def aabb_to_oobb(aabb: Aabb) -> Oobb:
  """Make an oriented bounds from an axis-aligned bounding box."""
  return Oobb(
      position=(aabb.min3 + aabb.max3) / 2,
      rotation=np.array([1.0, 0.0, 0.0, 0.0]),
      half_extents=(aabb.max3 - aabb.min3) / 2,
  )


def _rotate_vector3(rotation: np.ndarray, vector3: np.ndarray) -> np.ndarray:
  """Rotate a point about the origin by the given rotation.

  Args:
    rotation: A quaternion representing the rotation to apply.
    vector3: The 3d vector to rotate.

  Returns:
    The rotated point.
  """
  rotated_point = np.array([0.0] * 3)
  mujoco.mju_rotVecQuat(rotated_point, vector3, rotation)
  return rotated_point


def _transform_point(
    position: np.ndarray, rotation: np.ndarray, point: np.ndarray
):
  """Transform a point by a translation and rotation.

  Args:
    position: The 3d vector translation to apply.
    rotation: The quaternion rotation to apply.
    point: The point to transform.

  Returns:
    The transformed point.
  """
  return position + _rotate_vector3(rotation, point)


def get_vertices_oobb(oobb: Oobb) -> np.ndarray:
  """Get the corners of an OOBB. These are the aabb vertices, transformed."""

  pts = get_vertices_aabb(Aabb(-oobb.half_extents, oobb.half_extents))
  return np.array(
      [_transform_point(oobb.position, oobb.rotation, v) for v in pts]
  )


def get_oobb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_idx: int,
    geoms: Iterable[int] | None = None,
) -> list[Oobb]:
  """Get a good oriented bounding box for the given body.

  Args:
    model: The Mujoco model.
    data: The Mujoco data.
    body_idx: The index of the body to get the OOBB for.
    geoms: The geoms to get the OOBB for. If None, we manually pull all the
      geoms associated with the body.

  Returns:
    A list of OOBBs for the given body. We take an oobb for each child geom.
  """
  # This is complicated. What we really want is for Mujoco to give us a nice,
  # single oobb in world-space, but instead we have to work hard, reproducing
  # the code here:
  # https://github.com/google-deepmind/mujoco/blob/56bb3ca5de8512715b88cf12bb3c4d99c58e610c/src/engine/engine_vis_visualize.c#L643

  bvh_idx = model.body_bvhadr[body_idx]
  bvh_child = model.bvh_child[bvh_idx]

  def get_oobb_impl(xpos: np.ndarray, xmat: np.ndarray, oobb_cs: np.ndarray):
    rotation = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_mat2Quat(rotation, xmat)

    bounds_center_rotated = np.array([0.0] * 3)
    mujoco.mju_rotVecQuat(bounds_center_rotated, oobb_cs[:3], rotation)
    translate = bounds_center_rotated + xpos

    return Oobb(position=translate, rotation=rotation, half_extents=oobb_cs[3:])

  geoms_to_use = (
      geoms
      if geoms is not None
      else [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_idx]
  )
  leaf = bvh_child[0] == -1 and bvh_child[1] == -1
  if leaf:
    return [
        get_oobb_impl(
            data.geom_xpos[g],
            data.geom_xmat[g],
            model.geom_aabb[g],
        )
        for g in geoms_to_use
    ]
  else:
    return [
        get_oobb_impl(
            data.xipos[body_idx],
            data.ximat[body_idx],
            model.bvh_aabb[bvh_idx],
        )
    ]


def transform_oobb(
    oobb: Oobb, translation: np.ndarray, rotation: np.ndarray
) -> Oobb:
  """Apply a standard translation/rotation transform to an OOBB.

  First we apply the rotation transform to the position and rotation, then we
  apply the translation transform.

  Args:
    oobb: The OOBB to transform.
    translation: The translation to apply.
    rotation: The rotation to apply.

  Returns:
    The transformed OOBB.
  """
  rotated_local_pos = np.array([0.0] * 3)
  mujoco.mju_rotVecQuat(rotated_local_pos, oobb.position, rotation)
  new_rotation = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mju_mulQuat(new_rotation, rotation, oobb.rotation)
  return Oobb(
      position=translation + rotated_local_pos,
      rotation=new_rotation,
      half_extents=oobb.half_extents,
  )


def overlap_aabb_oobb(aabb: Aabb, oobb: Oobb) -> bool:
  """Returns true if the aabb and oobb overlap.

  Note that we expect the aabb and oobb to be in the same coordinate system.
  We will use SAT to determine if they overlap.
  https://programmerart.weebly.com/separating-axis-theorem.html

  Args:
    aabb: The AABB.
    oobb: The OOBB. Must be in the same coordinate system as the AABB.

  Returns:
    True if the aabb and oobb overlap, False otherwise.
  """

  # For all 6 axes, we will project the points onto the 1d lines, checking for
  # overlaps/separations.

  aabb_vertices = get_vertices_aabb(aabb)
  oobb_vertices = get_vertices_oobb(oobb)

  def has_separating_axis(axis: np.ndarray) -> bool:
    pts0_max = max(np.dot(v, axis) for v in aabb_vertices)
    pts0_min = min(np.dot(v, axis) for v in aabb_vertices)
    pts1_max = max(np.dot(v, axis) for v in oobb_vertices)
    pts1_min = min(np.dot(v, axis) for v in oobb_vertices)
    return pts0_max < pts1_min or pts0_min > pts1_max

  unit_x = np.array([1.0, 0.0, 0.0])
  unit_y = np.array([0.0, 1.0, 0.0])
  unit_z = np.array([0.0, 0.0, 1.0])

  # aabb axes.
  if has_separating_axis(unit_x):
    return False
  if has_separating_axis(unit_y):
    return False
  if has_separating_axis(unit_z):
    return False
  if has_separating_axis(_rotate_vector3(oobb.rotation, unit_x)):
    return False
  if has_separating_axis(_rotate_vector3(oobb.rotation, unit_y)):
    return False
  if has_separating_axis(_rotate_vector3(oobb.rotation, unit_z)):
    return False

  return True


def overlap_oobb_oobb(oobb0: Oobb, oobb1: Oobb) -> bool:
  """Test for overlap between two OOBBs.

  First we transform everything so oobb0 is an Aabb symmetric around the origin.
  Then we test for overlap with that Aabb.

  Args:
    oobb0: The first OOBB.
    oobb1: The second OOBB.

  Returns:
    True if the OOBBs overlap, False otherwise.
  """
  inv_rot = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mju_negQuat(inv_rot, oobb0.rotation)
  rotated_pos = np.array([0.0] * 3)
  mujoco.mju_rotVecQuat(rotated_pos, oobb1.position - oobb0.position, inv_rot)
  rotation = np.array([1.0, 0.0, 0.0, 0.0])
  mujoco.mju_mulQuat(rotation, inv_rot, oobb1.rotation)
  return overlap_aabb_oobb(
      Aabb(-oobb0.half_extents, oobb0.half_extents),
      Oobb(rotated_pos, rotation, oobb1.half_extents),
  )
