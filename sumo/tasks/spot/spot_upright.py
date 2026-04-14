# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass
from typing import Callable

import numpy as np
from judo.tasks.base import TaskConfig
from mujoco import MjModel, mjtGeom

Z_AXIS = np.array([0.0, 0.0, 1.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
GROUND_CLEARANCE_MARGIN = 0.02
MAX_RESET_ORIENTATION_ATTEMPTS = 1000


@dataclass
class SpotUprightConfig(TaskConfig):
    """Configuration for Sumo's simplified Spot upright analysis tasks."""

    w_orientation: float = 100.0
    w_gripper_proximity: float = 0.5


def random_unit_quat() -> np.ndarray:
    """Sample a random unit quaternion in MuJoCo's wxyz convention."""
    quat = np.random.normal(size=4)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat = -quat
    return quat


def z_axis_is_upright(quat: np.ndarray, tolerance: float) -> bool:
    """Return whether the body z-axis already satisfies an upright task."""
    object_z_axis = quat_to_mat(quat) @ Z_AXIS
    return bool(object_z_axis[2] >= 1.0 - tolerance)


def y_axis_is_horizontal(quat: np.ndarray, tolerance: float) -> bool:
    """Return whether the body y-axis already satisfies a tire upright task."""
    object_y_axis = quat_to_mat(quat) @ Y_AXIS
    return bool(abs(object_y_axis[2]) <= tolerance)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a rotation matrix."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _geom_min_z_in_body_frame(model: MjModel, geom_id: int, body_rot: np.ndarray) -> float:
    geom_pos = np.asarray(model.geom_pos[geom_id])
    geom_rot = quat_to_mat(np.asarray(model.geom_quat[geom_id]))
    center_z = float((body_rot @ geom_pos)[2])
    z_in_geom = geom_rot.T @ body_rot.T @ Z_AXIS
    geom_type = model.geom_type[geom_id]
    geom_size = model.geom_size[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        vert_start = model.mesh_vertadr[mesh_id]
        vert_end = vert_start + model.mesh_vertnum[mesh_id]
        vertices = np.asarray(model.mesh_vert[vert_start:vert_end])
        geom_points = geom_pos[:, None] + geom_rot @ vertices.T
        return float((body_rot @ geom_points)[2].min())

    if geom_type == mjtGeom.mjGEOM_BOX:
        support = np.abs(geom_size * z_in_geom).sum()
    elif geom_type == mjtGeom.mjGEOM_SPHERE:
        support = geom_size[0]
    elif geom_type == mjtGeom.mjGEOM_CAPSULE:
        support = geom_size[0] + geom_size[1] * abs(z_in_geom[2])
    elif geom_type == mjtGeom.mjGEOM_CYLINDER:
        support = geom_size[0] * np.linalg.norm(z_in_geom[:2]) + geom_size[1] * abs(z_in_geom[2])
    elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
        support = np.linalg.norm(geom_size * z_in_geom)
    else:
        support = model.geom_rbound[geom_id]

    return center_z - float(support)


def ground_clearance_height(model: MjModel, body_name: str, quat: np.ndarray) -> float:
    """Return a conservative free-body z value that keeps all body geoms above the ground."""
    body_id = model.body(body_name).id
    rot = quat_to_mat(quat)
    min_z = np.inf

    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        min_z = min(min_z, _geom_min_z_in_body_frame(model, geom_id, rot))

    if not np.isfinite(min_z):
        return GROUND_CLEARANCE_MARGIN
    return max(GROUND_CLEARANCE_MARGIN, -min_z + GROUND_CLEARANCE_MARGIN)


def random_object_pose(
    model: MjModel,
    body_name: str,
    xy: np.ndarray,
    reject_orientation: Callable[[np.ndarray], bool] | None = None,
) -> np.ndarray:
    """Build a free-joint pose with random attitude and ground clearance."""
    for _ in range(MAX_RESET_ORIENTATION_ATTEMPTS):
        quat = random_unit_quat()
        if reject_orientation is None or not reject_orientation(quat):
            break
    else:
        msg = f"Failed to sample a valid reset orientation for {body_name}"
        raise RuntimeError(msg)

    z = ground_clearance_height(model, body_name, quat)
    return np.array([xy[0], xy[1], z, *quat])


def sample_annulus_xy(radius_min: float, radius_max: float, noise_scale: float = 0.1) -> np.ndarray:
    """Sample an object position in an annulus around the origin."""
    radius = radius_min + (radius_max - radius_min) * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    return np.array([radius * np.cos(theta), radius * np.sin(theta)]) + noise_scale * np.random.randn(2)


def z_axis_orientation_reward(config: SpotUprightConfig, object_z_axis: np.ndarray) -> np.ndarray:
    """Reward object z-axis alignment with world z."""
    alignment = np.sum(object_z_axis * Z_AXIS, axis=-1)
    return -config.w_orientation * (1.0 - alignment).mean(axis=-1)


def horizontal_axis_orientation_reward(config: SpotUprightConfig, object_axis: np.ndarray) -> np.ndarray:
    """Reward an object axis being horizontal."""
    return -config.w_orientation * np.abs(np.sum(object_axis * Z_AXIS, axis=-1)).mean(axis=-1)


def gripper_distance_reward(config: SpotUprightConfig, gripper_distance: np.ndarray) -> np.ndarray:
    """Reward gripper proximity to the object."""
    return -config.w_gripper_proximity * gripper_distance.mean(axis=-1)
