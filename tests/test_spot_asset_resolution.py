from __future__ import annotations

import sys
from pathlib import Path

import pytest
from judo.tasks.spot.spot_base import SpotBase as JudoSpotBase
from judo.tasks.spot.spot_box_push import SpotBoxPush as JudoSpotBoxPush

from sumo import MODEL_PATH as SUMO_MODEL_PATH
from sumo.tasks.spot.spot_base import SpotBase as SumoSpotBase
from sumo.tasks.spot.spot_box_push import SpotBoxPush as SumoSpotBoxPush
from sumo.tasks.spot.spot_chair_push import SpotChairPush


def _mesh_file(task: object, mesh_name: str) -> str:
    for mesh in task.spec.meshes:
        if mesh.name == mesh_name:
            return mesh.file
    raise AssertionError(f"Mesh {mesh_name!r} not found")


def _texture_file(task: object, texture_name: str) -> str:
    for texture in task.spec.textures:
        if texture.name == texture_name:
            return texture.file
    raise AssertionError(f"Texture {texture_name!r} not found")


def test_sumo_spot_base_matches_judo_robot_assets() -> None:
    sumo_task = SumoSpotBase()
    judo_task = JudoSpotBase()

    assert _mesh_file(sumo_task, "body_0_visual") == _mesh_file(judo_task, "body_0_visual")
    assert _mesh_file(sumo_task, "front_left_hip_visual") == _mesh_file(judo_task, "front_left_hip_visual")
    assert _texture_file(sumo_task, "bdai_texture") == _texture_file(judo_task, "bdai_texture")


def test_wrapped_spot_task_matches_judo_robot_assets() -> None:
    sumo_task = SumoSpotBoxPush()
    judo_task = JudoSpotBoxPush()

    assert _mesh_file(sumo_task, "body_0_visual") == _mesh_file(judo_task, "body_0_visual")
    assert _mesh_file(sumo_task, "arm_link_sh0_visual") == _mesh_file(judo_task, "arm_link_sh0_visual")
    assert _texture_file(sumo_task, "bdai_texture") == _texture_file(judo_task, "bdai_texture")


def test_custom_sumo_spot_task_keeps_local_object_meshes() -> None:
    task = SpotChairPush()

    expected = SUMO_MODEL_PATH / "meshes" / "yellow_chair" / "meshes" / "visual" / "visual_v2.obj"
    assert Path(_mesh_file(task, "yellow_chair_mesh")) == expected


def test_sumo_spot_tasks_fail_clearly_without_robot_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "robot_descriptions", None)

    with pytest.raises(RuntimeError, match="robot_descriptions"):
        SumoSpotBase()
