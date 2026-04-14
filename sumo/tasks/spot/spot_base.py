# Copyright (c) 2025-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from judo import MODEL_PATH as JUDO_MODEL_PATH
from judo.tasks.base import TaskConfig
from judo.tasks.spot.spot_base import SpotBase as _JudoSpotBase
from judo.tasks.spot.spot_base import SpotBaseConfig

from sumo import MODEL_PATH

XML_PATH = str(JUDO_MODEL_PATH / "xml" / "spot_primitive" / "robot.xml")

ConfigT = TypeVar("ConfigT", bound=TaskConfig)

_INCLUDE_RE = re.compile(r'(<include\s+file=")([^"]+)(")')
_SPOT_PRIMITIVE_FILES = {
    "default.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "default.xml",
    "assets.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "assets.xml",
    "body.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "body.xml",
    "legs.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "legs.xml",
    "arm.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "arm.xml",
    "actuator.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "actuator.xml",
    "contact.xml": JUDO_MODEL_PATH / "xml" / "spot_primitive" / "contact.xml",
}
_SUMO_SPOT_EXTENSION_FILES = {
    "sensor.xml": MODEL_PATH / "xml" / "spot_components" / "sensor.xml",
}
_PUBLIC_OBJECT_COMPAT = {
    "tire_rubber.xml": JUDO_MODEL_PATH / "xml" / "objects" / "tire" / "tire.xml",
    "tire_rubber_defs.xml": JUDO_MODEL_PATH / "xml" / "objects" / "tire" / "tire_defs.xml",
}


def _get_spot_menagerie_dir() -> Path:
    try:
        from robot_descriptions import spot_mj_description  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised via failure-path test
        raise RuntimeError(
            "Spot tasks require `robot_descriptions` to resolve Spot robot assets. "
            "Run from the root pixi environment (for example `pixi run sumo`) "
            "or install `robot_descriptions` in the active environment."
        ) from exc

    return Path(spot_mj_description.PACKAGE_PATH)


def _resolve_public_object_asset(relpath: str) -> Path | None:
    if "objects/" not in relpath:
        return None

    suffix = relpath[relpath.rindex("objects/") :]
    candidates = [suffix]
    if suffix.startswith("objects/tire/meshes/"):
        compat_suffix = suffix.replace("objects/tire/meshes/", "objects/tire/", 1)
        if compat_suffix.endswith("/visual/tire.obj"):
            compat_suffix = compat_suffix[: -len("/visual/tire.obj")] + "/visual/tire_rubber.obj"
        candidates.insert(0, compat_suffix)

    for root in (JUDO_MODEL_PATH / "meshes", MODEL_PATH / "meshes"):
        for candidate_suffix in candidates:
            candidate = root / candidate_suffix
            if candidate.exists():
                return candidate
    return None


class SpotAssetMixin:
    """Align Spot robot assets with judo while preserving local object compatibility."""

    spec: Any  # MjSpec, provided by SpotBase via MRO

    def _process_spec(self) -> None:
        menagerie_dir = _get_spot_menagerie_dir()
        menagerie_assets = menagerie_dir / "assets"

        for mesh in self.spec.meshes:
            if "spot/meshes/" in mesh.file:
                mesh.file = str(menagerie_assets / Path(mesh.file).name)
                continue

            asset_path = _resolve_public_object_asset(mesh.file)
            if asset_path is not None:
                mesh.file = str(asset_path)

        for texture in self.spec.textures:
            if "spot/textures/" in texture.file:
                texture.file = str(menagerie_dir / "spot.png")


class SpotBase(SpotAssetMixin, _JudoSpotBase, Generic[ConfigT]):
    """Sumo SpotBase wrapper that composes local task XML with public Spot definitions."""

    config_t: type[ConfigT]  # pyright: ignore[reportIncompatibleVariableOverride]
    config: ConfigT

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    @classmethod
    def _resolve_include_path(cls, include_path: str, source_dir: Path) -> Path:
        include = Path(include_path)
        basename = include.name

        if basename in _SPOT_PRIMITIVE_FILES:
            return _SPOT_PRIMITIVE_FILES[basename]
        if basename in _SUMO_SPOT_EXTENSION_FILES:
            return _SUMO_SPOT_EXTENSION_FILES[basename]
        if basename in _PUBLIC_OBJECT_COMPAT:
            return _PUBLIC_OBJECT_COMPAT[basename]

        candidate = include if include.is_absolute() else (source_dir / include).resolve()
        if candidate.exists():
            return candidate

        flat_object_candidate = MODEL_PATH / "xml" / "objects" / basename
        if flat_object_candidate.exists():
            return flat_object_candidate

        nested_object_candidate = MODEL_PATH / "xml" / "objects" / include.stem / basename
        if nested_object_candidate.exists():
            return nested_object_candidate

        raise FileNotFoundError(f"Unable to resolve include '{include_path}' from '{source_dir}'.")

    @classmethod
    def _materialize_model_path(cls, model_path: str | Path) -> Path:
        path = Path(model_path).expanduser().resolve()
        if cls._is_relative_to(path, JUDO_MODEL_PATH):
            return path

        source = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)

        def replace_include(match: re.Match[str]) -> str:
            resolved = cls._resolve_include_path(match.group(2), path.parent)
            return f"{match.group(1)}{resolved.as_posix()}{match.group(3)}"

        rewritten = _INCLUDE_RE.sub(replace_include, source)
        cache_dir = Path(tempfile.gettempdir()) / "sumo-spot-models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(rewritten.encode("utf-8")).hexdigest()[:12]
        materialized_path = cache_dir / f"{path.stem}-{digest}.xml"
        if not materialized_path.exists():
            materialized_path.write_text(rewritten)
        return materialized_path

    def __init__(
        self,
        model_path: str = XML_PATH,
        use_arm: bool = True,
        use_gripper: bool = False,
        use_legs: bool = False,
        use_torso: bool = False,
        config: ConfigT | None = None,
    ) -> None:
        super().__init__(
            model_path=str(self._materialize_model_path(model_path)),
            use_arm=use_arm,
            use_gripper=use_gripper,
            use_legs=use_legs,
            use_torso=use_torso,
            config=cast(Any, config),
        )


__all__ = ["SpotAssetMixin", "SpotBase", "SpotBaseConfig"]
