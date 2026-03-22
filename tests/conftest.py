import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_JUDO_ROOT = REPO_ROOT / ".judo-src"
PRIVATE_JUDO_ROOT = REPO_ROOT / "judo-private"

if PUBLIC_JUDO_ROOT.exists():
    sys.path.insert(0, str(PUBLIC_JUDO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def bootstrap_public_judo_assets() -> None:
    import judo

    judo_path = Path(judo.__file__).resolve()
    if PRIVATE_JUDO_ROOT in judo_path.parents:
        raise RuntimeError(f"Expected public judo-rai, got local private checkout at {judo_path}")

    mesh_root = judo.MODEL_PATH / "meshes"
    if not (mesh_root / "spot").exists() or not (mesh_root / "objects" / "tire").exists():
        from judo.utils.assets import download_and_extract_meshes

        download_and_extract_meshes(
            extract_root=str(judo.MODEL_PATH), repo="bdaiinstitute/judo", asset_name="meshes.zip"
        )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require g1_extensions if it's not available."""
    try:
        import g1_extensions  # noqa: F401
    except ImportError:
        skip_cpp = pytest.mark.skip(reason="g1_extensions not built")
        for item in items:
            if "g1_extensions" in item.keywords:
                item.add_marker(skip_cpp)
