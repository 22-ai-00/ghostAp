import hashlib
import stat
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

from src.feishu.image_handler import FeishuImageHandler


def _successful_image_response(payload: bytes = b"image") -> MagicMock:
    response = MagicMock()
    response.success.return_value = True
    response.file = BytesIO(payload)
    return response


def test_image_cache_dir_is_shared_under_the_user_cache(monkeypatch, tmp_path: Path):
    user_home = tmp_path / "user-home"
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("HOME", str(user_home))

    save_dir = Path(FeishuImageHandler.get_image_save_dir())

    assert save_dir == user_home / ".cache" / "ghostAp" / "picturechat"
    assert project_root not in save_dir.parents


def test_global_image_cache_uses_full_message_identity_for_isolation(tmp_path: Path):
    api_client = MagicMock()
    api_client.im.v1.message_resource.get.side_effect = [
        _successful_image_response(b"first"),
        _successful_image_response(b"second"),
    ]
    handler = FeishuImageHandler(lambda: api_client, MagicMock())

    first_id = "om_first_sharedsuffix"
    second_id = "om_second_sharedsuffix"
    save_dir = tmp_path / "picturechat"
    first = handler.download_images(first_id, ["img_1"], str(save_dir))
    second = handler.download_images(second_id, ["img_2"], str(save_dir))

    first_path = Path(first.saved_paths[0])
    second_path = Path(second.saved_paths[0])
    assert first_path.parent.name == f"msg_{hashlib.sha256(first_id.encode()).hexdigest()}"
    assert second_path.parent.name == f"msg_{hashlib.sha256(second_id.encode()).hexdigest()}"
    assert first_path.parent != second_path.parent
    assert first_path.read_bytes() == b"first"
    assert second_path.read_bytes() == b"second"
    assert stat.S_IMODE(save_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(first_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(first_path.stat().st_mode) == 0o600
