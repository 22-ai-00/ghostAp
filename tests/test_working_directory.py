"""Direct-host working-directory contracts."""

from __future__ import annotations

import threading
from collections import OrderedDict
from types import SimpleNamespace

from src.feishu.handlers.base import BaseHandler


def test_working_directory_accepts_any_existing_host_directory(tmp_path) -> None:
    handler = object.__new__(BaseHandler)
    handler.ctx = SimpleNamespace(
        working_dir_lock=threading.Lock(),
        working_dirs=OrderedDict(),
    )

    success, selected = handler.set_working_dir("oc_direct_host", str(tmp_path))

    assert success is True
    assert selected == str(tmp_path)
    assert handler.get_working_dir("oc_direct_host") == str(tmp_path)
