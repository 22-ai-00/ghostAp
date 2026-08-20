"""GhostAP — Feishu chatbot remote development service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ghostap")
except PackageNotFoundError:
    # Fallback for development mode (not installed as package)
    __version__ = "0.2.1"
