VERSION = "0.1.0"

from .task import task, init, DistriputeNotConnectedError, _RemoteResult

__all__ = ["task", "init", "DistriputeNotConnectedError", "VERSION"]
