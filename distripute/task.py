import functools
import inspect
from pathlib import Path

from .client import _Client, _RemoteResult

_global_client = None


class DistriputeNotConnectedError(RuntimeError):
    pass


class RemoteFunction:
    def __init__(self, func, name, source_file, requirements=None):
        self._func = func
        self._name = name
        self._source_file = source_file
        self._requirements = requirements or []
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.remote(*args, **kwargs)

    def remote(self, *args, **kwargs):
        if _global_client is None:
            raise DistriputeNotConnectedError(
                "distripute mesh not connected. Call distripute.init() first."
            )
        source = Path(self._source_file).read_text()
        return _global_client.submit(
            self._name, source, self._requirements, args, kwargs,
        )


def task(func=None, *, requirements=None):
    if func is None:
        return lambda f: task(f, requirements=requirements)
    try:
        source_file = inspect.getfile(func)
    except (TypeError, OSError):
        source_file = None
    if not source_file:
        raise RuntimeError(
            "cannot determine source file. Define the function in a .py file."
        )
    return RemoteFunction(func, func.__name__, source_file, requirements=requirements)


def init(master_addr: str, network_id: str):
    global _global_client
    _global_client = _Client(master_addr, network_id)
