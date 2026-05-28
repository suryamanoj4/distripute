import functools
import inspect
from pathlib import Path

from .client import _Client

_global_client = None


class DistriputeNotConnectedError(RuntimeError):
    pass


class _RemoteResult:
    def __init__(self, task_id: str):
        self._task_id = task_id
        self._value = None
        self._error = None
        self._done = False

    def get(self, timeout: float | None = None):
        import threading
        event = threading.Event()
        def check():
            while not self._done:
                if event.wait(timeout=timeout or 0.1):
                    break
            return self._value
        waited = 0.0
        while not self._done:
            if timeout and waited >= timeout:
                raise TimeoutError(f"task {self._task_id} timed out")
            event.wait(timeout=0.1)
            waited += 0.1
        if self._error:
            raise self._error
        return self._value

    def _resolve(self, value):
        self._value = value
        self._done = True

    def _reject(self, error):
        self._error = error
        self._done = True


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
