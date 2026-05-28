import pytest
import tempfile
from pathlib import Path

import distripute
from distripute.task import DistriputeNotConnectedError


class TestTaskDecorator:
    def test_task_raises_without_init(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def helper():\n    return 1\n")
            f.flush()
            source_file = f.name

        import importlib.util
        spec = importlib.util.spec_from_file_location("test_mod", source_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        @distripute.task
        def my_func():
            return 42

        with pytest.raises(DistriputeNotConnectedError):
            my_func()

        Path(source_file).unlink()

    def test_remote_function_wraps_original(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            source_file = f.name

        import importlib.util
        spec = importlib.util.spec_from_file_location("test_mod2", source_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        @distripute.task
        def add(a, b):
            return a + b

        assert add.__name__ == "add"
        assert callable(add)

        Path(source_file).unlink()

    def test_task_with_requirements(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            source_file = f.name

        import importlib.util
        spec = importlib.util.spec_from_file_location("test_mod3", source_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        @distripute.task(requirements=["whisper", "torch"])
        def transcribe(path):
            return "text"

        assert transcribe._requirements == ["whisper", "torch"]

        Path(source_file).unlink()

    def test_task_is_accessible_via_distripute_namespace(self):
        assert hasattr(distripute, "task")
        assert callable(distripute.task)
        assert hasattr(distripute, "init")
        assert callable(distripute.init)
