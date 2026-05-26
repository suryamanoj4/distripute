import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


DEFAULTS = {
    "whisper-tiny":       {"family": "whisper", "num_layers": 4,  "param_count": 39_000_000,    "min_gpu_mem": 1_000_000_000, "min_ram": 2_000_000_000},
    "whisper-base":       {"family": "whisper", "num_layers": 6,  "param_count": 74_000_000,    "min_gpu_mem": 1_000_000_000, "min_ram": 2_000_000_000},
    "whisper-small":      {"family": "whisper", "num_layers": 12, "param_count": 244_000_000,   "min_gpu_mem": 2_000_000_000, "min_ram": 4_000_000_000},
    "whisper-medium":     {"family": "whisper", "num_layers": 24, "param_count": 769_000_000,   "min_gpu_mem": 4_000_000_000, "min_ram": 8_000_000_000},
    "whisper-large":      {"family": "whisper", "num_layers": 32, "param_count": 1_550_000_000, "min_gpu_mem": 6_000_000_000, "min_ram": 12_000_000_000},
    "whisper-large-v2":   {"family": "whisper", "num_layers": 32, "param_count": 1_550_000_000, "min_gpu_mem": 6_000_000_000, "min_ram": 12_000_000_000},
    "whisper-large-v3":   {"family": "whisper", "num_layers": 32, "param_count": 1_550_000_000, "min_gpu_mem": 6_000_000_000, "min_ram": 12_000_000_000},
    "paddleocr":          {"family": "paddleocr", "num_layers": 0, "param_count": 0, "min_gpu_mem": 2_000_000_000, "min_ram": 4_000_000_000},
    "easyocr":            {"family": "easyocr", "num_layers": 0, "param_count": 0, "min_gpu_mem": 2_000_000_000, "min_ram": 4_000_000_000},
    "llama-3.1-8b":       {"family": "llama", "num_layers": 32, "param_count": 8_000_000_000, "min_gpu_mem": 16_000_000_000, "min_ram": 32_000_000_000},
    "llama-3.1-70b":      {"family": "llama", "num_layers": 80, "param_count": 70_000_000_000, "min_gpu_mem": 140_000_000_000, "min_ram": 256_000_000_000},
}


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, dict] = dict(DEFAULTS)

    @classmethod
    def from_yaml(cls, *paths: str):
        if yaml is None:
            raise ImportError("PyYAML is required to load model configs")
        reg = cls()
        for path in paths:
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    data = yaml.safe_load(f) or {}
                    for name, info in data.items():
                        reg._models[name] = dict(info)
        return reg

    def load_user_config(self):
        for path in (
            Path.home() / ".config" / "distripute" / "models.yaml",
            Path.home() / ".config" / "distripute" / "models.yml",
            Path.cwd() / ".distripute-models.yaml",
            Path.cwd() / ".distripute-models.yml",
        ):
            if path.exists() and yaml:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                    for name, info in data.items():
                        self._models[name] = dict(info)
                break
        return self

    def get(self, name: str) -> dict:
        if name in self._models:
            return dict(self._models[name])
        raise KeyError(f"unknown model: {name}")

    def register(self, name: str, info: dict):
        self._models[name] = dict(info)

    def list(self) -> dict[str, dict]:
        return dict(self._models)

    def compute_shards(self, name: str, num_workers: int) -> list[tuple[int, int]]:
        info = self.get(name)
        layers = info["num_layers"]
        if layers == 0:
            return [(0, 0) for _ in range(num_workers)]
        per_worker = max(1, layers // num_workers)
        shards = []
        start = 0
        for i in range(num_workers):
            end = layers if i == num_workers - 1 else start + per_worker
            shards.append((start, end))
            start = end
        return shards

    def fits_on(self, name: str, gpu_mem: int, ram: int) -> bool:
        info = self.get(name)
        return gpu_mem >= info["min_gpu_mem"] and ram >= info["min_ram"]


_default_registry = ModelRegistry().load_user_config()


def get_model_info(name: str) -> dict:
    return _default_registry.get(name)


def register_model(name: str, info: dict):
    _default_registry.register(name, info)


def compute_shards(name: str, num_workers: int) -> list[tuple[int, int]]:
    return _default_registry.compute_shards(name, num_workers)


def memory_fits(name: str, gpu_mem: int, ram: int) -> bool:
    return _default_registry.fits_on(name, gpu_mem, ram)
