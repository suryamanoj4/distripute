import pytest
from distripute.registry import ModelRegistry


@pytest.fixture
def reg():
    return ModelRegistry()


class TestRegistry:
    def test_get_known_model(self, reg):
        info = reg.get("whisper-tiny")
        assert info["family"] == "whisper"
        assert info["num_layers"] == 4

    def test_get_unknown_model_raises(self, reg):
        with pytest.raises(KeyError):
            reg.get("nonexistent-model")

    def test_register_custom_model(self, reg):
        reg.register("my-model", dict(family="test", num_layers=8, param_count=100,
                                       min_gpu_mem=1_000_000_000, min_ram=2_000_000_000))
        info = reg.get("my-model")
        assert info["num_layers"] == 8

    def test_compute_shards_even(self, reg):
        shards = reg.compute_shards("whisper-small", 3)  # 12 layers / 3 = 4 per worker
        assert len(shards) == 3
        assert shards[0] == (0, 4)
        assert shards[1] == (4, 8)
        assert shards[2] == (8, 12)

    def test_compute_shards_uneven(self, reg):
        shards = reg.compute_shards("whisper-small", 5)  # 12 layers / 5
        assert len(shards) == 5
        assert shards[0][0] == 0
        assert shards[-1][1] == 12

    def test_compute_shards_zero_layers(self, reg):
        reg.register("noshard", dict(family="test", num_layers=0, param_count=0,
                                      min_gpu_mem=1, min_ram=1))
        shards = reg.compute_shards("noshard", 3)
        assert len(shards) == 3
        assert all(s == (0, 0) for s in shards)

    def test_fits_on_gpu(self, reg):
        info = reg.get("whisper-tiny")
        assert info["min_gpu_mem"] == 1_000_000_000
        assert reg.fits_on("whisper-tiny", 2_000_000_000, 4_000_000_000)
        assert not reg.fits_on("whisper-tiny", 500_000_000, 4_000_000_000)

    def test_list_models(self, reg):
        models = reg.list()
        assert "whisper-small" in models
        assert "llama-3.1-8b" in models
