import pytest
from aiohttp import web

from distripute.master import MasterNode


@pytest.fixture
def master():
    return MasterNode(host="127.0.0.1", port=0)


@pytest.fixture
async def client(aiohttp_client, master):
    return await aiohttp_client(master._http)


class TestScheduler:
    async def test_register_worker(self, client, master):
        resp = await client.post("/register", json={
            "network_id": master.network_id,
            "worker_id": "w1", "cpu_cores": 8, "gpu_count": 1, "hardware": "gpu",
        })
        data = await resp.json()
        assert data["worker_id"] == "w1"

    async def test_register_rejects_bad_network_id(self, client):
        resp = await client.post("/register", json={
            "network_id": "wrong", "worker_id": "w1",
        })
        assert resp.status == 403

    async def test_info(self, client, master):
        resp = await client.get("/info")
        data = await resp.json()
        assert data["network_id"] == master.network_id
        assert data["version"] == "0.1.0"

    async def test_create_job(self, client, tmp_path):
        d = tmp_path / "in"
        d.mkdir()
        (d / "test.wav").write_text("data")
        resp = await client.post("/job", json={
            "task_type": "asr", "model": "whisper-tiny",
            "strategy": "data_parallel",
            "input_source": str(d),
            "output_sink": str(tmp_path / "out"),
        })
        data = await resp.json()
        assert "job_id" in data
        assert data["task_count"] == 1

    async def test_create_job_bad_input(self, client):
        resp = await client.post("/job", json={
            "task_type": "asr", "model": "whisper-tiny",
            "strategy": "data_parallel",
            "input_source": "/nonexistent",
            "output_sink": "/tmp/out",
        })
        assert resp.status == 400

    async def test_poll_and_result(self, client, master, tmp_path):
        d = tmp_path / "in"
        d.mkdir()
        (d / "x.wav").write_text("data")

        await client.post("/job", json={
            "task_type": "asr", "model": "whisper-tiny",
            "strategy": "data_parallel",
            "input_source": str(d),
            "output_sink": str(tmp_path / "out"),
        })

        await client.post("/register", json={
            "network_id": master.network_id,
            "worker_id": "w1", "hardware": "cpu",
        })

        # worker polls
        resp = await client.post("/tasks/poll", json={
            "worker_id": "w1", "max_tasks": 4,
        })
        tasks = (await resp.json())["tasks"]
        assert len(tasks) == 1
        tid = tasks[0]["id"]

        # worker submits result
        resp = await client.post("/tasks/result", json={
            "task_id": tid, "success": True,
            "output": "hello world", "duration": 1.0,
        })
        assert (await resp.json())["ok"]

    async def test_poll_unknown_worker(self, client):
        resp = await client.post("/tasks/poll", json={
            "worker_id": "nonexistent", "max_tasks": 4,
        })
        assert resp.status == 403

    async def test_job_list_and_status(self, client, master, tmp_path):
        d = tmp_path / "in"
        d.mkdir()
        (d / "a.wav").write_text("d1")
        (d / "b.wav").write_text("d2")

        resp = await client.post("/job", json={
            "task_type": "asr", "model": "whisper-tiny",
            "strategy": "data_parallel",
            "input_source": str(d),
            "output_sink": str(tmp_path / "out"),
        })
        jid = (await resp.json())["job_id"]

        resp = await client.get(f"/job/{jid}")
        j = await resp.json()
        assert j["total"] == 2
        assert j["status"] == "running"

        resp = await client.get("/jobs")
        assert len((await resp.json())["jobs"]) == 1

    async def test_models_endpoint(self, client):
        resp = await client.get("/models")
        data = await resp.json()
        assert "whisper-tiny" in data["models"]

    async def test_register_model_api(self, client):
        resp = await client.post("/models/register", json={
            "name": "custom-v1",
            "info": {"family": "custom", "num_layers": 4, "param_count": 100,
                     "min_gpu_mem": 1, "min_ram": 1},
        })
        assert (await resp.json())["ok"]

        resp = await client.get("/models")
        data = await resp.json()
        assert "custom-v1" in data["models"]

    async def test_workers_endpoint(self, client, master):
        await client.post("/register", json={
            "network_id": master.network_id,
            "worker_id": "gpu1", "hardware": "gpu", "gpu_count": 2,
        })
        resp = await client.get("/workers")
        data = await resp.json()
        assert len(data["workers"]) == 1
        assert data["workers"][0]["id"] == "gpu1"

    async def test_result_completes_job(self, client, master, tmp_path):
        d = tmp_path / "in"
        d.mkdir()
        (d / "x.wav").write_text("d")
        (d / "y.wav").write_text("d")

        resp = await client.post("/job", json={
            "task_type": "asr", "model": "whisper-tiny",
            "strategy": "data_parallel",
            "input_source": str(d),
            "output_sink": str(tmp_path / "out"),
        })
        jid = (await resp.json())["job_id"]

        await client.post("/register", json={
            "network_id": master.network_id,
            "worker_id": "w1", "hardware": "cpu",
        })

        resp = await client.post("/tasks/poll", json={
            "worker_id": "w1", "max_tasks": 4,
        })
        tasks = (await resp.json())["tasks"]

        for t in tasks:
            await client.post("/tasks/result", json={
                "task_id": t["id"], "success": True,
                "output": "ok", "duration": 1.0,
            })

        resp = await client.get(f"/job/{jid}")
        j = await resp.json()
        assert j["status"] == "completed"
        assert j["done"] == 2
