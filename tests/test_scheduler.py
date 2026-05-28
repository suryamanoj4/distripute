import pytest
import json

from distripute.master import MasterNode


@pytest.fixture
def master():
    return MasterNode(host="127.0.0.1", port=0)


@pytest.fixture
async def client(aiohttp_client, master):
    return await aiohttp_client(master._http)


class TestMasterTaskEndpoint:
    async def test_submit_task(self, client, master):
        resp = await client.post("/task", json={
            "network_id": master.network_id,
            "task_id": "t1",
            "func_name": "test_func",
            "source": "def test_func():\n    return 42",
            "requirements": [],
            "payload": "",
        })
        data = await resp.json()
        assert data["task_id"] == "t1"
        assert data["status"] == "pending"

    async def test_task_rejects_bad_network_id(self, client):
        resp = await client.post("/task", json={
            "network_id": "wrong",
            "task_id": "t1",
            "func_name": "test",
            "source": "",
            "requirements": [],
            "payload": "",
        })
        assert resp.status == 403

    async def test_get_task_status(self, client, master):
        await client.post("/task", json={
            "network_id": master.network_id,
            "task_id": "t2",
            "func_name": "test_func",
            "source": "def test_func():\n    return 42",
            "requirements": [],
            "payload": "",
        })
        resp = await client.get("/task/t2")
        data = await resp.json()
        assert data["task_id"] == "t2"
        assert data["status"] in ("pending", "running")

    async def test_get_unknown_task(self, client):
        resp = await client.get("/task/nonexistent")
        assert resp.status == 404

    async def test_worker_polls_function_task(self, client, master):
        await client.post("/task", json={
            "network_id": master.network_id,
            "task_id": "t3",
            "func_name": "my_fn",
            "source": "def my_fn():\n    return 99",
            "requirements": [],
            "payload": "gASVHQAAAAAAAABdlChLAEsAdS4=",
        })

        await client.post("/register", json={
            "network_id": master.network_id,
            "worker_id": "w1", "hardware": "cpu",
        })

        resp = await client.post("/tasks/poll", json={
            "worker_id": "w1", "max_tasks": 4,
        })
        tasks = (await resp.json())["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["func_name"] == "my_fn"
        assert "source" in tasks[0]

    async def test_worker_submits_result(self, client, master):
        await client.post("/task", json={
            "network_id": master.network_id,
            "task_id": "t4",
            "func_name": "fn",
            "source": "",
            "requirements": [],
            "payload": "",
        })
        resp = await client.post("/tasks/result", json={
            "task_id": "t4", "success": True,
            "output": "hello world", "duration": 1.0,
        })
        assert (await resp.json())["ok"]

        resp = await client.get("/task/t4")
        data = await resp.json()
        assert data["status"] == "done"

    async def test_worker_info_endpoint(self, client, master):
        resp = await client.get("/info")
        data = await resp.json()
        assert data["network_id"] == master.network_id
        assert data["version"] == "0.1.0"
