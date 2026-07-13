import asyncio
import json
import time
import pytest
import grpc

import distripute.master as master_mod
import distripute.relay as relay_mod
import distripute.worker as worker_mod
from distripute.grpc import pb, grpc as rpcmod
from distripute.master import MasterServicer


@pytest.fixture
def network_id():
    return "test-net-123"


@pytest.fixture
def servicer(network_id, tmp_path, monkeypatch):
    monkeypatch.setattr(master_mod, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(worker_mod, "TASK_CACHE", tmp_path / "tasks")
    return MasterServicer(network_id)


@pytest.fixture
async def channel(servicer):
    server = grpc.aio.server()
    rpcmod.add_MasterServicer_to_server(servicer, server)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    async with grpc.aio.insecure_channel(f"localhost:{port}") as ch:
        yield ch
    await server.stop(None)


@pytest.fixture
def stub(channel):
    return rpcmod.MasterStub(channel)


@pytest.fixture
async def relay_addr():
    server = grpc.aio.server()
    rpcmod.add_RelayServicer_to_server(relay_mod.RelayServicer(), server)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    try:
        yield f"localhost:{port}"
    finally:
        await server.stop(None)


async def _wait_for_result(stub, task_id: str, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = await stub.GetTaskResult(pb.TaskResultRequest(task_id=task_id))
        if result.status in {"done", "failed"}:
            return result
        await asyncio.sleep(0.1)
    raise AssertionError(f"task {task_id} did not finish within {timeout}s")


async def _wait_for_workers(stub, expected: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = await stub.GetInfo(pb.Empty())
        if info.workers >= expected:
            return info
        await asyncio.sleep(0.1)
    raise AssertionError(f"expected at least {expected} workers within {timeout}s")


class TestMasterGRPC:
    async def test_register(self, stub, network_id):
        resp = await stub.Register(pb.RegisterRequest(
            network_id=network_id, worker_id="w1",
            cpu_cores=8, gpu_count=1, hardware="gpu",
        ))
        assert resp.worker_id == "w1"
        assert resp.heartbeat_interval == 10

    async def test_register_rejects_bad_network(self, stub):
        with pytest.raises(grpc.RpcError) as e:
            await stub.Register(pb.RegisterRequest(network_id="wrong", worker_id="w1"))
        assert e.value.code() == grpc.StatusCode.PERMISSION_DENIED

    async def test_heartbeat(self, stub, network_id):
        await stub.Register(pb.RegisterRequest(network_id=network_id, worker_id="w1"))
        resp = await stub.Heartbeat(pb.HeartbeatRequest(worker_id="w1", load=0.5))
        assert resp.acknowledged

    async def test_submit_and_get_task(self, stub, network_id):
        resp = await stub.SubmitTask(pb.TaskSubmit(
            network_id=network_id, task_id="t1",
            func_name="test_fn", source="def test_fn():\n  return 42",
            payload=b"test",
        ))
        assert resp.task_id == "t1"
        assert resp.status == "pending"

        r = await stub.GetTaskResult(pb.TaskResultRequest(task_id="t1"))
        assert r.task_id == "t1"

    async def test_get_unknown_task(self, stub):
        with pytest.raises(grpc.RpcError) as e:
            await stub.GetTaskResult(pb.TaskResultRequest(task_id="nonexistent"))
        assert e.value.code() == grpc.StatusCode.NOT_FOUND

    async def test_poll_task(self, stub, network_id):
        await stub.SubmitTask(pb.TaskSubmit(
            network_id=network_id, task_id="t2",
            func_name="fn", source="",
        ))
        await stub.Register(pb.RegisterRequest(network_id=network_id, worker_id="w1"))

        resp = await stub.PollTasks(pb.PollRequest(worker_id="w1", max_tasks=4))
        assert len(resp.tasks) == 1
        assert resp.tasks[0].id == "t2"

    async def test_submit_result(self, stub, network_id):
        await stub.SubmitTask(pb.TaskSubmit(
            network_id=network_id, task_id="t3", func_name="fn", source="",
        ))
        resp = await stub.SubmitResult(pb.TaskResult(
            task_id="t3", success=True, output="hello", duration=1.0,
        ))
        assert resp.ok

        r = await stub.GetTaskResult(pb.TaskResultRequest(task_id="t3"))
        assert r.status == "done"
        assert r.result == "hello"

    async def test_create_batch(self, stub, network_id):
        resp = await stub.CreateBatch(pb.BatchCreateRequest(
            network_id=network_id, func_name="fn", source="def fn():\n  pass",
            arg_payloads=[b"args1", b"args2", b"args3"],
            job_id="batch1",
        ))
        assert resp.job_id == "batch1"
        assert resp.task_count == 3
        assert resp.status == "running"

        status = await stub.GetBatchStatus(pb.BatchStatusRequest(job_id="batch1"))
        assert status.total == 3
        assert status.pending == 3

    async def test_get_info(self, stub, network_id):
        resp = await stub.GetInfo(pb.Empty())
        assert resp.network_id == network_id
        assert resp.version == "0.1.0"

    async def test_file_upload_and_download(self, stub):
        chunks = [pb.FileChunk(job_id="fj1", filename="test.txt",
                               data=b"hello world", offset=0, total_size=11)]

        async def _iter():
            for c in chunks:
                yield c

        up_resp = await stub.UploadFile(_iter())
        assert up_resp.total_size == 11

        down = []
        async for chunk in stub.GetFile(pb.FileRequest(job_id="fj1", filename="test.txt")):
            down.append(chunk)
        assert b"".join(c.data for c in down) == b"hello world"

    async def test_requeues_task_from_stale_worker(self, servicer, stub, network_id):
        await stub.SubmitTask(pb.TaskSubmit(
            network_id=network_id, task_id="t-requeue", func_name="fn", source="",
        ))
        await stub.Register(pb.RegisterRequest(network_id=network_id, worker_id="w1"))
        await stub.Register(pb.RegisterRequest(network_id=network_id, worker_id="w2"))

        first = await stub.PollTasks(pb.PollRequest(worker_id="w1", max_tasks=1))
        assert [task.id for task in first.tasks] == ["t-requeue"]

        servicer.cache._local_workers["w1"]["_ts"] = time.time() - 31
        servicer.cache._local_workers["w1"]["last_seen"] = time.time() - 31

        second = await stub.PollTasks(pb.PollRequest(worker_id="w2", max_tasks=1))
        assert [task.id for task in second.tasks] == ["t-requeue"]

        result = await stub.GetTaskResult(pb.TaskResultRequest(task_id="t-requeue"))
        assert result.status == "running"
        assert result.worker_id == "w2"

    async def test_worker_left_frame_requeues_task(self, servicer, stub, network_id):
        await stub.SubmitTask(pb.TaskSubmit(
            network_id=network_id, task_id="t-relay-left", func_name="fn", source="",
        ))
        await stub.Register(pb.RegisterRequest(network_id=network_id, worker_id="w1"))
        await stub.Register(pb.RegisterRequest(network_id=network_id, worker_id="w2"))
        assigned = await stub.PollTasks(pb.PollRequest(worker_id="w1", max_tasks=1))
        assert [task.id for task in assigned.tasks] == ["t-relay-left"]

        queue = asyncio.Queue()
        await master_mod._handle_relay_frame(
            servicer,
            pb.RelayFrame(
                network_id=network_id,
                sender_id="w1",
                sender_role="worker",
                routing_key="worker_left",
            ),
            network_id,
            queue,
        )

        reassigned = await stub.PollTasks(pb.PollRequest(worker_id="w2", max_tasks=1))
        assert [task.id for task in reassigned.tasks] == ["t-relay-left"]

    async def test_register_abort_returns_relay_error_frame(self, servicer, network_id):
        queue = asyncio.Queue()
        await master_mod._handle_relay_frame(
            servicer,
            pb.RelayFrame(
                network_id=network_id,
                sender_id="w1",
                sender_role="worker",
                routing_key="register",
                payload=pb.RegisterRequest(
                    network_id="wrong-network",
                    worker_id="w1",
                ).SerializeToString(),
            ),
            network_id,
            queue,
        )

        response = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert response.routing_key == "register_error"
        payload = json.loads(response.payload.decode())
        assert payload["code"] == "PERMISSION_DENIED"
        assert "invalid network_id" in payload["message"]

    async def test_relay_frame_handler_schedules_execution_without_blocking(self, monkeypatch, network_id):
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_run(source, func_name, payload, requirements, filename="", file_path=""):
            started.set()
            await release.wait()
            return {"success": True, "output": "ok", "error": "", "duration": 0.01}

        monkeypatch.setattr(worker_mod, "_run_with_uv", fake_run)

        outbound = asyncio.Queue()
        registered = asyncio.Event()
        registered.set()
        inflight_tasks = set()

        outcome = await worker_mod._handle_relay_frame(
            pb.RelayFrame(
                network_id=network_id,
                sender_id="master",
                sender_role="master",
                routing_key="poll_response",
                payload=pb.PollResponse(tasks=[pb.TaskDef(
                    id="relay-task",
                    func_name="demo",
                    source="def demo():\n    return 'ok'\n",
                )]).SerializeToString(),
            ),
            outbound=outbound,
            network_id=network_id,
            worker_id="w1",
            registered=registered,
            inflight_tasks=inflight_tasks,
        )

        assert outcome is None
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert inflight_tasks

        release.set()
        await asyncio.gather(*list(inflight_tasks), return_exceptions=True)
        response = await asyncio.wait_for(outbound.get(), timeout=1.0)
        assert response.routing_key == "submit_result"

    async def test_worker_register_error_frame_is_terminal(self, caplog, network_id):
        outbound = asyncio.Queue()
        registered = asyncio.Event()
        inflight_tasks = set()

        with caplog.at_level("ERROR"):
            outcome = await worker_mod._handle_relay_frame(
                pb.RelayFrame(
                    network_id=network_id,
                    sender_id="master",
                    sender_role="master",
                    routing_key="register_error",
                    payload=b'{"code":"PERMISSION_DENIED","message":"invalid network_id"}',
                ),
                outbound=outbound,
                network_id=network_id,
                worker_id="w1",
                registered=registered,
                inflight_tasks=inflight_tasks,
            )

        assert outcome == "register_error"
        assert not registered.is_set()
        assert "relay registration failed" in caplog.text

    async def test_relay_worker_executes_task_end_to_end(self, servicer, stub, network_id, relay_addr, monkeypatch):
        async def fake_run(source, func_name, payload, requirements, filename="", file_path=""):
            return {"success": True, "output": f"relay:{func_name}", "error": "", "duration": 0.01}

        monkeypatch.setattr(worker_mod, "_run_with_uv", fake_run)

        relay_task = asyncio.create_task(master_mod._relay_loop(servicer, relay_addr, network_id))
        worker_task = asyncio.create_task(worker_mod.run_worker(
            relay_addr=relay_addr,
            network_id=network_id,
        ))

        try:
            await _wait_for_workers(stub, 1, timeout=8.0)
            await stub.SubmitTask(pb.TaskSubmit(
                network_id=network_id,
                task_id="relay-task",
                func_name="demo",
                source="def demo():\n    return 'ok'\n",
            ))

            result = await _wait_for_result(stub, "relay-task", timeout=8.0)
            assert result.status == "done"
            assert result.result == "relay:demo"
            assert result.worker_id
        finally:
            worker_task.cancel()
            relay_task.cancel()
            await asyncio.gather(worker_task, relay_task, return_exceptions=True)

    async def test_relay_file_backed_task_fails_explicitly(self):
        result = await worker_mod._execute_task(
            pb.TaskDef(
                id="file-task",
                func_name="demo",
                source="",
                filename="input.txt",
                file_size=10,
            ),
            allow_file_download=False,
        )
        assert not result.success
        assert result.error == worker_mod.RELAY_UNSUPPORTED_FILE_ERROR
