import asyncio
import pytest
import grpc

from distripute.master import MasterServicer
from distripute.grpc import pb, grpc as rpcmod


@pytest.fixture
def network_id():
    return "test-net-123"


@pytest.fixture
def servicer(network_id):
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
