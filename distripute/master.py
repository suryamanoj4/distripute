import asyncio
import logging
import time
import uuid
from pathlib import Path

import click
import grpc
import cloudpickle
import base64

from . import VERSION
from .grpc import pb, grpc as rpcmod
from .registry import _default_registry as reg

logger = logging.getLogger("distripute.master")

DATA_DIR = Path.home() / ".cache" / "distripute" / "data"


class MasterServicer(rpcmod.MasterServicer):
    def __init__(self, network_id: str):
        self.network_id = network_id
        self.workers: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self._pending: list[str] = []
        self._lock = asyncio.Lock()

    # ── Worker Registration ──────────────────────────────

    async def Register(self, request, context):
        if request.network_id != self.network_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid network_id")
        wid = request.worker_id or uuid.uuid4().hex[:8]
        self.workers[wid] = dict(
            id=wid, cpu_cores=request.cpu_cores, ram_bytes=request.ram_bytes,
            gpu_count=request.gpu_count, gpu_mem_bytes=request.gpu_mem_bytes,
            hardware=request.hardware or "cpu",
            supported_models=list(request.supported_models),
            load=0.0, active_tasks=0, last_seen=time.time(),
        )
        logger.info(f"worker registered: {wid} ({self.workers[wid]['hardware']})")
        return pb.RegisterResponse(worker_id=wid, heartbeat_interval=10, version=VERSION)

    async def Heartbeat(self, request, context):
        w = self.workers.get(request.worker_id)
        if w:
            w.update(last_seen=time.time(), load=request.load,
                     active_tasks=request.active_tasks)
        return pb.HeartbeatResponse(acknowledged=True)

    # ── Task Scheduling ──────────────────────────────────

    async def PollTasks(self, request, context):
        wid = request.worker_id
        async with self._lock:
            tasks = self._schedule(wid, request.max_tasks)
        return pb.PollResponse(tasks=tasks)

    def _schedule(self, worker_id: str, max_tasks: int) -> list[pb.TaskDef]:
        worker = self.workers.get(worker_id)
        if not worker or time.time() - worker.get("last_seen", 0) > 30:
            return []
        assigned = []
        remaining = []
        for tid in self._pending:
            task = self.tasks.get(tid)
            if not task:
                continue
            if len(assigned) >= max_tasks:
                remaining.append(tid)
                continue
            task["worker_id"] = worker_id
            task["assigned_at"] = time.time()
            worker["active_tasks"] = worker.get("active_tasks", 0) + 1
            assigned.append(self._task_to_proto(task))
            remaining.append(tid)
        self._pending = remaining
        return assigned

    def _task_to_proto(self, task: dict) -> pb.TaskDef:
        return pb.TaskDef(
            id=task["id"],
            func_name=task.get("func_name", ""),
            source=task.get("source", ""),
            requirements=task.get("requirements", []),
            payload=task.get("payload", b""),
            job_id=task.get("job_id", ""),
            filename=task.get("filename", ""),
            file_size=task.get("file_size", 0),
        )

    async def SubmitResult(self, request, context):
        tid = request.task_id
        task = self.tasks.get(tid)
        if task:
            task["status"] = "done" if request.success else "failed"
            task["result"] = request.output
            task["error"] = request.error
            task["duration"] = request.duration
            wid = task.get("worker_id", "")
            if wid in self.workers:
                self.workers[wid]["active_tasks"] = max(0, self.workers[wid]["active_tasks"] - 1)
            jid = task.get("job_id", "")
            if jid in self.jobs:
                job = self.jobs[jid]
                if request.success:
                    job["done"] = job.get("done", 0) + 1
                else:
                    job["failed"] = job.get("failed", 0) + 1
                if job["done"] + job["failed"] >= job["total"]:
                    job["status"] = "completed"
                    logger.info(f"batch {jid} done: {job['done']}/{job['total']}")
        return pb.Ack(ok=True)

    # ── Task Submission (Client) ──────────────────────────

    async def SubmitTask(self, request, context):
        if request.network_id != self.network_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid network_id")
        tid = request.task_id or uuid.uuid4().hex[:8]
        self.tasks[tid] = dict(
            id=tid, func_name=request.func_name, source=request.source,
            requirements=list(request.requirements), payload=request.payload,
            job_id="", filename="", file_size=0,
            status="pending", worker_id="", result="", error="", duration=0.0,
        )
        async with self._lock:
            self._pending.append(tid)
        return pb.TaskSubmitResponse(task_id=tid, status="pending")

    async def GetTaskResult(self, request, context):
        task = self.tasks.get(request.task_id)
        if not task:
            await context.abort(grpc.StatusCode.NOT_FOUND, "task not found")
        return pb.TaskResultResponse(
            task_id=task["id"], status=task.get("status", "unknown"),
            result=task.get("result", ""), error=task.get("error", ""),
            worker_id=task.get("worker_id", ""),
        )

    # ── Batch Jobs ───────────────────────────────────────

    async def CreateBatch(self, request, context):
        if request.network_id != self.network_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid network_id")
        jid = request.job_id or uuid.uuid4().hex[:8]
        job_dir = DATA_DIR / jid
        job_dir.mkdir(parents=True, exist_ok=True)

        task_count = max(len(request.filenames), len(request.arg_payloads))
        for i in range(task_count):
            tid = f"{jid}-{i:04d}"
            filename = request.filenames[i] if i < len(request.filenames) else ""
            fsize = (job_dir / filename).stat().st_size if filename and (job_dir / filename).exists() else 0
            arg_payload = request.arg_payloads[i] if i < len(request.arg_payloads) else b""

            self.tasks[tid] = dict(
                id=tid, func_name=request.func_name, source=request.source,
                requirements=list(request.requirements), payload=arg_payload,
                job_id=jid, filename=filename, file_size=fsize,
                status="pending", worker_id="", result="", error="", duration=0.0,
            )
            self._pending.append(tid)

        self.jobs[jid] = dict(id=jid, status="running", total=task_count, done=0, failed=0)
        logger.info(f"batch {jid}: {task_count} tasks")
        return pb.BatchCreateResponse(job_id=jid, task_count=task_count, status="running")

    async def GetBatchStatus(self, request, context):
        j = self.jobs.get(request.job_id)
        if not j:
            await context.abort(grpc.StatusCode.NOT_FOUND, "job not found")
        pending = j["total"] - j["done"] - j["failed"]
        return pb.BatchStatusResponse(
            job_id=j["id"], status=j["status"],
            total=j["total"], done=j["done"],
            failed=j["failed"], pending=pending,
        )

    # ── File Serving ─────────────────────────────────────

    async def GetFile(self, request, context):
        filepath = DATA_DIR / request.job_id / request.filename
        if not filepath.exists() or not filepath.is_file():
            await context.abort(grpc.StatusCode.NOT_FOUND, "file not found")
        total = filepath.stat().st_size
        offset = request.offset or 0
        async with asyncio.Lock():
            with open(filepath, "rb") as f:
                f.seek(offset)
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    yield pb.FileChunk(
                        filename=request.filename, data=chunk,
                        offset=offset, total_size=total,
                    )
                    offset += len(chunk)

    async def UploadFile(self, request_iterator, context):
        job_id = ""
        filename = ""
        path = None
        first = True
        async for chunk in request_iterator:
            job_id = chunk.job_id or job_id
            filename = chunk.filename or filename
            if not path:
                dirpath = DATA_DIR / job_id
                dirpath.mkdir(parents=True, exist_ok=True)
                path = dirpath / filename
                if path.exists():
                    path.unlink()
            mode = "wb" if first else "ab"
            with open(path, mode) as f:
                f.write(chunk.data)
            first = False
        total = path.stat().st_size if path else 0
        return pb.FileUploadResponse(path=str(path), total_size=total)

    # ── Model Registry ───────────────────────────────────

    async def RegisterModel(self, request, context):
        info = {k: v for k, v in request.info.items()}
        reg.register(request.name, info)
        return pb.Ack(ok=True)

    async def ListModels(self, request, context):
        models = reg.list()
        flat = {}
        for name, info in models.items():
            flat[name] = str(info)
        return pb.ModelListResponse(models=flat)

    # ── Info ───────────────────────────────────────────────

    async def GetInfo(self, request, context):
        return pb.InfoResponse(
            network_id=self.network_id, version=VERSION,
            workers=len(self.workers), pending_tasks=len(self._pending),
        )


async def serve(host="0.0.0.0", port=9090, relay_addr=""):
    network_id = uuid.uuid4().hex[:12]
    print(f"NETWORK_ID={network_id}", flush=True)
    logger.info(f"master starting on {host}:{port}  network_id={network_id}")

    server = grpc.aio.server()
    servicer = MasterServicer(network_id)
    rpcmod.add_MasterServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()

    if relay_addr:
        logger.info(f"connecting to relay at {relay_addr}")
        asyncio.create_task(_relay_loop(servicer, relay_addr, network_id))

    await server.wait_for_termination()


async def _relay_loop(servicer: MasterServicer, relay_addr: str, network_id: str):
    while True:
        try:
            async with grpc.aio.insecure_channel(relay_addr) as channel:
                stub = rpcmod.RelayStub(channel)
                async def _gen():
                    yield pb.RelayFrame(
                        network_id=network_id, sender_id="master",
                        sender_role="master", routing_key="hello",
                        payload=b"",
                    )
                    while True:
                        await asyncio.sleep(30)
                        yield pb.RelayFrame(
                            network_id=network_id, sender_id="master",
                            sender_role="master", routing_key="ping",
                        )

                async for frame in stub.Connect(_gen()):
                    if frame.routing_key == "poll":
                        wid = frame.sender_id
                        tasks = servicer._schedule(wid, 1)
                        # results go back via direct gRPC if worker connected directly
        except Exception as e:
            logger.warning(f"relay disconnected: {e}")
        await asyncio.sleep(5)


@click.command("master")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9090, type=int)
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--log-level", default="INFO")
def main(host, port, relay, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(serve(host=host, port=port, relay_addr=relay))
