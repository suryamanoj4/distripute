import asyncio
import logging
import time
import uuid
from pathlib import Path

import click
import grpc

from . import VERSION
from .grpc import pb, grpc as rpcmod
from .cache import Cache
from .registry import _default_registry as reg

logger = logging.getLogger("distripute.master")

DATA_DIR = Path.home() / ".cache" / "distripute" / "data"


class MasterServicer(rpcmod.MasterServicer):
    def __init__(self, network_id: str, redis_url: str = ""):
        self.network_id = network_id
        self.cache = Cache(redis_url)
        self._lock = asyncio.Lock()

    # ── Worker Registration ──────────────────────────────

    async def Register(self, request, context):
        if request.network_id != self.network_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid network_id")
        wid = request.worker_id or uuid.uuid4().hex[:8]
        self.cache.worker_set(wid, dict(
            id=wid, cpu_cores=request.cpu_cores, ram_bytes=request.ram_bytes,
            gpu_count=request.gpu_count, gpu_mem_bytes=request.gpu_mem_bytes,
            hardware=request.hardware or "cpu",
            supported_models=list(request.supported_models),
            load=0.0, active_tasks=0, last_seen=time.time(),
        ))
        logger.info(f"worker registered: {wid} ({request.hardware})")
        return pb.RegisterResponse(worker_id=wid, heartbeat_interval=10, version=VERSION)

    async def Heartbeat(self, request, context):
        self.cache.worker_update(request.worker_id,
                                 last_seen=time.time(),
                                 load=request.load,
                                 active_tasks=request.active_tasks)
        return pb.HeartbeatResponse(acknowledged=True)

    # ── Task Scheduling ──────────────────────────────────

    async def PollTasks(self, request, context):
        async with self._lock:
            tasks = self._schedule(request.worker_id, request.max_tasks)
        return pb.PollResponse(tasks=tasks)

    def _schedule(self, worker_id: str, max_tasks: int) -> list[pb.TaskDef]:
        worker = self.cache.worker_get(worker_id)
        if not worker:
            return []
        if time.time() - worker.get("last_seen", 0) > 30:
            return []

        # reclaim tasks from dead workers
        for w in self.cache.worker_list():
            if time.time() - w.get("_ts", 0) > 30:
                self.cache.worker_delete(w["id"])

        assigned = []
        assigned_ids = []
        pending_ids = self.cache.pending_peek(max_tasks)
        for tid in pending_ids:
            self.cache.pending_remove(tid)
            t = self.cache.task_get(tid)
            if not t:
                continue
            t["status"] = "running"
            t["worker_id"] = worker_id
            self.cache.task_set(tid, t)
            worker["active_tasks"] = worker.get("active_tasks", 0) + 1
            self.cache.worker_update(worker_id, active_tasks=worker["active_tasks"])
            assigned.append(self._task_to_proto(t))
            assigned_ids.append(tid)

        return assigned

    def _task_to_proto(self, task: dict) -> pb.TaskDef:
        return pb.TaskDef(
            id=task.get("id", ""),
            func_name=task.get("func_name", ""),
            source=task.get("source", ""),
            requirements=task.get("requirements", []),
            payload=task.get("payload", b""),
            job_id=task.get("job_id", ""),
            filename=task.get("filename", ""),
            file_size=int(task.get("file_size", 0)),
        )

    async def SubmitResult(self, request, context):
        tid = request.task_id
        t = self.cache.task_get(tid)
        if t:
            t["status"] = "done" if request.success else "failed"
            t["result"] = request.output
            t["error"] = request.error
            t["duration"] = request.duration
            self.cache.task_set(tid, t)

            wid = t.get("worker_id", "")
            w = self.cache.worker_get(wid)
            if w:
                at = max(0, w.get("active_tasks", 0) - 1)
                self.cache.worker_update(wid, active_tasks=at)

            jid = t.get("job_id", "")
            j = self.cache.job_get(jid)
            if j:
                if request.success:
                    j["done"] = j.get("done", 0) + 1
                else:
                    j["failed"] = j.get("failed", 0) + 1
                if j["done"] + j["failed"] >= j["total"]:
                    j["status"] = "completed"
                    logger.info(f"batch {jid} done: {j['done']}/{j['total']}")
                self.cache.job_set(jid, j)

        return pb.Ack(ok=True)

    # ── Task Submission (Client) ──────────────────────────

    async def SubmitTask(self, request, context):
        if request.network_id != self.network_id:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "invalid network_id")
        tid = request.task_id or uuid.uuid4().hex[:8]
        self.cache.task_set(tid, dict(
            id=tid, func_name=request.func_name, source=request.source,
            requirements=list(request.requirements), payload=request.payload,
            job_id="", filename="", file_size=0,
            status="pending", worker_id="", result="", error="", duration=0.0,
        ))
        async with self._lock:
            self.cache.pending_push(tid)
        return pb.TaskSubmitResponse(task_id=tid, status="pending")

    async def GetTaskResult(self, request, context):
        t = self.cache.task_get(request.task_id)
        if not t:
            await context.abort(grpc.StatusCode.NOT_FOUND, "task not found")
        return pb.TaskResultResponse(
            task_id=t["id"], status=t.get("status", "unknown"),
            result=t.get("result", ""), error=t.get("error", ""),
            worker_id=t.get("worker_id", ""),
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
            fname = request.filenames[i] if i < len(request.filenames) else ""
            fp = job_dir / fname if fname else None
            fsize = fp.stat().st_size if fp and fp.exists() else 0
            arg_payload = request.arg_payloads[i] if i < len(request.arg_payloads) else b""

            self.cache.task_set(tid, dict(
                id=tid, func_name=request.func_name, source=request.source,
                requirements=list(request.requirements), payload=arg_payload,
                job_id=jid, filename=fname, file_size=fsize,
                status="pending", worker_id="", result="", error="", duration=0.0,
            ))
            self.cache.pending_push(tid)

        self.cache.job_set(jid, dict(id=jid, status="running", total=task_count, done=0, failed=0))
        logger.info(f"batch {jid}: {task_count} tasks")
        return pb.BatchCreateResponse(job_id=jid, task_count=task_count, status="running")

    async def GetBatchStatus(self, request, context):
        j = self.cache.job_get(request.job_id)
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
        fp = DATA_DIR / request.job_id / request.filename
        if not fp.exists() or not fp.is_file():
            await context.abort(grpc.StatusCode.NOT_FOUND, "file not found")
        total = fp.stat().st_size
        offset = request.offset or 0
        with open(fp, "rb") as f:
            f.seek(offset)
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                yield pb.FileChunk(
                    job_id=request.job_id, filename=request.filename,
                    data=chunk, offset=offset, total_size=total,
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
                d = DATA_DIR / job_id
                d.mkdir(parents=True, exist_ok=True)
                path = d / filename
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
            workers=self.cache.worker_count(),
            pending_tasks=self.cache.pending_count(),
        )


async def serve(host="0.0.0.0", port=9090, relay_addr="", redis_url=""):
    network_id = uuid.uuid4().hex[:12]
    print(f"NETWORK_ID={network_id}", flush=True)

    if redis_url:
        logger.info(f"redis cache at {redis_url}")
    else:
        logger.info("no redis — using in-memory cache")

    server = grpc.aio.server()
    servicer = MasterServicer(network_id, redis_url=redis_url)
    rpcmod.add_MasterServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info(f"master on {host}:{port}  network_id={network_id}")

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
                    )
                    while True:
                        await asyncio.sleep(30)
                        yield pb.RelayFrame(
                            network_id=network_id, sender_id="master",
                            sender_role="master", routing_key="ping",
                        )
                async for frame in stub.Connect(_gen()):
                    pass
        except Exception as e:
            logger.warning(f"relay disconnected: {e}")
        await asyncio.sleep(5)


@click.command("master")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9090, type=int)
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--redis", default="", envvar="DISTRIPUTE_REDIS", help="Redis URL (redis://localhost:6379)")
@click.option("--log-level", default="INFO")
def main(host, port, relay, redis, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(serve(host=host, port=port, relay_addr=relay, redis_url=redis))
