import asyncio
import json
import logging
import uuid
from pathlib import Path

import click
from aiohttp import web, ClientSession, WSMsgType

from . import VERSION
from .registry import _default_registry as reg

logger = logging.getLogger("distripute.master")

TASK_TYPES = {"asr", "ocr", "llm"}
STRATEGIES = {"data_parallel", "model_parallel", "pipeline_parallel"}


class MasterNode:
    def __init__(self, host="0.0.0.0", port=9090, relay_addr=""):
        self.host = host
        self.port = port
        self.relay_addr = relay_addr
        self.network_id = uuid.uuid4().hex[:12]
        self.workers: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        self._pending: list[str] = []
        self._http = web.Application()
        self._runner = None
        self._ws: ClientSession | None = None
        self._ws_conn = None
        self._build_routes()

    def _build_routes(self):
        self._http["master"] = self
        r = self._http.router
        r.add_post("/register", self._handle_register)
        r.add_post("/heartbeat", self._handle_heartbeat)
        r.add_post("/tasks/poll", self._handle_poll)
        r.add_post("/tasks/result", self._handle_result)
        r.add_post("/task", self._handle_submit_task)
        r.add_get("/task/{tid}", self._handle_get_task)
        r.add_post("/job", self._handle_create_job)
        r.add_get("/job/{jid}", self._handle_get_job)
        r.add_get("/jobs", self._handle_list_jobs)
        r.add_get("/workers", self._handle_list_workers)
        r.add_get("/info", self._handle_info)
        r.add_post("/models/register", self._handle_register_model)
        r.add_get("/models", self._handle_list_models)

    def _eligible_workers(self, model: str) -> list[dict]:
        now = asyncio.get_event_loop().time()
        candidates = []
        for w in self.workers.values():
            if now - w.get("last_seen", 0) > 30:
                continue
            sm = w.get("supported_models", [])
            if sm and model not in sm:
                continue
            candidates.append(w)
        candidates.sort(key=lambda w: w.get("load", 1.0))
        return candidates

    def _schedule(self, worker_id: str, max_tasks: int) -> list[dict]:
        worker = self.workers.get(worker_id)
        if not worker:
            return []
        now = asyncio.get_event_loop().time()
        if now - worker.get("last_seen", 0) > 30:
            return []

        assigned = []
        remaining = []
        for tid in self._pending:
            task = self.tasks.get(tid)
            if not task:
                continue
            candidates = self._eligible_workers(task.get("model", ""))
            if any(c["id"] == worker_id for c in candidates):
                if len(assigned) >= max_tasks:
                    remaining.append(tid)
                    continue
                task["status"] = "running"
                task["worker_id"] = worker_id
                worker["active_tasks"] = worker.get("active_tasks", 0) + 1
                assigned.append(dict(task))
            else:
                remaining.append(tid)
        self._pending = remaining
        return assigned

    async def _scan_input(self, input_source: str, task_type: str) -> list[Path]:
        p = Path(input_source)
        if p.is_file():
            return [p]
        if not p.is_dir():
            raise ValueError(f"input not found: {input_source}")
        if task_type == "asr":
            exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
        elif task_type == "ocr":
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pdf"}
        else:
            exts = {".txt", ".json"}
        return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])

    async def create_job(self, task_type: str, model: str, strategy: str,
                         input_source: str, output_sink: str,
                         params: dict | None = None) -> dict:
        jid = uuid.uuid4().hex[:8]
        out = Path(output_sink)
        out.mkdir(parents=True, exist_ok=True)
        files = await self._scan_input(input_source, task_type)
        tasks = []
        for i, f in enumerate(files):
            tid = f"{jid}-{i:04d}"
            task = dict(
                id=tid, job_id=jid, task_type=task_type, model=model,
                strategy=strategy, input_path=str(f.resolve()),
                output_path=str(out / f"{f.stem}_result.txt"),
                params=params or {}, status="pending",
                worker_id="", result="", error="", duration=0.0,
            )
            tasks.append(task)
            self.tasks[tid] = task
            self._pending.append(tid)
        job = dict(
            id=jid, task_type=task_type, model=model, strategy=strategy,
            input_source=input_source, output_sink=output_sink,
            params=params or {}, status="running",
            total=len(tasks), done=0, failed=0,
            created_at=asyncio.get_event_loop().time(),
        )
        self.jobs[jid] = job
        logger.info(f"job {jid}: {len(tasks)} tasks, model={model}, strategy={strategy}")
        return job

    # ── HTTP handlers ──────────────────────────────────────

    async def _handle_register(self, request):
        data = await request.json()
        if data.get("network_id") != self.network_id:
            return web.json_response({"error": "invalid network_id"}, status=403)
        wid = data.get("worker_id") or uuid.uuid4().hex[:8]
        self.workers[wid] = dict(
            id=wid, address=f"{request.remote}:{data.get('port', 0)}",
            cpu_cores=data.get("cpu_cores", 0), ram_bytes=data.get("ram_bytes", 0),
            gpu_count=data.get("gpu_count", 0), gpu_mem_bytes=data.get("gpu_mem_bytes", 0),
            hardware=data.get("hardware", "cpu"),
            supported_models=data.get("supported_models", []),
            load=0.0, active_tasks=0, last_seen=asyncio.get_event_loop().time(),
        )
        logger.info(f"worker registered: {wid} ({self.workers[wid]['hardware']})")
        return web.json_response(dict(worker_id=wid, heartbeat_interval=10, version=VERSION))

    async def _handle_heartbeat(self, request):
        data = await request.json()
        wid = data.get("worker_id", "")
        if wid in self.workers:
            s = asyncio.get_event_loop().time()
            self.workers[wid].update(last_seen=s, load=data.get("load", 0.0),
                                     active_tasks=data.get("active_tasks", 0))
        return web.json_response({"ok": True})

    async def _handle_poll(self, request):
        data = await request.json()
        wid = data.get("worker_id", "")
        if wid not in self.workers:
            return web.json_response({"error": "unknown worker"}, status=403)
        tasks = self._schedule_func_task(wid, data.get("max_tasks", 4))
        return web.json_response({"tasks": tasks})

    async def _handle_result(self, request):
        data = await request.json()
        tid = data.get("task_id", "")
        task = self.tasks.get(tid)
        if not task:
            return web.json_response({"error": "unknown task"}, status=404)
        task["status"] = "done" if data.get("success") else "failed"
        task["result"] = data.get("output", "")
        task["error"] = data.get("error", "")
        task["duration"] = data.get("duration", 0.0)
        wid = task.get("worker_id", "")
        if wid in self.workers:
            self.workers[wid]["active_tasks"] = max(0, self.workers[wid]["active_tasks"] - 1)
        # update job progress if this task belongs to a job
        jid = task.get("job_id")
        if jid and jid in self.jobs:
            job = self.jobs[jid]
            if data.get("success"):
                job["done"] = job.get("done", 0) + 1
            else:
                job["failed"] = job.get("failed", 0) + 1
            if job["done"] + job["failed"] >= job["total"]:
                job["status"] = "completed" if job["failed"] == 0 else "failed"
        return web.json_response({"ok": True})

    async def _handle_create_job(self, request):
        data = await request.json()
        try:
            job = await self.create_job(
                task_type=data["task_type"],
                model=data.get("model", "whisper-large-v3"),
                strategy=data.get("strategy", "data_parallel"),
                input_source=data["input_source"],
                output_sink=data["output_sink"],
                params=data.get("model_params"),
            )
            return web.json_response(dict(job_id=job["id"], task_count=job["total"], status=job["status"]))
        except Exception as e:
            logger.exception("create_job")
            return web.json_response({"error": str(e)}, status=400)

    async def _handle_get_job(self, request):
        j = self.jobs.get(request.match_info["jid"])
        if not j:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(j)

    async def _handle_list_jobs(self, _r):
        return web.json_response({"jobs": list(self.jobs.values())})

    async def _handle_list_workers(self, _r):
        return web.json_response({"workers": list(self.workers.values())})

    async def _handle_info(self, _r):
        return web.json_response(dict(
            network_id=self.network_id, version=VERSION,
            workers=len(self.workers), jobs=len(self.jobs),
        ))

    async def _handle_register_model(self, request):
        data = await request.json()
        reg.register(data["name"], data["info"])
        logger.info(f"model registered: {data['name']}")
        return web.json_response({"ok": True})

    async def _handle_list_models(self, _r):
        return web.json_response({"models": reg.list()})

    # ── @distripute.task handlers ────────────────────────────

    async def _handle_submit_task(self, request):
        data = await request.json()
        if data.get("network_id") != self.network_id:
            return web.json_response({"error": "invalid network_id"}, status=403)

        tid = data.get("task_id", uuid.uuid4().hex[:8])
        task = dict(
            id=tid,
            func_name=data.get("func_name", "unknown"),
            source=data.get("source", ""),
            requirements=data.get("requirements", []),
            payload=data.get("payload", ""),
            args=data.get("args", []),
            kwargs=data.get("kwargs", {}),
            status="pending",
            worker_id="",
            result=None,
            error="",
        )
        self.tasks[tid] = task
        self._pending.append(tid)
        logger.info(f"task {tid}: {task['func_name']} queued")
        return web.json_response({"task_id": tid, "status": "pending"})

    async def _handle_get_task(self, request):
        tid = request.match_info["tid"]
        task = self.tasks.get(tid)
        if not task:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({
            "task_id": task["id"],
            "status": task["status"],
            "result": task.get("result"),
            "error": task.get("error"),
            "worker_id": task.get("worker_id", ""),
        })

    def _schedule_func_task(self, worker_id: str, max_tasks: int) -> list[dict]:
        worker = self.workers.get(worker_id)
        if not worker:
            return []
        now = asyncio.get_event_loop().time()
        if now - worker.get("last_seen", 0) > 30:
            return []

        assigned = []
        remaining = []
        for tid in self._pending:
            task = self.tasks.get(tid)
            if not task:
                continue
            if not task.get("source"):
                remaining.append(tid)
                continue
            if len(assigned) >= max_tasks:
                remaining.append(tid)
                continue
            task["status"] = "running"
            task["worker_id"] = worker_id
            worker["active_tasks"] = worker.get("active_tasks", 0) + 1
            assigned.append(dict(task))
            remaining.append(tid)

        self._pending = remaining
        return assigned

    # ── Relay WebSocket client ──────────────────────────────

    async def _relay_loop(self):
        session = ClientSession()
        while True:
            try:
                async with session.ws_connect(f"ws://{self.relay_addr}/ws") as ws:
                    await ws.send_json({"type": "master_hello", "network_id": self.network_id, "version": VERSION})
                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            typ = data.get("type")
                            if typ == "worker_joined":
                                logger.info(f"relay: worker joined {data.get('worker_id')}")
                            elif typ == "worker_left":
                                logger.info(f"relay: worker left {data.get('worker_id')}")
                            elif typ == "forward":
                                payload = data.get("payload", {})
                                await self._dispatch_relay_request(payload, ws)
            except Exception as e:
                logger.warning(f"relay connection failed: {e}")
            await asyncio.sleep(5)

    async def _dispatch_relay_request(self, payload: dict, ws):
        if not isinstance(payload, dict):
            return
        action = payload.get("action", "")
        body = payload.get("body", {})

        if action == "register":
            # simulate an HTTP request locally
            wid = await self._register_via_relay(body, ws)
            if wid:
                logger.info(f"relay: worker {wid} registered via relay")
        elif action == "heartbeat":
            wid = body.get("worker_id", "")
            if wid in self.workers:
                self.workers[wid].update(last_seen=asyncio.get_event_loop().time(),
                                         load=body.get("load", 0.0),
                                         active_tasks=body.get("active_tasks", 0))
        elif action == "poll":
            wid = body.get("worker_id", "")
            tasks = self._schedule_func_task(wid, body.get("max_tasks", 4))
            await ws.send_json({"type": "forward", "network_id": self.network_id,
                                "target_worker": wid,
                                "payload": {"action": "poll_result", "tasks": tasks}})
        elif action == "result":
            await self._handle_result_relay(body, ws)

    async def _register_via_relay(self, body: dict, ws) -> str | None:
        if body.get("network_id") != self.network_id:
            return None
        wid = body.get("worker_id") or uuid.uuid4().hex[:8]
        self.workers[wid] = dict(
            id=wid, address=f"relay:{wid}",
            cpu_cores=body.get("cpu_cores", 0), ram_bytes=body.get("ram_bytes", 0),
            gpu_count=body.get("gpu_count", 0), gpu_mem_bytes=body.get("gpu_mem_bytes", 0),
            hardware=body.get("hardware", "cpu"),
            supported_models=body.get("supported_models", []),
            load=0.0, active_tasks=0, last_seen=asyncio.get_event_loop().time(),
        )
        await ws.send_json({"type": "forward", "target_worker": wid,
                            "payload": {"action": "register_ack", "worker_id": wid,
                                        "heartbeat_interval": 10, "version": VERSION}})
        return wid

    async def _handle_result_relay(self, body: dict, ws):
        tid = body.get("task_id", "")
        task = self.tasks.get(tid)
        if not task:
            return
        task["status"] = "done" if body.get("success") else "failed"
        task["result"] = body.get("output", "")
        task["error"] = body.get("error", "")
        task["duration"] = body.get("duration", 0.0)
        wid = task.get("worker_id", "")
        if wid in self.workers:
            self.workers[wid]["active_tasks"] = max(0, self.workers[wid]["active_tasks"] - 1)
        job = self.jobs.get(task["job_id"])
        if job:
            if task["status"] == "done":
                job["done"] += 1
            else:
                job["failed"] += 1
            if job["done"] + job["failed"] >= job["total"]:
                job["status"] = "completed" if job["failed"] == 0 else "failed"

    # ── Start ──────────────────────────────────────────────

    async def run_forever(self):
        self._runner = web.AppRunner(self._http)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(f"NETWORK_ID={self.network_id}", flush=True)
        logger.info(f"master on {self.host}:{self.port}  network_id={self.network_id}")
        if self.relay_addr:
            logger.info(f"connecting to relay at {self.relay_addr}")
            asyncio.create_task(self._relay_loop())
        while True:
            await asyncio.sleep(3600)


@click.command("master")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9090, type=int)
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY", help="Relay address (host:port)")
@click.option("--log-level", default="INFO")
def main(host, port, relay, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    node = MasterNode(host=host, port=port, relay_addr=relay)
    asyncio.run(node.run_forever())
