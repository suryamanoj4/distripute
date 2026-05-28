import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import click
from aiohttp import ClientSession, WSMsgType

from . import VERSION

logger = logging.getLogger("distripute.worker")

HARDWARE = "cpu"
GPU_COUNT = 0
GPU_MEM = 0
CPU_CORES = os.cpu_count() or 1
RAM_BYTES = 0

try:
    import psutil as _ps
    RAM_BYTES = _ps.virtual_memory().total
except ImportError:
    RAM_BYTES = 8_000_000_000

try:
    from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    GPU_COUNT = nvmlDeviceGetCount()
    if GPU_COUNT > 0:
        HARDWARE = "gpu"
        handle = nvmlDeviceGetHandleByIndex(0)
        GPU_MEM = nvmlDeviceGetMemoryInfo(handle).total
except Exception:
    pass


async def _run_with_uv(source: str, func_name: str, args: tuple, kwargs: dict,
                       requirements: list[str]) -> dict:
    cache_dir = Path.home() / ".cache" / "distripute" / "tasks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(dir=cache_dir))

    try:
        source_path = work_dir / "source.py"
        source_path.write_text(source)

        runner_path = work_dir / "runner.py"
        runner_path.write_text(f"""
import sys, json, importlib.util
spec = importlib.util.spec_from_file_location("user_mod", "{source_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
func = getattr(mod, {json.dumps(func_name)})
args, kwargs = json.loads(sys.argv[1])
result = func(*args, **kwargs)
print(json.dumps({{"result": result}}))
""")

        if requirements:
            deps = " ".join(f"--with {r}" for r in requirements)
            cmd = f"uv run {deps} python3 {runner_path} '{json.dumps([list(args), kwargs])}'"
        else:
            cmd = f"uv run python3 {runner_path} '{json.dumps([list(args), kwargs])}'"

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(work_dir),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return dict(success=False, error=stderr.decode().strip() or f"exit {proc.returncode}",
                        duration=0.0)

        result_data = json.loads(stdout.decode())
        return dict(success=True, output=result_data.get("result"), duration=0.0)

    except Exception as e:
        return dict(success=False, error=str(e), duration=0.0)
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


class WorkerNode:
    def __init__(self, master_addr="", relay_addr="", network_id="",
                 supported_models=None):
        self.master_addr = master_addr
        self.relay_addr = relay_addr
        self.network_id = network_id
        self.supported_models = supported_models or []
        self.worker_id = uuid.uuid4().hex[:8]
        self._session: ClientSession | None = None
        self._active_tasks: set[str] = set()

    async def _register_http(self):
        async with self._session.post(
            f"http://{self.master_addr}/register",
            json=dict(
                network_id=self.network_id, worker_id=self.worker_id,
                cpu_cores=CPU_CORES, ram_bytes=RAM_BYTES,
                gpu_count=GPU_COUNT, gpu_mem_bytes=GPU_MEM,
                hardware=HARDWARE, supported_models=self.supported_models,
            ),
        ) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"registration failed: {data['error']}")
            logger.info(f"registered as {data['worker_id']}")
            return data

    async def _register_relay(self, ws):
        await ws.send_json(dict(
            type="worker_hello", network_id=self.network_id,
            worker_id=self.worker_id, version=VERSION,
        ))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "worker_ack":
                    logger.info(f"registered via relay as {data['worker_id']}")
                    return True
        return False

    async def _heartbeat_loop_http(self):
        while True:
            try:
                async with self._session.post(
                    f"http://{self.master_addr}/heartbeat",
                    json=dict(worker_id=self.worker_id, load=0.5,
                              active_tasks=len(self._active_tasks)),
                ):
                    pass
            except Exception:
                pass
            await asyncio.sleep(10)

    async def _heartbeat_loop_relay(self, ws):
        while True:
            try:
                await ws.send_json(dict(
                    type="forward", network_id=self.network_id,
                    payload=dict(action="heartbeat",
                                 body=dict(worker_id=self.worker_id, load=0.5,
                                           active_tasks=len(self._active_tasks))),
                ))
            except Exception:
                pass
            await asyncio.sleep(10)

    async def _execute_task(self, task: dict) -> dict:
        func_name = task.get("func_name", "unknown")
        source = task.get("source", "")
        requirements = task.get("requirements", [])
        payload = task.get("payload", "")

        if not source:
            return dict(task_id=task["id"], success=False,
                        error="no source code in task", duration=0.0)

        import base64, cloudpickle
        try:
            args, kwargs = cloudpickle.loads(base64.b64decode(payload))
        except Exception:
            args = task.get("args", [])
            kwargs = task.get("kwargs", {})

        logger.info(f"executing {func_name} (reqs: {requirements})")
        result = await _run_with_uv(source, func_name, tuple(args), kwargs, requirements)
        result["task_id"] = task["id"]
        return result

    async def _poll_and_work_http(self):
        while True:
            try:
                async with self._session.post(
                    f"http://{self.master_addr}/tasks/poll",
                    json=dict(worker_id=self.worker_id, max_tasks=1),
                ) as resp:
                    data = await resp.json()
                    for task in data.get("tasks", []):
                        self._active_tasks.add(task["id"])
                        result = await self._execute_task(task)
                        async with self._session.post(
                            f"http://{self.master_addr}/tasks/result", json=result,
                        ):
                            pass
                        self._active_tasks.discard(task["id"])
            except Exception as e:
                logger.debug(f"poll error: {e}")
            await asyncio.sleep(1)

    async def _poll_and_work_relay(self, ws):
        while True:
            try:
                body = dict(worker_id=self.worker_id, max_tasks=1)
                await ws.send_json(dict(
                    type="forward", network_id=self.network_id,
                    payload=dict(action="poll", body=body),
                ))
            except Exception:
                pass
            await asyncio.sleep(1)

    async def _handle_relay_msg(self, msg, ws):
        if msg.type != WSMsgType.TEXT:
            return
        data = json.loads(msg.data)
        payload = data.get("payload", {})
        action = payload.get("action", "")

        if action == "poll_result":
            for task in payload.get("tasks", []):
                self._active_tasks.add(task["id"])
                result = await self._execute_task(task)
                await ws.send_json(dict(
                    type="forward", network_id=self.network_id,
                    payload=dict(action="result", body=result),
                ))
                self._active_tasks.discard(task["id"])

    async def run_forever(self):
        self._session = ClientSession()
        if self.master_addr:
            await self._register_http()
            asyncio.create_task(self._heartbeat_loop_http())
            await self._poll_and_work_http()
        elif self.relay_addr:
            while True:
                try:
                    async with self._session.ws_connect(f"ws://{self.relay_addr}/ws") as ws:
                        if await self._register_relay(ws):
                            asyncio.create_task(self._heartbeat_loop_relay(ws))
                            async for msg in ws:
                                await self._handle_relay_msg(msg, ws)
                except Exception as e:
                    logger.warning(f"relay disconnected: {e}")
                await asyncio.sleep(5)
        await self._session.close()


@click.command("worker")
@click.option("--master", default="", envvar="DISTRIPUTE_MASTER", help="Master address")
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY", help="Relay address")
@click.option("--network-id", default="", envvar="DISTRIPUTE_NETWORK_ID", required=True)
@click.option("--log-level", default="INFO")
def main(master, relay, network_id, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not master and not relay:
        raise click.UsageError("specify --master or --relay")
    node = WorkerNode(master_addr=master, relay_addr=relay, network_id=network_id)
    logger.info(f"worker {node.worker_id} starting (hw={HARDWARE}, gpus={GPU_COUNT})")
    asyncio.run(node.run_forever())
