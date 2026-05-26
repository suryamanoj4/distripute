import asyncio
import json
import logging
import os
import subprocess
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


class InferenceRunner:
    def __init__(self, plugin_cmd: str | None):
        self._cmd = plugin_cmd
        self._proc: subprocess.Popen | None = None

    async def ensure_ready(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self._cmd:
            return
        self._proc = await asyncio.create_subprocess_shell(
            self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    async def infer(self, task: dict) -> dict:
        if self._cmd:
            return await self._infer_subprocess(task)
        return await self._infer_fallback(task)

    async def _infer_subprocess(self, task: dict) -> dict:
        await self.ensure_ready()
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return dict(task_id=task["id"], success=False, error="plugin not ready", duration=0)

        line = json.dumps(task) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()
        resp = await self._proc.stdout.readline()
        result = json.loads(resp.decode())
        return result

    async def _infer_fallback(self, task: dict) -> dict:
        input_path = task.get("input_path", "")
        output_path = task.get("output_path", "")

        if task.get("task_type") == "asr":
            result = f"[mock-asr] transcribed: {Path(input_path).name}"
        elif task.get("task_type") == "ocr":
            result = f"[mock-ocr] ocr: {Path(input_path).name}"
        else:
            result = f"[mock-llm] generated: {task.get('params', {}).get('prompt', '')}"

        if output_path:
            Path(output_path).write_text(result)

        return dict(
            task_id=task["id"], success=True, output=result,
            error="", duration=0.0,
        )

    async def close(self):
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()

    @classmethod
    def detect_task_type(cls, supported_models: list[str]) -> str:
        for m in supported_models:
            m = m.lower()
            if "whisper" in m:
                return "asr"
            if "ocr" in m or "paddle" in m or "easyocr" in m:
                return "ocr"
        return "asr"


class WorkerNode:
    def __init__(self, master_addr="", relay_addr="", network_id="",
                 supported_models=None, plugin_cmd=None):
        self.master_addr = master_addr
        self.relay_addr = relay_addr
        self.network_id = network_id
        self.supported_models = supported_models or []
        self.worker_id = uuid.uuid4().hex[:8]
        self.runner = InferenceRunner(plugin_cmd)
        self._session: ClientSession | None = None
        self._hb_task: asyncio.Task | None = None
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

    async def _poll_and_work_http(self):
        task_type = InferenceRunner.detect_task_type(self.supported_models)
        while True:
            try:
                async with self._session.post(
                    f"http://{self.master_addr}/tasks/poll",
                    json=dict(worker_id=self.worker_id, max_tasks=2),
                ) as resp:
                    data = await resp.json()
                    for task in data.get("tasks", []):
                        self._active_tasks.add(task["id"])
                        result = await self.runner.infer(task)
                        result["task_type"] = task_type
                        async with self._session.post(
                            f"http://{self.master_addr}/tasks/result", json=result,
                        ):
                            pass
                        self._active_tasks.discard(task["id"])
            except Exception as e:
                logger.debug(f"poll error: {e}")
            await asyncio.sleep(2)

    async def _poll_and_work_relay(self, ws):
        task_type = InferenceRunner.detect_task_type(self.supported_models)
        while True:
            try:
                body = dict(worker_id=self.worker_id, max_tasks=2)
                await ws.send_json(dict(
                    type="forward", network_id=self.network_id,
                    payload=dict(action="poll", body=body),
                ))
            except Exception:
                pass
            await asyncio.sleep(2)

    async def _handle_relay_msg(self, msg, ws):
        if msg.type != WSMsgType.TEXT:
            return
        data = json.loads(msg.data)
        payload = data.get("payload", {})
        action = payload.get("action", "")

        if action == "poll_result":
            for task in payload.get("tasks", []):
                self._active_tasks.add(task["id"])
                result = await self.runner.infer(task)
                await ws.send_json(dict(
                    type="forward", network_id=self.network_id,
                    payload=dict(action="result", body=result),
                ))
                self._active_tasks.discard(task["id"])
        elif action == "register_ack":
            logger.info(f"relay confirmed registration: {payload.get('worker_id')}")

    async def run_forever(self):
        self._session = ClientSession()
        if self.master_addr:
            await self._register_http()
            self._hb_task = asyncio.create_task(self._heartbeat_loop_http())
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
        await self.runner.close()
        await self._session.close()


@click.command("worker")
@click.option("--master", default="", envvar="DISTRIPUTE_MASTER", help="Master address (host:port)")
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY", help="Relay address (host:port)")
@click.option("--network-id", default="", envvar="DISTRIPUTE_NETWORK_ID", required=True)
@click.option("--models", default="", help="Comma-separated supported models")
@click.option("--plugin", default="", help="Plugin command (e.g. 'python plugins/asr_infer.py')")
@click.option("--log-level", default="INFO")
def main(master, relay, network_id, models, plugin, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not master and not relay:
        raise click.UsageError("specify --master or --relay")
    supported = [m.strip() for m in models.split(",") if m.strip()]
    node = WorkerNode(master_addr=master, relay_addr=relay, network_id=network_id,
                      supported_models=supported, plugin_cmd=plugin or None)
    logger.info(f"worker {node.worker_id} starting (hw={HARDWARE}, gpus={GPU_COUNT})")
    asyncio.run(node.run_forever())
