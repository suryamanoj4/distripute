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
import grpc
import cloudpickle

from . import VERSION
from .grpc import pb, grpc as rpcmod

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

TASK_CACHE = Path.home() / ".cache" / "distripute" / "tasks"
RELAY_UNSUPPORTED_FILE_ERROR = "relay mode does not yet support file-backed task transfer"
RELAY_REGISTER_TIMEOUT = 5.0


async def _run_with_uv(source: str, func_name: str, payload: bytes,
                       requirements: list[str], filename: str = "",
                       file_path: str = "") -> dict:
    work_dir = Path(tempfile.mkdtemp(dir=TASK_CACHE))
    try:
        source_path = work_dir / "source.py"
        source_path.write_text(source)

        # Write arg payload
        args_path = work_dir / "args.bin"
        args_path.write_bytes(payload)

        # Write file data if present
        file_arg = ""
        if file_path:
            dest = work_dir / filename
            dest.write_bytes(Path(file_path).read_bytes())
            file_arg = str(dest)

        runner = work_dir / "runner.py"
        runner.write_text(f"""
import sys, json, cloudpickle
sys.path.insert(0, '{work_dir}')
import source
func = getattr(source, {json.dumps(func_name)})
with open('{args_path}', 'rb') as f:
    args, kwargs = cloudpickle.loads(f.read())
result = func({file_arg + ', ' if file_arg else ''}*args, **kwargs)
print(json.dumps({{"result": result}}))
""")

        deps = " ".join(f"--with {r}" for r in requirements)
        cmd = f"uv run {deps} python3 {runner}" if requirements else f"python3 {runner}"
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(work_dir),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode().strip() or f"exit {proc.returncode}"
            return dict(success=False, output="", error=err, duration=0.0)
        result_data = json.loads(stdout.decode())
        return dict(success=True, output=result_data.get("result", ""), error="", duration=0.0)
    except Exception as e:
        return dict(success=False, output="", error=str(e), duration=0.0)
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


async def _download_file(channel, job_id: str, filename: str, dest: Path) -> str:
    stub = rpcmod.MasterStub(channel)
    filepath = dest / filename
    async with stub.GetFile(pb.FileRequest(job_id=job_id, filename=filename)) as stream:
        with open(filepath, "wb") as f:
            async for chunk in stream:
                f.write(chunk.data)
    return str(filepath)


async def _execute_task(
    task: pb.TaskDef,
    channel=None,
    allow_file_download: bool = True,
    worker_id: str = "",
) -> pb.TaskResult:
    file_path = ""
    if task.filename and task.file_size > 0:
        if not allow_file_download or channel is None:
            return pb.TaskResult(
                task_id=task.id,
                success=False,
                output="",
                error=RELAY_UNSUPPORTED_FILE_ERROR,
                duration=0.0,
                worker_id=worker_id,
            )
        dest = TASK_CACHE / task.job_id
        dest.mkdir(parents=True, exist_ok=True)
        file_path = await _download_file(channel, task.job_id, task.filename, dest)

    result = await _run_with_uv(
        task.source,
        task.func_name,
        task.payload,
        list(task.requirements),
        task.filename,
        file_path,
    )
    return pb.TaskResult(
        task_id=task.id,
        success=result["success"],
        output=str(result["output"]),
        error=result.get("error", ""),
        duration=result.get("duration", 0.0),
        worker_id=worker_id,
    )


async def _run_relay_task(
    relay_task: pb.TaskDef,
    outbound: asyncio.Queue[pb.RelayFrame | None],
    network_id: str,
    worker_id: str,
):
    result = await _execute_task(
        relay_task,
        allow_file_download=False,
        worker_id=worker_id,
    )
    await outbound.put(pb.RelayFrame(
        network_id=network_id,
        sender_id=worker_id,
        sender_role="worker",
        routing_key="submit_result",
        payload=result.SerializeToString(),
    ))


def _track_background_task(task_set: set[asyncio.Task], task: asyncio.Task):
    task_set.add(task)

    def _cleanup(done: asyncio.Task):
        task_set.discard(done)
        try:
            exc = done.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning(f"relay task execution failed: {exc}")

    task.add_done_callback(_cleanup)
    return task


async def _handle_relay_frame(
    frame: pb.RelayFrame,
    *,
    outbound: asyncio.Queue[pb.RelayFrame | None],
    network_id: str,
    worker_id: str,
    registered: asyncio.Event,
    inflight_tasks: set[asyncio.Task],
):
    if frame.routing_key == "poll_response":
        resp = pb.PollResponse.FromString(frame.payload)
        for relay_task in resp.tasks:
            task = asyncio.create_task(_run_relay_task(
                relay_task,
                outbound,
                network_id,
                worker_id,
            ))
            _track_background_task(inflight_tasks, task)
        return None

    if frame.routing_key == "register_response":
        resp = pb.RegisterResponse.FromString(frame.payload)
        registered.set()
        logger.info(f"registered as {resp.worker_id} via relay")
        return None

    if frame.routing_key == "register_error":
        try:
            payload = json.loads(frame.payload.decode() or "{}")
        except json.JSONDecodeError:
            payload = {"message": frame.payload.decode(errors="replace")}
        message = payload.get("message", "registration failed")
        code = payload.get("code", "UNKNOWN")
        logger.error(f"relay registration failed ({code}): {message}")
        return "register_error"

    if frame.routing_key == "master_left":
        logger.warning("master disconnected from relay")
        return "master_left"

    return None


async def run_worker(master_addr="", relay_addr="", network_id="", supported_models=None):
    wid = uuid.uuid4().hex[:8]
    logger.info(f"worker {wid} starting (hw={HARDWARE}, gpus={GPU_COUNT})")
    TASK_CACHE.mkdir(parents=True, exist_ok=True)

    if master_addr:
        async with grpc.aio.insecure_channel(master_addr) as channel:
            stub = rpcmod.MasterStub(channel)
            try:
                resp = await stub.Register(pb.RegisterRequest(
                    network_id=network_id, worker_id=wid,
                    cpu_cores=CPU_CORES, ram_bytes=RAM_BYTES,
                    gpu_count=GPU_COUNT, gpu_mem_bytes=GPU_MEM,
                    hardware=HARDWARE, supported_models=supported_models or [],
                ))
                logger.info(f"registered as {resp.worker_id}")
            except grpc.RpcError as e:
                logger.error(f"registration failed: {e}")
                return

            # Heartbeat loop
            async def _heartbeat():
                while True:
                    try:
                        await stub.Heartbeat(pb.HeartbeatRequest(worker_id=wid))
                    except Exception:
                        pass
                    await asyncio.sleep(10)

            asyncio.create_task(_heartbeat())

            # Work loop
            while True:
                try:
                    resp = await stub.PollTasks(pb.PollRequest(worker_id=wid, max_tasks=1))
                    for task in resp.tasks:
                        result = await _execute_task(
                            task,
                            channel=channel,
                            allow_file_download=True,
                            worker_id=wid,
                        )
                        await stub.SubmitResult(result)
                except Exception as e:
                    logger.debug(f"poll error: {e}")
                await asyncio.sleep(1)

    elif relay_addr:
        logger.info(f"connecting to relay at {relay_addr}")
        while True:
            heartbeat_task = None
            poll_task = None
            register_timeout_task = None
            inflight_tasks: set[asyncio.Task] = set()
            fatal_registration_error = False
            try:
                async with grpc.aio.insecure_channel(relay_addr) as channel:
                    stub = rpcmod.RelayStub(channel)
                    outbound: asyncio.Queue[pb.RelayFrame | None] = asyncio.Queue()
                    registered = asyncio.Event()

                    async def _gen():
                        await outbound.put(pb.RelayFrame(
                            network_id=network_id, sender_id=wid,
                            sender_role="worker", routing_key="register",
                            payload=pb.RegisterRequest(
                                network_id=network_id, worker_id=wid,
                                cpu_cores=CPU_CORES, ram_bytes=RAM_BYTES,
                                gpu_count=GPU_COUNT, gpu_mem_bytes=GPU_MEM,
                                hardware=HARDWARE, supported_models=supported_models or [],
                            ).SerializeToString(),
                        ))
                        while True:
                            frame = await outbound.get()
                            if frame is None:
                                return
                            yield frame

                    async def _registration_timeout():
                        await asyncio.sleep(RELAY_REGISTER_TIMEOUT)
                        if not registered.is_set():
                            logger.error("relay registration timed out")
                            await channel.close()

                    async def _heartbeat():
                        await registered.wait()
                        while True:
                            await asyncio.sleep(10)
                            await outbound.put(pb.RelayFrame(
                                network_id=network_id,
                                sender_id=wid,
                                sender_role="worker",
                                routing_key="heartbeat",
                                payload=pb.HeartbeatRequest(worker_id=wid).SerializeToString(),
                            ))

                    async def _poll():
                        await registered.wait()
                        while True:
                            if inflight_tasks:
                                await asyncio.sleep(0.1)
                                continue
                            await outbound.put(pb.RelayFrame(
                                network_id=network_id,
                                sender_id=wid,
                                sender_role="worker",
                                routing_key="poll",
                                payload=pb.PollRequest(worker_id=wid, max_tasks=1).SerializeToString(),
                            ))
                            await asyncio.sleep(1)

                    register_timeout_task = asyncio.create_task(_registration_timeout())
                    heartbeat_task = asyncio.create_task(_heartbeat())
                    poll_task = asyncio.create_task(_poll())

                    async for frame in stub.Connect(_gen()):
                        outcome = await _handle_relay_frame(
                            frame,
                            outbound=outbound,
                            network_id=network_id,
                            worker_id=wid,
                            registered=registered,
                            inflight_tasks=inflight_tasks,
                        )
                        if outcome in {"register_error", "master_left"}:
                            fatal_registration_error = outcome == "register_error"
                            break
            except Exception as e:
                logger.warning(f"relay disconnected: {e}")
            finally:
                for task in (register_timeout_task, heartbeat_task, poll_task, *list(inflight_tasks)):
                    if task:
                        task.cancel()
                if register_timeout_task or heartbeat_task or poll_task or inflight_tasks:
                    await asyncio.gather(
                        *(
                            task for task in (
                                register_timeout_task,
                                heartbeat_task,
                                poll_task,
                                *list(inflight_tasks),
                            ) if task is not None
                        ),
                        return_exceptions=True,
                    )
            if fatal_registration_error:
                return
            await asyncio.sleep(5)


@click.command("worker")
@click.option("--master", default="", envvar="DISTRIPUTE_MASTER")
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--network-id", default="", envvar="DISTRIPUTE_NETWORK_ID", required=True)
@click.option("--log-level", default="INFO")
def main(master, relay, network_id, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not master and not relay:
        raise click.UsageError("specify --master or --relay")
    asyncio.run(run_worker(master_addr=master, relay_addr=relay, network_id=network_id))


if __name__ == "__main__":
    main()
