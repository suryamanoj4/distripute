import asyncio
import json
import sys

import click
from aiohttp import ClientSession, WSMsgType

from . import VERSION
from . import master as master_mod
from . import worker as worker_mod
from . import relay as _relay_mod
from .registry import _default_registry as reg


def master_addr(ctx):
    """Return the master address from the --master / -m option."""
    return ctx.parent.params.get("master") or "localhost:9090"


def relay_addr(ctx):
    """Return the relay address from --relay option."""
    return ctx.parent.params.get("relay") or ""


def use_relay(ctx):
    return bool(relay_addr(ctx))


@click.group()
@click.option("--master", "-m", default="", envvar="DISTRIPUTE_MASTER", help="Master address")
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY", help="Relay address")
@click.option("--network-id", default="", envvar="DISTRIPUTE_NETWORK_ID")
@click.version_option(version=VERSION)
@click.pass_context
def cli(ctx, master, relay, network_id):
    ctx.ensure_object(dict)
    ctx.obj["master"] = master
    ctx.obj["relay"] = relay
    ctx.obj["network_id"] = network_id


# ── job ──────────────────────────────────────────────────

@cli.group()
def job():
    """Manage inference jobs."""


@job.command("create")
@click.option("--type", "task_type", type=click.Choice(["asr", "ocr", "llm"]), required=True)
@click.option("--model", default="whisper-large-v3")
@click.option("--strategy", default="data_parallel", type=click.Choice(["data_parallel", "model_parallel", "pipeline_parallel"]))
@click.option("--input", "input_source", required=True)
@click.option("--output", "output_sink", required=True)
@click.option("--param", "-p", multiple=True, help="Model params: key=value")
@click.pass_context
def job_create(ctx, task_type, model, strategy, input_source, output_sink, param):
    """Submit a new inference job."""
    params = {}
    for p in param:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k] = v

    body = dict(
        task_type=task_type, model=model, strategy=strategy,
        input_source=input_source, output_sink=output_sink,
        model_params=params,
    )

    async def _do():
        if use_relay(ctx):
            async with ClientSession() as session:
                async with session.ws_connect(f"ws://{relay_addr(ctx)}/ws") as ws:
                    await ws.send_json(dict(
                        type="worker_hello", network_id=ctx.obj["network_id"],
                        worker_id="cli", version=VERSION,
                    ))
                    await ws.__anext__()
                    await ws.send_json(dict(
                        type="forward", network_id=ctx.obj["network_id"],
                        payload=dict(action="create_job", body=body),
                    ))
                    # best-effort — user should check via job list
        else:
            async with ClientSession() as session:
                async with session.post(f"http://{master_addr(ctx)}/job", json=body) as resp:
                    return await resp.json()

    result = asyncio.run(_do())
    if result:
        click.echo(json.dumps(result, indent=2))


@job.command("list")
@click.pass_context
def job_list(ctx):
    """List all jobs."""
    async def _do():
        async with ClientSession() as session:
            async with session.get(f"http://{master_addr(ctx)}/jobs") as resp:
                return await resp.json()
    result = asyncio.run(_do())
    for j in result.get("jobs", []):
        click.echo(f"  {j['id']:12s} {j['status']:10s} {j['model']:20s} "
                    f"{j['done']}/{j['total']}  {j.get('task_type', '')}")


@job.command("status")
@click.argument("job_id")
@click.pass_context
def job_status(ctx, job_id):
    """Get job status."""
    async def _do():
        async with ClientSession() as session:
            async with session.get(f"http://{master_addr(ctx)}/job/{job_id}") as resp:
                return await resp.json()
    result = asyncio.run(_do())
    click.echo(json.dumps(result, indent=2))


# ── workers ────────────────────────────────────────────────

@cli.group("workers")
def workers_cmd():
    """Manage workers."""


@workers_cmd.command("list")
@click.pass_context
def worker_list(ctx):
    """List registered workers."""
    async def _do():
        async with ClientSession() as session:
            async with session.get(f"http://{master_addr(ctx)}/workers") as resp:
                return await resp.json()
    result = asyncio.run(_do())
    for w in result.get("workers", []):
        click.echo(f"  {w['id']:12s} {w['hardware']:5s} "
                    f"gpus={w['gpu_count']} cores={w['cpu_cores']} "
                    f"load={w['load']:.1f} tasks={w['active_tasks']}")


# ── info ──────────────────────────────────────────────────

@cli.command()
@click.pass_context
def info(ctx):
    """Show network info."""
    async def _do():
        async with ClientSession() as session:
            async with session.get(f"http://{master_addr(ctx)}/info") as resp:
                return await resp.json()
    result = asyncio.run(_do())
    click.echo(json.dumps(result, indent=2))


# ── model ─────────────────────────────────────────────────

@cli.group()
def model():
    """Manage model registry."""


@model.command("list")
@click.pass_context
def model_list(ctx):
    """List known models."""
    async def _do():
        async with ClientSession() as session:
            async with session.get(f"http://{master_addr(ctx)}/models") as resp:
                return await resp.json()
    result = asyncio.run(_do())
    for name in result.get("models", {}):
        click.echo(f"  {name}")


@model.command("add")
@click.argument("name")
@click.option("--family", required=True)
@click.option("--layers", type=int, required=True)
@click.option("--param-count", type=int, default=0)
@click.option("--gpu-mem", type=int, default=1, help="Min GPU memory in GB")
@click.option("--ram", type=int, default=2, help="Min RAM in GB")
@click.pass_context
def model_add(ctx, name, family, layers, param_count, gpu_mem, ram):
    """Register a custom model."""
    info = dict(
        family=family, num_layers=layers, param_count=param_count,
        min_gpu_mem=gpu_mem * 1_000_000_000,
        min_ram=ram * 1_000_000_000,
    )
    reg.register(name, info)
    click.echo(f"model registered: {name}")


# ── relay ─────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9091, type=int)
@click.option("--log-level", default="INFO")
def relay(host, port, log_level):
    """Start a relay server."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    srv = _relay_mod.RelayServer(host=host, port=port)
    asyncio.run(srv.run_forever())


# ── master ────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9090, type=int)
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--log-level", default="INFO")
def master(host, port, relay, log_level):
    """Start a master node."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    node = master_mod.MasterNode(host=host, port=port, relay_addr=relay)
    asyncio.run(node.run_forever())


# ── entry ─────────────────────────────────────────────────

@cli.command()
@click.option("--master", default="", envvar="DISTRIPUTE_MASTER")
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--network-id", default="", envvar="DISTRIPUTE_NETWORK_ID", required=True)
@click.option("--models", default="")
@click.option("--plugin", default="")
@click.option("--log-level", default="INFO")
def worker(master, relay, network_id, models, plugin, log_level):
    """Start a worker node."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not master and not relay:
        raise click.UsageError("specify --master or --relay")
    supported = [m.strip() for m in models.split(",") if m.strip()]
    node = worker_mod.WorkerNode(
        master_addr=master, relay_addr=relay, network_id=network_id,
        supported_models=supported, plugin_cmd=plugin or None,
    )
    logger = logging.getLogger("distripute.cli")
    logger.info(f"worker starting (hw={worker_mod.HARDWARE}, gpus={worker_mod.GPU_COUNT})")
    asyncio.run(node.run_forever())


def main():
    cli()
