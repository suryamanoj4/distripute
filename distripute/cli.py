import asyncio
import json
import logging
import sys

import click
from aiohttp import ClientSession, WSMsgType

from . import VERSION
from . import master as master_mod
from . import worker as worker_mod
from . import relay as relay_mod


@click.group()
@click.version_option(version=VERSION)
def cli():
    """Distripute — distributed task execution mesh."""


# ── master ─────────────────────────────────────────────────

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


# ── worker ─────────────────────────────────────────────────

@cli.command()
@click.option("--master", default="", envvar="DISTRIPUTE_MASTER")
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--network-id", default="", envvar="DISTRIPUTE_NETWORK_ID", required=True)
@click.option("--log-level", default="INFO")
def worker(master, relay, network_id, log_level):
    """Start a worker node."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not master and not relay:
        raise click.UsageError("specify --master or --relay")
    node = worker_mod.WorkerNode(
        master_addr=master, relay_addr=relay, network_id=network_id,
    )
    logger = logging.getLogger("distripute.cli")
    logger.info(f"worker starting (hw={worker_mod.HARDWARE}, gpus={worker_mod.GPU_COUNT})")
    asyncio.run(node.run_forever())


# ── relay ──────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9091, type=int)
@click.option("--log-level", default="INFO")
def relay(host, port, log_level):
    """Start a relay server for cross-internet connectivity."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    srv = relay_mod.RelayServer(host=host, port=port)
    asyncio.run(srv.run_forever())


# ── info ───────────────────────────────────────────────────

@cli.command()
@click.option("--master", "-m", default="localhost:9090", envvar="DISTRIPUTE_MASTER")
def info(master):
    """Show mesh network info."""
    async def _do():
        async with ClientSession() as session:
            async with session.get(f"http://{master}/info") as resp:
                return await resp.json()
    result = asyncio.run(_do())
    click.echo(json.dumps(result, indent=2))


# ── entry ──────────────────────────────────────────────────

def main():
    cli()
