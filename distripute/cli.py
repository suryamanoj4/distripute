import asyncio
import json
import logging

import click
import grpc

from . import VERSION
from . import master as master_mod
from . import worker as worker_mod
from . import relay as relay_mod
from .grpc import pb, grpc as rpcmod


@click.group()
@click.version_option(version=VERSION)
def cli():
    """Distripute — distributed task execution mesh."""


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9090, type=int)
@click.option("--relay", default="", envvar="DISTRIPUTE_RELAY")
@click.option("--log-level", default="INFO")
def master(host, port, relay, log_level):
    """Start a master node."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(master_mod.serve(host=host, port=port, relay_addr=relay))


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
    asyncio.run(worker_mod.run_worker(
        master_addr=master, relay_addr=relay, network_id=network_id,
    ))


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9091, type=int)
@click.option("--log-level", default="INFO")
def relay(host, port, log_level):
    """Start a relay server for cross-internet connectivity."""
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(relay_mod.serve(host=host, port=port))


@cli.command()
@click.option("--master", "-m", default="localhost:9090", envvar="DISTRIPUTE_MASTER")
def info(master):
    """Show mesh network info."""
    async def _do():
        async with grpc.aio.insecure_channel(master) as channel:
            stub = rpcmod.MasterStub(channel)
            resp = await stub.GetInfo(pb.Empty())
            return dict(network_id=resp.network_id, version=resp.version,
                        workers=resp.workers, pending=resp.pending_tasks)
    result = asyncio.run(_do())
    click.echo(json.dumps(result, indent=2))


def main():
    cli()
