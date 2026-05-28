import asyncio
import logging
from collections import defaultdict

import click
import grpc

from .grpc import pb, grpc as rpcmod

logger = logging.getLogger("distripute.relay")


class RelayServicer(rpcmod.RelayServicer):
    def __init__(self):
        self._masters: dict[str, grpc.aio.StreamStreamCall] = {}
        self._workers: dict[str, list[tuple[str, grpc.aio.StreamStreamCall]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def Connect(self, request_iterator, context):
        first = await request_iterator.__anext__()
        network_id = first.network_id
        sender_id = first.sender_id
        role = first.sender_role

        if role == "master":
            async with self._lock:
                self._masters[network_id] = context
            logger.info(f"master registered: {network_id}")

            async def _from_master():
                async for frame in request_iterator:
                    yield frame

            # forward worker messages to master
            return self._forward_workers_to_master(_from_master(), network_id, context)

        elif role == "worker":
            async with self._lock:
                self._workers[network_id].append((sender_id, context))
            logger.info(f"worker joined: {sender_id} -> network_id={network_id}")

            # notify master
            async with self._lock:
                master_ctx = self._masters.get(network_id)
            if master_ctx:
                await master_ctx.write(pb.RelayFrame(
                    network_id=network_id, sender_id=sender_id,
                    sender_role="worker", routing_key="worker_joined",
                ))

            return request_iterator

        return request_iterator

    async def _forward_workers_to_master(self, master_iter, network_id, master_ctx):
        try:
            async for frame in master_iter:
                pass  # handle master->worker messages
        finally:
            async with self._lock:
                self._masters.pop(network_id, None)
                workers = self._workers.pop(network_id, [])
                for wid, wctx in workers:
                    await wctx.write(pb.RelayFrame(
                        network_id=network_id, sender_id=wid,
                        sender_role="worker", routing_key="worker_left",
                    ))


async def serve(host="0.0.0.0", port=9091):
    logger.info(f"relay starting on {host}:{port}")
    server = grpc.aio.server()
    servicer = RelayServicer()
    rpcmod.add_RelayServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    await server.wait_for_termination()


@click.command("relay")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9091, type=int)
@click.option("--log-level", default="INFO")
def main(host, port, log_level):
    logging.basicConfig(level=getattr(logging, log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    asyncio.run(serve(host=host, port=port))
