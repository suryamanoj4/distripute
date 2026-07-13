import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field

import click
import grpc

from .grpc import pb, grpc as rpcmod

logger = logging.getLogger("distripute.relay")


@dataclass
class _Peer:
    network_id: str
    peer_id: str
    role: str
    outbound: asyncio.Queue[pb.RelayFrame | None] = field(default_factory=asyncio.Queue)


class RelayServicer(rpcmod.RelayServicer):
    def __init__(self):
        self._masters: dict[str, _Peer] = {}
        self._workers: dict[str, dict[str, _Peer]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def Connect(self, request_iterator, context):
        first = await request_iterator.__anext__()
        peer = _Peer(
            network_id=first.network_id,
            peer_id=first.sender_id,
            role=first.sender_role,
        )

        await self._register_peer(peer)
        consumer = asyncio.create_task(self._consume(peer, request_iterator))

        try:
            async for frame in self._yield_frames(peer):
                yield frame
        finally:
            consumer.cancel()
            results = await asyncio.gather(consumer, return_exceptions=True)
            for result in results:
                if result is None or isinstance(result, asyncio.CancelledError):
                    continue
                logger.warning(
                    "relay consumer failed for %s %s on %s: %s",
                    peer.role,
                    peer.peer_id,
                    peer.network_id,
                    result,
                )
            await self._disconnect_peer(peer)

    async def _yield_frames(self, peer: _Peer):
        while True:
            frame = await peer.outbound.get()
            if frame is None:
                return
            yield frame

    async def _register_peer(self, peer: _Peer):
        async with self._lock:
            if peer.role == "master":
                self._masters[peer.network_id] = peer
            elif peer.role == "worker":
                self._workers[peer.network_id][peer.peer_id] = peer
            else:
                raise ValueError(f"unsupported relay role: {peer.role}")

        if peer.role == "master":
            logger.info(f"master registered: {peer.network_id}")
        else:
            logger.info(f"worker joined: {peer.peer_id} -> network_id={peer.network_id}")

    async def _consume(self, peer: _Peer, request_iterator):
        async for frame in request_iterator:
            await self._route_frame(peer, frame)

    async def _route_frame(self, peer: _Peer, frame: pb.RelayFrame):
        if peer.role == "worker":
            await self._send_to_master(peer.network_id, frame)
            return

        if frame.target_id:
            worker = await self._get_worker(peer.network_id, frame.target_id)
            if worker:
                await worker.outbound.put(frame)
            return

        for worker in await self._list_workers(peer.network_id):
            await worker.outbound.put(frame)

    async def _disconnect_peer(self, peer: _Peer):
        async with self._lock:
            if peer.role == "master":
                current = self._masters.get(peer.network_id)
                if current is peer:
                    self._masters.pop(peer.network_id, None)
                workers = list(self._workers.pop(peer.network_id, {}).values())
            else:
                workers = []
                self._workers[peer.network_id].pop(peer.peer_id, None)
                if not self._workers[peer.network_id]:
                    self._workers.pop(peer.network_id, None)

        if peer.role == "master":
            for worker in workers:
                await worker.outbound.put(pb.RelayFrame(
                    network_id=peer.network_id,
                    sender_id="master",
                    sender_role="master",
                    routing_key="master_left",
                ))
                await worker.outbound.put(None)
            logger.info(f"master disconnected: {peer.network_id}")
            return

        await self._send_to_master(peer.network_id, pb.RelayFrame(
            network_id=peer.network_id,
            sender_id=peer.peer_id,
            sender_role="worker",
            routing_key="worker_left",
        ))
        logger.info(f"worker left: {peer.peer_id} -> network_id={peer.network_id}")

    async def _send_to_master(self, network_id: str, frame: pb.RelayFrame):
        async with self._lock:
            master = self._masters.get(network_id)
        if master:
            await master.outbound.put(frame)

    async def _get_worker(self, network_id: str, worker_id: str) -> _Peer | None:
        async with self._lock:
            return self._workers.get(network_id, {}).get(worker_id)

    async def _list_workers(self, network_id: str) -> list[_Peer]:
        async with self._lock:
            return list(self._workers.get(network_id, {}).values())


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
