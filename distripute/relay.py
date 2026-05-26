import asyncio
import json
import logging
from collections import defaultdict

import click
from aiohttp import web, WSMsgType

logger = logging.getLogger("distripute.relay")


class RelayServer:
    def __init__(self, host="0.0.0.0", port=9091):
        self.host = host
        self.port = port
        self._masters: dict[str, web.WebSocketResponse] = {}
        self._workers: dict[str, list[tuple[str, web.WebSocketResponse]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def _handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        role = None
        network_id = None
        node_id = "unknown"

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                cmd = data.get("type")

                if cmd == "master_hello":
                    network_id = data["network_id"]
                    async with self._lock:
                        old = self._masters.get(network_id)
                        if old and not old.closed:
                            await old.close()
                        self._masters[network_id] = ws
                    role = "master"
                    logger.info(f"master registered: network_id={network_id}")
                    await ws.send_json({"type": "master_ack"})

                elif cmd == "worker_hello":
                    network_id = data["network_id"]
                    node_id = data.get("worker_id", "unknown")
                    async with self._lock:
                        self._workers[network_id].append((node_id, ws))
                    role = "worker"
                    master_ws = self._masters.get(network_id)
                    if master_ws and not master_ws.closed:
                        await master_ws.send_json({
                            "type": "worker_joined", "worker_id": node_id,
                        })
                    await ws.send_json({"type": "worker_ack", "worker_id": node_id})
                    logger.info(f"worker joined: {node_id} -> network_id={network_id}")

                elif cmd == "forward":
                    target_net = data.get("network_id", network_id)
                    payload = data.get("payload", {})
                    async with self._lock:
                        dest = self._masters.get(target_net) if role == "worker" else None
                        if role == "master" and data.get("target_worker"):
                            workers = self._workers.get(target_net, [])
                            for wid, wws in workers:
                                if wid == data["target_worker"]:
                                    dest = wws
                                    break
                    if dest and not dest.closed:
                        await dest.send_json(payload)
                    else:
                        await ws.send_json({"type": "error", "message": "target not connected"})

                elif cmd == "ping":
                    await ws.send_json({"type": "pong"})

        except Exception:
            pass
        finally:
            async with self._lock:
                if role == "master" and network_id:
                    self._masters.pop(network_id, None)
                    logger.info(f"master disconnected: network_id={network_id}")
                elif role == "worker" and network_id:
                    self._workers[network_id] = [
                        (n, w) for n, w in self._workers[network_id] if w is not ws
                    ]
                    master_ws = self._masters.get(network_id)
                    if master_ws and not master_ws.closed:
                        await master_ws.send_json({
                            "type": "worker_left", "worker_id": node_id,
                        })
                    logger.info(f"worker disconnected: {node_id}")
        return ws

    def build_app(self):
        app = web.Application()
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/health", lambda _r: web.json_response({"ok": True}))
        return app

    async def run_forever(self):
        runner = web.AppRunner(self.build_app())
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"relay listening on ws://{self.host}:{self.port}")
        while True:
            await asyncio.sleep(3600)


@click.command("relay")
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=9091, type=int)
@click.option("--log-level", default="INFO")
def main(host, port, log_level):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    srv = RelayServer(host=host, port=port)
    asyncio.run(srv.run_forever())
