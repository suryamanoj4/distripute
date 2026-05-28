import asyncio
import base64
import json
import logging
import threading
import time
import uuid

import cloudpickle
from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger("distripute.client")

POLL_INTERVAL = 0.5


class _RemoteResult:
    def __init__(self, task_id: str):
        self._task_id = task_id
        self._value = None
        self._error = None
        self._done = False

    def get(self, timeout: float | None = None):
        import threading
        waited = 0.0
        while not self._done:
            if timeout and waited >= timeout:
                raise TimeoutError(f"task {self._task_id} timed out")
            threading.Event().wait(timeout=0.1)
            waited += 0.1
        if self._error:
            raise self._error
        return self._value

    def _resolve(self, value):
        self._value = value
        self._done = True

    def _reject(self, error):
        self._error = error
        self._done = True


class _Client:
    def __init__(self, master_addr: str, network_id: str):
        self.master_addr = master_addr
        self.network_id = network_id
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._bg_thread: threading.Thread | None = None
        self._start()

    def _start(self):
        self._loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._bg_thread.start()
        self._session = ClientSession(loop=self._loop)

    def submit(self, func_name: str, source: str, requirements: list[str],
               args: tuple, kwargs: dict) -> _RemoteResult:
        task_id = uuid.uuid4().hex[:8]
        payload = cloudpickle.dumps((args, kwargs))
        payload_b64 = base64.b64encode(payload).decode()

        result = _RemoteResult(task_id)

        async def _do():
            async with self._session.post(
                f"http://{self.master_addr}/task",
                json=dict(
                    network_id=self.network_id,
                    task_id=task_id,
                    func_name=func_name,
                    source=source,
                    requirements=requirements,
                    payload=payload_b64,
                ),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    result._reject(RuntimeError(f"task submission failed: {err}"))
                    return

            # poll for result
            while True:
                async with self._session.get(
                    f"http://{self.master_addr}/task/{task_id}",
                ) as resp:
                    data = await resp.json()
                    status = data.get("status")
                    if status == "done":
                        result._resolve(data.get("result"))
                        return
                    elif status == "failed":
                        result._reject(RuntimeError(data.get("error", "unknown error")))
                        return
                await asyncio.sleep(POLL_INTERVAL)

        asyncio.run_coroutine_threadsafe(_do(), self._loop)
        return result
