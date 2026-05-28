import asyncio
import logging
import threading
import time
import uuid

import grpc
import cloudpickle

from .grpc import pb, grpc as rpcmod

logger = logging.getLogger("distripute.client")


class _RemoteResult:
    def __init__(self, task_id: str):
        self._task_id = task_id
        self._value = None
        self._error = None
        self._done = False

    def get(self, timeout: float | None = None):
        waited = 0.0
        while not self._done:
            if timeout and waited >= timeout:
                raise TimeoutError(f"task {self._task_id} timed out")
            time.sleep(0.1)
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
        self._channel = grpc.aio.insecure_channel(master_addr)
        self._stub = rpcmod.MasterStub(self._channel)

    def submit(self, func_name: str, source: str, requirements: list[str],
               args: tuple, kwargs: dict) -> _RemoteResult:
        task_id = uuid.uuid4().hex[:8]
        payload = cloudpickle.dumps((args, kwargs))
        result = _RemoteResult(task_id)

        async def _do():
            try:
                resp = await self._stub.SubmitTask(pb.TaskSubmit(
                    network_id=self.network_id, task_id=task_id,
                    func_name=func_name, source=source,
                    requirements=requirements, payload=payload,
                ))
                while True:
                    r = await self._stub.GetTaskResult(pb.TaskResultRequest(task_id=task_id))
                    if r.status == "done":
                        result._resolve(r.result)
                        return
                    elif r.status == "failed":
                        result._reject(RuntimeError(r.error or "task failed"))
                        return
                    await asyncio.sleep(0.5)
            except Exception as e:
                result._reject(e)

        asyncio.create_task(_do())
        return result

    def submit_batch(self, func_name: str, source: str, requirements: list[str],
                     args_list: list[tuple]) -> str:
        """Create a batch job. Returns job_id."""
        job_id = uuid.uuid4().hex[:8]
        payloads = [cloudpickle.dumps(args) for args in args_list]

        async def _do():
            resp = await self._stub.CreateBatch(pb.BatchCreateRequest(
                network_id=self.network_id, func_name=func_name, source=source,
                requirements=requirements, arg_payloads=payloads, job_id=job_id,
            ))
            return resp.job_id

        return asyncio.create_task(_do())

    def get_batch_status(self, job_id: str):
        async def _do():
            resp = await self._stub.GetBatchStatus(pb.BatchStatusRequest(job_id=job_id))
            return dict(job_id=resp.job_id, status=resp.status,
                        total=resp.total, done=resp.done, failed=resp.failed)
        return asyncio.create_task(_do())
