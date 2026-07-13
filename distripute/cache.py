"""
Cache abstraction backed by Redis or in-memory dicts (fallback).

The master uses this to track:
- Task queue (pending list)
- Task state (status, result, worker, etc.)
- Worker state (capabilities, heartbeat, load)
- Job state (batch progress)
"""
import json
import logging
import time

logger = logging.getLogger("distripute.cache")

try:
    import redis as _redis
except ImportError:
    _redis = None
    logger.warning("redis package not installed — using in-memory fallback")


def _now():
    return time.time()


class Cache:
    def __init__(self, redis_url: str = ""):
        self._use_redis = False
        self._r = None
        self._local_tasks: dict[str, dict] = {}
        self._local_pending: list[str] = []
        self._local_workers: dict[str, dict] = {}
        self._local_jobs: dict[str, dict] = {}

        if redis_url and _redis:
            try:
                self._r = _redis.from_url(redis_url, decode_responses=True)
                self._r.ping()
                self._use_redis = True
            except Exception:
                pass

    # ── Pending Queue ────────────────────────────────────

    def pending_push(self, task_id: str):
        if self._use_redis:
            self._r.lpush("pending", task_id)
        else:
            self._local_pending.append(task_id)

    def pending_pop(self, count: int = 1) -> list[str]:
        if self._use_redis:
            ids = []
            for _ in range(count):
                tid = self._r.rpop("pending")
                if tid is None:
                    break
                ids.append(tid)
            return ids
        else:
            ids = self._local_pending[:count]
            self._local_pending = self._local_pending[count:]
            return ids

    def pending_remove(self, task_id: str):
        if self._use_redis:
            self._r.lrem("pending", 0, task_id)
        else:
            try:
                self._local_pending.remove(task_id)
            except ValueError:
                pass

    def pending_count(self) -> int:
        if self._use_redis:
            return self._r.llen("pending") or 0
        return len(self._local_pending)

    def pending_peek(self, count: int) -> list[str]:
        if self._use_redis:
            return self._r.lrange("pending", 0, count - 1) or []
        return self._local_pending[:count]

    # ── Tasks ─────────────────────────────────────────────

    def task_set(self, task_id: str, data: dict):
        if self._use_redis:
            self._r.hset(f"task:{task_id}", mapping={k: _val(v) for k, v in data.items()})
            self._r.sadd("tasks", task_id)
        else:
            self._local_tasks[task_id] = dict(data)

    def task_get(self, task_id: str) -> dict | None:
        if self._use_redis:
            raw = self._r.hgetall(f"task:{task_id}")
            if not raw:
                return None
            return {k: _unval(v) for k, v in raw.items()}
        return self._local_tasks.get(task_id)

    def task_update(self, task_id: str, **fields):
        if self._use_redis:
            if fields:
                self._r.hset(f"task:{task_id}", mapping={k: _val(v) for k, v in fields.items()})
        else:
            t = self._local_tasks.get(task_id)
            if t:
                t.update(fields)

    def task_exists(self, task_id: str) -> bool:
        if self._use_redis:
            return bool(self._r.exists(f"task:{task_id}"))
        return task_id in self._local_tasks

    def task_delete(self, task_id: str):
        if self._use_redis:
            self._r.delete(f"task:{task_id}")
            self._r.srem("tasks", task_id)
        else:
            self._local_tasks.pop(task_id, None)

    def task_list(self) -> list[dict]:
        if self._use_redis:
            ids = self._r.smembers("tasks") or set()
            tasks = []
            for tid in ids:
                task = self.task_get(tid)
                if task:
                    tasks.append(task)
                else:
                    self._r.srem("tasks", tid)
            return tasks
        return list(self._local_tasks.values())

    # ── Workers ───────────────────────────────────────────

    def worker_set(self, worker_id: str, data: dict):
        data["_ts"] = _now()
        if self._use_redis:
            self._r.hset(f"worker:{worker_id}", mapping={k: _val(v) for k, v in data.items()})
            self._r.sadd("workers", worker_id)
        else:
            self._local_workers[worker_id] = dict(data)

    def worker_get(self, worker_id: str) -> dict | None:
        if self._use_redis:
            raw = self._r.hgetall(f"worker:{worker_id}")
            if not raw:
                return None
            return {k: _unval(v) for k, v in raw.items()}
        return self._local_workers.get(worker_id)

    def worker_update(self, worker_id: str, **fields):
        if self._use_redis:
            if fields:
                fields["_ts"] = _now()
                self._r.hset(f"worker:{worker_id}", mapping={k: _val(v) for k, v in fields.items()})
        else:
            w = self._local_workers.get(worker_id)
            if w:
                fields["_ts"] = _now()
                w.update(fields)

    def worker_delete(self, worker_id: str):
        if self._use_redis:
            self._r.delete(f"worker:{worker_id}")
            self._r.srem("workers", worker_id)
        else:
            self._local_workers.pop(worker_id, None)

    def worker_list(self) -> list[dict]:
        if self._use_redis:
            ids = self._r.smembers("workers") or set()
            workers = []
            for wid in ids:
                w = self.worker_get(wid)
                if w:
                    workers.append(w)
            return workers
        return list(self._local_workers.values())

    def worker_count(self) -> int:
        if self._use_redis:
            return self._r.scard("workers") or 0
        return len(self._local_workers)

    # ── Jobs ──────────────────────────────────────────────

    def job_set(self, job_id: str, data: dict):
        if self._use_redis:
            self._r.hset(f"job:{job_id}", mapping={k: _val(v) for k, v in data.items()})
            self._r.sadd("jobs", job_id)
        else:
            self._local_jobs[job_id] = dict(data)

    def job_get(self, job_id: str) -> dict | None:
        if self._use_redis:
            raw = self._r.hgetall(f"job:{job_id}")
            if not raw:
                return None
            return {k: _unval(v) for k, v in raw.items()}
        return self._local_jobs.get(job_id)

    def job_update(self, job_id: str, **fields):
        if self._use_redis:
            if fields:
                self._r.hset(f"job:{job_id}", mapping={k: _val(v) for k, v in fields.items()})
        else:
            j = self._local_jobs.get(job_id)
            if j:
                j.update(fields)

    def job_list(self) -> list[dict]:
        if self._use_redis:
            ids = self._r.smembers("jobs") or set()
            jobs = []
            for jid in ids:
                j = self.job_get(jid)
                if j:
                    jobs.append(j)
            return jobs
        return list(self._local_jobs.values())


def _val(v):
    if isinstance(v, float):
        return f"__f__{v}"
    if isinstance(v, int):
        return f"__i__{v}"
    if isinstance(v, list):
        return f"__l__{json.dumps(v)}"
    if isinstance(v, dict):
        return f"__d__{json.dumps(v)}"
    return str(v) if v is not None else ""


def _unval(s: str):
    if not s:
        return ""
    if s.startswith("__f__"):
        return float(s[5:])
    if s.startswith("__i__"):
        return int(s[5:])
    if s.startswith("__l__"):
        return json.loads(s[5:])
    if s.startswith("__d__"):
        return json.loads(s[5:])
    return s
