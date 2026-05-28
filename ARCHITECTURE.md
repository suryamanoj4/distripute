# Distripute Architecture

## Overview

Distripute is a distributed task execution mesh. Three node types form the network:

```
                         ┌──────────────┐
                         │    Relay     │  ← gRPC bidirectional stream hub
                         │  public VPS  │     routes by network_id
                         └──────┬───────┘
                          ╱              ╲
                 ┌────────▼──┐     ┌─────▼────────┐
                 │   Master  │     │   Workers     │
                 │  gRPC srv │     │  gRPC client  │
                 │ scheduler │     │ uv run --with │
                 │ file host │     │ exec(source)  │
                 └────────────┘     └──────────────┘
```

Nodes communicate via **gRPC** (protobuf). Two connectivity modes:

| Mode | Connection | Use case |
|---|---|---|
| **Direct** | Worker → Master gRPC (unary/streaming) | Same LAN, or master has public IP |
| **Relay** | All nodes → Relay bidirectional stream | Machines across the internet (NAT) |

---

## 1. Protocol Definition

All services are defined in [`proto/distripute.proto`](../proto/distripute.proto).

### Master Service (12 RPCs)

```protobuf
service Master {
  // Worker lifecycle
  rpc Register(RegisterRequest) returns (RegisterResponse);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);

  // Task scheduling
  rpc PollTasks(PollRequest) returns (PollResponse);
  rpc SubmitResult(TaskResult) returns (Ack);

  // Client task submission
  rpc SubmitTask(TaskSubmit) returns (TaskSubmitResponse);
  rpc GetTaskResult(TaskResultRequest) returns (TaskResultResponse);

  // Batch jobs
  rpc CreateBatch(BatchCreateRequest) returns (BatchCreateResponse);
  rpc GetBatchStatus(BatchStatusRequest) returns (BatchStatusResponse);

  // File transfer (streaming)
  rpc GetFile(FileRequest) returns (stream FileChunk);
  rpc UploadFile(stream FileChunk) returns (FileUploadResponse);

  // Model registry
  rpc RegisterModel(ModelRegisterRequest) returns (Ack);
  rpc ListModels(Empty) returns (ModelListResponse);

  // Info
  rpc GetInfo(Empty) returns (InfoResponse);
}
```

### Relay Service (1 RPC)

```protobuf
service Relay {
  rpc Connect(stream RelayFrame) returns (stream RelayFrame);
}
```

Both master and workers call `Connect()`. The relay pairs them by `network_id` and forward frames between the two streams. Frames carry routing metadata:

```protobuf
message RelayFrame {
  string network_id = 1;
  string sender_id = 2;
  string sender_role = 3;  // "master" or "worker"
  string target_id = 4;
  string routing_key = 5;  // "register", "heartbeat", "poll", "result", "task"
  bytes payload = 6;
}
```

### Key Messages

```protobuf
message TaskDef {
  string id = 1;
  string func_name = 2;
  string source = 3;          // entire source file content
  repeated string requirements = 4; // pip deps for uv run --with
  bytes payload = 5;          // cloudpickled (args, kwargs)
  string job_id = 6;
  string filename = 7;        // data file for batch jobs
  int64 file_size = 8;
}

message FileChunk {
  string job_id = 1;
  string filename = 2;
  bytes data = 3;
  int64 offset = 4;
  int64 total_size = 5;
}
```

---

## 2. Task Lifecycle

```
User Script                     Master                    Worker
    │                              │                        │
    │ distripute.init(addr, nid)   │                        │
    │─────────────────────────────►│                        │
    │                              │                        │
    │ @distripute.task             │                        │
    │ def transcribe(path):        │                        │
    │   ...                        │                        │
    │                              │                        │
    │ result = transcribe(f).get() │                        │
    │                              │                        │
    │──── SubmitTask ─────────────►│                        │
    │  {func_name, source,         │                        │
    │   requirements, payload}     │                        │
    │                              │                        │
    │                              │──── PollTasks ─────────►│
    │                              │◄─── {TaskDef} ─────────│
    │                              │                        │
    │                              │                        │── uv run --with deps
    │                              │                        │   python3 runner.py
    │                              │                        │
    │                              │◄─── SubmitResult ─────│
    │                              │  {success, output}     │
    │                              │                        │
    │◄─── GetTaskResult ──────────│                        │
    │  {status: "done", result}   │                        │
    │                              │                        │
```

### Step-by-step

1. **init()**: Creates a gRPC channel to the master, stores it in a module-level global.

2. **@distripute.task**: `inspect.getfile(func)` reads the source file path. `RemoteFunction` wraps the function, storing its name and source file.

3. **Call**: `transcribe("/data/audio.wav")` calls `RemoteFunction.remote()`:
   - Reads `source_file` content via `Path.read_text()`
   - Cloudpickles `(args, kwargs)` into `payload`
   - Calls `SubmitTask(func_name, source, requirements, payload)` via gRPC
   - Returns `_RemoteResult` (with `.get()` blocking)

4. **Master.SubmitTask**: Validates `network_id`, creates a task entry in `self.tasks` dict, appends to `_pending` list, returns `{task_id, status: "pending"}`.

5. **Worker.PollTasks**: Worker sends `PollRequest(worker_id, max_tasks)`. Master calls `_schedule()`:
   - Filters `_pending` tasks
   - Finds eligible workers (heartbeat < 30s old)
   - Assigns tasks to worker, returns `TaskDef` list

6. **Worker execution**: For each received `TaskDef`:
   - If `filename` is set and `file_size > 0`: worker calls `GetFile(job_id, filename)` to stream the data file from master, saves to `~/.cache/distripute/tasks/{job_id}/{filename}`
   - Writes `source.py` and `runner.py` to a temp directory
   - Runs `uv run --with dep1 --with dep2 python3 runner.py`
   - Captures stdout → parses JSON result
   - Calls `SubmitResult(task_id, success, output, duration)`

7. **Client result polling**: After calling `SubmitTask`, the client polls `GetTaskResult(task_id)` every 500ms until `status == "done"`. Blocks the `.get()` call.

### Retry / Fault Tolerance

- Workers heartbeat every 10s
- Master drops workers after 30s of no heartbeat
- Tasks assigned to dead workers stay in `_pending` (TODO: re-queue on disconnect)
- No checkpoint/restart (in-memory state)

---

## 3. Data Parallelism (`@distripute.task`)

### Single task

Every call to a `@distripute.task` function creates one task in the master queue. N calls produce N tasks, distributed across workers by the scheduler.

```python
@distripute.task
def transcribe(path):
    import whisper
    return whisper.load_model("base").transcribe(path)["text"]

files = ["a.wav", "b.wav", "c.wav", "d.wav"]
results = [transcribe(f).get() for f in files]
```

The scheduler assigns tasks round-robin by worker load. Each worker has the full source code — only the arguments differ.

### Batch jobs

For large datasets, use the batch API to submit N tasks at once with shared source:

```python
client = distripute._global_client
job_id = client.submit_batch(
    func_name="transcribe",
    source=Path("my_script.py").read_text(),
    requirements=["openai-whisper"],
    args_list=[("/data/a.wav",), ("/data/b.wav",), ...],
)
status = client.get_batch_status(job_id)
```

Batch jobs also support file upload:
1. Client uploads files via `UploadFile(stream FileChunk)`
2. Files stored at `~/.cache/distripute/data/{job_id}/{filename}`
3. Each task includes `filename` + `file_size`
4. Worker downloads via `GetFile` before execution

### Data distribution matrix

| Input type | How worker gets data |
|---|---|
| File path on client | **Upload** to master via gRPC stream, then **download** from master |
| URL (S3, HTTP) | Passed as string arg; worker fetches directly |
| Small data (< 1MB) | Cloudpickled into `payload` directly |
| Shared filesystem | Both client and worker mount same path |

---

## 4. Network Modes

### Direct Mode

```
Worker ──gRPC──► Master
```

Worker knows master's IP:port. All RPCs go directly. Used when:
- Master has a public IP or port forwarding
- Workers are on the same LAN

### Relay Mode

```
Worker ──gRPC──► Relay ◄──gRPC── Master
```

Both master and workers connect **outbound** to the relay via gRPC bidirectional streaming (`Relay.Connect`). The relay:

1. Reads the first `RelayFrame` from each connection to determine role and network_id
2. Stores a reference to the master's stream keyed by network_id
3. Stores worker streams in a list per network_id
4. Forwards frames between matched streams
5. On disconnect, notifies the counterpart via `worker_left` / `master_left` frames

No port forwarding needed on any node. The relay only needs a public IP (cheap VPS).

```
relay --host 0.0.0.0 --port 9091              # public VPS
master --relay relay.io:9091                   # behind NAT
worker --relay relay.io:9091 --network-id xyz  # anywhere
```

---

## 5. Task Cache (Redis)

The master uses a `Cache` abstraction backed by **Redis** or in-memory dicts (fallback). All task queue, worker state, and job state go through this cache.

### Data structures in Redis

| Redis key | Type | Purpose |
|---|---|---|
| `pending` | **List** | Task ID queue. `LPUSH` by client, `RPOP` by scheduler. |
| `task:{task_id}` | **Hash** | Task state: status, func_name, source, payload, worker_id, result, error, job_id, filename |
| `worker:{worker_id}` | **Hash** | Worker state: cpu_cores, gpu_count, hardware, load, active_tasks, last_seen |
| `workers` | **Set** | All registered worker IDs (for listing) |
| `job:{job_id}` | **Hash** | Batch job state: status, total, done, failed |
| `jobs` | **Set** | All job IDs (for listing) |

### Why not persistence

Redis is configured as a pure cache with `maxmemory` + `allkeys-lru` eviction. Task results are consumed and discarded. If Redis restarts, the master falls back to in-memory mode (dev) or re-queues tasks from the gRPC client side.

### Starting with Redis

```bash
# Start Redis locally
redis-server

# Start master with Redis
distripute master --redis redis://localhost:6379

# Without --redis flag, falls back to in-memory dicts
```

### Cache fallback

When no `--redis` URL is provided, `Cache` uses Python dicts/lists. Same API, same behavior — no Redis required for development or single-node testing. All 28 tests run without Redis.

## 6. Worker Execution Environment

### `uv run` dependency management

Workers only need `uv` installed (single binary, no runtime deps). When executing a task:

1. Create temp dir `~/.cache/distripute/tasks/{uuid}/`
2. Write `source.py` (user's entire source file)
3. Write `runner.py`:
   ```python
   import sys, json, cloudpickle
   sys.path.insert(0, '{work_dir}')
   import source
   func = getattr(source, '{func_name}')
   with open('{args_path}', 'rb') as f:
       args, kwargs = cloudpickle.loads(f.read())
   # If file_arg exists, prepend to args
   result = func(*args, **kwargs)
   print(json.dumps({"result": result}))
   ```
4. Run: `uv run --with dep1 --with dep2 python3 runner.py`
5. `uv run` auto-creates a temporary venv, installs deps from uv's global cache, runs the script
6. Captures stdout → JSON.parse → submit result

Subsequent runs with the same deps are instant (cache hit).

### Hardware detection

On startup, the worker auto-detects:
- **CPU cores**: `os.cpu_count()`
- **RAM**: `psutil.virtual_memory().total`
- **GPU count / memory**: `pynvml` (NVIDIA)

Reports capabilities on `Register()` for future scheduling decisions.

---

## 7. Code Structure

```
distripute/
├── proto/
│   └── distripute.proto          # Protobuf service definitions
├── distripute/
│   ├── __init__.py                # Package exports: task, init
│   ├── task.py                    # @distripute.task decorator + init()
│   ├── client.py                  # _Client gRPC stub + _RemoteResult
│   ├── master.py                  # MasterServicer gRPC implementation
│   ├── worker.py                  # Worker gRPC client + uv runner
│   ├── relay.py                   # RelayServicer gRPC bidirectional stream
│   ├── registry.py                # Model metadata registry (optional)
│   ├── cli.py                     # CLI entry points
│   └── grpc/
│       ├── __init__.py
│       ├── distripute_pb2.py      # Generated protobuf message classes
│       └── distripute_pb2_grpc.py # Generated gRPC client/server stubs
├── tests/
│   ├── test_scheduler.py          # gRPC master service tests
│   ├── test_task.py               # @distripute.task decorator tests
│   ├── test_registry.py           # Model registry tests
│   └── test_cli.py                # CLI command tests
└── Makefile                       # proto generation target
```

---

## 8. Planned: Model Parallelism (`@distripute.shard`)

For models too large for one GPU, sharding splits transformer layers across workers:

```
input → Worker A  →  Worker B  →  Worker C  → output
        (layers     (layers      (layers
          0-10)      11-21)       22-32)
```

Planned API:
```python
@distripute.shard(stage=0, num_stages=3)
def whisper_stage(inputs):
    # Each worker receives the same source but different layer range
    # Stage 0 processes layers 0-10, passes to next worker
    # Stage 1 receives activations, processes layers 11-21
    ...
```

**Implementation approach:** The master assigns `shard_index + num_shards` to each worker. Workers pass activations between each other via direct gRPC connections. The `RelayFrame` routing supports worker→worker forwarding.

**Layer assignment** uses the model registry (`registry.py`) which knows `num_layers` per model. `compute_shards(name, num_workers)` returns `[(start, end), ...]` tuples.

---

## 9. Planned: Pipeline Parallelism (`@distripute.pipeline`)

For multi-step workflows where different stages run on different workers:

```
Audio → segment → transcribe → translate → output
         [W1]       [W2]        [W3]
```

Planned API:
```python
@distripute.pipeline(stage=0, name="segments", next="transcribe")
def split_audio(path: str) -> list:
    ...

@distripute.pipeline(stage=1, name="transcribe", next="translate")
def transcribe_segment(segment: dict) -> str:
    ...

@distripute.pipeline(stage=2, name="translate")
def translate_text(text: str) -> str:
    ...
```

Each stage runs on a dedicated worker. Output of stage N is routed to stage N+1. Master coordinates the pipeline topology.

---

## 10. Comparison to Existing Systems

| Feature | Ray | Petals | Exo | **Distripute** |
|---|---|---|---|---|
| API | `@ray.remote` | Client-server | REST API | **`@distripute.task`** |
| Transport | gRPC (plasma store) | gRPC (custom) | HTTP + WebSocket | **gRPC** |
| Code distribution | Shared FS / serialized | Built-in layers | Built-in | **Source shipped per-task** |
| Dependency mgmt | pip on cluster | pip | pip + Nix | **`uv run --with` auto** |
| Cross-internet | ❌ (GCS/Redis) | ✅ DHT + relays | ✅ libp2p | **✅ gRPC relay** |
| Data parallelism | ✅ Tasks | ❌ | ❌ | **✅ @distripute.task** |
| Model parallelism | ❌ | ✅ Pipeline | ✅ Tensor | **🔜 @distripute.shard** |
| Pipeline | ❌ | Same as MP | ❌ | **🔜 @distripute.pipeline** |
| Worker requirement | Python + deps | Python + CUDA | Python + MLX | **Only `uv`** |
| Language | Python | Python | Python | **Python + any via source** |

---

## 11. Configuration Reference

| Env | CLI flag | Default | Component | Description |
|---|---|---|---|---|
| `DISTRIPUTE_MASTER` | `--master` | — | Worker/Client | Master gRPC address |
| `DISTRIPUTE_RELAY` | `--relay` | — | Master/Worker | Relay gRPC address |
| `DISTRIPUTE_REDIS` | `--redis` | — | Master | Redis URL (redis://host:6379) |
| `DISTRIPUTE_NETWORK_ID` | `--network-id` | — | Worker/Client | Network join token |
| `DISTRIPUTE_LOG_LEVEL` | `--log-level` | INFO | All | Log verbosity |

---

## 12. Generating Protobuf Code

```bash
make proto
# or manually:
uv run python3 -m grpc_tools.protoc \
    -I proto \
    --python_out=distripute/grpc \
    --grpc_python_out=distripute/grpc \
    proto/distripute.proto
# then fix the relative import:
sed -i 's/^import distripute_pb2/from . import distripute_pb2/' distripute/grpc/distripute_pb2_grpc.py
```
