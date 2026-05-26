# Distripute PRD — Distributed Inference System

## Problem Statement

Running ML inference on large datasets (thousands of audio files for ASR, document images for OCR, or prompts for LLMs) is slow on a single machine. Existing distributed inference solutions target specific niches:

- **Petals** is P2P pipeline-parallel only, for LLMs, GPU-only, no batch job support
- **Ray** is a general framework requiring cluster setup (GCS/Redis), no built-in model management
- **vLLM** / **TGI** / **DeepSpeed** are single-node multi-GPU serving systems, not batch job engines
- **SLURM / HTCondor** are HPC schedulers, not inference-aware

None handle the full workflow of: ad-hoc worker joining via a simple token → auto model deployment → data or model parallelism across heterogeneous nodes (CPU + GPU) → batch job completion with streamed results.

A researcher who wants to transcribe 5000 hours of audio using whisper-large-v3 on spare lab machines, or OCR 100K documents across friends' laptops, has no simple tool for this.

## Solution

Distripute is a distributed inference job system with three distribution strategies:

### Data Parallelism (default)
Every worker loads the full model. Master shards input files across workers. Use when model fits on one node. Zero latency overhead — each file is processed independently.

### Model Parallelism (tensor sharding)
Model layers split across workers. Each worker hosts a slice of the model. Use when model exceeds single node VRAM. Forward pass requires inter-worker communication per layer.

### Pipeline Parallelism (Petals-style)
Model layers chained sequentially. Data flows worker → worker → worker. Use for maximum model size at higher latency.

### Core Innovations

1. **Network ID** — Master prints a hex token on startup. Workers join with it. No DNS, no Redis, no static config.
2. **mDNS discovery** — Workers on the same LAN auto-discover the master by network_id.
3. **Model Registry** — Declarative knowledge of model architectures (whisper layer count, transformer block size, etc.) enables automatic shard assignment.
4. **Plugin protocol** — Inference backends are standalone executables communicating via JSON over stdin/stdout. Any language (Python, Go, Rust, C++) can be a plugin.
5. **Heterogeneous pools** — Workers report CPU cores, RAM, GPU count, GPU memory. Scheduler matches tasks to appropriate workers.

## User Stories

1. As a researcher, I want to start a master node on my machine and get a network_id, so that I know how others can join.
2. As a lab member, I want to join a distripute network by passing the network_id and master address, so that my GPU contributes to the inference workload.
3. As a colleague on the same LAN, I want workers to auto-discover the master via mDNS, so that I don't need to type the IP address.
4. As a job submitter, I want to submit an ASR job with an input directory of audio files, so that all files get transcribed.
5. As a job submitter, I want to submit an OCR job with an input directory of scanned documents, so that all documents get OCR'd.
6. As a job submitter, I want to submit an LLM inference job with prompts, so that text generation runs across the cluster.
7. As a job submitter, I want to choose between data_parallel, model_parallel, and pipeline_parallel strategies, so that I can handle models of any size.
8. As a job submitter, I want to see job progress (done/total, active workers), so that I know when results will be ready.
9. As a job submitter, I want results written to a specified output directory as each task completes, so that I can stream partial results.
10. As a worker operator, I want my machine's capabilities (CPU cores, GPU count, GPU memory) auto-detected and reported on registration, so that I don't manually configure them.
11. As a worker operator, I want to specify which models I can serve, so that I don't receive tasks for models I don't have.
12. As a worker operator, I want the worker to automatically download and cache models, so that I don't manually install them.
13. As a system admin, I want the master to detect worker disconnection (no heartbeat > 30s), so that tasks are re-assigned to alive workers.
14. As a developer, I want to write a custom inference plugin in any language, so that I can add support for any model.
15. As a developer, I want the plugin protocol to be well-documented JSON over stdio, so that implementing a plugin is trivial.
16. As a developer, I want a CLI tool to manage jobs (create, list, status) and workers (list), so that I can interact with the system from scripts.
17. As a developer, I want the model registry to know common model architectures, so that automatic sharding works for popular models.
18. As a tester, I want unit tests for the scheduler, registry, and CLI, so that I can validate correctness.
19. As a tester, I want integration tests with real master + worker processes, so that I can verify E2E behavior.
20. As a contributor, I want clear documentation of the API and architecture, so that I can extend the system.

## Implementation Decisions

### Language & Runtime
- **Python 3.14+** — full ML ecosystem access (PyTorch, transformers, whisper, vLLM)
- **aiohttp** for HTTP/JSON — simple, no protobuf compilation needed; can upgrade to gRPC later
- **asyncio** throughout — non-blocking I/O for network + subprocess management
- **click** for CLI — standard Python CLI framework
- Optional hot-path subprocesses can be Go/Rust later

### Project Structure
```
distripute/
├── pyproject.toml          # uv-managed project config
├── README.md
├── PRD.md
├── distripute/
│   ├── __init__.py
│   ├── master.py           # HTTP server, scheduler, registry
│   ├── worker.py           # Worker agent, heartbeat, task loop
│   ├── cli.py              # CLI commands (job, worker, info)
│   ├── registry.py         # Model registry (architecture knowledge)
│   └── runner.py           # Plugin subprocess runner
├── plugins/
│   ├── asr_infer.py        # whisper-based ASR plugin
│   └── ocr_infer.py        # paddleocr-based OCR plugin
└── tests/
    ├── test_scheduler.py
    ├── test_registry.py
    └── test_cli.py
```

### Master API (HTTP + JSON)

| Method | Path | Purpose |
|---|---|---|
| POST | `/register` | Worker joins with network_id + capabilities |
| POST | `/heartbeat` | Worker heartbeat (every 10s) |
| POST | `/tasks/poll` | Worker fetches pending tasks |
| POST | `/tasks/result` | Worker submits task result |
| POST | `/job` | Create a new inference job |
| GET | `/job/{id}` | Get job status/progress |
| GET | `/jobs` | List all jobs |
| GET | `/workers` | List all registered workers |
| GET | `/info` | Network info + status |

### Scheduler Design

The scheduler runs in-memory on the master. No persistence for v0.1.

```
schedule(worker_id, max_tasks):
  1. Filter pending tasks
  2. For each task, find eligible workers:
     - Worker must have heartbeat in last 30s
     - Worker must support the task's model (or have empty supported_models)
  3. Sort eligible workers by load (ascending)
  4. Assign up to max_tasks where the requesting worker is eligible
  5. Return assigned tasks
```

### Model Registry (automatic sharding)

The registry maps model names to architecture metadata:

```python
{
  "whisper-large-v3": {
    "family": "whisper",
    "num_layers": 32,
    "layer_type": "transformer_decoder",
    "param_count": 1550_000_000,
    "min_gpu_mem": 4_000_000_000,  # bytes
    "min_ram": 8_000_000_000,
  },
  "whisper-small": {
    "family": "whisper",
    "num_layers": 12,
    "layer_type": "transformer_decoder",
    "param_count": 244_000_000,
    "min_gpu_mem": 1_000_000_000,
    "min_ram": 2_000_000_000,
  },
}
```

For `model_parallel` strategy, the master divides `num_layers` across eligible workers. For `pipeline_parallel`, workers are ordered into a chain, each assigned a contiguous range of layers.

### Plugin Protocol

Worker spawns a persistent subprocess per plugin type. Communication via JSON lines on stdin/stdout:

**Worker → Plugin (stdin):**
```json
{"type":"infer","task_id":"abc-0001","task_type":"asr",
 "input_path":"/data/audio.wav","output_path":"/results/audio.txt",
 "params":{"model":"whisper-large-v3","language":"en"},
 "shard_info":null}
```

`shard_info` is present when strategy == model_parallel:
```json
{"shard_info":{"model_path":"/models/whisper","layer_start":0,"layer_end":16,"num_shards":2}}
```

**Plugin → Worker (stdout):**
```json
{"type":"result","task_id":"abc-0001","success":true,
 "output":"transcribed text","error":"","duration":3.21}
```

### Worker Discovery

1. **Primary:** `--master <host:port> --network-id <token>` on the worker CLI
2. **Optional:** `--discover` flag uses mDNS to find a master on LAN broadcasting the matching network_id
3. Master validates network_id on `/register` — rejects mismatched tokens

### Heartbeat & Fault Tolerance

- Workers heartbeat every 10s to `/heartbeat` with current load and active task count
- Master marks workers as dead after 30s of no heartbeat
- Tasks assigned to dead workers are returned to the pending queue for re-assignment
- Results from dead workers are discarded (task retried on another worker)

## Testing Decisions

### Test Philosophy
- Test external behavior, not implementation details
- Use real HTTP requests against the master server
- Mock the subprocess runner to avoid requiring GPU/ML libraries in CI
- Use `pytest` with `pytest-asyncio` for async tests

### Modules to Test

**test_scheduler.py**
- Worker joins → can poll tasks
- Tasks assigned to eligible workers only
- Dead workers don't receive tasks
- Tasks round-robin across workers with equal load
- Busy workers get fewer new tasks
- Tasks re-queued when worker disconnects mid-task

**test_registry.py**
- Known model returns correct layer count and memory requirements
- Unknown model raises KeyError
- Shard computation divides layers correctly
- Shard computation handles non-divisible layers

**test_cli.py** (integration)
- `distripute job create` sends correct API request
- `distripute job list` parses response correctly
- `distripute info` returns network_id

### Prior Art
Standard aiohttp test patterns using `aiohttp.test_utils.AioHTTPTestCase` or `pytest-aiohttp`. CLI tests use `click.testing.CliRunner`.

## Out of Scope (v0.1)

- Persistent job state (no restart safety — all in-memory)
- Authentication beyond network_id (no per-user auth)
- Encrypted inter-node communication (plain HTTP for v0.1)
- Web UI (CLI only)
- Dynamic model shard rebalancing (static assignment per job)
- Checkpoint/resume for long-running jobs (lost on master restart)
- Container orchestration (docker-compose example only)

## Further Notes

- Network_id is a 12-char hex string (48 bits of entropy) — sufficient for lab/ad-hoc clusters. Not cryptographically secure.
- The mDNS discovery uses `zeroconf` library for Python.
- Plugin subprocess is kept alive across tasks. Worker sends tasks as JSON lines. Plugin responds per task. Plugin can batch internally if beneficial.
- For model_parallel + pipeline_parallel, the master orchestrates layer assignment. The actual inter-worker communication (sending activations between shards) is handled by the plugins — the distripute control plane only assigns which layers each worker hosts.
- The worker auto-detects hardware via `psutil` (CPU cores, RAM) and `pynvml` (GPU count, GPU memory).
