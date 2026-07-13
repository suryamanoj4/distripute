# Distripute PRD — Distributed Inference System

## Problem Statement

Running ML inference on large datasets (thousands of audio files for ASR, document images for OCR, or prompts for LLMs) is slow on a single machine. Existing distributed inference solutions target specific niches:

- **Petals** is P2P pipeline-parallel only, for LLMs, GPU-only, no batch job support
- **Ray** is a general framework requiring cluster setup (GCS/Redis), no built-in model management
- **vLLM** / **TGI** / **DeepSpeed** are single-node multi-GPU serving systems, not batch job engines
- **SLURM / HTCondor** are HPC schedulers, not inference-aware

None handle the full workflow of: ad-hoc worker joining via a simple token → auto-spawn execution on device-aware nodes → data or model parallelism across heterogeneous nodes (CPU + GPU) → batch job completion with streamed results.

A researcher who wants to transcribe 5000 hours of audio using whisper-large-v3 on spare lab machines, or OCR 100K documents across friends' laptops, has no simple tool for this.

## Solution

Distripute is a **Python orchestration framework** that auto-discovers worker device capabilities (CPU cores, RAM, GPU count/memory) and **auto-spawns processes** on workers. It uses a **relay-based architecture** for cross-internet connectivity:

```
Relay (public VPS, gRPC bidirectional stream)
  ├── Master (behind NAT) connects outbound
  ├── Workers (anywhere) connect outbound
  └── Relay bridges by network_id
```

Current architecture: **distributed** — master schedules tasks to workers.
Roadmap: **federated** — peers negotiate work among themselves with no central coordinator.

### Core Innovations

1. **Relay-based connectivity** — Lightweight gRPC bidirectional stream on a public VPS bridges master ↔ workers. All nodes connect outbound. No port forwarding, no VPN, no static IPs.
2. **Network ID** — Master prints a hex token on startup. Workers join with it. The relay routes by network_id.
3. **Device-aware execution** — Workers report CPU cores, RAM, GPU count, GPU memory. The scheduler matches tasks to appropriate workers.
4. **Source shipping** — `@distripute.task` reads your entire source file and ships it to workers. No pre-installed code or containers needed.
5. **Zero-friction worker setup** — Workers only need `uv` installed. Dependencies auto-install via `uv run --with`.
6. **Model Registry** — Declarative knowledge of model architectures enables automatic shard assignment for future model/pipeline parallelism.

## User Stories

1. As a researcher, I want to start a master node on my machine and get a network_id, so that I know how others can join.
2. As a lab member, I want to join a distripute network by passing the network_id and master address, so that my GPU contributes to the inference workload.
3. As a colleague on the same LAN, I want workers to auto-discover the master via mDNS, so that I don't need to type the IP address.
4. As a job submitter, I want to submit a function call with arguments, so that it executes on a remote worker.
5. As a job submitter, I want to submit batch jobs with multiple input files, so that many tasks run across the cluster.
6. As a job submitter, I want to see job progress (done/total, active workers), so that I know when results will be ready.
7. As a worker operator, I want my machine's capabilities (CPU cores, GPU count, GPU memory) auto-detected and reported on registration, so that I don't manually configure them.
8. As a worker operator, I want workers to auto-install dependencies via `uv`, so that I don't manually set up environments.
9. As a system admin, I want the master to detect worker disconnection (no heartbeat > 30s), so that tasks are re-assigned to alive workers.
10. As a developer, I want a simple `@distripute.task` decorator to make any function run remotely, so that I don't learn a new framework.
11. As a developer, I want a CLI tool to manage nodes (master, worker, relay) and query info, so that I can interact with the system from scripts.
12. As a tester, I want unit tests for the scheduler, registry, and CLI, so that I can validate correctness.
13. As a tester, I want integration tests with real master + worker processes, so that I can verify E2E behavior.

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for:
- gRPC service definitions and protobuf messages
- Task lifecycle: submit → schedule → execute → result
- Relay protocol: bidirectional streaming frame routing
- Cache: Redis-backed with in-memory fallback
- Worker execution: `uv run --with` dependency management
- Model registry and shard computation
- Configuration reference

## Out of Scope (v0.1)

- Persistent job state (no restart safety — all in-memory)
- Authentication beyond network_id (no per-user auth)
- Encrypted inter-node communication (plain gRPC for v0.1)
- Web UI (CLI only)
- Model parallelism (`@distripute.shard`) — planned
- Pipeline parallelism (`@distripute.pipeline`) — planned
- Dynamic model shard rebalancing
- Checkpoint/resume for long-running jobs
- Container orchestration

## Further Notes

- Network_id is a 12-char hex string (48 bits of entropy) — sufficient for lab/ad-hoc clusters. Not cryptographically secure.
- The worker auto-detects hardware via `psutil` (CPU cores, RAM) and `pynvml` (GPU count, GPU memory).
- For model_parallel + pipeline_parallel, the master orchestrates layer assignment. The actual inter-worker communication is handled by the plugins — the distripute control plane only assigns which layers each worker hosts.
