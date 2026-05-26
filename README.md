# Distripute

> Distributed inference for ASR, OCR, and LLM workloads — data parallelism, model parallelism, and pipeline parallelism across a dynamic worker pool spanning the internet.

**Architecture**

```
                         ┌──────────────┐
                         │  Relay Node  │  ← WebSocket bridge (public VPS)
                         │  ws://host   │     routes by network_id
                         └──────┬───────┘
                          ╱              ╲
                 ┌────────▼──┐     ┌─────▼────────┐
                 │   Master  │     │   Workers     │
                 │ (HTTP +   │     │ (behind NAT)  │
                 │  WebSocket)│    │ whisper/ocr   │
                 │ network_id │     │ join via relay│
                 └────────────┘     └──────────────┘
```

Three connectivity modes:

| Mode | Workers → Master | Use case |
|---|---|---|
| **Direct** | Workers connect via HTTP to master's IP:port | Same LAN, or master has public IP |
| **Relay** | All nodes connect to relay via WebSocket | Machines across the internet (NAT) |
| **Hybrid** | Local clients use HTTP, remote workers use relay | Mixed scenarios |

## Quick Start

```bash
# ── On a public VPS — start relay ──
uv run distripute-relay --host 0.0.0.0 --port 9091

# ── On machine A (can be behind NAT) — start master ──
uv run distripute-master \
  --relay relay.example.com:9091
# → prints: NETWORK_ID=dstr-a1b2c3d4e5

# ── On machines B, C, D anywhere on the internet — join as workers ──
uv run distripute-worker \
  --relay relay.example.com:9091 \
  --network-id dstr-a1b2c3d4e5

# ── Submit a job (from anywhere) ──
distripute job create \
  --master relay.example.com:9091 \
  --relay \
  --type asr \
  --input ./audio_files/ \
  --output ./transcripts/ \
  --model whisper-large-v3
```

## Distribution Strategies

### Data Parallelism
Each worker loads the full model. Master shards input data across workers.

```
Input files:  [a.wav] [b.wav] [c.wav] [d.wav]
                   │       │       │       │
Worker 1 (whisper) Worker 2 (whisper) Worker 3 (whisper)
```

Use when: model fits on one node's GPU/CPU.

### Model Parallelism (Tensor Sharding)
Model layers are split across workers. Each worker hosts a subset.

```
Worker 1:  [layer 1] [layer 2] [layer 3]
Worker 2:  [layer 4] [layer 5] [layer 6]
Worker 3:  [layer 7] [layer 8] [layer 9]
```

Use when: model exceeds single node VRAM.

### Pipeline Parallelism (Petals-style)
Model layers chained sequentially across workers. Data flows through the chain.

```
input → Worker A → Worker B → Worker C → output
        (layers 1-4)  (5-8)     (9-12)
```

Use when: high latency is acceptable, need maximum model size.

## Task Types

| Type | Model Examples | Input | Output |
|---|---|---|---|
| `asr` | whisper, wav2vec2, deepspeech | audio (wav, mp3, flac) | text |
| `ocr` | paddleocr, easyocr, tesseract | images (png, jpg, pdf) | text |
| `llm` | llama, qwen, mistral, etc. | prompt text | generated text |

## Network Discovery

Workers discover and join the master using a **network ID** — a short hex token printed at master startup.

### Direct Mode (same LAN / public IP)

Workers connect directly to the master via HTTP:

```
master --port 9090
worker --master 192.168.1.100:9090 --network-id dstr-a1b2c3d4e5
```

### Relay Mode (internet, behind NAT)

A lightweight **relay server** on a publicly accessible machine bridges all connections via WebSocket. Both master and workers connect **outbound** to the relay — no port forwarding needed on any node:

```
relay --port 9091              # on a public VPS
master --relay relay.io:9091   # behind NAT, connects outbound
worker --relay relay.io:9091 --network-id dstr-a1b2c3d4e5  # anywhere
```

The relay routes JSON messages by network_id. It does not see task payloads — just forwards WebSocket frames between paired peers.

### Flow

1. Start relay → listens on `ws://host:9091`
2. Start master → connects to relay via WebSocket, registers `network_id`
3. Worker starts → connects to relay, requests to join `network_id`
4. Relay bridges worker ↔ master
5. All subsequent API calls (register, poll, result) flow through the relay
6. Workers heartbeat every 10s; master drops workers after 30s of silence

## API Reference

### Master Node (HTTP)

| Method | Path | Description |
|---|---|---|
| POST | `/register` | Worker joins the network |
| POST | `/heartbeat` | Worker status update |
| POST | `/tasks/poll` | Worker fetches pending tasks |
| POST | `/tasks/result` | Worker submits result |
| POST | `/job` | Client creates a job |
| GET | `/job/{id}` | Job status |
| GET | `/jobs` | List all jobs |
| GET | `/workers` | List all workers |
| GET | `/info` | Network info |

### CLI

```bash
# Create a job (direct mode)
distripute job create \
  --master localhost:9090 \
  --type asr \
  --input ./audio/ \
  --output ./results/ \
  --model whisper-large-v3

# Create a job (via relay)
distripute job create \
  --master relay.example.com:9091 \
  --relay \
  --type asr \
  --input ./audio/ \
  --output ./results/ \
  --model whisper-large-v3

# List jobs
distripute job list --master localhost:9090

# Get job status
distripute job status <job-id> --master localhost:9090

# List workers
distripute worker list --master localhost:9090

# Network info
distripute info --master localhost:9090

# Start relay
distripute relay --host 0.0.0.0 --port 9091
```

## Plugin System

Inference plugins are standalone Python scripts or any executable that communicate via JSON over stdin/stdout.

### Contract

Input (stdin):
```json
{
  "type": "infer",
  "task_id": "abc-0001",
  "task_type": "asr",
  "input_path": "/data/audio.wav",
  "output_path": "/results/audio.txt",
  "params": {"model": "whisper-large-v3", "language": "en"}
}
```

Output (stdout):
```json
{
  "type": "result",
  "task_id": "abc-0001",
  "success": true,
  "output": "transcribed text here",
  "duration": 3.21
}
```

Plugins can be written in any language. The worker spawns a persistent subprocess per plugin type and sends/receives JSON lines.

## Comparison to Existing Systems

| Feature | Petals | Ray | DeepSpeed | ColossalAI | vLLM | **Distripute** |
|---|---|---|---|---|---|---|
| Topology | P2P/swarm | Master-worker | Single-job | Single-job | Single-server | **Master-worker** |
| Model parallelism | Pipeline | N/A | Tensor+Pipeline | Tensor+Pipeline | Tensor+Pipeline | **Data + Model + Pipeline** |
| Data parallelism | No | Yes | Yes | Yes | Yes | **Yes + auto model deploy** |
| CPU support | No | Yes | Partial | Partial | No | **Yes** |
| Job/batch system | No | Yes | No | Yes | No | **Yes** |
| Network ID join | DHT libp2p | GCS/Redis | Static | Static | Static | **Hex network ID** |
| Language | Python | Python | Python | Python | Python+Go | **Python + plugin languages** |
| Target | LLMs | Any | LLMs | LLMs | LLMs | **ASR + OCR + LLM** |

## Configuration

Set via CLI flags or environment variables:

### Master

| Env | CLI flag | Default | Description |
|---|---|---|---|
| `DISTRIPUTE_HOST` | `--host` | `0.0.0.0` | Master HTTP bind address |
| `DISTRIPUTE_PORT` | `--port` | `9090` | Master HTTP port |
| `DISTRIPUTE_RELAY` | `--relay` | `""` | Relay address (host:port) for internet mode |
| `DISTRIPUTE_LOG_LEVEL` | `--log-level` | `INFO` | Log verbosity |

### Worker

| Env | CLI flag | Default | Description |
|---|---|---|---|
| `DISTRIPUTE_MASTER` | `--master` | `""` | Master address (host:port) for direct mode |
| `DISTRIPUTE_RELAY` | `--relay` | `""` | Relay address (host:port) for relay mode |
| `DISTRIPUTE_NETWORK_ID` | `--network-id` | `""` | Network ID to join |
| `DISTRIPUTE_LOG_LEVEL` | `--log-level` | `INFO` | Log verbosity |

### Relay

| Env | CLI flag | Default | Description |
|---|---|---|---|
| `DISTRIPUTE_HOST` | `--host` | `0.0.0.0` | Relay WebSocket bind address |
| `DISTRIPUTE_PORT` | `--port` | `9091` | Relay WebSocket port |
| `DISTRIPUTE_LOG_LEVEL` | `--log-level` | `INFO` | Log verbosity |

## Comparison to Existing Systems

| Feature | Petals | Ray | DeepSpeed | ColossalAI | vLLM | **Distripute** |
|---|---|---|---|---|---|---|
| Topology | P2P/swarm | Master-worker | Single-job | Single-job | Single-server | **Master-worker + relay** |
| Model parallelism | Pipeline | N/A | Tensor+Pipeline | Tensor+Pipeline | Tensor+Pipeline | **Data + Model + Pipeline** |
| Data parallelism | No | Yes | Yes | Yes | Yes | **Yes + auto model deploy** |
| CPU support | No | Yes | Partial | Partial | No | **Yes** |
| Cross-internet | ✅ DHT relays | ❌ direct/VPN | ❌ | ❌ | ❌ | **✅ WebSocket relay** |
| Job/batch system | No | Yes | No | Yes | No | **Yes** |
| Network ID join | DHT libp2p | GCS/Redis | Static | Static | Static | **Hex network ID** |
| Language | Python | Python | Python | Python | Python+Go | **Python + plugin languages** |
| Target | LLMs | Any | LLMs | LLMs | LLMs | **ASR + OCR + LLM** |

## Development

```bash
git clone && cd distripute
uv sync             # install deps
uv sync --dev       # include optional inference deps

# start relay (terminal 1)
uv run distripute-relay --port 9091 --log-level DEBUG

# start master (terminal 2)
uv run distripute-master --relay localhost:9091 --log-level DEBUG

# start worker (terminal 3)
uv run distripute-worker --relay localhost:9091 --network-id <from_master> --log-level DEBUG

# run tests
uv run pytest
```

## License

MIT
