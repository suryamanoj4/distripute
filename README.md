# Distripute

> Distributed inference for ASR, OCR, and LLM workloads — data parallelism, model parallelism, and pipeline parallelism across a dynamic worker pool.

**Architecture**

```
                  ┌────────────────┐
                  │   Master Node  │  ← HTTP server, scheduler, registry
                  │  network_id    │     network_id for worker discovery
                  └───────┬────────┘
                          │ HTTP + JSON
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Worker 1 │    │ Worker 2 │    │ Worker 3 │  ← join via network_id
   │ (GPU)    │    │ (CPU)    │    │ (GPU)    │     auto-download models
   │ whisper  │    │ paddleocr│    │ vLLM     │
   └──────────┘    └──────────┘    └──────────┘
```

## Quick Start

```bash
# On machine A — start master
uv run distripute-master
# → prints: NETWORK_ID=dstr-a1b2c3d4e5

# On machines B, C, D — join as workers
uv run distripute-worker \
  --master 192.168.1.100:9090 \
  --network-id dstr-a1b2c3d4e5

# Submit a job from any machine
distripute job create \
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

Workers discover and join the master using a **network ID** — a short hex token printed at master startup. The master does not require static IPs or DNS:

1. Start master → get `NETWORK_ID=dstr-a1b2c3d4e5`
2. Workers join with `--network-id dstr-a1b2c3d4e5`
3. Master validates the ID and registers the worker
4. Workers heartbeat every 10s
5. Workers drop off after 30s of no heartbeat

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
# Create a job
distripute job create \
  --master localhost:9090 \
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

| Env | Default | Description |
|---|---|---|
| `DISTRIPUTE_HOST` | `0.0.0.0` | Master bind address |
| `DISTRIPUTE_PORT` | `9090` | Master port |
| `DISTRIPUTE_LOG_LEVEL` | `INFO` | Log verbosity |

## Development

```bash
uv sync          # install deps
uv sync --dev    # include optional inference deps
distripute-master --port 9090 --log-level DEBUG
```

## License

MIT
