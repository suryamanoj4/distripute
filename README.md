# Distripute

> Distributed task execution mesh — run any Python function across workers anywhere on the internet.

```python
import distripute

distripute.init("localhost:9090", "dstr-a1b2c3d4e5")

@distripute.task(requirements=["openai-whisper"])
def transcribe(path: str) -> str:
    import whisper
    return whisper.load_model("base").transcribe(path)["text"]

result = transcribe("/data/audio.wav").get()
```

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
                 │  WebSocket)│    │ uv run --with │
                 │ network_id │     │ exec(source)  │
                 └────────────┘     └──────────────┘
```

## Distribution Strategies

The three forms of parallelism are the core idea of Distripute. Each maps to a different `@distripute.*` decorator, controlling how computation is split across the mesh.

### Data Parallelism (`@distripute.task`)

Every call to a `@distripute.task` function runs on **one worker**. Call it N times with different inputs and the master distributes those calls across all available workers. Each worker has the full function code — only the data (arguments) differs.

```
Files: [a.wav, b.wav, c.wav, d.wav]
Calls: transcribe(a)  transcribe(b)  transcribe(c)  transcribe(d)
         └──────┘      └──────┘      └──────┘      └──────┘
       Worker 1      Worker 2      Worker 1      Worker 3   ← each has full source
```

```python
results = [transcribe(f).get() for f in files]  # 4 calls → 3 workers
```

**Use when:** Model fits on one node. Zero communication overhead.

### Model Parallelism (future: `@distripute.shard`)

Model layers are split across workers. Each worker hosts a contiguous range of layers. A forward pass goes through all workers sequentially — Worker A does layers 0–5, passes activations to Worker B for layers 6–10, etc.

```
input → Worker A  →  Worker B  →  Worker C  → output
        (layers     (layers      (layers
          0-5)       6-10)        11-16)
```

Planned API:
```python
@distripute.shard(stage=0, num_stages=3, model="whisper-large-v3")
def encoder_stage(inputs):
    # Worker 0: layers 0-10
    # Worker 1: layers 11-21
    # Worker 2: layers 22-32
    pass
```

**Use when:** Model exceeds single GPU/CPU VRAM.

### Pipeline Parallelism (future: `@distripute.pipeline`)

Independent pipeline stages where each stage runs on a dedicated worker and data streams through them continuously. Like model parallelism but stages can be different functions, not just layer slices.

```
Audio chunks → segment → transcribe → translate → output
                [W1]       [W2]        [W3]       [W4]
```

Planned API:
```python
@distripute.pipeline(stage=0, name="segments")
def split_audio(path): ...

@distripute.pipeline(stage=1, name="transcribe")
def transcribe_segment(segment): ...
```

**Use when:** Multi-step inference pipelines where different workers handle different steps.

### Current Status

| Strategy | Decorator | Status |
|---|---|---|
| Data Parallel | `@distripute.task` | ✅ Implemented |
| Model Parallel | `@distripute.shard` (planned) | 🔜 |
| Pipeline Parallel | `@distripute.pipeline` (planned) | 🔜 |

The mesh (relay + master + worker) and task infrastructure are built. Model and pipeline parallelism will reuse the same networking layer with new decorators for inter-worker coordination and activation passing.

## Quick Start

```bash
# ── 1. On a public VPS — start relay ──
uv run distripute relay --host 0.0.0.0 --port 9091

# ── 2. On machine A — start master ──
uv run distripute master --relay relay.example.com:9091
# → prints: NETWORK_ID=dstr-a1b2c3d4e5

# ── 3. On machines B, C, D anywhere — start workers ──
uv run distripute worker \
  --relay relay.example.com:9091 \
  --network-id dstr-a1b2c3d4e5

# ── 4. Write and run your script (on any machine) ──
cat > my_task.py << 'EOF'
import distripute
distripute.init("relay.example.com:9091", "dstr-a1b2c3d4e5")

@distripute.task(requirements=["openai-whisper"])
def transcribe(path):
    import whisper
    return whisper.load_model("base").transcribe(path)["text"]

result = transcribe("/home/data/audio.wav").get()
print(result)
EOF

uv run python3 my_task.py
```

## How it Works

### `@distripute.task` decorator

Any function decorated with `@distripute.task` is executed on a worker node, not locally. The decorator:

1. Reads the **entire source file** containing the decorated function
2. Ships the source + function name + arguments to the master
3. The master queues it; a worker polls and receives it
4. The worker writes the source to a temp file and runs it via `uv run --with <deps>`
5. Returns the result back through master to the caller

If `distripute.init()` has not been called, calling a `@distripute.task` function raises `DistriputeNotConnectedError`.

### Worker execution with `uv run`

Workers only need `uv` installed. Dependencies are auto-installed and cached by uv:

```
Worker receives: {func_name, source, requirements, args}
  └── writes source.py + runner.py to temp dir
  └── uv run --with dep1 --with dep2 python3 runner.py
  └── captures stdout → parses result → submits to master
```

No pre-installed packages on workers. No SSH. No pip.

## Network Modes

| Mode | Workers → Master | When |
|---|---|---|
| **Direct** | HTTP to master's IP:port | Same LAN, or master has public IP |
| **Relay** | WebSocket to relay (outbound) | Machines anywhere on the internet |

```
# Direct mode
master --port 9090
worker --master 192.168.1.100:9090 --network-id dstr-a1b2c3d4e5

# Relay mode (NAT traversal)
relay --host 0.0.0.0 --port 9091
master --relay relay.io:9091
worker --relay relay.io:9091 --network-id dstr-a1b2c3d4e5
```

## CLI Reference

```bash
distripute master --port 9090                 # Start master
distripute master --relay relay.io:9091       # Master behind NAT
distripute worker --master host:9090 ...      # Join via direct
distripute worker --relay host:9091 ...       # Join via relay
distripute relay --port 9091                  # Start relay
distripute info --master host:9090            # Mesh status
```

## Python API

| Function | Description |
|---|---|
| `distripute.init(addr, network_id)` | Connect to the mesh |
| `@distripute.task(requirements=[...])` | Mark function for distributed execution |
| `.get()` | Wait for result (blocks) |
| `DistriputeNotConnectedError` | Raised when calling `@task` without `init()` |

## Configuration

| Env | CLI | Default | Description |
|---|---|---|---|
| `DISTRIPUTE_MASTER` | `--master` | — | Master address (worker/client) |
| `DISTRIPUTE_RELAY` | `--relay` | — | Relay address (master/worker) |
| `DISTRIPUTE_NETWORK_ID` | `--network-id` | — | Network join token |
| `DISTRIPUTE_LOG_LEVEL` | `--log-level` | INFO | Log verbosity |

## Development

```bash
uv sync
uv run pytest              # 25+ tests
uv run distripute relay --port 9091 --log-level DEBUG &
uv run distripute master --relay localhost:9091 --log-level DEBUG &
uv run distripute worker --relay localhost:9091 --network-id $(tail -1) --log-level DEBUG &
```

## Comparison

| Feature | Ray | Petals | Exo | **Distripute** |
|---|---|---|---|---|
| API | `@ray.remote` | Client-server | API-based | **`@distripute.task`** |
| Code distribution | Shared filesystem | Built-in | Built-in | **Source shipped per-task** |
| Deploy | pip + cluster config | pip | pip + macOS app | **`uv run` auto-installs deps** |
| Cross-internet | ❌ | ✅ DHT relays | ✅ libp2p | **✅ WebSocket relay** |
| Worker setup | pip install deps | pip | pip + nix | **Only `uv` needed** |
| Target | Any compute | LLMs only | LLMs (MLX) | **Any Python function** |

## License

MIT
