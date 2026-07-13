# Distripute

> Distribute any Python function across machines anywhere on the internet — data parallelism today, model and pipeline parallelism coming next.

```python
import distripute

distripute.init("localhost:9090", "dstr-a1b2c3d4e5")

@distripute.task(requirements=["openai-whisper"])
def transcribe(path: str) -> str:
    import whisper
    return whisper.load_model("base").transcribe(path)["text"]

result = transcribe("/data/audio.wav").get()
print(result)
```

## Vision

Distripute is a distributed execution mesh for Python functions. You write normal Python code, decorate a function with `@distripute.task`, and it runs on a worker somewhere on the mesh — not locally. The mesh spans machines across the internet, handles NAT traversal, auto-installs dependencies, and returns results as if the function ran locally.

Three forms of parallelism form the core:

| Strategy | Decorator | What it does | Status |
|---|---|---|---|
| **Data Parallel** | `@distripute.task` | Distribute function calls across workers, each with full code + different data | ✅ |
| **Model Parallel** | `@distripute.shard` (planned) | Split model layers across workers, activations passed between them | 🔜 |
| **Pipeline Parallel** | `@distripute.pipeline` (planned) | Chain workers into stages, data streams through sequentially | 🔜 |

## Quick Start

```bash
# Terminal 1 — relay (public VPS, for cross-internet)
uv run distripute relay --host 0.0.0.0 --port 9091

# Terminal 2 — master (can be behind NAT)
uv run distripute master --relay relay.example.com:9091
# → NETWORK_ID=dstr-a1b2c3d4e5

# Terminal 3 — worker anywhere
uv run distripute worker --relay relay.example.com:9091 --network-id dstr-a1b2c3d4e5

# Terminal 4 — run a task
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

## CLI

```bash
distripute master --port 9090                # Start master node
distripute master --relay host:9091          # Master behind NAT
distripute worker --master host:9090 ...     # Join via direct connection
distripute worker --relay host:9091 ...      # Join via relay
distripute relay --port 9091                 # Start relay server
distripute info --master host:9090           # Show mesh status
```

## How It Works

1. **`distripute.init()`** connects to the mesh via gRPC
2. **`@distripute.task`** reads your entire source file and wraps the function
3. On call, the decorator ships source + function name + arguments (cloudpickled) to the master via gRPC
4. Master queues the task; a worker polls and receives it
5. Worker executes via `uv run --with deps` — dependencies auto-install
6. Result flows back: worker → master → your script

Without `init()`, calling a `@distripute.task` function raises `DistriputeNotConnectedError`.

## Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full deep dive — gRPC service definitions, relay protocol, data flow diagrams, task lifecycle, batch jobs, and the planned model/pipeline parallelism designs.

## Comparison

| Feature | Ray | Petals | Exo | **Distripute** |
|---|---|---|---|---|
| API | `@ray.remote` | Client-server | API-based | **`@distripute.task`** |
| Code distribution | Shared FS | Built-in | Built-in | **Source shipped per-task** |
| Deploy | pip + cluster config | pip | pip + macOS app | **`uv run` auto-installs deps** |
| Cross-internet | ❌ | ✅ DHT relays | ✅ libp2p | **✅ gRPC relay** |
| Worker setup | pip install deps | pip | pip + nix | **Only `uv` needed** |
| Transport | gRPC (object store) | gRPC (custom) | HTTP + WebSocket | **gRPC** |
| Target | Any compute | LLMs only | LLMs (MLX) | **Any Python function** |

## Development

```bash
uv sync
uv run pytest              # 37 tests
uv run python -m grpc_tools.protoc -I proto --python_out=distripute/grpc --grpc_python_out=distripute/grpc proto/distripute.proto  # Regenerate protobuf
```

## License

MIT
