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
