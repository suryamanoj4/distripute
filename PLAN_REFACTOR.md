# Refactor: Ray-Style `@distripute.task` Architecture

## Problem

Current Distripute has a rigid "plugin + job" model:
- Inference code must be pre-installed on every worker
- Jobs are file-oriented (scan directory, create task per file)
- No Python API for submitting work — CLI only
- Plugin system lacks versioning and distribution

## Solution: `@distripute.task`

Distripute becomes three layers:

1. **Mesh layer** — master + worker + relay nodes (already built, keep as-is)
2. **SDK layer** — `distripute.init()` + `@distripute.task` decorator (new)
3. **Execution layer** — workers execute received source code via `uv run --with` (new)

## Architecture

```
User Machine                          Worker Machine
┌──────────────────────┐              ┌──────────────────────┐
│  my_infer.py          │              │  distripute-worker   │
│                       │  func_name   │                       │
│  @distripute.task     │  entire_file │  uv run --with dep1  │
│  def transcribe():    │  requirements│    --with dep2        │
│                       │  args,kwargs │    python3 source.py  │
│  fut = transcribe()   ├──────────────►  func(result)         │
│  fut.get()            │◄─────────────│  return result        │
└──────────────────────┘              └──────────────────────┘
         │                                    │
         │ POST /task   GET /task/{id}         │ POST /tasks/poll
         ▼                                    ▼
   ┌──────────────────────────────────────────────┐
   │              Master                           │
   │  /task → queue → /tasks/poll → /tasks/result │
   └──────────────────────────────────────────────┘
```

## Key Components

### 1. `distripute.init(addr, network_id)`
Connects to the mesh. Without this, `@distripute.task` calls raise `DistriputeNotConnectedError`.

### 2. `@distripute.task(requirements=["whisper", "torch"])`
- Reads the **entire source file** containing the decorated function
- Wraps it in a `RemoteFunction` that submits to the mesh when called
- `requirements` list tells workers which deps to install via `uv run --with`

### 3. Worker execution with `uv run --with`
- Worker receives: `func_name`, `entire_source`, `requirements`, `args`, `kwargs`
- Writes source to `~/.cache/distripute/jobs/{hash}/source.py`
- Runs: `uv run --with dep1 --with dep2 python3 source.py`
- The source file has a `__main__` section appended by the decorator that:
  - Reads args from env var / stdin
  - Calls the named function
  - Prints JSON result to stdout
- Worker captures stdout → submits result to master

### 4. No pre-installed deps needed
Workers only need `uv`. Dependencies are auto-installed and cached by uv.

## Task Payload

```json
{
  "task_id": "abc123",
  "func_name": "transcribe",
  "entire_source": "import whisper\ndef transcribe(path):\n    ...",
  "requirements": ["openai-whisper", "torch"],
  "args": [["/data/audio.wav"], {}]
}
```

## Files to Create

| File | Purpose |
|---|---|
| `distripute/task.py` | `@distripute.task` decorator, `init()`, `RemoteFunction` |
| `distripute/client.py` | `_Client` — connects to master, submits tasks, polls results |

## Files to Modify

| File | Change |
|---|---|
| `distripute/master.py` | Add `POST /task` (client submits), `GET /task/{id}` (client polls). Tasks carry `func_name` + `entire_source` + `requirements` + serialized args. Remove job API. |
| `distripute/worker.py` | Replace plugin subprocess with `uv run`-based execution. Write source + runner, exec with `uv run --with`, capture result. |
| `distripute/relay.py` | Forward task submission/polling through WebSocket. |
| `distripute/cli.py` | Remove `job`, `plugin` commands. Keep `master`, `worker`, `relay`, `info`. |
| `distripute/__init__.py` | Export `task`, `init` at package level. |

## User Script Example

```python
import distripute

distripute.init("localhost:9090", "dstr-a1b2c3d4e5")

@distripute.task(requirements=["openai-whisper"])
def transcribe(path: str) -> str:
    import whisper
    model = whisper.load_model("base")
    return model.transcribe(path)["text"]

result = transcribe("/data/audio.wav").get()
print(result)

# Without init():
# @distripute.task def f(): ...
# f() → raises DistriputeNotConnectedError
```
