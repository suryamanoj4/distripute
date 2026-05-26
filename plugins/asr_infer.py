"""
ASR inference plugin for Distripute.

Usage: python plugins/asr_infer.py

Communicates via JSON lines over stdin/stdout (Distripute Plugin Protocol).
"""
import json
import sys
import time


def transcribe(input_path: str, model: str = "whisper-large-v3", **kwargs) -> str:
    try:
        import whisper
        model_obj = whisper.load_model(model)
        result = model_obj.transcribe(input_path, **kwargs)
        return result.get("text", "")
    except ImportError:
        return f"[mock-asr] {model}: {input_path}"


def main():
    for line in sys.stdin:
        task = json.loads(line)
        task_id = task.get("task_id", "")
        input_path = task.get("input_path", "")
        params = task.get("params", {})
        model = params.get("model", "whisper-large-v3")
        start = time.time()
        try:
            output = transcribe(input_path, model=model)
            result = dict(
                task_id=task_id, success=True, output=output,
                error="", duration=time.time() - start,
            )
        except Exception as e:
            result = dict(
                task_id=task_id, success=False, output="",
                error=str(e), duration=time.time() - start,
            )
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
