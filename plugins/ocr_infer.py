"""
OCR inference plugin for Distripute.

Usage: python plugins/ocr_infer.py

Communicates via JSON lines over stdin/stdout (Distripute Plugin Protocol).
"""
import json
import sys
import time


def ocr_image(input_path: str, **kwargs) -> str:
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang=kwargs.get("lang", "en"))
        result = ocr.ocr(input_path, cls=True)
        texts = []
        for line_group in result or []:
            for line in line_group or []:
                texts.append(line[1][0])
        return "\n".join(texts)
    except ImportError:
        return f"[mock-ocr] paddleocr: {input_path}"


def main():
    for line in sys.stdin:
        task = json.loads(line)
        task_id = task.get("task_id", "")
        input_path = task.get("input_path", "")
        params = task.get("params", {})
        start = time.time()
        try:
            output = ocr_image(input_path, **params)
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
