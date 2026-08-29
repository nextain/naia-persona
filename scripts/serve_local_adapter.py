#!/usr/bin/env python3
"""Serve a local Transformers base plus PEFT adapter through a small OpenAI API."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-id", default="naia-qwen3.8-27b-h22-candidate")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-gib", type=int, default=23)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: f"{args.gpu_memory_gib}GiB", "cpu": "96GiB"},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                self.send_json(HTTPStatus.OK, {"object": "list", "data": [{
                    "id": args.model_id, "object": "model", "owned_by": "nextain"
                }]})
                return
            if self.path.rstrip("/") == "/health":
                self.send_json(HTTPStatus.OK, {"status": "ok", "model": args.model_id})
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                messages = request["messages"]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                max_tokens = min(int(request.get("max_tokens", 256)), 512)
                temperature = float(request.get("temperature", 0.7))
                sampling = temperature > 0
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=sampling,
                        temperature=max(temperature, 1e-5) if sampling else None,
                        top_p=float(request.get("top_p", 0.9)) if sampling else None,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                token_ids = output[0, inputs.input_ids.shape[1]:]
                answer = tokenizer.decode(token_ids, skip_special_tokens=True)
                completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                payload = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": args.model_id,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": int(inputs.input_ids.shape[1]),
                        "completion_tokens": int(token_ids.shape[0]),
                        "total_tokens": int(inputs.input_ids.shape[1] + token_ids.shape[0]),
                    },
                    "naia_metrics": {
                        "generation_seconds": time.perf_counter() - started,
                        "completion_tokens_per_second": int(token_ids.shape[0]) / max(time.perf_counter() - started, 1e-9),
                    },
                }
                if request.get("stream"):
                    stream_payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": payload["created"],
                        "model": args.model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }],
                    }
                    body = (
                        f"data: {json.dumps(stream_payload, ensure_ascii=False)}\n\n"
                        "data: [DONE]\n\n"
                    ).encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_json(HTTPStatus.OK, payload)
            except Exception as error:  # API boundary: return structured failure.
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(error), "type": type(error).__name__}})

        def log_message(self, fmt: str, *values: object) -> None:
            print(f"{self.address_string()} - {fmt % values}", flush=True)

    print(json.dumps({"status": "ready", "model": args.model_id, "port": args.port}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
