#!/usr/bin/env python3
"""Read-only probe for an OpenAI-compatible Naia serving endpoint."""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.error, urllib.request

def get(url: str) -> dict:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
            try: payload = json.loads(body)
            except json.JSONDecodeError: payload = {"body": body[:500]}
            return {"ok": True, "status": response.status, "latency_ms": round((time.monotonic()-started)*1000, 1), "payload": payload}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "latency_ms": round((time.monotonic()-started)*1000, 1), "error": str(exc)}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("SERVING_URL", "http://127.0.0.1:11435"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    result = {"serving_url": base, "models": get(f"{base}/v1/models")}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["models"]["ok"] else 1

if __name__ == "__main__": sys.exit(main())
