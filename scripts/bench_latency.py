"""Latency of the served API: /recommend p50 / p95 at k = 10 for reranked, hybrid and sasrec, and /similar, in-process
(TestClient) and over HTTP (a uvicorn subprocess, access log off), cycling through the dev shoppers' histories so history
lengths vary like real traffic. Also the server-side latency_ms each response reports. Target: p50 < 50 ms.
Writes eval/results/latency_local.json.

    python scripts/bench_latency.py [--calls 300] [--no-http]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_P50_MS = 50
MODELS = ("reranked", "hybrid", "sasrec")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stats(ms: list[float], server: list[float] | None = None) -> dict:
    a = np.asarray(ms)
    out = {"n": len(a), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2), "mean_ms": round(float(a.mean()), 2)}
    if server:
        out["server_p50_ms"] = round(float(np.percentile(server, 50)), 2)
    return out


def bench(get, post, histories: list[list[int]], items: list[int], calls: int, label: str) -> dict:
    out = {}
    for model in MODELS:
        for i in range(20):  # warm-up
            post("/recommend", {"history": histories[i % len(histories)], "k": 10, "model": model})
        ms, server = [], []
        for i in range(calls):
            t0 = time.perf_counter()
            r = post("/recommend", {"history": histories[i % len(histories)], "k": 10, "model": model})
            ms.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200, r.text
            server.append(r.json()["latency_ms"])
        out[model] = _stats(ms, server)
        print(f"{label:10s} /recommend {model:9s} p50 {out[model]['p50_ms']:6.2f} ms  p95 {out[model]['p95_ms']:6.2f} ms  (server-side p50 {out[model]['server_p50_ms']} ms)", flush=True)
    ms, server = [], []
    for i in range(calls):
        t0 = time.perf_counter()
        r = get(f"/similar/{items[i % len(items)]}?k=10")
        ms.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text
        server.append(r.json()["latency_ms"])
    out["similar"] = _stats(ms, server)
    print(f"{label:10s} /similar            p50 {out['similar']['p50_ms']:6.2f} ms  p95 {out['similar']['p95_ms']:6.2f} ms", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "latency_local.json")
    ap.add_argument("--no-http", action="store_true")
    a = ap.parse_args(argv)
    from fastapi.testclient import TestClient

    import app as app_module

    for name in ("shop_recsys.api", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    svc = app_module.service
    known = [s for s in svc.shoppers if svc.history_matrix(s["history"])[1]["any_known"]]
    histories = [s["history"] for s in known]
    items = [x for s in known for x in s["history"][-1:] if svc.item_info(x)]
    client = TestClient(app_module.app)
    result = {"machine": {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(), "cpus": os.cpu_count(), "python": platform.python_version(),
                          "torch_threads": __import__("torch").get_num_threads()},
              "version": svc.version, "k": 10, "calls_per_endpoint": a.calls, "histories": f"{len(histories)} dev shoppers with a known item, cycled",
              "in_process": bench(client.get, lambda p, b: client.post(p, json=b), histories, items, a.calls, "in-process")}
    if not a.no_http:
        port = _free_port()
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning", "--no-access-log"],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 120  # readiness: /health answers 200 (bounded; the process exits non-zero if it never does)
            while time.time() < deadline:
                try:
                    if httpx.get(base + "/health", timeout=2.0).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if proc.poll() is not None:
                    raise RuntimeError(f"uvicorn exited with {proc.returncode}")
                time.sleep(0.5)
            else:
                raise RuntimeError("uvicorn did not answer /health within 120 s")
            with httpx.Client(base_url=base, timeout=60.0) as http:
                result["http"] = bench(http.get, lambda p, b: http.post(p, json=b), histories, items, a.calls, "http")
        finally:
            proc.terminate()
            proc.wait(timeout=20)
    result["target_p50_ms"] = TARGET_P50_MS
    result["meets_target"] = all(result[mode][m]["p50_ms"] < TARGET_P50_MS for mode in ("in_process", "http") if mode in result for m in (*MODELS, "similar"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"meets p50 < {TARGET_P50_MS} ms: {result['meets_target']} → {a.out.relative_to(ROOT)}")
    return 0 if result["meets_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
