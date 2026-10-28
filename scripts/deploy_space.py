"""Create or update the Hugging Face Space (Docker SDK, cpu-basic, public) and upload exactly what the service reads: app.py,
the recsys package, the exported bundle in serving/, the committed test results and figures the UI shows, the Dockerfile and
requirements-space.txt, and space/README.md as the Space README. Then wait for RUNNING (the Space status API, polled with a
deadline), print the build log tail (it carries the torch + faiss + LightGBM probe line), and smoke-test the live service:
/health, one /recommend per model, /similar, and 30 timed /recommend calls per benchmarked model.
Never prints a token: it relies on the logged-in session (`hf auth login`).

    python scripts/deploy_space.py [--repo DavutKursun/shop-recsys] [--wait-minutes 30] [--no-smoke] [--calls 30]
Writes eval/results/latency_space.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FILES = ["app.py", "requirements-space.txt", "Dockerfile", ".dockerignore",
         *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "recsys").glob("*.py")),
         "eval/results/test-summary.json", "eval/results/dev_vs_test.json", "eval/results/ann_sasrec.json", "eval/results/ann_two_tower.json",
         "docs/img/reranked_by_history_test.png", "docs/img/recall_by_history_test.png", "docs/img/sampled_vs_full_test.png", "docs/img/reranker_features.png"]
SERVED = ("popularity", "repeat", "hybrid", "markov", "item_knn", "two_tower", "sasrec", "reranked")
TIMED = ("reranked", "hybrid", "sasrec")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{10,}|gh[po]_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,})")


def redact(s: str) -> str:
    return SECRET.sub("[redacted]", s)


def stage(tmp: Path) -> int:
    serving = ROOT / "serving"
    if not (serving / "manifest.json").exists():
        raise FileNotFoundError("serving/ is empty; run python scripts/export_serving.py first")
    rels = FILES + [str(p.relative_to(ROOT)) for p in sorted(serving.iterdir()) if p.is_file()]
    for rel in rels:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(rel)
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(ROOT / "space" / "README.md", tmp / "README.md")
    return sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())


def build_log(repo: str, lines: int = 60) -> list[str]:
    from huggingface_hub.utils import build_hf_headers

    out: list[str] = []
    try:
        with httpx.stream("GET", f"https://huggingface.co/api/spaces/{repo}/logs/build", headers=build_hf_headers(), timeout=20.0) as r:
            for raw in r.iter_lines():
                if raw.startswith("data:"):
                    try:
                        d = json.loads(raw[5:].strip())
                        out.append(redact(str(d.get("data", d))).rstrip())
                    except json.JSONDecodeError:
                        out.append(redact(raw[5:].strip()))
                if len(out) > 5000:
                    break
    except (httpx.HTTPError, httpx.StreamError) as e:
        out.append(f"(log stream ended: {type(e).__name__})")
    return out[-lines:] if lines else out


def _stats(ms: list[float], server: list[float]) -> dict:
    a = np.asarray(ms)
    return {"n": len(a), "round_trip_p50_ms": round(float(np.percentile(a, 50)), 1), "round_trip_p95_ms": round(float(np.percentile(a, 95)), 1),
            "server_p50_ms": round(float(np.percentile(server, 50)), 2), "server_p95_ms": round(float(np.percentile(server, 95)), 2)}


def smoke(base: str, calls: int, out: Path, probe_ok: bool) -> dict:
    shoppers = json.loads((ROOT / "serving" / "shoppers.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "serving" / "manifest.json").read_text(encoding="utf-8"))
    demo = shoppers[manifest.get("demo_shopper") or 0]
    histories = [s["history"] for s in shoppers if s["next_in_vocabulary"]][:200]
    with httpx.Client(base_url=base, timeout=60.0) as http:
        health = None
        deadline = time.time() + 300
        while time.time() < deadline:  # the container can take a moment after RUNNING to answer
            try:
                r = http.get("/health")
                if r.status_code == 200:
                    health = r.json()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        if health is None:
            raise RuntimeError("the live /health never answered 200")
        print("health:", json.dumps({k: health[k] for k in ("status", "models", "n_items", "version", "libraries")}))
        per_model = {}
        for model in SERVED:
            r = http.post("/recommend", json={"history": demo["history"], "k": 10, "model": model})
            assert r.status_code == 200, r.text
            b = r.json()
            per_model[model] = {"items": len(b["items"]), "first": b["items"][0]["item"], "seen_before": sum(x["seen_before"] for x in b["items"]),
                                "sources_present": all(x["sources"] for x in b["items"]), "latency_ms": b["latency_ms"]}
            print(f"recommend {model:10s} {len(b['items'])} items, {per_model[model]['seen_before']} seen before, sources on every item "
                  f"{per_model[model]['sources_present']}, server {b['latency_ms']} ms")
        sim = http.get(f"/similar/{demo['history'][-1]}?k=5")
        assert sim.status_code == 200, sim.text
        print(f"similar: {[x['item'] for x in sim.json()['items']]} in {sim.json()['latency_ms']} ms; unknown item → {http.get('/similar/-1').status_code}")
        lat = {}
        for model in TIMED:
            ms, server = [], []
            for i in range(calls):
                t0 = time.perf_counter()
                r = http.post("/recommend", json={"history": histories[i % len(histories)], "k": 10, "model": model})
                ms.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200, r.text
                server.append(r.json()["latency_ms"])
            lat[model] = _stats(ms, server)
            print(f"live {model:9s} server-side p50 {lat[model]['server_p50_ms']} ms, p95 {lat[model]['server_p95_ms']} ms; "
                  f"round trip p50 {lat[model]['round_trip_p50_ms']} ms over {calls} calls")
    result = {"space": base, "version": health["version"], "hardware": "cpu-basic (2 vCPU)", "calls_per_model": calls, "k": 10,
              "latency": lat, "target_server_p50_ms": 50, "meets_target": all(v["server_p50_ms"] < 50 for v in lat.values()),
              "per_model_smoke": per_model, "build_probe_torch_faiss_lightgbm_ok": probe_ok, "libraries": health["libraries"],
              "note": "server-side = latency_ms reported by the service; round trip = from the developer's laptop, network included"}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="DavutKursun/shop-recsys")
    ap.add_argument("--wait-minutes", type=float, default=30)
    ap.add_argument("--no-upload", action="store_true", help="only wait, read the build log and smoke-test")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "latency_space.json")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi

    api = HfApi()
    print("logged in as", api.whoami()["name"])
    if not a.no_upload:
        url = api.create_repo(repo_id=a.repo, repo_type="space", space_sdk="docker", space_hardware="cpu-basic", private=False, exist_ok=True)
        print("space:", redact(str(url)))
        with tempfile.TemporaryDirectory() as tmp:
            size = stage(Path(tmp))
            print(f"staged {size / 1e6:.1f} MB")
            info = api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="space", commit_message="deploy shop-recsys")
            print("uploaded:", redact(str(getattr(info, "commit_url", info))))
    owner, name = a.repo.split("/")
    base = f"https://{owner.lower()}-{name.lower()}.hf.space"
    deadline, last = time.time() + a.wait_minutes * 60, None
    while time.time() < deadline:  # the Space status API, not a log line
        stage_ = api.get_space_runtime(a.repo).stage
        if stage_ != last:
            print(f"{time.strftime('%H:%M:%S')} stage {stage_}", flush=True)
            last = stage_
        if stage_ == "RUNNING":
            break
        if stage_ in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED", "DELETING"):
            print("\n".join(build_log(a.repo, 80)))
            return 1
        time.sleep(15)
    else:
        print("timed out waiting for RUNNING")
        return 1
    log = build_log(a.repo, 0)
    probe_ok = any("OPENMP_PROBE_OK" in line for line in log)
    print("--- build log (tail) ---")
    print("\n".join(log[-30:]))
    print(f"build probe (torch + faiss + LightGBM in one process on Linux): {'OPENMP_PROBE_OK found' if probe_ok else 'probe line not found in the log'}")
    if not a.no_smoke:
        res = smoke(base, a.calls, a.out, probe_ok)
        print(f"meets server-side p50 < 50 ms: {res['meets_target']} → {a.out.relative_to(ROOT)}")
    print("live:", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
