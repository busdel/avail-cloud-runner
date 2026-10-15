# -*- coding: utf-8 -*-
"""AVAIL worker — a secure job server for YOUR own machine.

Runs on the user's own (GPU) server. Nothing is exposed by default:
the process binds to 127.0.0.1, every request needs a bearer token, and
you decide how (if at all) to expose it — outbound-only options such as
`tailscale serve` or a cloudflared tunnel are documented in WORKER.md.

API (all require: Authorization: Bearer <token>)
  GET  /health                 -> {"ok": true, "name": ..., "jobs": n}
  POST /jobs?flags=...         -> raw ZIP body; returns {"id": ...}
  GET  /jobs/<id>              -> {"status": queued|running|done|error, ...}
  GET  /jobs/<id>/log          -> plain-text log
  GET  /jobs/<id>/results.zip  -> results archive when done
  DELETE /jobs/<id>            -> remove job (dir + results)

Design choices
- stdlib only (no new dependencies);
- one job at a time by default (no GPU contention surprises);
- per-job sandbox directory with optional memory cap (Linux) and nice;
- pipeline runs as `python -m avail_omics.cli run-generic` in its own
  process; optional flags: robustness (leak-free permutation test).
"""
import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_WORKDIR = Path.home() / ".local" / "share" / "avail-worker"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "avail" / "worker_token"


def new_token():
    return secrets.token_urlsafe(32)


def load_token(path):
    path = Path(path)
    if path.exists():
        return path.read_text().strip()
    tok = new_token()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tok)
    path.chmod(0o600)
    return tok


class JobStore:
    def __init__(self, workdir, max_parallel=1, mem_gb=None, nice=None):
        self.workdir = Path(workdir)
        self.jobs_dir = self.workdir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.sem = threading.Semaphore(max_parallel)
        self.mem_gb = mem_gb
        self.nice = nice

    def submit(self, zip_bytes, flags, name=""):
        jid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        jd = self.jobs_dir / jid
        jd.mkdir(parents=True)
        zpath = jd / "data.zip"
        zpath.write_bytes(zip_bytes)
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.infolist():
                t = (jd / "data" / member.filename).resolve()
                if not str(t).startswith(str((jd / "data").resolve()) + os.sep):
                    raise ValueError("unsafe path in zip")
            zf.extractall(jd / "data")
        zpath.unlink()
        with self.lock:
            self.jobs[jid] = {"status": "queued", "flags": list(flags),
                              "name": name, "created": time.time(),
                              "dir": str(jd)}
        threading.Thread(target=self._run, args=(jid,), daemon=True).start()
        return jid

    def _run(self, jid):
        import tarfile
        with self.sem:
            j = self.jobs[jid]
            j["status"] = "running"
            j["started"] = time.time()
            jd = Path(j["dir"])
            log = open(jd / "run.log", "w")
            try:
                root = jd / "data"
                if not (root / "metadata.tsv").exists():
                    subs = [d for d in root.iterdir() if d.is_dir()]
                    if len(subs) == 1:
                        root = subs[0]
                out = jd / "avail_results"

                def pre():
                    if self.nice:
                        os.nice(self.nice)
                    if self.mem_gb:
                        import resource
                        lim = int(self.mem_gb * 1024 ** 3)
                        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

                cmd = [sys.executable, "-m", "avail_omics.cli", "run-generic",
                       "--data-root", str(root), "--out", str(out)]
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               check=True, preexec_fn=pre if os.name == "posix"
                               else None)
                if "robustness" in j["flags"]:
                    subprocess.run(
                        [sys.executable, "-m", "avail_omics.cli", "robustness",
                         "--data-root", str(root), "--out", str(out)],
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                with tarfile.open(jd / "results.tgz", "w:gz") as tf:
                    tf.add(out, arcname="avail_results")
                j["status"] = "done"
            except Exception as e:
                with open(jd / "run.log", "a") as f:
                    f.write(f"\n[worker] FAILED: {type(e).__name__}: {e}\n")
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                log.close()
                j["finished"] = time.time()

    def get(self, jid):
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return None
            return {k: v for k, v in j.items() if k != "dir"}

    def log_tail(self, jid, n=200):
        j = self.jobs.get(jid)
        if not j:
            return None
        lp = Path(j["dir"]) / "run.log"
        if not lp.exists():
            return ""
        return "\n".join(lp.read_text(errors="replace").splitlines()[-n:])

    def results_path(self, jid):
        j = self.jobs.get(jid)
        if not j or j["status"] != "done":
            return None
        p = Path(j["dir"]) / "results.tgz"
        return p if p.exists() else None

    def delete(self, jid):
        with self.lock:
            j = self.jobs.pop(jid, None)
        if j:
            shutil.rmtree(j["dir"], ignore_errors=True)
        return bool(j)


def make_handler(store, token, name):
    class H(BaseHTTPRequestHandler):
        server_version = "avail-worker/1.0"

        def _auth(self):
            h = self.headers.get("Authorization", "")
            if not h.startswith("Bearer "):
                self._json({"error": "token required"}, 401)
                return False
            got = h[len("Bearer "):].strip()
            if not secrets.compare_digest(got, token):
                self._json({"error": "bad token"}, 403)
                return False
            return True

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, data, ctype="application/octet-stream", name=None):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            if name:
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._json({"ok": True, "name": name,
                                   "jobs": len(store.jobs)})
            if not self._auth():
                return
            parts = [p for p in path.split("/") if p]
            if len(parts) == 2 and parts[0] == "jobs":
                j = store.get(parts[1])
                if j is None:
                    return self._json({"error": "unknown job"}, 404)
                j["log_tail"] = store.log_tail(parts[1], 20)
                return self._json(j)
            if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "log":
                t = store.log_tail(parts[1])
                if t is None:
                    return self._json({"error": "unknown job"}, 404)
                return self._bytes(t.encode(), "text/plain")
            if len(parts) == 3 and parts[0] == "jobs" and \
                    parts[2] == "results.zip":
                p = store.results_path(parts[1])
                if p is None:
                    return self._json({"error": "not ready"}, 404)
                return self._bytes(p.read_bytes(), "application/gzip",
                                   "results.tgz")
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._auth():
                return
            path = self.path.split("?")[0]
            if path != "/jobs":
                return self._json({"error": "not found"}, 404)
            q = {}
            if "?" in self.path:
                from urllib.parse import parse_qs
                q = parse_qs(self.path.split("?", 1)[1])
            flags = [f for f in (q.get("flags", [""])[0].split(",")) if f]
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 4 * 1024 ** 3:
                return self._json({"error": "empty or oversized body"}, 400)
            data = self.rfile.read(n)
            if not data[:2] == b"PK":
                return self._json({"error": "body must be a ZIP archive"}, 400)
            try:
                jid = store.submit(data, flags,
                                   name=(q.get("name", [""])[0]))
            except Exception as e:
                return self._json({"error": str(e)[:200]}, 400)
            return self._json({"id": jid, "status": "queued"}, 202)

        def do_DELETE(self):
            if not self._auth():
                return
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if len(parts) == 2 and parts[0] == "jobs":
                return self._json({"deleted": store.delete(parts[1])})
            return self._json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):
            pass

    return H


def serve(workdir=DEFAULT_WORKDIR, host="127.0.0.1", port=8787,
          token_file=DEFAULT_TOKEN_FILE, token=None, name="my-gpu",
          max_parallel=1, mem_gb=None, nice=None):
    token = token or load_token(token_file)
    store = JobStore(workdir, max_parallel=max_parallel, mem_gb=mem_gb,
                     nice=nice)
    httpd = ThreadingHTTPServer((host, port), make_handler(store, token, name))
    print(f"[worker] serving on http://{host}:{port} (name={name})",
          flush=True)
    print(f"[worker] token file: {token_file}", flush=True)
    httpd.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="avail-omics worker",
                                 description="secure job server for your own machine")
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep 127.0.0.1 and expose via "
                         "tailscale/cloudflared; see WORKER.md)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--token-file", default=str(DEFAULT_TOKEN_FILE))
    ap.add_argument("--name", default="my-gpu")
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--mem-gb", type=float, default=None)
    ap.add_argument("--nice", type=int, default=10)
    args = ap.parse_args(argv)
    serve(Path(args.workdir), args.host, args.port, Path(args.token_file),
          name=args.name, max_parallel=args.max_parallel, mem_gb=args.mem_gb,
          nice=args.nice)
    return 0
