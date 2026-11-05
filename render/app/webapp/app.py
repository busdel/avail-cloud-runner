#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AVAIL-AI web uploader.

People upload a ZIP with either the generic layout or the full study layout; the server
automatically runs the full pipeline (base learners -> meta-learner
benchmark -> figures -> LLM advisor) and exposes the results for download.

Config via environment:
  AVAIL_WEB_ROOT     base dir containing run_pipeline.py (default: /media/suleyman_disk/AVAIL_GPU)
  AVAIL_PYTHON       python interpreter for pipeline jobs (default: sys.executable)
  AVAIL_MAX_UPLOAD_MB  max upload size in MB (default: 2048)
  AVAIL_WEB_PORT     listen port (default: 8080)
  AVAIL_WEB_TOKEN    optional simple auth token (X-Auth-Token header or ?token=)
"""
import io
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import (Flask, Response, abort, jsonify, redirect,
                   request, send_file, send_from_directory, session, url_for)
from jinja2 import Environment, FileSystemLoader, select_autoescape

BASE = Path(os.environ.get("AVAIL_WEB_ROOT", "/media/suleyman_disk/AVAIL_GPU")).resolve()
PY = os.environ.get("AVAIL_PYTHON") or sys.executable
RADIOMICS_PY = os.environ.get(
    "AVAIL_RADIOMICS_PYTHON", str(BASE / ".venv_radiomics" / "bin" / "python"))
RUNNER = BASE / "run_pipeline.py"
GENERIC_RUNNER = BASE / "run_generic.py"
TUNNEL_LOG = Path(os.environ.get("AVAIL_TUNNEL_LOG",
                                 "/var/log/avail-tunnel.log"))
SERVER_RUN = os.environ.get("AVAIL_ALLOW_SERVER_RUN", "0") == "1"
DEMO_RUN = os.environ.get("AVAIL_ALLOW_DEMO_RUN", "1") == "1"  # built-in demo only

GH_TOKEN = os.environ.get("AVAIL_GH_TOKEN", "")
GH_REPO = os.environ.get("AVAIL_GH_REPO", "")          # owner/repo
GH_BRANCH = os.environ.get("AVAIL_GH_BRANCH", "main")
GH_WORKFLOW = os.environ.get("AVAIL_GH_WORKFLOW", "run.yml")
UPLOADS = BASE / "uploads"
STATIC_DIR = BASE / "webapp" / "static"
MAX_MB = int(os.environ.get("AVAIL_MAX_UPLOAD_MB", "2048"))

REQUIRED_HINTS = [
    "busra_pd_multisite",
    "fMRI correlation and feature selection",
    "paper_pipeline",
]
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


@app.errorhandler(413)
def too_large(e):
    return (
        f"<h3>Upload too large</h3>"
        f"<p>The uploaded file exceeds the server limit of {MAX_MB} MB "
        f"(tunnel uploads are additionally capped near 100 MB by the "
        f"network path).</p>"
        f"<p>For imaging cohorts, run the pipeline locally instead - "
        f"the local runner needs no upload at all: download "
        f"<code>avail_local_runner.zip</code> from any job page, run "
        f"<code>python run_me.py mydata --out results</code>, then upload "
        f"only the small <code>results/</code> ZIP back for the AI report."
        f"</p>", 413)
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify(error="not found"), 404
    return _render("error.html", code=404, title="Page not found",
                   message="The page you requested does not exist."), 404


app.config["SESSION_COOKIE_SECURE"] = os.environ.get("AVAIL_COOKIE_SECURE", "1") != "0"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
    return resp

PASSWORD = os.environ.get("AVAIL_WEB_PASSWORD", "")
CONTACT_EMAIL = os.environ.get("AVAIL_CONTACT_EMAIL", "")
PUBLIC_DEMO = os.environ.get("AVAIL_PUBLIC_DEMO", "0") == "1"
PUBLIC_UPLOADS = os.environ.get("AVAIL_PUBLIC_UPLOADS", "0") == "1"
DEMO_ONLY = not SERVER_RUN and not (GH_TOKEN and GH_REPO)
if PASSWORD:
    app.secret_key = hashlib.sha256(("avail:" + PASSWORD).encode()).digest()
else:
    app.secret_key = hashlib.sha256(os.urandom(32)).digest()

job_queue = queue.Queue()
_ask_times = {}


def _rate_limit_exceeded(key, limit=30):
    now = time.time()
    ts = [t for t in _ask_times.get(key, []) if now - t < 3600]
    if len(ts) >= limit:
        return True
    ts.append(now)
    _ask_times[key] = ts
    if len(_ask_times) > 5000:
        for k in [k for k, v in list(_ask_times.items())
                  if not any(now - t < 3600 for t in v)]:
            _ask_times.pop(k, None)
    return False   # session -> list of timestamps (rate limit)


# ---------------------------------------------------------------- helpers
@app.before_request
def _gate():
    if not PASSWORD or PUBLIC_DEMO:
        return
    if session.get("auth"):
        return
    if request.path.startswith("/dl/"):
        return
    if request.path.startswith("/static/"):
        return
    if request.path == "/login":
        return
    if request.path in ("/privacy", "/terms", "/byo"):
        return
    return redirect(url_for("login"))



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("pw") == PASSWORD:
            session["auth"] = True
            session["_id"] = uuid.uuid4().hex
            return redirect(url_for("index"))
        return _render("login.html", bad=True)
    if session.get("auth"):
        return redirect(url_for("index"))
    return _render("login.html", bad=False)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/privacy")
def privacy():
    return _render("privacy.html", contact=CONTACT_EMAIL)


@app.route("/terms")
def terms():
    return _render("terms.html", contact=CONTACT_EMAIL)


@app.route("/byo")
def byo():
    return _render("byo.html")


@app.route("/comparisons")
def comparisons_old():
    return redirect(url_for("results"))


@app.route("/results")
def results():
    th = ""
    tn2 = BASE / "experiments/THEORY_NOTES.md"
    if tn2.exists():
        th = _md_to_html(tn2.read_text())
    return _render("results.html", theory=th,
                   external=_external_validation_summary())


def _external_validation_summary():
    """Public-cohort validation numbers, read live from experiment results.
    Internal cohort numbers are NEVER exposed here."""
    import pandas as _pd
    EXP = BASE / "experiments" / "results"
    out = []
    specs = [
        ("Wallen 2022 (PD vs HC)", "wallen_avail_v2", "wallen_avail",
         "n=724 · 3 omics layers (species/pathways/KO) · independent shotgun cohort",
         "https://pubmed.ncbi.nlm.nih.gov/35851583/"),
        ("Preclinical AD (amyloid +/-)", "ad_avail", "ad_avail",
         "n=164 · microbiome + pathways + imaging features · real missingness",
         "https://pubmed.ncbi.nlm.nih.gov/37315112/"),
        ("HMP2 IBD (CD / UC / nonIBD)", "hmp2_avail", "hmp2_avail",
         "n=130 · 5 omics layers · genuine sample loss (106/130, 90/130)",
         "https://pubmed.ncbi.nlm.nih.gov/31142855/"),
        ("MetaCardis (IHD vs CTRL)", "metacardis_results", "metacardis_results",
         "n=1087 · clinical + serum + gut metagenome + urine · informative "
         "missingness (LVEF measured in 84% of IHD vs 42% of controls, "
         "chi2 p=9e-41) · availability-aware model keeps the full cohort",
         "https://doi.org/10.1038/s41591-022-01688-4"),
        ("MetaCardis · complete-case arm",
         "metacardis_cc/out", "metacardis_cc/out",
         "n=536 · same pipeline restricted to subjects with all 7 modalities "
         "· discards 51% of the cohort non-randomly · paired comparison to "
         "the availability-aware arm above",
         "https://doi.org/10.1038/s41591-022-01688-4"),
        ("ds004392 · PD cognitive spectrum (Neurocon)",
         "ds004392_results", "ds004392_results",
         "n=58 · rs-fMRI (Schaefer-100 summaries) + Hoehn & Yahr + "
         "cognitive domains · PD-NC / PD-MCI / PDD · same pipeline, no code "
         "changes (clinical + imaging composite)",
         "https://openneuro.org/datasets/ds004392"),
        ("ds005892 · PD-MCI (OpenNeuro)",
         "ds005892_results", "ds005892_results",
         "n=55 · rs-fMRI · HC / PD-NC / PD-MCI portability of the "
         "imaging-only stack",
         "https://openneuro.org/datasets/ds005892"),
    ]
    for title, d1, d2, desc, url in specs:
        best = None
        for d in (d1, d2):
            p = EXP / d / "meta_benchmark_results.tsv"
            if p.exists():
                try:
                    df = _pd.read_csv(p, sep="\t")
                    sub = df[(df.subgroup == "all") &
                             (~df.model.str.contains("GNN"))]
                    if len(sub):
                        r = sub.sort_values("macro_auc", ascending=False) \
                            .iloc[0]
                        best = {"model": r.model,
                                "auc": round(float(r.macro_auc), 3)}
                        break
                except Exception:
                    pass
        out.append({"title": title, "desc": desc, "url": url,
                    "best": best})
    # label-free reconstruction (no labels needed)
    rec = EXP / "reconstruction_benchmark_vascellari.tsv"
    rec_best = None
    if rec.exists():
        try:
            df = _pd.read_csv(rec, sep="\t")
            g = df.groupby("strategy")["rmse"].mean().sort_values()
            rec_best = {"strategy": g.index[0], "rmse": round(float(g.iloc[0]), 2)}
        except Exception:
            pass
    return {"cohorts": out, "reconstruction": rec_best}


def _worker_state_path(job_id):
    return _job_dir(job_id) / "state.json"


def _worker_api(url, path, token, method="GET", data=None, timeout=30):
    req = urllib.request.Request(url.rstrip("/") + path, method=method,
                                 data=data)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "avail-ui/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return r.status, (json.loads(body) if body[:1] in (b"{", b"[")
                          else body)


_worker_poller_started = False


def _ensure_worker_poller():
    global _worker_poller_started
    if _worker_poller_started:
        return
    _worker_poller_started = True

    def loop():
        while True:
            try:
                for jd in sorted(UPLOADS.glob("*/state.json")):
                    st = json.loads(jd.read_text())
                    if st.get("status") != "worker":
                        continue
                    url = st.get("worker_url", "")
                    tok = st.get("worker_token", "")
                    wj = st.get("worker_job", "")
                    if not (url and tok and wj):
                        continue
                    try:
                        _, j = _worker_api(url, f"/jobs/{wj}", tok)
                        if j.get("status") == "done":
                            _, blob = _worker_api(
                                url, f"/jobs/{wj}/results.zip", tok,
                                timeout=600)
                            import io as _io, tarfile
                            with tarfile.open(fileobj=_io.BytesIO(blob),
                                              mode="r:gz") as tf:
                                base = (jd.parent / "avail_results").resolve()
                                base.mkdir(parents=True, exist_ok=True)
                                for m in tf.getmembers():
                                    if m.name.startswith("avail_results/"):
                                        m.name = m.name[len("avail_results/"):]
                                    if not m.name:
                                        continue
                                    t = (base / m.name).resolve()
                                    if str(t).startswith(str(base)):
                                        tf.extract(m, base)
                            st.update(status="done", note="finished on your server",
                                      finished=datetime.now().isoformat(
                                          timespec="seconds"))
                        elif j.get("status") == "error":
                            st.update(status="error",
                                      error=j.get("error", "worker error"),
                                      note="your server reported an error")
                        else:
                            st["note"] = f"your server: {j.get('status')}"
                        jd.write_text(json.dumps(st))
                    except Exception as e:
                        st["note"] = f"worker unreachable: {type(e).__name__}"
                        jd.write_text(json.dumps(st))
            except Exception:
                pass
            time.sleep(20)

    threading.Thread(target=loop, daemon=True).start()


def _llm_error_text(e):
    t = str(e)
    low = t.lower()
    if "quota" in low or "rate" in low or "429" in low or "limit" in low:
        return ("AI providers are temporarily rate-limited (primary daily quota "
                "or fallback tokens-per-minute). Your data, results and jobs are "
                "unaffected. Please retry in a minute, or configure another "
                "fallback key.")
    return f"AI unavailable: {type(e).__name__}: {t[:180]}"


@app.route("/job/<job_id>/run-worker", methods=["POST"])
def run_worker(job_id):
    if not _can_access(job_id):
        abort(404)
    st = _read_state(job_id)
    if st.get("status") not in ("validated", "error", "done"):
        abort(409, "job cannot be started from this state")
    url = (request.form.get("worker_url") or "").strip()
    token = (request.form.get("worker_token") or "").strip()
    if not url.startswith(("http://", "https://")) or not token:
        abort(400, "worker URL (http/https) and token are required")
    if url.startswith("http://") and not url.startswith(
            ("http://127.0.0.1", "http://localhost")):
        print(f"[worker] warning: plain-http worker url {url}",
              flush=True)
    data_dir = _data_root_for(_job_dir(job_id) / "data")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        base = Path(data_dir)
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.stat().st_size < 500 * 1024 ** 2:
                zf.write(p, p.relative_to(base))
    flags = ",".join(f for f in ("robustness", "ci", "decision")
                     if st.get(f))
    try:
        code, resp = _worker_api(url, f"/jobs?flags={flags}&name={job_id}",
                                 token, method="POST", data=buf.getvalue(),
                                 timeout=300)
    except Exception as e:
        abort(502, f"worker connection failed: {type(e).__name__}: {e}")
    _write_state(job_id, status="worker", worker_url=url,
                 worker_token=token, worker_job=resp.get("id", ""),
                 note="queued on your own server")
    _ensure_worker_poller()
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/video-compat")
def video_compat_page():
    return _render("video_compat.html")


@app.route("/tour")
def tour_page():
    return _render("tour.html")


@app.route("/static/avail_video.mp4")
@app.route("/static/avail_video_2026.mp4")
def legacy_video_redirect():
    return redirect(url_for("video_page"))


@app.route("/video")
def video_page():
    return _render("video.html")


@app.route("/imaging")
def imaging_page():
    dd = BASE / "data_demos"
    datasets = []
    ftsv = dd / "fc_ds004392" / "fc_features.tsv"
    if ftsv.exists():
        z = dd / "fc_ds004392.zip"
        datasets.append({
            "name": "ds004392 · rs-fMRI connectivity features",
            "desc": ("50 subjects x 342 Schaefer-100 (17-network) connectivity "
                     "summaries from the rs-fMRI of ds004392 (CC0): full "
                     "Pearson (Fisher-z) and partial (Ledoit-Wolf) "
                     "within/between network features."),
            "size": f"{z.stat().st_size/1e3:.0f} KB" if z.exists() else "",
            "zip": "fc_ds004392.zip" if z.exists() else None,
            "raw": "fc_ds004392/fc_features.tsv",
            "meta": "fc_ds004392/metadata.tsv"})
    rtsv = dd / "radiomics_ds004392" / "radiomics.tsv"
    if rtsv.exists():
        z = dd / "radiomics_ds004392.zip"
        datasets.append({
            "name": "ds004392 · radiomics feature table",
            "desc": ("49 subjects x 386 PyRadiomics features from MNI T1w "
                     "volumes of OpenNeuro ds004392 (CC0): whole-brain + "
                     "network-group anatomical features and ALFF/fALFF/ReHo "
                     "texture. Ships with metadata (PD-NC / PD-MCI / PDD) "
                     "and a README."),
            "size": f"{z.stat().st_size/1e3:.0f} KB" if z.exists() else "",
            "zip": "radiomics_ds004392.zip" if z.exists() else None,
            "raw": "radiomics_ds004392/radiomics.tsv",
            "meta": "radiomics_ds004392/metadata.tsv"})
    ex = dd / "ds004392_excerpt.zip"
    if ex.exists():
        datasets.append({
            "name": "ds004392 · imaging excerpt (rs-fMRI + clinical)",
            "desc": ("Full excerpt: Schaefer connectivity features, "
                     "Hoehn & Yahr, cognitive domains and tests — the exact "
                     "dataset of the external-validation entry."),
            "size": f"{ex.stat().st_size/1e6:.1f} MB",
            "zip": "ds004392_excerpt.zip", "raw": None, "meta": None})
    else:
        datasets.append({
            "name": "ds004392 · imaging excerpt (rs-fMRI + clinical)",
            "desc": ("Being processed on the campaign server; the download "
                     "appears here automatically when complete (CC0)."),
            "size": "", "zip": None, "raw": None, "meta": None,
            "pending": True})
    import pandas as _pd

    def _preview(path, label):
        cols_all = list(_pd.read_csv(path, sep="\t", nrows=0).columns)
        df = _pd.read_csv(path, sep="\t", nrows=5)
        keep = [c for c in cols_all if c == "SampleID"][:1] + \
               [c for c in cols_all if c != "SampleID"][:7]
        with open(path) as f:
            nrows = sum(1 for _ in f) - 1
        return {"name": label, "shape": nrows, "ncols": len(cols_all),
                "cols": keep,
                "rows": df[keep].round(4).astype(str).values.tolist()}

    previews = []
    if ftsv.exists():
        previews.append(_preview(ftsv, "ds004392 fc_features.tsv"))
    if rtsv.exists():
        previews.append(_preview(rtsv, "ds004392 radiomics.tsv"))
    return _render("imaging.html", datasets=datasets, previews=previews)


IMG_DATA_ALLOW = {"radiomics_ds004392.zip", "ds004392_excerpt.zip",
                  "radiomics_ds004392/radiomics.tsv",
                  "radiomics_ds004392/metadata.tsv",
                  "radiomics_ds004392/README.txt",
                  "fc_ds004392.zip", "fc_ds004392/fc_features.tsv",
                  "fc_ds004392/metadata.tsv", "fc_ds004392/README.txt"}


@app.route("/imaging/data/<path:name>")
def imaging_data(name):
    if name not in IMG_DATA_ALLOW:
        abort(404)
    p = BASE / "data_demos" / name
    if not p.is_file():
        abort(404)
    return send_file(p, as_attachment=True)


def _imaging_context():
    """Grounding context for the imaging assistant: dataset facts + docs."""
    import pandas as _pd
    dd = BASE / "data_demos"
    parts = [
        "PIPELINE (imaging adapter): 1) AFNI preprocessing - T1 to MNI "
        "registration (robust LPC two-pass), EPI motion/band-pass cleanup "
        "0.01-0.10 Hz, 6 mm smoothing. 2) Connectivity: Schaefer-100 "
        "(17-network taxonomy; 18 named groups because SomMotB splits into "
        "auditory + rest) -> full Pearson (Fisher-z) and partial "
        "(Ledoit-Wolf) correlations -> within/between network summaries "
        "(171 + 171 = 342 features, keys full_* / part_*). 3) Anatomical "
        "radiomics: PyRadiomics on MNI T1 inside the Schaefer mask -> "
        "whole-brain (wb_*) and per-network (net_*) first-order/texture/"
        "shape features, bin width 25. 4) Functional radiomics: ALFF = "
        "stdev of the band-passed bold_clean, fALFF = ALFF/stdev(bold_blur), "
        "ReHo = 3dReHo (27-neighbourhood); maps rescaled to [0,1], bin "
        "width 0.05, coverage-masked (atlas AND map>0). 5) Assembly: one "
        "row per subject -> standard metadata.tsv + features/*.tsv layout. "
        "Two ways for rs-fMRI: template-space (MNI) BOLD is extracted "
        "automatically; raw native-space fMRI needs the documented AFNI "
        "registration step first. Worker: 'avail-omics worker' runs jobs on "
        "the user's own machine (bearer token, 127.0.0.1, sandboxed). Both feature tables below are downloadable from this page (zip = "
        "features + metadata + README). The pipeline is CPU-only; no GPU "
        "is required.",
    ]
    src = dd / "radiomics_ds004392/radiomics.tsv"
    fc = dd / "fc_ds004392/fc_features.tsv"
    for label, path in [("RADIOMICS dataset", src),
                        ("FC dataset", fc)]:
        if path.exists():
            cols = list(_pd.read_csv(path, sep="\t", nrows=0).columns)
            n = sum(1 for _ in open(path)) - 1
            wb = [c for c in cols if c.startswith(("wb_", "full_"))][:6]
            parts.append(
                f"{label}: {path.parent.name}/{path.name}, "
                f"{n} rows x {len(cols)-1} features; example columns: "
                f"{', '.join(wb)}")
            md = path.parent / "metadata.tsv"
            if md.exists():
                m = _pd.read_csv(md, sep="\t")
                if "Group" in m:
                    parts.append(f"  groups: {m.Group.value_counts().to_dict()}")
    for doc in ["IMAGING_ADAPTER.md", "WORKER.md"]:
        p = BASE / "docs" / doc
        if p.exists():
            parts.append(f"DOC {doc} (excerpt): " +
                         p.read_text()[:2200].replace("\n", " "))
    return "\n".join(parts)


@app.route("/api/imaging-ask", methods=["POST"])
def api_imaging_ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()[:1500]
    if not q:
        abort(400, "question required")
    key = session.get("_id") or request.remote_addr
    if _rate_limit_exceeded(key):
        return jsonify(answer="Rate limit reached (30 questions/hour). "
                              "Please try again later.")
    context = _imaging_context()
    try:
        from avail_ai.llm import MultiLLMClient, build_clients
        llm = MultiLLMClient(build_clients())
        ans = llm.complete(
            "You are Dr. Avail's imaging assistant for the AVAIL platform. "
            "Answer questions about the imaging pipeline, the radiomics and "
            "connectivity feature tables, and how to process one's own "
            "imaging data, using ONLY the provided context. Be concise and "
            "concrete; if the context does not cover something, say so "
            "briefly and point to WORKER.md / IMAGING_ADAPTER.md. "
            "Always reply in English. Write plain "
            "text only: no markdown asterisks, headers or tables.",
            f"Context:\n{context}\n\nQuestion: {q}", max_tokens=700)
        return jsonify(answer=ans.strip())
    except Exception as e:
        return jsonify(answer=_llm_error_text(e))


@app.route("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC_DIR, name)


def _job_dir(job_id):
    return UPLOADS / job_id


def _state_path(job_id):
    return _job_dir(job_id) / "state.json"


def _read_state(job_id):
    try:
        return json.loads(_state_path(job_id).read_text())
    except Exception:
        return {"status": "unknown"}


def _write_state(job_id, **kw):
    st = _read_state(job_id)
    st.update(kw)
    st["updated"] = datetime.now().isoformat(timespec="seconds")
    _state_path(job_id).write_text(json.dumps(st, indent=2))


def _can_access(job_id):
    """Session-scoped job isolation: demos are public; user jobs are owner-only."""
    st = _read_state(job_id)
    if st.get("status") == "unknown":
        return False
    if st.get("demo"):
        return True
    owner = st.get("owner")
    return bool(owner) and owner == session.get("_id")


def _detect_mode(data_root):
    """generic -> metadata.tsv + features/*.tsv; imaging -> DICOM/NIfTI;
    else the full study layout."""
    if (data_root / "metadata.tsv").exists() or (data_root / "metadata.csv").exists():
        feats = data_root / "features"
        if feats.exists() and any(f.suffix.lower() in (".tsv", ".csv")
                                  for f in feats.iterdir()):
            return "generic"
        if _has_imaging_files(data_root):
            return "imaging"
    if _has_imaging_files(data_root):
        return "imaging"
    return "avail"


def _has_imaging_files(data_root):
    for ext in ("*.dcm", "*.nii", "*.nii.gz", "*.ima"):
        for _ in Path(data_root).rglob(ext):
            return True
    return False


def _validate_layout(data_root):
    """Return (ok, message)."""
    mode = _detect_mode(data_root)
    if mode == "generic":
        return True, "generic layout detected (metadata + features/)"
    if mode == "imaging":
        if not ((data_root / "metadata.tsv").exists() or
                (data_root / "metadata.csv").exists()):
            return False, ("imaging uploads need a metadata.tsv/csv "
                           "(SampleID + Group) next to the DICOM/NIfTI files")
        return True, "imaging layout detected (metadata + DICOM/NIfTI)"
    hits = 0
    for hint in REQUIRED_HINTS:
        if any(p.name == hint or hint in p.name for p in data_root.iterdir() if p.is_dir()):
            hits += 1
    if hits == 0:
        return False, ("layout not recognized: expected either "
                       "'metadata.tsv + features/' (generic layout) or the "
                       "full study layout (see the format notes on the upload page)")
    return True, f"AVAIL layout detected ({hits}/3 markers found)"


def _safe_extract(zip_path, dest):
    """Extract with zip-slip protection. Returns list of top-level names."""
    with zipfile.ZipFile(zip_path) as zf:
        cap = max(5 * MAX_MB, 10240) * 1024 * 1024
        total = 0
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve()) + os.sep):
                raise ValueError(f"unsafe path in zip: {member.filename}")
            total += member.file_size
        if total > cap:
            raise ValueError(
                f"zip expands to {total/1e9:.1f} GB which exceeds the "
                f"{cap/1e9:.1f} GB safety cap; use the local runner for "
                f"large imaging cohorts")
        zf.extractall(dest)
    return [p.name for p in dest.iterdir()]


def _data_root_for(job_dir):
    """If the zip contains a single top-level folder, use it as the data root."""
    tops = [p for p in job_dir.iterdir()]
    if len(tops) == 1 and tops[0].is_dir():
        return tops[0]
    return job_dir


def _check_min_subjects(data_root, lf):
    """Friendly pre-flight check: the pipeline needs >=2 groups with >=3
    subjects each (internal cross-validation)."""
    meta_path = data_root / "metadata.tsv"
    if not meta_path.exists():
        meta_path = data_root / "metadata.csv"
    if not meta_path.exists():
        return
    try:
        sep = "\t" if str(meta_path).endswith(".tsv") else ","
        meta = pd.read_csv(meta_path, sep=sep)
        grp_col = "Group" if "Group" in meta.columns else meta.columns[1]
        cnt = meta[grp_col].astype(str).value_counts()
        small = [f"{k}={v}" for k, v in cnt.items() if v < 3]
        if len(cnt) < 2:
            raise RuntimeError(
                "the pipeline needs at least 2 groups for classification; "
                f"your metadata has only: {dict(cnt)}")
        if small:
            raise RuntimeError(
                "the pipeline needs at least 3 subjects per group "
                f"(internal cross-validation); too small: {small}")
    except Exception as e:
        lf.write(f"[validation] {e}\n")
        raise


def _run_imaging(data_root, lf):
    """DICOM/NIfTI -> ROI features (classical FC kept as the main layer;
    TimesFM-3 features are stashed for the optional benchmark)."""
    imp = BASE / "webapp" / "imaging.py"
    proc = subprocess.run(
        [PY, "-u", str(imp), str(data_root), str(data_root / "features")],
        cwd=str(BASE), stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        tf = data_root / "features" / "imaging_timesfm.tsv"
        if tf.exists():
            tf_dir = data_root / "features_timesfm"
            tf_dir.mkdir(exist_ok=True)
            shutil.move(str(tf), str(tf_dir / "imaging_timesfm.tsv"))
    return proc


def _run_timesfm_benchmark(data_root, jd, lf):
    """Second pipeline run using ONLY the TimesFM-3 representation layer."""
    root2 = jd / "data_timesfm"
    root2.mkdir(exist_ok=True)
    src_meta = data_root / "metadata.tsv"
    if not src_meta.exists():
        src_meta = data_root / "metadata.csv"
    if src_meta.exists():
        shutil.copy(str(src_meta), str(root2 / src_meta.name))
    tf_dir = data_root / "features_timesfm"
    if tf_dir.exists():
        dest = root2 / "features"
        dest.mkdir(exist_ok=True)
        for f in tf_dir.iterdir():
            if f.is_file():
                shutil.copy(str(f), str(dest / f.name))
        cmd = [PY, "-u", str(BASE / "run_generic.py"),
               "--data-root", str(root2), "--out", str(jd / "avail_results_timesfm")]
        subprocess.run(cmd, cwd=str(BASE), stdout=lf, stderr=subprocess.STDOUT)


def _worker():
    while True:
        job_id = job_queue.get()
        jd = _job_dir(job_id)
        try:
            _write_state(job_id, status="running",
                         started=datetime.now().isoformat(timespec="seconds"))
            data_root = _data_root_for(jd / "data")
            mode = _detect_mode(data_root)
            _write_state(job_id, mode=mode)
            out = jd / "avail_results"
            log_path = jd / "run.log"
            with open(log_path, "w") as lf:
                if mode == "imaging":
                    proc = _run_imaging(data_root, lf)
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"imaging preprocessing failed (exit "
                            f"{proc.returncode})")
                    if _read_state(job_id).get("radiomics"):
                        rp = Path(RADIOMICS_PY)
                        lf.write("\n=== RADIOMICS (optional block) ===\n")
                        if rp.exists():
                            subprocess.run(
                                [str(rp), "-m", "avail_omics.cli",
                                 "radiomics", "--data-root", str(data_root),
                                 "--out", str(data_root / "features"),
                                 "--mode", "wb"],
                                cwd=str(BASE), stdout=lf,
                                stderr=subprocess.STDOUT)
                        else:
                            lf.write("radiomics tier not installed on this "
                                     "server (pip install "
                                     "'avail-omics[radiomics]')\n")
                _check_min_subjects(data_root, lf)
                runner = "run_generic.py" if mode in ("generic", "imaging") \
                    else "run_pipeline.py"
                cmd = [PY, "-u", str(BASE / runner), "--data-root",
                       str(data_root), "--out", str(out)]
                env = dict(os.environ)
                if _read_state(job_id).get("demo"):
                    env["AVAIL_META_MODELS"] = (
                        "SimpleMean,WSimpleMean,ElasticNetLR,XGBoost")
                    lf.write("[demo] fast mode: 4 meta-learners "
                             "(advanced comparisons still available)\n")
                proc = subprocess.run(cmd, cwd=str(BASE), env=env,
                                      stdout=lf, stderr=subprocess.STDOUT)
                if proc.returncode == 0 and mode == "imaging" and \
                        _read_state(job_id).get("timesfm"):
                    lf.write("\n=== TIMESFM-3 REPRESENTATION BENCHMARK ===\n")
                    _run_timesfm_benchmark(data_root, jd, lf)
            if proc.returncode == 0:
                _write_state(job_id, status="done",
                             finished=datetime.now().isoformat(timespec="seconds"))
                # optional comparison suite (CPU, background)
                st = _read_state(job_id)
                if mode == "generic" and st.get("robustness"):
                    _write_state(job_id, note="running missing-data comparison suite")
                    imp = BASE / "experiments/imputation_benchmark_v4.py"
                    if imp.exists():
                        subprocess.Popen(
                            [PY, "-u", str(imp), "--data-root", str(data_root),
                             "--out", str(jd / "avail_comparisons")],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("ci"):
                    ci = BASE / "experiments/robustness_ci.py"
                    if ci.exists():
                        subprocess.Popen(
                            [PY, "-u", str(ci), str(out),
                             "--meta-root", str(data_root)],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("missingness"):
                    mr = BASE / "experiments/missingness_recovery.py"
                    if mr.exists():
                        subprocess.Popen(
                            [PY, "-u", str(mr), "--data-root", str(data_root),
                             "--out", str(jd / "avail_comparisons")],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("tuning"):
                    tn = BASE / "experiments/tuning.py"
                    if tn.exists():
                        subprocess.Popen(
                            [PY, "-u", str(tn), "--data-root", str(data_root),
                             "--out", str(jd / "avail_comparisons"),
                             "--trials", "30"],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                cmp_dir = jd / "avail_comparisons"
                cmp_dir.mkdir(exist_ok=True)
                if st.get("calibration"):
                    cb = BASE / "experiments/calibration_experiment.py"
                    if cb.exists():
                        subprocess.Popen(
                            [PY, "-u", str(cb), "--data-root", str(data_root),
                             "--out", str(cmp_dir)],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("ordinal"):
                    od = BASE / "experiments/ordinal_experiment.py"
                    if od.exists():
                        subprocess.Popen(
                            [PY, "-u", str(od),
                             "--base-preds", str(out / "base_predictions_generic.tsv"),
                             "--loo", str(out / "loo_predictions.tsv"),
                             "--out", str(cmp_dir)],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("conformal"):
                    cf = BASE / "experiments/conformal_experiment.py"
                    if cf.exists():
                        subprocess.Popen(
                            [PY, "-u", str(cf), "--res-dir", str(out),
                             "--out", str(cmp_dir)],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("hybrid"):
                    hy = BASE / "experiments/hybrid_comparison.py"
                    if hy.exists():
                        subprocess.Popen(
                            [PY, "-u", str(hy), "--data-root", str(data_root),
                             "--out", str(cmp_dir)],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
                if st.get("decision"):
                    de = BASE / "experiments/decision_experiment.py"
                    if de.exists():
                        subprocess.Popen(
                            [PY, "-u", str(de),
                             "--loo", str(out / "loo_predictions.tsv"),
                             "--out", str(cmp_dir)],
                            cwd=str(BASE),
                            stdout=open(jd / "comparisons.log", "a"),
                            stderr=subprocess.STDOUT)
            else:
                _write_state(job_id, status="error", returncode=proc.returncode,
                              finished=datetime.now().isoformat(timespec="seconds"))
        except Exception as e:
            _write_state(job_id, status="error",
                         error=f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            job_queue.task_done()


# ---------------------------------------------------------------- helpers (github, pubmed-free, validation)
def _public_url():
    """Reviewer-facing public URL (explicit override, Render, else tunnel)."""
    for key in ("AVAIL_PUBLIC_URL", "RENDER_EXTERNAL_URL"):
        env = os.environ.get(key, "").strip().rstrip("/")
        if env:
            return env
    try:
        txt = TUNNEL_LOG.read_text(errors="replace")
        urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
        return urls[-1] if urls else None
    except Exception:
        return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Disable automatic redirects so we can drop the Authorization header
    on the cross-host hop to Azure blob storage."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIR_OPENER = urllib.request.build_opener(_NoRedirect)


def _gh(method, path, data=None):
    """Minimal GitHub API helper (Bearer token)."""
    if not GH_TOKEN or not GH_REPO:
        return {"_error": "GitHub not configured (AVAIL_GH_TOKEN/AVAIL_GH_REPO)"}
    req = urllib.request.Request(
        f"https://api.github.com{path}", method=method,
        headers={"Authorization": f"Bearer {GH_TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "avail-web"})
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, body, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt.strip() else {}
    except urllib.error.HTTPError as e:
        return {"_error": e.code, "_body": e.read().decode()[:300]}
    except Exception as e:
        return {"_error": str(e)}


def _gh_find_run(job_id):
    r = _gh("GET", f"/repos/{GH_REPO}/actions/runs?event=workflow_dispatch&per_page=30")
    if isinstance(r, dict) and "_error" in r:
        return None
    for run in r.get("workflow_runs", []):
        if run.get("name") == job_id:
            return run
    return None


def _gh_download(url):
    """GET with Bearer auth; follows redirects WITHOUT the auth header
    (Azure blob storage rejects SAS URLs that carry an Authorization header)."""
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {GH_TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "avail-web"})
    try:
        with _NO_REDIR_OPENER.open(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307):
            loc = e.headers.get("Location")
            if loc:
                req2 = urllib.request.Request(loc, headers={"User-Agent": "avail-web"})
                with urllib.request.urlopen(req2, timeout=180) as r2:
                    return r2.read()
        return {"_error": e.code, "_body": e.read().decode()[:300]}
    except Exception as e:
        return {"_error": str(e)}


def _gh_download_artifact(run_id, job_dir):
    r = _gh("GET", f"/repos/{GH_REPO}/actions/runs/{run_id}/artifacts")
    if isinstance(r, dict) and "_error" in r:
        return False
    arts = r.get("artifacts", [])
    for a in arts:
        if a.get("name") == "results":
            data = _gh_download(
                f"https://api.github.com/repos/{GH_REPO}/actions/artifacts/{a['id']}/zip")
            if isinstance(data, bytes):
                zip_path = job_dir / "results_cloud.zip"
                zip_path.write_bytes(data)
                res_dir = job_dir / "avail_results"
                res_dir.mkdir(parents=True, exist_ok=True)
                tmp = job_dir / "artifact_tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True, exist_ok=True)
                _safe_extract(zip_path, tmp)
                zip_path.unlink(missing_ok=True)
                src = tmp
                for cand in ["results", "avail_results"]:
                    if (tmp / cand).is_dir():
                        src = tmp / cand
                        break
                for p in sorted(src.rglob("*")):
                    if p.is_file():
                        shutil.move(str(p), str(res_dir / p.relative_to(src)))
                shutil.rmtree(tmp, ignore_errors=True)
                return True
    return False


def _gh_poller():
    while True:
        time.sleep(60)
        if not GH_TOKEN or not UPLOADS.exists():
            continue
        for d in sorted(UPLOADS.iterdir()):
            if not d.is_dir():
                continue
            st = _read_state(d.name)
            if st.get("status") != "cloud":
                continue
            try:
                run = _gh_find_run(d.name)
                if run is None:
                    continue
                if run.get("status") in ("queued", "in_progress", "waiting"):
                    continue
                concl = run.get("conclusion")
                if concl == "success":
                    ok = _gh_download_artifact(run["id"], d)
                    if ok:
                        _write_state(d.name, status="done",
                                     note="completed in free cloud (GitHub Actions)")
                        llm_runner = BASE / "run_llm_advisor.py"
                        if llm_runner.exists():
                            subprocess.Popen(
                                [PY, str(llm_runner), "--out",
                                 str(d / "avail_results")], cwd=str(BASE),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
                    else:
                        _write_state(d.name, status="error",
                                     error="cloud run finished but artifact download failed")
                else:
                    _write_state(d.name, status="error", returncode=1,
                                 note=f"cloud run concluded: {concl}")
            except Exception as e:
                _write_state(d.name, status="error",
                             error=f"cloud poll: {type(e).__name__}: {str(e)[:200]}")


def _dl_secret(job_id):
    st = _read_state(job_id)
    sec = st.get("dl_secret")
    if sec:
        return sec
    sec = secrets.token_hex(8)
    _write_state(job_id, dl_secret=sec)
    return sec


PALETTE = ["#2E5A87", "#5B9279", "#3D2E5F", "#C77D9D", "#D9A441",
           "#E76F51", "#8A7FB8", "#4B7BA6"]


def _mod_color(name):
    return PALETTE[sum(ord(c) for c in name) % len(PALETTE)]


def _read_tab(path):
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _summarize(data_root, mode):
    """Validation summary of a freshly uploaded dataset (no model runs)."""
    s = {"mode": mode, "classes": {}, "n": 0, "modalities": [], "warnings": []}
    if mode == "imaging":
        meta_path = data_root / "metadata.tsv"
        if not meta_path.exists():
            meta_path = data_root / "metadata.csv"
        try:
            sep = "\t" if str(meta_path).endswith(".tsv") else ","
            meta = pd.read_csv(meta_path, sep=sep)
            id_col = "SampleID" if "SampleID" in meta.columns else meta.columns[0]
            grp_col = "Group" if "Group" in meta.columns else meta.columns[1]
            s["n"] = int(len(meta))
            s["classes"] = {str(k): int(v) for k, v in
                            meta[grp_col].astype(str).value_counts().items()}
            s["modalities"].append({"name": "brain_imaging",
                                    "n_samples": s["n"],
                                    "n_features": 5203, "coverage": 100,
                                    "color": _mod_color("brain_imaging")})
        except Exception:
            s["warnings"].append("could not parse imaging metadata")
        return s
    if mode != "generic":
        s["warnings"].append(
            "AVAIL layout detected: the full oral-gut-brain analysis will run "
            "(generic validation summary not applicable).")
        return s
    meta_path = data_root / "metadata.tsv"
    if not meta_path.exists():
        meta_path = data_root / "metadata.csv"
    meta = _read_tab(meta_path)
    if meta.shape[1] < 2:
        s["warnings"].append("metadata needs at least two columns "
                             "(sample id + group)")
        return s
    id_col = "SampleID" if "SampleID" in meta.columns else meta.columns[0]
    group_col = "Group" if "Group" in meta.columns else meta.columns[1]
    meta[id_col] = meta[id_col].astype(str).str.strip()
    meta[group_col] = meta[group_col].astype(str).str.strip()
    ids = set(meta[id_col])
    dups = meta[id_col][meta[id_col].duplicated()]
    if len(dups):
        s["warnings"].append(
            f"{len(dups)} duplicate SampleID(s) in metadata "
            f"(e.g. {dups.iloc[0]}) - later rows override earlier ones.")
    counts = meta[group_col].value_counts().to_dict()
    s["n"] = int(len(meta))
    s["classes"] = {str(k): int(v) for k, v in counts.items()}

    if len(counts) > 3:
        s["warnings"].append(
            f"{len(counts)} classes found; AVAIL keeps the 3 most frequent "
            f"({', '.join(list(counts)[:3])}). Others are dropped.")
    if len(counts) < 2:
        s["warnings"].append("Only one class found  -  classification is not possible.")
    elif max(counts.values()) / max(1, min(counts.values())) > 3:
        s["warnings"].append(
            "Strong class imbalance (ratio > 3:1)  -  consider class weights or "
            "more samples for the minority class.")
    if s["n"] < 30:
        s["warnings"].append(
            "Small sample size (n < 30): expect noisy AUC estimates; treat results "
            "as exploratory.")

    feats_dir = data_root / "features"
    for f in sorted(feats_dir.iterdir()):
        if f.suffix.lower() not in (".tsv", ".csv"):
            continue
        try:
            df = _read_tab(f)
        except Exception as e:
            s["warnings"].append(f"{f.name} could not be parsed: {e}")
            continue
        if df.shape[1] < 2:
            s["warnings"].append(f"{f.name} has no feature columns  -  skipped.")
            continue
        df[df.columns[0]] = df[df.columns[0]].astype(str).str.strip()
        try:
            num = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            nonnum = int(num.isna().sum().sum()) - int(df.iloc[:, 1:].isna().sum().sum())
            if nonnum > 0:
                s["warnings"].append(
                    f"{f.stem}: {nonnum} non-numeric feature cell(s) "
                    "(empty or unparseable, e.g. decimal commas) - coerced to 0.")
            nan_frac = num.isna().mean()
            bad = nan_frac[nan_frac > 0.5]
            if len(bad):
                s["warnings"].append(
                    f"{f.stem}: {len(bad)} feature(s) >50% missing "
                    f"(e.g. {', '.join(map(str, bad.index[:3]))}) - mostly imputed.")
            zero_var = [c for c in num.columns if num[c].dropna().nunique() <= 1]
            if zero_var:
                s["warnings"].append(
                    f"{f.stem}: {len(zero_var)} zero-variance feature(s) "
                    f"(e.g. {', '.join(map(str, zero_var[:3]))}) - carry no signal.")
        except Exception:
            pass
        overlap = set(df[df.columns[0]]) & ids
        cov = len(overlap) / max(1, s["n"])
        entry = {
            "name": f.stem,
            "n_samples": int(len(overlap)),
            "n_features": int(df.shape[1] - 1),
            "coverage": round(cov * 100),
            "color": _mod_color(f.stem),
        }
        s["modalities"].append(entry)
        if cov < 0.5:
            s["warnings"].append(
                f"{f.stem} covers only {len(overlap)}/{s['n']} subjects (<50%)  -  "
                "availability flags will handle it, but signal is limited.")
    if not s["modalities"]:
        s["warnings"].append("features/ is empty  -  add at least one modality file.")
    return s


def _ai_feedback(summary):
    """LLM data-quality feedback on the validation summary (best effort)."""
    try:
        from avail_ai.llm import MultiLLMClient, build_clients, load_api_key
        key = load_api_key()
        if not key:
            return None
        llm = MultiLLMClient(build_clients())
        user = ("Dataset validation summary:\n" + json.dumps(summary, indent=2) +
                "\n\nGive concise, practical data-quality feedback (max 4 bullet "
                "points) for running a missing-modality multimodal classifier on "
                "this data. Plain English.")
        return llm.complete("You are a rigorous multimodal ML data-quality reviewer.",
                            user).strip()
    except Exception as e:
        return f"AI feedback unavailable: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------- templates
from markupsafe import Markup

try:
    from webapp.icons import render as _icon_render
except ImportError:
    def _icon_render(name, cls=""):
        return ""


def _icon(name, cls=""):
    return Markup(_icon_render(name, cls))


def _plain_text(s):
    import re as _re
    s = s or ""
    s = _re.sub(r"\*\*(.*?)\*\*", r"\1", s, flags=_re.S)
    s = _re.sub(r"`([^`]*)`", r"\1", s)
    s = _re.sub(r"^\s*#{1,6}\s*", "", s, flags=_re.M)
    s = _re.sub(r"\$([^$]+)\$", r"\1", s)
    s = _re.sub(r"^\s*\*\s+", "• ", s, flags=_re.M)
    return s.replace("**", "")


TPL = Environment(loader=FileSystemLoader(str(BASE / 'webapp' / 'templates')),
                  autoescape=select_autoescape(['html']))
TPL.globals['url_for'] = url_for
TPL.globals['icon'] = _icon
TPL.filters['plain'] = _plain_text


def _render(name, **kw):
    return TPL.get_template(name).render(**kw)


def _list_jobs():
    out = []
    sid = session.get("_id")
    if UPLOADS.exists():
        for d in sorted(UPLOADS.iterdir(), reverse=True):
            if d.is_dir():
                st = _read_state(d.name)
                if not (st.get("demo") or (st.get("owner") and st.get("owner") == sid)):
                    continue
                out.append({"id": d.name,
                            "status": st.get("status", "unknown"),
                            "created": st.get("created", ""),
                            "mode": st.get("mode", "-")})
    return out


@app.route("/")
def index():
    demo_bundle = (BASE / "data_demos" / "ds004392_excerpt.zip").exists()
    demo_own = (BASE / "data_demos" / "own_cohort_excerpt.zip").exists()
    return _render("index.html", jobs=_list_jobs(), max_mb=MAX_MB,
                   public_url=_public_url(), demo_bundle=demo_bundle,
                   demo_own=demo_own, server_run=SERVER_RUN,
                   demo_only=DEMO_ONLY, contact=CONTACT_EMAIL)


@app.route("/template.zip")
def template_zip():
    """Synthetic multi-omics example dataset (generic layout) for demo purposes."""
    import numpy as np

    buf = io.BytesIO()
    rng = np.random.RandomState(42)
    n = 60
    classes = ["HC"] * 30 + ["MCI"] * 15 + ["PDD"] * 15
    rng.shuffle(classes)
    ids = [f"SUBJ{i:03d}" for i in range(1, n + 1)]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = ["SampleID\tGroup\tAge\tEducation\tSex"]
        for i, sid in enumerate(ids):
            g = classes[i]
            age = int(rng.uniform(48, 82))
            edu = int(rng.uniform(5, 20))
            sex = "M" if i % 2 else "F"
            lines.append(f"{sid}\t{g}\t{age}\t{edu}\t{sex}")
        zf.writestr("example_omics/metadata.tsv", "\n".join(lines))

        layers = [
            ("brain_connectivity", 120, 0.85, 0.0),
            ("stool_16S_genus", 150, 0.95, 1.0),
            ("saliva_shotgun_MGS", 200, 0.75, 1.0),
            ("plasma_metabolomics", 80, 0.55, 0.0),
        ]
        for mname, nf, coverage, comp in layers:
            feat_lines = ["SampleID"] + [f"{mname}_{j}" for j in range(1, nf + 1)]
            rows = [feat_lines]
            for i, sid in enumerate(ids):
                if rng.rand() > coverage:
                    continue
                cls = {"HC": 0, "MCI": 1, "PDD": 2}[classes[i]]
                base = np.array([cls] * nf, dtype=float)
                vec = base * 0.9 + rng.normal(0, 1.6, nf)
                if comp:
                    vec = np.exp(vec)
                rows.append([sid] + [f"{v:.4f}" for v in vec])
            zf.writestr(f"example_omics/features/{mname}.tsv",
                        "\n".join("\t".join(r) for r in rows))
        zf.writestr("example_omics/compositional.txt",
                    "stool_16S_genus\nsaliva_shotgun_MGS\n")
        zf.writestr(
            "example_omics/README.txt",
            "SYNTHETIC example dataset - for demonstration only.\n"
            "Feature NAMES are real (Schaefer network edges, gut genera, "
            "oral species, plasma metabolites); all VALUES are synthetic.\n"
            "Metric values from this data are NOT scientific.\n\n"
            "AVAIL generic layout example.\n\n"
            "* metadata.tsv: SampleID, Group, optional Age/Sex/Education.\n"
            "* features/*.tsv: one file per modality, first column SampleID;\n"
            "  availability is read from which subjects appear.\n"
            "* compositional.txt (optional): one modality name per line for\n"
            "  compositional modalities (relative abundances / counts);\n"
            "  these receive a log/CLR transform. Unlisted modalities are\n"
            "  treated as continuous measurements. Every classification is\n"
            "  logged by the pipeline; disagreement with the heuristic is\n"
            "  reported, never silent.\n")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="avail_example_omics.zip",
                     mimetype="application/zip")


@app.route("/demo", methods=["POST"])
def demo():
    """One-click demo: create a job from the built-in example dataset and
    start it directly (server GPU if enabled, else free cloud fast)."""
    fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    ip = fwd or request.remote_addr or "unknown"
    if _rate_limit_exceeded("demo:" + ip, limit=10):
        return _render("error.html", code=429, title="Demo limit reached",
                       message="Too many demo runs from this network. "
                               "Please try again in an hour."), 429
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    jd = _job_dir(job_id)
    data_dir = jd / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # build the same example zip server-side
    from flask import redirect as _red
    import numpy as np
    rng = np.random.RandomState(42)
    n = 60
    classes = ["HC"] * 30 + ["MCI"] * 15 + ["PDD"] * 15
    rng.shuffle(classes)
    ids = [f"SUBJ{i:03d}" for i in range(1, n + 1)]
    zip_path = jd / "demo.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = ["SampleID\tGroup\tAge\tEducation\tSex"]
        for i, sid in enumerate(ids):
            lines.append(f"{sid}\t{classes[i]}\t{int(rng.uniform(48, 82))}\t"
                         f"{int(rng.uniform(5, 20))}\t{'M' if i % 2 else 'F'}")
        zf.writestr("example_omics/metadata.tsv", "\n".join(lines))
        # real feature names (values are still synthetic)
        try:
            _fc = BASE / "data_demos" / "fc_ds004392" / "fc_features.tsv"
            _cols = pd.read_csv(_fc, sep="\t", nrows=0).columns.tolist()[1:]
            brain_names = [c for c in _cols
                           if c.startswith(("full_", "part_"))][:60]
        except Exception:
            brain_names = []
        if not brain_names:
            _nets = ["DMN", "FPN", "DAN", "VAN", "SMN", "VIS", "LIM", "CON"]
            brain_names = [f"full_{a}-{b}" for i, a in enumerate(_nets)
                           for b in _nets[i:]][:60]
        stool_genera = [
            "Faecalibacterium", "Roseburia", "Lachnospira", "Agathobacter",
            "Fusicatenibacter", "Bifidobacterium", "Blautia", "Oscillibacter",
            "Alistipes", "Bacteroides", "Prevotella", "Parabacteroides",
            "Eubacterium", "Coprococcus", "Ruminococcus", "Subdoligranulum",
            "Akkermansia", "Escherichia", "Klebsiella", "Enterococcus",
            "Lactobacillus", "Streptococcus", "Veillonella", "Dorea",
            "Anaerostipes", "Butyricicoccus", "Phascolarctobacterium",
            "Sutterella", "Acidaminococcus", "Barnesiella", "Odoribacter",
            "Butyricimonas", "Desulfovibrio", "Methanobrevibacter", "Bilophila",
            "Eggerthella", "Collinsella", "Slackia", "Adlercreutzia",
            "Gordonibacter"]
        saliva_species = [
            "Streptococcus salivarius", "Streptococcus mutans",
            "Prevotella melaninogenica", "Veillonella parvula",
            "Rothia mucilaginosa", "Actinomyces naeslundii",
            "Neisseria subflava", "Haemophilus parainfluenzae",
            "Porphyromonas gingivalis", "Fusobacterium nucleatum",
            "Capnocytophaga ochracea", "Tannerella forsythia",
            "Treponema denticola", "Campylobacter rectus",
            "Eikenella corrodens", "Aggregatibacter actinomycetemcomitans",
            "Lactobacillus salivarius", "Staphylococcus epidermidis",
            "Corynebacterium durum", "Kingella oralis",
            "Leptotrichia buccalis", "Selenomonas noxia",
            "Prevotella intermedia", "Prevotella nigrescens",
            "Streptococcus sanguinis", "Streptococcus gordonii",
            "Rothia dentocariosa", "Bifidobacterium dentium",
            "Scardovia wiggsiae", "Olsenella uli"]
        plasma_metabolites = [
            "butyrate", "acetate", "propionate", "isobutyrate", "valerate",
            "TMAO", "indole-3-acetate", "hippurate",
            "phenylacetylglutamine", "valine", "leucine", "isoleucine",
            "carnitine", "acetylcarnitine", "choline", "betaine", "taurine",
            "glutamine", "glutamate", "serine", "glycine", "alanine",
            "lactate", "pyruvate", "citrate", "succinate", "fumarate",
            "malate", "alpha-ketoglutarate", "creatinine"]
        for mname, _names, coverage, comp in [
                ("brain_connectivity", brain_names, 0.85, 0.0),
                ("stool_16S_genus", stool_genera, 0.95, 1.0),
                ("saliva_shotgun_MGS", saliva_species, 0.75, 1.0),
                ("plasma_metabolomics", plasma_metabolites, 0.55, 0.0)]:
            nf = len(_names)
            feat_lines = ["SampleID"] + list(_names)
            rows = [feat_lines]
            for i, sid in enumerate(ids):
                if rng.rand() > coverage:
                    continue
                cls = {"HC": 0, "MCI": 1, "PDD": 2}[classes[i]]
                vec = np.array([cls] * nf, dtype=float) * 0.9 + \
                    rng.normal(0, 1.6, nf)
                if comp:
                    vec = np.exp(vec)
                rows.append([sid] + [f"{v:.4f}" for v in vec])
            zf.writestr(f"example_omics/features/{mname}.tsv",
                        "\n".join("\t".join(r) for r in rows))
        zf.writestr("example_omics/compositional.txt",
                    "stool_16S_genus\nsaliva_shotgun_MGS\n")
        zf.writestr(
            "example_omics/README.txt",
            "SYNTHETIC example dataset - for demonstration only.\n"
            "Metric values from this data are NOT scientific.\n\n"
            "AVAIL generic layout example.\n\n"
            "* metadata.tsv: SampleID, Group, optional Age/Sex/Education.\n"
            "* features/*.tsv: one file per modality, first column SampleID;\n"
            "  availability is read from which subjects appear.\n"
            "* compositional.txt (optional): one modality name per line for\n"
            "  compositional modalities (relative abundances / counts);\n"
            "  these receive a log/CLR transform. Unlisted modalities are\n"
            "  treated as continuous measurements.\n")
    demo_set = (request.form.get("demo_set") or "").strip()
    if not demo_set:
        demo_set = ("own_cohort"
                    if (BASE / "data_demos" / "own_cohort_excerpt.zip").exists()
                    else "synthetic")
    if demo_set == "ds004392":
        bundle = BASE / "data_demos" / "ds004392_excerpt.zip"
        if not bundle.exists():
            abort(404, "real-cohort demo bundle is not available yet")
        zip_path.write_bytes(bundle.read_bytes())
    elif demo_set == "own_cohort":
        bundle = BASE / "data_demos" / "own_cohort_excerpt.zip"
        if not bundle.exists():
            abort(404, "own-cohort excerpt is not available")
        zip_path.write_bytes(bundle.read_bytes())
    _safe_extract(zip_path, data_dir)
    zip_path.unlink(missing_ok=True)
    data_root = _data_root_for(data_dir)
    if PUBLIC_UPLOADS:
        issues = _public_upload_issues(data_root)
        if issues:
            shutil.rmtree(jd, ignore_errors=True)
            abort(400, "Public-upload check failed: " + "; ".join(issues))
    mode = _detect_mode(data_root)
    summary = _summarize(data_root, mode)
    feedback = _ai_feedback(summary)
    (jd / "validation.json").write_text(json.dumps(summary, indent=2))
    (jd / "ai_feedback.txt").write_text(feedback or "")
    _write_state(job_id, status="validated", created=job_id.replace("_", " "),
                 owner=session.get("_id"),
                 layout_msg=("real public cohort (OpenNeuro ds004392, CC0)"
                             if demo_set == "ds004392" else
                             ("our own cohort (de-identified excerpt)"
                              if demo_set == "own_cohort"
                              else "built-in example dataset")),
                 mode=mode, demo=True,
                 robustness=request.form.get("robustness") == "1",
                 ci=request.form.get("ci") == "1",
                 missingness=request.form.get("missingness") == "1",
                 tuning=request.form.get("tuning") == "1",
                 calibration=request.form.get("calibration") == "1",
                 ordinal=request.form.get("ordinal") == "1",
                 conformal=request.form.get("conformal") == "1",
                 hybrid=request.form.get("hybrid") == "1",
                 decision=request.form.get("decision") == "1",
                 radiomics=request.form.get("radiomics") == "1",
                 timesfm=request.form.get("timesfm") == "1")

    # start directly (demo is always allowed: fixed built-in data, bounded run)
    if SERVER_RUN or DEMO_RUN:
        _write_state(job_id, status="queued", note="demo run (server)")
        job_queue.put(job_id)
    elif GH_TOKEN and GH_REPO:
        base = _public_url()
        data_url = f"{base}/dl/{job_id}/{_dl_secret(job_id)}/data.zip"
        r = _gh("POST",
                f"/repos/{GH_REPO}/actions/workflows/{GH_WORKFLOW}/dispatches",
                {"ref": GH_BRANCH,
                 "inputs": {"job_id": job_id, "data_url": data_url,
                            "full": "false"}})
        if isinstance(r, dict) and "_error" in r:
            _write_state(job_id, status="validated",
                         note="cloud backend unavailable - confirm manually")
        else:
            _write_state(job_id, status="cloud",
                         note="demo run (free cloud)")
    return redirect(url_for("job_page", job_id=job_id))


def _public_upload_issues(data_root):
    """Rule-based guard for public uploads: strict metadata schema plus
    identifier patterns. It reduces risk but is not a legal anonymisation
    guarantee."""
    issues = []
    meta = data_root / "metadata.tsv"
    if not meta.exists():
        return ["metadata.tsv is required"]
    try:
        df = pd.read_csv(meta, sep="\t", dtype=str)
    except Exception as e:
        return [f"metadata.tsv could not be read: {type(e).__name__}"]
    allowed = {"sampleid", "group", "age", "sex", "education"}
    bad_words = {"name", "patient", "mrn", "identity", "address", "phone",
                 "email", "birth", "dob", "tckn", "tc_kimlik"}

    def looks_like_identifier(col):
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(col)).lower()
        return any(w in bad_words for w in re.findall(r"[a-z]+", s))

    for c in df.columns:
        lc = str(c).strip().lower()
        if lc not in allowed:
            issues.append(f"metadata column '{c}' is not allowed on the public "
                          "instance; use SampleID/Group/Age/Sex/Education only")
        if looks_like_identifier(c):
            issues.append(f"metadata column '{c}' looks like an identifier")
    for c in df.columns:
        for v in df[c].dropna().astype(str).head(5000):
            s = v.strip()
            if str(c).strip().lower() == "sampleid" and re.search(r"\s", s):
                issues.append("SampleID values must not contain spaces "
                              "(possible names)")
                break
            if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", s):
                issues.append(f"email-like value in column '{c}'")
                break
            if re.fullmatch(r"\d{11}", s):
                issues.append(f"11-digit identifier-like value in column '{c}'")
                break
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                issues.append(f"exact date-like value in column '{c}'")
                break
    feats = data_root / "features"
    if not feats.exists():
        issues.append("features/ directory is required")
    else:
        for p in feats.glob("*"):
            if p.suffix.lower() not in (".tsv",):
                issues.append(f"feature file '{p.name}' must be .tsv")
                continue
            try:
                cols = pd.read_csv(p, sep="\t", nrows=0).columns
            except Exception:
                issues.append(f"feature file '{p.name}' could not be read")
                continue
            for c in cols:
                if looks_like_identifier(c):
                    issues.append(f"feature header '{c}' looks like an identifier")
    return issues[:20]


@app.route("/upload", methods=["POST"])
def upload():
    if DEMO_ONLY and not PUBLIC_UPLOADS:
        return _render("error.html", code=403, title="Uploads disabled",
                       message="This public instance is a research demo. "
                               "Do not upload personal or health data. "
                               "Use the local runner or pip install avail-omics "
                               "to analyse your own data."), 403
    if PUBLIC_UPLOADS:
        fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        ip = fwd or request.remote_addr or "unknown"
        if _rate_limit_exceeded("upload:" + ip, limit=5):
            return _render("error.html", code=429, title="Upload limit reached",
                           message="Too many uploads from this network. "
                                   "Please try again later."), 429
        if request.form.get("deid_confirm") != "1":
            abort(400, "Please confirm that the data are de-identified/derived "
                       "and contain no names, identifiers, or personal health "
                       "information.")
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".zip"):
        abort(400, "please upload a .zip file")
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    jd = _job_dir(job_id)
    data_dir = jd / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = jd / "upload.zip"
    f.save(zip_path)
    try:
        _safe_extract(zip_path, data_dir)
    except Exception as e:
        shutil.rmtree(jd, ignore_errors=True)
        abort(400, f"bad zip: {e}")
    zip_path.unlink(missing_ok=True)
    data_root = _data_root_for(data_dir)
    ok, msg = _validate_layout(data_root)
    if not ok:
        shutil.rmtree(jd, ignore_errors=True)
        abort(400, msg)
    if PUBLIC_UPLOADS:
        issues = _public_upload_issues(data_root)
        if issues:
            shutil.rmtree(jd, ignore_errors=True)
            abort(400, "Public-upload check failed: " + "; ".join(issues))
    mode = _detect_mode(data_root)
    summary = _summarize(data_root, mode)
    feedback = _ai_feedback(summary)
    (jd / "validation.json").write_text(json.dumps(summary, indent=2))
    (jd / "ai_feedback.txt").write_text(feedback or "")
    _write_state(job_id, status="validated", created=job_id.replace("_", " "),
                 owner=session.get("_id"),
                 layout_msg=msg, mode=mode,
                 robustness=request.form.get("robustness") == "1",
                 ci=request.form.get("ci") == "1",
                 missingness=request.form.get("missingness") == "1",
                 tuning=request.form.get("tuning") == "1",
                 calibration=request.form.get("calibration") == "1",
                 ordinal=request.form.get("ordinal") == "1",
                 conformal=request.form.get("conformal") == "1",
                 hybrid=request.form.get("hybrid") == "1",
                 decision=request.form.get("decision") == "1",
                 radiomics=request.form.get("radiomics") == "1",
                 timesfm=request.form.get("timesfm") == "1")
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>/run", methods=["POST"])
def run_job(job_id):
    if not _can_access(job_id):
        abort(404)
    if not SERVER_RUN:
        abort(409, "server-side GPU runs are disabled; run on your own computer")
    st = _read_state(job_id)
    if st.get("status") != "validated":
        abort(409, "job cannot be started from this state")
    flags = {}
    for f in ["robustness", "ci", "missingness", "tuning",
              "calibration", "ordinal", "conformal", "hybrid", "decision",
              "radiomics", "timesfm"]:
        flags[f] = (request.form.get(f) == "1") or bool(st.get(f))
    _write_state(job_id, status="queued", **flags)
    job_queue.put(job_id)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/runner.zip")
def runner_zip():
    """One-click local runner bundle (compute on the user's own machine)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Thin bundle: launcher + README + free atlas (data) only.
        # The pipeline is installed from PyPI into a private venv on first run.
        rp = BASE / "local_runner" / "run_me.py"
        if rp.exists():
            zf.write(rp, arcname="run_me.py")
        rd = BASE / "local_runner" / "README.txt"
        if rd.exists():
            zf.write(rd, arcname="README.txt")
        atlas = BASE / "atlas" / "schaefer_2018"
        if atlas.exists():
            for p in sorted(atlas.iterdir()):
                if p.is_file():
                    zf.write(p, arcname=f"atlas/schaefer_2018/{p.name}")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="avail_local_runner.zip",
                     mimetype="application/zip")


@app.route("/docs/byo")
def docs_byo():
    md = BASE / "docs" / "BYO_COMPUTE.md"
    if not md.exists():
        abort(404)
    return _render("doc.html", title="Bring your own compute",
                   html=_md_to_html(md.read_text()),
                   back_url=url_for("index"))


@app.route("/docs/data")
def docs_data():
    md = BASE / "docs" / "DATA_HANDLING.md"
    if not md.exists():
        abort(404)
    return _render("doc.html", title="Data handling",
                   html=_md_to_html(md.read_text()),
                   back_url=url_for("index"))


@app.route("/demo-data/own_cohort_excerpt.zip")
def demo_own_bundle():
    p = BASE / "data_demos" / "own_cohort_excerpt.zip"
    if not p.exists():
        abort(404, "own-cohort excerpt is not available")
    return send_file(p, as_attachment=True,
                     download_name="own_cohort_excerpt.zip")


@app.route("/demo-data/ds004392_excerpt.zip")
def demo_data_bundle():
    p = BASE / "data_demos" / "ds004392_excerpt.zip"
    if not p.exists():
        abort(404, "real-cohort demo bundle is not available yet")
    return send_file(p, as_attachment=True,
                     download_name="ds004392_excerpt.zip")


@app.route("/dl/<job_id>/<secret>/data.zip")
def dl_data(job_id, secret):
    if secret != _read_state(job_id).get("dl_secret"):
        abort(403)
    jd = _job_dir(job_id)
    cache = jd / "data_served.zip"
    if not cache.exists():
        data_root = _data_root_for(jd / "data")
        with zipfile.ZipFile(cache, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(data_root.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(data_root))
    return send_file(cache, as_attachment=False,
                     download_name="data.zip", mimetype="application/zip")


@app.route("/job/<job_id>/run-cloud", methods=["POST"])
def run_cloud(job_id):
    if not _can_access(job_id):
        abort(404)
    st = _read_state(job_id)
    if st.get("status") not in ("validated", "error", "cloud"):
        abort(409, "job cannot be started from this state")
    if not GH_TOKEN or not GH_REPO:
        abort(409, "free cloud backend is not configured yet")
    base = _public_url()
    if not base:
        abort(503, "public tunnel is down  -  cannot serve data to the cloud runner")
    data_url = f"{base}/dl/{job_id}/{_dl_secret(job_id)}/data.zip"
    full = request.form.get("full") == "1" or request.args.get("full") == "1"
    r = _gh("POST",
            f"/repos/{GH_REPO}/actions/workflows/{GH_WORKFLOW}/dispatches",
            {"ref": GH_BRANCH,
             "inputs": {"job_id": job_id, "data_url": data_url,
                        "full": "true" if full else "false"}})
    if isinstance(r, dict) and "_error" in r:
        abort(502, f"GitHub dispatch failed: {r}")
    _write_state(job_id, status="cloud",
                 note=("submitted to free cloud (GitHub Actions, full deep models)"
                       if full else "submitted to free cloud (GitHub Actions)"))
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>/upload-results", methods=["POST"])
def upload_results(job_id):
    if not _can_access(job_id):
        abort(404)
    """Users upload their locally-computed results back for the AI report."""
    st = _read_state(job_id)
    if st.get("status") == "unknown":
        abort(404)
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".zip"):
        abort(400, "please upload a .zip of your results/ folder")
    res_dir = _job_dir(job_id) / "avail_results"
    res_dir.mkdir(parents=True, exist_ok=True)
    zpath = _job_dir(job_id) / "results_upload.zip"
    f.save(zpath)
    tmp = _job_dir(job_id) / "upload_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        _safe_extract(zpath, tmp)
    except Exception as e:
        abort(400, f"bad zip: {e}")
    src = tmp
    for cand in ["results", "avail_results"]:
        if (tmp / cand).is_dir():
            src = tmp / cand
            break
    for p in sorted(src.rglob("*")):
        if p.is_file():
            dest = res_dir / p.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    zpath.unlink(missing_ok=True)
    _write_state(job_id, status="done", note="results uploaded (computed locally)")
    llm_runner = BASE / "run_llm_advisor.py"
    if llm_runner.exists():
        subprocess.Popen([PY, str(llm_runner), "--out", str(res_dir)],
                         cwd=str(BASE),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>/delete", methods=["POST"])
def delete_job(job_id):
    if not _can_access(job_id):
        abort(404)
    st = _read_state(job_id)
    if st.get("demo"):
        abort(403, "demo jobs are shared and cannot be deleted here")
    if st.get("status") in ("queued", "running", "cloud"):
        abort(409, "wait for the run to finish (or stop it) before deleting")
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return redirect(url_for("index"))


@app.route("/job/<job_id>")
def job_page(job_id):
    if not _can_access(job_id):
        abort(404)
    state = _effective_state(job_id)
    if state.get("status") == "unknown":
        abort(404)
    log = ""
    lp = _job_dir(job_id) / "run.log"
    if lp.exists():
        log = "\n".join(lp.read_text(errors="replace").splitlines()[-120:])
    files = []
    res = _job_dir(job_id) / "avail_results"
    if res.exists():
        for p in sorted(res.iterdir()):
            if p.is_file():
                files.append({"name": p.name, "size": f"{p.stat().st_size/1e3:.1f} KB"})
    figs = [f for f in files if f["name"].lower().endswith(".png")]
    files = [f for f in files
             if not f["name"].lower().endswith((".pdf", ".tiff", ".npz"))]
    cmp_files = []
    cdir = _job_dir(job_id) / "avail_comparisons"
    if cdir.exists():
        for p in sorted(cdir.iterdir()):
            if p.is_file():
                cmp_files.append({"name": p.name,
                                  "size": f"{p.stat().st_size/1e3:.1f} KB"})
    summary = None
    vp = _job_dir(job_id) / "validation.json"
    if vp.exists():
        try:
            summary = json.loads(vp.read_text())
        except Exception:
            summary = None
    feedback = ""
    fp = _job_dir(job_id) / "ai_feedback.txt"
    if fp.exists():
        feedback = fp.read_text(errors="replace")
    decision = None
    ap = res / "llm_advisor.json"
    if ap.exists():
        try:
            decision = json.loads(ap.read_text())
        except Exception:
            decision = None
    top_models = []
    for name in ["meta_benchmark_results.tsv", "meta_benchmark_3class_results.tsv"]:
        bp = res / name
        if bp.exists():
            try:
                bdf = pd.read_csv(bp, sep="\t")
                sub = bdf[bdf.subgroup == "all"].sort_values(
                    "macro_auc", ascending=False).head(5)
                top_models = [{"model": r.model, "auc": round(r.macro_auc, 3)}
                              for r in sub.itertuples()]
            except Exception:
                top_models = []
            break
    return _render("job.html", job_id=job_id, state=state,
                   log=log, files=files, figs=figs, cmp_files=cmp_files,
                   summary=summary,
                   feedback=feedback, server_run=SERVER_RUN,
                   cloud_enabled=bool(GH_TOKEN and GH_REPO),
                   decision=decision, top_models=top_models)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    question = (data.get("question") or "").strip()[:2000]
    if not job_id or not question:
        abort(400, "job_id and question required")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{6,80}", job_id) or \
            not (_job_dir(job_id) / "state.json").exists():
        return jsonify(answer="No such job.", sources=[])
    if not _can_access(job_id):
        return jsonify(answer="No such job.", sources=[])
    # simple per-session rate limit (30 questions / hour)
    key = session.get("_id") or request.remote_addr
    if _rate_limit_exceeded(key):
        return jsonify(answer="Rate limit reached (30 questions/hour). "
                              "Please try again later.", sources=[])
    res = _job_dir(job_id) / "avail_results"
    context = ""
    sources = []
    try:
        from avail_ai.rag import build_context
        context, sources = build_context(question, _job_dir(job_id), BASE)
        _friendly = {
            "avail_results/biomarkers_literature.tsv": "biomarker literature",
            "avail_results/llm_advisor_report.md": "advisor report",
            "experiments/METHODS_NOTES.md": "methods notes",
        }
        sources = [_friendly.get(s, s) for s in sources]
    except Exception:
        pass
    if not context:
        for name in ["meta_benchmark_results.tsv", "meta_benchmark_3class_results.tsv"]:
            p = res / name
            if p.exists():
                context += p.read_text(errors="replace")[:6000]
                sources.append(name)
                break
        adv = res / "llm_advisor.json"
        if adv.exists():
            try:
                a = json.loads(adv.read_text())
                context += "\nAI interpretation: " + a.get("interpretation", "")[:2000]
                sources.append("llm_advisor.json")
            except Exception:
                pass
    if not context:
        return jsonify(answer="No results available yet for this job.")
    try:
        from avail_ai.llm import MultiLLMClient, build_clients
        llm = MultiLLMClient(build_clients())
        ans = llm.complete(
            "You are an AI assistant explaining AVAIL-AI multimodal analysis "
            "results. Answer concisely using only the provided context. Cite "
            "the sources (table names / PMIDs) that support each claim. If the "
            "context is insufficient for a claim, say so explicitly. Write plain "
            "text only: no markdown asterisks, headings, tables or LaTeX.",
            f"Context:\n{context}\n\nQuestion: {question}", max_tokens=900)
        ans = ans.strip()
        # verification: every PMID in the answer must exist in the evidence
        cited = re.findall(r"PMID[:\s]+(\d+)", ans)
        unverified = [p for p in cited if p not in context]
        if unverified:
            ans += ("\n\nNote: some references above were not present in the "
                    "retrieved evidence; verify before citing.")
        return jsonify(answer=ans, sources=sources)
    except Exception as e:
        return jsonify(answer=_llm_error_text(e),
                       sources=sources)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    if not _can_access(job_id):
        abort(404)
    state = _effective_state(job_id)
    if state.get("status") == "unknown":
        abort(404)
    log = ""
    lp = _job_dir(job_id) / "run.log"
    if lp.exists():
        log = "\n".join(lp.read_text(errors="replace").splitlines()[-200:])
    return jsonify(status=state.get("status"), log=log,
                   returncode=state.get("returncode"), error=state.get("error"))


@app.route("/api/agent", methods=["POST"])
def api_agent():
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or "").strip()
    message = (data.get("message") or "").strip()[:4000]
    if not job_id or not message:
        abort(400, "job_id and message required")
    if not _can_access(job_id):
        abort(404)
    key = session.get("_id") or request.remote_addr
    if _rate_limit_exceeded(key):
        return jsonify(reply="Rate limit reached (30 messages/hour). "
                             "Please try again later.", steps=[], pending=None)
    try:
        from webapp.agent import run_agent
        reply, steps, pending = run_agent(job_id, message)
    except Exception as e:
        return jsonify(reply=_llm_error_text(e), steps=[], pending=None), 200
    return jsonify(reply=reply, steps=steps, pending=pending)


@app.route("/job/<job_id>/cmp/<name>")
def comparison_file(job_id, name):
    if not _can_access(job_id):
        abort(404)
    base = (_job_dir(job_id) / "avail_comparisons").resolve()
    p = (base / name).resolve()
    if not str(p).startswith(str(base)):
        abort(403)
    if not p.is_file():
        abort(404)
    return send_file(p, as_attachment=True)


@app.route("/job/<job_id>/results.zip")
def download_results(job_id):
    if not _can_access(job_id):
        abort(404)
    if _effective_state(job_id).get("status") != "done":
        abort(409, "job not finished yet")
    res = _job_dir(job_id) / "avail_results"
    if not res.exists():
        abort(404)
    zip_path = _job_dir(job_id) / "results.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(res.iterdir()):
                if p.is_file():
                    zf.write(p, arcname=f"avail_results/{p.name}")
    return send_file(zip_path, as_attachment=True,
                     download_name=f"{job_id}_results.zip")


@app.route("/job/<job_id>/report")
def report(job_id):
    if not _can_access(job_id):
        abort(404)
    md = _job_dir(job_id) / "avail_results" / "llm_advisor_report.md"
    if not md.exists():
        abort(404, "no LLM report yet")
    res = _job_dir(job_id) / "avail_results"
    figs = [f for f in ['fig1_meta_benchmark.png', 'fig2_biomarkers.png',
                        'fig3_continuum.png'] if (res / f).exists()]
    html = _md_to_html(md.read_text())
    import re as _re
    html = _re.sub(r"PMID[:\s]*(\d{4,9})",
                   lambda m: ('<a href="https://pubmed.ncbi.nlm.nih.gov/'
                              + m.group(1) + '/" target="_blank" '
                              'rel="noopener">PMID ' + m.group(1) + '</a>'),
                   html)
    return _render("report.html", job_id=job_id, html=html, figs=figs)


@app.route("/job/<job_id>/table/<name>")
def table_view(job_id, name):
    if not _can_access(job_id):
        abort(404)
    """Read-only HTML preview of a result table (same file users download)."""
    import html as _html
    base = (_job_dir(job_id) / "avail_results").resolve()
    f = (base / name).resolve()
    if not str(f).startswith(str(base)) or not f.is_file():
        abort(404)
    ext = f.suffix.lower()
    if ext not in (".tsv", ".csv", ".txt", ".md"):
        abort(400, "not a text table")
    if ext in (".tsv", ".csv"):
        sep = "\t" if ext == ".tsv" else ","
        try:
            with open(f, "r", errors="replace") as fh:
                n_total = max(0, sum(1 for _ in fh) - 1)
            df = pd.read_csv(f, sep=sep, nrows=100)
            body = df.to_html(index=False, classes="tbl", border=0, escape=True)
            note = (f"showing first 100 of {n_total} rows" if n_total > 100
                    else f"all {n_total} rows")
        except Exception as e:
            body = f"<pre>{_html.escape(f.read_text(errors='replace')[:20000])}</pre>"
            note = f"text preview ({type(e).__name__})"
    else:
        body = f"<pre>{_html.escape(f.read_text(errors='replace')[:20000])}</pre>"
        note = "text preview (first 20 kB)"
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{_html.escape(f.name)}</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#081220;color:#e8f0fa;margin:2rem}}
h1{{font-size:1.1rem}} a{{color:#38e1c6;text-decoration:none}}
.tbl{{border-collapse:collapse;font-size:.78rem;display:block;overflow:auto;max-height:70vh}}
.tbl td,.tbl th{{border:1px solid rgba(255,255,255,.14);padding:.3rem .5rem;white-space:nowrap}}
.tbl th{{position:sticky;top:0;background:#0e1e33}}
pre{{overflow:auto;max-height:70vh;white-space:pre-wrap}}
.note{{color:#8fa3bd;font-size:.85rem}}
</style></head><body>
<p><a href="/job/{job_id}">&larr; back to job</a> &middot;
<a href="/job/{job_id}/file/{_html.escape(name)}">download original</a></p>
<h1>{_html.escape(f.name)} <span class="note">({note})</span></h1>
{body}
<p class="note">This is the exact table behind the reported numbers and figures;
missing values appear as stored in the file.</p>
</body></html>"""
    return page, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/job/<job_id>/file/<name>")
def result_file(job_id, name):
    if not _can_access(job_id):
        abort(404)
    p = (_job_dir(job_id) / "avail_results" / name).resolve()
    if not str(p).startswith(str((_job_dir(job_id) / "avail_results").resolve())):
        abort(403)
    if not p.is_file():
        abort(404)
    return send_file(p, as_attachment=True)


_HTML_DANGER = re.compile(
    r"<\s*(script|iframe|object|embed|style)\b.*?(<\s*/\s*\1\s*>|$)",
    re.I | re.S)
_HTML_EVENT = re.compile(
    r"[\s/]+on\w+\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.I)
_HTML_JSURL = re.compile(r"(javascript|vbscript)\s*:", re.I)


def _sanitize_html(html_str):
    try:
        import nh3
        return nh3.clean(html_str)
    except Exception:
        pass
    try:
        import bleach
        return bleach.clean(html_str, strip=True)
    except Exception:
        pass
    html_str = _HTML_DANGER.sub("", html_str)
    html_str = _HTML_EVENT.sub("", html_str)
    html_str = _HTML_JSURL.sub("#", html_str)
    return html_str


def _md_to_html(text):
    try:
        import markdown
        return _sanitize_html(markdown.markdown(text, extensions=["tables"]))
    except ImportError:
        import html
        return "<pre>" + html.escape(text) + "</pre>"


def _log_finished(job_id):
    lp = _job_dir(job_id) / "run.log"
    if not lp.exists():
        return False
    txt = lp.read_text(errors="replace")
    return ("END-TO-END RUN COMPLETE" in txt or "[generic] DONE" in txt)


def _effective_state(job_id):
    """State that survives app restarts: infer completion from the log marker."""
    st = _read_state(job_id)
    if st.get("status") in ("running", "queued") and _log_finished(job_id):
        _write_state(job_id, status="done")
        st = _read_state(job_id)
    return st


def _requeue_interrupted():
    """Reconcile interrupted jobs periodically (also after server restarts)."""
    import time as _time
    while True:
        try:
            if UPLOADS.exists():
                for d in sorted(UPLOADS.iterdir()):
                    if not d.is_dir():
                        continue
                    st = _read_state(d.name)
                    if st.get("status") not in ("running", "queued"):
                        continue
                    if _log_finished(d.name):
                        _write_state(
                            d.name, status="done",
                            finished=datetime.now().isoformat(timespec="seconds"))
                        continue
                    alive = subprocess.run(
                        ["pgrep", "-f", f"python.*uploads/{d.name}"],
                        capture_output=True).returncode == 0
                    if alive:
                        continue
                    _write_state(d.name, status="queued",
                                 note="requeued after worker loss")
                    job_queue.put(d.name)
        except Exception:
            pass
        _time.sleep(120)


def _cleanup_old_jobs():
    hours = float(os.environ.get("AVAIL_JOB_RETENTION_HOURS", "24"))
    while True:
        now = time.time()
        try:
            for d in UPLOADS.iterdir():
                if not d.is_dir():
                    continue
                st = _read_state(d.name)
                if st.get("demo"):
                    continue
                if now - d.stat().st_mtime > hours * 3600:
                    shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        time.sleep(3600)


def create_app():
    """App factory: performs startup (dirs, threads) and returns the Flask app.

    Used by waitress (production): waitress-serve --call webapp.app:create_app
    """
    UPLOADS.mkdir(parents=True, exist_ok=True)
    if not RUNNER.exists() and not GENERIC_RUNNER.exists():
        raise RuntimeError(f"runner not found under {BASE}")
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_requeue_interrupted, daemon=True).start()
    threading.Thread(target=_cleanup_old_jobs, daemon=True).start()
    if GH_TOKEN and GH_REPO:
        threading.Thread(target=_gh_poller, daemon=True).start()
    return app


def main():
    create_app()
    port = int(os.environ.get("AVAIL_WEB_PORT", "8080"))
    print(f"AVAIL-AI uploader on :{port} (root={BASE})", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
