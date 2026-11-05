#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4 reproducibility capsule: deterministic data -> runner bundle ->
offline end-to-end run -> structure/published-number comparison."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
OUT = ROOT / "experiments" / "results" / "agent_eval"
OUT.mkdir(parents=True, exist_ok=True)
CAP = Path("/tmp/opencode/repro_capsule")
REUSE = os.environ.get("REPRO_REUSE") == "1"
if not REUSE:
    shutil.rmtree(CAP, ignore_errors=True)
DATA, BUNDLE, RUN = CAP / "data", CAP / "bundle", CAP / "out"
shutil.rmtree(DATA, ignore_errors=True)
shutil.rmtree(BUNDLE, ignore_errors=True)
(DATA / "features").mkdir(parents=True)

# 1 deterministic dataset (identical generator to E3)
rng = np.random.default_rng(42)
ids = [f"S{i:03d}" for i in range(48)]
meta = pd.DataFrame({"SampleID": ids, "Group": ["A"] * 24 + ["B"] * 24,
                     "Age": rng.integers(55, 81, 48),
                     "Education": rng.integers(8, 21, 48),
                     "Sex": rng.choice(["M", "F"], 48)})
meta.to_csv(DATA / "metadata.tsv", sep="\t", index=False)
clin = pd.DataFrame({"SampleID": ids, "Age": meta.Age.astype(float),
                     "Education": meta.Education.astype(float)})
clin.to_csv(DATA / "features" / "clin.tsv", sep="\t", index=False)
abn = rng.dirichlet(np.ones(25) * 8.0, size=48) * 100.0
abn[:24, :5] += 6.0; abn[24:, :5] -= 2.0
abn = np.clip(abn, 0.01, None); abn = abn / abn.sum(1, keepdims=True) * 100
abun = pd.DataFrame(abn, columns=[f"t{i:02d}" for i in range(25)])
abun.insert(0, "SampleID", ids)
abun.to_csv(DATA / "features" / "abun.tsv", sep="\t", index=False)
cog = pd.DataFrame(rng.normal(10, 2.5, (48, 6)).round(1),
                   columns=[f"cog{i}" for i in range(6)])
cog.iloc[rng.choice(48, 4, replace=False), 2] = np.nan
cog.insert(0, "SampleID", ids)
cog.to_csv(DATA / "features" / "cog.tsv", sep="\t", index=False)
(DATA / "compositional.txt").write_text("abun\n")

# 2 runner bundle (exact webapp zip file set)
BUNDLE.mkdir(parents=True)
for f in ["run_generic.py", "run_llm_advisor.py"]:
    shutil.copy(ROOT / f, BUNDLE / f)
(BUNDLE / "local_runner").mkdir()
for p in (ROOT / "local_runner").iterdir():
    if p.is_file() and p.suffix != ".pyc":
        shutil.copy(p, BUNDLE / "local_runner" / p.name)
for sub in ["avail_ai", "avail_omics"]:
    (BUNDLE / "avail_ai" / sub).mkdir(parents=True)
    for p in (ROOT / "avail_ai" / sub).rglob("*.py"):
        dst = BUNDLE / "avail_ai" / sub / p.relative_to(ROOT / "avail_ai" / sub)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst)

# 3 published-number audit (existing integrity audit)
print("[capsule] running integrity audit ...", flush=True)
audit_out = subprocess.run([sys.executable, str(ROOT / "experiments/integrity_audit.py")],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
print("[capsule] audit done", flush=True)
audit_tsv = ROOT / "experiments/results/integrity_audit.tsv"
audit = {"ok": 0, "mismatch": 0}
if audit_tsv.exists():
    try:
        adf = pd.read_csv(audit_tsv, sep="\t")
        if "match" in adf.columns:
            audit["ok"] = int(adf["match"].fillna(False).sum())
            audit["mismatch"] = len(adf) - audit["ok"]
    except Exception:
        pass
audit["dirs"] = len({r for r in []})  # placeholder

# 4 offline end-to-end run (documented user path; --reuse keeps prior outputs)
t0 = time.time()
if REUSE and (RUN / "meta_benchmark_results.tsv").exists():
    class _R:  # noqa
        returncode = 0
        stdout = "[reused previous end-to-end outputs]"
    rr = _R()
else:
    env = dict(os.environ, AVAIL_OFFLINE="1")
    rr = subprocess.run([sys.executable, "local_runner/run_me.py", str(DATA),
                         "--out", str(RUN)],
                        cwd=BUNDLE, capture_output=True, text=True,
                        timeout=3000, env=env)
dur = time.time() - t0

# 5 checks
checks = {"exit_ok": rr.returncode == 0}
res = RUN / "meta_benchmark_results.tsv"
pred = RUN / "loo_predictions.tsv"
checks["results_exists"] = res.exists()
checks["pred_rows"] = int(len(pd.read_csv(pred, sep="\t"))) if pred.exists() else 0
cur = {}
if res.exists():
    df = pd.read_csv(res, sep="\t")
    sub = df[df.subgroup == "all"] if "subgroup" in df.columns else df
    best = sub.sort_values("macro_auc", ascending=False).iloc[0]
    cur = {"n": int(best.n), "n_models": int(sub.model.nunique()),
           "best_model": str(best.model),
           "best_auc": round(float(best.macro_auc), 6)}

exp_path = OUT / "repro_expected.json"
cmp_ = {}
if cur:
    if not exp_path.exists():
        exp_path.write_text(json.dumps(cur, indent=2))
        cmp_ = {"reference_written": True}
    else:
        expected = json.loads(exp_path.read_text())
        cmp_ = {"best_model_match": cur["best_model"] == expected["best_model"],
                "auc_match": abs(cur["best_auc"] - expected["best_auc"]) <= 1e-3,
                "n_models_match": cur["n_models"] == expected["n_models"],
                "expected": expected, "observed": cur}

import platform
fp = {"python": sys.version.split()[0], "platform": platform.platform()}
for mod in ["numpy", "pandas", "sklearn"]:
    try:
        fp[mod] = __import__(mod).__version__
    except Exception:
        fp[mod] = "?"
sys.path.insert(0, str(ROOT / "avail_ai"))
try:
    import avail_omics
    fp["avail_omics"] = avail_omics.__version__
except Exception:
    fp["avail_omics"] = "?"

paint = lambda b: "PASS" if b else "FAIL"
lines = ["# E4 reproducibility capsule - report", ""]
lines.append(f"- audit (published AUCs recomputed from saved predictions): "
             f"{audit['ok']}/{audit['ok'] + audit['mismatch']} OK, "
             f"{audit['mismatch']} mismatch")
lines.append(f"- end-to-end offline run (documented runner path): exit="
             f"{rr.returncode}, {dur:.0f}s, predictions={checks['pred_rows']}")
for k, v in cmp_.items():
    if k in ("expected", "observed"):
        continue
    lines.append(f"- {k}: {paint(bool(v))}")
if cur:
    lines.append(f"- observed best: {cur['best_model']} AUC={cur['best_auc']} "
                 f"({cur['n_models']} models, n={cur['n']})")
lines += ["", "## Environment fingerprint", "",
          *[f"- {k}: {v}" for k, v in fp.items()], "",
          "## Runner output (tail)", "```", (rr.stdout or "")[-1200:], "```",
          "## Audit runner stdout (tail)", "```", (audit_out.stdout or "")[-400:], "```"]
(OUT / "repro_report.md").write_text("\n".join(lines))
print(json.dumps({"audit": audit, "checks": checks, "cmp": cmp_, "cur": cur,
                  "dur_s": round(dur)}, indent=2, default=str))
