#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless agent evaluation #3: data-quality injection battery.

Builds a clean 3-modality base dataset (n=48, 2 groups, class signal in
the compositional modality), then injects 12 realistic corruptions.  For
each injection we record the outcome of the four defence layers:
  L1 fail-closed  : loader/metadata raises (good)
  L2 auto-correct : normalization handles it correctly (good)
  L3 flagged      : platform validation (_summarize) warns (good)
  L4 semantic     : permutation semantics stay leak-free (good)
  GAP             : silent acceptance (documented + mitigation below)
"""
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import LeaveOneOut, cross_val_predict  # noqa: E402

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
sys.path.insert(0, str(ROOT))
from avail_omics.generic import load_generic, detect_compositional  # noqa: E402

try:
    sys.path.insert(0, str(ROOT / "webapp"))
    import app as webapp_app  # noqa: E402
    HAVE_WEBAPP = True
except Exception as e:
    HAVE_WEBAPP = False
    print(f"[warn] webapp _summarize unavailable: {type(e).__name__}")

WORK = Path("/tmp/opencode/dq_eval")
OUT = ROOT / "experiments/results/agent_eval"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

def write_ds(root, meta, tables, comp=None):
    root = Path(root); shutil.rmtree(root, ignore_errors=True)
    (root / "features").mkdir(parents=True)
    meta.to_csv(root / "metadata.tsv", sep="\t", index=False)
    for name, df in tables.items():
        df.to_csv(root / "features" / f"{name}.tsv", sep="\t", index=False)
    if comp:
        (root / "compositional.txt").write_text("\n".join(comp) + "\n")
    return root

def base_tables():
    ids = [f"S{i:03d}" for i in range(48)]
    grp = ["A"] * 24 + ["B"] * 24
    meta = pd.DataFrame({"SampleID": ids, "Group": grp,
                         "Age": rng.integers(55, 81, 48),
                         "Education": rng.integers(8, 21, 48),
                         "Sex": rng.choice(["M", "F"], 48)})
    clin = pd.DataFrame({"SampleID": ids,
                         "Age": meta.Age.astype(float),
                         "Education": meta.Education.astype(float)})
    taxa = [f"t{i:02d}" for i in range(25)]
    abn = rng.dirichlet(np.ones(25) * 8.0, size=48) * 100.0
    abn[:24, :5] += 6.0; abn[24:, :5] -= 2.0
    abn = np.clip(abn, 0.01, None); abn = abn / abn.sum(1, keepdims=True) * 100
    abun = pd.DataFrame(abn, columns=taxa); abun.insert(0, "SampleID", ids)
    cog = pd.DataFrame(rng.normal(10, 2.5, (48, 6)).round(1),
                       columns=[f"cog{i}" for i in range(6)])
    cog.iloc[rng.choice(48, 4, replace=False), 2] = np.nan
    cog.insert(0, "SampleID", ids)
    return meta, {"clin": clin, "abun": abun, "cog": cog}

def quick_auc(mods, ids, label, micro):
    X = np.vstack([mods["abun"][i] for i in ids])
    if "abun" in micro:
        X = np.log1p(X / X.sum(1, keepdims=True) * 100)
    Z = PCA(5, random_state=0).fit_transform(X)
    y = np.array([label[i] for i in ids])
    p = cross_val_predict(LogisticRegression(max_iter=2000),
                          Z, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

def summarize(ds):
    if not HAVE_WEBAPP:
        return []
    try:
        s = webapp_app._summarize(Path(ds), "generic")
        return s.get("warnings", [])
    except Exception as e:
        return [f"ERROR:{type(e).__name__}"]

rows = []
def record(tag, layer, outcome, detail=""):
    rows.append({"injection": tag, "layer": layer, "outcome": outcome,
                 "detail": detail[:160]})
    print(f"[E3] {tag:22s} {layer:14s} {outcome:16s} {detail[:80]}", flush=True)

META, TAB = base_tables()
COMP = ["abun"]

# ---- 1 no metadata ----
ds = write_ds(WORK / "j01", META, TAB, COMP); (ds / "metadata.tsv").unlink()
try:
    load_generic(ds); record("no_metadata", "L1", "GAP")
except FileNotFoundError as e:
    record("no_metadata", "L1", "fail_closed", str(e)[:60])

# ---- 2 missing Group column ----
ds = write_ds(WORK / "j02", META.drop(columns=["Group"]), TAB, COMP)
try:
    load_generic(ds); record("missing_group_col", "L1", "GAP")
except ValueError as e:
    record("missing_group_col", "L1", "fail_closed", str(e)[:60])

# ---- 3 sex variants ----
m = META.copy()
m.loc[:7, "Sex"] = ["male ", "FEMALE", "1", "0", "M", "f", "None", ""]
ds = write_ds(WORK / "j03", m, TAB, COMP)
g, conf, mods_, _ = load_generic(ds)
sex = [conf[i][2] for i in m.SampleID[:8]]
ok = sex[:4] == [1.0, 0.0, 1.0, 0.0] and np.isnan(sex[6]) and np.isnan(sex[7])
record("sex_variants", "L2", "auto_correct" if ok else "GAP", f"sex={sex}")

# ---- 4 negative values in compositional ----
t4 = {k: v.copy() for k, v in TAB.items()}
t4["abun"].iloc[3, 1:3] = -1.5
ds = write_ds(WORK / "j04a", META, t4, COMP)
g, conf, mods_, _ = load_generic(ds)
micro = detect_compositional(ds, mods_, g.keys(), verbose=False)
rec = "compositional.txt wins" if "abun" in micro else "lost"
record("neg_abun+file", "L2", "partial", f"CLR still applied ({rec}); negatives become NaN in log")
ds = write_ds(WORK / "j04b", META, t4, None)
micro = detect_compositional(ds, mods_, g.keys(), verbose=False)
record("neg_abun-no-file", "GAP?", "GAP" if "abun" not in micro else "ok",
       "heuristic silently rejects comp modality (needs explicit contract)")

# ---- 5 constant feature block ----
t5 = {k: v.copy() for k, v in TAB.items()}
for j in range(30):
    t5["cog"][f"dead{j}"] = 7.7
ds = write_ds(WORK / "j05", META, t5, COMP)
ws = summarize(ds)
record("const_block(30)", "L3", "flagged" if ws else "GAP", f"warnings={len(ws)}")

# ---- 6 NaN flood ----
t6 = {k: v.copy() for k, v in TAB.items()}
t6["cog"].iloc[:43, 3] = np.nan
ds = write_ds(WORK / "j06", META, t6, COMP)
ws = summarize(ds)
record("nan_flood(90%)", "L3", "flagged" if ws else "GAP", f"warnings={len(ws)}")

# ---- 7 duplicate SampleIDs ----
m7 = pd.concat([META, META.iloc[:6].assign(Group="B")], ignore_index=True)
ds = write_ds(WORK / "j07", m7, TAB, COMP)
g, conf, mods_, _ = load_generic(ds)
ws = summarize(ds)
dup_hit = any("duplicate" in w.lower() for w in ws)
record("dup_sample_ids", "L3", "flagged" if dup_hit else "GAP",
       f"loader last-wins (n_meta={len(m7)} vs {len(mods_['abun'])}) "
       f"PLATFORM {'flags' if dup_hit else 'silent'}")

# ---- 8 European decimal comma ----
t8 = {k: v.copy() for k, v in TAB.items()}
_c0 = t8["cog"].columns[1]
_vv = t8["cog"][_c0].tolist()
for i in range(10):
    _vv[i] = "12,5"
t8["cog"][_c0] = _vv
ds = write_ds(WORK / "j08", META, t8, COMP)
g, conf, mods_, _ = load_generic(ds)
v = mods_["cog"]["S000"][0]
ws = summarize(ds)
dn_hit = any("non-numeric" in w.lower() for w in ws)
record("decimal_comma", "L3", "flagged" if dn_hit else "GAP",
       f"loader coerces '12,5' -> {v}; PLATFORM {'flags' if dn_hit else 'silent'}")

# ---- 9 tiny class ----
m9 = META.copy(); m9.loc[:1, "Group"] = "C"
t9 = {k: v.copy() for k, v in TAB.items()}
for k in t9:
    t9[k] = t9[k].iloc[:48].copy()
ds = write_ds(WORK / "j09", m9, t9, COMP)
ws = summarize(ds)
hit = any("imbalance" in w.lower() for w in ws)
record("tiny_class(n=2)", "L3", "flagged" if hit else "GAP", f"warnings={len(ws)}")

# ---- 10 single-column modality ----
t10 = {k: v.copy() for k, v in TAB.items()}
t10["extra"] = pd.DataFrame({"SampleID": META.SampleID})
ds = write_ds(WORK / "j10", META, t10, COMP)
g, conf, mods_, _ = load_generic(ds)
ws = summarize(ds)
record("one_col_modality", "L3", "flagged" if "extra" not in mods_ and
       any("extra" in w for w in ws) else "GAP", f"skipped={'extra' not in mods_}")

# ---- 11 clinical-scale misfire (no compositional.txt) ----
ds = write_ds(WORK / "j11", META, TAB, None)
g, conf, mods_, _ = load_generic(ds)
micro = detect_compositional(ds, mods_, g.keys(), verbose=False)
misfire = "clin" in micro
record("clin_sum_50-150", "L4", "flagged_run_log" if misfire else "ok",
       f"CLR'd={sorted(micro)} -> visible in run log; contract fixes")

# ---- 12 label permutation semantics ----
ids = sorted(META.SampleID)
label = {i: 0 if gg == "A" else 1 for i, gg in zip(META.SampleID, META.Group)}
ds = write_ds(WORK / "j12", META, TAB, COMP)
g, conf, mods_, _ = load_generic(ds)
real = quick_auc(mods_, ids, label, {"abun"})
pas = []
for r in range(5):
    vals = list(label.values())
    np.random.default_rng(100 + r).shuffle(vals)
    pas.append(quick_auc(mods_, ids, dict(zip(ids, vals)), {"abun"}))
ok = real > 0.65 and 0.35 < np.mean(pas) < 0.65
record("label_permutation", "L4", "semantic_pass" if ok else "GAP",
       f"real AUC={real:.2f}  permuted mean={np.mean(pas):.2f} "
       f"[{min(pas):.2f},{max(pas):.2f}] (5 shuffles)")

df = pd.DataFrame(rows)
df.to_csv(OUT / "data_quality_detection.tsv", sep="\t", index=False)
print("\n=== E3 ÖZET ===")
print(df.outcome.value_counts().to_string())
