#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLP-style quick demo for avail-omics.

Demo-only: three fast meta-learners; no clinical data collection. The full
11-model pipeline is available from the Python package or a user-owned worker.
"""
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

# Keep the demo fast; generic.py reads this at import time.
os.environ.setdefault("AVAIL_META_MODELS", "SimpleMean,WSimpleMean,ElasticNetLR")

import gradio as gr
import numpy as np
import pandas as pd

from avail_omics.generic import run_generic_pipeline

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
MAX_UPLOAD_MB = int(os.environ.get("AVAIL_DEMO_MAX_MB", "100"))


def _read_doc(name):
    p = DOCS / name
    return p.read_text() if p.exists() else f"# {name}"


def _safe_extract(zip_path, dest):
    total = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            total += info.file_size
            if total > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"ZIP exceeds {MAX_UPLOAD_MB} MB uncompressed")
            target = (Path(dest) / info.filename).resolve()
            if not str(target).startswith(str(Path(dest).resolve()) + os.sep):
                raise ValueError("unsafe ZIP path")
            if info.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _find_root(root):
    root = Path(root)
    if (root / "metadata.tsv").exists() and (root / "features").is_dir():
        return root
    hits = [p.parent for p in root.rglob("metadata.tsv")
            if (p.parent / "features").is_dir()]
    if not hits:
        raise ValueError("metadata.tsv and features/ were not found in the ZIP")
    return sorted(hits, key=lambda p: len(p.parts))[0]


def _make_synthetic(root):
    rng = np.random.default_rng(7)
    root = Path(root)
    (root / "features").mkdir(parents=True, exist_ok=True)
    ids = [f"S{i:03d}" for i in range(1, 49)]
    groups = ["HC", "MCI", "PDD"]
    meta = pd.DataFrame({
        "SampleID": ids,
        "Group": [groups[i % 3] for i in range(len(ids))],
        "Age": rng.integers(50, 82, len(ids)),
        "Sex": rng.choice(["F", "M"], len(ids)),
    })
    meta.to_csv(root / "metadata.tsv", sep="\t", index=False)
    for name, nfeat, keep in [("brain", 24, 0.72), ("stool16S", 40, 0.85),
                              ("saliva16S", 30, 0.80)]:
        X = rng.normal(size=(len(ids), nfeat))
        keep_mask = rng.random(len(ids)) < keep
        d = pd.DataFrame(X, columns=[f"{name}_f{j:02d}" for j in range(nfeat)])
        d.insert(0, "SampleID", ids)
        d = d[keep_mask]
        d.to_csv(root / "features" / f"{name}.tsv", sep="\t", index=False)
    return root


def _run(data_root, label):
    data_root = _find_root(data_root)
    out = Path(tempfile.mkdtemp(prefix="avail_demo_out_"))
    res = run_generic_pipeline(str(data_root), str(out), verbose=False)
    df = pd.read_csv(out / "meta_benchmark_results.tsv", sep="\t")
    try:
        piv = df.pivot(index="model", columns="subgroup", values="macro_auc")
        table = piv.round(3).reset_index()
    except Exception:
        table = df.round(3)
    summary = (
        f"**{label}**  \n"
        f"Subjects: **{len(res['ids'])}**; classes: **{', '.join(res['classes'])}**  \n"
        f"Demo meta-learners: **{len(df.model.unique())}**  \n"
        f"Output: `{out}`"
    )
    zip_path = out.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out))
    return summary, table, str(zip_path)


def _on_upload(file):
    if not file:
        return "Upload a ZIP first.", None, gr.update(visible=False)
    tmp = tempfile.mkdtemp(prefix="avail_demo_in_")
    try:
        _safe_extract(file, tmp)
        return _run(tmp, "Uploaded dataset")
    except Exception as e:
        return f"**Error:** {e}", None, gr.update(visible=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _on_synthetic():
    tmp = tempfile.mkdtemp(prefix="avail_demo_syn_")
    try:
        _make_synthetic(tmp)
        return _run(tmp, "Synthetic demo")
    except Exception as e:
        return f"**Error:** {e}", None, gr.update(visible=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


with gr.Blocks(title="AVAIL-omics demo") as demo:
    gr.Markdown("# AVAIL-omics\n**Availability-aware multimodal integration — quick demo**")
    gr.Markdown("Demo-only: 3 fast meta-learners; no patient-identifiable data.")

    with gr.Tab("Demo"):
        with gr.Row():
            upload = gr.File(label="Upload ZIP (metadata.tsv + features/*.tsv)",
                             file_types=[".zip"], type="filepath")
            with gr.Column():
                run_btn = gr.Button("Run quick benchmark", variant="primary")
                syn_btn = gr.Button("Run synthetic demo")
        summary = gr.Markdown()
        table = gr.Dataframe(label="Meta-benchmark (macro-AUC)", interactive=False)
        download = gr.File(label="Download results ZIP", visible=False)
        run_btn.click(_on_upload, upload, [summary, table, download])
        syn_btn.click(_on_synthetic, None, [summary, table, download])

    with gr.Tab("Documentation"):
        gr.Markdown(_read_doc("DOCUMENTATION.md"))
    with gr.Tab("Tutorial"):
        gr.Markdown(_read_doc("TUTORIAL.md"))
    with gr.Tab("Download"):
        gr.Markdown(_read_doc("DOWNLOAD.md"))
    with gr.Tab("About"):
        gr.Markdown(_read_doc("ABOUT.md"))

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
