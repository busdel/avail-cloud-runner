# -*- coding: utf-8 -*-
"""AVAIL-AI agent: LLM tool-calling loop over the existing pipeline functions.

The agent can inspect jobs, run comparisons, generate figures and answer
questions grounded in the job's results. Heavy actions (starting runs,
running comparisons) require explicit user confirmation.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

BASE = Path(os.environ.get("AVAIL_WEB_ROOT", "/media/suleyman_disk/AVAIL_GPU")).resolve()
PY = os.environ.get("AVAIL_PYTHON") or "python3"
UPLOADS = BASE / "uploads"
MAX_ITERS = 6

SYSTEM = """You are Dr. Avail, the built-in research assistant of AVAIL-AI, an
invite-only multimodal missing-data machine-learning platform.

You can call tools to inspect and drive a user's analysis job. Rules:
- Work step by step: check state before acting, report numbers exactly as tools return them.
- Never invent numbers, filenames or results. If a tool fails, say so and propose a fix.
- Keep answers concise (2-6 sentences + bullet lists when useful).
- Never mention any internal dataset or cohort names, absolute filesystem paths,
  or counts/summaries of any study data that was not produced by the user's own job.
- Heavy actions: start_job and run_comparison must always be proposed to the user
  with a short explanation; the system handles the confirmation step.
- Answer in English. Use plain text with at most **bold** and `code`; no
  headings, tables or LaTeX."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "job_status",
            "description": "Status, mode and recent log lines of a job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_results",
            "description": "List result files produced for a job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "benchmark_summary",
            "description": "Top models and macro-AUC from the meta-learner benchmark of a finished job.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_results_question",
            "description": "Answer a question about a job's results, grounded in retrieved evidence (RAG).",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "question": {"type": "string"},
                },
                "required": ["job_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_job",
            "description": "Start the main pipeline run of a validated job on the server GPU. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "robustness": {"type": "boolean"},
                    "ci": {"type": "boolean"},
                    "missingness": {"type": "boolean"},
                    "tuning": {"type": "boolean"},
                    "calibration": {"type": "boolean"},
                    "ordinal": {"type": "boolean"},
                    "conformal": {"type": "boolean"},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_comparison",
            "description": "Run one optional comparison on a finished job. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "name": {
                        "type": "string",
                        "enum": ["robustness", "ci", "missingness", "tuning",
                                 "calibration", "ordinal", "conformal", "hybrid"],
                    },
                },
                "required": ["job_id", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_figure",
            "description": "Generate publication-ready figure(s) from a finished job's results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "figure": {
                        "type": "string",
                        "enum": ["benchmark", "biomarkers", "continuum", "shap", "all"],
                    },
                },
                "required": ["job_id", "figure"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "imaging_pipeline",
            "description": "Report the imaging preprocessing protocol used for a job (atlas, feature blocks, optional TimesFM benchmark) and its outputs.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "theory_notes",
            "description": "Search the platform's methods/theory notes for a topic (e.g. leakage control, conformal prediction, missingness).",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
]

COMPARISON_SCRIPTS = {
    "robustness": "imputation_benchmark_v4.py",
    "ci": "robustness_ci.py",
    "missingness": "missingness_recovery.py",
    "tuning": "tuning.py",
    "calibration": "calibration_experiment.py",
    "ordinal": "ordinal_experiment.py",
    "conformal": "conformal_experiment.py",
    "hybrid": "hybrid_comparison.py",
}


# ---------------------------------------------------------------- session
def _clean_job_id(job_id):
    return re.sub(r"[^0-9a-zA-Z_\-]", "", job_id or "")


def _job_dir(job_id):
    return UPLOADS / job_id


def _session_path(job_id):
    return _job_dir(job_id) / "agent_session.json"


def _read_state(job_id):
    try:
        return json.loads((_job_dir(job_id) / "state.json").read_text())
    except Exception:
        return {"status": "unknown"}


def _write_state(job_id, **kw):
    from datetime import datetime
    st = _read_state(job_id)
    st.update(kw)
    st["updated"] = datetime.now().isoformat(timespec="seconds")
    (_job_dir(job_id) / "state.json").write_text(json.dumps(st, indent=2))


def load_session(job_id):
    p = _session_path(job_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"messages": [], "pending": None}


def save_session(job_id, sess):
    p = _session_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sess, ensure_ascii=False))


# ---------------------------------------------------------------- tool handlers
def _log_tail(job_id, n=4000):
    lp = _job_dir(job_id) / "run.log"
    if not lp.exists():
        return ""
    return "\n".join(lp.read_text(errors="replace").splitlines()[-120:])[-n:]


def tool_job_status(job_id):
    st = _read_state(job_id)
    if st.get("status") == "unknown":
        return {"error": f"no job with id {job_id}"}
    return {
        "status": st.get("status"),
        "mode": st.get("mode"),
        "created": st.get("created", ""),
        "note": st.get("note", ""),
        "error": st.get("error", ""),
        "log_tail": _log_tail(job_id),
    }


def tool_list_results(job_id):
    res = _job_dir(job_id) / "avail_results"
    if not res.exists():
        return {"files": [], "message": "no results yet"}
    files = []
    for p in sorted(res.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "kb": round(p.stat().st_size / 1e3, 1)})
    return {"files": files}


def tool_benchmark_summary(job_id):
    res = _job_dir(job_id) / "avail_results"
    import pandas as pd
    for name in ["meta_benchmark_results.tsv", "meta_benchmark_3class_results.tsv"]:
        p = res / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, sep="\t")
            sub = df[df.subgroup == "all"].sort_values(
                "macro_auc", ascending=False).head(5)
            return {"table": name, "top_models": [
                {"model": r.model, "macro_auc": round(float(r.macro_auc), 4)}
                for r in sub.itertuples()]}
        except Exception as e:
            return {"error": f"could not parse {name}: {e}"}
    return {"error": "benchmark table not found (job may not be finished)"}


def tool_answer_results_question(job_id, question):
    from avail_ai.rag import build_context
    context, sources = build_context(question, _job_dir(job_id), BASE)
    if not context:
        res = _job_dir(job_id) / "avail_results"
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
        return {"error": "no results available for this job yet"}
    from avail_ai.llm import MultiLLMClient, build_clients
    llm = MultiLLMClient(build_clients())
    ans = llm.complete(
        "You answer questions about AVAIL-AI analysis results using ONLY the "
        "provided context. Cite table names / PMIDs that support each claim. "
        "If the context is insufficient, say so.",
        f"Context:\n{context}\n\nQuestion: {question}", max_tokens=900)
    return {"answer": ans.strip(), "sources": sources[:6]}


def tool_imaging_pipeline(job_id):
    jd = _job_dir(job_id)
    if not jd.exists():
        return {"error": "no such job"}
    mats = list((jd / "data").rglob("*_fc_100x100.npy"))
    tfm_dir = jd / "avail_results_timesfm"
    return {
        "protocol": ("Schaefer-100 parcellation (17 networks, same atlas as "
                     "the study ROI map). Classical layer: FC edges + "
                     "within/between-network summaries + ALFF. Optional "
                     "separate benchmark: Google TimesFM-3 zero-shot "
                     "forecast representations."),
        "fc_matrices_saved": len(mats),
        "timesfm_benchmark": "done" if tfm_dir.exists() else "not run",
    }


def tool_theory_notes(topic):
    found = []
    for name in ["THEORY_NOTES.md", "METHODS_NOTES.md"]:
        p = BASE / "experiments" / name
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        lines = text.splitlines()
        hits = [i for i, ln in enumerate(lines)
                if topic.lower() in ln.lower()]
        if not hits:
            continue
        start = max(0, hits[0] - 3)
        end = min(len(lines), hits[-1] + 12)
        found.append({"file": name, "excerpt": "\n".join(lines[start:end])[:2500]})
    if not found:
        return {"message": f"no theory notes mention '{topic}'"}
    return {"notes": found}


def _confirm(job_id, kind, label, payload):
    sess = load_session(job_id)
    sess["pending"] = {"kind": kind, "label": label, "payload": payload,
                       "ts": time.time()}
    save_session(job_id, sess)
    return {"needs_confirmation": True, "kind": kind, "label": label}


def tool_start_job(job_id, robustness=False, ci=False, missingness=False,
                   tuning=False, calibration=False, ordinal=False,
                   conformal=False):
    st = _read_state(job_id)
    if st.get("status") != "validated":
        return {"error": f"job is '{st.get('status')}', not validated; "
                         "it cannot be started now"}
    flags = {k: bool(v) for k, v in {
        "robustness": robustness, "ci": ci, "missingness": missingness,
        "tuning": tuning, "calibration": calibration,
        "ordinal": ordinal, "conformal": conformal}.items()}
    sel = [k for k, v in flags.items() if v] or ["pipeline only"]
    return _confirm(job_id, "start_job", job_id,
                    {"flags": flags, "summary": "comparisons: " + ", ".join(sel)})


def tool_run_comparison(job_id, name):
    if name not in COMPARISON_SCRIPTS:
        return {"error": f"unknown comparison '{name}'"}
    st = _read_state(job_id)
    if st.get("status") != "done":
        return {"error": f"job is '{st.get('status')}'; comparisons need a finished run"}
    return _confirm(job_id, "run_comparison", f"{name} on {job_id}",
                    {"name": name})


def tool_make_figure(job_id, figure):
    res = _job_dir(job_id) / "avail_results"
    if not res.exists() or not any(res.iterdir()):
        return {"error": "no results available for this job"}
    try:
        from avail_ai.figures_publication import (
            fig_benchmark, fig_biomarkers, fig_continuum, fig_shap)
        mapping = {"benchmark": fig_benchmark, "biomarkers": fig_biomarkers,
                   "continuum": fig_continuum, "shap": fig_shap}
        if figure == "all":
            for fn in (fig_benchmark, fig_biomarkers, fig_continuum, fig_shap):
                try:
                    fn(res)
                except Exception as e:
                    return {"partial": True,
                            "error": f"{fn.__name__}: {type(e).__name__}: {str(e)[:200]}"}
        else:
            mapping[figure](res)
        made = [p.name for p in sorted(res.iterdir())
                if p.suffix.lower() in (".png", ".pdf", ".tiff") and
                p.name.startswith(("fig", "Figure", "SHAP"))]
        return {"ok": True, "files": made[-12:]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}


# ---------------------------------------------------------------- execution
def _comparison_cmd(job_id, name):
    jd = _job_dir(job_id)
    data_root = jd / "data"
    for child in data_root.iterdir():
        if child.is_dir():
            data_root = child
            break
    out = jd / "avail_results"
    cmp_out = jd / "avail_comparisons"
    exp = BASE / "experiments"
    script = COMPARISON_SCRIPTS[name]
    if name == "robustness":
        return [PY, "-u", str(exp / script), "--data-root", str(data_root),
                "--out", str(cmp_out)]
    if name == "ci":
        return [PY, "-u", str(exp / script), str(out), "--meta-root", str(data_root)]
    if name == "missingness":
        return [PY, "-u", str(exp / script), "--data-root", str(data_root),
                "--out", str(cmp_out)]
    if name == "tuning":
        return [PY, "-u", str(exp / script), "--data-root", str(data_root),
                "--out", str(cmp_out), "--trials", "30"]
    if name == "calibration":
        return [PY, "-u", str(exp / script), "--data-root", str(data_root),
                "--out", str(cmp_out)]
    if name == "ordinal":
        return [PY, "-u", str(exp / script),
                "--base-preds", str(out / "base_predictions_generic.tsv"),
                "--loo", str(out / "loo_predictions.tsv"),
                "--out", str(cmp_out)]
    if name == "conformal":
        return [PY, "-u", str(exp / script), "--res-dir", str(out),
                "--out", str(cmp_out)]
    if name == "hybrid":
        return [PY, "-u", str(exp / script), "--data-root", str(data_root),
                "--out", str(cmp_out)]
    raise ValueError(name)


def _execute(job_id, kind, payload):
    jd = _job_dir(job_id)
    (jd / "avail_comparisons").mkdir(parents=True, exist_ok=True)
    if kind == "start_job":
        import queue as _q
        from webapp.app import job_queue, SERVER_RUN
        if not SERVER_RUN:
            return {"error": "server-side GPU runs are disabled on this instance"}
        flags = payload["flags"]
        _write_state(job_id, status="queued", **flags)
        job_queue.put(job_id)
        return {"ok": True, "message": "job queued on the server GPU"}
    if kind == "run_comparison":
        name = payload["name"]
        cmd = _comparison_cmd(job_id, name)
        logf = open(jd / "comparisons.log", "a")
        subprocess.Popen(cmd, cwd=str(BASE), stdout=logf, stderr=subprocess.STDOUT)
        _write_state(job_id, note=f"agent started comparison: {name}")
        return {"ok": True, "message": f"{name} comparison started in background",
                "log": str(jd / "comparisons.log")}
    return {"error": f"unknown pending kind {kind}"}


# ---------------------------------------------------------------- agent loop
TOOL_MAP = {
    "job_status": lambda a: tool_job_status(a["job_id"]),
    "list_results": lambda a: tool_list_results(a["job_id"]),
    "benchmark_summary": lambda a: tool_benchmark_summary(a["job_id"]),
    "answer_results_question": lambda a: tool_answer_results_question(
        a["job_id"], a["question"]),
    "start_job": lambda a: tool_start_job(
        a["job_id"], a.get("robustness", False), a.get("ci", False),
        a.get("missingness", False), a.get("tuning", False),
        a.get("calibration", False), a.get("ordinal", False),
        a.get("conformal", False)),
    "run_comparison": lambda a: tool_run_comparison(a["job_id"], a["name"]),
    "make_figure": lambda a: tool_make_figure(a["job_id"], a["figure"]),
    "theory_notes": lambda a: tool_theory_notes(a["topic"]),
    "imaging_pipeline": lambda a: tool_imaging_pipeline(a["job_id"]),
}


def _get_llm():
    from avail_ai.llm import MultiLLMClient, build_clients
    return MultiLLMClient(build_clients())


def _append_result_message(messages, call_id, content, name=None):
    m = {"role": "tool", "tool_call_id": call_id,
         "content": json.dumps(content, ensure_ascii=False)}
    if name:
        m["name"] = name
    messages.append(m)


def _sanitize(messages):
    out = []
    for m in messages:
        m2 = {k: v for k, v in m.items() if not k.startswith("_")}
        out.append(m2)
    return out


def _trim_messages(messages, max_total=16000, keep_last=14):
    """Keep provider token budgets small while preserving tool-call order.

    Providers require a tool-call turn to follow a user turn; trimming must
    never start the window on an assistant/tool message.
    """
    if not messages:
        return messages
    sysmsg = messages[0] if messages[0].get("role") == "system" else None
    rest = messages[1:] if sysmsg else messages

    def _from_user(ms):
        start = 1 if (ms and ms[0].get("role") == "system") else 0
        for i in range(start, len(ms)):
            if ms[i].get("role") == "user":
                return ([ms[0]] if start else []) + ms[i:]
        return ([ms[0]] if start else []) or ms

    tail = rest[-keep_last:]
    out = ([sysmsg] if sysmsg else []) + tail
    out = _from_user(out)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            continue
        m2 = dict(m)
        c = m2.get("content")
        if isinstance(c, str) and len(c) > 4000:
            m2["content"] = c[:4000] + " …[truncated]"
        out[i] = m2

    def _size(ms):
        return sum(len(str(x.get("content") or "")) for x in ms)
    while len(out) > 4 and _size(out) > max_total:
        head = out[:1] if out and out[0].get("role") == "system" else []
        body = out[1:] if head else out
        nxt = next((i for i, m in enumerate(body) if m.get("role") == "user"), None)
        if nxt is None or nxt == 0:
            break
        out = head + body[nxt:]
    return out


def run_agent(job_id, user_message):
    """One user turn of the agent. Returns (reply_text, steps, pending)."""
    job_id = _clean_job_id(job_id)
    if not job_id or not (UPLOADS / job_id).exists():
        return "No such job.", [], None
    sess = load_session(job_id)
    messages = sess["messages"]
    sys_note = (SYSTEM + f"\n\nCurrent job id: {job_id}. When the user refers to "
                f"'the job', 'my job', 'this job' or 'it', use this id for "
                f"job-scoped tools.")
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": sys_note}
    else:
        messages.insert(0, {"role": "system", "content": sys_note})
    steps = []
    pending = None

    if user_message.strip().lower() in ("confirm", "yes", "confirm now",
                                        "[confirm]", "approve"):
        pend = sess.get("pending")
        if not pend:
            return "Nothing pending to confirm.", [], None
        res = _execute(job_id, pend["kind"], pend["payload"])
        sess["pending"] = None
        ptc = pend.get("tool_call")
        if ptc:
            out = dict(res) if isinstance(res, dict) else {"result": res}
            if pend["kind"] == "start_job" and out.get("ok"):
                out["note"] = "job queued; use job_status to follow it"
            _append_result_message(messages, ptc.get("id", "t0"), out,
                                   name=ptc.get("name", pend["kind"]))
        else:
            messages.append({"role": "user",
                             "content": f"[system] Confirmed action finished: "
                                        f"{json.dumps(res)}"})
        steps.append({"tool": pend["kind"], "label": pend["label"], "result": res})
        save_session(job_id, sess)
    else:
        messages.append({"role": "user", "content": user_message})

    llm = _get_llm()
    for _ in range(MAX_ITERS):
        msg = llm.chat_tools(_sanitize(_trim_messages(messages)), TOOLS,
                             temperature=0.0, max_tokens=900)
        tcalls = msg.get("tool_calls") or []
        if not tcalls:
            reply = (msg.get("content") or "").strip() or "I have nothing to add."
            messages.append({"role": "assistant", "content": reply})
            save_session(job_id, sess)
            return reply, steps, sess.get("pending")
        messages.append(msg)
        pending_here = None
        for tc in tcalls:
            fn = tc.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if name in ("job_status", "list_results", "benchmark_summary",
                        "start_job", "make_figure", "run_comparison",
                        "answer_results_question"):
                jid = str(args.get("job_id") or "")
                if not jid or not (UPLOADS / _clean_job_id(jid)).exists():
                    args["job_id"] = job_id
            if name not in TOOL_MAP:
                res = {"error": f"unknown tool {name}"}
            else:
                try:
                    res = TOOL_MAP[name](args)
                except Exception as e:
                    res = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            tc_id = tc.get("id", "t0")
            if isinstance(res, dict) and res.get("needs_confirmation"):
                sess["pending"] = {"kind": res["kind"], "label": res["label"],
                                   "payload": args, "ts": time.time(),
                                   "tool_call": {"id": tc_id, "name": name}}
                _append_result_message(
                    messages, tc_id,
                    {"status": "awaiting user confirmation",
                     "label": res["label"]}, name=name)
                steps.append({"tool": name, "label": res["label"],
                              "result": {"pending": True}})
                pending_here = res["label"]
                continue
            _append_result_message(messages, tc_id, res, name=name)
            steps.append({"tool": name, "label": name, "result": res})
        if pending_here:
            save_session(job_id, sess)
            reply = (f"I need your confirmation for: **{pending_here}**. "
                     f"Click Confirm to proceed, or tell me to change the plan.")
            return reply, steps, sess["pending"]
    reply = "I stopped after several steps without a final answer. Please narrow the request."
    messages.append({"role": "assistant", "content": reply})
    save_session(job_id, sess)
    return reply, steps, sess.get("pending")
