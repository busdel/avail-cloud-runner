# -*- coding: utf-8 -*-
"""AVAIL-AI Advisor: LLM-driven model selection, hyperparameter guidance, and
English interpretation of results.

The advisor never runs the models itself; it reasons over the metrics you feed
it and returns structured recommendations (JSON) that your pipeline can act on.
"""
import json
import pandas as pd


class Advisor:
    def __init__(self, llm):
        self.llm = llm

    def recommend_model(self, results_df, primary_metric="macro_auc",
                        subgroup="all", objective="maximize"):
        """Recommend the best meta-learner from a results table and explain why.

        results_df: columns must include 'model', 'subgroup', and the metric.
        """
        sub = results_df[results_df["subgroup"] == subgroup]
        ranked = sub.sort_values(primary_metric, ascending=(objective != "maximize"))
        table = ranked[["model", primary_metric]].to_csv(index=False)
        user = (
            f"Benchmark results (metric: {primary_metric}, subgroup: {subgroup}, "
            f"objective: {objective}):\n\n{table}\n\n"
            "Recommend the single best model for small-sample multimodal clinical data, "
            "give a one-sentence statistical rationale, and flag any caveat "
            "(e.g. overfitting risk for deep models at small n)."
        )
        schema = ('{"recommended_model": str, "runner_up": str, '
                  '"rationale": str, "caveat": str}')
        out = self.llm.complete_json(
            "You are a rigorous ML research advisor for clinical multimodal papers.",
            user + "\n\nReturn JSON with this schema: " + schema)
        return out

    def suggest_hyperparams(self, space, history, n=1):
        """Propose the next hyperparameter configuration(s) to evaluate.

        space : dict {param: [choices]} (a small discrete grid).
        history : list of (params_dict, metric) tuples already evaluated.
        """
        hist_txt = "\n".join(f"- {json.dumps(p)} -> {m}" for p, m in history)
        space_txt = json.dumps(space)
        user = (
            f"Hyperparameter search space: {space_txt}\n\n"
            f"Already evaluated (config -> metric, higher is better):\n{hist_txt or '(none)'}\n\n"
            f"Propose {n} untried configuration(s) from the space most likely to improve "
            "the metric, balancing exploration and exploitation."
        )
        schema = '{"configs": [{"config": {param: choice}, "why": str}]}'
        out = self.llm.complete_json(
            "You are a hyperparameter optimization expert using Bayesian reasoning "
            "over a small grid. Only propose values present in the given space.",
            user + "\n\nReturn JSON with this schema: " + schema)
        return out.get("configs", [])

    def interpret(self, summary_text):
        """Write a concise English results paragraph for a manuscript."""
        out = self.llm.complete(
            "You write Results sections for high-impact clinical neuroinformatics journals.",
            f"Summarize these results into a single rigorous paragraph:\n\n{summary_text}")
        return out.strip()
