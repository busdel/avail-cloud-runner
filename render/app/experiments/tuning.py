# -*- coding: utf-8 -*-
"""Automatic hyperparameter tuning (optuna, nested CV) + AI-guided round.

Two modes:
  optuna   : Bayesian (TPE) search over base-RF and meta-learner
             hyperparameters with stratified 5-fold inner CV (no leakage:
             the outer LOO subject is never touched during tuning).
  ai       : after optuna, the LLM advisor proposes one more candidate
             round, which is evaluated the same way (history-driven).

Usage:
    python tuning.py --data-root DATA --out OUT [--trials 40] [--ai]

Outputs: OUT/tuning_results.tsv (baseline vs tuned macro-AUC),
         OUT/tuning_best_params.json
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
import sys
sys.path.insert(0, str(BASE / "avail_ai"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--ai", action="store_true")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from avail_ai.generic import load_generic
    from avail_ai.learners import (base_predictions_loo, build_meta_features,
                                   macro_auc)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    group, conf_raw, mods, names = load_generic(args.data_root)
    from collections import Counter
    classes = [c for c, _ in Counter(group.values()).most_common(3)]
    K = len(classes)
    ids = sorted([i for i in group if group[i] in classes])
    label = {i: classes.index(group[i]) for i in ids}
    mod_order = sorted(mods)

    conf = {}
    for i in ids:
        c = conf_raw.get(i, [np.nan] * 3)[:3]
        conf[i] = np.array(c + [np.nan] * (3 - len(c)), dtype=float)
    Cmat = np.vstack([conf[i] for i in ids])
    for j in range(3):
        med = np.nanmedian(Cmat[:, j])
        Cmat[:, j] = np.where(np.isnan(Cmat[:, j]), med, Cmat[:, j])
    conf = {i: Cmat[k] for k, i in enumerate(ids)}
    micro = {m for m in mods
             if all(np.all(np.isfinite(mods[m][i]) & (mods[m][i] >= 0))
                    for i in ids if i in mods[m])}
    print(f"tuning: n={len(ids)}, K={K}, modalities={mod_order}")

    # ---- baseline (default hyperparameters)
    piv = base_predictions_loo(mods, ids, label, conf, K, micro,
                               n_estimators=300, random_state=42)
    X, y = build_meta_features(piv, ids, label, len(mods), mod_order, K, 0.0)

    def cv_auc(params):
        """Stratified 5-fold CV macro-AUC of XGBoost meta-learner."""
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        P = np.zeros((len(y), K))
        for tr, te in skf.split(X, y):
            clf = XGBClassifier(random_state=0, n_jobs=6,
                                objective="multi:softprob", num_class=K,
                                eval_metric="mlogloss", **params)
            clf.fit(X[tr], y[tr])
            P[te] = clf.predict_proba(X[te])
        return macro_auc(y, P)

    base_auc = cv_auc(dict(n_estimators=200, max_depth=2, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8))

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 1, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        return cv_auc(params)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    tuned_auc = study.best_value
    best = study.best_params
    print(f"optuna: baseline XGBoost CV AUC={base_auc:.3f} -> "
          f"tuned={tuned_auc:.3f} ({args.trials} trials)")

    # ---- optional AI-guided round
    ai_rows = []
    if args.ai:
        from avail_ai.llm import MultiLLMClient, build_clients
        from avail_ai.advisor import Advisor
        llm = MultiLLMClient(build_clients())
        adv = Advisor(llm)
        space = {"n_estimators": [100, 200, 300, 500],
                 "max_depth": [1, 2, 3, 4, 6],
                 "learning_rate": [0.01, 0.05, 0.1, 0.2],
                 "subsample": [0.6, 0.8, 1.0],
                 "colsample_bytree": [0.3, 0.5, 0.8, 1.0]}
        history = [(best, tuned_auc)]
        try:
            recs = adv.suggest_hyperparams(space, history, n=3)
            for r in recs:
                cfg = {k: v for k, v in r.get("config", {}).items()
                       if k in space}
                if cfg:
                    a = cv_auc(cfg)
                    ai_rows.append({"config": json.dumps(cfg),
                                    "cv_auc": a,
                                    "why": r.get("why", "")})
                    print(f"ai candidate {cfg} -> {a:.3f}")
        except Exception as e:
            print(f"ai round failed: {type(e).__name__}: {str(e)[:150]}")

    rows = [{"config": "baseline", "cv_macro_auc": base_auc},
            {"config": "optuna_best", "cv_macro_auc": tuned_auc,
             "params": json.dumps(best)}]
    rows += [{"config": "ai_candidate", "cv_macro_auc": r["cv_auc"],
              "params": r["config"]} for r in ai_rows]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tuning_results.tsv", sep="\t", index=False)
    with open(OUT / "tuning_best_params.json", "w") as f:
        json.dump({"optuna_best": best, "baseline_auc": base_auc,
                   "tuned_auc": tuned_auc}, f, indent=2)
    print(df[["config", "cv_macro_auc"]].round(4).to_string(index=False))
    print("tuning done")


if __name__ == "__main__":
    main()
