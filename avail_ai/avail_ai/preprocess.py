# -*- coding: utf-8 -*-
"""AVAIL-AI preprocessing: transforms, feature selectors, confounder handling."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests
import numpy as np


class CLRTransformer(BaseEstimator, TransformerMixin):
    """Centered log-ratio transform with per-feature zero replacement.

    Works for compositional data (16S counts) AND non-compositional
    abundance-frequency data (shotgun MGS), because the zero-replacement
    delta is adapted to each feature's own scale.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.astype(np.float64)
        out = np.zeros_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            nz = col[col > 0]
            delta = nz.min() / 2.0 if len(nz) else 1e-9
            out[:, j] = np.where(col == 0, delta, col)
        gm = np.log(out).mean(axis=1, keepdims=True)
        return np.log(out) - gm


class NetworkFeatureExtractor(BaseEstimator, TransformerMixin):
    """Variance-based top-k feature filter (for connectivity matrices)."""
    def __init__(self, keep_top_k_raw=50):
        self.keep_top_k_raw = keep_top_k_raw

    def fit(self, X, y=None):
        self.top_idx_ = np.argsort(np.var(X, axis=0))[::-1][:self.keep_top_k_raw]
        return self

    def transform(self, X):
        return X[:, self.top_idx_]


class ANOVAFDRSelector(BaseEstimator, TransformerMixin):
    """ANOVA F-test + Benjamini-Hochberg FDR feature selection."""
    def __init__(self, alpha=0.05, min_features=10):
        self.alpha = alpha
        self.min_features = min_features

    def fit(self, X, y):
        self.support_mask_ = np.ones(X.shape[1], dtype=bool)
        try:
            _, p = f_classif(X, y)
            reject, _, _, _ = multipletests(p, method='fdr_bh', alpha=self.alpha)
            self.support_mask_ = reject
            if self.support_mask_.sum() < self.min_features:
                self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
                self.support_mask_[np.argsort(p)[:self.min_features]] = True
        except Exception:
            pass
        return self

    def transform(self, X):
        return X[:, self.support_mask_]


class RFImportanceCounter(BaseEstimator, TransformerMixin):
    """Random-forest importance top-n selector (multiclass-safe)."""
    def __init__(self, top_n=15):
        self.top_n = top_n

    def fit(self, X, y):
        imp = np.zeros(X.shape[1])
        for tr, _ in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
            m = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
            m.fit(X[tr], y[tr])
            imp += m.feature_importances_
        self.top_idx_ = np.argsort(imp)[::-1][:self.top_n]
        return self

    def transform(self, X):
        return X[:, self.top_idx_]


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """LassoCV-based feature selector (binary targets only)."""
    def __init__(self, max_features=30):
        self.max_features = max_features

    def fit(self, X, y):
        self.support_mask_ = np.ones(X.shape[1], dtype=bool)
        try:
            from sklearn.linear_model import LassoCV
            m = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X, y)
            nz = np.abs(m.coef_) > 1e-5
            if nz.sum() > 0:
                if nz.sum() <= self.max_features:
                    self.support_mask_ = nz
                else:
                    self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
                    self.support_mask_[np.argsort(np.abs(m.coef_))[::-1][:self.max_features]] = True
        except Exception:
            pass
        return self

    def transform(self, X):
        return X[:, self.support_mask_]


class ConfounderResidualizer(BaseEstimator, TransformerMixin):
    """Regress confounders (age/sex/education) out of features via OLS."""
    def fit(self, X, C):
        C_ = np.column_stack([np.ones(C.shape[0]), C])
        self.coef_ = np.linalg.lstsq(C_, X, rcond=None)[0]
        return self

    def transform(self, X, C):
        return X - np.column_stack([np.ones(C.shape[0]), C]).dot(self.coef_)


def preproc(X_tr, y_tr, C_tr, is_microbiome, residualize=False):
    """Fold-internal preprocessing.

    - microbiome: CLR -> RF-importance top-100
    - fMRI: variance top-50 -> MinMax -> ANOVA-FDR -> RF-importance top-15
    - optional confounder residualization, then standardization.
    """
    if is_microbiome:
        sel = CLRTransformer()
        Xt = sel.transform(X_tr)
        sel2 = RFImportanceCounter(top_n=100)
        Xt = sel2.fit_transform(Xt, y_tr)
    else:
        sel = Pipeline([('feat', NetworkFeatureExtractor()), ('scale', MinMaxScaler()),
                        ('anova', ANOVAFDRSelector()), ('rf', RFImportanceCounter(top_n=15))])
        Xt = sel.fit_transform(X_tr, y_tr)
        sel2 = None
    res = None
    if residualize:
        res = ConfounderResidualizer().fit(Xt, C_tr)
        Xt = res.transform(Xt, C_tr)
    sc = StandardScaler().fit(Xt)
    return sel, sel2, res, sc, sc.transform(Xt)


def apply_preproc(sel, sel2, res, sc, X_te, C_te):
    Xt = sel.transform(X_te)
    if sel2 is not None:
        Xt = sel2.transform(Xt)
    if res is not None:
        Xt = res.transform(Xt, C_te)
    return sc.transform(Xt)
