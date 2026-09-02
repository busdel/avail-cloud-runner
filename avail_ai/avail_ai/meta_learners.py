# -*- coding: utf-8 -*-
"""AVAIL-AI meta-learners (neural + tabular) for the LOO meta-benchmark.

Extracted from the AVAIL_meta_benchmark notebooks so that both the notebook
flow and the generic data pipeline (run_generic.py / web uploader) share the
exact same model implementations.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # optional: neural/tabular-deep models are skipped
    torch = None
    nn = None
    HAS_TORCH = False

MODEL_NAMES = ['SimpleMean', 'WSimpleMean', 'ElasticNetLR', 'MLP', 'MoE', 'GNN',
               'Transformer', 'XGBoost', 'TabNet', 'TabPFN', 'TabICL']


def _device():
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device('cuda')
    return 'cpu'


class _MLP(nn.Module if HAS_TORCH else object):
    def __init__(self, d_in, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 24), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(24, 12), nn.ReLU(), nn.Linear(12, K))

    def forward(self, x):
        return self.net(x)


class _MoE(nn.Module if HAS_TORCH else object):
    def __init__(self, n_mods, K):
        super().__init__()
        self.n_mods = n_mods
        self.K = K
        per = K + 1
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(per, 12), nn.ReLU(), nn.Linear(12, K))
            for _ in range(n_mods)])
        self.gate = nn.Sequential(nn.Linear(n_mods * per, 24), nn.ReLU(),
                                  nn.Linear(24, n_mods))

    def forward(self, x):
        B, _ = x.shape
        n = self.n_mods
        per = self.K + 1
        flags = torch.stack([x[:, c * per + self.K] for c in range(n)], dim=1)
        exp_out = torch.stack([
            self.experts[i](x[:, i * per:(i + 1) * per]) for i in range(n)], dim=1)
        g = torch.softmax(self.gate(x), dim=1) * flags
        g = g / g.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return (g.unsqueeze(-1) * exp_out).sum(dim=1)


class _GNN(nn.Module if HAS_TORCH else object):
    def __init__(self, n_nodes=5, K=3, d=8):
        super().__init__()
        self.K = K
        self.embed = nn.Linear(K + 1, d)
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, d)
        self.readout = nn.Sequential(nn.Linear(2 * d, 12), nn.ReLU(), nn.Linear(12, K))
        A = np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)
        A = A / A.sum(axis=1, keepdims=True)
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.n_nodes = n_nodes

    def forward(self, x):
        B, _ = x.shape
        n = self.n_nodes
        per = self.K + 1
        h = self.embed(x.view(B, n, per))
        h = torch.relu(torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), self.w1(h)))
        h = torch.relu(torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), self.w2(h)))
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.readout(pooled)


class _Transformer(nn.Module if HAS_TORCH else object):
    def __init__(self, n_nodes=5, K=3, d=16, nhead=2, nlayers=1):
        super().__init__()
        self.embed = nn.Linear(K + 1, d)
        self.pos = nn.Parameter(torch.randn(1, n_nodes, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                           dim_feedforward=48, dropout=0.2,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.Linear(d, 12), nn.ReLU(), nn.Linear(12, K))
        self.n_nodes = n_nodes
        self.K = K

    def forward(self, x):
        B, _ = x.shape
        n = self.n_nodes
        per = self.K + 1
        tok = self.embed(x.view(B, n, per)) + self.pos
        out = self.enc(tok)
        return self.head(out.mean(dim=1))


def train_nn(model, X, y, seed, K, device=None, epochs=300, patience=30,
             lr=1e-3, wd=1e-3):
    if not HAS_TORCH:
        raise RuntimeError("torch not installed")
    device = device or _device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    counts = np.bincount(y, minlength=K).astype(float)
    w = torch.tensor(len(y) / (K * counts), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rng = np.random.RandomState(seed)
    val = np.concatenate([
        rng.choice(np.where(y == c)[0],
                   size=max(1, int(0.2 * (y == c).sum())), replace=False)
        for c in range(K)])
    tr = np.array([i for i in range(len(y)) if i not in set(val)])
    best = np.inf
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt[tr]), yt[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xt[val]), yt[val]).item()
        if vl < best - 1e-4:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_nn(model, X, device=None):
    if not HAS_TORCH:
        raise RuntimeError("torch not installed")
    device = device or _device()
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float32, device=device))
        return torch.softmax(logit, dim=1).cpu().numpy()


def augment(X, y, n_mods, K, aug_p=0.3, random_state=42):
    Xa, ya = [X], [y]
    rng = np.random.RandomState(random_state)
    Xn = X.copy()
    per = K + 1
    for r in range(X.shape[0]):
        for c in range(n_mods):
            base = c * per
            if Xn[r, base + K] == 1 and rng.rand() < aug_p:
                Xn[r, base:base + K] = 0.0
                Xn[r, base + K] = 0.0
    Xa.append(Xn)
    ya.append(y)
    return np.vstack(Xa), np.concatenate(ya)


def _tabular_predict(name, Xtr, ytr, Xte, K):
    dev = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'
    if not HAS_TORCH and name in ('TabPFN', 'TabICL'):
        raise RuntimeError("torch not installed")

    def _jit(X):
        Xa = np.asarray(X, dtype=float).copy()
        const = Xa.std(axis=0) == 0
        if const.any():
            rng = np.random.RandomState(0)
            Xa[:, const] += rng.normal(0, 1e-4, (Xa.shape[0], int(const.sum())))
        return Xa

    if name == 'XGBoost':
        from xgboost import XGBClassifier
        kw = dict(n_estimators=200, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1)
        if K == 2:
            clf = XGBClassifier(eval_metric='logloss', **kw)
        else:
            clf = XGBClassifier(objective='multi:softprob', num_class=K,
                                eval_metric='mlogloss', **kw)
        clf.fit(Xtr, ytr)
    elif name == 'TabNet':
        from pytorch_tabnet.tab_model import TabNetClassifier
        clf = TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=0,
                               verbose=0, device_name='cpu')
        clf.fit(Xtr.astype(np.float32), np.asarray(ytr).astype(np.int64),
                max_epochs=50, batch_size=32)
    elif name == 'TabPFN':
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=dev, n_estimators=4, random_state=0)
        clf.fit(_jit(Xtr), ytr)
    elif name == 'TabICL':
        from tabicl import TabICLClassifier
        from sklearn.multiclass import OneVsRestClassifier
        base_tabicl = TabICLClassifier(random_state=0, device=dev)
        clf = base_tabicl if K == 2 else OneVsRestClassifier(base_tabicl)
        clf.fit(_jit(Xtr), ytr)
    else:
        raise ValueError(name)
    p = np.asarray(clf.predict_proba(Xte))
    if p.ndim == 3:
        p = p[0]
    return np.asarray(p, dtype=float).reshape(1, -1)[0]


def _modality_weights(Xtr, ytr, n_mods, K):
    per = K + 1
    w = []
    for m in range(n_mods):
        base = m * per
        obs = Xtr[:, base + K] == 1
        if obs.sum() >= 10:
            P = Xtr[obs, base:base + K]
            aucs = []
            for c in range(K):
                yc = (ytr[obs] == c).astype(int)
                if len(np.unique(yc)) == 2:
                    aucs.append(roc_auc_score(yc, P[:, c]))
            auc = np.mean(aucs) if aucs else 0.5
            w.append(max(auc, 0.5))
        else:
            w.append(0.5)
    return np.array(w)


def run_meta_loo(X, y, n_mods, K, model_names=MODEL_NAMES, device=None,
                 random_state=42, n_seeds=10, augment_flag=True, aug_p=0.3,
                 verbose=True):
    """LOO meta-learner benchmark. Returns {model_name: (n, K) probabilities}."""
    device = device or _device()
    n = len(y)
    preds = {m: np.zeros((n, K)) for m in model_names}
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        te = [fold]
        Xtr, ytr = X[tr], y[tr]
        Xte = X[te]
        if augment_flag:
            Xtr, ytr = augment(Xtr, ytr, n_mods, K, aug_p=aug_p,
                               random_state=random_state)

        per = K + 1
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(n_mods):
            base = c * per
            if Xte[0, base + K] == 1:
                acc += Xte[0, base:base + K]
                wsum += 1.0
        preds['SimpleMean'][fold] = acc / wsum if wsum > 0 else np.full(K, 1.0 / K)

        w = _modality_weights(Xtr, ytr, n_mods, K)
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(n_mods):
            base = c * per
            if Xte[0, base + K] == 1:
                acc += w[c] * Xte[0, base:base + K]
                wsum += w[c]
        preds['WSimpleMean'][fold] = acc / wsum if wsum > 0 else np.full(K, 1.0 / K)

        try:
            lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5,
                                    C=0.3, max_iter=5000, class_weight='balanced',
                                    random_state=0)
            lr.fit(Xtr, ytr)
            preds['ElasticNetLR'][fold] = lr.predict_proba(Xte)[0]
        except Exception:
            preds['ElasticNetLR'][fold] = preds['SimpleMean'][fold]

        for mname, factory in [('MLP', lambda: _MLP(X.shape[1], K)),
                               ('MoE', lambda: _MoE(n_mods, K)),
                               ('GNN', lambda: _GNN(n_nodes=n_mods, K=K)),
                               ('Transformer', lambda: _Transformer(n_nodes=n_mods, K=K))]:
            if not HAS_TORCH:
                preds[mname][fold] = preds['SimpleMean'][fold]
                continue
            ps = []
            for s in range(n_seeds):
                try:
                    mdl = factory()
                    train_nn(mdl, Xtr, ytr, seed=random_state + s, K=K, device=device)
                    ps.append(predict_nn(mdl, Xte, device=device)[0])
                except Exception:
                    ps.append(np.full(K, 1.0 / K))
            preds[mname][fold] = np.mean(ps, axis=0)

        for mname in ['XGBoost', 'TabNet', 'TabPFN', 'TabICL']:
            try:
                pr = _tabular_predict(mname, Xtr, ytr, Xte, K)
                preds[mname][fold] = pr if pr.shape[0] == K else np.full(K, 1.0 / K)
            except Exception as e:
                if fold == 0 and verbose:
                    print(f'    [{mname}] uyarisi: {type(e).__name__}: {str(e)[:120]}')
                preds[mname][fold] = preds['SimpleMean'][fold]

        if verbose and (fold + 1) % 25 == 0:
            print(f'    meta fold {fold + 1}/{n}')
    return preds
