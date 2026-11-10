"""Model registry. Every model is a full scikit-learn Pipeline: [feature selector] -> preprocessing -> classifier,
so cross-validation refits every step on the training fold only."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import TopKFeatureSelector, make_preprocessor
from .torch_models import TORCH_AVAILABLE, TabNetWrapper, TorchTabularClassifier

CLASSICAL = ["logreg", "svm", "rf", "xgb"]
NEURAL = ["dense_paper", "conv1d", "tabnet"]
DISPLAY = {"logreg": "Logistic regression", "svm": "SVM (RBF)", "rf": "Random forest",
           "xgb": "XGBoost", "dense_paper": "Dense NN (base paper)", "conv1d": "Conv1D CNN",
           "tabnet": "TabNet"}


def available_models(cfg_list: list[str]) -> list[str]:
    out = []
    for m in cfg_list:
        if m in ("dense_paper", "conv1d") and not TORCH_AVAILABLE:
            print(f"[warn] PyTorch missing -> skipping {m}")
            continue
        if m == "tabnet":
            try:
                import pytorch_tabnet  # noqa: F401
            except Exception:  # noqa: BLE001
                print("[warn] pytorch-tabnet missing -> skipping tabnet")
                continue
        out.append(m)
    return out


def make_model(name: str, cfg: dict, seed: int = 0, top_k: int | None = None,
               missing_indicator: bool = False) -> Pipeline:
    t = cfg["torch"]
    pre_kind = "onehot"
    if name == "logreg":
        clf = LogisticRegression(C=0.5, max_iter=5000, random_state=seed)
    elif name == "svm":
        # Platt scaling via cross-fitted calibration (SVC(probability=True) is deprecated in scikit-learn >= 1.9)
        clf = CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", random_state=seed),
                                     method="sigmoid", cv=5, ensemble=False)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=1)
    elif name == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                            random_state=seed, n_jobs=1, verbosity=0)
    elif name == "dense_paper":
        pre_kind = "ordinal"
        clf = TorchTabularClassifier(arch="dense", hidden=(128, 64, 32, 64, 32), dropout=0.2,
                                     epochs=t["epochs"], patience=t["patience"], lr=t["lr"],
                                     batch_size=t["batch_size"], weight_decay=t["weight_decay"],
                                     val_fraction=t["val_fraction"], device=t["device"], random_state=seed)
    elif name == "conv1d":
        pre_kind = "ordinal"
        clf = TorchTabularClassifier(arch="conv1d", dropout=0.25, epochs=t["epochs"], patience=t["patience"],
                                     lr=t["lr"] / 2, batch_size=t["batch_size"], weight_decay=t["weight_decay"],
                                     val_fraction=t["val_fraction"], device=t["device"], random_state=seed)
    elif name == "tabnet":
        pre_kind = "ordinal"
        clf = TabNetWrapper(device=t["device"], random_state=seed)
    else:
        raise ValueError(f"unknown model {name}")
    steps = []
    if top_k:
        steps.append(("select", TopKFeatureSelector(k=top_k, random_state=seed)))
    steps += [("pre", make_preprocessor(pre_kind, missing_indicator)), ("clf", clf)]
    return Pipeline(steps)


def make_paper_replica(cfg: dict, seed: int = 42) -> Pipeline:
    """The base paper's training protocol: dense 128-64-32-64-32, Adam, batch 64, 2000 epochs,
    best epoch chosen by validation accuracy (no early stopping)."""
    t = cfg["torch"]
    clf = TorchTabularClassifier(arch="dense", hidden=(128, 64, 32, 64, 32), dropout=0.2,
                                 epochs=t["paper_epochs"], patience=None, lr=1e-3,
                                 batch_size=t["paper_batch_size"], weight_decay=0.0,
                                 val_fraction=0.10, select_by="val_acc", device=t["device"],
                                 random_state=seed)
    return Pipeline([("pre", make_preprocessor("ordinal")), ("clf", clf)])
