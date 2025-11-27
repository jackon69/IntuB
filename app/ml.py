
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from app.models import IntubationRecord

FEATURES = [
    "age",
    "weight",
    "dtm",
    "dii",
    "mallampati",
    "stop_bang",
    "alganzouri",
]

def _record_to_features(rec: IntubationRecord) -> List[float]:
    vals: List[float] = []
    for name in FEATURES:
        v = getattr(rec, name)
        if v is None:
            vals.append(np.nan)
        else:
            vals.append(float(v))
    return vals

def train_logistic_model() -> Tuple[LogisticRegression, Dict[str, Any]]:
    records = IntubationRecord.query.all()
    if len(records) < 20:
        raise RuntimeError("Not enough records to train a model (need at least 20).")

    X = np.array([_record_to_features(r) for r in records], dtype=float)
    y = np.array([1 if r.difficult_binary else 0 for r in records], dtype=int)

    # simple imputation: replace NaNs with column means
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "n_samples": int(len(records)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "class_balance": {
            "easy": int((y == 0).sum()),
            "difficult": int((y == 1).sum()),
        },
        "feature_names": FEATURES,
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }

    return model, metrics
