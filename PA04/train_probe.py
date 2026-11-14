"""Frozen-feature experiments: logistic regression on ResNet18 features for every arm.

    python train_probe.py

Arms
    real{p}       IDID training crops with p % of the broken ones (p = 10, 25, 50, 100)
    real{p}+syn   the same crops plus the 420 synthetic crops
    syn-only      the 420 synthetic crops only

For test fold k, fold k+1 is the validation fold: the threshold is set so that at most 5 % of
its normal crops are flagged, and the test fold and Mendeley are scored at that threshold.
syn-only has no real validation data and uses 0.5.

Writes results/probe_runs.csv (one line per run), checkpoints/<arm>.npz and
results/domain_probe.json. Runs in a few seconds once the features exist.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CHECKPOINTS, FEATURES, FRACTIONS, N_FOLDS, RESULTS, SEEDS
from common import arm_name, fold_split, load_set, result_row, subsample_broken, threshold_from_normals, write_rows


def fit_probe(X, y):
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    return probe.fit(X, y)


def score(probe, X):
    return probe.predict_proba(X)[:, 1]     # P(broken)


def domain_probe(real, syn):
    """Can a linear model tell synthetic crops from real ones? (AUC on a held-out half)"""
    X = np.concatenate([real.X, syn.X])
    domain = np.concatenate([np.zeros(len(real.X), int), np.ones(len(syn.X), int)])
    X_tr, X_te, d_tr, d_te = train_test_split(X, domain, test_size=0.5, stratify=domain, random_state=0)
    auc = roc_auc_score(d_te, score(fit_probe(X_tr, d_tr), X_te))
    return {"task": "synthetic (1) vs real IDID (0) crops, normal and broken pooled",
            "train": len(d_tr), "test": len(d_te), "test_auc": round(float(auc), 4)}


if __name__ == "__main__":
    features = np.load(FEATURES)
    idid, mendeley, syn = (load_set(name, features=features) for name in ("idid", "mendeley", "synthetic"))
    rows = []
    saved = {}      # arm -> list of (fold, seed, threshold, probe)

    # synthetic only: one model, scored on each IDID test fold and on Mendeley
    probe = fit_probe(syn.X, syn.y)
    saved["syn-only"] = [(0, 0, 0.5, probe)]
    for fold in range(N_FOLDS):
        _, _, test = fold_split(idid, fold)
        rows.append(result_row("syn-only", fold, 0, 0, 0, len(syn.y), 0.5,
                               (idid.y[test], score(probe, idid.X[test])), (mendeley.y, score(probe, mendeley.X))))

    for fold in range(N_FOLDS):
        train, val, test = fold_split(idid, fold)
        val_normal = val[idid.y[val] == 0]
        for fraction in FRACTIONS:
            # with all broken crops kept the seed changes nothing, so one run per fold is enough
            for seed in (SEEDS if fraction < 1 else SEEDS[:1]):
                subset = subsample_broken(train, idid.y, fraction, seed)
                n_normal, n_broken = int((idid.y[subset] == 0).sum()), int((idid.y[subset] == 1).sum())
                for synthetic in (False, True):
                    X, y = idid.X[subset], idid.y[subset]
                    if synthetic:
                        X, y = np.concatenate([X, syn.X]), np.concatenate([y, syn.y])
                    arm = arm_name(fraction, synthetic)
                    probe = fit_probe(X, y)
                    threshold = threshold_from_normals(score(probe, idid.X[val_normal]))
                    saved.setdefault(arm, []).append((fold, seed, threshold, probe))
                    rows.append(result_row(arm, fold, seed, n_normal, n_broken, len(syn.y) if synthetic else 0, threshold,
                                           (idid.y[test], score(probe, idid.X[test])),
                                           (mendeley.y, score(probe, mendeley.X))))
        print(f"fold {fold} done")

    write_rows(RESULTS / "probe_runs.csv", rows)
    print(f"{len(rows)} lines in results/probe_runs.csv")

    # one checkpoint file per arm: scaler and regression weights of every run
    CHECKPOINTS.mkdir(exist_ok=True)
    for arm, runs in saved.items():
        np.savez(CHECKPOINTS / f"{arm}.npz",
                 fold=[r[0] for r in runs], seed=[r[1] for r in runs], threshold=[r[2] for r in runs],
                 mean=[r[3][0].mean_ for r in runs], scale=[r[3][0].scale_ for r in runs],
                 coef=[r[3][1].coef_[0] for r in runs], intercept=[r[3][1].intercept_[0] for r in runs])

    result = domain_probe(idid, syn)
    (RESULTS / "domain_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print("domain probe:", result)
