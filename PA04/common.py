"""Settings and helpers shared by the PA04 scripts."""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pa04_insulator_crops_v1"
FEATURES = ROOT / "data" / "features_resnet18.npz"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

# Experiment design, the same for the frozen-feature and the fine-tuned models.
N_FOLDS = 5
FRACTIONS = [0.10, 0.25, 0.50, 1.00]   # share of the real broken training crops that is kept
SEEDS = [0, 1, 2]                      # which broken crops are kept
MAX_FALSE_ALARM = 0.05                 # the threshold may flag at most 5 % of the validation normals
IMAGE_SIZE = 224

SETS = ["idid", "mendeley", "synthetic"]
LABELS = {"normal": 0, "broken": 1}    # flashover / ambiguous / shaded crops are not used


def read_labels(name, data=DATA):
    with open(Path(data) / name / "labels.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class CropSet:
    name: str
    files: np.ndarray        # crop paths relative to the set folder, e.g. crops/000000795_001.png
    paths: np.ndarray        # full paths of the same crops
    y: np.ndarray            # 0 normal, 1 broken
    fold: np.ndarray         # IDID fold 0-4, -1 for the other sets
    X: np.ndarray = None     # (n, 512) ResNet18 features, if a feature file was given


def load_set(name, data=DATA, features=None):
    """Normal and broken crops of one set, in labels.csv order."""
    rows = read_labels(name, data)
    keep = np.array([r["label"] in LABELS for r in rows])
    files = np.array([r["file"] for r in rows])
    y = np.array([LABELS.get(r["label"], -1) for r in rows])
    fold = np.array([int(r.get("fold") or -1) for r in rows])
    X = None
    if features is not None:
        assert list(features[f"{name}_files"]) == list(files), "feature file does not match labels.csv"
        X = features[f"{name}_X"][keep]
    paths = np.array([str(Path(data).resolve() / name / f) for f in files])
    return CropSet(name, files[keep], paths[keep], y[keep], fold[keep], X)


def fold_split(idid, test_fold):
    """Train / validation / test indices when `test_fold` is held out.

    The next fold is used for validation (to set the threshold) and the other three
    for training. Folds are grouped by source photo.
    """
    val_fold = (test_fold + 1) % N_FOLDS
    test = np.flatnonzero(idid.fold == test_fold)
    val = np.flatnonzero(idid.fold == val_fold)
    train = np.flatnonzero((idid.fold != test_fold) & (idid.fold != val_fold))
    return train, val, test


def subsample_broken(train, y, fraction, seed):
    """Keep all normal training crops and a random `fraction` of the broken ones."""
    normal = train[y[train] == 0]
    broken = train[y[train] == 1]
    n_keep = max(1, round(fraction * len(broken)))
    rng = np.random.default_rng(seed)
    kept = rng.choice(broken, size=n_keep, replace=False) if n_keep < len(broken) else broken
    return np.sort(np.concatenate([normal, kept]))


def arm_name(fraction, synthetic):
    return f"real{round(fraction * 100)}" + ("+syn" if synthetic else "")


def threshold_from_normals(normal_scores):
    """Smallest threshold that flags at most MAX_FALSE_ALARM of these normal crops."""
    return float(np.quantile(normal_scores, 1 - MAX_FALSE_ALARM, method="higher"))


def evaluate(y, scores, threshold):
    """Recall on broken crops, false-alarm rate on normal crops, and ROC AUC."""
    flagged = scores > threshold
    broken, normal = y == 1, y == 0
    return {
        "n_broken": int(broken.sum()),
        "n_normal": int(normal.sum()),
        "missed": int((broken & ~flagged).sum()),
        "false_alarms": int((normal & flagged).sum()),
        "recall": float(flagged[broken].mean()),
        "false_alarm": float(flagged[normal].mean()),
        "auc": float(roc_auc_score(y, scores)),
    }


def result_row(arm, fold, seed, n_normal, n_broken, n_syn, threshold, test, mendeley):
    """One line of results/*_runs.csv. `test` and `mendeley` are (labels, scores)."""
    row = {"arm": arm, "fold": fold, "seed": seed, "train_normal_real": n_normal,
           "train_broken_real": n_broken, "train_synthetic": n_syn, "threshold": threshold}
    for prefix, (y, scores) in (("idid", test), ("mendeley", mendeley)):
        for key, value in evaluate(y, scores, threshold).items():
            row[f"{prefix}_{key}"] = value
    return row


def write_rows(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
