"""Score the fine-tuned runs from their prediction files.

    mkdir -p data && tar xzf results/finetune_predictions.tar.gz -C data
    python evaluate_finetune.py                      # the released runs
    python evaluate_finetune.py --runs OUT           # your own runs from train_finetune.py

Uses the same rule as train_probe.py: the threshold flags at most 5 % of the validation fold's
normal crops (syn-only: 0.5). Writes results/finetune_runs.csv with the same columns as
results/probe_runs.csv.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import N_FOLDS, RESULTS, ROOT, read_labels, result_row, threshold_from_normals, write_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "data" / "pa04_finetune_runs_v1")
    args = parser.parse_args()
    fold_of = {"idid/" + r["file"]: int(r["fold"]) for r in read_labels("idid")}

    rows = []
    for info_file in sorted(args.runs.glob("*/fold*_seed*/run.json")):
        info = json.loads(info_file.read_text())
        if info.get("smoke"):
            continue
        parts = defaultdict(lambda: {"file": [], "y": [], "p": []})
        with open(info_file.parent / "predictions.csv", newline="") as f:
            for r in csv.DictReader(f):
                parts[r["part"]]["file"].append(r["file"])
                parts[r["part"]]["y"].append(int(r["label"]))
                parts[r["part"]]["p"].append(float(r["p_broken"]))
        parts = {k: {"file": v["file"], "y": np.array(v["y"]), "p": np.array(v["p"])} for k, v in parts.items()}
        mendeley = (parts["mendeley"]["y"], parts["mendeley"]["p"])
        n_syn = 420 if info["arm"].endswith("syn") else 0
        counts = (info["train_normal_real"], info["train_broken_real"], n_syn)

        if info["arm"] == "syn-only":       # one model per seed, scored on each IDID fold
            y, p = parts["idid"]["y"], parts["idid"]["p"]
            folds = np.array([fold_of[f] for f in parts["idid"]["file"]])
            for fold in range(N_FOLDS):
                rows.append(result_row("syn-only", fold, info["seed"], *counts, 0.5,
                                       (y[folds == fold], p[folds == fold]), mendeley))
        else:
            val = parts["val"]
            threshold = threshold_from_normals(val["p"][val["y"] == 0])
            rows.append(result_row(info["arm"], info["fold"], info["seed"], *counts, threshold,
                                   (parts["test"]["y"], parts["test"]["p"]), mendeley))

    write_rows(RESULTS / "finetune_runs.csv", rows)
    print(f"{len(rows)} lines in results/finetune_runs.csv")
