"""Summary tables from the run files in results/.

    python make_report.py

Reads results/probe_runs.csv (and results/finetune_runs.csv if it exists) and writes
results/summary.md: mean and SD of every metric per arm, and the paired differences
between each arm with and without the synthetic crops.
"""
import csv
import json
from collections import defaultdict
from statistics import mean, stdev

from common import RESULTS

METRICS = ["idid_recall", "idid_false_alarm", "idid_auc", "mendeley_recall", "mendeley_false_alarm", "mendeley_auc"]
PERCENTS = [10, 25, 50, 100]
ARMS = ["syn-only"] + [f"real{p}{s}" for p in PERCENTS for s in ("", "+syn")]


def load_runs(name):
    runs = defaultdict(list)
    with open(RESULTS / name, newline="") as f:
        for row in csv.DictReader(f):
            runs[row["arm"]].append({k: (v if k == "arm" else float(v)) for k, v in row.items()})
    return runs


def mean_sd(values):
    return mean(values), (stdev(values) if len(values) > 1 else 0.0)


def summarize(runs):
    """Mean and SD of every metric per arm."""
    rows = []
    for arm in ARMS:
        r = runs[arm]
        row = {"arm": arm, "runs": len(r), "train_broken_real": mean(x["train_broken_real"] for x in r)}
        for m in METRICS:
            row[m + "_mean"], row[m + "_sd"] = mean_sd([x[m] for x in r])
        rows.append(row)
    return rows


def paired_differences(runs):
    """real{p}+syn minus real{p}, paired by fold and seed."""
    out = []
    for p in PERCENTS:
        base = {(x["fold"], x["seed"]): x for x in runs[f"real{p}"]}
        for metric in ("idid_recall", "idid_auc", "mendeley_auc"):
            d = [x[metric] - base[(x["fold"], x["seed"])][metric] for x in runs[f"real{p}+syn"]]
            m, s = mean_sd(d)
            out.append({"percent": p, "metric": metric, "mean": m, "sd": s, "better": sum(v > 1e-12 for v in d),
                        "same": sum(abs(v) <= 1e-12 for v in d), "worse": sum(v < -1e-12 for v in d)})
    return out


def summary_section(title, note, rows, diffs):
    lines = [f"## {title}", "", note, "",
             "| Arm | Runs | Real broken | IDID recall | IDID false alarm | IDID AUC | Mendeley recall | Mendeley false alarm | Mendeley AUC |",
             "|---|---:|---:|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{100 * r[m + '_mean']:.1f} ± {100 * r[m + '_sd']:.1f}" if "auc" not in m
                 else f"{r[m + '_mean']:.3f} ± {r[m + '_sd']:.3f}" for m in METRICS]
        lines.append(f"| {r['arm']} | {r['runs']} | {r['train_broken_real']:.0f} | " + " | ".join(cells) + " |")
    lines += ["", "Adding the synthetic crops, paired difference (real+syn − real):", "",
              "| Real broken kept | Metric | Mean Δ | SD | Better | Same | Worse |", "|---:|---|---:|---:|---:|---:|---:|"]
    for d in diffs:
        scale, unit = (100, " pp") if d["metric"].endswith("recall") else (1, "")
        lines.append(f"| {d['percent']} % | {d['metric']} | {scale * d['mean']:+.3f}{unit} | {scale * d['sd']:.3f} | "
                     f"{d['better']} | {d['same']} | {d['worse']} |")
    return lines + [""]


if __name__ == "__main__":
    probe_runs = load_runs("probe_runs.csv")
    intro = "Mean ± SD over runs. Recall and false alarm in %, at the threshold fixed on the validation fold."
    md = ["# Results", "", intro, ""]
    md += summary_section("Frozen ResNet18 features + logistic regression",
                          "5 folds × 3 seeds; 5 runs for the 100 % arms and syn-only.",
                          summarize(probe_runs), paired_differences(probe_runs))
    domain = json.loads((RESULTS / "domain_probe.json").read_text())
    md += [f"Domain probe (synthetic vs real crops, held-out half): AUC {domain['test_auc']}", ""]
    if (RESULTS / "finetune_runs.csv").exists():
        ft_runs = load_runs("finetune_runs.csv")
        md += summary_section("Fine-tuned ResNet18",
                              "5 folds × 3 seeds for every arm; syn-only: 3 seeds, each scored on the 5 IDID folds.",
                              summarize(ft_runs), paired_differences(ft_runs))
    (RESULTS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote results/summary.md")
