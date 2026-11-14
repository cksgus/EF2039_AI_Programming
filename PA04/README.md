# PA04: Does rendered data help when real defects are scarce?

EF2039 PA04. Code, data, trained weights and results of a data-centric study of broken-insulator
classification.

The model is fixed (ImageNet ResNet18 features + logistic regression) and each insulator crop is
classified as **broken** or **normal**. Only the training data changes:

- **RQ1.** How does recall depend on the number of real broken examples, and do 420 rendered
  (synthetic) crops make up for missing ones?
- **RQ2.** Does the classifier transfer from rendered to real crops, and from one real dataset (IDID)
  to another (Mendeley)?

Short answer: more real broken crops help (recall 56 % with 18 → about 65 % with 88 or more). The
renders do not replace them: with frozen features they lower recall when real defects are scarce
(−16.9 points with 18), and with a fully fine-tuned ResNet18 they make no difference.

## Files

| | |
|---|---|
| `common.py` | experiment settings, data loading, fold split, metrics |
| `download_data.py` | download the crops (and optionally the features) from the release |
| `extract_features.py` | frozen ResNet18 features → `data/features_resnet18.npz` |
| `train_probe.py` | main experiment: all arms × folds × seeds → `results/probe_runs.csv`, `checkpoints/` |
| `train_finetune.py` | fine-tuning check on a GPU → prediction files |
| `evaluate_finetune.py` | score the fine-tuned runs → `results/finetune_runs.csv` |
| `make_report.py` | summary tables → `results/summary.md` |
| `predict.py` | classify new crops with a trained model |
| `checkpoints/` | logistic-regression weights, one file per arm (all folds and seeds inside) |
| `results/` | one line per run (`*_runs.csv`), `summary.md`, raw fine-tune predictions |
| `DATA.md` | data sources, licences, preprocessing, class counts, folds |

## Reproduce

Needs only a CPU: about 3 minutes for the features on 2 cores, then a few seconds. About 650 MB of disk.

```bash
git clone https://github.com/cksgus/EF2039_AI_Programming.git
cd EF2039_AI_Programming/PA04
conda env create -f environment.yml
conda activate ef2039-pa04

python download_data.py        # 308 MB into data/
python extract_features.py     # ~3 min, downloads the torchvision ResNet18 weights on first use
python train_probe.py          # 105 runs
python make_report.py          # results/summary.md
```

`python download_data.py --features` downloads the features behind the reported results instead of extracting
them; with those, every number below is reproduced exactly. Features extracted on another machine can
differ in the last digits, which may move a threshold by one crop in a few runs.

## Setup of the experiment

- **Data** (details in [DATA.md](DATA.md)): IDID V1.2 disc crops, 3,265 normal and 295 broken, in 5
  folds grouped by source photo. Mendeley porcelain-disc crops, 392 normal and 238 broken, used only as
  an external test. 420 synthetic renders (210 normal, 210 broken), used only for training.
- **Split**: for test fold k, fold k+1 is the validation fold and the other three are for training
  (on average 1,959 normal and 177 broken crops).
- **Model**: torchvision ResNet18 (IMAGENET1K_V1), frozen, 224 px input, 512-d features →
  standardization → logistic regression (C = 1, balanced class weights).
- **Arms**: `real{p}` keeps p = 10, 25, 50 or 100 % of the broken training crops (3 random subsets) and
  all normal ones; `real{p}+syn` adds the 420 renders; `syn-only` uses only the renders.
- **Threshold**: set on the validation fold so that at most 5 % of its normal crops are flagged, then
  used unchanged on the test fold and on Mendeley (`syn-only`: 0.5).
- **Metrics**: recall (broken crops flagged), false-alarm rate (normal crops flagged), ROC AUC.

## Results

Mean ± SD over runs, recall and false alarm in %. Full tables with the paired differences are in
[`results/summary.md`](results/summary.md).

| Training data | Real broken | IDID recall | IDID false alarm | IDID AUC | Mendeley AUC |
|---|---:|---|---|---|---|
| Synthetic only | 0 | 85.4 ± 5.2 | 69.7 ± 2.4 | 0.676 ± 0.048 | 0.523 ± 0.000 |
| Real 10 % | 18 | 56.0 ± 9.4 | 4.9 ± 2.8 | 0.863 ± 0.031 | 0.745 ± 0.092 |
| Real 10 % + synthetic | 18 | 39.1 ± 10.1 | 5.1 ± 2.0 | 0.795 ± 0.041 | 0.231 ± 0.061 |
| Real 25 % | 44 | 60.3 ± 5.8 | 6.3 ± 3.0 | 0.877 ± 0.023 | 0.718 ± 0.159 |
| Real 25 % + synthetic | 44 | 50.1 ± 7.0 | 6.0 ± 2.4 | 0.833 ± 0.029 | 0.252 ± 0.114 |
| Real 50 % | 88 | 65.6 ± 6.3 | 5.3 ± 2.9 | 0.895 ± 0.026 | 0.707 ± 0.132 |
| Real 50 % + synthetic | 88 | 60.1 ± 6.9 | 5.9 ± 3.0 | 0.879 ± 0.026 | 0.308 ± 0.101 |
| Real 100 % | 177 | 64.1 ± 6.2 | 5.5 ± 2.9 | 0.895 ± 0.031 | 0.791 ± 0.081 |
| Real 100 % + synthetic | 177 | 65.4 ± 11.3 | 6.4 ± 3.9 | 0.884 ± 0.033 | 0.458 ± 0.141 |

A logistic regression separates synthetic from real crops with AUC 1.00 on the same features
(`results/domain_probe.json`), so a linear classifier can tell the two apart without looking at damage.

## Fine-tuning check

The same arms with the whole ResNet18 fine-tuned (10 epochs, batch 32, AdamW 3e-4 with cosine decay,
class-weighted cross-entropy, flips and colour jitter; 5 folds × 3 seeds, 123 runs). This ran on the
school GPU server (part of an A100, about 60 s per run, 124 minutes in total).

| Training data | Real broken | IDID recall | IDID false alarm | IDID AUC | Mendeley AUC |
|---|---:|---|---|---|---|
| Synthetic only | 0 | 87.2 ± 6.1 | 64.8 ± 9.8 | 0.693 ± 0.054 | 0.039 ± 0.015 |
| Real 10 % | 18 | 71.0 ± 10.1 | 5.4 ± 2.5 | 0.919 ± 0.020 | 0.365 ± 0.128 |
| Real 10 % + synthetic | 18 | 73.8 ± 8.7 | 4.1 ± 2.1 | 0.925 ± 0.035 | 0.399 ± 0.136 |
| Real 25 % | 44 | 84.4 ± 5.3 | 4.8 ± 1.2 | 0.953 ± 0.020 | 0.365 ± 0.134 |
| Real 25 % + synthetic | 44 | 84.3 ± 5.0 | 5.9 ± 2.4 | 0.951 ± 0.012 | 0.304 ± 0.110 |
| Real 50 % | 88 | 88.7 ± 4.1 | 4.3 ± 1.7 | 0.975 ± 0.013 | 0.398 ± 0.127 |
| Real 50 % + synthetic | 88 | 88.2 ± 4.4 | 4.9 ± 2.0 | 0.974 ± 0.011 | 0.346 ± 0.120 |
| Real 100 % | 177 | 91.8 ± 2.9 | 5.1 ± 2.4 | 0.979 ± 0.012 | 0.362 ± 0.088 |
| Real 100 % + synthetic | 177 | 91.5 ± 4.1 | 5.6 ± 2.6 | 0.981 ± 0.010 | 0.420 ± 0.118 |

Here the renders change recall by +2.8, −0.1, −0.5 and −0.2 points (10, 25, 50, 100 %), so they neither
hurt nor help. Mendeley AUC is below chance with or without renders; its normal boxes are whole strings
and its broken boxes are small units, so the labelling differs from IDID (see DATA.md).

The predictions of all 123 runs are in `results/finetune_predictions.tar.gz`, so the table can be
recomputed without a GPU:

```bash
mkdir -p data && tar xzf results/finetune_predictions.tar.gz -C data
python evaluate_finetune.py
python make_report.py
```

To train again (GPU, about 2 hours; numbers will be close but not identical because GPU training is
not deterministic):

```bash
python train_finetune.py check
python train_finetune.py run --out data/my_finetune_runs
python evaluate_finetune.py --runs data/my_finetune_runs
```

The fine-tuned weights (seed 0 of every arm and fold, 41 models, half precision) are in the release as
`pa04_finetune_checkpoints_v1.tar` (0.9 GB). Unpack it into `data/` to use them with `predict.py`.

## Predict

The two example crops are from IDID fold 0, which the fold-0 models did not see in training or validation.

```bash
python predict.py checkpoints/real100.npz --fold 0 \
    data/pa04_insulator_crops_v1/idid/crops/000001268_001.png \
    data/pa04_insulator_crops_v1/idid/crops/000001268_002.png
```

```
threshold 0.8677
BROKEN  P(broken)=1.0000  data/pa04_insulator_crops_v1/idid/crops/000001268_001.png
normal  P(broken)=0.0144  data/pa04_insulator_crops_v1/idid/crops/000001268_002.png
```

With a fine-tuned model:

```bash
python predict.py data/pa04_finetune_runs_v1/real100/fold0_seed0/model.pt \
    data/pa04_insulator_crops_v1/idid/crops/000001268_001.png \
    data/pa04_insulator_crops_v1/idid/crops/000001268_002.png
```

```
threshold 0.0848
BROKEN  P(broken)=0.8029  data/pa04_insulator_crops_v1/idid/crops/000001268_001.png
normal  P(broken)=0.0005  data/pa04_insulator_crops_v1/idid/crops/000001268_002.png
```

## Data licences

The data package contains crops derived from IDID V1.2 (EPRI, CC BY 4.0) and from the Mendeley set by
S. Jiskani (CC BY 4.0); attribution and the changes made are described in [DATA.md](DATA.md). The
synthetic crops are released under CC BY-NC 4.0.
