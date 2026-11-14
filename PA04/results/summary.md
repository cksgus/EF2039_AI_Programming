# Results

Mean ± SD over runs. Recall and false alarm in %, at the threshold fixed on the validation fold.

## Frozen ResNet18 features + logistic regression

5 folds × 3 seeds; 5 runs for the 100 % arms and syn-only.

| Arm | Runs | Real broken | IDID recall | IDID false alarm | IDID AUC | Mendeley recall | Mendeley false alarm | Mendeley AUC |
|---|---:|---:|---|---|---|---|---|---|
| syn-only | 5 | 0 | 85.4 ± 5.2 | 69.7 ± 2.4 | 0.676 ± 0.048 | 86.1 ± 0.0 | 81.1 ± 0.0 | 0.523 ± 0.000 |
| real10 | 15 | 18 | 56.0 ± 9.4 | 4.9 ± 2.8 | 0.863 ± 0.031 | 45.0 ± 18.7 | 15.5 ± 12.7 | 0.745 ± 0.092 |
| real10+syn | 15 | 18 | 39.1 ± 10.1 | 5.1 ± 2.0 | 0.795 ± 0.041 | 26.4 ± 10.2 | 66.8 ± 15.1 | 0.231 ± 0.061 |
| real25 | 15 | 44 | 60.3 ± 5.8 | 6.3 ± 3.0 | 0.877 ± 0.023 | 58.3 ± 23.7 | 31.1 ± 22.7 | 0.718 ± 0.159 |
| real25+syn | 15 | 44 | 50.1 ± 7.0 | 6.0 ± 2.4 | 0.833 ± 0.029 | 31.4 ± 17.3 | 67.7 ± 12.6 | 0.252 ± 0.114 |
| real50 | 15 | 88 | 65.6 ± 6.3 | 5.3 ± 2.9 | 0.895 ± 0.026 | 67.9 ± 23.0 | 41.6 ± 22.1 | 0.707 ± 0.132 |
| real50+syn | 15 | 88 | 60.1 ± 6.9 | 5.9 ± 3.0 | 0.879 ± 0.026 | 36.4 ± 20.7 | 62.6 ± 12.8 | 0.308 ± 0.101 |
| real100 | 5 | 177 | 64.1 ± 6.2 | 5.5 ± 2.9 | 0.895 ± 0.031 | 75.1 ± 21.8 | 37.0 ± 19.2 | 0.791 ± 0.081 |
| real100+syn | 5 | 177 | 65.4 ± 11.3 | 6.4 ± 3.9 | 0.884 ± 0.033 | 54.4 ± 28.6 | 57.6 ± 13.9 | 0.458 ± 0.141 |

Adding the synthetic crops, paired difference (real+syn − real):

| Real broken kept | Metric | Mean Δ | SD | Better | Same | Worse |
|---:|---|---:|---:|---:|---:|---:|
| 10 % | idid_recall | -16.949 pp | 11.655 | 1 | 1 | 13 |
| 10 % | idid_auc | -0.068 | 0.029 | 0 | 0 | 15 |
| 10 % | mendeley_auc | -0.515 | 0.110 | 0 | 0 | 15 |
| 25 % | idid_recall | -10.282 pp | 7.673 | 1 | 0 | 14 |
| 25 % | idid_auc | -0.044 | 0.016 | 0 | 0 | 15 |
| 25 % | mendeley_auc | -0.466 | 0.098 | 0 | 0 | 15 |
| 50 % | idid_recall | -5.537 pp | 6.516 | 2 | 1 | 12 |
| 50 % | idid_auc | -0.015 | 0.013 | 2 | 0 | 13 |
| 50 % | mendeley_auc | -0.399 | 0.094 | 0 | 0 | 15 |
| 100 % | idid_recall | +1.356 pp | 9.237 | 2 | 1 | 2 |
| 100 % | idid_auc | -0.011 | 0.007 | 0 | 0 | 5 |
| 100 % | mendeley_auc | -0.332 | 0.069 | 0 | 0 | 5 |

Domain probe (synthetic vs real crops, held-out half): AUC 1.0

## Fine-tuned ResNet18

5 folds × 3 seeds for every arm; syn-only: 3 seeds, each scored on the 5 IDID folds.

| Arm | Runs | Real broken | IDID recall | IDID false alarm | IDID AUC | Mendeley recall | Mendeley false alarm | Mendeley AUC |
|---|---:|---:|---|---|---|---|---|---|
| syn-only | 15 | 0 | 87.2 ± 6.1 | 64.8 ± 9.8 | 0.693 ± 0.054 | 11.5 ± 0.2 | 94.3 ± 0.9 | 0.039 ± 0.015 |
| real10 | 15 | 18 | 71.0 ± 10.1 | 5.4 ± 2.5 | 0.919 ± 0.020 | 17.0 ± 15.6 | 37.7 ± 9.3 | 0.365 ± 0.128 |
| real10+syn | 15 | 18 | 73.8 ± 8.7 | 4.1 ± 2.1 | 0.925 ± 0.035 | 23.2 ± 23.0 | 39.2 ± 10.3 | 0.399 ± 0.136 |
| real25 | 15 | 44 | 84.4 ± 5.3 | 4.8 ± 1.2 | 0.953 ± 0.020 | 32.1 ± 20.7 | 51.2 ± 16.5 | 0.365 ± 0.134 |
| real25+syn | 15 | 44 | 84.3 ± 5.0 | 5.9 ± 2.4 | 0.951 ± 0.012 | 24.3 ± 15.1 | 54.0 ± 14.1 | 0.304 ± 0.110 |
| real50 | 15 | 88 | 88.7 ± 4.1 | 4.3 ± 1.7 | 0.975 ± 0.013 | 35.2 ± 19.7 | 51.8 ± 11.3 | 0.398 ± 0.127 |
| real50+syn | 15 | 88 | 88.2 ± 4.4 | 4.9 ± 2.0 | 0.974 ± 0.011 | 31.0 ± 19.2 | 52.7 ± 10.9 | 0.346 ± 0.120 |
| real100 | 15 | 177 | 91.8 ± 2.9 | 5.1 ± 2.4 | 0.979 ± 0.012 | 38.4 ± 16.7 | 57.1 ± 7.8 | 0.362 ± 0.088 |
| real100+syn | 15 | 177 | 91.5 ± 4.1 | 5.6 ± 2.6 | 0.981 ± 0.010 | 45.0 ± 20.7 | 54.6 ± 7.3 | 0.420 ± 0.118 |

Adding the synthetic crops, paired difference (real+syn − real):

| Real broken kept | Metric | Mean Δ | SD | Better | Same | Worse |
|---:|---|---:|---:|---:|---:|---:|
| 10 % | idid_recall | +2.825 pp | 15.060 | 7 | 1 | 7 |
| 10 % | idid_auc | +0.006 | 0.035 | 7 | 0 | 8 |
| 10 % | mendeley_auc | +0.034 | 0.139 | 8 | 0 | 7 |
| 25 % | idid_recall | -0.113 pp | 5.818 | 6 | 3 | 6 |
| 25 % | idid_auc | -0.001 | 0.018 | 7 | 0 | 8 |
| 25 % | mendeley_auc | -0.061 | 0.076 | 3 | 0 | 12 |
| 50 % | idid_recall | -0.452 pp | 2.753 | 6 | 4 | 5 |
| 50 % | idid_auc | -0.001 | 0.011 | 7 | 0 | 8 |
| 50 % | mendeley_auc | -0.052 | 0.133 | 4 | 0 | 11 |
| 100 % | idid_recall | -0.226 pp | 2.708 | 4 | 6 | 5 |
| 100 % | idid_auc | +0.002 | 0.009 | 8 | 0 | 7 |
| 100 % | mendeley_auc | +0.059 | 0.112 | 10 | 0 | 5 |
