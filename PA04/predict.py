"""Classify crop images as normal or broken with a trained model.

    # frozen features + logistic regression: one file per arm, pick the run with --fold / --seed
    python predict.py checkpoints/real100.npz --fold 0 crop1.png crop2.png

    # fine-tuned ResNet18 (pa04_finetune_checkpoints_v1.tar from the release, unpacked into data/)
    python predict.py data/pa04_finetune_runs_v1/real100/fold0_seed0/model.pt crop1.png crop2.png

A crop should show one insulator disc, roughly centred, like the crops in the data package.
The threshold is the one fixed on the run's validation fold (for a fine-tuned model it is read
from results/finetune_runs.csv); --threshold overrides it.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

from common import RESULTS
from extract_features import embed, eval_transform, load_backbone


def probe_scores(checkpoint, fold, seed, images):
    ckpt = np.load(checkpoint)
    i = np.flatnonzero((ckpt["fold"] == fold) & (ckpt["seed"] == seed))[0]
    features = embed(images, load_backbone())
    z = (features - ckpt["mean"][i]) / ckpt["scale"][i]
    scores = 1 / (1 + np.exp(-(z @ ckpt["coef"][i] + ckpt["intercept"][i])))
    return scores, float(ckpt["threshold"][i])


def finetuned_scores(checkpoint, images):
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})     # stored in half precision
    model.eval()
    transform = eval_transform()
    with torch.no_grad():
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in images])
        scores = torch.softmax(model(batch), 1)[:, 1].numpy()
    # .../<arm>/fold<k>_seed<s>/model.pt -> threshold of that run
    arm = checkpoint.parent.parent.name
    fold, seed = checkpoint.parent.name.removeprefix("fold").split("_seed")
    with open(RESULTS / "finetune_runs.csv", newline="") as f:
        for row in csv.DictReader(f):
            if (row["arm"], row["fold"], row["seed"]) == (arm, fold, seed):
                return scores, float(row["threshold"])
    return scores, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    if args.checkpoint.suffix == ".pt":
        scores, threshold = finetuned_scores(args.checkpoint, args.images)
    else:
        scores, threshold = probe_scores(args.checkpoint, args.fold, args.seed, args.images)
    if args.threshold is not None:
        threshold = args.threshold
    if threshold is None:
        parser.error("no threshold found for this checkpoint, pass --threshold")
    print(f"threshold {threshold:.4f}")
    for path, score in zip(args.images, scores):
        print(f"{'BROKEN' if score > threshold else 'normal':6s}  P(broken)={score:.4f}  {path}")
