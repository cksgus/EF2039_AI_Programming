"""Fine-tuning check: train the whole ResNet18 for every arm, fold and seed (needs a GPU).

    python train_finetune.py check                  # only checks data and GPU, trains nothing
    python train_finetune.py smoke --out OUT        # one tiny run (1 epoch, 64 crops)
    python train_finetune.py run   --out OUT        # all 123 runs, finished runs are skipped on restart

Arms, folds and broken-crop subsets are the same as in train_probe.py (plus 3 seeds at 100 %).
Each run writes OUT/<arm>/fold<k>_seed<s>/
    predictions.csv   P(broken) for the validation fold, the test fold and Mendeley
    run.json          crop counts, loss per epoch, time
    model.pt          half-precision weights (seed 0 only)
Thresholds and metrics are computed afterwards with evaluate_finetune.py.

--data points to the crops if they are not in data/, --weights to a local resnet18-f37072fd.pth
on a machine without internet, and --workers 0 helps if the data loader runs out of shared memory.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from common import DATA, FRACTIONS, IMAGE_SIZE, N_FOLDS, SEEDS, arm_name, fold_split, load_set, subsample_broken

# training recipe, fixed before any run
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4            # AdamW, cosine decay per step
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
train_transform = transforms.Compose([resize, transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(0.2, 0.2, 0.2, 0.02), transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)])
test_transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Crops(torch.utils.data.Dataset):
    def __init__(self, files, labels, transform):
        self.files, self.labels, self.transform = list(files), list(labels), transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(self.files[i]).convert("RGB")), int(self.labels[i])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def new_model(weights_file=None):
    if weights_file:
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
    else:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def train(model, files, labels, seed, device, workers, epochs=EPOCHS):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        Crops(files, labels, train_transform), batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=workers, generator=torch.Generator().manual_seed(seed),
        worker_init_fn=seed_worker, persistent_workers=workers > 0)
    # cross-entropy with balanced class weights
    counts = np.maximum(np.bincount(labels, minlength=2), 1)
    weights = torch.tensor(len(labels) / (2 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loader))
    model.to(device).train()
    history = []
    for epoch in range(epochs):
        start, loss_sum, seen = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
        history.append({"epoch": epoch + 1, "train_loss": round(loss_sum / seen, 5),
                        "seconds": round(time.time() - start, 1)})
    return history      # the model after the last epoch is used


@torch.no_grad()
def predict(model, files, device, workers):
    loader = torch.utils.data.DataLoader(Crops(files, np.zeros(len(files), int), test_transform),
                                         batch_size=128, shuffle=False, num_workers=workers)
    model.to(device).eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x.to(device)).float()
        out.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def plan_runs(idid, syn):
    runs = [{"arm": "syn-only", "fold": 0, "seed": seed, "train_seed": seed, "train": (syn.paths, syn.y), "val": None}
            for seed in SEEDS]
    for fold in range(N_FOLDS):
        train_idx, val, test = fold_split(idid, fold)
        for fraction in FRACTIONS:
            for seed in SEEDS:
                subset = subsample_broken(train_idx, idid.y, fraction, seed)
                for synthetic in (False, True):
                    files, y = idid.paths[subset], idid.y[subset]
                    if synthetic:
                        files, y = np.concatenate([files, syn.paths]), np.concatenate([y, syn.y])
                    runs.append({"arm": arm_name(fraction, synthetic), "fold": fold, "seed": seed,
                                 "train_seed": 100 * fold + seed, "train": (files, y), "val": val, "test": test,
                                 "train_broken_real": int((idid.y[subset] == 1).sum()),
                                 "train_normal_real": int((idid.y[subset] == 0).sum())})
    return runs


def do_run(run, idid, mendeley, out, args, device, epochs=EPOCHS, limit=None):
    folder = out / run["arm"] / f"fold{run['fold']}_seed{run['seed']}"
    if (folder / "run.json").exists():
        print(f"skip {folder.relative_to(out)} (finished)")
        return
    folder.mkdir(parents=True, exist_ok=True)
    files, labels = run["train"]
    if limit:
        pick = np.random.default_rng(0).choice(len(files), size=min(limit, len(files)), replace=False)
        files, labels = files[pick], labels[pick]
    start = time.time()
    model = new_model(args.weights)
    history = train(model, files, labels, run["train_seed"], device, args.workers, epochs)

    if run["val"] is None:      # syn-only is scored on all IDID crops; evaluate_finetune.py splits them by fold
        parts = [("idid", idid, np.arange(len(idid.y)))]
    else:
        parts = [("val", idid, run["val"]), ("test", idid, run["test"])]
    parts.append(("mendeley", mendeley, np.arange(len(mendeley.y))))
    with open(folder / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "file", "label", "p_broken"])
        for part, crops, idx in parts:
            for file, y, p in zip(crops.files[idx], crops.y[idx], predict(model, crops.paths[idx], device, args.workers)):
                writer.writerow([part, f"{crops.name}/{file}", int(y), f"{p:.6f}"])
    if run["seed"] == 0 or limit:
        torch.save({k: v.half() for k, v in model.state_dict().items()}, folder / "model.pt")
    info = {"arm": run["arm"], "fold": run["fold"], "seed": run["seed"], "train_seed": run["train_seed"],
            "train_crops": len(labels), "train_broken": int(labels.sum()),
            "train_broken_real": run.get("train_broken_real", 0), "train_normal_real": run.get("train_normal_real", 0),
            "epochs": len(history), "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "history": history,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "torch": torch.__version__,
            "seconds": round(time.time() - start, 1), "smoke": bool(limit)}
    (folder / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"done {folder.relative_to(out)} in {info['seconds']} s, last loss {history[-1]['train_loss']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "smoke", "run"])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idid, mendeley, syn = (load_set(name, args.data) for name in ("idid", "mendeley", "synthetic"))
    runs = plan_runs(idid, syn)

    if args.command == "check":
        for s in (idid, mendeley, syn):
            missing = sum(not Path(p).is_file() for p in s.paths)
            print(f"{s.name}: {len(s.y)} crops (normal {int((s.y == 0).sum())}, broken {int((s.y == 1).sum())}), "
                  f"missing files {missing}")
        print("broken crops per IDID fold:", [int(((idid.fold == k) & (idid.y == 1)).sum()) for k in range(N_FOLDS)])
        print("runs planned:", len(runs))
        print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if device == "cuda" else "cpu (no GPU)")
        model = new_model(args.weights).to(device).eval()
        with torch.no_grad():
            print("forward pass on 2 crops:", tuple(model(torch.zeros(2, 3, 224, 224, device=device)).shape))
        print("check passed: nothing was trained")
    elif args.out is None:
        parser.error("--out is required for smoke and run")
    elif args.command == "smoke":
        smoke = next(r for r in runs if r["arm"] == "real25+syn")
        do_run(smoke, idid, mendeley, args.out / "smoke", args, device, epochs=1, limit=64)
    else:
        start = time.time()
        for i, run in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {run['arm']} fold {run['fold']} seed {run['seed']}", flush=True)
            do_run(run, idid, mendeley, args.out, args, device)
        print(f"all {len(runs)} runs finished in {(time.time() - start) / 60:.1f} min")
