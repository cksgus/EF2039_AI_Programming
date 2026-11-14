"""Extract frozen ResNet18 (ImageNet) features for every crop.

    python extract_features.py

Writes data/features_resnet18.npz with <set>_files and <set>_X (n x 512) for each set.
Takes about 3 minutes on a 2-core CPU.
"""
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from common import DATA, FEATURES, IMAGE_SIZE, SETS, read_labels

BATCH_SIZE = 64


def load_backbone():
    """ImageNet ResNet18 without its classification layer (512-d output)."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model.eval()


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def embed(paths, model):
    transform = eval_transform()
    out = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i + BATCH_SIZE]])
        out.append(model(batch).numpy().astype(np.float32))
    return np.concatenate(out)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = load_backbone()
    arrays = {}
    for name in SETS:
        files = [r["file"] for r in read_labels(name)]
        start = time.time()
        arrays[f"{name}_files"] = np.array(files)
        arrays[f"{name}_X"] = embed([DATA / name / f for f in files], model)
        print(f"{name}: {len(files)} crops, {time.time() - start:.0f} s")
    np.savez_compressed(FEATURES, **arrays)
    print(f"saved {FEATURES}")
