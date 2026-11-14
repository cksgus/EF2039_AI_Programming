"""Download the crops (and optionally the ResNet18 features behind the reported results) from the GitHub release.

    python download_data.py               # crops and labels, 308 MB
    python download_data.py --features    # also data/features_resnet18.npz, 10 MB
"""
import argparse
import hashlib
import urllib.request
import zipfile

from common import DATA, FEATURES

RELEASE = "https://github.com/cksgus/EF2039_AI_Programming/releases/download/pa04-data-v1/"
FILES = {   # file name -> SHA-256
    "pa04_insulator_crops_v1.zip": "495069b331bb84e60654ab778b8a6eb4b8c6830d2a0619bce14892909d52a91f",
    "features_resnet18.npz": "4be72b8079ab483bf23ea693507666dc8e6f4c2e18bbb48e394eed7d194af781",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, target):
    if target.exists() and sha256(target) == FILES[name]:
        print(f"{target.name} is already here")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(RELEASE + name, target)
    if sha256(target) != FILES[name]:
        raise RuntimeError(f"{name}: checksum mismatch, please download again")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", action="store_true", help="also download the extracted features")
    args = parser.parse_args()

    archive = DATA.parent / "pa04_insulator_crops_v1.zip"
    fetch(archive.name, archive)
    if not (DATA / "idid" / "labels.csv").exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(DATA)
    print(f"crops are in {DATA}")
    if args.features:
        fetch(FEATURES.name, FEATURES)
