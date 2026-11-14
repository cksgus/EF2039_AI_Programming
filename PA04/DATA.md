# Data card: `pa04_insulator_crops_v1`

Crops of porcelain disc insulators for a binary task, **broken** vs **normal**. The archive is a
release asset of this repository; `python download_data.py` fetches and checks it. The same text is
the archive's `README.md`.

| Set | Source | Licence | Crops | Normal | Broken | Not used in the task | Role |
|---|---|---|---:|---:|---:|---|---|
| `idid` | IDID V1.2, EPRI [1] | CC BY 4.0 | 4,277 | 3,265 | 295 | 641 flashover, 76 ambiguous | training, validation and test (5 folds) |
| `mendeley` | Porcelain Disk type insulators, Mendeley Data [2] | CC BY 4.0 | 973 | 392 | 238 | 271 flashover, 72 shaded | external test only |
| `synthetic` | rendered by the author | CC BY-NC 4.0 | 420 | 210 | 210 | none | training only |

## Layout

```
pa04_insulator_crops_v1/
  idid/crops/*.png        idid/labels.csv
  mendeley/crops/*.png    mendeley/labels.csv
  synthetic/crops/*.png   synthetic/labels.csv
  README.md               (this data card)
```

Every crop is a 256 x 256 RGB PNG without metadata.

## Real crops: how they were made

- **Boxes.** IDID: the per-disc boxes of `Train/labels_v1.2.json` (399 of the 400 base photos have
  disc boxes; the string boxes, the 1,200 augmented copies and the 88 unlabelled test photos are not
  used). Mendeley: the publisher's YOLO polygons, as their bounding boxes, from all three publisher
  splits (255 photos).
- **Crop.** Each box grows by 15 % of its size on every side, is clipped to the photo, is centred on a
  square canvas padded with RGB (124, 116, 104), and is resized to 256 px with LANCZOS. No colour change,
  no EXIF rotation.
- **Labels.** IDID condition `No issues` → normal, `Broken` → broken, `Flashover damage` → flashover,
  `notbroken-notflashed` → ambiguous. Mendeley class `Insulator` → normal, `Broken` → broken,
  `Dirty/Flashed` → flashover, `Shaded` → shaded. Only normal and broken are used.
- **Annotation scope differs.** IDID boxes are single discs. Mendeley `Insulator` boxes mostly outline a
  whole string, while its `Broken` boxes are small regions (median box long side 67 px, against 451 px
  for IDID broken discs and 510 px for Mendeley normal boxes). Part of any IDID → Mendeley drop therefore comes from labelling, not only
  from appearance.

## IDID folds

`idid/labels.csv` gives each crop's `fold` (0–4). Folds are grouped by source photo: all crops of one
photo are in the same fold, so no photo is shared between training and test. Photos were assigned
greedily to balance the class counts.

| Fold | Normal | Broken | Flashover | Ambiguous |
|---:|---:|---:|---:|---:|
| 0 | 698 | 59 | 128 | 12 |
| 1 | 627 | 59 | 129 | 17 |
| 2 | 688 | 59 | 128 | 18 |
| 3 | 626 | 59 | 128 | 9 |
| 4 | 626 | 59 | 128 | 20 |

## Columns

- `idid/labels.csv`: `file`, `label`, `fold`, `source_image` (path inside the IDID release),
  `box_x`, `box_y`, `box_w`, `box_h` (source pixels), `idid_condition`.
- `mendeley/labels.csv`: `file`, `label`, `source_image`, `publisher_split`, `publisher_class`,
  `box_x`, `box_y`, `box_w`, `box_h`.
- `synthetic/labels.csv`: `file`, `label`.

## Synthetic crops

420 renders of a single porcelain disc from a parametric 3D model (210 intact, 210 with a chipped or
broken shed), on plain grey backgrounds, cropped with the same protocol as the real crops. They are an
early, deliberately simple render set: one disc, no real background, no clutter. File names are
shuffled, and only the binary label is given.

## Attribution and changes

[1] EPRI, *Insulator Defect Image Dataset (IDID) V1.2*, IEEE DataPort, DOI 10.21227/vkdw-x769.
Licensed CC BY 4.0 per the dataset's ReadMe (EPRI product 3002017949, 2020). The original images need an
IEEE DataPort login; this archive redistributes derived crops only. EPRI's ReadMe asks users to comply
with U.S. export regulations.

[2] S. Jiskani, *Porcelain Disk type insulators (Classification and Detection task)*, Mendeley Data, V1,
2026, DOI 10.17632/5pwrd7gmw2.1. CC BY 4.0.

Changes to [1] and [2]: cropped, padded and resized as above; labels mapped as above. The crops are
provided as-is, without warranty.
