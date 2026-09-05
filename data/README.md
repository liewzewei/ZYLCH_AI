# Data

The full dataset (~3,000 original photos, expanded to ~15,000 augmented images,
~20 GB on disk) is **not** included in this repository.

- `sample/` — a small handful of real photos (2 per class) so you can try the
  model and see what the inputs look like.

## Rebuilding the full dataset

1. Put your original photos into one folder per class, e.g.:
   ```
   data/raw/
   ├── KEK_LAPIS/
   ├── KUIH_KASWI_PANDAN/
   └── ... (8 classes)
   ```
2. (Optional) Balance classes with `notebooks/1_trim.ipynb`.
3. Augment ×11 and split 80/10/10:
   ```bash
   python src/augment.py data/raw --ratio 0.8 0.1 0.1
   ```
   This creates `data/raw_Output/` (augmented) and `data/raw_Split_Output/`
   (train/val/test).

See `docs/PROJECT_DOCUMENTATION.md` §4 for the full data pipeline and the exact
per-class counts used in the competition.
