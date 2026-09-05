# Kuih Classifier — Complete Project Documentation

> A deep-learning image classifier that recognises **8 types of traditional
> Malaysian/Nyonya kuih** (steamed cakes and sweets) from a photograph, built
> with **transfer learning on ResNet‑50** and tuned with **Bayesian
> Optimization**.

Built for the **National AI Competition (NAIC) 2024**, Senior Technical Track —
organised by **Hyperbyte AI**, sponsored by **Sunway University** — where it placed
**🏆 2nd Runner-Up out of 60+ teams**. The given problem statement was to build an
image-classification model for 8 kuih. Every part of the project — collecting and
photographing the dataset, augmentation, architecture, tuning, evaluation — was
done **manually, with no AI assistance**, by **Team ZYLCH AI**.

This document is the full technical reference and the story behind the project:
what it does, how we built the dataset, how the model was designed and tuned, how
it performed, and the months of iteration that got us there.

> **On this write-up:** the competition project itself — the dataset, the code, the
> model, every decision — was built entirely by hand in 2024 with no AI assistance.
> This repository's *presentation* (this documentation and the README) was later
> organised and polished with [Claude](https://claude.ai) to make the work clear to
> readers. The substance is the team's; the write-up is AI-assisted.

---

## Table of Contents

1. [At a glance](#1-at-a-glance)
2. [Headline results](#2-headline-results)
3. [The 8 classes](#3-the-8-classes)
4. [Building the dataset from scratch](#4-building-the-dataset-from-scratch)
5. [Model architecture](#5-model-architecture)
6. [Hyperparameter optimization (Bayesian Optimization)](#6-hyperparameter-optimization-bayesian-optimization)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [The journey — iterations, trials & errors](#9-the-journey--iterations-trials--errors)
10. [Repository structure](#10-repository-structure)
11. [Notebook & script reference](#11-notebook--script-reference)
12. [The trained model & how to use it](#12-the-trained-model--how-to-use-it)
13. [Tech stack](#13-tech-stack)
14. [Reproducing the project](#14-reproducing-the-project)
15. [Design decisions & honest caveats](#15-design-decisions--honest-caveats)
16. [Glossary](#16-glossary)
17. [Credits](#17-credits)

---

## 1. At a glance

| | |
|---|---|
| **Competition** | National AI Competition (NAIC) 2024 — Senior Technical Track (Hyperbyte AI · Sunway University) |
| **Placement** | 🏆 **2nd Runner-Up**, out of 60+ teams |
| **Team** | ZYLCH AI — Hao Yuan, Lilliana, Wei Han, Ze Wei |
| **Task** | Multi‑class image classification (single label per image) |
| **Domain** | Traditional Malaysian / Nyonya *kuih* (cakes & sweets) |
| **Classes** | 8 (see [§3](#3-the-8-classes)) |
| **Approach** | Transfer learning — ImageNet‑pretrained **ResNet‑50** with a custom classifier head; the deepest residual stages are fine‑tuned |
| **Framework** | PyTorch (`torch`, `torchvision`) — migrated from an initial TensorFlow/Keras prototype |
| **Tuning** | Bayesian Optimization via **Hyperopt** (TPE), with a custom objective that penalises the train/validation generalization gap |
| **Data** | ~3,000 photos taken by the team, class‑balanced, then offline‑augmented ×11 into ~15,000 images, plus online (on‑the‑fly) augmentation |
| **Result** | **Macro F1 ≈ 0.979**, accuracy **274/280 = 97.9%** on a held‑out test set |

---

## 2. Headline results

Measured on a **held‑out test set of 280 images (35 per class)** that the model
never saw during training or validation.

| Metric (macro‑averaged) | Score |
|---|---|
| Precision | **0.9790** |
| Recall | **0.9786** |
| F1 | **0.9786** |
| Accuracy | **274 / 280 = 0.9786** |

**Per‑class F1:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Kek Lapis | 0.972 | 1.000 | 0.986 |
| Kuih Kaswi Pandan | 0.943 | 0.943 | 0.943 |
| Kuih Ketayap | 1.000 | 1.000 | **1.000** |
| Kuih Lapis | 1.000 | 0.971 | 0.986 |
| Kuih Seri Muka | 1.000 | 0.971 | 0.986 |
| Kuih Talam | 0.946 | 1.000 | 0.972 |
| Kuih Ubi Kayu | 1.000 | 1.000 | **1.000** |
| Onde‑Onde | 0.971 | 0.943 | 0.957 |

### Confusion matrix

![Confusion matrix](../results/confusion_matrix.png)

Only **6 misclassifications out of 280**. Four classes are perfect (35/35), and
every mistake is intuitive — they happen between visually similar green/coconut
kuih:

- Onde‑Onde → Kuih Kaswi Pandan (×2)
- Kuih Kaswi Pandan → Kuih Talam (×1), Onde‑Onde (×1)
- Kuih Seri Muka → Kuih Talam (×1)
- Kuih Lapis → Kek Lapis (×1)

### ROC & Precision–Recall curves

![ROC and PR curves](../results/roc_prc_curves.png)

Per‑class AUC ≈ 1.00 across the board.

### Training curves

![Training and validation curves](../results/training_curves.png)

Training accuracy climbs from ~0.94 to ~0.999 over 10 epochs; validation accuracy
sits at ~0.99 from the first epoch (a consequence of strong pretraining + heavy
augmentation) and tracks training closely — good generalization, with only mild
late‑epoch overfitting.

*(All plots are regenerated by [`notebooks/5_evaluate.ipynb`](../notebooks/5_evaluate.ipynb)
and [`notebooks/4_train.ipynb`](../notebooks/4_train.ipynb), and live in
[`results/`](../results).)*

---

## 3. The 8 classes

The goal is to identify which of eight traditional kuih appears in a photo. These
desserts are genuinely hard to tell apart by eye — several share the same palette
(pandan green, coconut white, gula‑melaka brown) and a similar steamed, layered,
or glutinous texture — which makes it a meaningful fine‑grained classification
problem rather than a toy one.

The label set ([`labels.txt`](../labels.txt)) — index → name:

| Index | Class | Notes |
|---|---|---|
| 0 | **Kek Lapis** | Layered ("thousand‑layer") cake, often colourful bands |
| 1 | **Kuih Kaswi Pandan** | Chewy pandan‑flavoured steamed kuih |
| 2 | **Kuih Ketayap** | Green pandan crêpe rolled around coconut + gula melaka |
| 3 | **Kuih Lapis** | Steamed layered rice‑flour kuih (soft, peelable layers) |
| 4 | **Kuih Seri Muka** | Two‑layer: glutinous rice base + green custard top |
| 5 | **Kuih Talam** | Two‑layer: pandan/gula base + salty coconut top |
| 6 | **Kuih Ubi Kayu** | Tapioca (cassava) cake |
| 7 | **Onde‑Onde** | Glutinous rice balls with gula‑melaka centre, rolled in coconut |

A couple of example photos per class are in
[`data/sample/`](../data/sample) (folder names use `UPPER_SNAKE_CASE`, e.g.
`KEK_LAPIS`).

> **Why the indices line up:** the code derives labels from the sorted class‑folder
> names via scikit‑learn's `LabelEncoder`, which orders them alphabetically — and
> that alphabetical order happens to match the index order in `labels.txt`.

---

## 4. Building the dataset from scratch

There was no ready-made kuih dataset — so we made one. This was one of the most
demanding parts of the project, and the pipeline has six stages:

```
 (1) COLLECT           (2) SORT            (3) BALANCE          (4) AUGMENT (offline)     (5) SPLIT            (6) AUGMENT (online)
 ~3,000 photos    →    into 8 classes  →   cap each class  →    11 variants / image   →   train/val/test  →   random transforms
 (4 of us)                                 (balance)            (~15,000 images)          (80/10/10)           every epoch
```

> The full image set (~20 GB after augmentation) is **not** shipped in this repo —
> only a small [`data/sample/`](../data/sample). [`data/README.md`](../data/README.md)
> explains how to rebuild it, and the whole pipeline is reproducible from the
> notebooks and scripts described below.

### Stage 1 — Collection
Four of us photographed kuih ourselves. The raw contribution split:

| Contributor | Photos |
|---|---|
| Ze Wei | ~1,320 |
| Wei Han | ~858 |
| Lilliana | ~448 |
| Hao Yuan | ~445 |
| **Total** | **~3,000** (plus a small set sourced online) |

Photos came in mixed real‑world formats — `.jpg`, `.jpeg`, `.png`, and iPhone
`.HEIC` (decoded explicitly via `pillow‑heif`).

### Stage 2 — Sorting
Every photo was sorted into its class folder. As expected from real collection,
the classes were imbalanced — from ~200 (Kuih Ketayap) to ~700 (Kuih Talam) photos.

### Stage 3 — Balancing — [`notebooks/1_trim.ipynb`](../notebooks/1_trim.ipynb)
To reduce that imbalance, each class was randomly **down‑sampled to a cap** (using
`pandas.DataFrame.sample()` + `shutil.copyfile`), landing each class in the
~150–400 range.

### Stage 4 — Offline augmentation — [`notebooks/2_augment.ipynb`](../notebooks/2_augment.ipynb) / [`src/augment.py`](../src/augment.py)
Each source image is resized to **224×224** and expanded into **11 variants** (the
original plus 10 transforms):

| # | Variant | How |
|---|---|---|
| 1 | Original | resize to 224×224 |
| 2 | Translated | random shift x,y ∈ [−75, 75] px |
| 3 | Rotated | random angle ∈ [−90°, 90°] |
| 4 | Flipped | random horizontal/vertical |
| 5 | Random crop | crop a 60% region, re‑centre on a black canvas (no interpolation) |
| 6 | Gaussian blur | 9×9 kernel |
| 7 | Gaussian noise | mean 0, std 25 |
| 8 | Salt‑and‑pepper noise | 2% of pixels |
| 9 | Brightness/contrast | contrast ×2.3, +10 brightness |
| 10 | Sharpen | 3×3 sharpening kernel |
| 11 | Colour enhance | HSV hue ×0.7, sat ×1.5, val ×0.5 |

This turns ~2,400 balanced photos into **~15,000 images** and teaches the model to
be robust to blur, lighting, orientation, and noise.

### Stage 5 — Split
The augmented set is split **80 / 10 / 10** into train / validation / test with the
`split‑folders` library and a fixed seed (`1427`) so the split is reproducible.

### Stage 6 — Online (on‑the‑fly) augmentation
On top of the offline set, the training `DataLoader` applies further random
augmentation **every epoch**, so the model rarely sees the exact same tensor twice:

```python
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.RandomErasing(p=0.3),      # random occlusion
    transforms.ColorJitter(brightness=0.3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])
```

Validation/test transforms skip the random parts (only resize + normalize).

### Final dataset sizes
| Split | Images |
|---|---|
| Train | ~14,400 (1,800 / class) |
| Validation | ~320–600 |
| Test (held‑out) | 280 (35 / class) |

---

## 5. Model architecture

`CustomResNet` — see [`src/model.py`](../src/model.py):

```
Input: 3 × 224 × 224
│
├── ResNet‑50 backbone (ImageNet‑pretrained, IMAGENET1K_V1)
│     • early layers  → FROZEN (requires_grad = False)
│     • deepest stage(s) (layer4, optionally layer3) → TRAINABLE (fine‑tuned)
│     • original fc   → replaced with nn.Identity()   → 2048‑dim feature vector
│
└── Custom classifier head (all trainable)
      Flatten
      Linear(2048 → 1024) → ReLU → Dropout(≈0.44)
      Linear(1024 → 512)  → ReLU → Dropout(≈0.44)
      Linear(512 → 8)                       → 8 class logits
```

**Why these choices:**

- **Transfer learning, not from scratch.** With only a few thousand original
  photos, training a 25M‑parameter CNN from scratch would overfit badly. Reusing
  ImageNet features and fine‑tuning is the right call for a small dataset.
- **Partial freezing.** Early conv layers detect generic edges/textures that
  transfer across domains, so they stay frozen. The deepest stage(s) learn
  dataset‑specific, higher‑level features and are unfrozen — adapting the network
  to kuih while limiting overfitting.
- **A deeper head with dropout.** The 2048→1024→512→8 head with dropout gives the
  classifier capacity while regularising heavily against the small dataset.
- **`nn.Identity()` trick.** Replacing ResNet's final FC with identity cleanly
  exposes the 2048‑dim feature vector to the custom head.

**Parameter counts** (fine‑tuning the two deepest stages):

| | Params |
|---|---|
| Total | 26,135,112 |
| Trainable | 24,690,184 |
| Frozen | 1,444,928 |

---

## 6. Hyperparameter optimization (Bayesian Optimization)

Rather than guessing hyperparameters or grid‑searching, we tuned them with
**Bayesian Optimization** using [Hyperopt](https://github.com/hyperopt/hyperopt)
and the **TPE** (Tree‑structured Parzen Estimator) algorithm — the most
technically ambitious part of the project. See
[`notebooks/3_bayesian_optimization.ipynb`](../notebooks/3_bayesian_optimization.ipynb).

### The search space

```python
space = {
    'learning_rate': hp.loguniform('learning_rate', -10, -4),  # ≈ 5e‑5 … 2e‑2
    'batch_size':    hp.choice('batch_size', [16, 32, 64, 128]),
    'dropout':       hp.uniform('dropout', 0.1, 0.6),
    'weight_decay':  hp.loguniform('weight_decay', -8, -3),    # ≈ 3e‑4 … 5e‑2
    'momentum':      hp.uniform('momentum', 0.90, 0.99),
}
```

### The objective — the clever bit
Each trial builds a fresh model, trains it for 5 epochs (with a
`ReduceLROnPlateau` scheduler), and returns a **loss that explicitly penalises
overfitting**:

```python
generalization_gap = avg_val_loss - avg_train_loss
combined_loss      = avg_val_loss + 0.5 * generalization_gap
```

Instead of optimizing validation loss alone, the objective adds half the
train↔val gap, so a model that memorises the training set but generalises poorly
is punished. The search is steered toward hyperparameters that *generalise*.

### Analysis — [`notebooks/6_bo_analysis.ipynb`](../notebooks/6_bo_analysis.ipynb)
The saved trials object ([`results/final_bo.pkl`](../results/final_bo.pkl)) is
analysed after the fact: loss‑vs‑iteration, statistical outlier removal (|z| < 2),
the **running‑minimum ("best so far") loss** that shows convergence
([`results/running_min_loss.png`](../results/running_min_loss.png)), and a ranked
table of the top trials ([`results/bo_loss_per_iteration.csv`](../results/bo_loss_per_iteration.csv)).

One recorded search ran **50 trials over ~4h21m** on GPU. A representative best
configuration:

```python
{'batch_size': 128, 'dropout': 0.4164, 'learning_rate': 0.005668,
 'momentum': 0.9350, 'weight_decay': 0.008656}
```

> The precise, "ugly" numbers hard‑coded in the training/eval notebooks
> (e.g. `weight_decay = 0.001471975670040013`, `Dropout(p=0.4455…)`) are the
> fingerprints of BO outputs — they were *searched*, not chosen by hand.

---

## 7. Training

See [`notebooks/4_train.ipynb`](../notebooks/4_train.ipynb).

| Setting | Value |
|---|---|
| Loss | `CrossEntropyLoss` |
| Optimizer | `SGD` (momentum + weight decay); `Adam` was also trialled |
| Learning rate | `1e‑4` |
| Batch size | 16 |
| Epochs | 10 |
| Weight decay (L2) | `0.001471975670040013` (from BO) |
| Momentum | 0.9 |
| Device | CUDA if available, else CPU |

**Two implementation details worth noting:**

- **`param.grad = None` instead of `optimizer.zero_grad()`.** A deliberate
  micro‑optimization — setting gradients to `None` skips a memory write instead of
  zeroing tensors, which is marginally lighter.
- Per‑batch and per‑epoch loss/accuracy are tracked and plotted
  ([`results/training_curves.png`](../results/training_curves.png),
  [`results/training_batches.png`](../results/training_batches.png)).

The training run reaches ~0.96 train / ~0.98 validation accuracy over 10 epochs,
then the weights are saved.

---

## 8. Evaluation

See [`notebooks/5_evaluate.ipynb`](../notebooks/5_evaluate.ipynb). It loads the
trained weights and runs the held‑out test set through the model, computing a full
report with scikit‑learn:

- **Precision / recall / F1** per class and macro‑averaged.
- **Confusion matrix** (seaborn heatmap → [`results/confusion_matrix.png`](../results/confusion_matrix.png)).
- **ROC curves + AUC** per class (one‑vs‑rest, on softmax probabilities).
- **Precision‑Recall curves + AUC** per class → [`results/roc_prc_curves.png`](../results/roc_prc_curves.png).

Results are in [§2](#2-headline-results).

---

## 9. The journey — iterations, trials & errors

The final numbers are clean, but getting there was not a single lucky run — it was
months of iteration. This section is the honest history.

**We started in TensorFlow, then moved to PyTorch.** The very first prototype was
a Keras model (`ImageDataGenerator` + a frozen ResNet‑50 with a
`GlobalAveragePooling` head) trained on a tiny 5‑class set of ~40 images. It hit
100% validation accuracy instantly — which taught us an early lesson about how
meaningless metrics are on a trivial dataset. We rebuilt everything in PyTorch for
finer control over the architecture, the freezing strategy, and the training loop.

**The dataset was rebuilt seven times.** Internally the datasets were versioned
`testOne` → `testSeven`. Each version was a full regeneration as we collected more
photos, improved class balance, and tuned the augmentation. Early sets were tiny
and noisy; the final one (`testSeven`, plus a separate held‑out `evalFINAL` test
set) was large, balanced, and clean.

**We ran 30+ training experiments.** Model checkpoints were tagged `[a]`, `[b]`, …
all the way through `[z]`, `[z_b]`, `[z_c]`, `[z_d]` — each a distinct run with its
own training curves. We compared freezing strategies (freeze everything, unfreeze
`layer4`, unfreeze `layer3`+`layer4`), head depths, dropout rates, and optimizers
(SGD vs Adam), keeping the plots for each so we could see what actually helped.

**Bayesian Optimization evolved too.** It began by tuning just two knobs (learning
rate + batch size), then grew into the five‑parameter search with the
overfitting‑aware objective described in [§6](#6-hyperparameter-optimization-bayesian-optimization).
Several search runs were saved and compared before we settled on the final config.

**Peer cross-evaluation.** The competition had teams evaluate one another's models
on held‑out test sets. We built a Colab notebook that pulled each team's model and
test images from Google Drive (via `gdown`), aligned their label indices with ours,
ran inference, and produced a full metrics report — the shared "submission format"
that let scores be compared fairly across teams.

That accumulated effort — the rebuilds, the tagged runs, the searches — is what the
2nd‑Runner‑Up result is built on.

---

## 10. Repository structure

```
ZYLCH_AI/
├── README.md                 # Project front page: results & quickstart
├── LICENSE                   # MIT
├── requirements.txt          # Dependencies
├── labels.txt                # index → class name
├── docs/
│   └── PROJECT_DOCUMENTATION.md   # this document
├── notebooks/                # the pipeline, in order
│   ├── 1_trim.ipynb                    # class balancing
│   ├── 2_augment.ipynb                 # offline augmentation
│   ├── 3_bayesian_optimization.ipynb   # Hyperopt TPE search
│   ├── 4_train.ipynb                   # training
│   ├── 5_evaluate.ipynb                # metrics, confusion matrix, ROC/PRC
│   └── 6_bo_analysis.ipynb             # BO convergence analysis
├── src/
│   ├── model.py              # CustomResNet — single source of truth
│   ├── augment.py            # augmentation as a command-line script
│   └── infer.py              # load the model + predict on an image
├── results/                  # result plots, BO loss table & trials object
│   ├── confusion_matrix.png
│   ├── roc_prc_curves.png
│   ├── running_min_loss.png
│   ├── training_curves.png
│   ├── training_batches.png
│   ├── bo_loss_per_iteration.csv
│   └── final_bo.pkl
├── data/
│   ├── README.md             # how to rebuild the full dataset
│   └── sample/               # a few example photos per class
└── models/
    └── README.md             # how to obtain the trained weights
```

> The trained weights (~100 MB) and the full ~20 GB image dataset are not stored in
> git — [`models/README.md`](../models/README.md) and [`data/README.md`](../data/README.md)
> explain how to obtain/rebuild them.

---

## 11. Notebook & script reference

| File | Purpose |
|---|---|
| [`notebooks/1_trim.ipynb`](../notebooks/1_trim.ipynb) | Class balancing — randomly down‑samples each class folder to a cap. |
| [`notebooks/2_augment.ipynb`](../notebooks/2_augment.ipynb) | Offline augmentation — 11 variants per image, then an 80/10/10 split. |
| [`notebooks/3_bayesian_optimization.ipynb`](../notebooks/3_bayesian_optimization.ipynb) | Bayesian Optimization — 5‑parameter Hyperopt TPE search with the generalization‑gap objective. |
| [`notebooks/4_train.ipynb`](../notebooks/4_train.ipynb) | Training — data loading, the `CustomResNet`, the SGD training loop, and curve plots. |
| [`notebooks/5_evaluate.ipynb`](../notebooks/5_evaluate.ipynb) | Evaluation — metrics, confusion matrix, ROC/PRC on the held‑out test set. |
| [`notebooks/6_bo_analysis.ipynb`](../notebooks/6_bo_analysis.ipynb) | BO analysis — convergence plots and a ranked table of the best trials. |
| [`src/model.py`](../src/model.py) | `CustomResNet` definition + the canonical class names. Imported by the scripts. |
| [`src/augment.py`](../src/augment.py) | The augmentation pipeline as a clean CLI (`python src/augment.py <folder>`). |
| [`src/infer.py`](../src/infer.py) | Load the model and predict on one or more images from the command line. |

---

## 12. The trained model & how to use it

The final weights are distributed as a PyTorch **`state_dict`**
(`models/final_statedict.pt`, ~100 MB). Because that's over GitHub's file‑size
limit, it isn't committed to git — [`models/README.md`](../models/README.md)
explains where to download it (GitHub Release / Git LFS / Google Drive).

**Predict from the command line:**
```bash
python src/infer.py data/sample/ONDE_ONDE/*.jpg --weights models/final_statedict.pt
```

**Or in code:**
```python
import torch
from src.model import CustomResNet, CLASS_NAMES

model = CustomResNet()
model.load_state_dict(torch.load("models/final_statedict.pt", map_location="cpu"))
model.eval()
```

> **Two ways to save a PyTorch model.** A **`state_dict`** stores just the learned
> weights (portable, version‑safe — recommended). Saving the **entire model**
> pickles the whole object, which is convenient but ties loading to the exact class
> definition and library versions. The model can also be exported to **TorchScript**
> for Python‑independent deployment. This project ships the `state_dict`.

---

## 13. Tech stack

**Core:** Python 3 · **PyTorch** + **torchvision** (ResNet‑50, transforms) · CUDA GPU · torchsummary

**Data & augmentation:** OpenCV · Pillow + **pillow‑heif** (HEIC support) · NumPy · pandas · **split‑folders**

**ML tooling:** scikit‑learn (`LabelEncoder`, metrics, ROC/PRC) · **Hyperopt** (Bayesian Optimization) · SciPy

**Visualization:** matplotlib · seaborn

**Sharing:** gdown (pull models/test sets from Google Drive)

*Originally prototyped in TensorFlow/Keras before migrating to PyTorch.* All
dependencies are listed in [`requirements.txt`](../requirements.txt).

---

## 14. Reproducing the project

The `src/` scripts take paths as arguments; the notebooks are kept close to their
original competition form, so a few contain example paths you'll point at your own
data.

1. **Environment**
   ```bash
   conda create -n kuih python=3.10 && conda activate kuih
   pip install -r requirements.txt
   ```
2. **Build the dataset** — put your photos in one folder per class, then:
   - `notebooks/1_trim.ipynb` to balance classes,
   - `python src/augment.py data/raw --ratio 0.8 0.1 0.1` to augment ×11 and split.
3. **(Optional) Tune** — run `notebooks/3_bayesian_optimization.ipynb`; inspect
   convergence with `notebooks/6_bo_analysis.ipynb`.
4. **Train** — run `notebooks/4_train.ipynb` (point `input_path` at your split
   output). Saves the weights + training curves.
5. **Evaluate** — run `notebooks/5_evaluate.ipynb` against a held‑out test split.
6. **Infer** — `python src/infer.py <image> --weights models/final_statedict.pt`.

---

## 15. Design decisions & honest caveats

A trustworthy project states its limitations.

1. **Two fine‑tuning configurations exist.** During experimentation we tried
   unfreezing `layer3`+`layer4` (as in the training notebook) and unfreezing only
   `layer4` (as in the final evaluation), with slightly different dropout values
   (≈0.42 vs ≈0.44). Both load the same `state_dict` cleanly; the evaluation
   configuration is the authoritative one for the reported results.
2. **ImageNet normalization on a non‑ImageNet domain.** We normalise with ImageNet
   mean/std. It works well, but dataset‑specific statistics would be a natural
   refinement (the code notes this).
3. **Small, self‑collected dataset.** ~3,000 originals is small for deep learning;
   the heavy two‑stage augmentation and strong regularization (dropout, weight
   decay, the generalization‑gap objective) are what make the results hold up.
4. **Notebook outputs reflect real competition runs.** Some cells show outputs from
   specific past runs; re‑running a notebook top‑to‑bottom regenerates consistent
   numbers on your own data.
5. **Test‑set scale.** The held‑out test set is 280 images (35/class) — enough to
   be meaningful, but small enough that a single image shifts a per‑class metric by
   ~3 points.

---

## 16. Glossary

**Kuih terms**
- **Kuih / Kueh** — bite‑sized traditional Malaysian/Indonesian/Singaporean cakes
  and sweets, often steamed and made with rice flour, coconut, and pandan.
- **Pandan** — a fragrant green leaf used as flavouring/colouring.
- **Gula melaka** — palm sugar, the caramel‑brown sweetener in many kuih.
- **Nyonya / Peranakan** — the Straits‑Chinese culture these desserts come from.

**ML terms**
- **Transfer learning** — reusing a network pretrained on a large dataset
  (ImageNet) and adapting it to a new, smaller task.
- **Fine‑tuning / freezing** — unfreezing only some layers so pretrained features
  are partly retrained; frozen layers keep their weights.
- **Data augmentation (offline vs online)** — synthetically expanding a dataset
  with transformed copies; offline = pre‑generated to disk, online = applied
  randomly at load time each epoch.
- **Bayesian Optimization / TPE** — a sample‑efficient way to search
  hyperparameters by modelling which regions of the space look promising.
- **Generalization gap** — the difference between validation and training loss; a
  proxy for overfitting.
- **state_dict** — a PyTorch dictionary of a model's learned tensors (the
  recommended way to save/load weights).
- **Macro average** — a metric averaged equally across classes, so rare classes
  count as much as common ones.

---

## 17. Credits

**Team ZYLCH AI**
- Ze Wei
- Hao Yuan
- Lilliana
- Wei Han

The competition project was built entirely by hand (no AI tools); this
repository's README and documentation were later polished with
[Claude](https://claude.ai).

Licensed under the [MIT License](../LICENSE).
