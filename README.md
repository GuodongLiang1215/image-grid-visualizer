# Image Grid Visualizer

An interactive browser dashboard for exploring a small image collection through
**ResNet101 visual features**, a **t-SNE 2D projection**, and a **classifier
trained directly on the dataset's actual labels**. Built with vanilla HTML +
D3 + a Node static server; the heavy ML runs offline as a one-shot Python
pipeline.

The current dataset is **600 images / 10 classes** drawn from
[Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) (a Stanford-curated
subset of ImageNet at 64×64 resolution). Concretely:

| WordNet ID | Class | Images |
|---|---|---|
| n01443537 | goldfish | 78 |
| n01629819 | European fire salamander | 18 |
| n02843684 | birdhouse | 180 |
| n02909870 | bucket | 108 |
| n03670208 | limousine | 18 |
| n03983396 | pop bottle | 24 |
| n04254777 | sock | 18 |
| n04376876 | syringe | 18 |
| n07753592 | banana | 42 |
| n09193705 | alp | 96 |

The 10 classes are a hand-picked subset; the project is intentionally small so
the whole pipeline runs in minutes on a CPU and the JSON outputs ship with the
repo.

## Quick Start

```bash
node server.js
# open http://localhost:4173/app/
```

`server.js` is a 60-line Node static server. It auto-increments the port if
`4173` is busy and prints the URL it bound to.

The shipped `data/*.json` files are already up to date, so the dashboard works
out of the box. Re-running the Python pipeline is only required if you change
the images, the model, or the classifier.

## What's In The Repo

```
app/                     vanilla HTML + JS + D3 dashboard
  index.html             page skeleton (top bar, sidebar, main + lower panels)
  app.js                 all interaction logic (filters, sort, color modes,
                         tooltips, bar chart, timeline, brush selection)
  styles.css
  vendor/d3.v7.js        local D3 (no CDN at runtime)

data/
  images/                600 JPEGs at 64×64 native resolution (Tiny ImageNet)
  features.json          600 × 2048 ResNet101 avg-pool features
  pca_results.json       600 × 50 PCA-reduced features
  tsne_results.json      600 × 2 t-SNE coordinates (drives the grid layout)
  classification_results.json
                         per-image top-5 from the Tiny-ImageNet-aware
                         classifier (see "Classification" below)
  words.txt              WordNet-ID → human-readable label table

scripts/
  pca_reduce.py          ResNet101 → features → PCA(50) → t-SNE(2)
                         (TensorFlow + Keras; needs Python 3.11)
  train_tiny_classifier.py
                         trains a 10-class head on top of features.json and
                         overwrites classification_results.json with
                         cross-validated probabilities
                         (only needs numpy + scikit-learn)

server.js                Node static file server
```

## Data Pipeline (Two Stages)

### Stage 1 — `scripts/pca_reduce.py` (heavy, run once)

A single ResNet101 backbone (ImageNet-pretrained) is loaded with
`include_top=True`. The same forward pass produces:

1. **2048-D features** taken from the `avg_pool` layer → `features.json`
2. **Top-5 ImageNet predictions** from the 1000-class head (these get
   replaced in stage 2; they are saved as a baseline for comparison)
3. **PCA(50)** noise reduction → `pca_results.json`
4. **t-SNE(2)** with `init="pca"`, `learning_rate="auto"`, `random_state=42`
   → `tsne_results.json`

Requires: TensorFlow + scikit-learn + numpy + matplotlib in a Python 3.11
virtual environment. On Windows:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install tensorflow scikit-learn matplotlib numpy
cd data
..\.venv311\Scripts\python.exe ..\scripts\pca_reduce.py
cd ..
```

The script uses relative paths (`input_dir = 'images'`), so run it from inside
`data/`. Re-runs are deterministic thanks to fixed seeds.

### Stage 2 — `scripts/train_tiny_classifier.py` (light, fast)

Tiny ImageNet thumbnails (64×64) upsampled 3.5× to 224×224 do not match the
distribution that ImageNet-pretrained heads were trained on. The 1000-class
head from stage 1 collapses on this input and predicts texture classes like
`window_screen`, `jigsaw_puzzle`, `crossword_puzzle` — top-1 fuzzy accuracy is
about **3.7%**. That is a data/resolution problem, not a model bug.

This script fixes it without touching the backbone. It:

1. Loads the existing 2048-D features from `data/features.json`.
2. Reads each filename's WordNet-ID prefix as the ground-truth label.
3. Per fold, fits a `StandardScaler` (column-wise zero-mean / unit-variance)
   on training features only.
4. Trains `LogisticRegression(C=1.0, class_weight='balanced')` on the scaled
   features against the actual classes present in `data/images/`.
5. Uses 5-fold stratified cross-validation, so every prediction stored in
   `classification_results.json` comes from a fold where that image was held
   out (no train/test leakage).
6. Overwrites `class_id`, `class_name`, `probability`, and `top5` in
   `classification_results.json`. The schema is unchanged, so the front-end
   needs no edits.

```powershell
.\.venv311\Scripts\python.exe scripts\train_tiny_classifier.py
```

Only needs `numpy` + `scikit-learn`. Runs in seconds.

## Classification Accuracy

| Stage | Model | Top-1 (CV) |
|---|---|---|
| Stage 1 only | ResNet101 ImageNet-1k head, 1000-way | ~3.7% (fuzzy) |
| Stage 2 — naive | `LogReg(C=1)` on raw features | 84.7% |
| **Stage 2 — current** | **`StandardScaler` + `LogReg(C=1, balanced)`** | **85.8%** |

Per-class breakdown of the current head:

| Class | n | Accuracy |
|---|---|---|
| pop bottle | 24 | 45.8% |
| sock | 18 | 50.0% |
| syringe | 18 | 66.7% |
| salamander | 18 | 77.8% |
| limousine | 18 | 83.3% |
| bucket | 108 | 81.5% |
| banana | 42 | 90.5% |
| birdhouse | 180 | 90.6% |
| goldfish | 78 | 92.3% |
| alp | 96 | 96.9% |

The ceiling here is bounded by data quantity — minority classes only have 18
samples each, and no choice of linear head can manufacture information that
isn't there. A much wider ablation (kNN / SVM ensembles, L2 row-norm,
hyperparameter grids) was tried; the simple StandardScaler + balanced
LogReg combination above is the only configuration that consistently beats
84.7% on this data.

## UI Features

**Two main views** (toggle in the top bar):

- `Image Grid` — t-SNE neighborhood-preserving grid. Each image is greedily
  assigned to the nearest free grid cell from its t-SNE coordinate, so
  visually-similar images stay close while filling the grid uniformly. Can
  alternatively be grouped by dataset label, model prediction, or
  match/mismatch.
- `Embedding Map` — raw t-SNE scatter plot, with `Pan` / `Box Select` modes.

**Sidebar controls:**

- Filter by predicted class, source label, confidence threshold,
  match / mismatch.
- Sort by confidence / date / filename.
- Adjust tile size.
- Three color modes: by predicted class, by dataset label, by agreement
  (prediction vs. source label).

**Lower panels:**

- Bar chart — distribution of predicted classes in the current filtered view.
- Timeline — simulated dates (generated with seed 42, useful only as
  a "metadata exists" demo).

**Label comparison logic** is intentionally lenient: a prediction counts as a
match when the top-1 WordNet ID equals the source WordNet ID, or the top-1 /
top-5 text overlaps any of the source synonyms. This avoids penalizing
near-synonyms like *tabby cat* vs. *Egyptian cat*.

## Future Directions

The current pipeline is a baseline. The two highest-impact next steps:

### Tier 2 — Stronger backbone for small images

ResNet101 was trained on 224×224 photos and is suboptimal for 64×64 input.
Replacing it with a backbone that handles small / varied inputs better would
give cleaner features:

- **DINOv2 (small)** — self-supervised, robust across resolutions, strong
  feature geometry; drop-in replacement for the ResNet101 stage.
- **CLIP (ViT-B/32)** — slightly larger, but adds a free open-vocabulary
  capability if the project later wants text search / zero-shot labels.

Only `scripts/pca_reduce.py` needs to change. `features.json`,
`pca_results.json`, `tsne_results.json`, and `classification_results.json`
get regenerated; the front end and the stage-2 classifier work unchanged
because the JSON schema is the same.

Expected accuracy gain: **+3 to +5 points** (ceiling ~92%).

### Tier 3 — Train a backbone on full Tiny ImageNet

The most principled fix: stop using a 224-pretrained backbone on 64×64
inputs at all. Instead, train a small CIFAR-style CNN (e.g. ResNet-20 /
WideResNet-16) **directly on the full 200-class Tiny ImageNet training
set** (~100k images). The resulting backbone is purpose-built for 64×64,
and its features feed the same stage-2 head.

Only the `train/` split is needed (≈ 200 MB on disk). `val/` and `test/`
are not used because evaluation is done on the project's own 600-image
subset via 5-fold CV.

Expected accuracy gain: **85.8% → 92–95%**.

### Tier 4 — Live single-image upload

Add a `POST /api/predict` endpoint (FastAPI sidecar) that takes an uploaded
image, runs the same backbone + stage-2 head, and returns top-5 predictions.
Optionally project the new image into the existing t-SNE map via k-NN on
features. This is mostly UX work; modeling is unchanged.

### Smaller-scope ideas

- Add UMAP as an alternative projection (faster, better global structure).
- Add trustworthiness / nearest-neighbor-overlap projection metrics.
- Add freehand lasso selection in the embedding map.
- Replace the simulated dates with real metadata once a real source
  is available.

## Notes

- The dates in `classification_results.json` are simulated random dates
  generated with seed 42. They are intentional placeholders and not real
  metadata.
- Random seeds are fixed (`random`, `numpy`, `tf`, `sklearn`) so any
  re-generation of the data files is reproducible.
- `data/images/` ships in the repo so the dashboard works without
  re-running anything. The full Tiny ImageNet 200-class dataset is not
  included.
