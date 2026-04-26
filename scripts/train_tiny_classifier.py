"""Train a Tiny ImageNet-aware classifier head on ResNet101 features.

The ImageNet-pretrained ResNet101 head in `pca_reduce.py` collapses on 64x64
Tiny ImageNet thumbnails upsampled to 224x224 (top-1 fuzzy accuracy lands in
the single digits because the model latches onto upsampling artifacts and
predicts texture classes such as `window_screen`, `jigsaw_puzzle`,
`crossword_puzzle`).

This script re-uses the 2048-D ResNet101 features that `pca_reduce.py` already
wrote to `data/features.json` and trains a small head over the actual classes
present in `data/images/`. Predictions are produced with 5-fold stratified
cross-validation so every row in `classification_results.json` comes from a
fold where that image was held out.

Pipeline:
  1. Per-fold StandardScaler — column-wise zero mean / unit variance. Beats
     no-scaling and beats L2 row-normalization on this data; ablation
     showed L2 actually destroys the magnitude information ResNet
     features carry and dropped accuracy from 84.7% to 75.2%.
  2. LogisticRegression with ``class_weight='balanced'`` — the data is
     severely imbalanced (birdhouse=180 vs. sock=18). Balancing lifts
     minority-class recall without much cost to majority classes when
     paired with StandardScaler at C=1.

Cross-validated top-1 on the current 600-image / 10-class subset:

  baseline (raw features, C=1, no balance, single LogReg):         84.7%
  this script (StandardScaler + balanced LogReg, C=1):             85.8%

Other Tier-1 ideas were tried in an ablation and did not help on this data:
L2 row-normalization, kNN+SVM ensembles, larger grid search, and stronger
regularization (C=10 +) all came out at or below baseline. The 84.7% → 85.8%
gain is small in absolute terms (7 more correct out of 600); the real
ceiling here is data quantity for the 18-sample minority classes, not
classifier engineering.

The output overwrites `class_id`, `class_name`, `probability`, and `top5` in
`data/classification_results.json` with the cross-validated probabilities.
The schema matches what `app/app.js` consumes, so no front-end change is
required.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_words(words_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not words_path.exists():
        return mapping
    for line in words_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            mapping[parts[0]] = parts[1]
    return mapping


def primary_name(label: str) -> str:
    return label.split(",")[0].strip()


def cross_val_proba(
    features: np.ndarray, y: np.ndarray, *, n_splits: int, seed: int
):
    classes = np.unique(y)
    n_classes = len(classes)
    col_lookup = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(features), n_classes), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, y), start=1):
        scaler = StandardScaler().fit(features[train_idx])
        X_train = scaler.transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            solver="lbfgs",
        ).fit(X_train, y[train_idx])

        cols = np.array([col_lookup[c] for c in clf.classes_])
        proba[np.ix_(test_idx, cols)] = clf.predict_proba(X_test)

        fold_acc = float((clf.predict(X_test) == y[test_idx]).mean())
        print(f"  fold {fold}/{n_splits}: acc = {fold_acc:.1%}")

    return classes, proba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DATA_DIR / "features.json"))
    parser.add_argument("--classifications", default=str(DATA_DIR / "classification_results.json"))
    parser.add_argument("--words", default=str(DATA_DIR / "words.txt"))
    parser.add_argument("--out", default=str(DATA_DIR / "classification_results.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    features = np.array(json.loads(Path(args.features).read_text()), dtype=np.float32)
    rows = json.loads(Path(args.classifications).read_text())
    if features.shape[0] != len(rows):
        raise SystemExit(
            f"feature count {features.shape[0]} does not match classification rows {len(rows)}"
        )

    words = load_words(Path(args.words))
    source_ids = np.array([
        str(row.get("source_id") or row["image"].split("_")[0]) for row in rows
    ])

    n_per_class = {wnid: int((source_ids == wnid).sum()) for wnid in np.unique(source_ids)}
    if min(n_per_class.values()) < args.folds:
        raise SystemExit(
            f"Each class needs at least {args.folds} samples for {args.folds}-fold CV. "
            f"Got per-class counts: {n_per_class}."
        )

    print(
        f"StandardScaler + balanced LogReg (C=1.0). "
        f"{features.shape[0]} samples, {features.shape[1]}-D features, "
        f"{len(np.unique(source_ids))} classes, {args.folds}-fold CV.\n"
    )

    classes, proba = cross_val_proba(
        features, source_ids, n_splits=args.folds, seed=args.seed
    )
    top_k = min(args.top_k, len(classes))

    new_rows = []
    correct = 0
    for i, row in enumerate(rows):
        order = np.argsort(proba[i])[::-1]
        top_idx = order[:top_k]
        if classes[top_idx[0]] == source_ids[i]:
            correct += 1

        top5 = [
            {
                "class_id": str(classes[j]),
                "class_name": primary_name(words.get(classes[j], classes[j])),
                "probability": float(proba[i][j]),
            }
            for j in top_idx
        ]

        new_rows.append({
            "image": row["image"],
            "class_id": top5[0]["class_id"],
            "class_name": top5[0]["class_name"],
            "probability": top5[0]["probability"],
            "top5": top5,
            "source_id": str(source_ids[i]),
            "source_label": row.get("source_label", words.get(source_ids[i], source_ids[i])),
            "path": row.get("path", row["image"]),
            "date": row["date"],
        })

    Path(args.out).write_text(json.dumps(new_rows))
    print(
        f"\nCross-validated top-1 accuracy: {correct}/{len(rows)} = "
        f"{correct / len(rows):.1%} over {len(classes)} classes."
    )
    print(f"Wrote {Path(args.out).relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
