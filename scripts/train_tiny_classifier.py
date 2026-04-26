"""Train a Tiny ImageNet-aware classifier head on ResNet101 features.

The ImageNet-pretrained ResNet101 head in `pca_reduce.py` is well known to
collapse on 64x64 Tiny ImageNet thumbnails upsampled to 224x224 (top-1 fuzzy
accuracy lands in the single digits because the model latches onto upsampling
artifacts and predicts texture classes such as `window_screen`,
`jigsaw_puzzle`, `crossword_puzzle`).

This script re-uses the 2048-D ResNet101 features that `pca_reduce.py` already
wrote to `data/features.json` and trains a multinomial logistic regression
over the actual classes present in `data/images/`. Predictions are produced
with 5-fold stratified cross-validation so every row in
`classification_results.json` comes from a fold where that image was held
out.

The output overwrites `class_id`, `class_name`, `probability`, and `top5` in
`data/classification_results.json` with the cross-validated probabilities.
The schema matches what `app/app.js` consumes, so no front-end change is
required.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


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
    # Tiny ImageNet labels are often comma-separated synonyms ("goldfish,
    # Carassius auratus"). The first token reads cleaner in the UI.
    return label.split(",")[0].strip()


def cross_val_proba(features: np.ndarray, y: np.ndarray, *, n_splits: int, seed: int):
    classes = np.unique(y)
    proba = np.zeros((len(features), len(classes)), dtype=np.float64)

    # Each fold may not see every class, so map column indices back to the
    # global class order before assigning into `proba`.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(features, y):
        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
        )
        clf.fit(features[train_idx], y[train_idx])
        fold_classes = clf.classes_
        col_lookup = {c: i for i, c in enumerate(classes)}
        cols = np.array([col_lookup[c] for c in fold_classes])
        proba[np.ix_(test_idx, cols)] = clf.predict_proba(features[test_idx])
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

    features_path = Path(args.features)
    classifications_path = Path(args.classifications)
    words_path = Path(args.words)
    out_path = Path(args.out)

    features = np.array(json.loads(features_path.read_text()), dtype=np.float32)
    rows = json.loads(classifications_path.read_text())
    if features.shape[0] != len(rows):
        raise SystemExit(
            f"feature count {features.shape[0]} does not match classification rows {len(rows)}"
        )

    words = load_words(words_path)
    source_ids = np.array([
        str(row.get("source_id") or row["image"].split("_")[0]) for row in rows
    ])

    n_per_class = {wnid: int((source_ids == wnid).sum()) for wnid in np.unique(source_ids)}
    if min(n_per_class.values()) < args.folds:
        raise SystemExit(
            f"Each class needs at least {args.folds} samples for {args.folds}-fold CV. "
            f"Got per-class counts: {n_per_class}."
        )

    classes, proba = cross_val_proba(features, source_ids, n_splits=args.folds, seed=args.seed)
    top_k = min(args.top_k, len(classes))

    new_rows = []
    correct = 0
    for i, row in enumerate(rows):
        order = np.argsort(proba[i])[::-1]
        top_idx = order[:top_k]
        top1 = classes[top_idx[0]]
        if top1 == source_ids[i]:
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

    out_path.write_text(json.dumps(new_rows))
    print(
        f"Cross-validated top-1 accuracy: {correct}/{len(rows)} = "
        f"{correct / len(rows):.1%} over {len(classes)} classes."
    )
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
