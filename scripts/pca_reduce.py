import os
import json
import argparse
import random
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess,
    decode_predictions,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    return image.img_to_array(img)


def get_random_date(start_date, end_date):
    delta = end_date - start_date
    return start_date + datetime.timedelta(days=random.randint(0, delta.days))


def load_tiny_imagenet_words(words_path):
    labels = {}
    if not os.path.exists(words_path):
        return labels
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="images")
    parser.add_argument("--words", default="../tiny-imagenet-200/words.txt")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot", action="store_true",
                        help="Save a t-SNE preview PNG to the output directory.")
    args = parser.parse_args()

    if args.plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

    os.makedirs(args.out_dir, exist_ok=True)
    output_features_file = os.path.join(args.out_dir, "features.json")
    output_classification_file = os.path.join(args.out_dir, "classification_results.json")
    output_tsne_file = os.path.join(args.out_dir, "tsne_results.json")
    output_pca_file = os.path.join(args.out_dir, "pca_results.json")

    resnet_model = ResNet101(weights="imagenet", include_top=False, pooling="avg")
    mobilenet_model = MobileNetV2(weights="imagenet")

    source_labels = load_tiny_imagenet_words(args.words)
    image_names = sorted(
        n for n in os.listdir(args.input_dir)
        if n.lower().endswith(("jpg", "jpeg", "png"))
    )
    if not image_names:
        raise SystemExit(f"No images found in {args.input_dir}")

    start_date = datetime.date(2023, 7, 1)
    end_date = datetime.date(2024, 7, 1)

    features_list = []
    classification_results = []

    for batch_names in batched(image_names, args.batch_size):
        raw = np.stack([load_image(os.path.join(args.input_dir, n)) for n in batch_names])
        resnet_features = resnet_model.predict(resnet_preprocess(raw.copy()), verbose=0)
        mobilenet_logits = mobilenet_model.predict(mobilenet_preprocess(raw.copy()), verbose=0)
        top5_batch = decode_predictions(mobilenet_logits, top=5)

        for img_name, feature, top5 in zip(batch_names, resnet_features, top5_batch):
            features_list.append(feature.flatten().tolist())
            top1 = top5[0]
            source_id = img_name.split("_")[0]
            classification_results.append({
                "image": img_name,
                "class_id": top1[0],
                "class_name": top1[1],
                "probability": float(top1[2]),
                "top5": [
                    {"class_id": p[0], "class_name": p[1], "probability": float(p[2])}
                    for p in top5
                ],
                "source_id": source_id,
                "source_label": source_labels.get(source_id, source_id),
                "path": os.path.join(args.input_dir, img_name).replace("\\", "/"),
                "date": get_random_date(start_date, end_date).isoformat(),
            })

    features_array = np.array(features_list)
    pca_results = PCA(n_components=50, random_state=SEED).fit_transform(features_array)
    tsne_results = TSNE(
        n_components=2,
        perplexity=min(30, max(5, (len(pca_results) - 1) // 3)),
        init="pca",
        learning_rate="auto",
        random_state=SEED,
    ).fit_transform(pca_results).tolist()

    with open(output_features_file, "w") as f:
        json.dump(features_list, f)
    with open(output_classification_file, "w") as f:
        json.dump(classification_results, f)
    with open(output_pca_file, "w") as f:
        json.dump(pca_results.tolist(), f)
    with open(output_tsne_file, "w") as f:
        json.dump(tsne_results, f)

    print(
        f"Wrote {len(features_list)} rows to "
        f"{output_features_file}, {output_classification_file}, "
        f"{output_pca_file}, {output_tsne_file}"
    )

    if args.plot:
        arr = np.array(tsne_results)
        plt.figure(figsize=(10, 8))
        plt.scatter(arr[:, 0], arr[:, 1], s=20, alpha=0.5)
        plt.title("t-SNE Results")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        out_png = os.path.join(args.out_dir, "tsne_preview.png")
        plt.savefig(out_png, dpi=150)
        print(f"Saved preview to {out_png}")


if __name__ == "__main__":
    main()
