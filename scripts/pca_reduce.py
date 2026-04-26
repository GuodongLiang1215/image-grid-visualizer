import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import (
    preprocess_input as resnet_preprocess,
    decode_predictions,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import datetime


# Function to load and preprocess image
def load_and_preprocess_image(img_path, target_size=(224, 224), preprocess_func=None):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    if preprocess_func:
        img_array = preprocess_func(img_array)
    return img_array

# Function to generate a random date between two dates
def get_random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + datetime.timedelta(days=random_days)
    return random_date


def load_tiny_imagenet_words(words_path="../tiny-imagenet-200/words.txt"):
    labels = {}
    if not os.path.exists(words_path):
        return labels

    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels

# Single ResNet101 backbone serves both classification (top=True) and feature
# extraction (avg-pool layer). One forward pass per image, one set of weights,
# and the banner ("ResNet101 features / predictions") is now truthful.
classifier_model = ResNet101(weights='imagenet', include_top=True)
feature_model = tf.keras.Model(
    inputs=classifier_model.input,
    outputs=classifier_model.get_layer('avg_pool').output,
)

# Directories
input_dir = 'images'  # Update this path to your images directory
output_features_file = 'features.json'
output_classification_file = 'classification_results.json'
output_tsne_file = 'tsne_results.json'
output_pca_file = 'pca_results.json'

# Initialize results
features_list = []
classification_results = []
source_labels = load_tiny_imagenet_words()

# Define start and end dates for random date generation
start_date = datetime.date(2023, 7, 1)
end_date = datetime.date(2024, 7, 1)

# Process images
for img_name in sorted(os.listdir(input_dir)):
    if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
        img_path = os.path.join(input_dir, img_name)
        source_id = img_name.split("_")[0]
        source_label = source_labels.get(source_id, source_id)

        img_array = load_and_preprocess_image(img_path, preprocess_func=resnet_preprocess)
        if img_array is not None:
            features = feature_model.predict(img_array, verbose=0).flatten().tolist()
            features_list.append(features)

            predictions = classifier_model.predict(img_array, verbose=0)
            top_predictions = decode_predictions(predictions, top=5)[0]
            decoded_predictions = top_predictions[0]

            random_date = get_random_date(start_date, end_date)

            classification_results.append({
                'image': img_name,
                'class_id': decoded_predictions[0],
                'class_name': decoded_predictions[1],
                'probability': float(decoded_predictions[2]),
                'top5': [
                    {
                        'class_id': pred[0],
                        'class_name': pred[1],
                        'probability': float(pred[2])
                    }
                    for pred in top_predictions
                ],
                'source_id': source_id,
                'source_label': source_label,
                'path': img_name,
                'date': random_date.isoformat()
            })


# Convert features_list to a NumPy array
features_array = np.array(features_list)

# Perform PCA
pca = PCA(n_components=50)  # Reduce to 50 components first
pca_results = pca.fit_transform(features_array)

# Perform t-SNE on PCA-reduced data.
# init="pca" and learning_rate="auto" follow current scikit-learn guidance for
# more stable embeddings than the old fully default configuration.
tsne = TSNE(
    n_components=2,
    perplexity=min(30, max(5, (len(pca_results) - 1) // 3)),
    init="pca",
    learning_rate="auto",
    random_state=42
)
tsne_results = tsne.fit_transform(pca_results).tolist()

# Save features to JSON
with open(output_features_file, 'w') as f:
    json.dump(features_list, f)

# Save classification results to JSON
with open(output_classification_file, 'w') as f:
    json.dump(classification_results, f)

# Save PCA results to JSON
with open(output_pca_file, 'w') as f:
    json.dump(pca_results.tolist(), f)

# Save t-SNE results to JSON
with open(output_tsne_file, 'w') as f:
    json.dump(tsne_results, f)

print(f'Features, classification results, PCA results, and t-SNE results saved to {output_features_file}, {output_classification_file}, {output_pca_file}, and {output_tsne_file} respectively.')

# Optional: Plot t-SNE and PCA results
def plot_results(results, title):
    results = np.array(results)
    plt.figure(figsize=(10, 8))
    plt.scatter(results[:, 0], results[:, 1], s=50, alpha=0.5)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

plot_results(tsne_results, 't-SNE Results')
